import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import wandb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

"""
HETEROGENEOUS GRAPH TRAINING FOR WAYMO TRAJECTORY PREDICTION

This file uses heterogeneous graphs with:
- Node types: agent, lane_segment, traffic_control
- Edge types: 8 different relationships
- Rich features: Frenet coordinates, lane topology, traffic signals

Uses EvolveGCN-H model for temporal evolution of heterogeneous graphs.
"""

# Add local Waymo module path
src_path = os.path.abspath(os.path.join(os.getcwd(), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from waymo_open_dataset.protos import scenario_pb2
from EvolveGCNH import EvolveGCNH

print(f"PyTorch version: {torch.__version__}")
print(f"TensorFlow version: {tf.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print("Torch CUDA Available: ", torch.cuda.is_available())
    torch.rand(10, device=device)
    print(torch.cuda.get_device_name(0))


# ============================================================================
# Helper Functions
# ============================================================================
def parse_scenario_file(file_path):
    """Parse a Waymo TFRecord file and return list of scenarios."""
    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    scenarios = []
    for raw_record in dataset:
        try:
            scenario = scenario_pb2.Scenario.FromString(raw_record.numpy())
            scenarios.append(scenario)
        except Exception as e:
            print(f"Error parsing scenario: {e}")
            break
    return scenarios


def compute_frenet_coordinates(agent_x, agent_y, agent_heading, lane_polyline):
    """
    Compute Frenet coordinates (s, d) for an agent relative to a lane polyline.
    
    Args:
        agent_x, agent_y: Agent position
        agent_heading: Agent heading angle
        lane_polyline: Nx3 array of [x, y, z] points
    
    Returns:
        s: longitudinal coordinate (progress along lane)
        d: lateral offset (distance from lane centerline)
        heading_diff: difference between agent heading and lane heading
    """
    # Find closest point on polyline
    distances = np.sqrt((lane_polyline[:, 0] - agent_x)**2 + 
                       (lane_polyline[:, 1] - agent_y)**2)
    closest_idx = np.argmin(distances)
    d = distances[closest_idx]  # Lateral offset
    
    # Compute longitudinal coordinate (accumulated distance along polyline)
    s = 0.0
    for i in range(closest_idx):
        s += np.sqrt((lane_polyline[i+1, 0] - lane_polyline[i, 0])**2 + 
                    (lane_polyline[i+1, 1] - lane_polyline[i, 1])**2)
    
    # Compute lane heading at closest point:
    if closest_idx < len(lane_polyline) - 1:
        dx = lane_polyline[closest_idx + 1, 0] - lane_polyline[closest_idx, 0]
        dy = lane_polyline[closest_idx + 1, 1] - lane_polyline[closest_idx, 1]
        lane_heading = np.arctan2(dy, dx)
    elif closest_idx > 0:   # closest point is the last point
        dx = lane_polyline[closest_idx, 0] - lane_polyline[closest_idx - 1, 0]
        dy = lane_polyline[closest_idx, 1] - lane_polyline[closest_idx - 1, 1]
        lane_heading = np.arctan2(dy, dx)
    else:
        lane_heading = 0.0
    
    heading_diff = agent_heading - lane_heading
    
    heading_diff_normalized = (heading_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
    
    return s, d, heading_diff_normalized


def visualize_scenario_snapshot(scenario, timestep, predictions=None, ground_truth=None, 
                                agent_ids=None, future_steps=8, save_path=None, 
                                show_past=True, show_lanes=True, title_suffix=""):
    """
    Visualize a scenario at a specific timestep with predictions and ground truth.
    
    Args:
        scenario: Waymo scenario protobuf
        timestep: Current timestep to visualize
        predictions: Tensor of shape [num_agents, future_steps*2] with predicted offsets (normalized)
        ground_truth: Tensor of shape [num_agents, future_steps*2] with true offsets (normalized)
        agent_ids: List of agent IDs corresponding to predictions/ground_truth
        future_steps: Number of future timesteps
        save_path: Path to save figure (if None, displays instead)
        show_past: Whether to show past trajectories
        show_lanes: Whether to show lane polylines
        title_suffix: Additional text for title
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    
    # Color scheme
    colors = {1: '#e74c3c', 2: '#3498db', 3: '#f39c12', 4: '#9b59b6'}  # Vehicle, Pedestrian, Cyclist, Other
    type_names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Other'}
    
    # ========================================================================
    # 1. Draw Lane Polylines (if enabled)
    # ========================================================================
    if show_lanes:
        for map_feature in scenario.map_features:
            if map_feature.HasField('lane'):
                lane = map_feature.lane
                polyline = [[p.x, p.y] for p in lane.polyline]
                if len(polyline) > 0:
                    polyline_np = np.array(polyline)
                    ax.plot(polyline_np[:, 0], polyline_np[:, 1], 
                           'lightgray', linewidth=1.5, alpha=0.5, zorder=1)
            
            elif map_feature.HasField('road_line'):
                road_line = map_feature.road_line
                polyline = [[p.x, p.y] for p in road_line.polyline]
                if len(polyline) > 0:
                    polyline_np = np.array(polyline)
                    ax.plot(polyline_np[:, 0], polyline_np[:, 1], 
                           'gray', linewidth=1, alpha=0.6, zorder=1)
    
    # ========================================================================
    # 2. Build agent ID to index mapping (for predictions/ground_truth)
    # ========================================================================
    agent_id_to_pred_idx = {}
    if agent_ids is not None:
        agent_id_to_pred_idx = {aid: idx for idx, aid in enumerate(agent_ids)}
    
    # ========================================================================
    # 3. Draw Agents and Trajectories
    # ========================================================================
    sdc_idx = getattr(scenario, 'sdc_track_index', None)
    sdc_id = None
    if sdc_idx is not None and 0 <= sdc_idx < len(scenario.tracks):
        sdc_id = scenario.tracks[sdc_idx].id
    
    for track in scenario.tracks:
        if timestep >= len(track.states):
            continue
        
        state = track.states[timestep]
        if not state.valid:
            continue
        
        agent_color = colors.get(track.object_type, '#95a5a6')
        current_x, current_y = state.center_x, state.center_y
        
        # Draw past trajectory (if enabled)
        if show_past:
            past_x, past_y = [], []
            for t in range(max(0, timestep - 10), timestep + 1):
                if t < len(track.states) and track.states[t].valid:
                    past_x.append(track.states[t].center_x)
                    past_y.append(track.states[t].center_y)
            
            if len(past_x) > 1:
                ax.plot(past_x, past_y, color=agent_color, linewidth=2, 
                       alpha=0.4, zorder=5, label=f'Past {type_names.get(track.object_type, "Other")}')
        
        # Draw ground truth future trajectory (dashed green)
        if ground_truth is not None and track.id in agent_id_to_pred_idx:
            pred_idx = agent_id_to_pred_idx[track.id]
            gt_offsets = ground_truth[pred_idx].reshape(future_steps, 2) * 10.0  # Denormalize
            
            gt_future_x = [current_x]
            gt_future_y = [current_y]
            for offset in gt_offsets:
                gt_future_x.append(gt_future_x[-1] + offset[0].item())
                gt_future_y.append(gt_future_y[-1] + offset[1].item())
            
            ax.plot(gt_future_x, gt_future_y, color='green', linewidth=2.5, 
                   linestyle='--', alpha=0.8, zorder=10, label='Ground Truth')
            
            # Mark future positions
            ax.scatter(gt_future_x[1:], gt_future_y[1:], color='green', 
                      s=30, alpha=0.6, zorder=11, marker='o')
        
        # Draw predicted future trajectory (solid red)
        if predictions is not None and track.id in agent_id_to_pred_idx:
            pred_idx = agent_id_to_pred_idx[track.id]
            pred_offsets = predictions[pred_idx].reshape(future_steps, 2) * 10.0  # Denormalize
            
            pred_future_x = [current_x]
            pred_future_y = [current_y]
            for offset in pred_offsets:
                pred_future_x.append(pred_future_x[-1] + offset[0].item())
                pred_future_y.append(pred_future_y[-1] + offset[1].item())
            
            ax.plot(pred_future_x, pred_future_y, color='red', linewidth=1.5, 
                   linestyle='-', alpha=0.9, zorder=12, label='Prediction')
            
            # Mark predicted positions
            #ax.scatter(pred_future_x[1:], pred_future_y[1:], color='red', 
            #          s=30, alpha=0.7, zorder=13, marker='x')
        
        # Draw current agent position (larger if SDC)
        if sdc_id is not None and track.id == sdc_id:
            ax.scatter(current_x, current_y, c='#27ae60', s=150, 
                      edgecolors='#08532b', linewidths=3, zorder=20, marker='D')
            ax.annotate('SDC', xy=(current_x, current_y), xytext=(10, 10),
                       textcoords='offset points', fontsize=10, fontweight='bold',
                       color='#08532b', zorder=21)
        else:
            ax.scatter(current_x, current_y, c=agent_color, s=100, 
                      edgecolors='white', linewidths=2, zorder=15, marker='o')
        
        # Agent ID label
        """ax.annotate(str(track.id), xy=(current_x, current_y), xytext=(0, -20),
                   textcoords='offset points', fontsize=8, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   zorder=16)"""
    
    # ========================================================================
    # 4. Configure Plot
    # ========================================================================
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    
    title = f'Scenario: {scenario.scenario_id}\nTimestep: {timestep}'
    if title_suffix:
        title += f' | {title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Auto-scale with margin
    all_x = [s.center_x for t in scenario.tracks for s in t.states if s.valid]
    all_y = [s.center_y for t in scenario.tracks for s in t.states if s.valid]
    if all_x and all_y:
        margin = 30
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    # Create custom legend (avoiding duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved visualization to: {save_path}")
    
    return fig, ax


def scenario_to_heterogeneous_graph(scenario, timestep, interaction_radius=30.0, future_steps=8):
    """
    Convert a Waymo scenario at a specific timestep to a PyG HeteroData object.
    
    Creates a heterogeneous graph with:
    - Node types: agent (A), lane_segment (LS), traffic_control (TC)
    - Edge types: various relationships between nodes
    - Rich node and edge features
    
    Args:
        scenario: Waymo scenario protobuf
        timestep: Current timestep to extract
        interaction_radius: Radius for agent-agent interactions
        future_steps: Number of future timesteps to predict
    
    Returns:
        HeteroData object with heterogeneous graph structure
    """
    hetero_data = HeteroData()
    
    # ========================================================================
    # 1. Build Agent Nodes (A)
    # ========================================================================
    agent_features = []
    agent_positions = []
    agent_ids_list = []
    agent_headings = []
    
    for agent in scenario.tracks:
        if timestep >= len(agent.states):
            continue
        state = agent.states[timestep]
        if not state.valid:
            continue
        
        # Agent features: [x, y, vx, vy, valid, type_1, type_2, type_3, type_4]
        properties = [
            state.center_x / 100.0,  # Normalize position
            state.center_y / 100.0,
            state.velocity_x / 10.0,  # Normalize velocity
            state.velocity_y / 10.0,
            float(state.valid)  # Valid flag
        ]
        
        # Object type one-hot encoding (vehicle, pedestrian, cyclist, other)
        # can use this because agents are known in advance - transductive setting
        type_onehot = [
            1 if agent.object_type == 1 else 0,
            1 if agent.object_type == 2 else 0,
            1 if agent.object_type == 3 else 0,
            1 if agent.object_type == 4 else 0
        ]
        
        features = properties + type_onehot  # append and get 9 features
        
        agent_features.append(features)
        agent_positions.append([state.center_x, state.center_y])
        agent_ids_list.append(agent.id)
        agent_headings.append(state.heading)
    
    if len(agent_features) > 0:
        hetero_data['agent'].x = torch.tensor(agent_features, dtype=torch.float32)
        hetero_data['agent'].pos = torch.tensor(agent_positions, dtype=torch.float32)
        hetero_data['agent'].agent_ids = agent_ids_list
    else:
        return None     # No valid agents at this timestep
    
    # ========================================================================
    # 2. Build Agent Labels (Future Trajectories)
    # ========================================================================
    labels = []
    id_to_agent = {t.id: t for t in scenario.tracks}
    for i, agent_id in enumerate(agent_ids_list):
        agent = id_to_agent[agent_id]
        future_pos = []
        for t in range(1, future_steps + 1):
            future_t = timestep + t
            if future_t < len(agent.states) and agent.states[future_t].valid:
                future_pos.append([
                    agent.states[future_t].center_x,
                    agent.states[future_t].center_y
                ])
            else:
                last = agent.states[min(future_t, len(agent.states) - 1)]
                future_pos.append([last.center_x, last.center_y])
        
        current_pos = torch.tensor(agent_positions[i], dtype=torch.float32)
        future_tensor = torch.tensor(future_pos, dtype=torch.float32)
        offsets = (future_tensor - current_pos) / 10.0
        labels.append(offsets.flatten())
    
    hetero_data['agent'].y = torch.stack(labels)
    
    # ========================================================================
    # 3. Build Lane Segment Nodes (LS)
    # ========================================================================
    lane_features = []
    lane_polylines = []  # Store for Frenet computation
    lane_ids_list = []
    
    # Lane type encoding (one-hot or similar)
    lane_type_map = {
        0: [1, 0, 0],  # TYPE_UNDEFINED
        1: [0, 1, 0],  # TYPE_FREEWAY
        2: [0, 0, 1],  # TYPE_SURFACE_STREET
    }
    
    for map_feature in scenario.map_features:
        if map_feature.HasField('lane'):
            lane = map_feature.lane
            
            # Extract polyline
            polyline = [[p.x, p.y, p.z] for p in lane.polyline]
            if len(polyline) == 0:
                continue
            
            polyline_np = np.array(polyline)
            lane_polylines.append(polyline_np)
            lane_ids_list.append(map_feature.id)
            
            # Compute lane features: start/end points and center
            start = polyline_np[0, :2] / 100.0  # Normalize
            end = polyline_np[-1, :2] / 100.0
            center = polyline_np[len(polyline_np)//2, :2] / 100.0
            
            # Lane type
            lane_type_encoding = lane_type_map.get(lane.type, [0, 0, 0])
            
            # Feature: [start_x, start_y, center_x, center_y, end_x, end_y, type_encoding...]
            features = list(start) + list(center) + list(end) + lane_type_encoding
            lane_features.append(features)
    
    if len(lane_features) > 0:
        hetero_data['lane_segment'].x = torch.tensor(lane_features, dtype=torch.float32)
        hetero_data['lane_segment'].lane_ids = lane_ids_list
        hetero_data['lane_segment'].polylines = lane_polylines
    
    # ========================================================================
    # 4. Build Traffic Control Nodes (TC)
    # ========================================================================
    tc_features = []
    tc_positions = []
    tc_ids_list = []
    
    for map_feature in scenario.map_features:
        if map_feature.HasField('stop_sign'):
            stop_sign = map_feature.stop_sign
            pos = [stop_sign.position.x / 100.0, stop_sign.position.y / 100.0]
            tc_features.append(pos)
            tc_positions.append([stop_sign.position.x, stop_sign.position.y])
            tc_ids_list.append(map_feature.id)
        
        elif map_feature.HasField('crosswalk'):
            crosswalk = map_feature.crosswalk
            if len(crosswalk.polygon) > 0:
                center_x = np.mean([p.x for p in crosswalk.polygon]) / 100.0
                center_y = np.mean([p.y for p in crosswalk.polygon]) / 100.0
                tc_features.append([center_x, center_y])
                tc_positions.append([center_x * 100.0, center_y * 100.0])
                tc_ids_list.append(map_feature.id)
    
    if len(tc_features) > 0:
        hetero_data['traffic_control'].x = torch.tensor(tc_features, dtype=torch.float32)
        hetero_data['traffic_control'].pos = torch.tensor(tc_positions, dtype=torch.float32)
        hetero_data['traffic_control'].tc_ids = tc_ids_list
    
    # ========================================================================
    # 5. Build Edges: Lane Segment Topology
    # ========================================================================
    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids_list)}
    
    ls_succeeds_ls_src, ls_succeeds_ls_dst = [], []
    ls_precedes_ls_src, ls_precedes_ls_dst = [], []
    ls_left_adj_src, ls_left_adj_dst = [], []
    ls_right_adj_src, ls_right_adj_dst = [], []
    
    for map_feature in scenario.map_features:
        if map_feature.HasField('lane'):
            if map_feature.id not in lane_id_to_idx:
                continue
            src_idx = lane_id_to_idx[map_feature.id]
            lane = map_feature.lane
            
            for entry_id in lane.entry_lanes:
                if entry_id in lane_id_to_idx:
                    entry_idx = lane_id_to_idx[entry_id]
                    ls_precedes_ls_src.append(entry_idx)
                    ls_precedes_ls_dst.append(src_idx)
            
            for exit_id in lane.exit_lanes:
                if exit_id in lane_id_to_idx:
                    exit_idx = lane_id_to_idx[exit_id]
                    ls_succeeds_ls_src.append(src_idx)
                    ls_succeeds_ls_dst.append(exit_idx)
            
            for left_id in lane.left_neighbors:
                if left_id.feature_id in lane_id_to_idx:
                    left_idx = lane_id_to_idx[left_id.feature_id]
                    ls_left_adj_src.append(src_idx)
                    ls_left_adj_dst.append(left_idx)
            
            for right_id in lane.right_neighbors:
                if right_id.feature_id in lane_id_to_idx:
                    right_idx = lane_id_to_idx[right_id.feature_id]
                    ls_right_adj_src.append(src_idx)
                    ls_right_adj_dst.append(right_idx)
    
    if len(ls_succeeds_ls_src) > 0:
        hetero_data['lane_segment', 'succeeds', 'lane_segment'].edge_index = torch.tensor(
            [ls_succeeds_ls_src, ls_succeeds_ls_dst], dtype=torch.long)
    
    if len(ls_precedes_ls_src) > 0:
        hetero_data['lane_segment', 'precedes', 'lane_segment'].edge_index = torch.tensor(
            [ls_precedes_ls_src, ls_precedes_ls_dst], dtype=torch.long)
    
    if len(ls_left_adj_src) > 0:
        hetero_data['lane_segment', 'left_of', 'lane_segment'].edge_index = torch.tensor(
            [ls_left_adj_src, ls_left_adj_dst], dtype=torch.long)
    
    if len(ls_right_adj_src) > 0:
        hetero_data['lane_segment', 'right_of', 'lane_segment'].edge_index = torch.tensor(
            [ls_right_adj_src, ls_right_adj_dst], dtype=torch.long)
    
    # ========================================================================
    # 6. Build Edges: Agent-Lane (A on_lane LS)
    # ========================================================================
    a_onlane_ls_src, a_onlane_ls_dst = [], []
    a_onlane_ls_edge_attr = []
    
    for agent_idx, (agent_pos, agent_heading) in enumerate(zip(agent_positions, agent_headings)):
        for lane_idx, polyline in enumerate(lane_polylines):
            distances = np.sqrt((polyline[:, 0] - agent_pos[0])**2 + 
                              (polyline[:, 1] - agent_pos[1])**2)
            min_dist = np.min(distances)
            
            if min_dist < 5.0:
                s, d, heading_diff = compute_frenet_coordinates(
                    agent_pos[0], agent_pos[1], agent_heading, polyline)
                
                a_onlane_ls_src.append(agent_idx)
                a_onlane_ls_dst.append(lane_idx)
                a_onlane_ls_edge_attr.append([s / 100.0, d / 10.0, heading_diff])
    
    if len(a_onlane_ls_src) > 0:
        hetero_data['agent', 'on_lane', 'lane_segment'].edge_index = torch.tensor(
            [a_onlane_ls_src, a_onlane_ls_dst], dtype=torch.long)
        hetero_data['agent', 'on_lane', 'lane_segment'].edge_attr = torch.tensor(
            a_onlane_ls_edge_attr, dtype=torch.float32)
    
    # ========================================================================
    # 7. Build Edges: Agent-Traffic Control (A observing TC)
    # ========================================================================
    if len(tc_positions) > 0:
        a_observing_tc_src, a_observing_tc_dst = [], []
        
        for agent_idx, agent_pos in enumerate(agent_positions):
            for tc_idx, tc_pos in enumerate(tc_positions):
                dist = np.sqrt((agent_pos[0] - tc_pos[0])**2 + 
                             (agent_pos[1] - tc_pos[1])**2)
                if dist < 50.0:
                    a_observing_tc_src.append(agent_idx)
                    a_observing_tc_dst.append(tc_idx)
        
        if len(a_observing_tc_src) > 0:
            hetero_data['agent', 'observing', 'traffic_control'].edge_index = torch.tensor(
                [a_observing_tc_src, a_observing_tc_dst], dtype=torch.long)
    
    # ========================================================================
    # 8. Build Edges: Agent-Agent Interactions (A interacting_with A)
    # ========================================================================
    a_interact_a_src, a_interact_a_dst = [], []
    a_interact_a_edge_attr = []
    
    agent_positions_np = np.array(agent_positions)
    for i in range(len(agent_positions)):
        for j in range(len(agent_positions)):
            if i == j:
                continue
            
            delta_x = agent_positions_np[j, 0] - agent_positions_np[i, 0]
            delta_y = agent_positions_np[j, 1] - agent_positions_np[i, 1]
            dist = np.sqrt(delta_x**2 + delta_y**2)
            
            if dist < interaction_radius:
                a_interact_a_src.append(i)
                a_interact_a_dst.append(j)
                a_interact_a_edge_attr.append([delta_x / 100.0, delta_y / 100.0])
    
    if len(a_interact_a_src) > 0:
        hetero_data['agent', 'interacting_with', 'agent'].edge_index = torch.tensor(
            [a_interact_a_src, a_interact_a_dst], dtype=torch.long)
        hetero_data['agent', 'interacting_with', 'agent'].edge_attr = torch.tensor(
            a_interact_a_edge_attr, dtype=torch.float32)
    
    hetero_data.scenario_id = scenario.scenario_id
    hetero_data.timestep = timestep
    
    return hetero_data


# ============================================================================
# Heterogeneous GNN Model - Using EvolveGCN-H with Agent-only output
# ============================================================================

class HeteroEvolveGCN(torch.nn.Module):
    """
    Wrapper around EvolveGCN-H that handles heterogeneous graphs and
    only outputs predictions for agent nodes.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, topk=30):
        super().__init__()
        
        # Input normalization layer
        self.input_norm = torch.nn.LayerNorm(input_dim)
        
        # Use EvolveGCN-H as backbone
        self.evolvegcn = EvolveGCNH(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output hidden_dim, not final output_dim
            num_layers=num_layers,
            dropout=dropout,
            topk=topk
        )
        
        # Output head with batch normalization
        self.output_norm = torch.nn.LayerNorm(hidden_dim)
        self.output_fc = torch.nn.Linear(hidden_dim, output_dim)
        
        # Initialize output layer with small weights
        torch.nn.init.xavier_uniform_(self.output_fc.weight, gain=0.01)
        torch.nn.init.zeros_(self.output_fc.bias)
        
    def forward(self, x, edge_index, node_counts=None):
        """
        Forward pass that only returns predictions for agent nodes.
        
        Args:
            x: Node features for all nodes
            edge_index: Edge connectivity
            node_counts: Dictionary with (start, end) indices for each node type
        
        Returns:
            Predictions for agent nodes only
        """
        # Normalize input
        x = self.input_norm(x)
        
        # Get embeddings for all nodes
        embeddings = self.evolvegcn(x, edge_index)
        
        # Extract agent embeddings
        if node_counts is not None and 'agent' in node_counts:
            start_idx, end_idx = node_counts['agent']
            agent_embeddings = embeddings[start_idx:end_idx]
        else:
            agent_embeddings = embeddings
        
        # Apply output head with normalization
        agent_embeddings = self.output_norm(agent_embeddings)
        out = self.output_fc(agent_embeddings)
        
        return out
    
    @property
    def gru_h(self):
        return self.evolvegcn.gru_h
    
    @gru_h.setter
    def gru_h(self, value):
        self.evolvegcn.gru_h = value


# Note: EvolveGCN-H expects homogeneous graphs, so we flatten heterogeneous
# graphs into a single node type with heterogeneous edge features preserved

def hetero_to_homo_temporal_graphs(hetero_graphs):
    """
    Convert sequence of HeteroData objects to homogeneous temporal graphs.
    Flattens all node types into a single node type while preserving structure.
    
    Args:
        hetero_graphs: List of HeteroData objects (temporal sequence)
    
    Returns:
        List of homogeneous Data objects compatible with EvolveGCN-H
    """
    from torch_geometric.data import Data
    
    homo_graphs = []
    
    for hetero_data in hetero_graphs:
        # Collect all node features
        node_features = []
        node_type_encoding = []
        node_counts = {}
        offset = 0
        
        # Determine max feature dimension across all node types
        max_dim = 0
        if 'agent' in hetero_data.node_types:
            max_dim = max(max_dim, hetero_data['agent'].x.shape[1])
        if 'lane_segment' in hetero_data.node_types:
            max_dim = max(max_dim, hetero_data['lane_segment'].x.shape[1])
        if 'traffic_control' in hetero_data.node_types:
            max_dim = max(max_dim, hetero_data['traffic_control'].x.shape[1])
        
        # Agent nodes (type 0)
        if 'agent' in hetero_data.node_types:
            num_agents = hetero_data['agent'].x.shape[0]
            node_counts['agent'] = (offset, offset + num_agents)
            
            # Pad agent features to max_dim
            agent_feats = hetero_data['agent'].x
            if agent_feats.shape[1] < max_dim:
                padding = torch.zeros(num_agents, max_dim - agent_feats.shape[1])
                agent_feats = torch.cat([agent_feats, padding], dim=1)
            node_features.append(agent_feats)
            
            # Type encoding: [1, 0, 0] for agents
            node_type_encoding.append(torch.zeros(num_agents, 3))
            node_type_encoding[-1][:, 0] = 1
            offset += num_agents
        
        # Lane segment nodes (type 1)
        if 'lane_segment' in hetero_data.node_types:
            num_lanes = hetero_data['lane_segment'].x.shape[0]
            node_counts['lane_segment'] = (offset, offset + num_lanes)
            
            # Pad lane features to max_dim
            lane_feats = hetero_data['lane_segment'].x
            if lane_feats.shape[1] < max_dim:
                padding = torch.zeros(num_lanes, max_dim - lane_feats.shape[1])
                lane_feats = torch.cat([lane_feats, padding], dim=1)
            node_features.append(lane_feats)
            
            # Type encoding: [0, 1, 0] for lanes
            node_type_encoding.append(torch.zeros(num_lanes, 3))
            node_type_encoding[-1][:, 1] = 1
            offset += num_lanes
        
        # Traffic control nodes (type 2)
        if 'traffic_control' in hetero_data.node_types:
            num_tc = hetero_data['traffic_control'].x.shape[0]
            node_counts['traffic_control'] = (offset, offset + num_tc)
            
            # Pad traffic control features to max_dim
            tc_feats = hetero_data['traffic_control'].x
            if tc_feats.shape[1] < max_dim:
                padding = torch.zeros(num_tc, max_dim - tc_feats.shape[1])
                tc_feats = torch.cat([tc_feats, padding], dim=1)
            node_features.append(tc_feats)
            
            # Type encoding: [0, 0, 1] for traffic controls
            node_type_encoding.append(torch.zeros(num_tc, 3))
            node_type_encoding[-1][:, 2] = 1
            offset += num_tc
        
        # Concatenate all features (now all have same dimension)
        all_features = torch.cat(node_features, dim=0)
        all_type_encodings = torch.cat(node_type_encoding, dim=0)
        
        # Combine features with type encoding
        x = torch.cat([all_features, all_type_encodings], dim=1)  # [num_nodes, max_dim + 3]
        
        # Collect all edges
        edge_indices = []
        
        # Agent-agent interactions
        if ('agent', 'interacting_with', 'agent') in hetero_data.edge_types:
            edge_idx = hetero_data['agent', 'interacting_with', 'agent'].edge_index
            # Offset already correct (both are agents, offset 0)
            if 'agent' in node_counts:
                edge_indices.append(edge_idx + node_counts['agent'][0])
        
        # Agent-lane edges
        if ('agent', 'on_lane', 'lane_segment') in hetero_data.edge_types:
            edge_idx = hetero_data['agent', 'on_lane', 'lane_segment'].edge_index
            if 'agent' in node_counts and 'lane_segment' in node_counts:
                src = edge_idx[0] + node_counts['agent'][0]
                dst = edge_idx[1] + node_counts['lane_segment'][0]
                edge_indices.append(torch.stack([src, dst]))
        
        # Lane topology edges
        for rel in ['succeeds', 'precedes', 'left_of', 'right_of']:
            if ('lane_segment', rel, 'lane_segment') in hetero_data.edge_types:
                edge_idx = hetero_data['lane_segment', rel, 'lane_segment'].edge_index
                if 'lane_segment' in node_counts:
                    edge_indices.append(edge_idx + node_counts['lane_segment'][0])
        
        # Agent-traffic control edges
        if ('agent', 'observing', 'traffic_control') in hetero_data.edge_types:
            edge_idx = hetero_data['agent', 'observing', 'traffic_control'].edge_index
            if 'agent' in node_counts and 'traffic_control' in node_counts:
                src = edge_idx[0] + node_counts['agent'][0]
                dst = edge_idx[1] + node_counts['traffic_control'][0]
                edge_indices.append(torch.stack([src, dst]))
        
        # Concatenate all edges
        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create labels (only for agent nodes)
        if 'agent' in node_counts:
            y = hetero_data['agent'].y
        else:
            y = torch.zeros((0, 16))  # Dummy labels
        
        # Create homogeneous graph
        homo_data = Data(x=x, edge_index=edge_index, y=y)
        homo_data.node_counts = node_counts
        homo_data.scenario_id = hetero_data.scenario_id
        homo_data.timestep = hetero_data.timestep
        
        homo_graphs.append(homo_data)
    
    return homo_graphs


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def scenario_to_temporal_heterogeneous_graphs(scenario, start_timestep, end_timestep, interaction_radius=30.0, future_steps=8):
    """
    Convert a scenario to a sequence of temporal heterogeneous graphs.
    
    Args:
        scenario: Waymo scenario protobuf
        start_timestep: Starting timestep
        end_timestep: Ending timestep (exclusive)
        interaction_radius: Radius for agent-agent interactions
        future_steps: Number of future timesteps to predict
    
    Returns:
        List of HeteroData objects (temporal sequence)
    """
    graphs = []
    for t in range(start_timestep, end_timestep):
        hetero_graph = scenario_to_heterogeneous_graph(
            scenario, t, interaction_radius, future_steps
        )
        if hetero_graph is not None:
            graphs.append(hetero_graph)
    return graphs


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, train_scenarios, optimizer, criterion, device, config, epoch):
    """Train model for one epoch using temporal heterogeneous graphs."""
    model.train()
    total_loss = 0
    num_predictions = 0
    
    # Learning rate warmup
    if epoch < config['warmup_epochs']:
        warmup_factor = (epoch + 1) / config['warmup_epochs']
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['learning_rate'] * warmup_factor
    
    for scenario in train_scenarios:
        # Get temporal sequence of HISTORY graphs (timesteps 0-9)
        hetero_history_graphs = scenario_to_temporal_heterogeneous_graphs(
            scenario,
            start_timestep=0,
            end_timestep=config['timestep'],  # Up to but NOT including target
            interaction_radius=config['radius'],
            future_steps=config['future_steps']
        )
        
        # Get TARGET graph (timestep 10) separately
        hetero_target_graph = scenario_to_heterogeneous_graph(
            scenario,
            timestep=config['timestep'],
            interaction_radius=config['radius'],
            future_steps=config['future_steps']
        )
        
        if len(hetero_history_graphs) == 0 or hetero_target_graph is None:
            continue
        
        # Convert to homogeneous graphs
        history_graphs = hetero_to_homo_temporal_graphs(hetero_history_graphs)
        target_graph = hetero_to_homo_temporal_graphs([hetero_target_graph])[0]
        
        if len(history_graphs) == 0:
            continue
        
        # Reset GRU hidden state for each scenario
        model.gru_h = None
        optimizer.zero_grad()
        
        # Process ALL graphs (history + target) to build up temporal state
        # Gradients flow through the entire sequence!
        all_graphs = history_graphs + [target_graph]
        target_out = None

        for idx, graph in enumerate(all_graphs):
            graph = graph.to(device)
            out = model(graph.x, graph.edge_index, graph.node_counts)

            if idx == len(all_graphs) - 1:
                target_graph = graph  # GPU copy of the target graph
                target_out = out

        # Compute loss ONLY on the final (target) prediction
        target_y = target_graph.y
        loss = criterion(target_out, target_y)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
        
        # Clip GRU hidden states to prevent explosion
        if model.gru_h is not None:
            with torch.no_grad():
                for i in range(len(model.gru_h)):
                    model.gru_h[i].clamp_(-10, 10)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_predictions += 1
    
    return total_loss / num_predictions if num_predictions > 0 else 0


def validate(model, val_scenarios, criterion, device, config):
    """Validate model on temporal heterogeneous graphs."""
    model.eval()
    total_loss = 0
    total_mae = 0
    num_predictions = 0
    
    with torch.no_grad():
        for scenario in val_scenarios:
            hetero_history_graphs = scenario_to_temporal_heterogeneous_graphs(
                scenario,
                start_timestep=0,
                end_timestep=config['timestep'],
                interaction_radius=config['radius'],
                future_steps=config['future_steps']
            )
            
            hetero_target_graph = scenario_to_heterogeneous_graph(
                scenario,
                timestep=config['timestep'],
                interaction_radius=config['radius'],
                future_steps=config['future_steps']
            )
            
            if len(hetero_history_graphs) == 0 or hetero_target_graph is None:
                continue
            
            history_graphs = hetero_to_homo_temporal_graphs(hetero_history_graphs)
            target_graph = hetero_to_homo_temporal_graphs([hetero_target_graph])[0]
            
            model.gru_h = None
            
            for graph in history_graphs:
                graph = graph.to(device)
                _ = model(graph.x, graph.edge_index, graph.node_counts)
            
            target_graph = target_graph.to(device)
            out = model(target_graph.x, target_graph.edge_index, target_graph.node_counts)
            
            # Predictions are already for agent nodes only
            loss = criterion(out, target_graph.y)
            mae = torch.nn.L1Loss()(out, target_graph.y)
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_predictions += 1
    
    avg_loss = total_loss / num_predictions if num_predictions > 0 else 0
    avg_mae = total_mae / num_predictions if num_predictions > 0 else 0
    
    return avg_loss, avg_mae


def test(model, test_scenarios, criterion, device, config, visualize_top_n=5):
    """
    Test model with detailed metrics on temporal heterogeneous graphs.
    
    Args:
        model: Trained model
        test_scenarios: List of test scenarios
        criterion: Loss function
        device: Device to run on
        config: Configuration dictionary
        visualize_top_n: Number of scenarios to visualize (default: 5)
    
    Returns:
        Dictionary with test metrics and results
    """
    model.eval()
    total_mse = 0
    total_mae = 0
    all_predictions = []
    all_targets = []
    per_scenario_metrics = []
    num_predictions = 0
    
    # Create visualization directory
    viz_dir = './test_visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, scenario in enumerate(test_scenarios):
            hetero_history_graphs = scenario_to_temporal_heterogeneous_graphs(
                scenario,
                start_timestep=0,
                end_timestep=config['timestep'],
                interaction_radius=config['radius'],
                future_steps=config['future_steps']
            )
            
            hetero_target_graph = scenario_to_heterogeneous_graph(
                scenario,
                timestep=config['timestep'],
                interaction_radius=config['radius'],
                future_steps=config['future_steps']
            )
            
            if len(hetero_history_graphs) == 0 or hetero_target_graph is None:
                continue
            
            history_graphs = hetero_to_homo_temporal_graphs(hetero_history_graphs)
            target_graph = hetero_to_homo_temporal_graphs([hetero_target_graph])[0]
            
            model.gru_h = None
            
            for graph in history_graphs:
                graph = graph.to(device)
                _ = model(graph.x, graph.edge_index, graph.node_counts)
            
            target_graph = target_graph.to(device)
            out = model(target_graph.x, target_graph.edge_index, target_graph.node_counts)
            
            # Predictions are already for agent nodes only
            mse = criterion(out, target_graph.y)
            mae = torch.nn.L1Loss()(out, target_graph.y)
            
            scenario_preds = out.cpu()
            scenario_targets = target_graph.y.cpu()
            
            pred_final = scenario_preds.reshape(-1, config['future_steps'], 2)[:, -1, :]
            target_final = scenario_targets.reshape(-1, config['future_steps'], 2)[:, -1, :]
            scenario_ade = torch.mean(torch.sqrt(torch.sum((pred_final - target_final) ** 2, dim=1)))
            
            num_agents = scenario_preds.shape[0]
            per_scenario_metrics.append({
                'scenario_idx': idx,
                'mse': mse.item(),
                'mae': mae.item(),
                'ade': scenario_ade.item(),
                'num_agents': num_agents
            })
            
            # Visualize first N scenarios
            if idx < visualize_top_n:
                # Get agent IDs from hetero graph
                if hasattr(hetero_target_graph['agent'], 'agent_ids'):
                    agent_ids = hetero_target_graph['agent'].agent_ids
                else:
                    agent_ids = None
                
                viz_path = os.path.join(viz_dir, f'scenario_{idx:03d}_pred_vs_gt.png')
                ade_meters = scenario_ade.item() * 10.0  # Denormalize
                
                visualize_scenario_snapshot(
                    scenario=scenario,
                    timestep=config['timestep'],
                    predictions=scenario_preds,
                    ground_truth=scenario_targets,
                    agent_ids=agent_ids,
                    future_steps=config['future_steps'],
                    save_path=viz_path,
                    show_past=True,
                    show_lanes=True,
                    title_suffix=f'ADE: {ade_meters:.2f}m | Agents: {num_agents}'
                )
            
            total_mse += mse.item()
            total_mae += mae.item()
            
            all_predictions.append(scenario_preds)
            all_targets.append(scenario_targets)
            num_predictions += 1
    
    avg_mse = total_mse / num_predictions if num_predictions > 0 else 0
    avg_mae = total_mae / num_predictions if num_predictions > 0 else 0
    
    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        final_step_predictions = all_predictions.reshape(-1, config['future_steps'], 2)[:, -1, :]
        final_step_targets = all_targets.reshape(-1, config['future_steps'], 2)[:, -1, :]
        ade = torch.mean(torch.sqrt(torch.sum((final_step_predictions - final_step_targets) ** 2, dim=1)))
        fde = torch.mean(torch.sqrt(torch.sum((final_step_predictions - final_step_targets) ** 2, dim=1)))
        
        all_timesteps_pred = all_predictions.reshape(-1, config['future_steps'], 2)
        all_timesteps_target = all_targets.reshape(-1, config['future_steps'], 2)
        timestep_errors = []
        for t in range(config['future_steps']):
            t_error = torch.mean(torch.sqrt(torch.sum((all_timesteps_pred[:, t, :] - all_timesteps_target[:, t, :]) ** 2, dim=1)))
            timestep_errors.append(t_error.item())
    else:
        ade = torch.tensor(0.0)
        fde = torch.tensor(0.0)
        timestep_errors = []
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'ade': ade.item(),
        'fde': fde.item(),
        'predictions': all_predictions,
        'targets': all_targets,
        'per_scenario_metrics': per_scenario_metrics,
        'timestep_errors': timestep_errors,
        'num_scenarios_tested': num_predictions
    }


# ============================================================================
# Main Training Script
# ============================================================================
def main():
    config = {
        "learning_rate": 0.002,  # Lower initial LR with warmup
        "weight_decay": 1e-4,  # Stronger regularization
        "epochs": 150,  # More epochs with lower LR
        "hidden_channels": 64,
        "dropout": 0.25,  # Increased dropout
        "dataset": "Waymo Open Motion Dataset",
        "architecture": "EvolveGCN-H (Heterogeneous)",
        "radius": 30.0,
        "future_steps": 8,
        "timestep": 10,
        "num_scenarios_per_split": 10,
        "topk": 30,
        "num_layers": 5,
        "val_frequency": 3,
        "warmup_epochs": 5,  # Gradual LR warmup
        "max_grad_norm": 0.5  # Stronger gradient clipping
    }
    
    print(f"Configuration: {config}")
    print(f"Using device: {device}")
    
    print("="*60)
    print("Loading Waymo Open Motion Dataset")
    print("="*60)
    
    def load_scenarios_from_dir(data_dir, num_scenarios):
        """Load scenarios from a directory of TFRecord files."""
        all_files = os.listdir(data_dir)
        tfrecord_files = [
            os.path.join(data_dir, f) 
            for f in all_files
            if f.startswith('uncompressed_') and 'tfrecord' in f and not f.endswith('.index')
        ]
        
        if not tfrecord_files:
            return []
        
        scenarios = []
        for tfrecord_file in tfrecord_files:
            if len(scenarios) >= num_scenarios:
                break
            file_scenarios = parse_scenario_file(tfrecord_file)
            scenarios.extend(file_scenarios[:num_scenarios - len(scenarios)])
        
        return scenarios
    
    train_dir = './data/scenario/training'
    val_dir = './data/scenario/validation'
    test_dir = './data/scenario/testing'
    
    print(f"Loading training scenarios from: {train_dir}")
    train_scenarios = load_scenarios_from_dir(train_dir, config['num_scenarios_per_split'])
    
    print(f"Loading validation scenarios from: {val_dir}")
    val_scenarios = load_scenarios_from_dir(val_dir, config['num_scenarios_per_split'])
    
    print(f"Loading test scenarios from: {test_dir}")
    test_scenarios = load_scenarios_from_dir(test_dir, config['num_scenarios_per_split'])
    
    print(f"\nDataset splits:")
    print(f"  Training scenarios: {len(train_scenarios)}")
    print(f"  Validation scenarios: {len(val_scenarios)}")
    print(f"  Test scenarios: {len(test_scenarios)}")
    
    if len(train_scenarios) == 0:
        print("\nNo training scenarios found. Exiting.")
        return
    
    # Create sample heterogeneous graph
    print(f"\nCreating sample heterogeneous graph...")
    sample_hetero_graph = scenario_to_heterogeneous_graph(
        train_scenarios[0],
        timestep=config['timestep'],
        interaction_radius=config['radius'],
        future_steps=config['future_steps']
    )
    
    if sample_hetero_graph:
        print(f"\n{'='*60}")
        print("HETEROGENEOUS GRAPH STRUCTURE")
        print(f"{'='*60}")
        print(f"\nNode Types:")
        print(f"  • Agents: {sample_hetero_graph['agent'].x.shape[0]} nodes, {sample_hetero_graph['agent'].x.shape[1]}D features")
        if 'lane_segment' in sample_hetero_graph.node_types:
            print(f"  • Lane Segments: {sample_hetero_graph['lane_segment'].x.shape[0]} nodes, {sample_hetero_graph['lane_segment'].x.shape[1]}D features")
        if 'traffic_control' in sample_hetero_graph.node_types:
            print(f"  • Traffic Controls: {sample_hetero_graph['traffic_control'].x.shape[0]} nodes, {sample_hetero_graph['traffic_control'].x.shape[1]}D features")
        
        print(f"\nEdge Types ({len(sample_hetero_graph.edge_types)} total):")
        for edge_type in sample_hetero_graph.edge_types:
            src, rel, dst = edge_type
            num_edges = sample_hetero_graph[edge_type].edge_index.shape[1]
            print(f"  • {src} --[{rel}]--> {dst}: {num_edges} edges")
        
        # Convert to homogeneous and show structure
        sample_homo_graphs = hetero_to_homo_temporal_graphs([sample_hetero_graph])
        sample_homo = sample_homo_graphs[0]
        print(f"\nAfter flattening to homogeneous graph:")
        print(f"  • Total nodes: {sample_homo.x.shape[0]}")
        print(f"  • Node feature dim: {sample_homo.x.shape[1]}D (includes 3D type encoding)")
        print(f"  • Total edges: {sample_homo.edge_index.shape[1]}")
        print(f"{'='*60}\n")
    else:
        print("Failed to create sample graph. Exiting.")
        return
    
    # Initialize EvolveGCN-H model with wrapper
    # Input dim: max_dim (9) + type_encoding (3) = 12
    input_dim = 12
    output_dim = config['future_steps'] * 2
    
    model = HeteroEvolveGCN(
        input_dim=input_dim,
        hidden_dim=config['hidden_channels'],
        output_dim=output_dim,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        topk=config['topk']
    ).to(device)
    
    print(f"Model initialized: EvolveGCN-H (Heterogeneous)")
    print(f"  Input dimension: {input_dim} (max 9D features + 3D type encoding)")
    print(f"    Agent: 9D (x,y,vx,vy,valid + 4D type one-hot)")
    print(f"    Lane: 9D (x,y,dir_x,dir_y,type + 4D type one-hot)")
    print(f"    Traffic: 2D (x,y) → padded to 9D")
    print(f"  Hidden channels: {config['hidden_channels']}")
    print(f"  Output dimension: {output_dim} ({config['future_steps']} timesteps × 2D)")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Top-k nodes: {config['topk']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    wandb.login()
    wandb.init(
        project="waymo-trajectory-prediction",
        config=config,
        name=f"HeteroEvolveGCN_h{config['hidden_channels']}_l{config['num_layers']}"
    )
    print("W&B initialized")
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    criterion = torch.nn.MSELoss()
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    print(f"\n{'='*60}")
    print(f"Training for {config['epochs']} epochs")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_scenarios, optimizer, criterion, device, config, epoch)
        
        if epoch % config['val_frequency'] == 0:
            val_loss, val_mae = validate(model, val_scenarios, criterion, device, config)
            train_eval_loss, train_eval_mae = validate(model, train_scenarios, criterion, device, config)
            
            model.train() # Go back to training mode after validation

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "train_eval_loss": train_eval_loss,
                "train_eval_mae": train_eval_mae,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            print(f"Epoch {epoch}/{config['epochs']} | Train Loss: {train_loss:.4f} | Train Eval Loss: {train_eval_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_hetero_model.pt')
                print(f"  -> Saved new best model with val_loss: {best_val_loss:.4f}")
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            print(f"Epoch {epoch}/{config['epochs']} | Train Loss: {train_loss:.4f}")
    
    print(f"{'='*60}")
    print("Training complete")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")
    
    print("\nTesting on test set...")
    model.load_state_dict(torch.load('best_hetero_model.pt'))
    test_results = test(model, test_scenarios, criterion, device, config)
    
    print(f"\nTest Results (across {test_results['num_scenarios_tested']} scenarios):")
    print(f"  MSE: {test_results['mse']:.4f}")
    print(f"  MAE: {test_results['mae']:.4f}")
    print(f"  ADE: {test_results['ade']:.4f} (normalized)")
    print(f"  FDE: {test_results['fde']:.4f} (normalized)")
    
    test_log = {
        "test_mse": test_results['mse'],
        "test_mae": test_results['mae'],
        "test_ade": test_results['ade'],
        "test_fde": test_results['fde'],
        "test_num_scenarios": test_results['num_scenarios_tested']
    }
    
    if len(test_results['timestep_errors']) > 0:
        print(f"\nError by timestep (normalized):")
        for t, error in enumerate(test_results['timestep_errors']):
            denorm_error = error * 10.0  # Denormalize to meters
            print(f"  t+{t+1}: {error:.4f} (norm) = {denorm_error:.2f} meters")
            test_log[f"test_error_timestep_{t+1}"] = denorm_error
    
    if len(test_results['per_scenario_metrics']) > 0:
        scenario_ades = [m['ade'] for m in test_results['per_scenario_metrics']]
        print(f"\nPer-scenario ADE statistics:")
        print(f"  Min: {min(scenario_ades):.4f}, Max: {max(scenario_ades):.4f}, Std: {np.std(scenario_ades):.4f}")
        
        test_log["test_ade_min"] = min(scenario_ades)
        test_log["test_ade_max"] = max(scenario_ades)
        test_log["test_ade_std"] = np.std(scenario_ades)
        
        scenario_table = wandb.Table(
            columns=["scenario_idx", "mse", "mae", "ade", "num_agents"],
            data=[[m['scenario_idx'], m['mse'], m['mae'], m['ade'], m['num_agents']] 
                  for m in test_results['per_scenario_metrics']]
        )
        test_log["test_per_scenario_metrics"] = scenario_table
    
    # Log visualization images to W&B
    viz_dir = './test_visualizations'
    if os.path.exists(viz_dir):
        viz_images = []
        for viz_file in sorted(os.listdir(viz_dir)):
            if viz_file.endswith('.png'):
                viz_path = os.path.join(viz_dir, viz_file)
                viz_images.append(wandb.Image(viz_path, caption=viz_file))
        
        if viz_images:
            test_log["test_visualizations"] = viz_images
            print(f"\nLogged {len(viz_images)} visualization images to W&B")
    
    wandb.log(test_log, step=config['epochs'] - 1)
    
    print(f"\n{'='*60}")
    wandb.finish()
    print("W&B run finished")


if __name__ == "__main__":
    main()
