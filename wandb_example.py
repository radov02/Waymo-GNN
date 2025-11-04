import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset, DataLoader, HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
import wandb
import tensorflow as tf
import numpy as np
from EvolveGCNH import EvolveGCNH

"""
GRAPH REPRESENTATION:
This file now supports BOTH homogeneous and heterogeneous graph representations:

1. HOMOGENEOUS GRAPHS (Currently used for training):
   - Function: scenario_to_graph(), scenario_to_temporal_graphs()
   - Simple agent-only representation with spatial proximity edges
   - Used by train_epoch(), validate(), test() functions
   - Works with current EvolveGCN-H model

2. HETEROGENEOUS GRAPHS (Demonstrated in main()):
   - Function: scenario_to_heterogeneous_graph()
   - Rich representation with multiple node and edge types:
     * Node types: agent, lane_segment, traffic_control
     * Edge types: 8 different relationships (agent-agent, agent-lane, lane-lane, etc.)
     * Features: Frenet coordinates, lane topology, traffic signals
   - Requires heterogeneous GNN model (e.g., HeteroGNN, HGT) for training
   
The main() function demonstrates the heterogeneous graph structure but still 
trains using homogeneous graphs. To fully utilize heterogeneous graphs, you'll 
need to implement a heterogeneous GNN model architecture.
"""

# Add local Waymo module path
src_path = os.path.abspath(os.path.join(os.getcwd(), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from waymo_open_dataset.protos import scenario_pb2

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

def initial_feature_vector(agent, state_index):
    """Create feature vector for an agent at a specific timestep with normalization."""
    state = agent.states[state_index]
    
    properties = [
        state.center_x / 100.0,
        state.center_y / 100.0,
        state.velocity_x / 10.0,
        state.velocity_y / 10.0,
        float(state.valid)
    ]
    
    type_onehot = [
        1 if agent.object_type == 1 else 0,
        1 if agent.object_type == 2 else 0,
        1 if agent.object_type == 3 else 0,
        1 if agent.object_type == 4 else 0
    ]
    
    return torch.tensor(properties + type_onehot, dtype=torch.float32)

def build_edge_index_radius(positions, radius=30.0, valid_mask=None):
    """Build graph edges based on spatial proximity."""
    pairwise_distances = torch.cdist(positions, positions)
    
    if valid_mask is not None:
        vm = torch.as_tensor(valid_mask, dtype=torch.bool)
        valid_pair = vm[:, None] & vm[None, :]
        pairwise_distances = pairwise_distances.clone()
        pairwise_distances[~valid_pair] = float('inf')
    
    # Remove self-loops
    pairwise_distances.fill_diagonal_(float('inf'))
    
    # Create edges for agents within radius
    edges_mask = pairwise_distances <= radius
    src, dst = torch.where(edges_mask)
    edge_index = torch.stack([src, dst], dim=0)
    
    return edge_index

# ============================================================================
# HOMOGENEOUS GRAPH FUNCTIONS (Still used by train/val/test functions)
# Note: These are simpler agent-only graphs. The heterogeneous graph function
# below provides richer representation but requires model architecture updates.
# ============================================================================

def scenario_to_graph(scenario, timestep, radius=30.0, future_steps=1):
    """Convert a Waymo scenario at a specific timestep to PyG Data."""
    node_features = []
    positions = []
    agent_ids = []
    valid_mask = []
    
    for agent in scenario.tracks:
        if timestep >= len(agent.states):
            continue
            
        state = agent.states[timestep]
        if not state.valid:
            continue
        
        node_features.append(initial_feature_vector(agent, timestep))
        positions.append([state.center_x, state.center_y])
        agent_ids.append(agent.id)
        valid_mask.append(1)
    
    if len(node_features) == 0:
        return None
    
    x = torch.stack(node_features)
    pos = torch.tensor(positions, dtype=torch.float32)
    edge_index = build_edge_index_radius(pos, radius, valid_mask)
    
    labels = []
    id_to_agent = {t.id: t for t in scenario.tracks}
    for i, agent_id in enumerate(agent_ids):
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
        
        current_pos = torch.tensor(positions[i], dtype=torch.float32)
        future_tensor = torch.tensor(future_pos, dtype=torch.float32)
        offsets = (future_tensor - current_pos) / 10.0
        labels.append(offsets.flatten())
    
    y = torch.stack(labels)
    
    data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    data.agent_ids = agent_ids
    data.scenario_id = scenario.scenario_id
    data.timestep = timestep
    
    return data


def scenario_to_temporal_graphs(scenario, start_timestep, end_timestep, radius=30.0, future_steps=1):
    """Convert a scenario to a sequence of temporal graphs for EvolveGCN."""
    graphs = []
    for t in range(start_timestep, end_timestep):
        graph = scenario_to_graph(scenario, t, radius, future_steps)
        if graph is not None:
            graphs.append(graph)
    return graphs


# ============================================================================
# HETEROGENEOUS GRAPH FUNCTION (New, richer representation)
# ============================================================================


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
    
    # Compute lane heading at closest point
    if closest_idx < len(lane_polyline) - 1:
        dx = lane_polyline[closest_idx + 1, 0] - lane_polyline[closest_idx, 0]
        dy = lane_polyline[closest_idx + 1, 1] - lane_polyline[closest_idx, 1]
        lane_heading = np.arctan2(dy, dx)
    elif closest_idx > 0:
        dx = lane_polyline[closest_idx, 0] - lane_polyline[closest_idx - 1, 0]
        dy = lane_polyline[closest_idx, 1] - lane_polyline[closest_idx - 1, 1]
        lane_heading = np.arctan2(dy, dx)
    else:
        lane_heading = 0.0
    
    heading_diff = agent_heading - lane_heading
    # Normalize to [-pi, pi]
    heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
    
    return s, d, heading_diff


def scenario_to_heterogeneous_graph(scenario, timestep, interaction_radius=30.0):
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
        
        # Agent features: [x, y, vx, vy, heading, length, width]
        features = [
            state.center_x / 100.0,  # Normalize position
            state.center_y / 100.0,
            state.velocity_x / 10.0,  # Normalize velocity
            state.velocity_y / 10.0,
            state.heading,  # Keep heading as is
            state.length / 10.0,  # Normalize dimensions
            state.width / 10.0
        ]
        
        agent_features.append(features)
        agent_positions.append([state.center_x, state.center_y])
        agent_ids_list.append(agent.id)
        agent_headings.append(state.heading)
    
    if len(agent_features) > 0:
        hetero_data['agent'].x = torch.tensor(agent_features, dtype=torch.float32)
        hetero_data['agent'].pos = torch.tensor(agent_positions, dtype=torch.float32)
        hetero_data['agent'].agent_ids = agent_ids_list
    else:
        # No valid agents at this timestep
        return None
    
    # ========================================================================
    # 2. Build Lane Segment Nodes (LS)
    # ========================================================================
    lane_features = []
    lane_polylines = []  # Store for Frenet computation
    lane_ids_list = []
    
    # Lane type encoding (one-hot or similar)
    lane_type_map = {
        0: [1, 0, 0],  # TYPE_UNDEFINED
        1: [0, 1, 0],  # TYPE_FREEWAY
        2: [0, 0, 1],  # TYPE_SURFACE_STREET
        # Add more as needed
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
            
            # Compute lane features: flattened polyline + lane type encoding
            # For simplicity, use start/end points and center
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
    # 3. Build Traffic Control Nodes (TC)
    # ========================================================================
    tc_features = []
    tc_positions = []
    tc_ids_list = []
    
    for map_feature in scenario.map_features:
        # Traffic signals
        if map_feature.HasField('stop_sign'):
            stop_sign = map_feature.stop_sign
            pos = [stop_sign.position.x / 100.0, stop_sign.position.y / 100.0]
            tc_features.append(pos)
            tc_positions.append([stop_sign.position.x, stop_sign.position.y])
            tc_ids_list.append(map_feature.id)
        
        # Crosswalks
        elif map_feature.HasField('crosswalk'):
            crosswalk = map_feature.crosswalk
            if len(crosswalk.polygon) > 0:
                # Use center of crosswalk polygon
                center_x = np.mean([p.x for p in crosswalk.polygon]) / 100.0
                center_y = np.mean([p.y for p in crosswalk.polygon]) / 100.0
                tc_features.append([center_x, center_y])
                tc_positions.append([center_x * 100.0, center_y * 100.0])
                tc_ids_list.append(map_feature.id)
    
    # Dynamic states (traffic lights) - these change per timestep
    for dynamic_state in scenario.dynamic_map_states:
        if timestep < len(dynamic_state.lane_states):
            lane_state = dynamic_state.lane_states[timestep]
            # Traffic light position from map features
            for map_feature in scenario.map_features:
                if map_feature.id == lane_state.lane and map_feature.HasField('lane'):
                    # Use lane end position as traffic light location
                    lane = map_feature.lane
                    if len(lane.polyline) > 0:
                        pos = lane.polyline[-1]
                        tc_features.append([pos.x / 100.0, pos.y / 100.0])
                        tc_positions.append([pos.x, pos.y])
                        tc_ids_list.append(map_feature.id)
                        break
    
    if len(tc_features) > 0:
        hetero_data['traffic_control'].x = torch.tensor(tc_features, dtype=torch.float32)
        hetero_data['traffic_control'].pos = torch.tensor(tc_positions, dtype=torch.float32)
        hetero_data['traffic_control'].tc_ids = tc_ids_list
    
    # ========================================================================
    # 4. Build Edges: Lane Segment Topology
    # ========================================================================
    # Build lane ID to index mapping
    lane_id_to_idx = {lane_id: idx for idx, lane_id in enumerate(lane_ids_list)}
    
    # LS -> LS edges (succeeds, precedes, left/right adjacent)
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
            
            # Entry lanes (precedes)
            for entry_id in lane.entry_lanes:
                if entry_id in lane_id_to_idx:
                    entry_idx = lane_id_to_idx[entry_id]
                    ls_precedes_ls_src.append(entry_idx)
                    ls_precedes_ls_dst.append(src_idx)
            
            # Exit lanes (succeeds)
            for exit_id in lane.exit_lanes:
                if exit_id in lane_id_to_idx:
                    exit_idx = lane_id_to_idx[exit_id]
                    ls_succeeds_ls_src.append(src_idx)
                    ls_succeeds_ls_dst.append(exit_idx)
            
            # Left neighbors
            for left_id in lane.left_neighbors:
                if left_id.feature_id in lane_id_to_idx:
                    left_idx = lane_id_to_idx[left_id.feature_id]
                    ls_left_adj_src.append(src_idx)
                    ls_left_adj_dst.append(left_idx)
            
            # Right neighbors
            for right_id in lane.right_neighbors:
                if right_id.feature_id in lane_id_to_idx:
                    right_idx = lane_id_to_idx[right_id.feature_id]
                    ls_right_adj_src.append(src_idx)
                    ls_right_adj_dst.append(right_idx)
    
    # Add lane topology edges
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
    # 5. Build Edges: Agent-Lane (A on_lane LS)
    # ========================================================================
    a_onlane_ls_src, a_onlane_ls_dst = [], []
    a_onlane_ls_edge_attr = []
    
    for agent_idx, (agent_pos, agent_heading) in enumerate(zip(agent_positions, agent_headings)):
        # Find nearest lane segments (within reasonable distance)
        for lane_idx, polyline in enumerate(lane_polylines):
            # Compute distance to lane
            distances = np.sqrt((polyline[:, 0] - agent_pos[0])**2 + 
                              (polyline[:, 1] - agent_pos[1])**2)
            min_dist = np.min(distances)
            
            # If agent is close to this lane (within 5 meters)
            if min_dist < 5.0:
                # Compute Frenet coordinates
                s, d, heading_diff = compute_frenet_coordinates(
                    agent_pos[0], agent_pos[1], agent_heading, polyline)
                
                a_onlane_ls_src.append(agent_idx)
                a_onlane_ls_dst.append(lane_idx)
                # Edge features: [frenet_s, frenet_d, heading_diff]
                a_onlane_ls_edge_attr.append([s / 100.0, d / 10.0, heading_diff])
    
    if len(a_onlane_ls_src) > 0:
        hetero_data['agent', 'on_lane', 'lane_segment'].edge_index = torch.tensor(
            [a_onlane_ls_src, a_onlane_ls_dst], dtype=torch.long)
        hetero_data['agent', 'on_lane', 'lane_segment'].edge_attr = torch.tensor(
            a_onlane_ls_edge_attr, dtype=torch.float32)
    
    # ========================================================================
    # 6. Build Edges: Lane-Traffic Control (LS controlled_by TC)
    # ========================================================================
    if len(tc_ids_list) > 0 and len(lane_ids_list) > 0:
        tc_id_to_idx = {tc_id: idx for idx, tc_id in enumerate(tc_ids_list)}
        ls_controlled_tc_src, ls_controlled_tc_dst = [], []
        
        # Map traffic lights to lanes
        for dynamic_state in scenario.dynamic_map_states:
            if timestep < len(dynamic_state.lane_states):
                lane_state = dynamic_state.lane_states[timestep]
                if lane_state.lane in lane_id_to_idx and lane_state.lane in tc_id_to_idx:
                    lane_idx = lane_id_to_idx[lane_state.lane]
                    tc_idx = tc_id_to_idx[lane_state.lane]
                    ls_controlled_tc_src.append(lane_idx)
                    ls_controlled_tc_dst.append(tc_idx)
        
        if len(ls_controlled_tc_src) > 0:
            hetero_data['lane_segment', 'controlled_by', 'traffic_control'].edge_index = torch.tensor(
                [ls_controlled_tc_src, ls_controlled_tc_dst], dtype=torch.long)
    
    # ========================================================================
    # 7. Build Edges: Agent-Traffic Control (A observing TC)
    # ========================================================================
    if len(tc_positions) > 0:
        a_observing_tc_src, a_observing_tc_dst = [], []
        
        for agent_idx, agent_pos in enumerate(agent_positions):
            for tc_idx, tc_pos in enumerate(tc_positions):
                # If traffic control is in agent's observation range (e.g., 50m)
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
            
            # Compute distance
            delta_x = agent_positions_np[j, 0] - agent_positions_np[i, 0]
            delta_y = agent_positions_np[j, 1] - agent_positions_np[i, 1]
            dist = np.sqrt(delta_x**2 + delta_y**2)
            
            # If within interaction radius
            if dist < interaction_radius:
                a_interact_a_src.append(i)
                a_interact_a_dst.append(j)
                # Edge features: [delta_x, delta_y] (normalized)
                a_interact_a_edge_attr.append([delta_x / 100.0, delta_y / 100.0])
    
    if len(a_interact_a_src) > 0:
        hetero_data['agent', 'interacting_with', 'agent'].edge_index = torch.tensor(
            [a_interact_a_src, a_interact_a_dst], dtype=torch.long)
        hetero_data['agent', 'interacting_with', 'agent'].edge_attr = torch.tensor(
            a_interact_a_edge_attr, dtype=torch.float32)
    
    # ========================================================================
    # Store metadata
    # ========================================================================
    hetero_data.scenario_id = scenario.scenario_id
    hetero_data.timestep = timestep
    
    return hetero_data


# ============================================================================
# Data Download Utilities
# ============================================================================
def get_gs_filename(dataset, i, dataset_type="scenario"):
    """
    Generate Google Cloud Storage filename for Waymo dataset.
    
    Args:
        dataset: "training", "validation", or "testing"
        i: file index
        dataset_type: "scenario" or "tf_example"
    """
    if dataset_type == "scenario":
        template = "uncompressed/scenario/{dataset}/uncompressed_scenario_{dataset}_{dataset}.tfrecord-{i}-of-{length}"
    else:
        template = "uncompressed/tf_example/{dataset}/{dataset}_tfexample.tfrecord-{i}-of-{length}"
    
    if dataset == "training":
        length = "01000" if dataset_type == "tf_example" else "00150"
    elif dataset == "validation" or dataset == "testing":
        length = "00150"
    else:
        raise ValueError("Invalid dataset name")
    
    return template.format(dataset=dataset, i=str(i).zfill(5), length=length)

def get_local_filename(dataset, i, dataset_type="scenario"):
    """Generate local filename for downloaded dataset."""
    return f"data/{dataset_type}/{dataset}/{str(i).zfill(5)}.tfrecord"

def download_waymo_data(num_files=1, dataset="training", dataset_type="scenario", 
                       project_id="waymo-gnn-475616", 
                       bucket_name="waymo_open_dataset_motion_v_1_3_0"):
    """
    Download Waymo dataset files from Google Cloud Storage.
    
    Args:
        num_files: Number of files to download
        dataset: "training", "validation", or "testing"
        dataset_type: "scenario" or "tf_example"
        project_id: GCP project ID
        bucket_name: GCS bucket name
    
    Requires:
        ADC_B64 environment variable with base64-encoded credentials
    """
    try:
        from google.cloud import storage
        import base64
        import json
        from google.oauth2.credentials import Credentials as UserCredentials
    except ImportError:
        print("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
        return
    
    adc_b64 = os.environ.get("ADC_B64")
    if not adc_b64:
        print("Missing ADC_B64 environment variable")
        print("  Run: gcloud auth application-default login")
        return
    
    try:
        adc_json = base64.b64decode(adc_b64).decode("utf-8")
        adc_info = json.loads(adc_json)
        credentials = UserCredentials.from_authorized_user_info(adc_info)
        storage_client = storage.Client(project=project_id, credentials=credentials)
    except Exception as e:
        print(f"Failed to authenticate: {e}")
        return
    
    bucket = storage_client.bucket(bucket_name, user_project=project_id)
    
    for i in range(num_files):
        gs_filename = get_gs_filename(dataset, i, dataset_type)
        blob = bucket.blob(gs_filename)
        
        local_filename = get_local_filename(dataset, i, dataset_type)
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        
        try:
            blob.download_to_filename(filename=local_filename)
            print("Downloaded:", local_filename)
        except Exception as e:
            print(f"Failed to download {local_filename}: {repr(e)}")

def create_tfrecord_indices(data_dir):
    """
    Create index files for TFRecord files to enable random access.
    
    Args:
        data_dir: Directory containing .tfrecord files
        
    Requires:
        tfrecord package (pip install tfrecord)
    """
    import subprocess
    try:
        subprocess.run(
            ["python", "-m", "tfrecord.tools.tfrecord2idx", data_dir],
            check=True
        )
        print(f"Created indices for {data_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create indices: {e}")
    except FileNotFoundError:
        print("tfrecord package not found. Install with: pip install tfrecord")



# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, train_scenarios, optimizer, criterion, device, config):
    """Train model for one epoch using temporal sequences."""
    model.train()
    total_loss = 0
    num_sequences = 0
    
    for scenario in train_scenarios:
        temporal_graphs = scenario_to_temporal_graphs(
            scenario, 
            start_timestep=0, 
            end_timestep=config['timestep'], 
            radius=config['radius'], 
            future_steps=config['future_steps']
        )
        
        if len(temporal_graphs) == 0:
            continue
        
        model.gru_h = None
        optimizer.zero_grad()
        sequence_loss = 0
        
        for graph in temporal_graphs:
            graph = graph.to(device)
            out = model(graph.x, graph.edge_index)
            loss = criterion(out, graph.y)
            sequence_loss += loss
        
        avg_sequence_loss = sequence_loss / len(temporal_graphs)
        avg_sequence_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += avg_sequence_loss.item()
        num_sequences += 1
    
    return total_loss / num_sequences if num_sequences > 0 else 0


def validate(model, val_scenarios, criterion, device, config):
    """Validate: use timesteps 0-9 to predict timestep 10."""
    model.eval()
    total_loss = 0
    total_mae = 0
    num_predictions = 0
    
    with torch.no_grad():
        for scenario in val_scenarios:
            history_graphs = scenario_to_temporal_graphs(
                scenario, 
                start_timestep=0, 
                end_timestep=config['timestep'], 
                radius=config['radius'], 
                future_steps=config['future_steps']
            )
            
            target_graph = scenario_to_graph(
                scenario,
                timestep=config['timestep'],
                radius=config['radius'],
                future_steps=config['future_steps']
            )
            
            if len(history_graphs) == 0 or target_graph is None:
                continue
            
            model.gru_h = None
            
            for graph in history_graphs:
                graph = graph.to(device)
                _ = model(graph.x, graph.edge_index)
            
            target_graph = target_graph.to(device)
            out = model(target_graph.x, target_graph.edge_index)
            
            loss = criterion(out, target_graph.y)
            mae = torch.nn.L1Loss()(out, target_graph.y)
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_predictions += 1
    
    avg_loss = total_loss / num_predictions if num_predictions > 0 else 0
    avg_mae = total_mae / num_predictions if num_predictions > 0 else 0
    
    return avg_loss, avg_mae


def test(model, test_scenarios, criterion, device, config):
    """Test: use timesteps 0-9 to predict timestep 10 with detailed metrics."""
    model.eval()
    total_mse = 0
    total_mae = 0
    all_predictions = []
    all_targets = []
    per_scenario_metrics = []
    num_predictions = 0
    
    with torch.no_grad():
        for idx, scenario in enumerate(test_scenarios):
            history_graphs = scenario_to_temporal_graphs(
                scenario, 
                start_timestep=0, 
                end_timestep=config['timestep'], 
                radius=config['radius'], 
                future_steps=config['future_steps']
            )
            
            target_graph = scenario_to_graph(
                scenario,
                timestep=config['timestep'],
                radius=config['radius'],
                future_steps=config['future_steps']
            )
            
            if len(history_graphs) == 0 or target_graph is None:
                continue
            
            model.gru_h = None
            
            for graph in history_graphs:
                graph = graph.to(device)
                _ = model(graph.x, graph.edge_index)
            
            target_graph = target_graph.to(device)
            out = model(target_graph.x, target_graph.edge_index)
            
            mse = criterion(out, target_graph.y)
            mae = torch.nn.L1Loss()(out, target_graph.y)
            
            scenario_preds = out.cpu()
            scenario_targets = target_graph.y.cpu()
            
            pred_final = scenario_preds.reshape(-1, config['future_steps'], 2)[:, -1, :]
            target_final = scenario_targets.reshape(-1, config['future_steps'], 2)[:, -1, :]
            scenario_ade = torch.mean(torch.sqrt(torch.sum((pred_final - target_final) ** 2, dim=1)))
            
            per_scenario_metrics.append({
                'scenario_idx': idx,
                'mse': mse.item(),
                'mae': mae.item(),
                'ade': scenario_ade.item(),
                'num_agents': target_graph.num_nodes
            })
            
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
        "learning_rate": 0.0001,
        "epochs": 30,
        "hidden_channels": 32,
        "dropout": 0.3,
        "dataset": "Waymo Open Motion Dataset",
        "architecture": "EvolveGCN-H",
        "radius": 30.0,
        "future_steps": 8,
        "timestep": 10,
        "batch_size": 32,
        "num_scenarios_per_split": 10,
        "topk": 30,
        "num_layers": 2,
        "download_data": False,
        "val_frequency": 5
    }
    
    print(f"Configuration: {config}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    # Download Data (Optional):    
    if config["download_data"]:
        print("Downloading Waymo Dataset from Google Cloud Storage")
        
        # Check if credentials are available
        if not os.environ.get("ADC_B64"):
            print("\n  ADC_B64 environment variable not set")
            print("Skipping download. Data will be loaded from local files.")
            print("\nTo enable download:")
            print("  1. Run: gcloud auth application-default login")
            print("  2. Encode credentials: [System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes(\"$env:APPDATA\\gcloud\\application_default_credentials.json\"))")
            print("  3. Set: $env:ADC_B64=\"<base64_string>\"")
        else:
            download_waymo_data(
                num_files=1,
                dataset="training",
                dataset_type="scenario"
            )
            
            # Create indices for efficient random access (optional, requires tfrecord package)
            try:
                create_tfrecord_indices("data/scenario/training")
            except Exception as e:
                print(f"  Could not create indices: {e}")
                print("Install tfrecord package for indexed access: pip install tfrecord")

    
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
    
    if len(train_scenarios) == 0:
        print("\nNo training scenarios found. Exiting.")
        return
    
    scenario = train_scenarios[0]
    
    agent_types = {}
    type_names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Other'}
    for track in scenario.tracks:
        type_name = type_names.get(track.object_type, 'Unknown')
        agent_types[type_name] = agent_types.get(type_name, 0) + 1
    
    print(f"\nFirst scenario - Agent types:")
    for agent_type, count in agent_types.items():
        print(f"  {agent_type}: {count}")
    
    print(f"Number of timesteps in first scenario: {len(scenario.timestamps_seconds)}")
    
    print(f"\nConverting to heterogeneous PyG graph at timestep {config['timestep']}...")
    hetero_graph_data = scenario_to_heterogeneous_graph(
        scenario, 
        timestep=config['timestep'],
        interaction_radius=config['radius']
    )
    
    if hetero_graph_data:
        print(f"\n{'='*60}")
        print("HETEROGENEOUS GRAPH STRUCTURE")
        print(f"{'='*60}")
        
        # Node information
        print(f"\nNode Types:")
        print(f"  • Agents: {hetero_graph_data['agent'].x.shape[0]} nodes")
        print(f"    - Features: {hetero_graph_data['agent'].x.shape[1]}D [x, y, vx, vy, heading, length, width]")
        
        if 'lane_segment' in hetero_graph_data.node_types:
            print(f"  • Lane Segments: {hetero_graph_data['lane_segment'].x.shape[0]} nodes")
            print(f"    - Features: {hetero_graph_data['lane_segment'].x.shape[1]}D [start, center, end, lane_type]")
        
        if 'traffic_control' in hetero_graph_data.node_types:
            print(f"  • Traffic Controls: {hetero_graph_data['traffic_control'].x.shape[0]} nodes")
            print(f"    - Features: {hetero_graph_data['traffic_control'].x.shape[1]}D [x, y]")
        
        # Edge information
        print(f"\nEdge Types ({len(hetero_graph_data.edge_types)} total):")
        for edge_type in hetero_graph_data.edge_types:
            src, rel, dst = edge_type
            edge_index = hetero_graph_data[edge_type].edge_index
            num_edges = edge_index.shape[1]
            print(f"  • {src} --[{rel}]--> {dst}: {num_edges} edges")
            
            # Show edge features if they exist
            if hasattr(hetero_graph_data[edge_type], 'edge_attr'):
                edge_attr = hetero_graph_data[edge_type].edge_attr
                print(f"    - Edge features: {edge_attr.shape[1]}D")
        
        print(f"{'='*60}\n")
    else:
        print("  No valid graph at this timestep")
        return
    
    print("="*60)
    

    # Use agent features for model input dimensions
    input_dim = hetero_graph_data['agent'].x.shape[1]
    # Output: predict 2D trajectory offsets for future_steps timesteps
    output_dim = config['future_steps'] * 2
    
    print(f"\nHeterogeneous Graph Model Configuration:")
    print(f"  Agent feature dimension: {input_dim}")
    print(f"  Output dimension (trajectory): {output_dim} ({config['future_steps']} timesteps × 2D)")
    
    model = EvolveGCNH(
        input_dim=input_dim,
        hidden_dim=config['hidden_channels'],
        output_dim=output_dim,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        topk=config['topk']
    ).to(device)
    
    print(f"\nModel initialized")
    print(f"  Input: {input_dim}, Hidden: {config['hidden_channels']}, Output: {output_dim}")
    print(f"  Layers: {config['num_layers']}, Top-k: {config['topk']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    

    wandb.login()
    wandb.init(
        project="waymo-trajectory-prediction",
        config=config,
        name=f"EvolveGCN-H_r{config['radius']}_h{config['hidden_channels']}_k{config['topk']}"
    )
    print("W&B initialized")
    

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()
    
    print(f"\n{'='*60}")
    print(f"Training for {config['epochs']} epochs")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_scenarios, optimizer, criterion, device, config)
        
        if (epoch + 1) % config['val_frequency'] == 0 or epoch == config['epochs'] - 1:
            val_loss, val_mae = validate(model, val_scenarios, criterion, device, config)
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | Train Loss: {train_loss:.1f} | Val Loss: {val_loss:.1f} | Val MAE: {val_mae:.1f}")
        else:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss
            })
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | Train Loss: {train_loss:.1f}")
    
    print(f"{'='*60}")
    print("Training complete")
    print(f"Best validation loss: {best_val_loss:.1f}")
    print(f"{'='*60}")
    
    print("\nFinal validation on best model...")
    model.load_state_dict(torch.load('best_model.pt'))
    final_val_loss, final_val_mae = validate(model, val_scenarios, criterion, device, config)
    print(f"Best model - Val Loss: {final_val_loss:.1f} | Val MAE: {final_val_mae:.1f}")
    
    print("\nTesting on test set...")
    test_results = test(model, test_scenarios, criterion, device, config)
    
    print(f"\nTest Results (across {test_results['num_scenarios_tested']} scenarios):")
    print(f"  MSE: {test_results['mse']:.1f}")
    print(f"  MAE: {test_results['mae']:.1f}")
    print(f"  ADE (avg across all timesteps): {test_results['ade']:.1f} meters")
    print(f"  FDE (final displacement error): {test_results['fde']:.1f} meters")
    
    # Build test log dict
    test_log = {
        "test_mse": test_results['mse'],
        "test_mae": test_results['mae'],
        "test_ade": test_results['ade'],
        "test_fde": test_results['fde'],
        "test_num_scenarios": test_results['num_scenarios_tested']
    }
    
    if len(test_results['timestep_errors']) > 0:
        print(f"\nError by timestep (0.1s intervals):")
        for t, error in enumerate(test_results['timestep_errors']):
            denorm_error = error * 10.0
            print(f"  t+{t+1}: {denorm_error:.2f} meters")
            test_log[f"test_error_timestep_{t+1}"] = denorm_error
    
    if len(test_results['per_scenario_metrics']) > 0:
        scenario_ades = [m['ade'] for m in test_results['per_scenario_metrics']]
        scenario_maes = [m['mae'] for m in test_results['per_scenario_metrics']]
        
        print(f"\nPer-scenario statistics:")
        print(f"  ADE - Min: {min(scenario_ades):.1f}, Max: {max(scenario_ades):.1f}, Std: {np.std(scenario_ades):.1f}")
        print(f"  MAE - Min: {min(scenario_maes):.1f}, Max: {max(scenario_maes):.1f}, Std: {np.std(scenario_maes):.1f}")
        
        test_log["test_ade_min"] = min(scenario_ades)
        test_log["test_ade_max"] = max(scenario_ades)
        test_log["test_ade_std"] = np.std(scenario_ades)
        
        scenario_table = wandb.Table(
            columns=["scenario_idx", "mse", "mae", "ade", "num_agents"],
            data=[[m['scenario_idx'], m['mse'], m['mae'], m['ade'], m['num_agents']] 
                  for m in test_results['per_scenario_metrics']]
        )
        test_log["test_per_scenario_metrics"] = scenario_table
    
    # Log test results with explicit step to match training timeline
    wandb.log(test_log, step=config['epochs'] - 1)
    
    if len(test_scenarios) > 0 and 'predictions' in test_results:
        predictions = test_results['predictions']
        targets = test_results['targets']
        
        print(f"\nSample predictions from test set (first scenario, normalized units):")
        num_samples = min(3, predictions.shape[0])
        
        for i in range(num_samples):
            pred = predictions[i].cpu().numpy()
            true = targets[i].cpu().numpy()
            
            pred_traj = pred.reshape(-1, 2)
            true_traj = true.reshape(-1, 2)
            
            pred_denorm = pred_traj[-1] * 10.0
            true_denorm = true_traj[-1] * 10.0
            
            print(f"  Agent {i}:")
            print(f"    Predicted offset: ({pred_denorm[0]:.2f}, {pred_denorm[1]:.2f}) m")
            print(f"    True offset:      ({true_denorm[0]:.2f}, {true_denorm[1]:.2f}) m")
            print(f"    Error: {np.linalg.norm(pred_denorm - true_denorm):.2f} m")
    
    print(f"\n{'='*60}")
    
    wandb.finish()
    print("\nW&B run finished")
    print("View results at: https://wandb.ai")


if __name__ == "__main__":
    main()
