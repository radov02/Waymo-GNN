
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from datetime import datetime
from helper_functions.graph_creation_functions import timestep_to_pyg_data
from config import sequence_length, radius, graph_creation_method


def load_scenario_for_visualization(scenario_id=None):
    """
    Load a Waymo scenario for visualization purposes.
    
    Args:
        scenario_id: Optional scenario ID. If None, loads first available scenario.
    
    Returns:
        scenario object or None
    """
    try:
        from helper_functions.graph_creation_functions import get_data_files, parse_scenario_file
        
        # Try to load from training data
        training_files = get_data_files(".\\data\\scenario\\training")
        if not training_files:
            print("Warning: No training scenario files found")
            return None
        
        # Load first file
        scenarios = parse_scenario_file(training_files[0])
        if scenarios:
            if scenario_id is not None:
                # Find specific scenario
                for scenario in scenarios:
                    if scenario.scenario_id == scenario_id:
                        return scenario
            # Return first scenario
            return scenarios[0]
        return None
    except Exception as e:
        print(f"Warning: Could not load scenario for visualization: {e}")
        return None


def plot_lane_polylines(scenario, figsize=(14, 12), show_lane_ids=False, show_lane_types=True):
    """
    Plot only lane polylines from a Waymo scenario.
    
    Args:
        scenario: Waymo scenario object
        figsize: Figure size tuple (width, height)
        show_lane_ids: Whether to show lane IDs on the plot
        show_lane_types: Whether to color-code lanes by type
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Lane type names and colors
    lane_type_names = {
        0: 'UNDEFINED',
        1: 'FREEWAY',
        2: 'SURFACE_STREET',
        3: 'BIKE_LANE'
    }
    
    lane_type_colors = {
        0: '#95a5a6',  # Gray for undefined
        1: '#e74c3c',  # Red for freeway
        2: '#3498db',  # Blue for surface street
        3: '#2ecc71'   # Green for bike lane
    }
    
    lane_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print(f"Plotting lane polylines for scenario: {scenario.scenario_id}")
    
    # Plot lane polylines
    for feature in scenario.map_features:
        feature_type = feature.WhichOneof('feature_data')
        
        if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
            lane_type = feature.lane.type
            lane_counts[lane_type] += 1
            
            # Extract coordinates
            x_coords = [point.x for point in feature.lane.polyline]
            y_coords = [point.y for point in feature.lane.polyline]
            
            # Choose color
            if show_lane_types:
                color = lane_type_colors.get(lane_type, '#95a5a6')
            else:
                color = '#3498db'  # Default blue
            
            # Plot the lane
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
            
            # Optionally show lane ID at the midpoint
            if show_lane_ids and len(x_coords) > 0:
                mid_idx = len(x_coords) // 2
                ax.annotate(
                    str(feature.id),
                    xy=(x_coords[mid_idx], y_coords[mid_idx]),
                    fontsize=8,
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
    
    # Configure plot
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Lane Polylines - Scenario: {scenario.scenario_id}', 
                 fontsize=14, fontweight='bold')
    
    # Create legend if showing lane types
    if show_lane_types:
        for lane_type, count in lane_counts.items():
            if count > 0:
                type_name = lane_type_names.get(lane_type, f'Type_{lane_type}')
                color = lane_type_colors.get(lane_type, '#95a5a6')
                ax.plot([], [], color=color, linewidth=2, alpha=0.7, 
                       label=f'{type_name} ({count})')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Print summary
    total_lanes = sum(lane_counts.values())
    print(f"Total lanes plotted: {total_lanes}")
    for lane_type, count in lane_counts.items():
        if count > 0:
            type_name = lane_type_names.get(lane_type, f'Type_{lane_type}')
            print(f"  - {type_name}: {count}")
    
    return fig, ax


# Example usage (commented out - uncomment when you have training_dataset loaded):
# fig, ax = plot_lane_polylines(
#     training_dataset[0][0],
#     figsize=(14, 12),
#     show_lane_ids=False,
#     show_lane_types=True
# )




# VISUALIZE SCENARIO WITH GRAPH EDGES
import matplotlib.pyplot as plt

def show_distances(pos, src_idx, dst_idx, x_coords, y_coords, ax):
    dist = np.sqrt((pos[src_idx, 0] - pos[dst_idx, 0])**2 + (pos[src_idx, 1] - pos[dst_idx, 1])**2)
    mid_x = (x_coords[0] + x_coords[1]) / 2
    mid_y = (y_coords[0] + y_coords[1]) / 2
    ax.annotate(f'{dist:.1f}m', xy=(mid_x, mid_y), fontsize=6, color='blue', ha='center', alpha=0.6)

def visualize_scenario_with_graph(scenario, timestep, radius, figsize=(14, 12), show_future=True, show_edge_distances=False, show_scenario_analysis=False):

    if show_scenario_analysis:
        analyze_scenario(scenario)
        analyze_scenario_agents(scenario)

    graph_data = timestep_to_pyg_data(scenario, timestep, radius, future_states=10, method='star')
    if graph_data is None:
        print("No valid agents at this timestep!")
        return None, None, None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = {1: '#e74c3c', 2: '#3498db', 3: '#f39c12', 4: '#9b59b6'}
    type_names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Other'}
    print(f"VISUALIZING SCENARIO WITH GRAPH: {scenario.scenario_id}")
    print(f" Time: {scenario.timestamps_seconds[timestep]:.1f}s")
    print(f" Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    render_map_features(scenario, ax)

    print(f" Rendering {graph_data.num_edges} graph edges...")
    edge_index = graph_data.edge_index
    pos = graph_data.pos.numpy()
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        x_coords = [pos[src_idx, 0], pos[dst_idx, 0]]
        y_coords = [pos[src_idx, 1], pos[dst_idx, 1]]
        ax.plot(x_coords, y_coords, 'blue', linewidth=1.2, alpha=0.4, zorder=5)
        if show_edge_distances:
            show_distances(pos, src_idx, dst_idx, x_coords, y_coords, ax)

    render_agents(scenario, timestep, colors, ax, show_future=show_future)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Waymo Scenario with Graph Edges (radius={radius}m)\n'
                f'Scenario: {scenario.scenario_id}, Time: {scenario.timestamps_seconds[timestep]:.1f}s\n'
                f'{graph_data.num_nodes} nodes, {graph_data.num_edges} edges', 
                fontsize=14, fontweight='bold')
    all_x = [state.center_x for track in scenario.tracks for state in track.states if state.valid]
    all_y = [state.center_y for track in scenario.tracks for state in track.states if state.valid]
    if all_x and all_y:
        margin = 20
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    for obj_type, color in colors.items():
        if obj_type in [t.object_type for t in scenario.tracks]:
            ax.scatter([], [], c=color, s=50, label=type_names[obj_type])
    ax.plot([], [], 'blue', linewidth=2, alpha=0.4, label=f'Graph edges (≤{radius}m)')
    ax.legend()
    plt.tight_layout()

    agent_counts = {}
    for track in scenario.tracks:
        if timestep < len(track.states) and track.states[timestep].valid:
            agent_type = type_names.get(track.object_type, 'Other')
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
    print(" Agents in graph:")
    for agent_type, count in agent_counts.items():
        print(f"   - {count} {agent_type}{'s' if count != 1 else ''}")
    if graph_data.num_edges > 0:
        avg_degree = graph_data.num_edges / graph_data.num_nodes
        print(f" Average degree: {avg_degree:.2f} edges per node")
    
    return fig, ax, graph_data


# TEST: Visualize scenario with graph edges at different radii
# time = 0
# fig1, ax1, graph1 = visualize_scenario_with_graph(training_dataset[0][0], timestep=time, radius=30.0, show_future=True)


def visualize_predictions_vs_actual(model, batch_dict, epoch, save_dir='training_visualizations', 
                                     device='cpu', max_nodes_to_plot=50, num_timesteps_to_show=None):
    """
    Visualize model predictions vs actual positions across a temporal sequence.
    Creates a .png file showing trajectories at different timesteps.
    
    Args:
        model: The trained EvolveGCN model
        batch_dict: Dictionary with 'batch' (list of Batch objects), 'B', 'T'
        epoch: Current epoch number (for filename)
        save_dir: Directory to save visualization files
        device: Device to run inference on
        max_nodes_to_plot: Maximum number of nodes to visualize (to avoid clutter)
        num_timesteps_to_show: Number of timesteps to visualize (None = all)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    batched_graph_sequence = batch_dict["batch"]
    T = batch_dict["T"]
    
    if num_timesteps_to_show is None:
        num_timesteps_to_show = T
    else:
        num_timesteps_to_show = min(num_timesteps_to_show, T)
    
    # Reset model hidden states
    model.reset_gru_hidden_states()
    
    # Move graphs to device
    for t in range(T):
        batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
    
    # Collect actual and predicted positions
    actual_positions = []  # List of [num_nodes, 2] tensors
    predicted_positions = []  # List of [num_nodes, 2] tensors
    
    with torch.no_grad():
        for t in range(T):
            batched_graph = batched_graph_sequence[t]
            
            # Get current positions (from node features or pos attribute)
            if hasattr(batched_graph, 'pos') and batched_graph.pos is not None:
                current_pos = batched_graph.pos.cpu()
            else:
                # Extract x, y from features (assuming first 2 features are positions)
                current_pos = batched_graph.x[:, :2].cpu()
            
            actual_positions.append(current_pos)
            
            # Get predictions
            predictions = model(batched_graph.x, batched_graph.edge_index)
            predicted_positions.append(predictions.cpu())
    
    # Select subset of nodes to visualize
    num_nodes = actual_positions[0].shape[0]
    if num_nodes > max_nodes_to_plot:
        # Sample nodes evenly
        indices = np.linspace(0, num_nodes - 1, max_nodes_to_plot, dtype=int)
    else:
        indices = np.arange(num_nodes)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left plot: Actual trajectories
    ax_actual = axes[0]
    # Right plot: Predicted trajectories
    ax_pred = axes[1]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))
    
    for idx, node_idx in enumerate(indices):
        color = colors[idx]
        
        # Extract trajectory for this node across timesteps
        actual_traj = np.array([actual_positions[t][node_idx].numpy() for t in range(num_timesteps_to_show)])
        pred_traj = np.array([predicted_positions[t][node_idx].numpy() for t in range(num_timesteps_to_show)])
        
        # Plot actual trajectory
        ax_actual.plot(actual_traj[:, 0], actual_traj[:, 1], 'o-', color=color, 
                      alpha=0.6, linewidth=1.5, markersize=4, label=f'Node {node_idx}' if idx < 10 else None)
        ax_actual.scatter(actual_traj[0, 0], actual_traj[0, 1], color=color, s=100, 
                         marker='s', edgecolors='black', linewidths=2, zorder=10)
        
        # Plot predicted trajectory
        ax_pred.plot(pred_traj[:, 0], pred_traj[:, 1], 'o-', color=color, 
                    alpha=0.6, linewidth=1.5, markersize=4, label=f'Node {node_idx}' if idx < 10 else None)
        ax_pred.scatter(pred_traj[0, 0], pred_traj[0, 1], color=color, s=100, 
                       marker='s', edgecolors='black', linewidths=2, zorder=10)
    
    # Configure actual positions plot
    ax_actual.set_aspect('equal', adjustable='box')
    ax_actual.grid(True, alpha=0.3)
    ax_actual.set_xlabel('X Position (m)', fontsize=12)
    ax_actual.set_ylabel('Y Position (m)', fontsize=12)
    ax_actual.set_title(f'Actual Trajectories\n(First {num_timesteps_to_show} timesteps)', 
                       fontsize=14, fontweight='bold')
    if len(indices) <= 10:
        ax_actual.legend(loc='best', fontsize=8)
    
    # Configure predicted positions plot
    ax_pred.set_aspect('equal', adjustable='box')
    ax_pred.grid(True, alpha=0.3)
    ax_pred.set_xlabel('X Position (m)', fontsize=12)
    ax_pred.set_ylabel('Y Position (m)', fontsize=12)
    ax_pred.set_title(f'Predicted Trajectories\n(First {num_timesteps_to_show} timesteps)', 
                     fontsize=14, fontweight='bold')
    if len(indices) <= 10:
        ax_pred.legend(loc='best', fontsize=8)
    
    # Add epoch info
    fig.suptitle(f'Epoch {epoch} - Trajectory Comparison\n'
                f'{len(indices)} nodes shown out of {num_nodes} total', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'epoch_{epoch:03d}_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved visualization: {filepath}")
    
    model.train()
    return filepath


def visualize_predictions_overlay(model, batch_dict, epoch, save_dir='training_visualizations', 
                                   device='cpu', max_nodes_to_plot=30, num_timesteps_to_show=None):
    """
    Visualize model predictions overlaid with actual positions to show discrepancy.
    Creates a .png file with actual (solid line) and predicted (dashed line) trajectories.
    
    Args:
        model: The trained EvolveGCN model
        batch_dict: Dictionary with 'batch' (list of Batch objects), 'B', 'T'
        epoch: Current epoch number (for filename)
        save_dir: Directory to save visualization files
        device: Device to run inference on
        max_nodes_to_plot: Maximum number of nodes to visualize
        num_timesteps_to_show: Number of timesteps to visualize (None = all)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    batched_graph_sequence = batch_dict["batch"]
    T = batch_dict["T"]
    
    if num_timesteps_to_show is None:
        num_timesteps_to_show = T
    else:
        num_timesteps_to_show = min(num_timesteps_to_show, T)
    
    # Reset model hidden states
    model.reset_gru_hidden_states()
    
    # Move graphs to device
    for t in range(T):
        batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
    
    # Collect actual and predicted positions
    actual_positions = []
    predicted_positions = []
    
    with torch.no_grad():
        for t in range(T):
            batched_graph = batched_graph_sequence[t]
            
            # Get current positions
            if hasattr(batched_graph, 'pos') and batched_graph.pos is not None:
                current_pos = batched_graph.pos.cpu()
            else:
                current_pos = batched_graph.x[:, :2].cpu()
            
            actual_positions.append(current_pos)
            
            # Get predictions
            predictions = model(batched_graph.x, batched_graph.edge_index)
            predicted_positions.append(predictions.cpu())
    
    # Select subset of nodes
    num_nodes = actual_positions[0].shape[0]
    if num_nodes > max_nodes_to_plot:
        indices = np.linspace(0, num_nodes - 1, max_nodes_to_plot, dtype=int)
    else:
        indices = np.arange(num_nodes)
    
    # Create single overlay plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))
    
    for idx, node_idx in enumerate(indices):
        color = colors[idx]
        
        # Extract trajectories
        actual_traj = np.array([actual_positions[t][node_idx].numpy() for t in range(num_timesteps_to_show)])
        pred_traj = np.array([predicted_positions[t][node_idx].numpy() for t in range(num_timesteps_to_show)])
        
        # Plot actual trajectory (solid line)
        ax.plot(actual_traj[:, 0], actual_traj[:, 1], '-', color=color, 
               alpha=0.7, linewidth=2, label=f'Actual {node_idx}' if idx < 5 else None)
        
        # Plot predicted trajectory (dashed line)
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], '--', color=color, 
               alpha=0.7, linewidth=2, label=f'Pred {node_idx}' if idx < 5 else None)
        
        # Mark starting position
        ax.scatter(actual_traj[0, 0], actual_traj[0, 1], color=color, s=120, 
                  marker='s', edgecolors='black', linewidths=2, zorder=10)
        
        # Show error arrows at each timestep
        for t in range(0, num_timesteps_to_show, max(1, num_timesteps_to_show // 5)):
            ax.annotate('', xy=pred_traj[t], xytext=actual_traj[t],
                       arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.3))
    
    # Configure plot
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Epoch {epoch} - Actual vs Predicted Trajectories\n'
                f'Solid = Actual, Dashed = Predicted, Red arrows = Error\n'
                f'{len(indices)} nodes, {num_timesteps_to_show} timesteps', 
                fontsize=14, fontweight='bold')
    
    if len(indices) <= 5:
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'epoch_{epoch:03d}_overlay_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved overlay visualization: {filepath}")
    
    model.train()
    return filepath


def visualize_training_progress(model, batch_dict, epoch, scenario=None, save_dir='training_visualizations',
                                device='cpu', max_nodes_per_graph=10, show_timesteps=8):
    """
    Visualization showing actual vs predicted trajectories with map features.
    Creates a B × T grid where rows are graphs and columns are timesteps.
    
    Args:
        model: The trained model
        batch_dict: Batch dictionary with temporal graph sequence
        epoch: Current epoch number
        scenario: Optional Waymo scenario object for map features
        save_dir: Directory to save visualizations
        device: Device for inference
        max_nodes_per_graph: Maximum nodes to show per individual graph
        show_timesteps: Number of timesteps to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    batched_graph_sequence = batch_dict["batch"]
    B = batch_dict["B"]  # Number of graphs in batch
    T = min(show_timesteps, batch_dict["T"])
    
    # Reset hidden states
    model.reset_gru_hidden_states()
    
    # Move to device
    for t in range(len(batch_dict["batch"])):
        batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
    
    # Collect data for each timestep
    all_actual_pos = []
    all_pred_pos = []
    all_batch_indices = []
    
    with torch.no_grad():
        for t in range(T):
            graph = batched_graph_sequence[t]
            
            # Current positions
            if hasattr(graph, 'pos') and graph.pos is not None:
                curr = graph.pos.cpu()
            else:
                curr = graph.x[:, :2].cpu()
            all_actual_pos.append(curr)
            
            # Predictions
            pred = model(graph.x, graph.edge_index).cpu()
            all_pred_pos.append(pred)
            
            # Batch indices (which graph each node belongs to)
            all_batch_indices.append(graph.batch.cpu())
    
    # Create figure with B × T grid (rows = graphs, columns = timesteps)
    fig, axes = plt.subplots(B, T, figsize=(2.5*T, 2.5*B))
    
    # Make axes always 2D array
    if B == 1 and T == 1:
        axes = np.array([[axes]])
    elif B == 1:
        axes = axes.reshape(1, -1)
    elif T == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each graph in the batch
    total_error = 0
    error_counts = 0
    
    # Lane type colors for map features
    lane_type_colors = {
        0: '#95a5a6',  # Gray for undefined
        1: '#e74c3c',  # Red for freeway
        2: '#3498db',  # Blue for surface street
        3: '#2ecc71'   # Green for bike lane
    }
    
    # Process each graph (row) and timestep (column)
    for graph_idx in range(B):
        # Get nodes for this graph across all timesteps
        graph_data_per_timestep = []
        
        for t in range(T):
            batch_mask = all_batch_indices[t] == graph_idx
            if batch_mask.sum() > 0:
                actual_pos = all_actual_pos[t][batch_mask]
                pred_pos = all_pred_pos[t][batch_mask]
                graph_data_per_timestep.append({
                    'actual': actual_pos,
                    'pred': pred_pos,
                    'num_nodes': actual_pos.shape[0]
                })
            else:
                graph_data_per_timestep.append(None)
        
        # Determine consistent number of nodes for this graph
        valid_counts = [d['num_nodes'] for d in graph_data_per_timestep if d is not None]
        if not valid_counts:
            continue
        
        # Use minimum node count across all timesteps
        num_nodes = min(valid_counts)
        
        # Select subset of nodes to visualize
        if num_nodes > max_nodes_per_graph:
            node_indices = np.linspace(0, num_nodes - 1, max_nodes_per_graph, dtype=int)
        else:
            node_indices = np.arange(num_nodes)
        
        # Assign consistent colors to each node
        node_colors = plt.cm.tab20(np.linspace(0, 1, len(node_indices)))
        
        # Calculate axis limits for this graph sequence (entire row)
        all_x_coords = []
        all_y_coords = []
        for data in graph_data_per_timestep:
            if data is not None and data['num_nodes'] >= num_nodes:
                actual_pos = data['actual'][:num_nodes]
                pred_pos = data['pred'][:num_nodes]
                all_x_coords.extend(actual_pos[:, 0].tolist())
                all_y_coords.extend(actual_pos[:, 1].tolist())
                all_x_coords.extend(pred_pos[:, 0].tolist())
                all_y_coords.extend(pred_pos[:, 1].tolist())
        
        # Standardized axis limits with padding
        if all_x_coords and all_y_coords:
            x_min, x_max = min(all_x_coords), max(all_x_coords)
            y_min, y_max = min(all_y_coords), max(all_y_coords)
            x_padding = (x_max - x_min) * 0.1 or 10  # 10% padding or 10m minimum
            y_padding = (y_max - y_min) * 0.1 or 10
            x_lim = (x_min - x_padding, x_max + x_padding)
            y_lim = (y_min - y_padding, y_max + y_padding)
        else:
            x_lim, y_lim = None, None
        
        # Plot each timestep for this graph
        for t in range(T):
            ax = axes[graph_idx, t]
            
            # Plot map features (only once per subplot)
            if scenario is not None:
                for feature in scenario.map_features:
                    feature_type = feature.WhichOneof('feature_data')
                    
                    if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
                        lane_type = feature.lane.type
                        x_coords = [point.x for point in feature.lane.polyline]
                        y_coords = [point.y for point in feature.lane.polyline]
                        color = lane_type_colors.get(lane_type, '#95a5a6')
                        ax.plot(x_coords, y_coords, color=color, linewidth=0.5, alpha=0.2, zorder=1)
                    
                    elif feature_type == 'road_edge' and hasattr(feature.road_edge, 'polyline'):
                        x_coords = [point.x for point in feature.road_edge.polyline]
                        y_coords = [point.y for point in feature.road_edge.polyline]
                        ax.plot(x_coords, y_coords, color='black', linewidth=0.5, alpha=0.2, zorder=1)
                    
                    elif feature_type == 'crosswalk' and hasattr(feature.crosswalk, 'polygon'):
                        x_coords = [point.x for point in feature.crosswalk.polygon]
                        y_coords = [point.y for point in feature.crosswalk.polygon]
                        x_coords.append(x_coords[0])
                        y_coords.append(y_coords[0])
                        ax.fill(x_coords, y_coords, color='yellow', alpha=0.15, zorder=1)
            
            # Get data for this timestep
            data = graph_data_per_timestep[t]
            if data is None or data['num_nodes'] < num_nodes:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_aspect('equal')
                continue
            
            # Truncate to consistent node count
            actual_pos = data['actual'][:num_nodes]
            pred_pos = data['pred'][:num_nodes]
            
            # Plot each selected node
            for idx, node_idx in enumerate(node_indices):
                color = node_colors[idx]
                
                actual_xy = actual_pos[node_idx].numpy()
                pred_xy = pred_pos[node_idx].numpy()
                
                # Plot actual position (filled circle)
                ax.scatter(actual_xy[0], actual_xy[1], color=color, s=30, 
                          marker='o', edgecolors='black', linewidths=0.8, 
                          alpha=0.9, zorder=5, label='Actual' if idx == 0 else None)
                
                # Plot predicted position (hollow circle)
                ax.scatter(pred_xy[0], pred_xy[1], facecolors='none', edgecolors=color, s=30, 
                          marker='o', linewidths=1.5, alpha=0.8, zorder=4,
                          label='Predicted' if idx == 0 else None)
                
                # Draw error line
                ax.plot([actual_xy[0], pred_xy[0]], [actual_xy[1], pred_xy[1]], 
                       color='red', linewidth=0.8, alpha=0.4, zorder=3)
                
                # Calculate error
                error = np.linalg.norm(actual_xy - pred_xy)
                total_error += error
                error_counts += 1
            
            # Calculate average error for this cell
            cell_errors = []
            for idx in node_indices:
                actual_xy = actual_pos[idx].numpy()
                pred_xy = pred_pos[idx].numpy()
                cell_errors.append(np.linalg.norm(actual_xy - pred_xy))
            cell_avg_error = np.mean(cell_errors)
            
            # Formatting
            ax.set_aspect('equal', adjustable='box')
            if x_lim and y_lim:
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=7)
            
            # Add title with timestep and error
            ax.set_title(f't={t} | Err:{cell_avg_error:.1f}m', fontsize=8, fontweight='bold')
            
            # Add row and column labels
            if t == 0:
                ax.set_ylabel(f'Graph {graph_idx+1}', fontsize=9, fontweight='bold')
            if graph_idx == 0:
                ax.text(0.5, 1.15, f'Timestep {t}', transform=ax.transAxes,
                       ha='center', fontsize=9, fontweight='bold')
    
    # Overall average error
    avg_error = total_error / max(1, error_counts)
    
    # Add overall title
    map_note = ' (with map)' if scenario is not None else ''
    fig.suptitle(f'Epoch {epoch} - Training Progress{map_note}\n'
                f'Rows = Graphs (1-{B}) | Columns = Timesteps (0-{T-1})\n'
                f'● = Actual | ○ = Predicted | Red line = Error | Overall Avg Error: {avg_error:.2f}m', 
                fontsize=11, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    filename = f'epoch_{epoch:03d}_progress.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved training progress visualization: {filepath}")
    print(f"    Grid: {B} graphs × {T} timesteps | Avg Error: {avg_error:.2f}m | {len(node_indices)} nodes per cell")
    
    model.train()
    return filepath, avg_error


def visualize_graph_sequence_creation(scenario, graph_sequence, save_dir='graph_visualizations', figsize=(16, 10)):
    """
    Visualize a temporal sequence of graphs created from a scenario.
    Shows how the graph structure evolves over time.
    
    Args:
        scenario: Waymo scenario object
        graph_sequence: List of PyG Data objects (temporal sequence)
        radius: Radius used for graph creation
        graph_creation_method: Graph creation method ('radius' or 'star')
        save_dir: Directory to save visualization
        figsize: Figure size
    
    Returns:
        filepath: Path to saved visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    T = len(graph_sequence)
    
    # Get SDC track ID
    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_id = sdc_track.id
    
    # Create grid: show subset of timesteps (e.g., every 2nd or 3rd)
    max_timesteps_to_show = 8
    if T > max_timesteps_to_show:
        timestep_indices = np.linspace(0, T-1, max_timesteps_to_show, dtype=int)
    else:
        timestep_indices = np.arange(T)
    
    num_plots = len(timestep_indices)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Agent type colors
    type_colors = {1: '#e74c3c', 2: '#3498db', 3: '#f39c12', 4: '#9b59b6'}
    type_names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Other'}
    
    # Lane colors for map
    lane_type_colors = {
        0: '#95a5a6', 1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'
    }
    
    for plot_idx, t in enumerate(timestep_indices):
        ax = axes[plot_idx]
        graph = graph_sequence[t]
        
        # Render map features (lightweight)
        for feature in scenario.map_features:
            feature_type = feature.WhichOneof('feature_data')
            
            if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
                lane_type = feature.lane.type
                x_coords = [point.x for point in feature.lane.polyline]
                y_coords = [point.y for point in feature.lane.polyline]
                color = lane_type_colors.get(lane_type, '#95a5a6')
                ax.plot(x_coords, y_coords, color=color, linewidth=0.5, alpha=0.15, zorder=1)
            
            elif feature_type == 'road_edge' and hasattr(feature.road_edge, 'polyline'):
                x_coords = [point.x for point in feature.road_edge.polyline]
                y_coords = [point.y for point in feature.road_edge.polyline]
                ax.plot(x_coords, y_coords, color='black', linewidth=0.5, alpha=0.15, zorder=1)
        
        # Render graph edges
        edge_index = graph.edge_index
        pos = graph.pos.numpy()
        
        for i in range(edge_index.shape[1]):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()
            x_coords = [pos[src_idx, 0], pos[dst_idx, 0]]
            y_coords = [pos[src_idx, 1], pos[dst_idx, 1]]
            ax.plot(x_coords, y_coords, 'blue', linewidth=0.8, alpha=0.3, zorder=3)
        
        # Check if SDC is present in this graph
        sdc_found = False
        if hasattr(graph, 'agent_ids'):
            # graph has agent_ids attribute (from Data creation)
            agent_ids = graph.agent_ids
            if sdc_id in agent_ids:
                sdc_node_idx = agent_ids.index(sdc_id)
                sdc_found = True
        
        # Render nodes (agent positions)
        # Get agent types from node features (last 4 dimensions are one-hot)
        node_features = graph.x.numpy()
        for node_idx in range(graph.num_nodes):
            x, y = pos[node_idx]
            
            # Determine agent type from one-hot encoding (indices 5-8)
            type_onehot = node_features[node_idx, 5:9]
            agent_type = np.argmax(type_onehot) + 1  # 1-indexed
            color = type_colors.get(agent_type, '#95a5a6')
            
            # Check if this is the SDC node
            is_sdc = False
            if sdc_found and hasattr(graph, 'agent_ids') and node_idx == sdc_node_idx:
                is_sdc = True
            
            if is_sdc:
                # Special rendering for SDC: larger, green border, star marker
                ax.scatter(x, y, c=color, s=100, marker='*', 
                          edgecolors='#27ae60', linewidths=2.5, alpha=1.0, zorder=10)
                # Add SDC label
                ax.annotate('SDC', xy=(x, y), xytext=(8, 8), 
                           textcoords='offset points', fontsize=7, 
                           fontweight='bold', color='#27ae60',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='#27ae60', alpha=0.9))
            else:
                # Normal agent rendering
                ax.scatter(x, y, c=color, s=60, marker='o', 
                          edgecolors='white', linewidths=1, alpha=0.9, zorder=5)
        
        # Check SDC validity and add warning if invalid
        sdc_valid = t < len(sdc_track.states) and sdc_track.states[t].valid
        
        if not sdc_valid or not sdc_found:
            # Add red warning banner
            warning_text = "⚠ SDC INVALID" if not sdc_valid else "⚠ SDC NOT IN GRAPH"
            ax.text(0.5, 0.95, warning_text, 
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=8, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.9))
        
        # Add SDC coordinates in bottom right corner
        if sdc_found and sdc_valid:
            # Get SDC position from the graph
            sdc_x, sdc_y = pos[sdc_node_idx]
            coord_text = f'SDC: ({sdc_x:.1f}, {sdc_y:.1f})'
            ax.text(0.98, 0.02, coord_text, 
                   transform=ax.transAxes, ha='right', va='bottom',
                   fontsize=7, fontweight='bold', color='#27ae60',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='#27ae60', alpha=0.85, linewidth=1.5))
        
        # Formatting
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=7)
        
        # Title
        time_sec = scenario.timestamps_seconds[t] if t < len(scenario.timestamps_seconds) else 0
        ax.set_title(f't={t} ({time_sec:.1f}s)\n{graph.num_nodes}N, {graph.num_edges}E', 
                    fontsize=8, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    method_str = f'{graph_creation_method} (r={radius}m)' if graph_creation_method == 'radius' else graph_creation_method
    fig.suptitle(f'Graph Sequence Creation - Scenario: {scenario.scenario_id}\n'
                f'Method: {method_str} | Total Timesteps: {T} | Showing: {num_plots} snapshots', 
                fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    filename = f'graph_sequence_{scenario.scenario_id}_{graph_creation_method}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    return filepath

def create_graph_sequence_visualization(scenario, save_dir='../../visualizations/'):
    """Creates .png showing graphs for given scenario sequence"""
    from config import radius, graph_creation_method, sequence_length
    from helper_functions.graph_creation_functions import timestep_to_pyg_data
    
    print(f"Creating graph sequence visualization (from scenario {scenario.scenario_id}) into {save_dir} directory...")
    try:
        graph_sequence = []
        for t in range(min(sequence_length, len(scenario.timestamps_seconds))):
            graph = timestep_to_pyg_data(scenario, t, radius, use_valid_only=True, method=graph_creation_method)
            if graph is not None:
                graph_sequence.append(graph)
            
        if graph_sequence:
            visualize_graph_sequence_creation(scenario, graph_sequence, save_dir=save_dir)
        else:
            print("No valid graphs created for visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    print(f"Sequence visualized successfully at {save_dir}/graph_sequence_{scenario.scenario_id}_{graph_creation_method}.png")
