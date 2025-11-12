
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from datetime import datetime
from helper_functions.graph_creation_functions import timestep_to_pyg_data
from config import sequence_length, radius, graph_creation_method, viz_scenario_dir, viz_training_dir, visualize_every_n_epochs, max_nodes_per_graph_viz, show_timesteps_viz
import matplotlib.pyplot as plt

# Cache for loaded scenarios to avoid reloading the same scenario multiple times
_scenario_cache = {}

def load_scenario_by_id(scenario_id, scenario_dir="./data/scenario/training"):
    """
    Load a Waymo scenario from tfrecord files by scenario_id.
    Uses caching to avoid reloading the same scenario multiple times.
    
    Args:
        scenario_id: The scenario ID to load
        scenario_dir: Directory containing tfrecord files
        
    Returns:
        Scenario object or None if not found
    """
    # Check cache first
    if scenario_id in _scenario_cache:
        return _scenario_cache[scenario_id]
    
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2
        from pathlib import Path
        
        scenario_path = Path(scenario_dir)
        tfrecord_files = sorted(scenario_path.glob("*.tfrecord"))
        
        for tfrecord_file in tfrecord_files:
            scenario_dataset = tf.data.TFRecordDataset(str(tfrecord_file), compression_type='')
            
            for raw_record in scenario_dataset:
                scenario = scenario_pb2.Scenario.FromString(raw_record.numpy())
                if scenario.scenario_id == scenario_id:
                    # Cache and return
                    _scenario_cache[scenario_id] = scenario
                    print(f"  Loaded scenario {scenario_id} with {len(scenario.map_features)} map features")
                    return scenario
        
        print(f"  Warning: Scenario {scenario_id} not found in {scenario_dir}")
        return None
        
    except Exception as e:
        print(f"  Warning: Could not load scenario {scenario_id}: {e}")
        return None

def visualize_training_progress(model, batch_dict, epoch, scenario_id=None, save_dir=viz_training_dir,
                                device='cpu', max_nodes_per_graph=10, show_timesteps=8):
    """
    Visualization showing actual vs predicted trajectories with map features.
    Creates a B × T grid where rows are graphs and columns are timesteps.
    
    Args:
        model: The trained model
        batch_dict: Batch dictionary with temporal graph sequence
        epoch: Current epoch number
        scenario_id: Scenario ID to load map features from (if None, will try to get from batch data)
        save_dir: Directory to save visualizations
        device: Device for inference
        max_nodes_per_graph: Maximum nodes to show per individual graph
        show_timesteps: Number of timesteps to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load scenario for map features if scenario_id provided or available in batch
    scenario = None
    if scenario_id is None:
        # Get scenario_id from batch_dict (added by collate function)
        if 'scenario_ids' in batch_dict and batch_dict['scenario_ids']:
            scenario_ids_list = batch_dict['scenario_ids']
            # Get first valid scenario_id
            scenario_id = next((sid for sid in scenario_ids_list if sid is not None), None)
            if scenario_id:
                print(f"  Found scenario_id from batch_dict: {scenario_id}")
    
    if scenario_id is not None:
        print(f"  Loading scenario {scenario_id} for map visualization...")
        scenario = load_scenario_by_id(scenario_id)
    else:
        print(f"  Warning: No scenario_id found, visualization will not include map features")
    
    model.eval()
    batched_graph_sequence = batch_dict["batch"]
    B = batch_dict["B"]  # Number of graphs in batch
    total_T = batch_dict["T"]
    
    # Sample timesteps evenly across the sequence
    if show_timesteps < total_T:
        timestep_indices = np.linspace(0, total_T - 1, show_timesteps, dtype=int)
        T = len(timestep_indices)
    else:
        timestep_indices = np.arange(total_T)
        T = total_T
    
    # Reset hidden states with correct batch size
    model.reset_gru_hidden_states(batch_size=B)
    
    # Move to device
    for t in range(len(batch_dict["batch"])):
        batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
    
    # Denormalization factor (we normalized by dividing by 100)
    POSITION_SCALE = 100.0
    
    # Collect data for each timestep
    all_curr_pos = []
    all_actual_next_pos = []
    all_pred_next_pos = []
    all_batch_indices = []
    
    with torch.no_grad():
        # IMPORTANT: Process ALL timesteps to maintain GRU hidden state evolution
        # but only save data for the sampled timesteps
        for t in range(total_T):
            graph = batched_graph_sequence[t]
            
            # Debug: Check if pos exists in the batched graph
            if t == 0:
                print(f"  DEBUG: First graph - hasattr(graph, 'pos'): {hasattr(graph, 'pos')}, graph.pos is not None: {hasattr(graph, 'pos') and graph.pos is not None}")
                if hasattr(graph, 'pos') and graph.pos is not None:
                    print(f"  DEBUG: graph.pos.shape: {graph.pos.shape}")
            
            # Current positions
            # Note: graph.pos contains RAW world coordinates (meters), not normalized
            if hasattr(graph, 'pos') and graph.pos is not None:
                curr_pos = graph.pos.cpu()  # Already in meters, no need to denormalize
            else:
                # Fallback: if pos not available, positions are not in the feature vector anymore
                # This shouldn't happen with properly saved HDF5 files
                raise ValueError("graph.pos is None - HDF5 file may need to be regenerated")
            
            # Get predicted displacement (normalized) - MUST run for all timesteps to evolve GRU
            pred_displacement_norm = model(graph.x, graph.edge_index, graph.batch, batch_size=B).cpu()
            
            # Denormalize displacement and add to current position to get predicted next position
            pred_displacement = pred_displacement_norm * POSITION_SCALE
            pred_next_pos = curr_pos + pred_displacement
            
            # Get actual next position from ground truth labels
            # graph.y contains the normalized displacement to next position
            if graph.y is not None:
                actual_displacement_norm = graph.y.cpu()
                actual_displacement = actual_displacement_norm * POSITION_SCALE
                actual_next_pos = curr_pos + actual_displacement
            else:
                # If no ground truth, use current position as fallback
                actual_next_pos = curr_pos
            
            # Only save data for sampled timesteps
            if t in timestep_indices:
                all_curr_pos.append(curr_pos)
                all_pred_next_pos.append(pred_next_pos)
                all_actual_next_pos.append(actual_next_pos)
                all_batch_indices.append(graph.batch.cpu())
    
    # Create figure with B × T grid (rows = graphs, columns = timesteps)
    # Larger figure size for better visibility
    fig, axes = plt.subplots(B, T, figsize=(5*T, 5*B))
    
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
    
    # Gray colors for all map features
    map_feature_gray = '#808080'  # Medium gray for all map features
    
    # Process each graph (row) and timestep (column)
    for graph_idx in range(B):
        # Get nodes for this graph across all timesteps
        graph_data_per_timestep = []
        
        for t in range(T):
            batch_mask = all_batch_indices[t] == graph_idx
            if batch_mask.sum() > 0:
                actual_next_pos = all_actual_next_pos[t][batch_mask]
                pred_next_pos = all_pred_next_pos[t][batch_mask]
                curr_pos = all_curr_pos[t][batch_mask]
                graph_data_per_timestep.append({
                    'actual_next': actual_next_pos,
                    'pred_next': pred_next_pos,
                    'curr': curr_pos,
                    'num_nodes': actual_next_pos.shape[0]
                })
            else:
                graph_data_per_timestep.append(None)
        
        # Determine consistent number of nodes for this graph
        valid_counts = [d['num_nodes'] for d in graph_data_per_timestep if d is not None]
        if not valid_counts:
            continue
        
        # Use minimum node count across all timesteps
        num_nodes = min(valid_counts)
        
        # Find SDC index by checking agent_ids in the first valid timestep
        sdc_idx = None
        first_valid_graph = batched_graph_sequence[0]
        if hasattr(first_valid_graph, 'agent_ids'):
            # Get SDC id from scenario
            if scenario is not None:
                sdc_track = scenario.tracks[scenario.sdc_track_index]
                sdc_id = sdc_track.id
                
                # Find which nodes belong to current graph
                batch_mask = first_valid_graph.batch == graph_idx
                graph_agent_ids = [first_valid_graph.agent_ids[i] for i in range(len(first_valid_graph.agent_ids)) if batch_mask[i]]
                
                if sdc_id in graph_agent_ids:
                    sdc_idx = graph_agent_ids.index(sdc_id)
        
        # Select subset of nodes to visualize, ensuring SDC is included
        if num_nodes > max_nodes_per_graph:
            if sdc_idx is not None and sdc_idx < num_nodes:
                # Include SDC first, then evenly sample others
                other_indices = [i for i in range(num_nodes) if i != sdc_idx]
                num_others = max_nodes_per_graph - 1
                if len(other_indices) > num_others:
                    other_selected = np.linspace(0, len(other_indices) - 1, num_others, dtype=int)
                    other_indices = [other_indices[i] for i in other_selected]
                node_indices = np.array([sdc_idx] + other_indices)
            else:
                node_indices = np.linspace(0, num_nodes - 1, max_nodes_per_graph, dtype=int)
        else:
            node_indices = np.arange(num_nodes)
            # Ensure SDC is in the list
            if sdc_idx is not None and sdc_idx not in node_indices and sdc_idx < num_nodes:
                node_indices = np.append([sdc_idx], node_indices[:-1])
        
        # Assign consistent colors to each node, with green for SDC
        node_colors = []
        for idx, node_idx in enumerate(node_indices):
            if node_idx == sdc_idx:
                node_colors.append('#27ae60')  # Green for SDC
            else:
                # Use tab20 colormap for other agents
                color_idx = idx / max(len(node_indices) - 1, 1)
                node_colors.append(plt.cm.tab20(color_idx))
        node_colors = np.array(node_colors)
        
        # Calculate axis limits for this graph sequence (entire row)
        # Include map features if available for proper zoom
        all_x_coords = []
        all_y_coords = []
        
        # Add agent positions
        for data in graph_data_per_timestep:
            if data is not None and data['num_nodes'] >= num_nodes:
                curr_pos = data['curr'][:num_nodes]
                actual_next_pos = data['actual_next'][:num_nodes]
                pred_next_pos = data['pred_next'][:num_nodes]
                all_x_coords.extend(curr_pos[:, 0].tolist())
                all_y_coords.extend(curr_pos[:, 1].tolist())
                all_x_coords.extend(actual_next_pos[:, 0].tolist())
                all_y_coords.extend(actual_next_pos[:, 1].tolist())
                all_x_coords.extend(pred_next_pos[:, 0].tolist())
                all_y_coords.extend(pred_next_pos[:, 1].tolist())
        
        # Add map feature bounds if scenario available
        if scenario is not None:
            for feature in scenario.map_features:
                feature_type = feature.WhichOneof('feature_data')
                
                if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
                    all_x_coords.extend([point.x for point in feature.lane.polyline])
                    all_y_coords.extend([point.y for point in feature.lane.polyline])
                
                elif feature_type == 'road_edge' and hasattr(feature.road_edge, 'polyline'):
                    all_x_coords.extend([point.x for point in feature.road_edge.polyline])
                    all_y_coords.extend([point.y for point in feature.road_edge.polyline])
        
        # Standardized axis limits with minimal padding to fit map features
        if all_x_coords and all_y_coords:
            x_min, x_max = min(all_x_coords), max(all_x_coords)
            y_min, y_max = min(all_y_coords), max(all_y_coords)
            
            # Calculate range and use minimal padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Use 5% padding or 2m minimum (tight fit for map features)
            x_padding = max(x_range * 0.05, 2.0)
            y_padding = max(y_range * 0.05, 2.0)
            
            x_lim = (x_min - x_padding, x_max + x_padding)
            y_lim = (y_min - y_padding, y_max + y_padding)
        else:
            x_lim, y_lim = None, None
        
        # Plot each timestep for this graph
        for t in range(T):
            ax = axes[graph_idx, t]
            
            # Plot map features (only once per subplot)
            # Map features are in raw world coordinates (meters), not normalized
            if scenario is not None:
                map_features_drawn = 0
                for feature in scenario.map_features:
                    feature_type = feature.WhichOneof('feature_data')
                    
                    if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
                        x_coords = [point.x for point in feature.lane.polyline]
                        y_coords = [point.y for point in feature.lane.polyline]
                        ax.plot(x_coords, y_coords, color=map_feature_gray, linewidth=1.5, alpha=0.4, zorder=1)
                        map_features_drawn += 1
                    
                    elif feature_type == 'road_edge' and hasattr(feature.road_edge, 'polyline'):
                        x_coords = [point.x for point in feature.road_edge.polyline]
                        y_coords = [point.y for point in feature.road_edge.polyline]
                        ax.plot(x_coords, y_coords, color=map_feature_gray, linewidth=2.0, alpha=0.5, zorder=1)
                        map_features_drawn += 1
                    
                    elif feature_type == 'road_line' and hasattr(feature.road_line, 'polyline'):
                        x_coords = [point.x for point in feature.road_line.polyline]
                        y_coords = [point.y for point in feature.road_line.polyline]
                        # Use dashed line for road markings
                        ax.plot(x_coords, y_coords, color=map_feature_gray, linewidth=1.0, 
                               linestyle='--', alpha=0.4, zorder=1)
                        map_features_drawn += 1
                    
                    elif feature_type == 'crosswalk' and hasattr(feature.crosswalk, 'polygon'):
                        x_coords = [point.x for point in feature.crosswalk.polygon]
                        y_coords = [point.y for point in feature.crosswalk.polygon]
                        x_coords.append(x_coords[0])
                        y_coords.append(y_coords[0])
                        ax.fill(x_coords, y_coords, color=map_feature_gray, alpha=0.3, zorder=1)
                        map_features_drawn += 1
                
                if t == 0 and graph_idx == 0:
                    print(f"  Drew {map_features_drawn} map features on first subplot")
            
            # Get data for this timestep
            data = graph_data_per_timestep[t]
            if data is None or data['num_nodes'] < num_nodes:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_aspect('equal')
                continue
            
            # Truncate to consistent node count
            curr_pos = data['curr'][:num_nodes]
            actual_next_pos = data['actual_next'][:num_nodes]
            pred_next_pos = data['pred_next'][:num_nodes]
            
            # Draw trajectories first (so they're behind the points)
            if t > 0:
                prev_data = graph_data_per_timestep[t-1]
                if prev_data is not None and prev_data['num_nodes'] >= num_nodes:
                    prev_curr_pos = prev_data['curr'][:num_nodes]
                    
                    # Draw trajectory lines connecting previous timestep to current position
                    for idx, node_idx in enumerate(node_indices):
                        color = node_colors[idx]
                        
                        # Actual trajectory (solid line) - from previous current position to current position
                        ax.plot([prev_curr_pos[node_idx, 0], curr_pos[node_idx, 0]], 
                               [prev_curr_pos[node_idx, 1], curr_pos[node_idx, 1]], 
                               color=color, linewidth=2.5, alpha=0.9, linestyle='-', zorder=2,
                               label='Actual Trajectory' if idx == 0 and t == 1 else None)
            
            # Draw predicted trajectories from current position to predicted next
            for idx, node_idx in enumerate(node_indices):
                color = node_colors[idx]
                
                curr_xy = curr_pos[node_idx].numpy()
                pred_next_xy = pred_next_pos[node_idx].numpy()
                
                # Predicted trajectory (dotted line) - from current position to predicted next
                ax.plot([curr_xy[0], pred_next_xy[0]], 
                       [curr_xy[1], pred_next_xy[1]], 
                       color=color, linewidth=2.5, alpha=0.9, linestyle=':', zorder=2,
                       label='Predicted Trajectory' if idx == 0 and t == 0 else None)
            
            # Plot each selected node
            for idx, node_idx in enumerate(node_indices):
                color = node_colors[idx]
                is_sdc = (node_idx == sdc_idx)
                
                curr_xy = curr_pos[node_idx].numpy()
                actual_next_xy = actual_next_pos[node_idx].numpy()
                pred_next_xy = pred_next_pos[node_idx].numpy()
                
                # Plot current position (star for SDC, circle for others)
                if is_sdc:
                    ax.scatter(curr_xy[0], curr_xy[1], color=color, s=100, 
                              marker='*', alpha=1.0, zorder=3, 
                              edgecolors='darkgreen', linewidths=2,
                              label='SDC Current' if idx == 0 else None)
                else:
                    ax.scatter(curr_xy[0], curr_xy[1], color=color, s=30, 
                              marker='o', alpha=0.6, zorder=3, 
                              label='Current' if idx == 1 else None)
                
                # Plot predicted next position (hollow circle)
                marker_size = 25 if is_sdc else 20
                linewidth = 2.0 if is_sdc else 1.5
                ax.scatter(pred_next_xy[0], pred_next_xy[1], facecolors='none', edgecolors=color, 
                          s=marker_size, marker='o', linewidths=linewidth, alpha=0.9, zorder=4,
                          label='Predicted Next' if idx == 0 else None)
                
                # Draw thicker error line between actual and predicted next positions
                #ax.plot([actual_next_xy[0], pred_next_xy[0]], [actual_next_xy[1], pred_next_xy[1]], 
                #       color='red', linewidth=2.0, alpha=0.7, zorder=3)
                
                # Calculate error (distance between actual and predicted NEXT positions)
                error = np.linalg.norm(actual_next_xy - pred_next_xy)
                total_error += error
                error_counts += 1
            
            # Calculate average error for this cell
            cell_errors = []
            for idx in node_indices:
                actual_next_xy = actual_next_pos[idx].numpy()
                pred_next_xy = pred_next_pos[idx].numpy()
                cell_errors.append(np.linalg.norm(actual_next_xy - pred_next_xy))
            cell_avg_error = np.mean(cell_errors)
            
            # Formatting
            ax.set_aspect('equal', adjustable='box')
            if x_lim and y_lim:
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=10)
            
            # Add axis labels
            ax.set_xlabel('X position (m)', fontsize=10)
            ax.set_ylabel('Y position (m)', fontsize=10)
            
            # Add title with actual timestep number and error (larger font)
            actual_t = timestep_indices[t]
            ax.set_title(f't={actual_t} | Err:{cell_avg_error:.1f}m', fontsize=11, fontweight='bold')
            
            # Add row and column labels
            if t == 0:
                ax.set_ylabel(f'Graph {graph_idx+1}\n\nY position (m)', fontsize=12, fontweight='bold')
            if graph_idx == 0:
                ax.text(0.5, 1.12, f'Timestep {actual_t}', transform=ax.transAxes,
                       ha='center', fontsize=12, fontweight='bold')
    
    # Overall average error
    avg_error = total_error / max(1, error_counts)
    
    # Create custom legend with all visualization elements
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#27ae60', markersize=12, 
               label='SDC (Self-Driving Car)', markeredgecolor='darkgreen', markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, 
               label='Current Position', markeredgewidth=0),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=8,
               markeredgecolor='tab:blue', markeredgewidth=1.5, label='Predicted Next Position'),
        Line2D([0], [0], color='tab:blue', linewidth=2, linestyle='-', 
               label='Actual Trajectory (solid)', alpha=0.7),
        Line2D([0], [0], color='tab:blue', linewidth=2, linestyle=':', 
               label='Predicted Trajectory (dotted)', alpha=0.7),
        Line2D([0], [0], color='red', linewidth=2, label='Prediction Error', alpha=0.7),
    ]
    
    # Add map feature legend items if scenario is present
    if scenario is not None:
        legend_elements.extend([
            Line2D([0], [0], color=map_feature_gray, linewidth=1.5, alpha=0.4, label='Lane'),
            Line2D([0], [0], color=map_feature_gray, linewidth=2.0, alpha=0.5, label='Road Edge'),
            Line2D([0], [0], color=map_feature_gray, linewidth=1.0, linestyle='--', alpha=0.4, label='Road Marking'),
            Patch(facecolor=map_feature_gray, alpha=0.3, label='Crosswalk'),
        ])
    
    # Add legend to the figure (outside the subplots)
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.98),
               ncol=3 if scenario is not None else 2, fontsize=9, framealpha=0.95,
               title='Legend', title_fontsize=10)
    
    # Add overall title with more spacing
    map_note = ' with Map Features' if scenario is not None else ''
    title_text = (f'Epoch {epoch} - Training Progress{map_note}\n'
                  f'Showing {T} of {total_T} timesteps (evenly sampled) | '
                  f'Overall Avg Prediction Error: {avg_error:.2f}m')
    
    fig.suptitle(title_text, fontsize=13, fontweight='bold', y=0.99)
    
    # Adjust layout to prevent title and legend overlap
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Save
    filename = f'epoch_{epoch:03d}_progress.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved training progress visualization: {filepath}")
    print(f"    Grid: {B} graphs × {T} timesteps | Avg Error: {avg_error:.2f}m | {len(node_indices)} nodes per cell")
    
    model.train()
    return filepath, avg_error

def visualize_graph_sequence_creation(scenario, graph_sequence, max_timesteps_to_show, save_dir=viz_scenario_dir, figsize=(16, 10)):
    """Visualize a temporal sequence of graphs created from a scenario. Shows how the graph structure evolves over time."""
    os.makedirs(save_dir, exist_ok=True)
    T = len(graph_sequence)

    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_id = sdc_track.id
    
    max_timesteps_to_show = max_timesteps_to_show   # Create grid: show subset of timesteps (e.g., every 2nd or 3rd)
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
    
    type_colors = {1: '#e74c3c', 2: '#3498db', 3: '#f39c12', 4: '#9b59b6'}
    type_names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Other'}
    
    # Gray color for all map features
    map_feature_gray = '#808080'
    
    for plot_idx, t in enumerate(timestep_indices):
        ax = axes[plot_idx]
        graph = graph_sequence[t]
        
        for feature in scenario.map_features:       # Render map features (lightweight)
            feature_type = feature.WhichOneof('feature_data')
            
            if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
                x_coords = [point.x for point in feature.lane.polyline]
                y_coords = [point.y for point in feature.lane.polyline]
                ax.plot(x_coords, y_coords, color=map_feature_gray, linewidth=0.5, alpha=0.3, zorder=1)
            
            elif feature_type == 'road_edge' and hasattr(feature.road_edge, 'polyline'):
                x_coords = [point.x for point in feature.road_edge.polyline]
                y_coords = [point.y for point in feature.road_edge.polyline]
                ax.plot(x_coords, y_coords, color=map_feature_gray, linewidth=0.5, alpha=0.4, zorder=1)
        
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

def create_graph_sequence_visualization(scenario, save_dir=viz_scenario_dir, num_timesteps=15):
    """Creates .png showing graphs for given scenario sequence"""
    from config import radius, graph_creation_method, sequence_length
    from helper_functions.graph_creation_functions import timestep_to_pyg_data
    
    print(f"Creating graph sequence visualization (from scenario {scenario.scenario_id}) into {save_dir} directory...")
    try:
        graph_sequence = []
        print(f"sequence length: {sequence_length}, scenario.timestamps_seconds length: {len(scenario.timestamps_seconds)}")
        for t in range(min(sequence_length-1, len(scenario.timestamps_seconds))):
            graph = timestep_to_pyg_data(scenario, t, radius, use_valid_only=True, method=graph_creation_method)
            if graph is not None:
                graph_sequence.append(graph)
            
        if graph_sequence:
            visualize_graph_sequence_creation(scenario, graph_sequence, num_timesteps, save_dir=save_dir)
        else:
            print("No valid graphs created for visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    print(f"Sequence visualized successfully at {save_dir}/graph_sequence_{scenario.scenario_id}_{graph_creation_method}.png\n")

def visualize_epoch(epoch, viz_batch, model, device, wandb):
    should_visualize = (epoch + 1) % visualize_every_n_epochs == 0 or epoch == 0
        
    if should_visualize and viz_batch is not None:
        print(f"  Generating visualization of first batch for epoch {epoch+1}...")
        try:
            filepath, avg_error = visualize_training_progress(
                model, viz_batch, epoch=epoch+1,
                scenario_id=None,  # Will auto-detect from batch data
                save_dir=viz_training_dir,
                device=device,
                max_nodes_per_graph=max_nodes_per_graph_viz,
                show_timesteps=show_timesteps_viz
            )
            # Log visualization error to wandb
            wandb.log({"epoch": epoch, "viz_avg_error": avg_error})
        except Exception as e:
            print(f"  Warning: Visualization failed: {e}")
            import traceback
            traceback.print_exc()
        print("="*30)