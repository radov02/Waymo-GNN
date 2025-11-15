
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
            #if t == 0:
            #    print(f"  DEBUG: First graph - hasattr(graph, 'pos'): {hasattr(graph, 'pos')}, graph.pos is not None: {hasattr(graph, 'pos') and graph.pos is not None}")
            #    if hasattr(graph, 'pos') and graph.pos is not None:
            #        print(f"  DEBUG: graph.pos.shape: {graph.pos.shape}")
            
            # Current positions
            # Note: graph.pos contains RAW world coordinates (meters), not normalized
            if hasattr(graph, 'pos') and graph.pos is not None:
                curr_pos = graph.pos.cpu()  # Already in meters, no need to denormalize
            else:
                # Fallback: if pos not available, positions are not in the feature vector anymore
                # This shouldn't happen with properly saved HDF5 files
                raise ValueError("graph.pos is None - HDF5 file may need to be regenerated")
            
            # Get predicted displacement (normalized) - MUST run for all timesteps to evolve GRU
            pred_displacement_norm = model(graph.x, graph.edge_index, 
                                          edge_weight=graph.edge_attr, 
                                          batch=graph.batch, batch_size=B).cpu()
            
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
    
    # Import config for vehicle-only filter
    from config import viz_vehicles_only, max_nodes_per_graph_viz
    
    # Process each graph (row) and timestep (column)
    for graph_idx in range(B):
        # Get agent_ids from first timestep to track same agents across time
        first_graph = batched_graph_sequence[0]
        batch_mask_first = first_graph.batch == graph_idx
        
        if not hasattr(first_graph, 'agent_ids'):
            print(f"Warning: graph missing agent_ids, skipping graph {graph_idx}")
            continue
        
        # Get agent IDs for this graph sequence
        all_agent_ids = [first_graph.agent_ids[i] for i in range(len(first_graph.agent_ids)) if batch_mask_first[i]]
        
        # Filter to vehicles only if configured
        if viz_vehicles_only and hasattr(first_graph, 'x'):
            # Get node features for this graph
            node_features_first = first_graph.x[batch_mask_first]
            # Check if agent is vehicle (feature index 5 is vehicle one-hot)
            vehicle_mask = node_features_first[:, 5] == 1.0  # Index 5 is 'vehicle' in one-hot
            vehicle_indices = torch.where(vehicle_mask)[0].numpy()
            all_agent_ids = [all_agent_ids[i] for i in vehicle_indices]
        
        if len(all_agent_ids) == 0:
            print(f"Warning: No agents found for graph {graph_idx}")
            continue
        
        # Find SDC
        sdc_idx_in_list = None
        sdc_id = None
        if scenario is not None:
            sdc_track = scenario.tracks[scenario.sdc_track_index]
            sdc_id = sdc_track.id
            if sdc_id in all_agent_ids:
                sdc_idx_in_list = all_agent_ids.index(sdc_id)
        
        # Calculate movement score for each agent (how much they move)
        # This helps us select the most interesting agents to visualize
        agent_movement_scores = {}
        movement_threshold = 0.5  # meters - movement less than this is considered "stopped"
        
        for agent_id in all_agent_ids:
            # Collect all positions for this agent across timesteps
            positions = []
            for t_idx in range(total_T):
                graph = batched_graph_sequence[t_idx]
                batch_mask = graph.batch == graph_idx
                
                if batch_mask.sum() > 0 and hasattr(graph, 'agent_ids'):
                    timestep_agent_ids = [graph.agent_ids[i] for i in range(len(graph.agent_ids)) if batch_mask[i]]
                    if agent_id in timestep_agent_ids:
                        local_idx = timestep_agent_ids.index(agent_id)
                        global_idx = torch.where(batch_mask)[0][local_idx]
                        pos = graph.pos[global_idx].cpu().numpy()
                        positions.append(pos)
            
            # Calculate total movement (sum of distances between consecutive valid positions)
            total_movement = 0.0
            valid_steps = 0
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    dist = np.linalg.norm(positions[i+1] - positions[i])
                    if dist > movement_threshold:  # Only count significant movements
                        total_movement += dist
                        valid_steps += 1
            
            agent_movement_scores[agent_id] = (total_movement, valid_steps)
        
        # Select subset of agent IDs to track, prioritizing moving agents
        if len(all_agent_ids) > max_nodes_per_graph_viz:
            # Always include SDC
            if sdc_idx_in_list is not None:
                other_ids = [aid for i, aid in enumerate(all_agent_ids) if i != sdc_idx_in_list]
                num_others = max_nodes_per_graph_viz - 1
                
                # Sort other agents by movement score (total_movement, then valid_steps)
                other_ids_sorted = sorted(other_ids, 
                                         key=lambda aid: agent_movement_scores[aid], 
                                         reverse=True)
                
                # Take the most moving agents
                selected_agent_ids = [all_agent_ids[sdc_idx_in_list]] + other_ids_sorted[:num_others]
            else:
                # No SDC, just take the most moving agents
                all_agent_ids_sorted = sorted(all_agent_ids, 
                                             key=lambda aid: agent_movement_scores[aid], 
                                             reverse=True)
                selected_agent_ids = all_agent_ids_sorted[:max_nodes_per_graph_viz]
        else:
            selected_agent_ids = all_agent_ids
        
        # Assign consistent colors to each agent ID
        # Use more vibrant colors that contrast well with gray map features
        vibrant_colors = [
            '#FF1744',  # Bright red
            '#2979FF',  # Bright blue
            '#FF9100',  # Bright amber (green removed - reserved for SDC only)
            '#FF6D00',  # Bright orange
            '#D500F9',  # Bright purple
            '#FFEA00',  # Bright yellow
            '#00E5FF',  # Bright cyan
            '#FF4081',  # Bright pink
        ]
        
        agent_colors = {}
        for idx, agent_id in enumerate(selected_agent_ids):
            if agent_id == sdc_id:
                agent_colors[agent_id] = '#00C853'  # Bright green for SDC (more vibrant than before)
            else:
                # Cycle through vibrant colors
                color_idx = (idx - 1) % len(vibrant_colors) if sdc_id in selected_agent_ids else idx % len(vibrant_colors)
                agent_colors[agent_id] = vibrant_colors[color_idx]
        
        # Get nodes for this graph across all timesteps
        graph_data_per_timestep = []
        
        # Get nodes for this graph across all timesteps
        graph_data_per_timestep = []
        
        for t in range(T):
            graph = batched_graph_sequence[timestep_indices[t]]
            batch_mask = all_batch_indices[t] == graph_idx
            
            if batch_mask.sum() == 0 or not hasattr(graph, 'agent_ids'):
                graph_data_per_timestep.append(None)
                continue
            
            # Get agent IDs for this timestep
            timestep_agent_ids = [graph.agent_ids[i] for i in range(len(graph.agent_ids)) if batch_mask[i]]
            
            # Find indices of our selected agents in this timestep
            agent_indices_in_timestep = []
            for agent_id in selected_agent_ids:
                if agent_id in timestep_agent_ids:
                    local_idx = timestep_agent_ids.index(agent_id)
                    # Get global index in the batched tensors
                    global_indices = torch.where(batch_mask)[0]
                    agent_indices_in_timestep.append(global_indices[local_idx].item())
                else:
                    agent_indices_in_timestep.append(None)  # Agent not present
            
            # Extract positions for selected agents
            curr_pos_t = all_curr_pos[t]
            actual_next_pos_t = all_actual_next_pos[t]
            pred_next_pos_t = all_pred_next_pos[t]
            
            # Build position tensors for selected agents (fill with NaN if agent missing)
            selected_curr = []
            selected_actual = []
            selected_pred = []
            
            for global_idx in agent_indices_in_timestep:
                if global_idx is not None:
                    selected_curr.append(curr_pos_t[global_idx])
                    selected_actual.append(actual_next_pos_t[global_idx])
                    selected_pred.append(pred_next_pos_t[global_idx])
                else:
                    # Agent not present - use NaN
                    selected_curr.append(torch.tensor([float('nan'), float('nan')]))
                    selected_actual.append(torch.tensor([float('nan'), float('nan')]))
                    selected_pred.append(torch.tensor([float('nan'), float('nan')]))
            
            if selected_curr:
                graph_data_per_timestep.append({
                    'curr': torch.stack(selected_curr),
                    'actual_next': torch.stack(selected_actual),
                    'pred_next': torch.stack(selected_pred),
                    'agent_ids': selected_agent_ids
                })
            else:
                graph_data_per_timestep.append(None)
        
        # Calculate axis limits for this graph sequence (entire row)
        # Focus on ACTUAL agent positions only (not predictions)
        # Predictions that go outside will just be clipped/not shown
        all_x_coords = []
        all_y_coords = []
        
        # Add only ACTUAL agent positions (current and actual_next)
        # Do NOT include predictions - let them fall outside if they're wrong
        for data in graph_data_per_timestep:
            if data is not None:
                curr_pos = data['curr']
                actual_next_pos = data['actual_next']
                # Only use actual positions, not predictions
                all_x_coords.extend(curr_pos[:, 0].tolist())
                all_y_coords.extend(curr_pos[:, 1].tolist())
                all_x_coords.extend(actual_next_pos[:, 0].tolist())
                all_y_coords.extend(actual_next_pos[:, 1].tolist())
        
        # DON'T include predictions or all map features in zoom calculation
        # Only zoom to actual agent movements
        
        # Standardized axis limits with padding around actual agent positions
        if all_x_coords and all_y_coords:
            x_min, x_max = min(all_x_coords), max(all_x_coords)
            y_min, y_max = min(all_y_coords), max(all_y_coords)
            
            # Calculate range and use generous padding to show surrounding context
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Use 20% padding or 20m minimum (show context around action)
            x_padding = max(x_range * 0.2, 20.0)
            y_padding = max(y_range * 0.2, 20.0)
            
            x_lim = (x_min - x_padding, x_max + x_padding)
            y_lim = (y_min - y_padding, y_max + y_padding)
        else:
            x_lim, y_lim = None, None
        
        # Plot each timestep for this graph
        for t in range(T):
            ax = axes[graph_idx, t]
            
            # Plot map features (only once per subplot)
            # Map features are in raw world coordinates (meters), not normalized
            # Only draw features that are within or near the action area (x_lim, y_lim)
            if scenario is not None and x_lim is not None and y_lim is not None:
                map_features_drawn = 0
                for feature in scenario.map_features:
                    feature_type = feature.WhichOneof('feature_data')
                    
                    if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
                        x_coords = [point.x for point in feature.lane.polyline]
                        y_coords = [point.y for point in feature.lane.polyline]
                        # Check if any point is within the visible area
                        if any(x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1] 
                               for x, y in zip(x_coords, y_coords)):
                            ax.plot(x_coords, y_coords, color=map_feature_gray, linewidth=1.5, alpha=0.15, zorder=1)
                            map_features_drawn += 1
                    
                    elif feature_type == 'road_edge' and hasattr(feature.road_edge, 'polyline'):
                        x_coords = [point.x for point in feature.road_edge.polyline]
                        y_coords = [point.y for point in feature.road_edge.polyline]
                        # Check if any point is within the visible area
                        if any(x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1] 
                               for x, y in zip(x_coords, y_coords)):
                            ax.plot(x_coords, y_coords, color=map_feature_gray, linewidth=2.0, alpha=0.2, zorder=1)
                            map_features_drawn += 1
                    
                    elif feature_type == 'road_line' and hasattr(feature.road_line, 'polyline'):
                        x_coords = [point.x for point in feature.road_line.polyline]
                        y_coords = [point.y for point in feature.road_line.polyline]
                        # Check if any point is within the visible area
                        if any(x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1] 
                               for x, y in zip(x_coords, y_coords)):
                            # Use dashed line for road markings
                            ax.plot(x_coords, y_coords, color=map_feature_gray, linewidth=1.0, 
                                   linestyle='--', alpha=0.15, zorder=1)
                            map_features_drawn += 1
                    
                    elif feature_type == 'crosswalk' and hasattr(feature.crosswalk, 'polygon'):
                        x_coords = [point.x for point in feature.crosswalk.polygon]
                        y_coords = [point.y for point in feature.crosswalk.polygon]
                        # Check if any point is within the visible area
                        if any(x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1] 
                               for x, y in zip(x_coords, y_coords)):
                            x_coords.append(x_coords[0])
                            y_coords.append(y_coords[0])
                            ax.fill(x_coords, y_coords, color=map_feature_gray, alpha=0.1, zorder=1)
                            map_features_drawn += 1
                
                if t == 0:  # Print for first timestep of each graph
                    print(f"  Drew {map_features_drawn} map features for graph {graph_idx} (filtered to action area)")
            
            # Get data for this timestep
            data = graph_data_per_timestep[t]
            if data is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_aspect('equal')
                continue
            
            # Get positions for selected agents
            curr_pos = data['curr']
            actual_next_pos = data['actual_next']
            pred_next_pos = data['pred_next']
            agent_ids_in_data = data['agent_ids']
            
            # Draw actual historical trajectory (all previous positions up to current)
            for idx, agent_id in enumerate(agent_ids_in_data):
                color = agent_colors[agent_id]
                is_sdc = (agent_id == sdc_id)
                
                # Collect all past positions for this agent (only valid ones)
                historical_positions = []
                for past_t in range(t + 1):  # 0 to current timestep
                    past_data = graph_data_per_timestep[past_t]
                    if past_data is not None and agent_id in past_data['agent_ids']:
                        past_idx = past_data['agent_ids'].index(agent_id)
                        past_pos = past_data['curr'][past_idx]
                        # Only add position if valid (not NaN) - stops trajectory when agent becomes invalid
                        if not torch.isnan(past_pos).any():
                            historical_positions.append(past_pos.numpy())
                        else:
                            # Agent became invalid, clear history to break the trajectory
                            historical_positions = []
                
                # Draw actual trajectory line through all historical positions (past)
                if len(historical_positions) > 1:
                    historical_positions = np.array(historical_positions)
                    linewidth = 1.2 if is_sdc else 1.0
                    ax.plot(historical_positions[:, 0], historical_positions[:, 1], 
                           color=color, linewidth=linewidth, alpha=0.8, linestyle='-', zorder=3,
                           label='Actual Past Trajectory' if idx == 0 and t == 0 else None)
                
                # Collect all future positions for this agent (ground truth future)
                future_positions = []
                for future_t in range(t, T):  # current to last displayed timestep
                    future_data = graph_data_per_timestep[future_t]
                    if future_data is not None and agent_id in future_data['agent_ids']:
                        future_idx = future_data['agent_ids'].index(agent_id)
                        future_pos = future_data['curr'][future_idx]
                        if not torch.isnan(future_pos).any():
                            future_positions.append(future_pos.numpy())
                        else:
                            # Agent became invalid, stop future trajectory
                            break
                
                # Draw actual future trajectory line (ground truth)
                if len(future_positions) > 1:
                    future_positions = np.array(future_positions)
                    linewidth = 1.2 if is_sdc else 1.0
                    ax.plot(future_positions[:, 0], future_positions[:, 1], 
                           color=color, linewidth=linewidth, alpha=0.5, linestyle='--', zorder=3,
                           label='Actual Future Trajectory' if idx == 0 and t == 0 else None)
                
                # Draw 1-step prediction
                curr_xy = curr_pos[idx]
                pred_next_xy = pred_next_pos[idx]
                
                # Skip if agent not present (NaN)
                if torch.isnan(curr_xy).any():
                    continue
                
                curr_xy = curr_xy.numpy()
                pred_next_xy = pred_next_xy.numpy()
                
                # Predicted next position (black dashed line) - from current to predicted next
                linewidth = 1.2 if is_sdc else 1.0
                ax.plot([curr_xy[0], pred_next_xy[0]], 
                       [curr_xy[1], pred_next_xy[1]], 
                       color='black', linewidth=linewidth, alpha=0.6, linestyle='--', zorder=3,
                       label='Predicted Step' if idx == 0 and t == 0 else None)
            
            # Plot each selected agent
            for idx, agent_id in enumerate(agent_ids_in_data):
                color = agent_colors[agent_id]
                is_sdc = (agent_id == sdc_id)
                
                curr_xy = curr_pos[idx]
                actual_next_xy = actual_next_pos[idx]
                pred_next_xy = pred_next_pos[idx]
                
                # Skip if agent not present (NaN)
                if torch.isnan(curr_xy).any():
                    continue
                
                curr_xy = curr_xy.numpy()
                actual_next_xy = actual_next_xy.numpy()
                pred_next_xy = pred_next_xy.numpy()
                
                # Check if agent is stopped at this timestep
                is_stopped = False
                if t > 0:  # Check movement from previous timestep
                    prev_data = graph_data_per_timestep[t - 1]
                    if prev_data is not None and agent_id in prev_data['agent_ids']:
                        prev_idx = prev_data['agent_ids'].index(agent_id)
                        prev_pos = prev_data['curr'][prev_idx]
                        if not torch.isnan(prev_pos).any():
                            prev_xy = prev_pos.numpy()
                            # Check if agent moved less than 0.1m from previous timestep
                            movement = np.linalg.norm(curr_xy - prev_xy)
                            is_stopped = movement < 0.1  # Stopped threshold: 0.1 meters
                else:  # t=0: check if next movement is negligible
                    if not np.isnan(actual_next_xy).any():
                        next_movement = np.linalg.norm(actual_next_xy - curr_xy)
                        is_stopped = next_movement < 0.1  # Stopped if next step < 0.1m
                
                # Plot current position (very small dots for all agents)
                if is_sdc:
                    ax.scatter(curr_xy[0], curr_xy[1], color=color, s=12, 
                              marker='o', alpha=1.0, zorder=5, 
                              edgecolors='darkgreen', linewidths=1.0,
                              label='SDC Current' if idx == 0 else None)
                    # Add "s" text inside marker if stopped (black color)
                    if is_stopped:
                        ax.text(curr_xy[0], curr_xy[1], 's', 
                               fontsize=4, ha='center', va='center', 
                               color='black', weight='bold', zorder=6)
                else:
                    ax.scatter(curr_xy[0], curr_xy[1], color=color, s=6, 
                              marker='o', alpha=1.0, zorder=5, 
                              edgecolors='white', linewidths=0.5,
                              label='Current' if idx == 1 else None)
                    # Add "s" text inside marker if stopped (black color)
                    if is_stopped:
                        ax.text(curr_xy[0], curr_xy[1], 's', 
                               fontsize=3, ha='center', va='center', 
                               color='black', weight='bold', zorder=6)
                
                # Plot actual next position as a small thin arrow from current to actual next
                # Arrow shows where agent actually moves in the next timestep
                dx = actual_next_xy[0] - curr_xy[0]
                dy = actual_next_xy[1] - curr_xy[1]
                arrow_width = 0.3 if is_sdc else 0.2
                head_width = 1.5 if is_sdc else 1.0
                head_length = 1.2 if is_sdc else 0.8
                ax.arrow(curr_xy[0], curr_xy[1], dx, dy, 
                        head_width=head_width, head_length=head_length, 
                        fc='black', ec='black', alpha=0.8, linewidth=arrow_width, 
                        zorder=5, length_includes_head=True,
                        label='Actual Next Step' if idx == 0 else None)
                
                # Plot predicted next position (small hollow circle)
                marker_size = 10 if is_sdc else 6
                linewidth = 1.0 if is_sdc else 0.8
                ax.scatter(pred_next_xy[0], pred_next_xy[1], facecolors='none', edgecolors='black', 
                          s=marker_size, marker='o', linewidths=linewidth, alpha=0.6, zorder=5,
                          label='Predicted Next' if idx == 0 else None)
                
                # Calculate error (distance between actual and predicted NEXT positions)
                error = np.linalg.norm(actual_next_xy - pred_next_xy)
                total_error += error
                error_counts += 1
            
            # Calculate average error for this cell
            cell_errors = []
            for idx, agent_id in enumerate(agent_ids_in_data):
                curr_xy = curr_pos[idx]
                if torch.isnan(curr_xy).any():
                    continue
                actual_next_xy = actual_next_pos[idx].numpy()
                pred_next_xy = pred_next_pos[idx].numpy()
                cell_errors.append(np.linalg.norm(actual_next_xy - pred_next_xy))
            cell_avg_error = np.mean(cell_errors) if cell_errors else 0.0
            
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
    
    # Create custom legend - only show trajectory/line types, not points or map features
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color='tab:blue', linewidth=2, linestyle='-', 
               label='Actual Past Trajectory', alpha=0.8),
        Line2D([0], [0], color='tab:blue', linewidth=2, linestyle='--', 
               label='Actual Future Trajectory', alpha=0.5),
        Line2D([0], [0], color='black', linewidth=2, linestyle='--', 
               label='Predicted Next Step', alpha=0.6),
    ]
    
    # Add legend to the figure (outside the subplots)
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.01, 0.98),
               ncol=3, fontsize=10, framealpha=0.95,
               title='Trajectory Types', title_fontsize=11)
    
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
    print(f"    Grid: {B} graphs × {T} timesteps | Avg Error: {avg_error:.2f}m | {max_nodes_per_graph_viz} agents per graph")
    
    model.train()
    return filepath, avg_error

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