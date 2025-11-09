
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from datetime import datetime
from helper_functions.graph_creation_functions import timestep_to_pyg_data
from config import sequence_length, radius, graph_creation_method, viz_scenario_dir, viz_training_dir, visualize_every_n_epochs, max_nodes_per_graph_viz, show_timesteps_viz
import matplotlib.pyplot as plt

def visualize_training_progress(model, batch_dict, epoch, scenario=None, save_dir=viz_training_dir,
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
                ax.text(0.5, 1.25, f'Timestep {t}', transform=ax.transAxes,
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
    
    lane_type_colors = {
        0: '#95a5a6', 1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'
    }
    
    for plot_idx, t in enumerate(timestep_indices):
        ax = axes[plot_idx]
        graph = graph_sequence[t]
        
        for feature in scenario.map_features:       # Render map features (lightweight)
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

def visualize_epoch(epoch, viz_batch, model, viz_scenario, device, wandb):
    should_visualize = (epoch + 1) % visualize_every_n_epochs == 0 or epoch == 0
        
    if should_visualize and viz_batch is not None:
        print(f"  Generating visualization of first batch for epoch {epoch+1}...")
        try:
            filepath, avg_error = visualize_training_progress(
                model, viz_batch, epoch=epoch+1,
                scenario=viz_scenario,  # Pass scenario for map features
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