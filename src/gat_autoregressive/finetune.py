"""Fine-tune GAT single-step model for autoregressive multi-step prediction.

This module takes a pre-trained GAT single-step model (from training.py) and fine-tunes it
for autoregressive rollout using scheduled sampling.

Scheduled Sampling:
- Start with teacher forcing (use ground truth as input)
- Gradually increase probability of using model's own predictions
- Helps model learn to handle its own prediction errors
"""

import sys
import os
# Add parent directory (src/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from SpatioTemporalGAT import SpatioTemporalGAT
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, num_gru_layers,
                    input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    gradient_clip_value,
                    num_gpus, use_data_parallel, setup_model_parallel, get_model_for_saving, load_model_state,
                    autoreg_num_rollout_steps, autoreg_num_epochs, autoreg_sampling_strategy,
                    autoreg_visualize_every_n_epochs,
                    pin_memory, prefetch_factor, use_amp,
                    early_stopping_patience, early_stopping_min_delta,
                    cache_validation_data, max_validation_scenarios)
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from helper_functions.visualization_functions import (load_scenario_by_id, draw_map_features,
                                                       MAP_FEATURE_GRAY, VIBRANT_COLORS, SDC_COLOR)
from torch.nn.utils import clip_grad_norm_
import random

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/gat_autoregressive/finetune.py

# GAT-specific directories
CHECKPOINT_DIR = 'checkpoints/gat'
CHECKPOINT_DIR_AUTOREG = 'checkpoints/gat/autoregressive'
VIZ_DIR = 'visualizations/autoreg/gat/finetune'


class EarlyStopping:
    """Early stopping to stop finetuning when validation loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


def load_pretrained_model(checkpoint_path, device):
    """Load pre-trained GAT model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading pre-trained GAT model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model with saved config or use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = SpatioTemporalGAT(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_channels'],
            output_dim=config.get('output_dim', output_dim),
            num_gat_layers=config['num_layers'],
            num_gru_layers=config.get('num_gru_layers', 1),
            dropout=config['dropout'],
            num_heads=config.get('num_heads', 4)
        )
    else:
        # Checkpoint doesn't have config - use default values from config.py
        print("  Warning: Checkpoint missing 'config' key, using default values from config.py")
        model = SpatioTemporalGAT(
            input_dim=input_dim,
            hidden_dim=hidden_channels,
            output_dim=output_dim,
            num_gat_layers=num_layers,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
            num_heads=4  # Default value
        )
    
    # Setup multi-GPU if available
    model, is_parallel = setup_model_parallel(model, device)
    
    # Load weights
    load_model_state(model, checkpoint['model_state_dict'], is_parallel)
    
    print(f"✓ Loaded GAT model from epoch {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Pre-trained val_loss: {checkpoint['val_loss']:.6f}")
    
    return model, is_parallel, checkpoint


def update_graph_with_prediction(graph, pred_displacement, device):
    """
    Update node features based on predicted displacement for next autoregressive step.
    
    The model predicts normalized displacement (dx/100, dy/100).
    Node features expect normalized velocity (vx/30, vy/30) and acceleration (ax/10, ay/10).
    Also updates position-dependent features (relative to SDC).
    """
    updated_graph = graph.clone()
    dt = 0.1  # 0.1 second timestep
    
    # Normalization constants (must match graph_creation_functions.py)
    POSITION_SCALE = 100.0  # displacement normalization
    MAX_SPEED = 30.0  # velocity normalization
    MAX_ACCEL = 10.0  # acceleration normalization
    MAX_DIST_SDC = 100.0  # max distance to SDC for normalization
    
    # Convert normalized displacement to normalized velocity
    velocity_scale = POSITION_SCALE / dt / MAX_SPEED  # = 100 / 0.1 / 30 = 33.33
    
    new_vx_norm = pred_displacement[:, 0] * velocity_scale
    new_vy_norm = pred_displacement[:, 1] * velocity_scale
    
    # Calculate normalized acceleration
    old_vx_norm = updated_graph.x[:, 0]
    old_vy_norm = updated_graph.x[:, 1]
    accel_scale = MAX_SPEED / dt / MAX_ACCEL  # = 30 / 0.1 / 10 = 30
    ax_norm = (new_vx_norm - old_vx_norm) * accel_scale
    ay_norm = (new_vy_norm - old_vy_norm) * accel_scale
    
    # Clamp to reasonable ranges to prevent explosion
    new_vx_norm = torch.clamp(new_vx_norm, -2.0, 2.0)
    new_vy_norm = torch.clamp(new_vy_norm, -2.0, 2.0)
    ax_norm = torch.clamp(ax_norm, -2.0, 2.0)
    ay_norm = torch.clamp(ay_norm, -2.0, 2.0)
    
    # Update velocity/acceleration features
    updated_graph.x[:, 0] = new_vx_norm
    updated_graph.x[:, 1] = new_vy_norm
    updated_graph.x[:, 2] = torch.sqrt(new_vx_norm**2 + new_vy_norm**2)  # normalized speed
    updated_graph.x[:, 3] = torch.atan2(new_vy_norm, new_vx_norm) / np.pi  # heading [-1, 1]
    updated_graph.x[:, 5] = ax_norm
    updated_graph.x[:, 6] = ay_norm
    
    # Update positions and position-dependent features
    if hasattr(updated_graph, 'pos') and updated_graph.pos is not None:
        # pred_displacement is normalized, multiply by 100 to get actual displacement
        actual_displacement = pred_displacement * POSITION_SCALE
        updated_graph.pos = updated_graph.pos + actual_displacement
        
        # Update relative position to SDC (features 7-8) and distance to SDC (feature 9)
        if hasattr(updated_graph, 'batch') and updated_graph.batch is not None:
            batch_ids = updated_graph.batch.unique()
            for batch_id in batch_ids:
                batch_mask = (updated_graph.batch == batch_id)
                batch_positions = updated_graph.pos[batch_mask]
                
                # Assume SDC is the first node in each batch
                sdc_pos = batch_positions[0]
                
                # Calculate relative positions for this batch
                rel_pos = batch_positions - sdc_pos
                dist_to_sdc = torch.sqrt((rel_pos ** 2).sum(dim=1))
                
                # Normalize and clamp
                rel_x_norm = torch.clamp(rel_pos[:, 0] / MAX_DIST_SDC, -1.0, 1.0)
                rel_y_norm = torch.clamp(rel_pos[:, 1] / MAX_DIST_SDC, -1.0, 1.0)
                dist_norm = torch.clamp(dist_to_sdc / MAX_DIST_SDC, 0.0, 1.0)
                
                # Update features for this batch
                batch_indices = torch.where(batch_mask)[0]
                updated_graph.x[batch_indices, 7] = rel_x_norm
                updated_graph.x[batch_indices, 8] = rel_y_norm
                updated_graph.x[batch_indices, 9] = dist_norm
    
    return updated_graph


def scheduled_sampling_probability(epoch, total_epochs, strategy='linear'):
    """Calculate probability of using model's own prediction vs ground truth."""
    progress = epoch / total_epochs
    
    if strategy == 'linear':
        return progress
    elif strategy == 'exponential':
        return 1 - np.exp(-5 * progress)
    elif strategy == 'inverse_sigmoid':
        k = 10
        return 1 / (1 + np.exp(-k * (progress - 0.5)))
    else:
        return progress


def visualize_autoregressive_rollout(model, batch_dict, epoch, num_rollout_steps, device, 
                                     is_parallel, save_dir=None):
    """Visualize autoregressive rollout predictions vs ground truth."""
    if save_dir is None:
        save_dir = VIZ_DIR
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    base_model = model.module if is_parallel else model
    
    batched_graph_sequence = batch_dict["batch"]
    B, T = batch_dict["B"], batch_dict["T"]
    scenario_ids = batch_dict.get("scenario_ids", ["unknown"])
    
    # Load scenario for map features (with timeout to avoid hanging)
    scenario = None
    if scenario_ids and scenario_ids[0] and scenario_ids[0] != "unknown":
        try:
            print(f"  Loading scenario {scenario_ids[0]} for map visualization...")
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Scenario loading timed out")
            
            # Set 10 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            
            scenario = load_scenario_by_id(scenario_ids[0])
            signal.alarm(0)  # Cancel alarm
            
        except (TimeoutError, Exception) as e:
            signal.alarm(0)  # Ensure alarm is cancelled
            print(f"  Warning: Could not load scenario ({type(e).__name__}: {e}). Proceeding without map features.")
    
    # Move to device
    for t in range(T):
        batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
    
    # Reset GRU
    num_nodes = batched_graph_sequence[0].num_nodes
    base_model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
    
    start_t = T // 3
    actual_rollout_steps = min(num_rollout_steps, T - start_t - 1)
    
    if actual_rollout_steps < 1:
        print(f"  [ERROR] Not enough timesteps for rollout. T={T}, start_t={start_t}")
        return None
    
    POSITION_SCALE = 100.0
    
    # Find SDC
    sdc_id = None
    if scenario is not None:
        sdc_track = scenario.tracks[scenario.sdc_track_index]
        sdc_id = sdc_track.id
    
    agent_trajectories = {}
    
    with torch.no_grad():
        # Warm up GRU
        for t in range(start_t):
            graph = batched_graph_sequence[t]
            _ = model(graph.x, graph.edge_index, batch=graph.batch, batch_size=B, batch_num=0, timestep=t)
        
        start_graph = batched_graph_sequence[start_t]
        
        # Filter to only first scenario (batch_idx=0) when B > 1
        if B > 1 and hasattr(start_graph, 'batch'):
            start_batch_mask = (start_graph.batch == 0)
            start_scenario_indices = torch.where(start_batch_mask)[0].cpu().numpy()
        else:
            start_scenario_indices = np.arange(start_graph.num_nodes)
        
        if hasattr(start_graph, 'pos') and start_graph.pos is not None:
            all_start_pos = start_graph.pos.cpu().numpy()
            start_pos = all_start_pos[start_scenario_indices]
        else:
            start_pos = np.zeros((len(start_scenario_indices), 2))
        
        if hasattr(start_graph, 'agent_ids'):
            all_start_agent_ids = list(start_graph.agent_ids)
            start_agent_ids = [all_start_agent_ids[i] for i in start_scenario_indices if i < len(all_start_agent_ids)]
        else:
            start_agent_ids = [i for i in start_scenario_indices]
        
        # Find persistent agents (only from first scenario in batch)
        persistent_agent_ids = set(start_agent_ids)
        for step in range(actual_rollout_steps):
            target_t = start_t + step + 1
            if target_t >= T:
                break
            target_graph = batched_graph_sequence[target_t]
            
            # Filter to only first scenario
            if B > 1 and hasattr(target_graph, 'batch'):
                target_batch_mask = (target_graph.batch == 0)
                target_scenario_indices_check = torch.where(target_batch_mask)[0].cpu().numpy()
            else:
                target_scenario_indices_check = np.arange(target_graph.num_nodes)
            
            if hasattr(target_graph, 'agent_ids'):
                all_target_agent_ids = list(target_graph.agent_ids)
                target_agent_ids_set = set(all_target_agent_ids[i] for i in target_scenario_indices_check if i < len(all_target_agent_ids))
            else:
                target_agent_ids_set = set(target_scenario_indices_check)
            
            persistent_agent_ids = persistent_agent_ids.intersection(target_agent_ids_set)
        
        # Initialize trajectories
        for agent_id in persistent_agent_ids:
            if agent_id in start_agent_ids:
                local_idx = start_agent_ids.index(agent_id)
                agent_trajectories[agent_id] = {
                    'gt': [(start_t, start_pos[local_idx].copy())],
                    'pred': [(start_t, start_pos[local_idx].copy())]
                }
        
        # Collect GT positions (only from first scenario)
        for step in range(actual_rollout_steps):
            target_t = start_t + step + 1
            if target_t >= T:
                break
            target_graph = batched_graph_sequence[target_t]
            
            # Filter to only first scenario
            if B > 1 and hasattr(target_graph, 'batch'):
                target_batch_mask = (target_graph.batch == 0)
                target_scenario_indices = torch.where(target_batch_mask)[0].cpu().numpy()
            else:
                target_scenario_indices = np.arange(target_graph.num_nodes)
            
            if hasattr(target_graph, 'agent_ids'):
                all_target_agent_ids = list(target_graph.agent_ids)
                target_agent_ids = [all_target_agent_ids[i] for i in target_scenario_indices if i < len(all_target_agent_ids)]
            else:
                target_agent_ids = [i for i in target_scenario_indices]
            
            if hasattr(target_graph, 'pos') and target_graph.pos is not None:
                all_target_pos = target_graph.pos.cpu().numpy()
                target_pos = all_target_pos[target_scenario_indices]
            else:
                target_pos = None
            
            for local_idx, agent_id in enumerate(target_agent_ids):
                if agent_id in agent_trajectories:
                    if target_pos is not None:
                        pos = target_pos[local_idx].copy()
                    else:
                        last_pos = agent_trajectories[agent_id]['gt'][-1][1]
                        if target_graph.y is not None and local_idx < target_graph.y.shape[0]:
                            disp = target_graph.y[local_idx].cpu().numpy() * POSITION_SCALE
                            pos = last_pos + disp
                        else:
                            pos = last_pos
                    agent_trajectories[agent_id]['gt'].append((target_t, pos))
        
        # Autoregressive rollout
        agent_pred_positions = {}
        for agent_id in persistent_agent_ids:
            if agent_id in start_agent_ids:
                local_idx = start_agent_ids.index(agent_id)
                agent_pred_positions[agent_id] = start_pos[local_idx].copy()
        
        graph_for_prediction = start_graph
        
        for step in range(actual_rollout_steps):
            target_t = start_t + step + 1
            if target_t >= T:
                break
            
            target_graph = batched_graph_sequence[target_t]
            
            # Filter to only first scenario for predictions
            if B > 1 and hasattr(graph_for_prediction, 'batch'):
                pred_batch_mask = (graph_for_prediction.batch == 0)
                pred_scenario_indices = torch.where(pred_batch_mask)[0].cpu().numpy()
            else:
                pred_scenario_indices = np.arange(graph_for_prediction.num_nodes)
            
            if hasattr(graph_for_prediction, 'agent_ids'):
                all_current_agent_ids = list(graph_for_prediction.agent_ids)
                current_agent_ids = [all_current_agent_ids[i] for i in pred_scenario_indices if i < len(all_current_agent_ids)]
            else:
                current_agent_ids = [i for i in pred_scenario_indices]
            
            # Check node count match (no need to check since we're working with batched graphs that should maintain structure)
            # Note: When B > 1, both graphs have nodes from all scenarios, so total count should match
            # if graph_for_prediction.x.shape[0] != target_graph.num_nodes:
            #     break
            
            # GAT doesn't use edge weights
            pred = model(graph_for_prediction.x, graph_for_prediction.edge_index,
                        batch=graph_for_prediction.batch, batch_size=B, batch_num=0, timestep=start_t + step)
            
            pred_disp = pred.cpu().numpy() * POSITION_SCALE
            
            # DEBUG: Print prediction stats on first step
            if step == 0:
                print(f"  [DEBUG VIZ] Raw pred (normalized): min={pred.min().item():.4f}, max={pred.max().item():.4f}, mean={pred.mean().item():.4f}")
                print(f"  [DEBUG VIZ] Pred disp (meters): min={pred_disp.min():.2f}, max={pred_disp.max():.2f}, mean={pred_disp.mean():.2f}")
                # Compare to ground truth if available
                target_graph = batched_graph_sequence[target_t]
                if target_graph.y is not None:
                    gt = target_graph.y.cpu().numpy() * POSITION_SCALE
                    print(f"  [DEBUG VIZ] GT disp (meters): min={gt.min():.2f}, max={gt.max():.2f}, mean={gt.mean():.2f}")
            
            # Map predictions to persistent agents (only first scenario)
            for local_idx, agent_id in enumerate(current_agent_ids):
                if agent_id in agent_pred_positions:
                    # Get global index in the full batched graph
                    global_idx = pred_scenario_indices[local_idx]
                    if global_idx < pred_disp.shape[0]:
                        new_pos = agent_pred_positions[agent_id] + pred_disp[global_idx]
                        agent_pred_positions[agent_id] = new_pos
                        
                        # DEBUG: Print position updates on first few steps
                        if step < 3 and local_idx == 0:
                            print(f"  [DEBUG VIZ] Step {step}, Agent {agent_id}: disp={pred_disp[global_idx]}, new_pos={new_pos}")
                        agent_trajectories[agent_id]['pred'].append((target_t, new_pos.copy()))
            
            graph_for_prediction = update_graph_with_prediction(graph_for_prediction, pred, device)
    
    # Calculate errors
    horizon_errors = []
    for step in range(actual_rollout_steps):
        target_t = start_t + step + 1
        errors_at_step = []
        for agent_id, traj in agent_trajectories.items():
            gt_at_t = [pos for t, pos in traj['gt'] if t == target_t]
            pred_at_t = [pos for t, pos in traj['pred'] if t == target_t]
            if gt_at_t and pred_at_t:
                error = np.sqrt(((gt_at_t[0] - pred_at_t[0]) ** 2).sum())
                errors_at_step.append(error)
        if errors_at_step:
            horizon_errors.append(np.mean(errors_at_step))
    
    # Select agents to visualize
    max_agents_viz = 10
    agent_movements = {}
    for agent_id, traj in agent_trajectories.items():
        gt_positions_list = sorted(traj['gt'], key=lambda x: x[0])
        total_movement = 0.0
        for i in range(len(gt_positions_list) - 1):
            pos1 = gt_positions_list[i][1]
            pos2 = gt_positions_list[i + 1][1]
            dist = np.sqrt(((pos2 - pos1) ** 2).sum())
            total_movement += dist
        agent_movements[agent_id] = total_movement
    
    selected_agent_ids = []
    if sdc_id is not None and sdc_id in agent_trajectories:
        selected_agent_ids.append(sdc_id)
    
    other_agent_ids = [aid for aid in agent_trajectories.keys() if aid != sdc_id]
    other_agent_ids_sorted = sorted(other_agent_ids, key=lambda aid: agent_movements.get(aid, 0), reverse=True)
    remaining_slots = max_agents_viz - len(selected_agent_ids)
    selected_agent_ids.extend(other_agent_ids_sorted[:remaining_slots])
    
    # Assign colors
    agent_colors = {}
    color_idx = 0
    for agent_id in selected_agent_ids:
        if agent_id == sdc_id:
            agent_colors[agent_id] = SDC_COLOR
        else:
            agent_colors[agent_id] = VIBRANT_COLORS[color_idx % len(VIBRANT_COLORS)]
            color_idx += 1
    
    # Calculate axis limits
    all_x_coords, all_y_coords = [], []
    for agent_id in selected_agent_ids:
        traj = agent_trajectories[agent_id]
        for t, pos in traj['gt']:
            all_x_coords.append(pos[0])
            all_y_coords.append(pos[1])
        for t, pos in traj['pred']:
            all_x_coords.append(pos[0])
            all_y_coords.append(pos[1])
    
    if all_x_coords and all_y_coords:
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)
        x_range, y_range = x_max - x_min, y_max - y_min
        x_padding = max(x_range * 0.2, 15.0)
        y_padding = max(y_range * 0.2, 15.0)
        x_lim = (x_min - x_padding, x_max + x_padding)
        y_lim = (y_min - y_padding, y_max + y_padding)
    else:
        x_lim, y_lim = None, None
    
    final_error = horizon_errors[-1] if horizon_errors else 0
    print(f"  Final horizon error ({actual_rollout_steps * 0.1:.1f}s): {final_error:.2f}m")
    
    # Create visualization
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    
    if scenario is not None:
        map_features_drawn = draw_map_features(ax1, scenario, x_lim, y_lim)
        print(f"  Drew {map_features_drawn} map features")
    
    from matplotlib.lines import Line2D
    agent_legend_elements = []
    
    for agent_id in selected_agent_ids:
        color = agent_colors[agent_id]
        is_sdc = (agent_id == sdc_id)
        agent_label = 'SDC' if is_sdc else f'Agent {agent_id}'
        
        traj = agent_trajectories[agent_id]
        gt_sorted = sorted(traj['gt'], key=lambda x: x[0])
        pred_sorted = sorted(traj['pred'], key=lambda x: x[0])
        
        gt_traj_x = [pos[0] for t, pos in gt_sorted]
        gt_traj_y = [pos[1] for t, pos in gt_sorted]
        pred_traj_x = [pos[0] for t, pos in pred_sorted]
        pred_traj_y = [pos[1] for t, pos in pred_sorted]
        
        if len(pred_traj_x) > 1:
            ax1.plot(pred_traj_x, pred_traj_y, '--', color=color, linewidth=2.5, alpha=0.9, zorder=3)
            # Mark predicted trajectory endpoint (same color, square marker)
            ax1.scatter(pred_traj_x[-1], pred_traj_y[-1], c=color, marker='s', s=50, 
                       edgecolors='black', linewidths=0.5, zorder=6)
        
        if len(gt_traj_x) > 1:
            ax1.plot(gt_traj_x, gt_traj_y, '-', color='black', linewidth=1.5, alpha=0.8, zorder=4)
            # Mark ground truth endpoint (black x marker)
            ax1.scatter(gt_traj_x[-1], gt_traj_y[-1], c='black', marker='x', s=50, zorder=6)
        
        if len(gt_traj_x) > 0:
            ax1.scatter(gt_traj_x[0], gt_traj_y[0], c=color, marker='o', s=40, 
                       edgecolors='black', linewidths=0.5, zorder=5)
        
        if len(agent_legend_elements) < 10:
            agent_legend_elements.append(Line2D([0], [0], color=color, linewidth=2, 
                                                linestyle='--', label=agent_label))
    
    line_legend_elements = [
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', label='Ground Truth'),
        Line2D([0], [0], marker='o', color='gray', markersize=6, linestyle='None', 
               markeredgecolor='black', label='Start position'),
        Line2D([0], [0], marker='s', color='gray', markersize=6, linestyle='None', 
               markeredgecolor='black', label='Predicted endpoint'),
        Line2D([0], [0], marker='x', color='black', markersize=6, linestyle='None', 
               label='GT endpoint'),
    ]
    
    ax1.set_xlabel('X position (meters)', fontsize=11)
    ax1.set_ylabel('Y position (meters)', fontsize=11)
    ax1.set_title(f'GAT Autoregressive Rollout: {actual_rollout_steps} steps ({actual_rollout_steps * 0.1:.1f}s)\n'
                  f'Epoch {epoch+1}', fontsize=12, fontweight='bold')
    
    all_legend_elements = agent_legend_elements + line_legend_elements
    ax1.legend(handles=all_legend_elements, loc='upper left', fontsize=7, framealpha=0.9, ncol=2)
    
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    if x_lim and y_lim:
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_str = scenario_ids[0] if scenario_ids else "unknown"
    filename = f'gat_autoreg_epoch{epoch+1:03d}_{scenario_str}_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    try:
        wandb.log({f"gat_autoreg_visualization": wandb.Image(filepath)})
    except:
        pass
    
    print(f"  → Saved visualization to {filepath}")
    return filepath


def train_epoch_autoregressive(model, dataloader, optimizer, device, 
                               sampling_prob, num_rollout_steps, is_parallel, scaler=None, epoch=0):
    """Train one epoch with scheduled sampling for autoregressive prediction (with optional AMP).
    
    Processes multiple scenarios in parallel using PyG's batching mechanism.
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    steps = 0
    count = 0
    
    base_model = model.module if is_parallel else model
    use_amp_local = scaler is not None and torch.cuda.is_available()
    
    for batch_idx, batch_dict in enumerate(dataloader):
        batched_graph_sequence = batch_dict["batch"]
        B, T = batch_dict["B"], batch_dict["T"]
        
        for t in range(T):
            batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
        
        num_nodes = batched_graph_sequence[0].num_nodes
        num_edges = batched_graph_sequence[0].edge_index.size(1)
        
        # Print parallel execution info for autoregressive rollout (every 20 batches)
        if batch_idx % 20 == 0:
            if torch.cuda.is_available():
                print(f"\n[Batch {batch_idx}] Scenarios: {B} | Total Nodes: {num_nodes} | Edges: {num_edges}")
                print(f"[PARALLEL] Processing {B} scenarios simultaneously | Rollout: {num_rollout_steps} steps")
            else:
                print(f"Batch {batch_idx}: B={B}, Nodes={num_nodes}, Rollout={num_rollout_steps} steps")
        
        base_model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
        
        optimizer.zero_grad()
        accumulated_loss = None
        valid_steps = 0
        
        effective_rollout = min(num_rollout_steps, T - 1)
        if effective_rollout < 1:
            continue
        
        for t in range(max(1, T - effective_rollout)):
            current_graph = batched_graph_sequence[t]
            if current_graph.y is None:
                continue
            
            rollout_loss = None
            graph_for_prediction = current_graph
            
            for step in range(effective_rollout):
                target_t = t + step + 1
                if target_t >= T:
                    break
                
                target_graph = batched_graph_sequence[target_t]
                if target_graph.y is None:
                    break
                
                if graph_for_prediction.x.shape[0] != target_graph.y.shape[0]:
                    graph_for_prediction = target_graph
                    continue
                
                # GAT forward pass with optional AMP - no edge weights
                with torch.amp.autocast('cuda', enabled=use_amp_local):
                    pred = model(
                        graph_for_prediction.x,
                        graph_for_prediction.edge_index,
                        batch=graph_for_prediction.batch,
                        batch_size=B,
                        batch_num=batch_idx,
                        timestep=t + step
                    )
                    
                    target = target_graph.y.to(pred.dtype)
                    
                    mse_loss = F.mse_loss(pred, target)
                    pred_norm = F.normalize(pred, p=2, dim=1, eps=1e-6)
                    target_norm = F.normalize(target, p=2, dim=1, eps=1e-6)
                    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
                    cosine_loss = (1 - cos_sim).mean()
                    
                    discount = 0.95 ** step
                    step_loss = (0.5 * mse_loss + 0.5 * cosine_loss) * discount
                
                if rollout_loss is None:
                    rollout_loss = step_loss
                else:
                    rollout_loss = rollout_loss + step_loss
                
                with torch.no_grad():
                    total_mse += mse_loss.item()
                    total_cosine += cos_sim.mean().item()
                    count += 1
                
                if step < num_rollout_steps - 1:
                    use_prediction = np.random.random() < sampling_prob
                    
                    if use_prediction:
                        graph_for_prediction = update_graph_with_prediction(
                            graph_for_prediction, pred.detach(), device
                        )
                    else:
                        graph_for_prediction = batched_graph_sequence[target_t]
            
            if rollout_loss is not None:
                if accumulated_loss is None:
                    accumulated_loss = rollout_loss
                else:
                    accumulated_loss = accumulated_loss + rollout_loss
                valid_steps += 1
        
        if valid_steps > 0 and accumulated_loss is not None:
            avg_loss = accumulated_loss / valid_steps
            # Backward pass with optional AMP scaling
            if use_amp_local:
                scaler.scale(avg_loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), gradient_clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                avg_loss.backward()
                clip_grad_norm_(model.parameters(), gradient_clip_value)
                optimizer.step()
            total_loss += accumulated_loss.item()
            steps += 1
        
        if batch_idx % 20 == 0:
            loss_val = accumulated_loss.item() if hasattr(accumulated_loss, 'item') else accumulated_loss
            avg_mse_so_far = total_mse / max(1, count)
            avg_cos_so_far = total_cosine / max(1, count)
            rmse_meters = (avg_mse_so_far ** 0.5) * 100.0
            print(f"  Batch {batch_idx}: Loss={loss_val/max(1,valid_steps):.4f}, "
                  f"Sampling prob={sampling_prob:.2f}")
            
            # Log per-step metrics to wandb
            wandb.log({
                "batch": steps,
                "train/batch_epoch": epoch + 1,  # Which epoch this batch is from
                "train/batch_loss": loss_val / max(1, valid_steps),
                "train/batch_mse": avg_mse_so_far,
                "train/batch_rmse_meters": rmse_meters,
                "train/batch_cosine_sim": avg_cos_so_far,
                "train/batch_sampling_prob": sampling_prob,
            })
    
    return {
        'loss': total_loss / max(1, steps),
        'mse': total_mse / max(1, count),
        'cosine_sim': total_cosine / max(1, count)
    }


def evaluate_autoregressive(model, dataloader, device, num_rollout_steps, is_parallel):
    """Evaluate GAT model with full autoregressive rollout (no teacher forcing)."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    steps = 0
    count = 0
    
    horizon_mse = [0.0] * num_rollout_steps
    horizon_cosine = [0.0] * num_rollout_steps
    horizon_counts = [0] * num_rollout_steps
    
    base_model = model.module if is_parallel else model
    
    with torch.no_grad():
        for batch_dict in dataloader:
            batched_graph_sequence = batch_dict["batch"]
            B, T = batch_dict["B"], batch_dict["T"]
            
            for t in range(T):
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
            
            num_nodes = batched_graph_sequence[0].num_nodes
            base_model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
            
            start_t = T // 3
            effective_rollout = min(num_rollout_steps, T - start_t - 1)
            if effective_rollout < 1:
                continue
            
            for t in range(start_t, max(start_t + 1, T - effective_rollout)):
                current_graph = batched_graph_sequence[t]
                if current_graph.y is None:
                    continue
                
                graph_for_prediction = current_graph
                rollout_loss = 0.0
                
                for step in range(effective_rollout):
                    target_t = t + step + 1
                    if target_t >= T:
                        break
                    
                    target_graph = batched_graph_sequence[target_t]
                    if target_graph.y is None:
                        break
                    
                    if graph_for_prediction.x.shape[0] != target_graph.y.shape[0]:
                        graph_for_prediction = target_graph
                        continue
                    
                    # GAT forward - no edge weights
                    pred = model(
                        graph_for_prediction.x,
                        graph_for_prediction.edge_index,
                        batch=graph_for_prediction.batch,
                        batch_size=B,
                        batch_num=0,
                        timestep=t + step
                    )
                    
                    target = target_graph.y.to(pred.dtype)
                    mse = F.mse_loss(pred, target)
                    
                    pred_norm = F.normalize(pred, p=2, dim=1, eps=1e-6)
                    target_norm = F.normalize(target, p=2, dim=1, eps=1e-6)
                    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                    
                    rollout_loss += mse
                    total_mse += mse.item()
                    total_cosine += cos_sim.item()
                    count += 1
                    
                    horizon_mse[step] += mse.item()
                    horizon_cosine[step] += cos_sim.item()
                    horizon_counts[step] += 1
                    
                    if step < num_rollout_steps - 1:
                        graph_for_prediction = update_graph_with_prediction(
                            graph_for_prediction, pred, device
                        )
                
                total_loss += rollout_loss.item() if isinstance(rollout_loss, torch.Tensor) else rollout_loss
                steps += 1
    
    horizon_avg_mse = [horizon_mse[h] / max(1, horizon_counts[h]) for h in range(num_rollout_steps)]
    horizon_avg_cosine = [horizon_cosine[h] / max(1, horizon_counts[h]) for h in range(num_rollout_steps)]
    
    return {
        'loss': total_loss / max(1, steps),
        'mse': total_mse / max(1, count),
        'cosine_sim': total_cosine / max(1, count),
        'horizon_mse': horizon_avg_mse,
        'horizon_cosine': horizon_avg_cosine
    }


def run_autoregressive_finetuning(
    pretrained_checkpoint="./checkpoints/gat/best_model.pt",
    pretrained_checkpoint_2="./checkpoints/gat/best_model_batched.pt",
    train_path="./data/graphs/training/training_seqlen90.hdf5",
    val_path="./data/graphs/validation/validation_seqlen90.hdf5",
    num_rollout_steps=5,
    num_epochs=50,
    sampling_strategy='linear'
):
    """Fine-tune pre-trained GAT model for autoregressive multi-step prediction."""
    
    # Load pre-trained model
    if pretrained_checkpoint is not None and os.path.exists(pretrained_checkpoint):
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint, device)
    elif pretrained_checkpoint_2 is not None and os.path.exists(pretrained_checkpoint_2):
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint_2, device)
    else:
        raise FileNotFoundError(
            f"No valid checkpoint found. Checked:\n"
            f"  - {pretrained_checkpoint}\n"
            f"  - {pretrained_checkpoint_2}"
        )


    # Create directories
    os.makedirs(CHECKPOINT_DIR_AUTOREG, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # Initialize wandb
    wandb.login()
    run = wandb.init(
        project=project_name,
        config={
            "model": "SpatioTemporalGAT_Autoregressive",
            "pretrained_from": pretrained_checkpoint,
            "batch_size": batch_size,
            "learning_rate": learning_rate * 0.1,
            "dataset": dataset_name,
            "num_rollout_steps": num_rollout_steps,
            "prediction_horizon": f"{num_rollout_steps * 0.1}s",
            "sampling_strategy": sampling_strategy,
            "epochs": num_epochs,
            "use_gat": True,
            "early_stopping_patience": early_stopping_patience
        },
        name=f"GAT_Autoregressive_finetune_{num_rollout_steps}steps",
        dir="../wandb"
    )
    
    wandb.watch(model, log='all', log_freq=10)
    
    # Define custom x-axes for wandb metrics
    wandb.define_metric("epoch")
    wandb.define_metric("batch")  # Global batch counter across all epochs
    
    # Epoch-based metrics (plotted vs epoch)
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("train_mse", step_metric="epoch")
    wandb.define_metric("train_cosine_sim", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("val_mse", step_metric="epoch")
    wandb.define_metric("val_cosine_sim", step_metric="epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")
    wandb.define_metric("sampling_probability", step_metric="epoch")
    wandb.define_metric("val_mse_horizon_*", step_metric="epoch")
    wandb.define_metric("val_cosine_horizon_*", step_metric="epoch")
    
    # Batch-based metrics (plotted vs batch - fine-grained progress)
    wandb.define_metric("train/batch_loss", step_metric="batch")
    wandb.define_metric("train/batch_mse", step_metric="batch")
    wandb.define_metric("train/batch_rmse_meters", step_metric="batch")
    wandb.define_metric("train/batch_cosine_sim", step_metric="batch")
    wandb.define_metric("train/batch_sampling_prob", step_metric="batch")
    wandb.define_metric("train/batch_epoch", step_metric="batch")  # Which epoch this batch belongs to
    
    finetune_lr = learning_rate * 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                   min_lr=1e-7)
    
    # Initialize early stopping
    early_stopper = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        verbose=True
    )
    
    # Load datasets
    try:
        train_dataset = HDF5ScenarioDataset(train_path, seq_len=sequence_length)
        print(f"Loaded training dataset: {len(train_dataset)} scenarios")
    except FileNotFoundError:
        print(f"ERROR: {train_path} not found!")
        run.finish()
        return None
    
    # Load validation dataset with optional caching and subsampling
    val_dataset = None
    try:
        val_dataset_full = HDF5ScenarioDataset(
            val_path, 
            seq_len=sequence_length,
            cache_in_memory=cache_validation_data,
            max_cache_size=max_validation_scenarios
        )
        
        # Subsample validation set for faster evaluation during finetuning
        max_val_scenarios = max_validation_scenarios  # From config.py
        if len(val_dataset_full) > max_val_scenarios:
            val_indices = random.sample(range(len(val_dataset_full)), max_val_scenarios)
            val_dataset = Subset(val_dataset_full, val_indices)
            print(f"Loaded validation dataset: {max_val_scenarios} scenarios (subsampled from {len(val_dataset_full)})")
        else:
            val_dataset = val_dataset_full
            print(f"Loaded validation dataset: {len(val_dataset)} scenarios")
        
        if cache_validation_data:
            print(f"  [Cache] Validation caching enabled for faster evaluation")
    except FileNotFoundError:
        print(f"WARNING: {val_path} not found! Finetuning without validation.")
        val_dataset = None
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_graph_sequences_to_batch,
        drop_last=True, persistent_workers=False,
        pin_memory=pin_memory, prefetch_factor=None
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_graph_sequences_to_batch,
            drop_last=True, persistent_workers=False,
            pin_memory=pin_memory, prefetch_factor=None
        )
    
    print(f"\n{'='*80}")
    print(f"GAT AUTOREGRESSIVE FINE-TUNING (BATCHED)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Pre-trained model: {pretrained_checkpoint}")
    print(f"Batch size: {batch_size} scenarios processed in parallel")
    print(f"Rollout steps: {num_rollout_steps} ({num_rollout_steps * 0.1}s horizon)")
    print(f"Sampling strategy: {sampling_strategy}")
    print(f"Fine-tuning LR: {finetune_lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    print(f"Validation: {'Enabled' if val_loader else 'Disabled'}")
    print(f"Mixed Precision (AMP): {'Enabled' if use_amp and torch.cuda.is_available() else 'Disabled'}")
    print(f"Checkpoints: {CHECKPOINT_DIR_AUTOREG}")
    print(f"Visualizations: {VIZ_DIR}")
    print(f"{'='*80}\n")
    
    # Initialize GradScaler for AMP (only if enabled and CUDA available)
    scaler = None
    if use_amp and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        print("[AMP] Mixed Precision Training ENABLED - using float16 for forward pass")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        sampling_prob = scheduled_sampling_probability(epoch, num_epochs, sampling_strategy)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} (sampling_prob={sampling_prob:.3f})")
        
        train_metrics = train_epoch_autoregressive(
            model, train_loader, optimizer, device, 
            sampling_prob, num_rollout_steps, is_parallel, scaler=scaler, epoch=epoch
        )
        
        val_metrics = None
        if val_loader:
            val_metrics = evaluate_autoregressive(
                model, val_loader, device, num_rollout_steps, is_parallel
            )
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step(train_metrics['loss'])
        
        current_lr = optimizer.param_groups[0]['lr']
        
        log_dict = {
            "epoch": epoch,
            "sampling_probability": sampling_prob,
            "train_loss": train_metrics['loss'],
            "train_mse": train_metrics['mse'],
            "train_cosine_sim": train_metrics['cosine_sim'],
            "learning_rate": current_lr
        }
        
        if val_metrics:
            log_dict.update({
                "val_loss": val_metrics['loss'],
                "val_mse": val_metrics['mse'],
                "val_cosine_sim": val_metrics['cosine_sim']
            })
            for h in range(num_rollout_steps):
                log_dict[f"val_mse_horizon_{h+1}"] = val_metrics['horizon_mse'][h]
                log_dict[f"val_cosine_horizon_{h+1}"] = val_metrics['horizon_cosine'][h]
        
        wandb.log(log_dict)
        
        if val_metrics:
            horizon_str = " | ".join([f"H{h+1}:{val_metrics['horizon_mse'][h]:.4f}" 
                                      for h in range(num_rollout_steps)])
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Train Cos: {train_metrics['cosine_sim']:.4f} | Val Cos: {val_metrics['cosine_sim']:.4f}")
            print(f"  Per-horizon MSE: {horizon_str}")
            
            # Early stopping check
            early_stopper(val_metrics['loss'])
            
            if (epoch % autoreg_visualize_every_n_epochs == 0) or (epoch == num_epochs - 1) or (epoch == 0):
                try:
                    viz_batch = next(iter(val_loader))
                    visualize_autoregressive_rollout(
                        model, viz_batch, epoch, num_rollout_steps, 
                        device, is_parallel, save_dir=VIZ_DIR
                    )
                except Exception as e:
                    print(f"  Warning: Visualization failed: {e}")
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_filename = f'best_gat_autoreg_{num_rollout_steps}step_B{batch_size}_{sampling_strategy}_E{num_epochs}.pt'
                save_path = os.path.join(CHECKPOINT_DIR_AUTOREG, save_filename)
                model_to_save = get_model_for_saving(model, is_parallel)
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'train_loss': train_metrics['loss'],
                    'num_rollout_steps': num_rollout_steps,
                    'sampling_strategy': sampling_strategy
                }
                # Preserve original config if it exists
                if 'config' in checkpoint:
                    checkpoint_data['config'] = checkpoint['config']
                torch.save(checkpoint_data, save_path)
                print(f"  → Saved best model (val_loss: {val_metrics['loss']:.4f})")
            
            # Check early stopping
            if early_stopper.early_stop:
                print(f"\n Early stopping triggered at epoch {epoch+1}!")
                break
        else:
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Cos: {train_metrics['cosine_sim']:.4f}")
            # Early stopping on train loss if no validation
            early_stopper(train_metrics['loss'])
            if early_stopper.early_stop:
                print(f"\n Early stopping triggered at epoch {epoch+1}!")
                break
    
    # Save final model
    final_filename = f'final_gat_autoreg_{num_rollout_steps}step_B{batch_size}_{sampling_strategy}_E{num_epochs}.pt'
    final_path = os.path.join(CHECKPOINT_DIR_AUTOREG, final_filename)
    model_to_save = get_model_for_saving(model, is_parallel)
    final_checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_rollout_steps': num_rollout_steps,
        'early_stopped': early_stopper.early_stop
    }
    # Preserve original config if it exists
    if 'config' in checkpoint:
        final_checkpoint_data['config'] = checkpoint['config']
    torch.save(final_checkpoint_data, final_path)
    
    print(f"\n{'='*80}")
    print(f"GAT FINE-TUNING COMPLETE!")
    print(f"{'='*80}")
    best_filename = f'best_gat_autoreg_{num_rollout_steps}step_B{batch_size}_{sampling_strategy}_E{num_epochs}.pt'
    print(f"Best model: {os.path.join(CHECKPOINT_DIR_AUTOREG, best_filename)}")
    print(f"Final model: {final_path}")
    if val_metrics:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Early stopped: {early_stopper.early_stop}")
    print(f"{'='*80}\n")
    
    run.finish()
    return model


if __name__ == '__main__':
    model = run_autoregressive_finetuning(
        pretrained_checkpoint=None,  # Uses checkpoints/gat/best_model.pt by default
        num_rollout_steps=autoreg_num_rollout_steps,
        num_epochs=autoreg_num_epochs,
        sampling_strategy=autoreg_sampling_strategy
    )
