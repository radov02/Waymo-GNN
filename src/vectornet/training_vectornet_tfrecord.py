"""VectorNet training with TFRecord dataset. Predicts full trajectory at once
python src/vectornet/training_vectornet_tfrecord.py --data_dir data/scenario
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import wandb
import warnings
import logging
import argparse
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.nn.utils import clip_grad_norm_
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'helper_functions'))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime as dt
from helper_functions.visualization_functions import VIBRANT_COLORS, SDC_COLOR, MAP_FEATURE_GRAY
# suppress warnings
warnings.filterwarnings("ignore", message=".*skipping cudagraphs.*")
warnings.filterwarnings("ignore", message=".*cudagraph.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
# multiprocessing setup
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
from VectorNet import VectorNetTFRecord, AGENT_VECTOR_DIM, MAP_VECTOR_DIM
from vectornet_tfrecord_dataset import (VectorNetTFRecordDataset, vectornet_collate_fn)
from config import (
    device, vectornet_prefetch_factor,
    use_amp, use_bf16, use_torch_compile, torch_compile_mode,
    vectornet_dropout, vectornet_prediction_horizon, vectornet_history_length,
    vectornet_num_agents_to_predict, vectornet_min_lr,
    vectornet_loss_alpha, vectornet_loss_beta, vectornet_loss_gamma, vectornet_loss_delta,
    vectornet_checkpoint_dir, vectornet_viz_dir, vectornet_visualize_every_n_epochs, 
    vectornet_wandb_project, vectornet_wandb_name, vectornet_gradient_clip)

def parse_args():
    parser = argparse.ArgumentParser(description='Train VectorNet on TFRecord data')

    parser.add_argument('--data_dir', type=str, default=None,
                       help='Base directory for TFRecord data (default: src/vectornet/data)')
    parser.add_argument('--max_train', type=int, default=None,
                       help='Max training scenarios (None = all)')
    parser.add_argument('--max_val', type=int, default=1000,
                       help='Max validation scenarios')

    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_polyline_layers', type=int, default=3)
    parser.add_argument('--num_global_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Resume from checkpoint')
    return parser.parse_args()

def multistep_trajectory_loss(pred, target, valid_mask=None, 
                              alpha=0.2, beta=0.5, gamma=0.1, delta=0.2):
    """loss function for multi-step trajectory prediction, using:
        pred: Predictions [B, T, 2] - predicted positions
        target: Targets [B, T, 2] - ground truth positions
        valid_mask: [B, T] validity mask
        alpha: Weight for angular error
        beta: Weight for MSE
        gamma: Weight for velocity consistency
        delta: Weight for endpoint error
    and returning combined loss"""
    
    if valid_mask is not None:  # apply validity mask if provided
        mask = valid_mask.unsqueeze(-1).expand_as(pred)
        pred = pred * mask
        target = target * mask
    
    
    mse = F.mse_loss(pred, target)      # MSE loss
    pred_flat = pred.reshape(-1, 2)
    target_flat = target.reshape(-1, 2)
    pred_angle = torch.atan2(pred_flat[:, 1], pred_flat[:, 0] + 1e-6)
    target_angle = torch.atan2(target_flat[:, 1], target_flat[:, 0] + 1e-6)
    angle_diff = torch.atan2(torch.sin(pred_angle - target_angle),torch.cos(pred_angle - target_angle))
    angular_loss = torch.mean(torch.abs(angle_diff))        # Angular/direction loss
    if pred.shape[1] > 1:       # Velocity consistency (smooth trajectory)
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        target_vel = target[:, 1:, :] - target[:, :-1, :]
        velocity_loss = F.mse_loss(pred_vel, target_vel)
    else:
        velocity_loss = torch.tensor(0.0, device=pred.device)
    endpoint_loss = F.mse_loss(pred[:, -1, :], target[:, -1, :])    # Endpoint error (final displacement error)
    
    total_loss = (
        beta * mse +
        alpha * angular_loss +
        gamma * velocity_loss +
        delta * endpoint_loss
    )
    return total_loss

def compute_metrics(pred, target, valid_mask=None):
    """Compute ADE (Average Displacement Error) and FDE (Final Displacement Error) metrics using:
        pred: [B, T, 2] predictions
        target: [B, T, 2] ground truth
        valid_mask: [B, T] optional validity mask"""
    # displacement at each timestep:
    disp = torch.norm(pred - target, dim=-1)  # [B, T]
    
    if valid_mask is not None:
        # masked ADE
        valid_count = valid_mask.sum()
        if valid_count > 0:
            ade = (disp * valid_mask).sum() / valid_count
        else:
            ade = torch.tensor(0.0, device=pred.device)
        
        # FDE: last valid position per sample, for simplicity, just use last timestep 
        fde = disp[:, -1].mean()
    else:
        ade = disp.mean()
        fde = disp[:, -1].mean()
    
    return ade, fde

def visualize_vectornet_predictions(predictions, targets, valid_mask, scenario_ids,
                                     agent_vectors, agent_polyline_ids, agent_batch_idx,
                                     map_vectors, map_polyline_ids, map_batch_idx,
                                     target_polyline_indices, target_scenario_batch, num_targets_per_scenario,
                                     epoch, output_dir, max_scenarios=2):
    """visualize VectorNet trajectory predictions with map features and multi-agent support using:
        predictions: [total_targets, T, 2] predicted positions for all targets
        targets: [total_targets, T, 2] ground truth positions for all targets
        valid_mask: [total_targets, T] validity mask for all targets
        scenario_ids: List of scenario IDs (length = batch_size)
        agent_vectors: [N_agents, 16] agent polyline vectors
        agent_polyline_ids: [N_agents] polyline IDs for agents
        agent_batch_idx: [N_agents] batch index for each agent vector
        map_vectors: [N_map, 13] map polyline vectors
        map_polyline_ids: [N_map] polyline IDs for map features
        map_batch_idx: [N_map] batch index for each map vector
        target_polyline_indices: [total_targets] polyline index for each target agent
        target_scenario_batch: [total_targets] which scenario each target belongs to
        num_targets_per_scenario: [batch_size] how many targets per scenario
        epoch: Current epoch number
        output_dir: Directory to save visualizations
        max_scenarios: Maximum number of scenarios to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle numpy/tensor conversion
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    if torch.is_tensor(valid_mask):
        valid_mask = valid_mask.cpu().numpy()
    if torch.is_tensor(target_scenario_batch):
        target_scenario_batch = target_scenario_batch.cpu().numpy()
    if torch.is_tensor(num_targets_per_scenario):
        num_targets_per_scenario = num_targets_per_scenario.cpu().numpy()
    if torch.is_tensor(target_polyline_indices):
        target_polyline_indices = target_polyline_indices.cpu().numpy()
    if torch.is_tensor(agent_vectors):
        agent_vectors = agent_vectors.cpu().numpy()
    if torch.is_tensor(agent_polyline_ids):
        agent_polyline_ids = agent_polyline_ids.cpu().numpy()
    if torch.is_tensor(agent_batch_idx):
        agent_batch_idx = agent_batch_idx.cpu().numpy()
    if torch.is_tensor(map_vectors):
        map_vectors = map_vectors.cpu().numpy()
    if torch.is_tensor(map_polyline_ids):
        map_polyline_ids = map_polyline_ids.cpu().numpy()
    if torch.is_tensor(map_batch_idx):
        map_batch_idx = map_batch_idx.cpu().numpy()
    
    batch_size = len(scenario_ids)
    num_scenarios_to_viz = min(batch_size, max_scenarios)
    T = predictions.shape[1]
    
    # Map feature type colors (from one-hot encoding in map_vectors)
    # type_onehot is at indices 6:13 (7 types)
    # Using MAP_FEATURE_GRAY for most features, red for stop signs (same as finetune.py)
    MAP_COLORS = {
        0: (MAP_FEATURE_GRAY, 'Lanes'),        # lane - gray
        1: (MAP_FEATURE_GRAY, 'Road Lines'),   # road_line - gray
        2: (MAP_FEATURE_GRAY, 'Road Edges'),   # road_edge - gray
        3: ('#FF0000', 'Stop Signs'),          # stop_sign - red
        4: (MAP_FEATURE_GRAY, 'Crosswalks'),   # crosswalk - gray
        5: (MAP_FEATURE_GRAY, 'Speed Bumps'),  # speed_bump - gray
        6: (MAP_FEATURE_GRAY, 'Driveways'),    # driveway - gray
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_scenarios_to_viz, figsize=(10*num_scenarios_to_viz, 10))
    if num_scenarios_to_viz == 1:
        axes = [axes]
    
    for scenario_idx in range(num_scenarios_to_viz):
        ax = axes[scenario_idx]
        
        # Get target indices for this scenario
        target_mask = target_scenario_batch == scenario_idx
        scenario_predictions = predictions[target_mask]  # [num_targets_this_scenario, T, 2]
        scenario_targets = targets[target_mask]           # [num_targets_this_scenario, T, 2]
        scenario_valid = valid_mask[target_mask]          # [num_targets_this_scenario, T]
        scenario_target_polylines = target_polyline_indices[target_mask]  # [num_targets_this_scenario]
        num_targets = scenario_predictions.shape[0]
        
        # Get current positions for each target agent from their polylines
        # Current position = end point of last vector in agent's polyline
        scenario_agent_mask = (agent_batch_idx == scenario_idx)
        scenario_agent_vectors = agent_vectors[scenario_agent_mask]
        scenario_agent_polyline_ids = agent_polyline_ids[scenario_agent_mask]
        
        current_positions = []
        for t_idx in range(num_targets):
            target_poly_id = scenario_target_polylines[t_idx]
            # Find vectors belonging to this polyline
            poly_mask = scenario_agent_polyline_ids == target_poly_id
            poly_vectors = scenario_agent_vectors[poly_mask]
            if len(poly_vectors) > 0:
                # Current position is de_x, de_y of last vector (indices 3, 4)
                current_pos = poly_vectors[-1, 3:5]  # [de_x, de_y]
                current_positions.append(current_pos)
            else:
                # Fallback to first valid target position
                current_positions.append(scenario_targets[t_idx, 0, :2])
        current_positions = np.array(current_positions)  # [num_targets, 2]
        
        # ===== Plot Map Features =====
        if map_batch_idx is not None and len(map_batch_idx) > 0:
            scenario_map_mask = (map_batch_idx == scenario_idx)
            scenario_map_vectors = map_vectors[scenario_map_mask]
            scenario_map_polyline_ids = map_polyline_ids[scenario_map_mask]
            
            # Group by polyline ID and plot
            unique_polylines = np.unique(scenario_map_polyline_ids)
            plotted_types = set()
            
            for poly_id in unique_polylines:
                poly_mask = scenario_map_polyline_ids == poly_id
                poly_vectors = scenario_map_vectors[poly_mask]
                
                if len(poly_vectors) == 0:
                    continue
                
                # Get type from one-hot (indices 6:13)
                type_onehot = poly_vectors[0, 6:13]
                type_idx = np.argmax(type_onehot)
                color, label = MAP_COLORS.get(type_idx, ('#CCCCCC', 'Unknown'))
                
                # Extract connected polyline points
                # Each vector is a segment: (ds_x, ds_y) -> (de_x, de_y)
                # Build continuous path by following start->end of each segment
                points = []
                for i, vec in enumerate(poly_vectors):
                    if i == 0:
                        points.append([vec[0], vec[1]])  # First start point
                    points.append([vec[3], vec[4]])  # End point of each segment
                points = np.array(points)
                
                show_label = type_idx not in plotted_types
                ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5, 
                       alpha=0.6, label=label if show_label else None, zorder=1)
                plotted_types.add(type_idx)
        
        # ===== Agent Histories Removed =====
        # Previously plotted green trajectories - removed for cleaner visualization
        
        # ===== Plot Multi-Agent Predictions and Ground Truth =====
        total_ade = 0.0
        total_fde = 0.0
        
        # Build legend elements
        agent_legend_elements = []
        
        for t_idx in range(num_targets):
            pred = scenario_predictions[t_idx]  # [T, 2]
            tgt = scenario_targets[t_idx]       # [T, 2]
            mask = scenario_valid[t_idx]        # [T]
            
            valid_idx = mask > 0.5
            pred_valid = pred[valid_idx]
            tgt_valid = tgt[valid_idx]
            
            if len(pred_valid) == 0:
                continue
            
            # Color for this agent - use VIBRANT_COLORS (same as finetune.py)
            # First agent (SDC) gets SDC_COLOR, others get VIBRANT_COLORS
            if t_idx == 0:
                agent_color = SDC_COLOR
                agent_label = 'SDC (Pred)'
            else:
                color_idx = (t_idx - 1) % len(VIBRANT_COLORS)
                agent_color = VIBRANT_COLORS[color_idx]
                agent_label = f'Agent {t_idx+1} (Pred)'
            
            # Get current position for this agent
            curr_pos = current_positions[t_idx]
            
            # Build complete trajectories starting from current position
            gt_traj = np.vstack([curr_pos.reshape(1, 2), tgt_valid])  # [T+1, 2]
            pred_traj = np.vstack([curr_pos.reshape(1, 2), pred_valid])  # [T+1, 2]
            
            # Plot Ground Truth (solid black line, same as finetune.py)
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], '-', color='black', 
                    linewidth=1.5, alpha=0.8, zorder=10,
                    label='Ground Truth' if t_idx == 0 else None)
            
            # Plot start position (circle marker at current position) - smaller
            ax.scatter(curr_pos[0], curr_pos[1], c=agent_color, s=35, 
                       marker='o', zorder=15, edgecolors='black', linewidth=0.5,
                       label=None)
            
            # Plot GT endpoint (x marker in black)
            ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='black', s=60, 
                       marker='x', zorder=15, linewidth=2,
                       label=None)
            
            # Plot Prediction (dashed line with agent color)
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], '--', color=agent_color,
                    linewidth=2, alpha=0.9, zorder=11)
            
            # Plot predicted endpoint (square marker) - smaller and more transparent
            ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], c=agent_color, s=30, 
                       marker='s', zorder=16, alpha=0.5, edgecolors='black', linewidth=0.5)
            
            # Add to legend
            agent_legend_elements.append(
                Line2D([0], [0], color=agent_color, linewidth=2, linestyle='--',
                       marker='s', markersize=6, label=agent_label)
            )
            
            # Compute metrics
            disp = np.linalg.norm(pred_valid - tgt_valid, axis=1)
            total_ade += disp.mean()
            total_fde += disp[-1] if len(disp) > 0 else 0
        
        avg_ade = total_ade / max(1, num_targets)
        avg_fde = total_fde / max(1, num_targets)
        
        # Get scenario ID
        sid = scenario_ids[scenario_idx] if scenario_idx < len(scenario_ids) else f"scenario_{scenario_idx}"
        if isinstance(sid, bytes):
            sid = sid.decode('utf-8')
        
        ax.set_title(f'Scenario: {sid[:25]}...\n{num_targets} agents | Avg ADE: {avg_ade:.2f}m | Avg FDE: {avg_fde:.2f}m', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X position (meters)', fontsize=11)
        ax.set_ylabel('Y position (meters)', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # Build complete legend with line type elements (same as finetune.py)
        line_legend_elements = [
            Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', label='Ground Truth'),
            Line2D([0], [0], marker='o', color='gray', markersize=6, linestyle='None', 
                   markeredgecolor='black', label='Start position'),
            Line2D([0], [0], marker='s', color='gray', markersize=6, linestyle='None', 
                   markeredgecolor='black', label='Predicted endpoint'),
            Line2D([0], [0], marker='x', color='black', markersize=6, linestyle='None', 
                   label='GT endpoint'),
        ]
        
        all_legend_elements = agent_legend_elements + line_legend_elements
        ax.legend(handles=all_legend_elements, loc='upper left', fontsize=7, framealpha=0.9, ncol=2)
    
    plt.suptitle(f'VectorNet Multi-Agent Predictions - Epoch {epoch+1}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
    filename = f'vectornet_epoch{epoch+1:03d}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Log visualization to wandb (per-epoch)
    try:
        wandb.log({f"vectornet_visualization": wandb.Image(filepath)})
    except:
        pass
    
    print(f"  Saved visualization: {filepath}")
    return filepath

def train_epoch(model, dataloader, optimizer, loss_alpha, loss_beta, loss_gamma, loss_delta,
                gradient_clip, scaler=None, visualize_callback=None):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0.0
    total_traj_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    steps = 0
    last_batch = None
    
    amp_enabled = use_amp and torch.cuda.is_available()
    amp_dtype = torch.float16 if not use_bf16 else torch.bfloat16
    accumulation_steps = 1
    
    log_every = max(1, len(dataloader) // 10)
    
    optimizer.zero_grad(set_to_none=True)   # zero gradients at start
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % log_every == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Agents: {batch['agent_vectors'].shape[0]} | "
                  f"Map: {batch['map_vectors'].shape[0]}")
        
        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
            
            predictions = model(batch)  # forward pass, get [B, future_len, 2]
            
            # Get targets:
            targets = batch['future_positions'].to(predictions.device)  # [B, future_len, 2]
            valid_mask = batch['future_valid'].to(predictions.device)   # [B, future_len]
            
            # use trajectory loss
            traj_loss = multistep_trajectory_loss(
                predictions, targets, valid_mask,
                alpha=loss_alpha,
                beta=loss_beta,
                gamma=loss_gamma,
                delta=loss_delta
            )
            
            loss = traj_loss
            loss = loss / accumulation_steps      # scale loss for gradient accumulation
            if torch.isnan(loss):
                print(f"  Warning: NaN loss at batch {batch_idx}")
                continue
        
        # backward pass (accumulate gradients)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step optimizer every accumulation_steps batches:
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)   # zero gradients after optimizer step
        
        with torch.no_grad():
            ade, fde = compute_metrics(predictions, targets, valid_mask)    # compute metrics using unscaled loss
        
        total_loss += (loss.item() * accumulation_steps)  # unscale for logging
        total_traj_loss += traj_loss.item()
        total_ade += ade.item()
        total_fde += fde.item()
        steps += 1
        
        # store last batch for visualization:
        if batch_idx == len(dataloader) - 1:
            last_batch = {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    metrics = {
        'loss': total_loss / max(1, steps),
        'traj_loss': total_traj_loss / max(1, steps),
        'ade': total_ade / max(1, steps),
        'fde': total_fde / max(1, steps),
    }
    
    # call visualization callback after epoch completes
    if visualize_callback is not None and last_batch is not None:
        try:
            visualize_callback(last_batch)
        except Exception as e:
            print(f"  WARNING: Visualization failed: {e}")
    return metrics

@torch.no_grad()
def validate(model, dataloader, loss_alpha, loss_beta, loss_gamma, loss_delta):
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    steps = 0
    
    amp_enabled = use_amp and torch.cuda.is_available()
    amp_dtype = torch.float16 if not use_bf16 else torch.bfloat16
    
    for batch in dataloader:
        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
            predictions = model(batch)
            targets = batch['future_positions'].to(predictions.device)
            valid_mask = batch['future_valid'].to(predictions.device)
            
            loss = multistep_trajectory_loss(
                predictions, targets, valid_mask,
                alpha=loss_alpha,
                beta=loss_beta,
                gamma=loss_gamma,
                delta=loss_delta
            )
            
            ade, fde = compute_metrics(predictions, targets, valid_mask)
        
        total_loss += loss.item()
        total_ade += ade.item()
        total_fde += fde.item()
        steps += 1
    
    return {
        'loss': total_loss / max(1, steps),
        'ade': total_ade / max(1, steps),
        'fde': total_fde / max(1, steps),
    }

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle DataParallel
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})

def main():
    """main training loop"""
    args = parse_args()
    data_dir = args.data_dir if args.data_dir else os.path.join(os.path.dirname(__file__), 'data')
    max_train_scenarios = args.max_train
    max_val_scenarios = args.max_val
    hidden_dim = args.hidden_dim
    num_polyline_layers = args.num_polyline_layers
    num_global_layers = args.num_global_layers
    num_heads = args.num_heads
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    num_workers = args.num_workers
    
    train_device = torch.device(str(device))
    print(f"Using device: {train_device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if not args.no_wandb:
        run_name = vectornet_wandb_name or f"vectornet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=vectornet_wandb_project,
            name=run_name,
            config={
                'data_dir': data_dir,
                'hidden_dim': hidden_dim,
                'num_polyline_layers': num_polyline_layers,
                'num_global_layers': num_global_layers,
                'num_heads': num_heads,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'num_workers': num_workers,
                'max_train_scenarios': max_train_scenarios,
                'max_val_scenarios': max_val_scenarios,
            },
        )
    else:
        wandb.init(mode='disabled')
    

    print("\n" + "="*60)
    print("Loading TFRecord datasets...")
    print("="*60)
    train_dataset = VectorNetTFRecordDataset(
        tfrecord_dir=data_dir,
        split='training',
        history_len=vectornet_history_length,
        future_len=vectornet_prediction_horizon,
        max_scenarios=max_train_scenarios,
        num_agents_to_predict=vectornet_num_agents_to_predict,
    )
    val_dataset = VectorNetTFRecordDataset(
        tfrecord_dir=data_dir,
        split='validation',
        history_len=vectornet_history_length,
        future_len=vectornet_prediction_horizon,
        max_scenarios=max_val_scenarios,
        num_agents_to_predict=vectornet_num_agents_to_predict,
    )
    print(f"Training scenarios: {len(train_dataset)}")
    print(f"Validation scenarios: {len(val_dataset)}")
    
    # Create dataloaders with persistent workers for speed:
    persistent_workers = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=vectornet_collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=vectornet_prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vectornet_collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=vectornet_prefetch_factor if num_workers > 0 else None,
    )
    
    print("\n" + "="*60)
    print("Creating VectorNet model...")
    print("="*60)
    model = VectorNetTFRecord(
        agent_input_dim=AGENT_VECTOR_DIM,
        map_input_dim=MAP_VECTOR_DIM,
        hidden_dim=hidden_dim,
        output_dim=2,
        prediction_horizon=vectornet_prediction_horizon,
        num_polyline_layers=num_polyline_layers,
        num_global_layers=num_global_layers,
        num_heads=num_heads,
        dropout=vectornet_dropout,
    ).to(train_device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # torch.compile for speedup (PyTorch 2.0+)
    # Note: 'reduce-overhead' mode can cause pow_by_natural warnings with dynamic shapes
    # Using 'default' mode which is more stable with variable batch sizes
    if use_torch_compile and hasattr(torch, 'compile'):
        try:
            compile_mode = torch_compile_mode if torch_compile_mode != 'reduce-overhead' else 'default'
            print(f"Compiling model with torch.compile (mode={compile_mode})...")
            model = torch.compile(model, mode=compile_mode, dynamic=True)
            print("  Model compiled successfully")
        except Exception as e:
            print(f"  torch.compile failed: {e}")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,)
    
    # Scheduler
    scheduler_type = 'cosine'
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=vectornet_min_lr)
    else:
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=len(train_loader),)
    
    scaler = torch.amp.GradScaler() if use_amp and torch.cuda.is_available() else None  # AMP scaler
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    best_model_state = None
    best_optimizer_state = None
    best_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, metrics = load_checkpoint(args.resume, model, optimizer, scheduler)
        best_val_loss = metrics.get('val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch+1}")
    
    
    # training loop:
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    os.makedirs(vectornet_checkpoint_dir, exist_ok=True)
    os.makedirs(vectornet_viz_dir, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # visualization callback - run every epoch
        viz_callback = None
        if (epoch + 1) % vectornet_visualize_every_n_epochs == 0 or epoch == 0:
            def viz_callback(batch):
                print(f"  Generating visualization for epoch {epoch+1}...")
                model.eval()
                with torch.no_grad():
                    # Move batch back to device
                    batch_gpu = {k: v.to(train_device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    predictions = model(batch_gpu)
                    
                    # Visualize predictions with map features (multi-agent version)
                    visualize_vectornet_predictions(
                        predictions=predictions.cpu(),
                        targets=batch['future_positions'],
                        valid_mask=batch['future_valid'],
                        scenario_ids=batch.get('scenario_ids', [f'scenario_{i}' for i in range(batch['batch_size'])]),
                        agent_vectors=batch['agent_vectors'],
                        agent_polyline_ids=batch['agent_polyline_ids'],
                        agent_batch_idx=batch['agent_batch'],
                        map_vectors=batch['map_vectors'],
                        map_polyline_ids=batch['map_polyline_ids'],
                        map_batch_idx=batch['map_batch'],
                        target_polyline_indices=batch['target_polyline_indices'],
                        target_scenario_batch=batch['target_scenario_batch'],
                        num_targets_per_scenario=batch['num_targets_per_scenario'],
                        epoch=epoch,
                        output_dir=vectornet_viz_dir,
                        max_scenarios=2
                    )
                model.train()
        
        # train:
        train_metrics = train_epoch(model, train_loader, optimizer,
                                   vectornet_loss_alpha, vectornet_loss_beta,
                                   vectornet_loss_gamma, vectornet_loss_delta,
                                   vectornet_gradient_clip, scaler, visualize_callback=viz_callback)
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
              f"ADE: {train_metrics['ade']:.4f} | FDE: {train_metrics['fde']:.4f}")
        
        # validate:
        val_metrics = validate(model, val_loader,
                              vectornet_loss_alpha, vectornet_loss_beta,
                              vectornet_loss_gamma, vectornet_loss_delta)
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"ADE: {val_metrics['ade']:.4f} | FDE: {val_metrics['fde']:.4f}")
        
        # update scheduler
        if scheduler_type == 'cosine':
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_metrics['loss'],
            "train/traj_loss": train_metrics['traj_loss'],
            "train/ade": train_metrics['ade'],
            "train/fde": train_metrics['fde'],
            "val/loss": val_metrics['loss'],
            "val/ade": val_metrics['ade'],
            "val/fde": val_metrics['fde'],
            "lr": current_lr,
            # Per-epoch aggregated metrics for better tracking
            "train_epoch/loss": train_metrics['loss'],
            "train_epoch/traj_loss": train_metrics['traj_loss'],
            "train_epoch/ade": train_metrics['ade'],
            "train_epoch/fde": train_metrics['fde'],
            "val_epoch/loss": val_metrics['loss'],
            "val_epoch/ade": val_metrics['ade'],
            "val_epoch/fde": val_metrics['fde'],
            "learning_rate": current_lr,
        }, commit=True)
        
        # Save best model state in memory:
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            # Handle DataParallel and torch.compile
            if hasattr(model, 'module'):
                save_model = model.module
            elif hasattr(model, '_orig_mod'):
                save_model = model._orig_mod
            else:
                save_model = model
            best_model_state = save_model.state_dict().copy()
            best_optimizer_state = optimizer.state_dict().copy()
            print(f"   New best validation loss: {val_metrics['loss']:.4f} at epoch {epoch+1}")
    
    # Save best model checkpoint after training completes:
    if best_model_state is not None:
        print(f"\n{'='*60}")
        print(f"Saving best model from epoch {best_epoch}...")
        best_checkpoint_path = os.path.join(vectornet_checkpoint_dir, 'best_vectornet_tfrecord.pt')
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'val_loss': best_val_loss,
            'config': {
                'hidden_dim': hidden_dim,
                'num_polyline_layers': num_polyline_layers,
                'num_global_layers': num_global_layers,
                'num_heads': num_heads,
                'dropout': vectornet_dropout,
                'future_len': vectornet_prediction_horizon,
                'history_len': vectornet_history_length,
            }
        }, best_checkpoint_path)
        print(f"Best VectorNet model saved to best_vectornet_tfrecord.pt")
        print(f"Epoch: {best_epoch} | Val Loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
    else:
        print("\nNo best model checkpoint saved (no improvement detected)\n")
    
    # Save final model (current state at last epoch):
    print(f"\n{'='*60}")
    print(f"Saving final model from epoch {epoch+1}...")
    # Handle DataParallel and torch.compile
    if hasattr(model, 'module'):
        save_model = model.module
    elif hasattr(model, '_orig_mod'):
        save_model = model._orig_mod
    else:
        save_model = model
    final_checkpoint_path = os.path.join(vectornet_checkpoint_dir, 'final_vectornet_tfrecord.pt')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': save_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'config': {
            'hidden_dim': hidden_dim,
            'num_polyline_layers': num_polyline_layers,
            'num_global_layers': num_global_layers,
            'num_heads': num_heads,
            'dropout': vectornet_dropout,
            'future_len': vectornet_prediction_horizon,
            'history_len': vectornet_history_length,
        }
    }, final_checkpoint_path)
    print(f"Final VectorNet model saved to final_vectornet_tfrecord.pt")
    print(f"Epoch: {epoch+1} | Val Loss: {val_metrics['loss']:.4f}")
    print(f"{'='*60}\n")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print("="*60)
    
    wandb.finish()


if __name__ == '__main__':
    main()
