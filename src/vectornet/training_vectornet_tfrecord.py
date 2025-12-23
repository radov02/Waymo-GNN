"""VectorNet Training Script for TFRecord Dataset (Waymo Open Motion Dataset).

This is the main training script for VectorNet using direct TFRecord loading.
It uses VectorNetTFRecordDataset which extracts all necessary data including:
- Agent trajectory polylines
- Map feature polylines (lanes, roads, crosswalks)
- Multi-step future trajectory labels

VectorNet predicts the FULL FUTURE TRAJECTORY at once (not autoregressive).

Usage:
    python src/vectornet/training_vectornet_tfrecord.py --data_dir data/scenario

Checkpoints: checkpoints/vectornet/
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

# Import visualization functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'helper_functions'))
import matplotlib.pyplot as plt
from datetime import datetime as dt

# Suppress warnings
warnings.filterwarnings("ignore", message=".*skipping cudagraphs.*")
warnings.filterwarnings("ignore", message=".*cudagraph.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# Multiprocessing setup
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Import VectorNet modules
from VectorNet import VectorNetTFRecord, AGENT_VECTOR_DIM, MAP_VECTOR_DIM
from vectornet_tfrecord_dataset import (
    VectorNetTFRecordDataset, 
    vectornet_collate_fn,
    create_vectornet_dataloaders
)


# ============== Configuration ==============
class Config:
    """Training configuration."""
    
    # Data
    data_dir: str = os.path.join(os.path.dirname(__file__), 'data')
    history_len: int = 10
    future_len: int = 50
    max_train_scenarios: int = None  # None = all
    max_val_scenarios: int = 1000
    
    # Model
    hidden_dim: int = 128
    num_polyline_layers: int = 3
    num_global_layers: int = 1
    num_heads: int = 8
    dropout: float = 0.1
    use_node_completion: bool = True
    node_completion_ratio: float = 0.15
    
    # Training
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    gradient_clip: float = 1.0
    warmup_epochs: int = 5
    
    # Loss weights
    loss_alpha: float = 0.2   # Angular error
    loss_beta: float = 0.5    # MSE
    loss_gamma: float = 0.1   # Velocity consistency  
    loss_delta: float = 0.2   # Endpoint error
    node_completion_weight: float = 1.0
    
    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine' or 'onecycle'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    
    # Logging
    wandb_project: str = 'waymo-vectornet-tfrecord'
    wandb_name: str = None  # Auto-generated
    
    # Checkpoints
    checkpoint_dir: str = 'checkpoints/vectornet'
    save_every: int = 5
    
    # Visualization
    viz_dir: str = 'visualizations/autoreg/vectornet'
    visualize_every_n_epochs: int = 1  # Visualize every epoch
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp: bool = True
    amp_dtype: str = 'float16'  # 'float16' or 'bfloat16'
    
    # Parallelization
    use_torch_compile: bool = True  # Use torch.compile for speedup
    gradient_accumulation_steps: int = 1  # Accumulate gradients for larger effective batch
    persistent_workers: bool = True  # Keep workers alive between epochs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VectorNet on TFRecord data')
    
    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Base directory for TFRecord data (default: src/vectornet/data)')
    parser.add_argument('--max_train', type=int, default=None,
                       help='Max training scenarios (None = all)')
    parser.add_argument('--max_val', type=int, default=1000,
                       help='Max validation scenarios')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_polyline_layers', type=int, default=3)
    parser.add_argument('--num_global_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Misc
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Resume from checkpoint')
    
    return parser.parse_args()


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  Early stopping: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False


def multistep_trajectory_loss(pred, target, valid_mask=None, 
                              alpha=0.2, beta=0.5, gamma=0.1, delta=0.2):
    """Loss function for multi-step trajectory prediction.
    
    Args:
        pred: Predictions [B, T, 2] - predicted positions
        target: Targets [B, T, 2] - ground truth positions
        valid_mask: [B, T] validity mask
        alpha: Weight for angular error
        beta: Weight for MSE
        gamma: Weight for velocity consistency
        delta: Weight for endpoint error
        
    Returns:
        Combined loss
    """
    # Apply validity mask if provided
    if valid_mask is not None:
        # Expand mask for 2D coordinates
        mask = valid_mask.unsqueeze(-1).expand_as(pred)
        pred = pred * mask
        target = target * mask
    
    # MSE loss
    mse = F.mse_loss(pred, target)
    
    # Angular/direction loss
    pred_flat = pred.reshape(-1, 2)
    target_flat = target.reshape(-1, 2)
    
    pred_angle = torch.atan2(pred_flat[:, 1], pred_flat[:, 0] + 1e-6)
    target_angle = torch.atan2(target_flat[:, 1], target_flat[:, 0] + 1e-6)
    angle_diff = torch.atan2(
        torch.sin(pred_angle - target_angle),
        torch.cos(pred_angle - target_angle)
    )
    angular_loss = torch.mean(torch.abs(angle_diff))
    
    # Velocity consistency (smooth trajectory)
    if pred.shape[1] > 1:
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        target_vel = target[:, 1:, :] - target[:, :-1, :]
        velocity_loss = F.mse_loss(pred_vel, target_vel)
    else:
        velocity_loss = torch.tensor(0.0, device=pred.device)
    
    # Endpoint error (final displacement error)
    endpoint_loss = F.mse_loss(pred[:, -1, :], target[:, -1, :])
    
    # Combine losses
    total_loss = (
        beta * mse +
        alpha * angular_loss +
        gamma * velocity_loss +
        delta * endpoint_loss
    )
    
    return total_loss


def compute_metrics(pred, target, valid_mask=None):
    """Compute ADE and FDE metrics.
    
    Args:
        pred: [B, T, 2] predictions
        target: [B, T, 2] ground truth
        valid_mask: [B, T] optional validity mask
        
    Returns:
        ade: Average Displacement Error
        fde: Final Displacement Error
    """
    # Displacement at each timestep
    disp = torch.norm(pred - target, dim=-1)  # [B, T]
    
    if valid_mask is not None:
        # Masked ADE
        valid_count = valid_mask.sum()
        if valid_count > 0:
            ade = (disp * valid_mask).sum() / valid_count
        else:
            ade = torch.tensor(0.0, device=pred.device)
        
        # FDE: last valid position per sample
        # For simplicity, just use last timestep
        fde = disp[:, -1].mean()
    else:
        ade = disp.mean()
        fde = disp[:, -1].mean()
    
    return ade, fde


def visualize_vectornet_predictions(predictions, targets, valid_mask, scenario_ids,
                                     agent_vectors, agent_polyline_ids, agent_batch_idx,
                                     map_vectors, map_polyline_ids, map_batch_idx,
                                     epoch, output_dir, max_scenarios=2):
    """Visualize VectorNet trajectory predictions with map features.
    
    Args:
        predictions: [B, T, 2] predicted positions
        targets: [B, T, 2] ground truth positions  
        valid_mask: [B, T] validity mask
        scenario_ids: List of scenario IDs
        agent_vectors: [N_agents, 16] agent polyline vectors
        agent_polyline_ids: [N_agents] polyline IDs for agents
        agent_batch_idx: [N_agents] batch index for each agent vector
        map_vectors: [N_map, 13] map polyline vectors
        map_polyline_ids: [N_map] polyline IDs for map features
        map_batch_idx: [N_map] batch index for each map vector
        epoch: Current epoch number
        output_dir: Directory to save visualizations
        max_scenarios: Maximum number of scenarios to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    B = min(predictions.shape[0], max_scenarios)
    T = predictions.shape[1]
    
    # Map feature type colors (from one-hot encoding in map_vectors)
    # type_onehot is at indices 6:13 (7 types)
    # All grey except stop signs (red)
    MAP_COLORS = {
        0: ('#A0A0A0', 'Lanes'),        # lane - gray
        1: ('#A0A0A0', 'Road Lines'),   # road_line - gray
        2: ('#A0A0A0', 'Road Edges'),   # road_edge - gray
        3: ('#FF0000', 'Stop Signs'),   # stop_sign - red
        4: ('#A0A0A0', 'Crosswalks'),   # crosswalk - gray
        5: ('#A0A0A0', 'Speed Bumps'),  # speed_bump - gray
        6: ('#A0A0A0', 'Driveways'),    # driveway - gray
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, B, figsize=(10*B, 10))
    if B == 1:
        axes = [axes]
    
    for i in range(B):
        ax = axes[i]
        pred = predictions[i].numpy()  # [T, 2]
        tgt = targets[i].numpy()       # [T, 2]
        mask = valid_mask[i].numpy() if valid_mask is not None else np.ones(T)
        
        # Filter by valid mask
        valid_idx = mask > 0.5
        pred_valid = pred[valid_idx]
        tgt_valid = tgt[valid_idx]
        
        # ===== Plot Map Features =====
        # Get map vectors for this scenario
        if map_batch_idx is not None and len(map_batch_idx) > 0:
            scenario_map_mask = (map_batch_idx == i)
            scenario_map_vectors = map_vectors[scenario_map_mask].numpy()
            scenario_map_polyline_ids = map_polyline_ids[scenario_map_mask].numpy()
            
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
                
                # Extract start and end points (ds_x, ds_y at 0:2, de_x, de_y at 3:5)
                # Build polyline from consecutive vectors
                points = []
                for vec in poly_vectors:
                    points.append([vec[0], vec[1]])  # ds_x, ds_y
                # Add last endpoint
                if len(poly_vectors) > 0:
                    points.append([poly_vectors[-1, 3], poly_vectors[-1, 4]])  # de_x, de_y
                
                points = np.array(points)
                
                # Plot with label only once per type
                show_label = type_idx not in plotted_types
                ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1, 
                       alpha=0.4, label=label if show_label else None)
                plotted_types.add(type_idx)
        
        # ===== Plot Agent Trajectories (from agent_vectors) =====
        if agent_batch_idx is not None and len(agent_batch_idx) > 0:
            scenario_agent_mask = (agent_batch_idx == i)
            scenario_agent_vectors = agent_vectors[scenario_agent_mask].numpy()
            scenario_agent_polyline_ids = agent_polyline_ids[scenario_agent_mask].numpy()
            
            # Group by polyline ID and plot history trajectories
            unique_agent_polylines = np.unique(scenario_agent_polyline_ids)
            
            for j, poly_id in enumerate(unique_agent_polylines[:20]):  # Limit to 20 agents
                poly_mask = scenario_agent_polyline_ids == poly_id
                poly_vectors = scenario_agent_vectors[poly_mask]
                
                if len(poly_vectors) == 0:
                    continue
                
                # Build trajectory from vectors
                points = []
                for vec in poly_vectors:
                    points.append([vec[0], vec[1]])  # ds_x, ds_y (start point)
                # Add last endpoint
                if len(poly_vectors) > 0:
                    points.append([poly_vectors[-1, 3], poly_vectors[-1, 4]])  # de_x, de_y
                
                points = np.array(points)
                
                # Plot agent history (lighter for other agents)
                alpha = 0.8 if j == 0 else 0.3
                lw = 2 if j == 0 else 1
                ax.plot(points[:, 0], points[:, 1], 'g-', linewidth=lw, alpha=alpha,
                       label='Agent History' if j == 0 else None)
        
        # ===== Plot Ground Truth Future Trajectory =====
        ax.plot(tgt_valid[:, 0], tgt_valid[:, 1], 'b-', linewidth=3, 
                label='GT Future', alpha=0.9, zorder=10)
        ax.scatter(tgt_valid[0, 0], tgt_valid[0, 1], c='blue', s=150, 
                   marker='o', zorder=15, edgecolors='white', linewidth=2)
        ax.scatter(tgt_valid[-1, 0], tgt_valid[-1, 1], c='blue', s=150, 
                   marker='s', zorder=15, edgecolors='white', linewidth=2)
        
        # ===== Plot Predicted Future Trajectory =====
        ax.plot(pred_valid[:, 0], pred_valid[:, 1], 'r--', linewidth=3, 
                label='Predicted', alpha=0.9, zorder=10)
        ax.scatter(pred_valid[-1, 0], pred_valid[-1, 1], c='red', s=150, 
                   marker='x', zorder=15, linewidth=3)
        
        # Compute ADE/FDE for this scenario
        disp = np.linalg.norm(pred_valid - tgt_valid, axis=1)
        ade = disp.mean()
        fde = disp[-1] if len(disp) > 0 else 0
        
        # Get scenario ID
        sid = scenario_ids[i] if i < len(scenario_ids) else f"scenario_{i}"
        if isinstance(sid, bytes):
            sid = sid.decode('utf-8')
        
        ax.set_title(f'Scenario: {sid[:25]}...\nADE: {ade:.2f}m | FDE: {fde:.2f}m', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9, ncol=2)
    
    plt.suptitle(f'VectorNet Predictions - Epoch {epoch+1}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
    filename = f'vectornet_epoch{epoch+1:03d}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Saved visualization: {filepath}")
    return filepath


def train_epoch(model, dataloader, optimizer, config, scaler=None, visualize_callback=None):
    """Train for one epoch with gradient accumulation support.
    
    Args:
        model: VectorNetTFRecord model
        dataloader: Training DataLoader
        optimizer: Optimizer
        config: Training config
        scaler: GradScaler for AMP
        visualize_callback: Optional callback for visualization
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_traj_loss = 0.0
    total_node_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    steps = 0
    last_batch = None
    
    device = config.device
    amp_enabled = config.use_amp and torch.cuda.is_available()
    amp_dtype = torch.float16 if config.amp_dtype == 'float16' else torch.bfloat16
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    
    log_every = max(1, len(dataloader) // 10)
    
    # Zero gradients at start
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % log_every == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Agents: {batch['agent_vectors'].shape[0]} | "
                  f"Map: {batch['map_vectors'].shape[0]}")
        
        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
            # Forward pass
            predictions = model(batch)  # [B, future_len, 2]
            
            # Get targets
            targets = batch['future_positions'].to(device)  # [B, future_len, 2]
            valid_mask = batch['future_valid'].to(device)   # [B, future_len]
            
            # Trajectory loss
            traj_loss = multistep_trajectory_loss(
                predictions, targets, valid_mask,
                alpha=config.loss_alpha,
                beta=config.loss_beta,
                gamma=config.loss_gamma,
                delta=config.loss_delta
            )
            
            # Node completion loss
            node_loss_val = model.get_node_completion_loss() if hasattr(model, 'get_node_completion_loss') else model._orig_mod.get_node_completion_loss()
            if isinstance(node_loss_val, torch.Tensor) and node_loss_val.device != device:
                node_loss_val = node_loss_val.to(device)
            
            loss = traj_loss + config.node_completion_weight * node_loss_val
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            if torch.isnan(loss):
                print(f"  Warning: NaN loss at batch {batch_idx}")
                continue
        
        # Backward pass (accumulate gradients)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step optimizer every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        # Compute metrics (use unscaled loss)
        with torch.no_grad():
            ade, fde = compute_metrics(predictions, targets, valid_mask)
        
        total_loss += (loss.item() * accumulation_steps)  # Unscale for logging
        total_traj_loss += traj_loss.item()
        total_node_loss += node_loss_val.item() if isinstance(node_loss_val, torch.Tensor) else node_loss_val
        total_ade += ade.item()
        total_fde += fde.item()
        steps += 1
        
        # Log to wandb periodically
        if steps % 10 == 0:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_ade": ade.item(),
                "train/batch_fde": fde.item(),
            }, commit=False)
        
        # Store last batch for visualization
        if batch_idx == len(dataloader) - 1:
            last_batch = {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    metrics = {
        'loss': total_loss / max(1, steps),
        'traj_loss': total_traj_loss / max(1, steps),
        'node_loss': total_node_loss / max(1, steps),
        'ade': total_ade / max(1, steps),
        'fde': total_fde / max(1, steps),
    }
    
    # Call visualization callback after epoch completes
    if visualize_callback is not None and last_batch is not None:
        try:
            visualize_callback(last_batch)
        except Exception as e:
            print(f"  WARNING: Visualization failed: {e}")
    
    return metrics


@torch.no_grad()
def validate(model, dataloader, config):
    """Validate the model.
    
    Args:
        model: VectorNetTFRecord model
        dataloader: Validation DataLoader
        config: Training config
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    steps = 0
    
    device = config.device
    amp_enabled = config.use_amp and torch.cuda.is_available()
    amp_dtype = torch.float16 if config.amp_dtype == 'float16' else torch.bfloat16
    
    for batch in dataloader:
        with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
            predictions = model(batch)
            targets = batch['future_positions'].to(device)
            valid_mask = batch['future_valid'].to(device)
            
            loss = multistep_trajectory_loss(
                predictions, targets, valid_mask,
                alpha=config.loss_alpha,
                beta=config.loss_beta,
                gamma=config.loss_gamma,
                delta=config.loss_delta
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


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Handle DataParallel
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)
    print(f"  Saved checkpoint: {path}")


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
    """Main training loop."""
    args = parse_args()
    config = Config()
    
    # Update config from args
    if args.data_dir is None:
        config.data_dir = os.path.join(os.path.dirname(__file__), 'data')
    else:
        config.data_dir = args.data_dir
    config.max_train_scenarios = args.max_train
    config.max_val_scenarios = args.max_val
    config.hidden_dim = args.hidden_dim
    config.num_polyline_layers = args.num_polyline_layers
    config.num_global_layers = args.num_global_layers
    config.num_heads = args.num_heads
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.epochs = args.epochs
    config.num_workers = args.num_workers
    
    # Set device
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize wandb
    if not args.no_wandb:
        run_name = config.wandb_name or f"vectornet_tfrecord_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=vars(config),
        )
    else:
        wandb.init(mode='disabled')
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading TFRecord datasets...")
    print("="*60)
    
    train_dataset = VectorNetTFRecordDataset(
        tfrecord_dir=config.data_dir,
        split='training',
        history_len=config.history_len,
        future_len=config.future_len,
        max_scenarios=config.max_train_scenarios,
    )
    
    val_dataset = VectorNetTFRecordDataset(
        tfrecord_dir=config.data_dir,
        split='validation',
        history_len=config.history_len,
        future_len=config.future_len,
        max_scenarios=config.max_val_scenarios,
    )
    
    print(f"Training scenarios: {len(train_dataset)}")
    print(f"Validation scenarios: {len(val_dataset)}")
    
    # Create dataloaders with persistent workers for speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=vectornet_collate_fn,
        pin_memory=True,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=vectornet_collate_fn,
        pin_memory=True,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
    
    # Create model
    print("\n" + "="*60)
    print("Creating VectorNet model...")
    print("="*60)
    
    model = VectorNetTFRecord(
        agent_input_dim=AGENT_VECTOR_DIM,
        map_input_dim=MAP_VECTOR_DIM,
        hidden_dim=config.hidden_dim,
        output_dim=2,
        prediction_horizon=config.future_len,
        num_polyline_layers=config.num_polyline_layers,
        num_global_layers=config.num_global_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        use_node_completion=config.use_node_completion,
        node_completion_ratio=config.node_completion_ratio,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # torch.compile for speedup (PyTorch 2.0+)
    if config.use_torch_compile and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile (reduce-overhead mode)...")
            model = torch.compile(model, mode='reduce-overhead')
            print("  ✓ Model compiled successfully")
        except Exception as e:
            print(f"  ✗ torch.compile failed: {e}")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler
    if config.scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr
        )
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
        )
    
    # AMP scaler
    scaler = torch.amp.GradScaler() if config.use_amp and torch.cuda.is_available() else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, metrics = load_checkpoint(args.resume, model, optimizer, scheduler)
        best_val_loss = metrics.get('val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch+1}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.viz_dir, exist_ok=True)
    
    for epoch in range(start_epoch, config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"{'='*60}")
        
        # Visualization callback - run every epoch
        viz_callback = None
        if (epoch + 1) % config.visualize_every_n_epochs == 0 or epoch == 0:
            def viz_callback(batch):
                print(f"  Generating visualization for epoch {epoch+1}...")
                model.eval()
                with torch.no_grad():
                    # Move batch back to device
                    batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    predictions = model(batch_gpu)
                    
                    # Visualize predictions with map features
                    visualize_vectornet_predictions(
                        predictions=predictions.cpu(),
                        targets=batch['future_positions'],
                        valid_mask=batch['future_valid'],
                        scenario_ids=batch.get('scenario_ids', [f'scenario_{i}' for i in range(predictions.shape[0])]),
                        agent_vectors=batch['agent_vectors'],
                        agent_polyline_ids=batch['agent_polyline_ids'],
                        agent_batch_idx=batch['agent_batch_idx'],
                        map_vectors=batch['map_vectors'],
                        map_polyline_ids=batch['map_polyline_ids'],
                        map_batch_idx=batch['map_batch_idx'],
                        epoch=epoch,
                        output_dir=config.viz_dir,
                        max_scenarios=2  # Only 2 scenarios
                    )
                model.train()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, scaler, visualize_callback=viz_callback)
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
              f"ADE: {train_metrics['ade']:.4f} | FDE: {train_metrics['fde']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, config)
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"ADE: {val_metrics['ade']:.4f} | FDE: {val_metrics['fde']:.4f}")
        
        # Update scheduler
        if config.scheduler_type == 'cosine':
            scheduler.step()
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_metrics['loss'],
            "train/traj_loss": train_metrics['traj_loss'],
            "train/node_loss": train_metrics['node_loss'],
            "train/ade": train_metrics['ade'],
            "train/fde": train_metrics['fde'],
            "val/loss": val_metrics['loss'],
            "val/ade": val_metrics['ade'],
            "val/fde": val_metrics['fde'],
            "lr": optimizer.param_groups[0]['lr'],
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': val_metrics['loss'], 'val_ade': val_metrics['ade'], 'val_fde': val_metrics['fde']},
                os.path.join(config.checkpoint_dir, 'best_vectornet_tfrecord.pt')
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_loss': val_metrics['loss']},
                os.path.join(config.checkpoint_dir, f'vectornet_tfrecord_epoch_{epoch+1}.pt')
            )
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        {'val_loss': val_metrics['loss']},
        os.path.join(config.checkpoint_dir, 'final_vectornet_tfrecord.pt')
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    wandb.finish()


if __name__ == '__main__':
    main()
