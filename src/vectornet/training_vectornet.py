"""VectorNet Multi-Step Training Script for Waymo Open Motion Dataset.

VectorNet predicts the FULL FUTURE TRAJECTORY at once (not autoregressive).
This is the correct approach as per the VectorNet paper.

The model:
1. Encodes history timesteps (default: 10 steps = 1 second)
2. Aggregates temporal information using attention
3. Predicts all future timesteps at once (default: 50 steps = 5 seconds)

Features:
- Multi-step trajectory prediction (not autoregressive)
- Node completion auxiliary task
- Mixed precision training (AMP)
- Learning rate scheduling with warmup
- Early stopping
- WandB logging
- Checkpoint saving

Usage:
    python src/vectornet/training_vectornet.py

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
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.nn.utils import clip_grad_norm_

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

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass

# Import VectorNet modules
from VectorNet import VectorNetMultiStep
from vectornet_dataset import VectorNetDataset, collate_vectornet_batch, worker_init_fn
from vectornet_config import (
    device, num_workers, pin_memory, prefetch_factor,
    use_amp, use_bf16, use_torch_compile, torch_compile_mode,
    use_gradient_checkpointing, print_gpu_info, setup_model_parallel,
    get_model_for_saving, get_vectornet_config,
    vectornet_input_dim, vectornet_hidden_dim, vectornet_output_dim,
    vectornet_num_polyline_layers, vectornet_num_global_layers,
    vectornet_num_heads, vectornet_dropout,
    vectornet_use_node_completion, vectornet_node_completion_ratio,
    vectornet_node_completion_weight,
    vectornet_batch_size, vectornet_learning_rate, vectornet_epochs,
    vectornet_gradient_clip, vectornet_scheduler_patience,
    vectornet_scheduler_factor, vectornet_min_lr,
    vectornet_early_stopping_patience, vectornet_early_stopping_min_delta,
    vectornet_loss_alpha, vectornet_loss_beta, vectornet_loss_gamma, vectornet_loss_delta,
    vectornet_prediction_horizon, vectornet_history_length,
    vectornet_max_validation_scenarios,
    vectornet_checkpoint_dir, vectornet_viz_dir,
    vectornet_wandb_project, vectornet_wandb_name, vectornet_wandb_tags,
    print_vectornet_config
)


# ============== PATHS ==============
TRAIN_HDF5 = 'data/graphs/training/scenario_graphs.h5'
VAL_HDF5 = 'data/graphs/validation/scenario_graphs.h5'


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


def get_module(model):
    """Get underlying model (unwrap DataParallel/compiled models)."""
    if hasattr(model, 'module'):
        return model.module
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def multistep_trajectory_loss(pred, target, alpha=0.2, beta=0.5, gamma=0.1, delta=0.2):
    """Loss function for multi-step trajectory prediction.
    
    Args:
        pred: Predictions [N, T, 2] - predicted displacements
        target: Targets [N, T, 2] - ground truth displacements
        alpha: Weight for angular error
        beta: Weight for MSE
        gamma: Weight for velocity consistency
        delta: Weight for endpoint error
        
    Returns:
        Combined loss
    """
    # Handle shape mismatch
    if pred.shape != target.shape:
        min_t = min(pred.shape[1], target.shape[1])
        pred = pred[:, :min_t, :]
        target = target[:, :min_t, :]
    
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


def train_epoch(model, dataloader, optimizer, device, epoch,
                scaler=None, amp_dtype=torch.float16,
                node_completion_weight=1.0, prediction_horizon=50):
    """Train for one epoch with multi-step prediction.
    
    Args:
        model: VectorNet model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Target device
        epoch: Current epoch number
        scaler: GradScaler for AMP
        amp_dtype: AMP data type
        node_completion_weight: Weight for node completion loss
        prediction_horizon: Number of future timesteps to predict
        
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
    
    use_amp_local = torch.cuda.is_available() and amp_dtype in (torch.float16, torch.bfloat16)
    log_every_n = max(1, len(dataloader) // 10)
    
    for batch_idx, batch_dict in enumerate(dataloader):
        batched_sequence = batch_dict["batch"]
        B = batch_dict["B"]
        T = batch_dict["T"]
        
        if batch_idx % log_every_n == 0:
            total_nodes = batched_sequence[0].num_nodes if len(batched_sequence) > 0 else 0
            print(f"\n[Epoch {epoch+1}] Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Scenarios: {B} | Nodes: {total_nodes} | T={T}")
        
        # Move to device
        for t in range(T):
            batched_sequence[t] = batched_sequence[t].to(device, non_blocking=True)
        
        # Reset model state
        base_model = get_module(model)
        base_model.reset_temporal_buffer()
        
        optimizer.zero_grad(set_to_none=True)
        
        # Determine history and future split
        history_len = min(T - 1, 10)  # Use up to 10 history steps
        
        if T <= 1:
            continue  # Need at least 2 timesteps
        
        with torch.amp.autocast('cuda', enabled=use_amp_local, dtype=amp_dtype):
            # Process history steps
            for t in range(history_len):
                data = batched_sequence[t]
                base_model.forward_single_step(
                    data.x, data.edge_index, 
                    edge_weight=data.edge_attr,
                    is_last_step=False
                )
            
            # Final step: get predictions
            last_data = batched_sequence[history_len]
            predictions = base_model.forward_single_step(
                last_data.x, last_data.edge_index,
                edge_weight=last_data.edge_attr,
                is_last_step=True
            )
            
            if predictions is None:
                continue
            
            # Build multi-step targets from remaining timesteps
            future_len = min(T - history_len - 1, prediction_horizon)
            N = last_data.x.shape[0]
            
            if future_len <= 0:
                continue
            
            targets = torch.zeros(N, future_len, 2, device=device)
            for future_t in range(future_len):
                t_idx = history_len + 1 + future_t
                if t_idx < T:
                    future_data = batched_sequence[t_idx]
                    if future_data.y is not None and future_data.y.shape[0] >= N:
                        targets[:, future_t, :] = future_data.y[:N, :]
            
            # Truncate predictions to match targets
            predictions = predictions[:, :future_len, :]
            
            # Trajectory loss
            traj_loss = multistep_trajectory_loss(
                predictions, targets,
                alpha=vectornet_loss_alpha,
                beta=vectornet_loss_beta,
                gamma=vectornet_loss_gamma,
                delta=vectornet_loss_delta
            )
            
            # Node completion loss
            node_loss = torch.tensor(0.0, device=device)
            if hasattr(base_model, 'get_node_completion_loss'):
                node_loss = base_model.get_node_completion_loss()
                if node_loss.device != device:
                    node_loss = node_loss.to(device)
            
            loss = traj_loss + node_completion_weight * node_loss
            
            if torch.isnan(loss):
                print(f"  Warning: NaN loss at batch {batch_idx}")
                continue
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), vectornet_gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), vectornet_gradient_clip)
            optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            # ADE: Average displacement error
            ade = torch.mean(torch.norm(predictions - targets, dim=-1))
            # FDE: Final displacement error
            fde = torch.mean(torch.norm(predictions[:, -1, :] - targets[:, -1, :], dim=-1))
        
        total_loss += loss.item()
        total_traj_loss += traj_loss.item()
        total_node_loss += node_loss.item() if isinstance(node_loss, torch.Tensor) else node_loss
        total_ade += ade.item()
        total_fde += fde.item()
        steps += 1
        
        # Log to wandb
        if steps % 10 == 0:
            wandb.log({
                "batch": epoch * len(dataloader) + batch_idx,
                "train/batch_loss": loss.item(),
                "train/batch_traj_loss": traj_loss.item(),
                "train/batch_ade": ade.item(),
                "train/batch_fde": fde.item(),
            })
        
        # Clear CUDA cache
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    return {
        'loss': total_loss / max(1, steps),
        'traj_loss': total_traj_loss / max(1, steps),
        'node_loss': total_node_loss / max(1, steps),
        'ade': total_ade / max(1, steps),
        'fde': total_fde / max(1, steps),
    }


@torch.no_grad()
def validate(model, dataloader, device, amp_dtype=torch.float16, prediction_horizon=50):
    """Validate the model with multi-step prediction.
    
    Args:
        model: VectorNet model
        dataloader: Validation data loader
        device: Target device
        amp_dtype: AMP data type
        prediction_horizon: Number of future timesteps
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    steps = 0
    
    use_amp_local = torch.cuda.is_available() and amp_dtype in (torch.float16, torch.bfloat16)
    base_model = get_module(model)
    
    for batch_dict in dataloader:
        batched_sequence = batch_dict["batch"]
        T = batch_dict["T"]
        
        # Move to device
        for t in range(T):
            batched_sequence[t] = batched_sequence[t].to(device, non_blocking=True)
        
        base_model.reset_temporal_buffer()
        
        history_len = min(T - 1, 10)
        if T <= 1:
            continue
        
        with torch.amp.autocast('cuda', enabled=use_amp_local, dtype=amp_dtype):
            # Process history
            for t in range(history_len):
                data = batched_sequence[t]
                base_model.forward_single_step(
                    data.x, data.edge_index,
                    edge_weight=data.edge_attr,
                    is_last_step=False
                )
            
            # Get predictions
            last_data = batched_sequence[history_len]
            predictions = base_model.forward_single_step(
                last_data.x, last_data.edge_index,
                edge_weight=last_data.edge_attr,
                is_last_step=True
            )
            
            if predictions is None:
                continue
            
            # Build targets
            future_len = min(T - history_len - 1, prediction_horizon)
            N = last_data.x.shape[0]
            
            if future_len <= 0:
                continue
            
            targets = torch.zeros(N, future_len, 2, device=device)
            for future_t in range(future_len):
                t_idx = history_len + 1 + future_t
                if t_idx < T:
                    future_data = batched_sequence[t_idx]
                    if future_data.y is not None and future_data.y.shape[0] >= N:
                        targets[:, future_t, :] = future_data.y[:N, :]
            
            predictions = predictions[:, :future_len, :]
            
            loss = multistep_trajectory_loss(predictions, targets)
            ade = torch.mean(torch.norm(predictions - targets, dim=-1))
            fde = torch.mean(torch.norm(predictions[:, -1, :] - targets[:, -1, :], dim=-1))
        
        total_loss += loss.item()
        total_ade += ade.item()
        total_fde += fde.item()
        steps += 1
    
    return {
        'loss': total_loss / max(1, steps),
        'ade': total_ade / max(1, steps),
        'fde': total_fde / max(1, steps),
    }


def create_model(config, device):
    """Create VectorNet model from config."""
    model = VectorNetMultiStep(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        prediction_horizon=config['prediction_horizon'],
        num_polyline_layers=config['num_polyline_layers'],
        num_global_layers=config['num_global_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_node_completion=config['use_node_completion'],
        node_completion_ratio=config['node_completion_ratio'],
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model


def main():
    """Main training function."""
    print("\n" + "=" * 70)
    print("VectorNet Multi-Step Trajectory Prediction Training")
    print("=" * 70)
    
    # Print configuration
    print_vectornet_config()
    
    # Print GPU info
    print_gpu_info()
    
    # Get config
    config = get_vectornet_config()
    
    # Check data exists
    if not os.path.exists(TRAIN_HDF5):
        print(f"\nERROR: Training data not found at {TRAIN_HDF5}")
        print("Please ensure the data files are in the correct location.")
        return
    
    if not os.path.exists(VAL_HDF5):
        print(f"\nWARNING: Validation data not found at {VAL_HDF5}")
        print("Will use a subset of training data for validation.")
        use_train_for_val = True
    else:
        use_train_for_val = False
    
    # Create datasets
    print("\n--- Loading Datasets ---")
    train_dataset = VectorNetDataset(
        TRAIN_HDF5,
        seq_len=config['prediction_horizon'] + config['history_length'],
        cache_in_memory=False
    )
    print(f"Training scenarios: {len(train_dataset)}")
    
    if use_train_for_val:
        # Use 10% of training data for validation
        val_size = max(1, len(train_dataset) // 10)
        val_indices = list(range(len(train_dataset) - val_size, len(train_dataset)))
        val_dataset = Subset(train_dataset, val_indices)
    else:
        val_dataset = VectorNetDataset(
            VAL_HDF5,
            seq_len=config['prediction_horizon'] + config['history_length'],
            cache_in_memory=True
        )
    
    val_size = min(len(val_dataset), vectornet_max_validation_scenarios)
    if len(val_dataset) > val_size:
        val_indices = list(range(val_size))
        val_dataset = Subset(val_dataset, val_indices)
    print(f"Validation scenarios: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_vectornet_batch,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_vectornet_batch,
        worker_init_fn=worker_init_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n--- Creating Model ---")
    model = create_model(config, device)
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=vectornet_min_lr
    )
    
    # AMP setup
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=vectornet_early_stopping_patience,
        min_delta=vectornet_early_stopping_min_delta
    )
    
    # Initialize WandB
    wandb.init(
        project=vectornet_wandb_project,
        name=f"{vectornet_wandb_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags=vectornet_wandb_tags + ['multi-step'],
        config={
            **config,
            'use_amp': use_amp,
            'amp_dtype': str(amp_dtype),
            'device': str(device),
        }
    )
    
    # Create checkpoint directory
    os.makedirs(vectornet_checkpoint_dir, exist_ok=True)
    
    # Training loop
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*60}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            scaler=scaler, amp_dtype=amp_dtype,
            node_completion_weight=vectornet_node_completion_weight,
            prediction_horizon=config['prediction_horizon']
        )
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
              f"ADE: {train_metrics['ade']:.4f} | FDE: {train_metrics['fde']:.4f}")
        
        # Validate
        val_metrics = validate(
            model, val_loader, device, amp_dtype=amp_dtype,
            prediction_horizon=config['prediction_horizon']
        )
        
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"ADE: {val_metrics['ade']:.4f} | FDE: {val_metrics['fde']:.4f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_metrics['loss'],
            "train/ade": train_metrics['ade'],
            "train/fde": train_metrics['fde'],
            "val/loss": val_metrics['loss'],
            "val/ade": val_metrics['ade'],
            "val/fde": val_metrics['fde'],
            "learning_rate": optimizer.param_groups[0]['lr'],
        })
        
        # Step scheduler
        scheduler.step()
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(vectornet_checkpoint_dir, 'best_vectornet_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': get_model_for_saving(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_ade': val_metrics['ade'],
                'val_fde': val_metrics['fde'],
                'config': config,
            }, checkpoint_path)
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print("\nEarly stopping triggered!")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(vectornet_checkpoint_dir, f'vectornet_epoch_{epoch+1:03d}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': get_model_for_saving(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
            }, checkpoint_path)
    
    # Save final model
    checkpoint_path = os.path.join(vectornet_checkpoint_dir, 'final_vectornet_model.pt')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': get_model_for_saving(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'config': config,
    }, checkpoint_path)
    print(f"\nSaved final model to {checkpoint_path}")
    
    # Finish wandb
    wandb.finish()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
