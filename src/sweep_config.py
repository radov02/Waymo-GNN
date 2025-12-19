"""Wandb Sweep configuration for hyperparameter tuning.

Usage:
    1. Create sweep:   wandb sweep src/sweep_config.py
    2. Run agent:      wandb agent <sweep_id>
    
Or run directly:
    python src/sweep_config.py
"""

import wandb
import torch
import os
import torch.nn.functional as F
from autoregressive_predictions.SpatioTemporalGNN import SpatioTemporalGNN
from gat_autoregressive.SpatioTemporalGAT import SpatioTemporalGAT
from dataset import HDF5ScenarioDataset
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from config import (device, num_workers, input_dim, output_dim, sequence_length, 
                    project_name, checkpoint_dir, use_edge_weights, use_gat,
                    gat_num_heads, loss_alpha, loss_beta, loss_gamma, loss_delta,
                    num_gpus, use_data_parallel, setup_model_parallel, get_model_for_saving,
                    pin_memory, prefetch_factor, use_amp)
from helper_functions.helpers import advanced_directional_loss
import random

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/sweep_config.py


# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================
sweep_config = {
    'method': 'bayes',  # 'grid', 'random', or 'bayes' (recommended)
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        # Model architecture
        'hidden_channels': {
            'values': [64, 128, 256]
        },
        'num_gcn_layers': {
            'values': [2, 3, 4]
        },
        'num_gru_layers': {
            'values': [1, 2]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.3
        },
        
        # Training hyperparameters
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'gradient_clip': {
            'values': [0.5, 1.0, 2.0, 5.0]
        },
        
        # Model type
        'use_gat': {
            'values': [True, False]  # Compare GAT vs GCN
        },
        
        # Loss weights
        'loss_alpha': {  # Angle loss weight
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.7
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
        's': 2
    }
}


def train_sweep():
    """Training function for wandb sweep - compares GCN vs GAT models."""
    
    # Initialize wandb run (sweep will set hyperparameters)
    run = wandb.init()
    config = wandb.config
    
    print(f"\n{'='*60}")
    print(f"SWEEP RUN: {run.name}")
    print(f"{'='*60}")
    print(f"Hyperparameters:")
    for key, value in dict(config).items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Create model based on use_gat parameter
    model_uses_gat = config.get('use_gat', use_gat)
    
    if model_uses_gat:
        model = SpatioTemporalGAT(
            input_dim=input_dim,
            hidden_dim=config.hidden_channels,
            output_dim=output_dim,
            num_gat_layers=config.num_gcn_layers,
            num_gru_layers=config.num_gru_layers,
            dropout=config.dropout,
            num_heads=gat_num_heads
        )
        print("Using SpatioTemporalGAT (Graph Attention Networks)")
    else:
        model = SpatioTemporalGNN(
            input_dim=input_dim,
            hidden_dim=config.hidden_channels,
            output_dim=output_dim,
            num_gcn_layers=config.num_gcn_layers,
            num_gru_layers=config.num_gru_layers,
            dropout=config.dropout,
            use_gat=False
        )
        print("Using SpatioTemporalGNN (Graph Convolutional Networks)")
    
    # Setup multi-GPU if available
    model, is_parallel = setup_model_parallel(model, device)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    wandb.log({'total_parameters': total_params})
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, 
                                   verbose=True, min_lr=1e-6)
    
    # Load datasets (use subset for faster sweeps)
    train_path = "./data/graphs/training/training.hdf5"
    val_path = "./data/graphs/validation/validation.hdf5"
    
    try:
        train_dataset_full = HDF5ScenarioDataset(train_path, seq_len=sequence_length)
        val_dataset_full = HDF5ScenarioDataset(val_path, seq_len=sequence_length)
    except FileNotFoundError as e:
        print(f"ERROR: Dataset not found - {e}")
        wandb.finish()
        return
    
    # Use subsets for faster hyperparameter search
    max_train = 500  # Use 500 training scenarios
    max_val = 100    # Use 100 validation scenarios
    
    if len(train_dataset_full) > max_train:
        train_indices = random.sample(range(len(train_dataset_full)), max_train)
        train_dataset = Subset(train_dataset_full, train_indices)
    else:
        train_dataset = train_dataset_full
    
    if len(val_dataset_full) > max_val:
        val_indices = random.sample(range(len(val_dataset_full)), max_val)
        val_dataset = Subset(val_dataset_full, val_indices)
    else:
        val_dataset = val_dataset_full
    
    print(f"Using {len(train_dataset)} train, {len(val_dataset)} val scenarios")
    
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch,
        drop_last=True, persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin_memory, prefetch_factor=prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch,
        drop_last=True, persistent_workers=True if num_workers > 0 else False,
        pin_memory=pin_memory, prefetch_factor=prefetch_factor
    )
    
    # Initialize GradScaler for AMP (only if enabled and CUDA available)
    scaler = None
    if use_amp and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        print("[AMP] Mixed Precision Training ENABLED")
    
    # Training loop (shorter for sweeps)
    num_epochs = 20  # Fewer epochs for hyperparameter search
    best_val_loss = float('inf')
    loss_fn = advanced_directional_loss
    use_amp_local = scaler is not None
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_cosine = 0.0
        train_steps = 0
        
        for batch_dict in train_loader:
            batched_graph_sequence = batch_dict["batch"]
            T = batch_dict["T"]
            
            for t in range(T):
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
            
            num_nodes = batched_graph_sequence[0].num_nodes
            model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
            
            optimizer.zero_grad()
            accumulated_loss = 0.0
            
            for t, graph in enumerate(batched_graph_sequence):
                if graph.y is None or torch.all(graph.y == 0):
                    continue
                
                # Forward pass with optional AMP
                with torch.amp.autocast('cuda', enabled=use_amp_local):
                    out = model(graph.x, graph.edge_index, batch=graph.batch,
                               batch_size=1, batch_num=0, timestep=t)
                    
                    loss_t = loss_fn(out, graph.y.to(out.dtype), graph.x,
                                    alpha=loss_alpha, beta=loss_beta,
                                    gamma=loss_gamma, delta=loss_delta)
                accumulated_loss += loss_t
                
                with torch.no_grad():
                    pred_norm = F.normalize(out, p=2, dim=1, eps=1e-6)
                    target_norm = F.normalize(graph.y.to(out.dtype), p=2, dim=1, eps=1e-6)
                    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                    train_cosine += cos_sim.item()
            
            # Backward pass with optional AMP scaling
            if use_amp_local:
                scaler.scale(accumulated_loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                accumulated_loss.backward()
                clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
            
            train_loss += accumulated_loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / max(1, train_steps)
        avg_train_cosine = train_cosine / max(1, train_steps * T)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_cosine = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch_dict in val_loader:
                batched_graph_sequence = batch_dict["batch"]
                T = batch_dict["T"]
                
                for t in range(T):
                    batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
                
                num_nodes = batched_graph_sequence[0].num_nodes
                model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
                
                accumulated_loss = 0.0
                
                for t, graph in enumerate(batched_graph_sequence):
                    if graph.y is None or torch.all(graph.y == 0):
                        continue
                    
                    out = model(graph.x, graph.edge_index, batch=graph.batch,
                               batch_size=1, batch_num=0, timestep=t)
                    
                    loss_t = loss_fn(out, graph.y.to(out.dtype), graph.x,
                                    alpha=loss_alpha, beta=loss_beta,
                                    gamma=loss_gamma, delta=loss_delta)
                    accumulated_loss += loss_t
                    
                    pred_norm = F.normalize(out, p=2, dim=1, eps=1e-6)
                    target_norm = F.normalize(graph.y.to(out.dtype), p=2, dim=1, eps=1e-6)
                    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                    val_cosine += cos_sim.item()
                
                val_loss += accumulated_loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / max(1, val_steps)
        avg_val_cosine = val_cosine / max(1, val_steps * T)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_cosine_sim': avg_train_cosine,
            'val_loss': avg_val_loss,
            'val_cosine_sim': avg_val_cosine,
            'learning_rate': current_lr,
            'model_type': 'GAT' if model_uses_gat else 'GCN'
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        # Track best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
    # Log final best metric
    wandb.log({'best_val_loss': best_val_loss})
    print(f"\nBest val_loss: {best_val_loss:.4f}")
    
    wandb.finish()


def run_sweep(count=20):
    """Create and run a wandb sweep.
    
    Args:
        count: Number of sweep runs to execute
    """
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"\nCreated sweep: {sweep_id}")
    print(f"View at: https://wandb.ai/{project_name}/sweeps/{sweep_id}")
    
    # Run sweep agent
    print(f"\nRunning {count} sweep iterations...")
    wandb.agent(sweep_id, function=train_sweep, count=count)


if __name__ == '__main__':
    print("="*60)
    print("WANDB HYPERPARAMETER SWEEP")
    print("="*60)
    print("\nThis will search for optimal hyperparameters using Bayesian optimization.")
    print("Parameters being tuned:")
    print("  - hidden_channels: [64, 128, 256]")
    print("  - num_gcn_layers: [2, 3, 4]")
    print("  - num_gru_layers: [1, 2]")
    print("  - dropout: [0.0 - 0.3]")
    print("  - learning_rate: [1e-4 - 1e-2]")
    print("  - gradient_clip: [0.5, 1.0, 2.0, 5.0]")
    print("  - use_gat: [True, False]")
    print("  - loss_alpha: [0.3 - 0.7]")
    print("\n" + "="*60)
    
    # Run sweep with 20 iterations (adjust as needed)
    run_sweep(count=20)
