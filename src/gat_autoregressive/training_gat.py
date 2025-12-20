"""Training script for GAT-based autoregressive trajectory prediction.

This module trains the SpatioTemporalGAT model from scratch using
Graph Attention Networks for spatial feature extraction.

Checkpoints are saved to checkpoints/gat/
Visualizations are saved to visualizations/autoreg/gat/
"""

import sys
import os
# Add parent directory (src/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
import torch.nn.functional as F
from SpatioTemporalGAT import SpatioTemporalGAT
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, num_gru_layers, epochs, 
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    visualize_every_n_epochs, debug_mode,
                    gradient_clip_value, loss_alpha, loss_beta, loss_gamma, loss_delta,
                    scheduler_patience, scheduler_factor, min_lr,
                    early_stopping_patience, early_stopping_min_delta,
                    num_gpus, use_data_parallel, setup_model_parallel, get_model_for_saving,
                    print_gpu_info, pin_memory, prefetch_factor, use_amp, use_bf16,
                    use_torch_compile, torch_compile_mode, use_gradient_checkpointing,
                    cache_validation_data, max_validation_scenarios)
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from helper_functions.visualization_functions import visualize_epoch, visualize_training_progress
from helper_functions.helpers import advanced_directional_loss, compute_metrics
import config as cfg  # Import config to override viz directory

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python src/gat_autoregressive/training_gat.py

# GAT-specific directories
CHECKPOINT_DIR = 'checkpoints/gat'
VIZ_DIR = 'visualizations/autoreg/gat'

# Override config's viz_training_dir for GAT
cfg.viz_training_dir = VIZ_DIR


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
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


def train_single_epoch(model, dataloader, optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, 
                       device, epoch, scaler=None, amp_dtype=torch.float16, visualize_callback=None):
    """Train for one epoch with optional AMP (Automatic Mixed Precision).
    
    Optimizations:
    - Reduced logging overhead (every N batches instead of every batch)
    - Pre-allocated metric accumulators
    - Efficient tensor operations
    """
    model.train()
    total_loss_epoch = 0.0
    steps = 0
    
    # Track direction metrics
    total_cosine_sim = 0.0
    total_mse = 0.0
    total_angular = 0.0
    metric_count = 0
    
    # Will store the last batch for visualization
    last_batch_dict = None
    
    # Determine if we're using AMP (either with scaler for FP16, or just autocast for BF16)
    use_amp_local = torch.cuda.is_available() and amp_dtype in (torch.float16, torch.bfloat16)
    
    # Logging frequency (reduce overhead)
    log_every_n_batches = max(1, len(dataloader) // 10)  # Log ~10 times per epoch
    total_batches = len(dataloader)
    
    for batch, batch_dict in enumerate(dataloader):
        batched_graph_sequence = batch_dict["batch"]
        B = batch_dict["B"]
        T = batch_dict["T"]
        
        num_nodes = batched_graph_sequence[0].num_nodes
        num_edges = batched_graph_sequence[0].edge_index.size(1)
        
        # Reduced logging - only log every N batches
        if batch % log_every_n_batches == 0:
            # Compute running averages for display
            avg_mse_so_far = total_mse / max(1, metric_count)
            avg_cos_so_far = total_cosine_sim / max(1, metric_count)
            rmse_meters = (avg_mse_so_far ** 0.5) * 100.0  # Convert normalized to meters
            
            if torch.cuda.is_available():
                print(f"\n[Epoch {epoch+1}] Batch {batch}/{total_batches} | Nodes: {num_nodes} | Edges: {num_edges} | T={T}")
                if metric_count > 0:
                    print(f"[METRICS] MSE={avg_mse_so_far:.6f} | RMSE={rmse_meters:.2f}m | CosSim={avg_cos_so_far:.4f}")
            else:
                print(f"Batch {batch}/{total_batches}: T={T} timesteps, Nodes={num_nodes}")
                if metric_count > 0:
                    print(f"  MSE={avg_mse_so_far:.6f} | RMSE={rmse_meters:.2f}m | CosSim={avg_cos_so_far:.4f}")

        # Move all timesteps to device in batch for efficiency
        for t in range(T): 
            batched_graph_sequence[t] = batched_graph_sequence[t].to(device, non_blocking=True)

        # Reset GRU hidden states for this scenario
        model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
        
        # Use set_to_none=True for slightly faster zeroing
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        
        for t, batched_graph in enumerate(batched_graph_sequence):
            if batched_graph.y is None or torch.all(batched_graph.y == 0):
                continue
            
            # Check for NaN in inputs
            if torch.isnan(batched_graph.x).any() or torch.isnan(batched_graph.y).any():
                print(f"  Warning: NaN detected in input data at batch {batch}, timestep {t}, skipping...")
                continue
            
            # Forward pass with optional AMP (supports FP16 and BF16)
            with torch.amp.autocast('cuda', enabled=use_amp_local, dtype=amp_dtype):
                # Forward pass - GAT doesn't use edge weights
                out_predictions = model(batched_graph.x, batched_graph.edge_index,
                                      batch=batched_graph.batch, batch_size=1, 
                                      batch_num=batch, timestep=t)
                
                # Check for NaN in predictions
                if torch.isnan(out_predictions).any():
                    print(f"  Warning: NaN in model output at batch {batch}, timestep {t}, skipping...")
                    continue
                
                loss_t = loss_fn(out_predictions, batched_graph.y.to(out_predictions.dtype), 
                               batched_graph.x, alpha=loss_alpha, beta=loss_beta, 
                               gamma=loss_gamma, delta=loss_delta)
                
                # Check for NaN in loss
                if torch.isnan(loss_t):
                    print(f"  Warning: NaN loss at batch {batch}, timestep {t}, skipping...")
                    continue
                    
            accumulated_loss += loss_t
            
            with torch.no_grad():
                mse = F.mse_loss(out_predictions, batched_graph.y.to(out_predictions.dtype))
                pred_norm = F.normalize(out_predictions, p=2, dim=1, eps=1e-6)
                target_norm = F.normalize(batched_graph.y.to(out_predictions.dtype), p=2, dim=1, eps=1e-6)
                cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                
                pred_angle = torch.atan2(out_predictions[:, 1], out_predictions[:, 0])
                target_angle = torch.atan2(batched_graph.y[:, 1], batched_graph.y[:, 0])
                angle_diff = pred_angle - target_angle
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                mean_angle_error = torch.abs(angle_diff).mean()
                
                total_mse += mse.item()
                total_cosine_sim += cos_sim.item()
                total_angular += mean_angle_error.item()
                metric_count += 1
        
        # Skip backward pass if no valid loss was accumulated
        if accumulated_loss == 0.0:
            print(f"  Warning: No valid loss at batch {batch}, skipping backward pass")
            continue
            
        # Check for NaN in accumulated loss before backward
        if isinstance(accumulated_loss, torch.Tensor) and torch.isnan(accumulated_loss):
            print(f"  Warning: NaN accumulated loss at batch {batch}, skipping backward pass")
            continue
        
        # Backward pass with optional AMP scaling (only needed for FP16, not BF16)
        if scaler is not None:
            scaler.scale(accumulated_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            accumulated_loss.backward()
            clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
            
        total_loss_epoch += accumulated_loss.item()
        steps += 1
        
        # Store this batch for potential visualization
        last_batch_dict = {
            'batch': [b.cpu() for b in batch_dict['batch']],
            'B': batch_dict['B'],
            'T': batch_dict['T'],
            'scenario_ids': batch_dict.get('scenario_ids', [])
        }

    avg_loss_epoch = total_loss_epoch / max(1, steps)
    avg_cosine_sim = total_cosine_sim / max(1, metric_count)
    avg_mse = total_mse / max(1, metric_count)
    avg_angle_error = total_angular / max(1, metric_count)
    
    # Print epoch summary with RMSE in meters
    rmse_meters = (avg_mse ** 0.5) * 100.0
    print(f"\n[TRAIN EPOCH {epoch+1} SUMMARY]")
    print(f"  Loss: {avg_loss_epoch:.6f} | MSE: {avg_mse:.6f} | RMSE: {rmse_meters:.2f}m")
    print(f"  CosSim: {avg_cosine_sim:.4f} | AngleErr: {avg_angle_error:.4f} rad")
    
    if visualize_callback is not None and last_batch_dict is not None:
        visualize_callback(epoch, last_batch_dict, model, device, wandb)
    
    return avg_loss_epoch, avg_mse, avg_cosine_sim, avg_angle_error, last_batch_dict


def validate_single_epoch(model, dataloader, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device):
    """Run validation loop and return metrics without gradient updates."""
    model.eval()
    total_loss = 0.0
    steps = 0
    
    total_cosine_sim = 0.0
    total_mse = 0.0
    total_angular = 0.0
    metric_count = 0
    
    total_batches = len(dataloader)
    print(f"  Running validation on {total_batches} scenarios...")
    
    with torch.no_grad():
        for batch, batch_dict in enumerate(dataloader):
            if batch % 500 == 0:
                print(f"    Validation progress: {batch}/{total_batches} ({100*batch/total_batches:.1f}%)")
            
            batched_graph_sequence = batch_dict["batch"]
            B = batch_dict["B"]
            T = batch_dict["T"]

            for t in range(T): 
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device)

            # Reset GRU for this validation scenario
            num_nodes = batched_graph_sequence[0].num_nodes
            model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
            
            accumulated_loss = 0.0
            
            for t, batched_graph in enumerate(batched_graph_sequence):
                if batched_graph.y is None or torch.all(batched_graph.y == 0):
                    continue
                
                # GAT doesn't use edge weights
                out_predictions = model(batched_graph.x, batched_graph.edge_index,
                                      batch=batched_graph.batch, batch_size=1, 
                                      batch_num=batch, timestep=t)
                
                loss_t = loss_fn(out_predictions, batched_graph.y.to(out_predictions.dtype), 
                               batched_graph.x, alpha=loss_alpha, beta=loss_beta, 
                               gamma=loss_gamma, delta=loss_delta)
                accumulated_loss += loss_t
                
                # Track metrics
                mse = F.mse_loss(out_predictions, batched_graph.y.to(out_predictions.dtype))
                pred_norm = F.normalize(out_predictions, p=2, dim=1, eps=1e-6)
                target_norm = F.normalize(batched_graph.y.to(out_predictions.dtype), p=2, dim=1, eps=1e-6)
                cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                
                pred_angle = torch.atan2(out_predictions[:, 1], out_predictions[:, 0])
                target_angle = torch.atan2(batched_graph.y[:, 1], batched_graph.y[:, 0])
                angle_diff = pred_angle - target_angle
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                mean_angle_error = torch.abs(angle_diff).mean()
                
                total_mse += mse.item()
                total_cosine_sim += cos_sim.item()
                total_angular += mean_angle_error.item()
                metric_count += 1
            
            total_loss += accumulated_loss.item()
            steps += 1

    avg_loss = total_loss / max(1, steps)
    avg_cosine_sim = total_cosine_sim / max(1, metric_count)
    avg_mse = total_mse / max(1, metric_count)
    avg_angle_error = total_angular / max(1, metric_count)
    
    # Print validation summary with RMSE in meters
    rmse_meters = (avg_mse ** 0.5) * 100.0
    print(f"\n[VALIDATION SUMMARY]")
    print(f"  Loss: {avg_loss:.6f} | MSE: {avg_mse:.6f} | RMSE: {rmse_meters:.2f}m")
    print(f"  CosSim: {avg_cosine_sim:.4f} | AngleErr: {avg_angle_error:.4f} rad")
    
    return avg_loss, avg_mse, avg_cosine_sim, avg_angle_error


def run_training(dataset_path="./data/graphs/training/training_seqlen90.hdf5", 
                validation_path="./data/graphs/validation/validation_seqlen90.hdf5",
                wandb_run=None, return_viz_batch=False):
    """Run full GAT training loop."""
    
    # Print GPU info at startup
    print_gpu_info()
    
    # Create GAT-specific directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    should_finish_wandb = False
    if wandb_run is None:
        wandb.login()
        wandb_run = wandb.init(
            project=project_name,
            config={
                "model": "SpatioTemporalGAT",
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dataset": dataset_name,
                "dropout": dropout,
                "hidden_channels": hidden_channels,
                "num_gat_layers": num_layers,
                "num_gru_layers": num_gru_layers,
                "use_gat": True,  # Always true for this module
                "num_heads": 4,
                "epochs": epochs,
                "loss_alpha": loss_alpha,
                "loss_beta": loss_beta,
                "loss_gamma": loss_gamma,
                "loss_delta": loss_delta,
                "scheduler": "ReduceLROnPlateau",
                "early_stopping_patience": early_stopping_patience
            },
            name=f"SpatioTemporalGAT_r{radius}_h{hidden_channels}",
            dir="../wandb"
        )
        should_finish_wandb = True

    # Initialize SpatioTemporalGAT model with optional gradient checkpointing
    model = SpatioTemporalGAT(
        input_dim=input_dim,
        hidden_dim=hidden_channels,
        output_dim=output_dim,
        num_gat_layers=num_layers,
        num_gru_layers=num_gru_layers,
        dropout=dropout,
        num_heads=4,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    
    if use_gradient_checkpointing:
        print("[Memory] Gradient checkpointing ENABLED - trading compute for memory")
    
    # Apply torch.compile() for H100/A100 optimization (PyTorch 2.0+)
    if use_torch_compile and hasattr(torch, 'compile'):
        try:
            print(f"[torch.compile] Compiling model with mode='{torch_compile_mode}'...")
            model = torch.compile(model, mode=torch_compile_mode)
            print("[torch.compile] Model compiled successfully!")
        except Exception as e:
            print(f"[torch.compile] Failed to compile: {e}. Running without compilation.")
    
    # Setup multi-GPU if available
    model, is_parallel = setup_model_parallel(model, device)
    wandb.watch(model, log='all', log_freq=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, 
                                  patience=scheduler_patience, verbose=True, min_lr=min_lr)
    loss_fn = advanced_directional_loss

    # Load training dataset
    try:
        dataset = HDF5ScenarioDataset(dataset_path, seq_len=sequence_length)
    except FileNotFoundError:
        print(f"ERROR: {dataset_path} not found!")
        print("Please run graph_creation_and_saving.py first to create HDF5 files.")
        if should_finish_wandb:
            wandb_run.finish()
        exit(1)
    
    # Load validation dataset with optional caching
    val_dataloader = None
    max_val_scenarios = max_validation_scenarios  # From config.py
    try:
        # Enable caching for validation data if configured
        val_dataset_full = HDF5ScenarioDataset(
            validation_path, 
            seq_len=sequence_length,
            cache_in_memory=cache_validation_data,
            max_cache_size=max_val_scenarios
        )
        
        if len(val_dataset_full) > max_val_scenarios:
            val_indices = random.sample(range(len(val_dataset_full)), max_val_scenarios)
            val_dataset = Subset(val_dataset_full, val_indices)
            print(f"Validation: using {max_val_scenarios} random scenarios (from {len(val_dataset_full)} total)")
            if cache_validation_data:
                print(f"  [Cache] Validation caching enabled - scenarios will be cached in RAM")
        else:
            val_dataset = val_dataset_full
            print(f"Validation dataset size: {len(val_dataset)} scenarios")
        
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers,
                                    collate_fn=collate_graph_sequences_to_batch, 
                                    drop_last=True,
                                    pin_memory=pin_memory,
                                    prefetch_factor=prefetch_factor,
                                    persistent_workers=True if num_workers > 0 else False)
    except FileNotFoundError:
        print(f"WARNING: {validation_path} not found! Training without validation.")

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers,
                            collate_fn=collate_graph_sequences_to_batch, 
                            drop_last=True,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            persistent_workers=True if num_workers > 0 else False) 

    # Debug: Check first batch for NaN values
    print("\n[DEBUG] Checking first batch for NaN values...")
    debug_batch = next(iter(dataloader))
    for t, graph in enumerate(debug_batch['batch'][:3]):  # Check first 3 timesteps
        has_nan_x = torch.isnan(graph.x).any()
        has_nan_y = torch.isnan(graph.y).any() if graph.y is not None else False
        has_inf_x = torch.isinf(graph.x).any()
        print(f"  Timestep {t}: x_nan={has_nan_x}, y_nan={has_nan_y}, x_inf={has_inf_x}, "
              f"x_range=[{graph.x.min():.4f}, {graph.x.max():.4f}]")
    print("[DEBUG] First batch check complete\n")

    # Initialize early stopping
    early_stopper = EarlyStopping(
        patience=early_stopping_patience, 
        min_delta=early_stopping_min_delta, 
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print(f"GAT TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Model: SpatioTemporalGAT (Graph Attention Networks)")
    print(f"Dataset size: {len(dataset)} scenarios")
    print(f"Batch size: {batch_size}, num workers: {num_workers}")
    print(f"Sequence length: {sequence_length}")
    print(f"Learning rate: {learning_rate}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Visualizations: {VIZ_DIR}")
    print(f"Mixed Precision (AMP): {'BF16' if use_bf16 else 'FP16' if use_amp else 'Disabled'}")
    print(f"{'='*80}\n")
    
    # Initialize GradScaler for AMP (only if enabled and CUDA available)
    # Note: BF16 doesn't need gradient scaling (same exponent range as FP32)
    scaler = None
    amp_dtype = torch.float16  # Default to FP16
    if use_amp and torch.cuda.is_available():
        if use_bf16:
            amp_dtype = torch.bfloat16
            print("[AMP] Mixed Precision Training ENABLED - using BFLOAT16 (H100/A100 optimized)")
            # BF16 doesn't need scaler, but we still use autocast
        else:
            scaler = torch.amp.GradScaler('cuda')
            print("[AMP] Mixed Precision Training ENABLED - using FLOAT16")
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    last_viz_batch = None

    for epoch in range(epochs):
        avg_loss_epoch, avg_mse, avg_cosine_sim, avg_angle_error, last_viz_batch = train_single_epoch(
            model, dataloader, optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta,
            device, epoch, scaler=scaler, amp_dtype=amp_dtype, visualize_callback=visualize_epoch
        )
        
        avg_angle_error_deg = avg_angle_error * 180 / 3.14159
        
        # Validation loop
        val_loss, val_mse, val_cosine_sim, val_angle_error = 0.0, 0.0, 0.0, 0.0
        if val_dataloader is not None:
            val_loss, val_mse, val_cosine_sim, val_angle_error = validate_single_epoch(
                model, val_dataloader, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device
            )
            val_angle_error_deg = val_angle_error * 180 / 3.14159
            
            scheduler.step(val_loss)
            early_stopper(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
                model_to_save = get_model_for_saving(model, is_parallel)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': avg_loss_epoch,
                    'config': {
                        'input_dim': input_dim,
                        'hidden_channels': hidden_channels,
                        'output_dim': output_dim,
                        'num_layers': num_layers,
                        'num_gru_layers': num_gru_layers,
                        'dropout': dropout,
                        'use_gat': True,  # Always true for GAT model
                        'num_heads': 4
                    }
                }, checkpoint_path)
                print(f"  -> Saved best GAT model (val_loss: {val_loss:.6f}) to {checkpoint_path}")
        else:
            scheduler.step(avg_loss_epoch)
            early_stopper(avg_loss_epoch)
            if avg_loss_epoch < best_train_loss:
                best_train_loss = avg_loss_epoch
                checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
                model_to_save = get_model_for_saving(model, is_parallel)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_loss_epoch,
                    'config': {
                        'input_dim': input_dim,
                        'hidden_channels': hidden_channels,
                        'output_dim': output_dim,
                        'num_layers': num_layers,
                        'num_gru_layers': num_gru_layers,
                        'dropout': dropout,
                        'use_gat': True,
                        'num_heads': 4
                    }
                }, checkpoint_path)
                print(f"  -> Saved best GAT model (train_loss: {avg_loss_epoch:.6f}) to {checkpoint_path}")
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics to wandb
        log_dict = {
            "epoch": epoch,
            "train_loss": avg_loss_epoch,
            "train_cosine_similarity": avg_cosine_sim,
            "train_mse": avg_mse,
            "train_angle_error_rad": avg_angle_error,
            "train_angle_error_deg": avg_angle_error_deg,
            "learning_rate": current_lr
        }
        
        if val_dataloader is not None:
            log_dict.update({
                "val_loss": val_loss,
                "val_cosine_similarity": val_cosine_sim,
                "val_mse": val_mse,
                "val_angle_error_rad": val_angle_error,
                "val_angle_error_deg": val_angle_error_deg
            })
        
        wandb.log(log_dict)
        
        # Print progress
        if val_dataloader is not None:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_loss_epoch:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Train Cos: {avg_cosine_sim:.4f} | Val Cos: {val_cosine_sim:.4f} | LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss_epoch:.6f} | Cosine Sim: {avg_cosine_sim:.4f} | LR: {current_lr:.2e}")
        
        if early_stopper.early_stop:
            print(f"\n Early stopping triggered at epoch {epoch+1}!")
            break
    
    # Save final model
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'final_model.pt')
    model_to_save = get_model_for_saving(model, is_parallel)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss if val_dataloader is not None else None,
        'final_train_loss': avg_loss_epoch,
        'early_stopped': early_stopper.early_stop,
        'config': {
            'input_dim': input_dim,
            'hidden_channels': hidden_channels,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'num_gru_layers': num_gru_layers,
            'dropout': dropout,
            'use_gat': True,
            'num_heads': 4
        }
    }, final_checkpoint_path)
    
    print(f"\n{'='*80}")
    print(f"GAT TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best model: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")
    print(f"Final model: {final_checkpoint_path}")
    print(f"{'='*80}\n")

    if should_finish_wandb:
        wandb_run.finish()
    
    if return_viz_batch:
        return model, last_viz_batch
    return model


if __name__ == '__main__':
    run_training()
