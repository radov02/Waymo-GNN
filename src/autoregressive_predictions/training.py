import sys
import os
# Add parent directory (src/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
import torch.nn.functional as F
from SpatioTemporalGNN import SpatioTemporalGNN
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, num_gru_layers, epochs, 
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    visualize_every_n_epochs, debug_mode,
                    gradient_clip_value, loss_alpha, loss_beta, loss_gamma, loss_delta, use_edge_weights,
                    checkpoint_dir, scheduler_patience, scheduler_factor, min_lr,
                    early_stopping_patience, early_stopping_min_delta,
                    num_gpus, use_data_parallel, setup_model_parallel, get_model_for_saving,
                    print_gpu_info, pin_memory, prefetch_factor, use_amp, use_bf16,
                    use_torch_compile, torch_compile_mode,
                    max_validation_scenarios, cache_validation_data)
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from helper_functions.visualization_functions import visualize_epoch, visualize_training_progress
from helper_functions.helpers import advanced_directional_loss, compute_metrics
import config  # Import config to override viz directory

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/autoregressive_predictions/training.py

# Set viz directory for autoregressive training (GCN model - GAT has separate module)
VIZ_DIR = 'visualizations/autoreg'
config.viz_training_dir = VIZ_DIR


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
    """Train for one epoch with optional AMP (Automatic Mixed Precision)."""
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
    
    for batch, batch_dict in enumerate(dataloader):
        batched_graph_sequence = batch_dict["batch"]  # list of length T of Batch objects
        B = batch_dict["B"]  # always 1 for per-scenario temporal processing
        T = batch_dict["T"]
        
        # Print parallel execution info
        num_nodes = batched_graph_sequence[0].num_nodes
        num_edges = batched_graph_sequence[0].edge_index.size(1)
        if torch.cuda.is_available():
            print(f"\n[CUDA] Batch {batch} | Device: {device} | Nodes: {num_nodes} | Edges: {num_edges} | Timesteps: {T}")
            print(f"[CUDA] Launching parallel GCN convolution across {num_nodes} nodes, {num_edges} edges...")
            torch.cuda.synchronize()  # Ensure accurate timing
        else:
            print(f"Batch {batch}: Processing scenario, T={T} timesteps, Nodes={num_nodes}")

        for t in range(T): 
            batched_graph_sequence[t] = batched_graph_sequence[t].to(device)

        # Reset GRU hidden states for this scenario (per-agent hidden states)
        model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for t, batched_graph in enumerate(batched_graph_sequence):      # process entire sequence for this scenario
            if batched_graph.y is None or torch.all(batched_graph.y == 0):
                continue
            
            # Forward pass with optional AMP (supports FP16 and BF16)
            with torch.amp.autocast('cuda', enabled=use_amp_local, dtype=amp_dtype):
                # forward pass (batch_size=1, so all nodes belong to single scenario):
                edge_w = batched_graph.edge_attr if use_edge_weights else None
                out_predictions = model(batched_graph.x, batched_graph.edge_index,
                                      edge_weight=edge_w,
                                      batch=batched_graph.batch, batch_size=1, 
                                      batch_num=batch, timestep=t)
                
                loss_t = loss_fn(out_predictions, batched_graph.y.to(out_predictions.dtype), 
                               batched_graph.x, alpha=loss_alpha, beta=loss_beta, 
                               gamma=loss_gamma, delta=loss_delta)
            accumulated_loss += loss_t
            
            with torch.no_grad():       # track metrics:
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
        
        # Store this batch for potential visualization (last batch will be kept)
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
    
    # Call visualization callback with the last batch of this epoch
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
                
                edge_w = batched_graph.edge_attr if use_edge_weights else None
                out_predictions = model(batched_graph.x, batched_graph.edge_index,
                                      edge_weight=edge_w,
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
    
    return avg_loss, avg_mse, avg_cosine_sim, avg_angle_error


def run_training(dataset_path="./data/graphs/training/training_seqlen90.hdf5", 
                validation_path="./data/graphs/validation/validation_seqlen90.hdf5",
                wandb_run=None, return_viz_batch=False):
    """run full training loop for given epochs."""
    
    # Print GPU info at startup
    print_gpu_info()
    
    should_finish_wandb = False
    if wandb_run is None:
        wandb.login()
        wandb_run = wandb.init(
            project=project_name,
            config={
                "model": "SpatioTemporalGNN",
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dataset": dataset_name,
                "dropout": dropout,
                "hidden_channels": hidden_channels,
                "num_gcn_layers": num_layers,
                "num_gru_layers": num_gru_layers,
                "epochs": epochs,
                "loss_alpha": loss_alpha,
                "loss_beta": loss_beta,
                "loss_gamma": loss_gamma,
                "loss_delta": loss_delta,
                "use_edge_weights": use_edge_weights,
                "scheduler": "ReduceLROnPlateau",
                "early_stopping_patience": early_stopping_patience
            },
            name=f"SpatioTemporalGNN_r{radius}_h{hidden_channels}{'_ew' if use_edge_weights else ''}",
            dir="../wandb"
        )
        should_finish_wandb = True

    # Initialize SpatioTemporalGNN (GCN-based - GAT has separate module in gat_autoregressive/)
    model = SpatioTemporalGNN(
        input_dim=input_dim,
        hidden_dim=hidden_channels,
        output_dim=output_dim,
        num_gcn_layers=num_layers,
        num_gru_layers=num_gru_layers,
        dropout=dropout
    )
    
    # Apply torch.compile() for H100/A100 optimization (PyTorch 2.0+)
    if use_torch_compile and hasattr(torch, 'compile'):
        try:
            print(f"[torch.compile] Compiling model with mode='{torch_compile_mode}'...")
            model = torch.compile(model, mode=torch_compile_mode)
            print("[torch.compile] Model compiled successfully!")
        except Exception as e:
            print(f"[torch.compile] Failed to compile: {e}. Running without compilation.")
    
    # Setup multi-GPU if available (only on GPU, not CPU)
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
        print("Please run graph_creation_and_saving.py first to create HDF5 files from .tfrecord files.")
        print("Make sure you have .tfrecord files in ./data/scenario/training/ folder.")
        if should_finish_wandb:
            wandb_run.finish()
        exit(1)
    
    # Load validation dataset (use random subset for speed)
    val_dataloader = None
    max_val_scenarios = max_validation_scenarios  # From config.py
    try:
        val_dataset_full = HDF5ScenarioDataset(
            validation_path, 
            seq_len=sequence_length,
            cache_in_memory=cache_validation_data,
            max_cache_size=max_val_scenarios
        )
        
        # Use random subset if dataset is large
        if len(val_dataset_full) > max_val_scenarios:
            val_indices = random.sample(range(len(val_dataset_full)), max_val_scenarios)
            val_dataset = Subset(val_dataset_full, val_indices)
            print(f"Validation: using {max_val_scenarios} random scenarios (from {len(val_dataset_full)} total)")
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
        print("To enable validation, run graph_creation_and_saving.py for validation data.")

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers,    # number of working threads that are used for data loading
                            collate_fn=collate_graph_sequences_to_batch, 
                            drop_last=True,         # discard last minibatch if it does not have enough snapshots
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            persistent_workers=True if num_workers > 0 else False) 

    # Create checkpoint and visualization directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # Initialize early stopping
    early_stopper = EarlyStopping(
        patience=early_stopping_patience, 
        min_delta=early_stopping_min_delta, 
        verbose=True
    )
    
    print(f"Training on {device}")
    print(f"Dataset size: {len(dataset)} scenarios")
    print(f"Batch size: {batch_size}, num workers: {num_workers}")
    print(f"Sequence length: {sequence_length}")
    print(f"Learning rate: {learning_rate} (with ReduceLROnPlateau scheduler)")
    print(f"Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    print(f"Mixed Precision (AMP): {'BF16' if use_bf16 else 'FP16' if use_amp else 'Disabled'}")
    print(f"Model checkpoints will be saved to: {checkpoint_dir}\n")
    
    # Initialize GradScaler for AMP (only if enabled and CUDA available)
    # Note: BF16 doesn't need gradient scaling (same exponent range as FP32)
    scaler = None
    amp_dtype = torch.float16  # Default to FP16
    if use_amp and torch.cuda.is_available():
        if use_bf16:
            amp_dtype = torch.bfloat16
            print("[AMP] Mixed Precision Training ENABLED - using BFLOAT16 (H100/A100 optimized)")
        else:
            scaler = torch.amp.GradScaler('cuda')
            print("[AMP] Mixed Precision Training ENABLED - using FLOAT16")
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    last_viz_batch = None  # Will store the last batch from the final epoch

    for epoch in range(epochs):
        # Train and visualize the last batch of each epoch
        avg_loss_epoch, avg_mse, avg_cosine_sim, avg_angle_error, last_viz_batch = train_single_epoch(
            model, dataloader, optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta,
            device, epoch, scaler=scaler, amp_dtype=amp_dtype, visualize_callback=visualize_epoch
        )
        
        avg_angle_error_deg = avg_angle_error * 180 / 3.14159  # convert to degrees for readability
        
        # Validation loop
        val_loss, val_mse, val_cosine_sim, val_angle_error = 0.0, 0.0, 0.0, 0.0
        if val_dataloader is not None:
            val_loss, val_mse, val_cosine_sim, val_angle_error = validate_single_epoch(
                model, val_dataloader, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device
            )
            val_angle_error_deg = val_angle_error * 180 / 3.14159
            
            # Update learning rate scheduler based on validation loss
            scheduler.step(val_loss)
            
            # Check early stopping
            early_stopper(val_loss)
            
            # Track best model and save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                # Get underlying model (unwrap DataParallel if needed)
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
                        'dropout': dropout
                    }
                }, checkpoint_path)
                print(f"  -> Saved best model (val_loss: {val_loss:.6f}) to {checkpoint_path}")
        else:
            # If no validation, use training loss for scheduler and save best based on train loss
            scheduler.step(avg_loss_epoch)
            early_stopper(avg_loss_epoch)  # Use train loss for early stopping if no validation
            if avg_loss_epoch < best_train_loss:
                best_train_loss = avg_loss_epoch
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                # Get underlying model (unwrap DataParallel if needed)
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
                        'dropout': dropout
                    }
                }, checkpoint_path)
                print(f"  -> Saved best model (train_loss: {avg_loss_epoch:.6f}) to {checkpoint_path}")
        
        # Get current learning rate
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
                  f"Train Cos: {avg_cosine_sim:.4f} | Val Cos: {val_cosine_sim:.4f} | "
                  f"Train Angle: {avg_angle_error_deg:.1f}° | Val Angle: {val_angle_error_deg:.1f}° | LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss_epoch:.6f} | Cosine Sim: {avg_cosine_sim:.4f} | "
                  f"Angle Err: {avg_angle_error_deg:.1f}° | LR: {current_lr:.2e}")
        
        # Check early stopping
        if early_stopper.early_stop:
            print(f"\n Early stopping triggered at epoch {epoch+1}!")
            print(f"   Best {'validation' if val_dataloader else 'training'} loss: {early_stopper.best_loss:.6f}")
            break
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pt')
    # Get underlying model (unwrap DataParallel if needed)
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
            'dropout': dropout
        }
    }, final_checkpoint_path)
    print(f"\n Training complete! Final model saved to {final_checkpoint_path}")
    print(f" Best model saved to {os.path.join(checkpoint_dir, 'best_model.pt')}")

    if should_finish_wandb:
        wandb_run.finish()
    
    if return_viz_batch:
        return model, last_viz_batch
    return model


if __name__ == '__main__':
    run_training()
