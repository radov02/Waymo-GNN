"""Batched Training script for GCN-based autoregressive trajectory prediction.

This version supports batch_size > 1 for better GPU utilization.
Key improvements:
- Process multiple scenarios in parallel
- 2-4x speedup on modern GPUs
- Same model quality as batch_size=1

Usage:
    python src/autoregressive_predictions/training_batched.py

Checkpoints: checkpoints/
Visualizations: visualizations/autoreg/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp
import wandb
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
import threading
import warnings
warnings.filterwarnings("ignore", message=r"skipping cudagraphs due to graph with symbolic shapes*")


# Ensure safe start method (avoids inheriting open HDF5 handles on Linux)
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Use file-system backed shared memory to avoid exhausting file descriptors
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("[mp] set_sharing_strategy('file_system')")
except Exception:
    pass

# Configure torch.inductor for dynamic graph sizes (reduce cudagraph overhead)
try:
    if hasattr(torch, '_inductor'):
        cfg = torch._inductor.config
        if hasattr(cfg, 'triton'):
            cfg.triton.cudagraph_skip_dynamic_graphs = True
            cfg.triton.cudagraph_dynamic_shape_warn_limit = None
            print("[inductor] configured cudagraph_skip_dynamic_graphs=True")
except Exception:
    pass

# Global visualization thread pool (reused across epochs)
_viz_executor = None

def get_viz_executor():
    """Get or create a thread pool for async visualization."""
    global _viz_executor
    if _viz_executor is None:
        _viz_executor = threading.Thread
    return _viz_executor
from SpatioTemporalGNN_batched import SpatioTemporalGNNBatched
from dataset import HDF5ScenarioDataset
from config import (device, num_workers, num_layers, num_gru_layers, epochs, 
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    visualize_every_n_epochs, debug_mode,
                    gradient_clip_value, loss_alpha, loss_beta, loss_gamma, loss_delta, use_edge_weights,
                    checkpoint_dir, scheduler_patience, scheduler_factor, min_lr,
                    early_stopping_patience, early_stopping_min_delta,
                    num_gpus, use_data_parallel, setup_model_parallel, get_model_for_saving,
                    print_gpu_info, pin_memory, prefetch_factor, use_amp, use_bf16,
                    use_torch_compile, torch_compile_mode, use_gradient_checkpointing,
                    max_validation_scenarios, cache_validation_data)
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from helper_functions.visualization_functions import visualize_epoch, visualize_training_progress
from helper_functions.helpers import advanced_directional_loss, compute_metrics
import config
from config import batch_size

# ============== BATCHED TRAINING CONFIGURATION ==============
# batch_size is imported from config.py
BATCHED_BATCH_SIZE = batch_size  # Use batch_size from config.py

# Set viz directory for autoregressive training
VIZ_DIR = 'visualizations/autoreg'
config.viz_training_dir = VIZ_DIR


# ============== WORKER INIT FUNCTION (MODULE LEVEL FOR PICKLING) ==============
def worker_init_fn(worker_id):
    """Initialize worker process with fresh HDF5 file handle.
    Must be at module level to be pickleable for spawn multiprocessing.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset
    # Unwrap Subset if present
    if hasattr(dataset_obj, 'dataset'):
        dataset_obj = dataset_obj.dataset
    if hasattr(dataset_obj, 'init_worker'):
        dataset_obj.init_worker()


class OverfittingDetector:
    """Detect overfitting by tracking when train loss decreases but val loss increases.
    
    Stops training after patience consecutive epochs of overfitting.
    """
    
    def __init__(self, patience=4, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.overfit_counter = 0
        self.best_val_loss = None
        self.prev_train_loss = None
        self.prev_val_loss = None
        self.should_stop = False
        
    def __call__(self, train_loss, val_loss):
        """Check if we're overfitting: train loss decreasing but val loss increasing."""
        
        # Initialize on first call
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.prev_train_loss = train_loss
            self.prev_val_loss = val_loss
            return
        
        # Track best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        # Detect overfitting: train improves but val gets worse
        train_improved = train_loss < (self.prev_train_loss - self.min_delta)
        val_worsened = val_loss > (self.prev_val_loss + self.min_delta)
        
        if train_improved and val_worsened:
            self.overfit_counter += 1
            if self.verbose:
                print(f"  Overfitting detected: train↓ {self.prev_train_loss:.4f}→{train_loss:.4f}, "
                      f"val↑ {self.prev_val_loss:.4f}→{val_loss:.4f} "
                      f"({self.overfit_counter}/{self.patience})")
            
            if self.overfit_counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"\n  STOPPING: Overfitting detected for {self.patience} consecutive epochs!")
        else:
            # Reset counter if not overfitting
            if self.overfit_counter > 0 and self.verbose:
                print(f"  No overfitting this epoch, counter reset")
            self.overfit_counter = 0
        
        self.prev_train_loss = train_loss
        self.prev_val_loss = val_loss
            
    def reset(self):
        self.overfit_counter = 0
        self.best_val_loss = None
        self.prev_train_loss = None
        self.prev_val_loss = None
        self.should_stop = False


def get_module(model):
    """Get underlying model (unwrap DataParallel/compiled models)."""
    if hasattr(model, 'module'):
        return model.module
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def train_single_epoch_batched(model, dataloader, optimizer, loss_fn, 
                                loss_alpha, loss_beta, loss_gamma, loss_delta,
                                device, epoch, scaler=None, amp_dtype=torch.float16,
                                visualize_callback=None):
    """Train for one epoch with BATCHED scenario processing.
    
    Key difference from original: Processes B scenarios in parallel per batch.
    """
    model.train()
    total_loss_epoch = 0.0
    steps = 0
    
    total_cosine_sim = 0.0
    total_mse = 0.0
    total_angular = 0.0
    metric_count = 0
    
    last_batch_dict = None
    use_amp_local = torch.cuda.is_available() and amp_dtype in (torch.float16, torch.bfloat16)
    
    log_every_n_batches = max(1, len(dataloader) // 10)
    total_batches = len(dataloader)
    
    for batch_idx, batch_dict in enumerate(dataloader):
        batched_graph_sequence = batch_dict["batch"]  # List of T Batch objects
        B = batch_dict["B"]  # Number of scenarios in this batch
        T = batch_dict["T"]  # Sequence length
        
        # Get node counts for this batch
        total_nodes = batched_graph_sequence[0].num_nodes
        total_edges = batched_graph_sequence[0].edge_index.size(1)
        
        if batch_idx % log_every_n_batches == 0:
            # Compute running averages for display
            avg_mse_so_far = total_mse / max(1, metric_count)
            avg_cos_so_far = total_cosine_sim / max(1, metric_count)
            rmse_meters = (avg_mse_so_far ** 0.5) * 100.0  # Convert normalized to meters
            
            if torch.cuda.is_available():
                print(f"\n[Epoch {epoch+1}] Batch {batch_idx}/{total_batches} | "
                      f"Scenarios: {B} | Total Nodes: {total_nodes} | Edges: {total_edges} | T={T}")
                print(f"[PARALLEL] Processing {B} scenarios simultaneously on GPU")
                if metric_count > 0:
                    print(f"[METRICS] MSE={avg_mse_so_far:.6f} | RMSE={rmse_meters:.2f}m | CosSim={avg_cos_so_far:.4f}")
            else:
                print(f"Batch {batch_idx}/{total_batches}: B={B}, T={T}, Nodes={total_nodes}")
                if metric_count > 0:
                    print(f"  MSE={avg_mse_so_far:.6f} | RMSE={rmse_meters:.2f}m | CosSim={avg_cos_so_far:.4f}")

        # Move all timesteps to device
        for t in range(T):
            batched_graph_sequence[t] = batched_graph_sequence[t].to(device, non_blocking=True)

        # Reset GRU hidden states for this batch of scenarios
        base_model = get_module(model)
        base_model.reset_gru_hidden_states(num_agents=total_nodes, device=device)
        
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        
        for t, batched_graph in enumerate(batched_graph_sequence):
            if batched_graph.y is None or torch.all(batched_graph.y == 0):
                continue
            
            if torch.isnan(batched_graph.x).any() or torch.isnan(batched_graph.y).any():
                print(f"  Warning: NaN in input at batch {batch_idx}, t={t}, skipping...")
                continue
            
            with torch.amp.autocast('cuda', enabled=use_amp_local, dtype=amp_dtype):
                # Forward pass with batch info
                edge_w = batched_graph.edge_attr if use_edge_weights else None
                out_predictions = model(
                    batched_graph.x, 
                    batched_graph.edge_index,
                    edge_weight=edge_w,
                    batch=batched_graph.batch,  # Which scenario each node belongs to
                    batch_size=B,  # Number of scenarios
                    batch_num=batch_idx, 
                    timestep=t
                )
                
                if torch.isnan(out_predictions).any():
                    print(f"  Warning: NaN in predictions at batch {batch_idx}, t={t}, skipping...")
                    continue
                
                loss_t = loss_fn(out_predictions, batched_graph.y.to(out_predictions.dtype), 
                               batched_graph.x, alpha=loss_alpha, beta=loss_beta, 
                               gamma=loss_gamma, delta=loss_delta)
                
                if torch.isnan(loss_t):
                    print(f"  Warning: NaN loss at batch {batch_idx}, t={t}, skipping...")
                    continue
                    
            accumulated_loss += loss_t
            
            with torch.no_grad():
                mse = F.mse_loss(out_predictions, batched_graph.y.to(out_predictions.dtype))
                pred_norm = F.normalize(out_predictions, p=2, dim=1, eps=1e-6)
                target_norm = F.normalize(batched_graph.y.to(out_predictions.dtype), p=2, dim=1, eps=1e-6)
                cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                
                pred_angle = torch.atan2(out_predictions[:, 1], out_predictions[:, 0])
                target_angle = torch.atan2(batched_graph.y[:, 1], batched_graph.y[:, 0])
                angle_diff = torch.atan2(torch.sin(pred_angle - target_angle), 
                                        torch.cos(pred_angle - target_angle))
                mean_angle_error = torch.abs(angle_diff).mean()
                
                # Only accumulate if values are valid (not NaN)
                if not torch.isnan(mse):
                    total_mse += mse.item()
                if not torch.isnan(cos_sim):
                    total_cosine_sim += cos_sim.item()
                if not torch.isnan(mean_angle_error):
                    total_angular += mean_angle_error.item()
                metric_count += 1
        
        if accumulated_loss == 0.0:
            print(f"  Warning: No valid loss at batch {batch_idx}, skipping backward")
            continue
            
        if isinstance(accumulated_loss, torch.Tensor) and torch.isnan(accumulated_loss):
            print(f"  Warning: NaN accumulated loss at batch {batch_idx}, skipping")
            continue
        
        # Backward pass
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
        
        # Log per-step metrics to wandb for detailed monitoring
        if steps % 10 == 0:  # Log every 10 steps to avoid too much data
            wandb.log({
                "step": epoch * len(dataloader) + batch_idx,
                "train/step_loss": accumulated_loss.item(),
                "train/step_mse": mse.item() if not torch.isnan(mse) else 0.0,
                "train/step_cosine_similarity": cos_sim.item() if not torch.isnan(cos_sim) else 0.0,
                "train/step_angle_error": mean_angle_error.item() if not torch.isnan(mean_angle_error) else 0.0,
            })
        
        last_batch_dict = {
            'batch': [b.cpu() for b in batch_dict['batch']],
            'B': batch_dict['B'],
            'T': batch_dict['T'],
            'scenario_ids': batch_dict.get('scenario_ids', [])
        }
        
        # Clear CUDA cache periodically to prevent memory fragmentation
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    avg_loss_epoch = total_loss_epoch / max(1, steps)
    avg_cosine_sim = total_cosine_sim / max(1, metric_count)
    avg_mse = total_mse / max(1, metric_count)
    avg_angle_error = total_angular / max(1, metric_count)
    
    # Print epoch summary with RMSE in meters
    rmse_meters = (avg_mse ** 0.5) * 100.0
    print(f"\n[TRAIN EPOCH {epoch+1} SUMMARY]")
    print(f"  Loss: {avg_loss_epoch:.6f} | MSE: {avg_mse:.6f} | RMSE: {rmse_meters:.2f}m")
    print(f"  CosSim: {avg_cosine_sim:.4f} | AngleErr: {avg_angle_error:.4f} rad")
    
    # Call visualization AFTER epoch completes (model is still in train mode but no backward pending)
    if visualize_callback is not None and last_batch_dict is not None:
        visualize_callback(epoch, last_batch_dict, model, device, wandb)
    
    return avg_loss_epoch, avg_mse, avg_cosine_sim, avg_angle_error, last_batch_dict


def validate_single_epoch_batched(model, dataloader, loss_fn, 
                                   loss_alpha, loss_beta, loss_gamma, loss_delta, device):
    """Run batched validation loop."""
    model.eval()
    total_loss = 0.0
    steps = 0
    
    total_cosine_sim = 0.0
    total_mse = 0.0
    total_angular = 0.0
    metric_count = 0
    
    total_batches = len(dataloader)
    print(f"  Running batched validation on {total_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            if batch_idx % max(1, total_batches // 5) == 0:
                print(f"    Validation: {batch_idx}/{total_batches} ({100*batch_idx/total_batches:.1f}%)")
            
            batched_graph_sequence = batch_dict["batch"]
            B = batch_dict["B"]
            T = batch_dict["T"]

            for t in range(T):
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device, non_blocking=True)

            # Reset GRU for this validation batch
            total_nodes = batched_graph_sequence[0].num_nodes
            base_model = get_module(model)
            base_model.reset_gru_hidden_states(num_agents=total_nodes, device=device)
            
            accumulated_loss = 0.0
            
            for t, batched_graph in enumerate(batched_graph_sequence):
                if batched_graph.y is None or torch.all(batched_graph.y == 0):
                    continue
                
                edge_w = batched_graph.edge_attr if use_edge_weights else None
                out_predictions = model(
                    batched_graph.x, 
                    batched_graph.edge_index,
                    edge_weight=edge_w,
                    batch=batched_graph.batch,
                    batch_size=B,
                    batch_num=batch_idx, 
                    timestep=t
                )
                
                loss_t = loss_fn(out_predictions, batched_graph.y.to(out_predictions.dtype), 
                               batched_graph.x, alpha=loss_alpha, beta=loss_beta, 
                               gamma=loss_gamma, delta=loss_delta)
                accumulated_loss += loss_t
                
                mse = F.mse_loss(out_predictions, batched_graph.y.to(out_predictions.dtype))
                pred_norm = F.normalize(out_predictions, p=2, dim=1, eps=1e-6)
                target_norm = F.normalize(batched_graph.y.to(out_predictions.dtype), p=2, dim=1, eps=1e-6)
                cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                
                pred_angle = torch.atan2(out_predictions[:, 1], out_predictions[:, 0])
                target_angle = torch.atan2(batched_graph.y[:, 1], batched_graph.y[:, 0])
                angle_diff = torch.atan2(torch.sin(pred_angle - target_angle),
                                        torch.cos(pred_angle - target_angle))
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


def run_training_batched(dataset_path="./data/graphs/training/training_seqlen90.hdf5", 
                         validation_path="./data/graphs/validation/validation_seqlen90.hdf5",
                         batch_size=BATCHED_BATCH_SIZE,
                         wandb_run=None, return_viz_batch=False):
    """Run batched GCN training with batch_size > 1 for better GPU utilization."""
    
    print_gpu_info()
    
    print(f"\n{'='*60}")
    print(f"BATCHED GCN TRAINING MODE")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size} scenarios processed in parallel")
    print(f"Expected speedup: ~{batch_size}x over batch_size=1")
    print(f"{'='*60}\n")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    should_finish_wandb = False
    if wandb_run is None:
        wandb.login()
        wandb_run = wandb.init(
            project=project_name,
            config={
                "model": "SpatioTemporalGNN_Batched",
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
                "early_stopping_patience": early_stopping_patience,
                "batched_training": True
            },
            name=f"SpatioTemporalGNN_Batched_B{batch_size}_r{radius}_h{hidden_channels}{'_ew' if use_edge_weights else ''}",
            dir="../wandb"
        )
        should_finish_wandb = True
    
    # Define custom x-axes for wandb metrics
    wandb.define_metric("epoch")
    wandb.define_metric("step")
    
    # Epoch-based metrics (plotted vs epoch)
    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("train/mse", step_metric="epoch")
    wandb.define_metric("train/cosine_similarity", step_metric="epoch")
    wandb.define_metric("train/angle_error", step_metric="epoch")
    wandb.define_metric("val/loss", step_metric="epoch")
    wandb.define_metric("val/mse", step_metric="epoch")
    wandb.define_metric("val/cosine_similarity", step_metric="epoch")
    wandb.define_metric("val/angle_error", step_metric="epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")
    wandb.define_metric("train_val_gap", step_metric="epoch")
    
    # Step-based metrics (plotted vs step)
    wandb.define_metric("train/step_loss", step_metric="step")
    wandb.define_metric("train/step_mse", step_metric="step")
    wandb.define_metric("train/step_cosine_similarity", step_metric="step")
    wandb.define_metric("train/step_angle_error", step_metric="step")

    # Initialize batched model
    model = SpatioTemporalGNNBatched(
        input_dim=input_dim,
        hidden_dim=hidden_channels,
        output_dim=output_dim,
        num_gcn_layers=num_layers,
        num_gru_layers=num_gru_layers,
        dropout=dropout,
        use_gat=False,  # GCN model
        use_gradient_checkpointing=use_gradient_checkpointing,
        max_agents_per_scenario=128
    )
    
    if use_gradient_checkpointing:
        print("[Memory] Gradient checkpointing ENABLED")
    
    if use_torch_compile and hasattr(torch, 'compile'):
        try:
            print(f"[torch.compile] Compiling batched model with mode='{torch_compile_mode}'...")
            model = torch.compile(model, mode=torch_compile_mode)
            print("[torch.compile] Model compiled successfully!")
        except Exception as e:
            print(f"[torch.compile] Failed: {e}")
    
    model, is_parallel = setup_model_parallel(model, device)
    wandb.watch(model, log='all', log_freq=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, 
                                  patience=scheduler_patience, min_lr=min_lr)
    loss_fn = advanced_directional_loss

    # Load datasets
    try:
        dataset = HDF5ScenarioDataset(dataset_path, seq_len=sequence_length)
    except FileNotFoundError:
        print(f"ERROR: {dataset_path} not found!")
        print("Please run graph_creation_and_saving.py first to create HDF5 files.")
        if should_finish_wandb:
            wandb_run.finish()
        exit(1)
    
    # Validation dataset
    val_dataloader = None
    max_val_scenarios = max_validation_scenarios
    try:
        val_dataset_full = HDF5ScenarioDataset(
            validation_path, 
            seq_len=sequence_length,
            cache_in_memory=cache_validation_data,
            max_cache_size=max_val_scenarios
        )
        
        if len(val_dataset_full) > max_val_scenarios:
            val_indices = random.sample(range(len(val_dataset_full)), max_val_scenarios)
            val_dataset = Subset(val_dataset_full, val_indices)
            print(f"Validation: {max_val_scenarios} random scenarios (from {len(val_dataset_full)} total)")
        else:
            val_dataset = val_dataset_full
            print(f"Validation: {len(val_dataset)} scenarios")
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_graph_sequences_to_batch, 
            drop_last=False,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=worker_init_fn if num_workers > 0 else None
        )
    except FileNotFoundError:
        print(f"WARNING: {validation_path} not found! Training without validation.")

    # Training dataloader with batched scenarios
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,  # KEY: Process multiple scenarios in parallel!
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_graph_sequences_to_batch, 
        drop_last=True,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        worker_init_fn=worker_init_fn if num_workers > 0 else None
    )

    overfit_detector = OverfittingDetector(
        patience=4,  # Stop after 4 consecutive epochs of overfitting
        min_delta=early_stopping_min_delta, 
        verbose=True
    )
    
    print(f"Training on {device}")
    print(f"Dataset: {len(dataset)} scenarios")
    print(f"Batch size: {batch_size} scenarios/batch (GPU parallel)")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Learning rate: {learning_rate}")
    print(f"Edge weights: {'Enabled' if use_edge_weights else 'Disabled'}")
    print(f"Mixed Precision: {'BF16' if use_bf16 else 'FP16' if use_amp else 'Disabled'}")
    print(f"Checkpoints: {checkpoint_dir}\n")
    
    # AMP setup
    scaler = None
    amp_dtype = torch.float16
    if use_amp and torch.cuda.is_available():
        if use_bf16:
            amp_dtype = torch.bfloat16
            print("[AMP] Using BF16 (no gradient scaling needed)")
        else:
            scaler = torch.cuda.amp.GradScaler()
            print("[AMP] Using FP16 with GradScaler")
    
    best_val_loss = float('inf')
    last_viz_batch = None
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # Visualization callback (runs AFTER epoch, synchronously to avoid data corruption)
        def viz_callback(ep, batch_dict, mdl, dev, wb):
            nonlocal last_viz_batch
            if ep % visualize_every_n_epochs == 0:
                # Run visualization synchronously to avoid multiprocessing conflicts
                # Background threads were causing data loader corruption due to
                # concurrent TensorFlow dataset access during scenario loading
                try:
                    print(f"  [VIZ] Starting visualization for epoch {ep+1}...")
                    
                    # Ensure model is in eval mode for visualization
                    was_training = mdl.training
                    mdl.eval()
                    
                    with torch.no_grad():
                        filepath, avg_error = visualize_training_progress(
                            mdl, batch_dict, epoch=ep+1,
                            scenario_id=None,
                            save_dir=VIZ_DIR,
                            device=dev,
                            max_nodes_per_graph=config.max_nodes_per_graph_viz,
                            show_timesteps=config.show_timesteps_viz
                        )
                    
                    # Restore original training state
                    if was_training:
                        mdl.train()
                    
                    wb.log({"epoch": ep, "viz_avg_error": avg_error})
                    print(f"  [VIZ] Saved: {filepath} | Avg Error: {avg_error:.2f}m")
                except Exception as e:
                    print(f"  [VIZ] Error: {e}")
                    import traceback
                    traceback.print_exc()
            last_viz_batch = batch_dict
        
        # Training
        train_loss, train_mse, train_cos, train_angle, _ = train_single_epoch_batched(
            model, dataloader, optimizer, loss_fn,
            loss_alpha, loss_beta, loss_gamma, loss_delta,
            device, epoch, scaler=scaler, amp_dtype=amp_dtype,
            visualize_callback=viz_callback
        )
        
        print(f"\n[Train] Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | "
              f"Cos: {train_cos:.4f} | Angle: {train_angle:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/mse": train_mse,
            "train/cosine_similarity": train_cos,
            "train/angle_error": train_angle,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Validation
        if val_dataloader is not None:
            val_loss, val_mse, val_cos, val_angle = validate_single_epoch_batched(
                model, val_dataloader, loss_fn,
                loss_alpha, loss_beta, loss_gamma, loss_delta, device
            )
            
            print(f"[Valid] Loss: {val_loss:.4f} | MSE: {val_mse:.4f} | "
                  f"Cos: {val_cos:.4f} | Angle: {val_angle:.4f}")
            
            wandb.log({
                "val/loss": val_loss,
                "val/mse": val_mse,
                "val/cosine_similarity": val_cos,
                "val/angle_error": val_angle,
                "train_val_gap": train_loss - val_loss  # Track overfitting gap
            })
            
            scheduler.step(val_loss)
            
            # Check for overfitting with both train and val loss
            overfit_detector(train_loss, val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model = get_model_for_saving(model, is_parallel)
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model_batched.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'batch_size': batch_size
                }, checkpoint_path)
                print(f"   Best model saved (val_loss: {val_loss:.4f})")
        else:
            scheduler.step(train_loss)
            print("   No validation data - cannot detect overfitting")
        
        if overfit_detector.should_stop:
            print(f"\n{'='*60}")
            print(f"Training stopped at epoch {epoch+1} due to overfitting")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"{'='*60}")
            break
    
    if should_finish_wandb:
        wandb_run.finish()
    
    if return_viz_batch:
        return model, last_viz_batch
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batched GCN Training')
    parser.add_argument('--batch-size', type=int, default=BATCHED_BATCH_SIZE,
                        help=f'Batch size (default: {BATCHED_BATCH_SIZE})')
    parser.add_argument('--train-path', type=str, 
                        default='./data/graphs/training/training_seqlen90.hdf5',
                        help='Training data path')
    parser.add_argument('--val-path', type=str,
                        default='./data/graphs/validation/validation_seqlen90.hdf5',
                        help='Validation data path')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# BATCHED GCN TRAINING")
    print(f"# Batch size: {args.batch_size} scenarios in parallel")
    print(f"{'#'*60}\n")
    
    run_training_batched(
        dataset_path=args.train_path,
        validation_path=args.val_path,
        batch_size=args.batch_size
    )
