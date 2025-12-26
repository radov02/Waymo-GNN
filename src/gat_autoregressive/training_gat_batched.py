"""Batched GAT training for trajectory prediction. Supports batch_size > 1.

Usage: python src/gat_autoregressive/training_gat_batched.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp
import wandb
import torch.nn.functional as F
import warnings
import logging

# Suppress CUDA graph compilation warnings (non-critical, expected with dynamic shapes)
warnings.filterwarnings("ignore", message=".*skipping cudagraphs.*")
warnings.filterwarnings("ignore", message=".*_maybe_guard_rel.*")
warnings.filterwarnings("ignore", message=".*cudagraph.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

# Also suppress at logging level
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch.fx").setLevel(logging.ERROR)


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

from SpatioTemporalGAT_batched import SpatioTemporalGATBatched
from dataset import HDF5ScenarioDataset
from config import (device, gat_num_workers, num_layers, num_gru_layers, epochs, 
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    visualize_every_n_epochs, debug_mode,
                    gradient_clip_value, loss_alpha, loss_beta, loss_gamma, loss_delta,
                    scheduler_patience, scheduler_factor, min_lr, gat_prefetch_factor,
                    num_gpus, use_data_parallel, setup_model_parallel, get_model_for_saving,
                    print_gpu_info, pin_memory, gat_prefetch_factor, use_amp, use_bf16,
                    use_torch_compile, torch_compile_mode, use_gradient_checkpointing,
                    cache_validation_data, max_validation_scenarios, batch_size, gat_checkpoint_dir)
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from helper_functions.visualization_functions import visualize_epoch, visualize_training_progress
from helper_functions.helpers import advanced_directional_loss, compute_metrics
import config as cfg
from config import (batch_size, gat_viz_dir)

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
        # The model internally handles per-scenario hidden states
        base_model = get_module(model)
        base_model.gru_hidden = None  # Reset for new batch
        
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        valid_timesteps = 0  # Track valid timesteps for averaging
        
        for t, batched_graph in enumerate(batched_graph_sequence):
            if batched_graph.y is None or torch.all(batched_graph.y == 0):
                continue
            
            if torch.isnan(batched_graph.x).any() or torch.isnan(batched_graph.y).any():
                print(f"  Warning: NaN in input at batch {batch_idx}, t={t}, skipping...")
                continue
            
            with torch.amp.autocast('cuda', enabled=use_amp_local, dtype=amp_dtype):
                # Forward pass with batch info
                out_predictions = model(
                    batched_graph.x, 
                    batched_graph.edge_index,
                    batch=batched_graph.batch,  # Which scenario each node belongs to
                    batch_size=B,  # Number of scenarios
                    batch_num=batch_idx, 
                    timestep=t
                )
                
                if torch.isnan(out_predictions).any():
                    print(f"  Warning: NaN in predictions at batch {batch_idx}, t={t}, skipping...")
                    continue
                
                # SIMPLIFIED LOSS: Model predicts 2D velocity only
                # Convert velocity to displacement for loss computation
                dt = 0.1  # timestep in seconds
                from config import MAX_SPEED, POSITION_SCALE
                
                # Model outputs 2D velocity (vx_norm, vy_norm)
                pred_vx_norm = out_predictions[:, 0]
                pred_vy_norm = out_predictions[:, 1]
                
                # Convert velocity to displacement
                pred_vx = pred_vx_norm * MAX_SPEED
                pred_vy = pred_vy_norm * MAX_SPEED
                pred_dx_norm = (pred_vx * dt) / POSITION_SCALE
                pred_dy_norm = (pred_vy * dt) / POSITION_SCALE
                pred_disp_for_loss = torch.stack([pred_dx_norm, pred_dy_norm], dim=1)  # [N, 2]
                
                # Displacement loss
                loss_t = loss_fn(pred_disp_for_loss, batched_graph.y.to(out_predictions.dtype), 
                                batched_graph.x, alpha=loss_alpha, beta=loss_beta, 
                                gamma=loss_gamma, delta=loss_delta)
                
                if torch.isnan(loss_t):
                    print(f"  Warning: NaN loss at batch {batch_idx}, t={t}, skipping...")
                    continue
                    
            accumulated_loss += loss_t
            valid_timesteps += 1
            
            # Detailed loss debugging every 100 batches on first timestep
            if batch_idx % 100 == 0 and t == 0:
                with torch.no_grad():
                    from helper_functions.helpers import _compute_angle_loss, _compute_cosine_loss
                    angle_loss_val = _compute_angle_loss(pred_disp_for_loss, batched_graph.y)
                    mse_val = F.mse_loss(pred_disp_for_loss, batched_graph.y) * 100.0
                    cos_loss_val = _compute_cosine_loss(pred_disp_for_loss, batched_graph.y)
                    print(f"  [Loss Debug] angle={angle_loss_val:.4f}, mse={mse_val:.4f}, cos={cos_loss_val:.4f}, total={loss_t.item():.4f}")
            
            with torch.no_grad():
                mse = F.mse_loss(pred_disp_for_loss, batched_graph.y.to(out_predictions.dtype))
                pred_norm = F.normalize(pred_disp_for_loss, p=2, dim=1, eps=1e-6)
                target_norm = F.normalize(batched_graph.y.to(out_predictions.dtype), p=2, dim=1, eps=1e-6)
                cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                
                pred_angle = torch.atan2(pred_disp_for_loss[:, 1], pred_disp_for_loss[:, 0])
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
        
        # CRITICAL FIX: Average loss across timesteps instead of raw sum
        # Previously accumulated_loss was sum over ~89 timesteps, causing loss values ~18-19
        # Now we average to get per-timestep loss for proper learning signal
        if valid_timesteps > 0:
            accumulated_loss = accumulated_loss / valid_timesteps
        
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
    print(f"  AVERAGED LOSS (per timestep): {avg_loss_epoch:.6f}")
    print(f"  MSE: {avg_mse:.6f} | RMSE: {rmse_meters:.2f}m")
    print(f"  CosSim: {avg_cosine_sim:.4f} (target: >0.7) | AngleErr: {avg_angle_error:.4f} rad")
    if avg_cosine_sim < 0.5:
        print(f"  ⚠️  WARNING: Low cosine similarity indicates poor direction learning!")
    
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
            base_model = get_module(model)
            base_model.gru_hidden = None
            
            accumulated_loss = 0.0
            valid_timesteps = 0  # Track valid timesteps for averaging
            
            for t, batched_graph in enumerate(batched_graph_sequence):
                if batched_graph.y is None or torch.all(batched_graph.y == 0):
                    continue
                
                out_predictions = model(
                    batched_graph.x, 
                    batched_graph.edge_index,
                    batch=batched_graph.batch,
                    batch_size=B,
                    batch_num=batch_idx, 
                    timestep=t
                )
                
                # Multi-dimensional loss computation (same as training)
                dt = 0.1
                from config import MAX_SPEED, POSITION_SCALE
                
                pred_vx_norm = out_predictions[:, 0]
                pred_vy_norm = out_predictions[:, 1]
                pred_vx = pred_vx_norm * MAX_SPEED
                pred_vy = pred_vy_norm * MAX_SPEED
                pred_dx_norm = (pred_vx * dt) / POSITION_SCALE
                pred_dy_norm = (pred_vy * dt) / POSITION_SCALE
                pred_disp_for_loss = torch.stack([pred_dx_norm, pred_dy_norm], dim=1)
                
                gt_vx_norm = batched_graph.x[:, 0]
                gt_vy_norm = batched_graph.x[:, 1]
                gt_speed_norm = batched_graph.x[:, 2]
                gt_heading = batched_graph.x[:, 3]
                
                # Simplified displacement loss only
                loss_t = loss_fn(pred_disp_for_loss, batched_graph.y.to(out_predictions.dtype), 
                                batched_graph.x, alpha=loss_alpha, beta=loss_beta, 
                                gamma=loss_gamma, delta=loss_delta)
                accumulated_loss += loss_t
                valid_timesteps += 1
                
                mse = F.mse_loss(pred_disp_for_loss, batched_graph.y.to(out_predictions.dtype))
                pred_norm = F.normalize(pred_disp_for_loss, p=2, dim=1, eps=1e-6)
                target_norm = F.normalize(batched_graph.y.to(out_predictions.dtype), p=2, dim=1, eps=1e-6)
                cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                
                pred_angle = torch.atan2(pred_disp_for_loss[:, 1], pred_disp_for_loss[:, 0])
                target_angle = torch.atan2(batched_graph.y[:, 1], batched_graph.y[:, 0])
                angle_diff = torch.atan2(torch.sin(pred_angle - target_angle),
                                        torch.cos(pred_angle - target_angle))
                mean_angle_error = torch.abs(angle_diff).mean()
                
                total_mse += mse.item()
                total_cosine_sim += cos_sim.item()
                total_angular += mean_angle_error.item()
                metric_count += 1
            
            # Average loss across timesteps
            if valid_timesteps > 0:
                accumulated_loss = accumulated_loss / valid_timesteps
            
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
                         batch_size=batch_size,
                         wandb_run=None, return_viz_batch=False):
    """Run batched GAT training with batch_size > 1 for better GPU utilization."""
    
    print_gpu_info()
    
    print(f"\n{'='*60}")
    print(f"BATCHED TRAINING MODE")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size} scenarios processed in parallel")
    print(f"Expected speedup: ~{batch_size}x over batch_size=1")
    print(f"{'='*60}\n")
    
    os.makedirs(gat_checkpoint_dir, exist_ok=True)
    os.makedirs(gat_viz_dir, exist_ok=True)
    
    should_finish_wandb = False
    if wandb_run is None:
        wandb.login()
        wandb_run = wandb.init(
            project=project_name,
            config={
                "model": "SpatioTemporalGAT_Batched",
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dataset": dataset_name,
                "dropout": dropout,
                "hidden_channels": hidden_channels,
                "num_gat_layers": num_layers,
                "num_gru_layers": num_gru_layers,
                "use_gat": True,
                "num_heads": 4,
                "epochs": epochs,
                "loss_alpha": loss_alpha,
                "loss_beta": loss_beta,
                "loss_gamma": loss_gamma,
                "loss_delta": loss_delta,
                "scheduler": "ReduceLROnPlateau",
                "batched_training": True
            },
            name=f"SpatioTemporalGAT_Batched_B{batch_size}_r{radius}_h{hidden_channels}",
            dir="../wandb"
        )
        should_finish_wandb = True    
    # Define custom x-axes for wandb metrics
    wandb.define_metric("epoch")
    
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
    # Initialize batched model
    num_attention_heads = 4  # Number of attention heads for GAT
    model = SpatioTemporalGATBatched(
        input_dim=input_dim,
        hidden_dim=hidden_channels,
        output_dim=output_dim,
        num_gat_layers=num_layers,
        num_gru_layers=num_gru_layers,
        dropout=dropout,
        num_heads=num_attention_heads,
        use_gradient_checkpointing=use_gradient_checkpointing,
        max_agents_per_scenario=128  # Adjust based on your data
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
            batch_size=batch_size,  # Use same batch size for validation
            shuffle=False, 
            num_workers=gat_num_workers,
            collate_fn=collate_graph_sequences_to_batch, 
            drop_last=False,  # Don't drop last batch in validation
            pin_memory=pin_memory,
            prefetch_factor=gat_prefetch_factor,
            persistent_workers=True if gat_num_workers > 0 else False,
            worker_init_fn=worker_init_fn if gat_num_workers > 0 else None
        )
    except FileNotFoundError:
        print(f"WARNING: {validation_path} not found! Training without validation.")

    # Training dataloader with batched scenarios
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,  # KEY: Process multiple scenarios in parallel!
        shuffle=True, 
        num_workers=gat_num_workers,
        collate_fn=collate_graph_sequences_to_batch, 
        drop_last=True,
        pin_memory=pin_memory,
        prefetch_factor=gat_prefetch_factor,
        persistent_workers=True if gat_num_workers > 0 else False,
        worker_init_fn=worker_init_fn if gat_num_workers > 0 else None
    )
    
    print(f"Training on {device}")
    print(f"Dataset: {len(dataset)} scenarios")
    print(f"Batch size: {batch_size} scenarios/batch (GPU parallel)")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Sequence length: {sequence_length}")
    print(f"Learning rate: {learning_rate}")
    print(f"Mixed Precision: {'BF16' if use_bf16 else 'FP16' if use_amp else 'Disabled'}")
    print(f"Checkpoints: {gat_checkpoint_dir}\n")
    
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
    best_model_state = None
    best_optimizer_state = None
    best_epoch = 0
    checkpoint_filename = f'best_gat_batched_B{batch_size}_h{hidden_channels}_lr{learning_rate:.0e}_heads{num_attention_heads}_E{epochs}.pt'
    checkpoint_path = os.path.join(gat_checkpoint_dir, checkpoint_filename)
    last_viz_batch = None
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # Visualization callback
        def viz_callback(ep, batch_dict, mdl, dev, wb):
            nonlocal last_viz_batch
            if ep % visualize_every_n_epochs == 0:
                try:
                    # Use visualize_training_progress directly with gat_viz_dir
                    filepath, avg_error = visualize_training_progress(
                        mdl, batch_dict, epoch=ep+1,
                        scenario_id=None,
                        save_dir=gat_viz_dir,  # Use GAT-specific directory
                        device=dev,
                        max_nodes_per_graph=cfg.max_nodes_per_graph_viz,
                        show_timesteps=cfg.show_timesteps_viz
                    )
                    wb.log({"epoch": ep, "viz_avg_error": avg_error})
                except Exception as e:
                    print(f"  Visualization error: {e}")
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
        
        # Prepare logging dict with epoch and training metrics
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/mse": train_mse,
            "train/cosine_similarity": train_cos,
            "train/angle_error": train_angle,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        # Validation
        if val_dataloader is not None:
            val_loss, val_mse, val_cos, val_angle = validate_single_epoch_batched(
                model, val_dataloader, loss_fn,
                loss_alpha, loss_beta, loss_gamma, loss_delta, device
            )
            
            print(f"[Valid] Loss: {val_loss:.4f} | MSE: {val_mse:.4f} | "
                  f"Cos: {val_cos:.4f} | Angle: {val_angle:.4f}")
            
            # Add validation metrics to same log_dict
            log_dict.update({
                "val/loss": val_loss,
                "val/mse": val_mse,
                "val/cosine_similarity": val_cos,
                "val/angle_error": val_angle,
                "train_val_gap": train_loss - val_loss
            })
        
        # Single wandb.log call with all metrics for this epoch
        wandb.log(log_dict)
        
        # Continue with scheduler and checkpoint saving
        if val_dataloader is not None:
            scheduler.step(val_loss)
        
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                save_model = get_model_for_saving(model, is_parallel)
                # Store best model state in memory (will save to disk at end)
                best_model_state = save_model.state_dict().copy()
                best_optimizer_state = optimizer.state_dict().copy()
                print(f"   New best validation loss: {val_loss:.4f} at epoch {epoch+1}")
        else:
            scheduler.step(train_loss)
    
    # Save best model checkpoint after training completes
    if best_model_state is not None:
        print(f"\n{'='*60}")
        print(f"Saving best model from epoch {best_epoch}...")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'val_loss': best_val_loss,
            'batch_size': batch_size,
            'config': {
                'batch_size': batch_size,
                'hidden_channels': hidden_channels,
                'learning_rate': learning_rate,
                'num_attention_heads': num_attention_heads,
                'num_layers': num_layers,
                'num_gru_layers': num_gru_layers,
                'dropout': dropout,
                'use_gat': True
            }
        }, checkpoint_path)
        print(f"Best GAT model saved to {checkpoint_filename}")
        print(f"Epoch: {best_epoch} | Val Loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
    else:
        print("\nNo best model checkpoint saved (no validation data or no improvement)\n")
    
    # Save final model (current state at last epoch)
    print(f"\n{'='*60}")
    print(f"Saving final model from epoch {epoch+1}...")
    save_model = get_model_for_saving(model, is_parallel)
    final_checkpoint_filename = f'final_gat_batched_B{batch_size}_h{hidden_channels}_lr{learning_rate:.0e}_heads{num_attention_heads}_E{epochs}.pt'
    final_checkpoint_path = os.path.join(gat_checkpoint_dir, final_checkpoint_filename)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': save_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss if val_dataloader is not None else None,
        'batch_size': batch_size,
        'config': {
            'batch_size': batch_size,
            'hidden_channels': hidden_channels,
            'learning_rate': learning_rate,
            'num_attention_heads': num_attention_heads,
            'num_layers': num_layers,
            'num_gru_layers': num_gru_layers,
            'dropout': dropout,
            'use_gat': True
        }
    }, final_checkpoint_path)
    print(f"Final GAT model saved to {final_checkpoint_filename}")
    print(f"Epoch: {epoch+1}")
    print(f"{'='*60}\n")
    
    if should_finish_wandb:
        wandb_run.finish()
    
    if return_viz_batch:
        return model, last_viz_batch
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batched GAT Training')
    parser.add_argument('--batch-size', type=int, default=batch_size,
                        help=f'Batch size (default: {batch_size})')
    parser.add_argument('--train-path', type=str, 
                        default='./data/graphs/training/training_seqlen90.hdf5',
                        help='Training data path')
    parser.add_argument('--val-path', type=str,
                        default='./data/graphs/validation/validation_seqlen90.hdf5',
                        help='Validation data path')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# BATCHED GAT TRAINING")
    print(f"# Batch size: {args.batch_size} scenarios in parallel")
    print(f"{'#'*60}\n")
    
    run_training_batched(
        dataset_path=args.train_path,
        validation_path=args.val_path,
        batch_size=args.batch_size
    )
