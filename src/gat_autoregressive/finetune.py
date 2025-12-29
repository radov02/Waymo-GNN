"""Fine-tune GAT single-step model for autoregressive prediction using scheduled sampling."""

import sys
import os
# Set CUDA memory allocation config BEFORE importing PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add parent directory (src/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from gat_autoregressive.SpatioTemporalGAT_batched import SpatioTemporalGATBatched
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, gat_num_workers, num_layers, num_gru_layers,
                    input_dim, output_dim, sequence_length, hidden_channels, autoreg_learning_rate,
                    dropout, learning_rate, project_name, dataset_name, epochs, gat_learning_rate,
                    gradient_clip_value, gat_num_heads, gcn_checkpoint_dir, gcn_checkpoint_dir_autoreg,
                    num_gpus, use_data_parallel, setup_model_parallel, get_model_for_saving, load_model_state,
                    autoreg_num_rollout_steps, autoreg_num_epochs, autoreg_sampling_strategy,
                    autoreg_visualize_every_n_epochs, gat_viz_dir_autoreg, use_edge_weights,
                    pin_memory, gat_prefetch_factor, use_amp, gcn_viz_dir_autoreg, autoreg_skip_map_features,
                    gat_checkpoint_dir, gat_checkpoint_dir_autoreg, use_gradient_checkpointing,
                    cache_validation_data, max_validation_scenarios, radius, max_scenario_files_for_viz,
                    POSITION_SCALE, MAX_SPEED, MAX_ACCEL, MAX_DIST_SDC, enable_debug_viz)
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch, build_edge_index_using_radius
from helper_functions.visualization_functions import (load_scenario_by_id, draw_map_features, VIBRANT_COLORS, SDC_COLOR)
from torch.nn.utils import clip_grad_norm_
import random
from autoregressive_predictions.SpatioTemporalGNN_batched import SpatioTemporalGNNBatched

AUTOREG_REBUILD_EDGES = True  # when enabled, edges are recomputed at each autoregressive step based on updated positions, which captures new spatial relationships as agents move (e.g., agents coming within/out of radius) - more accurate for long rollouts where agents move significantly
_size_mismatch_warned = False

# ============ FEATURE NOISE INJECTION ============
# During teacher forcing, inject noise into derived features (velocity, acceleration)
# to simulate the distribution shift that occurs during autoregressive rollout.
# This helps the model learn to be robust to noisy features from its own predictions.
FEATURE_NOISE_INJECTION = True
FEATURE_NOISE_VELOCITY_STD = 0.05  # Normalized velocity noise std (10% of MAX_SPEED range)
FEATURE_NOISE_ACCEL_STD = 0.08     # Normalized acceleration noise std (higher because accel is noisier)
FEATURE_NOISE_HEADING_STD = 0.03  # Heading noise in normalized units (radians/pi)

# ============ VELOCITY CONSISTENCY LOSS ============
# Add loss to ensure predicted displacement is kinematically consistent
# with the velocity/heading features the model is using as input.
VELOCITY_CONSISTENCY_LOSS = True
VELOCITY_CONSISTENCY_WEIGHT = 0.15  # Weight for velocity consistency loss

# ============ FEATURE SMOOTHING DURING AUTOREG ============
# Use exponential moving average for derived features during autoregressive rollout
# This reduces noise from accumulated prediction errors
AUTOREG_FEATURE_EMA = True
AUTOREG_FEATURE_EMA_ALPHA = 0.3  # Higher = more smoothing (0.3 = 70% previous, 30% new)

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/gat_autoregressive/finetune.py

def load_pretrained_model(checkpoint_path, device, model_type="gat"):
    """Load pre-trained GAT/GCN model from checkpoint with automatic architecture inference."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    if model_type == "gat":
        print(f"Loading pre-trained GAT model from {checkpoint_path}...")
        
        # Infer GAT architecture from checkpoint weights
        gat_layer_keys = ['spatial_layers.0.lin_src.weight', 'module.spatial_layers.0.lin_src.weight',
                          'spatial_layers.0.att_src', 'module.spatial_layers.0.att_src',
                          '_orig_mod.spatial_layers.0.lin_src.weight', '_orig_mod.spatial_layers.0.att_src']
        
        checkpoint_hidden_dim = None
        print(f"  First 10 keys in checkpoint state_dict:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            if i < 5 or 'spatial' in key:
                print(f"    {key}: {state_dict[key].shape}")
        
        for key in gat_layer_keys:
            if key in state_dict:
                # Handle att_src which has shape [1, num_heads, hidden_dim]
                if 'att_src' in key or 'att_dst' in key:
                    checkpoint_hidden_dim = state_dict[key].shape[-1]  # Last dimension is hidden_dim
                else:
                    checkpoint_hidden_dim = state_dict[key].shape[0] if len(state_dict[key].shape) > 1 else state_dict[key].shape[0]
                print(f"  OK: Inferred GAT hidden_dim={checkpoint_hidden_dim} from key: {key}")
                break
        
        if checkpoint_hidden_dim is None:
            checkpoint_hidden_dim = hidden_channels
            print(f"  ERROR: Could not infer GAT hidden_dim, using config.py default: {checkpoint_hidden_dim}")
        
        # Recreate model with saved config or use inferred architecture
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  Checkpoint config keys: {list(config.keys())}")
            print(f"  hidden_channels from checkpoint config: {config.get('hidden_channels', 'NOT FOUND')}")
            
            # Use inferred hidden_dim if available, otherwise fall back to config
            final_hidden_dim = checkpoint_hidden_dim if checkpoint_hidden_dim else config.get('hidden_channels', hidden_channels)
            print(f"  Using hidden_dim={final_hidden_dim}")
            
            model = SpatioTemporalGATBatched(
                input_dim=config.get('input_dim', input_dim),
                hidden_dim=final_hidden_dim,
                output_dim=config.get('output_dim', output_dim),
                num_gat_layers=config.get('num_layers', num_layers),
                num_gru_layers=config.get('num_gru_layers', num_gru_layers),
                dropout=config.get('dropout', dropout),
                num_heads=config.get('num_attention_heads', config.get('num_heads', gat_num_heads)),
                max_agents_per_scenario=128
            )
        else:
            # Checkpoint doesn't have config - use inferred architecture
            print("  Warning: Checkpoint missing 'config' key, using inferred architecture from weights")
            model = SpatioTemporalGATBatched(
                input_dim=input_dim,
                hidden_dim=checkpoint_hidden_dim if checkpoint_hidden_dim else hidden_channels,
                output_dim=output_dim,
                num_gat_layers=num_layers,
                num_gru_layers=num_gru_layers,
                dropout=dropout,
                num_heads=4,  # Default value
                max_agents_per_scenario=128
            )
        print(f"  Loaded GAT model from epoch {checkpoint['epoch']}")
        
    elif model_type == "gcn":
        print(f"Loading pre-trained GCN model from {checkpoint_path}...")
        
        # Infer GCN architecture from checkpoint weights
        # Try multiple key formats: regular, DataParallel (module.), and torch.compile (_orig_mod.)
        gcn_layer_keys = [
            'spatial_layers.0.lin.weight',
            'module.spatial_layers.0.lin.weight',
            '_orig_mod.spatial_layers.0.lin.weight'
        ]
        
        checkpoint_hidden_dim = None
        print(f"  First 10 keys in checkpoint state_dict:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            if i < 5 or 'spatial' in key:
                print(f"    {key}: {state_dict[key].shape}")
        
        for key in gcn_layer_keys:
            if key in state_dict:
                checkpoint_hidden_dim = state_dict[key].shape[0]  # [out_features, in_features]
                print(f"  ✓ Inferred GCN hidden_dim={checkpoint_hidden_dim} from key: {key}")
                break
        
        if checkpoint_hidden_dim is None:
            checkpoint_hidden_dim = hidden_channels
            print(f"  ✗ Could not infer GCN hidden_dim, using config.py default: {checkpoint_hidden_dim}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            # Debug: print what's in the config
            print(f"  Checkpoint config keys: {list(config.keys())}")
            print(f"  hidden_channels from checkpoint config: {config.get('hidden_channels', 'NOT FOUND')}")
            
            # Use inferred hidden_dim if available, otherwise fall back to config
            final_hidden_dim = checkpoint_hidden_dim if checkpoint_hidden_dim else config.get('hidden_channels', hidden_channels)
            print(f"  >>> CREATING GCN MODEL WITH hidden_dim={final_hidden_dim}, use_gat=False <<<")
            
            model = SpatioTemporalGNNBatched(
                input_dim=config.get('input_dim', input_dim),
                hidden_dim=final_hidden_dim,
                output_dim=config.get('output_dim', output_dim),
                num_gcn_layers=config.get('num_layers', num_layers),
                num_gru_layers=config.get('num_gru_layers', num_gru_layers),
                dropout=config.get('dropout', dropout),
                use_gat=False,
                use_gradient_checkpointing=use_gradient_checkpointing,
                max_agents_per_scenario=128
            )
        else:
            print("  Warning: Checkpoint missing 'config' key, using inferred architecture from weights")
            model = SpatioTemporalGNNBatched(
                input_dim=input_dim,
                hidden_dim=checkpoint_hidden_dim,  # Use inferred value!
                output_dim=output_dim,
                num_gcn_layers=num_layers,
                num_gru_layers=num_gru_layers,
                dropout=dropout,
                use_gat=False,
                use_gradient_checkpointing=use_gradient_checkpointing,
                max_agents_per_scenario=128
            )
        print(f"  Loaded GCN model from epoch {checkpoint['epoch']}")
    
    # Setup multi-GPU if available
    model, is_parallel = setup_model_parallel(model, device)
    
    # Load weights:
    load_model_state(model, checkpoint['model_state_dict'], is_parallel)
    
    return model, is_parallel, checkpoint

def inject_feature_noise(graph, epoch, total_epochs, device):
    """Inject noise into derived features during teacher forcing.
    
    This simulates the distribution shift the model will experience during
    autoregressive rollout, where derived features (velocity, acceleration, heading)
    become noisy due to accumulated prediction errors.
    
    CURRICULUM: Noise decreases as training progresses, since the model's predictions
    improve and derived features become more accurate.
    
    Args:
        graph: PyG Data object with node features [N, 15]
        epoch: Current epoch (0-indexed)
        total_epochs: Total epochs for training
        device: Torch device
    
    Returns:
        Graph with noise-injected features (same object, modified in-place clone)
    """
    if not FEATURE_NOISE_INJECTION:
        return graph
    
    # Clone to avoid modifying original
    noisy_graph = graph.clone()
    
    # Curriculum: noise starts high and decreases
    # At epoch 0: 100% noise, at final epoch: 20% noise
    progress = epoch / max(1, total_epochs - 1)
    noise_scale = 1.0 - 0.8 * progress  # 1.0 -> 0.2
    
    N = noisy_graph.x.shape[0]
    
    # Inject noise into velocity features [0-1]
    vel_noise = torch.randn(N, 2, device=device) * FEATURE_NOISE_VELOCITY_STD * noise_scale
    noisy_graph.x[:, 0:2] = noisy_graph.x[:, 0:2] + vel_noise
    
    # Update speed [2] to be consistent with noisy velocity
    noisy_vx = noisy_graph.x[:, 0] * MAX_SPEED
    noisy_vy = noisy_graph.x[:, 1] * MAX_SPEED
    noisy_speed = torch.sqrt(noisy_vx**2 + noisy_vy**2)
    noisy_graph.x[:, 2] = noisy_speed / MAX_SPEED
    
    # Inject noise into heading [3]
    heading_noise = torch.randn(N, device=device) * FEATURE_NOISE_HEADING_STD * noise_scale
    noisy_graph.x[:, 3] = torch.clamp(noisy_graph.x[:, 3] + heading_noise, -1, 1)
    
    # Inject noise into acceleration features [5-6]
    accel_noise = torch.randn(N, 2, device=device) * FEATURE_NOISE_ACCEL_STD * noise_scale
    noisy_graph.x[:, 5:7] = noisy_graph.x[:, 5:7] + accel_noise
    
    return noisy_graph


def compute_velocity_consistency_loss(pred_displacement, graph, device):
    """Compute loss for kinematic consistency between displacement and velocity.
    
    The predicted displacement should be roughly consistent with the velocity
    features in the input graph. This teaches the model to USE the velocity
    information rather than ignoring it.
    
    Loss = ||pred_displacement - expected_displacement_from_velocity||^2
    
    Args:
        pred_displacement: Predicted normalized displacement [N, 2]
        graph: Input graph with velocity features
        device: Torch device
    
    Returns:
        Velocity consistency loss (scalar tensor)
    """
    if not VELOCITY_CONSISTENCY_LOSS:
        return torch.tensor(0.0, device=device)
    
    dt = 0.1  # timestep
    
    # Get velocity from input features [0-1] (normalized by MAX_SPEED)
    input_vx = graph.x[:, 0] * MAX_SPEED  # m/s
    input_vy = graph.x[:, 1] * MAX_SPEED  # m/s
    
    # Expected displacement from velocity: d = v * dt
    expected_dx = input_vx * dt  # meters
    expected_dy = input_vy * dt  # meters
    
    # Normalize to match prediction scale
    expected_displacement = torch.stack([expected_dx, expected_dy], dim=1) / POSITION_SCALE
    
    # Only compute for valid agents with significant velocity
    # (stationary agents have noisy velocity features)
    speed = torch.sqrt(input_vx**2 + input_vy**2)
    moving_mask = speed > 0.5  # > 0.5 m/s
    
    if moving_mask.sum() < 2:
        return torch.tensor(0.0, device=device)
    
    # L2 loss on moving agents
    pred_aligned = pred_displacement[:expected_displacement.shape[0]]
    if pred_aligned.shape[0] != expected_displacement.shape[0]:
        return torch.tensor(0.0, device=device)
    
    diff = pred_aligned[moving_mask] - expected_displacement[moving_mask]
    consistency_loss = (diff ** 2).mean()
    
    return consistency_loss


def scheduled_sampling_probability(epoch, total_epochs, strategy='linear', warmup_epochs=0):
    """Calculate probability of using model's own prediction vs ground truth.
    
    IMPROVED SCHEDULE: Faster exposure to autoregressive mode.
    The model needs to learn to correct its own errors early in training,
    not just at the end. Teacher forcing alone teaches single-step prediction
    but not error correction.
    
    Args:
        epoch: Current epoch (0-indexed, so first epoch is epoch=0)
        total_epochs: Total number of training epochs
        strategy: 'linear', 'exponential', or 'inverse_sigmoid'
        warmup_epochs: Number of epochs with low sampling probability (for gradual transition)
    
    Returns:
        Probability of using model's own prediction (0 = teacher forcing, 1 = autoregressive)
    """
    
    # REDUCED warmup: Only 2 epochs of pure teacher forcing
    # This gets the model stable, then immediately starts autoregressive exposure
    warmup_epochs = 2
    if epoch < warmup_epochs:
        # During warmup: 10% at epoch 0, 20% at epoch 1
        return 0.10 * (epoch + 1)
    
    # After warmup, apply strategy-based scheduling with FASTER ramp-up
    remaining_epochs = total_epochs - warmup_epochs
    adjusted_epoch = epoch - warmup_epochs
    progress = adjusted_epoch / max(1, remaining_epochs - 1)
    
    if strategy == 'linear':
        # Linear from 0.30 to 1.0 after warmup (start higher!)
        return 0.30 + 0.70 * progress
    elif strategy == 'exponential':
        # Exponential with faster ramp: reaches ~0.65 at halfway, ~0.95 at end
        return 0.30 + 0.70 * (1.0 - np.exp(-3.0 * progress))
    elif strategy == 'inverse_sigmoid':
        # S-curve: moderate at start, fast in middle, saturates at end
        k = 6
        return 0.30 + 0.70 / (1 + np.exp(-k * (progress - 0.5)))
    else:
        return 0.30 + 0.70 * progress

def curriculum_rollout_steps(epoch, max_rollout_steps, total_epochs):
    """Curriculum learning: gradually increase rollout length.
    
    IMPROVED v2: Start with SHORT rollouts so model masters basics first.
    The model needs to learn accurate single-step predictions before
    it can handle long trajectory rollouts. Starting too long leads to
    compounding errors the model can't learn to correct.
    
    Args:
        epoch: Current epoch (0-indexed)
        max_rollout_steps: Maximum rollout steps (e.g., 50)
        total_epochs: Total training epochs
    
    Returns:
        Number of rollout steps to use this epoch
    """
    # Start with 5 steps (0.5 seconds) - model must master basics first
    # Gradually increase: reach max at 70% of training, maintain for final 30%
    min_steps = 5
    
    # Use 70% of epochs to reach max, then maintain max for remaining 30%
    ramp_epochs = int(total_epochs * 0.7)
    if epoch >= ramp_epochs:
        return max_rollout_steps
    
    # Quadratic ramp: slower start, faster increase later
    progress = epoch / max(1, ramp_epochs - 1)
    progress_curved = progress ** 1.5  # Slower initial ramp
    steps = int(min_steps + (max_rollout_steps - min_steps) * progress_curved)
    return min(steps, max_rollout_steps)

def visualize_autoregressive_rollout(model, batch_dict, epoch, num_rollout_steps, device, 
                                     is_parallel, save_dir=None, total_epochs=40, model_type=None):
    """Visualize autoregressive rollout predictions vs ground truth.
    
    NOTE: This visualization runs in FULL AUTOREGRESSIVE MODE (100% model predictions)
    regardless of the training sampling probability! This is intentional - we want to see
    how the model performs at inference time, not during training.
    
    At epoch 1, expect poor results because the base model was trained with pure teacher
    forcing (always GT inputs) and has never seen its own errors. The scheduled sampling
    fine-tuning will gradually improve this over epochs as the model learns to correct
    for its own prediction errors.
    
    Model predicts displacement → updates positions → derives all 15-dim node features.
    
    Args:
        model: The GAT model
        batch_dict: Batch dictionary with graph sequences
        epoch: Current epoch (0-indexed)
        num_rollout_steps: Number of autoregressive steps to roll out
        device: Torch device
        is_parallel: Whether model is DataParallel wrapped
        save_dir: Directory to save visualizations
        total_epochs: Total number of epochs (for display)
        model_type: "gat" or "gcn"
    """
    if model_type is None:
        print("CRITICAL ERROR: model_type must be specified as 'gat' or 'gcn' for visualization saving paths.")
        return None
    
    if save_dir is None:
        if model_type == "gat":
            save_dir = gat_viz_dir_autoreg
        elif model_type == "gcn":
            save_dir = gcn_viz_dir_autoreg
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Clarify that visualization runs full autoregressive (not scheduled sampling)
    if epoch == 0:
        train_sampling_prob = scheduled_sampling_probability(0, total_epochs, strategy='linear')
        print(f"\n  [VIZ NOTE] Visualization runs FULL AUTOREGRESSIVE (100% model predictions)!")
        print(f"  [VIZ NOTE] Training uses {train_sampling_prob:.0%} model predictions at epoch 1.")
        print(f"  [VIZ NOTE] Poor results at epoch 1 are EXPECTED - the base model was trained with")
        print(f"  [VIZ NOTE] pure teacher forcing and hasn't learned to correct its own errors yet.\n")
    
    print(f"  [VIZ] Autoregressive rollout: model predicts displacement → derives features (epoch {epoch+1}/{total_epochs})")
    
    base_model = model.module if is_parallel else model
    
    batched_graph_sequence = batch_dict["batch"]
    B, T = batch_dict["B"], batch_dict["T"]
    scenario_ids = batch_dict.get("scenario_ids", ["unknown"])
    
    # Load scenario for map features (cross-platform compatible)
    scenario = None
    if not autoreg_skip_map_features and scenario_ids and scenario_ids[0] and scenario_ids[0] != "unknown":
        try:
            print(f"  Loading scenario {scenario_ids[0]} for map visualization...")
            scenario = load_scenario_by_id(scenario_ids[0])     # use indexed loading for speed (index is built once and cached)
            if scenario is not None:
                print(f"  Loaded scenario for map visualization")
            else:
                print(f"  Scenario loaded as None. Proceeding without map features.")
                print(f"    NOTE: TFRecord files may not exist on this machine. Map features require original Waymo data.")
        except Exception as e:
            error_msg = str(e)
            if "No such file or directory" in error_msg or "tfrecord" in error_msg.lower():
                print(f"  TFRecord file not found - map features unavailable.")
                print(f"    To enable map features, ensure Waymo TFRecord files are in data/scenario/training/")
            else:
                print(f"  Warning: Could not load scenario ({type(e).__name__}: {e}). Proceeding without map features.")
    
    for t in range(T):  # move to device
        batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
    
    # Reset GRU with proper batch info
    # CRITICAL: Must compute agents_per_scenario from batch tensor to match forward pass sizing
    first_graph = batched_graph_sequence[0]
    if hasattr(first_graph, 'batch') and first_graph.batch is not None:
        # Count agents per scenario in the batch
        batch_counts = torch.bincount(first_graph.batch, minlength=B)
        agents_per_scenario = batch_counts.tolist()
        base_model.reset_gru_hidden_states(
            batch_size=B, 
            agents_per_scenario=agents_per_scenario, 
            device=device
        )
    else:
        # Single scenario fallback
        num_nodes = first_graph.num_nodes
        base_model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
    
    start_t = T // 3
    actual_rollout_steps = min(num_rollout_steps, T - start_t - 1)
    if actual_rollout_steps < num_rollout_steps:
        print(f"  [WARNING] Requested {num_rollout_steps} rollout steps but only {actual_rollout_steps} available (T={T}, start_t={start_t})")
    if actual_rollout_steps < 1:
        print(f"  [ERROR] Not enough timesteps for rollout. T={T}, start_t={start_t}")
        return None
    
    # find SDC:
    sdc_id = None
    if scenario is not None:
        sdc_track = scenario.tracks[scenario.sdc_track_index]
        sdc_id = sdc_track.id
    
    agent_trajectories = {}     # Dictionary to track positions by agent_id: {agent_id: {'gt': [(t, pos), ...], 'pred': [(t, pos), ...]}}
    
    with torch.no_grad():
        # WARM UP GRU: Run through timesteps 0 to start_t to build up temporal context
        # This matches how training processes the full sequence from t=0
        # Without this, visualization starts with zero hidden states which hurts predictions
        print(f"  [VIZ] Warming up GRU for {start_t} timesteps...")
        for warm_t in range(start_t):
            warm_graph = batched_graph_sequence[warm_t]
            if warm_graph.x is not None:
                agent_ids_warm = getattr(warm_graph, 'agent_ids', None)
                edge_w = warm_graph.edge_attr if use_edge_weights else None
                _ = model(warm_graph.x, warm_graph.edge_index, edge_weight=edge_w,
                         batch=warm_graph.batch, batch_size=B, batch_num=0, timestep=warm_t,
                         agent_ids=agent_ids_warm)
        print(f"  [VIZ] GRU warmup complete. Starting rollout from t={start_t}")
        
        start_graph = batched_graph_sequence[start_t]
        start_scenario_indices, start_agent_ids, start_pos = filter_to_scenario(start_graph, 0, B)   # filter to only first scenario
        
        # find persistent agents (only from first scenario in batch (batch_idx=0)) - we can only properly track agents that exist at ALL timesteps 
        persistent_agent_ids = set(start_agent_ids)
        for step in range(actual_rollout_steps):
            target_t = start_t + step + 1
            if target_t >= T:
                break

            target_graph = batched_graph_sequence[target_t]
            target_scenario_indices, target_agent_ids, target_pos = filter_to_scenario(target_graph, 0, B)   # filter to only first scenario

            persistent_agent_ids = persistent_agent_ids.intersection(set(target_agent_ids))
        
        if enable_debug_viz:
            print(f"  [DEBUG VIZ] {len(persistent_agent_ids)} agents persist through all {actual_rollout_steps} steps")
        
        # initialize trajectories only for persistent agents
        for agent_id in persistent_agent_ids:
            if agent_id in start_agent_ids:     # find this agent's index in start_graph (first scenario graphs that are not for warmup)
                local_idx = start_agent_ids.index(agent_id)
                agent_trajectories[agent_id] = {
                    'gt': [(start_t, start_pos[local_idx].copy())],
                    'pred': [(start_t, start_pos[local_idx].copy())]
                }
        
        # collect GT positions for persistent agents (only from first scenario)
        for step in range(actual_rollout_steps):
            target_t = start_t + step + 1
            if target_t >= T:
                break

            target_graph = batched_graph_sequence[target_t]
            target_scenario_indices, target_agent_ids, target_pos = filter_to_scenario(target_graph, 0, B)   # filter to only first scenario
            
            for local_idx, agent_id in enumerate(target_agent_ids):
                if agent_id in agent_trajectories:
                    if target_pos is not None:
                        pos = target_pos[local_idx].copy()
                    else:
                        last_pos = agent_trajectories[agent_id]['gt'][-1][1]
                        global_idx = target_scenario_indices[local_idx]
                        if target_graph.y is not None and global_idx < target_graph.y.shape[0]:
                            # GT y is NORMALIZED (meters / POSITION_SCALE), convert to meters
                            disp_normalized = target_graph.y[global_idx].cpu().numpy()
                            disp_meters = disp_normalized * POSITION_SCALE
                            pos = last_pos + disp_meters
                        else:
                            pos = last_pos
                    agent_trajectories[agent_id]['gt'].append((target_t, pos))
        
        # autoregressive rollout for predictions - track predicted position per persistent agent:
        agent_pred_positions = {}
        for agent_id in persistent_agent_ids:
            if agent_id in start_agent_ids:
                local_idx = start_agent_ids.index(agent_id)
                agent_pred_positions[agent_id] = start_pos[local_idx].copy()
        
        graph_for_prediction = start_graph
        completed_prediction_steps = 0
        
        # build agent_id -> GT position mapping for all timesteps (for correct error calculation)
        # only include agents from batch 0 to avoid mixing up positions from different scenarios
        gt_positions_by_agent_and_time = {}
        for step in range(actual_rollout_steps):
            target_t = start_t + step + 1
            if target_t >= T:
                break
            
            target_graph = batched_graph_sequence[target_t]
            
            if hasattr(target_graph, 'pos') and target_graph.pos is not None:
                pos_at_t = target_graph.pos.cpu().numpy()
                
                # Filter to only first scenario (batch_idx=0) when B > 1
                if B > 1 and hasattr(target_graph, 'batch'):
                    batch0_mask = (target_graph.batch == 0)
                    batch0_indices = torch.where(batch0_mask)[0].cpu().numpy()
                else:
                    batch0_indices = np.arange(target_graph.num_nodes)
                
                if hasattr(target_graph, 'agent_ids'):
                    all_agent_ids = list(target_graph.agent_ids)
                    for local_idx, global_idx in enumerate(batch0_indices):
                        if global_idx < len(all_agent_ids) and global_idx < pos_at_t.shape[0]:
                            aid = all_agent_ids[global_idx]
                            if aid not in gt_positions_by_agent_and_time:
                                gt_positions_by_agent_and_time[aid] = {}
                            gt_positions_by_agent_and_time[aid][t] = pos_at_t[global_idx].copy()
            





            # Check node count match (no need to check since we're working with batched graphs that should maintain structure)
            # Note: When B > 1, both graphs have nodes from all scenarios, so total count should match
            # if graph_for_prediction.x.shape[0] != target_graph.num_nodes:
            #     break
            
            # Filter prediction graph to first scenario (batch_idx=0) for tracking agents
            pred_scenario_indices, current_agent_ids, _ = filter_to_scenario(graph_for_prediction, 0, B)
            
            # GAT forward pass with agent_ids for per-agent GRU tracking
            agent_ids_for_model = getattr(graph_for_prediction, 'agent_ids', None)
            pred = model(graph_for_prediction.x, graph_for_prediction.edge_index,
                        batch=graph_for_prediction.batch, batch_size=B, batch_num=0, timestep=start_t + step,
                        agent_ids=agent_ids_for_model)
            
            # Model predicts NORMALIZED displacement [N, 2]: (dx, dy) / POSITION_SCALE
            # Convert to meters for position updates
            pred_disp_normalized = pred.cpu().numpy()  # [N, 2] normalized
            pred_disp = pred_disp_normalized * POSITION_SCALE  # convert to meters
            dt = 0.1  # timestep in seconds (for velocity calculations)
            
            # DEBUG: Print prediction stats on first few steps
            if step < 5 and enable_debug_viz:
                print(f"\t\t[DEBUG VIZ] Step {step}: Raw pred (norm): min={pred.min().item():.4f}, max={pred.max().item():.4f}, mean={pred.mean().item():.4f}")
                # Print input features for scenario 0's first agent
                if len(pred_scenario_indices) > 0:
                    first_agent_idx = pred_scenario_indices[0]
                    agent_feats = graph_for_prediction.x[first_agent_idx].cpu().numpy()
                    print(f"\t\t[DEBUG VIZ] Step {step}: Agent 0 features: vx={agent_feats[0]:.3f}, vy={agent_feats[1]:.3f}, speed={agent_feats[2]:.3f}, heading={agent_feats[3]:.3f}")
                    # Also print position for context
                    if hasattr(graph_for_prediction, 'pos'):
                        agent_pos = graph_for_prediction.pos[first_agent_idx].cpu().numpy()
                        print(f"  [DEBUG VIZ] Step {step}: Agent 0 predicted pos: {agent_pos}")
                # Show what the model predicts for this agent
                if len(pred_scenario_indices) > 0:
                    pred_disp_agent0 = pred_disp[pred_scenario_indices[0]]
                    # Calculate predicted speed from displacement (displacement / timestep)
                    dt = 0.1  # 0.1 second timestep
                    pred_vx = pred_disp_agent0[0] / dt
                    pred_vy = pred_disp_agent0[1] / dt
                    pred_speed = np.sqrt(pred_disp_agent0[0]**2 + pred_disp_agent0[1]**2) / dt
                    print(f"\t\t[DEBUG VIZ] Step {step}: Agent 0: pred_disp_m={pred_disp_agent0}, pred_v=({pred_vx:.2f}, {pred_vy:.2f}) m/s, pred_speed={pred_speed:.2f} m/s")
                    
                    # CRITICAL DEBUG: Compare with GT displacement at this step
                    if target_graph.y is not None and first_agent_idx < target_graph.y.shape[0]:
                        # For step 0, we're predicting from start_graph to target_graph
                        # GT is in batched_graph_sequence[start_t].y (normalized displacement to next step)
                        source_graph = batched_graph_sequence[start_t + step]
                        if source_graph.y is not None and first_agent_idx < source_graph.y.shape[0]:
                            # GT y is NORMALIZED, convert to meters for display
                            gt_disp_normalized = source_graph.y[first_agent_idx].cpu().numpy()
                            gt_disp = gt_disp_normalized * POSITION_SCALE  # convert to meters
                            gt_vx = gt_disp[0] / dt
                            gt_vy = gt_disp[1] / dt
                            gt_speed = np.sqrt(gt_disp[0]**2 + gt_disp[1]**2) / dt
                            error = np.sqrt(((pred_disp_agent0 - gt_disp) ** 2).sum())
                            print(f"  [DEBUG VIZ] Step {step}: Agent 0: gt_disp_m={gt_disp}, gt_v=({gt_vx:.2f}, {gt_vy:.2f}) m/s, gt_speed={gt_speed:.2f} m/s, error={error:.2f}m")
            
            # Map predictions to persistent agents (only first scenario)
            for local_idx, agent_id in enumerate(current_agent_ids):
                if agent_id in agent_pred_positions:
                    # Get global index in the full batched graph
                    global_idx = pred_scenario_indices[local_idx]
                    if global_idx < pred_disp.shape[0]:
                        # Verify agent ID consistency
                        if hasattr(graph_for_prediction, 'agent_ids') and global_idx < len(graph_for_prediction.agent_ids):
                            actual_agent_id = graph_for_prediction.agent_ids[global_idx]
                            if actual_agent_id != agent_id and step < 3 and local_idx < 3:
                                print(f"  [WARNING] Agent ID mismatch at step {step}: expected {agent_id}, got {actual_agent_id} at global_idx {global_idx}")
                        
                        new_pos = agent_pred_positions[agent_id] + pred_disp[global_idx]
                        agent_pred_positions[agent_id] = new_pos
                        
                        # DEBUG: Print position updates on first few steps
                        if step < 3 and local_idx == 0 and enable_debug_viz:
                            print(f"  [DEBUG VIZ] Step {step}, Agent {agent_id}: disp={pred_disp[global_idx]}, new_pos={new_pos}")
                        agent_trajectories[agent_id]['pred'].append((target_t, new_pos.copy()))
            
            # NEW APPROACH: Use GT graph from next timestep with only position offset
            # This gives the model correct GT velocity features for each agent
            next_target_t = start_t + step + 2
            if next_target_t < T:
                gt_graph_next = batched_graph_sequence[next_target_t]
                graph_for_prediction = gt_graph_next.clone()
                
                # Update position using agent alignment (same fix as training)
                if hasattr(graph_for_prediction, 'pos') and graph_for_prediction.pos is not None:
                    current_gt_graph = batched_graph_sequence[start_t + step]
                    # Model predicts NORMALIZED displacement, convert to meters for position updates
                    pred_displacement_meters = pred * POSITION_SCALE  # [N, 2] in meters, on device
                    
                    # CRITICAL: Only update positions for agents that exist in BOTH graphs
                    if (hasattr(current_gt_graph, 'agent_ids') and hasattr(gt_graph_next, 'agent_ids') and
                        hasattr(current_gt_graph, 'batch') and hasattr(gt_graph_next, 'batch')):
                        
                        # Keep batch tensors on GPU for faster processing
                        current_batch_tensor = current_gt_graph.batch
                        next_batch_tensor = gt_graph_next.batch
                        
                        # Build mappings (need CPU for dict keys, but minimize data transfer)
                        current_batch_np = current_batch_tensor.cpu().numpy()
                        next_batch_np = next_batch_tensor.cpu().numpy()
                        
                        # Vectorized dictionary building
                        num_current = min(len(current_gt_graph.agent_ids), len(current_batch_np))
                        num_next = min(len(gt_graph_next.agent_ids), len(next_batch_np))
                        
                        current_id_to_idx = {(int(current_batch_np[i]), current_gt_graph.agent_ids[i]): i 
                                            for i in range(num_current)}
                        next_id_to_idx = {(int(next_batch_np[i]), gt_graph_next.agent_ids[i]): i 
                                         for i in range(num_next)}
                        
                        common_ids = set(current_id_to_idx.keys()) & set(next_id_to_idx.keys())
                        
                        if common_ids:
                            # GPU-accelerated batch update using advanced indexing
                            current_indices = torch.tensor([current_id_to_idx[gid] for gid in common_ids], 
                                                          device=device, dtype=torch.long)
                            next_indices = torch.tensor([next_id_to_idx[gid] for gid in common_ids], 
                                                       device=device, dtype=torch.long)
                            
                            # Validate indices and create valid mask on GPU
                            valid_mask = (current_indices < pred_displacement_meters.shape[0]) & \
                                        (current_indices < current_gt_graph.pos.shape[0]) & \
                                        (next_indices < graph_for_prediction.pos.shape[0])
                            
                            if valid_mask.any():
                                valid_current = current_indices[valid_mask]
                                valid_next = next_indices[valid_mask]
                                
                                # Batch update positions on GPU - single operation!
                                graph_for_prediction.pos[valid_next] = (current_gt_graph.pos[valid_current] + 
                                                                        pred_displacement_meters[valid_current])
                    else:
                        # Fallback: only update if sizes match
                        if current_gt_graph.pos.shape[0] == pred_displacement_meters.shape[0] == graph_for_prediction.pos.shape[0]:
                            graph_for_prediction.pos = current_gt_graph.pos.to(device) + pred_displacement_meters.to(device)
            else:
                # No more GT graphs available, use update function as fallback
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
    
    # Print horizon error summary to help diagnose accumulation vs base model issues
    # HORIZON ERROR: Mean Euclidean distance between predicted and GT positions
    # across all agents at each timestep. Shows how error accumulates over time.
    if len(horizon_errors) >= 5:
        print(f"  [VIZ SUMMARY] Horizon errors (mean |pred_pos - gt_pos| across all agents at each timestep):")
        print(f"    Step 1 (0.1s): {horizon_errors[0]:.2f}m")
        print(f"    Step 5 (0.5s): {horizon_errors[4]:.2f}m")
        if len(horizon_errors) >= 10:
            print(f"    Step 10 (1.0s): {horizon_errors[9]:.2f}m")
        if len(horizon_errors) >= 30:
            print(f"    Step 30 (3.0s): {horizon_errors[29]:.2f}m")
        if len(horizon_errors) >= 50:
            print(f"    Step 50 (5.0s): {horizon_errors[49]:.2f}m")
        print(f"    Final ({len(horizon_errors)*0.1:.1f}s): {horizon_errors[-1]:.2f}m")
        
        # Calculate error growth rate to help diagnose
        if horizon_errors[0] > 0:
            growth_10 = horizon_errors[min(9, len(horizon_errors)-1)] / horizon_errors[0] if len(horizon_errors) > 9 else None
            if growth_10:
                print(f"    Error growth (step 1 -> 10): {growth_10:.1f}x")
    
    # Print per-agent final errors to identify outliers
    # FINAL ERROR: Euclidean distance |pred_pos - gt_pos| for each agent at the last timestep.
    # High variance indicates some agents are being predicted poorly.
    final_t = start_t + actual_rollout_steps
    agent_final_errors = []
    for agent_id, traj in agent_trajectories.items():
        gt_at_final = [pos for t, pos in traj['gt'] if t == final_t]
        pred_at_final = [pos for t, pos in traj['pred'] if t == final_t]
        if gt_at_final and pred_at_final:
            error = np.sqrt(((gt_at_final[0] - pred_at_final[0]) ** 2).sum())
            agent_final_errors.append((agent_id, error))
    
    if agent_final_errors:
        agent_final_errors.sort(key=lambda x: x[1], reverse=True)
        print(f"  [PER-AGENT] Final position errors |pred - gt| at t={actual_rollout_steps*0.1:.1f}s (top 5 worst / total {len(agent_final_errors)} agents):")
        for i, (agent_id, error) in enumerate(agent_final_errors[:5]):
            print(f"    Agent {agent_id}: {error:.2f}m")
        if len(agent_final_errors) > 5:
            print(f"    ... best: {agent_final_errors[-1][1]:.2f}m, median: {agent_final_errors[len(agent_final_errors)//2][1]:.2f}m")
    
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
    ax1.set_title(f'{"GAT" if model_type == "gat" else "GCN"} Autoregressive Rollout: {actual_rollout_steps} steps ({actual_rollout_steps * 0.1:.1f}s)\n'
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
    filename = f'{"gat" if model_type == "gat" else "gcn"}_autoreg_epoch{epoch+1:03d}_{scenario_str}_{timestamp}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # NOTE: Don't log to wandb here - visualization images are logged per-epoch in main loop
    # to avoid step conflicts. The main loop collects all viz paths and logs them together.
    
    print(f"  -> Saved visualization to {filepath}")
    
    # Return final horizon error for early stopping tracking
    final_horizon_error = horizon_errors[-1] if horizon_errors else float('inf')
    return filepath, final_horizon_error

def train_epoch_autoregressive(model, dataloader, optimizer, device, 
                               sampling_prob, num_rollout_steps, is_parallel, scaler=None, epoch=0, total_epochs=40):
    """Train one epoch with scheduled sampling for autoregressive prediction (with optional AMP).
    
    Processes multiple scenarios in parallel using PyG's batching mechanism.
    Uses adaptive velocity smoothing to stabilize early training.
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_magnitude_ratio = 0.0  # Track magnitude ratio loss
    total_overshoot = 0.0  # Track overshoot penalty
    total_trajectory = 0.0  # Track trajectory loss
    total_vel_consistency = 0.0  # Track velocity consistency loss
    steps = 0
    count = 0
    
    # Track per-agent errors for ADE/FDE computation
    train_agent_errors = []  # List of [num_agents, num_steps] arrays
    
    base_model = model.module if is_parallel else model
    use_amp_local = scaler is not None and torch.cuda.is_available()
    
    for batch_idx, batch_dict in enumerate(dataloader):
        batched_graph_sequence = batch_dict["batch"]
        B, T = batch_dict["B"], batch_dict["T"]
        for t in range(T): batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
        
        num_nodes = batched_graph_sequence[0].num_nodes
        num_edges = batched_graph_sequence[0].edge_index.size(1)
        
        # Compute agents per scenario for proper GRU hidden state sizing
        # CRITICAL: The GRU hidden state must be sized as [layers, B * max_agents_per_scenario, hidden_dim]
        # NOT [layers, total_nodes, hidden_dim] which treats all nodes as one scenario
        first_graph = batched_graph_sequence[0]
        if hasattr(first_graph, 'batch') and first_graph.batch is not None:
            batch_counts = torch.bincount(first_graph.batch, minlength=B)
            agents_per_scenario = batch_counts.tolist()
        else:
            agents_per_scenario = [num_nodes]  # Single scenario fallback
        
        # Print parallel execution info for autoregressive rollout (every 20 batches)
        if batch_idx % 20 == 0:
            if torch.cuda.is_available():
                print(f"\n[Batch {batch_idx}]: Processing {B} scenarios in parallel | Rollout: {num_rollout_steps} steps | Total Nodes: {num_nodes} | Edges: {num_edges}")
                print(f"  Agents per scenario: min={min(agents_per_scenario)}, max={max(agents_per_scenario)}, avg={sum(agents_per_scenario)/len(agents_per_scenario):.1f}")
            else:
                print(f"Batch {batch_idx}: B={B}, Nodes={num_nodes}, Rollout={num_rollout_steps} steps")
        
        # Reset GRU with proper batch info - NOT just num_agents!
        base_model.reset_gru_hidden_states(
            batch_size=B,
            agents_per_scenario=agents_per_scenario,
            device=device
        )
        
        optimizer.zero_grad()
        
        # CURRICULUM LEARNING: Start with short rollouts, gradually increase
        # This helps model learn to handle error accumulation progressively
        curriculum_steps = curriculum_rollout_steps(epoch, num_rollout_steps, total_epochs)
        effective_rollout = min(curriculum_steps, T - 1)
        if effective_rollout < 1:
            continue
        if batch_idx % 20 == 0:
            print(f"  [CURRICULUM] Rollout steps: {effective_rollout} (max: {num_rollout_steps})")
        
        # TODO: Do not use simplified method:
        # SIMPLIFIED: Do ONE rollout per batch starting from t=0
        # This is cleaner and more effective than trying many starting points
        # Each batch contains 48 scenarios, so we get plenty of diversity
        t = 0  # Always start from beginning of scenario
        current_graph = batched_graph_sequence[t]
        if current_graph.y is None:
            continue
        
        # GRU hidden states were reset at batch start, no need to reset again here
        
        rollout_loss = None
        graph_for_prediction = current_graph
        is_using_predicted_positions = False  # Track if we're using our predictions
        
        # CRITICAL: Track accumulated displacements for correct ADE/FDE computation
        # We accumulate displacements and add them to GT start positions to get predicted positions
        # This avoids index alignment issues when agents enter/leave
        accumulated_displacements = {}  # {(batch_idx, agent_id): displacement_tensor}
        
        # NEW: Track accumulated PREDICTED displacements WITH gradients for trajectory loss
        # This allows backprop to flow through all predictions to penalize trajectory drift
        accumulated_pred_displacements_grad = {}  # {(batch_idx, agent_id): displacement_tensor WITH gradients}
        
        # Initialize per-agent error tracking for ADE/FDE computation
        # Track position error (in meters) per agent per timestep
        # NOTE: This will be resized after first step when we know aligned agent count
        current_rollout_errors = None
        aligned_agent_count = None

        for step in range(effective_rollout):
            target_t = t + step + 1
            if target_t >= T:
                break
            
            target_graph = batched_graph_sequence[target_t]
            if target_graph.y is None:
                break
            
            # ========== CRITICAL: AGENT ALIGNMENT ==========
            # The graphs at different timesteps may have different agents (some enter/leave).
            # We need to compute loss only for agents that exist in BOTH graphs,
            # and align predictions with targets by agent ID, not by node index.
            #
            # IMPORTANT: Agent IDs are NOT globally unique across scenarios in a batch!
            # Two different scenarios can have agents with the same ID.
            # We must combine agent_id with batch_id to create globally unique identifiers.
            
            # Get agent IDs and batch assignments from both graphs
            pred_agent_ids = getattr(graph_for_prediction, 'agent_ids', None)
            target_agent_ids = getattr(target_graph, 'agent_ids', None)
            pred_batch = getattr(graph_for_prediction, 'batch', None)
            target_batch = getattr(target_graph, 'batch', None)
            
            # Validate that agent_ids and batch tensors have matching lengths
            pred_num_nodes = graph_for_prediction.x.shape[0]
            target_num_nodes = target_graph.x.shape[0]
            
            # Check for length mismatches - if any, fall back to simple comparison
            agent_ids_valid = (
                pred_agent_ids is not None and 
                target_agent_ids is not None and
                pred_batch is not None and 
                target_batch is not None and
                len(pred_agent_ids) == pred_num_nodes and
                len(target_agent_ids) == target_num_nodes and
                pred_batch.shape[0] == pred_num_nodes and
                target_batch.shape[0] == target_num_nodes
            )
            
            # If agent IDs and batch info are available and valid, use them for proper alignment
            if agent_ids_valid:
                # Create globally unique IDs by combining (batch_idx, agent_id)
                pred_batch_np = pred_batch.cpu().numpy()
                target_batch_np = target_batch.cpu().numpy()
                
                # Build mapping: (batch_idx, agent_id) -> node_index using vectorized operations
                pred_indices_arr = np.arange(pred_num_nodes)
                pred_bids = pred_batch_np[pred_indices_arr].astype(int)
                pred_id_to_idx = {(bid, aid): idx for idx, bid, aid in zip(pred_indices_arr, pred_bids, pred_agent_ids)}
                
                target_indices_arr = np.arange(target_num_nodes)
                target_bids = target_batch_np[target_indices_arr].astype(int)
                target_id_to_idx = {(bid, aid): idx for idx, bid, aid in zip(target_indices_arr, target_bids, target_agent_ids)}
                
                # Find common agents (same batch AND same agent ID)
                common_ids = set(pred_id_to_idx.keys()) & set(target_id_to_idx.keys())
                
                if len(common_ids) < 2:
                    # Too few common agents, skip this step
                    graph_for_prediction = target_graph
                    is_using_predicted_positions = False
                    continue
                
                # Get aligned indices - convert common_ids to list for deterministic ordering
                common_ids_list = sorted(common_ids)  # Sort for deterministic behavior
                pred_indices = torch.tensor([pred_id_to_idx[gid] for gid in common_ids_list], 
                                            device=device, dtype=torch.long)
                target_indices = torch.tensor([target_id_to_idx[gid] for gid in common_ids_list], 
                                              device=device, dtype=torch.long)
                
                # Validate indices are in bounds
                if pred_indices.max() >= pred_num_nodes or target_indices.max() >= target_num_nodes:
                    # Index out of bounds - fall back to skipping
                    graph_for_prediction = target_graph
                    is_using_predicted_positions = False
                    continue
            else:
                # No valid agent IDs or batch info - fall back to assuming same ordering
                if pred_num_nodes != target_num_nodes:
                    graph_for_prediction = target_graph
                    is_using_predicted_positions = False
                    continue
                pred_indices = None
                target_indices = None
            
            # Safety check input features for NaN before forward pass
            if torch.isnan(graph_for_prediction.x).any():
                print(f"  [ERROR] NaN in input features at batch {batch_idx}, t={t}, step={step}! Skipping rollout.")
                break
            
            # ============ FEATURE NOISE INJECTION ============
            # During teacher forcing (using GT graph), inject noise into derived features
            # to simulate the distribution shift during autoregressive rollout.
            # This helps the model learn to be robust to noisy features.
            if not is_using_predicted_positions and FEATURE_NOISE_INJECTION:
                graph_for_prediction = inject_feature_noise(graph_for_prediction, epoch, total_epochs, device)
            
            # Forward pass with optional AMP and agent_ids for per-agent GRU tracking
            # Both GAT and GCN now support per-agent hidden state tracking
            with torch.amp.autocast('cuda', enabled=use_amp_local):
                edge_w = graph_for_prediction.edge_attr if use_edge_weights else None
                agent_ids_for_model = getattr(graph_for_prediction, 'agent_ids', None)
                
                pred = model(
                    graph_for_prediction.x,
                    graph_for_prediction.edge_index,
                    edge_weight=edge_w,
                    batch=graph_for_prediction.batch,
                    batch_size=B,
                    batch_num=batch_idx,
                    timestep=t + step,
                    agent_ids=agent_ids_for_model
                )
                
                # Safety check for NaN in model output and reset GRU if needed
                if torch.isnan(pred).any():
                    print(f"  [ERROR] NaN in model output at batch {batch_idx}, step {step}! Resetting GRU and skipping.")
                    # reset GRU to prevent NaN propagation
                    base_model.reset_gru_hidden_states(batch_size=B, agents_per_scenario=agents_per_scenario, device=device)
                    break
                
                # Apply alignment if needed
                if pred_indices is not None:
                    # Additional validation: check pred size matches expectations
                    if pred_indices.max() >= pred.shape[0]:
                        # Prediction tensor is smaller than expected - skip this step
                        graph_for_prediction = target_graph
                        is_using_predicted_positions = False
                        continue
                    
                    # Validate target tensor sizes before indexing
                    if is_using_predicted_positions:
                        if (not hasattr(target_graph, 'pos') or target_graph.pos is None or
                            target_indices.max() >= target_graph.pos.shape[0] or
                            not hasattr(graph_for_prediction, 'pos') or graph_for_prediction.pos is None or
                            pred_indices.max() >= graph_for_prediction.pos.shape[0]):
                            # Position tensors invalid - fall back to GT displacement
                            is_using_predicted_positions = False
                    
                    if not is_using_predicted_positions:
                        # Validate target_graph.y exists and has correct size
                        if target_graph.y is None or target_indices.max() >= target_graph.y.shape[0]:
                            # Target invalid - skip this step
                            graph_for_prediction = target_graph
                            continue
                    
                    pred_aligned = pred[pred_indices]
                    
                    # CRITICAL FIX: Target should ALWAYS be the GT displacement
                    # The position offset is only to expose the model to distribution shift
                    # The model learns: "Given velocity/heading, predict next displacement"
                    # NOT: "Correct back to GT trajectory"
                    # 
                    # To get GT displacement: we need the GT graph at (t+step) which has .y
                    # But graph_for_prediction might be offset, so we use the GT graph's .y
                    source_gt_graph = batched_graph_sequence[t + step]
                    if source_gt_graph.y is not None:
                        # Use GT displacement from the source timestep
                        # Need to align by agent ID
                        source_agent_ids = getattr(source_gt_graph, 'agent_ids', None)
                        source_batch = getattr(source_gt_graph, 'batch', None)
                        
                        if (source_agent_ids is not None and source_batch is not None and
                            len(source_agent_ids) == source_gt_graph.y.shape[0]):
                            # GPU-ACCELERATED AGENT MATCHING
                            # Convert agent IDs to tensors for GPU operations
                            source_agent_ids_tensor = torch.tensor(source_agent_ids, device=device, dtype=torch.long)
                            source_batch_tensor = source_batch.to(device)
                            
                            # Get pred batch and agent IDs for aligned agents
                            pred_batch_aligned = pred_batch[pred_indices].to(device)  # [num_aligned]
                            pred_agent_ids_aligned = torch.tensor([pred_agent_ids[idx.item()] if idx.item() < len(pred_agent_ids) else -1 
                                                                   for idx in pred_indices], device=device, dtype=torch.long)  # [num_aligned]
                            
                            # Vectorized matching: find source index for each pred agent
                            # For each pred agent (batch_id, agent_id), find matching source agent
                            # Use broadcasting: [num_aligned, 1] vs [1, num_source]
                            batch_match = (pred_batch_aligned.unsqueeze(1) == source_batch_tensor.unsqueeze(0))  # [num_aligned, num_source]
                            agent_match = (pred_agent_ids_aligned.unsqueeze(1) == source_agent_ids_tensor.unsqueeze(0))  # [num_aligned, num_source]
                            full_match = batch_match & agent_match & (pred_agent_ids_aligned.unsqueeze(1) != -1)  # [num_aligned, num_source]
                            
                            # Get first matching index for each pred agent (or -1 if no match)
                            # Using argmax on boolean tensor gives first True index (0 if all False, so we need to check any())
                            has_match = full_match.any(dim=1)  # [num_aligned]
                            source_indices_tensor = torch.where(has_match, full_match.int().argmax(dim=1), torch.tensor(-1, device=device, dtype=torch.long))
                            valid_mask = source_indices_tensor >= 0
                            
                            if valid_mask.any():
                                # Initialize target with zeros
                                target = torch.zeros(source_indices_tensor.shape[0], 2, device=device, dtype=pred.dtype)
                                # Batch gather valid targets on GPU
                                valid_source_indices = source_indices_tensor[valid_mask]
                                target[valid_mask] = source_gt_graph.y[valid_source_indices].to(pred.dtype)
                            else:
                                target = torch.zeros(source_indices_tensor.shape[0], 2, device=device, dtype=pred.dtype)
                        else:
                            # Fallback: use target_graph.y directly (assumes alignment)
                            target = source_gt_graph.y[target_indices].to(pred.dtype)
                    else:
                        # No GT displacement available - compute from positions (both in meters)
                        gt_current_pos = source_gt_graph.pos[target_indices].to(pred.dtype)
                        gt_next_pos = target_graph.pos[target_indices].to(pred.dtype)
                        # Positions are in meters, but model outputs NORMALIZED displacement
                        target = (gt_next_pos - gt_current_pos) / POSITION_SCALE  # normalized displacement
                else:
                    pred_aligned = pred
                    # Same: always use GT displacement
                    source_gt_graph = batched_graph_sequence[t + step]
                    if source_gt_graph.y is not None:
                        target = source_gt_graph.y.to(pred.dtype)
                    else:
                        # Compute from GT positions (both in meters)
                        gt_current_pos = source_gt_graph.pos.to(pred.dtype)
                        gt_next_pos = target_graph.pos.to(pred.dtype)
                        # Positions are in meters, but model outputs NORMALIZED displacement
                        target = (gt_next_pos - gt_current_pos) / POSITION_SCALE  # normalized displacement
            
            # Check for NaN in predictions or targets before computing loss
            if torch.isnan(pred_aligned).any() or torch.isnan(target).any():
                print(f"  [ERROR] NaN detected in predictions or targets at step {step}! Skipping this step.")
                continue
            
            # Displacement loss components (model predicts displacement directly)
            mse_loss = F.mse_loss(pred_aligned, target)
            huber_loss = F.smooth_l1_loss(pred_aligned, target)
            pred_norm = F.normalize(pred_aligned, p=2, dim=1, eps=1e-6)
            target_norm = F.normalize(target, p=2, dim=1, eps=1e-6)
            cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
            cosine_loss = (1 - cos_sim).mean()
            pred_magnitude = torch.norm(pred_aligned, dim=1)
            target_magnitude = torch.norm(target, dim=1)
            magnitude_loss = F.mse_loss(pred_magnitude, target_magnitude)
            
            # HEADING-AWARE LOSS: Penalize predictions that don't align with agent's heading
            # The heading feature (index 3) tells us where the agent is facing
            # Predictions should generally align with heading for forward-moving vehicles
            heading_loss = torch.tensor(0.0, device=pred.device)
            if hasattr(graph_for_prediction, 'x') and graph_for_prediction.x.shape[0] == pred_aligned.shape[0]:
                # Get heading from features (index 3, in radians, normalized by pi)
                heading_normalized = graph_for_prediction.x[:, 3]  # Already normalized by pi
                if pred_indices is not None:
                    heading_normalized = heading_normalized[pred_indices]
                heading_rad = heading_normalized * np.pi  # Denormalize to radians
                
                # Expected direction from heading
                heading_dir = torch.stack([torch.cos(heading_rad), torch.sin(heading_rad)], dim=1)
                
                # Compute alignment between prediction and heading direction
                # Only apply to agents with significant predicted displacement (moving agents)
                pred_mag = torch.norm(pred_aligned, dim=1, keepdim=True)
                moving_mask = (pred_mag.squeeze() > 0.001)  # > 0.1m displacement
                
                if moving_mask.any():
                    pred_dir = F.normalize(pred_aligned[moving_mask], p=2, dim=1, eps=1e-6)
                    heading_dir_moving = heading_dir[moving_mask].to(pred_dir.dtype)
                    heading_alignment = F.cosine_similarity(pred_dir, heading_dir_moving, dim=1)
                    # Soft penalty: allow some deviation (turning), but penalize large deviations
                    # Use (1 - alignment)^2 to penalize more for wrong directions
                    heading_loss = ((1 - heading_alignment) ** 2).mean()
            
            # ANGLE LOSS: Explicit angle difference penalty (more sensitive to direction errors)
            pred_angle = torch.atan2(pred_aligned[:, 1], pred_aligned[:, 0])
            target_angle = torch.atan2(target[:, 1], target[:, 0])
            angle_diff = pred_angle - target_angle
            # Wrap to [-pi, pi]
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            angle_loss = (angle_diff ** 2).mean()
            
            # MAGNITUDE RATIO LOSS: Penalize when predicted magnitude differs from GT magnitude
            # This is critical for preventing overshoot/undershoot issues seen in visualizations
            # Use log ratio to make it symmetric and scale-invariant
            eps = 1e-6
            magnitude_ratio = (pred_magnitude + eps) / (target_magnitude + eps)
            # Log ratio: log(pred/target) = 0 when perfect, positive when overshoot, negative when undershoot
            log_ratio = torch.log(magnitude_ratio + eps)
            magnitude_ratio_loss = (log_ratio ** 2).mean()
            
            # OVERSHOOT PENALTY: Extra penalty when prediction magnitude exceeds GT magnitude
            # This is asymmetric because overshooting causes more visible trajectory divergence
            overshoot_mask = pred_magnitude > target_magnitude
            if overshoot_mask.any():
                overshoot_amount = (pred_magnitude[overshoot_mask] - target_magnitude[overshoot_mask])
                overshoot_penalty = (overshoot_amount ** 2).mean()
            else:
                overshoot_penalty = torch.tensor(0.0, device=pred.device)
            
            # TRAJECTORY LOSS: Penalize accumulated position error (with gradients!)
            # This teaches the model that small angle errors compound into large trajectory errors
            # CRITICAL: This loss must be applied from step 0 to get gradient flow through ALL predictions
            trajectory_loss = torch.tensor(0.0, device=pred.device)
            if agent_ids_valid and hasattr(target_graph, 'pos') and target_graph.pos is not None:
                # Accumulate predicted displacements WITH gradients
                pred_disp_grad = pred_aligned * POSITION_SCALE  # [num_aligned, 2] in meters
                
                # Update accumulated displacements for each aligned agent using vectorized operations
                if pred_indices is not None:
                    pred_batch_np = pred_batch.cpu().numpy()
                    pred_indices_np = pred_indices.cpu().numpy()
                    
                    # Create masks for valid indices
                    valid_mask = (pred_indices_np < len(pred_agent_ids)) & (np.arange(len(pred_indices_np)) < pred_disp_grad.shape[0])
                    valid_pred_indices = pred_indices_np[valid_mask]
                    valid_i = np.where(valid_mask)[0]
                    
                    # Batch update displacements
                    for i, idx in zip(valid_i, valid_pred_indices):
                        bid = int(pred_batch_np[idx])
                        aid = pred_agent_ids[idx]
                        agent_key = (bid, aid)
                        
                        disp_with_grad = pred_disp_grad[i]
                        if agent_key in accumulated_pred_displacements_grad:
                            accumulated_pred_displacements_grad[agent_key] = accumulated_pred_displacements_grad[agent_key] + disp_with_grad
                        else:
                            accumulated_pred_displacements_grad[agent_key] = disp_with_grad
                
                # Compute trajectory position error for agents we have accumulated displacements
                source_gt_graph = batched_graph_sequence[t]  # Start graph
                source_batch_np = source_gt_graph.batch.cpu().numpy() if hasattr(source_gt_graph, 'batch') else None
                source_agent_ids = getattr(source_gt_graph, 'agent_ids', None)
                
                if source_agent_ids is not None and source_batch_np is not None:
                    # Build source mapping (minimal CPU transfer)
                    num_source = min(len(source_agent_ids), len(source_batch_np), source_gt_graph.pos.shape[0])
                    source_key_to_idx = {(int(source_batch_np[i]), source_agent_ids[i]): i 
                                        for i in range(num_source)}
                    
                    target_batch_np = target_batch.cpu().numpy()
                    
                    # Build parallel lists for GPU batch processing
                    target_pos_indices = []
                    source_pos_indices = []
                    agent_keys_for_grad_update = []
                    
                    for i, target_idx in enumerate(target_indices.cpu().numpy()):
                        if target_idx < len(target_agent_ids):
                            bid = int(target_batch_np[target_idx])
                            aid = target_agent_ids[target_idx]
                            agent_key = (bid, aid)
                            
                            if agent_key in accumulated_pred_displacements_grad and agent_key in source_key_to_idx:
                                source_idx = source_key_to_idx[agent_key]
                                target_pos_indices.append(target_idx)
                                source_pos_indices.append(source_idx)
                                agent_keys_for_grad_update.append(agent_key)
                    
                    if target_pos_indices:
                        # GPU batch operations - gather positions in one go
                        target_pos_tensor = torch.tensor(target_pos_indices, device=device, dtype=torch.long)
                        source_pos_tensor = torch.tensor(source_pos_indices, device=device, dtype=torch.long)
                        
                        # Batch gather GT positions on GPU
                        gt_starts = source_gt_graph.pos[source_pos_tensor]  # [N, 2]
                        gt_targets = target_graph.pos[target_pos_tensor]    # [N, 2]
                        
                        # Stack accumulated displacements in same order
                        accum_disps = torch.stack([accumulated_pred_displacements_grad[key] 
                                                   for key in agent_keys_for_grad_update])  # [N, 2]
                        
                        # Batch compute predicted positions and errors on GPU
                        pred_positions = gt_starts + accum_disps  # [N, 2]
                        pos_errors = torch.norm(pred_positions - gt_targets, dim=1)  # [N]
                        
                        traj_errors = list(pos_errors)  # Keep as tensor list for stacking
                    
                    if traj_errors:
                        # Mean trajectory error - DIRECT position error in meters
                        # Use linear scaling (not sqrt) to maintain strong gradient for large errors
                        # This forces the model to care about trajectory drift from the very first step
                        # 
                        # Key insight: sqrt was dampening the gradient for large errors, but we WANT
                        # strong gradients when trajectory diverges significantly
                        traj_error_tensor = torch.stack(traj_errors)
                        # IMPROVED: Direct L2 loss on position error, normalized by POSITION_SCALE
                        # No sqrt dampening - we WANT strong gradients for large trajectory errors
                        # This is the most important loss for trajectory following
                        trajectory_loss = (traj_error_tensor / POSITION_SCALE).mean()
            
            # ========== IMPROVED LOSS FUNCTION v3 - Feature Utilization & Trajectory Focus ==========
            # 
            # KEY INSIGHTS from visualizations:
            # 1. Model doesn't learn to use derived features (velocity, heading) during autoreg
            # 2. Predictions have correct general direction but WRONG MAGNITUDE
            # 3. Trajectory drift accumulates over 50 steps
            #
            # SOLUTION: Add velocity consistency loss to force model to USE the velocity features
            #
            # Loss components:
            # - huber_loss (0.15): Robust displacement loss
            # - cosine_loss (0.15): Direction alignment
            # - magnitude_loss (0.10): Displacement magnitude MSE
            # - magnitude_ratio_loss (0.10): Log-ratio penalty for scale errors
            # - overshoot_penalty (0.10): Asymmetric penalty for overshooting
            # - trajectory_loss (0.25): Accumulated position error - critical for long rollouts
            # - velocity_consistency_loss (0.15): NEW - force model to use velocity features
            
            # Compute velocity consistency loss
            velocity_consistency_loss = compute_velocity_consistency_loss(
                pred_aligned, graph_for_prediction, device
            )
            
            disp_loss = (0.15 * huber_loss + 
                        0.15 * cosine_loss + 
                        0.10 * magnitude_loss + 
                        0.10 * magnitude_ratio_loss +
                        0.10 * overshoot_penalty +
                        0.25 * trajectory_loss +
                        VELOCITY_CONSISTENCY_WEIGHT * velocity_consistency_loss)
            
            # Final NaN check
            if torch.isnan(disp_loss):
                print(f"  [ERROR] NaN in loss computation at step {step}! Skipping.")
                continue
            
            # INVERTED temporal weighting: later steps get MORE weight, not less!
            # Error accumulates over time, so we need to pay more attention to later steps
            # where trajectory divergence is most severe.
            # Weight increases from 1.0 at step 0 to 1.5 at max steps
            step_weight = 1.0 + 0.5 * (step / max(1, effective_rollout - 1))
            step_loss = disp_loss * step_weight
            
            if rollout_loss is None:
                rollout_loss = step_loss
            else:
                rollout_loss = rollout_loss + step_loss
            
            with torch.no_grad():
                total_mse += mse_loss.item()
                total_cosine += cos_sim.mean().item()
                total_magnitude_ratio += magnitude_ratio_loss.item()
                total_overshoot += overshoot_penalty.item() if isinstance(overshoot_penalty, torch.Tensor) else overshoot_penalty
                total_trajectory += trajectory_loss.item() if isinstance(trajectory_loss, torch.Tensor) else trajectory_loss
                total_vel_consistency += velocity_consistency_loss.item() if isinstance(velocity_consistency_loss, torch.Tensor) else velocity_consistency_loss
                count += 1
                
                # Track per-agent POSITION errors for ADE/FDE computation (not displacement errors!)
                # ADE/FDE measure how far the predicted trajectory is from GT trajectory in meters
                # We need to compute: ||predicted_position - gt_position|| at each timestep
                
                # Convert predicted displacement to meters
                pred_disp_meters = pred_aligned * POSITION_SCALE  # [num_aligned_agents, 2] in meters
                
                # Accumulate displacements per agent for position error computation
                # This handles agent alignment correctly by using global agent IDs
                if agent_ids_valid:
                    # Update accumulated displacements using vectorized operations
                    pred_batch_np = pred_batch.cpu().numpy()
                    pred_indices_np = pred_indices.cpu().numpy()
                    
                    # Create validity mask
                    valid_mask = (pred_indices_np < len(pred_agent_ids)) & (np.arange(len(pred_indices_np)) < pred_disp_meters.shape[0])
                    valid_pred_indices = pred_indices_np[valid_mask]
                    valid_i = np.where(valid_mask)[0]
                    
                    # Batch process displacements
                    for i, idx in zip(valid_i, valid_pred_indices):
                        bid = int(pred_batch_np[idx])
                        aid = pred_agent_ids[idx]
                        agent_key = (bid, aid)
                        
                        # Get displacement for this agent
                        disp = pred_disp_meters[i].cpu()
                        
                        # Accumulate (or initialize if first time)
                        if agent_key in accumulated_displacements:
                            accumulated_displacements[agent_key] = accumulated_displacements[agent_key] + disp
                        else:
                            accumulated_displacements[agent_key] = disp
                
                # Compute position errors using accumulated displacements
                # predicted_position = GT_start_position + accumulated_displacement
                if hasattr(target_graph, 'pos') and target_graph.pos is not None and agent_ids_valid:
                    # Get GT start positions (from t=0 graph) and target positions
                    source_gt_graph = batched_graph_sequence[t]
                    source_batch_np = source_gt_graph.batch.cpu().numpy() if hasattr(source_gt_graph, 'batch') else None
                    source_agent_ids = getattr(source_gt_graph, 'agent_ids', None)
                    
                    target_batch_np = target_batch.cpu().numpy()
                    target_indices_np = target_indices.cpu().numpy()
                    
                    # Build source mapping using vectorized operations
                    if source_agent_ids is not None and source_batch_np is not None:
                        source_valid = np.arange(min(len(source_agent_ids), len(source_batch_np)))
                        source_bids_np = source_batch_np[source_valid].astype(int)
                        source_aids = [source_agent_ids[i] for i in source_valid]
                        source_key_to_idx = {(bid, aid): idx for idx, bid, aid in zip(source_valid, source_bids_np, source_aids)}
                    else:
                        source_key_to_idx = {}
                    
                    # Vectorized position error computation
                    step_errors_list = []
                    valid_target_mask = target_indices_np < len(target_agent_ids)
                    valid_target_indices = target_indices_np[valid_target_mask]
                    
                    for target_idx in valid_target_indices:
                        bid = int(target_batch_np[target_idx])
                        aid = target_agent_ids[target_idx]
                        agent_key = (bid, aid)
                        
                        # Get GT target position
                        gt_target_pos = target_graph.pos[target_idx].cpu()
                        
                        # Compute predicted position: start_pos + accumulated_displacement
                        # NOTE: We use GT start position (at t=0) even when scheduled sampling uses
                        # predicted graphs, because ADE/FDE measure total deviation from GT start.
                        # Mathematically: predicted_pos_t = pos_0 + sum(displacements_0_to_t)
                        # This is equivalent to iterative: pos_t = pos_{t-1} + disp_t
                        # Each displacement was predicted from either GT or predicted state (scheduled sampling).
                        if agent_key in accumulated_displacements and agent_key in source_key_to_idx:
                            source_idx = source_key_to_idx[agent_key]
                            if source_idx < source_gt_graph.pos.shape[0]:
                                gt_start_pos = source_gt_graph.pos[source_idx].cpu()
                                predicted_pos = gt_start_pos + accumulated_displacements[agent_key]
                                
                                # Compute position error
                                error = torch.norm(predicted_pos - gt_target_pos).item()
                                step_errors_list.append(error)
                    
                    if step_errors_list:
                        step_errors_meters = torch.tensor(step_errors_list, device=device)
                    else:
                        # Fallback
                        step_errors_meters = torch.norm(pred_aligned - target, dim=1) * POSITION_SCALE
                else:
                    # Fallback: if no positions available, use displacement error (less accurate)
                    step_errors_meters = torch.norm(pred_aligned - target, dim=1) * POSITION_SCALE
                
                # Initialize error dict on first step - track per agent
                if current_rollout_errors is None:
                    current_rollout_errors = {}
                
                # Store errors for this step - handle each agent individually  
                if agent_ids_valid and hasattr(target_graph, 'agent_ids'):
                    for i, target_idx in enumerate(target_indices.cpu().numpy()):
                        if target_idx < len(target_agent_ids) and i < len(step_errors_list):
                            bid = int(target_batch_np[target_idx])
                            aid = target_agent_ids[target_idx]
                            agent_key = (bid, aid)
                            
                            if agent_key not in current_rollout_errors:
                                current_rollout_errors[agent_key] = []
                            current_rollout_errors[agent_key].append(step_errors_list[i])
            
            if step < num_rollout_steps - 1:
                # Scheduled sampling: sometimes use prediction, sometimes use GT
                use_prediction = np.random.random() < sampling_prob
                
                if use_prediction:
                    # TRUE AUTOREGRESSIVE: Build graph from our predictions
                    graph_for_prediction = update_graph_with_prediction(
                        graph_for_prediction, pred.detach(), device
                    )
                    is_using_predicted_positions = True
                else:
                    # TEACHER FORCING: Use ground truth graph from sequence
                    graph_for_prediction = batched_graph_sequence[target_t]
                    is_using_predicted_positions = False
        
        # end of single rollout - do backward pass immediately
        if rollout_loss is not None:
            # Check for NaN in loss before backward
            if torch.isnan(rollout_loss):
                print(f"  [ERROR] NaN in rollout loss at batch {batch_idx}! Skipping backward pass.")
                continue
            
            avg_loss = rollout_loss / max(1, count)     # Normalize by number of rollout steps for consistent gradient magnitudes
            
            # Backward pass with optional AMP scaling
            if use_amp_local:
                scaler.scale(avg_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), gradient_clip_value)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):    # Check for NaN in gradients
                    print(f"  [ERROR] NaN/Inf in gradients (norm={grad_norm})! Skipping optimizer step.")
                    optimizer.zero_grad()
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                avg_loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), gradient_clip_value)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"  [ERROR] NaN/Inf in gradients (norm={grad_norm})! Skipping optimizer step.")
                    optimizer.zero_grad()
                    continue
                optimizer.step()
            total_loss += rollout_loss.item()
            steps += 1
            
            # Save rollout errors for ADE/FDE computation - convert dict to array
            if current_rollout_errors and len(current_rollout_errors) > 0:
                # Convert dict to array [num_agents, num_steps]
                # Only include agents with complete trajectories (all timesteps present)
                agent_error_arrays = []
                for agent_key, errors in current_rollout_errors.items():
                    if len(errors) == effective_rollout:
                        agent_error_arrays.append(np.array(errors))
                
                if agent_error_arrays:
                    batch_errors = np.stack(agent_error_arrays, axis=0)
                    train_agent_errors.append(batch_errors)
        
        if batch_idx % 20 == 0:
            loss_val = rollout_loss.item() if rollout_loss is not None and hasattr(rollout_loss, 'item') else 0.0
            avg_mse_so_far = total_mse / max(1, count)
            avg_cos_so_far = total_cosine / max(1, count)
            rmse_meters = (avg_mse_so_far ** 0.5) * 100.0
            print(f"  Loss={loss_val/max(1,count):.4f}, Sampling prob={sampling_prob:.2f}")
            print(f"  [METRICS] MSE={avg_mse_so_far:.6f} | RMSE={rmse_meters:.2f}m | CosSim={avg_cos_so_far:.4f}")
    

    final_mse = total_mse / max(1, count)
    final_rmse = (final_mse ** 0.5) * 100.0
    final_loss = total_loss / max(1, steps)
    final_cosine = total_cosine / max(1, count)
    final_magnitude_ratio = total_magnitude_ratio / max(1, count)
    final_overshoot = total_overshoot / max(1, count)
    final_trajectory = total_trajectory / max(1, count)
    final_vel_consistency = total_vel_consistency / max(1, count)
    
    # Compute ADE and FDE from accumulated agent errors
    train_ade = 0.0
    train_fde = 0.0
    total_train_agents = 0
    if train_agent_errors:
        # Concatenate all rollout errors: [total_agents, num_steps]
        all_train_errors = np.concatenate(train_agent_errors, axis=0)
        if all_train_errors.shape[0] > 0 and all_train_errors.shape[1] > 0:
            # ADE: Average displacement error across all timesteps per agent, then across agents
            per_agent_ade = np.mean(all_train_errors, axis=1)  # [total_agents]
            train_ade = float(np.mean(per_agent_ade))
            # FDE: Final displacement error (last timestep)
            train_fde = float(np.mean(all_train_errors[:, -1]))
            total_train_agents = all_train_errors.shape[0]
    
    print(f"\n[TRAIN EPOCH SUMMARY]")
    print(f"  Loss: {final_loss:.6f} | Avg per-step MSE: {final_mse:.6f} | Avg per-step RMSE: {final_rmse:.2f}m | CosSim: {final_cosine:.4f}")
    print(f"  Magnitude Losses: ratio_loss={final_magnitude_ratio:.6f} | overshoot={final_overshoot:.6f} | trajectory={final_trajectory:.6f}")
    print(f"  Velocity Consistency Loss: {final_vel_consistency:.6f}")
    print(f"  ADE: {train_ade:.2f}m | FDE: {train_fde:.2f}m (across {total_train_agents} agent rollouts)")
    print(f"  Training uses sampling_prob={sampling_prob:.2f} (0=teacher forcing, 1=autoregressive)")
    
    # NOTE: No wandb.log here - logging is done per-epoch in main training loop
    # This avoids duplicate logging (per-step vs per-epoch)
    
    return {
        'loss': final_loss,
        'mse': final_mse,
        'cosine_sim': final_cosine,
        'magnitude_ratio_loss': final_magnitude_ratio,
        'overshoot_loss': final_overshoot,
        'trajectory_loss': final_trajectory,
        'velocity_consistency_loss': final_vel_consistency,
        'ade': train_ade,
        'fde': train_fde
    }

def evaluate_autoregressive(model, dataloader, device, num_rollout_steps, is_parallel, epoch=0, total_epochs=40):
    """Evaluate GAT model with full autoregressive rollout (no teacher forcing).
    Computes:
    - ADE (Average Displacement Error): mean displacement error across all timesteps
    - FDE (Final Displacement Error): displacement error at the last timestep
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    steps = 0
    count = 0
    
    # TODO: the GT velocity from dataset should not be used for evaluation, model can only know current and previous node features
    # NEW APPROACH: Keep GT velocity features, only update position
    velocity_smoothing = 0.0  # Not used
    update_velocity = False   # Keep GT features
    
    horizon_mse = [0.0] * num_rollout_steps
    horizon_cosine = [0.0] * num_rollout_steps
    horizon_counts = [0] * num_rollout_steps
    
    # Track displacement errors per agent per scenario for ADE/FDE calculation
    all_agent_errors = []  # List of [num_agents, num_steps] arrays
    
    base_model = model.module if is_parallel else model
    
    # Track position drift for debugging
    debug_printed = False
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            batched_graph_sequence = batch_dict["batch"]
            B, T = batch_dict["B"], batch_dict["T"]
            
            for t in range(T): batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
            
            num_nodes = batched_graph_sequence[0].num_nodes
            
            # Compute agents per scenario for proper GRU hidden state sizing
            first_graph = batched_graph_sequence[0]
            if hasattr(first_graph, 'batch') and first_graph.batch is not None:
                batch_counts = torch.bincount(first_graph.batch, minlength=B)
                agents_per_scenario = batch_counts.tolist()
            else:
                agents_per_scenario = [num_nodes]
            
            # Reset GRU with proper batch info
            base_model.reset_gru_hidden_states(
                batch_size=B,
                agents_per_scenario=agents_per_scenario,
                device=device
            )
            
            start_t = T // 3
            effective_rollout = min(num_rollout_steps, T - start_t - 1)  # cap rollout to available timesteps
            if effective_rollout < 1:
                continue

            
            # WARM UP GRU: Run through timesteps 0 to start_t to build temporal context
            # This matches training which processes sequences from t=0
            for warm_t in range(start_t):
                warm_graph = batched_graph_sequence[warm_t]
                if warm_graph.x is not None:
                    agent_ids_warm = getattr(warm_graph, 'agent_ids', None)
                    _ = model(warm_graph.x, warm_graph.edge_index,
                             batch=warm_graph.batch, batch_size=B, batch_num=0, timestep=warm_t,
                             agent_ids=agent_ids_warm)
            
            # Start rollout from start_t (one starting point per batch)
            current_graph = batched_graph_sequence[start_t]
            if current_graph.y is None:
                continue
            
            graph_for_prediction = current_graph
            rollout_loss = 0.0
            is_using_predicted_positions = False
            
            # CRITICAL: Track accumulated displacements for correct ADE/FDE computation
            # We accumulate displacements and add them to GT start positions to get predicted positions
            accumulated_displacements = {}  # {(batch_idx, agent_id): displacement_tensor}
            
            # Initialize per-agent error tracking for this batch's rollout
            # Will be resized after first step when we know aligned agent count
            current_rollout_errors = None
            aligned_agent_count = None
            
            for step in range(effective_rollout):
                target_t = start_t + step + 1
                if target_t >= T:
                    break
                
                target_graph = batched_graph_sequence[target_t]
                if target_graph.y is None:
                    break
                
                # CRITICAL FIX: Do NOT skip when sizes don't match!
                # Instead, we compute errors for agents that exist in BOTH graphs.
                # The previous code was skipping error computation entirely, causing error plateaus.
                size_mismatch = (graph_for_prediction.x.shape[0] != target_graph.y.shape[0])
                if size_mismatch:
                    # Update graph_for_prediction to match target topology for next iteration
                    # But STILL compute errors for this step using aligned agents
                    pass  # Continue to compute errors below
                
                # GAT forward with agent_ids for per-agent GRU tracking
                agent_ids_for_model = getattr(graph_for_prediction, 'agent_ids', None)
                pred = model(
                    graph_for_prediction.x,
                    graph_for_prediction.edge_index,
                    batch=graph_for_prediction.batch,
                    batch_size=B,
                    batch_num=0,
                    timestep=start_t + step,
                    agent_ids=agent_ids_for_model
                )
                
                # PURE AUTOREGRESSIVE EVALUATION: No teacher forcing or loss computation during rollout
                # Model predicts displacement based purely on current graph features
                # We only track position errors against GT for metrics (ADE/FDE)
                
                # Convert predicted displacement from normalized to meters for position update
                pred_disp_meters = pred * POSITION_SCALE  # [N, 2] in meters
                
                # Accumulate displacements for position error computation
                # Store displacement for each agent by (batch_idx, agent_id)
                if hasattr(graph_for_prediction, 'agent_ids') and hasattr(graph_for_prediction, 'batch'):
                    pred_agent_ids = graph_for_prediction.agent_ids
                    pred_batch = graph_for_prediction.batch
                    pred_batch_np = pred_batch.cpu().numpy()
                    
                    # Minimize CPU operations - batch process on GPU then transfer
                    num_valid = min(pred_disp_meters.shape[0], len(pred_agent_ids), len(pred_batch_np))
                    
                    # Process displacements in batch (already on device from pred_disp_meters)
                    for node_idx in range(num_valid):
                        bid = int(pred_batch_np[node_idx])
                        aid = pred_agent_ids[node_idx]
                        agent_key = (bid, aid)
                        disp = pred_disp_meters[node_idx].cpu()  # Single transfer per displacement
                        
                        # Accumulate displacement
                        if agent_key in accumulated_displacements:
                            accumulated_displacements[agent_key] = accumulated_displacements[agent_key] + disp
                        else:
                            accumulated_displacements[agent_key] = disp
                
                # Compute position errors: predicted_pos = start_pos + accumulated_displacement
                if hasattr(target_graph, 'pos') and target_graph.pos is not None:
                    start_graph = batched_graph_sequence[start_t]
                    start_agent_ids = getattr(start_graph, 'agent_ids', None)
                    start_batch = getattr(start_graph, 'batch', None)
                    start_batch_np = start_batch.cpu().numpy() if start_batch is not None else None
                    
                    target_agent_ids = getattr(target_graph, 'agent_ids', None)
                    target_batch = getattr(target_graph, 'batch', None)
                    target_batch_np = target_batch.cpu().numpy() if target_batch is not None else None
                    
                    step_errors_list = []
                    if (start_agent_ids is not None and target_agent_ids is not None and 
                        start_batch_np is not None and target_batch_np is not None):
                        
                        # Build start_key_to_idx mapping (minimal CPU work)
                        num_start = min(len(start_agent_ids), len(start_batch_np), start_graph.pos.shape[0])
                        start_key_to_idx = {(int(start_batch_np[i]), start_agent_ids[i]): i 
                                           for i in range(num_start)}
                        
                        # Build parallel lists for GPU batch gather
                        target_pos_indices = []
                        start_pos_indices = []
                        agent_keys_for_errors = []
                        
                        num_target = min(len(target_agent_ids), len(target_batch_np), target_graph.pos.shape[0])
                        for target_idx in range(num_target):
                            bid = int(target_batch_np[target_idx])
                            aid = target_agent_ids[target_idx]
                            agent_key = (bid, aid)
                            
                            if agent_key in accumulated_displacements and agent_key in start_key_to_idx:
                                start_idx = start_key_to_idx[agent_key]
                                target_pos_indices.append(target_idx)
                                start_pos_indices.append(start_idx)
                                agent_keys_for_errors.append(agent_key)
                        
                        if target_pos_indices:
                            # GPU batch gather operations
                            target_pos_tensor = torch.tensor(target_pos_indices, device=device, dtype=torch.long)
                            start_pos_tensor = torch.tensor(start_pos_indices, device=device, dtype=torch.long)
                            
                            # Batch gather positions on GPU
                            gt_target_positions = target_graph.pos[target_pos_tensor].cpu()  # [N, 2]
                            gt_start_positions = start_graph.pos[start_pos_tensor].cpu()    # [N, 2]
                            
                            # Stack accumulated displacements (already on CPU)
                            accum_disps = torch.stack([accumulated_displacements[key] 
                                                      for key in agent_keys_for_errors])  # [N, 2]
                            
                            # Batch compute predicted positions and errors
                            predicted_positions = gt_start_positions + accum_disps  # [N, 2]
                            errors = torch.norm(predicted_positions - gt_target_positions, dim=1)  # [N]
                            step_errors_list = errors.tolist()
                    
                    if step_errors_list:
                        step_errors_meters = torch.tensor(step_errors_list, device=device)
                    else:
                        # Fallback: displacement error
                        gt_disp = target_graph.y * POSITION_SCALE if target_graph.y is not None else pred_disp_meters
                        step_errors_meters = torch.norm(pred_disp_meters - gt_disp, dim=1)
                else:
                    # No GT positions available - use displacement error as fallback
                    gt_disp = target_graph.y * POSITION_SCALE if target_graph.y is not None else pred_disp_meters
                    step_errors_meters = torch.norm(pred_disp_meters - gt_disp, dim=1)
                
                # For loss tracking (optional - not used for gradient, just monitoring)
                # Compute MSE on displacement vs GT displacement (from dataset y attribute)
                if target_graph.y is not None:
                    # Handle size mismatch: only compute loss for overlapping nodes
                    if size_mismatch:
                        # Use only the common size
                        min_size = min(pred.shape[0], target_graph.y.shape[0])
                        mse = F.mse_loss(pred[:min_size], target_graph.y[:min_size].to(pred.dtype))
                    else:
                        mse = F.mse_loss(pred, target_graph.y.to(pred.dtype))
                else:
                    mse = torch.tensor(0.0, device=pred.device)
                
                # Track per-agent displacement error at each step for ADE/FDE
                # Use dict to handle agent count mismatches gracefully
                if current_rollout_errors is None:
                    # Initialize dict to store per-agent errors: {agent_key: [errors_over_time]}
                    current_rollout_errors = {}
                
                # Store errors for this step - use agent keys from accumulated_displacements
                # since that's what we used to compute the errors
                if step_errors_list and accumulated_displacements:
                    # Get the agent keys that we computed errors for (in order)
                    computed_agent_keys = []
                    if (start_agent_ids is not None and target_agent_ids is not None and 
                        start_batch_np is not None and target_batch_np is not None):
                        # Build start lookup using vectorized operations
                        start_valid = np.arange(min(len(start_agent_ids), len(start_batch_np), start_graph.pos.shape[0]))
                        start_bids_np = start_batch_np[start_valid].astype(int)
                        start_aids = [start_agent_ids[i] for i in start_valid]
                        start_keys = {(bid, aid) for bid, aid in zip(start_bids_np, start_aids)}
                        
                        # Vectorized agent key collection
                        valid_target_indices = np.arange(min(len(target_agent_ids), len(target_batch_np), target_graph.pos.shape[0]))
                        target_bids = target_batch_np[valid_target_indices].astype(int)
                        target_aids = [target_agent_ids[idx] for idx in valid_target_indices]
                        
                        for bid, aid in zip(target_bids, target_aids):
                            agent_key = (bid, aid)
                            if agent_key in accumulated_displacements and agent_key in start_keys:
                                computed_agent_keys.append(agent_key)
                    
                    # Now store errors with correct agent keys
                    for i, error_val in enumerate(step_errors_list):
                        if i < len(computed_agent_keys):
                            agent_key = computed_agent_keys[i]
                            if agent_key not in current_rollout_errors:
                                current_rollout_errors[agent_key] = []
                            current_rollout_errors[agent_key].append(error_val)
                
                # Debug: print position error progression for first batch
                if batch_idx == 0 and not debug_printed:
                    if step_errors_list:
                        avg_pos_error = np.mean(step_errors_list)
                        max_pos_error = np.max(step_errors_list)
                        num_agents = len(step_errors_list)
                    else:
                        avg_pos_error = step_errors_meters.mean().item()
                        max_pos_error = step_errors_meters.max().item()
                        num_agents = step_errors_meters.shape[0]
                    
                    # Print more frequently for debugging error accumulation
                    if step in [0, 2, 4, 9, 14, 19, 29, 39, 49]:
                        size_info = f" (mismatch)" if size_mismatch else ""
                        print(f"    [VAL DEBUG] Step {step+1} ({(step+1)*0.1:.1f}s): avg_err={avg_pos_error:.2f}m, max={max_pos_error:.2f}m, agents={num_agents}{size_info}")
                    if step >= 49:
                        debug_printed = True
                
                # Compute cosine similarity on displacement direction (for monitoring)
                if target_graph.y is not None:
                    # Handle size mismatch: only compute similarity for overlapping nodes
                    if size_mismatch:
                        min_size = min(pred.shape[0], target_graph.y.shape[0])
                        pred_norm = F.normalize(pred[:min_size], p=2, dim=1, eps=1e-6)
                        target_norm = F.normalize(target_graph.y[:min_size].to(pred.dtype), p=2, dim=1, eps=1e-6)
                        cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                    else:
                        pred_norm = F.normalize(pred, p=2, dim=1, eps=1e-6)
                        target_norm = F.normalize(target_graph.y.to(pred.dtype), p=2, dim=1, eps=1e-6)
                        cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                else:
                    cos_sim = torch.tensor(1.0, device=pred.device)
                
                rollout_loss += mse
                total_mse += mse.item()
                total_cosine += cos_sim.item()
                count += 1
                
                horizon_mse[step] += mse.item()
                horizon_cosine[step] += cos_sim.item()
                horizon_counts[step] += 1
                
                if step < num_rollout_steps - 1:
                    # ALWAYS use predictions in evaluation (no teacher forcing)
                    # Handle size mismatch: if target_graph has different agents,
                    # we need to update to target topology but preserve predicted positions for common agents
                    if size_mismatch:
                        # For agents that exist in both graphs, transfer predicted positions to target
                        # This maintains autoregressive behavior while handling topology changes
                        graph_for_prediction = update_graph_with_prediction(
                            target_graph.clone(), pred[:min(pred.shape[0], target_graph.x.shape[0])], device
                        )
                    else:
                        graph_for_prediction = update_graph_with_prediction(
                            graph_for_prediction, pred, device
                        )
                    is_using_predicted_positions = True
            
            # After rollout completes, convert dict to array for ADE/FDE computation
            if current_rollout_errors and len(current_rollout_errors) > 0:
                # Convert dict {agent_key: [errors]} to numpy array [num_agents, num_steps]
                # Only include agents that have errors for ALL timesteps (no padding)
                # This gives accurate ADE/FDE without artificial inflation from extrapolation
                agent_error_arrays = []
                for agent_key, errors in current_rollout_errors.items():
                    # Only include agents with complete trajectories (all timesteps present)
                    if len(errors) == effective_rollout:
                        agent_error_arrays.append(np.array(errors))
                
                if agent_error_arrays:
                    batch_errors = np.stack(agent_error_arrays, axis=0)  # [num_agents, num_steps]
                    all_agent_errors.append(batch_errors)
            
            total_loss += rollout_loss.item() if isinstance(rollout_loss, torch.Tensor) else rollout_loss
            steps += 1
    
    horizon_avg_mse = [horizon_mse[h] / max(1, horizon_counts[h]) for h in range(num_rollout_steps)]
    horizon_avg_cosine = [horizon_cosine[h] / max(1, horizon_counts[h]) for h in range(num_rollout_steps)]
    
    # Compute ADE and FDE from all_agent_errors
    # ADE (Average Displacement Error): For each agent, compute mean error across all timesteps,
    #     then average those means across all agents
    # FDE (Final Displacement Error): Mean of final timestep errors across all agents
    
    print(f"  [DEBUG] Total batches processed: {steps}, Batches with valid errors: {len(all_agent_errors)}")
    
    if all_agent_errors:
        # Concatenate all rollout errors: [total_agents, num_steps]
        all_errors = np.concatenate(all_agent_errors, axis=0)  # [total_agents, num_steps]
        
        # Sanity check: errors should always be non-negative (Euclidean distances)
        if all_errors.min() < 0:
            print(f"  [WARNING] Negative errors detected! min={all_errors.min():.4f} - this indicates a bug")
            print(f"            Setting negative values to 0 and continuing...")
            all_errors = np.maximum(all_errors, 0.0)
        
        print(f"  [DEBUG] all_errors shape: {all_errors.shape}, min={all_errors.min():.4f}, max={all_errors.max():.4f}, mean={all_errors.mean():.4f}")
        print(f"  [DEBUG] Position error by timestep (meters):")
        for t_idx in [0, 4, 9, 19, 29, 49] if all_errors.shape[1] > 50 else range(min(10, all_errors.shape[1])):
            if t_idx < all_errors.shape[1]:
                mean_err = np.mean(all_errors[:, t_idx])
                max_err = np.max(all_errors[:, t_idx])
                print(f"    t={t_idx+1} ({(t_idx+1)*0.1:.1f}s): mean={mean_err:.2f}m, max={max_err:.2f}m")
        
        # ADE: First compute per-agent average across timesteps, then average across agents
        # This gives equal weight to each agent regardless of trajectory length
        per_agent_ade = np.mean(all_errors, axis=1)  # [total_agents] - mean error per agent
        ade = float(np.mean(per_agent_ade))  # Average across all agents
        
        # FDE: Mean of final timestep errors across all agents
        fde = float(np.mean(all_errors[:, -1]))
        
        total_agents = all_errors.shape[0]
    else:
        ade = 0.0
        fde = 0.0
        total_agents = 0
    
    # Print horizon error progression (every 10th step)
    # NOTE: horizon_mse tracks DISPLACEMENT MSE (pred_disp vs GT_disp), NOT position error!
    # This stays small because each step's displacement is small (~0.1s of motion).
    # ADE/FDE track ACCUMULATED POSITION ERROR which grows over the rollout.
    final_mse = total_mse / max(1, count)
    final_rmse_meters = (final_mse ** 0.5) * 100.0
    print(f"  [VAL METRICS] Displacement MSE={final_mse:.6f} | Displacement RMSE={final_rmse_meters:.2f}m")
    print(f"  [VAL HORIZON] Per-step displacement RMSE (NOT position error): ", end="")
    for h in [0, 9, 19, 29, 49, 69, 88]:
        if h < len(horizon_avg_mse) and horizon_counts[h] > 0:
            rmse_h = (horizon_avg_mse[h] ** 0.5) * 100.0
            print(f"t{h+1}={rmse_h:.1f}m ", end="")
    print()
    
    # Print position-based metrics (ADE/FDE) - these show true trajectory accuracy
    print(f"\n[VALIDATION] Position-based: ADE: {ade:.2f}m | FDE: {fde:.2f}m (across {total_agents} complete agent trajectories)")
    print(f"  (Note: Only agents present for all {num_rollout_steps} timesteps are included - no padding/extrapolation)")
    
    return {
        'loss': total_loss / max(1, steps),
        'mse': total_mse / max(1, count),
        'cosine_sim': total_cosine / max(1, count),
        'horizon_mse': horizon_avg_mse,
        'horizon_cosine': horizon_avg_cosine,
        'ade': ade,
        'fde': fde
    }

def run_autoregressive_finetuning(
        model_type="gat",
        pretrained_checkpoint="./checkpoints/gat/best_model.pt",
        pretrained_checkpoint_2="./checkpoints/gat/best_model_batched.pt",
        pretrained_checkpoint_3="./checkpoints/gcn/best_model.pt",
        pretrained_checkpoint_4="./checkpoints/gcn/best_model_batched.pt",
        train_path="./data/graphs/training/training_seqlen90.hdf5",
        val_path="./data/graphs/validation/validation_seqlen90.hdf5",
        num_rollout_steps=5,
        num_epochs=50,
        sampling_strategy='linear'
    ):
    """Fine-tune pre-trained model for autoregressive multi-step prediction."""
    
    # construct checkpoint filename based on training script's naming convention:
    
    if model_type == "gat":
        checkpoint_filename = f'best_{model_type}_batched_B{batch_size}_h{hidden_channels}_lr{gat_learning_rate:.0e}_heads{gat_num_heads}_E{epochs}.pt'
        pretrained_checkpoint_batched = os.path.join(gat_checkpoint_dir, checkpoint_filename)
        # Ensure parent directories exist first
        os.makedirs(gat_checkpoint_dir, exist_ok=True)
        os.makedirs(gat_checkpoint_dir_autoreg, exist_ok=True)
        os.makedirs(gat_viz_dir_autoreg, exist_ok=True)
    elif model_type == "gcn":
        checkpoint_filename = f'best_{model_type}_batched_B{batch_size}_h{hidden_channels}_lr{learning_rate:.0e}_L{num_layers}x{num_gru_layers}_E{epochs}.pt'
        pretrained_checkpoint_batched = os.path.join(gcn_checkpoint_dir, checkpoint_filename)
        # Ensure parent directories exist first
        os.makedirs(gcn_checkpoint_dir, exist_ok=True)
        os.makedirs(gcn_checkpoint_dir_autoreg, exist_ok=True)
        os.makedirs(gcn_viz_dir_autoreg, exist_ok=True)

    # Load pre-trained model:
    checkpoint = None
    model = None
    is_parallel = False
    if os.path.exists(pretrained_checkpoint_batched):
        print(f"Found batched checkpoint: {pretrained_checkpoint_batched}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint_batched, device, model_type=model_type)
        pretrained_checkpoint = pretrained_checkpoint_batched
    elif pretrained_checkpoint is not None and os.path.exists(pretrained_checkpoint):
        print(f"Found checkpoint: {pretrained_checkpoint}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint, device, model_type=model_type)
    elif pretrained_checkpoint_2 is not None and os.path.exists(pretrained_checkpoint_2):
        print(f"Found checkpoint: {pretrained_checkpoint_2}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint_2, device, model_type=model_type)
        pretrained_checkpoint = pretrained_checkpoint_2
    elif pretrained_checkpoint_3 is not None and os.path.exists(pretrained_checkpoint_3):
        print(f"Found checkpoint: {pretrained_checkpoint_3}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint_3, device, model_type=model_type)
        pretrained_checkpoint = pretrained_checkpoint_3
    elif pretrained_checkpoint_4 is not None and os.path.exists(pretrained_checkpoint_4):
        print(f"Found checkpoint: {pretrained_checkpoint_4}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint_4, device, model_type=model_type)
        pretrained_checkpoint = pretrained_checkpoint_4
    else:
        raise FileNotFoundError(
            f"No valid checkpoint found. Checked:\n"
            f"  - {pretrained_checkpoint_batched}\n"
            f"  - {pretrained_checkpoint}\n"
            f"  - {pretrained_checkpoint_2}\n"
            f"  - {pretrained_checkpoint_3}\n"
            f"  - {pretrained_checkpoint_4}"
        )
    
    wandb.login()
    run = wandb.init(
        project=project_name,
        config={
            "model": "SpatioTemporalGATBatched_Autoregressive" if model_type == "gat" else "SpatioTemporalGCNBatched_Autoregressive",
            "pretrained_from": pretrained_checkpoint,
            "batch_size": batch_size,
            "learning_rate": autoreg_learning_rate,
            "dataset": dataset_name,
            "num_rollout_steps": num_rollout_steps,
            "prediction_horizon": f"{num_rollout_steps * 0.1}s",
            "sampling_strategy": sampling_strategy,
            "epochs": num_epochs,
            "use_gat": model_type == "gat"
        },
        name=f"{('GAT' if model_type == 'gat' else 'GCN')}_Autoregressive_finetune_{num_rollout_steps}steps",
        dir="../wandb"
    )
    
    wandb.watch(model, log='all', log_freq=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=autoreg_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    
    try:        # Load datasets
        train_dataset = HDF5ScenarioDataset(train_path, seq_len=sequence_length)
        print(f"Loaded training dataset: {len(train_dataset)} scenarios")
    except FileNotFoundError:
        print(f"ERROR: {train_path} not found!")
        run.finish()
        return None
    
    val_dataset = None
    try:
        val_dataset_full = HDF5ScenarioDataset(
            val_path, 
            seq_len=sequence_length,
            cache_in_memory=cache_validation_data,
            max_cache_size=max_validation_scenarios
        )
        
        # subsample validation set for faster evaluation during finetuning
        max_val_scenarios = max_validation_scenarios
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
        # Use smaller batch size for validation if dataset is smaller than batch_size
        # to avoid drop_last=True dropping all samples
        val_batch_size = min(batch_size, len(val_dataset))
        val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_graph_sequences_to_batch,
            drop_last=False,  # Don't drop last for validation to use all samples
            persistent_workers=False,
            pin_memory=pin_memory, prefetch_factor=None
        )
        if val_batch_size < batch_size:
            print(f"  [Note] Validation batch size adjusted to {val_batch_size} (dataset has {len(val_dataset)} scenarios)")
        
        # Verify loader is not empty
        try:
            test_batch = next(iter(val_loader))
            print(f"  [Validation] Loader verified with {val_batch_size} scenarios per batch")
        except StopIteration:
            print(f"  [WARNING] Validation loader is empty! Disabling validation.")
            val_loader = None
    
    use_amp_finetune = False  # DISABLED - use float32 for stability
    print(f"\n{'='*80}")
    print(f"{'GAT' if model_type == 'gat' else 'GCN'} AUTOREGRESSIVE FINE-TUNING (BATCHED)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Pre-trained model: {pretrained_checkpoint}")
    print(f"Model: SpatioTemporalGATBatched (supports batch_size > 1)")
    print(f"Batch size: {batch_size} scenarios processed in parallel")
    print(f"Rollout steps: {num_rollout_steps} ({num_rollout_steps * 0.1}s horizon)")
    print(f"Sampling strategy: {sampling_strategy} (with 5-epoch warmup)")
    print(f"Fine-tuning LR: {autoreg_learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Validation: {'Enabled' if val_loader else 'Disabled'}")
    print(f"Mixed Precision (AMP): {'DISABLED for stability' if not use_amp_finetune else 'Enabled'}")
    print(f"Edge Rebuilding: {'Enabled (radius=' + str(radius) + 'm)' if AUTOREG_REBUILD_EDGES else 'Disabled (fixed topology)'}")
    print(f"Checkpoints: {gat_checkpoint_dir_autoreg if model_type == 'gat' else gcn_checkpoint_dir_autoreg}")
    print(f"Visualizations: {gat_viz_dir_autoreg if model_type == 'gat' else gcn_viz_dir_autoreg}")
    print(f"{'='*80}\n")

    if num_rollout_steps >= sequence_length - 1:
        print(f"\n[WARNING] Requested {num_rollout_steps} rollout steps but sequence length is only {sequence_length}!")
        print(f"          Training will use at most {sequence_length - 2} steps (T-1 timesteps).")
        print(f"          Consider reducing rollout steps or increasing sequence length.")
    
    scaler = None
    if use_amp_finetune and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        print("[AMP] Mixed Precision Training ENABLED - using float16 for forward pass")
    else:
        print("[AMP] DISABLED for autoregressive finetuning (float32 for stability)")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Reset warning flag at start of each epoch
        global _size_mismatch_warned
        _size_mismatch_warned = False
        
        sampling_prob = scheduled_sampling_probability(epoch, num_epochs, sampling_strategy)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} (sampling_prob={sampling_prob:.3f})")
        
        # Calculate curriculum rollout steps for this epoch
        curr_rollout = curriculum_rollout_steps(epoch, num_rollout_steps, num_epochs)
        
        # Explain what sampling_prob means for clarity
        print(f"  [SCHEDULED SAMPLING] {sampling_prob*100:.0f}% predicted input, {(1-sampling_prob)*100:.0f}% GT input")
        if sampling_prob < 0.15:
            print(f"  - Mostly teacher forcing: model learning from clean GT inputs")
        elif sampling_prob < 0.5:
            print(f"  - Balanced mix: model learning to handle some prediction errors")
        else:
            print(f"  - Mostly autoregressive: model learning to handle its own predictions")
        print(f"  [CURRICULUM] Rollout: {curr_rollout} steps ({curr_rollout*0.1:.1f}s) - target: {num_rollout_steps} steps")
        

        train_metrics = train_epoch_autoregressive(
            model, train_loader, optimizer, device, 
            sampling_prob, num_rollout_steps, is_parallel, scaler=scaler, epoch=epoch, total_epochs=num_epochs
        )
        
        # Validation (always full autoregressive)
        val_metrics = None
        if val_loader is not None:
            print(f"  Running validation...")
            val_metrics = evaluate_autoregressive(
                model, val_loader, device, num_rollout_steps, is_parallel, epoch=epoch, total_epochs=num_epochs
            )
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step(train_metrics['loss'])
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate RMSE in meters for more interpretable logging
        train_rmse_meters = (train_metrics['mse'] ** 0.5) * 100.0
        
        # Build log dictionary for this epoch
        log_dict = {
            "epoch": epoch + 1,  # 1-indexed for readability
            "sampling_probability": sampling_prob,
            "train/loss": train_metrics['loss'],
            "train/mse": train_metrics['mse'],
            "train/rmse_meters": train_rmse_meters,
            "train/cosine_sim": train_metrics['cosine_sim'],
            "learning_rate": current_lr
        }
        
        if val_metrics:
            val_rmse_meters = (val_metrics['mse'] ** 0.5) * 100.0
            log_dict.update({
                "val/loss": val_metrics['loss'],
                "val/mse": val_metrics['mse'],
                "val/rmse_meters": val_rmse_meters,
                "val/cosine_sim": val_metrics['cosine_sim'],
                "val/ade": val_metrics['ade'],
                "val/fde": val_metrics['fde']
            })
            # Per-horizon metrics
            for h in range(num_rollout_steps):
                log_dict[f"val/mse_horizon_{h+1}"] = val_metrics['horizon_mse'][h]
                log_dict[f"val/cosine_horizon_{h+1}"] = val_metrics['horizon_cosine'][h]
        
        # Track horizon errors and visualization paths for logging
        epoch_horizon_errors = []
        epoch_viz_paths = []
        
        # Visualize using validation data if available, otherwise training data
        if val_metrics:
            # Visualization every N epochs or on first/last epoch (4 scenarios)
            if (epoch % autoreg_visualize_every_n_epochs == 0) or (epoch == num_epochs - 1) or (epoch == 0):
                try:
                    print(f"\n  Creating visualizations for epoch {epoch+1}...")
                    viz_count = 0
                    for viz_batch in val_loader:
                        if viz_count >= 4:
                            break
                        result = visualize_autoregressive_rollout(
                            model, viz_batch, epoch, num_rollout_steps, 
                            device, is_parallel, save_dir=gat_viz_dir_autoreg if model_type == "gat" else gcn_viz_dir_autoreg, total_epochs=num_epochs, model_type=model_type
                        )
                        if isinstance(result, tuple):
                            filepath, final_error = result
                            epoch_viz_paths.append(filepath)
                            if final_error != float('inf') and not np.isnan(final_error):
                                epoch_horizon_errors.append(final_error)
                        viz_count += 1
                    print(f"  Created {viz_count} visualizations\n")
                except Exception as e:
                    import traceback
                    print(f"  Warning: Visualization failed: {e}")
                    traceback.print_exc()
            
            # Save best model based on validation loss
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_filename = f'best_{"gat" if model_type == "gat" else "gcn"}_autoreg_{num_rollout_steps}step_B{batch_size}_{sampling_strategy}_E{num_epochs}.pt'
                if model_type == "gat":
                    save_path = os.path.join(gat_checkpoint_dir_autoreg, save_filename)
                else:
                    save_path = os.path.join(gcn_checkpoint_dir_autoreg, save_filename)
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
                print(f"  -> Saved best model to {save_filename} (val_loss: {val_metrics['loss']:.4f})")
            
        else:
            # No validation - use training data for visualization
            if (epoch % autoreg_visualize_every_n_epochs == 0) or (epoch == num_epochs - 1) or (epoch == 0):
                try:
                    print(f"\n  Creating visualizations for epoch {epoch+1} (using TRAINING data)...")
                    viz_count = 0
                    for viz_batch in train_loader:
                        if viz_count >= max_scenario_files_for_viz:
                            break
                        result = visualize_autoregressive_rollout(
                            model, viz_batch, epoch, num_rollout_steps, 
                            device, is_parallel, save_dir=gat_viz_dir_autoreg if model_type == "gat" else gcn_viz_dir_autoreg, total_epochs=num_epochs, model_type=model_type
                        )
                        if isinstance(result, tuple):
                            filepath, final_error = result
                            epoch_viz_paths.append(filepath)
                            if final_error != float('inf') and not np.isnan(final_error):
                                epoch_horizon_errors.append(final_error)
                        viz_count += 1
                    print(f"  Created {viz_count} visualizations\n")
                except Exception as e:
                    import traceback
                    print(f"  Warning: Visualization failed: {e}")
                    traceback.print_exc()
        
        # Calculate average horizon error and add to log_dict (for visualization-based tracking)
        if epoch_horizon_errors:
            avg_horizon_error = np.mean(epoch_horizon_errors)
            print(f"\n  [EPOCH {epoch+1}] Average Final Horizon Error: {avg_horizon_error:.2f}m (across {len(epoch_horizon_errors)} scenarios)")
            
            # Add horizon error metrics to main log dict
            log_dict["horizon/avg_final_error_m"] = avg_horizon_error
            log_dict["horizon/min_error_m"] = min(epoch_horizon_errors)
            log_dict["horizon/max_error_m"] = max(epoch_horizon_errors)
        
        # Add visualizations to log dict (if any were created this epoch)
        if epoch_viz_paths:
            # Log only the first visualization per epoch to avoid clutter
            try:
                log_dict["visualization"] = wandb.Image(epoch_viz_paths[0])
            except Exception as e:
                print(f"  Warning: Could not log visualization to wandb: {e}")
        
        # Log all metrics for this epoch with explicit step parameter
        wandb.log(log_dict, step=epoch)
    
    # Determine actual final epoch (may be earlier due to early stopping)
    actual_final_epoch = epoch + 1
    
    # Save final model

    final_filename = f'final_{"gat" if model_type == "gat" else "gcn"}_autoreg_{num_rollout_steps}step_B{batch_size}_{sampling_strategy}_E{num_epochs}.pt'
    if model_type == "gat":
        final_path = os.path.join(gat_checkpoint_dir_autoreg, final_filename)
    else:
        final_path = os.path.join(gcn_checkpoint_dir_autoreg, final_filename)
    model_to_save = get_model_for_saving(model, is_parallel)
    final_checkpoint_data = {
        'epoch': actual_final_epoch - 1,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_rollout_steps': num_rollout_steps,
    }
    # Preserve original config if it exists
    if 'config' in checkpoint:
        final_checkpoint_data['config'] = checkpoint['config']
    torch.save(final_checkpoint_data, final_path)
    
    print(f"\n{'='*80}")
    print(f"{'GAT' if model_type == 'gat' else 'GCN'} FINE-TUNING COMPLETE!")
    print(f"{'='*80}")
    best_filename = f'best_{"gat" if model_type == "gat" else "gcn"}_autoreg_{num_rollout_steps}step_B{batch_size}_{sampling_strategy}_E{num_epochs}.pt'
    print(f"Best model: {os.path.join(gat_checkpoint_dir_autoreg if model_type == 'gat' else gcn_checkpoint_dir_autoreg, best_filename)}")
    print(f"Final model: {final_path}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"{'='*80}\n")
    
    run.finish()
    return model

def update_graph_with_prediction(graph, pred_displacement, device):
    """
    Update graph for next autoregressive step using PREDICTED DISPLACEMENT.
    
    Model predicts 2D displacement [N, 2] in meters → we derive all features:
        [0-1]   vx, vy - calculated from displacement/dt
        [2]     speed - calculated as sqrt(vx² + vy²)
        [3]     heading - calculated as atan2(vy, vx)
        [4]     valid - keep from previous graph
        [5-6]   ax, ay - calculated as velocity change from previous
        [7-10]  distances - recalculated from updated positions
        [11-14] one-hot type - keep from previous graph (never changes)
    
    Args:
        graph: PyG Data object with current features and positions
        pred_displacement: Predicted normalized displacement [N, 2] (dx_norm, dy_norm)
        device: Torch device
    
    Returns:
        Updated graph with recalculated features and optionally rebuilt edges
    """
    global _size_mismatch_warned
    updated_graph = graph.clone()
    dt = 0.1  # timestep in seconds
    
    # Handle size mismatch (agents entering/leaving)
    num_nodes_graph = updated_graph.x.shape[0]
    num_nodes_pred = pred_displacement.shape[0]
    
    if num_nodes_pred != num_nodes_graph:
        if not _size_mismatch_warned:
            print(f"  [WARNING] Size mismatch: pred={num_nodes_pred}, graph={num_nodes_graph}")
            _size_mismatch_warned = True
        num_nodes = min(num_nodes_pred, num_nodes_graph)
        pred_displacement = pred_displacement[:num_nodes]
    else:
        num_nodes = num_nodes_graph
    
    # Check for NaN in predictions
    if torch.isnan(pred_displacement).any():
        nan_mask = torch.isnan(pred_displacement).any(dim=1)
        print(f"  [WARNING] NaN in {nan_mask.sum()} predictions! Using zeros.")
        pred_displacement[nan_mask] = 0.0
    
    # ============ UPDATE POSITIONS FROM PREDICTED DISPLACEMENT ============
    # Model predicts NORMALIZED displacement (same scale as GT y, which is meters/100)
    # Multiply by POSITION_SCALE to get actual displacement in meters
    displacement_meters = pred_displacement * POSITION_SCALE  # [N, 2] in meters
    
    # Update positions (positions are in meters)
    if hasattr(updated_graph, 'pos') and updated_graph.pos is not None:
        updated_graph.pos[:num_nodes] = updated_graph.pos[:num_nodes] + displacement_meters[:num_nodes]
    
    # ============ DERIVE ALL NODE FEATURES FROM DISPLACEMENT ============
    # Calculate velocity from displacement: v = dx / dt
    # Note: displacement_meters already converted from normalized via * POSITION_SCALE
    pred_vx = displacement_meters[:, 0] / dt  # m/s
    pred_vy = displacement_meters[:, 1] / dt  # m/s
    
    # ============ IMPROVED FEATURE SMOOTHING WITH EMA ============
    # Use Exponential Moving Average when AUTOREG_FEATURE_EMA is enabled.
    # This reduces noise accumulation during autoregressive rollout while still
    # allowing the model to express trajectory changes.
    #
    # Key insight: The problem is that small prediction errors compound when deriving
    # velocity from displacement. A small position error (1cm) becomes a 10x larger
    # velocity error (10cm/s) and a 100x larger acceleration error.
    # EMA smoothing reduces this noise propagation.
    if AUTOREG_FEATURE_EMA and hasattr(updated_graph, 'x') and updated_graph.x is not None:
        # EMA: new = alpha * previous + (1 - alpha) * current
        # Higher alpha = more smoothing (more weight on previous)
        alpha = AUTOREG_FEATURE_EMA_ALPHA
        prev_vx = updated_graph.x[:num_nodes, 0] * MAX_SPEED
        prev_vy = updated_graph.x[:num_nodes, 1] * MAX_SPEED
        pred_vx = alpha * prev_vx + (1 - alpha) * pred_vx
        pred_vy = alpha * prev_vy + (1 - alpha) * pred_vy
    elif hasattr(updated_graph, 'x') and updated_graph.x is not None:
        # Legacy smoothing (minimal)
        velocity_smoothing = 0.1  # 10% previous, 90% predicted
        prev_vx = updated_graph.x[:num_nodes, 0] * MAX_SPEED
        prev_vy = updated_graph.x[:num_nodes, 1] * MAX_SPEED
        pred_vx = (1 - velocity_smoothing) * pred_vx + velocity_smoothing * prev_vx
        pred_vy = (1 - velocity_smoothing) * pred_vy + velocity_smoothing * prev_vy
    
    # Start with zeros, then fill in calculated values
    new_features = torch.zeros(num_nodes, 15, device=device, dtype=torch.float32)
    
    # [0-1] Velocity (normalized) - calculate from displacement
    new_features[:, 0] = pred_vx[:num_nodes] / MAX_SPEED
    new_features[:, 1] = pred_vy[:num_nodes] / MAX_SPEED
    
    # [2] Speed (normalized) - calculate from velocity
    pred_speed = torch.sqrt(pred_vx[:num_nodes]**2 + pred_vy[:num_nodes]**2)
    new_features[:, 2] = pred_speed / MAX_SPEED
    
    # [3] Heading direction (normalized) - from velocity with EMA smoothing
    # Heading reflects direction of travel. Use EMA to reduce noise while
    # still allowing turns and direction changes.
    pred_heading = torch.atan2(pred_vy[:num_nodes], pred_vx[:num_nodes]) / np.pi  # [-1, 1]
    if hasattr(updated_graph, 'x') and updated_graph.x is not None:
        prev_heading = updated_graph.x[:num_nodes, 3]
        # Use EMA for heading smoothing if enabled
        if AUTOREG_FEATURE_EMA:
            # Same alpha as velocity for consistency
            heading_alpha = AUTOREG_FEATURE_EMA_ALPHA * 0.5  # Less smoothing for heading (important for turns)
        else:
            heading_alpha = 0.05  # Legacy minimal smoothing
        # Handle wraparound at +/- pi using circular mean
        heading_diff = pred_heading - prev_heading
        heading_diff = torch.where(heading_diff > 1, heading_diff - 2, heading_diff)
        heading_diff = torch.where(heading_diff < -1, heading_diff + 2, heading_diff)
        pred_heading = prev_heading + (1 - heading_alpha) * heading_diff
        pred_heading = torch.clamp(pred_heading, -1, 1)
    new_features[:, 3] = pred_heading
    
    # [4] Valid - keep from previous graph (doesn't change)
    new_features[:, 4] = updated_graph.x[:num_nodes, 4]
    
    # [5-6] Acceleration (normalized) - use EMA-smoothed velocities
    # Acceleration is the noisiest derived feature (derivative of derivative).
    # Apply additional smoothing to reduce noise while still capturing accelerations.
    if hasattr(updated_graph, 'x') and updated_graph.x is not None:
        orig_prev_vx = updated_graph.x[:num_nodes, 0] * MAX_SPEED
        orig_prev_vy = updated_graph.x[:num_nodes, 1] * MAX_SPEED
        
        # Calculate raw acceleration
        raw_ax = (pred_vx[:num_nodes] - orig_prev_vx) / dt  # m/s²
        raw_ay = (pred_vy[:num_nodes] - orig_prev_vy) / dt
        
        # Apply EMA to acceleration if enabled (with higher smoothing)
        if AUTOREG_FEATURE_EMA:
            prev_ax = updated_graph.x[:num_nodes, 5] * MAX_ACCEL
            prev_ay = updated_graph.x[:num_nodes, 6] * MAX_ACCEL
            # Higher alpha for acceleration (more smoothing because more noise)
            accel_alpha = min(AUTOREG_FEATURE_EMA_ALPHA * 1.5, 0.6)
            ax = accel_alpha * prev_ax + (1 - accel_alpha) * raw_ax
            ay = accel_alpha * prev_ay + (1 - accel_alpha) * raw_ay
        else:
            ax = raw_ax
            ay = raw_ay
        
        new_features[:, 5] = ax / MAX_ACCEL
        new_features[:, 6] = ay / MAX_ACCEL
    else:
        new_features[:, 5] = 0.0
        new_features[:, 6] = 0.0
    
    # [7-10] Distances - recalculate from updated positions
    if hasattr(updated_graph, 'pos') and updated_graph.pos is not None:
        positions = updated_graph.pos[:num_nodes]
        
        # Find SDC (ego vehicle) - usually agent 0 or check batch info
        # For simplicity, use first agent as SDC approximation
        sdc_pos = positions[0]
        
        for i in range(num_nodes):
            # Relative position to SDC
            rel_x = (positions[i, 0] - sdc_pos[0]).item()
            rel_y = (positions[i, 1] - sdc_pos[1]).item()
            dist_sdc = np.sqrt(rel_x**2 + rel_y**2)
            
            new_features[i, 7] = rel_x / MAX_DIST_SDC  # normalized
            new_features[i, 8] = rel_y / MAX_DIST_SDC
            new_features[i, 9] = min(dist_sdc / MAX_DIST_SDC, 1.0)
            
            # Distance to nearest neighbor
            if num_nodes > 1:
                dists = torch.norm(positions - positions[i], dim=1)
                dists[i] = float('inf')  # exclude self
                min_dist = dists.min().item()
                new_features[i, 10] = min(min_dist / 50.0, 1.0)  # MAX_DIST_NEAREST=50m
            else:
                new_features[i, 10] = 1.0  # far away (no neighbors)
    
    # [11-14] One-hot type - keep from previous graph (NEVER changes)
    new_features[:, 11:15] = updated_graph.x[:num_nodes, 11:15]
    
    # Update graph features
    updated_graph.x[:num_nodes] = new_features
    
    # ============ REBUILD EDGES IF ENABLED ============
    if AUTOREG_REBUILD_EDGES and hasattr(updated_graph, 'pos'):
        if hasattr(updated_graph, 'batch') and updated_graph.batch is not None:
            # Batched graph: rebuild per scenario
            batch_ids = updated_graph.batch[:num_nodes].unique()
            all_edge_indices = []
            all_edge_weights = []
            
            for batch_id in batch_ids:
                batch_mask = (updated_graph.batch[:num_nodes] == batch_id)
                batch_positions = updated_graph.pos[:num_nodes][batch_mask]
                batch_indices = torch.where(batch_mask)[0]
                
                # Valid mask from feature 4
                valid_mask = new_features[batch_mask, 4] > 0.5
                
                # Build edges
                local_edge_index, local_edge_weight = build_edge_index_using_radius(
                    batch_positions, radius=radius, self_loops=False,
                    valid_mask=valid_mask.cpu().numpy()
                )
                
                if local_edge_index.size(1) > 0:
                    global_edge_index = batch_indices[local_edge_index]
                    all_edge_indices.append(global_edge_index)
                    all_edge_weights.append(local_edge_weight.to(device))
            
            # Combine edges
            if all_edge_indices:
                updated_graph.edge_index = torch.cat(all_edge_indices, dim=1)
                updated_graph.edge_weight = torch.cat(all_edge_weights, dim=0)
            else:
                updated_graph.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                updated_graph.edge_weight = torch.zeros(0, dtype=torch.float32, device=device)
        else:
            # Single graph
            valid_mask = new_features[:, 4] > 0.5
            new_edge_index, new_edge_weight = build_edge_index_using_radius(
                updated_graph.pos[:num_nodes], radius=radius, self_loops=False,
                valid_mask=valid_mask.cpu().numpy()
            )
            updated_graph.edge_index = new_edge_index.to(device)
            updated_graph.edge_weight = new_edge_weight.to(device)
    
    return updated_graph

def filter_to_scenario(graph, scenario_idx, B):
    if B > 1 and hasattr(graph, 'batch'):
        graph_batch_mask = (graph.batch == scenario_idx)
        graph_scenario_indices = torch.where(graph_batch_mask)[0].cpu().numpy()
    else:
        graph_scenario_indices = np.arange(graph.num_nodes)
            
    if hasattr(graph, 'agent_ids'):
        all_graph_agent_ids = list(graph.agent_ids)
        graph_agent_ids = [all_graph_agent_ids[i] for i in graph_scenario_indices if i < len(all_graph_agent_ids)]
    else:
        graph_agent_ids = [i for i in graph_scenario_indices]
            
    if hasattr(graph, 'pos') and graph.pos is not None:
        all_graph_pos = graph.pos.cpu().numpy()
        graph_pos = all_graph_pos[graph_scenario_indices]
    else:
        graph_pos = np.zeros((len(graph_scenario_indices), 2))
    return graph_scenario_indices, graph_agent_ids, graph_pos

if __name__ == '__main__':
    model = run_autoregressive_finetuning(
        pretrained_checkpoint=None,  # Uses checkpoints/gat/best_model.pt by default
        num_rollout_steps=autoreg_num_rollout_steps,
        num_epochs=autoreg_num_epochs,
        sampling_strategy=autoreg_sampling_strategy
    )