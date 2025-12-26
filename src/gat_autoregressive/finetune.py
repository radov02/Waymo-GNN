"""Fine-tune GAT single-step model for autoregressive prediction using scheduled sampling."""

import sys
import os
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
from .SpatioTemporalGAT_batched import SpatioTemporalGATBatched
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, gat_num_workers, num_layers, num_gru_layers,
                    input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name, epochs,
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
                          'spatial_layers.0.att_src', 'module.spatial_layers.0.att_src']
        
        checkpoint_hidden_dim = None
        print(f"  First 10 keys in checkpoint state_dict:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            if i < 5 or 'spatial' in key:
                print(f"    {key}: {state_dict[key].shape}")
        
        for key in gat_layer_keys:
            if key in state_dict:
                checkpoint_hidden_dim = state_dict[key].shape[0] if len(state_dict[key].shape) > 1 else state_dict[key].shape[0]
                print(f"  ✓ Inferred GAT hidden_dim={checkpoint_hidden_dim} from key: {key}")
                break
        
        if checkpoint_hidden_dim is None:
            checkpoint_hidden_dim = hidden_channels
            print(f"  ✗ Could not infer GAT hidden_dim, using config.py default: {checkpoint_hidden_dim}")
        
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
        gcn_layer_keys = ['spatial_layers.0.lin.weight', 'module.spatial_layers.0.lin.weight']
        
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
            use_gat_from_checkpoint = config.get('use_gat', False)
            print(f"  >>> CREATING GCN MODEL WITH hidden_dim={final_hidden_dim}, use_gat={use_gat_from_checkpoint} <<<")
            
            model = SpatioTemporalGNNBatched(
                input_dim=config.get('input_dim', input_dim),
                hidden_dim=final_hidden_dim,
                output_dim=config.get('output_dim', output_dim),
                num_gcn_layers=config.get('num_layers', num_layers),
                num_gru_layers=config.get('num_gru_layers', num_gru_layers),
                dropout=config.get('dropout', dropout),
                use_gat=use_gat_from_checkpoint,
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

def scheduled_sampling_probability(epoch, total_epochs, strategy='linear', warmup_epochs=5):
    """Calculate probability of using model's own prediction vs ground truth.
    
    Ensures sampling_prob reaches exactly 1.0 at the final epoch (epoch == total_epochs - 1).
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of training epochs
        strategy: 'linear', 'exponential', 'inverse_sigmoid', or 'delayed_linear'
        warmup_epochs: Number of epochs before increasing scheduled sampling
    
    Returns:
        Probability of using model's own prediction (0 = teacher forcing, 1 = autoregressive)
    """
    # Guarantee final epoch is 1.0 (100% autoregressive)
    if epoch >= total_epochs - 1:
        return 1.0
    
    # TODO: make scheduled sampling linearly increase from 0.0 to 1.0 over total_epochs...
    
    # During warmup, use small amount of model predictions to start learning
    if epoch < warmup_epochs:
        # Linear ramp from 0.0 to 0.1 during warmup for smoother transition
        return 0.1 * (epoch / max(1, warmup_epochs))
    
    # Calculate progress from warmup_epochs to total_epochs-1
    # At epoch=warmup_epochs: progress=0, at epoch=total_epochs-1: progress=1
    remaining_epochs = total_epochs - warmup_epochs
    adjusted_epoch = epoch - warmup_epochs
    progress = adjusted_epoch / max(1, remaining_epochs - 1)
    progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
    
    # Scale from 0.1 (end of warmup) to 1.0 (final epoch)
    min_sampling = 0.1
    max_sampling = 1.0
    
    if strategy == 'linear' or strategy == 'delayed_linear':
        return min_sampling + progress * (max_sampling - min_sampling)
    elif strategy == 'exponential':
        return min_sampling + (1 - np.exp(-5 * progress)) * (max_sampling - min_sampling)
    elif strategy == 'inverse_sigmoid':
        k = 10
        return min_sampling + (1 / (1 + np.exp(-k * (progress - 0.5)))) * (max_sampling - min_sampling)
    else:
        return min_sampling + progress * (max_sampling - min_sampling)

def curriculum_rollout_steps(epoch, max_rollout_steps, total_epochs):
    """Curriculum learning: gradually increase rollout length.
    
    Start with short rollouts (easier) and increase over training.
    This helps the model learn to handle error accumulation gradually.
    
    Args:
        epoch: Current epoch (0-indexed)
        max_rollout_steps: Maximum rollout steps (e.g., 89)
        total_epochs: Total training epochs
    
    Returns:
        Number of rollout steps to use this epoch
    """
    # Start with 10 steps, linearly increase to max over training
    min_steps = 10
    progress = epoch / max(1, total_epochs - 1)
    steps = int(min_steps + (max_rollout_steps - min_steps) * progress)
    return min(steps, max_rollout_steps)

def visualize_autoregressive_rollout(model, batch_dict, epoch, num_rollout_steps, device, 
                                     is_parallel, save_dir=None, total_epochs=40, model_type="gat"):
    """Visualize autoregressive rollout predictions vs ground truth.
    
    Args:
        model: The GAT model
        batch_dict: Batch dictionary with graph sequences
        epoch: Current epoch (0-indexed)
        num_rollout_steps: Number of autoregressive steps to roll out
        device: Torch device
        is_parallel: Whether model is DataParallel wrapped
        save_dir: Directory to save visualizations
        total_epochs: Total number of epochs (for computing velocity smoothing)
    """
    if save_dir is None:
        if model_type == "gat":
            save_dir = gat_viz_dir_autoreg
        elif model_type == "gcn":
            save_dir = gcn_viz_dir_autoreg
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # TODO (w.r.t. update_graph_with_prediction)
    # NEW APPROACH: Keep GT velocity features, only update position
    # This matches training and gives model correct kinematic features
    velocity_smoothing = 0.0  # Not used
    update_velocity = False   # Keep GT features
    print(f"  [VIZ] Using GT velocity features for visualization (epoch {epoch+1}/{total_epochs})")
    
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
                            disp = target_graph.y[global_idx].cpu().numpy() * POSITION_SCALE
                            pos = last_pos + disp
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
            
            # GAT forward pass with agent_ids for per-agent GRU tracking
            agent_ids_for_model = getattr(graph_for_prediction, 'agent_ids', None)
            pred = model(graph_for_prediction.x, graph_for_prediction.edge_index,
                        batch=graph_for_prediction.batch, batch_size=B, batch_num=0, timestep=start_t + step,
                        agent_ids=agent_ids_for_model)
            
            # Extract 2D displacement from predicted 15-dim feature vectors
            # Features [0-1] are normalized velocities vx_norm, vy_norm
            # Integrate: displacement = velocity * dt
            dt = 0.1  # timestep in seconds
            pred_vx_norm = pred[:, 0]  # normalized velocity x
            pred_vy_norm = pred[:, 1]  # normalized velocity y
            pred_vx = pred_vx_norm * MAX_SPEED  # denormalize: m/s
            pred_vy = pred_vy_norm * MAX_SPEED
            pred_dx = pred_vx * dt  # displacement in meters
            pred_dy = pred_vy * dt
            pred_disp = torch.stack([pred_dx, pred_dy], dim=1).cpu().numpy()  # [N, 2] in meters
            
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
                            gt_disp = source_graph.y[first_agent_idx].cpu().numpy() * POSITION_SCALE
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
                    # Extract 2D displacement from predicted 15-dim feature vectors
                    dt = 0.1
                    pred_vx = (pred[:, 0] * MAX_SPEED).cpu()  # denormalize velocity
                    pred_vy = (pred[:, 1] * MAX_SPEED).cpu()
                    pred_dx = pred_vx * dt  # displacement in meters
                    pred_dy = pred_vy * dt
                    pred_displacement = torch.stack([pred_dx, pred_dy], dim=1)  # [N, 2] in meters
                    
                    # CRITICAL: Only update positions for agents that exist in BOTH graphs
                    if (hasattr(current_gt_graph, 'agent_ids') and hasattr(gt_graph_next, 'agent_ids') and
                        hasattr(current_gt_graph, 'batch') and hasattr(gt_graph_next, 'batch')):
                        
                        current_batch_np = current_gt_graph.batch.cpu().numpy()
                        next_batch_np = gt_graph_next.batch.cpu().numpy()
                        
                        current_id_to_idx = {}
                        for idx in range(len(current_gt_graph.agent_ids)):
                            if idx < len(current_batch_np):
                                bid = int(current_batch_np[idx])
                                aid = current_gt_graph.agent_ids[idx]
                                current_id_to_idx[(bid, aid)] = idx
                        
                        next_id_to_idx = {}
                        for idx in range(len(gt_graph_next.agent_ids)):
                            if idx < len(next_batch_np):
                                bid = int(next_batch_np[idx])
                                aid = gt_graph_next.agent_ids[idx]
                                next_id_to_idx[(bid, aid)] = idx
                        
                        common_ids = set(current_id_to_idx.keys()) & set(next_id_to_idx.keys())
                        
                        for gid in common_ids:
                            current_idx = current_id_to_idx[gid]
                            next_idx = next_id_to_idx[gid]
                            if current_idx < pred_displacement.shape[0] and current_idx < current_gt_graph.pos.shape[0]:
                                graph_for_prediction.pos[next_idx] = current_gt_graph.pos[current_idx] + pred_displacement[current_idx]
                    else:
                        # Fallback: only update if sizes match
                        if current_gt_graph.pos.shape[0] == pred_displacement.shape[0] == graph_for_prediction.pos.shape[0]:
                            graph_for_prediction.pos = current_gt_graph.pos + pred_displacement
            else:
                # No more GT graphs available, use update function as fallback
                graph_for_prediction = update_graph_with_prediction(graph_for_prediction, pred, device, 
                                                                      velocity_smoothing=velocity_smoothing,
                                                                      update_velocity=update_velocity)
    
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
    steps = 0
    count = 0

    # TODO: velocity smoothing or only position update?    

    
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
                
                # Build mapping: (batch_idx, agent_id) -> node_index
                pred_id_to_idx = {}
                for idx in range(pred_num_nodes):
                    bid = int(pred_batch_np[idx])
                    aid = pred_agent_ids[idx]
                    pred_id_to_idx[(bid, aid)] = idx
                
                target_id_to_idx = {}
                for idx in range(target_num_nodes):
                    bid = int(target_batch_np[idx])
                    aid = target_agent_ids[idx]
                    target_id_to_idx[(bid, aid)] = idx
                
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
            
            # GAT forward pass with optional AMP and agent_ids for per-agent GRU tracking
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
                            # Build mapping for source graph
                            source_batch_np = source_batch.cpu().numpy()
                            source_id_to_idx = {}
                            for idx in range(len(source_agent_ids)):
                                bid = int(source_batch_np[idx])
                                aid = source_agent_ids[idx]
                                source_id_to_idx[(bid, aid)] = idx
                            
                            # Get GT displacement for each aligned agent
                            target_list = []
                            for i, idx in enumerate(pred_indices.cpu().numpy()):
                                # Get the agent ID from pred graph
                                if idx < len(pred_agent_ids):
                                    aid = pred_agent_ids[idx]
                                    bid = int(pred_batch.cpu().numpy()[idx])
                                    gid = (bid, aid)
                                    
                                    if gid in source_id_to_idx:
                                        source_idx = source_id_to_idx[gid]
                                        target_list.append(source_gt_graph.y[source_idx])
                                    else:
                                        # Agent not in source - shouldn't happen, use zeros
                                        target_list.append(torch.zeros(2, device=device))
                                else:
                                    target_list.append(torch.zeros(2, device=device))
                            
                            target = torch.stack(target_list).to(pred.dtype)
                        else:
                            # Fallback: use target_graph.y directly (assumes alignment)
                            target = source_gt_graph.y[target_indices].to(pred.dtype)
                    else:
                        # No GT displacement available - compute from positions
                        gt_current_pos = source_gt_graph.pos[target_indices].to(pred.dtype)
                        gt_next_pos = target_graph.pos[target_indices].to(pred.dtype)
                        target = (gt_next_pos - gt_current_pos) / POSITION_SCALE
                else:
                    pred_aligned = pred
                    # Same: always use GT displacement
                    source_gt_graph = batched_graph_sequence[t + step]
                    if source_gt_graph.y is not None:
                        target = source_gt_graph.y.to(pred.dtype)
                    else:
                        # Compute from GT positions
                        gt_current_pos = source_gt_graph.pos.to(pred.dtype)
                        gt_next_pos = target_graph.pos.to(pred.dtype)
                        target = (gt_next_pos - gt_current_pos) / POSITION_SCALE
                
                # SIMPLIFIED LOSS: Model predicts 2D velocity (vx_norm, vy_norm)
                # Convert velocity to displacement for loss computation
                dt = 0.1
                
                # Extract predicted velocity (model outputs 2D)
                pred_vx_norm = pred_aligned[:, 0]
                pred_vy_norm = pred_aligned[:, 1]
                
                # Convert velocity to displacement
                pred_vx = pred_vx_norm * MAX_SPEED  # denormalize to m/s
                pred_vy = pred_vy_norm * MAX_SPEED
                pred_dx_norm = (pred_vx * dt) / POSITION_SCALE
                pred_dy_norm = (pred_vy * dt) / POSITION_SCALE
                pred_disp_for_loss = torch.stack([pred_dx_norm, pred_dy_norm], dim=1)
                
                # Check for NaN in predictions or targets before computing loss
                if torch.isnan(pred_disp_for_loss).any() or torch.isnan(target).any():
                    print(f"  [ERROR] NaN detected in predictions or targets at step {step}! Skipping this step.")
                    continue
                
                # Displacement loss components
                mse_loss = F.mse_loss(pred_disp_for_loss, target)
                huber_loss = F.smooth_l1_loss(pred_disp_for_loss, target)
                pred_norm = F.normalize(pred_disp_for_loss, p=2, dim=1, eps=1e-6)
                target_norm = F.normalize(target, p=2, dim=1, eps=1e-6)
                cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
                cosine_loss = (1 - cos_sim).mean()
                pred_magnitude = torch.norm(pred_disp_for_loss, dim=1)
                target_magnitude = torch.norm(target, dim=1)
                magnitude_loss = F.mse_loss(pred_magnitude, target_magnitude)
                
                # Combined displacement loss
                disp_loss = 0.4 * huber_loss + 0.2 * mse_loss + 0.3 * cosine_loss + 0.1 * magnitude_loss
                
                # Final NaN check
                if torch.isnan(disp_loss):
                    print(f"  [ERROR] NaN in loss computation at step {step}! Skipping.")
                    continue
                
                # Temporal discount for autoregressive stability
                discount = 0.97 ** step
                step_loss = disp_loss * discount
            
            if rollout_loss is None:
                rollout_loss = step_loss
            else:
                rollout_loss = rollout_loss + step_loss
            
            with torch.no_grad():
                total_mse += mse_loss.item()
                total_cosine += cos_sim.mean().item()
                count += 1
            
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
    
    print(f"\n[TRAIN EPOCH SUMMARY]")
    print(f"  Loss: {final_loss:.6f} | Avg per-step MSE: {final_mse:.6f} | Avg per-step RMSE: {final_rmse:.2f}m | CosSim: {final_cosine:.4f}")
    print(f"  Training uses sampling_prob={sampling_prob:.2f} (0=teacher forcing, 1=autoregressive)")
    
    # NOTE: No wandb.log here - logging is done per-epoch in main training loop
    # This avoids duplicate logging (per-step vs per-epoch)
    
    return {
        'loss': final_loss,
        'mse': final_mse,
        'cosine_sim': final_cosine
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
            
            for step in range(effective_rollout):
                target_t = start_t + step + 1
                if target_t >= T:
                    break
                
                target_graph = batched_graph_sequence[target_t]
                if target_graph.y is None:
                    break
                
                if graph_for_prediction.x.shape[0] != target_graph.y.shape[0]:
                    graph_for_prediction = target_graph
                    is_using_predicted_positions = False
                    continue
                
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
                
                # CRITICAL: Compute correct target based on current position
                if is_using_predicted_positions and hasattr(graph_for_prediction, 'pos') and graph_for_prediction.pos is not None:
                    # When using predicted positions, target is displacement from current predicted pos to GT next pos
                    gt_next_pos = target_graph.pos.to(pred.dtype)
                    current_pred_pos = graph_for_prediction.pos.to(pred.dtype)
                    target = (gt_next_pos - current_pred_pos) / POSITION_SCALE
                else:
                    # First step uses GT position
                    target = target_graph.y.to(pred.dtype)
                
                # Model outputs 2D velocity - convert to displacement for loss
                dt = 0.1
                pred_vx_norm = pred[:, 0]
                pred_vy_norm = pred[:, 1]
                pred_vx = pred_vx_norm * MAX_SPEED
                pred_vy = pred_vy_norm * MAX_SPEED
                pred_dx_norm = (pred_vx * dt) / POSITION_SCALE
                pred_dy_norm = (pred_vy * dt) / POSITION_SCALE
                pred_disp_for_loss = torch.stack([pred_dx_norm, pred_dy_norm], dim=1)
                
                # Simplified loss - just displacement
                mse = F.mse_loss(pred_disp_for_loss, target)
                
                # Track per-agent displacement error at each step for ADE/FDE
                # Error in meters: ||pred_disp - target_disp|| * POSITION_SCALE
                step_errors_meters = torch.norm(pred_disp_for_loss - target, dim=1) * POSITION_SCALE  # [num_agents]
                
                # Initialize or accumulate per-agent errors for this rollout
                if step == 0:
                    # Start new rollout tracking: [num_agents, num_steps]
                    current_rollout_errors = torch.zeros(pred.shape[0], effective_rollout, device=device)
                
                # Store errors for this step
                if step < effective_rollout and step_errors_meters.shape[0] == current_rollout_errors.shape[0]:
                    current_rollout_errors[:, step] = step_errors_meters
                
                # At final step, save rollout errors
                if step == effective_rollout - 1:
                    all_agent_errors.append(current_rollout_errors.cpu().numpy())
                
                # Debug: print position drift for first batch
                if batch_idx == 0 and not debug_printed:
                    pos_drift = 0.0
                    if hasattr(graph_for_prediction, 'pos') and hasattr(target_graph, 'pos'):
                        pos_drift = torch.norm(graph_for_prediction.pos - target_graph.pos, dim=1).mean().item()
                    if step in [0, 4, 9, 19, 49, 88]:
                        print(f"    [VAL DEBUG] Step {step}: mse={mse.item():.4f}, pos_drift={pos_drift:.1f}m, auto={is_using_predicted_positions}")
                    if step == 88:
                        debug_printed = True
                
                pred_norm = F.normalize(pred_disp_for_loss, p=2, dim=1, eps=1e-6)
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
                    # ALWAYS use predictions in evaluation (no teacher forcing)
                    graph_for_prediction = update_graph_with_prediction(
                        graph_for_prediction, pred, device
                    )
                    is_using_predicted_positions = True
            
            # After rollout completes
            total_loss += rollout_loss.item() if isinstance(rollout_loss, torch.Tensor) else rollout_loss
            steps += 1
    
    horizon_avg_mse = [horizon_mse[h] / max(1, horizon_counts[h]) for h in range(num_rollout_steps)]
    horizon_avg_cosine = [horizon_cosine[h] / max(1, horizon_counts[h]) for h in range(num_rollout_steps)]
    
    # Compute ADE and FDE from all_agent_errors
    # ADE (Average Displacement Error): For each agent, compute mean error across all timesteps,
    #     then average those means across all agents
    # FDE (Final Displacement Error): Mean of final timestep errors across all agents
    if all_agent_errors:
        # Concatenate all rollout errors: [total_agents, num_steps]
        all_errors = np.concatenate(all_agent_errors, axis=0)  # [total_agents, num_steps]
        
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
    final_mse = total_mse / max(1, count)
    final_rmse_meters = (final_mse ** 0.5) * 100.0
    print(f"  [VAL METRICS] Per-step MSE={final_mse:.6f} | Per-step RMSE={final_rmse_meters:.2f}m")
    print(f"  [VAL HORIZON] Step-wise RMSE (meters): ", end="")
    for h in [0, 9, 19, 29, 49, 69, 88]:
        if h < len(horizon_avg_mse) and horizon_counts[h] > 0:
            rmse_h = (horizon_avg_mse[h] ** 0.5) * 100.0
            print(f"t{h+1}={rmse_h:.1f}m ", end="")
    print()
    
    # Print key metrics (ADE/FDE like VectorNet)
    print(f"\n[VALIDATION] ADE: {ade:.2f}m | FDE: {fde:.2f}m (across {total_agents} agents)")
    
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
        checkpoint_filename = f'best_{model_type}_batched_B{batch_size}_h{hidden_channels}_lr{learning_rate:.0e}_heads{gat_num_heads}_E{epochs}.pt'
        pretrained_checkpoint_batched = os.path.join(gat_checkpoint_dir, checkpoint_filename)
        os.makedirs(gat_checkpoint_dir_autoreg, exist_ok=True)
        os.makedirs(gat_viz_dir_autoreg, exist_ok=True)
    elif model_type == "gcn":
        checkpoint_filename = f'best_{model_type}_batched_B{batch_size}_h{hidden_channels}_lr{learning_rate:.0e}_L{num_layers}x{num_gru_layers}_E{epochs}.pt'
        pretrained_checkpoint_batched = os.path.join(gcn_checkpoint_dir, checkpoint_filename)
        os.makedirs(gcn_checkpoint_dir_autoreg, exist_ok=True)
        os.makedirs(gcn_viz_dir_autoreg, exist_ok=True)

    # Load pre-trained model:
    checkpoint = None
    model = None
    is_parallel = False
    if os.path.exists(pretrained_checkpoint_batched):
        print(f"Found batched checkpoint: {pretrained_checkpoint_batched}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint_batched, device)
        pretrained_checkpoint = pretrained_checkpoint_batched
    elif pretrained_checkpoint is not None and os.path.exists(pretrained_checkpoint):
        print(f"Found checkpoint: {pretrained_checkpoint}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint, device)
    elif pretrained_checkpoint_2 is not None and os.path.exists(pretrained_checkpoint_2):
        print(f"Found checkpoint: {pretrained_checkpoint_2}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint_2, device)
        pretrained_checkpoint = pretrained_checkpoint_2
    elif pretrained_checkpoint_3 is not None and os.path.exists(pretrained_checkpoint_3):
        print(f"Found checkpoint: {pretrained_checkpoint_3}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint_3, device)
        pretrained_checkpoint = pretrained_checkpoint_3
    elif pretrained_checkpoint_4 is not None and os.path.exists(pretrained_checkpoint_4):
        print(f"Found checkpoint: {pretrained_checkpoint_4}")
        model, is_parallel, checkpoint = load_pretrained_model(pretrained_checkpoint_4, device)
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
            "learning_rate": learning_rate * 0.1,
            "dataset": dataset_name,
            "num_rollout_steps": num_rollout_steps,
            "prediction_horizon": f"{num_rollout_steps * 0.1}s",
            "sampling_strategy": sampling_strategy,
            "epochs": num_epochs,
            "use_gat": model_type == "gat"
        },
        name=f"GAT_Autoregressive_finetune_{num_rollout_steps}steps" if model_type == "gat" else f"GCN_Autoregressive_finetune_{num_rollout_steps}steps",
        dir="../wandb"
    )
    
    wandb.watch(model, log='all', log_freq=10)
    
    finetune_lr = learning_rate * 0.05      # make lower to help with stability (prevent NaN gradients during scheduled sampling)
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
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
    print(f"GAT AUTOREGRESSIVE FINE-TUNING (BATCHED)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Pre-trained model: {pretrained_checkpoint}")
    print(f"Model: SpatioTemporalGATBatched (supports batch_size > 1)")
    print(f"Batch size: {batch_size} scenarios processed in parallel")
    print(f"Rollout steps: {num_rollout_steps} ({num_rollout_steps * 0.1}s horizon)")
    print(f"Sampling strategy: {sampling_strategy} (with 5-epoch warmup)")
    print(f"Fine-tuning LR: {finetune_lr}")
    print(f"Epochs: {num_epochs}")
    print(f"Validation: {'Enabled' if val_loader else 'Disabled'}")
    print(f"Mixed Precision (AMP): {'DISABLED for stability' if not use_amp_finetune else 'Enabled'}")
    print(f"Edge Rebuilding: {'Enabled (radius=' + str(radius) + 'm)' if AUTOREG_REBUILD_EDGES else 'Disabled (fixed topology)'}")
    print(f"Checkpoints: {gat_checkpoint_dir_autoreg}")
    print(f"Visualizations: {gat_viz_dir_autoreg}")
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
        if sampling_prob < 0.01:
            print(f"  [TEACHER FORCING] Using GT graphs as input for each step.")
            print(f"  - Model learns single-step prediction with perfect input.")
            print(f"  - Visualization (autoregressive) tests multi-step performance.")
        elif sampling_prob > 0.49:
            print(f"  [MAX SCHEDULED SAMPLING] {sampling_prob*100:.0f}% predicted input (capped at 50%).")
        else:
            print(f"  [SCHEDULED SAMPLING] {sampling_prob*100:.0f}% predicted, {(1-sampling_prob)*100:.0f}% GT input.")
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
                            device, is_parallel, save_dir=gat_viz_dir_autoreg, total_epochs=num_epochs
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
                save_filename = f'best_gat_autoreg_{num_rollout_steps}step_B{batch_size}_{sampling_strategy}_E{num_epochs}.pt'
                save_path = os.path.join(gat_checkpoint_dir_autoreg, save_filename)
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
                            device, is_parallel, save_dir=gat_viz_dir_autoreg, total_epochs=num_epochs
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
    final_filename = f'final_gat_autoreg_{num_rollout_steps}step_B{batch_size}_{sampling_strategy}_E{num_epochs}.pt'
    final_path = os.path.join(gat_checkpoint_dir_autoreg, final_filename)
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
    print(f"GAT FINE-TUNING COMPLETE!")
    print(f"{'='*80}")
    print(f"Epochs completed: {actual_final_epoch}/{num_epochs}" + (" (early stopped)" if early_stopped else ""))
    best_filename = f'best_gat_autoreg_{num_rollout_steps}step_B{batch_size}_{sampling_strategy}_E{num_epochs}.pt'
    print(f"Best model: {os.path.join(gat_checkpoint_dir_autoreg, best_filename)}")
    print(f"Final model: {final_path}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"{'='*80}\n")
    
    run.finish()
    return model

def update_graph_with_prediction(graph, pred_velocity, device):
    """
    Update graph for next autoregressive step using PREDICTED VELOCITY.
    
    Model predicts 2D velocity [N, 2] → we derive all other features:
        [0-1]   vx, vy - from prediction
        [2]     speed - calculated as sqrt(vx² + vy²)
        [3]     heading - calculated as atan2(vy, vx)
        [4]     valid - keep from previous graph
        [5-6]   ax, ay - calculated as velocity change from previous
        [7-10]  distances - recalculated from updated positions
        [11-14] one-hot type - keep from previous graph (never changes)
    
    Args:
        graph: PyG Data object with current features and positions
        pred_velocity: Predicted velocity [N, 2] (vx_norm, vy_norm)
        device: Torch device
    
    Returns:
        Updated graph with recalculated features and optionally rebuilt edges
    """
    global _size_mismatch_warned
    updated_graph = graph.clone()
    dt = 0.1  # timestep in seconds
    
    # Handle size mismatch (agents entering/leaving)
    num_nodes_graph = updated_graph.x.shape[0]
    num_nodes_pred = pred_velocity.shape[0]
    
    if num_nodes_pred != num_nodes_graph:
        if not _size_mismatch_warned:
            print(f"  [WARNING] Size mismatch: pred={num_nodes_pred}, graph={num_nodes_graph}")
            _size_mismatch_warned = True
        num_nodes = min(num_nodes_pred, num_nodes_graph)
        pred_velocity = pred_velocity[:num_nodes]
    else:
        num_nodes = num_nodes_graph
    
    # Check for NaN in predictions
    if torch.isnan(pred_velocity).any():
        nan_mask = torch.isnan(pred_velocity).any(dim=1)
        print(f"  [WARNING] NaN in {nan_mask.sum()} predictions! Using previous velocity.")
        pred_velocity[nan_mask] = updated_graph.x[:num_nodes, 0:2][nan_mask]
    
    # ============ UPDATE POSITIONS FROM PREDICTED VELOCITY ============
    # Denormalize velocity
    pred_vx = pred_velocity[:, 0] * MAX_SPEED  # m/s
    pred_vy = pred_velocity[:, 1] * MAX_SPEED  # m/s
    
    # Integrate: displacement = velocity * dt
    dx = pred_vx * dt  # meters
    dy = pred_vy * dt  # meters
    
    # Update positions
    if hasattr(updated_graph, 'pos') and updated_graph.pos is not None:
        displacement = torch.stack([dx, dy], dim=1)
        updated_graph.pos[:num_nodes] = updated_graph.pos[:num_nodes] + displacement
    
    # ============ DERIVE ALL NODE FEATURES FROM PREDICTION ============
    # Start with zeros, then fill in calculated values
    new_features = torch.zeros(num_nodes, 15, device=device, dtype=torch.float32)
    
    # [0-1] Velocity (normalized) - from prediction
    new_features[:, 0:2] = pred_velocity[:num_nodes]
    
    # [2] Speed (normalized) - calculate from velocity
    pred_speed = torch.sqrt(pred_vx[:num_nodes]**2 + pred_vy[:num_nodes]**2)
    new_features[:, 2] = pred_speed / MAX_SPEED
    
    # [3] Heading direction (normalized) - calculate from velocity
    pred_heading = torch.atan2(pred_vy[:num_nodes], pred_vx[:num_nodes]) / np.pi  # [-1, 1]
    new_features[:, 3] = pred_heading
    
    # [4] Valid - keep from previous graph (doesn't change)
    new_features[:, 4] = updated_graph.x[:num_nodes, 4]
    
    # [5-6] Acceleration (normalized) - calculate as velocity change
    if hasattr(updated_graph, 'x'):
        prev_vx = updated_graph.x[:num_nodes, 0] * MAX_SPEED
        prev_vy = updated_graph.x[:num_nodes, 1] * MAX_SPEED
        ax = pred_vx[:num_nodes] - prev_vx
        ay = pred_vy[:num_nodes] - prev_vy
        new_features[:, 5] = ax / MAX_ACCEL
        new_features[:, 6] = ay / MAX_ACCEL
    
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