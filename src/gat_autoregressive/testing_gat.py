"""Testing script for GNN-based autoregressive trajectory prediction.

This module evaluates trained GCN/GAT models using autoregressive multi-step prediction
on the validation set (or test set), logs metrics to wandb, and creates visualizations.

Supports both GAT and GCN model types via the model_type parameter.

Uses validation dataset by default (90 timesteps) since test dataset may have shorter
sequences that don't support proper 5-second rollouts.

Usage:
    python src/gat_autoregressive/testing_gat.py
    python src/gat_autoregressive/testing_gat.py --model_type gcn
    python src/gat_autoregressive/testing_gat.py --checkpoint path/to/model.pt
    python src/gat_autoregressive/testing_gat.py --max_scenarios 50 --no_wandb
    python src/gat_autoregressive/testing_gat.py --test_data path/to/test.hdf5
"""

import sys
import os
import argparse
from datetime import datetime
import json
import glob

# Add parent directory (src/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import torch.nn.functional as F
import wandb

# Import both model types
from gat_autoregressive.SpatioTemporalGAT_batched import SpatioTemporalGATBatched
from autoregressive_predictions.SpatioTemporalGNN_batched import SpatioTemporalGNNBatched

from dataset import HDF5ScenarioDataset
from config import (device, batch_size, gat_num_workers, gcn_num_workers, num_layers, num_gru_layers,
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, gat_num_heads, project_name, print_gpu_info,
                    num_gpus, use_data_parallel, setup_model_parallel, load_model_state,
                    use_gradient_checkpointing, POSITION_SCALE,
                    gat_checkpoint_dir, gat_checkpoint_dir_autoreg, gat_viz_dir_testing,
                    gcn_checkpoint_dir, gcn_checkpoint_dir_autoreg, gcn_viz_dir_testing,
                    test_hdf5_path, test_num_rollout_steps, test_max_scenarios,
                    test_visualize, test_visualize_max, test_use_wandb, test_horizons,
                    autoreg_skip_map_features, max_scenario_files_for_viz, use_edge_weights)
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from helper_functions.helpers import compute_metrics
from helper_functions.visualization_functions import (load_scenario_by_id, draw_map_features,
                                                       MAP_FEATURE_GRAY, VIBRANT_COLORS, SDC_COLOR)
import matplotlib.pyplot as plt

# Validation dataset path (longer sequences for proper testing)
val_hdf5_path = f'data/graphs/validation/validation_seqlen{sequence_length}.hdf5'

# Import functions from finetuning to ensure consistency
from gat_autoregressive.finetune import (update_graph_with_prediction, 
                                          visualize_autoregressive_rollout,
                                          filter_to_scenario)

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/gat_autoregressive/testing_gat.py
# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/gat_autoregressive/testing_gat.py --model_type gcn


def get_model_config(model_type):
    """Get model-specific configuration based on model type.
    
    Args:
        model_type: Either 'gat' or 'gcn'
        
    Returns:
        dict with checkpoint_dir, checkpoint_dir_autoreg, viz_dir_testing, num_workers
    """
    if model_type == 'gat':
        return {
            'checkpoint_dir': gat_checkpoint_dir,
            'checkpoint_dir_autoreg': gat_checkpoint_dir_autoreg,
            'viz_dir_testing': gat_viz_dir_testing,
            'num_workers': gat_num_workers,
            'legacy_checkpoint_names': ['best_autoregressive_20step.pt', 'best_autoregressive_50step.pt'],
            'autoreg_pattern': 'best_gat_autoreg_*.pt',
        }
    else:  # gcn
        return {
            'checkpoint_dir': gcn_checkpoint_dir,
            'checkpoint_dir_autoreg': gcn_checkpoint_dir_autoreg,
            'viz_dir_testing': gcn_viz_dir_testing,
            'num_workers': gcn_num_workers,
            'legacy_checkpoint_names': ['finetuned_scheduled_sampling_best.pt'],
            'autoreg_pattern': 'best_gcn_autoreg_*.pt',
        }


def load_trained_model(checkpoint_path, device, model_type='gat'):
    """Load a trained GAT or GCN model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        model_type: Either 'gat' or 'gcn'
        
    Returns:
        (model, checkpoint) tuple
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model_name = model_type.upper()
    print(f"Loading {model_name} model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    
    if model_type == 'gat':
        model = SpatioTemporalGATBatched(
            input_dim=config.get('input_dim', input_dim),
            hidden_dim=config.get('hidden_channels', hidden_channels),
            output_dim=config.get('output_dim', output_dim),
            num_gat_layers=config.get('num_layers', num_layers),
            num_gru_layers=config.get('num_gru_layers', num_gru_layers),
            dropout=config.get('dropout', dropout),
            num_heads=config.get('num_heads', gat_num_heads),
            use_gradient_checkpointing=use_gradient_checkpointing,
            max_agents_per_scenario=128
        )
    else:  # gcn
        model = SpatioTemporalGNNBatched(
            input_dim=config.get('input_dim', input_dim),
            hidden_dim=config.get('hidden_channels', hidden_channels),
            output_dim=config.get('output_dim', output_dim),
            num_gcn_layers=config.get('num_layers', num_layers),
            num_gru_layers=config.get('num_gru_layers', num_gru_layers),
            dropout=config.get('dropout', dropout),
            use_gat=config.get('use_gat', False),
            use_gradient_checkpointing=use_gradient_checkpointing,
            max_agents_per_scenario=128
        )
    
    model, is_parallel = setup_model_parallel(model, device)
    load_model_state(model, checkpoint['model_state_dict'], is_parallel)
    model.eval()
    
    print(f"  {model_name} Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    if 'train_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['train_loss']:.6f}")
    
    return model, checkpoint


def get_base_model(model):
    """Get the base model (unwrap DataParallel if needed)."""
    if hasattr(model, 'module'):
        return model.module
    return model


def predict_single_step(model, graph, device, batch_size=1, timestep=0):
    """Predict one timestep ahead for all agents in a graph.
    
    Uses the same forward pass as training/finetuning code.
    """
    graph = graph.to(device)
    
    with torch.no_grad():
        # Get agent IDs if available
        agent_ids = getattr(graph, 'agent_ids', None)
        # Get edge weights if available
        edge_w = graph.edge_attr if use_edge_weights else None
        
        # Forward pass matching training/finetuning code
        predictions = model(
            graph.x, 
            graph.edge_index,
            edge_weight=edge_w,
            batch=graph.batch,
            batch_size=batch_size,
            batch_num=0,
            timestep=timestep,
            agent_ids=agent_ids
        )
    
    return predictions


# Removed - using update_graph_with_prediction from finetune.py instead


def autoregressive_rollout(model, initial_graph, num_steps, device, batch_size=1):
    """Autoregressive multi-step prediction using same logic as finetuning."""
    model.eval()
    current_graph = initial_graph.clone().to(device)
    
    all_predictions = []
    updated_graphs = [current_graph.cpu()]
    
    for step in range(num_steps):
        pred_displacement = predict_single_step(model, current_graph, device, 
                                               batch_size=batch_size, timestep=step)
        all_predictions.append(pred_displacement.cpu())
        # Use the same update function as finetuning for consistency
        current_graph = update_graph_with_prediction(current_graph, pred_displacement, device)
        updated_graphs.append(current_graph.cpu())
    
    return all_predictions, updated_graphs


def evaluate_autoregressive(model, dataloader, num_rollout_steps, device, max_scenarios=None, model_type='gat'):
    """Evaluate model using autoregressive rollout on test set.
    
    Args:
        model: The trained model (GAT or GCN)
        dataloader: DataLoader for test/validation data
        num_rollout_steps: Number of autoregressive rollout steps
        device: Device to run on
        max_scenarios: Maximum number of scenarios to evaluate
        model_type: Either 'gat' or 'gcn' (for logging)
    """
    model.eval()
    model_name = model_type.upper()
    
    # Use horizons from config, capped at rollout steps
    horizons = [h for h in test_horizons if h <= num_rollout_steps]
    if not horizons:
        horizons = [min(10, num_rollout_steps)]
    horizon_metrics = {h: {'ade': [], 'fde': [], 'angle_error': [], 'cosine_sim': []} for h in horizons}
    
    all_scenario_results = []
    scenario_count = 0
    
    print(f"\nEvaluating {model_name} with {num_rollout_steps}-step autoregressive rollout...")
    print(f"Horizons: {[f'{h*0.1}s' for h in horizons]}")
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            if max_scenarios and scenario_count >= max_scenarios:
                break
            
            batched_graph_sequence = batch_dict["batch"]
            B = batch_dict["B"]
            T = batch_dict["T"]
            scenario_ids = batch_dict.get("scenario_ids", [])
            
            for t in range(T):
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
            
            num_nodes = batched_graph_sequence[0].num_nodes
            
            # Compute agents per scenario for proper GRU sizing (matching finetuning)
            base_model = get_base_model(model)
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
            
            # Use T//3 for warmup to match finetuning validation approach
            start_t = T // 3
            actual_rollout = min(num_rollout_steps, T - start_t - 1)
            
            if actual_rollout <= 0:
                print(f"  Skipping scenario {scenario_count+1}: not enough timesteps (T={T}, need at least {start_t + 2})")
                continue
            
            print(f"Scenario {scenario_count+1} ({scenario_ids[0] if scenario_ids else 'unknown'}): "
                  f"T={T}, warmup={start_t} steps, rollout={actual_rollout} steps ({actual_rollout * 0.1:.1f}s)")
            
            # WARM UP GRU: Run through timesteps 0 to start_t to build temporal context
            # This matches training/finetuning which processes sequences from t=0
            for warm_t in range(start_t):
                warm_graph = batched_graph_sequence[warm_t]
                if warm_graph.x is not None:
                    agent_ids_warm = getattr(warm_graph, 'agent_ids', None)
                    edge_w = warm_graph.edge_attr if use_edge_weights else None
                    _ = model(warm_graph.x, warm_graph.edge_index, edge_weight=edge_w,
                             batch=warm_graph.batch, batch_size=B, batch_num=0, timestep=warm_t,
                             agent_ids=agent_ids_warm)
            
            # Start rollout from start_t (after warmup)
            initial_graph = batched_graph_sequence[start_t]
            predictions, _ = autoregressive_rollout(model, initial_graph, actual_rollout, device, batch_size=B)
            
            # Collect ground truth displacements for comparison
            ground_truths = []
            for step in range(1, actual_rollout + 1):
                if start_t + step < T:
                    gt_graph = batched_graph_sequence[start_t + step]
                    ground_truths.append(gt_graph.y.cpu())
                else:
                    break
            
            # Debug: Check if we have enough rollout steps
            if len(predictions) < min(horizons):
                print(f"  WARNING: Only {len(predictions)} predictions (need {min(horizons)} for shortest horizon)")
                print(f"           T={T}, start_t={start_t}, actual_rollout={actual_rollout}")
            
            scenario_metrics = {}
            for horizon in horizons:
                if horizon <= len(predictions) and horizon <= len(ground_truths):
                    ade = 0.0
                    for step in range(horizon):
                        pred = predictions[step] * 100.0
                        gt = ground_truths[step] * 100.0
                        displacement_error = torch.norm(pred - gt, dim=1).mean()
                        ade += displacement_error.item()
                    ade /= horizon
                    
                    pred_final = predictions[horizon-1] * 100.0
                    gt_final = ground_truths[horizon-1] * 100.0
                    fde = torch.norm(pred_final - gt_final, dim=1).mean().item()
                    
                    pred_angle = torch.atan2(predictions[horizon-1][:, 1], predictions[horizon-1][:, 0])
                    gt_angle = torch.atan2(ground_truths[horizon-1][:, 1], ground_truths[horizon-1][:, 0])
                    angle_diff = torch.atan2(torch.sin(pred_angle - gt_angle), torch.cos(pred_angle - gt_angle))
                    angle_error = torch.abs(angle_diff).mean().item() * 180 / np.pi
                    
                    pred_norm = F.normalize(predictions[horizon-1], p=2, dim=1, eps=1e-6)
                    gt_norm = F.normalize(ground_truths[horizon-1], p=2, dim=1, eps=1e-6)
                    cosine_sim = F.cosine_similarity(pred_norm, gt_norm, dim=1).mean().item()
                    
                    horizon_metrics[horizon]['ade'].append(ade)
                    horizon_metrics[horizon]['fde'].append(fde)
                    horizon_metrics[horizon]['angle_error'].append(angle_error)
                    horizon_metrics[horizon]['cosine_sim'].append(cosine_sim)
                    
                    scenario_metrics[f'{horizon*0.1}s'] = {
                        'ade': ade,
                        'fde': fde,
                        'angle_error': angle_error,
                        'cosine_sim': cosine_sim
                    }
            
            all_scenario_results.append({
                'scenario_id': scenario_ids[0] if scenario_ids else f'scenario_{batch_idx}',
                'metrics': scenario_metrics
            })
            
            print(f"  Metrics:")
            for h in horizons:
                if f'{h*0.1}s' in scenario_metrics:
                    m = scenario_metrics[f'{h*0.1}s']
                    print(f"    {h*0.1}s: ADE={m['ade']:.2f}m, FDE={m['fde']:.2f}m, "
                          f"Angle={m['angle_error']:.1f}deg, Cos={m['cosine_sim']:.3f}")
            
            scenario_count += 1
    
    print("\n" + "="*70)
    print(f"{model_name} TEST RESULTS")
    print("="*70)
    
    results = {
        'horizons': {},
        'scenarios': all_scenario_results
    }
    
    for horizon in horizons:
        if horizon_metrics[horizon]['ade']:
            results['horizons'][f'{horizon*0.1}s'] = {
                'ade_mean': np.mean(horizon_metrics[horizon]['ade']),
                'ade_std': np.std(horizon_metrics[horizon]['ade']),
                'fde_mean': np.mean(horizon_metrics[horizon]['fde']),
                'fde_std': np.std(horizon_metrics[horizon]['fde']),
                'angle_error_mean': np.mean(horizon_metrics[horizon]['angle_error']),
                'angle_error_std': np.std(horizon_metrics[horizon]['angle_error']),
                'cosine_sim_mean': np.mean(horizon_metrics[horizon]['cosine_sim']),
                'cosine_sim_std': np.std(horizon_metrics[horizon]['cosine_sim'])
            }
            
            r = results['horizons'][f'{horizon*0.1}s']
            print(f"\n{horizon*0.1}s Horizon ({horizon} steps):")
            print(f"  ADE: {r['ade_mean']:.2f} +/- {r['ade_std']:.2f} m")
            print(f"  FDE: {r['fde_mean']:.2f} +/- {r['fde_std']:.2f} m")
            print(f"  Angle Error: {r['angle_error_mean']:.1f} +/- {r['angle_error_std']:.1f} deg")
            print(f"  Cosine Sim: {r['cosine_sim_mean']:.3f} +/- {r['cosine_sim_std']:.3f}")
    
    print("="*70)
    
    return results


def visualize_test_scenario(model, batch_dict, scenario_idx, save_dir, device, num_rollout_steps=None):
    """Visualize predictions vs ground truth for a test scenario."""
    os.makedirs(save_dir, exist_ok=True)
    
    if num_rollout_steps is None:
        num_rollout_steps = test_num_rollout_steps
    
    batched_graph_sequence = batch_dict["batch"]
    T = batch_dict["T"]
    scenario_ids = batch_dict.get("scenario_ids", [])
    scenario_id = scenario_ids[0] if scenario_ids else f"scenario_{scenario_idx}"
    
    # Load scenario for map features (matching finetuning approach)
    scenario = None
    if not autoreg_skip_map_features and scenario_ids and scenario_ids[0] and scenario_ids[0] != "unknown":
        try:
            scenario = load_scenario_by_id(
                scenario_ids[0],
                scenario_dirs=['./data/scenario/testing'],
                max_files_for_index=max_scenario_files_for_viz
            )
            if scenario is not None:
                print(f"  Loaded scenario {scenario_ids[0]} for map visualization")
        except Exception as e:
            print(f"  Warning: Could not load scenario for map features: {e}")
            scenario = None
    
    for t in range(T):
        batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
    
    num_nodes = batched_graph_sequence[0].num_nodes
    B = batch_dict["B"]
    
    # Compute agents per scenario for proper GRU sizing
    base_model = get_base_model(model)
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
    
    # Find SDC (ego vehicle) for priority visualization
    sdc_id = None
    if scenario is not None:
        for track in scenario.tracks:
            if track.object_type == 3:  # TYPE_EGO_VEHICLE = 3
                sdc_id = track.id
                break
    
    # Find persistent agents (present throughout sequence)
    # Use agent_id if available, otherwise use node indices
    has_agent_ids = hasattr(batched_graph_sequence[0], 'agent_id') and batched_graph_sequence[0].agent_id is not None
    
    if has_agent_ids:
        persistent_agent_ids = None
        for t in range(T):
            graph = batched_graph_sequence[t]
            current_ids = set(graph.agent_id.cpu().numpy())
            if persistent_agent_ids is None:
                persistent_agent_ids = current_ids
            else:
                persistent_agent_ids = persistent_agent_ids & current_ids
        
        persistent_agent_ids = sorted(list(persistent_agent_ids))
    else:
        # Use node indices as agent IDs
        num_agents = batched_graph_sequence[0].num_nodes
        persistent_agent_ids = list(range(num_agents))
    
    if len(persistent_agent_ids) == 0:
        print(f"  No persistent agents found, skipping visualization")
        return
    
    # Collect ground truth positions
    gt_positions = {agent_id: [] for agent_id in persistent_agent_ids}
    
    for t in range(T):
        graph = batched_graph_sequence[t]
        positions_t = graph.pos.cpu().numpy()
        
        if has_agent_ids:
            agent_ids_t = graph.agent_id.cpu().numpy()
            for agent_id in persistent_agent_ids:
                idx = np.where(agent_ids_t == agent_id)[0]
                if len(idx) > 0:
                    gt_positions[agent_id].append(positions_t[idx[0]])
        else:
            # Use node indices directly
            for agent_id in persistent_agent_ids:
                if agent_id < len(positions_t):
                    gt_positions[agent_id].append(positions_t[agent_id])
    
    # Run autoregressive prediction with proper warmup (matching finetuning validation)
    start_t = T // 3
    # Use the test_num_rollout_steps from config (default 50) capped by available timesteps
    num_rollout_steps = min(test_num_rollout_steps, T - start_t - 1)
    
    if num_rollout_steps <= 0:
        print(f"  Not enough timesteps for visualization")
        return
    
    # WARM UP GRU: Run through timesteps 0 to start_t
    for warm_t in range(start_t):
        warm_graph = batched_graph_sequence[warm_t]
        if warm_graph.x is not None:
            agent_ids_warm = getattr(warm_graph, 'agent_ids', None)
            _ = model(warm_graph.x, warm_graph.edge_index,
                     batch=warm_graph.batch, batch_size=B, batch_num=0, timestep=warm_t,
                     agent_ids=agent_ids_warm)
    
    # Start rollout from start_t
    initial_graph = batched_graph_sequence[start_t]
    positions_init = initial_graph.pos.cpu().numpy()
    
    # Get agent mapping
    if has_agent_ids:
        agent_ids_init = initial_graph.agent_id.cpu().numpy()
    else:
        agent_ids_init = np.arange(len(positions_init))  # Use indices as IDs
    
    predictions, _ = autoregressive_rollout(model, initial_graph, num_rollout_steps, device, batch_size=B)
    
    # Build predicted trajectories and compute errors per agent
    pred_positions = {agent_id: [] for agent_id in persistent_agent_ids}
    agent_errors = {agent_id: [] for agent_id in persistent_agent_ids}  # Per-timestep errors
    
    for agent_id in persistent_agent_ids:
        if has_agent_ids:
            idx = np.where(agent_ids_init == agent_id)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]
        else:
            # Use agent_id as direct index
            idx = agent_id
            if idx >= len(positions_init):
                continue
        
        current_pos = positions_init[idx].copy()
        pred_positions[agent_id].append(current_pos.copy())
        
        # Compute errors against ground truth at each timestep
        for step_idx, step_pred in enumerate(predictions):
            pred_np = step_pred.cpu().numpy()
            if idx < len(pred_np):
                # Model outputs NORMALIZED displacement, multiply by POSITION_SCALE to get meters
                displacement_normalized = pred_np[idx]
                displacement_meters = displacement_normalized * POSITION_SCALE
                current_pos = current_pos + displacement_meters
                pred_positions[agent_id].append(current_pos.copy())
                
                # Get ground truth position at this timestep
                gt_timestep = start_t + step_idx + 1
                if gt_timestep < len(gt_positions[agent_id]):
                    gt_pos = gt_positions[agent_id][gt_timestep]
                    error_meters = np.linalg.norm(current_pos - gt_pos)
                    agent_errors[agent_id].append(error_meters)
    
    # Select agents to visualize (matching finetuning approach)
    # Prioritize: 1) SDC (ego), 2) Most active agents (by movement)
    max_agents_viz = 10
    agent_movements = {}
    
    for agent_id in persistent_agent_ids:
        if agent_id not in gt_positions or len(gt_positions[agent_id]) < 2:
            continue
        # Compute total movement (sum of distances between consecutive positions)
        positions = np.array(gt_positions[agent_id])
        total_movement = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        agent_movements[agent_id] = total_movement
    
    # Select agents: SDC first, then by movement
    selected_agent_ids = []
    if sdc_id is not None and sdc_id in agent_movements:
        selected_agent_ids.append(sdc_id)
    
    # Add most active agents (excluding SDC if already added)
    other_agents = [aid for aid in agent_movements.keys() if aid != sdc_id]
    other_agents_sorted = sorted(other_agents, key=lambda aid: agent_movements[aid], reverse=True)
    remaining_slots = max_agents_viz - len(selected_agent_ids)
    selected_agent_ids.extend(other_agents_sorted[:remaining_slots])
    
    if len(selected_agent_ids) == 0:
        print(f"  No agents with sufficient movement, skipping visualization")
        return
    
    # Create visualization with map features (matching finetuning)
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Draw map features if available (roads, lanes, etc.)
    if scenario is not None:
        draw_map_features(ax, scenario)
    
    # Assign colors (SDC gets special color, others get vibrant colors)
    agent_colors = {}
    color_idx = 0
    for agent_id in selected_agent_ids:
        if agent_id == sdc_id:
            agent_colors[agent_id] = SDC_COLOR
        else:
            agent_colors[agent_id] = VIBRANT_COLORS[color_idx % len(VIBRANT_COLORS)]
            color_idx += 1
    
    # Compute metrics across all agents
    all_ades = []
    all_fdes = []
    num_agents_plotted = 0
    
    for agent_id in selected_agent_ids:
        gt_traj = np.array(gt_positions[agent_id])
        pred_traj = np.array(pred_positions[agent_id])
        
        if len(gt_traj) < 2 or len(pred_traj) < 2:
            continue
        
        color = agent_colors[agent_id]
        
        # Create label with agent ID and type (SDC if applicable)
        if agent_id == sdc_id:
            label_prefix = f'SDC (ID {agent_id})'
        else:
            label_prefix = f'Agent {agent_id}'
        
        # Plot ground truth trajectory
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], '-', color=color, linewidth=2.0, alpha=0.8,
                label=f'{label_prefix} GT' if num_agents_plotted < 5 else None)
        
        # Plot predicted trajectory
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], '--', color=color, linewidth=2.0, alpha=0.8,
                label=f'{label_prefix} Pred' if num_agents_plotted < 5 else None)
        
        # Mark start point (circle)
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], color=color, marker='o', s=50, 
                   edgecolors='white', linewidths=1, zorder=5)
        
        # Mark GT end point (X)
        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color=color, marker='x', s=80, 
                   linewidths=2, zorder=5)
        
        # Mark predicted endpoint (square)
        ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], color=color, marker='s', s=60, 
                   edgecolors='black', linewidths=1, zorder=6)
        
        # Compute ADE and FDE for this agent
        if agent_id in agent_errors and len(agent_errors[agent_id]) > 0:
            agent_ade = np.mean(agent_errors[agent_id])
            agent_fde = agent_errors[agent_id][-1]  # Final displacement error
            all_ades.append(agent_ade)
            all_fdes.append(agent_fde)
        
        num_agents_plotted += 1
    
    # Compute overall metrics
    overall_ade = np.mean(all_ades) if all_ades else 0.0
    overall_fde = np.mean(all_fdes) if all_fdes else 0.0
    horizon_time = num_rollout_steps * 0.1  # Convert timesteps to seconds
    
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)
    ax.set_title(f'GAT Test Scenario {scenario_id}\n'
                 f'{num_agents_plotted} agents | Horizon: {num_rollout_steps} steps ({horizon_time:.1f}s)\n'
                 f'ADE: {overall_ade:.2f}m | FDE: {overall_fde:.2f}m | Start from t={start_t}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    save_path = os.path.join(save_dir, f'gat_test_scenario_{scenario_idx:04d}_{scenario_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {save_path} | ADE: {overall_ade:.2f}m | FDE: {overall_fde:.2f}m | Horizon: {horizon_time:.1f}s")
    return overall_ade


def run_testing(test_dataset_path=val_hdf5_path,  # Use validation dataset (has 90 timesteps vs test's 9)
                checkpoint_path=None,
                num_rollout_steps=20,
                max_scenarios=None,
                visualize=True,
                visualize_max=10,
                use_wandb=True,
                model_type='gat'):
    """Run testing with autoregressive multi-step prediction using GAT or GCN model.
    
    Args:
        test_dataset_path: Path to test HDF5 file
        checkpoint_path: Path to model checkpoint (default: auto-detect best)
        num_rollout_steps: Number of autoregressive rollout steps
        max_scenarios: Maximum scenarios to evaluate (None = all)
        visualize: Whether to generate visualizations
        visualize_max: Maximum scenarios to visualize
        use_wandb: Whether to log results to wandb
        model_type: Either 'gat' or 'gcn'
    """
    model_name = model_type.upper()
    model_config = get_model_config(model_type)
    
    print("\n" + "="*70)
    print(f"{model_name} MODEL TESTING - Autoregressive Multi-Step Prediction")
    print("="*70)
    
    # Print GPU info
    print_gpu_info()
    
    # Initialize wandb
    if use_wandb:
        wandb.login()
        run = wandb.init(
            project=project_name,
            name=f"{model_type}-testing-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": model_type,
                "task": "testing",
                "num_rollout_steps": num_rollout_steps,
                "max_scenarios": max_scenarios,
                "test_dataset": test_dataset_path,
            },
            tags=[model_type, "testing", "autoregressive"]
        )
    
    # Use autoregressive checkpoint by default
    if checkpoint_path is None:
        checkpoint_dir_autoreg = model_config['checkpoint_dir_autoreg']
        checkpoint_dir = model_config['checkpoint_dir']
        autoreg_pattern_name = model_config['autoreg_pattern']
        legacy_names = model_config['legacy_checkpoint_names']
        
        # Try autoregressive checkpoint first with new naming pattern, then legacy names
        autoreg_pattern = os.path.join(checkpoint_dir_autoreg, autoreg_pattern_name)
        autoreg_matches = glob.glob(autoreg_pattern)
        
        # Build legacy paths
        legacy_paths = [os.path.join(checkpoint_dir_autoreg, name) for name in legacy_names]
        legacy_paths += [os.path.join('checkpoints', name) for name in legacy_names]
        
        base_path = os.path.join(checkpoint_dir, 'best_model.pt')
        root_base_path = os.path.join('checkpoints', model_type, 'best_model.pt')
        
        if autoreg_matches:
            # Use most recent autoregressive checkpoint
            checkpoint_path = max(autoreg_matches, key=os.path.getmtime)
            print(f"Using {model_name} autoregressive checkpoint: {checkpoint_path}")
        else:
            # Try legacy paths
            found_legacy = False
            for legacy_path in legacy_paths:
                if os.path.exists(legacy_path):
                    checkpoint_path = legacy_path
                    print(f"Using {model_name} legacy checkpoint: {legacy_path}")
                    found_legacy = True
                    break
            
            if not found_legacy:
                if os.path.exists(base_path):
                    checkpoint_path = base_path
                    print(f"Using {model_name} base single-step checkpoint: {base_path}")
                elif os.path.exists(root_base_path):
                    checkpoint_path = root_base_path
                    print(f"Using {model_name} base single-step checkpoint: {root_base_path}")
                else:
                    print(f"ERROR: No {model_name} checkpoint found!")
                    print(f"  Checked pattern: {autoreg_pattern}")
                    print(f"  Checked legacy: {legacy_paths}")
                    print(f"  Checked base: {base_path}")
                    if use_wandb:
                        wandb.finish(exit_code=1)
                    return None
    
    # Load model
    model, checkpoint = load_trained_model(checkpoint_path, device, model_type=model_type)
    
    if use_wandb:
        wandb.config.update({"checkpoint_path": checkpoint_path})
    
    # Load test dataset
    if not os.path.exists(test_dataset_path):
        print(f"ERROR: {test_dataset_path} not found!")
        print("Please run graph_creation_and_saving.py to create test data.")
        if use_wandb:
            wandb.finish(exit_code=1)
        return None
    
    test_dataset = HDF5ScenarioDataset(test_dataset_path, seq_len=sequence_length)
    dataset_type = "validation" if "validation" in test_dataset_path else "test"
    print(f"Loaded {dataset_type} dataset: {len(test_dataset)} scenarios (seq_len={sequence_length})")
    
    # Use num_workers=0 for testing to avoid "Too many open files" errors with HDF5
    # Testing is sequential anyway and doesn't benefit much from multiprocessing
    viz_dir_testing = model_config['viz_dir_testing']
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Avoid file descriptor exhaustion during testing
        collate_fn=collate_graph_sequences_to_batch,
        drop_last=False
    )
    
    effective_rollout = min(num_rollout_steps, sequence_length - 2)
    
    print(f"\n{model_name} Test Configuration:")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Rollout steps: {effective_rollout} ({effective_rollout * 0.1:.1f}s)")
    print(f"  Max scenarios: {max_scenarios if max_scenarios else 'all'}")
    print(f"  W&B logging: {use_wandb}")
    
    # Create visualization directory
    os.makedirs(viz_dir_testing, exist_ok=True)
    
    # Run evaluation with visualization using finetuning's visualization function
    # This ensures consistent visualization with proper trajectory tracking
    viz_images = []
    is_parallel = hasattr(model, 'module')
    
    if visualize:
        print(f"\nGenerating visualizations using finetuning's visualization (max {visualize_max} scenarios)...")
        viz_count = 0
        
        for batch_idx, batch_dict in enumerate(test_dataloader):
            if viz_count >= visualize_max:
                break
            
            # Use finetuning's visualization function for proper trajectory tracking
            final_error = visualize_autoregressive_rollout(
                model=model,
                batch_dict=batch_dict,
                epoch=0,  # Treat as epoch 0 for naming
                num_rollout_steps=effective_rollout,
                device=device,
                is_parallel=is_parallel,
                save_dir=viz_dir_testing,
                total_epochs=1,
                model_type=model_type
            )
            
            if final_error is not None:
                viz_count += 1
                
                # Log visualization to wandb
                if use_wandb:
                    scenario_ids = batch_dict.get("scenario_ids", [])
                    scenario_id = scenario_ids[0] if scenario_ids else f"scenario_{batch_idx}"
                    # Find the most recent visualization file
                    pattern = os.path.join(viz_dir_testing, f'{model_type}_autoreg_epoch001_{scenario_id}_*.png')
                    matches = glob.glob(pattern)
                    if matches:
                        latest_viz = max(matches, key=os.path.getmtime)
                        viz_images.append(wandb.Image(latest_viz, caption=f"Scenario {scenario_id} (Error: {final_error:.2f}m)"))
        
        if viz_count > 0:
            print(f"\nVisualization Summary:")
            print(f"  Scenarios visualized: {viz_count}")
            print(f"  Saved to: {viz_dir_testing}")
            
            if use_wandb and viz_images:
                wandb.log({"test_visualizations": viz_images})
    
    # Run full evaluation
    results = evaluate_autoregressive(
        model, test_dataloader, effective_rollout, device, max_scenarios, model_type=model_type
    )
    
    # Log results to wandb
    if use_wandb and results:
        wandb_log = {"test/num_scenarios": len(results.get('scenarios', []))}
        
        for horizon_key, metrics in results.get('horizons', {}).items():
            wandb_log[f"test/{horizon_key}/ade_mean"] = metrics['ade_mean']
            wandb_log[f"test/{horizon_key}/ade_std"] = metrics['ade_std']
            wandb_log[f"test/{horizon_key}/fde_mean"] = metrics['fde_mean']
            wandb_log[f"test/{horizon_key}/fde_std"] = metrics['fde_std']
            wandb_log[f"test/{horizon_key}/angle_error_mean"] = metrics['angle_error_mean']
            wandb_log[f"test/{horizon_key}/cosine_sim_mean"] = metrics['cosine_sim_mean']
        
        wandb.log(wandb_log)
        
        # Log summary metrics
        if '5.0s' in results.get('horizons', {}):
            wandb.run.summary["final_ade_5s"] = results['horizons']['5.0s']['ade_mean']
            wandb.run.summary["final_fde_5s"] = results['horizons']['5.0s']['fde_mean']
        elif '2.0s' in results.get('horizons', {}):
            wandb.run.summary["final_ade_2s"] = results['horizons']['2.0s']['ade_mean']
            wandb.run.summary["final_fde_2s"] = results['horizons']['2.0s']['fde_mean']
    
    # Save results
    checkpoint_dir = model_config['checkpoint_dir']
    checkpoint_dir_autoreg = model_config['checkpoint_dir_autoreg']
    
    results_path = os.path.join(checkpoint_dir, f'{model_type}_test_results.pt')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_dir_autoreg, exist_ok=True)
    torch.save(results, results_path)
    print(f"Results saved to {results_path}")
    
    # Also save as JSON for easy reading
    json_results = {
        'model_type': model_type,
        'checkpoint': checkpoint_path,
        'test_dataset': test_dataset_path,
        'num_scenarios': len(results.get('scenarios', [])),
        'horizons': results.get('horizons', {}),
        'timestamp': datetime.now().isoformat()
    }
    json_path = os.path.join(checkpoint_dir_autoreg, f'{model_type}_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to {json_path}")
    
    if use_wandb:
        wandb.finish()
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GNN Model Testing (GAT/GCN)')
    parser.add_argument('--model_type', type=str, default='gat', choices=['gat', 'gcn'],
                        help='Model type: gat or gcn (default: gat)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: auto-detect)')
    parser.add_argument('--test_data', type=str, default=val_hdf5_path,  # Use validation dataset by default (90 timesteps)
                        help='Path to test/validation HDF5 file (default: validation dataset)')
    parser.add_argument('--num_rollout_steps', type=int, default=test_num_rollout_steps,
                        help='Number of rollout steps')
    parser.add_argument('--max_scenarios', type=int, default=test_max_scenarios,
                        help='Max scenarios to evaluate (None = all)')
    parser.add_argument('--visualize_max', type=int, default=test_visualize_max,
                        help='Max scenarios to visualize')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    results = run_testing(
        test_dataset_path=args.test_data,
        checkpoint_path=args.checkpoint,
        num_rollout_steps=args.num_rollout_steps,
        max_scenarios=args.max_scenarios,
        visualize=not args.no_visualize,
        visualize_max=args.visualize_max,
        use_wandb=not args.no_wandb,
        model_type=args.model_type
    )
    
    return results


if __name__ == '__main__':
    main()
