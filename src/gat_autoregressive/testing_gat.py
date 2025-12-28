"""Testing script for GAT-based autoregressive trajectory prediction.

This module evaluates trained GAT models using autoregressive multi-step prediction
on the test set, logs metrics to wandb, and creates visualizations.

Usage:
    python src/gat_autoregressive/testing_gat.py
    python src/gat_autoregressive/testing_gat.py --checkpoint path/to/model.pt
    python src/gat_autoregressive/testing_gat.py --max_scenarios 50 --no_wandb
"""

import sys
import os
import argparse
from datetime import datetime
import json

# Add parent directory (src/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import torch.nn.functional as F
import wandb
from SpatioTemporalGAT_batched import SpatioTemporalGATBatched
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, gat_num_workers, num_layers, num_gru_layers,
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, gat_num_heads, project_name, print_gpu_info,
                    num_gpus, use_data_parallel, setup_model_parallel, load_model_state,
                    use_gradient_checkpointing, POSITION_SCALE,
                    gat_checkpoint_dir, gat_checkpoint_dir_autoreg, gat_viz_dir_testing,
                    test_hdf5_path, test_num_rollout_steps, test_max_scenarios,
                    test_visualize, test_visualize_max, test_use_wandb, test_horizons)
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from helper_functions.helpers import compute_metrics
from helper_functions.visualization_functions import (load_scenario_by_id, draw_map_features,
                                                       MAP_FEATURE_GRAY, VIBRANT_COLORS, SDC_COLOR)
import matplotlib.pyplot as plt

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/gat_autoregressive/testing_gat.py

def load_trained_model(checkpoint_path, device):
    """Load a trained GAT model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading GAT model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
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
    
    model, is_parallel = setup_model_parallel(model, device)
    load_model_state(model, checkpoint['model_state_dict'], is_parallel)
    model.eval()
    
    print(f" GAT Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    if 'train_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['train_loss']:.6f}")
    
    return model, checkpoint


def predict_single_step(model, graph, device):
    """Predict one timestep ahead for all agents in a graph."""
    graph = graph.to(device)
    
    with torch.no_grad():
        # GAT doesn't use edge weights
        predictions = model(
            graph.x, 
            graph.edge_index,
            batch=graph.batch,
            batch_size=1,
            batch_num=0,
            timestep=0
        )
    
    return predictions


def update_graph_features(graph, pred_displacement, device):
    """Update node features based on predicted displacement."""
    updated_graph = graph.clone()
    
    if hasattr(updated_graph, 'pos'):
        updated_graph.pos = updated_graph.pos + pred_displacement * 100.0
    
    dt = 0.1
    
    new_vx = pred_displacement[:, 0] / dt
    new_vy = pred_displacement[:, 1] / dt
    
    old_vx = updated_graph.x[:, 0]
    old_vy = updated_graph.x[:, 1]
    ax = (new_vx - old_vx) / dt / 10.0
    ay = (new_vy - old_vy) / dt / 10.0
    
    updated_graph.x[:, 0] = new_vx
    updated_graph.x[:, 1] = new_vy
    updated_graph.x[:, 2] = torch.sqrt(new_vx**2 + new_vy**2)
    updated_graph.x[:, 3] = torch.atan2(new_vy, new_vx) / np.pi
    updated_graph.x[:, 5] = ax
    updated_graph.x[:, 6] = ay
    
    return updated_graph


def autoregressive_rollout(model, initial_graph, num_steps, device):
    """Autoregressive multi-step prediction."""
    model.eval()
    current_graph = initial_graph.clone().to(device)
    
    all_predictions = []
    updated_graphs = [current_graph.cpu()]
    
    for step in range(num_steps):
        pred_displacement = predict_single_step(model, current_graph, device)
        all_predictions.append(pred_displacement.cpu())
        current_graph = update_graph_features(current_graph, pred_displacement, device)
        updated_graphs.append(current_graph.cpu())
    
    return all_predictions, updated_graphs


def evaluate_autoregressive(model, dataloader, num_rollout_steps, device, max_scenarios=None):
    """Evaluate GAT model using autoregressive rollout on test set."""
    model.eval()
    
    # Use horizons from config, capped at rollout steps
    horizons = [h for h in test_horizons if h <= num_rollout_steps]
    if not horizons:
        horizons = [min(10, num_rollout_steps)]
    horizon_metrics = {h: {'ade': [], 'fde': [], 'angle_error': [], 'cosine_sim': []} for h in horizons}
    
    all_scenario_results = []
    scenario_count = 0
    
    print(f"\nEvaluating GAT with {num_rollout_steps}-step autoregressive rollout...")
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
            model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
            
            warmup_steps = T // 2
            print(f"Scenario {scenario_count+1} ({scenario_ids[0] if scenario_ids else 'unknown'}): "
                  f"Warming up with {warmup_steps} steps, then rolling out {num_rollout_steps} steps...")
            
            for t in range(warmup_steps):
                graph = batched_graph_sequence[t]
                _ = predict_single_step(model, graph, device)
            
            actual_rollout = min(num_rollout_steps, T - warmup_steps - 1)
            if actual_rollout <= 0:
                print(f"  Skipping: not enough timesteps for rollout")
                continue
            
            initial_graph = batched_graph_sequence[warmup_steps]
            predictions, _ = autoregressive_rollout(model, initial_graph, actual_rollout, device)
            
            ground_truths = []
            for step in range(1, actual_rollout + 1):
                if warmup_steps + step < T:
                    gt_graph = batched_graph_sequence[warmup_steps + step]
                    ground_truths.append(gt_graph.y.cpu())
                else:
                    break
            
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
    print("GAT TEST RESULTS")
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


def visualize_test_scenario(model, batch_dict, scenario_idx, save_dir, device):
    """Visualize predictions vs ground truth for a test scenario."""
    os.makedirs(save_dir, exist_ok=True)
    
    batched_graph_sequence = batch_dict["batch"]
    T = batch_dict["T"]
    scenario_ids = batch_dict.get("scenario_ids", [])
    scenario_id = scenario_ids[0] if scenario_ids else f"scenario_{scenario_idx}"
    
    for t in range(T):
        batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
    
    num_nodes = batched_graph_sequence[0].num_nodes
    model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
    
    # Find persistent agents
    persistent_agent_ids = None
    for t in range(T):
        graph = batched_graph_sequence[t]
        current_ids = set(graph.agent_id.cpu().numpy())
        if persistent_agent_ids is None:
            persistent_agent_ids = current_ids
        else:
            persistent_agent_ids = persistent_agent_ids & current_ids
    
    persistent_agent_ids = sorted(list(persistent_agent_ids))
    if len(persistent_agent_ids) == 0:
        print(f"  No persistent agents found, skipping visualization")
        return
    
    # Collect ground truth positions
    gt_positions = {agent_id: [] for agent_id in persistent_agent_ids}
    
    for t in range(T):
        graph = batched_graph_sequence[t]
        agent_ids_t = graph.agent_id.cpu().numpy()
        positions_t = graph.pos.cpu().numpy()
        
        for agent_id in persistent_agent_ids:
            idx = np.where(agent_ids_t == agent_id)[0]
            if len(idx) > 0:
                gt_positions[agent_id].append(positions_t[idx[0]])
    
    # Run autoregressive prediction
    warmup_steps = T // 2
    num_rollout_steps = min(T - warmup_steps - 1, 20)
    
    if num_rollout_steps <= 0:
        print(f"  Not enough timesteps for visualization")
        return
    
    for t in range(warmup_steps):
        graph = batched_graph_sequence[t]
        _ = predict_single_step(model, graph, device)
    
    initial_graph = batched_graph_sequence[warmup_steps]
    agent_ids_init = initial_graph.agent_id.cpu().numpy()
    positions_init = initial_graph.pos.cpu().numpy()
    
    predictions, _ = autoregressive_rollout(model, initial_graph, num_rollout_steps, device)
    
    # Build predicted trajectories
    pred_positions = {agent_id: [] for agent_id in persistent_agent_ids}
    
    for agent_id in persistent_agent_ids:
        idx = np.where(agent_ids_init == agent_id)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        
        current_pos = positions_init[idx].copy()
        pred_positions[agent_id].append(current_pos.copy())
        
        for step_pred in predictions:
            pred_np = step_pred.cpu().numpy()
            if idx < len(pred_np):
                # Model outputs NORMALIZED displacement, multiply by POSITION_SCALE to get meters
                displacement_normalized = pred_np[idx]
                displacement_meters = displacement_normalized * POSITION_SCALE
                current_pos = current_pos + displacement_meters
                pred_positions[agent_id].append(current_pos.copy())
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(persistent_agent_ids))))
    agent_colors = {agent_id: colors[i % len(colors)] for i, agent_id in enumerate(persistent_agent_ids)}
    
    total_error = 0
    num_agents_plotted = 0
    
    for agent_id in persistent_agent_ids:
        gt_traj = np.array(gt_positions[agent_id])
        pred_traj = np.array(pred_positions[agent_id])
        
        if len(gt_traj) < 2 or len(pred_traj) < 2:
            continue
        
        color = agent_colors[agent_id]
        
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], '-', color=color, linewidth=1.5,
                label=f'Agent {agent_id} GT' if num_agents_plotted < 5 else None)
        
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], '--', color=color, linewidth=1.5,
                label=f'Agent {agent_id} Pred' if num_agents_plotted < 5 else None)
        
        # Mark start and end points of ground truth
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], color=color, marker='o', s=30, zorder=5)
        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color=color, marker='x', s=30, zorder=5)
        
        # Mark predicted trajectory endpoint (same color as prediction, square marker)
        ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], color=color, marker='s', s=40, 
                   edgecolors='black', linewidths=0.5, zorder=6)
        
        gt_segment = np.array(gt_positions[agent_id][warmup_steps:warmup_steps+len(pred_traj)])
        if len(gt_segment) > 0 and len(pred_traj) > 0:
            min_len = min(len(gt_segment), len(pred_traj))
            errors = np.linalg.norm(gt_segment[:min_len] - pred_traj[:min_len], axis=1)
            agent_ade = np.mean(errors)
            total_error += agent_ade
            num_agents_plotted += 1
    
    avg_error = total_error / max(1, num_agents_plotted)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'GAT Test Scenario {scenario_id}\n'
                 f'{num_agents_plotted} agents, {num_rollout_steps} rollout steps\n'
                 f'Average Displacement Error: {avg_error:.2f}m')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, f'gat_test_scenario_{scenario_idx:04d}_{scenario_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {save_path} (ADE: {avg_error:.2f}m)")
    return avg_error


def run_testing(test_dataset_path=test_hdf5_path,
                checkpoint_path=None,
                num_rollout_steps=20,
                max_scenarios=None,
                visualize=True,
                visualize_max=10,
                use_wandb=True):
    """Run testing with autoregressive multi-step prediction using GAT model.
    
    Args:
        test_dataset_path: Path to test HDF5 file
        checkpoint_path: Path to model checkpoint (default: auto-detect best)
        num_rollout_steps: Number of autoregressive rollout steps
        max_scenarios: Maximum scenarios to evaluate (None = all)
        visualize: Whether to generate visualizations
        visualize_max: Maximum scenarios to visualize
        use_wandb: Whether to log results to wandb
    """
    print("\n" + "="*70)
    print("GAT MODEL TESTING - Autoregressive Multi-Step Prediction")
    print("="*70)
    
    # Print GPU info
    print_gpu_info()
    
    # Initialize wandb
    if use_wandb:
        wandb.login()
        run = wandb.init(
            project=project_name,
            name=f"gat-testing-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": "gat",
                "task": "testing",
                "num_rollout_steps": num_rollout_steps,
                "max_scenarios": max_scenarios,
                "test_dataset": test_dataset_path,
            },
            tags=["gat", "testing", "autoregressive"]
        )
    
    # Use autoregressive checkpoint by default
    if checkpoint_path is None:
        # Try autoregressive checkpoint first with new naming pattern, then legacy names
        # New pattern: best_gat_autoreg_{steps}step_B{batch}_{strategy}_E{epochs}.pt
        import glob
        autoreg_pattern = os.path.join(gat_checkpoint_dir_autoreg, 'best_gat_autoreg_*.pt')
        autoreg_matches = glob.glob(autoreg_pattern)
        
        # Legacy paths
        legacy_path = os.path.join(gat_checkpoint_dir_autoreg, 'best_autoregressive_20step.pt')
        root_autoreg_path = os.path.join('checkpoints', 'best_autoregressive_20step.pt')
        root_autoreg_50step = os.path.join('checkpoints', 'best_autoregressive_50step.pt')
        base_path = os.path.join(gat_checkpoint_dir, 'best_model.pt')
        root_base_path = os.path.join('checkpoints', 'gat', 'best_model.pt')
        
        if autoreg_matches:
            # Use most recent autoregressive checkpoint
            checkpoint_path = max(autoreg_matches, key=os.path.getmtime)
            print(f"Using GAT autoregressive checkpoint: {checkpoint_path}")
        elif os.path.exists(legacy_path):
            checkpoint_path = legacy_path
            print(f"Using legacy GAT autoregressive checkpoint: {legacy_path}")
        elif os.path.exists(root_autoreg_50step):
            checkpoint_path = root_autoreg_50step
            print(f"Using GAT autoregressive checkpoint: {root_autoreg_50step}")
        elif os.path.exists(root_autoreg_path):
            checkpoint_path = root_autoreg_path
            print(f"Using GAT autoregressive checkpoint: {root_autoreg_path}")
        elif os.path.exists(base_path):
            checkpoint_path = base_path
            print(f"Using GAT base single-step checkpoint: {base_path}")
        elif os.path.exists(root_base_path):
            checkpoint_path = root_base_path
            print(f"Using GAT base single-step checkpoint: {root_base_path}")
        else:
            print(f"ERROR: No GAT checkpoint found!")
            print(f"  Checked pattern: {autoreg_pattern}")
            print(f"  Checked: {legacy_path}")
            print(f"  Checked: {base_path}")
            if use_wandb:
                wandb.finish(exit_code=1)
            return None
    
    # Load model
    model, checkpoint = load_trained_model(checkpoint_path, device)
    
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
    print(f"Loaded test dataset: {len(test_dataset)} scenarios")
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=gat_num_workers,
        collate_fn=collate_graph_sequences_to_batch,
        drop_last=False
    )
    
    effective_rollout = min(num_rollout_steps, sequence_length - 2)
    
    print(f"\nGAT Test Configuration:")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Rollout steps: {effective_rollout} ({effective_rollout * 0.1:.1f}s)")
    print(f"  Max scenarios: {max_scenarios if max_scenarios else 'all'}")
    print(f"  W&B logging: {use_wandb}")
    
    # Create visualization directory
    os.makedirs(gat_viz_dir_testing, exist_ok=True)
    
    # Run evaluation with visualization
    viz_images = []
    if visualize:
        print(f"\nGenerating visualizations (max {visualize_max} scenarios)...")
        viz_count = 0
        total_ade = 0
        
        for batch_idx, batch_dict in enumerate(test_dataloader):
            if viz_count >= visualize_max:
                break
            
            ade = visualize_test_scenario(model, batch_dict, batch_idx, gat_viz_dir_testing, device)
            if ade is not None:
                total_ade += ade
                viz_count += 1
                
                # Log visualization to wandb
                if use_wandb:
                    scenario_ids = batch_dict.get("scenario_ids", [])
                    scenario_id = scenario_ids[0] if scenario_ids else f"scenario_{batch_idx}"
                    save_path = os.path.join(gat_viz_dir_testing, f'gat_test_scenario_{batch_idx:04d}_{scenario_id}.png')
                    if os.path.exists(save_path):
                        viz_images.append(wandb.Image(save_path, caption=f"Scenario {scenario_id} (ADE: {ade:.2f}m)"))
        
        if viz_count > 0:
            print(f"\nVisualization Summary:")
            print(f"  Scenarios visualized: {viz_count}")
            print(f"  Average ADE: {total_ade / viz_count:.2f}m")
            print(f"  Saved to: {gat_viz_dir_testing}")
            
            if use_wandb and viz_images:
                wandb.log({"test_visualizations": viz_images})
    
    # Run full evaluation
    results = evaluate_autoregressive(
        model, test_dataloader, effective_rollout, device, max_scenarios
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
    results_path = os.path.join(gat_checkpoint_dir, 'gat_test_results.pt')
    os.makedirs(gat_checkpoint_dir, exist_ok=True)
    torch.save(results, results_path)
    print(f"Results saved to {results_path}")
    
    # Also save as JSON for easy reading
    json_results = {
        'model_type': 'gat',
        'checkpoint': checkpoint_path,
        'test_dataset': test_dataset_path,
        'num_scenarios': len(results.get('scenarios', [])),
        'horizons': results.get('horizons', {}),
        'timestamp': datetime.now().isoformat()
    }
    json_path = os.path.join(gat_checkpoint_dir_autoreg, 'gat_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to {json_path}")
    
    if use_wandb:
        wandb.finish()
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GAT Model Testing')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: auto-detect)')
    parser.add_argument('--test_data', type=str, default=test_hdf5_path,
                        help='Path to test HDF5 file')
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
        use_wandb=not args.no_wandb
    )
    
    return results


if __name__ == '__main__':
    main()
