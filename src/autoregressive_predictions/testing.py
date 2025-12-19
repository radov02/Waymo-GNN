import sys
import os
# Add parent directory (src/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set matplotlib backend to non-interactive before importing pyplot
# This prevents Tkinter threading errors when saving plots
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import torch.nn.functional as F
from SpatioTemporalGNN import SpatioTemporalGNN
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, num_gru_layers,
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, checkpoint_dir, checkpoint_dir_autoreg, use_edge_weights,
                    num_gpus, use_data_parallel, setup_model_parallel, load_model_state,
                    autoreg_viz_dir)
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from helper_functions.helpers import compute_metrics
from helper_functions.visualization_functions import (load_scenario_by_id, draw_map_features,
                                                       MAP_FEATURE_GRAY, VIBRANT_COLORS, SDC_COLOR)
import matplotlib.pyplot as plt

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/autoregressive_predictions/testing.py

# Constants
POSITION_SCALE = 100.0  # For denormalizing displacements


def load_trained_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model with saved config
    config = checkpoint['config']
    model = SpatioTemporalGNN(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_channels'],
        output_dim=config.get('output_dim', 2),
        num_gcn_layers=config['num_layers'],
        num_gru_layers=config.get('num_gru_layers', 1),
        dropout=config['dropout'],
        use_gat=config.get('use_gat', False)  # Backward compat with old checkpoints
    )
    
    # Setup multi-GPU if available
    model, is_parallel = setup_model_parallel(model, device)
    
    # Load weights (handle DataParallel prefix)
    load_model_state(model, checkpoint['model_state_dict'], is_parallel)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    if 'train_loss' in checkpoint:
        print(f"  Training loss: {checkpoint['train_loss']:.6f}")
    
    return model, checkpoint

def predict_single_step(model, graph, device, edge_weights=None):
    """
    Predict one timestep ahead for all agents in a graph.
    
    Args:
        model: Trained EvolveGCNH model
        graph: PyG Data object with current state
        device: Torch device
        edge_weights: Optional edge weights
    
    Returns:
        predictions: [N, 2] tensor of predicted displacements
    """
    graph = graph.to(device)
    
    with torch.no_grad():
        edge_w = graph.edge_attr if edge_weights and hasattr(graph, 'edge_attr') else None
        predictions = model(
            graph.x, 
            graph.edge_index,
            edge_weight=edge_w,
            batch=graph.batch,
            batch_size=1,
            batch_num=0,
            timestep=0
        )
    
    return predictions


def autoregressive_rollout(model, initial_graph, num_steps, device, edge_weights=None):
    """
    Autoregressive multi-step prediction: predict future trajectory by iteratively
    feeding predictions back as input.
    
    This is the core of trajectory forecasting:
    1. Start with current state at time t
    2. Predict displacement for t+1
    3. Update node positions and velocities
    4. Use updated state to predict t+2
    5. Repeat for num_steps
    
    Args:
        model: Trained SpatioTemporalGNN model
        initial_graph: PyG Data object at time t
        num_steps: Number of future timesteps to predict
        device: Torch device
        edge_weights: Whether to use edge weights
    
    Returns:
        all_predictions: List of [N, 2] predictions for each step
        updated_graphs: List of graph states (for visualization)
    """
    model.eval()
    current_graph = initial_graph.clone().to(device)
    
    all_predictions = []
    updated_graphs = [current_graph.cpu()]
    
    for step in range(num_steps):
        # Predict next displacement
        pred_displacement = predict_single_step(model, current_graph, device, edge_weights)
        all_predictions.append(pred_displacement.cpu())
        
        # Update graph state for next prediction
        # This is the autoregressive part: we use our prediction as input for the next step
        current_graph = update_graph_features(current_graph, pred_displacement, device)
        updated_graphs.append(current_graph.cpu())
    
    return all_predictions, updated_graphs


def update_graph_features(graph, pred_displacement, device):
    """
    Update node features based on predicted displacement.
    This simulates the agent's state at the next timestep.
    
    Key updates:
    - Position (pos): add predicted displacement
    - Velocity (features 0-1): compute from displacement / dt
    - Speed (feature 2): magnitude of velocity
    - Heading (feature 3): direction of movement
    - Acceleration (features 5-6): velocity change
    
    Args:
        graph: Current graph state
        pred_displacement: [N, 2] predicted displacements
        device: Torch device
    
    Returns:
        updated_graph: Graph with updated features
    """
    # Clone to avoid modifying original
    updated_graph = graph.clone()
    
    # Update positions (stored separately in graph.pos)
    if hasattr(updated_graph, 'pos'):
        # Displacements are normalized by 100.0 in dataset, so denormalize
        updated_graph.pos = updated_graph.pos + pred_displacement * 100.0
    
    # Update node features
    # Feature indices: [0-1: vx, vy, 2: speed, 3: heading, 4: valid, 5-6: ax, ay, 
    #                   7-8: rel_pos_sdc, 9: dist_sdc, 10: dist_nearest, 11-14: type]
    dt = 0.1  # 0.1 second timestep
    
    # Calculate new velocity from displacement
    new_vx = pred_displacement[:, 0] / dt  # already normalized
    new_vy = pred_displacement[:, 1] / dt
    
    # Calculate acceleration (change in velocity)
    old_vx = updated_graph.x[:, 0]
    old_vy = updated_graph.x[:, 1]
    ax = (new_vx - old_vx) / dt / 10.0  # normalize acceleration
    ay = (new_vy - old_vy) / dt / 10.0
    
    # Update velocity features
    updated_graph.x[:, 0] = new_vx
    updated_graph.x[:, 1] = new_vy
    
    # Update speed
    updated_graph.x[:, 2] = torch.sqrt(new_vx**2 + new_vy**2)
    
    # Update heading (direction of movement)
    updated_graph.x[:, 3] = torch.atan2(new_vy, new_vx) / np.pi  # normalize to [-1, 1]
    
    # Update acceleration
    updated_graph.x[:, 5] = ax
    updated_graph.x[:, 6] = ay
    
    # Note: Relative features (7-10) and type (11-14) remain unchanged
    # In a full simulation, you'd recalculate relative positions to SDC and nearest neighbors
    
    return updated_graph


def evaluate_autoregressive(model, dataloader, num_rollout_steps, device, max_scenarios=None):
    """
    Evaluate model using autoregressive rollout on test set.
    
    For each scenario:
    1. Use first part of sequence as context (warm-up GRU states)
    2. From a certain timestep, start autoregressive prediction
    3. Compare predictions with ground truth
    4. Compute metrics at different horizons (1s, 3s, 5s, 8s)
    
    Args:
        model: Trained model
        dataloader: DataLoader for test set
        num_rollout_steps: How many steps to predict autoregressively
        device: Torch device
        max_scenarios: Maximum scenarios to evaluate (None = all)
    
    Returns:
        results: Dictionary with metrics
    """
    model.eval()
    
    # Metrics at different horizons (in timesteps: 1s=10 steps, 3s=30, 5s=50, 8s=80)
    horizons = [10, 30, 50, min(80, num_rollout_steps)]
    horizon_metrics = {h: {'ade': [], 'fde': [], 'angle_error': [], 'cosine_sim': []} for h in horizons}
    
    all_scenario_results = []
    scenario_count = 0
    
    print(f"\nEvaluating with {num_rollout_steps}-step autoregressive rollout...")
    print(f"Horizons: {[f'{h*0.1}s' for h in horizons]}")
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            if max_scenarios and scenario_count >= max_scenarios:
                break
            
            batched_graph_sequence = batch_dict["batch"]
            B = batch_dict["B"]
            T = batch_dict["T"]
            scenario_ids = batch_dict.get("scenario_ids", [])
            
            assert B == 1, f"batch_size must be 1 for evaluation! Got B={B}"
            
            # Move graphs to device
            for t in range(T):
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
            
            # Reset GRU hidden states for this scenario (per-agent)
            num_nodes = batched_graph_sequence[0].num_nodes
            model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
            
            # Warm-up: process first half of sequence to initialize GRU states
            warmup_steps = T // 2
            print(f"Scenario {scenario_count+1} ({scenario_ids[0] if scenario_ids else 'unknown'}): "
                  f"Warming up with {warmup_steps} steps, then rolling out {num_rollout_steps} steps...")
            
            for t in range(warmup_steps):
                graph = batched_graph_sequence[t]
                _ = predict_single_step(model, graph, device, use_edge_weights)
            
            # Start autoregressive rollout from warmup_steps
            if warmup_steps + num_rollout_steps > T:
                num_rollout_steps = T - warmup_steps - 1
            
            if num_rollout_steps <= 0:
                print(f"  Skipping: not enough timesteps for rollout")
                continue
            
            initial_graph = batched_graph_sequence[warmup_steps]
            predictions, _ = autoregressive_rollout(
                model, initial_graph, num_rollout_steps, device, use_edge_weights
            )
            
            # Collect ground truth
            ground_truths = []
            for step in range(1, num_rollout_steps + 1):
                if warmup_steps + step < T:
                    gt_graph = batched_graph_sequence[warmup_steps + step]
                    ground_truths.append(gt_graph.y.cpu())
                else:
                    break
            
            # Compute metrics at different horizons
            scenario_metrics = {}
            for horizon in horizons:
                if horizon <= len(predictions) and horizon <= len(ground_truths):
                    # Average Displacement Error (ADE): mean error across all timesteps up to horizon
                    ade = 0.0
                    for step in range(horizon):
                        pred = predictions[step] * 100.0  # denormalize
                        gt = ground_truths[step] * 100.0
                        displacement_error = torch.norm(pred - gt, dim=1).mean()
                        ade += displacement_error.item()
                    ade /= horizon
                    
                    # Final Displacement Error (FDE): error at the horizon
                    pred_final = predictions[horizon-1] * 100.0
                    gt_final = ground_truths[horizon-1] * 100.0
                    fde = torch.norm(pred_final - gt_final, dim=1).mean().item()
                    
                    # Angle error at horizon
                    pred_angle = torch.atan2(predictions[horizon-1][:, 1], predictions[horizon-1][:, 0])
                    gt_angle = torch.atan2(ground_truths[horizon-1][:, 1], ground_truths[horizon-1][:, 0])
                    angle_diff = torch.atan2(torch.sin(pred_angle - gt_angle), torch.cos(pred_angle - gt_angle))
                    angle_error = torch.abs(angle_diff).mean().item() * 180 / np.pi
                    
                    # Cosine similarity
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
            
            # Print scenario metrics
            print(f"  Metrics:")
            for h in horizons:
                if h in scenario_metrics.get(f'{h*0.1}s', {}):
                    m = scenario_metrics[f'{h*0.1}s']
                    print(f"    {h*0.1}s: ADE={m['ade']:.2f}m, FDE={m['fde']:.2f}m, "
                          f"Angle={m['angle_error']:.1f}°, Cos={m['cosine_sim']:.3f}")
            
            scenario_count += 1
    
    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATED TEST RESULTS")
    print("="*80)
    
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
            print(f"  ADE: {r['ade_mean']:.2f} ± {r['ade_std']:.2f} m")
            print(f"  FDE: {r['fde_mean']:.2f} ± {r['fde_std']:.2f} m")
            print(f"  Angle Error: {r['angle_error_mean']:.1f} ± {r['angle_error_std']:.1f}°")
            print(f"  Cosine Sim: {r['cosine_sim_mean']:.3f} ± {r['cosine_sim_std']:.3f}")
    
    print("\n" + "="*80)
    
    return results


def visualize_test_scenario(model, batch_dict, scenario_idx, save_dir, device):
    """
    Visualize predictions vs ground truth for a test scenario.
    
    Creates a single plot showing all persistent agents with:
    - Ground truth trajectories (solid lines)
    - Predicted trajectories (dashed lines)
    - Error metrics printed
    
    Args:
        model: Trained model
        batch_dict: Batch dictionary from dataloader
        scenario_idx: Index for filename
        save_dir: Directory to save visualization
        device: Torch device
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batched_graph_sequence = batch_dict["batch"]
    T = batch_dict["T"]
    scenario_ids = batch_dict.get("scenario_ids", [])
    scenario_id = scenario_ids[0] if scenario_ids else f"scenario_{scenario_idx}"
    
    # Move graphs to device
    for t in range(T):
        batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
    
    # Reset GRU hidden states
    num_nodes = batched_graph_sequence[0].num_nodes
    model.reset_gru_hidden_states(num_agents=num_nodes, device=device)
    
    # Find persistent agents (exist at ALL timesteps)
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
    
    # Collect ground truth positions for persistent agents
    gt_positions = {agent_id: [] for agent_id in persistent_agent_ids}
    
    for t in range(T):
        graph = batched_graph_sequence[t]
        agent_ids_t = graph.agent_id.cpu().numpy()
        positions_t = graph.pos.cpu().numpy()
        
        for agent_id in persistent_agent_ids:
            idx = np.where(agent_ids_t == agent_id)[0]
            if len(idx) > 0:
                gt_positions[agent_id].append(positions_t[idx[0]])
    
    # Run autoregressive prediction from middle of sequence
    warmup_steps = T // 2
    num_rollout_steps = min(T - warmup_steps - 1, 20)  # Max 20 steps (2s)
    
    if num_rollout_steps <= 0:
        print(f"  Not enough timesteps for visualization")
        return
    
    # Warm up GRU states
    for t in range(warmup_steps):
        graph = batched_graph_sequence[t]
        _ = predict_single_step(model, graph, device, use_edge_weights)
    
    # Get initial positions at warmup point
    initial_graph = batched_graph_sequence[warmup_steps]
    agent_ids_init = initial_graph.agent_id.cpu().numpy()
    positions_init = initial_graph.pos.cpu().numpy()
    
    # Run autoregressive rollout
    predictions, _ = autoregressive_rollout(
        model, initial_graph, num_rollout_steps, device, use_edge_weights
    )
    
    # Build predicted trajectories for persistent agents
    pred_positions = {agent_id: [] for agent_id in persistent_agent_ids}
    
    for agent_id in persistent_agent_ids:
        idx = np.where(agent_ids_init == agent_id)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        
        # Start from initial position
        current_pos = positions_init[idx].copy()
        pred_positions[agent_id].append(current_pos.copy())
        
        # Accumulate predictions
        for step_pred in predictions:
            pred_np = step_pred.cpu().numpy()
            if idx < len(pred_np):
                displacement = pred_np[idx] * POSITION_SCALE
                current_pos = current_pos + displacement
                pred_positions[agent_id].append(current_pos.copy())
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Assign colors to agents
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
        
        # Plot full ground truth trajectory
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], '-', color=color, linewidth=1.5,
                label=f'Agent {agent_id} GT' if num_agents_plotted < 5 else None)
        
        # Plot predicted trajectory (starting from warmup point)
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], '--', color=color, linewidth=1.5,
                label=f'Agent {agent_id} Pred' if num_agents_plotted < 5 else None)
        
        # Mark start and end points of ground truth
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], color=color, marker='o', s=30, zorder=5)
        ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color=color, marker='x', s=30, zorder=5)
        
        # Mark predicted trajectory endpoint (same color as prediction, square marker)
        ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], color=color, marker='s', s=40, 
                   edgecolors='black', linewidths=0.5, zorder=6)
        
        # Calculate error for the overlapping part
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
    ax.set_title(f'Test Scenario {scenario_id}\n'
                 f'{num_agents_plotted} agents, {num_rollout_steps} rollout steps\n'
                 f'Average Displacement Error: {avg_error:.2f}m')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Save
    save_path = os.path.join(save_dir, f'test_scenario_{scenario_idx:04d}_{scenario_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {save_path} (ADE: {avg_error:.2f}m)")
    return avg_error


def run_testing(test_dataset_path="./data/graphs/testing/testing_seqlen90.hdf5",
                checkpoint_path=None,
                num_rollout_steps=20,  # 2 seconds (limited by data)
                max_scenarios=None,
                visualize=True,
                visualize_max=10):
    """
    Run testing with autoregressive multi-step prediction.
    
    Args:
        test_dataset_path: Path to test HDF5 file
        checkpoint_path: Path to model checkpoint (default: best autoregressive model)
        num_rollout_steps: Number of steps to predict (default: 20 = 2s, max ~19 for T=29)
        max_scenarios: Max scenarios to evaluate (None = all)
        visualize: Whether to generate visualizations
        visualize_max: Max number of scenarios to visualize
    """
    
    # Use autoregressive checkpoint by default
    if checkpoint_path is None:
        # Try autoregressive checkpoint first, fall back to base checkpoint
        autoreg_path = os.path.join(checkpoint_dir_autoreg, 'finetuned_scheduled_sampling_best.pt')
        base_path = os.path.join(checkpoint_dir, 'best_model.pt')
        
        if os.path.exists(autoreg_path):
            checkpoint_path = autoreg_path
            print(f"Using autoregressive fine-tuned checkpoint")
        elif os.path.exists(base_path):
            checkpoint_path = base_path
            print(f"Using base single-step checkpoint")
        else:
            print(f"ERROR: No checkpoint found!")
            print(f"  Checked: {autoreg_path}")
            print(f"  Checked: {base_path}")
            exit(1)
    
    # Load model
    model, checkpoint = load_trained_model(checkpoint_path, device)
    
    # Load test dataset
    try:
        test_dataset = HDF5ScenarioDataset(test_dataset_path, seq_len=sequence_length)
        print(f"\n✓ Loaded test dataset: {len(test_dataset)} scenarios")
    except FileNotFoundError:
        print(f"ERROR: {test_dataset_path} not found!")
        print("Please run graph_creation_and_saving.py to create test data.")
        exit(1)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # Must be 1 for per-scenario temporal processing
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_sequences_to_batch,
        drop_last=False
    )
    
    # Cap rollout steps based on sequence length
    effective_rollout = min(num_rollout_steps, sequence_length - 2)
    
    # Check if model was trained with GAT (for backward compatibility with old checkpoints)
    model_uses_gat = checkpoint.get('config', {}).get('use_gat', False)
    
    print(f"\nTest Configuration:")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Rollout steps: {effective_rollout} ({effective_rollout * 0.1:.1f}s)")
    print(f"  Max scenarios: {max_scenarios if max_scenarios else 'all'}")
    print(f"  Edge weights: {use_edge_weights}")
    if model_uses_gat:
        print(f"  Note: Loaded old GAT checkpoint - consider using gat_autoregressive/testing_gat.py")
    
    # Create visualization directory (GCN model uses autoreg, not autoreg/gat)
    viz_dir = os.path.join('visualizations/autoreg', 'testing')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Run evaluation with visualization
    if visualize:
        print(f"\nGenerating visualizations (max {visualize_max} scenarios)...")
        viz_count = 0
        total_ade = 0
        
        for batch_idx, batch_dict in enumerate(test_dataloader):
            if viz_count >= visualize_max:
                break
            
            ade = visualize_test_scenario(model, batch_dict, batch_idx, viz_dir, device)
            if ade is not None:
                total_ade += ade
                viz_count += 1
        
        if viz_count > 0:
            print(f"\nVisualization Summary:")
            print(f"  Scenarios visualized: {viz_count}")
            print(f"  Average ADE: {total_ade / viz_count:.2f}m")
            print(f"  Saved to: {viz_dir}")
    
    # Run full evaluation
    results = evaluate_autoregressive(
        model, test_dataloader, effective_rollout, device, max_scenarios
    )
    
    # Save results
    results_path = os.path.join(checkpoint_dir_autoreg, 'test_results.pt')
    os.makedirs(checkpoint_dir_autoreg, exist_ok=True)
    torch.save(results, results_path)
    print(f"\n✓ Results saved to {results_path}")
    
    return results


if __name__ == '__main__':
    # Test on a small number of scenarios first
    results = run_testing(
        test_dataset_path="./data/graphs/testing/testing_seqlen90.hdf5",
        checkpoint_path=None,  # Auto-selects best autoregressive checkpoint
        num_rollout_steps=20,  # 2 seconds ahead (limited by sequence length)
        max_scenarios=10,       # Test on 10 scenarios first, set to None for all
        visualize=True,
        visualize_max=5
    )
