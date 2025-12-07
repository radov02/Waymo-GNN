import torch
import os
import numpy as np
import torch.nn.functional as F
from EvolveGCNH import EvolveGCNH
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, topk,
                    radius, input_dim, sequence_length, hidden_channels,
                    dropout, checkpoint_dir, use_edge_weights)
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from helper_functions.helpers import compute_metrics
import matplotlib.pyplot as plt

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/testing.py


def load_trained_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model with saved config
    config = checkpoint['config']
    model = EvolveGCNH(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_channels'],
        output_dim=config['output_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        topk=config['topk']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
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
        model: Trained EvolveGCNH model
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
            
            # Reset GRU hidden states for this scenario
            model.reset_gru_hidden_states(batch_size=1)
            
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


def run_testing(test_dataset_path="./data/graphs/testing/testing.hdf5",
                checkpoint_path=None,
                num_rollout_steps=80,  # 8 seconds
                max_scenarios=None):
    """
    Run testing with autoregressive multi-step prediction.
    
    Args:
        test_dataset_path: Path to test HDF5 file
        checkpoint_path: Path to model checkpoint (default: best_model.pt)
        num_rollout_steps: Number of steps to predict (default: 80 = 8 seconds)
        max_scenarios: Max scenarios to evaluate (None = all)
    """
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
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
        batch_size=1,  # Must be 1 for EvolveGCN
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_sequences_to_batch,
        drop_last=False
    )
    
    print(f"\nTest Configuration:")
    print(f"  Device: {device}")
    print(f"  Rollout steps: {num_rollout_steps} ({num_rollout_steps * 0.1}s)")
    print(f"  Max scenarios: {max_scenarios if max_scenarios else 'all'}")
    print(f"  Edge weights: {use_edge_weights}")
    
    # Run evaluation
    results = evaluate_autoregressive(
        model, test_dataloader, num_rollout_steps, device, max_scenarios
    )
    
    # Save results
    results_path = os.path.join(checkpoint_dir, 'test_results.pt')
    torch.save(results, results_path)
    print(f"\n✓ Results saved to {results_path}")
    
    return results


if __name__ == '__main__':
    # Test on a small number of scenarios first
    results = run_testing(
        test_dataset_path="./data/graphs/testing/testing.hdf5",
        checkpoint_path=None,  # Uses best_model.pt
        num_rollout_steps=80,  # 8 seconds ahead
        max_scenarios=5  # Test on 5 scenarios first, set to None for all
    )
