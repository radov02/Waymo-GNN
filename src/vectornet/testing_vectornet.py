"""VectorNet Multi-Step Testing Script for Waymo Open Motion Dataset.

This script evaluates trained VectorNet models that predict full trajectories
at once (multi-step, not autoregressive).

Features:
- Load trained VectorNet checkpoints
- Evaluate on test set with multi-step prediction
- Compute trajectory prediction metrics (ADE, FDE, MR)
- Generate trajectory visualizations
- Log metrics to Weights & Biases
- Save results to file

Usage:
    python src/vectornet/testing_vectornet.py
    python src/vectornet/testing_vectornet.py --checkpoint path/to/model.pt
    python src/vectornet/testing_vectornet.py --max_scenarios 50 --no_wandb
"""

import sys
import os

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, Subset
import random

# Import VectorNet modules
from VectorNet import VectorNetMultiStep
from vectornet_dataset import VectorNetDataset, collate_vectornet_batch, worker_init_fn

# Import configuration from centralized config.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    device, vectornet_num_workers, pin_memory, print_gpu_info, num_workers,
    vectornet_input_dim, vectornet_hidden_dim, vectornet_output_dim,
    vectornet_num_polyline_layers, vectornet_num_global_layers,
    vectornet_num_heads, vectornet_dropout, project_name,
    vectornet_prediction_horizon, vectornet_history_length,
    vectornet_use_node_completion, vectornet_node_completion_ratio,
    vectornet_checkpoint_dir, vectornet_viz_dir,
    use_gradient_checkpointing
)


# ============== TESTING CONFIGURATION ==============
TEST_HDF5 = 'data/graphs/testing/testing.hdf5'
DEFAULT_CHECKPOINT = os.path.join(vectornet_checkpoint_dir, 'best_vectornet_model.pt')
VIZ_DIR = 'visualizations/autoreg/vectornet/testing'

# Constants
POSITION_SCALE = 100.0  # For denormalizing displacements


def get_vectornet_config():
    """Get default VectorNet configuration."""
    return {
        'input_dim': vectornet_input_dim,
        'hidden_dim': vectornet_hidden_dim,
        'output_dim': vectornet_output_dim,
        'prediction_horizon': vectornet_prediction_horizon,
        'num_polyline_layers': vectornet_num_polyline_layers,
        'num_global_layers': vectornet_num_global_layers,
        'num_heads': vectornet_num_heads,
        'dropout': vectornet_dropout,
        'history_length': vectornet_history_length,
    }


def get_module(model):
    """Get underlying model (unwrap DataParallel/compiled)."""
    if hasattr(model, 'module'):
        return model.module
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def load_model(checkpoint_path, device):
    """Load trained VectorNet model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device
        
    Returns:
        Loaded model
    """
    print(f"\nLoading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', get_vectornet_config())
    
    # Create model
    model = VectorNetMultiStep(
        input_dim=config.get('input_dim', vectornet_input_dim),
        hidden_dim=config.get('hidden_dim', vectornet_hidden_dim),
        output_dim=config.get('output_dim', vectornet_output_dim),
        prediction_horizon=config.get('prediction_horizon', vectornet_prediction_horizon),
        num_polyline_layers=config.get('num_polyline_layers', vectornet_num_polyline_layers),
        num_global_layers=config.get('num_global_layers', vectornet_num_global_layers),
        num_heads=config.get('num_heads', vectornet_num_heads),
        dropout=config.get('dropout', vectornet_dropout),
        use_node_completion=False  # No aux task during testing
    )
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    
    # Handle DataParallel prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Handle torch.compile prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    print(f"  Loaded model from epoch {epoch}, val_loss={val_loss}")
    
    return model, config


def compute_trajectory_metrics(predictions, targets):
    """Compute trajectory prediction metrics.
    
    Args:
        predictions: [N, T, 2] predicted trajectories (displacements per step)
        targets: [N, T, 2] ground truth trajectories
        
    Returns:
        Dictionary of metrics
    """
    # Ensure same shape
    min_t = min(predictions.shape[1], targets.shape[1])
    predictions = predictions[:, :min_t, :]
    targets = targets[:, :min_t, :]
    
    # Convert displacements to cumulative positions
    pred_pos = torch.cumsum(predictions, dim=1)
    target_pos = torch.cumsum(targets, dim=1)
    
    # Compute per-timestep displacement errors
    errors = torch.sqrt(((pred_pos - target_pos) ** 2).sum(dim=-1))  # [N, T]
    
    # ADE: Average Displacement Error (mean over all timesteps)
    ade = errors.mean()
    
    # FDE: Final Displacement Error (error at last timestep)
    fde = errors[:, -1].mean()
    
    # Miss Rate: percentage of endpoints with error > 2.0 meters
    miss_rate = (errors[:, -1] > 2.0).float().mean()
    
    # Per-timestep ADE
    ade_per_step = errors.mean(dim=0)  # [T]
    
    return {
        'ade': ade.item(),
        'fde': fde.item(),
        'miss_rate': miss_rate.item(),
        'ade_per_step': ade_per_step.cpu().numpy().tolist(),
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device, prediction_horizon=50):
    """Evaluate model on test set.
    
    Args:
        model: VectorNet model
        dataloader: Test data loader
        device: Target device
        prediction_horizon: Number of timesteps to predict
        
    Returns:
        Dictionary of aggregate metrics
    """
    model.eval()
    base_model = get_module(model)
    
    all_predictions = []
    all_targets = []
    all_scenario_ids = []
    
    total_ade = 0.0
    total_fde = 0.0
    total_mr = 0.0
    num_batches = 0
    
    print("\n--- Evaluating ---")
    
    for batch_idx, batch_dict in enumerate(dataloader):
        batched_sequence = batch_dict["batch"]
        B = batch_dict["B"]
        T = batch_dict["T"]
        scenario_ids = batch_dict.get("scenario_ids", [None] * B)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}")
        
        # Move to device
        for t in range(T):
            batched_sequence[t] = batched_sequence[t].to(device, non_blocking=True)
        
        base_model.reset_temporal_buffer()
        
        # Determine history split
        history_len = min(T - 1, 10)
        if T <= 1:
            continue
        
        # Process history
        for t in range(history_len):
            data = batched_sequence[t]
            base_model.forward_single_step(
                data.x, data.edge_index,
                edge_weight=data.edge_attr,
                is_last_step=False
            )
        
        # Get predictions
        last_data = batched_sequence[history_len]
        predictions = base_model.forward_single_step(
            last_data.x, last_data.edge_index,
            edge_weight=last_data.edge_attr,
            is_last_step=True
        )
        
        if predictions is None:
            continue
        
        # Build targets
        future_len = min(T - history_len - 1, prediction_horizon)
        N = last_data.x.shape[0]
        
        if future_len <= 0:
            continue
        
        targets = torch.zeros(N, future_len, 2, device=device)
        for future_t in range(future_len):
            t_idx = history_len + 1 + future_t
            if t_idx < T:
                future_data = batched_sequence[t_idx]
                if future_data.y is not None and future_data.y.shape[0] >= N:
                    targets[:, future_t, :] = future_data.y[:N, :]
        
        predictions = predictions[:, :future_len, :]
        
        # Compute metrics
        metrics = compute_trajectory_metrics(predictions, targets)
        
        total_ade += metrics['ade']
        total_fde += metrics['fde']
        total_mr += metrics['miss_rate']
        num_batches += 1
        
        # Store for detailed analysis
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())
        all_scenario_ids.extend(scenario_ids)
    
    # Aggregate metrics
    avg_ade = total_ade / max(1, num_batches)
    avg_fde = total_fde / max(1, num_batches)
    avg_mr = total_mr / max(1, num_batches)
    
    results = {
        'ade': avg_ade,
        'fde': avg_fde,
        'miss_rate': avg_mr,
        'num_scenarios': len(all_scenario_ids),
        'num_batches': num_batches,
    }
    
    return results, all_predictions, all_targets


def visualize_predictions(predictions, targets, scenario_ids, save_dir, max_viz=10):
    """Generate visualization of predicted vs ground truth trajectories.
    
    Args:
        predictions: List of [N, T, 2] prediction tensors
        targets: List of [N, T, 2] target tensors
        scenario_ids: List of scenario IDs
        save_dir: Directory to save visualizations
        max_viz: Maximum number of visualizations to generate
    
    Returns:
        List of saved file paths and average ADE
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    total_ade = 0
    viz_count = 0
    
    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        if viz_count >= max_viz:
            break
        
        pred = pred.numpy() if isinstance(pred, torch.Tensor) else pred
        target = target.numpy() if isinstance(target, torch.Tensor) else target
        
        # Denormalize
        pred = pred * POSITION_SCALE
        target = target * POSITION_SCALE
        
        # Convert to cumulative positions
        pred_pos = np.cumsum(pred, axis=1)
        target_pos = np.cumsum(target, axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot up to 10 agents
        num_agents = min(pred_pos.shape[0], 10)
        colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
        
        agent_errors = []
        for agent_idx in range(num_agents):
            color = colors[agent_idx]
            
            # Ground truth
            ax.plot(target_pos[agent_idx, :, 0], target_pos[agent_idx, :, 1], 
                    '-', color=color, linewidth=2, label=f'Agent {agent_idx} GT' if agent_idx < 5 else None)
            
            # Prediction
            ax.plot(pred_pos[agent_idx, :, 0], pred_pos[agent_idx, :, 1], 
                    '--', color=color, linewidth=2, label=f'Agent {agent_idx} Pred' if agent_idx < 5 else None)
            
            # Mark endpoints
            ax.scatter(target_pos[agent_idx, -1, 0], target_pos[agent_idx, -1, 1], 
                      color=color, marker='x', s=50, zorder=5)
            ax.scatter(pred_pos[agent_idx, -1, 0], pred_pos[agent_idx, -1, 1], 
                      color=color, marker='s', s=50, edgecolors='black', linewidths=0.5, zorder=6)
            
            # Compute error
            error = np.linalg.norm(pred_pos[agent_idx] - target_pos[agent_idx], axis=1).mean()
            agent_errors.append(error)
        
        avg_error = np.mean(agent_errors) if agent_errors else 0
        total_ade += avg_error
        
        scenario_id = scenario_ids[idx] if idx < len(scenario_ids) else f"scenario_{idx}"
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'VectorNet Test Scenario {scenario_id}\n'
                     f'{num_agents} agents, {pred_pos.shape[1]} prediction steps\n'
                     f'Average Displacement Error: {avg_error:.2f}m')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, f'vectornet_test_scenario_{idx:04d}_{scenario_id}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        saved_paths.append((save_path, avg_error, scenario_id))
        viz_count += 1
        print(f"  Saved: {save_path} (ADE: {avg_error:.2f}m)")
    
    avg_ade = total_ade / max(1, viz_count)
    return saved_paths, avg_ade


def run_testing(test_dataset_path=TEST_HDF5,
                checkpoint_path=None,
                batch_size=32,
                max_scenarios=None,
                visualize=True,
                visualize_max=10,
                use_wandb=True):
    """Run testing with multi-step prediction using VectorNet model.
    
    Args:
        test_dataset_path: Path to test HDF5 file
        checkpoint_path: Path to model checkpoint (default: auto-detect best)
        batch_size: Batch size for evaluation
        max_scenarios: Maximum scenarios to evaluate (None = all)
        visualize: Whether to generate visualizations
        visualize_max: Maximum scenarios to visualize
        use_wandb: Whether to log results to wandb
    """
    print("\n" + "="*70)
    print("VECTORNET MODEL TESTING - Multi-Step Prediction")
    print("="*70)
    
    # Print GPU info
    print_gpu_info()
    
    # Initialize wandb
    if use_wandb:
        wandb.login()
        run = wandb.init(
            project=project_name,
            name=f"vectornet-testing-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_type": "vectornet",
                "task": "testing",
                "batch_size": batch_size,
                "max_scenarios": max_scenarios,
                "test_dataset": test_dataset_path,
            },
            tags=["vectornet", "testing", "multi-step"]
        )
    
    # Use default checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    
    # Check paths
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found at {checkpoint_path}")
        if use_wandb:
            wandb.finish(exit_code=1)
        return None
    
    if not os.path.exists(test_dataset_path):
        print(f"\nERROR: Test data not found at {test_dataset_path}")
        print("Trying alternative paths...")
        alt_paths = [
            'data/graphs/testing/testing.hdf5',
            'data/graphs/testing/scenario_graphs.h5',
            'data/graphs/validation/scenario_graphs.h5',
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                test_dataset_path = alt_path
                print(f"Found data at: {alt_path}")
                break
        else:
            print("No test data found. Please check data paths.")
            if use_wandb:
                wandb.finish(exit_code=1)
            return None
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    prediction_horizon = config.get('prediction_horizon', vectornet_prediction_horizon)
    history_length = config.get('history_length', vectornet_history_length)
    
    if use_wandb:
        wandb.config.update({
            "checkpoint_path": checkpoint_path,
            "prediction_horizon": prediction_horizon,
            "history_length": history_length,
        })
    
    # Load dataset
    print(f"\n--- Loading Test Data ---")
    print(f"Data: {test_dataset_path}")
    
    test_dataset = VectorNetDataset(
        test_dataset_path,
        seq_len=prediction_horizon + history_length,
        cache_in_memory=False
    )
    
    # Limit scenarios if needed
    if max_scenarios is not None and len(test_dataset) > max_scenarios:
        indices = list(range(max_scenarios))
        test_dataset = Subset(test_dataset, indices)
    
    print(f"Test scenarios: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_vectornet_batch,
        worker_init_fn=worker_init_fn,
        drop_last=False
    )
    
    print(f"\nVectorNet Test Configuration:")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Prediction horizon: {prediction_horizon} steps ({prediction_horizon * 0.1:.1f}s)")
    print(f"  History length: {history_length} steps ({history_length * 0.1:.1f}s)")
    print(f"  Max scenarios: {max_scenarios if max_scenarios else 'all'}")
    print(f"  W&B logging: {use_wandb}")
    
    # Evaluate
    results, predictions, targets = evaluate_model(
        model, test_loader, device,
        prediction_horizon=prediction_horizon
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("VECTORNET TEST RESULTS")
    print("=" * 60)
    print(f"  Average Displacement Error (ADE): {results['ade']:.4f}m")
    print(f"  Final Displacement Error (FDE):   {results['fde']:.4f}m")
    print(f"  Miss Rate @ 2.0m:                 {results['miss_rate']:.2%}")
    print(f"  Number of scenarios:              {results['num_scenarios']}")
    print("=" * 60)
    
    # Generate visualizations
    viz_images = []
    if visualize and predictions:
        print(f"\nGenerating visualizations (max {visualize_max} scenarios)...")
        os.makedirs(VIZ_DIR, exist_ok=True)
        
        # Get scenario IDs from the loader
        scenario_ids = [f"scenario_{i}" for i in range(len(predictions))]
        
        saved_paths, avg_viz_ade = visualize_predictions(
            predictions, targets, scenario_ids, VIZ_DIR, max_viz=visualize_max
        )
        
        print(f"\nVisualization Summary:")
        print(f"  Scenarios visualized: {len(saved_paths)}")
        print(f"  Average ADE: {avg_viz_ade:.2f}m")
        print(f"  Saved to: {VIZ_DIR}")
        
        # Log visualizations to wandb
        if use_wandb:
            for save_path, ade, scenario_id in saved_paths:
                if os.path.exists(save_path):
                    viz_images.append(wandb.Image(save_path, caption=f"Scenario {scenario_id} (ADE: {ade:.2f}m)"))
            
            if viz_images:
                wandb.log({"test_visualizations": viz_images})
    
    # Log results to wandb
    if use_wandb:
        wandb.log({
            "test/ade": results['ade'],
            "test/fde": results['fde'],
            "test/miss_rate": results['miss_rate'],
            "test/num_scenarios": results['num_scenarios'],
        })
        
        # Log summary metrics
        wandb.run.summary["final_ade"] = results['ade']
        wandb.run.summary["final_fde"] = results['fde']
        wandb.run.summary["final_miss_rate"] = results['miss_rate']
    
    # Save results
    os.makedirs(vectornet_checkpoint_dir, exist_ok=True)
    results_path = os.path.join(vectornet_checkpoint_dir, 'vectornet_test_results.pt')
    torch.save(results, results_path)
    print(f"\n✓ Results saved to {results_path}")
    
    # Also save as JSON for easy reading
    json_results = {
        'model_type': 'vectornet',
        'checkpoint': checkpoint_path,
        'test_dataset': test_dataset_path,
        'results': {
            'ade': results['ade'],
            'fde': results['fde'],
            'miss_rate': results['miss_rate'],
            'num_scenarios': results['num_scenarios'],
        },
        'config': {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
        'timestamp': datetime.now().isoformat()
    }
    json_path = os.path.join(vectornet_checkpoint_dir, 'vectornet_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ JSON results saved to {json_path}")
    
    if use_wandb:
        wandb.finish()
    
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='VectorNet Model Testing - Multi-Step Prediction')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: auto-detect best)')
    parser.add_argument('--test_data', type=str, default=TEST_HDF5,
                        help='Path to test HDF5 file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--max_scenarios', type=int, default=None,
                        help='Maximum scenarios to evaluate (default: all)')
    parser.add_argument('--visualize_max', type=int, default=10,
                        help='Maximum scenarios to visualize (default: 10)')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable visualization generation')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    results = run_testing(
        test_dataset_path=args.test_data,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        max_scenarios=args.max_scenarios,
        visualize=not args.no_visualize,
        visualize_max=args.visualize_max,
        use_wandb=not args.no_wandb
    )
    
    return results


if __name__ == "__main__":
    main()
