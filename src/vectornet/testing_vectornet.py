"""VectorNet Multi-Step Testing Script for Waymo Open Motion Dataset.

This script evaluates trained VectorNet models that predict full trajectories
at once (multi-step, not autoregressive).

Features:
- Load trained VectorNet checkpoints
- Evaluate on test set with multi-step prediction
- Compute trajectory prediction metrics (ADE, FDE, MR)
- Generate trajectory visualizations
- Save results to file

Usage:
    python src/vectornet/testing_vectornet.py
    python src/vectornet/testing_vectornet.py --checkpoint path/to/model.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from datetime import datetime
from torch.utils.data import DataLoader, Subset
import random

# Import VectorNet modules
from VectorNet import VectorNetMultiStep
from vectornet_dataset import VectorNetDataset, collate_vectornet_batch, worker_init_fn
from vectornet_config import (
    device, num_workers, pin_memory,
    vectornet_input_dim, vectornet_hidden_dim, vectornet_output_dim,
    vectornet_num_polyline_layers, vectornet_num_global_layers,
    vectornet_num_heads, vectornet_dropout,
    vectornet_prediction_horizon, vectornet_history_length,
    vectornet_use_node_completion,
    vectornet_checkpoint_dir, vectornet_viz_dir,
    get_vectornet_config,
    print_gpu_info
)


# ============== PATHS ==============
TEST_HDF5 = 'data/graphs/testing/scenario_graphs.h5'
DEFAULT_CHECKPOINT = os.path.join(vectornet_checkpoint_dir, 'best_vectornet_model.pt')


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


def main():
    parser = argparse.ArgumentParser(description='VectorNet Multi-Step Testing')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default=TEST_HDF5,
                        help='Path to test HDF5 file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--max_scenarios', type=int, default=100,
                        help='Maximum number of scenarios to evaluate')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("VectorNet Multi-Step Testing")
    print("=" * 70)
    
    # Print GPU info
    print_gpu_info()
    
    # Check paths
    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: Checkpoint not found at {args.checkpoint}")
        return
    
    if not os.path.exists(args.test_data):
        print(f"\nERROR: Test data not found at {args.test_data}")
        print("Trying validation data instead...")
        args.test_data = 'data/graphs/validation/scenario_graphs.h5'
        if not os.path.exists(args.test_data):
            print(f"Validation data also not found. Please check data paths.")
            return
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    prediction_horizon = config.get('prediction_horizon', vectornet_prediction_horizon)
    history_length = config.get('history_length', vectornet_history_length)
    
    # Load dataset
    print(f"\n--- Loading Test Data ---")
    print(f"Data: {args.test_data}")
    
    test_dataset = VectorNetDataset(
        args.test_data,
        seq_len=prediction_horizon + history_length,
        cache_in_memory=False
    )
    
    # Limit scenarios if needed
    if len(test_dataset) > args.max_scenarios:
        indices = list(range(args.max_scenarios))
        test_dataset = Subset(test_dataset, indices)
    
    print(f"Test scenarios: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_vectornet_batch,
        worker_init_fn=worker_init_fn,
        drop_last=False
    )
    
    # Evaluate
    results, predictions, targets = evaluate_model(
        model, test_loader, device,
        prediction_horizon=prediction_horizon
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Average Displacement Error (ADE): {results['ade']:.4f}")
    print(f"  Final Displacement Error (FDE):   {results['fde']:.4f}")
    print(f"  Miss Rate @ 2.0m:                 {results['miss_rate']:.2%}")
    print(f"  Number of scenarios:              {results['num_scenarios']}")
    print("=" * 60)
    
    # Save results
    if args.save_results:
        results_path = os.path.join(
            vectornet_checkpoint_dir,
            f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_path, 'w') as f:
            json.dump({
                'checkpoint': args.checkpoint,
                'test_data': args.test_data,
                'results': results,
                'config': config,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
