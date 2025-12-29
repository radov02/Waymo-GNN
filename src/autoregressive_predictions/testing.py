"""Testing script for GCN-based autoregressive trajectory prediction.

This module evaluates trained GCN models using autoregressive multi-step prediction
on the validation set (or test set), logs metrics to wandb, and creates visualizations.

This is a thin wrapper around the unified testing_gat.py which supports both GAT and GCN models.

Uses validation dataset by default (90 timesteps) since test dataset may have shorter
sequences that don't support proper 5-second rollouts.

Usage:
    python src/autoregressive_predictions/testing.py
    python src/autoregressive_predictions/testing.py --checkpoint path/to/model.pt
    python src/autoregressive_predictions/testing.py --max_scenarios 50 --no_wandb
    python src/autoregressive_predictions/testing.py --test_data path/to/test.hdf5
"""

import sys
import os
import argparse

# Add parent directory (src/) to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (sequence_length, test_num_rollout_steps, test_max_scenarios,
                    test_visualize_max)

# Import unified testing functions from testing_gat.py
from gat_autoregressive.testing_gat import (
    run_testing,
    load_trained_model,
    evaluate_autoregressive,
    autoregressive_rollout,
    predict_single_step,
    get_base_model,
    get_model_config,
    val_hdf5_path
)

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/autoregressive_predictions/testing.py


def run_gcn_testing(test_dataset_path=val_hdf5_path,
                    checkpoint_path=None,
                    num_rollout_steps=20,
                    max_scenarios=None,
                    visualize=True,
                    visualize_max=10,
                    use_wandb=True):
    """Run testing with autoregressive multi-step prediction for GCN model.
    
    This is a convenience wrapper that calls the unified run_testing with model_type='gcn'.
    
    Args:
        test_dataset_path: Path to test HDF5 file
        checkpoint_path: Path to model checkpoint (default: auto-detect best)
        num_rollout_steps: Number of autoregressive rollout steps
        max_scenarios: Maximum scenarios to evaluate (None = all)
        visualize: Whether to generate visualizations
        visualize_max: Maximum scenarios to visualize
        use_wandb: Whether to log results to wandb
        
    Returns:
        dict with evaluation results
    """
    return run_testing(
        test_dataset_path=test_dataset_path,
        checkpoint_path=checkpoint_path,
        num_rollout_steps=num_rollout_steps,
        max_scenarios=max_scenarios,
        visualize=visualize,
        visualize_max=visualize_max,
        use_wandb=use_wandb,
        model_type='gcn'  # Use GCN model
    )


def main():
    """Main entry point for GCN testing."""
    parser = argparse.ArgumentParser(description='GCN Model Testing')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: auto-detect)')
    parser.add_argument('--test_data', type=str, default=val_hdf5_path,
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
    
    results = run_gcn_testing(
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
