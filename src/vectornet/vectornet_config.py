"""VectorNet Configuration File.

This module contains all configuration parameters for VectorNet training
and evaluation on the Waymo Open Motion Dataset.

Configuration is organized into sections:
- Model architecture
- Training hyperparameters
- Dataset settings
- Logging and checkpoints
"""

import torch
import torch.nn as nn
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import base config for shared settings
from config import (
    device, num_workers, pin_memory, prefetch_factor,
    use_amp, use_bf16, use_torch_compile, torch_compile_mode,
    use_gradient_checkpointing, num_gpus, use_data_parallel,
    print_gpu_info, setup_model_parallel, get_model_for_saving,
    project_name, dataset_name
)

# ============== VectorNet Model Architecture ==============
# Input dimension: Same as other models (15 features)
vectornet_input_dim = 15

# Hidden dimension: Paper uses 64, we use 128 to match other models
vectornet_hidden_dim = 128

# Output dimension: 2 for (dx, dy) displacement
vectornet_output_dim = 2

# Number of layers in Polyline Subgraph Network
# Paper: 3 layers
vectornet_num_polyline_layers = 3

# Number of Global Interaction Graph layers
# Paper: 1 layer (for efficiency)
vectornet_num_global_layers = 1

# Number of attention heads in Global Interaction Graph
# Standard transformer head count
vectornet_num_heads = 8

# Number of GRU layers (for VectorNetTemporal)
vectornet_num_gru_layers = 1

# Dropout rate
vectornet_dropout = 0.1

# Whether to use node completion auxiliary task
# Paper: This helps model learn better context features
vectornet_use_node_completion = True

# Ratio of nodes to mask for node completion task
# Paper: Similar to BERT's 15% masking
vectornet_node_completion_ratio = 0.15

# Loss weight for node completion auxiliary task
# Paper: alpha = 1.0
vectornet_node_completion_weight = 1.0

# VectorNet predicts FULL TRAJECTORY at once (not autoregressive)
# This is the correct approach as per the paper
vectornet_mode = 'multi_step'

# Prediction horizon: number of future timesteps to predict
# Standard Waymo prediction horizon is 50 timesteps (5 seconds at 10Hz)
vectornet_prediction_horizon = 50

# History length: number of past timesteps to encode
# Standard is 10 timesteps (1 second at 10Hz)
vectornet_history_length = 10

# ============== VectorNet Training Hyperparameters ==============
# Batch size: Number of scenarios per batch
# VectorNet is more memory efficient than CNN, so can use larger batches
vectornet_batch_size = 48

# Learning rate
vectornet_learning_rate = 0.001

# Number of training epochs
vectornet_epochs = 30

# Gradient clipping
vectornet_gradient_clip = 1.0

# Learning rate scheduler settings
vectornet_scheduler_patience = 5
vectornet_scheduler_factor = 0.5
vectornet_min_lr = 1e-6

# Early stopping settings
vectornet_early_stopping_patience = 10
vectornet_early_stopping_min_delta = 0.001

# ============== Loss Configuration ==============
# Loss weights for trajectory prediction
# alpha: Angle loss weight
# beta: MSE loss weight (primary)
# gamma: Velocity consistency weight
# delta: Cosine similarity weight
vectornet_loss_alpha = 0.2
vectornet_loss_beta = 0.5
vectornet_loss_gamma = 0.1
vectornet_loss_delta = 0.2

# ============== Dataset Configuration ==============
# Sequence length for temporal processing
vectornet_sequence_length = 90

# Maximum validation scenarios per epoch
vectornet_max_validation_scenarios = 20

# Cache validation data in memory
vectornet_cache_validation = True

# ============== Checkpoint Configuration ==============
vectornet_checkpoint_dir = 'checkpoints/vectornet'
vectornet_checkpoint_prefix = 'vectornet'

# Best model checkpoint name
vectornet_best_model = 'best_vectornet_model.pt'

# Final model checkpoint name
vectornet_final_model = 'final_vectornet_model.pt'

# ============== Visualization Configuration ==============
vectornet_viz_dir = 'visualizations/autoreg/vectornet'
vectornet_visualize_every_n_epochs = 5

# ============== WandB Configuration ==============
vectornet_wandb_project = project_name
vectornet_wandb_name = 'vectornet-training'
vectornet_wandb_tags = ['vectornet', 'waymo', 'trajectory-prediction']

# ============== Multi-Step Prediction Settings ==============
# VectorNet does NOT need autoregressive fine-tuning!
# It predicts the full trajectory in a single forward pass.

# Number of scenarios to use for validation visualization
vectornet_viz_scenarios = 5


def get_vectornet_config():
    """Get VectorNet configuration as a dictionary."""
    return {
        # Model
        'input_dim': vectornet_input_dim,
        'hidden_dim': vectornet_hidden_dim,
        'output_dim': vectornet_output_dim,
        'num_polyline_layers': vectornet_num_polyline_layers,
        'num_global_layers': vectornet_num_global_layers,
        'num_heads': vectornet_num_heads,
        'num_gru_layers': vectornet_num_gru_layers,
        'dropout': vectornet_dropout,
        'use_node_completion': vectornet_use_node_completion,
        'node_completion_ratio': vectornet_node_completion_ratio,
        'node_completion_weight': vectornet_node_completion_weight,
        'mode': vectornet_mode,
        'prediction_horizon': vectornet_prediction_horizon,
        'history_length': vectornet_history_length,
        
        # Training
        'batch_size': vectornet_batch_size,
        'learning_rate': vectornet_learning_rate,
        'epochs': vectornet_epochs,
        'gradient_clip': vectornet_gradient_clip,
        'scheduler_patience': vectornet_scheduler_patience,
        'scheduler_factor': vectornet_scheduler_factor,
        'min_lr': vectornet_min_lr,
        'early_stopping_patience': vectornet_early_stopping_patience,
        'early_stopping_min_delta': vectornet_early_stopping_min_delta,
        
        # Loss
        'loss_alpha': vectornet_loss_alpha,
        'loss_beta': vectornet_loss_beta,
        'loss_gamma': vectornet_loss_gamma,
        'loss_delta': vectornet_loss_delta,
        
        # Dataset
        'sequence_length': vectornet_sequence_length,
        'max_validation_scenarios': vectornet_max_validation_scenarios,
        
        # Paths
        'checkpoint_dir': vectornet_checkpoint_dir,
        'viz_dir': vectornet_viz_dir,
    }


def create_vectornet_model(**kwargs):
    """Factory function to create VectorNet model.
    
    VectorNet predicts full trajectory at once (multi-step), not autoregressively.
    
    Args:
        **kwargs: Override default configuration
        
    Returns:
        VectorNet model instance
    """
    from VectorNet import VectorNetMultiStep
    
    config = get_vectornet_config()
    config.update(kwargs)
    
    # VectorNet always uses multi-step prediction (as per paper)
    model = VectorNetMultiStep(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        prediction_horizon=config['prediction_horizon'],
        num_polyline_layers=config['num_polyline_layers'],
        num_global_layers=config['num_global_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_node_completion=config['use_node_completion'],
        node_completion_ratio=config['node_completion_ratio'],
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    
    return model


def print_vectornet_config():
    """Print VectorNet configuration summary."""
    config = get_vectornet_config()
    
    print("\n" + "=" * 60)
    print("VECTORNET CONFIGURATION")
    print("=" * 60)
    
    print("\n--- Model Architecture ---")
    print(f"  Input Dimension: {config['input_dim']}")
    print(f"  Hidden Dimension: {config['hidden_dim']}")
    print(f"  Output Dimension: {config['output_dim']}")
    print(f"  Polyline Layers: {config['num_polyline_layers']}")
    print(f"  Global Layers: {config['num_global_layers']}")
    print(f"  Attention Heads: {config['num_heads']}")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Prediction Horizon: {config['prediction_horizon']} timesteps")
    print(f"  History Length: {config['history_length']} timesteps")
    
    print("\n--- Node Completion ---")
    print(f"  Enabled: {config['use_node_completion']}")
    print(f"  Masking Ratio: {config['node_completion_ratio']}")
    print(f"  Loss Weight: {config['node_completion_weight']}")
    
    print("\n--- Training ---")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Gradient Clip: {config['gradient_clip']}")
    
    print("\n--- Loss Weights ---")
    print(f"  Angle (alpha): {config['loss_alpha']}")
    print(f"  MSE (beta): {config['loss_beta']}")
    print(f"  Velocity (gamma): {config['loss_gamma']}")
    print(f"  Cosine (delta): {config['loss_delta']}")
    
    print("\n--- Paths ---")
    print(f"  Checkpoint Dir: {config['checkpoint_dir']}")
    print(f"  Visualization Dir: {config['viz_dir']}")
    
    print("=" * 60 + "\n")


# Create checkpoint directory if it doesn't exist
os.makedirs(vectornet_checkpoint_dir, exist_ok=True)
os.makedirs(vectornet_viz_dir, exist_ok=True)


__all__ = [
    # Configuration values
    'vectornet_input_dim',
    'vectornet_hidden_dim',
    'vectornet_output_dim',
    'vectornet_num_polyline_layers',
    'vectornet_num_global_layers',
    'vectornet_num_heads',
    'vectornet_dropout',
    'vectornet_use_node_completion',
    'vectornet_node_completion_ratio',
    'vectornet_node_completion_weight',
    'vectornet_mode',
    'vectornet_prediction_horizon',
    'vectornet_history_length',
    'vectornet_batch_size',
    'vectornet_learning_rate',
    'vectornet_epochs',
    'vectornet_gradient_clip',
    'vectornet_scheduler_patience',
    'vectornet_scheduler_factor',
    'vectornet_min_lr',
    'vectornet_early_stopping_patience',
    'vectornet_early_stopping_min_delta',
    'vectornet_checkpoint_dir',
    'vectornet_viz_dir',
    'vectornet_sequence_length',
    'vectornet_max_validation_scenarios',
    'vectornet_wandb_project',
    'vectornet_wandb_name',
    'vectornet_wandb_tags',
    
    # Loss weights
    'vectornet_loss_alpha',
    'vectornet_loss_beta',
    'vectornet_loss_gamma',
    'vectornet_loss_delta',
    
    # Functions
    'get_vectornet_config',
    'create_vectornet_model',
    'print_vectornet_config',
    
    # Imported from base config
    'device',
    'num_workers',
    'pin_memory',
    'prefetch_factor',
    'use_amp',
    'use_bf16',
    'use_torch_compile',
    'torch_compile_mode',
    'use_gradient_checkpointing',
    'print_gpu_info',
    'setup_model_parallel',
    'get_model_for_saving',
]
