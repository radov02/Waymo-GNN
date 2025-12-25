"""utility functions for VectorNet training, evaluation, and visualization"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import hashlib
from config import POSITION_SCALE

def compute_vectornet_metrics(predictions, targets, prefix=''):
    """Compute comprehensive metrics for VectorNet predictions using:
        predictions: [N, 2] or [N, T, 2] predicted displacements
        targets: [N, 2] or [N, T, 2] target displacements
        prefix: Optional prefix for metric names"""
    with torch.no_grad():
        # for both single-step and multi-step predictions
        if predictions.dim() == 2:
            pred = predictions
            targ = targets
        else:
            # flatten for aggregate metrics
            pred = predictions.reshape(-1, 2)
            targ = targets.reshape(-1, 2)
        
        mse = F.mse_loss(pred, targ)    # MSE
        rmse_meters = (mse ** 0.5) * POSITION_SCALE     # RMSE in meters
        pred_norm = F.normalize(pred, p=2, dim=1, eps=1e-6)
        targ_norm = F.normalize(targ, p=2, dim=1, eps=1e-6)
        cos_sim = F.cosine_similarity(pred_norm, targ_norm, dim=1).mean()   # cosine similarity
        pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
        targ_angle = torch.atan2(targ[:, 1], targ[:, 0])
        angle_diff = torch.atan2(
            torch.sin(pred_angle - targ_angle),
            torch.cos(pred_angle - targ_angle)
        )
        angle_error = torch.abs(angle_diff).mean()  # Mean absolute angle error in radians
        pred_mag = torch.norm(pred, dim=1)
        targ_mag = torch.norm(targ, dim=1)
        mag_error = torch.abs(pred_mag - targ_mag).mean()   # Mean absolute displacement magnitude error
        
    p = f"{prefix}_" if prefix else ""
    
    return {
        f'{p}mse': mse.item(),
        f'{p}rmse_meters': rmse_meters.item(),
        f'{p}cosine_sim': cos_sim.item(),
        f'{p}angle_error_rad': angle_error.item(),
        f'{p}angle_error_deg': np.degrees(angle_error.item()),
        f'{p}magnitude_error': mag_error.item(),
    }

def compute_ade_fde(predictions, targets):
    """compute Average and Final Displacement Errors using:
        predictions: [N, T, 2] predicted trajectory displacements
        targets: [N, T, 2] target trajectory displacements"""
    with torch.no_grad():
        # convert to cumulative positions
        pred_pos = torch.cumsum(predictions, dim=1)
        targ_pos = torch.cumsum(targets, dim=1)
        
        # displacement errors at each timestep:
        errors = torch.norm(pred_pos - targ_pos, dim=2)  # [N, T]
        ade = errors.mean()     # ADE: mean over all timesteps and agents
        
        # FDE: error at final timestep:
        fde = errors[:, -1].mean()
        
        # Miss rate at 2m threshold
        mr_2m = (errors[:, -1] > 2.0).float().mean() * POSITION_SCALE
        
    return {
        'ade': ade.item(),
        'fde': fde.item(),
        'miss_rate_2m': mr_2m.item(),
    }

def visualize_vectornet_predictions(predictions, targets, positions,
                                    agent_types=None, save_path=None,
                                    title="VectorNet Trajectory Prediction"):
    """Visualize predicted vs ground truth trajectories.
    
    Args:
        predictions: [N, T, 2] or [N, 2] predicted displacements
        targets: [N, T, 2] or [N, 2] target displacements
        positions: [N, 2] starting positions
        agent_types: Optional [N] tensor of agent type indices
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    
    # Handle both single-step and multi-step
    if predictions.ndim == 2:
        predictions = predictions[:, np.newaxis, :]
        targets = targets[:, np.newaxis, :]
    
    N = positions.shape[0]
    T = predictions.shape[1]
    
    # Convert displacements to cumulative positions
    pred_pos = positions[:, np.newaxis, :] + np.cumsum(predictions * 100, axis=1)  # Scale back
    targ_pos = positions[:, np.newaxis, :] + np.cumsum(targets * 100, axis=1)
    
    # Color map for agent types
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(min(N, 50)):  # Limit to 50 agents for clarity
        color_idx = agent_types[i] if agent_types is not None else i % 10
        color = colors[color_idx % 10]
        
        # Plot starting position
        ax.scatter(positions[i, 0], positions[i, 1], 
                   c=[color], s=100, marker='o', edgecolors='black', zorder=3)
        
        # Plot ground truth trajectory
        gt_full = np.vstack([positions[i:i+1], targ_pos[i]])
        ax.plot(gt_full[:, 0], gt_full[:, 1], 
                c='green', linestyle='-', linewidth=2, alpha=0.7, label='Ground Truth' if i == 0 else '')
        
        # Plot predicted trajectory
        pred_full = np.vstack([positions[i:i+1], pred_pos[i]])
        ax.plot(pred_full[:, 0], pred_full[:, 1], 
                c='red', linestyle='--', linewidth=2, alpha=0.7, label='Predicted' if i == 0 else '')
        
        # End points
        ax.scatter(targ_pos[i, -1, 0], targ_pos[i, -1, 1], 
                   c='green', s=60, marker='x', zorder=3)
        ax.scatter(pred_pos[i, -1, 0], pred_pos[i, -1, 1], 
                   c='red', s=60, marker='+', zorder=3)
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    
    plt.close(fig)
    
    return fig

def visualize_attention_weights(attention_weights, polyline_labels=None,
                                save_path=None, title="VectorNet Attention Weights"):
    """Visualize attention weights from global interaction graph.
    
    Args:
        attention_weights: [P, P] attention weight matrix
        polyline_labels: Optional list of labels for each polyline
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    P = attention_weights.shape[0]
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='hot', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    # Add labels if provided
    if polyline_labels:
        ax.set_xticks(range(P))
        ax.set_yticks(range(P))
        ax.set_xticklabels(polyline_labels[:P], rotation=45, ha='right')
        ax.set_yticklabels(polyline_labels[:P])
    
    ax.set_xlabel('Key Polyline')
    ax.set_ylabel('Query Polyline')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved attention visualization to {save_path}")
    
    plt.close(fig)
    
    return fig

def generate_run_id():
    """generate a unique run ID based on timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    return f"{timestamp}_{hash_suffix}"

def count_parameters(model):
    """count total and trainable parameters in a model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'total_mb': total * 4 / (1024 ** 2),  # Assuming float32
        'trainable_mb': trainable * 4 / (1024 ** 2)
    }

def print_model_summary(model):
    print("\n" + "=" * 60)
    print("VECTORNET MODEL SUMMARY")
    print("=" * 60)
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {params['total']:,} ({params['total_mb']:.2f} MB)")
    print(f"  Trainable: {params['trainable']:,} ({params['trainable_mb']:.2f} MB)")
    print(f"  Frozen: {params['frozen']:,}")
    
    # Print layer info
    print(f"\nArchitecture:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params:,} parameters")
    
    print("=" * 60 + "\n")

def create_checkpoint_name(prefix, epoch, run_id, extension='.pt'):
    """create a standardized checkpoint filename"""
    return f"{prefix}_epoch{epoch:03d}_{run_id}{extension}"

def load_checkpoint_config(checkpoint_path):
    """load only the config from a checkpoint without loading the full model"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint.get('config', {})

class MovingAverage:
    """exponential moving average for tracking training metrics"""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        self.value = None

class MetricTracker:
    """track and aggregate metrics during training."""
    
    def __init__(self, metric_names):
        self.metric_names = metric_names
        self.reset()
    
    def reset(self):
        self.sums = {name: 0.0 for name in self.metric_names}
        self.counts = {name: 0 for name in self.metric_names}
    
    def update(self, **kwargs):
        for name, value in kwargs.items():
            if name in self.metric_names:
                self.sums[name] += value
                self.counts[name] += 1
    
    def get_averages(self):
        return {
            name: self.sums[name] / max(1, self.counts[name])
            for name in self.metric_names
        }
    
    def get_average(self, name):
        return self.sums[name] / max(1, self.counts[name])

__all__ = [
    'compute_vectornet_metrics',
    'compute_ade_fde',
    'visualize_vectornet_predictions',
    'visualize_attention_weights',
    'generate_run_id',
    'count_parameters',
    'print_model_summary',
    'create_checkpoint_name',
    'load_checkpoint_config',
    'MovingAverage',
    'MetricTracker',
]
