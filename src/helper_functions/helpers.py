import torch
import torch.nn.functional as F
from src.config import MAX_SPEED

@torch.jit.script
def _compute_angle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """JIT-compiled angle loss computation with NaN protection."""
    # Filter out zero vectors to avoid atan2(0, 0) issues
    pred_mag = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + 1e-8)
    target_mag = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + 1e-8)
    valid_mask = (pred_mag > 1e-6) & (target_mag > 1e-6)
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    
    pred_angle = torch.atan2(pred_valid[:, 1], pred_valid[:, 0])
    target_angle = torch.atan2(target_valid[:, 1], target_valid[:, 0])
    angle_diff = pred_angle - target_angle
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    return (angle_diff ** 2).mean()


@torch.jit.script
def _compute_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """JIT-compiled cosine loss computation."""
    pred_norm = F.normalize(pred, p=2.0, dim=1, eps=1e-6)
    target_norm = F.normalize(target, p=2.0, dim=1, eps=1e-6)
    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
    return (1.0 - cos_sim).mean()


def advanced_directional_loss(pred, target, node_features, alpha=0.2, beta=0.5, gamma=0.1, delta=0.2):
    """Loss function combining MSE (magnitude) with directional losses.
    
    Default weights prioritize MSE (beta=0.5) to ensure model learns correct
    displacement magnitudes, while still considering direction.
    
    MSE is scaled by 100 to make it comparable to other loss components when
    working with normalized displacements (~0.01 magnitude).
    
    Args:
        alpha: Angle loss weight (default 0.2)
        beta: MSE loss weight (default 0.5) - PRIMARY for magnitude learning
        gamma: Velocity consistency weight (default 0.1)
        delta: Cosine similarity weight (default 0.2)
    """
    # ANGLE LOSS (JIT-compiled):
    angle_loss = _compute_angle_loss(pred, target)
    
    # MSE for magnitude - scaled by 100 to be comparable to angle/cosine losses
    # Since displacements are normalized by 100, typical values are ~0.01
    # MSE of 0.01 differences = 0.0001, which is too small vs angle loss ~0.01-0.1
    # Scaling by 100 makes MSE ~0.01 and comparable to other losses
    mse_loss = F.mse_loss(pred, target) * 100.0
    
    # Cosine similarity (JIT-compiled):
    cosine_loss = _compute_cosine_loss(pred, target)
    
    # heading direction alignment
    heading = node_features[:, 3]   # already computed as node feature (atan2(vy, vx) / pi) -> [-1, 1]
    speed = node_features[:, 2] * MAX_SPEED  # Denormalize speed (normalized by MAX_SPEED=30)
    
    moving_mask = speed > 0.5        # filter out stopped/very slow agents (speed < 0.5 m/s)
    
    if moving_mask.any():
        # Calculate angle between prediction and heading direction
        pred_angle_moving = torch.atan2(pred[moving_mask, 1], pred[moving_mask, 0])
        heading_moving = heading[moving_mask]
        
        # Angular difference between prediction and heading
        heading_angle_diff = pred_angle_moving - heading_moving
        heading_angle_diff = torch.atan2(torch.sin(heading_angle_diff), torch.cos(heading_angle_diff))
        
        # Heavily penalize predictions that deviate from heading (squared error)
        heading_direction_loss = (heading_angle_diff ** 2).mean()
    else:
        heading_direction_loss = torch.tensor(0.0, device=pred.device)
    
    # velocity magnitude consistency - displacement should roughly match velocity * dt
    # BUT: this is only valid for constant velocity motion, which doesn't hold for turning/accelerating vehicles
    # So we'll make this loss less strict and only apply to agents with low acceleration
    vx = node_features[:, 0] * 30.0  # denormalize velocity (normalized by MAX_SPEED=30)
    vy = node_features[:, 1] * 30.0
    velocity_vector = torch.stack([vx, vy], dim=1)
    # expected_disp in real meters, but pred is normalized (real/100)
    # So normalize expected_disp: (velocity * dt) / 100
    expected_disp = (velocity_vector * 0.1) / 100.0  # normalized expected displacement
    
    # Only apply velocity consistency to agents with low acceleration (approximately constant velocity)
    if node_features.shape[1] > 6:  # Check if acceleration features exist
        ax = node_features[:, 5] * 10.0  # denormalize acceleration (normalized by MAX_ACCEL=10)
        ay = node_features[:, 6] * 10.0
        accel_magnitude = torch.sqrt(ax**2 + ay**2 + 1e-6)
        low_accel_mask = accel_magnitude < 2.0  # Less than 2 m/s² acceleration
        
        if low_accel_mask.any():
            # Scale velocity loss same as MSE for consistency
            velocity_magnitude_loss = F.mse_loss(pred[low_accel_mask], expected_disp[low_accel_mask]) * 100.0
        else:
            velocity_magnitude_loss = torch.tensor(0.0, device=pred.device)
    else:
        # Fallback if no acceleration features (shouldn't happen with 15-dim features)
        velocity_magnitude_loss = F.mse_loss(pred, expected_disp) * 100.0
    
    # diversity penalty to prevent mode collapse:
    if len(pred) > 1:
        pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
        target_angle = torch.atan2(target[:, 1], target[:, 0])
        pred_angle_var = torch.var(pred_angle)
        target_angle_var = torch.var(target_angle)
        variance_ratio = pred_angle_var / (target_angle_var + 1e-6)
        diversity_loss = torch.exp(-variance_ratio)
    else:
        diversity_loss = torch.tensor(0.0, device=pred.device)
    
    total_loss = (alpha * angle_loss +                    # target angle (should have big importance)
                  #gamma * heading_direction_loss +        # heading alignment (also should have big importance)
                  gamma * velocity_magnitude_loss +       # velocity magnitude
                  beta * mse_loss +                       # MSE magnitude
                  delta * cosine_loss #+                    # backup direction
                  #0.05 * diversity_loss)                  # diversity
    )
    
    # NaN protection: if any component is NaN, return just MSE loss as fallback
    if torch.isnan(total_loss):
        print(f"  Warning: NaN in loss - angle:{angle_loss.item():.4f}, mse:{mse_loss.item():.4f}, "
              f"cos:{cosine_loss.item():.4f}, vel_mag:{velocity_magnitude_loss.item():.4f}")
        return mse_loss  # Safe fallback
    
    return total_loss

def compute_metrics(predictions, targets, features):
    """outputs MSE, cosine similarity and angle metrics for given prediction and target displacements"""
    with torch.no_grad():
        mse = F.mse_loss(predictions, targets)
        pred_norm = F.normalize(predictions, p=2, dim=1, eps=1e-6)
        target_norm = F.normalize(targets, p=2, dim=1, eps=1e-6)
        cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
        
        pred_angle = torch.atan2(predictions[:, 1], predictions[:, 0])
        target_angle = torch.atan2(targets[:, 1], targets[:, 0])
        angle_diff = pred_angle - target_angle
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_error = torch.abs(angle_diff).mean()
        
    return mse.item(), cos_sim.item(), angle_error.item()


# ============== Standardized Metrics for All Models ==============
# These functions provide consistent metric computation across GCN, GAT, and VectorNet

def compute_trajectory_metrics(pred_positions, target_positions, valid_mask=None, position_scale=100.0):
    """Compute standardized trajectory metrics for multi-step predictions.
    
    Works for both autoregressive rollouts (GCN/GAT finetuning) and 
    direct trajectory prediction (VectorNet).
    
    Args:
        pred_positions: Predicted positions [N, T, 2] or [T, N, 2] - in NORMALIZED scale
        target_positions: Ground truth positions [N, T, 2] or [T, N, 2] - in NORMALIZED scale
        valid_mask: Optional validity mask [N, T] or [T, N]
        position_scale: Scale factor to convert to meters (default 100.0)
    
    Returns:
        dict with keys: ade, fde, mse, rmse_meters, cosine_similarity, angle_error
    """
    with torch.no_grad():
        # Ensure 3D: [batch, time, 2]
        if pred_positions.dim() == 2:
            pred_positions = pred_positions.unsqueeze(0)
            target_positions = target_positions.unsqueeze(0)
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(0)
        
        # Handle [T, N, 2] format -> [N, T, 2]
        if pred_positions.shape[0] < pred_positions.shape[1] and pred_positions.shape[2] == 2:
            pass  # Already in [N, T, 2] format
        
        # Convert to meters for ADE/FDE
        pred_meters = pred_positions * position_scale
        target_meters = target_positions * position_scale
        
        # Displacement errors in meters
        displacement_errors = torch.sqrt(
            (pred_meters[..., 0] - target_meters[..., 0])**2 + 
            (pred_meters[..., 1] - target_meters[..., 1])**2 + 1e-8
        )  # [N, T]
        
        if valid_mask is not None:
            # Apply mask
            valid_mask = valid_mask.float()
            masked_errors = displacement_errors * valid_mask
            num_valid = valid_mask.sum(dim=-1).clamp(min=1)  # [N]
            ade_per_agent = masked_errors.sum(dim=-1) / num_valid  # [N]
            
            # FDE: error at last valid timestep
            # Find last valid index for each agent
            last_valid_idx = (valid_mask.cumsum(dim=-1) * valid_mask).argmax(dim=-1)  # [N]
            fde_per_agent = displacement_errors.gather(-1, last_valid_idx.unsqueeze(-1)).squeeze(-1)
        else:
            ade_per_agent = displacement_errors.mean(dim=-1)  # [N]
            fde_per_agent = displacement_errors[..., -1]  # [N]
        
        ade = ade_per_agent.mean()
        fde = fde_per_agent.mean()
        
        # MSE in normalized space (for loss comparison)
        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(-1).expand_as(pred_positions)
            mse = ((pred_positions - target_positions)**2 * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)
        else:
            mse = F.mse_loss(pred_positions, target_positions)
        
        # RMSE in meters
        rmse_meters = torch.sqrt(mse) * position_scale
        
        # Cosine similarity (direction alignment) - compute on displacements between timesteps
        if pred_positions.shape[1] > 1:
            pred_vel = pred_positions[:, 1:, :] - pred_positions[:, :-1, :]
            target_vel = target_positions[:, 1:, :] - target_positions[:, :-1, :]
            pred_vel_flat = pred_vel.reshape(-1, 2)
            target_vel_flat = target_vel.reshape(-1, 2)
        else:
            pred_vel_flat = pred_positions.reshape(-1, 2)
            target_vel_flat = target_positions.reshape(-1, 2)
        
        pred_norm = F.normalize(pred_vel_flat, p=2, dim=1, eps=1e-6)
        target_norm = F.normalize(target_vel_flat, p=2, dim=1, eps=1e-6)
        cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
        
        # Angle error (radians)
        pred_angle = torch.atan2(pred_vel_flat[:, 1], pred_vel_flat[:, 0] + 1e-8)
        target_angle = torch.atan2(target_vel_flat[:, 1], target_vel_flat[:, 0] + 1e-8)
        angle_diff = pred_angle - target_angle
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_error = torch.abs(angle_diff).mean()
        
        return {
            'ade': ade.item(),
            'fde': fde.item(),
            'mse': mse.item(),
            'rmse_meters': rmse_meters.item(),
            'cosine_similarity': cos_sim.item(),
            'angle_error': angle_error.item(),
        }


def compute_miss_rate(pred_positions, target_positions, valid_mask=None, 
                      position_scale=100.0, threshold_meters=2.0):
    """Compute miss rate: fraction of predictions with FDE > threshold.
    
    Args:
        pred_positions: Predicted positions [N, T, 2]
        target_positions: Ground truth positions [N, T, 2]
        valid_mask: Optional validity mask [N, T]
        position_scale: Scale factor to convert to meters
        threshold_meters: FDE threshold for miss (default 2.0m)
    
    Returns:
        float: Miss rate (0-1)
    """
    with torch.no_grad():
        if pred_positions.dim() == 2:
            pred_positions = pred_positions.unsqueeze(0)
            target_positions = target_positions.unsqueeze(0)
        
        pred_meters = pred_positions * position_scale
        target_meters = target_positions * position_scale
        
        # FDE for each agent
        if valid_mask is not None and valid_mask.dim() >= 2:
            # Find last valid index for each agent
            valid_mask = valid_mask.float()
            last_valid_idx = (valid_mask.cumsum(dim=-1) * valid_mask).argmax(dim=-1)
            pred_final = pred_meters.gather(1, last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)
            target_final = target_meters.gather(1, last_valid_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)).squeeze(1)
        else:
            pred_final = pred_meters[:, -1, :]
            target_final = target_meters[:, -1, :]
        
        fde = torch.sqrt(
            (pred_final[:, 0] - target_final[:, 0])**2 + 
            (pred_final[:, 1] - target_final[:, 1])**2 + 1e-8
        )
        
        miss_rate = (fde > threshold_meters).float().mean()
        return miss_rate.item()


def format_metrics_for_wandb(metrics, prefix='train', include_epoch=True, epoch=None, 
                              learning_rate=None, sampling_prob=None):
    """Format metrics dict for wandb logging with consistent key naming.
    
    Args:
        metrics: dict with keys like 'loss', 'ade', 'fde', 'mse', 'cosine_similarity', etc.
        prefix: 'train', 'val', or 'test'
        include_epoch: Whether to include epoch in output
        epoch: Current epoch number (1-indexed)
        learning_rate: Optional learning rate to log
        sampling_prob: Optional sampling probability (for finetuning)
    
    Returns:
        dict formatted for wandb.log()
    """
    wandb_dict = {}
    
    if include_epoch and epoch is not None:
        wandb_dict['epoch'] = epoch
    
    if learning_rate is not None:
        wandb_dict['learning_rate'] = learning_rate
    
    if sampling_prob is not None:
        wandb_dict['sampling_probability'] = sampling_prob
    
    # Map metric keys to wandb format
    for key, value in metrics.items():
        if value is not None:
            wandb_dict[f'{prefix}/{key}'] = value
    
    return wandb_dict


def print_epoch_metrics(epoch, train_metrics, val_metrics=None, learning_rate=None, 
                        sampling_prob=None, model_type='base'):
    """Print formatted epoch metrics to console.
    
    Args:
        epoch: Current epoch number
        train_metrics: dict with training metrics
        val_metrics: Optional dict with validation metrics
        learning_rate: Current learning rate
        sampling_prob: Sampling probability (for finetuning)
        model_type: 'base' for single-step, 'trajectory' for multi-step/finetuning
    """
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Summary")
    print(f"{'='*60}")
    
    if learning_rate is not None:
        print(f"Learning Rate: {learning_rate:.6f}")
    if sampling_prob is not None:
        print(f"Sampling Prob: {sampling_prob:.4f}")
    
    if model_type == 'base':
        # Single-step prediction (base training)
        print(f"\nTraining:")
        print(f"  Loss: {train_metrics.get('loss', 0):.6f}")
        print(f"  MSE: {train_metrics.get('mse', 0):.6f}")
        if 'rmse_meters' in train_metrics:
            print(f"  RMSE: {train_metrics['rmse_meters']:.4f} m")
        print(f"  CosSim: {train_metrics.get('cosine_similarity', 0):.4f}")
        print(f"  AngleErr: {train_metrics.get('angle_error', 0):.4f} rad")
        
        if val_metrics:
            print(f"\nValidation:")
            print(f"  Loss: {val_metrics.get('loss', 0):.6f}")
            print(f"  MSE: {val_metrics.get('mse', 0):.6f}")
            if 'rmse_meters' in val_metrics:
                print(f"  RMSE: {val_metrics['rmse_meters']:.4f} m")
            print(f"  CosSim: {val_metrics.get('cosine_similarity', 0):.4f}")
            print(f"  AngleErr: {val_metrics.get('angle_error', 0):.4f} rad")
    else:
        # Multi-step trajectory prediction (finetuning/vectornet)
        print(f"\nTraining:")
        print(f"  Loss: {train_metrics.get('loss', 0):.6f}")
        print(f"  ADE: {train_metrics.get('ade', 0):.4f} m")
        print(f"  FDE: {train_metrics.get('fde', 0):.4f} m")
        if 'mse' in train_metrics:
            print(f"  MSE: {train_metrics['mse']:.6f}")
        if 'cosine_similarity' in train_metrics:
            print(f"  CosSim: {train_metrics['cosine_similarity']:.4f}")
        
        if val_metrics:
            print(f"\nValidation:")
            print(f"  Loss: {val_metrics.get('loss', 0):.6f}")
            print(f"  ADE: {val_metrics.get('ade', 0):.4f} m")
            print(f"  FDE: {val_metrics.get('fde', 0):.4f} m")
            if 'mse' in val_metrics:
                print(f"  MSE: {val_metrics['mse']:.6f}")
            if 'cosine_similarity' in val_metrics:
                print(f"  CosSim: {val_metrics['cosine_similarity']:.4f}")


def print_test_summary(results, model_name='Model'):
    """Print formatted test results summary.
    
    Args:
        results: dict with test metrics (ade, fde, miss_rate, per-horizon metrics, etc.)
        model_name: Name of model for display
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Test Results")
    print(f"{'='*60}")
    
    print(f"\nOverall Metrics:")
    print(f"  ADE: {results.get('ade', 0):.4f} m")
    print(f"  FDE: {results.get('fde', 0):.4f} m")
    if 'miss_rate' in results:
        print(f"  Miss Rate: {results['miss_rate']*100:.2f}%")
    if 'cosine_similarity' in results:
        print(f"  CosSim: {results['cosine_similarity']:.4f}")
    if 'angle_error' in results:
        print(f"  AngleErr: {results['angle_error']:.4f} rad")
    
    # Per-horizon metrics if available
    if 'horizons' in results:
        print(f"\nPer-Horizon Metrics:")
        for h, h_metrics in results['horizons'].items():
            print(f"  {h}:")
            print(f"    ADE: {h_metrics.get('ade_mean', 0):.4f} ± {h_metrics.get('ade_std', 0):.4f} m")
            print(f"    FDE: {h_metrics.get('fde_mean', 0):.4f} ± {h_metrics.get('fde_std', 0):.4f} m")
