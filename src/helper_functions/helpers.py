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
