import torch
import torch.nn.functional as F

def advanced_directional_loss(pred, target, node_features, alpha=0.4, beta=0.1, gamma=0.25, delta=0.15):
    """loss that includes MSE, angle, heading direction, velocity magnitude, similarity"""
    # ANGLE LOSS:
    pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    angle_diff = pred_angle - target_angle
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    angle_loss = (angle_diff ** 2).mean()   # we use squared angle error to heavily penalize wrong directions
    
    # MSE for magnitude:
    mse_loss = F.mse_loss(pred, target)
    
    # cosine similarity for backup directional signal:
    pred_norm = F.normalize(pred, p=2, dim=1, eps=1e-6)
    target_norm = F.normalize(target, p=2, dim=1, eps=1e-6)
    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
    cosine_loss = (1 - cos_sim).mean()
    
    # heading direction alignment
    heading = node_features[:, 3]   # already computed as node feature (atan2(vy, vx) / pi) → [-1, 1]
    speed = node_features[:, 2] * 30.0  # Denormalize speed (normalized by MAX_SPEED=30)
    
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
    expected_disp = velocity_vector * 0.1      # 0.1s timestep
    
    # Only apply velocity consistency to agents with low acceleration (approximately constant velocity)
    if node_features.shape[1] > 6:  # Check if acceleration features exist
        ax = node_features[:, 5] * 10.0  # denormalize acceleration (normalized by MAX_ACCEL=10)
        ay = node_features[:, 6] * 10.0
        accel_magnitude = torch.sqrt(ax**2 + ay**2 + 1e-6)
        low_accel_mask = accel_magnitude < 2.0  # Less than 2 m/s² acceleration
        
        if low_accel_mask.any():
            velocity_magnitude_loss = F.mse_loss(pred[low_accel_mask], expected_disp[low_accel_mask])
        else:
            velocity_magnitude_loss = torch.tensor(0.0, device=pred.device)
    else:
        # Fallback if no acceleration features (shouldn't happen with 15-dim features)
        velocity_magnitude_loss = F.mse_loss(pred, expected_disp)
    
    # diversity penalty to prevent mode collapse:
    if len(pred) > 1:
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
