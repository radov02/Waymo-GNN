import torch
import torch.nn.functional as F

def advanced_directional_loss(pred, target, node_features, alpha=0.4, beta=0.1, gamma=0.25, delta=0.15):
    """
    HEADING-GUIDED DIRECTIONAL loss: Predictions must follow heading direction!
    
    Args:
        pred: Predicted displacements [N, 2]
        target: Ground truth displacements [N, 2]
        node_features: Node features [N, 9]
            - [0,1]: vx_norm, vy_norm
            - [2]: speed_norm
            - [3]: heading (already computed as atan2(vy, vx))
            - [4-8]: valid, vehicle, pedestrian, cyclist, other
        alpha: Weight for ANGLE (default 0.4 = 40%)
        beta: Weight for MSE magnitude (default 0.1 = 10%)
        gamma: Weight for HEADING alignment (default 0.25 = 25% - STRONG!)
        delta: Weight for velocity magnitude consistency (default 0.15 = 15%)
        Remaining 10% split: cosine 5%, diversity 5%
    
    Loss breakdown:
        40% - Target angle correctness
        25% - HEADING alignment (predictions must follow heading direction!)
        15% - Velocity magnitude consistency (physics: displacement ≈ velocity * dt)
        10% - MSE (magnitude)
        5%  - Cosine similarity (backup signal)
        5%  - Diversity penalty (prevent collapse)
    """
    # ANGLE LOSS - The dominant factor! (40%)
    pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    angle_diff = pred_angle - target_angle
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    
    # Use squared angle error to heavily penalize wrong directions
    angle_loss = (angle_diff ** 2).mean()
    
    # MSE for magnitude (10%)
    mse_loss = F.mse_loss(pred, target)
    
    # Cosine similarity for backup directional signal (5%)
    pred_norm = F.normalize(pred, p=2, dim=1, eps=1e-6)
    target_norm = F.normalize(target, p=2, dim=1, eps=1e-6)
    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
    cosine_loss = (1 - cos_sim).mean()
    
    # HEADING ALIGNMENT (25%) - Use heading feature directly!
    heading = node_features[:, 3]  # Already computed as atan2(vy, vx)
    speed = node_features[:, 2] * 10.0  # Denormalize speed
    
    # Filter out stopped/very slow agents (speed < 0.5 m/s)
    moving_mask = speed > 0.5
    
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
    
    # Velocity magnitude consistency (15%) - Displacement should match velocity * dt
    vx = node_features[:, 0] * 10.0  # Denormalize velocity
    vy = node_features[:, 1] * 10.0
    velocity_vector = torch.stack([vx, vy], dim=1)
    expected_disp = velocity_vector * 0.1  # 0.1s timestep
    velocity_magnitude_loss = F.mse_loss(pred, expected_disp)
    
    # Diversity penalty to prevent mode collapse (5%)
    if len(pred) > 1:
        pred_angle_var = torch.var(pred_angle)
        target_angle_var = torch.var(target_angle)
        variance_ratio = pred_angle_var / (target_angle_var + 1e-6)
        diversity_loss = torch.exp(-variance_ratio)
    else:
        diversity_loss = torch.tensor(0.0, device=pred.device)
    
    # Weighted combination: HEADING DIRECTION + TARGET ANGLE dominate!
    total_loss = (alpha * angle_loss +                    # 40% - Target angle
                  gamma * heading_direction_loss +        # 25% - HEADING alignment (predictions follow heading)
                  delta * velocity_magnitude_loss +       # 15% - Velocity magnitude
                  beta * mse_loss +                       # 10% - MSE magnitude
                  0.05 * cosine_loss +                    # 5%  - Backup direction
                  0.05 * diversity_loss)                  # 5%  - Diversity
    
    return total_loss

def compute_metrics(predictions, targets, features):
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
