import torch
import torch.nn.functional as F

def advanced_directional_loss(pred, target, node_features, alpha=0.6, beta=0.15, gamma=0.15, delta=0.1):
    """
    ANGLE-DOMINANT loss: Angle correctness gets 60% weight!
    
    Args:
        pred: Predicted displacements [N, 2]
        target: Ground truth displacements [N, 2]
        node_features: Node features [N, 9] with velocities in first 2 dims
        alpha: Weight for ANGLE (default 0.6 = 60% - DOMINANT!)
        beta: Weight for MSE magnitude (default 0.15 = 15%)
        gamma: Weight for velocity consistency (default 0.15 = 15%)
        delta: Weight for cosine similarity (default 0.1 = 10%)
    
    Loss breakdown:
        60% - Angle correctness (MAJOR WEIGHT!)
        15% - MSE (magnitude)
        15% - Velocity consistency (physics prior - increased from 5%)
        10% - Cosine similarity (backup directional signal)
        
    Note: Removed diversity loss - diversity should come from:
        - Different graph structures per scenario
        - Different node features (velocities, types)
        - Attention mechanism (different node selections)
    """
    # ANGLE LOSS - The dominant factor! (60%)
    pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    angle_diff = pred_angle - target_angle
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    
    # Use squared angle error to heavily penalize wrong directions
    angle_loss = (angle_diff ** 2).mean()
    
    # MSE for magnitude (15%)
    mse_loss = F.mse_loss(pred, target)
    
    # Cosine similarity for backup directional signal (10%)
    pred_norm = F.normalize(pred, p=2, dim=1, eps=1e-6)
    target_norm = F.normalize(target, p=2, dim=1, eps=1e-6)
    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
    cosine_loss = (1 - cos_sim).mean()
    
    # Velocity consistency (15% - increased to replace diversity loss)
    vx = node_features[:, 0] * 10.0
    vy = node_features[:, 1] * 10.0
    expected_disp = torch.stack([vx, vy], dim=1) * 0.1
    velocity_loss = F.mse_loss(pred, expected_disp)
    
    # Weighted combination: ANGLE DOMINATES with 60%!
    total_loss = (alpha * angle_loss +           # 60% - ANGLE!
                  beta * mse_loss +              # 15% - magnitude
                  delta * cosine_loss +          # 10% - backup direction
                  gamma * velocity_loss)         # 15% - physics (increased)
    
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
