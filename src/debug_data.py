"""Debug script to check data values and identify issues."""
import torch
import h5py
import numpy as np
from dataset import HDF5ScenarioDataset

def check_hdf5_data():
    """Check raw data in HDF5 file."""
    print("=" * 80)
    print("CHECKING HDF5 DATA")
    print("=" * 80)
    
    hdf5_path = "data/graphs/training/training.hdf5"
    
    with h5py.File(hdf5_path, "r") as f:
        scenario_ids = list(f["scenarios"].keys())
        print(f"\nTotal scenarios: {len(scenario_ids)}")
        
        # Check first scenario
        scenario_id = scenario_ids[0]
        print(f"\nExamining scenario: {scenario_id}")
        
        snaps = f["scenarios"][scenario_id]["snapshot_graphs"]
        timesteps = sorted(snaps.keys(), key=lambda x: int(x))
        
        print(f"Total timesteps: {len(timesteps)}")
        
        # Check first timestep
        t0 = snaps[timesteps[0]]
        x = t0["x"][:]
        y = t0["y"][:] if "y" in t0 else None
        
        print(f"\n--- Timestep 0 ---")
        print(f"Number of nodes: {x.shape[0]}")
        print(f"Feature dimension: {x.shape[1]}")
        
        print(f"\nNode features (first agent):")
        print(f"  vx_norm: {x[0, 0]:.6f}")
        print(f"  vy_norm: {x[0, 1]:.6f}")
        print(f"  speed_norm: {x[0, 2]:.6f}")
        print(f"  heading: {x[0, 3]:.6f}")
        print(f"  valid: {x[0, 4]:.6f}")
        print(f"  ax_norm: {x[0, 5]:.6f}")
        print(f"  ay_norm: {x[0, 6]:.6f}")
        print(f"  rel_x_sdc_norm: {x[0, 7]:.6f}")
        print(f"  rel_y_sdc_norm: {x[0, 8]:.6f}")
        print(f"  dist_sdc_norm: {x[0, 9]:.6f}")
        print(f"  dist_nearest_norm: {x[0, 10]:.6f}")
        print(f"  type (one-hot): [{x[0, 11]:.1f}, {x[0, 12]:.1f}, {x[0, 13]:.1f}, {x[0, 14]:.1f}]")
        
        print(f"\nFeature statistics across all agents:")
        print(f"  vx_norm: min={x[:, 0].min():.3f}, max={x[:, 0].max():.3f}, mean={x[:, 0].mean():.3f}")
        print(f"  vy_norm: min={x[:, 1].min():.3f}, max={x[:, 1].max():.3f}, mean={x[:, 1].mean():.3f}")
        print(f"  speed_norm: min={x[:, 2].min():.3f}, max={x[:, 2].max():.3f}, mean={x[:, 2].mean():.3f}")
        print(f"  rel_x_sdc: min={x[:, 7].min():.3f}, max={x[:, 7].max():.3f}, mean={x[:, 7].mean():.3f}")
        print(f"  rel_y_sdc: min={x[:, 8].min():.3f}, max={x[:, 8].max():.3f}, mean={x[:, 8].mean():.3f}")
        print(f"  dist_sdc: min={x[:, 9].min():.3f}, max={x[:, 9].max():.3f}, mean={x[:, 9].mean():.3f}")
        
        if y is not None:
            print(f"\nTarget labels (displacement in meters):")
            print(f"  dx: min={y[:, 0].min():.3f}, max={y[:, 0].max():.3f}, mean={y[:, 0].mean():.3f}, std={y[:, 0].std():.3f}")
            print(f"  dy: min={y[:, 1].min():.3f}, max={y[:, 1].max():.3f}, mean={y[:, 1].mean():.3f}, std={y[:, 1].std():.3f}")
            
            # Check for outliers
            displacement_magnitude = np.sqrt(y[:, 0]**2 + y[:, 1]**2)
            print(f"  displacement magnitude: min={displacement_magnitude.min():.3f}, max={displacement_magnitude.max():.3f}")
            
            # For 0.1s timestep, reasonable displacement is ~0.5-3m
            # (5-30 m/s × 0.1s = 0.5-3m)
            large_disp = displacement_magnitude > 5.0
            if large_disp.any():
                print(f"  WARNING: {large_disp.sum()} agents with displacement > 5m in 0.1s (> 50 m/s = 180 km/h)")
                print(f"  Max displacement: {displacement_magnitude.max():.3f}m")
        
        # Check multiple timesteps to see consistency
        print(f"\n--- Checking consistency across timesteps ---")
        for i in range(min(3, len(timesteps))):
            ti = snaps[timesteps[i]]
            xi = ti["x"][:]
            yi = ti["y"][:] if "y" in ti else None
            
            if yi is not None:
                disp_mag = np.sqrt(yi[:, 0]**2 + yi[:, 1]**2)
                print(f"Timestep {i}: {xi.shape[0]} agents, avg displacement: {disp_mag.mean():.3f}m, max: {disp_mag.max():.3f}m")

def check_model_initialization():
    """Check if model is initializing properly."""
    print("\n" + "=" * 80)
    print("CHECKING MODEL INITIALIZATION")
    print("=" * 80)
    
    from SpatioTemporalGNN import SpatioTemporalGNN
    from config import input_dim, output_dim, hidden_channels, num_layers, dropout
    
    model = SpatioTemporalGNN(
        input_dim=input_dim,
        hidden_dim=hidden_channels,
        output_dim=output_dim,
        num_gcn_layers=num_layers,
        num_gru_layers=1,
        dropout=dropout,
        use_gat=False
    )
    
    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_channels}")
    print(f"  Output dim: {output_dim}")
    print(f"  Num GCN layers: {num_layers}")
    
    # Create dummy input
    num_nodes = 8
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    
    model.reset_gru_hidden_states(num_agents=num_nodes)
    
    # Forward pass
    output = model(x, edge_index, batch_size=1, batch_num=0, timestep=0)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"  Output std: {output.std().item():.6f}")
    print(f"  Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    
    # Check if output is reasonable for displacement prediction
    # For 0.1s timestep, typical displacement should be -3 to +3 meters
    if abs(output.mean().item()) > 10 or output.std().item() > 10:
        print(f"  ⚠️  WARNING: Output values seem too large!")
        print(f"  Expected range for 0.1s displacement: roughly [-3, 3] meters")

def check_loss_computation():
    """Check if loss computation is reasonable."""
    print("\n" + "=" * 80)
    print("CHECKING LOSS COMPUTATION")
    print("=" * 80)
    
    from helper_functions.helpers import advanced_directional_loss
    from config import loss_alpha, loss_beta, loss_gamma, loss_delta
    
    # Create realistic dummy data
    num_agents = 8
    
    # Features: typical values
    features = torch.zeros(num_agents, 15)
    features[:, 0] = torch.randn(num_agents) * 0.3  # vx_norm ~ N(0, 0.3)
    features[:, 1] = torch.randn(num_agents) * 0.3  # vy_norm
    features[:, 2] = torch.abs(torch.randn(num_agents) * 0.5)  # speed_norm (positive)
    features[:, 3] = torch.randn(num_agents) * 0.5  # heading
    features[:, 4] = 1.0  # valid
    features[:, 11] = 1.0  # vehicle type
    
    # Predictions: small displacements (typical for 0.1s)
    pred = torch.randn(num_agents, 2) * 0.5  # ~0.5m std
    
    # Targets: also small displacements
    target = torch.randn(num_agents, 2) * 0.5
    
    loss = advanced_directional_loss(pred, target, features, 
                                    alpha=loss_alpha, beta=loss_beta, 
                                    gamma=loss_gamma, delta=loss_delta)
    
    print(f"\nLoss weights:")
    print(f"  alpha (angle): {loss_alpha}")
    print(f"  beta (MSE): {loss_beta}")
    print(f"  gamma (velocity): {loss_gamma}")
    print(f"  delta (cosine): {loss_delta}")
    
    print(f"\nTest loss computation:")
    print(f"  Prediction range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
    print(f"  Target range: [{target.min().item():.3f}, {target.max().item():.3f}]")
    print(f"  Loss value: {loss.item():.6f}")
    
    # Compute individual components
    import torch.nn.functional as F
    mse = F.mse_loss(pred, target)
    pred_norm = F.normalize(pred, p=2, dim=1, eps=1e-6)
    target_norm = F.normalize(target, p=2, dim=1, eps=1e-6)
    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
    
    pred_angle = torch.atan2(pred[:, 1], pred[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    angle_diff = pred_angle - target_angle
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    angle_loss = (angle_diff ** 2).mean()
    
    print(f"\nLoss components:")
    print(f"  MSE: {mse.item():.6f}")
    print(f"  Angle loss: {angle_loss.item():.6f}")
    print(f"  Cosine similarity: {cos_sim.item():.6f}")
    print(f"  Cosine loss: {(1 - cos_sim).item():.6f}")
    
    print(f"\nExpected total loss: ~{(loss_alpha * angle_loss + loss_beta * mse + loss_delta * (1-cos_sim)).item():.6f}")

if __name__ == "__main__":
    check_hdf5_data()
    check_model_initialization()
    check_loss_computation()
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
