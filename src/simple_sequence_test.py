"""Test if model can learn a simple SEQUENCE (mimics actual training)."""
import torch
import torch.nn.functional as F
from SpatioTemporalGNN import SpatioTemporalGNN  # New stable architecture
from config import input_dim, output_dim, hidden_channels, num_layers, dropout
from helper_functions.helpers import advanced_directional_loss, compute_metrics
from config import loss_alpha, loss_beta, loss_gamma, loss_delta

print("=" * 80)
print("SIMPLE SEQUENCE OVERFITTING TEST - NEW ARCHITECTURE")
print("=" * 80)
print("\nGoal: Check if SpatioTemporalGNN can overfit to a sequence")

# Create a simple sequence of 5 timesteps
num_nodes = 5
num_timesteps = 5

# Simple features for each timestep
sequence_x = []
sequence_targets = []

for t in range(num_timesteps):
    x = torch.zeros(num_nodes, input_dim)
    # Velocity changes over time
    x[:, 0] = torch.tensor([0.3, 0.2, 0.1, -0.1, 0.0]) * (0.05 + 0.01 * t)  # vx_norm
    x[:, 1] = torch.tensor([0.2, 0.3, 0.0, 0.1, -0.2]) * (0.05 + 0.01 * t)  # vy_norm
    x[:, 2] = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)  # speed_norm
    x[:, 4] = 1.0  # valid
    x[:, 11] = 1.0  # vehicle type
    
    # Target: displacement proportional to velocity
    target = x[:, 0:2] * 30.0 * 0.1  # denormalize velocity and multiply by dt
    
    sequence_x.append(x)
    sequence_targets.append(target)

# Simple edge index (same for all timesteps)
edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 2, 3, 4],
    [1, 2, 0, 3, 1, 4, 2, 3]
], dtype=torch.long)

print(f"\nSequence length: {num_timesteps} timesteps")
print(f"Number of nodes: {num_nodes}")
print(f"Target displacement range: {sequence_targets[0].min().item():.3f} to {sequence_targets[-1].max().item():.3f}m")

# Initialize new stable model
model = SpatioTemporalGNN(
    input_dim=input_dim,
    hidden_dim=hidden_channels,
    output_dim=output_dim,
    num_gcn_layers=num_layers,
    num_gru_layers=1,
    dropout=dropout,
    use_gat=False
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

print(f"\nTraining for 100 epochs...")
print(f"Each epoch processes the full sequence\n")

for epoch in range(100):
    # Reset GRU states at start of each sequence (per-agent hidden states)
    model.reset_gru_hidden_states(num_agents=num_nodes)
    
    optimizer.zero_grad()
    accumulated_loss = 0.0
    
    # Process sequence (like actual training)
    for t in range(num_timesteps):
        x = sequence_x[t]
        target = sequence_targets[t]
        
        output = model(x, edge_index, batch_size=1, batch_num=0, timestep=t)
        
        # Use simple MSE for this test
        loss_t = F.mse_loss(output, target)
        accumulated_loss += loss_t
    
    # Backprop through entire sequence
    accumulated_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 10 == 0:
        with torch.no_grad():
            # Evaluate on first timestep
            model.reset_gru_hidden_states(num_agents=num_nodes)
            output_t0 = model(sequence_x[0], edge_index, batch_size=1, batch_num=0, timestep=0)
            mse, cos_sim, angle_error = compute_metrics(output_t0, sequence_targets[0], sequence_x[0])
            
            avg_loss = accumulated_loss.item() / num_timesteps
            print(f"Epoch {epoch:3d}: AvgLoss={avg_loss:.6f}, MSE_t0={mse:.6f}, "
                  f"Cos={cos_sim:.4f}, Angle={angle_error*180/3.14159:.1f}°")

print(f"\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

model.eval()
with torch.no_grad():
    model.reset_gru_hidden_states(num_agents=num_nodes)
    
    print(f"\nPredictions for each timestep:")
    print(f"{'Time':<6} {'Target (m)':<20} {'Prediction (m)':<20} {'Error (m)':<10}")
    print("-" * 66)
    
    total_error = 0.0
    for t in range(num_timesteps):
        output = model(sequence_x[t], edge_index, batch_size=1, batch_num=0, timestep=t)
        
        # Show first agent
        tgt = sequence_targets[t][0]
        pred = output[0]
        err = torch.norm(tgt - pred).item()
        total_error += err
        
        print(f"t={t:<4} [{tgt[0]:>6.3f}, {tgt[1]:>6.3f}]    [{pred[0]:>6.3f}, {pred[1]:>6.3f}]    {err:>6.3f}")
    
    avg_error = total_error / num_timesteps
    print(f"\nAverage error: {avg_error:.3f}m")
    
    if avg_error < 0.05:
        print(f"\n✅ SUCCESS: Model can learn sequences! Proceed with full training.")
    elif avg_error < 0.2:
        print(f"\n⚠️  PARTIAL: Model learning but slowly. May work with more data/epochs.")
    else:
        print(f"\n❌ FAILURE: Model cannot learn even this simple sequence!")
        print(f"   Architecture issue with temporal evolution.")
