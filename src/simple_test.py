"""Simple test to check if model can learn a trivial task."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from SpatioTemporalGNN import SpatioTemporalGNN
from config import input_dim, output_dim, hidden_channels, num_layers, dropout
from helper_functions.helpers import advanced_directional_loss, compute_metrics
from config import loss_alpha, loss_beta, loss_gamma, loss_delta

print("=" * 80)
print("SIMPLE OVERFITTING TEST")
print("=" * 80)
print("\nGoal: Check if model can overfit to a single simple example")
print("If it can't, there's a fundamental problem with model/loss/data")

# Create a simple synthetic example
num_nodes = 5
num_edges = 8

# Simple features: mostly zeros except velocity
x = torch.zeros(num_nodes, input_dim)
x[:, 0] = torch.tensor([0.3, 0.2, 0.1, -0.1, 0.0]) * 0.1  # vx_norm (small values)
x[:, 1] = torch.tensor([0.2, 0.3, 0.0, 0.1, -0.2]) * 0.1  # vy_norm
x[:, 2] = torch.abs(x[:, 0:2]).sum(dim=1)  # speed_norm
x[:, 4] = 1.0  # valid
x[:, 11] = 1.0  # vehicle type

# Simple edge index (connected graph)
edge_index = torch.tensor([
    [0, 0, 1, 1, 2, 2, 3, 4],
    [1, 2, 0, 3, 1, 4, 2, 3]
], dtype=torch.long)

# Simple target: proportional to velocity (physics-based)
# For 0.1s timestep, displacement ≈ velocity * 0.1
target = x[:, 0:2] * 30.0 * 0.1  # denormalize velocity and multiply by dt
print(f"\nTarget displacements (meters):")
print(f"  Shape: {target.shape}")
print(f"  Values:\n{target}")
print(f"  Mean magnitude: {torch.norm(target, dim=1).mean().item():.4f}m")

# ==============================================================================
# TEST 1: Simple MLP (no graph structure) - baseline
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 1: Simple MLP (no graph structure)")
print("=" * 80)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

mlp = SimpleMLP(input_dim, hidden_channels, output_dim)
mlp_opt = torch.optim.Adam(mlp.parameters(), lr=0.01)

for iter in range(200):
    mlp_opt.zero_grad()
    out = mlp(x)
    loss = F.mse_loss(out, target)
    loss.backward()
    mlp_opt.step()
    
    if iter % 50 == 0:
        print(f"Iter {iter:3d}: MSE={loss.item():.6f}")

with torch.no_grad():
    final = mlp(x)
    mse = F.mse_loss(final, target).item()
    print(f"Final MSE: {mse:.6f}")
    if mse < 0.001:
        print("✅ MLP can overfit!")
    else:
        print("❌ MLP cannot overfit")

# ==============================================================================
# TEST 2: GCN only (no GRU)
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 2: GCN only (no GRU)")
print("=" * 80)

from torch_geometric.nn import GCNConv

class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index):
        h = F.relu(self.gcn1(x, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        return self.decoder(h)

gcn = SimpleGCN(input_dim, hidden_channels, output_dim)
gcn_opt = torch.optim.Adam(gcn.parameters(), lr=0.01)

for iter in range(200):
    gcn_opt.zero_grad()
    out = gcn(x, edge_index)
    loss = F.mse_loss(out, target)
    loss.backward()
    gcn_opt.step()
    
    if iter % 50 == 0:
        print(f"Iter {iter:3d}: MSE={loss.item():.6f}")

with torch.no_grad():
    final = gcn(x, edge_index)
    mse = F.mse_loss(final, target).item()
    print(f"Final MSE: {mse:.6f}")
    if mse < 0.001:
        print("✅ GCN can overfit!")
    else:
        print("❌ GCN cannot overfit")

# ==============================================================================
# TEST 3: Full SpatioTemporalGNN
# ==============================================================================
print("\n" + "=" * 80)
print("TEST 3: Full SpatioTemporalGNN")
print("=" * 80)

model = SpatioTemporalGNN(
    input_dim=input_dim,
    hidden_dim=hidden_channels,
    output_dim=output_dim,
    num_gcn_layers=num_layers,
    num_gru_layers=1,
    dropout=0.0,  # No dropout for overfitting test
    use_gat=False
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR

for iter in range(200):
    # Reset GRU states each iteration
    model.reset_gru_hidden_states(num_agents=num_nodes)
    
    optimizer.zero_grad()
    
    output = model(x, edge_index, batch_size=1, batch_num=0, timestep=0)
    loss = F.mse_loss(output, target)
    loss.backward()
    
    # Check gradients
    if iter == 0:
        total_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += p.grad.abs().mean().item()
        print(f"Average gradient magnitude: {total_grad/len(list(model.parameters())):.6f}")
    
    optimizer.step()
    
    if iter % 50 == 0:
        with torch.no_grad():
            model.reset_gru_hidden_states(num_agents=num_nodes)
            eval_out = model(x, edge_index, batch_size=1, batch_num=0, timestep=0)
            mse = F.mse_loss(eval_out, target).item()
            print(f"Iter {iter:3d}: MSE={mse:.6f}")

with torch.no_grad():
    model.reset_gru_hidden_states(num_agents=num_nodes)
    final = model(x, edge_index, batch_size=1, batch_num=0, timestep=0)
    mse = F.mse_loss(final, target).item()
    print(f"Final MSE: {mse:.6f}")
    if mse < 0.001:
        print("✅ SpatioTemporalGNN can overfit!")
    else:
        print("❌ SpatioTemporalGNN cannot overfit")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
If MLP works but GCN doesn't: Problem is in GCN layer
If GCN works but SpatioTemporalGNN doesn't: Problem is in GRU integration
If nothing works: Problem is in data representation
""")

# ==============================================================================
# TEST 4: SpatioTemporalGNN with training config
# ==============================================================================
print("=" * 80)
print("TEST 4: SpatioTemporalGNN with actual training config")
print("=" * 80)

from config import learning_rate, gradient_clip_value

model2 = SpatioTemporalGNN(
    input_dim=input_dim,
    hidden_dim=hidden_channels,
    output_dim=output_dim,
    num_gcn_layers=num_layers,
    num_gru_layers=1,
    dropout=dropout,
    use_gat=False
)

optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

for iter in range(500):  # More iterations since using actual config
    model2.reset_gru_hidden_states(num_agents=num_nodes)
    
    optimizer2.zero_grad()
    output = model2(x, edge_index, batch_size=1, batch_num=0, timestep=0)
    loss = F.mse_loss(output, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model2.parameters(), gradient_clip_value)
    optimizer2.step()
    
    if iter % 100 == 0:
        with torch.no_grad():
            model2.reset_gru_hidden_states(num_agents=num_nodes)
            eval_out = model2(x, edge_index, batch_size=1, batch_num=0, timestep=0)
            mse = F.mse_loss(eval_out, target).item()
            print(f"Iter {iter:3d}: MSE={mse:.6f}")

with torch.no_grad():
    model2.reset_gru_hidden_states(num_agents=num_nodes)
    final = model2(x, edge_index, batch_size=1, batch_num=0, timestep=0)
    mse = F.mse_loss(final, target).item()
    print(f"Final MSE: {mse:.6f}")
    if mse < 0.01:
        print("✅ SpatioTemporalGNN with training config can overfit!")
    else:
        print(f"⚠️  SpatioTemporalGNN with training config struggles. MSE={mse:.6f}")
        print(f"   Consider: higher LR, less grad clipping, less dropout")
