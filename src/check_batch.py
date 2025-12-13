"""Quick check of the first batch of training data."""
import torch
from dataset import HDF5ScenarioDataset
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch

print("=" * 80)
print("CHECKING FIRST TRAINING BATCH")
print("=" * 80)

# Load dataset
dataset = HDF5ScenarioDataset("data/graphs/training/training.hdf5", seq_len=30)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                       collate_fn=collate_graph_sequences_to_batch)

# Get first batch
for batch_dict in dataloader:
    batched_graph_sequence = batch_dict["batch"]
    B = batch_dict["B"]
    T = batch_dict["T"]
    
    print(f"\nBatch info:")
    print(f"  B (batch size): {B}")
    print(f"  T (timesteps): {T}")
    print(f"  Sequence length: {len(batched_graph_sequence)}")
    
    # Check first timestep
    first_graph = batched_graph_sequence[0]
    
    print(f"\nFirst timestep graph:")
    print(f"  Number of nodes: {first_graph.num_nodes}")
    print(f"  Number of edges: {first_graph.num_edges}")
    print(f"  Node features shape: {first_graph.x.shape}")
    print(f"  Edge index shape: {first_graph.edge_index.shape}")
    
    if first_graph.y is not None:
        print(f"  Target labels shape: {first_graph.y.shape}")
    
    # Check feature values
    x = first_graph.x
    y = first_graph.y
    
    print(f"\nFeature statistics (first timestep):")
    for i, name in enumerate(['vx_norm', 'vy_norm', 'speed_norm', 'heading', 'valid', 
                              'ax_norm', 'ay_norm', 'rel_x_sdc', 'rel_y_sdc', 
                              'dist_sdc', 'dist_nearest', 'type_veh', 'type_ped', 
                              'type_cyc', 'type_other']):
        vals = x[:, i]
        print(f"  {name:15s}: min={vals.min().item():>7.3f}, max={vals.max().item():>7.3f}, mean={vals.mean().item():>7.3f}")
    
    if y is not None:
        print(f"\nTarget statistics (first timestep):")
        print(f"  dx: min={y[:, 0].min().item():>7.3f}, max={y[:, 0].max().item():>7.3f}, mean={y[:, 0].mean().item():>7.3f}, std={y[:, 0].std().item():>7.3f}")
        print(f"  dy: min={y[:, 1].min().item():>7.3f}, max={y[:, 1].max().item():>7.3f}, mean={y[:, 1].mean().item():>7.3f}, std={y[:, 1].std().item():>7.3f}")
        
        displacement_mag = torch.norm(y, dim=1)
        print(f"  displacement magnitude: min={displacement_mag.min().item():.3f}, max={displacement_mag.max().item():.3f}, mean={displacement_mag.mean().item():.3f}")
        
        # Check for unrealistic displacements
        large_disp = displacement_mag > 5.0
        if large_disp.any():
            print(f"  ⚠️  WARNING: {large_disp.sum().item()} agents with displacement > 5m in 0.1s (> 50 m/s)")
        
        small_disp = displacement_mag < 0.01
        if small_disp.any():
            print(f"  ℹ️  INFO: {small_disp.sum().item()} agents with displacement < 0.01m (stopped/very slow)")
    
    # Check multiple timesteps
    print(f"\n--- Consistency across timesteps ---")
    for t in [0, T//2, T-1]:
        if t < len(batched_graph_sequence):
            graph_t = batched_graph_sequence[t]
            if graph_t.y is not None:
                disp_mag = torch.norm(graph_t.y, dim=1)
                print(f"Timestep {t:2d}: {graph_t.num_nodes:3d} nodes, avg disp: {disp_mag.mean().item():.3f}m, max: {disp_mag.max().item():.3f}m")
    
    # Only check first batch
    break

print("\n" + "=" * 80)
print("QUICK DIAGNOSTICS")
print("=" * 80)

# Check for data issues
if x.isnan().any():
    print("❌ PROBLEM: NaN values in features!")
else:
    print("✅ No NaN in features")

if y is not None and y.isnan().any():
    print("❌ PROBLEM: NaN values in targets!")
else:
    print("✅ No NaN in targets")

# Check if relative positions are properly normalized
rel_positions = x[:, 7:10]  # rel_x, rel_y, dist_sdc
if (rel_positions.abs() > 10).any():
    print("❌ PROBLEM: Relative positions not properly normalized (values > 10)!")
    print(f"   Max abs value: {rel_positions.abs().max().item():.3f}")
else:
    print("✅ Relative positions properly normalized")

# Check velocity normalization
velocities = x[:, 0:2]  # vx_norm, vy_norm
if (velocities.abs() > 2).any():
    print("❌ PROBLEM: Velocities not properly normalized (values > 2)!")
    print(f"   Max abs value: {velocities.abs().max().item():.3f}")
else:
    print("✅ Velocities properly normalized")

print("\n" + "=" * 80)
