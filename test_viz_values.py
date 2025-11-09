"""Debug script to check what values are being visualized"""
import sys
sys.path.append('./src')

import torch
import h5py
from dataset import HDF5ScenarioDataset
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch

# Load one batch
dataset = HDF5ScenarioDataset("./data/graphs/training/training.hdf5", seq_len=91)
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                       collate_fn=collate_graph_sequences_to_batch, drop_last=True)

batch_dict = next(iter(dataloader))
first_graph = batch_dict["batch"][0]

print("First graph info:")
print(f"Number of nodes: {first_graph.num_nodes}")
print(f"\nNode features (x) - first 3 nodes:")
print(first_graph.x[:3])
print(f"\nPositions (pos) - first 3 nodes:")
if hasattr(first_graph, 'pos') and first_graph.pos is not None:
    print(first_graph.pos[:3])
    print(f"\nPosition ranges:")
    print(f"  X: [{first_graph.pos[:, 0].min():.2f}, {first_graph.pos[:, 0].max():.2f}]")
    print(f"  Y: [{first_graph.pos[:, 1].min():.2f}, {first_graph.pos[:, 1].max():.2f}]")
    
    print(f"\nAfter multiplying by 100 (denormalized):")
    denorm_pos = first_graph.pos * 100
    print(f"  X: [{denorm_pos[:, 0].min():.2f}, {denorm_pos[:, 0].max():.2f}]")
    print(f"  Y: [{denorm_pos[:, 1].min():.2f}, {denorm_pos[:, 1].max():.2f}]")
else:
    print("No pos attribute, using first 2 features of x:")
    print(first_graph.x[:3, :2])

print(f"\nLabels (y) - first 3:")
if first_graph.y is not None:
    print(first_graph.y[:3])
    print(f"\nDisplacement ranges:")
    print(f"  dX: [{first_graph.y[:, 0].min():.4f}, {first_graph.y[:, 0].max():.4f}]")
    print(f"  dY: [{first_graph.y[:, 1].min():.4f}, {first_graph.y[:, 1].max():.4f}]")
