import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch.nn.utils import clip_grad_norm_
import h5py

"""
Key Points:

collate_graph_sequences_to_batch function: This is the critical piece that transforms your list of graph sequences into the batched format you need. It returns a dictionary with:

batched_ts: List of T Batch objects (one per timestep)
B: Batch size


HDF5ScenarioDataset: Your existing dataset class works perfectly with multi-worker loading because:

__getstate__ and __setstate__ properly handle file handles when workers are spawned
Each worker opens its own HDF5 file handle via _open()


Multi-worker compatibility: Added persistent_workers=True to keep worker processes alive between epochs, which is more efficient for HDF5 datasets since each worker maintains its file handle.
Training loop: Your existing training loop works as-is! The collate function ensures that batch_dict["batched_ts"] contains properly batched graphs at each timestep.

How the batching works with parallel workers:

Worker 1 loads scenarios [0, 8, 16, ...] → applies collate_graph_sequences_to_batch
Worker 2 loads scenarios [1, 9, 17, ...] → applies collate_graph_sequences_to_batch
Worker 3 loads scenarios [2, 10, 18, ...] → applies collate_graph_sequences_to_batch
Worker 4 loads scenarios [3, 11, 19, ...] → applies collate_graph_sequences_to_batch

Each worker independently reads from the HDF5 file and creates batches, then the main training process consumes these batches efficiently!
"""


class HDF5ScenarioDataset(Dataset):
    """Stores dataset in one HDF5 file, containing dataset items; 
    dataset item is a sequence of seq_len graphs/snapshots for one scenario"""
    
    def __init__(self, hdf5_path, seq_len=11):
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self._h5file = None

        with h5py.File(hdf5_path, "r") as f:
            self.scenario_ids = sorted(list(f["scenarios"].keys()), key=lambda x: int(x))

    def __len__(self):
        return len(self.scenario_ids)

    def __getstate__(self):
        # Don't pickle file handle when DataLoader spawns workers
        state = self.__dict__.copy()
        state["_h5file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _open(self):
        if self._h5file is None:
            self._h5file = h5py.File(self.hdf5_path, "r")

    def get_scenario(self, scenario_id):
        self._open()
        snaps = self._h5file["scenarios"][scenario_id]["snapshot_graphs"]
        keys = sorted(snaps.keys(), key=lambda x: int(x))
        data_list = []
        for snapshot_id in keys[:self.seq_len]:
            group = snaps[snapshot_id]
            x = torch.from_numpy(group["x"][:])
            edge_index = torch.from_numpy(group["edge_index"][:]).long()
            edge_weight = torch.from_numpy(group["edge_weight"][:]) if "edge_weight" in group else None
            y = torch.from_numpy(group["y"][:]) if "y" in group else None
            d = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
            d.snapshot_id = int(snapshot_id)
            data_list.append(d)
        return data_list  # List of graphs, each also having snapshot_id

    def __getitem__(self, idx):
        scenario_id = self.scenario_ids[idx]
        return self.get_scenario(scenario_id)


def collate_graph_sequences_to_batch(batch):
    """
    Collate function to batch temporal graph sequences.
    
    Args:
        batch: List of sequences, where each sequence is [graph_0, graph_1, ..., graph_T]
               Each graph is a PyG Data object.
    
    Returns:
        Dictionary containing:
        - 'batched_ts': List of T Batch objects, where batched_ts[t] contains all graphs 
                        at time t batched together
        - 'B': Batch size (number of sequences)
    """
    B = len(batch)
    T = len(batch[0])  # Number of time steps
    
    # Transpose: from [B, T] to [T, B]
    # Group all graphs at time t=0, then all at t=1, etc.
    transposed = list(zip(*batch))
    
    # Batch graphs at each time step using PyG's Batch
    batched_ts = []
    for t in range(T):
        graphs_at_t = list(transposed[t])
        batched_graph = Batch.from_data_list(graphs_at_t)
        batched_ts.append(batched_graph)
    
    return {
        'batched_ts': batched_ts,
        'B': B
    }


def save_scenarios_to_hdf5(graphs_of_scenarios_dict, h5_path, compression="lzf"):
    """Layout: /scenarios/{scenario_id}/snapshot_graphs/{i}/(x, edge_index, edge_weight?, y?)"""
    with h5py.File(h5_path, "w") as f:
        scenarios_group = f.create_group("scenarios")

        for scID, graphs in graphs_of_scenarios_dict.items():
            scenario_group = scenarios_group.create_group(str(scID))
            snapshots_group = scenario_group.create_group("snapshot_graphs")

            for i, graph in enumerate(graphs):
                snapshot_group = snapshots_group.create_group(str(i))

                if hasattr(graph, "x"):
                    x = graph.x.cpu().numpy()
                    edge_index = graph.edge_index.cpu().numpy()
                    edge_weight = getattr(graph, "edge_weight", None)
                    y = getattr(graph, "y", None)
                else:
                    x = graph["x"].cpu().numpy()
                    edge_index = graph["edge_index"].cpu().numpy()
                    edge_weight = graph.get("edge_weight")
                    y = graph.get("y")

                snapshot_group.create_dataset("x", data=x, compression=compression, chunks=True)
                snapshot_group.create_dataset("edge_index", data=edge_index, compression=compression, chunks=True)
                if edge_weight is not None:
                    snapshot_group.create_dataset("edge_weight", data=edge_weight.cpu().numpy(),
                                            compression=compression, chunks=True)
                if y is not None:
                    snapshot_group.create_dataset("y", data=y.cpu().numpy(),
                                            compression=compression, chunks=True)


if __name__ == '__main__':
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    epochs = 50
    batch_size = 8
    num_workers = 4
    sequence_length = 11
    radius = 30.0
    input_dim = 10
    output_dim = 5
    
    config = {
        "learning_rate": 0.01,
        "project_name": "waymo-project",
        "dataset_name": "waymo open motion dataset v 1_3_0",
        "epochs": epochs,
        "hidden_channels": 64,
        "dropout": 0.3,
        "radius": radius,
        "batch_size": batch_size,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "sequence_length": sequence_length,
        "num_layers": 2,
        "topk": 10
    }
    
    import wandb
    wandb.login()

    training_wandb_run = wandb.init(
        project=config['project_name'],
        config={
            "batch_size": config['batch_size'],
            "learning_rate": config['learning_rate'],
            "dataset": config['dataset_name'],
            "dropout": config['dropout'],
            "hidden_channels": config['hidden_channels'],
            "topk": config['topk'],
            "num_layers": config['num_layers'],
            "epochs": config['epochs']
        },
        name=f"GCN_r{config['radius']}_h{config['hidden_channels']}"
    )

    # Initialize your model (replace with your actual EvolveGCNH model)
    # model = EvolveGCNH(
    #     input_dim=config['input_dim'],
    #     hidden_dim=config['hidden_channels'],
    #     output_dim=output_dim,
    #     num_layers=config['num_layers'],
    #     dropout=config['dropout'],
    #     topk=config['topk']
    # ).to(device)
    
    # Placeholder model for demonstration
    from torch_geometric.nn import GCNConv
    class SimpleTemporalGNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.dropout = dropout
            
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
            
            x = self.conv1(x, edge_index, edge_weight)
            x = torch.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            x = torch.relu(x)
            x = self.fc(x)
            return x
    
    model = SimpleTemporalGNN(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_channels'],
        output_dim=output_dim,
        dropout=config['dropout']
    ).to(device)
    
    wandb.watch(model, log='all', log_freq=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    # Create dataset and dataloader
    dataset = HDF5ScenarioDataset("data/graphs/training.hdf5", seq_len=sequence_length)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        collate_fn=collate_graph_sequences_to_batch, 
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )

    """
    Graph minibatching:
    - To parallelize processing, PyG combines multiple graphs into a single graph 
      with many disconnected components (torch_geometric.data.Batch)
    - The batch attribute maps each node to its corresponding graph index: 
      [0, ..., 0, 1, ..., n-2, n-1, ..., n-1]
    """

    print(f"Training on {device}")
    print(f"Dataset size: {len(dataset)} scenarios")
    print(f"Batch size: {batch_size}, Num workers: {num_workers}")
    print(f"Sequence length: {sequence_length}\n")

    # Training loop
    for epoch in range(training_wandb_run.config['epochs']):
        model.train()
        total_loss_epoch = 0.0
        steps = 0
        
        for step, batch_dict in enumerate(dataloader):
            batched_ts = batch_dict["batched_ts"]  # List of length T of Batch objects
            B = batch_dict["B"]
            T = len(batched_ts)

            # Move each batched graph to device
            for t in range(T):
                batched_ts[t] = batched_ts[t].to(device)

            optimizer.zero_grad()
            loss_sum = 0.0

            for t in range(T):
                batch_t = batched_ts[t]

                out_predictions = model(batch_t)

                # Targets: batch_t.y expected to be node-level targets aligned with batch_t.x
                if batch_t.y is None:
                    # No ground truth stored -> skip loss (or handle differently)
                    continue

                # Ensure out_predictions shape matches batch_t.y
                # out_predictions: [num_nodes, out_dim]; batch_t.y: [num_nodes, out_dim]
                loss_t = loss_fn(out_predictions, batch_t.y.to(out_predictions.dtype))
                loss_sum = loss_sum + loss_t

            # Average loss across time (and across nodes by loss_fn semantics)
            loss = loss_sum / T
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_epoch += loss.item()
            steps += 1

        avg_loss_epoch = total_loss_epoch / max(1, steps)
        wandb.log({"epoch": epoch, "train_loss": avg_loss_epoch})
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config['epochs']} | Loss: {avg_loss_epoch:.6f}")

    print("\nTraining complete!")
    wandb.finish()