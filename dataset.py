import torch
import h5py
from torch.utils.data import Dataset
from config import device
from torch_geometric.data import Data

class LazyTemporalDataset(Dataset):
    def __init__(self, snapshot_paths):
        """snapshot_paths is a list of filepaths
           
            (node_features, edge_index, edge_weight, targets_y)"""
        self.snapshot_paths = snapshot_paths

    def __len__(self):
        return len(self.snapshot_paths)

    def __getitem__(self, idx):
        snapshot_path_string = self.snapshot_paths[idx]
        data = torch.load(snapshot_path_string, map_location=torch.device(device), 
                          weights_only=False)    # loads tensors to the device
        
        # Handle both HeteroData and dict formats
        if isinstance(data, dict):
            x = data['x']
            edge_index = data['edge_index']
            edge_weight = data.get('edge_weight')
            y = data.get('y')
            return (x, edge_index, edge_weight, y)
        else:
            # Assume it's a PyG Data or HeteroData object
            return data

    def snapshots(self):    # generator interface
        for idx in range(len(self)):
            yield self[idx]     # self[idx] triggers __getitem__()




class HDF5TemporalDataset(Dataset):
    def __init__(self, hdf5_path):
        """hdf5_path: path to single HDF5 file containing all snapshots.
        expected layout: /scenarios/{scenario_id}/snapshot_graphs/{i}/(x, edge_index, edge_weight?, y?)"""
        self.hdf5_path = hdf5_path
        self._h5file = None  # will be opened lazily per worker

        with h5py.File(hdf5_path, "r") as f:
            self.index = []
            for scenario_id in f["scenarios"].keys():
                scenario_group = f["scenarios"][scenario_id]
                for snapshot_id in sorted(scenario_group["snapshot_graphs"].keys(), key=lambda x: int(x)):
                    self.index.append((scenario_id, snapshot_id))
        # self.index = [(scenario1, snap0), (scenario1, snap1), (scenario2, snap0), ...]

    def __len__(self):
        return len(self.index)
    
    def __getstate__(self):
        # Don't pickle the h5file handle
        state = self.__dict__.copy()
        state['_h5file'] = None
        return state
    
    def __setstate__(self, state):
        # Restore state without h5file (it will be opened lazily)
        self.__dict__.update(state)
    
    def __del__(self):
        if self._h5file is not None:
            self._h5file.close()

    def __getitem__(self, idx):
        if self._h5file is None:
            # Open per worker - each worker process will open its own handle
            self._h5file = h5py.File(self.hdf5_path, "r")

        scenario_id, snapshot_id = self.index[idx]
        group = self._h5file["scenarios"][scenario_id]["snapshot_graphs"][snapshot_id]

        x = torch.from_numpy(group["x"][:])
        edge_index = torch.from_numpy(group["edge_index"][:]).long()
        edge_weight = torch.from_numpy(group["edge_weight"][:]) if "edge_weight" in group else None
        y = torch.from_numpy(group["y"][:]) if "y" in group else None

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        data.scenario_id = scenario_id      # attach metadata
        data.snapshot_id = int(snapshot_id)
        return data

    def snapshots(self, scenario_id=None):
        """generator interface that yields snapshots one-by-one without loading all at once; if scenario_id given, only yields that one."""
        with h5py.File(self.hdf5_path, "r") as f:
            if scenario_id is None:
                for sc_id, snapshot_id in self.index:
                    group = f["scenarios"][sc_id]["snapshot_graphs"][snapshot_id]
                    x = torch.from_numpy(group["x"][:])
                    edge_index = torch.from_numpy(group["edge_index"][:]).long()
                    edge_weight = torch.from_numpy(group["edge_weight"][:]) if "edge_weight" in group else None
                    y = torch.from_numpy(group["y"][:]) if "y" in group else None
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
                    data.scenario_id = sc_id
                    data.snapshot_id = int(snapshot_id)
                    yield data
            else:
                snaps = f["scenarios"][str(scenario_id)]["snapshot_graphs"]
                for snapshot_id in sorted(snaps.keys(), key=lambda x: int(x)):
                    group = snaps[snapshot_id]
                    x = torch.from_numpy(group["x"][:])
                    edge_index = torch.from_numpy(group["edge_index"][:]).long()
                    edge_weight = torch.from_numpy(group["edge_weight"][:]) if "edge_weight" in group else None
                    y = torch.from_numpy(group["y"][:]) if "y" in group else None
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
                    data.scenario_id = scenario_id
                    data.snapshot_id = int(snapshot_id)
                    yield data