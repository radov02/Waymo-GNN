import torch
import h5py
from torch.utils.data import Dataset
from config import device
from torch_geometric.data import Data

class HDF5ScenarioDataset(Dataset):
    # stores dataset in one HDF5 file, containing dataset items; 
    # dataset item is a sequence of seq_len graphs/snapshots for one scenario
    def __init__(self, hdf5_path, seq_len=10):
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self._h5file = None

        with h5py.File(hdf5_path, "r") as f:
            self.scenario_ids = list(f["scenarios"].keys())

    def __len__(self):
        return len(self.scenario_ids)

    def __getstate__(self):
        # don't pickle file handle when DataLoader spawns workers
        state = self.__dict__.copy()
        state["_h5file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _open(self):
        if self._h5file is None:
            self._h5file = h5py.File(self.hdf5_path, "r")

    def __getitem__(self, idx):
        self._open()
        scenario_id = self.scenario_ids[idx]
        snaps = self._h5file["scenarios"][scenario_id]["snapshot_graphs"]
        
        # Get sorted timestep keys
        timesteps = sorted(snaps.keys(), key=lambda x: int(x))[:self.seq_len]
        
        data_list = []
        for timestep in timesteps:
            group = snaps[timestep]
            
            # Load tensors
            x = torch.from_numpy(group["x"][:])
            edge_index = torch.from_numpy(group["edge_index"][:]).long()
            edge_weight = torch.from_numpy(group["edge_weight"][:]) if "edge_weight" in group else None
            y = torch.from_numpy(group["y"][:]) if "y" in group else None
            pos = torch.from_numpy(group["pos"][:]) if "pos" in group else None
            agent_ids = group["agent_ids"][:].tolist() if "agent_ids" in group else None
            
            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weight,
                y=y,
                pos=pos
            )
            data.scenario_id = scenario_id  # Add scenario_id to each graph
            data.snapshot_id = int(timestep)
            if agent_ids is not None:
                data.agent_ids = agent_ids  # Add agent_ids as list
            data_list.append(data)
        
        return data_list
    
    def __del__(self):
        # Clean up file handle when object is destroyed
        if self._h5file is not None:
            self._h5file.close()
