import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import threading


class HDF5ScenarioDataset(Dataset):
   
    def __init__(self, hdf5_path, seq_len=10, cache_in_memory=False, max_cache_size=100):
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self.cache_in_memory = cache_in_memory
        self.max_cache_size = max_cache_size
        self._local = threading.local()     # thread-local storage for file handles (one per worker)
        
        with h5py.File(hdf5_path, "r", swmr=True) as f:     # pre-load metadata to avoid repeated file opens
            self.scenario_ids = list(f["scenarios"].keys())
            self._timestep_cache = {}       # pre-compute sorted timestep keys for each scenario
            for sid in self.scenario_ids:
                snaps = f["scenarios"][sid]["snapshot_graphs"]
                self._timestep_cache[sid] = sorted(snaps.keys(), key=lambda x: int(x))[:self.seq_len]
        
        # optional in-memory cache for frequently accessed data:
        self._memory_cache = {} if cache_in_memory else None
        self._cache_lock = threading.Lock() if cache_in_memory else None

    def __len__(self):
        return len(self.scenario_ids)

    def __getstate__(self):
        # Don't pickle file handle, thread-local storage, or locks when DataLoader spawns workers
        state = self.__dict__.copy()
        state["_local"] = None
        state["_cache_lock"] = None  # threading.Lock cannot be pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()
        # Recreate lock if cache is enabled
        if self._memory_cache is not None:
            self._cache_lock = threading.Lock()
        else:
            self._cache_lock = None

    def _get_h5file(self):
        """Get thread-local HDF5 file handle, opening if necessary."""
        if not hasattr(self._local, 'h5file') or self._local.h5file is None:
            # Use swmr=True for concurrent read access
            self._local.h5file = h5py.File(self.hdf5_path, "r", swmr=True)
        return self._local.h5file
    
    def init_worker(self):
        """Called from DataLoader worker_init_fn to ensure fresh handle per worker."""
        # Close any existing handle (from parent process)
        if hasattr(self._local, 'h5file') and self._local.h5file is not None:
            try:
                self._local.h5file.close()
            except Exception:
                pass
            self._local.h5file = None
        # Next __getitem__ call will open a new handle for this worker

    def _load_tensor_fast(self, dataset, dtype=torch.float32):
        # read tensor directly into contiguous numpy array, then convert
        arr = np.ascontiguousarray(dataset[:])
        return torch.from_numpy(arr).to(dtype)

    def __getitem__(self, idx):
        scenario_id = self.scenario_ids[idx]
        if self._memory_cache is not None:   # check memory cache first
            with self._cache_lock:
                if scenario_id in self._memory_cache:
                    return self._memory_cache[scenario_id]
        
        h5file = self._get_h5file()
        snaps = h5file["scenarios"][scenario_id]["snapshot_graphs"]
        
        timesteps = self._timestep_cache[scenario_id]   # use pre-computed sorted timesteps
        
        data_list = []
        for timestep in timesteps:
            group = snaps[timestep]
            
            x = self._load_tensor_fast(group["x"], torch.float32)       # optimized tensor loading
            edge_index = self._load_tensor_fast(group["edge_index"], torch.long)
            
            edge_weight = None
            if "edge_weight" in group:
                edge_weight = self._load_tensor_fast(group["edge_weight"], torch.float32)
            
            y = None
            if "y" in group:
                y = self._load_tensor_fast(group["y"], torch.float32)
            
            pos = None
            if "pos" in group:
                pos = self._load_tensor_fast(group["pos"], torch.float32)
            
            agent_ids = None
            if "agent_ids" in group:
                agent_ids = group["agent_ids"][:].tolist()
            
            data = Data(        # create PyG Data object
                x=x,
                edge_index=edge_index,
                edge_attr=edge_weight,
                y=y,
                pos=pos
            )
            data.scenario_id = scenario_id
            data.snapshot_id = int(timestep)
            if agent_ids is not None:
                data.agent_ids = agent_ids
            data_list.append(data)
        
        if self._memory_cache is not None:      # cache if enabled and under limit
            with self._cache_lock:
                if len(self._memory_cache) < self.max_cache_size:
                    self._memory_cache[scenario_id] = data_list
        
        return data_list
    
    def __del__(self):
        """Clean up file handles."""
        if hasattr(self, '_local') and self._local is not None:
            if hasattr(self._local, 'h5file') and self._local.h5file is not None:
                try:
                    self._local.h5file.close()
                except:
                    pass
    
    def preload_to_memory(self, indices=None):
        """Preload specified scenarios (from indices list, or first max_cache_size if None) into memory cache"""
        if self._memory_cache is None:
            print("Warning: cache_in_memory=False, cannot preload")
            return
        
        if indices is None:
            indices = range(min(len(self), self.max_cache_size))
        
        for idx in indices:
            _ = self[idx]  # this will cache automatically
