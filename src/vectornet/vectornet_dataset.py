"""VectorNet Dataset Handler for Waymo Open Motion Dataset.

This module provides dataset classes for VectorNet that can work with
both the existing HDF5 graph format and a polyline-based representation.

Key features:
1. VectorNetDataset: Wrapper around HDF5ScenarioDataset for VectorNet compatibility
2. PolylineDataset: Native polyline-based dataset for VectorNet
3. Support for map features as additional polylines (if available)
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import h5py
import threading
import os
import sys

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VectorNetDataset(Dataset):
    """VectorNet-compatible wrapper for HDF5ScenarioDataset.
    
    This dataset works with the existing HDF5 graph format and adds
    polyline structure for VectorNet processing.
    
    Each agent's trajectory over time forms a polyline. For single-timestep
    processing, each node is treated as its own polyline.
    """
    
    def __init__(self, hdf5_path, seq_len=10, cache_in_memory=False, 
                 max_cache_size=100, include_map_features=False):
        """
        Args:
            hdf5_path: Path to HDF5 file containing scenario graphs
            seq_len: Maximum sequence length to load
            cache_in_memory: If True, cache loaded scenarios in RAM
            max_cache_size: Maximum number of scenarios to cache
            include_map_features: If True, include map polylines (requires scenario data)
        """
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self.cache_in_memory = cache_in_memory
        self.max_cache_size = max_cache_size
        self.include_map_features = include_map_features
        
        # Thread-local storage for file handles
        self._local = threading.local()
        
        # Pre-load metadata
        with h5py.File(hdf5_path, "r", swmr=True) as f:
            self.scenario_ids = list(f["scenarios"].keys())
            # Pre-compute sorted timestep keys
            self._timestep_cache = {}
            for sid in self.scenario_ids:
                snaps = f["scenarios"][sid]["snapshot_graphs"]
                self._timestep_cache[sid] = sorted(
                    snaps.keys(), key=lambda x: int(x)
                )[:self.seq_len]
        
        # In-memory cache
        self._memory_cache = {} if cache_in_memory else None
        self._cache_lock = threading.Lock() if cache_in_memory else None

    def __len__(self):
        return len(self.scenario_ids)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_local"] = None
        state["_cache_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()
        if self._memory_cache is not None:
            self._cache_lock = threading.Lock()
        else:
            self._cache_lock = None
    
    def _get_h5file(self):
        """Get thread-local HDF5 file handle."""
        if not hasattr(self._local, 'h5file') or self._local.h5file is None:
            self._local.h5file = h5py.File(self.hdf5_path, "r", swmr=True)
        return self._local.h5file
    
    def init_worker(self):
        """Initialize worker with fresh HDF5 handle."""
        if hasattr(self._local, 'h5file') and self._local.h5file is not None:
            try:
                self._local.h5file.close()
            except Exception:
                pass
            self._local.h5file = None
    
    def _load_tensor_fast(self, dataset, dtype=torch.float32):
        """Optimized tensor loading."""
        arr = np.ascontiguousarray(dataset[:])
        return torch.from_numpy(arr).to(dtype)
    
    def _create_polyline_ids(self, agent_ids, timestep_idx, total_timesteps):
        """Create polyline IDs for VectorNet.
        
        Each agent's trajectory across time forms a polyline.
        Polyline ID = agent_id (encoded as integer index)
        
        Args:
            agent_ids: List of agent IDs
            timestep_idx: Current timestep index
            total_timesteps: Total number of timesteps
            
        Returns:
            polyline_ids: Tensor [N] with polyline membership
        """
        # Map agent IDs to sequential indices
        unique_agents = sorted(set(agent_ids))
        agent_to_polyline = {aid: i for i, aid in enumerate(unique_agents)}
        
        polyline_ids = torch.tensor(
            [agent_to_polyline.get(aid, 0) for aid in agent_ids],
            dtype=torch.long
        )
        
        return polyline_ids
    
    def _add_vectornet_attributes(self, data, timestep_idx, total_timesteps, agent_ids):
        """Add VectorNet-specific attributes to PyG Data object.
        
        Args:
            data: PyG Data object
            timestep_idx: Current timestep index
            total_timesteps: Total timesteps in sequence
            agent_ids: List of agent IDs
            
        Returns:
            Enhanced Data object with polyline info
        """
        # Create polyline IDs
        if agent_ids is not None:
            data.polyline_ids = self._create_polyline_ids(
                agent_ids, timestep_idx, total_timesteps
            )
        else:
            # Fallback: each node is its own polyline
            data.polyline_ids = torch.arange(data.x.shape[0], dtype=torch.long)
        
        # Agent mask: all nodes are agents (no map features in current format)
        data.agent_mask = torch.ones(data.x.shape[0], dtype=torch.bool)
        
        # Timestep info (useful for temporal encoding)
        data.timestep = timestep_idx
        data.total_timesteps = total_timesteps
        
        return data
    
    def __getitem__(self, idx):
        """Get a scenario as a list of VectorNet-compatible Data objects."""
        scenario_id = self.scenario_ids[idx]
        
        # Check cache
        if self._memory_cache is not None:
            with self._cache_lock:
                if scenario_id in self._memory_cache:
                    return self._memory_cache[scenario_id]
        
        h5file = self._get_h5file()
        snaps = h5file["scenarios"][scenario_id]["snapshot_graphs"]
        timesteps = self._timestep_cache[scenario_id]
        T = len(timesteps)
        
        data_list = []
        for t_idx, timestep in enumerate(timesteps):
            group = snaps[timestep]
            
            # Load tensors
            x = self._load_tensor_fast(group["x"], torch.float32)
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
            
            # Create PyG Data object
            data = Data(
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
            
            # Add VectorNet attributes
            data = self._add_vectornet_attributes(data, t_idx, T, agent_ids)
            
            data_list.append(data)
        
        # Cache if enabled
        if self._memory_cache is not None:
            with self._cache_lock:
                if len(self._memory_cache) < self.max_cache_size:
                    self._memory_cache[scenario_id] = data_list
        
        return data_list


class VectorNetPolylineDataset(Dataset):
    """Native polyline-based dataset for VectorNet.
    
    This dataset creates proper polyline representations where:
    - Agent trajectories are polylines (sequence of position vectors)
    - Map features (lanes, crosswalks) are polylines
    
    Each vector in a polyline has features:
    [ds_x, ds_y, de_x, de_y, attributes..., polyline_id]
    
    where ds = start point, de = end point of the vector.
    """
    
    def __init__(self, hdf5_path, seq_len=10, history_len=10, future_len=50,
                 cache_in_memory=False, max_cache_size=100):
        """
        Args:
            hdf5_path: Path to HDF5 file
            seq_len: Maximum sequence length for loading
            history_len: Number of past timesteps for trajectory polylines
            future_len: Number of future timesteps to predict
            cache_in_memory: Whether to cache in RAM
            max_cache_size: Maximum cache size
        """
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self.history_len = history_len
        self.future_len = future_len
        self.cache_in_memory = cache_in_memory
        self.max_cache_size = max_cache_size
        
        self._local = threading.local()
        
        # Load metadata
        with h5py.File(hdf5_path, "r", swmr=True) as f:
            self.scenario_ids = list(f["scenarios"].keys())
            self._timestep_cache = {}
            for sid in self.scenario_ids:
                snaps = f["scenarios"][sid]["snapshot_graphs"]
                self._timestep_cache[sid] = sorted(
                    snaps.keys(), key=lambda x: int(x)
                )[:self.seq_len]
        
        self._memory_cache = {} if cache_in_memory else None
        self._cache_lock = threading.Lock() if cache_in_memory else None
    
    def __len__(self):
        return len(self.scenario_ids)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_local"] = None
        state["_cache_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()
        if self._memory_cache is not None:
            self._cache_lock = threading.Lock()
    
    def _get_h5file(self):
        if not hasattr(self._local, 'h5file') or self._local.h5file is None:
            self._local.h5file = h5py.File(self.hdf5_path, "r", swmr=True)
        return self._local.h5file
    
    def init_worker(self):
        if hasattr(self._local, 'h5file') and self._local.h5file is not None:
            try:
                self._local.h5file.close()
            except Exception:
                pass
            self._local.h5file = None
    
    def _build_trajectory_polylines(self, scenario_data, current_timestep):
        """Build polyline representations from agent trajectories.
        
        Each agent's past trajectory is represented as vectors connecting
        consecutive positions.
        
        Args:
            scenario_data: List of Data objects for each timestep
            current_timestep: Index of current timestep (for prediction target)
            
        Returns:
            vectors: [V, feature_dim] tensor of vector features
            polyline_ids: [V] tensor of polyline membership
            agent_polyline_ids: Set of polyline IDs that are agents
            future_targets: [A, future_len, 2] future trajectory targets
        """
        # Get history window
        start_t = max(0, current_timestep - self.history_len + 1)
        history_data = scenario_data[start_t:current_timestep + 1]
        
        # Build agent_id to trajectory mapping
        agent_trajectories = {}  # agent_id -> list of (pos, features) tuples
        
        for t, data in enumerate(history_data):
            if hasattr(data, 'agent_ids') and data.agent_ids:
                for i, aid in enumerate(data.agent_ids):
                    if aid not in agent_trajectories:
                        agent_trajectories[aid] = []
                    
                    pos = data.pos[i].numpy() if data.pos is not None else data.x[i, 7:9].numpy()
                    features = data.x[i].numpy()
                    agent_trajectories[aid].append((pos, features, t))
        
        # Build vectors from trajectories
        vectors = []
        polyline_ids = []
        polyline_to_agent = {}  # polyline_id -> agent_id
        
        polyline_id = 0
        for agent_id, trajectory in agent_trajectories.items():
            if len(trajectory) < 2:
                # Single point - create a zero-length vector
                pos, features, t = trajectory[0]
                vector_features = np.concatenate([
                    pos, pos,  # Start and end same
                    features,
                    [t / self.history_len]  # Normalized timestamp
                ])
                vectors.append(vector_features)
                polyline_ids.append(polyline_id)
            else:
                # Create vectors between consecutive positions
                for i in range(len(trajectory) - 1):
                    pos_start, feat_start, t_start = trajectory[i]
                    pos_end, feat_end, t_end = trajectory[i + 1]
                    
                    # Vector features: [start_pos, end_pos, attributes, timestamp]
                    vector_features = np.concatenate([
                        pos_start, pos_end,
                        feat_start,  # Use start point features as attributes
                        [(t_start + t_end) / (2 * self.history_len)]  # Avg timestamp
                    ])
                    vectors.append(vector_features)
                    polyline_ids.append(polyline_id)
            
            polyline_to_agent[polyline_id] = agent_id
            polyline_id += 1
        
        if len(vectors) == 0:
            # Fallback: return empty tensors
            return (torch.zeros(0, 20), torch.zeros(0, dtype=torch.long),
                    set(), torch.zeros(0, self.future_len, 2))
        
        vectors = torch.tensor(np.array(vectors), dtype=torch.float32)
        polyline_ids = torch.tensor(polyline_ids, dtype=torch.long)
        agent_polyline_ids = set(polyline_to_agent.keys())
        
        # Build future targets (if available)
        future_targets = self._build_future_targets(
            scenario_data, current_timestep, polyline_to_agent
        )
        
        return vectors, polyline_ids, agent_polyline_ids, future_targets
    
    def _build_future_targets(self, scenario_data, current_timestep, polyline_to_agent):
        """Build future trajectory targets for prediction.
        
        Args:
            scenario_data: Full scenario data
            current_timestep: Current timestep index
            polyline_to_agent: Mapping from polyline_id to agent_id
            
        Returns:
            targets: [num_agents, future_len, 2] displacement targets
        """
        num_agents = len(polyline_to_agent)
        targets = torch.zeros(num_agents, self.future_len, 2)
        
        if current_timestep >= len(scenario_data) - 1:
            return targets
        
        # Get current positions
        current_data = scenario_data[current_timestep]
        if not hasattr(current_data, 'agent_ids') or not current_data.agent_ids:
            return targets
        
        agent_current_pos = {}
        for i, aid in enumerate(current_data.agent_ids):
            if current_data.pos is not None:
                agent_current_pos[aid] = current_data.pos[i].numpy()
            else:
                agent_current_pos[aid] = current_data.x[i, 7:9].numpy()
        
        # Extract future positions
        for future_t in range(self.future_len):
            t_idx = current_timestep + 1 + future_t
            if t_idx >= len(scenario_data):
                break
            
            future_data = scenario_data[t_idx]
            if not hasattr(future_data, 'agent_ids') or not future_data.agent_ids:
                continue
            
            for i, aid in enumerate(future_data.agent_ids):
                if aid not in agent_current_pos:
                    continue
                
                # Find polyline_id for this agent
                poly_id = None
                for pid, agent_id in polyline_to_agent.items():
                    if agent_id == aid:
                        poly_id = pid
                        break
                
                if poly_id is None:
                    continue
                
                # Compute displacement from current position
                if future_data.pos is not None:
                    future_pos = future_data.pos[i].numpy()
                else:
                    future_pos = future_data.x[i, 7:9].numpy()
                
                displacement = (future_pos - agent_current_pos[aid]) / 100.0  # Normalize
                targets[poly_id, future_t] = torch.tensor(displacement)
        
        return targets
    
    def __getitem__(self, idx):
        """Get a scenario as a VectorNet polyline representation."""
        scenario_id = self.scenario_ids[idx]
        
        # Check cache
        if self._memory_cache is not None:
            with self._cache_lock:
                if scenario_id in self._memory_cache:
                    return self._memory_cache[scenario_id]
        
        h5file = self._get_h5file()
        snaps = h5file["scenarios"][scenario_id]["snapshot_graphs"]
        timesteps = self._timestep_cache[scenario_id]
        
        # Load all timesteps first
        scenario_data = []
        for timestep in timesteps:
            group = snaps[timestep]
            
            x = torch.from_numpy(np.ascontiguousarray(group["x"][:])).float()
            edge_index = torch.from_numpy(np.ascontiguousarray(group["edge_index"][:])).long()
            
            pos = None
            if "pos" in group:
                pos = torch.from_numpy(np.ascontiguousarray(group["pos"][:])).float()
            
            y = None
            if "y" in group:
                y = torch.from_numpy(np.ascontiguousarray(group["y"][:])).float()
            
            agent_ids = None
            if "agent_ids" in group:
                agent_ids = group["agent_ids"][:].tolist()
            
            data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
            if agent_ids:
                data.agent_ids = agent_ids
            data.scenario_id = scenario_id
            
            scenario_data.append(data)
        
        # Build polyline representation for each timestep
        polyline_data_list = []
        for t in range(len(scenario_data)):
            vectors, poly_ids, agent_poly_ids, targets = self._build_trajectory_polylines(
                scenario_data, t
            )
            
            if vectors.shape[0] > 0:
                poly_data = Data(
                    x=vectors,
                    polyline_ids=poly_ids,
                    agent_mask=torch.tensor([
                        i in agent_poly_ids for i in range(poly_ids.max().item() + 1)
                    ]),
                    future_targets=targets,
                    scenario_id=scenario_id,
                    timestep=t
                )
                polyline_data_list.append(poly_data)
        
        # Cache
        if self._memory_cache is not None:
            with self._cache_lock:
                if len(self._memory_cache) < self.max_cache_size:
                    self._memory_cache[scenario_id] = polyline_data_list
        
        return polyline_data_list


def collate_vectornet_batch(scenario_list):
    """Custom collate function for VectorNet batching.
    
    Similar to collate_graph_sequences_to_batch but handles VectorNet attributes.
    
    Args:
        scenario_list: List of scenario data lists
        
    Returns:
        Dictionary with batched data
    """
    B = len(scenario_list)
    T = len(scenario_list[0]) if scenario_list else 0
    
    # Transpose: list of sequences -> sequences of lists
    transposed = list(zip(*scenario_list))
    
    # Extract scenario IDs
    scenario_ids = []
    for sequence in scenario_list:
        if sequence and len(sequence) > 0 and hasattr(sequence[0], 'scenario_id'):
            scenario_ids.append(sequence[0].scenario_id)
        else:
            scenario_ids.append(None)
    
    # Batch each timestep
    batch = []
    for timestep in range(T):
        graphs_at_timestep = list(transposed[timestep])
        
        # Use Batch.from_data_list
        batched = Batch.from_data_list(graphs_at_timestep)
        
        # Preserve additional attributes
        all_agent_ids = []
        for graph in graphs_at_timestep:
            if hasattr(graph, 'agent_ids'):
                all_agent_ids.extend(graph.agent_ids)
        
        if all_agent_ids:
            batched.agent_ids = all_agent_ids
        
        batch.append(batched)
    
    return {
        'batch': batch,
        'B': B,
        'T': T,
        'scenario_ids': scenario_ids
    }


# Worker init function for DataLoader
def worker_init_fn(worker_id):
    """Initialize worker process with fresh HDF5 file handle."""
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset
    
    # Unwrap Subset if present
    if hasattr(dataset_obj, 'dataset'):
        dataset_obj = dataset_obj.dataset
    
    if hasattr(dataset_obj, 'init_worker'):
        dataset_obj.init_worker()


__all__ = [
    'VectorNetDataset',
    'VectorNetPolylineDataset',
    'collate_vectornet_batch',
    'worker_init_fn'
]
