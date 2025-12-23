"""VectorNet Dataset for Direct TFRecord Loading from Waymo Open Motion Dataset.

This module provides a proper VectorNet dataset that extracts:
1. Agent trajectory polylines (past trajectory as vectors)
2. Map feature polylines (lanes, road edges, crosswalks)
3. Traffic light information
4. Multi-step future trajectory labels

Unlike the HDF5-based dataset, this directly parses TFRecord files to get
all the information VectorNet needs, including HD map features.

Reference: "VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation"
(Gao et al., 2020)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import os
import sys
import glob
import math
from typing import Dict, List, Tuple, Optional, Any

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import TensorFlow for TFRecord parsing
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not available. Using tfrecord package instead.")

# Try to import Waymo Open Dataset protos
try:
    from waymo_open_dataset.protos import scenario_pb2
    HAS_WAYMO_SDK = True
except ImportError:
    HAS_WAYMO_SDK = False
    print("Warning: Waymo Open Dataset SDK not available.")


class VectorNetTFRecordDataset(Dataset):
    """VectorNet dataset that directly loads from Waymo TFRecord files.
    
    This dataset creates proper polyline representations for VectorNet:
    - Agent past trajectories as polylines
    - Map features (lanes, road edges, crosswalks) as polylines
    - Traffic light states
    - Multi-step future trajectory labels
    
    Each vector in a polyline: [ds_x, ds_y, ds_z, de_x, de_y, de_z, attributes..., polyline_id]
    """
    
    # Map feature type encoding
    MAP_FEATURE_TYPES = {
        'lane': 0,
        'road_line': 1, 
        'road_edge': 2,
        'stop_sign': 3,
        'crosswalk': 4,
        'speed_bump': 5,
        'driveway': 6,
    }
    
    # Agent type encoding
    AGENT_TYPES = {
        0: 'unset',
        1: 'vehicle',
        2: 'pedestrian', 
        3: 'cyclist',
        4: 'other',
    }
    
    def __init__(
        self,
        tfrecord_dir: str,
        split: str = 'training',
        history_len: int = 10,
        future_len: int = 50,
        max_agents: int = 128,
        max_map_polylines: int = 256,
        max_polyline_vectors: int = 20,
        normalize: bool = True,
        cache_scenarios: bool = False,
        max_scenarios: Optional[int] = None,
        num_agents_to_predict: Optional[int] = 8,
    ):
        """
        Args:
            tfrecord_dir: Base directory containing scenario/training, scenario/validation, etc.
            split: 'training', 'validation', or 'testing'
            history_len: Number of past timesteps to use (default 10 = 1 second at 10Hz)
            future_len: Number of future timesteps to predict (default 50 = 5 seconds at 10Hz)
            max_agents: Maximum number of agents to include
            max_map_polylines: Maximum number of map polylines to include
            max_polyline_vectors: Maximum vectors per polyline
            normalize: Whether to normalize coordinates relative to target agent
            cache_scenarios: Whether to cache parsed scenarios in memory
            max_scenarios: Maximum number of scenarios to load (None = all)
            num_agents_to_predict: Number of agents to predict for per scenario
                                   (None = all valid agents, int = limit to top N)
        """
        self.tfrecord_dir = tfrecord_dir
        self.split = split
        self.history_len = history_len
        self.future_len = future_len
        self.max_agents = max_agents
        self.max_map_polylines = max_map_polylines
        self.max_polyline_vectors = max_polyline_vectors
        self.normalize = normalize
        self.cache_scenarios = cache_scenarios
        self.max_scenarios = max_scenarios
        self.num_agents_to_predict = num_agents_to_predict
        
        # Scenario cache
        self._scenario_cache: Dict[int, Any] = {}
        
        # Find TFRecord files
        self.data_dir = os.path.join(tfrecord_dir, split)
        self.tfrecord_files = sorted(glob.glob(os.path.join(self.data_dir, "*.tfrecord")))
        
        if len(self.tfrecord_files) == 0:
            raise ValueError(f"No TFRecord files found in {self.data_dir}")
        
        print(f"Found {len(self.tfrecord_files)} TFRecord files in {self.data_dir}")
        print(f"Predicting for {num_agents_to_predict if num_agents_to_predict else 'all valid'} agents per scenario")
        
        # Build index: (file_idx, scenario_idx_in_file) -> global_idx
        self._build_index()
    
    def _build_index(self):
        """Build an index of all scenarios across all TFRecord files."""
        self.scenario_index: List[Tuple[int, int]] = []  # (file_idx, scenario_idx_in_file)
        
        print("Building scenario index...")
        for file_idx, tfrecord_path in enumerate(self.tfrecord_files):
            try:
                # Count scenarios in this file
                if HAS_TF:
                    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
                    scenario_count = 0
                    for _ in dataset:
                        scenario_count += 1
                        if self.max_scenarios and len(self.scenario_index) + scenario_count >= self.max_scenarios:
                            break
                else:
                    # Fallback: assume fixed number per file
                    scenario_count = 50  # Typical number per file
                
                for s_idx in range(scenario_count):
                    self.scenario_index.append((file_idx, s_idx))
                    if self.max_scenarios and len(self.scenario_index) >= self.max_scenarios:
                        break
                        
            except Exception as e:
                print(f"Warning: Could not read {tfrecord_path}: {e}")
            
            if self.max_scenarios and len(self.scenario_index) >= self.max_scenarios:
                break
        
        print(f"Indexed {len(self.scenario_index)} scenarios")
    
    def __len__(self) -> int:
        return len(self.scenario_index)
    
    def _parse_scenario(self, file_idx: int, scenario_idx: int) -> Any:
        """Parse a single scenario from TFRecord file."""
        if not HAS_TF or not HAS_WAYMO_SDK:
            raise RuntimeError("TensorFlow and Waymo SDK required for TFRecord parsing")
        
        tfrecord_path = self.tfrecord_files[file_idx]
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
        
        for idx, raw_record in enumerate(dataset):
            if idx == scenario_idx:
                scenario = scenario_pb2.Scenario.FromString(raw_record.numpy())
                return scenario
        
        raise IndexError(f"Scenario {scenario_idx} not found in {tfrecord_path}")
    
    def _get_valid_agents_for_prediction(self, scenario, num_agents_to_predict: Optional[int] = None) -> List[int]:
        """Get list of valid agent indices that can be predicted.
        
        An agent is valid for prediction if:
        1. Has valid state at current timestep (history_len - 1)
        2. Has at least some valid future states
        
        Args:
            scenario: Waymo scenario proto
            num_agents_to_predict: Max number of agents to return (None = all valid)
            
        Returns:
            List of valid agent indices (track indices)
        """
        valid_agents = []
        
        for i, track in enumerate(scenario.tracks):
            if not hasattr(track, 'states') or len(track.states) <= self.history_len:
                continue
                
            # Check if current timestep is valid
            if not track.states[self.history_len - 1].valid:
                continue
            
            # Check if agent has at least some valid future states
            has_future = False
            for t in range(self.history_len, min(self.history_len + self.future_len, len(track.states))):
                if track.states[t].valid:
                    has_future = True
                    break
            
            if not has_future:
                continue
            
            valid_agents.append(i)
        
        # Prioritize SDC if present
        sdc_idx = scenario.sdc_track_index
        if sdc_idx in valid_agents:
            valid_agents.remove(sdc_idx)
            valid_agents.insert(0, sdc_idx)
        
        # Limit number of agents if specified
        if num_agents_to_predict is not None and len(valid_agents) > num_agents_to_predict:
            valid_agents = valid_agents[:num_agents_to_predict]
        
        # Fallback: if no valid agents, use SDC anyway
        if len(valid_agents) == 0:
            valid_agents = [scenario.sdc_track_index]
        
        return valid_agents
    
    def _get_target_agent_state(self, scenario) -> Tuple[int, float, float, float, float]:
        """Get target agent index and current state for normalization.
        
        Uses SDC (self-driving car) as the reference for normalization.
        
        Returns:
            (target_idx, origin_x, origin_y, origin_z, origin_yaw)
        """
        # Use SDC as the reference for coordinate normalization
        target_idx = scenario.sdc_track_index
        
        # Get current state (last history timestep)
        track = scenario.tracks[target_idx]
        current_state = track.states[self.history_len - 1] if len(track.states) > self.history_len - 1 else track.states[-1]
        
        origin_x = current_state.center_x
        origin_y = current_state.center_y
        origin_z = current_state.center_z if hasattr(current_state, 'center_z') else 0.0
        origin_yaw = current_state.heading if hasattr(current_state, 'heading') else 0.0
        
        return target_idx, origin_x, origin_y, origin_z, origin_yaw
    
    def _rotate_point(self, x: float, y: float, yaw: float) -> Tuple[float, float]:
        """Rotate point by -yaw to align with target agent heading."""
        c = math.cos(-yaw)
        s = math.sin(-yaw)
        return c * x - s * y, s * x + c * y
    
    def _extract_agent_polylines(
        self, 
        scenario, 
        target_idx: int,
        origin_x: float, 
        origin_y: float, 
        origin_z: float,
        origin_yaw: float
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, Dict[int, int]]:
        """Extract agent trajectory polylines.
        
        Each agent's past trajectory becomes a polyline.
        Each vector: [ds_x, ds_y, ds_z, de_x, de_y, de_z, vx, vy, heading, width, length, type_onehot(4), timestamp, polyline_id]
        
        Returns:
            agent_vectors: [N_vectors, feature_dim]
            agent_polyline_ids: [N_vectors]
            target_polyline_idx: Index of target agent's polyline (for SDC/reference)
            num_polylines: Total number of agent polylines
            track_to_polyline: Dict mapping track index to polyline index
        """
        agent_vectors = []
        agent_polyline_ids = []
        polyline_id = 0
        target_polyline_idx = 0
        track_to_polyline: Dict[int, int] = {}  # Maps track index to polyline index
        
        for agent_idx, track in enumerate(scenario.tracks):
            if polyline_id >= self.max_agents:
                break
            
            # Check if agent has valid history
            valid_history = []
            for t in range(self.history_len):
                if t < len(track.states) and track.states[t].valid:
                    valid_history.append(t)
            
            if len(valid_history) < 2:  # Need at least 2 points for a polyline
                continue
            
            # Store mapping from track index to polyline index
            track_to_polyline[agent_idx] = polyline_id
            
            # Mark target agent (SDC for reference)
            if agent_idx == target_idx:
                target_polyline_idx = polyline_id
            
            # Create vectors from consecutive valid timesteps
            vectors_this_polyline = []
            for i in range(len(valid_history) - 1):
                if len(vectors_this_polyline) >= self.max_polyline_vectors:
                    break
                    
                t_start = valid_history[i]
                t_end = valid_history[i + 1]
                
                state_start = track.states[t_start]
                state_end = track.states[t_end]
                
                # Transform to local coordinates
                if self.normalize:
                    ds_x, ds_y = self._rotate_point(
                        state_start.center_x - origin_x,
                        state_start.center_y - origin_y,
                        origin_yaw
                    )
                    de_x, de_y = self._rotate_point(
                        state_end.center_x - origin_x,
                        state_end.center_y - origin_y,
                        origin_yaw
                    )
                    ds_z = (state_start.center_z if hasattr(state_start, 'center_z') else 0.0) - origin_z
                    de_z = (state_end.center_z if hasattr(state_end, 'center_z') else 0.0) - origin_z
                    
                    # Rotate velocity
                    vx, vy = self._rotate_point(
                        state_end.velocity_x,
                        state_end.velocity_y,
                        origin_yaw
                    )
                else:
                    ds_x, ds_y = state_start.center_x, state_start.center_y
                    de_x, de_y = state_end.center_x, state_end.center_y
                    ds_z = state_start.center_z if hasattr(state_start, 'center_z') else 0.0
                    de_z = state_end.center_z if hasattr(state_end, 'center_z') else 0.0
                    vx, vy = state_end.velocity_x, state_end.velocity_y
                
                # Heading (relative to origin if normalizing)
                heading = state_end.heading if hasattr(state_end, 'heading') else 0.0
                if self.normalize:
                    heading = heading - origin_yaw
                
                # Agent dimensions
                width = track.width if hasattr(track, 'width') else 2.0
                length = track.length if hasattr(track, 'length') else 4.5
                
                # Agent type one-hot
                obj_type = track.object_type if hasattr(track, 'object_type') else 0
                type_onehot = [0.0, 0.0, 0.0, 0.0]
                if 1 <= obj_type <= 4:
                    type_onehot[obj_type - 1] = 1.0
                
                # Timestamp (normalized to [0, 1] within history)
                timestamp = t_end / self.history_len
                
                vector = [
                    ds_x, ds_y, ds_z,
                    de_x, de_y, de_z,
                    vx, vy,
                    heading,
                    width, length,
                ] + type_onehot + [timestamp]
                
                vectors_this_polyline.append(vector)
            
            if len(vectors_this_polyline) > 0:
                for vec in vectors_this_polyline:
                    agent_vectors.append(vec)
                    agent_polyline_ids.append(polyline_id)
                polyline_id += 1
        
        if len(agent_vectors) == 0:
            # Return empty tensors with correct dimensions
            return (
                torch.zeros((0, 16), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                0,
                0,
                {}  # Empty track_to_polyline mapping
            )
        
        return (
            torch.tensor(agent_vectors, dtype=torch.float32),
            torch.tensor(agent_polyline_ids, dtype=torch.long),
            target_polyline_idx,
            polyline_id,
            track_to_polyline
        )
    
    def _extract_map_polylines(
        self,
        scenario,
        origin_x: float,
        origin_y: float, 
        origin_z: float,
        origin_yaw: float,
        start_polyline_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Extract map feature polylines (lanes, roads, crosswalks).
        
        Each vector: [ds_x, ds_y, ds_z, de_x, de_y, de_z, type_onehot(7), polyline_id]
        
        Returns:
            map_vectors: [N_vectors, feature_dim]
            map_polyline_ids: [N_vectors]
            num_polylines: Total number of map polylines
        """
        map_vectors = []
        map_polyline_ids = []
        polyline_id = start_polyline_id
        
        for feature in scenario.map_features:
            if polyline_id - start_polyline_id >= self.max_map_polylines:
                break
            
            feature_type = feature.WhichOneof('feature_data')
            
            # Get polyline points based on feature type
            points = []
            type_idx = self.MAP_FEATURE_TYPES.get(feature_type, 0)
            
            if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
                points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.lane.polyline]
            elif feature_type == 'road_line' and hasattr(feature.road_line, 'polyline'):
                points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.road_line.polyline]
            elif feature_type == 'road_edge' and hasattr(feature.road_edge, 'polyline'):
                points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.road_edge.polyline]
            elif feature_type == 'crosswalk' and hasattr(feature.crosswalk, 'polygon'):
                points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.crosswalk.polygon]
            elif feature_type == 'stop_sign':
                # Stop signs are points, not polylines
                if hasattr(feature.stop_sign, 'position'):
                    pos = feature.stop_sign.position
                    points = [(pos.x, pos.y, getattr(pos, 'z', 0.0))]
            
            if len(points) < 2:
                continue
            
            # Create vectors from consecutive points
            vectors_this_polyline = []
            for i in range(min(len(points) - 1, self.max_polyline_vectors)):
                p_start = points[i]
                p_end = points[i + 1]
                
                # Transform to local coordinates
                if self.normalize:
                    ds_x, ds_y = self._rotate_point(
                        p_start[0] - origin_x,
                        p_start[1] - origin_y,
                        origin_yaw
                    )
                    de_x, de_y = self._rotate_point(
                        p_end[0] - origin_x,
                        p_end[1] - origin_y,
                        origin_yaw
                    )
                    ds_z = p_start[2] - origin_z
                    de_z = p_end[2] - origin_z
                else:
                    ds_x, ds_y, ds_z = p_start
                    de_x, de_y, de_z = p_end
                
                # Type one-hot
                type_onehot = [0.0] * 7
                type_onehot[type_idx] = 1.0
                
                vector = [ds_x, ds_y, ds_z, de_x, de_y, de_z] + type_onehot
                vectors_this_polyline.append(vector)
            
            if len(vectors_this_polyline) > 0:
                for vec in vectors_this_polyline:
                    map_vectors.append(vec)
                    map_polyline_ids.append(polyline_id)
                polyline_id += 1
        
        if len(map_vectors) == 0:
            return (
                torch.zeros((0, 13), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                polyline_id
            )
        
        return (
            torch.tensor(map_vectors, dtype=torch.float32),
            torch.tensor(map_polyline_ids, dtype=torch.long),
            polyline_id
        )
    
    def _extract_future_trajectory(
        self,
        scenario,
        target_idx: int,
        origin_x: float,
        origin_y: float,
        origin_yaw: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract future trajectory labels for target agent.
        
        Returns:
            future_positions: [future_len, 2] - (x, y) positions
            future_valid: [future_len] - validity mask
        """
        track = scenario.tracks[target_idx]
        
        future_positions = []
        future_valid = []
        
        for t in range(self.future_len):
            future_t = self.history_len + t
            
            if future_t < len(track.states) and track.states[future_t].valid:
                state = track.states[future_t]
                
                if self.normalize:
                    x, y = self._rotate_point(
                        state.center_x - origin_x,
                        state.center_y - origin_y,
                        origin_yaw
                    )
                else:
                    x, y = state.center_x, state.center_y
                
                future_positions.append([x, y])
                future_valid.append(1.0)
            else:
                # Invalid future step - use zero and mark as invalid
                future_positions.append([0.0, 0.0])
                future_valid.append(0.0)
        
        return (
            torch.tensor(future_positions, dtype=torch.float32),
            torch.tensor(future_valid, dtype=torch.float32)
        )
    
    def _extract_multi_agent_futures(
        self,
        scenario,
        target_track_indices: List[int],
        track_to_polyline: Dict[int, int],
        origin_x: float,
        origin_y: float,
        origin_yaw: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract future trajectories for multiple target agents.
        
        Args:
            scenario: Waymo scenario proto
            target_track_indices: List of track indices to predict for
            track_to_polyline: Dict mapping track index to polyline index
            origin_x, origin_y, origin_yaw: Normalization origin
            
        Returns:
            target_polyline_indices: [num_targets] polyline indices in the scene
            future_positions: [num_targets, future_len, 2]
            future_valid: [num_targets, future_len]
        """
        target_polyline_indices = []
        all_future_positions = []
        all_future_valid = []
        
        for track_idx in target_track_indices:
            # Get the polyline index for this track
            if track_idx not in track_to_polyline:
                continue  # Agent didn't have valid history (was skipped)
            
            polyline_idx = track_to_polyline[track_idx]
            target_polyline_indices.append(polyline_idx)
            
            # Extract future trajectory for this agent
            future_pos, future_val = self._extract_future_trajectory(
                scenario, track_idx, origin_x, origin_y, origin_yaw
            )
            all_future_positions.append(future_pos)
            all_future_valid.append(future_val)
        
        if len(target_polyline_indices) == 0:
            # Fallback: no valid agents found
            return (
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0, self.future_len, 2), dtype=torch.float32),
                torch.zeros((0, self.future_len), dtype=torch.float32)
            )
        
        return (
            torch.tensor(target_polyline_indices, dtype=torch.long),
            torch.stack(all_future_positions, dim=0),  # [num_targets, future_len, 2]
            torch.stack(all_future_valid, dim=0)       # [num_targets, future_len]
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single scenario for VectorNet with multi-agent prediction.
        
        Returns a dictionary with:
            - agent_vectors: [N_agent_vectors, agent_feature_dim]
            - agent_polyline_ids: [N_agent_vectors]
            - map_vectors: [N_map_vectors, map_feature_dim] 
            - map_polyline_ids: [N_map_vectors]
            - target_polyline_indices: [num_targets] polyline indices for target agents
            - future_positions: [num_targets, future_len, 2]
            - future_valid: [num_targets, future_len]
            - num_targets: number of target agents
            - scenario_id: Scenario identifier
        """
        file_idx, scenario_idx = self.scenario_index[idx]
        
        # Check cache
        cache_key = (file_idx, scenario_idx)
        if self.cache_scenarios and cache_key in self._scenario_cache:
            return self._scenario_cache[cache_key]
        
        # Parse scenario
        scenario = self._parse_scenario(file_idx, scenario_idx)
        
        # Get SDC as the reference for coordinate normalization
        ref_idx, origin_x, origin_y, origin_z, origin_yaw = self._get_target_agent_state(scenario)
        
        # Extract agent polylines (returns track_to_polyline mapping)
        agent_vectors, agent_polyline_ids, sdc_polyline_idx, num_agent_polylines, track_to_polyline = \
            self._extract_agent_polylines(scenario, ref_idx, origin_x, origin_y, origin_z, origin_yaw)
        
        # Extract map polylines
        map_vectors, map_polyline_ids, total_polylines = \
            self._extract_map_polylines(scenario, origin_x, origin_y, origin_z, origin_yaw, num_agent_polylines)
        
        # Get valid agents for prediction
        target_track_indices = self._get_valid_agents_for_prediction(
            scenario, 
            num_agents_to_predict=self.num_agents_to_predict
        )
        
        # Extract future trajectories for all target agents
        target_polyline_indices, future_positions, future_valid = self._extract_multi_agent_futures(
            scenario, target_track_indices, track_to_polyline, origin_x, origin_y, origin_yaw
        )
        
        result = {
            'agent_vectors': agent_vectors,
            'agent_polyline_ids': agent_polyline_ids,
            'map_vectors': map_vectors,
            'map_polyline_ids': map_polyline_ids,
            'target_polyline_indices': target_polyline_indices,  # [num_targets]
            'num_agent_polylines': torch.tensor(num_agent_polylines, dtype=torch.long),
            'future_positions': future_positions,                # [num_targets, future_len, 2]
            'future_valid': future_valid,                        # [num_targets, future_len]
            'num_targets': torch.tensor(len(target_polyline_indices), dtype=torch.long),
            'scenario_id': scenario.scenario_id,
        }
        
        # Cache if enabled
        if self.cache_scenarios:
            self._scenario_cache[cache_key] = result
        
        return result


def vectornet_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for VectorNet dataset with multi-agent support.
    
    Handles variable-length polylines by concatenating and tracking batch indices.
    Multi-agent prediction is handled by flattening all target agents across scenarios.
    """
    batch_size = len(batch)
    
    # Concatenate agent vectors with batch tracking
    agent_vectors_list = []
    agent_polyline_ids_list = []
    agent_batch_list = []
    
    # Concatenate map vectors with batch tracking  
    map_vectors_list = []
    map_polyline_ids_list = []
    map_batch_list = []
    
    # Multi-agent target tracking
    # Each target gets: (polyline_idx, scenario_batch_idx)
    all_target_polyline_indices = []  # Flattened list of all target polyline indices
    all_target_scenario_batch = []     # Which scenario each target belongs to
    all_future_positions = []          # [total_targets, future_len, 2]
    all_future_valid = []              # [total_targets, future_len]
    num_targets_per_scenario = []      # How many targets in each scenario
    
    scenario_ids = []
    
    agent_polyline_offset = 0
    
    for b_idx, sample in enumerate(batch):
        # Agent data
        agent_vectors_list.append(sample['agent_vectors'])
        agent_polyline_ids_list.append(sample['agent_polyline_ids'] + agent_polyline_offset)
        agent_batch_list.append(torch.full((len(sample['agent_vectors']),), b_idx, dtype=torch.long))
        
        # Map data
        map_vectors_list.append(sample['map_vectors'])
        map_polyline_ids_list.append(sample['map_polyline_ids'] + agent_polyline_offset)
        map_batch_list.append(torch.full((len(sample['map_vectors']),), b_idx, dtype=torch.long))
        
        # Multi-agent targets
        target_indices = sample['target_polyline_indices']  # [num_targets]
        num_targets = len(target_indices)
        num_targets_per_scenario.append(num_targets)
        
        # Offset target polyline indices and flatten
        for t in range(num_targets):
            all_target_polyline_indices.append(target_indices[t].item() + agent_polyline_offset)
            all_target_scenario_batch.append(b_idx)
        
        # Flatten future positions and valid masks
        # sample['future_positions'] is [num_targets, future_len, 2]
        for t in range(num_targets):
            all_future_positions.append(sample['future_positions'][t])
            all_future_valid.append(sample['future_valid'][t])
        
        scenario_ids.append(sample['scenario_id'])
        
        # Update offset (only agent polylines, as map polylines are concatenated with agent polylines)
        num_agent_polylines = sample['num_agent_polylines'].item()
        agent_polyline_offset += num_agent_polylines
    
    # Handle empty batches
    total_targets = sum(num_targets_per_scenario)
    
    return {
        'agent_vectors': torch.cat(agent_vectors_list, dim=0) if agent_vectors_list else torch.zeros((0, 16)),
        'agent_polyline_ids': torch.cat(agent_polyline_ids_list, dim=0) if agent_polyline_ids_list else torch.zeros((0,), dtype=torch.long),
        'agent_batch': torch.cat(agent_batch_list, dim=0) if agent_batch_list else torch.zeros((0,), dtype=torch.long),
        'map_vectors': torch.cat(map_vectors_list, dim=0) if map_vectors_list else torch.zeros((0, 13)),
        'map_polyline_ids': torch.cat(map_polyline_ids_list, dim=0) if map_polyline_ids_list else torch.zeros((0,), dtype=torch.long),
        'map_batch': torch.cat(map_batch_list, dim=0) if map_batch_list else torch.zeros((0,), dtype=torch.long),
        # Multi-agent support
        'target_polyline_indices': torch.tensor(all_target_polyline_indices, dtype=torch.long),  # [total_targets]
        'target_scenario_batch': torch.tensor(all_target_scenario_batch, dtype=torch.long),      # [total_targets]
        'future_positions': torch.stack(all_future_positions, dim=0) if all_future_positions else torch.zeros((0, 50, 2)),  # [total_targets, future_len, 2]
        'future_valid': torch.stack(all_future_valid, dim=0) if all_future_valid else torch.zeros((0, 50)),                  # [total_targets, future_len]
        'num_targets_per_scenario': torch.tensor(num_targets_per_scenario, dtype=torch.long),    # [batch_size]
        'total_targets': total_targets,
        'batch_size': batch_size,
        'scenario_ids': scenario_ids,
    }


def create_vectornet_dataloaders(
    tfrecord_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    history_len: int = 10,
    future_len: int = 50,
    max_train_scenarios: Optional[int] = None,
    max_val_scenarios: Optional[int] = None,
    num_agents_to_predict: Optional[int] = 8,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for VectorNet.
    
    Args:
        tfrecord_dir: Base directory (e.g., 'data/scenario')
        batch_size: Batch size
        num_workers: Number of data loading workers
        history_len: Number of history timesteps
        future_len: Number of future timesteps to predict
        max_train_scenarios: Max training scenarios (None = all)
        max_val_scenarios: Max validation scenarios (None = all)
        num_agents_to_predict: Number of agents to predict for per scenario
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = VectorNetTFRecordDataset(
        tfrecord_dir=tfrecord_dir,
        split='training',
        history_len=history_len,
        future_len=future_len,
        max_scenarios=max_train_scenarios,
        num_agents_to_predict=num_agents_to_predict,
    )
    
    val_dataset = VectorNetTFRecordDataset(
        tfrecord_dir=tfrecord_dir,
        split='validation',
        history_len=history_len,
        future_len=future_len,
        max_scenarios=max_val_scenarios,
        num_agents_to_predict=num_agents_to_predict,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=vectornet_collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vectornet_collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# For backward compatibility with existing code
class VectorNetDatasetWrapper(Dataset):
    """Wrapper that converts TFRecord dataset output to PyG Data format.
    
    This allows using the TFRecord dataset with existing PyG-based training code.
    """
    
    def __init__(self, tfrecord_dataset: VectorNetTFRecordDataset):
        self.dataset = tfrecord_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Data:
        sample = self.dataset[idx]
        
        # Combine agent and map vectors
        all_vectors = torch.cat([
            sample['agent_vectors'],
            sample['map_vectors']
        ], dim=0) if sample['map_vectors'].numel() > 0 else sample['agent_vectors']
        
        all_polyline_ids = torch.cat([
            sample['agent_polyline_ids'],
            sample['map_polyline_ids']
        ], dim=0) if sample['map_polyline_ids'].numel() > 0 else sample['agent_polyline_ids']
        
        # Create agent mask (True for agent polylines, False for map)
        agent_mask = torch.zeros(all_vectors.shape[0], dtype=torch.bool)
        agent_mask[:len(sample['agent_vectors'])] = True
        
        data = Data(
            x=all_vectors,
            polyline_ids=all_polyline_ids,
            agent_mask=agent_mask,
            target_polyline_idx=sample['target_polyline_idx'],
            y=sample['future_positions'],  # [future_len, 2]
            y_valid=sample['future_valid'],  # [future_len]
            scenario_id=sample['scenario_id'],
        )
        
        return data


if __name__ == '__main__':
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/scenario',
                       help='Path to scenario directory')
    parser.add_argument('--split', type=str, default='training',
                       help='Data split to test')
    parser.add_argument('--max_scenarios', type=int, default=10,
                       help='Max scenarios to load for testing')
    args = parser.parse_args()
    
    print(f"Testing VectorNetTFRecordDataset with {args.data_dir}/{args.split}")
    
    dataset = VectorNetTFRecordDataset(
        tfrecord_dir=args.data_dir,
        split=args.split,
        max_scenarios=args.max_scenarios,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print("\nSample keys:", list(sample.keys()))
    print(f"Agent vectors shape: {sample['agent_vectors'].shape}")
    print(f"Agent polyline IDs shape: {sample['agent_polyline_ids'].shape}")
    print(f"Map vectors shape: {sample['map_vectors'].shape}")
    print(f"Map polyline IDs shape: {sample['map_polyline_ids'].shape}")
    print(f"Target polyline idx: {sample['target_polyline_idx']}")
    print(f"Future positions shape: {sample['future_positions'].shape}")
    print(f"Future valid shape: {sample['future_valid'].shape}")
    
    # Test dataloader
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=vectornet_collate_fn,
        num_workers=0,
    )
    
    batch = next(iter(loader))
    print("\nBatch keys:", list(batch.keys()))
    print(f"Batch agent vectors shape: {batch['agent_vectors'].shape}")
    print(f"Batch future positions shape: {batch['future_positions'].shape}")
