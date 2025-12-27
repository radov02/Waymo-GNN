"""VectorNet Dataset for Direct TFRecord Loading from Waymo Open Motion Dataset, extracting agent trajectory polylines,
map feature polylines, traffic light states and multi-step future trajectory labels."""
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import os
import sys
import glob
import math
from typing import Dict, List, Tuple, Optional, Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not available. Using tfrecord package instead.")
try:
    from waymo_open_dataset.protos import scenario_pb2
    HAS_WAYMO_SDK = True
except ImportError:
    HAS_WAYMO_SDK = False
    print("Warning: Waymo Open Dataset SDK not available.")


class VectorNetTFRecordDataset(Dataset):
    """VectorNet dataset that directly loads from Waymo TFRecord files.
    Each vector in a polyline: [ds_x, ds_y, ds_z, de_x, de_y, de_z, attributes..., polyline_id]"""
    
    MAP_FEATURE_TYPES = {
        'lane': 0,
        'road_line': 1, 
        'road_edge': 2,
        'stop_sign': 3,
        'crosswalk': 4,
        'speed_bump': 5,
        'driveway': 6,
    }
    
    AGENT_TYPES = {
        0: 'unset',
        1: 'vehicle',
        2: 'pedestrian', 
        3: 'cyclist',
        4: 'other',
    }
    
    def __init__(
        self,
        tfrecord_dir: str,      # base directory containing scenario/training, scenario/validation, etc.
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
        map_feature_radius: float = 100.0,  # Only include map features within this distance (meters)
    ):
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
        self.map_feature_radius = map_feature_radius
        self._scenario_cache: Dict[int, Any] = {}
        self.data_dir = os.path.join(tfrecord_dir, split)
        self.tfrecord_files = sorted(glob.glob(os.path.join(self.data_dir, "*.tfrecord")))
        if len(self.tfrecord_files) == 0:
            raise ValueError(f"No TFRecord files found in {self.data_dir}")
        print(f"Found {len(self.tfrecord_files)} TFRecord files in {self.data_dir}")
        print(f"Predicting for {num_agents_to_predict if num_agents_to_predict else 'all valid'} agents per scenario")
        print(f"Map feature filtering: {map_feature_radius:.1f}m radius (closest features prioritized)")
        self._build_index()     # build index: (file_idx, scenario_idx_in_file) -> global_idx
    
    def _build_index(self):
        """build an index of all scenarios across all TFRecord files - necessary for batching and random access"""
        self.scenario_index: List[Tuple[int, int]] = []  # (file_idx, scenario_idx_in_file)
        print("Building scenario index...")
        for file_idx, tfrecord_path in enumerate(self.tfrecord_files):
            try:
                if HAS_TF:
                    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
                    scenario_count = 0
                    for _ in dataset:
                        scenario_count += 1
                        if self.max_scenarios and len(self.scenario_index) + scenario_count >= self.max_scenarios:
                            break
                else:
                    scenario_count = 50  # Fallback: assume fixed number per file
                
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
        """Get list of valid agent indices that can be predicted: an agent is valid for prediction if: 
        has valid state at current timestep (history_len - 1) and has at least some valid future states"""
        valid_agents = []
        
        for i, track in enumerate(scenario.tracks):
            if not hasattr(track, 'states') or len(track.states) <= self.history_len:
                continue
                
            if not track.states[self.history_len - 1].valid:        # check if current timestep is valid
                continue
            
            has_future = False      # check if agent has at least some valid future states
            for t in range(self.history_len, min(self.history_len + self.future_len, len(track.states))):
                if track.states[t].valid:
                    has_future = True
                    break
            
            if not has_future:
                continue
            
            valid_agents.append(i)
        
        sdc_idx = scenario.sdc_track_index
        if sdc_idx in valid_agents:
            valid_agents.remove(sdc_idx)
            valid_agents.insert(0, sdc_idx)   # prioritize SDC if present
        
        # limit number of agents if specified:
        if num_agents_to_predict is not None and len(valid_agents) > num_agents_to_predict:
            valid_agents = valid_agents[:num_agents_to_predict]
        
        if len(valid_agents) == 0:      # if no valid agents, use SDC
            valid_agents = [scenario.sdc_track_index]
        
        return valid_agents
    
    def _get_target_agent_state(self, scenario) -> Tuple[int, float, float, float, float]:
        """get target agent index and current state for normalization, use SDC (self-driving car) as the reference for normalization"""
        target_idx = scenario.sdc_track_index
        track = scenario.tracks[target_idx]
        # get current state (last history timestep):
        current_state = track.states[self.history_len - 1] if len(track.states) > self.history_len - 1 else track.states[-1]
        
        origin_x = current_state.center_x
        origin_y = current_state.center_y
        origin_z = current_state.center_z if hasattr(current_state, 'center_z') else 0.0
        origin_yaw = current_state.heading if hasattr(current_state, 'heading') else 0.0
        
        return target_idx, origin_x, origin_y, origin_z, origin_yaw
    
    def _rotate_point(self, x: float, y: float, yaw: float) -> Tuple[float, float]:
        """rotate point by -yaw angle to align with target agent heading"""
        c = math.cos(-yaw)
        s = math.sin(-yaw)
        return c * x - s * y, s * x + c * y
    
    def _extract_agent_polylines(self, scenario, target_idx: int, origin_x: float, origin_y: float, origin_z: float, origin_yaw: float) -> Tuple[torch.Tensor, torch.Tensor, int, int, Dict[int, int]]:
        """extract agent trajectory polylines, each agent's past trajectory becomes a polyline.
        Each vector: [ds_x, ds_y, ds_z, de_x, de_y, de_z, vx, vy, heading, width, length, type_onehot(4), timestamp, polyline_id]"""
        agent_vectors = []
        agent_polyline_ids = []
        polyline_id = 0
        target_polyline_idx = 0
        track_to_polyline: Dict[int, int] = {}  # Maps track index to polyline index
        
        for agent_idx, track in enumerate(scenario.tracks):
            if polyline_id >= self.max_agents:
                break
            
            valid_history = []
            for t in range(self.history_len):       # check if agent has valid history
                if t < len(track.states) and track.states[t].valid:
                    valid_history.append(t)
            if len(valid_history) < 2:  # need at least 2 points for a polyline
                continue
            
            track_to_polyline[agent_idx] = polyline_id      # store mapping from track index to polyline index
            
            if agent_idx == target_idx:     # mark target agent (SDC for reference)
                target_polyline_idx = polyline_id
            
            # create vectors from consecutive valid timesteps:
            vectors_this_polyline = []
            for i in range(len(valid_history) - 1):
                if len(vectors_this_polyline) >= self.max_polyline_vectors:
                    break
                    
                t_start = valid_history[i]
                t_end = valid_history[i + 1]
                
                # use SDC state as origin for normalization:
                state_start = track.states[t_start]
                state_end = track.states[t_end]
                
                # transform to local coordinates (around SDC):
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
                
                # heading (relative to origin if normalizing)
                heading = state_end.heading if hasattr(state_end, 'heading') else 0.0
                if self.normalize:
                    heading = heading - origin_yaw
                
                # agent dimensions
                width = track.width if hasattr(track, 'width') else 2.0
                length = track.length if hasattr(track, 'length') else 4.5
                
                # agent type one-hot
                obj_type = track.object_type if hasattr(track, 'object_type') else 0
                type_onehot = [0.0, 0.0, 0.0, 0.0]
                if 1 <= obj_type <= 4:
                    type_onehot[obj_type - 1] = 1.0
                
                # timestamp (normalized to [0, 1] within history)
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
            # return empty tensors with correct dimensions
            return (
                torch.zeros((0, 16), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                0,
                0,
                {}  # Empty track_to_polyline mapping
            )
        
        return (
            torch.tensor(agent_vectors, dtype=torch.float32),       # [N_vectors, feature_dim]
            torch.tensor(agent_polyline_ids, dtype=torch.long),     # [N_vectors]
            target_polyline_idx,                             # index of target agent's polyline
            polyline_id,                                   # total number of agent polylines
            track_to_polyline                          # mapping from track index to polyline index
        )
    
    def _extract_map_polylines(self, scenario, origin_x: float, origin_y: float,  origin_z: float, origin_yaw: float, start_polyline_id: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Extract map feature polylines (lanes, roads, crosswalks) with SPATIAL FILTERING.
        Only includes features within map_feature_radius of the origin (SDC position).
        Each vector: [ds_x, ds_y, ds_z, de_x, de_y, de_z, type_onehot(7), polyline_id]"""
        
        # STEP 1: Filter map features by distance to origin, then sort by distance
        # This ensures we get the MOST RELEVANT features (closest to prediction area)
        feature_distances = []
        
        for feature in scenario.map_features:
            feature_type = feature.WhichOneof('feature_data')
            
            # get polyline points based on feature type:
            points = []
            if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
                points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.lane.polyline]
            elif feature_type == 'road_line' and hasattr(feature.road_line, 'polyline'):
                points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.road_line.polyline]
            elif feature_type == 'road_edge' and hasattr(feature.road_edge, 'polyline'):
                points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.road_edge.polyline]
            elif feature_type == 'crosswalk' and hasattr(feature.crosswalk, 'polygon'):
                points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.crosswalk.polygon]
            elif feature_type == 'stop_sign':
                # stop signs are points, not polylines
                if hasattr(feature.stop_sign, 'position'):
                    pos = feature.stop_sign.position
                    points = [(pos.x, pos.y, getattr(pos, 'z', 0.0))]
            if len(points) < 2:
                continue
            
            # Compute minimum distance from any point in polyline to origin
            min_dist = float('inf')
            for px, py, _ in points:
                dist = math.sqrt((px - origin_x)**2 + (py - origin_y)**2)
                min_dist = min(dist, min_dist)
            
            # Only include features within radius
            if min_dist <= self.map_feature_radius:
                feature_distances.append((min_dist, feature, feature_type, points))
        
        # Sort by distance (closest first) to prioritize nearby features
        feature_distances.sort(key=lambda x: x[0])
        
        # STEP 2: Extract vectors from filtered and sorted features
        map_vectors = []
        map_polyline_ids = []
        polyline_id = start_polyline_id
        
        for _, feature, feature_type, points in feature_distances:
            if polyline_id - start_polyline_id >= self.max_map_polylines:
                break
            
            type_idx = self.MAP_FEATURE_TYPES.get(feature_type, 0)
            
            # create vectors from consecutive points:
            vectors_this_polyline = []
            for i in range(min(len(points) - 1, self.max_polyline_vectors)):
                p_start = points[i]
                p_end = points[i + 1]
                
                # transform to local coordinates (around SDC):
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
                
                # type one-hot
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
            torch.tensor(map_vectors, dtype=torch.float32),     # [N_vectors, feature_dim]
            torch.tensor(map_polyline_ids, dtype=torch.long),   # [N_vectors]
            polyline_id                                  # total number of map polylines
        )
    
    def _extract_future_trajectory(self, scenario, target_idx: int, origin_x: float, origin_y: float, origin_yaw: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """extract future trajectory labels for target agent"""
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
            torch.tensor(future_positions, dtype=torch.float32),        # [future_len, 2] - (x, y) positions
            torch.tensor(future_valid, dtype=torch.float32)          # [future_len] - validity mask
        )
    
    def _extract_multi_agent_futures(self, scenario, target_track_indices: List[int], track_to_polyline: Dict[int, int], origin_x: float, origin_y: float, origin_yaw: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """extract future trajectories for multiple target agents"""
        target_polyline_indices = []
        all_future_positions = []
        all_future_valid = []
        
        for track_idx in target_track_indices:
            if track_idx not in track_to_polyline:
                continue  # agent didn't have valid history (was skipped)
            
            polyline_idx = track_to_polyline[track_idx]     # get the polyline index for this track
            target_polyline_indices.append(polyline_idx)
            
            # extract future trajectory for this agent:
            future_pos, future_val = self._extract_future_trajectory(scenario, track_idx, origin_x, origin_y, origin_yaw)
            all_future_positions.append(future_pos)
            all_future_valid.append(future_val)
        
        if len(target_polyline_indices) == 0:
            # fallback: no valid agents found
            return (
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0, self.future_len, 2), dtype=torch.float32),
                torch.zeros((0, self.future_len), dtype=torch.float32)
            )
        
        return (
            torch.tensor(target_polyline_indices, dtype=torch.long),        # [num_targets] polyline indices in the scene
            torch.stack(all_future_positions, dim=0),  # [num_targets, future_len, 2]
            torch.stack(all_future_valid, dim=0)       # [num_targets, future_len]
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """get a single scenario for VectorNet with multi-agent prediction"""
        file_idx, scenario_idx = self.scenario_index[idx]
        
        cache_key = (file_idx, scenario_idx)        # check cache
        if self.cache_scenarios and cache_key in self._scenario_cache:
            return self._scenario_cache[cache_key]
        
        scenario = self._parse_scenario(file_idx, scenario_idx)
        
        # get SDC as the reference for coordinate normalization:
        ref_idx, origin_x, origin_y, origin_z, origin_yaw = self._get_target_agent_state(scenario)
        
        # extract agent polylines:
        agent_vectors, agent_polyline_ids, sdc_polyline_idx, num_agent_polylines, track_to_polyline = self._extract_agent_polylines(scenario, ref_idx, origin_x, origin_y, origin_z, origin_yaw)
        
        # extract map polylines:
        map_vectors, map_polyline_ids, total_polylines = self._extract_map_polylines(scenario, origin_x, origin_y, origin_z, origin_yaw, num_agent_polylines)
        
        # get valid agents for prediction:
        target_track_indices = self._get_valid_agents_for_prediction(scenario, num_agents_to_predict=self.num_agents_to_predict)
        
        # extract future trajectories for all target agents:
        target_polyline_indices, future_positions, future_valid = self._extract_multi_agent_futures(scenario, target_track_indices, track_to_polyline, origin_x, origin_y, origin_yaw)
        
        result = {
            'agent_vectors': agent_vectors,
            'agent_polyline_ids': agent_polyline_ids,
            'map_vectors': map_vectors,
            'map_polyline_ids': map_polyline_ids,
            'target_polyline_indices': target_polyline_indices,         # [num_targets]
            'num_agent_polylines': torch.tensor(num_agent_polylines, dtype=torch.long),
            'future_positions': future_positions,                   # [num_targets, future_len, 2]
            'future_valid': future_valid,                               # [num_targets, future_len]
            'num_targets': torch.tensor(len(target_polyline_indices), dtype=torch.long),
            'scenario_id': scenario.scenario_id,
        }
        
        if self.cache_scenarios:        # cache if enabled
            self._scenario_cache[cache_key] = result
        
        return result

def vectornet_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """collate function for VectorNet dataset with multi-agent support, handles variable-length polylines by concatenating and 
    tracking batch indices, multi-agent prediction is handled by flattening all target agents across scenarios."""
    batch_size = len(batch)
    
    # concatenate agent vectors with batch tracking:
    agent_vectors_list = []
    agent_polyline_ids_list = []
    agent_batch_list = []
    
    # concatenate map vectors with batch tracking:
    map_vectors_list = []
    map_polyline_ids_list = []
    map_batch_list = []
    
    # multi-agent target tracking:
    # each target gets: (polyline_idx, scenario_batch_idx)
    all_target_polyline_indices = []  # flattened list of all target polyline indices
    all_target_scenario_batch = []     # which scenario each target belongs to
    all_future_positions = []          # [total_targets, future_len, 2]
    all_future_valid = []              # [total_targets, future_len]
    num_targets_per_scenario = []      # how many targets in each scenario
    
    scenario_ids = []
    
    agent_polyline_offset = 0
    
    for b_idx, scenario in enumerate(batch):
        # agent data
        agent_vectors_list.append(scenario['agent_vectors'])
        agent_polyline_ids_list.append(scenario['agent_polyline_ids'] + agent_polyline_offset)
        agent_batch_list.append(torch.full((len(scenario['agent_vectors']),), b_idx, dtype=torch.long))
        
        # map data
        map_vectors_list.append(scenario['map_vectors'])
        map_polyline_ids_list.append(scenario['map_polyline_ids'] + agent_polyline_offset)
        map_batch_list.append(torch.full((len(scenario['map_vectors']),), b_idx, dtype=torch.long))
        
        # multi-agent targets
        target_indices = scenario['target_polyline_indices']  # [num_targets]
        num_targets = len(target_indices)
        num_targets_per_scenario.append(num_targets)
        
        # offset target polyline indices and flatten
        for t in range(num_targets):
            all_target_polyline_indices.append(target_indices[t].item() + agent_polyline_offset)
            all_target_scenario_batch.append(b_idx)
        
        # flatten future positions and valid masks
        # scenario['future_positions'] is [num_targets, future_len, 2]
        for t in range(num_targets):
            all_future_positions.append(scenario['future_positions'][t])
            all_future_valid.append(scenario['future_valid'][t])
        
        scenario_ids.append(scenario['scenario_id'])
        
        # Update offset (only agent polylines, as map polylines are concatenated with agent polylines)
        num_agent_polylines = scenario['num_agent_polylines'].item()
        agent_polyline_offset += num_agent_polylines
    
    # handle empty batches
    total_targets = sum(num_targets_per_scenario)
    
    return {
        'agent_vectors': torch.cat(agent_vectors_list, dim=0) if agent_vectors_list else torch.zeros((0, 16)),
        'agent_polyline_ids': torch.cat(agent_polyline_ids_list, dim=0) if agent_polyline_ids_list else torch.zeros((0,), dtype=torch.long),
        'agent_batch': torch.cat(agent_batch_list, dim=0) if agent_batch_list else torch.zeros((0,), dtype=torch.long),
        'map_vectors': torch.cat(map_vectors_list, dim=0) if map_vectors_list else torch.zeros((0, 13)),
        'map_polyline_ids': torch.cat(map_polyline_ids_list, dim=0) if map_polyline_ids_list else torch.zeros((0,), dtype=torch.long),
        'map_batch': torch.cat(map_batch_list, dim=0) if map_batch_list else torch.zeros((0,), dtype=torch.long),
        # multi-agent:
        'target_polyline_indices': torch.tensor(all_target_polyline_indices, dtype=torch.long),  # [total_targets]
        'target_scenario_batch': torch.tensor(all_target_scenario_batch, dtype=torch.long),      # [total_targets]
        'future_positions': torch.stack(all_future_positions, dim=0) if all_future_positions else torch.zeros((0, 50, 2)),  # [total_targets, future_len, 2]
        'future_valid': torch.stack(all_future_valid, dim=0) if all_future_valid else torch.zeros((0, 50)),                  # [total_targets, future_len]
        'num_targets_per_scenario': torch.tensor(num_targets_per_scenario, dtype=torch.long),    # [batch_size]
        'total_targets': total_targets,
        'batch_size': batch_size,
        'scenario_ids': scenario_ids,
    }

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
