import json
import math
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from tqdm import tqdm

from waymo_open_dataset.protos import scenario_pb2


WAYMO_TYPE_TO_GRIP = {
    1: 1,  # vehicle -> car decoder
    2: 3,  # pedestrian
    3: 4,  # cyclist
}

FEATURE_DIM = 11


@dataclass
class GripPreprocessConfig:
    """Configuration for Waymo->GRIP preprocessing at native 10Hz.
    
    Frame allocation (Waymo Motion Challenge standard):
    - Train/Val: 10 history (incl. current) + 79 future = 89 frames @ 10Hz
    - Test: 10 history (incl. current) + 0 future = 10 frames @ 10Hz (futures hidden)
    """
    max_agents: int = 64
    neighbor_distance: float = 10.0
    
    def get_frame_counts(self, split: str) -> Tuple[int, int]:
        """Return (history_frames, future_frames) for the given split."""
        if split in ("train", "val"):
            return 10, 79  # 89 frames total
        elif split == "test":
            return 10, 0   # 10 frames total (futures hidden)
        else:
            raise ValueError(f"Unknown split: {split}")

@dataclass
class GripData:
    features: np.ndarray  # (N, C, T, V) - coordinates centered on SDC at history cutoff
    adjacency: np.ndarray  # (N, V, V)
    mean_xy: np.ndarray  # (N, 2) - SDC position at history cutoff; used to reconstruct absolute coordinates
    agent_mask: np.ndarray  # (N, V) - 1.0 for real agents, 0.0 for padding


def load_sdc_track_ids_from_hdf5(hdf5_path: str) -> Dict[str, int]:
    """Load SDC track IDs directly from HDF5 file (much faster than scanning TFRecords)."""
    sdc_map: Dict[str, int] = {}
    
    with h5py.File(hdf5_path, "r") as f:
        for scenario_id in f["scenarios"].keys():
            scenario_group = f["scenarios"][scenario_id]
            # Get first timestep snapshot (all snapshots in a scenario have the same SDC ID)
            first_timestep = list(scenario_group["snapshot_graphs"].keys())[0]
            snap = scenario_group["snapshot_graphs"][first_timestep]
            
            if "sdc_track_id" in snap:
                sdc_map[scenario_id] = int(snap["sdc_track_id"][()])
            else:
                # Fallback: use first agent ID (may not be SDC, but better than nothing)
                agent_ids = snap["agent_ids"][:]
                sdc_map[scenario_id] = int(agent_ids[0])
    
    return sdc_map


def _map_type(one_hot: np.ndarray) -> int:
    idx = int(np.argmax(one_hot))
    waymo_type = idx + 1  # vehicle, pedestrian, cyclist, other
    return WAYMO_TYPE_TO_GRIP.get(waymo_type, -1)


def _build_agent_series(
    scenario_group: h5py.Group,
    ds_timesteps: List[int],
) -> Tuple[Dict[int, Dict[str, np.ndarray]], np.ndarray]:
    num_frames = len(ds_timesteps)
    frame_ids = np.array(ds_timesteps, dtype=np.int32)
    agent_series: Dict[int, Dict[str, np.ndarray]] = {}

    snapshots = scenario_group["snapshot_graphs"]
    for frame_idx, timestep in enumerate(ds_timesteps):
        snap = snapshots[str(timestep)]
        agent_ids = snap["agent_ids"][:]
        positions = snap["pos"][:]
        features = snap["x"][:]

        for row_idx, agent_id in enumerate(agent_ids):
            type_code = _map_type(features[row_idx, 5:9])
            if type_code == -1:
                continue  # drop "other" type

            valid = 1.0 if features[row_idx, 4] > 0.5 else 0.0
            series = agent_series.get(agent_id)
            if series is None:
                series = {
                    "type": type_code,
                    "x": np.zeros(num_frames, dtype=np.float32),
                    "y": np.zeros(num_frames, dtype=np.float32),
                    "heading": np.zeros(num_frames, dtype=np.float32),
                    "valid": np.zeros(num_frames, dtype=np.float32),
                }
                agent_series[agent_id] = series

            series["x"][frame_idx] = positions[row_idx, 0]
            series["y"][frame_idx] = positions[row_idx, 1]
            series["heading"][frame_idx] = features[row_idx, 3]
            series["valid"][frame_idx] = valid

    return agent_series, frame_ids


def _select_agents(
    agent_series: Dict[int, Dict[str, np.ndarray]],
    hist_frame_idx: int,
    sdc_id: int,
    max_agents: int,
) -> List[int]:
    if sdc_id not in agent_series:
        return []

    sdc = agent_series[sdc_id]
    if sdc["valid"][hist_frame_idx] < 0.5:
        return []

    base_x = sdc["x"][hist_frame_idx]
    base_y = sdc["y"][hist_frame_idx]

    candidates: List[Tuple[float, int]] = []
    for agent_id, series in agent_series.items():
        if agent_id == sdc_id:
            continue
        if series["valid"][hist_frame_idx] < 0.5:
            continue
        dx = series["x"][hist_frame_idx] - base_x
        dy = series["y"][hist_frame_idx] - base_y
        dist = math.hypot(float(dx), float(dy))
        candidates.append((dist, agent_id))

    candidates.sort(key=lambda item: item[0])
    selected = [sdc_id]
    for _, agent_id in candidates:
        if len(selected) >= max_agents:
            break
        selected.append(agent_id)

    return selected



def _build_adjacency(
    selected_ids: List[int],
    hist_frame_idx: int,
    agent_series: Dict[int, Dict[str, np.ndarray]],
    config: GripPreprocessConfig,
) -> np.ndarray:
    adj = np.zeros((config.max_agents, config.max_agents), dtype=np.float32)
    if not selected_ids:
        return adj

    coords = []
    for agent_id in selected_ids:
        series = agent_series[agent_id]
        coords.append([series["x"][hist_frame_idx], series["y"][hist_frame_idx]])
    coords = np.array(coords, dtype=np.float32)

    if len(coords) == 1:
        return adj

    dist_matrix = distance.cdist(coords, coords)
    mask = (dist_matrix <= config.neighbor_distance).astype(np.float32)
    mask = mask - np.eye(len(coords), dtype=np.float32)
    adj[: len(coords), : len(coords)] = np.clip(mask, 0.0, 1.0)
    return adj


def _generate_windows_for_scenario(
    scenario_group: h5py.Group,
    sdc_id: int,
    config: GripPreprocessConfig,
    split: str = "train",
    scenario_id: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Generate windows for a scenario at native 10Hz.
    
    Args:
        scenario_group: HDF5 group for one scenario
        sdc_id: Self-driving car track ID
        config: Preprocessing config
        split: "train", "val", or "test" - determines frame allocation
        scenario_id: Scenario ID for logging (optional)
    
    Returns:
        features, adjacencies, means, agent_masks - one per window
    """
    timesteps = sorted(int(key) for key in scenario_group["snapshot_graphs"].keys())
    
    # Get split-specific frame counts
    history_frames, future_frames = config.get_frame_counts(split)
    effective_total = history_frames + future_frames
    
    if len(timesteps) < effective_total:
        if scenario_id:
            print(f"  [SKIP] Scenario {scenario_id}: has {len(timesteps)} frames, need {effective_total}")
        return [], [], [], []

    agent_series, frame_ids = _build_agent_series(scenario_group, timesteps)
    if sdc_id not in agent_series:
        if scenario_id:
            print(f"  [SKIP] Scenario {scenario_id}: SDC ID {sdc_id} not found in agent series")
        return [], [], [], []

    features: List[np.ndarray] = []
    adjacencies: List[np.ndarray] = []
    means: List[np.ndarray] = []
    agent_masks: List[np.ndarray] = []

    for start in range(0, len(timesteps) - effective_total + 1):
        hist_frame_idx = start + history_frames - 1
        selected_ids = _select_agents(agent_series, hist_frame_idx, sdc_id, config.max_agents)
        if not selected_ids:
            continue

        # Build tensor with effective history only
        num_frames = effective_total
        tensor = np.zeros((config.max_agents, num_frames, FEATURE_DIM), dtype=np.float32)
        valid_mask = np.zeros((config.max_agents, num_frames), dtype=np.float32)
        agent_mask = np.zeros(config.max_agents, dtype=np.float32)

        for node_idx, agent_id in enumerate(selected_ids):
            agent_mask[node_idx] = 1.0
            series = agent_series[agent_id]
            for offset in range(num_frames):
                frame_idx = start + offset
                row = tensor[node_idx, offset]
                row[0] = frame_ids[frame_idx]
                row[1] = float(agent_id)
                row[2] = float(series["type"])
                row[9] = series["heading"][frame_idx]

                if series["valid"][frame_idx] >= 0.5:
                    row[3] = series["x"][frame_idx]
                    row[4] = series["y"][frame_idx]
                    row[10] = 1.0
                    valid_mask[node_idx, offset] = 1.0

        # Center on SDC position at history cutoff frame
        sdc_series = agent_series[selected_ids[0]]
        mean_xy = np.array(
            [sdc_series["x"][hist_frame_idx], sdc_series["y"][hist_frame_idx]], dtype=np.float32
        )

        tensor[: len(selected_ids), :, 3] -= mean_xy[0]
        tensor[: len(selected_ids), :, 4] -= mean_xy[1]

        # Transpose to (C, T, V) and append
        tensor = np.transpose(tensor, (2, 1, 0)).astype(np.float32)
        adjacency = _build_adjacency(selected_ids, hist_frame_idx, agent_series, config)

        features.append(tensor)
        adjacencies.append(adjacency)
        means.append(mean_xy)
        agent_masks.append(agent_mask)

    return features, adjacencies, means, agent_masks


def _stack_lists(
    feature_list: List[np.ndarray],
    adjacency_list: List[np.ndarray],
    mean_list: List[np.ndarray],
    agent_mask_list: List[np.ndarray],
) -> GripData:
    if not feature_list:
        raise ValueError("No windows extracted; check your configuration or input data.")

    features = np.stack(feature_list, axis=0)  # (N, C, T, V) - already transposed
    adjacency = np.stack(adjacency_list, axis=0)
    mean_xy = np.stack(mean_list, axis=0)
    agent_masks = np.stack(agent_mask_list, axis=0)  # (N, V)

    features = features.astype(np.float32)
    adjacency = adjacency.astype(np.float32)
    mean_xy = mean_xy.astype(np.float32)
    agent_masks = agent_masks.astype(np.float32)

    return GripData(features=features, adjacency=adjacency, mean_xy=mean_xy, agent_mask=agent_masks)


def _process_split(
    hdf5_path: str,
    split: str,
    config: GripPreprocessConfig,
) -> GripData:
    """Process a single split (train/val/test) from HDF5 and return stacked GripData.
    
    Args:
        hdf5_path: Path to split-specific HDF5 file (now contains SDC IDs)
        split: "train", "val", or "test"
        config: Preprocessing config
    
    Returns:
        GripData object for the split
    """
    sdc_map = load_sdc_track_ids_from_hdf5(hdf5_path)

    all_features: List[np.ndarray] = []
    all_adj: List[np.ndarray] = []
    all_mean: List[np.ndarray] = []
    all_agent_masks: List[np.ndarray] = []

    with h5py.File(hdf5_path, "r") as f:
        scenario_ids = list(f["scenarios"].keys())

        for scenario_id in tqdm(scenario_ids, desc=f"Converting {split} scenarios"):
            if scenario_id not in sdc_map:
                print(f"  [SKIP] Scenario {scenario_id}: no SDC ID found")
                continue  # Skip scenarios without SDC ID (shouldn't happen with new HDF5 format)

            scenario_group = f["scenarios"][scenario_id]
            feat, adj, mean, agent_masks = _generate_windows_for_scenario(
                scenario_group, sdc_map[scenario_id], config, split, scenario_id
            )
            all_features.extend(feat)
            all_adj.extend(adj)
            all_mean.extend(mean)
            all_agent_masks.extend(agent_masks)

    dataset = _stack_lists(all_features, all_adj, all_mean, all_agent_masks)
    return dataset


def convert_waymo_to_grip(
    hdf5_paths: Dict[str, str],
    output_dir: str,
    config: GripPreprocessConfig,
) -> Dict[str, str]:
    """Convert Waymo data to GRIP++ format for train/val/test splits.
    
    Args:
        hdf5_paths: Dict mapping split -> HDF5 path (must contain SDC track IDs)
                   e.g. {"train": "train.hdf5", "val": "val.hdf5", "test": "test.hdf5"}
        output_dir: Directory to save train/val/test pickles and config
        config: GripPreprocessConfig
    
    Returns:
        Dict mapping split -> output pickle path
        e.g. {"train": "train_data.pkl", "val": "val_data.pkl", "test": "test_data.pkl"}
    """
    print(f"Starting Waymo to GRIP++ conversion. Output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    split_samples = {}
    
    for split in ["train", "val", "test"]:
        if split not in hdf5_paths:
            print(f"Skipping {split} split: no HDF5 path provided.")
            continue
            
        hdf5_path = hdf5_paths[split]
        
        print(f"Processing {split} split from {hdf5_path}")
        dataset = _process_split(hdf5_path, split, config)
        
        output_path = os.path.join(output_dir, f"{split}_data.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(
                [dataset.features, dataset.adjacency, dataset.mean_xy, dataset.agent_mask],
                f,
            )
        
        output_paths[split] = output_path
        split_samples[split] = int(dataset.features.shape[0])
    
    # Save unified config with all split metadata
    config_path = os.path.join(output_dir, "preprocess_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json_dict = asdict(config)
        json_dict["splits"] = {}
        
        for split, num_samples in split_samples.items():
            json_dict["splits"][split] = {
                "hdf5_path": os.path.abspath(hdf5_paths.get(split, "")),
                "num_samples": num_samples,
            }
        
        json.dump(json_dict, f, indent=2)
    
    return output_paths
