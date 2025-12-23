"""Prepare VectorNet Data from Waymo TFRecord Files.

This script converts raw Waymo scenario TFRecord files into preprocessed
VectorNet-format TFRecord files containing agent polylines and map features.

Output files are saved as: #####.vectornet.tfrecord

VectorNetTFRecordDataset will read from:
- src/vectornet/data/training/*.vectornet.tfrecord
- src/vectornet/data/validation/*.vectornet.tfrecord
- src/vectornet/data/testing/*.vectornet.tfrecord

Usage:
    # Process all splits
    python src/vectornet/prepare_vectornet_data.py --source_dir data/scenario
    
    # Process specific split
    python src/vectornet/prepare_vectornet_data.py --source_dir data/scenario --split training
    
    # Limit number of scenarios
    python src/vectornet/prepare_vectornet_data.py --source_dir data/scenario --max_scenarios 1000
"""

import os
import sys
import glob
import argparse
import pickle
import math
from pathlib import Path
from tqdm import tqdm

# Try to import TensorFlow for TFRecord
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    print("ERROR: TensorFlow not available. Install with: pip install tensorflow")
    sys.exit(1)

# Try to import Waymo SDK
try:
    from waymo_open_dataset.protos import scenario_pb2
    HAS_WAYMO_SDK = True
except ImportError:
    print("ERROR: Waymo Open Dataset SDK not available.")
    print("Install with: pip install waymo-open-dataset-tf-2-11-0")
    sys.exit(1)


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


def rotate_point(x, y, yaw):
    """Rotate point by -yaw."""
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return c * x - s * y, s * x + c * y


def extract_agent_polylines(scenario, history_len=10, normalize=True):
    """Extract agent trajectory polylines from scenario.
    
    Returns dict with:
        - agent_vectors: list of vectors [ds_x, ds_y, ds_z, de_x, de_y, de_z, vx, vy, heading, width, length, type(4), timestamp]
        - agent_polyline_ids: list of polyline IDs for each vector
        - target_agent_idx: index of target agent
        - origin_x, origin_y, origin_z, origin_yaw: normalization origin
    """
    # Find target agent (SDC or first valid track)
    target_idx = scenario.sdc_track_index
    track = scenario.tracks[target_idx]
    
    # Get normalization origin from current state (last history timestep)
    current_state = track.states[history_len - 1] if len(track.states) > history_len - 1 else track.states[-1]
    origin_x = current_state.center_x
    origin_y = current_state.center_y
    origin_z = getattr(current_state, 'center_z', 0.0)
    origin_yaw = getattr(current_state, 'heading', 0.0)
    
    agent_vectors = []
    agent_polyline_ids = []
    polyline_id = 0
    
    # Process each track
    for agent_idx, track in enumerate(scenario.tracks):
        # Collect valid history states
        valid_history = []
        for t in range(history_len):
            if t < len(track.states) and track.states[t].valid:
                valid_history.append(t)
        
        if len(valid_history) < 2:
            continue
        
        # Create vectors from consecutive valid timesteps
        for i in range(len(valid_history) - 1):
            t_start = valid_history[i]
            t_end = valid_history[i + 1]
            
            state_start = track.states[t_start]
            state_end = track.states[t_end]
            
            # Transform to local coordinates
            if normalize:
                ds_x, ds_y = rotate_point(
                    state_start.center_x - origin_x,
                    state_start.center_y - origin_y,
                    origin_yaw
                )
                de_x, de_y = rotate_point(
                    state_end.center_x - origin_x,
                    state_end.center_y - origin_y,
                    origin_yaw
                )
                ds_z = (getattr(state_start, 'center_z', 0.0)) - origin_z
                de_z = (getattr(state_end, 'center_z', 0.0)) - origin_z
                
                vx, vy = rotate_point(
                    state_end.velocity_x,
                    state_end.velocity_y,
                    origin_yaw
                )
            else:
                ds_x, ds_y = state_start.center_x, state_start.center_y
                de_x, de_y = state_end.center_x, state_end.center_y
                ds_z = getattr(state_start, 'center_z', 0.0)
                de_z = getattr(state_end, 'center_z', 0.0)
                vx, vy = state_end.velocity_x, state_end.velocity_y
            
            heading = getattr(state_end, 'heading', 0.0)
            if normalize:
                heading = heading - origin_yaw
            
            width = getattr(track, 'width', 2.0)
            length = getattr(track, 'length', 4.5)
            
            # Agent type one-hot
            obj_type = getattr(track, 'object_type', 0)
            type_onehot = [0.0, 0.0, 0.0, 0.0]
            if 1 <= obj_type <= 4:
                type_onehot[obj_type - 1] = 1.0
            
            timestamp = t_end / history_len
            
            vector = [
                ds_x, ds_y, ds_z,
                de_x, de_y, de_z,
                vx, vy, heading,
                width, length,
            ] + type_onehot + [timestamp]
            
            agent_vectors.append(vector)
            agent_polyline_ids.append(polyline_id)
        
        if len(valid_history) >= 2:
            polyline_id += 1
    
    return {
        'agent_vectors': agent_vectors,
        'agent_polyline_ids': agent_polyline_ids,
        'target_agent_idx': target_idx,
        'num_agent_polylines': polyline_id,
        'origin_x': origin_x,
        'origin_y': origin_y,
        'origin_z': origin_z,
        'origin_yaw': origin_yaw,
    }


def extract_map_polylines(scenario, origin_x, origin_y, origin_z, origin_yaw, normalize=True):
    """Extract map feature polylines from scenario.
    
    Returns dict with:
        - map_vectors: list of vectors [ds_x, ds_y, ds_z, de_x, de_y, de_z, type_onehot(7)]
        - map_polyline_ids: list of polyline IDs for each vector
    """
    map_vectors = []
    map_polyline_ids = []
    polyline_id = 0
    
    for feature in scenario.map_features:
        feature_type = feature.WhichOneof('feature_data')
        
        # Get polyline points
        points = []
        type_idx = MAP_FEATURE_TYPES.get(feature_type, 0)
        
        if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
            points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.lane.polyline]
        elif feature_type == 'road_line' and hasattr(feature.road_line, 'polyline'):
            points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.road_line.polyline]
        elif feature_type == 'road_edge' and hasattr(feature.road_edge, 'polyline'):
            points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.road_edge.polyline]
        elif feature_type == 'crosswalk' and hasattr(feature.crosswalk, 'polygon'):
            points = [(p.x, p.y, getattr(p, 'z', 0.0)) for p in feature.crosswalk.polygon]
        
        if len(points) < 2:
            continue
        
        # Create vectors from consecutive points
        for i in range(len(points) - 1):
            p_start = points[i]
            p_end = points[i + 1]
            
            if normalize:
                ds_x, ds_y = rotate_point(
                    p_start[0] - origin_x,
                    p_start[1] - origin_y,
                    origin_yaw
                )
                de_x, de_y = rotate_point(
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
            map_vectors.append(vector)
            map_polyline_ids.append(polyline_id)
        
        if len(points) >= 2:
            polyline_id += 1
    
    return {
        'map_vectors': map_vectors,
        'map_polyline_ids': map_polyline_ids,
        'num_map_polylines': polyline_id,
    }


def extract_future_trajectory(scenario, target_idx, origin_x, origin_y, origin_yaw, 
                             history_len=10, future_len=50, normalize=True):
    """Extract future trajectory labels.
    
    Returns dict with:
        - future_positions: list of [x, y] positions
        - future_valid: list of validity flags
    """
    track = scenario.tracks[target_idx]
    
    future_positions = []
    future_valid = []
    
    for t in range(future_len):
        future_t = history_len + t
        
        if future_t < len(track.states) and track.states[future_t].valid:
            state = track.states[future_t]
            
            if normalize:
                x, y = rotate_point(
                    state.center_x - origin_x,
                    state.center_y - origin_y,
                    origin_yaw
                )
            else:
                x, y = state.center_x, state.center_y
            
            future_positions.append([x, y])
            future_valid.append(1.0)
        else:
            future_positions.append([0.0, 0.0])
            future_valid.append(0.0)
    
    return {
        'future_positions': future_positions,
        'future_valid': future_valid,
    }


def process_scenario(scenario, history_len=10, future_len=50):
    """Process a single scenario and extract VectorNet features."""
    try:
        # Extract agent polylines
        agent_data = extract_agent_polylines(scenario, history_len=history_len)
        
        # Extract map polylines
        map_data = extract_map_polylines(
            scenario,
            agent_data['origin_x'],
            agent_data['origin_y'],
            agent_data['origin_z'],
            agent_data['origin_yaw']
        )
        
        # Extract future trajectory
        future_data = extract_future_trajectory(
            scenario,
            agent_data['target_agent_idx'],
            agent_data['origin_x'],
            agent_data['origin_y'],
            agent_data['origin_yaw'],
            history_len=history_len,
            future_len=future_len
        )
        
        # Combine all data
        return {
            'scenario_id': scenario.scenario_id,
            'agent_vectors': agent_data['agent_vectors'],
            'agent_polyline_ids': agent_data['agent_polyline_ids'],
            'num_agent_polylines': agent_data['num_agent_polylines'],
            'map_vectors': map_data['map_vectors'],
            'map_polyline_ids': map_data['map_polyline_ids'],
            'num_map_polylines': map_data['num_map_polylines'],
            'future_positions': future_data['future_positions'],
            'future_valid': future_data['future_valid'],
            'target_agent_idx': agent_data['target_agent_idx'],
        }
    except Exception as e:
        print(f"Error processing scenario {scenario.scenario_id}: {e}")
        return None


def write_vectornet_tfrecord(data_list, output_path):
    """Write processed VectorNet data to TFRecord file."""
    with tf.io.TFRecordWriter(output_path) as writer:
        for data in data_list:
            # Serialize the data dict using pickle
            serialized = pickle.dumps(data)
            
            # Create TFRecord example
            example = tf.train.Example(features=tf.train.Features(feature={
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized]))
            }))
            
            writer.write(example.SerializeToString())


def find_tfrecord_files(base_dir):
    """Find all TFRecord files in directory structure.
    
    Looks for patterns like:
    - training/*.tfrecord
    - validation/*.tfrecord  
    - testing/*.tfrecord
    """
    splits = {}
    
    # Try to find scenario subdirectory structure
    scenario_dir = os.path.join(base_dir, 'scenario')
    if os.path.exists(scenario_dir):
        base_dir = scenario_dir
    
    for split in ['training', 'validation', 'testing']:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            # Exclude .vectornet.tfrecord files (our output format)
            files = sorted([f for f in glob.glob(os.path.join(split_dir, '*.tfrecord'))
                          if not f.endswith('.vectornet.tfrecord')])
            if files:
                splits[split] = files
                print(f"Found {len(files)} source TFRecord files in {split_dir}")
        else:
            # Try without split subdirectory
            pattern = os.path.join(base_dir, f'*{split}*.tfrecord')
            files = sorted([f for f in glob.glob(pattern)
                          if not f.endswith('.vectornet.tfrecord')])
            if files:
                splits[split] = files
                print(f"Found {len(files)} {split} source TFRecord files in {base_dir}")
    
    return splits


def process_tfrecord_file(input_file, history_len=10, future_len=50):
    """Process a single TFRecord file and extract all scenarios."""
    dataset = tf.data.TFRecordDataset(input_file, compression_type='')
    processed_scenarios = []
    
    for raw_record in dataset:
        try:
            scenario = scenario_pb2.Scenario.FromString(raw_record.numpy())
            data = process_scenario(scenario, history_len=history_len, future_len=future_len)
            if data is not None:
                processed_scenarios.append(data)
        except Exception as e:
            print(f"  Error parsing record: {e}")
            continue
    
    return processed_scenarios


def setup_target_directories(target_base=None):
    """Create target directory structure."""
    if target_base is None:
        target_base = os.path.join(os.path.dirname(__file__), 'data')
    for split in ['training', 'validation', 'testing']:
        target_dir = os.path.join(target_base, split)
        os.makedirs(target_dir, exist_ok=True)
    return target_base


def process_split(source_files, target_dir, split_name, history_len=10, future_len=50, max_scenarios=None):
    """Process all files in a split."""
    print(f"\nProcessing {split_name.upper()}:")
    print(f"  Source files: {len(source_files)}")
    print(f"  Target directory: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    total_scenarios = 0
    
    for file_idx, source_file in enumerate(tqdm(source_files, desc=f"  {split_name}")):
        # Generate output filename
        basename = os.path.basename(source_file)
        file_id = os.path.splitext(basename)[0]
        output_file = os.path.join(target_dir, f"{file_id}.vectornet.tfrecord")
        
        # Check if already exists
        if os.path.exists(output_file):
            print(f"  Skipping {output_file} (already exists)")
            continue
        
        # Process the file
        try:
            processed_data = process_tfrecord_file(source_file, history_len, future_len)
            
            if processed_data:
                # Apply max_scenarios limit
                if max_scenarios and total_scenarios + len(processed_data) > max_scenarios:
                    remaining = max_scenarios - total_scenarios
                    processed_data = processed_data[:remaining]
                
                # Write to output
                write_vectornet_tfrecord(processed_data, output_file)
                total_scenarios += len(processed_data)
                
                if max_scenarios and total_scenarios >= max_scenarios:
                    print(f"  Reached max_scenarios limit ({max_scenarios})")
                    break
        except Exception as e:
            print(f"  ERROR processing {source_file}: {e}")
            continue
    
    print(f"  Processed {total_scenarios} scenarios")
    return total_scenarios


def verify_setup(target_base=None):
    """Verify that VectorNet TFRecord files were created."""
    if target_base is None:
        target_base = os.path.join(os.path.dirname(__file__), 'data')
    print("\n" + "="*60)
    print("Verifying VectorNet data...")
    print("="*60)
    
    total_files = 0
    for split in ['training', 'validation', 'testing']:
        target_dir = os.path.join(target_base, split)
        files = sorted(glob.glob(os.path.join(target_dir, '*.vectornet.tfrecord')))
        print(f"{split:12s}: {len(files):4d} .vectornet.tfrecord files")
        total_files += len(files)
    
    print(f"\nTotal: {total_files} VectorNet TFRecord files")
    
    if total_files == 0:
        print("\n⚠️  WARNING: No VectorNet TFRecord files found!")
        return False
    else:
        print("\n✅ VectorNet data preparation complete!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert Waymo TFRecord files to VectorNet format'
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Source directory containing original TFRecord files (e.g., data/scenario)'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default=None,
        help='Target directory for VectorNet files (default: src/vectornet/data)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['training', 'validation', 'testing'],
        help='Process only this split (default: all splits)'
    )
    parser.add_argument(
        '--history_len',
        type=int,
        default=10,
        help='Number of history timesteps (default: 10)'
    )
    parser.add_argument(
        '--future_len',
        type=int,
        default=50,
        help='Number of future timesteps to predict (default: 50)'
    )
    parser.add_argument(
        '--max_scenarios',
        type=int,
        default=None,
        help='Maximum scenarios per split (default: all)'
    )
    
    args = parser.parse_args()
    
    # Set default target directory
    if args.target_dir is None:
        args.target_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    print("="*60)
    print("VectorNet Data Preparation")
    print("="*60)
    print(f"Source: {args.source_dir}")
    print(f"Target: {args.target_dir}")
    print(f"History length: {args.history_len}")
    print(f"Future length: {args.future_len}")
    if args.max_scenarios:
        print(f"Max scenarios per split: {args.max_scenarios}")
    print("="*60)
    
    # Find source files
    print("\nSearching for source TFRecord files...")
    splits = find_tfrecord_files(args.source_dir)
    
    if not splits:
        print(f"\n❌ ERROR: No TFRecord files found in {args.source_dir}")
        print("\nPlease check:")
        print("1. The path is correct")
        print("2. TFRecord files exist in training/validation/testing subdirectories")
        print("3. Files have .tfrecord extension (not .vectornet.tfrecord)")
        return 1
    
    # Setup target directories
    print("\nSetting up target directories...")
    target_base = setup_target_directories(args.target_dir)
    
    # Process each split
    if args.split:
        # Process only specified split
        if args.split in splits:
            target_dir = os.path.join(target_base, args.split)
            process_split(
                splits[args.split],
                target_dir,
                args.split,
                args.history_len,
                args.future_len,
                args.max_scenarios
            )
        else:
            print(f"\n❌ ERROR: Split '{args.split}' not found in source directory")
            return 1
    else:
        # Process all splits
        for split_name, files in splits.items():
            target_dir = os.path.join(target_base, split_name)
            process_split(
                files,
                target_dir,
                split_name,
                args.history_len,
                args.future_len,
                args.max_scenarios
            )
    
    # Verify
    verify_setup(target_base)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
