import torch
import sys
import os
import h5py
import time
import math
import numpy as np
import tensorflow as tf
from config import batch_size, gcn_num_workers, sequence_length, radius, graph_creation_method, MAX_SPEED, MAX_ACCEL, MAX_DIST_SDC, MAX_DIST_NEAREST 
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

def initialize():
    print("Initializing...")
    print(f" - Python version: {sys.version}")
    print(f" - Python executable: {sys.executable}")

    src_path = os.path.join(os.getcwd(), 'src')
    src_path_abs = os.path.abspath(src_path)

    waymo_module_path = None
    local_src = os.path.abspath(os.path.join(os.getcwd(), 'src', 'waymo_open_dataset'))
    if os.path.exists(local_src):
        waymo_module_path = local_src
        src_dir = os.path.abspath(os.path.join(os.getcwd(), 'src'))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)   # ensure imports from src work
        print(f" - FOUND waymo_open_dataset AT local src: {waymo_module_path}")
    else:
        # 2) search existing sys.path entries (original behavior)
        for path in sys.path:
            potential_path = os.path.join(path, 'waymo_open_dataset')
            if os.path.exists(potential_path):
                waymo_module_path = potential_path
                print(f"FOUND waymo_open_dataset IN sys.path AT: {waymo_module_path}.")
                break
    if not waymo_module_path:
        print(" - waymo_open_dataset DIRECTORY NOT FOUND IN ANY sys.path LOCATION.")


    try:    # Check protobuf version
        import google.protobuf
        protobuf_version = google.protobuf.__version__
        print(f" - Protobuf version: {protobuf_version}")
        
        if protobuf_version.startswith('3.20'):
            print(" - CORRECT protobuf VERSION!")
        else:
            print(f" - Wrong protobuf version ({protobuf_version}), need 3.20.3")
            print("Run this in terminal: conda activate waymo; pip install protobuf==3.20.3 --force-reinstall")
    except Exception as e:
        print(f" - Protobuf ERROR: {e}")


    try:    # Test basic imports
        print(f" - NumPy version: {np.__version__}")
        
        import tensorflow as tf
        print(f" - TensorFlow version: {tf.__version__}")
        print("BASIC IMPORTS SUCCESSFUL.")
    except Exception as e:
        print(f" - Basic imports FAILED: {e}")


    try:    # Import Waymo modules
        from waymo_open_dataset import dataset_pb2
        print(" - dataset_pb2 imported")
        
        from waymo_open_dataset.protos import scenario_pb2
        print(" - scenario_pb2 imported")
        
        from waymo_open_dataset.utils import womd_camera_utils
        print(" - womd_camera_utils imported")
        
        # Import additional utility modules for data processing
        from waymo_open_dataset.utils import range_image_utils
        from waymo_open_dataset.utils import transform_utils
        from waymo_open_dataset.utils import frame_utils
        print(" - Additional utils imported")
        
        print(" - ALL WAYMO IMPORTS SUCCESSFUL!")
    except ImportError as e:
        print(f"\tWaymo import FAILED: {e}")
        print("\nSOLUTIONS:")
        print("1. Make sure you're in the correct directory (waymo-open-dataset/tutorial)")
        print("In /src/waymo_open_dataset you should have libraries (like math, metrics, protos, utils), __pycache__, bazel etc.")
        print("IF NOT: clone the repo: https://github.com/waymo-research/waymo-open-dataset.git")
        print("2. Compile proto files first:")
        print("   - Change to src directory: os.chdir('../src')")
        print("   - Run: subprocess.run(['python', '-m', 'grpc_tools.protoc', '--python_out=.', '--proto_path=.'] + glob.glob('waymo_open_dataset/**/*.proto', recursive=True))")
        print("3. Or run this compilation now:")
        try:    # Try to compile proto files automatically
            import subprocess
            import glob
            current_dir = os.getcwd() # Change to src directory
            src_dir = None
            for potential_src in [os.path.join(current_dir, '..', 'src'),   # Find src directory
                                os.path.join(current_dir, 'src'),
                                r'c:\Users\radov\dev\waymo-open-dataset\src']:
                if os.path.exists(potential_src):
                    src_dir = potential_src
                    break
            if src_dir:
                print(f"   Found src directory: {src_dir}")
                os.chdir(src_dir)
                # Get proto files
                proto_files = glob.glob('waymo_open_dataset/**/*.proto', recursive=True)
                print(f"   Found {len(proto_files)} proto files")
                if proto_files:
                    # Compile proto files
                    cmd = ['python', '-m', 'grpc_tools.protoc', '--python_out=.', '--proto_path=.'] + proto_files
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("   Proto FILES COMPILED SUCCESSFULLY!")
                        os.chdir(current_dir)  # Return to original directory
                        print("\tPlease restart the kernel and run this cell again")
                    else:
                        print(f"\tProto compilation FAILED: {result.stderr}")
                        os.chdir(current_dir)
                else:
                    print("\tNo proto files found")
                    os.chdir(current_dir)
            else:
                print("\tCould not find src directory")
        except Exception as compile_error:
            print(f"\tAuto-compilation FAILED: {compile_error}")
    except Exception as e:
        print(f"\tUnexpected ERROR: {e}")

    print(f"...INITIALIZATION COMPLETE!\n")

def get_data_files(filepath):
    print("Files to get", end=": ")
    files = []
    try:
        for filename in os.listdir(filepath):
            if filename != '.gitkeep':
                print(filename, end=", ")
                full_path = os.path.join(filepath, filename)  # Use a different variable name!
                if os.path.isfile(full_path):
                    files.append(full_path)
                else:
                    print(f"{full_path} is not file")
        print(files, "\n")
        return files
    except Exception as e:
        print(e)

def parse_scenario_file(file):
    from waymo_open_dataset.protos import scenario_pb2
    
    # Check if file exists and has content
    if not os.path.exists(file):
        print(f"  ERROR: File does not exist: {file}")
        return []
    
    file_size = os.path.getsize(file)
    print(f"  File size: {file_size} bytes")
    
    if file_size == 0:
        print(f"  ERROR: File is empty: {file}")
        return []
    
    scenario_dataset = tf.data.TFRecordDataset(file, compression_type='')
    scenarios = []
    record_count = 0

    try:
        for idx, raw_record in enumerate(scenario_dataset):
            record_count += 1
            try:
                scenario = scenario_pb2.Scenario.FromString(raw_record.numpy())
                scenarios.append(scenario)
            except Exception as e:
                print(f"  ERROR parsing scenario {idx} in {file}: {e}")
                continue  # Continue instead of break to process remaining scenarios
    except Exception as e:
        print(f"  ERROR iterating TFRecordDataset for {file}: {e}")
    
    print(f"  Found {record_count} records, successfully parsed {len(scenarios)} scenarios")
    
    if len(scenarios) == 0 and record_count == 0:
        print(f"  WARNING: No records found in {file} - file may be corrupted or wrong format")
    
    return scenarios

def create_scenario_dataset_dict(files, prinT=False):
    """returns dict of scenarios (fileID: [scenarios...]), extracted from given file"""
    dataset_dict = {}
    try:
        for file in enumerate(files):
            file_index = file[0]
            file_path = file[1]

            dataset_dict[file_index] = parse_scenario_file(file_path)
            if prinT:
                print(f"Added {len(dataset_dict[file_index])} scenarios to dataset_dict of file with ID = {file_index} (from {file_path}).")
        return dataset_dict
    except Exception as e:
        print(f"Error processing files: {e}")

def build_edge_index_using_radius(position_tensor, radius, self_loops=False, valid_mask=None, min_distance=0.0):
    """Optimized edge index computation using radius neighborhood.
    
    Converts position tensor to edge tensor (edge_index) of shape (2, E) for graph using radius.
    Also computes edge weights based on distance: weight = exp(-distance^2 / (2 * radius^2)).
    
    Optimizations:
    - Pre-allocated tensors where possible
    - Vectorized mask operations
    - Avoids unnecessary tensor copies
    """
    # Use float32 for distance computation (sufficient precision, faster)
    pos = position_tensor.float()
    
    # Compute pairwise squared distances (avoid sqrt until needed)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    pos_norm_sq = (pos ** 2).sum(dim=1, keepdim=True)
    pairwise_sq_distances = pos_norm_sq + pos_norm_sq.t() - 2 * torch.mm(pos, pos.t())
    pairwise_sq_distances = pairwise_sq_distances.clamp(min=0)  # Numerical stability
    
    # Convert to actual distances for edge weight computation later
    pairwise_norm2_distances = torch.sqrt(pairwise_sq_distances + 1e-8)
    
    # Apply valid mask if provided
    if valid_mask is not None:
        vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=pairwise_norm2_distances.device)
        if vm.numel() != pairwise_norm2_distances.size(0):
            raise ValueError("valid_mask length must match number of positions")
        # Create invalid mask in-place
        invalid_mask = ~(vm[:, None] & vm[None, :])
        pairwise_norm2_distances.masked_fill_(invalid_mask, float('inf'))
    
    if not self_loops:
        pairwise_norm2_distances.fill_diagonal_(float('inf'))
    
    # Combined mask for radius and min_distance
    edges_mask = (pairwise_norm2_distances <= float(radius)) & (pairwise_norm2_distances > float(min_distance))
    
    # Get edge indices
    edge_indices = edges_mask.nonzero(as_tuple=True)
    edge_index = torch.stack([edge_indices[0], edge_indices[1]], dim=0)
    
    # Compute edge weights efficiently
    edge_distances = pairwise_norm2_distances[edges_mask]
    sigma_sq_2 = 2 * radius * radius
    edge_weight = torch.exp(-edge_distances.pow(2) / sigma_sq_2)
    
    # Normalize edge weights
    if edge_weight.numel() > 0:
        edge_weight = edge_weight / edge_weight.max()
    
    return edge_index, edge_weight

def build_edge_index_using_star_graph(position_tensor, scenario, agent_ids=None, valid_mask=None, min_distance=0.0):
    """Create star graph topology with SDC as center node. All other agents connect to SDC.
    Also computes edge weights based on distance from SDC."""
    
    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_id = sdc_track.id
    
    num_nodes = position_tensor.shape[0]
    
    if agent_ids is not None:
        try:
            sdc_idx = agent_ids.index(sdc_id)
        except ValueError:
            print(f"Warning: SDC (id={sdc_id}) not found in agent_ids, using index 0")
            sdc_idx = 0
    else:
        # use scenario.sdc_track_index (may be incorrect if agents are filtered)
        sdc_idx = scenario.sdc_track_index
        if sdc_idx >= num_nodes:
            print(f"Warning: SDC index {sdc_idx} >= num_nodes {num_nodes}, using index 0")
            sdc_idx = 0
    
    # Create valid mask tensor if provided
    if valid_mask is not None:
        vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=position_tensor.device)
        if vm.numel() != num_nodes:
            raise ValueError("valid_mask length must match number of positions")
    else:
        vm = torch.ones(num_nodes, dtype=torch.bool, device=position_tensor.device)
    
    # Build star graph edges (bidirectional) and compute weights
    edge_list = []
    edge_distances = []
    for i in range(num_nodes):
        if i == sdc_idx:
            continue  # Skip self-loop to SDC
        
        if not vm[i]:  # Skip invalid agents
            continue
        
        # Calculate distance from SDC to agent
        dist = torch.norm(position_tensor[sdc_idx] - position_tensor[i])
        
        # Check minimum distance if specified
        if min_distance > 0.0:
            if dist <= min_distance:
                continue
        
        # Add bidirectional edges:
        edge_list.append([sdc_idx, i])
        edge_list.append([i, sdc_idx])
        edge_distances.append(dist.item())
        edge_distances.append(dist.item())
    
    if len(edge_list) == 0:
        return (torch.zeros((2, 0), dtype=torch.long, device=position_tensor.device),    # Return empty edge_index and edge_weight with correct shape
                torch.zeros(0, dtype=torch.float32, device=position_tensor.device))
    
    # Convert to edge_index format [2, num_edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=position_tensor.device).t()
    
    edge_distances_tensor = torch.tensor(edge_distances, dtype=torch.float32, device=position_tensor.device)    # compute edge weights: exp(-distance^2 / (2*sigma^2)) for reasonable scale
    sigma = 30.0    # Similar to radius
    edge_weight = torch.exp(-edge_distances_tensor**2 / (2 * sigma**2))
    
    return edge_index, edge_weight

def initial_feature_vector(agent, timestep, scenario=None, all_positions=None):
    """
    Creates initial feature vector for given agent at given timestep.
    Returns tensor of shape [15] with features:
        [0-1]   vx_norm, vy_norm (normalized velocity)
        [2]     speed_norm (speed magnitude)
        [3]     heading_direc (heading direction, atan2)
        [4]     valid (validity flag)
        [5-6]   ax_norm, ay_norm (acceleration - velocity change from previous timestep)
        [7-8]   rel_x_to_sdc_norm, rel_y_to_sdc_norm (normalized relative position to ego vehicle)
        [9]     dist_to_sdc_norm (normalized distance to ego vehicle)
        [10]    dist_to_nearest_norm (normalized distance to nearest neighbor)
        [11-14] one-hot type (vehicle, pedestrian, cyclist, other)
    """
    # Bounds check for testing data (only 11 timesteps)
    if timestep >= len(agent.states):
        # Return zero vector if timestep out of bounds
        return torch.zeros(15, dtype=torch.float32)
    
    state = agent.states[timestep]
    
    # Basic velocity features (normalized)
    vx = state.velocity_x
    vy = state.velocity_y
    speed = (vx**2 + vy**2)**0.5
    
    vx_norm = vx / MAX_SPEED
    vy_norm = vy / MAX_SPEED
    speed_norm = speed / MAX_SPEED
    heading_direc = math.atan2(vy, vx) / math.pi  # normalize to [-1, 1]
    
    valid = 1.0 if state.valid else 0.0
    
    # Acceleration features (velocity change from previous timestep)
    ax_norm, ay_norm = 0.0, 0.0
    if timestep > 0:
        prev_state = agent.states[timestep - 1]
        if prev_state.valid and state.valid:
            ax = state.velocity_x - prev_state.velocity_x
            ay = state.velocity_y - prev_state.velocity_y
            ax_norm = ax / MAX_ACCEL
            ay_norm = ay / MAX_ACCEL
    
    # Relative position to SDC (ego vehicle) - normalized
    # NOTE: Waymo dataset uses CENTIMETERS, so divide by 100.0 to convert to meters first
    rel_x_to_sdc_norm, rel_y_to_sdc_norm, dist_to_sdc_norm = 0.0, 0.0, 0.0
    if scenario is not None:
        sdc_track_index = scenario.sdc_track_index
        if sdc_track_index < len(scenario.tracks):
            sdc_track = scenario.tracks[sdc_track_index]
            sdc_state = sdc_track.states[timestep]
            if sdc_state.valid and state.valid:
                # Convert from centimeters to meters
                rel_x = (state.center_x - sdc_state.center_x) / 100.0
                rel_y = (state.center_y - sdc_state.center_y) / 100.0
                dist_to_sdc = (rel_x**2 + rel_y**2)**0.5
                
                rel_x_to_sdc_norm = rel_x / MAX_DIST_SDC
                rel_y_to_sdc_norm = rel_y / MAX_DIST_SDC
                dist_to_sdc_norm = min(dist_to_sdc / MAX_DIST_SDC, 1.0)  # clip to [0, 1]
    
    # Distance to nearest neighbor - normalized
    # NOTE: all_positions are in CENTIMETERS, convert to meters
    dist_to_nearest_norm = 1.0  # default: far away (normalized max)
    if all_positions is not None and state.valid and len(all_positions) > 1:
        current_pos = (state.center_x, state.center_y)
        min_dist = float('inf')
        for pos in all_positions:
            if pos != current_pos:
                # Distance in centimeters, convert to meters
                d = math.sqrt((pos[0] - current_pos[0])**2 + (pos[1] - current_pos[1])**2) / 100.0
                min_dist = min(min_dist, d)
        if min_dist != float('inf'):
            dist_to_nearest_norm = min(min_dist / MAX_DIST_NEAREST, 1.0)  # clip to [0, 1]
    
    # One-hot object type
    obj_type = agent.object_type
    type_vehicle = 1.0 if obj_type == 1 else 0.0
    type_pedestrian = 1.0 if obj_type == 2 else 0.0
    type_cyclist = 1.0 if obj_type == 3 else 0.0
    type_other = 1.0 if obj_type not in [1, 2, 3] else 0.0
    
    return torch.tensor([
        vx_norm, vy_norm,                           # [0-1] normalized velocity
        speed_norm,                                  # [2] normalized speed
        heading_direc,                               # [3] heading in [-1, 1]
        valid,                                       # [4] validity
        ax_norm, ay_norm,                            # [5-6] normalized acceleration
        rel_x_to_sdc_norm, rel_y_to_sdc_norm,       # [7-8] normalized relative position to SDC
        dist_to_sdc_norm,                            # [9] normalized distance to SDC
        dist_to_nearest_norm,                        # [10] normalized distance to nearest
        type_vehicle, type_pedestrian,               # [11-12] one-hot type
        type_cyclist, type_other                     # [13-14] one-hot type
    ], dtype=torch.float32)

def get_data_from_agents(agents, node_features, positions_2D, agent_ids, valid_mask, timestep, use_valid_only, scenario=None):
    """returns lists node_features [N, d], positions_2D [N, 2] and agent_ids and valid_mask"""
    # First pass: collect all positions for nearest neighbor calculation
    all_positions = []
    valid_agents = []
    for agent in agents:
        if timestep >= len(agent.states):
            state = agent.states[-1]
        else:
            state = agent.states[timestep]

        valid = state.valid
        if use_valid_only and not valid:
            continue
        
        all_positions.append((state.center_x, state.center_y))
        valid_agents.append((agent, state, valid))
    
    # Second pass: build features with all_positions available
    for agent, state, valid in valid_agents:
        node_features.append(initial_feature_vector(agent, timestep, scenario=scenario, all_positions=all_positions))
        positions_2D.append([state.center_x, state.center_y])
        agent_ids.append(agent.id)
        valid_mask.append(1 if valid else 0)

def get_future_2D_trajectory_labels(scenario, agent_ids, timestep, normalize=True):
    """outputs tensor of shape [N, 2], for each agent we have position displacements stored as target labels"""
    id_to_agent_track = {track.id: track for track in scenario.tracks}
    future_2d_position_displacements = []

    # get node positions for next timestep
    # calculate displacements (current graph node position - next graph node position)
    # put displacements into list which is used as y of the graph
    
    for agent_id in agent_ids:
        agent = id_to_agent_track.get(agent_id)   # agent data has its position at current timestep
        if agent is None:
            future_2d_position_displacements.append([0, 0])
            continue
        
        try:
            current_timestep_positions = agent.states[timestep]
            next_timestep_positions = agent.states[timestep+1]

            if next_timestep_positions.valid:
                dx = (next_timestep_positions.center_x - current_timestep_positions.center_x) / 100.0
                dy = (next_timestep_positions.center_y - current_timestep_positions.center_y) / 100.0
                
                displacement = [dx, dy]
                future_2d_position_displacements.append(displacement)
            else:
                future_2d_position_displacements.append([0, 0])     # use current position (agent did not move)
        except (IndexError, KeyError) as e:
            print(f"ERROR AT TIMESTEPS IN get_future_2D_trajectory_labels FUNCTION!\nTimestep {timestep} or {timestep+1} out of bounds for agent {agent_id}: {e}")
            print(f"\tAgent only has: {len(agent.states)} states!")
            future_2d_position_displacements.append([0, 0])
    return torch.tensor(future_2d_position_displacements, dtype=torch.float32)

def timestep_to_pyg_data(scenario, timestep, radius, use_valid_only=True, method='radius'):
    """Converts timestep snapshot into PyG data - graph, which has these properties:
    - tensor x [N, d] of node features
    - tensor edge_index [2, E] of edges
    - tensor pos [N, 2] of node/agent positions
    - tensor y [N, 2] of position displacements in next graph in the sequence
    - tensor agent_ids [N]
    - tensor valid_mask [N]"""
    x = []
    positions_2D = []
    agent_ids = []
    valid_mask = []
    get_data_from_agents(scenario.tracks, x, positions_2D, agent_ids, valid_mask, timestep, use_valid_only, scenario=scenario)

    if len(x) == 0:
        return None
    
    positions_2D_tensor = torch.tensor(positions_2D, dtype=torch.float32)

    if method == 'radius':
        edge_index, edge_weight = build_edge_index_using_radius(positions_2D_tensor, radius, valid_mask=valid_mask, min_distance=0.1)
    elif method == 'star':
        edge_index, edge_weight = build_edge_index_using_star_graph(positions_2D_tensor, scenario, agent_ids=agent_ids, valid_mask=valid_mask)

    y = get_future_2D_trajectory_labels(scenario, agent_ids, timestep)

    data = Data(
        x=torch.stack(x) if isinstance(x[0], torch.Tensor) else torch.tensor(x, dtype=torch.float32), 
        edge_index=edge_index,
        edge_attr=edge_weight,  # Add edge weights based on distance
        pos=positions_2D_tensor, 
        y=y,
        scenario_id=scenario.scenario_id,
        agent_ids=agent_ids,
        valid_mask=torch.tensor(valid_mask, dtype=torch.bool)
    )
    
    return data

def save_scenarios_to_hdf5_streaming(files, h5_path, compression="lzf", max_num_scenarios_per_tfrecord_file=None):
    """Stream directly from TFRecord files to HDF5 without storing in memory, HDF5 file structure:
    scenarios/{scenario_id}/snapshot_graphs/{timestep}/x|edge_index|edge_weight|y"""
    print("Saving scenarios to HDF5 file:")
    with h5py.File(h5_path, "w") as f:
        scenarios_group = f.create_group("scenarios")

        total_scenarios = 0
        start_time = time.perf_counter()
        for file_idx, file_path in enumerate(files):
            print(f"\tProcessing file {file_idx + 1}/{len(files)}: {file_path}...")
            start = time.perf_counter()

            scenarios = parse_scenario_file(file_path)      # Parse scenarios from this file only

            if max_num_scenarios_per_tfrecord_file is not None:
                scenarios = scenarios[:max_num_scenarios_per_tfrecord_file]

            for scenario in scenarios:
                scenario_group = scenarios_group.create_group(str(scenario.scenario_id))
                snapshots_group = scenario_group.create_group("snapshot_graphs")
                
                # Detect actual sequence length from this scenario (testing=11, training=91)
                actual_timesteps = len(scenario.tracks[0].states) if len(scenario.tracks) > 0 else 0
                max_timestep = min(actual_timesteps - 1, sequence_length - 1)  # Use min of actual and config
                
                # Create and save graphs for this scenario
                for timestep in range(max_timestep):
                    graph = timestep_to_pyg_data(scenario, timestep, radius, method=graph_creation_method)
                    snapshot_group = snapshots_group.create_group(str(timestep))
                    
                    # Extract data
                    if hasattr(graph, "x"):
                        x = graph.x.cpu().numpy()
                        edge_index = graph.edge_index.cpu().numpy()
                        edge_weight = getattr(graph, "edge_weight", None)
                        y = getattr(graph, "y", None)
                        pos = getattr(graph, "pos", None)
                        scenario_id = graph.scenario_id
                    else:
                        x = graph["x"].cpu().numpy()
                        edge_index = graph["edge_index"].cpu().numpy()
                        edge_weight = graph.get("edge_weight")
                        y = graph.get("y")
                        pos = graph.get("pos")
                        scenario_id = graph.get("scenario_id")
                    
                    # Save datasets
                    snapshot_group.create_dataset("x", data=x, compression=compression, chunks=True)
                    snapshot_group.create_dataset("edge_index", data=edge_index, compression=compression, chunks=True)
                    
                    if edge_weight is not None:
                        snapshot_group.create_dataset("edge_weight", data=edge_weight.cpu().numpy(), 
                                                     compression=compression, chunks=True)
                    if y is not None:
                        snapshot_group.create_dataset("y", data=y.cpu().numpy(), 
                                                     compression=compression, chunks=True)
                    
                    # Save positions (normalized, will be denormalized in visualization)
                    if pos is not None:
                        snapshot_group.create_dataset("pos", data=pos.cpu().numpy(),
                                                     compression=compression, chunks=True)
                    else:
                        if timestep == 0 and total_scenarios == 0:
                            print("  Warning: pos is None for first graph - this will cause visualization issues!")
                    
                    # Save agent_ids as numpy array of integers
                    agent_ids = getattr(graph, "agent_ids", None) if hasattr(graph, "agent_ids") else graph.get("agent_ids")
                    if agent_ids is not None:
                        snapshot_group.create_dataset("agent_ids", data=np.array(agent_ids, dtype=np.int64))
                    
                    # Save scenario_id as string (no compression for scalar strings)
                    if scenario_id is not None:
                        snapshot_group.create_dataset("scenario_id", data=np.bytes_(scenario_id))
                
                total_scenarios += 1
            end = time.perf_counter()
            elapsed = end - start
            
            print(f"\tProcessed {len(scenarios)} scenarios from this file in {elapsed:.1f} seconds. (total: {total_scenarios})")
            del scenarios   #  Free memory after processing each file
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
            
    print(f"Completed in {elapsed_time:.1f} seconds! Total scenarios saved into HDF5 file ({h5_path}): {total_scenarios}.\n")

def test_hdf5_and_lazy_loading(path):
    print("=" * 80)
    print("TESTING HDF5 LAZY LOADING")
    print("=" * 80)

    # Initialize dataset
    from dataset import HDF5ScenarioDataset
    dataset = HDF5ScenarioDataset(path, seq_len=sequence_length)
    print(f"\n✓ Dataset loaded: {len(dataset)} total scenarios")

    # Test 1: Check a few individual scenario sequences
    print("\n" + "-" * 80)
    print("TEST 1: Individual Scenario Sequence Access (should be lazy)")
    print("-" * 80)
    for i in range(min(3, len(dataset))):
        data_list = dataset[i]
        print(f"Scenario {i}: {len(data_list)} timesteps")
        first_graph = data_list[0]
        print(f"  - scenario_id: {first_graph.scenario_id}")
        print(f"  - First timestep: nodes={first_graph.x.shape[0]}, edges={first_graph.edge_index.shape[1]}")
        print(f"  - x shape: {first_graph.x.shape}, dtype: {first_graph.x.dtype}")
        print(f"  - edge_index shape: {first_graph.edge_index.shape}")
        if first_graph.edge_attr is not None:
            print(f"  - edge_weight shape: {first_graph.edge_attr.shape}")
        if first_graph.y is not None:
            print(f"  - y shape: {first_graph.y.shape}")
        if first_graph.pos is not None:
            print(f"  - pos shape: {first_graph.pos.shape}")
        if hasattr(first_graph, 'agent_ids') and first_graph.agent_ids is not None:
            print(f"  - agent_ids: {len(first_graph.agent_ids)} agents")

    # Test 2: DataLoader with Sequence Batching (TRAINING-STYLE)
    print("\n" + "-" * 80)
    print("TEST 2: DataLoader with Sequence Batching (TRAINING-STYLE)")
    print("-" * 80)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=gcn_num_workers,
        collate_fn=collate_graph_sequences_to_batch,
        drop_last=True
    )
    print(f"DataLoader config: batch_size={batch_size}, num_workers={gcn_num_workers}, shuffle=False")
    print(f"Expected batches: ~{len(dataset) // batch_size}")
    print(f"Each batch contains {batch_size} different scenarios, each with {sequence_length} timesteps\n")

    for batch_idx, batch_dict in enumerate(dataloader):
        if batch_idx >= 3:  # Only show first 3 batches
            break
        
        batched_graph_sequence = batch_dict["batch"]
        B = batch_dict["B"]
        T = batch_dict["T"]
        scenario_ids = batch_dict.get("scenario_ids", [])
        
        print(f"Batch {batch_idx}:")
        print(f"  - B (# scenarios): {B}")
        print(f"  - T (# timesteps): {T}")
        print(f"  - Batched graphs: {len(batched_graph_sequence)}")
        print(f"  - Scenario IDs: {scenario_ids}")
        
        # Show first timestep details
        first_timestep = batched_graph_sequence[0]
        print(f"  - First timestep graph:")
        print(f"    - x: {first_timestep.x.shape}")
        print(f"    - edge_index: {first_timestep.edge_index.shape}")
        print(f"    - y: {first_timestep.y.shape if first_timestep.y is not None else None}")
        print(f"    - pos: {first_timestep.pos.shape if hasattr(first_timestep, 'pos') and first_timestep.pos is not None else None}")
        print(f"    - batch: {first_timestep.batch.shape} (node→graph mapping)")
        if hasattr(first_timestep, 'agent_ids') and first_timestep.agent_ids:
            print(f"    - agent_ids: {len(first_timestep.agent_ids)} total agents")
        print()

    print(f"Total batches processed: {batch_idx + 1}")
    print("\n✓ This matches training: Each batch has B={batch_size} different scenarios,")
    print(f"  and you iterate through T={sequence_length} timesteps sequentially in training loop")

    # Test 3: Verify Lazy Loading (memory check)
    print("\n" + "-" * 80)
    print("TEST 3: Verify Lazy Loading (memory check)")
    print("-" * 80)
    print("Accessing 5 random scenarios without loading all data...")
    import random
    indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    for idx in indices:
        data_list = dataset[idx]
        first_graph = data_list[0]
        print(f"  Index {idx}: scenario={first_graph.scenario_id}, timesteps={len(data_list)}, nodes={first_graph.x.shape[0]}")
    print("✓ If this was fast, lazy loading is working correctly!")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

def collate_graph_sequences_to_batch(scenario_list):
    """turns scenario list like [[scenario 1 T graphs], [scenario 2 T graphs], ...] into dict containing list 
    [[timestep 1 - B batched graphs], [timestep 2 - B batched graphs], ...] of T Batch objects and batch size (number of graph sequences in the batch):
    {
        'batch': [...],
        'B': num_of_sequences,
        'T': sequence_length
    }"""
    B = len(scenario_list)    # number of sequences
    T = len(scenario_list[0])   # sequence length
    transposed = list(zip(*scenario_list))  # transposes list-of-lists structure

    # extract scenario_ids from the first timestep (all timesteps have the same scenario_id per sequence):
    scenario_ids = []
    for sequence in scenario_list:
        if sequence and len(sequence) > 0 and hasattr(sequence[0], 'scenario_id'):
            sid = sequence[0].scenario_id
            scenario_ids.append(sid)
        else:
            scenario_ids.append(None)

    batch = []
    for timestep in range(T):
        graphs_at_timestep = list(transposed[timestep])
        
        batched_graphs_at_timestep = Batch.from_data_list(graphs_at_timestep)
        
        # manually preserve agent_ids (Batch.from_data_list doesn't preserve list attributes)
        all_agent_ids = []
        for graph in graphs_at_timestep:
            if hasattr(graph, 'agent_ids'):
                all_agent_ids.extend(graph.agent_ids)
        
        if all_agent_ids:
            batched_graphs_at_timestep.agent_ids = all_agent_ids
        
        batch.append(batched_graphs_at_timestep)
    
    return {'batch': batch, 'B': B, 'T': T, 'scenario_ids': scenario_ids}
