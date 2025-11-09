import torch
import sys
import os
import h5py
import time
import numpy as np
import tensorflow as tf
from config import batch_size, num_workers, sequence_length, radius, graph_creation_method
from torch_geometric.data import Data, Batch
from dataset import HDF5TemporalDataset
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
    scenario_dataset = tf.data.TFRecordDataset(file, compression_type='')
    scenarios = []
    from waymo_open_dataset.protos import scenario_pb2

    for raw_record in scenario_dataset:
        try:
            scenario = scenario_pb2.Scenario.FromString(raw_record.numpy())
            scenarios.append(scenario)
        except Exception as e:
            print(f"Error parsing scenario: {e}")
            break
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
    """Convert position tensor that holds (x,y) positions of agents into edge tensor (edge_index) OF SHAPE (2, E) for graph using radius."""
    pairwise_norm2_distances = torch.cdist(position_tensor, position_tensor)
    if valid_mask is not None:  # mask out pairs with invalid endpoints
        vm = torch.as_tensor(valid_mask, dtype=torch.bool, device=pairwise_norm2_distances.device)
        if vm.numel() != pairwise_norm2_distances.size(0):
            raise ValueError("valid_mask length must match number of positions")
        valid_pair = vm[:, None] & vm[None, :]   # True only if both endpoints valid
        pairwise_norm2_distances = pairwise_norm2_distances.clone()
        pairwise_norm2_distances[~valid_pair] = float('inf')
    if not self_loops:
        pairwise_norm2_distances.fill_diagonal_(float('inf'))
    edges_mask = (pairwise_norm2_distances <= float(radius)) & (pairwise_norm2_distances > float(min_distance))
    n1, n2 = torch.where(edges_mask)
    edge_index = torch.stack([n1, n2], dim=0)
    return edge_index

def build_edge_index_using_star_graph(position_tensor, scenario, agent_ids=None, valid_mask=None, min_distance=0.0):
    """Create star graph topology with SDC as center node. All other agents connect to SDC."""
    
    # Get SDC track using direct index access (O(1) instead of O(n) loop)
    sdc_track = scenario.tracks[scenario.sdc_track_index]
    sdc_id = sdc_track.id
    
    num_nodes = position_tensor.shape[0]
    
    # Find SDC's index in the position_tensor using agent_ids
    if agent_ids is not None:
        try:
            sdc_idx = agent_ids.index(sdc_id)
        except ValueError:
            print(f"Warning: SDC (id={sdc_id}) not found in agent_ids, using index 0")
            sdc_idx = 0
    else:
        # Fallback: use scenario.sdc_track_index (may be incorrect if agents are filtered)
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
    
    # Build star graph edges (bidirectional)
    edge_list = []
    for i in range(num_nodes):
        if i == sdc_idx:
            continue  # Skip self-loop to SDC
        
        if not vm[i]:  # Skip invalid agents
            continue
        
        # Check minimum distance if specified
        if min_distance > 0.0:
            dist = torch.norm(position_tensor[sdc_idx] - position_tensor[i])
            if dist <= min_distance:
                continue
        
        # Add bidirectional edges: SDC -> agent and agent -> SDC
        edge_list.append([sdc_idx, i])
        edge_list.append([i, sdc_idx])
    
    if len(edge_list) == 0:
        # Return empty edge_index with correct shape
        return torch.zeros((2, 0), dtype=torch.long, device=position_tensor.device)
    
    # Convert to edge_index format [2, num_edges]
    edge_index = torch.tensor(edge_list, dtype=torch.long, device=position_tensor.device).t()
    
    return edge_index

def initial_feature_vector(agent, timestep, normalize=True):
    """returns **list** of features for agent at given timestep
    
    Args:
        agent: Agent track from scenario
        timestep: Current timestep index
        normalize: If True, normalize position and velocity to reasonable scales
        
    Returns:
        List of 9 features: [x, y, vx, vy, valid, vehicle, pedestrian, cyclist, other]
    """
    # TODO: IMPLEMENT TECHNIQUES FROM FEATURE AUGMENTATION LECTURE (lec 7)
    # consider adding: Speed: sqrt(vx² + vy²) Heading: atan2(vy, vx) Acceleration: If you have velocity from previous timestep Normalized coordinates: Relative to ego vehicle or scene center Temporal encoding: Timestep information

    if timestep >= len(agent.states):
        timestep = len(agent.states) - 1

    state = agent.states[timestep]
    
    # Normalize positions and velocities to reasonable scales
    # Waymo positions are in meters (can be thousands), velocities in m/s
    if normalize:
        # Divide by typical scene size (100m) and velocity scale (10 m/s ~ 36 km/h)
        x_norm = state.center_x / 100.0
        y_norm = state.center_y / 100.0
        vx_norm = state.velocity_x / 10.0
        vy_norm = state.velocity_y / 10.0
    else:
        x_norm = state.center_x
        y_norm = state.center_y
        vx_norm = state.velocity_x
        vy_norm = state.velocity_y
    
    object_types = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Other'}
    properties = [x_norm, y_norm, vx_norm, vy_norm, float(state.valid)]
    type_onehot = [1 if object_types[agent.object_type] == 'Vehicle' else 0,
                   1 if object_types[agent.object_type] == 'Pedestrian' else 0,
                   1 if object_types[agent.object_type] == 'Cyclist' else 0,
                   1 if object_types[agent.object_type] == 'Other' else 0 ]
    return properties + type_onehot

def get_data_from_agents(agents, node_features, positions_2D, agent_ids, valid_mask, timestep, use_valid_only):
    """returns lists node_features [N, d], positions_2D [N, 2] and agent_ids and valid_mask"""
    for agent in agents:
        if timestep >= len(agent.states):
            state = agent.states[-1]
        else:
            state = agent.states[timestep]

        valid = state.valid
        if use_valid_only and not valid:
            continue

        node_features.append(initial_feature_vector(agent, timestep))
        positions_2D.append([state.center_x, state.center_y])
        agent_ids.append(agent.id)
        valid_mask.append(1 if valid else 0)

def get_future_2D_trajectory_labels(scenario, agent_ids, timestep, normalize=True):
    """outputs tensor of shape [N, 2], for each agent we have displacements stored
    
    Args:
        scenario: Waymo scenario
        agent_ids: List of agent IDs
        timestep: Current timestep
        normalize: If True, normalize displacements by dividing by 100.0 (typical scene scale)
        
    Returns:
        Tensor of shape [N, 2] with position displacements
    """
    id_to_agent = {t.id: t for t in scenario.tracks}
    future_2d_position_displacements = []

    # get node positions for next timestep
    # calculate displacements
    # put displacements into list which is used as y of the graph
    
    for agent_id in agent_ids:
        agent = id_to_agent.get(agent_id)   # agent data has its position at current timestep
        if agent is None:
            future_2d_position_displacements.append([0, 0])
            continue
        
        try:
            current_timestep_positions = agent.states[timestep]
            next_timestep_positions = agent.states[timestep+1]

            if next_timestep_positions.valid:
                dx = next_timestep_positions.center_x - current_timestep_positions.center_x
                dy = next_timestep_positions.center_y - current_timestep_positions.center_y
                
                # Normalize displacements to match input feature scale
                if normalize:
                    dx /= 100.0
                    dy /= 100.0
                
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
    get_data_from_agents(scenario.tracks, x, positions_2D, agent_ids, valid_mask, timestep, use_valid_only)

    if len(x) == 0:
        return None
    
    positions_2D_tensor = torch.tensor(positions_2D, dtype=torch.float32)

    if method == 'radius':
        edge_index = build_edge_index_using_radius(positions_2D_tensor, radius, valid_mask=valid_mask, min_distance=0.1)
    elif method == 'star':
        edge_index = build_edge_index_using_star_graph(positions_2D_tensor, scenario, agent_ids=agent_ids, valid_mask=valid_mask)

    y = get_future_2D_trajectory_labels(scenario, agent_ids, timestep)

    data = Data(x=torch.tensor(x, dtype=torch.float32), 
                edge_index=edge_index, 
                pos=positions_2D_tensor, 
                y=y)
    
    # Store agent_ids for visualization and debugging
    data.agent_ids = agent_ids
    data.valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
    
    return data

def get_graphs_for_scenarios(dataset_dict, radius, graph_creation_method, prinT=False):
    """returns dict like {scenarioID: [graphs...], ...}"""
    scenarios_and_their_graphs = {}     # {scenarioID: [graphs...]}
    for file in dataset_dict:
        for scenario in dataset_dict[file]:
            scenario_graphs = []
            for timestep in range(sequence_length-1):
                data = timestep_to_pyg_data(scenario, int(timestep), radius, method=graph_creation_method)
                scenario_graphs.append(data)
            scenarios_and_their_graphs[scenario.scenario_id] = scenario_graphs
    if prinT:
        #print(scenarios_and_their_graphs.values())
        print(f"All the scenario IDs: {scenarios_and_their_graphs.keys()}")
        print(f"Example of a graph from scenario with id {list(scenarios_and_their_graphs.keys())[0]}:\n\t{scenarios_and_their_graphs[list(scenarios_and_their_graphs.keys())[0]][0]}")
    return scenarios_and_their_graphs

from tqdm import tqdm
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
                
                # Create and save graphs for this scenario
                for timestep in range(sequence_length - 1):
                    graph = timestep_to_pyg_data(scenario, timestep, radius, method=graph_creation_method)
                    snapshot_group = snapshots_group.create_group(str(timestep))
                    
                    # Extract data
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
                    
                    # Save datasets
                    snapshot_group.create_dataset("x", data=x, compression=compression, chunks=True)
                    snapshot_group.create_dataset("edge_index", data=edge_index, compression=compression, chunks=True)
                    
                    if edge_weight is not None:
                        snapshot_group.create_dataset("edge_weight", data=edge_weight.cpu().numpy(), 
                                                     compression=compression, chunks=True)
                    if y is not None:
                        snapshot_group.create_dataset("y", data=y.cpu().numpy(), 
                                                     compression=compression, chunks=True)
                
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
    dataset = HDF5TemporalDataset(path)
    print(f"\n✓ Dataset loaded: {len(dataset)} total snapshots")

    # Test 1: Check a few individual snapshots
    print("\n" + "-" * 80)
    print("TEST 1: Individual Snapshot Access (should be lazy)")
    print("-" * 80)
    for i in range(min(3, len(dataset))):
        data = dataset[i]
        print(f"Snapshot {i}: scenario={data.scenario_id}, time={data.snapshot_id}, nodes={data.x.shape[0]}, edges={data.edge_index.shape[1]}")
        print(f"  - x shape: {data.x.shape}, dtype: {data.x.dtype}")
        print(f"  - edge_index shape: {data.edge_index.shape}")
        if data.edge_attr is not None:
            print(f"  - edge_weight shape: {data.edge_attr.shape}")
        if data.y is not None:
            print(f"  - y shape: {data.y.shape}")

    # Test 2: Iterate through all snapshots (no batching)
    print("\n" + "-" * 80)
    print("TEST 2: Full Dataset Iteration")
    print("-" * 80)
    scenario_counts = {}
    for data in dataset.snapshots():
        scenario_counts[data.scenario_id] = scenario_counts.get(data.scenario_id, 0) + 1

    print(f"Total scenarios: {len(scenario_counts)}")
    for sid, count in sorted(scenario_counts.items()):
        print(f"  - Scenario {sid}: {count} snapshots")

    # Test 3: DataLoader with standard batching (PyG style)
    print("\n" + "-" * 80)
    print("TEST 3: DataLoader with Sequence Batching (TRAINING-STYLE)")
    print("-" * 80)
    
    # Use the same dataset as training
    from dataset import HDF5ScenarioDataset
    scenario_dataset = HDF5ScenarioDataset(path, seq_len=sequence_length)
    
    dataloader = DataLoader(
        scenario_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_graph_sequences_to_batch,
        drop_last=True
    )
    print(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}, shuffle=False")
    print(f"Expected batches: ~{len(scenario_dataset) // batch_size}")
    print(f"Each batch contains {batch_size} different scenarios, each with {sequence_length} timesteps\n")

    for batch_idx, batch_dict in enumerate(dataloader):
        if batch_idx >= 3:  # Only show first 3 batches
            break
        
        batched_graph_sequence = batch_dict["batch"]
        B = batch_dict["B"]
        T = batch_dict["T"]
        
        print(f"Batch {batch_idx}:")
        print(f"  - B (# scenarios): {B}")
        print(f"  - T (# timesteps): {T}")
        print(f"  - Batched graphs: {len(batched_graph_sequence)}")
        
        # Show first timestep details
        first_timestep = batched_graph_sequence[0]
        print(f"  - First timestep graph:")
        print(f"    - x: {first_timestep.x.shape}")
        print(f"    - edge_index: {first_timestep.edge_index.shape}")
        print(f"    - y: {first_timestep.y.shape}")
        print(f"    - batch: {first_timestep.batch.shape} (node→graph mapping)")
        if hasattr(first_timestep, 'scenario_id'):
            unique_scenarios = len(set([first_timestep.scenario_id]))
            print(f"    - Unique scenarios in batch: {unique_scenarios}")
        print()

    print(f"Total batches processed: {batch_idx + 1}")
    print("\n✓ This matches training: Each batch has B={batch_size} different scenarios,")
    print(f"  and you iterate through T={sequence_length} timesteps sequentially in training loop")

    # Test 4: OLD-STYLE flat snapshot batching (for comparison)
    print("\n" + "-" * 80)
    print("TEST 4: DataLoader Flat Snapshot Batching (OLD STYLE - NOT USED IN TRAINING)")
    print("-" * 80)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=Batch.from_data_list  # PyG default batching
    )
    print(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}, shuffle=False")
    print(f"Expected batches: ~{len(dataset) // batch_size}")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < 3:  # Print first 3 batches
            print(f"\nBatch {batch_idx}:")
            print(f"  Type: {type(batch)}")
            # DataLoader returns a Batch object (batched PyG Data)
            print(f"  - x: {batch.x.shape}")
            print(f"  - edge_index: {batch.edge_index.shape}")
            print(f"  - edge_attr: {batch.edge_attr.shape if batch.edge_attr is not None else None}")
            print(f"  - y: {batch.y.shape if batch.y is not None else None}")
            print(f"  - batch size: {batch.num_graphs}")
            if hasattr(batch, 'scenario_id'):
                print(f"  - scenario_ids: {batch.scenario_id}")
            if hasattr(batch, 'snapshot_id'):
                print(f"  - snapshot_ids: {batch.snapshot_id}")

    print(f"\nTotal batches processed: {batch_idx + 1}")
    print("\n⚠ This batching treats snapshots independently (NOT used in training!)")

    # Test 5: OLD custom sequence batching test
    print("\n" + "-" * 80)
    print("TEST 5: DataLoader with Custom Collate (OLD - SKIPPED)")
    print("-" * 80)
    print("⚠ Skipping - not compatible with HDF5TemporalDataset")
    print("  (Use HDF5ScenarioDataset for sequence batching as shown in Test 3)")

    # Test 6: Verify Lazy Loading (memory check)
    print("\n" + "-" * 80)
    print("TEST 6: Verify Lazy Loading (memory check)")
    print("-" * 80)
    print("Accessing 5 random snapshots without loading all data...")
    import random
    indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    for idx in indices:
        data = dataset[idx]
        print(f"  Index {idx}: scenario={data.scenario_id}, time={data.snapshot_id}, nodes={data.x.shape[0]}")
    print("✓ If this was fast, lazy loading is working correctly!")

    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

def collate_graph_sequences_to_batch(scenario_list):
    """turns scenario list like [[scenario 1 T graphs], [scenario 2 T graphs], ...] into dict containing list 
    [[timestep 1 - B batched graphs], [timestep 2 - B batched graphs], ...] of T Batch objects and batch size (number of graph sequences in the batch):
    {
        'batched_ts': [...],
        'B': num_of_sequences,
        'T': sequence_length
    }"""
    B = len(scenario_list)    # number of sequences
    T = len(scenario_list[0])   # sequence length
    transposed = list(zip(*scenario_list))  # transposes list-of-lists structure

    batch = []
    for timestep in range(T):
        graphs_at_timestep = list(transposed[timestep])
        batched_graphs_at_timestep = Batch.from_data_list(graphs_at_timestep)
        batch.append(batched_graphs_at_timestep)
    
    return {'batch': batch, 'B': B, 'T': T}
