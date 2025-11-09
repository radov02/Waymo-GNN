"""Quick check to see if training data is normalized"""
import h5py
import numpy as np

hdf5_path = "./data/graphs/training/training.hdf5"

try:
    with h5py.File(hdf5_path, 'r') as f:
        # Check if it has the old or new structure
        if 'scenarios' in f:
            print("Found 'scenarios' group - checking structure...")
            scenarios_group = f['scenarios']
            scenario_ids = list(scenarios_group.keys())
            
            if not scenario_ids:
                print("No scenarios found in HDF5 file!")
                exit(1)
            
            print(f"Found {len(scenario_ids)} scenarios")
            first_scenario_id = scenario_ids[0]
            print(f"First scenario: {first_scenario_id}")
            
            first_scenario = scenarios_group[first_scenario_id]
            if 'snapshot_graphs' in first_scenario:
                snapshot_graphs = first_scenario['snapshot_graphs']
                snapshot_ids = list(snapshot_graphs.keys())
                print(f"Found {len(snapshot_ids)} snapshots")
                
                first_snapshot = snapshot_graphs[snapshot_ids[0]]
                
                # Check node features (x)
                x = first_snapshot['x'][:]
                print(f"\nNode features (x) shape: {x.shape}")
                print(f"Position values (first 5 nodes, x and y):")
                print(x[:5, :2])
                print(f"\nPosition range: x=[{x[:, 0].min():.2f}, {x[:, 0].max():.2f}], y=[{x[:, 1].min():.2f}, {x[:, 1].max():.2f}]")
                
                # Check labels (y)
                if 'y' in first_snapshot:
                    y = first_snapshot['y'][:]
                    print(f"\nDisplacement labels (y) shape: {y.shape}")
                    print(f"Displacement values (first 5):")
                    print(y[:5])
                    print(f"Displacement range: dx=[{y[:, 0].min():.2f}, {y[:, 0].max():.2f}], dy=[{y[:, 1].min():.2f}, {y[:, 1].max():.2f}]")
            else:
                print("No 'snapshot_graphs' found in scenario!")
                exit(1)
        else:
            print("Unknown HDF5 structure!")
            print(f"Top-level keys: {list(f.keys())}")
            exit(1)
        
        print("\n" + "="*60)
        if abs(x[:, 0].max()) > 50 or abs(x[:, 1].max()) > 50:
            print("❌ DATA IS NOT NORMALIZED!")
            print("Position values are in the hundreds/thousands (raw meters).")
            print("\nYou MUST regenerate the training data:")
            print("  python ./src/graph_creation_and_saving.py")
        else:
            print("✅ Data appears to be normalized (values < 50)")
            print("Position values are in reasonable range for normalized data.")
        print("="*60)
        
except FileNotFoundError:
    print(f"❌ File not found: {hdf5_path}")
    print("You need to generate training data first:")
    print("  python ./src/graph_creation_and_saving.py")
except Exception as e:
    print(f"Error reading HDF5 file: {e}")
