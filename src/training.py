import torch
import wandb
from EvolveGCNH import EvolveGCNH
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, topk, epochs, 
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    visualize_every_n_epochs, visualize_first_batch_only,
                    max_nodes_per_graph_viz, show_timesteps_viz)
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from helper_functions.visualization_functions import visualize_training_progress, load_scenario_for_visualization

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/training.py

if __name__ == '__main__':
    wandb.login()

    training_wandb_run = wandb.init(
        project=project_name,
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dataset": dataset_name,
            "dropout": dropout,
            "hidden_channels": hidden_channels,
            "topk": topk,
            "num_layers": num_layers,
            "epochs": epochs
        },
        name=f"GCN_r{radius}_h{hidden_channels}",
        dir="../wandb"  # Save wandb runs to project root, not src/wandb
    )

    model = EvolveGCNH(
            input_dim=input_dim,
            hidden_dim=hidden_channels,
            output_dim=2,       # we are predicting the position displacements
            num_layers=num_layers,
            dropout=dropout,
            topk=topk
        ).to(device)
    wandb.watch(model, log='all', log_freq=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    dataset = HDF5ScenarioDataset("../data/graphs/training/training.hdf5", seq_len=sequence_length)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers,    # number of working threads that are used for data loading
                            collate_fn=collate_graph_sequences_to_batch, 
                            drop_last=True,    # discard last minibatch if it does not have enough snapshots
                            persistent_workers=True if num_workers > 0 else False) 
    
    # TODO: do not shuffle for validation/test?...

    print(f"Training on {device}")
    print(f"Dataset size: {len(dataset)} scenarios")
    print(f"Batch size: {batch_size}, num workers: {num_workers}")
    print(f"Sequence length: {sequence_length}\n")

    # Keep a validation batch for visualization
    viz_batch = None
    viz_scenario = None
    viz_batch_saved = False  # Flag to track if we've saved the visualization batch

    for epoch in range(training_wandb_run.config['epochs']):
        model.train()
        total_loss_epoch = 0.0
        steps = 0

        for step, batch_dict in enumerate(dataloader):
            batched_graph_sequence = batch_dict["batch"]  # list of length T of Batch objects
            B = batch_dict["B"]
            T = batch_dict["T"]

            # Save first batch for visualization (only once)
            if viz_batch is None and (not visualize_first_batch_only or not viz_batch_saved):
                viz_batch = {
                    'batch': [b.cpu() for b in batch_dict['batch']],
                    'B': batch_dict['B'],
                    'T': batch_dict['T']
                }
                viz_batch_saved = True

            # reset GRU hidden states at the start of each new batch
            model.reset_gru_hidden_states()

            for t in range(T):  # move each batched graph to device
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device)

            optimizer.zero_grad()
            accumulated_loss = 0.0
            valid_timesteps = 0     # linked with skipping loss for when no ground truth is given

            for t, batched_graph in enumerate(batched_graph_sequence):  # batched_graph contains B graphs at timestep t, merged into one batch
                
                out_predictions = model(batched_graph.x, batched_graph.edge_index)      # out_predictions: [total_nodes_in_batch, output_dim]

                # Skip if no ground truth (e.g., last timestep has no future to predict)
                if batched_graph.y is None:
                    continue
                
                # Skip if labels are all zeros (invalid future state)
                if torch.all(batched_graph.y == 0):
                    continue

                # Shape validation (optional, remove after debugging)
                if step == 0 and t == 0 and epoch == 0:
                    print(f"out_predictions shape: {out_predictions.shape}")
                    print(f"batched_graph.y shape: {batched_graph.y.shape}")
                    assert out_predictions.shape == batched_graph.y.shape, \
                        f"Shape mismatch! Predictions: {out_predictions.shape}, Targets: {batched_graph.y.shape}"

                # Use batched_graph.y (position displacements/deltas) for loss function
                loss_t = loss_fn(out_predictions, batched_graph.y.to(out_predictions.dtype))
                
                # Accumulate as scalar, not tensor (prevents computational graph buildup)
                accumulated_loss += loss_t.item()
                
                # Backprop immediately for this timestep
                loss_t.backward(retain_graph=True)
                
                valid_timesteps += 1  # increment counter

            # Only step optimizer after all timesteps
            if valid_timesteps > 0:
                avg_loss_batch = accumulated_loss / valid_timesteps
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss_epoch += avg_loss_batch
                steps += 1

        avg_loss_epoch = total_loss_epoch / max(1, steps)
        wandb.log({"epoch": epoch, "train_loss": avg_loss_epoch})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("="*30)
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss_epoch:.6f}")
        
        # Generate visualization every N epochs (controlled by config)
        should_visualize = (epoch + 1) % visualize_every_n_epochs == 0 or epoch == 0
        
        if should_visualize and viz_batch is not None:
            print(f"  Generating visualization for epoch {epoch+1}...")
            try:
                filepath, avg_error = visualize_training_progress(
                    model, viz_batch, epoch=epoch+1,
                    scenario=viz_scenario,  # Pass scenario for map features
                    save_dir='../visualizations/training',
                    device=device,
                    max_nodes_per_graph=max_nodes_per_graph_viz,
                    show_timesteps=show_timesteps_viz
                )
                # Log visualization error to wandb
                wandb.log({"epoch": epoch, "viz_avg_error": avg_error})
            except Exception as e:
                print(f"  Warning: Visualization failed: {e}")
                import traceback
                traceback.print_exc()
            print("="*30)





    # TODO:
    # - more timesteps?
    # - GRU reset
    # - define the Data/HeteroData object:
    #   - x ... node feature matrix
    #   - edge_index ... edges, shape [2, num_edges]
    #   - edge_attr ... edge feature matrix, shape [num_edges, num_edge_features]
    #   - y ... truth labels
    #   - pos ... node position matrix, shape [num_nodes, num_dimensions]
    #   - time ... timestamps for each event, shape [num_edges] or [num_nodes]
    # - define correct loss_fn and when to calculate it (at each timestep or only at the end)!
    # - add for validation (note more timesteps) and testing
    # - implement whole model training pipeline using wandb (see Colab 2), in evaluation make visualizations
    # - use different graph creation methods
    # - do initial_feature_vector() method
    # - review build_edge_index_using_...() functions
    # - implement GAT/transformer model (see Colab 4) using HeteroData PyG object
    

    training_wandb_run.finish()











def train():
    print("to be implemented")































    """
    # EvolveGCN IMPLEMENTATION FOR WAYMO TEMPORAL GRAPHS
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data, Batch

    def prepare_temporal_graph_data(scenario, start_time=0, end_time=10, radius=30.0):
        Prepare temporal graph sequence from a Waymo scenario, returns list of PyG Data objects one per timestep
        temporal_graphs = []
        for t in range(start_time, min(end_time, len(scenario.timestamps_seconds))):
            graph_data = timestep_to_pyg_data(scenario, t, radius, future_states=1, use_valid_only=True)
            if graph_data is not None and graph_data.num_nodes > 0:
                temporal_graphs.append(graph_data)
            else:
                print(f"Warning: No valid graph at timestep {t}")
        return temporal_graphs

    def train_evolvegcn(model, temporal_graphs, optimizer, loss_fn, device='cpu'):
        Train EvolveGCN on a temporal sequence of graphs, returns average loss over the sequence.
        model.train()
        
        num_graphs = len(temporal_graphs)
        if num_graphs == 0:
            return -1
        
        loss_accum = 0.0
        for t, graph in enumerate(temporal_graphs):
            graph = graph.to(device)
            
            out = model(graph.x, graph.edge_index)
            
            if hasattr(graph, 'y') and graph.y is not None:   # Compute loss (predict future positions)
                loss = loss_fn(out, graph.y)
                loss_accum += loss
                
        optimizer.zero_grad()
        loss_accum.backward()
        optimizer.step()
        
        return loss_accum / num_graphs

    @torch.no_grad()
    def evaluate_evolvegcn(model, temporal_graphs, device='cpu'):
        Evaluate EvolveGCN on a temporal sequence, returns avg prediction error.
        model.eval()
        model.reset_parameters()
        
        total_error = 0
        num_predictions = 0
        
        for graph in temporal_graphs:
            graph = graph.to(device)
            
            out = model(graph.x, graph.edge_index)

            print("out")
            print(out)
            
            if hasattr(graph, 'y') and graph.y is not None:
                error = F.mse_loss(out, graph.y, reduction='sum')
                total_error += error.item()
                num_predictions += graph.num_nodes
        
        return total_error / num_predictions if num_predictions > 0 else 0.0

    # %%
    # TRAIN THE EVOLVEGCN:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    scenario = training_dataset[0][0]
    print(f"\nScenario: {scenario.scenario_id}")
    print(f"Duration: {scenario.timestamps_seconds[-1] - scenario.timestamps_seconds[0]:.1f}s")
    print(f"Number of timesteps: {len(scenario.timestamps_seconds)}")

    print("\nPreparing temporal graphs...")
    start_time = 0
    end_time = 20  # Use first 20 timesteps (2 seconds)
    radius = 30.0

    temporal_graphs = prepare_temporal_graph_data(scenario, start_time, end_time, radius)
    print(f"Created {len(temporal_graphs)} temporal graphs")


    if len(temporal_graphs) > 0:
        input_dim = temporal_graphs[0].x.shape[1]  # Feature dimension
        hidden_dim = 64
        output_dim = temporal_graphs[0].y.shape[1] if hasattr(temporal_graphs[0], 'y') else 2
        
        print(f"\nModel configuration:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        
        model_h = EvolveGCNH(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3).to(device)
        print(f"\nEvolveGCN-H parameters: {sum(p.numel() for p in model_h.parameters()):,}")
        
        # Training setup
        optimizer_h = torch.optim.Adam(model_h.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        epochs = 10
        for epoch in range(epochs):
            model_h.reset_parameters()   # Reset LSTM states at the start of sequence
            loss_h = train_evolvegcn(model_h, temporal_graphs, optimizer_h, loss_fn, device)
            
            if epoch % 2 == 0:
                eval_error_h = evaluate_evolvegcn(model_h, temporal_graphs, device)
                print(f"Epoch {epoch:2d}: Loss={loss_h:.6f}, Eval MSE={eval_error_h:.6f}")
        

        # Final evaluation
        final_error_h = evaluate_evolvegcn(model_h, temporal_graphs, device)
        
        print(f"\nFinal Results:")
        print(f"  EvolveGCN-H MSE: {final_error_h:.6f}")
    else:
        print("\nNo valid temporal graphs created. Try a different scenario or time range.")"""