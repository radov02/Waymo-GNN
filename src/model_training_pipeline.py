import torch
from EvolveGCNH import EvolveGCNH
from training import train
from testing import test





# TODO:
# define model
# train(model)
# eval(model)
# show results (and save them to wandb)








model = EvolveGCNH(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_channels'],
            output_dim=output_dim,
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            topk=config['topk']
        ).to(device)

model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = F.nll_loss

best_model = None
best_valid_acc = 0

for epoch in range(1, 1 + args["epochs"]):
    loss = train(model, data, train_idx, optimizer, loss_fn)
    result = test(model, data, split_idx, evaluator)
    train_acc, valid_acc, test_acc = result
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')
    





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