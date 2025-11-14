import torch
import wandb
import torch.nn.functional as F
from EvolveGCNH import EvolveGCNH
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, topk, epochs, 
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    visualize_every_n_epochs, visualize_first_batch_only, debug_mode, 
                    gradient_clip_value, loss_alpha, loss_beta, loss_gamma, loss_delta, use_edge_weights)
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from helper_functions.visualization_functions import visualize_epoch
from helper_functions.helpers import advanced_directional_loss

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
            "epochs": epochs,
            "loss_alpha": loss_alpha,
            "loss_beta": loss_beta,
            "loss_gamma": loss_gamma,
            "loss_delta": loss_delta,
            "loss_breakdown": "60% ANGLE (dominant!), 15% MSE, 15% velocity, 10% cosine",
            "use_edge_weights": use_edge_weights
        },
        name=f"GCN_r{radius}_h{hidden_channels}_angle_dom{'_ew' if use_edge_weights else ''}",
        dir="../wandb"
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
    loss_fn = advanced_directional_loss

    try:
        dataset = HDF5ScenarioDataset("./data/graphs/training/training.hdf5", seq_len=sequence_length)
    except FileNotFoundError:
        print("ERROR: training.hdf5 not found!")
        print("Please run graph_creation_and_saving.py first to create HDF5 files from .tfrecord files.")
        print("Make sure you have .tfrecord files in ./data/scenario/training/ folder.")
        training_wandb_run.finish()
        exit(1)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers,    # number of working threads that are used for data loading
                            collate_fn=collate_graph_sequences_to_batch, 
                            drop_last=True,    # discard last minibatch if it does not have enough snapshots
                            persistent_workers=True if num_workers > 0 else False) 

    print(f"Training on {device}")
    print(f"Dataset size: {len(dataset)} scenarios")
    print(f"Batch size: {batch_size}, num workers: {num_workers}")
    print(f"Sequence length: {sequence_length}\n")
    viz_batch = None
    viz_batch_saved = False

    for epoch in range(training_wandb_run.config['epochs']):
        model.train()
        total_loss_epoch = 0.0
        steps = 0
        
        # Track direction metrics
        total_cosine_sim = 0.0
        total_mse = 0.0
        total_angular = 0.0
        metric_count = 0
        
        # Reset GRU hidden states once per epoch
        # DON'T reset per batch - this breaks temporal evolution within scenarios!
        model.reset_gru_hidden_states(batch_size=batch_size)

        for batch, batch_dict in enumerate(dataloader):
            batched_graph_sequence = batch_dict["batch"]  # list of length T of Batch objects (each is a disconnected graph of B components)
            B = batch_dict["B"]
            T = batch_dict["T"]
            print(f"Batch {batch}: B={B} scenarios, T={T} timesteps")
            
            # Auto-reinitialize if batch size changed (should rarely happen with drop_last=True)
            if B != batch_size:
                print(f"Warning: Batch size mismatch (expected {batch_size}, got {B}). Reinitializing GRU states.")
                model.reset_gru_hidden_states(batch_size=B)

            if viz_batch is None and (not visualize_first_batch_only or not viz_batch_saved):   # Save first batch for visualization (only once)
                viz_batch = {
                    'batch': [b.cpu() for b in batch_dict['batch']],
                    'B': batch_dict['B'],
                    'T': batch_dict['T'],
                    'scenario_ids': batch_dict.get('scenario_ids', [])  # Include scenario_ids if available
                }
                viz_batch_saved = True

            for t in range(T): batched_graph_sequence[t] = batched_graph_sequence[t].to(device)

            optimizer.zero_grad()
            accumulated_loss = 0.0
            valid_timesteps = 0     # for skipping loss for when no ground truth is given

            # Process entire sequence with maintained hidden states
            for t, batched_graph in enumerate(batched_graph_sequence):  # batched_graph contains B graphs at timestep t, merged into one batch
                if batched_graph.y is None or torch.all(batched_graph.y == 0):     # skip if no ground truth y or if labels are all zeros (invalid future state)
                    continue

                # Forward: hidden GRU states W evolve for each of B scenarios
                # Pass edge_weight to utilize distance-based attention (closer agents have more influence)
                edge_w = batched_graph.edge_attr if use_edge_weights else None
                out_predictions = model(batched_graph.x, batched_graph.edge_index, 
                                       edge_weight=edge_w,
                                       batch=batched_graph.batch, batch_size=B, batch_num=batch, timestep=t)      # out_predictions: [total_nodes_in_batch, output_dim]
                # out_predictions holds predicted positional displacements for all nodes in the batched graph, shape [num_nodes_total, 2]

                loss_t = loss_fn(out_predictions, batched_graph.y.to(out_predictions.dtype), batched_graph.x, 
                               alpha=loss_alpha, beta=loss_beta, gamma=loss_gamma, delta=loss_delta)
                accumulated_loss += loss_t
                valid_timesteps += 1
                
                # Track metrics for monitoring
                with torch.no_grad():
                    mse = F.mse_loss(out_predictions, batched_graph.y.to(out_predictions.dtype))
                    pred_norm = F.normalize(out_predictions, p=2, dim=1, eps=1e-6)
                    target_norm = F.normalize(batched_graph.y.to(out_predictions.dtype), p=2, dim=1, eps=1e-6)
                    cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1).mean()
                    
                    # Calculate angle error
                    pred_angle = torch.atan2(out_predictions[:, 1], out_predictions[:, 0])
                    target_angle = torch.atan2(batched_graph.y[:, 1], batched_graph.y[:, 0])
                    angle_diff = pred_angle - target_angle
                    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                    mean_angle_error = torch.abs(angle_diff).mean()
                    
                    total_mse += mse.item()
                    total_cosine_sim += cos_sim.item()
                    total_angular += mean_angle_error.item()
                    metric_count += 1

            if valid_timesteps > 0:     # Compute loss for entire sequence, then backprop once (This prevents gradient accumulation issues with retain_graph=True)
                avg_loss_batch = accumulated_loss / valid_timesteps
                avg_loss_batch.backward()
                clip_grad_norm_(model.parameters(), gradient_clip_value)    # to mitigate gradient vanishing/exploding risk that comes with long sequences
                optimizer.step()

                total_loss_epoch += avg_loss_batch.item()  # Convert to scalar only for logging
                steps += 1

        avg_loss_epoch = total_loss_epoch / max(1, steps)
        avg_cosine_sim = total_cosine_sim / max(1, metric_count)
        avg_mse = total_mse / max(1, metric_count)
        avg_angle_error = total_angular / max(1, metric_count)
        avg_angle_error_deg = avg_angle_error * 180 / 3.14159  # Convert to degrees for readability
        
        wandb.log({
            "epoch": epoch,
            "train_avg_loss_per_batch": avg_loss_epoch,
            "avg_cosine_similarity": avg_cosine_sim,
            "avg_mse_component": avg_mse,
            "avg_angle_error_rad": avg_angle_error,
            "avg_angle_error_deg": avg_angle_error_deg
        })
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss_epoch:.6f} | Cosine Sim: {avg_cosine_sim:.4f} | Angle Err: {avg_angle_error_deg:.1f}°")
        visualize_epoch(epoch, viz_batch, model, device, wandb)

    training_wandb_run.finish()
