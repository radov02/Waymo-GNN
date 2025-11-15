import torch
import wandb
import torch.nn.functional as F
from EvolveGCNH import EvolveGCNH
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, topk, epochs, 
                    radius, input_dim, output_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    visualize_every_n_epochs, debug_mode, gradient_clip_value, 
                    loss_alpha, loss_beta, loss_gamma, loss_delta, use_edge_weights)
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from helper_functions.visualization_functions import visualize_epoch
from helper_functions.helpers import advanced_directional_loss, compute_metrics

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/model_training_pipeline.py

def train_epoch(model, dataloader, optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_angle = 0.0
    count = 0
    steps = 0
    
    model.reset_gru_hidden_states(batch_size=batch_size)
    
    for batch_dict in dataloader:
        batched_graph_sequence = batch_dict["batch"]
        B, T = batch_dict["B"], batch_dict["T"]
        
        for t in range(T):
            batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        valid_timesteps = 0
        
        for t, batched_graph in enumerate(batched_graph_sequence):
            if batched_graph.y is None or torch.all(batched_graph.y == 0):
                continue
            
            out_predictions = model(batched_graph.x, batched_graph.edge_index, 
                                   edge_weight=batched_graph.edge_attr if use_edge_weights else None,
                                   batch=batched_graph.batch, batch_size=B, batch_num=0, timestep=t)
            
            loss_t = loss_fn(out_predictions, batched_graph.y.to(out_predictions.dtype), 
                           batched_graph.x, alpha=loss_alpha, beta=loss_beta, gamma=loss_gamma, delta=loss_delta)
            accumulated_loss += loss_t
            valid_timesteps += 1
            
            mse, cos_sim, angle_err = compute_metrics(out_predictions, 
                                                     batched_graph.y.to(out_predictions.dtype),
                                                     batched_graph.x)
            total_mse += mse
            total_cosine += cos_sim
            total_angle += angle_err
            count += 1
        
        if valid_timesteps > 0:
            #avg_loss_batch = accumulated_loss / valid_timesteps
            accumulated_loss.backward()
            clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
            total_loss += accumulated_loss.item()
            steps += 1
    
    return (total_loss / max(1, steps), 
            total_mse / max(1, count), 
            total_cosine / max(1, count), 
            total_angle / max(1, count))

def evaluate(model, dataloader, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device, phase='val'):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_angle = 0.0
    count = 0
    steps = 0
    
    model.reset_gru_hidden_states(batch_size=batch_size)
    
    with torch.no_grad():
        for batch_dict in dataloader:
            batched_graph_sequence = batch_dict["batch"]
            B, T = batch_dict["B"], batch_dict["T"]
            
            for t in range(T):
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
            
            accumulated_loss = 0.0
            valid_timesteps = 0
            
            for t, batched_graph in enumerate(batched_graph_sequence):
                if batched_graph.y is None or torch.all(batched_graph.y == 0):
                    continue
                
                out_predictions = model(batched_graph.x, batched_graph.edge_index, 
                                       edge_weight=batched_graph.edge_attr if use_edge_weights else None,
                                       batch=batched_graph.batch, batch_size=B, batch_num=0, timestep=t)
                
                loss_t = loss_fn(out_predictions, batched_graph.y.to(out_predictions.dtype), 
                               batched_graph.x, alpha=loss_alpha, beta=loss_beta, gamma=loss_gamma, delta=loss_delta)
                accumulated_loss += loss_t
                valid_timesteps += 1
                
                mse, cos_sim, angle_err = compute_metrics(out_predictions, 
                                                         batched_graph.y.to(out_predictions.dtype),
                                                         batched_graph.x)
                total_mse += mse
                total_cosine += cos_sim
                total_angle += angle_err
                count += 1
            
            if valid_timesteps > 0:
                avg_loss_batch = accumulated_loss / valid_timesteps
                total_loss += avg_loss_batch.item()
                steps += 1
    
    return (total_loss / max(1, steps), 
            total_mse / max(1, count), 
            total_cosine / max(1, count), 
            total_angle / max(1, count))



# TODO:
    # - num_workers not working for batches
    # - update README.md
    # - multi-step prediction

    # - use different graph creation methods
    # - do initial_feature_vector() method
    # - review build_edge_index_using_...() functions
    # - implement GAT/transformer model (see Colab 4,5) using HeteroData PyG object (DeepSNAP?)
    # - define the Data/HeteroData object:
    #   - x ... node feature matrix
    #   - edge_index ... edges, shape [2, num_edges]
    #   - edge_attr ... edge feature matrix, shape [num_edges, num_edge_features]
    #   - y ... truth labels
    #   - pos ... node position matrix, shape [num_nodes, num_dimensions]
    #   - time ... timestamps for each event, shape [num_edges] or [num_nodes]



if __name__ == '__main__':
    wandb.login()
    
    wandb_run = wandb.init(
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
            "loss_breakdown": "40% target_angle, 25% VELOCITY_DIR (follow velocity!), 15% vel_mag, 10% MSE, 5% cosine, 5% diversity",
            "use_edge_weights": use_edge_weights
        },
        name=f"Pipeline_r{radius}_h{hidden_channels}_vel_guided{'_ew' if use_edge_weights else ''}",
        dir="../wandb"
    )
    
    model = EvolveGCNH(
        input_dim=input_dim,
        hidden_dim=hidden_channels,
        output_dim=2,
        num_layers=num_layers,
        dropout=dropout,
        topk=topk
    ).to(device)
    
    wandb.watch(model, log='all', log_freq=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = advanced_directional_loss
    
    try:
        train_dataset = HDF5ScenarioDataset("./data/graphs/training/training.hdf5", seq_len=sequence_length)
    except FileNotFoundError:
        print("ERROR: training.hdf5 not found!")
        print("Please run graph_creation_and_saving.py first to create HDF5 files from .tfrecord files.")
        print("Make sure you have .tfrecord files in ./data/scenario/training/ folder.")
        wandb_run.finish()
        exit(1)
    
    try:
        val_dataset = HDF5ScenarioDataset("./data/graphs/validation/validation.hdf5", seq_len=sequence_length)
    except FileNotFoundError:
        print("ERROR: validation.hdf5 not found!")
        print("Please run graph_creation_and_saving.py for validation data.")
        print("Make sure you have .tfrecord files in ./data/scenario/validation/ folder.")
        wandb_run.finish()
        exit(1)
    
    try:
        test_dataset = HDF5ScenarioDataset("./data/graphs/testing/testing.hdf5", seq_len=sequence_length)
    except FileNotFoundError:
        print("ERROR: testing.hdf5 not found!")
        print("Please run graph_creation_and_saving.py for testing data.")
        print("Make sure you have .tfrecord files in ./data/scenario/testing/ folder.")
        wandb_run.finish()
        exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch, 
                             drop_last=True, persistent_workers=True if num_workers > 0 else False)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch, 
                           drop_last=True, persistent_workers=True if num_workers > 0 else False)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch, 
                            drop_last=True, persistent_workers=True if num_workers > 0 else False)
    
    print(f"Training on {device}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} scenarios\n")
    
    viz_batch_train = None
    viz_batch_val = None
    
    for batch_dict in train_loader:
        viz_batch_train = {
            'batch': [b.cpu() for b in batch_dict['batch']],
            'B': batch_dict['B'],
            'T': batch_dict['T'],
            'scenario_ids': batch_dict.get('scenario_ids', [])
        }
        break
    
    for batch_dict in val_loader:
        viz_batch_val = {
            'batch': [b.cpu() for b in batch_dict['batch']],
            'B': batch_dict['B'],
            'T': batch_dict['T'],
            'scenario_ids': batch_dict.get('scenario_ids', [])
        }
        break
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss, train_mse, train_cos, train_angle = train_epoch(
            model, train_loader, optimizer, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device)
        
        val_loss, val_mse, val_cos, val_angle = evaluate(
            model, val_loader, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device, phase='val')
        
        train_angle_deg = train_angle * 180 / 3.14159
        val_angle_deg = val_angle * 180 / 3.14159
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "train_cosine_sim": train_cos,
            "train_angle_error_deg": train_angle_deg,
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_cosine_sim": val_cos,
            "val_angle_error_deg": val_angle_deg
        })
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} Angle: {train_angle_deg:.1f}° | Val Loss: {val_loss:.4f} Angle: {val_angle_deg:.1f}°")
        
        if epoch % visualize_every_n_epochs == 0:
            if viz_batch_train:
                visualize_epoch(epoch, viz_batch_train, model, device, wandb, prefix='train')
            if viz_batch_val and epoch % 5 == 0:
                visualize_epoch(epoch, viz_batch_val, model, device, wandb, prefix='val')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pt')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    
    print("\n=== Testing Best Model ===")
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mse, test_cos, test_angle = evaluate(
        model, test_loader, loss_fn, loss_alpha, loss_beta, loss_gamma, loss_delta, device, 'test')
    
    test_angle_deg = test_angle * 180 / 3.14159
    
    wandb.log({
        "test_loss": test_loss,
        "test_mse": test_mse,
        "test_cosine_sim": test_cos,
        "test_angle_error_deg": test_angle_deg
    })
    
    print(f"Test Loss: {test_loss:.4f} | MSE: {test_mse:.4f} | Cosine: {test_cos:.4f} | Angle: {test_angle_deg:.1f}°")
    
    wandb_run.finish()