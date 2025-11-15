import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from EvolveGCNH import EvolveGCNH
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, topk, epochs, 
                    radius, input_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    visualize_every_n_epochs, debug_mode, gradient_clip_value)
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
from helper_functions.visualization_functions import visualize_epoch


# - multi-step prediction (Once your model performs well at short horizons, you can extend to multi-step prediction (predicting the next 5-10 steps simultaneously): trajectory forecasting: Keep 0.1s timesteps, Predict multiple future positions: [t+0.1, t+0.2, t+0.3, t+0.4, t+0.5], This is called "multi-horizon" or "trajectory forecasting", loss would be: loss = MSE(pred[0:5], actual[0:5]))


class MultiStepEvolveGCNH(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_future_steps=5, num_layers=3, dropout=0.3, topk=7):
        super().__init__()
        self.num_future_steps = num_future_steps
        self.base_model = EvolveGCNH(input_dim, hidden_dim, 2, num_layers, dropout, topk)
        self.future_predictors = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(num_future_steps)
        ])
    
    def forward(self, x, edge_index, edge_weight=None, batch=None, batch_size=None, batch_num=-1, timestep=-1):
        # Use the base model's forward to get embeddings (excluding final layer)
        device = x.device
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        if batch_size is None:
            batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
            
        # Get intermediate embeddings from base model
        h = x
        for layer_idx in range(self.base_model.num_layers - 1):
            gcn = self.base_model.gcn_layers[layer_idx]
            Z = self.base_model._summarize_batch_embeddings(h, batch, batch_size, self.base_model.topk, self.base_model.p[layer_idx])
            self.base_model.gru_h[layer_idx] = self.base_model.gru_cells[layer_idx](Z, self.base_model.gru_h[layer_idx])
            evolved_params = self.base_model.gru_h[layer_idx].mean(dim=0)
            self.base_model._set_GCN_params_from_vector(gcn, evolved_params)
            h = gcn(h, edge_index, edge_weight=edge_weight)
            h = self.base_model.layer_norms[layer_idx](h)
            h = torch.nn.functional.relu(h)
            h = torch.nn.functional.dropout(h, p=self.base_model.dropout, training=self.training)
        
        # Now predict multiple future steps
        predictions = []
        for predictor in self.future_predictors:
            pred = predictor(h)
            predictions.append(pred)
        return torch.stack(predictions, dim=1)
    
    def reset_gru_hidden_states(self, batch_size):
        self.base_model.reset_gru_hidden_states(batch_size)

def multi_step_loss(predictions, targets, alpha=0.5):
    """
    Multi-step prediction loss.
    
    Args:
        predictions: [N, num_steps, 2] predicted displacements for multiple steps
        targets: [N, num_steps, 2] ground truth displacements for multiple steps
        alpha: Weight for MSE vs angle loss
    """
    num_steps = predictions.shape[1]
    total_loss = 0.0
    
    for step in range(num_steps):
        pred_step = predictions[:, step, :]
        target_step = targets[:, step, :]
        
        mse_loss = F.mse_loss(pred_step, target_step)
        
        pred_angle = torch.atan2(pred_step[:, 1], pred_step[:, 0])
        target_angle = torch.atan2(target_step[:, 1], target_step[:, 0])
        angle_diff = pred_angle - target_angle
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_loss = torch.abs(angle_diff).mean()
        
        step_loss = alpha * mse_loss + (1 - alpha) * angle_loss
        total_loss += step_loss * (1.0 / (step + 1))
    
    return total_loss / num_steps

def train_epoch_multistep(model, dataloader, optimizer, loss_fn, num_future_steps, device):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
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
        
        for t in range(T - num_future_steps):
            batched_graph = batched_graph_sequence[t]
            if batched_graph.y is None:
                continue
            
            future_targets = []
            for step in range(num_future_steps):
                future_graph = batched_graph_sequence[t + step + 1]
                if future_graph.y is not None:
                    future_targets.append(future_graph.y)
                else:
                    break
            
            if len(future_targets) != num_future_steps:
                continue
            
            out_predictions = model(batched_graph.x, batched_graph.edge_index, 
                                   edge_weight=batched_graph.edge_attr,
                                   batch=batched_graph.batch, batch_size=B, batch_num=0, timestep=t)
            
            targets_tensor = torch.stack(future_targets, dim=1).to(out_predictions.dtype)
            
            loss_t = loss_fn(out_predictions, targets_tensor)
            accumulated_loss += loss_t
            valid_timesteps += 1
            
            with torch.no_grad():
                mse = F.mse_loss(out_predictions, targets_tensor)
                total_mse += mse.item()
                count += 1
        
        if valid_timesteps > 0:
            avg_loss_batch = accumulated_loss / valid_timesteps
            avg_loss_batch.backward()
            clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
            total_loss += avg_loss_batch.item()
            steps += 1
    
    return total_loss / max(1, steps), total_mse / max(1, count)

def evaluate_multistep(model, dataloader, loss_fn, num_future_steps, device):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
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
            
            for t in range(T - num_future_steps):
                batched_graph = batched_graph_sequence[t]
                if batched_graph.y is None:
                    continue
                
                future_targets = []
                for step in range(num_future_steps):
                    future_graph = batched_graph_sequence[t + step + 1]
                    if future_graph.y is not None:
                        future_targets.append(future_graph.y)
                    else:
                        break
                
                if len(future_targets) != num_future_steps:
                    continue
                
                out_predictions = model(batched_graph.x, batched_graph.edge_index, 
                                       edge_weight=batched_graph.edge_attr,
                                       batch=batched_graph.batch, batch_size=B, batch_num=0, timestep=t)
                
                targets_tensor = torch.stack(future_targets, dim=1).to(out_predictions.dtype)
                
                loss_t = loss_fn(out_predictions, targets_tensor)
                accumulated_loss += loss_t
                valid_timesteps += 1
                
                mse = F.mse_loss(out_predictions, targets_tensor)
                total_mse += mse.item()
                count += 1
            
            if valid_timesteps > 0:
                avg_loss_batch = accumulated_loss / valid_timesteps
                total_loss += avg_loss_batch.item()
                steps += 1
    
    return total_loss / max(1, steps), total_mse / max(1, count)

if __name__ == '__main__':
    wandb.login()
    
    num_future_steps = 5
    
    run = wandb.init(
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
            "num_future_steps": num_future_steps,
            "prediction_horizon": f"{num_future_steps * 0.1}s"
        },
        name=f"MultiStep{num_future_steps}_r{radius}_h{hidden_channels}",
        dir="../wandb"
    )
    
    model = MultiStepEvolveGCNH(
        input_dim=input_dim,
        hidden_dim=hidden_channels,
        num_future_steps=num_future_steps,
        num_layers=num_layers,
        dropout=dropout,
        topk=topk
    ).to(device)
    
    wandb.watch(model, log='all', log_freq=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = multi_step_loss
    
    try:
        train_dataset = HDF5ScenarioDataset("./data/graphs/training/training.hdf5", seq_len=sequence_length)
    except FileNotFoundError:
        print("ERROR: training.hdf5 not found!")
        print("Please run graph_creation_and_saving.py first to create HDF5 files from .tfrecord files.")
        print("Make sure you have .tfrecord files in ./data/scenario/training/ folder.")
        run.finish()
        exit(1)
    
    try:
        val_dataset = HDF5ScenarioDataset("./data/graphs/validation/validation.hdf5", seq_len=sequence_length)
    except FileNotFoundError:
        print("ERROR: validation.hdf5 not found!")
        print("Please run graph_creation_and_saving.py for validation data.")
        print("Make sure you have .tfrecord files in ./data/scenario/validation/ folder.")
        run.finish()
        exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch, 
                             drop_last=True, persistent_workers=True if num_workers > 0 else False)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch, 
                           drop_last=True, persistent_workers=True if num_workers > 0 else False)
    
    print(f"Training Multi-Step Prediction Model on {device}")
    print(f"Predicting {num_future_steps} future steps ({num_future_steps * 0.1}s horizon)")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)} scenarios\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss, train_mse = train_epoch_multistep(
            model, train_loader, optimizer, loss_fn, num_future_steps, device)
        
        val_loss, val_mse = evaluate_multistep(
            model, val_loader, loss_fn, num_future_steps, device)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "val_loss": val_loss,
            "val_mse": val_mse
        })
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} MSE: {train_mse:.4f} | Val Loss: {val_loss:.4f} MSE: {val_mse:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'num_future_steps': num_future_steps
            }, f'best_multistep_{num_future_steps}_model.pt')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    
    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")
    run.finish()
