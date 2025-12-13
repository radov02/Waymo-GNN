"""Multi-step trajectory prediction using SpatioTemporalGNN.

This module implements multi-horizon trajectory forecasting:
- Predict multiple future timesteps (t+1, t+2, ..., t+K) simultaneously
- Uses autoregressive rollout during training for better generalization
"""

import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from SpatioTemporalGNN import SpatioTemporalGNN
from dataset import HDF5ScenarioDataset
from config import (device, batch_size, num_workers, num_layers, epochs, 
                    radius, input_dim, sequence_length, hidden_channels,
                    dropout, learning_rate, project_name, dataset_name,
                    gradient_clip_value, checkpoint_dir)
from torch.utils.data import DataLoader
from helper_functions.graph_creation_functions import collate_graph_sequences_to_batch
from torch.nn.utils import clip_grad_norm_
import os

# PS:>> $env:PYTHONWARNINGS="ignore"; $env:TF_CPP_MIN_LOG_LEVEL="3"; python ./src/multi_step_prediction.py


class MultiStepSpatioTemporalGNN(nn.Module):
    """Multi-step prediction model based on SpatioTemporalGNN.
    
    Architecture:
    1. Base SpatioTemporalGNN encodes spatio-temporal features
    2. Multiple prediction heads for different future horizons
    3. Optional: shared hidden representation with separate decoders per horizon
    """
    
    def __init__(self, input_dim, hidden_dim, num_future_steps=5, 
                 num_gcn_layers=2, num_gru_layers=1, dropout=0.2, use_gat=False):
        super().__init__()
        
        self.num_future_steps = num_future_steps
        self.hidden_dim = hidden_dim
        
        # Base model for spatio-temporal encoding
        self.base_model = SpatioTemporalGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output hidden features, not predictions
            num_gcn_layers=num_gcn_layers,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
            use_gat=use_gat
        )
        
        # Override decoder - we'll use our own multi-head decoder
        self.base_model.decoder = nn.Identity()
        
        # Multi-step prediction heads
        # Each head predicts displacement for a specific future timestep
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)  # Output: (dx, dy)
            )
            for _ in range(num_future_steps)
        ])
        
        # Temporal decay weights - closer predictions should be more accurate
        self.register_buffer(
            'horizon_weights',
            torch.tensor([1.0 / (i + 1) for i in range(num_future_steps)])
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize prediction head weights."""
        for head in self.prediction_heads:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def reset_gru_hidden_states(self, num_agents=None):
        """Reset GRU hidden states."""
        self.base_model.reset_gru_hidden_states(num_agents)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None, 
                batch_size=None, batch_num=-1, timestep=-1):
        """Forward pass predicting multiple future steps.
        
        Returns:
            predictions: [num_nodes, num_future_steps, 2] - displacement for each horizon
        """
        device = x.device
        num_nodes = x.size(0)
        
        # Get spatio-temporal features from base model
        # Note: base_model.decoder is Identity, so this returns hidden features
        temporal_features = self.base_model.spatial_encoding(x, edge_index, edge_weight)
        
        # Update GRU state
        if self.base_model.gru_hidden is None or self.base_model.gru_hidden.size(1) != num_nodes:
            self.base_model.reset_gru_hidden_states(num_nodes)
        if self.base_model.gru_hidden.device != device:
            self.base_model.gru_hidden = self.base_model.gru_hidden.to(device)
        
        gru_input = temporal_features.unsqueeze(0)
        gru_output, self.base_model.gru_hidden = self.base_model.gru(
            gru_input, self.base_model.gru_hidden
        )
        temporal_features = gru_output.squeeze(0)
        
        # Concatenate with original features for skip connection
        decoder_input = torch.cat([temporal_features, x], dim=-1)
        
        # Predict for each future horizon
        predictions = []
        for head in self.prediction_heads:
            pred = head(decoder_input)
            pred = torch.clamp(pred, min=-5.0, max=5.0)  # Reasonable bounds
            predictions.append(pred)
        
        # Stack: [num_nodes, num_future_steps, 2]
        return torch.stack(predictions, dim=1)


def multi_step_loss(predictions, targets, horizon_weights=None, alpha=0.6):
    """
    Multi-step prediction loss with temporal weighting.
    
    Args:
        predictions: [N, K, 2] - predicted displacements for K future steps
        targets: [N, K, 2] - ground truth displacements
        horizon_weights: [K] - weights for each horizon (closer = more important)
        alpha: Weight for MSE vs angle loss
    """
    num_steps = predictions.shape[1]
    
    if horizon_weights is None:
        horizon_weights = torch.ones(num_steps, device=predictions.device)
    
    total_loss = 0.0
    
    for step in range(num_steps):
        pred_step = predictions[:, step, :]
        target_step = targets[:, step, :]
        
        # MSE loss
        mse_loss = F.mse_loss(pred_step, target_step)
        
        # Angular loss
        pred_angle = torch.atan2(pred_step[:, 1], pred_step[:, 0])
        target_angle = torch.atan2(target_step[:, 1], target_step[:, 0])
        angle_diff = pred_angle - target_angle
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_loss = (angle_diff ** 2).mean()
        
        # Cosine similarity loss
        pred_norm = F.normalize(pred_step, p=2, dim=1, eps=1e-6)
        target_norm = F.normalize(target_step, p=2, dim=1, eps=1e-6)
        cos_sim = F.cosine_similarity(pred_norm, target_norm, dim=1)
        cosine_loss = (1 - cos_sim).mean()
        
        step_loss = alpha * mse_loss + (1 - alpha) * 0.5 * (angle_loss + cosine_loss)
        total_loss += step_loss * horizon_weights[step]
    
    return total_loss / horizon_weights.sum()


def train_epoch_multistep(model, dataloader, optimizer, loss_fn, num_future_steps, device):
    """Train for one epoch with multi-step prediction."""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    count = 0
    steps = 0
    
    for batch_dict in dataloader:
        batched_graph_sequence = batch_dict["batch"]
        B, T = batch_dict["B"], batch_dict["T"]
        
        # Move to device
        for t in range(T):
            batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
        
        # Reset GRU for this scenario
        num_nodes = batched_graph_sequence[0].num_nodes
        model.reset_gru_hidden_states(num_agents=num_nodes)
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        valid_timesteps = 0
        
        # Process sequence, predicting K steps ahead at each timestep
        for t in range(T - num_future_steps):
            batched_graph = batched_graph_sequence[t]
            if batched_graph.y is None:
                continue
            
            # Collect ground truth for K future steps
            future_targets = []
            valid_future = True
            for step in range(num_future_steps):
                future_graph = batched_graph_sequence[t + step + 1]
                if future_graph.y is not None:
                    future_targets.append(future_graph.y)
                else:
                    valid_future = False
                    break
            
            if not valid_future or len(future_targets) != num_future_steps:
                continue
            
            # Forward pass
            out_predictions = model(
                batched_graph.x, 
                batched_graph.edge_index,
                edge_weight=batched_graph.edge_attr,
                batch=batched_graph.batch, 
                batch_size=B, 
                batch_num=0, 
                timestep=t
            )
            
            # Stack targets: [N, K, 2]
            targets_tensor = torch.stack(future_targets, dim=1).to(out_predictions.dtype)
            
            loss_t = loss_fn(out_predictions, targets_tensor, model.horizon_weights)
            accumulated_loss += loss_t
            valid_timesteps += 1
            
            with torch.no_grad():
                mse = F.mse_loss(out_predictions, targets_tensor)
                total_mse += mse.item()
                count += 1
        
        if valid_timesteps > 0:
            accumulated_loss.backward()
            clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
            total_loss += accumulated_loss.item()
            steps += 1
    
    return total_loss / max(1, steps), total_mse / max(1, count)


def evaluate_multistep(model, dataloader, loss_fn, num_future_steps, device):
    """Evaluate multi-step prediction model."""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    
    # Per-horizon metrics
    horizon_mse = [0.0] * num_future_steps
    horizon_counts = [0] * num_future_steps
    
    count = 0
    steps = 0
    
    with torch.no_grad():
        for batch_dict in dataloader:
            batched_graph_sequence = batch_dict["batch"]
            B, T = batch_dict["B"], batch_dict["T"]
            
            for t in range(T):
                batched_graph_sequence[t] = batched_graph_sequence[t].to(device)
            
            num_nodes = batched_graph_sequence[0].num_nodes
            model.reset_gru_hidden_states(num_agents=num_nodes)
            
            accumulated_loss = 0.0
            valid_timesteps = 0
            
            for t in range(T - num_future_steps):
                batched_graph = batched_graph_sequence[t]
                if batched_graph.y is None:
                    continue
                
                future_targets = []
                valid_future = True
                for step in range(num_future_steps):
                    future_graph = batched_graph_sequence[t + step + 1]
                    if future_graph.y is not None:
                        future_targets.append(future_graph.y)
                    else:
                        valid_future = False
                        break
                
                if not valid_future or len(future_targets) != num_future_steps:
                    continue
                
                out_predictions = model(
                    batched_graph.x, 
                    batched_graph.edge_index,
                    edge_weight=batched_graph.edge_attr,
                    batch=batched_graph.batch, 
                    batch_size=B, 
                    batch_num=0, 
                    timestep=t
                )
                
                targets_tensor = torch.stack(future_targets, dim=1).to(out_predictions.dtype)
                
                loss_t = loss_fn(out_predictions, targets_tensor, model.horizon_weights)
                accumulated_loss += loss_t
                valid_timesteps += 1
                
                # Per-horizon MSE
                for h in range(num_future_steps):
                    h_mse = F.mse_loss(out_predictions[:, h, :], targets_tensor[:, h, :])
                    horizon_mse[h] += h_mse.item()
                    horizon_counts[h] += 1
                
                mse = F.mse_loss(out_predictions, targets_tensor)
                total_mse += mse.item()
                count += 1
            
            if valid_timesteps > 0:
                avg_loss_batch = accumulated_loss / valid_timesteps
                total_loss += avg_loss_batch.item()
                steps += 1
    
    # Compute per-horizon average MSE
    horizon_avg_mse = [
        horizon_mse[h] / max(1, horizon_counts[h]) 
        for h in range(num_future_steps)
    ]
    
    return total_loss / max(1, steps), total_mse / max(1, count), horizon_avg_mse


if __name__ == '__main__':
    wandb.login()
    
    num_future_steps = 5  # Predict 0.5 seconds ahead (5 × 0.1s)
    
    run = wandb.init(
        project=project_name,
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dataset": dataset_name,
            "dropout": dropout,
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "epochs": epochs,
            "num_future_steps": num_future_steps,
            "prediction_horizon": f"{num_future_steps * 0.1}s",
            "model": "MultiStepSpatioTemporalGNN"
        },
        name=f"MultiStep{num_future_steps}_h{hidden_channels}",
        dir="../wandb"
    )
    
    model = MultiStepSpatioTemporalGNN(
        input_dim=input_dim,
        hidden_dim=hidden_channels,
        num_future_steps=num_future_steps,
        num_gcn_layers=num_layers,
        num_gru_layers=1,
        dropout=dropout,
        use_gat=False
    ).to(device)
    
    wandb.watch(model, log='all', log_freq=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = multi_step_loss
    
    # Load datasets
    try:
        train_dataset = HDF5ScenarioDataset("./data/graphs/training/training.hdf5", seq_len=sequence_length)
    except FileNotFoundError:
        print("ERROR: training.hdf5 not found!")
        run.finish()
        exit(1)
    
    try:
        val_dataset = HDF5ScenarioDataset("./data/graphs/validation/validation.hdf5", seq_len=sequence_length)
    except FileNotFoundError:
        print("ERROR: validation.hdf5 not found!")
        run.finish()
        exit(1)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch, 
        drop_last=True, persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_graph_sequences_to_batch, 
        drop_last=True, persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Training Multi-Step Prediction Model on {device}")
    print(f"Predicting {num_future_steps} future steps ({num_future_steps * 0.1}s horizon)")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)} scenarios\n")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_loss, train_mse = train_epoch_multistep(
            model, train_loader, optimizer, loss_fn, num_future_steps, device
        )
        
        val_loss, val_mse, horizon_mse = evaluate_multistep(
            model, val_loader, loss_fn, num_future_steps, device
        )
        
        # Log metrics
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "val_loss": val_loss,
            "val_mse": val_mse
        }
        # Add per-horizon MSE
        for h in range(num_future_steps):
            log_dict[f"val_mse_horizon_{h+1}"] = horizon_mse[h]
        
        wandb.log(log_dict)
        
        horizon_str = " | ".join([f"H{h+1}:{horizon_mse[h]:.4f}" for h in range(num_future_steps)])
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {horizon_str}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'num_future_steps': num_future_steps,
                'config': {
                    'input_dim': input_dim,
                    'hidden_channels': hidden_channels,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'num_future_steps': num_future_steps
                }
            }, os.path.join(checkpoint_dir, f'best_multistep_{num_future_steps}_model.pt'))
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    
    print(f"\n✓ Training complete! Best val_loss: {best_val_loss:.4f}")
    run.finish()
