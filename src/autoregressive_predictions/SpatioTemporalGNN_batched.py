"""Batched Spatio-Temporal GNN for trajectory prediction.

Architecture: GCN (spatial) + GRU (temporal) + MLP (decoder)
Supports batch_size > 1 for better GPU utilization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.utils.checkpoint import checkpoint
from config import debug_mode


class SpatioTemporalGNNBatched(nn.Module):
    """GCN/GAT spatial encoder + GRU temporal encoder + MLP decoder.
    Supports multi-scenario batching via PyG batch tensors.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, 
                 num_gru_layers=1, dropout=0.2, use_gat=False, 
                 use_gradient_checkpointing=False, max_agents_per_scenario=128):
        super(SpatioTemporalGNNBatched, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_gru_layers = num_gru_layers
        self.dropout = dropout
        self.use_gat = use_gat
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_agents_per_scenario = max_agents_per_scenario
        
        # Spatial encoder: GCN/GAT layers
        self.spatial_layers = nn.ModuleList()
        
        if use_gat:
            self.spatial_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=False, 
                                               dropout=dropout, add_self_loops=True))
            for _ in range(num_gcn_layers - 1):
                self.spatial_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False,
                                                   dropout=dropout, add_self_loops=True))
        else:
            self.spatial_layers.append(GCNConv(input_dim, hidden_dim, improved=True, add_self_loops=True))
            for _ in range(num_gcn_layers - 1):
                self.spatial_layers.append(GCNConv(hidden_dim, hidden_dim, improved=True, add_self_loops=True))
        
        self.spatial_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gcn_layers)
        ])
        
        # Temporal encoder: GRU processes sequence of spatial features
        # batch_first=True for better memory layout: [batch, seq_len, features]
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,  # Changed for better GPU memory access patterns
            dropout=dropout if num_gru_layers > 1 else 0.0
        )
        
        # Decoder: MLP to predict displacement
        decoder_input_dim = hidden_dim + input_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.gru_hidden = None
        self._current_edge_index = None
        self._current_edge_weight = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all weights with small values for stability."""
        for layer in self.spatial_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reset_gru_hidden_states(self, num_agents=None, device=None, 
                                batch_size=None, agents_per_scenario=None):
        """Reset GRU hidden states for new scenario(s).
        
        Supports two calling conventions:
        1. Simple: reset_gru_hidden_states(num_agents=N) - treats as single flat tensor
        2. Batched: reset_gru_hidden_states(batch_size=B, agents_per_scenario=[n1,...])
        
        The GCN model doesn't use to_dense_batch, so it just needs total node count.
        
        NOTE: This also detaches any existing hidden state, effectively implementing
        truncated BPTT at batch boundaries.
        """
        # Handle batched call - just compute total nodes
        if batch_size is not None and agents_per_scenario is not None:
            num_agents = sum(agents_per_scenario)
        
        if num_agents is None:
            self.gru_hidden = None
        else:
            self.gru_hidden = torch.zeros(
                self.num_gru_layers, 
                num_agents, 
                self.hidden_dim,
                device=device
            )
    
    def detach_hidden_states(self):
        """Detach GRU hidden states from computation graph (for truncated BPTT).
        
        Call this periodically during long rollouts to prevent memory explosion
        while still allowing gradients to flow within the truncation window.
        """
        if self.gru_hidden is not None:
            self.gru_hidden = self.gru_hidden.detach()
    
    def _spatial_layer_forward(self, h, layer, norm, is_last):
        """Single spatial layer forward pass (for gradient checkpointing)."""
        if self.use_gat:
            h = layer(h, self._current_edge_index)
        else:
            h = layer(h, self._current_edge_index, edge_weight=self._current_edge_weight)
        
        h = norm(h)
        
        if not is_last:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def spatial_encoding(self, x, edge_index, edge_weight=None):
        """Extract spatial features using GCN/GAT layers."""
        h = x
        self._current_edge_index = edge_index
        self._current_edge_weight = edge_weight
        
        for i, layer in enumerate(self.spatial_layers):
            is_last = (i == len(self.spatial_layers) - 1)
            
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(
                    self._spatial_layer_forward,
                    h, layer, self.spatial_norms[i], is_last,
                    use_reentrant=False
                )
            else:
                if self.use_gat:
                    h = layer(h, edge_index)
                else:
                    h = layer(h, edge_index, edge_weight=edge_weight)
                
                h = self.spatial_norms[i](h)
                
                if not is_last:
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def forward(self, x, edge_index, edge_weight=None, batch=None, 
                batch_size=None, batch_num=-1, timestep=-1):
        """Forward pass for one timestep with batched scenarios.
        
        Args:
            x: Node features [total_nodes, input_dim] (concatenated from B scenarios)
            edge_index: Edge connectivity [2, total_edges]
            edge_weight: Optional edge weights [total_edges]
            batch: Batch assignment tensor [total_nodes]
            batch_size: Number of scenarios in batch (B)
            batch_num: Batch number (for logging)
            timestep: Timestep in sequence (for logging)
            
        Returns:
            Displacement predictions [total_nodes, output_dim]
        """
        device = x.device
        total_nodes = x.size(0)
        
        # Initialize GRU hidden state if needed
        # Hidden state shape: [num_layers, total_nodes, hidden_dim]
        if self.gru_hidden is None or self.gru_hidden.size(1) != total_nodes:
            self.reset_gru_hidden_states(total_nodes, device)
        
        if self.gru_hidden.device != device:
            self.gru_hidden = self.gru_hidden.to(device)
        
        # 1. Spatial encoding (already batched via PyG - very GPU efficient)
        spatial_features = self.spatial_encoding(x, edge_index, edge_weight)
        
        # 2. Temporal encoding
        # GRU with batch_first=True expects: [batch, seq_len, features]
        # We treat each node as a separate sequence, processing one timestep at a time
        # Input: [total_nodes, 1, hidden_dim] - all nodes process their next timestep in parallel
        gru_input = spatial_features.unsqueeze(1)  # [total_nodes, 1, hidden_dim]
        
        # GRU forward: processes all nodes in parallel on GPU
        # This is efficient because cuDNN GRU kernels are optimized for large batch sizes
        gru_output, new_hidden = self.gru(gru_input, self.gru_hidden)
        
        # CRITICAL: During training, maintain gradient flow for temporal learning
        # During evaluation, detach to save memory
        if self.training:
            self.gru_hidden = new_hidden
        else:
            self.gru_hidden = new_hidden.detach()
        
        # Output: [total_nodes, 1, hidden_dim] -> [total_nodes, hidden_dim]
        temporal_features = gru_output.squeeze(1)
        
        # 3. Skip connection + decode
        decoder_input = torch.cat([temporal_features, x], dim=-1)
        predictions = self.decoder(decoder_input)
        
        predictions = torch.clamp(predictions, min=-5.0, max=5.0)
        
        if debug_mode:
            print(f"------ Batched GNN Forward (B={batch_size}) at timestep {timestep}: ------")
            print(f"  Total nodes: {total_nodes}")
            print(f"  Spatial features: {spatial_features.shape}")
            print(f"  Temporal features: {temporal_features.shape}")
            print(f"  Predictions: {predictions.shape}")
        
        return predictions
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        self._init_weights()
    
    def forward_sequence(self, x_sequence, edge_index, edge_weight=None):
        """Forward pass for entire sequence (useful for testing).
        
        Processes timesteps sequentially, maintaining GRU hidden state.
        """
        num_nodes = x_sequence[0].size(0)
        device = x_sequence[0].device
        self.reset_gru_hidden_states(num_nodes, device)
        
        predictions = []
        for t, x_t in enumerate(x_sequence):
            pred_t = self.forward(x_t, edge_index, edge_weight, timestep=t)
            predictions.append(pred_t)
        
        return predictions
    
    def forward_sequence_parallel(self, x_sequence, edge_index, edge_weight=None):
        """Forward pass for entire sequence with maximum GPU parallelism.
        
        This method processes all timesteps' spatial encodings in parallel,
        then processes the temporal sequence through GRU. More GPU efficient
        for inference when we have the full sequence available.
        
        Args:
            x_sequence: List of T tensors, each [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Optional edge weights
            
        Returns:
            List of T prediction tensors, each [num_nodes, output_dim]
        """
        if len(x_sequence) == 0:
            return []
        
        T = len(x_sequence)
        num_nodes = x_sequence[0].size(0)
        device = x_sequence[0].device
        
        # 1. Batch all spatial encodings (very GPU efficient)
        # Stack all timesteps: [T * num_nodes, input_dim]
        x_stacked = torch.cat(x_sequence, dim=0)
        
        # Expand edge_index for all timesteps
        edge_indices = []
        for t in range(T):
            offset = t * num_nodes
            edge_indices.append(edge_index + offset)
        edge_index_stacked = torch.cat(edge_indices, dim=1)
        
        # Expand edge weights if present
        edge_weight_stacked = None
        if edge_weight is not None:
            edge_weight_stacked = edge_weight.repeat(T)
        
        # Single GCN forward for all timesteps at once
        spatial_features_stacked = self.spatial_encoding(
            x_stacked, edge_index_stacked, edge_weight_stacked
        )
        
        # Reshape to [T, num_nodes, hidden_dim]
        spatial_features = spatial_features_stacked.view(T, num_nodes, self.hidden_dim)
        
        # 2. Temporal encoding: GRU processes all timesteps at once
        # Transpose for batch_first=True: [num_nodes, T, hidden_dim]
        gru_input = spatial_features.transpose(0, 1)
        
        # Reset hidden states
        self.reset_gru_hidden_states(num_nodes, device)
        
        # GRU forward: [num_nodes, T, hidden_dim]
        gru_output, new_hidden = self.gru(gru_input, self.gru_hidden)
        self.gru_hidden = new_hidden.detach()
        
        # 3. Decode all timesteps at once
        # gru_output: [num_nodes, T, hidden_dim]
        # x_sequence stacked for skip connection: [num_nodes, T, input_dim]
        x_for_skip = torch.stack(x_sequence, dim=1)  # [num_nodes, T, input_dim]
        
        decoder_input = torch.cat([gru_output, x_for_skip], dim=-1)  # [num_nodes, T, hidden+input]
        
        # Flatten for decoder: [num_nodes * T, hidden+input]
        decoder_input_flat = decoder_input.view(-1, decoder_input.size(-1))
        predictions_flat = self.decoder(decoder_input_flat)
        predictions_flat = torch.clamp(predictions_flat, min=-5.0, max=5.0)
        
        # Reshape back: [num_nodes, T, output_dim] -> list of T tensors
        predictions = predictions_flat.view(num_nodes, T, self.output_dim)
        predictions_list = [predictions[:, t, :] for t in range(T)]
        
        return predictions_list
