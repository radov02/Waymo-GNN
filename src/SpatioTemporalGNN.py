"""Stable GNN architecture for trajectory prediction.

Architecture: Static GCN (spatial) + GRU (temporal) + MLP (decoder)
This is much more stable than EvolveGCN for trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from config import debug_mode


class SpatioTemporalGNN(nn.Module):
    """Spatio-Temporal GNN for trajectory prediction.
    
    Architecture:
    1. GCN layers extract spatial features from each timestep independently
    2. GRU processes temporal sequence of spatial features for each agent
    3. MLP decoder produces displacement predictions
    
    This is much more stable than EvolveGCN because:
    - GCN parameters are static (not evolved by GRU)
    - Each agent has its own GRU hidden state
    - Temporal modeling is separated from spatial modeling
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, 
                 num_gru_layers=1, dropout=0.2, use_gat=False):
        super(SpatioTemporalGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_gru_layers = num_gru_layers
        self.dropout = dropout
        self.use_gat = use_gat
        
        # Spatial encoder: GCN/GAT layers
        self.spatial_layers = nn.ModuleList()
        
        if use_gat:
            # GAT with attention for better neighbor weighting
            self.spatial_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
            for _ in range(num_gcn_layers - 1):
                self.spatial_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        else:
            # Standard GCN
            self.spatial_layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_gcn_layers - 1):
                self.spatial_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Layer normalization for each GCN layer
        self.spatial_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gcn_layers)
        ])
        
        # Temporal encoder: GRU processes sequence of spatial features
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=False,  # Input: [seq_len, batch, features]
            dropout=dropout if num_gru_layers > 1 else 0.0
        )
        
        # Decoder: MLP to predict displacement
        # Input: concatenated [temporal_features, original_node_features] for per-node specificity
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
        
        # No output scaling - model learns to predict raw displacements directly
        # For Waymo: typical displacements are 0-3m per 0.1s timestep
        
        # GRU hidden state (per agent, not per scenario)
        self.gru_hidden = None
        
        # Initialize with small weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all weights with small values for stability."""
        for layer in self.spatial_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # Small initialization for GRU
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize decoder with gain=1.0 for better small displacement learning
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reset_gru_hidden_states(self, num_agents=None):
        """Reset GRU hidden states for new scenario.
        
        Args:
            num_agents: Number of agents in the scenario (determines batch size for GRU)
        """
        if num_agents is None:
            self.gru_hidden = None
        else:
            # GRU hidden state: [num_layers, num_agents, hidden_dim]
            self.gru_hidden = torch.zeros(
                self.num_gru_layers, 
                num_agents, 
                self.hidden_dim
            )
    
    def spatial_encoding(self, x, edge_index, edge_weight=None):
        """Extract spatial features using GCN/GAT layers.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            Spatial features [num_nodes, hidden_dim]
        """
        h = x
        
        for i, layer in enumerate(self.spatial_layers):
            if self.use_gat:
                h = layer(h, edge_index)
            else:
                h = layer(h, edge_index, edge_weight=edge_weight)
            
            h = self.spatial_norms[i](h)
            
            if i < len(self.spatial_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def forward(self, x, edge_index, edge_weight=None, batch=None, 
                batch_size=None, batch_num=-1, timestep=-1):
        """Forward pass for one timestep.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            batch: Batch assignment (for multi-graph batches) [num_nodes]
            batch_size: Number of graphs in batch
            batch_num: Batch number (for logging)
            timestep: Timestep in sequence (for logging)
            
        Returns:
            Displacement predictions [num_nodes, output_dim]
        """
        device = x.device
        num_nodes = x.size(0)
        
        # Initialize GRU hidden state if needed
        if self.gru_hidden is None or self.gru_hidden.size(1) != num_nodes:
            self.reset_gru_hidden_states(num_nodes)
        
        # Move hidden state to correct device
        if self.gru_hidden.device != device:
            self.gru_hidden = self.gru_hidden.to(device)
        
        # 1. Spatial encoding: Extract spatial features for this timestep
        spatial_features = self.spatial_encoding(x, edge_index, edge_weight)
        
        # 2. Temporal encoding: Update GRU with spatial features
        # GRU input: [seq_len=1, batch=num_nodes, features=hidden_dim]
        gru_input = spatial_features.unsqueeze(0)  # Add sequence dimension
        
        # GRU forward pass
        gru_output, self.gru_hidden = self.gru(gru_input, self.gru_hidden)
        
        # Remove sequence dimension: [num_nodes, hidden_dim]
        temporal_features = gru_output.squeeze(0)
        
        # 3. Skip connection: Concatenate temporal features with original node features
        # This preserves per-node information that might be lost in GCN/GRU processing
        decoder_input = torch.cat([temporal_features, x], dim=-1)  # [num_nodes, hidden_dim + input_dim]
        
        # 4. Decode: Predict displacement with reasonable bounds
        predictions = self.decoder(decoder_input)
        
        # Clamp to prevent explosive predictions during autoregressive rollout
        # Max realistic displacement: 30 m/s * 0.1s = 3m, add margin → ±5m
        predictions = torch.clamp(predictions, min=-5.0, max=5.0)
        
        if debug_mode:
            print(f"------ Forward pass for batch {batch_num} at timestep {timestep}: ------")
            print(f"  Spatial features: {spatial_features.shape}, range [{spatial_features.min():.3f}, {spatial_features.max():.3f}]")
            print(f"  Temporal features: {temporal_features.shape}, range [{temporal_features.min():.3f}, {temporal_features.max():.3f}]")
            print(f"  Predictions: {predictions.shape}, range [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        return predictions
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        self._init_weights()
    
    def forward_sequence(self, x_sequence, edge_index, edge_weight=None):
        """Forward pass for entire sequence (useful for testing).
        
        Args:
            x_sequence: List of node features [T, num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges] (same for all timesteps)
            edge_weight: Optional edge weights [num_edges]
            
        Returns:
            List of predictions [T, num_nodes, output_dim]
        """
        num_nodes = x_sequence[0].size(0)
        self.reset_gru_hidden_states(num_nodes)
        
        predictions = []
        for t, x_t in enumerate(x_sequence):
            pred_t = self.forward(x_t, edge_index, edge_weight, timestep=t)
            predictions.append(pred_t)
        
        return predictions
