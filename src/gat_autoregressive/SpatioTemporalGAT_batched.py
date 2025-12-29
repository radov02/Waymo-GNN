"""Batched Spatio-Temporal GAT for trajectory prediction.

Architecture: GAT (spatial) + Per-Agent GRU (temporal) + MLP (decoder)
Maintains per-agent hidden states for temporal memory. Supports batch_size > 1.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
from torch.utils.checkpoint import checkpoint
from config import debug_mode


class SpatioTemporalGATBatched(nn.Module):
    """GAT spatial encoder + Per-Agent GRU temporal encoder + MLP decoder.
    Supports multi-scenario batching with per-agent hidden state tracking.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_gat_layers=2, 
                 num_gru_layers=1, dropout=0.2, num_heads=4, 
                 use_gradient_checkpointing=False, max_agents_per_scenario=128):
        super(SpatioTemporalGATBatched, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gat_layers = num_gat_layers
        self.num_gru_layers = num_gru_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_agents_per_scenario = max_agents_per_scenario
        
        # Spatial encoder: GAT layers with multi-head attention
        self.spatial_layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.spatial_layers.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, concat=False, 
                    dropout=dropout, add_self_loops=True, bias=True)
        )
        
        # Remaining layers: hidden_dim -> hidden_dim
        for _ in range(num_gat_layers - 1):
            self.spatial_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, 
                        dropout=dropout, add_self_loops=True, bias=True)
            )
        
        # Layer normalization for each GAT layer
        self.spatial_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
        ])
        
        # Temporal encoder: GRU processes sequence of spatial features
        # Now processes [seq_len=1, B*max_agents, hidden_dim] with proper masking
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=False,
            dropout=dropout if num_gru_layers > 1 else 0.0
        )
        
        # Decoder: MLP to predict 2D DISPLACEMENT (dx_norm, dy_norm)
        # Same as GCN model - predicts normalized displacement to next position
        decoder_input_dim = hidden_dim + input_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)  # 2D displacement (dx_norm, dy_norm)
        )
        
        # Per-agent GRU hidden states - TRACKED BY AGENT ID
        # Dictionary: (scenario_batch_idx, agent_id) -> hidden_state tensor [num_layers, 1, hidden_dim]
        # This ensures each agent maintains its own temporal memory regardless of order
        self.agent_hidden_states = {}
        
        # Legacy attributes for backward compatibility
        self.gru_hidden = None
        self.current_batch_size = 0
        self.current_agents_per_scenario = None
        self.current_max_agents = None
        
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
    
    def reset_gru_hidden_states(self, batch_size=None, agents_per_scenario=None, 
                                 num_agents=None, device=None):
        """Reset GRU hidden states for new batch of scenarios.
        
        CRITICAL: This clears all per-agent hidden states, starting fresh.
        Call this at the START of each new scenario/batch, not between timesteps!
        
        Supports two calling conventions for backward compatibility:
        1. Batched: reset_gru_hidden_states(batch_size=B, agents_per_scenario=[n1, n2, ...])
        2. Simple: reset_gru_hidden_states(num_agents=N) - treats as single scenario
        
        Args:
            batch_size: Number of scenarios in the batch (B)
            agents_per_scenario: List of agent counts per scenario [n1, n2, ..., nB]
            num_agents: Total number of agents (for simple single-scenario compatibility)
            device: Device to create tensor on
        """
        # CRITICAL: Clear per-agent hidden state dictionary for new scenarios
        self.agent_hidden_states = {}
        
        # Handle simple num_agents call (backward compatibility with visualization)
        if num_agents is not None and batch_size is None:
            batch_size = 1
            agents_per_scenario = [num_agents]
        
        if batch_size is None or agents_per_scenario is None:
            self.gru_hidden = None
            return
            
        self.current_batch_size = batch_size
        self.current_agents_per_scenario = agents_per_scenario
        max_agents = max(agents_per_scenario)
        self.current_max_agents = max_agents
        
        # Legacy hidden state for compatibility (will be replaced by per-agent tracking)
        self.gru_hidden = torch.zeros(
            self.num_gru_layers,
            batch_size * max_agents,
            self.hidden_dim,
            device=device
        )
    
    def spatial_encoding(self, x, edge_index):
        """Extract spatial features using GAT layers.
        
        This is already batched via PyG's Batch mechanism.
        """
        h = x
        self._current_edge_index = edge_index
        
        for i, layer in enumerate(self.spatial_layers):
            is_last = (i == len(self.spatial_layers) - 1)
            
            if self.use_gradient_checkpointing and self.training:
                h = checkpoint(
                    self._spatial_layer_forward,
                    h, layer, self.spatial_norms[i], is_last,
                    use_reentrant=False
                )
            else:
                h = layer(h, edge_index)
                h = self.spatial_norms[i](h)
                if not is_last:
                    h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def _spatial_layer_forward(self, h, layer, norm, is_last):
        """Single spatial layer forward pass (for gradient checkpointing)."""
        h = layer(h, self._current_edge_index)
        h = norm(h)
        if not is_last:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def forward(self, x, edge_index, edge_weight=None, batch=None, 
                batch_size=None, batch_num=-1, timestep=-1, agent_ids=None):
        """Forward pass for one timestep with batched scenarios.
        
        EFFICIENT VECTORIZED VERSION with per-agent hidden state tracking.
        
        Key insight: We use to_dense_batch to group by scenario, then track hidden
        states using a hash of (scenario_idx, agent_id) mapped to a contiguous index.
        This allows vectorized GRU processing while maintaining per-agent memory.
        
        Args:
            x: Node features [total_nodes, input_dim] (concatenated from B scenarios)
            edge_index: Edge connectivity [2, total_edges]
            edge_weight: Ignored for GAT
            batch: Batch assignment tensor [total_nodes] - which scenario each node belongs to
            batch_size: Number of scenarios in batch (B)
            batch_num: Batch number (for logging)
            timestep: Timestep in sequence (for logging)
            agent_ids: List of agent IDs for each node (for per-agent hidden state tracking)
            
        Returns:
            Displacement predictions [total_nodes, output_dim]
        """
        device = x.device
        total_nodes = x.size(0)
        
        # Handle single-scenario case (backward compatible)
        if batch is None or batch_size is None or batch_size == 1:
            return self._forward_single_scenario(x, edge_index, batch_num, timestep, agent_ids)
        
        # ===== EFFICIENT BATCHED PROCESSING WITH PER-AGENT GRU =====
        
        # 1. Spatial encoding (already batched via PyG)
        # GAT processes all agents together, using graph structure for attention
        spatial_features = self.spatial_encoding(x, edge_index)
        
        # 2. Build agent key mapping for hidden state tracking
        # Create unique keys: (scenario_idx, agent_id) for each node
        batch_np = batch.cpu().numpy()
        
        # Build agent keys and gather existing hidden states
        agent_keys = []
        for node_idx in range(total_nodes):
            scenario_idx = int(batch_np[node_idx])
            if agent_ids is not None and node_idx < len(agent_ids):
                agent_id = agent_ids[node_idx]
            else:
                agent_id = node_idx  # Fallback
            agent_keys.append((scenario_idx, agent_id))
        
        # 3. Gather hidden states into a tensor [num_layers, total_nodes, hidden_dim]
        # Initialize with zeros for new agents, use existing for known agents
        h_gathered = torch.zeros(
            self.num_gru_layers, total_nodes, self.hidden_dim,
            device=device, dtype=spatial_features.dtype
        )
        
        for node_idx, key in enumerate(agent_keys):
            if key in self.agent_hidden_states:
                h_gathered[:, node_idx, :] = self.agent_hidden_states[key].squeeze(1)
        
        # 4. VECTORIZED GRU forward: process all agents in parallel
        # Input: [1, total_nodes, hidden_dim], Hidden: [num_layers, total_nodes, hidden_dim]
        gru_input = spatial_features.unsqueeze(0)  # [1, total_nodes, hidden_dim]
        gru_output, new_hidden = self.gru(gru_input, h_gathered)
        
        # 5. Scatter new hidden states back to dictionary (detached)
        new_hidden_detached = new_hidden.detach()
        self.agent_hidden_states = {}
        for node_idx, key in enumerate(agent_keys):
            # Store as [num_layers, 1, hidden_dim] for consistency
            self.agent_hidden_states[key] = new_hidden_detached[:, node_idx:node_idx+1, :]
        
        temporal_features = gru_output.squeeze(0)  # [total_nodes, hidden_dim]
        
        # 6. Skip connection: concatenate temporal features with original node features
        decoder_input = torch.cat([temporal_features, x], dim=-1)
        
        # 7. Decode predictions - predicts 2D NORMALIZED displacement (dx, dy) / POSITION_SCALE
        # Model output matches GT y scale (actual_displacement_meters / 100)
        predictions = self.decoder(decoder_input)
        
        # Clamp displacement predictions to reasonable range in NORMALIZED space
        # [-0.05, 0.05] normalized = [-5, 5] meters per 0.1s timestep = [-50, 50] m/s
        predictions = torch.clamp(predictions, min=-0.05, max=0.05)
        
        if debug_mode:
            print(f"------ Batched GAT Forward (B={batch_size}) at timestep {timestep}: ------")
            print(f"  Total nodes: {total_nodes}")
            print(f"  Unique agents tracked: {len(self.agent_hidden_states)}")
            print(f"  Predictions: {predictions.shape}")
        
        return predictions
    
    def _forward_single_scenario(self, x, edge_index, batch_num=-1, timestep=-1, agent_ids=None):
        """Single-scenario forward pass with per-agent hidden state tracking.
        
        This version tracks hidden states per agent_id to handle agents entering/leaving.
        Falls back to position-based tracking if agent_ids not provided.
        """
        device = x.device
        num_nodes = x.size(0)
        
        spatial_features = self.spatial_encoding(x, edge_index)
        
        # If agent_ids provided, use per-agent hidden state tracking
        if agent_ids is not None:
            # Build agent keys (scenario_idx=0 for single scenario)
            agent_keys = []
            for node_idx in range(num_nodes):
                if node_idx < len(agent_ids):
                    agent_id = agent_ids[node_idx]
                else:
                    agent_id = node_idx
                agent_keys.append((0, agent_id))  # scenario_idx=0
            
            # Gather hidden states
            h_gathered = torch.zeros(
                self.num_gru_layers, num_nodes, self.hidden_dim,
                device=device, dtype=spatial_features.dtype
            )
            
            for node_idx, key in enumerate(agent_keys):
                if key in self.agent_hidden_states:
                    h_gathered[:, node_idx, :] = self.agent_hidden_states[key].squeeze(1)
            
            # Vectorized GRU forward
            gru_input = spatial_features.unsqueeze(0)
            gru_output, new_hidden = self.gru(gru_input, h_gathered)
            
            # Scatter back to dictionary
            new_hidden_detached = new_hidden.detach()
            self.agent_hidden_states = {}
            for node_idx, key in enumerate(agent_keys):
                self.agent_hidden_states[key] = new_hidden_detached[:, node_idx:node_idx+1, :]
            
            temporal_features = gru_output.squeeze(0)
        else:
            # Legacy position-based tracking (for backward compatibility)
            if self.gru_hidden is None or self.gru_hidden.size(1) != num_nodes:
                self.gru_hidden = torch.zeros(
                    self.num_gru_layers,
                    num_nodes,
                    self.hidden_dim,
                    device=device
                )
            
            if self.gru_hidden.device != device:
                self.gru_hidden = self.gru_hidden.to(device)
            
            gru_input = spatial_features.unsqueeze(0)
            gru_output, new_hidden = self.gru(gru_input, self.gru_hidden)
            self.gru_hidden = new_hidden.detach()
            temporal_features = gru_output.squeeze(0)
        
        decoder_input = torch.cat([temporal_features, x], dim=-1)
        predictions = self.decoder(decoder_input)
        
        # Clamp NORMALIZED displacement to reasonable range
        # [-0.05, 0.05] normalized = [-5, 5] meters per 0.1s timestep = [-50, 50] m/s
        predictions = torch.clamp(predictions, min=-0.05, max=0.05)
        
        return predictions
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        self._init_weights()


class SpatioTemporalGATBatchedV2(nn.Module):
    """Alternative batched implementation using PackedSequence for variable-length scenarios.
    
    This version is more memory-efficient when scenarios have very different agent counts.
    Uses torch.nn.utils.rnn.pack_padded_sequence for efficient GRU processing.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_gat_layers=2, 
                 num_gru_layers=1, dropout=0.2, num_heads=4,
                 use_gradient_checkpointing=False):
        super(SpatioTemporalGATBatchedV2, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gat_layers = num_gat_layers
        self.num_gru_layers = num_gru_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Spatial encoder: GAT layers
        self.spatial_layers = nn.ModuleList()
        self.spatial_layers.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, concat=False, 
                    dropout=dropout, add_self_loops=True, bias=True)
        )
        for _ in range(num_gat_layers - 1):
            self.spatial_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, 
                        dropout=dropout, add_self_loops=True, bias=True)
            )
        
        self.spatial_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
        ])
        
        # Per-scenario GRU (processes agents within each scenario)
        # This GRU treats each agent as a sequence element within a scenario
        self.agent_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,  # [B, num_agents, hidden_dim]
            dropout=dropout if num_gru_layers > 1 else 0.0
        )
        
        # Temporal GRU (processes timesteps for each agent)
        # Hidden state maintained across timesteps
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=False,
            dropout=dropout if num_gru_layers > 1 else 0.0
        )
        
        # Decoder
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
        
        self.temporal_hidden = None
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.spatial_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for gru in [self.agent_gru, self.temporal_gru]:
            for name, param in gru.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=0.5)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def reset_temporal_hidden(self, total_agents, device):
        """Reset temporal GRU hidden state for new batch of scenarios."""
        self.temporal_hidden = torch.zeros(
            self.num_gru_layers,
            total_agents,
            self.hidden_dim,
            device=device
        )
    
    def spatial_encoding(self, x, edge_index):
        h = x
        for i, layer in enumerate(self.spatial_layers):
            h = layer(h, edge_index)
            h = self.spatial_norms[i](h)
            if i < len(self.spatial_layers) - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def forward(self, x, edge_index, edge_weight=None, batch=None,
                batch_size=None, batch_num=-1, timestep=-1):
        """Forward pass with batched scenarios.
        
        This version processes spatial features, then applies temporal GRU
        directly on the concatenated node features (simpler than V1).
        """
        device = x.device
        total_nodes = x.size(0)
        
        # Initialize temporal hidden if needed
        if self.temporal_hidden is None or self.temporal_hidden.size(1) != total_nodes:
            self.reset_temporal_hidden(total_nodes, device)
        
        if self.temporal_hidden.device != device:
            self.temporal_hidden = self.temporal_hidden.to(device)
        
        # 1. Spatial encoding (batched via PyG)
        spatial_features = self.spatial_encoding(x, edge_index)
        
        # 2. Temporal encoding - process all agents together
        # GRU input: [1, total_nodes, hidden_dim]
        gru_input = spatial_features.unsqueeze(0)
        gru_output, new_hidden = self.temporal_gru(gru_input, self.temporal_hidden)
        self.temporal_hidden = new_hidden.detach()
        
        temporal_features = gru_output.squeeze(0)  # [total_nodes, hidden_dim]
        
        # 3. Skip connection + decode - outputs NORMALIZED displacement
        decoder_input = torch.cat([temporal_features, x], dim=-1)
        predictions = self.decoder(decoder_input)
        # Clamp in NORMALIZED space: [-0.05, 0.05] = [-5, 5] meters per 0.1s
        predictions = torch.clamp(predictions, min=-0.05, max=0.05)
        
        return predictions
