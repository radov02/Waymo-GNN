"""VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation

Implementation based on the paper by Gao et al. (2020):
https://arxiv.org/abs/2005.04259

VectorNet is a hierarchical graph neural network that:
1. First exploits spatial locality of individual road components represented by vectors
2. Then models high-order interactions among all components

Key components:
- Polyline Subgraph Network: Aggregates vectors within each polyline
- Global Interaction Graph: Models interactions between all polylines using self-attention
- Node Completion Auxiliary Task: Predicts masked node features for better context modeling

IMPORTANT: VectorNet predicts the FULL FUTURE TRAJECTORY at once (not autoregressive).
The model encodes history and directly outputs all future timesteps in a single forward pass.

This implementation is designed for the Waymo Open Motion Dataset.

Feature dimensions from VectorNetTFRecordDataset:
- Agent vectors: 16 features [ds_x, ds_y, ds_z, de_x, de_y, de_z, vx, vy, heading, width, length, type(4), timestamp]
- Map vectors: 13 features [ds_x, ds_y, ds_z, de_x, de_y, de_z, type_onehot(7)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Feature dimensions for TFRecord dataset
AGENT_VECTOR_DIM = 16  # [ds_x, ds_y, ds_z, de_x, de_y, de_z, vx, vy, heading, width, length, type(4), timestamp]
MAP_VECTOR_DIM = 13    # [ds_x, ds_y, ds_z, de_x, de_y, de_z, type_onehot(7)]


class PolylineSubgraphNetwork(nn.Module):
    """Polyline Subgraph Network for encoding vectors within a polyline.
    
    Takes vectors belonging to a polyline and produces a single polyline-level feature.
    Uses MLP + MaxPooling as described in the paper.
    
    Architecture per layer:
    - MLP encoder for each vector node
    - Max pooling aggregation across neighbors
    - Concatenation of encoded features with aggregated features
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        super(PolylineSubgraphNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # MLP encoders for each layer
        self.encoders = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.encoders.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        
        # Subsequent layers: 2*hidden_dim (concat) -> hidden_dim
        for _ in range(num_layers - 1):
            self.encoders.append(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for encoder in self.encoders:
            for module in encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, x, polyline_ids):
        """Forward pass through the polyline subgraph network.
        
        Args:
            x: Node features [N, input_dim] where N is total number of vectors
            polyline_ids: Tensor [N] indicating which polyline each vector belongs to
            
        Returns:
            polyline_features: [P, hidden_dim] where P is number of unique polylines
            unique_polylines: [P] unique polyline identifiers
        """
        h = x
        unique_polylines = torch.unique(polyline_ids)
        P = len(unique_polylines)
        
        # First layer
        h = self.encoders[0](h)  # [N, hidden_dim]
        
        # Remaining layers with aggregation
        for layer_idx in range(1, self.num_layers):
            # Aggregate within each polyline using max pooling
            agg_features = torch.zeros(h.shape[0], self.hidden_dim, device=h.device)
            
            for poly_id in unique_polylines:
                mask = polyline_ids == poly_id
                if mask.sum() > 0:
                    poly_features = h[mask]  # [M, hidden_dim]
                    max_pooled = poly_features.max(dim=0)[0]  # [hidden_dim]
                    agg_features[mask] = max_pooled
            
            # Concatenate and encode
            h = torch.cat([h, agg_features], dim=-1)  # [N, 2*hidden_dim]
            h = self.encoders[layer_idx](h)  # [N, hidden_dim]
        
        # Final aggregation to get polyline-level features
        polyline_features = torch.zeros(P, self.hidden_dim, device=h.device)
        for i, poly_id in enumerate(unique_polylines):
            mask = polyline_ids == poly_id
            if mask.sum() > 0:
                polyline_features[i] = h[mask].max(dim=0)[0]
        
        return polyline_features, unique_polylines


class GlobalInteractionGraph(nn.Module):
    """Global Interaction Graph using Self-Attention.
    
    Models high-order interactions among all polyline features using
    multi-head self-attention as described in the paper.
    """
    
    def __init__(self, hidden_dim, num_heads=8, num_layers=1, dropout=0.1):
        super(GlobalInteractionGraph, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.layer_norms1 = nn.ModuleList()
        self.layer_norms2 = nn.ModuleList()
        
        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                )
            )
            self.layer_norms1.append(nn.LayerNorm(hidden_dim))
            self.layer_norms2.append(nn.LayerNorm(hidden_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for layer in self.ffn_layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, polyline_features, mask=None):
        """Forward pass through global interaction graph.
        
        Args:
            polyline_features: [P, hidden_dim] or [B, P, hidden_dim] polyline features
            mask: Optional attention mask [P, P] or [B, P, P]
            
        Returns:
            Updated polyline features [P, hidden_dim] or [B, P, hidden_dim]
        """
        # Handle unbatched input
        if polyline_features.dim() == 2:
            polyline_features = polyline_features.unsqueeze(0)  # [1, P, hidden_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        h = polyline_features
        
        for layer_idx in range(self.num_layers):
            # Self-attention with residual connection
            attn_out, _ = self.attention_layers[layer_idx](h, h, h, attn_mask=mask)
            h = self.layer_norms1[layer_idx](h + attn_out)
            
            # Feed-forward with residual connection
            ffn_out = self.ffn_layers[layer_idx](h)
            h = self.layer_norms2[layer_idx](h + ffn_out)
        
        if squeeze_output:
            h = h.squeeze(0)
        
        return h


class NodeFeatureDecoder(nn.Module):
    """Decoder for node completion auxiliary task.
    
    Predicts masked node features from context as described in the paper.
    """
    
    def __init__(self, hidden_dim, output_dim):
        super(NodeFeatureDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.decoder(x)


class MultiStepTrajectoryDecoder(nn.Module):
    """MLP decoder for predicting full future trajectory at once.
    
    Takes agent polyline features and decodes to multi-step trajectory predictions.
    This is the key component that makes VectorNet non-autoregressive.
    """
    
    def __init__(self, hidden_dim, output_dim=2, prediction_horizon=50, dropout=0.1):
        super(MultiStepTrajectoryDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon
        
        # Multi-layer decoder for better capacity
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * prediction_horizon)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Decode trajectory from features.
        
        Args:
            x: [num_agents, hidden_dim] agent features
            
        Returns:
            trajectories: [num_agents, prediction_horizon, output_dim]
        """
        batch_size = x.shape[0]
        out = self.decoder(x)  # [num_agents, output_dim * prediction_horizon]
        out = out.view(batch_size, self.prediction_horizon, self.output_dim)
        return out


class VectorNetMultiStep(nn.Module):
    """VectorNet for Multi-Step Trajectory Prediction.
    
    This model predicts the FULL future trajectory at once, not autoregressively.
    It encodes history snapshots, aggregates temporal information, and directly
    outputs all future positions in a single forward pass.
    
    The model processes sequences of graph snapshots using:
    1. Per-timestep VectorNet encoding (Polyline + Global Attention)
    2. Temporal aggregation via attention
    3. Multi-step trajectory decoding
    
    Args:
        input_dim: Dimension of input vector features (default: 15)
        hidden_dim: Hidden dimension throughout the network (default: 128)
        output_dim: Output dimension per timestep (default: 2 for dx, dy)
        prediction_horizon: Number of future timesteps to predict (default: 50)
        num_polyline_layers: Number of layers in polyline subgraph network
        num_global_layers: Number of global interaction graph layers
        num_heads: Number of attention heads in global graph
        dropout: Dropout probability
        use_node_completion: Whether to use node completion auxiliary task
        node_completion_ratio: Ratio of nodes to mask for completion task
        use_gradient_checkpointing: Whether to use gradient checkpointing
    """
    
    def __init__(
        self,
        input_dim=15,
        hidden_dim=128,
        output_dim=2,
        prediction_horizon=50,
        num_polyline_layers=3,
        num_global_layers=1,
        num_heads=8,
        dropout=0.1,
        use_node_completion=True,
        node_completion_ratio=0.15,
        use_gradient_checkpointing=False
    ):
        super(VectorNetMultiStep, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon
        self.use_node_completion = use_node_completion
        self.node_completion_ratio = node_completion_ratio
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Polyline Subgraph Network
        self.polyline_net = PolylineSubgraphNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_polyline_layers,
            dropout=dropout
        )
        
        # Global Interaction Graph
        self.global_graph = GlobalInteractionGraph(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_global_layers,
            dropout=dropout
        )
        
        # Temporal aggregation (for processing history sequence)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        
        # FFN after temporal attention
        self.temporal_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Multi-step trajectory decoder
        self.trajectory_decoder = MultiStepTrajectoryDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            prediction_horizon=prediction_horizon,
            dropout=dropout
        )
        
        # Node Completion Decoder (for auxiliary task)
        if use_node_completion:
            self.node_decoder = NodeFeatureDecoder(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim
            )
        else:
            self.node_decoder = None
        
        # Identifier embedding
        self.id_embedding = nn.Linear(2, hidden_dim)
        
        # Store masked info for auxiliary loss computation
        self.masked_indices = None
        self.masked_features = None
        self.masked_predictions = None
        
        # Temporal feature storage for sequence processing
        self._temporal_buffer = []
    
    def _mask_polylines(self, polyline_features, unique_polylines):
        """Randomly mask polyline features for node completion task."""
        P = polyline_features.shape[0]
        num_to_mask = max(1, int(P * self.node_completion_ratio))
        
        perm = torch.randperm(P, device=polyline_features.device)
        mask_indices = perm[:num_to_mask]
        
        mask = torch.zeros(P, dtype=torch.bool, device=polyline_features.device)
        mask[mask_indices] = True
        
        self.masked_indices = mask_indices
        self.masked_features = polyline_features[mask].clone()
        
        masked_polyline_features = polyline_features.clone()
        masked_polyline_features[mask] = 0
        
        return masked_polyline_features, mask
    
    def _add_identifier_embedding(self, polyline_features, x, polyline_ids, unique_polylines):
        """Add identifier embedding to polyline features."""
        P = polyline_features.shape[0]
        id_embeds = torch.zeros(P, self.hidden_dim, device=polyline_features.device)
        
        for i, poly_id in enumerate(unique_polylines):
            mask = polyline_ids == poly_id
            if mask.sum() > 0:
                # Use positions from node features (indices 7, 8 are relative positions)
                if x.shape[1] > 8:
                    positions = x[mask, 7:9]
                else:
                    positions = x[mask, :2]
                min_pos = positions.min(dim=0)[0]
                id_embeds[i] = self.id_embedding(min_pos)
        
        return polyline_features + id_embeds
    
    def encode_timestep(self, x, polyline_ids=None):
        """Encode a single timestep using VectorNet.
        
        Args:
            x: [N, input_dim] node features
            polyline_ids: Optional [N] polyline membership
            
        Returns:
            global_features: [N, hidden_dim] node features after encoding
        """
        device = x.device
        N = x.shape[0]
        
        if polyline_ids is None:
            polyline_ids = torch.arange(N, device=device)
        
        # Polyline encoding
        if self.use_gradient_checkpointing and self.training:
            polyline_features, unique_polylines = checkpoint(
                self.polyline_net, x, polyline_ids, use_reentrant=False
            )
        else:
            polyline_features, unique_polylines = self.polyline_net(x, polyline_ids)
        
        polyline_features = F.normalize(polyline_features, p=2, dim=-1)
        
        # Add position embedding
        polyline_features = self._add_identifier_embedding(
            polyline_features, x, polyline_ids, unique_polylines
        )
        
        # Optional masking for node completion
        if self.training and self.use_node_completion:
            polyline_features_masked, mask = self._mask_polylines(
                polyline_features, unique_polylines
            )
        else:
            polyline_features_masked = polyline_features
            mask = None
        
        # Global interaction
        if self.use_gradient_checkpointing and self.training:
            global_features = checkpoint(
                self.global_graph, polyline_features_masked, use_reentrant=False
            )
        else:
            global_features = self.global_graph(polyline_features_masked)
        
        # Node completion prediction
        if self.training and self.use_node_completion and mask is not None:
            masked_outputs = global_features[mask]
            self.masked_predictions = self.node_decoder(masked_outputs)
        
        return global_features
    
    def reset_temporal_buffer(self):
        """Clear the temporal feature buffer."""
        self._temporal_buffer = []
    
    def forward_single_step(self, x, edge_index, edge_weight=None, 
                            is_last_step=False, batch=None, **kwargs):
        """Process a single timestep and optionally predict.
        
        This method accumulates temporal features. Call with is_last_step=True
        on the final history timestep to get trajectory predictions.
        
        Args:
            x: [N, input_dim] node features
            edge_index: Edge indices (for compatibility, not used directly)
            edge_weight: Optional edge weights
            is_last_step: If True, aggregate temporal features and predict
            batch: Optional batch tensor
            
        Returns:
            If is_last_step=True: [N, prediction_horizon, output_dim] predictions
            Otherwise: None (features stored in buffer)
        """
        # Encode this timestep
        global_features = self.encode_timestep(x)
        self._temporal_buffer.append(global_features)
        
        if not is_last_step:
            return None
        
        # Aggregate temporal features and predict
        N = x.shape[0]
        device = x.device
        T = len(self._temporal_buffer)
        
        # Stack temporal features
        if T == 1:
            agent_features = self._temporal_buffer[0]
        else:
            # Check if all timesteps have same number of nodes
            if all(f.shape[0] == N for f in self._temporal_buffer):
                temporal_stack = torch.stack(self._temporal_buffer, dim=0)  # [T, N, H]
                temporal_stack = temporal_stack.permute(1, 0, 2)  # [N, T, H]
                
                # Query from last timestep
                query = self._temporal_buffer[-1].unsqueeze(1)  # [N, 1, H]
                
                # Temporal attention
                attn_out, _ = self.temporal_attention(
                    query, temporal_stack, temporal_stack
                )
                attn_out = attn_out.squeeze(1)  # [N, H]
                
                # Residual connection
                agent_features = self.temporal_norm(
                    self._temporal_buffer[-1] + attn_out
                )
            else:
                # Fallback: just use last timestep
                agent_features = self._temporal_buffer[-1]
        
        # FFN for final processing
        agent_features = agent_features + self.temporal_ffn(agent_features)
        
        # Clear buffer
        self.reset_temporal_buffer()
        
        # Decode full trajectory
        predictions = self.trajectory_decoder(agent_features)
        
        return predictions
    
    def forward(self, x, edge_index, edge_weight=None, batch=None,
                batch_size=None, **kwargs):
        """Single-step forward for compatibility.
        
        Encodes features and directly predicts full trajectory.
        For sequence processing, use forward_single_step() iteratively.
        
        Args:
            x: [N, input_dim] node features
            edge_index: Edge indices (for compatibility)
            edge_weight: Optional edge weights
            batch: Optional batch tensor
            batch_size: Number of scenarios in batch
            
        Returns:
            predictions: [N, prediction_horizon, output_dim]
        """
        # Single-step encoding
        global_features = self.encode_timestep(x)
        
        # Temporal processing (single step = just FFN)
        agent_features = global_features + self.temporal_ffn(global_features)
        
        # Decode full trajectory
        predictions = self.trajectory_decoder(agent_features)
        
        return predictions
    
    def forward_sequence(self, sequence):
        """Process a sequence of graphs and predict full future trajectory.
        
        This is the main method for multi-step prediction from history.
        
        Args:
            sequence: List of PyG Data objects or dict with 'x', 'edge_index'
            
        Returns:
            predictions: [N, prediction_horizon, output_dim] trajectory predictions
        """
        self.reset_temporal_buffer()
        
        T = len(sequence)
        
        for t in range(T):
            data = sequence[t]
            
            if hasattr(data, 'x'):
                x = data.x
            else:
                x = data['x']
            
            if hasattr(data, 'edge_index'):
                edge_index = data.edge_index
            else:
                edge_index = data.get('edge_index', None)
            
            is_last = (t == T - 1)
            result = self.forward_single_step(x, edge_index, is_last_step=is_last)
            
            if is_last:
                return result
        
        return None
    
    def get_node_completion_loss(self):
        """Compute node completion auxiliary loss."""
        if self.masked_predictions is None or self.masked_features is None:
            return torch.tensor(0.0)
        
        loss = F.huber_loss(self.masked_predictions, self.masked_features)
        
        self.masked_predictions = None
        self.masked_features = None
        self.masked_indices = None
        
        return loss
    
    def reset_gru_hidden_states(self, **kwargs):
        """Compatibility method - VectorNet doesn't use GRU hidden states."""
        self.reset_temporal_buffer()


# Backward compatibility aliases
VectorNet = VectorNetMultiStep
VectorNetTemporal = VectorNetMultiStep


class VectorNetTFRecord(nn.Module):
    """VectorNet model designed for VectorNetTFRecordDataset.
    
    This model accepts the batch format from vectornet_collate_fn:
    - agent_vectors: [N_agent, 16] agent trajectory polyline vectors
    - agent_polyline_ids: [N_agent] polyline membership for agents
    - agent_batch: [N_agent] batch indices for agents
    - map_vectors: [N_map, 13] map feature polyline vectors
    - map_polyline_ids: [N_map] polyline membership for map
    - map_batch: [N_map] batch indices for map
    - target_polyline_idx: [B] target agent polyline index
    - future_positions: [B, future_len, 2] ground truth
    - future_valid: [B, future_len] validity mask
    
    The model:
    1. Separately encodes agent and map polylines
    2. Concatenates all polyline features
    3. Applies global self-attention
    4. Extracts target agent features
    5. Decodes multi-step trajectory
    """
    
    def __init__(
        self,
        agent_input_dim: int = 16,
        map_input_dim: int = 13,
        hidden_dim: int = 128,
        output_dim: int = 2,
        prediction_horizon: int = 50,
        num_polyline_layers: int = 3,
        num_global_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_node_completion: bool = True,
        node_completion_ratio: float = 0.15,
    ):
        super(VectorNetTFRecord, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon
        self.use_node_completion = use_node_completion
        self.node_completion_ratio = node_completion_ratio
        
        # Separate encoders for agents and map features
        self.agent_encoder = PolylineSubgraphNetwork(
            input_dim=agent_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_polyline_layers,
            dropout=dropout
        )
        
        self.map_encoder = PolylineSubgraphNetwork(
            input_dim=map_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_polyline_layers,
            dropout=dropout
        )
        
        # Global interaction graph
        self.global_graph = GlobalInteractionGraph(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_global_layers,
            dropout=dropout
        )
        
        # Trajectory decoder
        self.trajectory_decoder = MultiStepTrajectoryDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            prediction_horizon=prediction_horizon,
            dropout=dropout
        )
        
        # Node completion decoder
        if use_node_completion:
            self.node_decoder = NodeFeatureDecoder(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim
            )
        else:
            self.node_decoder = None
        
        # Storage for auxiliary loss
        self.masked_indices = None
        self.masked_features = None
        self.masked_predictions = None
    
    def _scatter_max(self, src, index, dim_size):
        """Scatter max operation for aggregating vectors to polylines."""
        out = torch.zeros(dim_size, src.shape[-1], device=src.device)
        for i in range(dim_size):
            mask = index == i
            if mask.sum() > 0:
                out[i] = src[mask].max(dim=0)[0]
        return out
    
    def _encode_polylines(self, vectors, polyline_ids, encoder):
        """Encode vectors into polyline-level features.
        
        Args:
            vectors: [N, feature_dim] vector features
            polyline_ids: [N] polyline membership
            encoder: PolylineSubgraphNetwork to use
            
        Returns:
            polyline_features: [P, hidden_dim] polyline features
            unique_ids: [P] unique polyline identifiers
        """
        if vectors.numel() == 0:
            return torch.zeros(0, self.hidden_dim, device=vectors.device), torch.tensor([], dtype=torch.long, device=vectors.device)
        
        polyline_features, unique_ids = encoder(vectors, polyline_ids)
        return polyline_features, unique_ids
    
    def _mask_polylines(self, polyline_features):
        """Randomly mask polyline features for node completion task."""
        P = polyline_features.shape[0]
        if P == 0:
            return polyline_features, None
            
        num_to_mask = max(1, int(P * self.node_completion_ratio))
        
        perm = torch.randperm(P, device=polyline_features.device)
        mask_indices = perm[:num_to_mask]
        
        mask = torch.zeros(P, dtype=torch.bool, device=polyline_features.device)
        mask[mask_indices] = True
        
        self.masked_indices = mask_indices
        self.masked_features = polyline_features[mask].clone()
        
        masked_polyline_features = polyline_features.clone()
        masked_polyline_features[mask] = 0
        
        return masked_polyline_features, mask
    
    def forward(self, batch):
        """Forward pass for TFRecord batch format.
        
        Args:
            batch: Dict from vectornet_collate_fn containing:
                - agent_vectors: [N_agent, 16]
                - agent_polyline_ids: [N_agent]
                - agent_batch: [N_agent]
                - map_vectors: [N_map, 13]
                - map_polyline_ids: [N_map]
                - map_batch: [N_map]
                - target_polyline_idx: [B]
                - batch_size: int
                
        Returns:
            predictions: [B, prediction_horizon, output_dim]
        """
        device = next(self.parameters()).device
        batch_size = batch['batch_size']
        
        agent_vectors = batch['agent_vectors'].to(device)
        agent_polyline_ids = batch['agent_polyline_ids'].to(device)
        map_vectors = batch['map_vectors'].to(device)
        map_polyline_ids = batch['map_polyline_ids'].to(device)
        target_polyline_idx = batch['target_polyline_idx'].to(device)
        
        # Encode agent polylines
        agent_polyline_features, agent_unique_ids = self._encode_polylines(
            agent_vectors, agent_polyline_ids, self.agent_encoder
        )
        
        # Encode map polylines
        map_polyline_features, map_unique_ids = self._encode_polylines(
            map_vectors, map_polyline_ids, self.map_encoder
        )
        
        # L2 normalize (paper requirement)
        if agent_polyline_features.numel() > 0:
            agent_polyline_features = F.normalize(agent_polyline_features, p=2, dim=-1)
        if map_polyline_features.numel() > 0:
            map_polyline_features = F.normalize(map_polyline_features, p=2, dim=-1)
        
        # Concatenate all polylines
        if map_polyline_features.numel() > 0:
            all_polylines = torch.cat([agent_polyline_features, map_polyline_features], dim=0)
        else:
            all_polylines = agent_polyline_features
        
        # Masking for node completion
        if self.training and self.use_node_completion:
            all_polylines_masked, mask = self._mask_polylines(all_polylines)
        else:
            all_polylines_masked = all_polylines
            mask = None
        
        # Global interaction (self-attention over all polylines)
        if all_polylines_masked.numel() > 0:
            global_features = self.global_graph(all_polylines_masked)  # [P, hidden_dim]
        else:
            global_features = torch.zeros(0, self.hidden_dim, device=device)
        
        # Node completion prediction
        if self.training and self.use_node_completion and mask is not None and mask.sum() > 0:
            masked_outputs = global_features[mask]
            self.masked_predictions = self.node_decoder(masked_outputs)
        
        # Extract target agent features
        # target_polyline_idx indexes into agent polylines, which are first in the concatenation
        target_features = global_features[target_polyline_idx]  # [B, hidden_dim]
        
        # Decode trajectories
        predictions = self.trajectory_decoder(target_features)  # [B, prediction_horizon, output_dim]
        
        return predictions
    
    def get_node_completion_loss(self):
        """Compute node completion auxiliary loss."""
        if self.masked_predictions is None or self.masked_features is None:
            return torch.tensor(0.0)
        
        loss = F.huber_loss(self.masked_predictions, self.masked_features)
        
        self.masked_predictions = None
        self.masked_features = None
        self.masked_indices = None
        
        return loss


# Export all models
__all__ = [
    'VectorNet',
    'VectorNetMultiStep',
    'VectorNetTemporal',  # Alias for compatibility
    'VectorNetTFRecord',  # New model for TFRecord dataset
    'PolylineSubgraphNetwork',
    'GlobalInteractionGraph',
    'MultiStepTrajectoryDecoder',
    'AGENT_VECTOR_DIM',
    'MAP_VECTOR_DIM',
]
