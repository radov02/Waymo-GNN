"""VectorNet model: hierarchical GNN for trajectory prediction,
predicts full future trajectory at once using Polyline Subgraph Network which aggregates 
vectors within each polyline and a Global Interaction Graph that applies self-attention between polylines"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Feature dimensions for TFRecord dataset
AGENT_VECTOR_DIM = 16  # [ds_x, ds_y, ds_z, de_x, de_y, de_z, vx, vy, heading, width, length, type(4), timestamp]
MAP_VECTOR_DIM = 13    # [ds_x, ds_y, ds_z, de_x, de_y, de_z, type_onehot(7)]


class PolylineSubgraphNetwork(nn.Module):
    """Encodes vectors within a polyline using MLP + MaxPooling."""
    
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
        unique_polylines, inverse_indices = torch.unique(polyline_ids, return_inverse=True)
        P = len(unique_polylines)
        N = h.shape[0]
        
        # First layer
        h = self.encoders[0](h)  # [N, hidden_dim]
        
        # Remaining layers with VECTORIZED aggregation (much faster than for-loops)
        for layer_idx in range(1, self.num_layers):
            # Vectorized max pooling per polyline using scatter_reduce
            # This replaces the slow for-loop with a single CUDA kernel
            polyline_max = torch.zeros(P, self.hidden_dim, device=h.device)
            polyline_max.scatter_reduce_(
                0, 
                inverse_indices.unsqueeze(-1).expand(-1, self.hidden_dim),
                h,
                reduce='amax',
                include_self=False
            )
            
            # Gather back to node level
            agg_features = polyline_max[inverse_indices]  # [N, hidden_dim]
            
            # Concatenate and encode
            h = torch.cat([h, agg_features], dim=-1)  # [N, 2*hidden_dim]
            h = self.encoders[layer_idx](h)  # [N, hidden_dim]
        
        # Final aggregation to get polyline-level features (vectorized)
        polyline_features = torch.zeros(P, self.hidden_dim, device=h.device)
        polyline_features.scatter_reduce_(
            0,
            inverse_indices.unsqueeze(-1).expand(-1, self.hidden_dim),
            h,
            reduce='amax',
            include_self=False
        )
        
        return polyline_features, unique_polylines

class GlobalInteractionGraph(nn.Module):
    """Multi-head self-attention for high-order interactions between polylines."""
    
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
        """Apply self-attention to polyline features [P, D] or [B, P, D]."""
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

class NodeCompletionHead(nn.Module):
    """Masked polyline feature reconstruction head for node completion auxiliary task.
    
    This implements the L_node loss from the VectorNet paper - predicting the original
    polyline features for masked polylines from the global context.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super(NodeCompletionHead, self).__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Reconstruct polyline features.
        
        Args:
            x: [P, hidden_dim] polyline features after global attention
            
        Returns:
            reconstructed: [P, hidden_dim] reconstructed polyline features
        """
        return self.head(x)


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
    3. Applies global self-attention (with optional masking for node completion)
    4. Extracts target agent features
    5. Decodes multi-step trajectory
    6. Optionally reconstructs masked polyline features (node completion task)
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
        node_mask_ratio: float = 0.15,
        mask_target_polyline: bool = False,
    ):
        super(VectorNetTFRecord, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon
        self.node_mask_ratio = node_mask_ratio
        self.mask_target_polyline = mask_target_polyline
        
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
        
        # Node completion head for masked polyline prediction
        self.node_completion_head = NodeCompletionHead(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def _scatter_max(self, src, index, dim_size):
        """Vectorized scatter max operation for aggregating vectors to polylines."""
        # Use scatter_reduce which is much faster than Python loops
        out = torch.full((dim_size, src.shape[-1]), float('-inf'), device=src.device, dtype=src.dtype)
        out.scatter_reduce_(
            0,
            index.unsqueeze(-1).expand(-1, src.shape[-1]),
            src,
            reduce='amax',
            include_self=True
        )
        # Replace -inf with 0 for empty polylines
        out = torch.where(out == float('-inf'), torch.zeros_like(out), out)
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
    
    def forward(self, batch, return_node_completion: bool = False):
        """Forward pass for TFRecord batch format.
        
        Args:
            batch: Dict from vectornet_collate_fn containing:
                - agent_vectors: [N_agent, 16]
                - agent_polyline_ids: [N_agent]
                - agent_batch: [N_agent]
                - map_vectors: [N_map, 13]
                - map_polyline_ids: [N_map]
                - map_batch: [N_map]
                - target_polyline_indices: [total_targets] - indices of all target agent polylines
                - target_scenario_batch: [total_targets] - which scenario each target belongs to
                - total_targets: int - total number of targets across all scenarios
                - batch_size: int
            return_node_completion: If True, also return node completion outputs for L_node loss
                
        Returns:
            predictions: [total_targets, prediction_horizon, output_dim]
            node_completion_dict: (optional) Dict with masked polyline reconstruction outputs
                - 'masked_indices': [num_masked] indices of masked polylines
                - 'reconstructed': [num_masked, hidden_dim] reconstructed features
                - 'original': [num_masked, hidden_dim] original features (pre-norm)
        """
        device = next(self.parameters()).device
        batch_size = batch['batch_size']
        
        agent_vectors = batch['agent_vectors'].to(device)
        agent_polyline_ids = batch['agent_polyline_ids'].to(device)
        map_vectors = batch['map_vectors'].to(device)
        map_polyline_ids = batch['map_polyline_ids'].to(device)
        target_polyline_indices = batch['target_polyline_indices'].to(device)  # [total_targets]
        
        # Encode agent polylines
        agent_polyline_features, agent_unique_ids = self._encode_polylines(
            agent_vectors, agent_polyline_ids, self.agent_encoder
        )
        
        # Encode map polylines
        map_polyline_features, map_unique_ids = self._encode_polylines(
            map_vectors, map_polyline_ids, self.map_encoder
        )
        
        # Store pre-normalized features for node completion target
        if map_polyline_features.numel() > 0:
            all_polylines_prenorm = torch.cat([agent_polyline_features, map_polyline_features], dim=0)
        else:
            all_polylines_prenorm = agent_polyline_features.clone()
        
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
        
        num_polylines = all_polylines.shape[0]
        node_completion_dict = None
        
        # Apply masking for node completion during training
        if self.training and return_node_completion and self.node_mask_ratio > 0.0 and num_polylines > 0:
            # Create mask - randomly select polylines to mask
            rand = torch.rand(num_polylines, device=device)
            mask = rand < self.node_mask_ratio
            
            # Optionally avoid masking target polylines to keep trajectory prediction stable
            if not self.mask_target_polyline:
                for idx in target_polyline_indices:
                    if idx < num_polylines:
                        mask[idx] = False
            
            # Ensure we don't mask everything
            if mask.sum() == num_polylines:
                # Keep at least one polyline unmasked
                mask[0] = False
            
            masked_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            
            if masked_indices.numel() > 0:
                # Zero out masked polyline features
                all_polylines_masked = all_polylines.clone()
                all_polylines_masked[masked_indices] = 0.0
                
                # Global interaction with masked features
                global_features = self.global_graph(all_polylines_masked)  # [P, hidden_dim]
                
                # Reconstruct masked polyline features
                reconstructed = self.node_completion_head(global_features[masked_indices])
                original = all_polylines_prenorm[masked_indices]
                
                node_completion_dict = {
                    'masked_indices': masked_indices,
                    'reconstructed': reconstructed,
                    'original': original,
                }
            else:
                # No polylines masked - regular forward pass
                global_features = self.global_graph(all_polylines)  # [P, hidden_dim]
        else:
            # Regular forward pass without masking
            if all_polylines.numel() > 0:
                global_features = self.global_graph(all_polylines)  # [P, hidden_dim]
            else:
                global_features = torch.zeros(0, self.hidden_dim, device=device)
        
        # Extract target agent features for ALL target agents
        # target_polyline_indices contains indices into agent polylines for all targets
        target_features = global_features[target_polyline_indices]  # [total_targets, hidden_dim]
        
        # Decode trajectories for all target agents
        predictions = self.trajectory_decoder(target_features)  # [total_targets, prediction_horizon, output_dim]
        
        if return_node_completion:
            return predictions, node_completion_dict
        return predictions


# Export:
__all__ = [
    'VectorNetTFRecord',
    'PolylineSubgraphNetwork',
    'GlobalInteractionGraph',
    'MultiStepTrajectoryDecoder',
    'NodeCompletionHead',
    'AGENT_VECTOR_DIM',
    'MAP_VECTOR_DIM',
]
