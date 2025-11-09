import torch
import torch.nn as nn
import torch.nn.functional as torchFunctional
from torch_geometric.nn import GCNConv
from config import debug_mode

class EvolveGCNH(nn.Module):
    """EvolveGCN-H: Evolving Graph Convolutional Network
    Based on the paper "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs"(AAAI 2020)"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, topk=None):
        super(EvolveGCNH, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.topk = topk if topk is not None else hidden_dim
        
        # GCN layers (parameters evolve through time for each sequence separately using GRU cells):
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn_layers.append(GCNConv(hidden_dim, output_dim))

        # GRU cells (actual learned parameters for general use):
        self.gru_cells = nn.ModuleList()
        for i, gcn_layer in enumerate(self.gcn_layers):
            num_trainable_params_for_layer = sum(parameters.numel() for parameters in gcn_layer.parameters())   # .parameters() returns an iterator over torch.nn.Parameter tensors: Parameter 0: Bias vector [out_channels] and Parameter 1: Weight matrix [out_channels, in_channels]; .numel() counts total scalar elements: bias_size + (out * in)
            
            # GRU input size must be topk * feature_dim (flattened top-k node embeddings)
            feature_dim = self.input_dim if i == 0 else self.hidden_dim
            gru_input_size = self.topk * feature_dim
            self.gru_cells.append(nn.GRUCell(gru_input_size, num_trainable_params_for_layer))
        
        # ATTENTION parameter for summarization:
        self.p = nn.ParameterList()     # Learnable parameter vector p (attention) for node summarization (one per layer): which nodes are currently important in the graph
        for i in range(num_layers):
            param = nn.Parameter(torch.randn(hidden_dim if i > 0 else input_dim))
            # Better initialization for attention parameters
            nn.init.normal_(param, mean=0, std=0.1)
            self.p.append(param)
        
        # Layer normalization for stabilization
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers - 1):  # Don't normalize final output
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # GRU hidden states:
        self.gru_h = None   # holds GRU hidden states for each sequence in batch, shape: list of [B, num_params_per_layer] per layer
    
    def reset_parameters(self):
        """Reset all learnable parameters."""
        for gru in self.gru_cells:
            gru.reset_parameters()
        for param in self.summarize_params:
            nn.init.normal_(param)
        self.gru_h = None

    def reset_gru_hidden_states(self, batch_size=None):
        if batch_size is None:
            self.gru_h = None
        else:
            # Initialize hidden states for B scenarios with actual GCN weights (more stable training, faster convergence)
            self.gru_h = []
            for gcn in self.gcn_layers:
                params_vector = self._get_params_vector(gcn)
                # Shape: [B, num_params] - one hidden state per scenario
                layer_hidden = params_vector.detach().unsqueeze(0).repeat(batch_size, 1)    # .detach() is important to avoid backpropagating through the initial parameters
                self.gru_h.append(layer_hidden)
    
    def _get_params_vector(self, gcn_layer):
        """Extract all parameters from a GCN layer as a single flat vector."""
        params = []
        for p in gcn_layer.parameters():
            params.append(p.view(-1))
        return torch.cat(params)
    
    def _set_GCN_params_from_vector(self, gcn_layer, new_parameters_vector):
        """Set GCN layer parameters from a flattened vector."""
        offset = 0
        for param in gcn_layer.parameters():
            numel = param.numel()
            param.data = new_parameters_vector[offset:offset + numel].view(param.shape)
            offset += numel
    
    def _summarize_node_embeddings(self, node_embeddings, topk, p):
        """Summarize node embeddings to k representative vectors using attention.
        1. Compute attention weights using learnable parameter p
        2. Select top-k nodes based on attention weights
        3. Apply weighted combination with tanh activation
        4. Pad with zeros if graph has fewer than k nodes
        
        node_embeddings: H^(l)_t of shape [num_nodes, feature_dim]
        topk: number of top nodes to select
        p: Learnable parameter vector p
            
        return: summarized embedding matrix of shape [topk, feature_dim]
        """
        
        p_norm = p / (torch.norm(p) + 1e-8)
        y_t = node_embeddings @ p_norm  # Compute attention scores y_t, of shape [num_nodes]
        
        # Get top-k indices
        actual_k = min(topk, y_t.size(0))  # Use available nodes if less than topk
        if actual_k < y_t.size(0):
            _, top_indices = torch.topk(y_t, actual_k)
        else:
            top_indices = torch.arange(y_t.size(0), device=y_t.device)  # use all nodes
        
        weights = torch.tanh(y_t).unsqueeze(1)  # apply weights from tanh, of shape [num_nodes, 1]
        weighted_embeddings = node_embeddings * weights  # [num_nodes, feature_dim]
        Z_t = weighted_embeddings[top_indices]  # [actual_k, feature_dim]
        
        # Pad with zeros if we have fewer nodes than topk
        if actual_k < topk:
            padding = torch.zeros(topk - actual_k, node_embeddings.size(1), 
                                device=node_embeddings.device, dtype=node_embeddings.dtype)
            Z_t = torch.cat([Z_t, padding], dim=0)  # [topk, feature_dim]
        
        return Z_t
    
    def _summarize_batch_embeddings(self, node_embeddings, batch, batch_size, topk, p):
        """Summarize node embeddings for each graph in the batch separately.
        
        Args:
            node_embeddings: [total_nodes, feature_dim] - all nodes from all graphs
            batch: [total_nodes] - indicates which graph each node belongs to
            batch_size: int - number of graphs in batch (B)
            topk: int - number of top nodes to select per graph
            p: parameter vector for attention
            
        Returns:
            summarized: [B, topk * feature_dim] - flattened summaries for all graphs
        """
        summaries = []
        for i in range(batch_size):
            # Get nodes belonging to graph i
            mask = (batch == i)
            graph_embeddings = node_embeddings[mask]  # [num_nodes_i, feature_dim]
            
            # Summarize this graph's embeddings
            Z_t = self._summarize_node_embeddings(graph_embeddings, topk, p)  # [topk, feature_dim]
            summaries.append(Z_t.flatten())  # [topk * feature_dim]
        
        return torch.stack(summaries)  # [B, topk * feature_dim]

    def forward(self, x, edge_index, batch=None, batch_size=None, batch_num=-1, timestep=-1):
        """Takes node feature matrix x [num_nodes_total, input_dim/feature_dim], edge_index [2, num_edges_total] and batch assignment vector [num_nodes_total] indicating which graph each node belongs to. 
        Returns node embeddings [num_nodes_total, output_dim], in our case [num_nodes_total, 2], predicted position displacements (in x and y directions) for each node."""

        device = x.device

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    
        if batch_size is None:
            batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1

        # Check if we need to reinitialize hidden states due to batch size change
        if self.gru_h is None or self.gru_h[0].size(0) != batch_size:
            self.reset_gru_hidden_states(batch_size)
        
        if self.gru_h[0].device != device:
            self.gru_h = [h.to(device) for h in self.gru_h]
        
        if debug_mode: print(f"------ Forward pass for batch: {batch_num} at time {timestep}: ------------------")

        h = x   # H^(0)_t = X_t (input node features)
        for layer_idx, gcn in enumerate(self.gcn_layers):
            if debug_mode: print(f"\tlayer: {layer_idx}, gcn: {gcn}")

            Z = self._summarize_batch_embeddings(h, batch, batch_size, self.topk, self.p[layer_idx])  # Shape: [B, topk * feature_dim]
            
            # GRU update: update/evolve hidden states (GCN weights) using learned GRU cell parameters
            self.gru_h[layer_idx] = self.gru_cells[layer_idx](Z, self.gru_h[layer_idx])     # W_t^(k) = GRU(Z_t^(k), W_{t-1}^(k))

            if debug_mode: print(f"\t\tself.gru_h[layer_idx] = W_t^(k) has shape [B, num_params]: {self.gru_h[layer_idx].shape}, \n\t\t\tself.gru_h[layer_idx][0][:10]: {self.gru_h[layer_idx][0][:10]}")
            
            # get parameters that were evolved by GRU (across scenarios for this timestep) and average them (or use another aggregation strategy)
            # set these parameters of GRU as GCN parameters
            evolved_params = self.gru_h[layer_idx].mean(dim=0)  # take mean across all batches for each trainable parameter
            self._set_GCN_params_from_vector(gcn, evolved_params)

            h = gcn(h, edge_index)     # put node embeddings and edge_index into GCNConv layer to perform GCN convolutions

            if layer_idx < self.num_layers - 1:
                # Apply layer normalization before activation for stability
                h = self.layer_norms[layer_idx](h)
                h = torchFunctional.relu(h)
                h = torchFunctional.dropout(h, p=self.dropout, training=self.training)
        
        return h  # Shape: [num_nodes_total, output_dim]




"""def _gru_matrix(self, input_matrix, hidden_matrix, gru_cell):
        
        Apply GRU cell in matrix form (process each column independently).
        
        Args:
            input_matrix: Input matrix of shape [topk, num_params]
            hidden_matrix: Hidden state matrix (GCN weights) of shape [num_params]
            gru_cell: GRUCell instance
            
        Returns:
            Updated hidden state (new GCN weights) of shape [num_params]
        
        # Flatten input matrix to vector by taking mean over topk dimension
        # This matches the GRU input dimension
        input_vector = input_matrix.mean(dim=0)  # [num_params]
        
        # Apply GRU cell
        new_hidden = gru_cell(input_vector.unsqueeze(0), hidden_matrix.unsqueeze(0))
        
        return new_hidden.squeeze(0)"""

"""for layer_index, (GCN, GRU, p) in enumerate(zip(self.gcn_layers, self.gru_cells, self.p)):        # Process each GCN layer
            
            # For each graph in the batch, compute summarized embeddings and evolve weights
            new_weights_batch = []
            
            for b in range(B):
                # Get nodes belonging to this graph
                node_mask = (batch == b)
                h_b = h[node_mask]  # Nodes for graph b
                
                # Compute and flatten summary nodes for this graph
                Z_t = self._summarize_node_embeddings(h_b, self.topk, p)  # [topk, feature_dim]
                Z_t_flat = Z_t.T.flatten()  # [feature_dim * topk]
                
                # Evolve the weights for this specific sequence:
                # W^(l)_t[b] = GRU(summarized(H^(l)_t[b]), W^(l)_{t-1}[b])
                new_weights_b = GRU(Z_t_flat.unsqueeze(0), self.gru_h[layer_index][b].unsqueeze(0))
                new_weights_batch.append(new_weights_b.squeeze(0))
            
            # Stack weights for all sequences: [B, num_params]
            self.gru_h[layer_index] = torch.stack(new_weights_batch, dim=0)
            
            # Apply GCN with evolved weights for each graph separately
            h_outputs = []
            for b in range(B):
                node_mask = (batch == b)
                h_b = h[node_mask]
                
                # Get edge indices for this graph
                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                edge_index_b = edge_index[:, edge_mask]
                
                # Remap node indices to local graph indices
                node_mapping = torch.zeros(x.size(0), dtype=torch.long, device=device)
                node_mapping[node_mask] = torch.arange(h_b.size(0), device=device)
                edge_index_b_local = node_mapping[edge_index_b]
                
                # Set GCN weights for this graph
                self._set_params_from_vector(GCN, self.gru_h[layer_index][b])
                
                # Apply graph convolution: H^(l+1)_t = σ(Â_t H^(l)_t W^(l)_t)
                h_b = GCN(h_b, edge_index_b_local)
                
                h_outputs.append(h_b)
            
            # Concatenate outputs from all graphs
            h = torch.cat(h_outputs, dim=0)

            # Apply activation and dropout:
            if layer_index < len(self.gcn_layers)-1:
                h = torchFunctional.relu(h)
                h = torchFunctional.dropout(h, p=self.dropout, training=self.training)"""