import torch
import torch.nn as nn
import torch.nn.functional as torchFunctional
from torch_geometric.nn import GCNConv

class EvolveGCNH(nn.Module):
    """
    EvolveGCN-H: Evolving Graph Convolutional Network
    Based on the paper "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs"(AAAI 2020)
    - Uses GRU to evolve GCN weight matrices over time
    - Treats GCN weights W^(l)_t as hidden states of the GRU
    - Takes node embeddings H^(l)_t as input to the GRU
    - Suitable when node features are informative
    Recurrent update: W^(l)_t = GRU(H^(l)_t, W^(l)_{t-1}) where the node embeddings are summarized to match the dimensionality.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, topk=None):
        super(EvolveGCNH, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.topk = topk if topk is not None else hidden_dim
        
        # GCN layers:
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn_layers.append(GCNConv(hidden_dim, output_dim))
        
        # GRU cells to evolve GCN weights for each layer:
        # - weights adapt based on which nodes are currently important in the graph
        # - input are top k most important node embeddings
        # - output are evolved GCN weights (all parameters new values)
        # https://www.researchgate.net/figure/The-basic-structure-of-a-GRU-cell-41_fig1_340821036

        # Standard GRU equations:
        #   r_t = σ(W_ir @ x_t + W_hr @ h_{t-1} + b_r)  # Reset gate
        #   z_t = σ(W_iz @ x_t + W_hz @ h_{t-1} + b_z)  # Update gate
        #   n_t = tanh(W_in @ x_t + r_t ⊙ (W_hn @ h_{t-1} + b_n))  # New gate
        #   h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}  # New hidden state
        #
        # How top-k nodes drive weight evolution:
        #   1. At each timestep, we have node embeddings H^(l)_t [num_nodes, feature_dim]
        #   2. Attention mechanism selects top-k most informative nodes
        #   3. These k nodes with their features are flattened: [topk, feature_dim] → [topk * feature_dim]
        #   4. This flattened vector becomes GRU input (x_t in equations above)
        #   5. GRU hidden state is the GCN weight vector W^(l)_{t-1} [num_params]
        #   6. GRU outputs new weights W^(l)_t [num_params] based on node information
        self.gru_cells = nn.ModuleList()
        for i, gcn_layer in enumerate(self.gcn_layers):
            num_params = sum(p.numel() for p in gcn_layer.parameters())
            # .parameters() returns an iterator over torch.nn.Parameter tensors:
            #   - Parameter 0: Bias vector [out_channels] 
            #   - Parameter 1: Weight matrix [out_channels, in_channels]
            # .numel() counts total scalar elements: bias_size + (out * in)
            
            # GRU input size must be topk * feature_dim (flattened top-k node embeddings)
            # Feature dim varies by layer: input_dim for first layer, hidden_dim for others
            feature_dim = self.input_dim if i == 0 else self.hidden_dim
            gru_input_size = self.topk * feature_dim
            self.gru_cells.append(nn.GRUCell(gru_input_size, num_params))
        
        # Learnable parameter vector p (attention) for node summarization (one per layer):
        self.p = nn.ParameterList()
        for i in range(num_layers):
            self.p.append(nn.Parameter(torch.randn(hidden_dim if i > 0 else input_dim)))
        
        # GRU hidden states: shape will be [B, num_layers, num_params_per_layer]
        # where B is batch size (number of independent sequences)
        self.gru_h = None   # holds GRU hidden states for each sequence in batch
    
    def reset_parameters(self):
        """Reset all learnable parameters including GCN layers and GRU cells."""
        for gcn in self.gcn_layers:
            gcn.reset_parameters()
        for gru in self.gru_cells:
            gru.reset_parameters()
        for param in self.summarize_params:
            nn.init.normal_(param)
        self.gru_h = None

    def reset_gru_hidden_states(self):
        self.gru_h = None
    
    def _get_params_vector(self, gcn_layer):
        """Extract all parameters from a GCN layer as a single flat vector."""
        params = []
        for p in gcn_layer.parameters():
            params.append(p.view(-1))
        return torch.cat(params)
    
    def _set_params_from_vector(self, gcn_layer, new_parameters_vector):
        """Set GCN layer parameters from a flattened vector."""
        offset = 0
        for p in gcn_layer.parameters():
            numel = p.numel()
            p.data = new_parameters_vector[offset:offset + numel].view(p.shape)
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
    
    def forward(self, x, edge_index=None, batch=None):
        """
        Args:
            x: Either node features [num_nodes, input_dim] OR a PyG Data/Batch object
            edge_index: Graph connectivity [2, num_edges] (optional if x is Data/Batch)
            batch: Batch assignment vector [num_nodes] indicating which graph each node belongs to
                   (optional if x is Data/Batch object)
        
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Handle PyG Data/Batch objects
        if edge_index is None:
            if hasattr(x, 'x') and hasattr(x, 'edge_index'):
                edge_index = x.edge_index
                batch = x.batch if hasattr(x, 'batch') else None
                x = x.x
            else:
                raise ValueError("If edge_index is not provided, x must be a PyG Data/Batch object")
        
        device = x.device
        
        # Determine batch size
        if batch is None:
            B = 1  # Single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        else:
            B = int(batch.max().item()) + 1
        
        # Initialize GRU hidden states from current GCN parameters if needed:
        # Shape: [B, num_layers, num_params_per_layer]
        if self.gru_h is None:
            self.gru_h = []
            for layer_idx, gcn in enumerate(self.gcn_layers):
                params_vector = self._get_params_vector(gcn)
                # Replicate initial weights for each sequence in batch
                layer_hidden = params_vector.detach().unsqueeze(0).repeat(B, 1)  # [B, num_params]
                self.gru_h.append(layer_hidden.to(device))
        else:
            # If batch size changed, reinitialize
            if self.gru_h[0].size(0) != B:
                self.gru_h = []
                for layer_idx, gcn in enumerate(self.gcn_layers):
                    params_vector = self._get_params_vector(gcn)
                    layer_hidden = params_vector.detach().unsqueeze(0).repeat(B, 1)
                    self.gru_h.append(layer_hidden.to(device))
        
        # Move hidden states to correct device if needed:
        if self.gru_h[0].device != device:
            self.gru_h = [h.to(device) for h in self.gru_h]
        
        h = x   # H^(0)_t = X_t (input node features)
        
        # Process each GCN layer
        for layer_index, (GCN, GRU, p) in enumerate(zip(self.gcn_layers, self.gru_cells, self.p)):
            
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
                h = torchFunctional.dropout(h, p=self.dropout, training=self.training)

        return h  # Shape: [num_nodes, output_dim]
