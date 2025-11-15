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
        for i in range(num_layers - 1):  # we don't normalize final output
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

    def forward(self, x, edge_index, edge_weight=None, batch=None, batch_size=None, batch_num=-1, timestep=-1):
        """Takes node feature matrix x [num_nodes_total, input_dim/feature_dim], edge_index [2, num_edges_total], 
        optional edge_weight [num_edges_total] and batch assignment vector [num_nodes_total] indicating which graph each node belongs to. 
        Returns node embeddings [num_nodes_total, output_dim], in our case [num_nodes_total, 2], predicted position displacements (in x and y directions) for each node.
        
        Edge weights allow the model to prioritize information from closer agents (higher weights).
        
        IMPORTANT: Each scenario in the batch has its own temporal evolution!
        The GRU hidden states maintain separate parameter trajectories for each scenario,
        ensuring temporal dependencies are scenario-specific, not shared across batch."""

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
        
        # we process each of B batched scenarios separately to maintain temporal specificity
        outputs = []
        for scenario_idx in range(batch_size):
        
            scenario_mask = (batch == scenario_idx)     # Get nodes for this scenario
            scenario_nodes = h[scenario_mask]  # [num_nodes_in_scenario, feature_dim]
            
            # get edges for this scenario (filter edges where both endpoints are in this scenario):
            node_indices = torch.where(scenario_mask)[0]
            edge_mask = torch.isin(edge_index[0], node_indices) & torch.isin(edge_index[1], node_indices)
            scenario_edge_index = edge_index[:, edge_mask]
            
            # remap edge indices to local scenario indices (0 to num_nodes_in_scenario-1)
            old_to_new = torch.full((h.size(0),), -1, dtype=torch.long, device=device)
            old_to_new[node_indices] = torch.arange(scenario_nodes.size(0), device=device)
            scenario_edge_index = old_to_new[scenario_edge_index]
            
            scenario_edge_weight = edge_weight[edge_mask] if edge_weight is not None else None
            
            # process through GCN layers with scenario-specific evolved parameters:
            h_scenario = scenario_nodes
            for layer_idx, gcn in enumerate(self.gcn_layers):
                if debug_mode and scenario_idx == 0: print(f"\tlayer: {layer_idx}, gcn: {gcn}")

                # Summarize this scenario's embeddings
                Z = self._summarize_node_embeddings(h_scenario, self.topk, self.p[layer_idx])  # [topk, feature_dim]
                Z_flat = Z.flatten().unsqueeze(0)  # [1, topk * feature_dim]
                
                # GRU update for THIS scenario only
                h_prev = self.gru_h[layer_idx][scenario_idx:scenario_idx+1]  # [1, num_params]
                h_new = self.gru_cells[layer_idx](Z_flat, h_prev)  # [1, num_params]
                self.gru_h[layer_idx][scenario_idx] = h_new.squeeze(0)

                if debug_mode and scenario_idx == 0: 
                    print(f"\t\tScenario {scenario_idx} evolved params shape: {h_new.shape}")
                
                # Set GCN parameters from THIS scenario's evolved state
                self._set_GCN_params_from_vector(gcn, h_new.squeeze(0))

                # GCN convolution
                h_scenario = gcn(h_scenario, scenario_edge_index, edge_weight=scenario_edge_weight)

                if layer_idx < self.num_layers - 1:
                    h_scenario = self.layer_norms[layer_idx](h_scenario)
                    h_scenario = torchFunctional.relu(h_scenario)
                    h_scenario = torchFunctional.dropout(h_scenario, p=self.dropout, training=self.training)
            
            outputs.append(h_scenario)
        
        # Concatenate outputs back into batch format
        h_out = torch.cat(outputs, dim=0)  # [num_nodes_total, output_dim]
        
        return h_out