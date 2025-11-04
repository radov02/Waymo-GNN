import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class TrajectoryGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, dropout=0.3):
        super(TrajectoryGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, output_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (trajectory prediction)
        x = self.conv3(x, edge_index)
        
        return x