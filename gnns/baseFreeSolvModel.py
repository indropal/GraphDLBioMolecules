import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv, GraphNorm
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader


# Implementing Baseline Graphical Neural network
class baseFreeSolvModel(torch.nn.Module):
    def __init__(self, n_features, embedding_size):
        super(baseFreeSolvModel,self).__init__()
        torch.manual_seed(95)
        
        # Initialise the Base Model Architecture
        self.in_conv = GCNConv(n_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.conv4 = GCNConv(embedding_size, embedding_size)
        
        # output layer of the Graph Neural Net
        self.out = torch.nn.Linear(embedding_size*2, 1)
        
    def forward(self, x, edge_index, batch_index):
        
        hidden = self.in_conv(x, edge_index)
        hidden = F.tanh(hidden)
        
        # Graph Message Passing
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        hidden = self.conv4(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        # Apply Pooling at Graph-level i.e. Global Pooling
        # Pooling mechanism to represent the Graph of any number of nodes in to an array of fixed size
        # Concatenate the Global Pooling of Mean (G.A.P) & Max (G.M.P) of these features into a single representation
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        
        # Apply a final (linear) regressor
        out = self.out(hidden)
        
        return out, hidden