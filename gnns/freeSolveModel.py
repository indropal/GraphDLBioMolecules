import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv, GraphNorm
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader


class freeSolveModel(torch.nn.Module):
    def __init__(self, n_features, embedding_size):
        super(freeSolveModel,self).__init__()
        torch.manual_seed(95)
        
        # Initialise the Base Model Architecture
        self.in_conv = GCNConv(n_features, embedding_size)
        self.conv1 = GATv2Conv(embedding_size, embedding_size, heads=4, concat=False, dropout=0.2)
        self.lin1 = torch.nn.Linear(embedding_size*4, embedding_size)
        self.gnorm = GraphNorm(embedding_size, eps=2e-5)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.gnorm2 = GraphNorm(embedding_size, eps=2e-5)
        self.conv3 = GATv2Conv(embedding_size, embedding_size, heads=2, concat=False, dropout=0.2)
        self.lin2 = torch.nn.Linear(embedding_size*2, embedding_size)
        self.conv4 = GCNConv(embedding_size, embedding_size)
        
        # output layer of the Graph Neural Net
        self.out = torch.nn.Linear(embedding_size*2, 1)
        
    def forward(self, x, edge_index, batch_index):
        
        hidden = self.in_conv(x, edge_index)
        hidden = F.tanh(hidden)
        
        # Graph Message Passing
        hidden = self.conv1(hidden, edge_index)
        hidden = self.lin1(hidden)
        hidden = F.tanh(hidden)
        
        hidden = self.gnorm(hidden, batch_index)
        
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        hidden = self.gnorm2(hidden, batch_index)
        
        hidden = self.conv3(hidden, edge_index)
        hidden = self.lin2(hidden)
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
    