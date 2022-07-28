import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv, GraphNorm
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader


class baseHIVModel(torch.nn.Module):
    def __init__(self, n_features, embedding_size):
        super(baseHIVModel,self).__init__()
        torch.manual_seed(95)
        
        # Initialise the Base Model Architecture
        
        # input transormation layer to convert Node-Feature Tensor to Embedding Vector of size 'embedding_size'
        self.in_conv = GCNConv(n_features, embedding_size)
        
        # Graph-Convolutions i.e Graph Message Passing Steps
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        
        # output layer of the Graph Neural Net -> Reshape the Output Tensor to meet Target Variable Shape
        # '2*embedding_size' as we concatenate 'Graph_Mean_Pooling' & 'Graph_Max_Pooling' Tensors, each of size 'embedding-size'
        self.out = torch.nn.Linear(embedding_size*2, 1)
        
    def forward(self, x, edge_index, batch_index):
        
        # x > constitutes the Node Information / Features present in the Graph ~ along with Edge Information 'edge_index'
        hidden = self.in_conv(x, edge_index)
        hidden = F.tanh(hidden) # Activation Function
        
        # Graph Message Passing
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        hidden = self.conv3(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        
        # Apply Pooling at Graph-level i.e. Global Pooling ~> combine Node-states of graph to one representation
        # Pooling mechanism to represent the Graph of any number of nodes in to an array of fixed size
        # Concatenate the Global Pooling of Mean (G.A.P) & Max (G.M.P) of these features into a single representation
        # The entire Graph-Information is encoded in this representation / Tensor.
        hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        #hidden = F.sigmoid(hidden)
        
        # Apply a final (linear) regressor
        out = self.out(hidden)
        
        return out, hidden