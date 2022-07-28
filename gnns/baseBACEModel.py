import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv, GraphNorm, TransformerConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import TopKPooling
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader


# Implementing Baseline Graphical Neural network
class baseBACEModel(torch.nn.Module):
    def __init__(self, feature_size, embedding_size):
        super(baseBACEModel, self).__init__()
        torch.manual_seed(95)
        
        # Initialise the Base Model Architecture
        self.in_conv = GCNConv(feature_size, embedding_size)
        
        # Graph Transformer Layer
        self.conv1 = TransformerConv(embedding_size, embedding_size, heads=2, dropout=0.2, beta=True)
        # Transformer Conv layer returns 'num_heads*embedding_size' Tensor -> condense to Tensor of dim 'embedding_size'
        self.lin1 = torch.nn.Linear(embedding_size*2, embedding_size)
        
        # Apply Normalization
        self.g_norm = GraphNorm(embedding_size, eps=2e-5)
        
        self.conv2 = GCNConv(embedding_size, embedding_size)
        
        self.pool = TopKPooling(embedding_size, ratio=0.3)
        
        # output layer of the Graph Neural Net
        self.out = torch.nn.Linear(embedding_size*2, 1)
        
    def forward(self, x, edge_index, batch_index):
        
        hidden = self.in_conv(x, edge_index)
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        
        hidden = self.lin1(hidden)
        
        hidden = self.g_norm(hidden)
        hidden = self.conv2(hidden, edge_index)
        
        pool_hidden, edge_index, edge_attr, batch_index, _, _ = self.pool( hidden, edge_index, None, batch_index )
        
        pool_graph_features = torch.cat([gap(pool_hidden, batch_index), gmp(pool_hidden, batch_index)], dim=1)
        
        out = self.out(pool_graph_features)
        # out = F.Sigmoid(out) #Using BCEWithLogitsLoss
        
        return out, pool_graph_features