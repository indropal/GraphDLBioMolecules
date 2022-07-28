import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv, GraphNorm, TransformerConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_add_pool as g_add_p
from torch_geometric.nn import TopKPooling
from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import DataLoader


# Implementing Baseline Graphical Neural network
class baceModel(torch.nn.Module):
    def __init__(self, feature_size, embedding_size):
        super(baceModel, self).__init__()
        torch.manual_seed(95)
        
        # Initialise the Base Model Architecture
        self.in_conv = GCNConv(feature_size, embedding_size)
        
        # Attention Conv layer returns 'num_heads*embedding_size' Tensor -> condense to Tensor of dim 'embedding_size'
        self.conv1 = GATv2Conv(embedding_size, embedding_size, heads=4, dropout=0.1)
        self.lin1 = torch.nn.Linear(embedding_size*4, embedding_size)
        
        # Transformer Conv Layer returns 'num_heads*embedding_size' Tensor -> condense to Tensor of dim 'embedding_size'
        self.conv2 = TransformerConv(embedding_size, embedding_size, heads=2, dropout=0.2, beta=True)
        self.lin2 = torch.nn.Linear(embedding_size*2, embedding_size)
        
        # Apply graph-level normalization
        self.g_norm = GraphNorm(embedding_size, eps=2e-5)
        
        self.pool = TopKPooling(embedding_size, ratio=0.4)
        
        self.conv3 = GATv2Conv(embedding_size, embedding_size, heads=2, dropout=0.1)
        self.lin3 = torch.nn.Linear(embedding_size*2, embedding_size)
        
        self.pool2 = TopKPooling(embedding_size, ratio=0.4)        
        
        # output layer of the Graph Neural Net
        self.out = torch.nn.Linear(embedding_size*3, 1)
        
    def forward(self, x, edge_index, batch_index):
        
        hidden = self.in_conv(x, edge_index)
        
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.lin1(hidden)
        
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.lin2(hidden)

        hidden = self.g_norm(hidden)
        
        pool_hidden, edge_index, edge_attr, batch_index, _, _ = self.pool( hidden, edge_index, None, batch_index )
        
        hidden = self.conv3(pool_hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.lin3(hidden)        
        
        pool_hidden, edge_index, edge_attr, batch_index, _, _ = self.pool2( hidden, edge_index, None, batch_index )
        
        pool_graph_features = torch.cat([gap(pool_hidden, batch_index),
                                         g_add_p(pool_hidden, batch_index),
                                         gmp(pool_hidden, batch_index)
                                        ], dim=1
                                       )
        #pool_graph_features = torch.cat([gap(hidden, batch_index), gmp(hidden, batch_index)], dim=1)
        
        out = self.out(pool_graph_features)
        # out = F.Sigmoid(out) #Using BCEWithLogitsLoss
        
        return out, pool_graph_features