import torch
from torch import Tensor
from typing import Union
import torch_geometric.nn as gnn
import torch.nn as nn

class GNLayer(gnn.MessagePassing):
    def __init__(self, indim, hiddendim, outdim, edgedim=0):
        super(GNLayer, self).__init__()
        self.in_channels = indim
        self.hidden_channels = hiddendim
        self.out_channels = outdim
        self.edge_dim = edgedim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * indim + edgedim, hiddendim),
            nn.PReLU(),
            nn.Linear(hiddendim, hiddendim),
            nn.PReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(indim + hiddendim, hiddendim),
            nn.PReLU(),
            nn.Linear(hiddendim, outdim),
        )

    def reset_parameters(self):
        super().reset_parameters()
        gnn.inits.reset(self.edge_mlp)
        gnn.inits.reset(self.node_mlp)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        if isinstance(x, tuple):
          x = x[1] if self.flow == "source_to_target" else x[0]
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        return out
    
    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is None:
            out = torch.cat([x_i, x_j], dim=1)
        else:
            out = torch.cat([x_i, x_j, edge_attr], dim=1)
        out = self.edge_mlp(out)
        return out

class GNNBlock(nn.Module):
    def __init__(self, indim, hiddendim, outdim, edgedim=0):
        super(GNNBlock, self).__init__()
        self.gnlayer = GNLayer(indim, hiddendim, outdim, edgedim)
        self.act = nn.PReLU()
        self.norm = nn.BatchNorm1d(outdim)
        
    def forward(self, x, edge_index, edge_attr=None):
        x = self.gnlayer(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.norm(x)
        return x

class Segger(nn.Module):
    def __init__(self, tokens, num_node_features, init_emb=16, hidden_channels=32, out_channels=32):
        super(Segger, self).__init__()        
        if tokens:
            self.node_init = nn.ModuleDict({
                'tx': nn.Embedding(num_node_features['tx'], init_emb),
                'bd': nn.Linear(num_node_features['bd'], init_emb),
            })
        else:
            self.node_init = nn.ModuleDict({
                'tx': nn.Linear(num_node_features['tx'], init_emb),
                'bd': nn.Linear(num_node_features['bd'], init_emb),
            })
        self.hetero_conv1 = gnn.HeteroConv({
            ('tx', 'neighbors', 'tx'): GNNBlock(init_emb, hidden_channels, hidden_channels),
            ('tx', 'belongs', 'bd'): GNNBlock(init_emb, hidden_channels, hidden_channels),
        }, aggr='sum')
        self.hetero_conv2 = gnn.HeteroConv({
            ('tx', 'neighbors', 'tx'): GNNBlock(hidden_channels, hidden_channels, hidden_channels),
            ('tx', 'belongs', 'bd'): GNNBlock(hidden_channels, hidden_channels, hidden_channels),
        }, aggr='sum')
        self.mlp = nn.ModuleDict({
            'tx': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.PReLU(),
                nn.Linear(hidden_channels, out_channels),
            ),
            'bd': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.PReLU(),
                nn.Linear(hidden_channels, out_channels),
            ),
        })
        self.edge_mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(2 * out_channels, out_channels),
            nn.PReLU(),
            nn.Linear(out_channels, 1),
        )

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = {node_type: self.node_init[node_type](x) for node_type, x in x_dict.items()}
        x_dict = self.hetero_conv1(x_dict, edge_index_dict)
        x_dict = self.hetero_conv2(x_dict, edge_index_dict)
        x_dict = {key: self.mlp[key](x) for key, x in x_dict.items()}
        edge_x = torch.cat([
            x_dict['tx'][edge_label_index[0]],
            x_dict['bd'][edge_label_index[1]],
        ], dim=1)
        edge_x = self.edge_mlp(edge_x)
        return x_dict, edge_x.squeeze(-1)
