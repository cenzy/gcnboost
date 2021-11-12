import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GCN, HeteroConv, to_hetero

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, 0.5)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class MGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super(MGNN, self).__init__()
        self.gnn_artist = GNN(hidden_channels, out_channels['artist'])
        self.gnn_artist = to_hetero(self.gnn_artist, metadata, aggr='sum')

        self.gnn_style = GNN(hidden_channels, out_channels['style'])
        self.gnn_style = to_hetero(self.gnn_style, metadata, aggr='sum')

        self.gnn_genre = GNN(hidden_channels, out_channels['genre'])
        self.gnn_genre = to_hetero(self.gnn_genre, metadata, aggr='sum')

    def forward(self, x, edge_index):
        return [self.gnn_artist(x, edge_index), self.gnn_style(x, edge_index), self.gnn_genre(x, edge_index)]