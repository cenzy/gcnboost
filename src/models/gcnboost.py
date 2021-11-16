import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GCN, HeteroConv, to_hetero

class GNN(torch.nn.Module):
    def __init__(self, operator=SAGEConv, hidden_channels=16, out_channels=300, num_layers=1, dropout=0.5):
        super(GNN, self).__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = operator((-1, -1), hidden_channels)
            self.convs.append(conv)
        self.conv_out = operator((-1, -1), out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, self.dropout)
        x = self.conv_out(x, edge_index)
        return F.log_softmax(x, dim=1)

class SGNN(torch.nn.Module):
    def __init__(self, operator, aggr, hidden_channels, out_channels, metadata, n_layers, dropout):
        super(MGNN, self).__init__()
        self.gnn = GNN(operator, hidden_channels, out_channels, n_layers, dropout)
        self.gnn = to_hetero(self.gnn_artist, metadata, aggr=aggr)

    def forward(self, x, edge_index):
        return self.gnn(x, edge_index)

class MGNN(torch.nn.Module):
    def __init__(self, operator, aggr, hidden_channels, out_channels, metadata, n_layers, dropout):
        super(MGNN, self).__init__()
        self.gnn_artist = GNN(operator, hidden_channels, out_channels['artist'], n_layers, dropout)
        self.gnn_artist = to_hetero(self.gnn_artist, metadata, aggr=aggr)

        self.gnn_style = GNN(operator, hidden_channels, out_channels['style'], n_layers, dropout)
        self.gnn_style = to_hetero(self.gnn_style, metadata, aggr=aggr)

        self.gnn_genre = GNN(operator, hidden_channels, out_channels['genre'], n_layers, dropout)
        self.gnn_genre = to_hetero(self.gnn_genre, metadata,aggr=aggr)

    def forward(self, x, edge_index):
        return [self.gnn_artist(x, edge_index), self.gnn_style(x, edge_index), self.gnn_genre(x, edge_index)]