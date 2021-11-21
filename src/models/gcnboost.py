import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, Linear, to_hetero

class HomoGNN(torch.nn.Module):
    def __init__(self, operator=GCNConv, input_channels=128, hidden_channels=16, out_channels=300, num_layers=1, dropout=0.5, skip=False):
        super(HomoGNN, self).__init__()
        self.dropout = dropout
        self.skip = skip
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        #self.convs.append(operator(input_channels, hidden_channels))
        for _ in range(num_layers):
            conv = operator(-1, hidden_channels)
            lin = Linear(-1, hidden_channels)
            self.convs.append(conv)
            self.lins.append(lin)
        self.conv_out = operator(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            if self.skip:
                x = conv(x, edge_index).relu() + self.lins[i](x)
            else:
                x = conv(x, edge_index).relu()
            x = F.dropout(x, self.dropout)
        x = self.conv_out(x, edge_index)
        return F.log_softmax(x, dim=1)

class HomoSGNN(torch.nn.Module):
    def __init__(self, operator, input_channels, hidden_channels, out_channels, n_layers, dropout, skip):
        super(HomoSGNN, self).__init__()
        self.gnn = HomoGNN(operator, input_channels, hidden_channels, out_channels, n_layers, dropout, skip)

    def forward(self, x, edge_index):
        return [self.gnn(x, edge_index)]

class HomoMGNN(torch.nn.Module):
    def __init__(self, operator, input_channels, hidden_channels, out_channels, n_layers, dropout, skip):
        super(HomoMGNN, self).__init__()
        self.gnn_artist = HomoGNN(operator, input_channels, hidden_channels, out_channels['artist'], n_layers, dropout, skip)
        self.gnn_style = HomoGNN(operator, input_channels, hidden_channels, out_channels['style'], n_layers, dropout, skip)
        self.gnn_genre = HomoGNN(operator, input_channels, hidden_channels, out_channels['genre'], n_layers, dropout, skip)

    def forward(self, x, edge_index):
        return [self.gnn_artist(x, edge_index), self.gnn_style(x, edge_index), self.gnn_genre(x, edge_index)]

class HeteroGNN(torch.nn.Module):
    def __init__(self, operator=SAGEConv, hidden_channels=16, out_channels=300, num_layers=1, dropout=0.5, skip=False):
        super(HeteroGNN, self).__init__()
        self.dropout = dropout
        self.skip = skip
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = operator((-1, -1), hidden_channels)
            lin = Linear(-1, hidden_channels)
            self.convs.append(conv)
            self.lins.append(lin)
        self.conv_out = operator((-1, -1), out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            if self.skip:
                x = conv(x, edge_index).relu() + self.lins[i](x)
            else:
                x = conv(x, edge_index).relu()
            x = F.dropout(x, self.dropout)
        x = self.conv_out(x, edge_index)
        return F.log_softmax(x, dim=1)

class HeteroSGNN(torch.nn.Module):
    def __init__(self, operator, aggr, hidden_channels, out_channels, metadata, n_layers, dropout, skip):
        super(HeteroSGNN, self).__init__()
        self.gnn = HeteroGNN(operator, hidden_channels, out_channels, n_layers, dropout, skip)
        self.gnn = to_hetero(self.gnn, metadata, aggr=aggr)

    def forward(self, x, edge_index):
        return [self.gnn(x, edge_index)]

class HeteroMGNN(torch.nn.Module):
    def __init__(self, operator, aggr, hidden_channels, out_channels, metadata, n_layers, dropout, skip):
        super(HeteroMGNN, self).__init__()
        self.gnn_artist = HeteroGNN(operator, hidden_channels, out_channels['artist'], n_layers, dropout, skip)
        self.gnn_artist = to_hetero(self.gnn_artist, metadata, aggr=aggr)

        self.gnn_style = HeteroGNN(operator, hidden_channels, out_channels['style'], n_layers, dropout, skip)
        self.gnn_style = to_hetero(self.gnn_style, metadata, aggr=aggr)

        self.gnn_genre = HeteroGNN(operator, hidden_channels, out_channels['genre'], n_layers, dropout, skip)
        self.gnn_genre = to_hetero(self.gnn_genre, metadata,aggr=aggr)

    def forward(self, x, edge_index):
        return [self.gnn_artist(x, edge_index), self.gnn_style(x, edge_index), self.gnn_genre(x, edge_index)]