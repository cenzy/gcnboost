from tqdm import tqdm
import argparse

import torch
import torch_geometric.transforms as T
import torch_geometric.nn

from data.artgraph import ArtGraph
from models.gcnboost import MGNN
from artgraph_gcnboost import ArtGraphGCNBoost

torch.manual_seed(1)
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=1, help='Number of layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--operator', type=str, default='SAGEConv', )
args = parser.parse_args()

operator_registry = {
    'SAGEConv': torch_geometric.nn.SAGEConv,
    'GraphConv': torch_geometric.nn.GraphConv
}

assert args.operator in ['SAGEConv', 'GraphConv']

base_data = ArtGraph("../data", preprocess='node2vec', transform=T.ToUndirected(), features=True, type='ekg')
data = base_data[0]

model = MGNN(operator=operator_registry[args.operator], hidden_channels=args.hidden, out_channels=base_data.num_classes, metadata=data.metadata(),
            n_layers=args.nlayers, dropout=args.dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

gcn = ArtGraphGCNBoost(model, data, optimizer)
out = gcn.train(args.epochs)
torch.save(out, f'../models/out{args.operator}.pt')