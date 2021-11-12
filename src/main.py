import torch
import torch_geometric.transforms as T

from data.artgraph import ArtGraph
from models.gcnboost import MGNN
from artgraph_gcnboost import ArtGraphGCNBoost
from tqdm import tqdm

import argparse

torch.manual_seed(1)
torch.cuda.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=2, help='Number of layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

base_data = ArtGraph("../data", preprocess='node2vec', transform=T.ToUndirected(), features=True, type='ekg')
data = base_data[0]
model = MGNN( hidden_channels=args.hidden, out_channels=base_data.num_classes, metadata=data.metadata())
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#gcn = ArtGraphGCNBoost(model, data, optimizer)
#out = gcn.train(args.epochs)
#torch.save(out, "out.pt")