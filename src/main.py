from tqdm import tqdm
import argparse
import os

import torch
import torch_geometric.transforms as T
import torch_geometric.nn
import mlflow

from data.artgraph import ArtGraph
from models.gcnboost import MGNN
from artgraph_gcnboost import ArtGraphGCNBoost

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=1, help='Number of layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--operator', type=str, default='SAGEConv', help='The graph convolutional operator.')
parser.add_argument('--aggr', type=str, default='sum', help='Aggregation function.')
args = parser.parse_args()

operator_registry = {
    'SAGEConv': torch_geometric.nn.SAGEConv,
    'GraphConv': torch_geometric.nn.GraphConv
}

assert args.operator in ['SAGEConv', 'GraphConv']

#Load the EKG
base_data = ArtGraph("../data", preprocess='node2vec', transform=T.ToUndirected(), features=True, type='ekg')
data = base_data[0]

#Build the GCNBoost model
model = MGNN(operator=operator_registry[args.operator], aggr=args.aggr, hidden_channels=args.hidden, out_channels=base_data.num_classes, metadata=data.metadata(),
            n_layers=args.nlayers, dropout=args.dropout)

#Set the GCNBoost system (ekg + model) 
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
gcn = ArtGraphGCNBoost(model, data, optimizer)

mlruns_path = 'file:///home/jbananafish/Desktop/Master/Thesis/code/gcnboost/src/mlruns'
mlflow.set_tracking_uri(mlruns_path)
#mlflow.set_experiment()
with mlflow.start_run() as run:
    print(run.info.run_id)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('learning_rate', args.lr)
    mlflow.log_param('hidden_units', args.hidden)
    mlflow.log_param('n_layers', args.nlayers)
    mlflow.log_param('dropout', args.dropout)
    mlflow.log_param('operator', args.operator)
    mlflow.log_param('aggr', args.aggr)
    for epoch in tqdm(range(0, args.epochs)):
        out, train_losses, train_accuracies = gcn.multi_task_training(epoch)
        val_losses, val_accuracies, test_losses, test_accuracies = gcn.test(out)

        for i, train_loss_acc in enumerate(zip(train_losses, train_accuracies)):
            mlflow.log_metric(f'{gcn.map_labels[i]}_train_loss', round(train_loss_acc[0].detach().item(), 4), step=epoch)
            mlflow.log_metric(f'{gcn.map_labels[i]}_train_accuracy', round(train_loss_acc[1].item(), 2) * 100, step=epoch)
       
        for i, val_loss_acc in enumerate(zip(val_losses, val_accuracies)):
            mlflow.log_metric(f'{gcn.map_labels[i]}_val_loss', round(val_loss_acc[0].detach().item(), 4) ,step=epoch)
            mlflow.log_metric(f'{gcn.map_labels[i]}_val_accuracy', round(val_loss_acc[1].item(), 2) * 100, step=epoch)

        for i, test_loss_acc in enumerate(zip(test_losses, test_accuracies)):
            mlflow.log_metric(f'{gcn.map_labels[i]}_test_loss', round(test_loss_acc[0].detach().item(), 4), step=epoch)
            mlflow.log_metric(f'{gcn.map_labels[i]}_test_accuracy', round(test_loss_acc[1].item(), 2) * 100, step=epoch)

    model_path = '../models'
    model_name = f'out-{args.operator}-{args.nlayers}-{args.hidden}-{args.lr}.pt'
    model_path_name = os.path.join(model_path, model_name)
    torch.save(out, model_path_name)
    mlflow.log_artifact(model_path_name)