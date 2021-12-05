from tqdm import tqdm
import argparse
import os

import torch
import torch_geometric.transforms as T
import torch_geometric.nn
import mlflow

from artgraph_gcnboost import ArtGraphGCNBoost

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='test', help='Experiment name.')
parser.add_argument('--type', type=str, default='hetero', help='Graph type (hetero|homo).')
parser.add_argument('--mode', type=str, default='multi_task', help='Training mode (multi_task|single_task).')
parser.add_argument('--label', type=str, default='all', help='Label to predict (artist|style|genre).')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nlayers', type=int, default=1, help='Number of layers.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--operator', type=str, default='SAGEConv', help='The graph convolutional operator.')
parser.add_argument('--activation', type=str, default='relu', help='The activation function.')
parser.add_argument('--aggr', type=str, default='sum', help='Aggregation function.')
parser.add_argument('--skip', action='store_true', default='False', help='Add skip connection.')
parser.add_argument('--bn', action='store_true', default='False', help='Add batch normalization.')
parser.add_argument('--wd', type=float, default=3e-4, help='Weight decay.')
args = parser.parse_args()

gcn = ArtGraphGCNBoost(args, graph_type=args.type, training_mode=args.mode)

mlruns_path = 'file://' +  os.path.abspath(os.getcwd()) +  '/mlruns'
#if not os.path.exists(mlruns_path):
#    os.makedirs(mlruns_path)
mlflow.set_tracking_uri(mlruns_path)
mlflow.set_experiment(args.exp)
with mlflow.start_run() as run:
    mlflow.log_param('type', args.type)
    mlflow.log_param('mode', args.mode)
    mlflow.log_param('label', args.label)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('learning_rate', args.lr)
    mlflow.log_param('hidden_units', args.hidden)
    mlflow.log_param('n_layers', args.nlayers)
    mlflow.log_param('dropout', args.dropout)
    mlflow.log_param('operator', args.operator)
    mlflow.log_param('aggr', args.aggr)
    mlflow.log_param('bn', args.bn)
    mlflow.log_param('skip', args.skip)
    mlflow.log_param('activation_function', args.activation)
    mlflow.log_param('weight_decay', args.wd)
    
    for epoch in tqdm(range(0, args.epochs)):
        if args.type == 'hetero':
            out, train_losses, train_accuracies = gcn.hetero_training()
            val_losses, val_accuracies, test_losses, test_accuracies = gcn.hetero_test()
            if args.mode == 'multi_task':
                for i, train_loss_acc in enumerate(zip(train_losses, train_accuracies)):
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_train_loss', round(train_loss_acc[0].detach().item(), 4), step=epoch)
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_train_accuracy', round(train_loss_acc[1].item(), 2) * 100, step=epoch)
                for i, val_loss_acc in enumerate(zip(val_losses, val_accuracies)):
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_val_loss', round(val_loss_acc[0].detach().item(), 4) ,step=epoch)
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_val_accuracy', round(val_loss_acc[1].item(), 2) * 100, step=epoch)
                for i, test_loss_acc in enumerate(zip(test_losses, test_accuracies)):
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_test_loss', round(test_loss_acc[0].detach().item(), 4), step=epoch)
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_test_accuracy', round(test_loss_acc[1].item(), 2) * 100, step=epoch)
            if args.mode == 'single_task':
                mlflow.log_metric(f'{args.label}_train_loss', round(train_losses[0].detach().item(), 4), step=epoch)
                mlflow.log_metric(f'{args.label}_train_accuracy', round(train_accuracies[0].item(), 2) * 100, step=epoch)
                mlflow.log_metric(f'{args.label}_val_loss', round(val_losses[0].detach().item(), 4), step=epoch)
                mlflow.log_metric(f'{args.label}_val_accuracy', round(val_accuracies[0].item(), 2) * 100, step=epoch)
                mlflow.log_metric(f'{args.label}_test_loss', round(test_losses[0].detach().item(), 4), step=epoch)
                mlflow.log_metric(f'{args.label}_test_accuracy', round(test_accuracies[0].item(), 2) * 100, step=epoch)
        if args.type == 'homo':
            out, train_losses, train_accuracies = gcn.homo_training()
            val_losses, val_accuracies, test_losses, test_accuracies = gcn.homo_test(out)
            if args.mode == 'multi_task':
                for i, train_loss_acc in enumerate(zip(train_losses, train_accuracies)):
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_train_loss', round(train_loss_acc[0].detach().item(), 4), step=epoch)
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_train_accuracy', round(train_loss_acc[1].item(), 2) * 100, step=epoch)
                for i, val_loss_acc in enumerate(zip(val_losses, val_accuracies)):
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_val_loss', round(val_loss_acc[0].detach().item(), 4) ,step=epoch)
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_val_accuracy', round(val_loss_acc[1].item(), 2) * 100, step=epoch)
                for i, test_loss_acc in enumerate(zip(test_losses, test_accuracies)):
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_test_loss', round(test_loss_acc[0].detach().item(), 4), step=epoch)
                    mlflow.log_metric(f'{gcn.map_id2labels[i]}_test_accuracy', round(test_loss_acc[1].item(), 2) * 100, step=epoch)
            if args.mode == 'single_task':
                mlflow.log_metric(f'{args.label}_train_loss', round(train_losses[0].detach().item(), 4), step=epoch)
                mlflow.log_metric(f'{args.label}_train_accuracy', round(train_accuracies[0].item(), 2) * 100, step=epoch)
                mlflow.log_metric(f'{args.label}_val_loss', round(val_losses[0].detach().item(), 4), step=epoch)
                mlflow.log_metric(f'{args.label}_val_accuracy', round(val_accuracies[0].item(), 2) * 100, step=epoch)
                mlflow.log_metric(f'{args.label}_test_loss', round(test_losses[0].detach().item(), 4), step=epoch)
                mlflow.log_metric(f'{args.label}_test_accuracy', round(test_accuracies[0].item(), 2) * 100, step=epoch)

       
    model_path = '../models'
    model_name = f'out-{args.operator}-{args.nlayers}-{args.hidden}-{args.lr}.pt'
    model_path_name = os.path.join(model_path, model_name)
    torch.save(out, model_path_name)
    mlflow.log_artifact(model_path_name)