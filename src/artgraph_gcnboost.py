import torch
from torch_geometric.loader import DataLoader
import torch_geometric.nn as operators
import torch_geometric.transforms as T
import torch.nn.functional as F
from models.gcnboost import *
from data.artgraph import ArtGraph

from tqdm import tqdm

class ArtGraphGCNBoost:

    operator_registry = {
        'SAGEConv': operators.SAGEConv,
        'GraphConv': operators.GraphConv,
        'GATConv': operators.GATConv,
        'GCNConv': operators.GCNConv
    }

    map_id2labels = {
        0: 'artist',
        1: 'style',
        2: 'genre'
    }

    map_labels2id = {
        'artist': 0,
        'style': 1,
        'genre': 2
    }

    def __init__(self, args, graph_type='hetero', training_mode='multi_task'):
        
        self.graph_type = graph_type
        self.traning_mode = training_mode
        assert graph_type in ['hetero', 'homo']
        assert training_mode in ['multi_task', 'single_task']
        assert args.operator in self.operator_registry.keys()

        self.base_data, self.data, self.y, self.model, self.optimizer = self._bootstrap(args)
        self.artworks = self.base_data[0]['artwork']
        self.train_mask = self.artworks.train_mask
        self.val_mask = self.artworks.val_mask
        self.test_mask = self.artworks.test_mask

    def _bootstrap(self, args):
        base_data = ArtGraph("../data", preprocess='node2vec', transform=T.ToUndirected(), features=True, type='ekg')
        data = base_data[0]
        def del_some_nodes():
            del data['gallery']
            del data['city']
            del data['country']
            del data['auction']
            del data['artwork', 'auction_rel', 'auction']
            del data['auction', 'rev_auction_rel', 'artwork']

            del data['city', 'country_rel', 'country']
            del data['country', 'rev_country_rel', 'city']

            del data['gallery', 'city_rel', 'city']
            del data['city', 'rev_city_rel', 'gallery']
            
            del data['gallery', 'country_rel', 'country']
            del data['country', 'rev_country_rel', 'gallery']
            
            del data['artwork', 'locatedin_rel', 'galley']
            del data['gallery', 'rev_locatedin_rel', 'artwork']

            del data['artwork', 'completedin_rel', 'city']
            del data['city', 'rev_completedin_rel', 'artwork']
            
        if self.graph_type == 'hetero':
            if self.traning_mode == 'multi_task':
                model = HeteroMGNN(operator=self.operator_registry[args.operator], 
                             aggr=args.aggr, 
                             hidden_channels=args.hidden, 
                             out_channels=base_data.num_classes, 
                             metadata=data.metadata(),
                             n_layers=args.nlayers, 
                             dropout=args.dropout,
                             skip=args.skip)
                y = torch.stack([base_data[0]['artwork'].y_artist, base_data[0]['artwork'].y_style, base_data[0]['artwork'].y_genre])
            if self.traning_mode == 'single_task':
                model = HeteroSGNN(operator=self.operator_registry[args.operator], 
                             aggr=args.aggr, 
                             hidden_channels=args.hidden, 
                             out_channels=base_data.num_classes[args.label], 
                             metadata=data.metadata(),
                             n_layers=args.nlayers, 
                             dropout=args.dropout,
                             skip=args.skip)
                y = torch.stack([base_data[0]['artwork'][f'y_{args.label}']])
        
        if self.graph_type == 'homo':
            data = data.to_homogeneous()
            if self.traning_mode == 'multi_task':
                model = HomoMGNN(operator=self.operator_registry[args.operator],
                                 input_channels=base_data.num_features,
                                 hidden_channels=args.hidden,
                                 out_channels=base_data.num_classes,
                                 n_layers=args.nlayers,
                                 dropout=args.dropout,
                                 skip=args.skip)
                y = torch.stack([base_data[0]['artwork'].y_artist, base_data[0]['artwork'].y_style, base_data[0]['artwork'].y_genre])
            if self.traning_mode == 'single_task':
                model = HomoSGNN(operator=self.operator_registry[args.operator],
                                 input_channels=base_data.num_features,
                                 hidden_channels=args.hidden,
                                 out_channels=base_data.num_classes[args.label],
                                 n_layers=args.nlayers,
                                 dropout=args.dropout,
                                 skip=args.skip)
                y = torch.stack([base_data[0]['artwork'][f'y_{args.label}']])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-4)
        
        return base_data, data, y, model, optimizer

    def get_accuracy(self, predicted, labels):
        return predicted.argmax(dim=1).eq(labels).sum()/predicted.shape[0]

    def get_accuracies(self, predicted, labels, mask):
        accuracies = [] 
        for i, _ in enumerate(labels):
            accuracies.append(self.get_accuracy(predicted[i]['artwork'][mask], labels[i][mask]))
        return accuracies

    def get_accuracies_homo(self, predicted, labels, mask):
        size = self.train_mask.shape[0]
        accuracies = [] 
        for i, _ in enumerate(labels):
            accuracies.append(self.get_accuracy(predicted[i][:size][mask], labels[i][mask]))
        return accuracies

    def get_loss(self, predicted, labels):
        return F.nll_loss(predicted, labels.type(torch.LongTensor))

    def get_losses(self, predicted, labels, mask):
        losses = []
        for i, _ in enumerate(labels):
            losses.append(self.get_loss(predicted[i]['artwork'][mask], labels[i][mask]))
        return losses
    
    def get_losses_homo(self, predicted, labels, mask):
        size = self.train_mask.shape[0]
        losses = []
        for i, _ in enumerate(labels):
            losses.append(self.get_loss(predicted[i][:size][mask], labels[i][mask]))
        return losses

    def hetero_training(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)

        train_losses = self.get_losses(out, self.y, self.train_mask)
        train_total_loss = sum(train_losses)

        train_total_loss.backward()
        self.optimizer.step()

        train_accuracies = self.get_accuracies(out, self.y, self.train_mask)

        return out, train_losses, train_accuracies

    def homo_training(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)

        train_losses = self.get_losses_homo(out, self.y, self.train_mask)
        train_total_loss = sum(train_losses)

        train_total_loss.backward()
        self.optimizer.step()

        train_accuracies = self.get_accuracies_homo(out, self.y, self.train_mask)

        return out, train_losses, train_accuracies
        
    def hetero_test(self, out):
        val_losses = self.get_losses(out, self.y, self.val_mask)
        test_losses = self.get_losses(out, self.y, self.test_mask)

        val_accuracies = self.get_accuracies(out, self.y, self.val_mask)
        test_accuracies = self.get_accuracies(out, self.y, self.test_mask)

        return val_losses, val_accuracies, test_losses, test_accuracies

    def homo_test(self, out):
        val_losses = self.get_losses_homo(out, self.y, self.val_mask)
        test_losses = self.get_losses_homo(out, self.y, self.test_mask)

        val_accuracies = self.get_accuracies_homo(out, self.y, self.val_mask)
        test_accuracies = self.get_accuracies_homo(out, self.y, self.test_mask)

        return val_losses, val_accuracies, test_losses, test_accuracies