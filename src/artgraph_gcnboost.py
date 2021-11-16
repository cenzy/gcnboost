import torch
import torch_geometric.transforms as T
import torch.nn.functional as F

from tqdm import tqdm

class ArtGraphGCNBoost:

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

    def __init__(self, model, data, optimizer):
        
        self.data = data
        self.artworks = data['artwork']
        self.y = torch.stack([data['artwork'].y_artist, data['artwork'].y_style, data['artwork'].y_genre])
        self.train_mask = self.artworks.train_mask
        self.val_mask = self.artworks.val_mask
        self.test_mask = self.artworks.test_mask

        self.model = model
        self.optimizer = optimizer

    def get_classes(self, label= 'artist', split='train'):
        pass 

    def get_accuracy(self, predicted, labels):
        return predicted.argmax(dim=1).eq(labels).sum()/predicted.shape[0]

    def get_accuracies(self, predicted, labels, mask):
        accuracies = [] 
        for id, _ in self.map_id2labels.items():
            accuracies.append(self.get_accuracy(predicted[id]['artwork'][mask], labels[id][mask]))
        return accuracies

    def get_loss(self, predicted, labels):
        return F.nll_loss(predicted, labels.type(torch.LongTensor))

    def get_losses(self, predicted, labels, mask):
        losses = []
        for id, _ in self.map_id2labels.items():
            losses.append(self.get_loss(predicted[id]['artwork'][mask], labels[id][mask]))
        return losses

    def multi_task_training(self, epoch):
        self.model.train()

        self.optimizer.zero_grad()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        train_losses = self.get_losses(out, self.y, self.train_mask)
        train_total_loss = 0.6*train_losses[0] + 0.2*train_losses[1] + 0.2*train_losses[2] 
        train_total_loss.backward()
        self.optimizer.step()

        train_accuracies = self.get_accuracies(out, self.y, self.train_mask)

        return out, train_losses, train_accuracies
        
    def test(self, out):
        val_losses = self.get_losses(out, self.y, self.val_mask)
        test_losses = self.get_losses(out, self.y, self.test_mask)

        val_accuracies = self.get_accuracies(out, self.y, self.val_mask)
        test_accuracies = self.get_accuracies(out, self.y, self.test_mask)

        return val_losses, val_accuracies, test_losses, test_accuracies

    def single_task_training(self, epoch, label='author'):
        self.model.train()

        self.optimizer.zero_grad()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        train_losses = self.get_losses(out, self.y, self.train_mask)
        train_total_loss = train_losses[self.map_labels2id[label]]
        train_total_loss.backward()
        self.optimizer.step()

        train_accuracies = self.get_accuracies(out, self.y, self.train_mask)[self.map_labels2id[label]]

        return out, [train_total_loss], [train_accuracies]
    
    def test_single(self, out, label='author'):
        val_losses = self.get_losses(out, self.y, self.val_mask)[self.map_labels2id[label]]
        test_losses = self.get_losses(out, self.y, self.test_mask)[self.map_labels2id[label]]

        val_accuracies = self.get_accuracies(out, self.y, self.val_mask)[self.map_labels2id[label]]
        test_accuracies = self.get_accuracies(out, self.y, self.test_mask)[self.map_labels2id[label]]

        return [val_losses], [val_accuracies], [test_losses], [test_accuracies]