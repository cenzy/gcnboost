import torch
import torch_geometric.transforms as T
import torch.nn.functional as F

from tqdm import tqdm

class ArtGraphGCNBoost:

    map_labels = {
        0: 'artist',
        1: 'style',
        2: 'genre'
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
        for id, _ in self.map_labels.items():
            accuracies.append(self.get_accuracy(predicted[id]['artwork'][mask], labels[id][mask]))
        return accuracies

    def get_loss(self, predicted, labels):
        return F.nll_loss(predicted, labels.type(torch.LongTensor))

    def get_losses(self, predicted, labels, mask):
        losses = []
        for id, _ in self.map_labels.items():
            losses.append(self.get_loss(predicted[id]['artwork'][mask], labels[id][mask]))
        return losses

    def train(self, epochs):
        for epoch in tqdm(range(0, epochs)):
            out = self.multi_task_training(epoch)
        return out

    def multi_task_training(self, epoch):
        self.model.train()

        self.optimizer.zero_grad()
        out = self.model(self.data.x_dict, self.data.edge_index_dict)
        train_losses = self.get_losses(out, self.y, self.train_mask)
        train_total_loss = sum(train_losses)
        train_total_loss.backward()
        self.optimizer.step()

        train_accuracies = self.get_accuracies(out, self.y, self.train_mask)

        print(f'Epoch: {epoch+1}')
        for i, train_loss_acc in enumerate(zip(train_losses, train_accuracies)):
            print(f'\t {self.map_labels[i]} \t {round(train_loss_acc[0].detach().item(), 4)} \t{round(train_loss_acc[1].item(), 2) * 100}%')

        if epoch % 5 == 0:
            self.test(out)

        return out

    def test(self, out):
        val_losses = self.get_losses(out, self.y, self.val_mask)
        test_losses = self.get_losses(out, self.y, self.test_mask)

        val_accuracies = self.get_accuracies(out, self.y, self.val_mask)
        test_accuracies = self.get_accuracies(out, self.y, self.test_mask)

        print(f'*\tOn validation')
        for i, val_loss_acc in enumerate(zip(val_losses, val_accuracies)):
            print(f'\t{self.map_labels[i]}\t {round(val_loss_acc[0].detach().item(), 4)} \t {round(val_loss_acc[1].item(), 2) * 100}%')

        print(f'*\tOn test')
        for i, test_loss_acc in enumerate(zip(test_losses, test_accuracies)):
            print(f'\t{self.map_labels[i]}\t {round(test_loss_acc[0].detach().item(), 4)} \t {round(test_loss_acc[1].item(), 2) * 100}%')