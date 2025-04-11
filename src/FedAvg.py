import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy
from tqdm import tqdm
import random



class FedAvg:
    def __init__(self, config, model, client_datasets, test_loader):
        self.config = config
        self.global_model = model
        self.client_datasets = client_datasets
        self.test_loader = test_loader
        self.device = device
        self.global_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.num_clients = config['num_clients']

    def client_update(self, client_model, optimizer, train_loader):
        client_model.train()
        for epoch in range(self.config['local_epochs']):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
        return client_model.state_dict()

    def aggregate(self, client_weights):
        global_weights = copy.deepcopy(client_weights[0])
        for key in global_weights.keys():
            for i in range(1, len(client_weights)):
                global_weights[key] += client_weights[i][key]
            global_weights[key] = torch.div(global_weights[key], len(client_weights))
        return global_weights

    def test(self):
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                loss = self.criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_loss = test_loss / len(self.test_loader)
        return accuracy, avg_loss

    def run(self):
        num_selected = max(1, int(self.config['client_fraction'] * self.num_clients))
        test_losses = []
        test_accuracies = []
        rounds_list = []

        for round_ in tqdm(range(self.config['rounds']), desc="Federated Training Rounds"):
            selected_clients = random.sample(range(self.num_clients), num_selected)
            client_weights = []

            for client_idx in selected_clients:
                local_model = copy.deepcopy(self.global_model)
                local_model.to(self.device)
                optimizer = optim.SGD(local_model.parameters(),
                                    lr=self.config['lr'],
                                    momentum=0.9,
                                    weight_decay=0.0004)

                client_dataset = self.client_datasets[client_idx]
                train_loader = DataLoader(client_dataset,
                                        batch_size=self.config['batch_size'],
                                        shuffle=True)

                local_weights = self.client_update(local_model, optimizer, train_loader)
                client_weights.append(local_weights)

            global_weights = self.aggregate(client_weights)
            self.global_model.load_state_dict(global_weights)

            if (round_ + 1) % self.config['test_freq'] == 0 or round_ == 0:
                accuracy, loss = self.test()
                test_accuracies.append(accuracy)
                test_losses.append(loss)
                rounds_list.append(round_ + 1)
                print(f"\nRound {round_+1}, Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")

        return test_accuracies, test_losses, rounds_list
