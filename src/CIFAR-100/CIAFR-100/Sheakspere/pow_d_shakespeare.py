import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import json
import random
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from datetime import datetime

# =====================
# Parameters
# =====================
K = 100  # Total clients
C = 0.1  # Client fraction
N = 200  # Fixed communication rounds
J = 16  # Fixed local epochs
lr = 0.01  # Learning rate
B = 64  # Batch size
d_value = 3  # Parameter for DynamicPowDClientSelector
skew_gamma = 5  # Parameter for Skewed client selection

test_freq = 10

# Paths to IID dataset
train_json_path = '/content/leaf/data/shakespeare/data/train/all_data_niid_0_keep_0_train_9.json'
test_json_path = '/content/leaf/data/shakespeare/data/test/all_data_niid_0_keep_0_test_9.json'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Data Loading & Preprocessing
# =====================
def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    records = []
    for user, info in data['user_data'].items():
        for x, y in zip(info['x'], info['y']):
            if len(x) > 0 and len(y) > 0 and y != '':
                records.append({'x': x, 'y': y})
    return pd.DataFrame(records).sample(frac=1).reset_index(drop=True)

print("Loading training data...")
train_df = load_json_data(train_json_path)
print("Loading test data...")
test_df = load_json_data(test_json_path)

# Build vocabulary
print("Building vocabulary...")
train_chars = set(''.join(train_df['x']) + ''.join(train_df['y']))
chars = ['<UNK>'] + sorted(train_chars)
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
print(f"Vocabulary size: {vocab_size}")

class ShakespeareDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_str = self.df.iloc[idx]['x'][:80].ljust(80)
        y_char = self.df.iloc[idx]['y'][0] if len(self.df.iloc[idx]['y']) > 0 else '<UNK>'
        x_indices = [char_to_idx.get(c, 0) for c in x_str]
        y_index = char_to_idx.get(y_char, 0)
        return torch.tensor(x_indices), torch.tensor(y_index)

print("Creating datasets...")
train_dataset = ShakespeareDataset(train_df)
test_dataset = ShakespeareDataset(test_df)

def iid_shard_data(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [Subset(dataset, indices[i::K]) for i in range(K)]

test_loader = DataLoader(test_dataset, batch_size=B, shuffle=True)

class FedAvg:
    def __init__(self, config, model, client_datasets, test_loader, selector):
        self.config = config
        self.global_model = model.to(device)
        self.client_datasets = client_datasets
        self.test_loader = test_loader
        self.selector = selector
        self.criterion = nn.CrossEntropyLoss()

    def client_update(self, model, client_idx):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'])
        loader = DataLoader(self.client_datasets[client_idx], batch_size=self.config['batch_size'], shuffle=True)

        for _ in range(self.config['local_epochs']):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = self.criterion(model(x), y)
                loss.backward()
                optimizer.step()
        return model.state_dict()

    def aggregate(self, client_weights):
        global_weights = copy.deepcopy(client_weights[0])
        for key in global_weights:
            for w in client_weights[1:]:
                global_weights[key] += w[key]
            global_weights[key] = torch.div(global_weights[key], len(client_weights))
        return global_weights

    def run(self):
        for round in range(self.config['rounds']):
            selected_clients = self.selector.select()
            client_weights = []
            for client in selected_clients:
                model = copy.deepcopy(self.global_model)
                weights = self.client_update(model, client)
                client_weights.append(weights)

            global_weights = self.aggregate(client_weights)
            self.global_model.load_state_dict(global_weights)

            if round % 10 == 0:
                print(f"Round {round}/{self.config['rounds']}")

if __name__ == "__main__":
    client_datasets = iid_shard_data(train_dataset)
    config = {'K': K, 'C': C, 'rounds': N, 'local_epochs': J, 'lr': lr, 'batch_size': B}

    print("\nRunning Uniform Client Selection")
    selector_uniform = ClientSelector({**config, 'scheme': 'uniform'})
    model = CharRNN().to(device)
    fedavg = FedAvg(config, model, client_datasets, test_loader, selector_uniform)
    fedavg.run()

    print("\nRunning Skewed Client Selection")
    selector_skewed = ClientSelector({**config, 'scheme': 'skewed', 'gamma': skew_gamma})
    model = CharRNN().to(device)
    fedavg = FedAvg(config, model, client_datasets, test_loader, selector_skewed)
    fedavg.run()

    print("\nRunning Dynamic PoW-d Client Selection")
    selector_pow_d = DynamicPowDClientSelector({'K': K, 'C': C, 'd': d_value})
    model = CharRNN().to(device)
    fedavg = FedAvg(config, model, client_datasets, test_loader, selector_pow_d)
    fedavg.run()
