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
N_base = 200  # Base communication rounds (adjusted per J)
J_values = [4, 8, 16]  # Local epochs
lr = 0.01  # Learning rate
B = 64  # Batch size
Nc_values = [1, 5, 10, 50]  # Number of labels per client (Non-IID levels)

# Paths to Non-IID datasets
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

# IID Sharding
def iid_shard_data(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return [Subset(dataset, indices[i::K]) for i in range(K)]

# Non-IID Sharding
def shard_data(dataset, Nc):
    label_to_indices = {}
    for idx in range(len(dataset)):
        label = dataset.df.iloc[idx]['y'][0]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    all_labels = list(label_to_indices.keys())
    random.shuffle(all_labels)
    client_data = [[] for _ in range(K)]

    for i in range(K):
        chosen_labels = random.sample(all_labels, Nc)
        for label in chosen_labels:
            client_data[i].extend(random.sample(label_to_indices[label], min(len(label_to_indices[label]), len(dataset) // (K * Nc))))

    return [Subset(dataset, indices) for indices in client_data]

# Test loader
test_loader = DataLoader(test_dataset, batch_size=B, shuffle=True)

# =====================
# Model
# =====================
class CharRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =====================
# Federated Learning
# =====================
class FedAvg:
    def __init__(self, config, model, client_datasets, test_loader):
        self.config = config
        self.global_model = model.to(device)
        self.client_datasets = client_datasets
        self.test_loader = test_loader
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

    def run(self):
        for round in range(self.config['rounds']):
            selected_clients = random.sample(range(K), max(1, int(C * K)))
            client_weights = [self.client_update(copy.deepcopy(self.global_model), c) for c in selected_clients]
            global_weights = {k: sum(w[k] for w in client_weights) / len(client_weights) for k in client_weights[0]}
            self.global_model.load_state_dict(global_weights)

            if round % 10 == 0:
                print(f"Round {round}/{self.config['rounds']}")

# =====================
# Experiment
# =====================
if __name__ == "__main__":
    for Nc in Nc_values:
        client_datasets = iid_shard_data(train_dataset) if Nc == 50 else shard_data(train_dataset, Nc)
        print(f"Running Experiment for Nc={Nc}")

        for J in J_values:
            rounds_scaled = N_base // (J // 4)  # Scale rounds based on J
            config = {'K': K, 'C': C, 'rounds': rounds_scaled, 'local_epochs': J, 'lr': lr, 'batch_size': B}
            print(f"\nRunning FedAvg for Nc={Nc}, J={J}, rounds={rounds_scaled}")
            model = CharRNN().to(device)
            fedavg = FedAvg(config, model, client_datasets, test_loader)
            fedavg.run()
