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
K = 100                   # Total clients
C = 0.1                   # Client fraction
N = 50                    # Communication rounds
J = 4                     # Local epochs
lr = 0.01                 # Learning rate
B = 64                    # Batch size
gamma_values = [0.1, 1, 3, 10]
sequence_length = 80      # Character sequence length
embed_dim = 128           # Embedding dimension
lstm_units = 256          # LSTM units
train_json_path = '/workspace/leaf/data/shakespeare/data/train/all_data_train_9.json'
test_json_path = '/workspace/leaf/data/shakespeare/data/test/all_data_test_9.json'

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
            # Filter empty sequences and invalid data
            if len(x) > 0 and len(y) > 0 and y != '':
                records.append({'x': x, 'y': y})
    return pd.DataFrame(records).sample(frac=1).reset_index(drop=True)

# Load and clean data
print("Loading training data...")
train_df = load_json_data(train_json_path)
print("Loading test data...")
test_df = load_json_data(test_json_path)

# Build vocabulary with UNK token
print("Building vocabulary...")
train_chars = set(''.join(train_df['x']) + ''.join(train_df['y']))
chars = ['<UNK>'] + sorted(train_chars)  # Index 0 reserved for unknown
vocab_size = len(chars)
char_to_idx = {c:i for i,c in enumerate(chars)}
print(f"Vocabulary size: {vocab_size}")

# Dataset class with safety checks
class ShakespeareDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Process input sequence
        x_str = self.df.iloc[idx]['x']
        x_str = x_str[:sequence_length].ljust(sequence_length)[:sequence_length]

        # Process target character with fallback
        y_char = self.df.iloc[idx]['y']
        y_char = y_char[0] if len(y_char) > 0 else '<UNK>'

        # Convert to indices with UNK handling
        x_indices = [char_to_idx.get(c, 0) for c in x_str]  # 0 is UNK index
        y_index = char_to_idx.get(y_char, 0)

        return torch.tensor(x_indices), torch.tensor(y_index)

# Create datasets
print("Creating datasets...")
train_dataset = ShakespeareDataset(train_df)
test_dataset = ShakespeareDataset(test_df)

# IID Sharding
print("Sharding data...")
all_indices = list(range(len(train_dataset)))
random.shuffle(all_indices)
client_datasets = [Subset(train_dataset, indices)
                  for indices in np.array_split(all_indices, K)]

# Test loader
test_loader = DataLoader(test_dataset, batch_size=B, shuffle=True)

# =====================
# Model
# =====================
class CharRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_units, num_layers=2, batch_first=True)
        self.fc = nn.Linear(lstm_units, vocab_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =====================
# Client Selection
# =====================
class ClientSelector:
    def __init__(self, config):
        self.K = config['K']
        self.C = config['C']
        self.scheme = config.get('scheme', 'uniform')
        self.num_select = max(1, int(self.C * self.K))

        if self.scheme == 'skewed':
            self.gamma = config['gamma']
            self.probs = np.random.dirichlet([self.gamma]*self.K)
            self.probs /= self.probs.sum()  # Normalize
        else:  # Uniform
            self.probs = np.ones(self.K)/self.K

    def select(self):
        return np.random.choice(self.K, self.num_select, p=self.probs, replace=False).tolist()

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
        self.selector = ClientSelector(config)

    def client_update(self, model, client_idx):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'])
        loader = DataLoader(self.client_datasets[client_idx],
                          batch_size=self.config['batch_size'],
                          shuffle=True)

        for _ in range(self.config['local_epochs']):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = self.criterion(outputs, y)
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

    def test(self):
        self.global_model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(device), y.to(device)
                outputs = self.global_model(x)
                loss = self.criterion(outputs, y)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return 100 * correct / total, test_loss / len(self.test_loader)

    def run(self):
        accuracies, losses = [], []
        for round in range(self.config['rounds']):
            # Client selection
            selected = self.selector.select()

            # Local training
            client_weights = []
            for client in selected:
                model = copy.deepcopy(self.global_model)
                weights = self.client_update(model, client)
                client_weights.append(weights)

            # Aggregation
            global_weights = self.aggregate(client_weights)
            self.global_model.load_state_dict(global_weights)

            # Evaluation
            acc, loss = self.test()
            accuracies.append(acc)
            losses.append(loss)
            print(f"Round {round+1}: Accuracy {acc:.2f}%, Loss {loss:.4f}")

        return accuracies, losses

# =====================
# Experiment
# =====================
if __name__ == "__main__":
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Base configuration
    base_config = {
        'K': K,
        'C': C,
        'rounds': N,  # 50 rounds per scheme
        'local_epochs': J,
        'lr': lr,
        'batch_size': B
    }

    # Run experiments
    results = {}

    # Uniform participation
    print("\n=== Running Uniform Participation ===")
    uniform_config = {**base_config, 'scheme': 'uniform'}
    model = CharRNN().to(device)
    fedavg = FedAvg(uniform_config, model, client_datasets, test_loader)
    results['Uniform'], losses_uniform = fedavg.run()

    # Skewed participation experiments
    losses_skewed = {}
    for gamma in gamma_values:
        print(f"\n=== Running Skewed Participation (\u03b3={gamma}) ===")
        skewed_config = {**base_config, 'scheme': 'skewed', 'gamma': gamma}
        model = CharRNN().to(device)
        fedavg = FedAvg(skewed_config, model, client_datasets, test_loader)
        results[f'\u03b3={gamma}'], losses_skewed[gamma] = fedavg.run()

    # Plot accuracy results
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    for (label, accs), color in zip(results.items(), colors):
        rounds = np.arange(1, len(accs)+1)
        plt.plot(rounds, accs, marker='o', linestyle='-', color=color, label=label)

    plt.title("Federated Learning Client Participation Comparison (Accuracy)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save accuracy plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'fl_participation_accuracy_{timestamp}.png', dpi=300)
    plt.show()

    # Plot loss results
    plt.figure(figsize=(12, 6))
    for gamma, loss in losses_skewed.items():
        rounds = np.arange(1, len(loss)+1)
        plt.plot(rounds, loss, marker='o', linestyle='-', label=f'\u03b3={gamma}')

    rounds = np.arange(1, len(losses_uniform)+1)
    plt.plot(rounds, losses_uniform, marker='o', linestyle='-', label='Uniform')

    plt.title("Federated Learning Client Participation Comparison (Loss)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save loss plot
    plt.savefig(f'fl_participation_loss_{timestamp}.png', dpi=300)
    plt.show()

    # Print final results
    print("\nFinal Results:")
    for scheme, accs in results.items():
        print(f"{scheme}: Accuracy {accs[-1]:.2f}%")
