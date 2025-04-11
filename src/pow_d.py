import copy
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
from tqdm import tqdm


class DynamicPowDClientSelector:
    def __init__(self, config):
        self.num_clients = config['K']
        self.sample_size = max(int(config['C'] * self.num_clients), 1)
        self.d = config.get('d', 3)
        self.client_indices = np.arange(self.num_clients)
        # 'qualities' will hold the local losses for each client
        self.qualities = np.zeros(self.num_clients, dtype=np.float32)

    def select(self):

        chosen_clients = []
        available = set(self.client_indices)

        for _ in range(self.sample_size):
            if not available:
                break
            # Sample d candidates (or fewer if available < d) from the remaining clients
            if len(available) <= self.d:
                d_candidates = list(available)
            else:
                d_candidates = np.random.choice(list(available), self.d, replace=False)

            # Pick the client with the largest 'quality'
            best_client = max(d_candidates, key=lambda x: self.qualities[x])
            chosen_clients.append(best_client)
            available.remove(best_client)

        return chosen_clients

    def update_quality(self, client_idx, local_loss):
        """
        Update the 'quality' based on local_loss.
        In this example, bigger local_loss => higher 'quality'.
        """
        self.qualities[client_idx] = local_loss


class EnhancedLeNet(nn.Module):
    def __init__(self):
        super(EnhancedLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 384)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(384, 192)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(192, 100)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class FedAvg:
    def __init__(self, config, model, client_datasets, test_loader, selector, device='cpu'):

        self.config = config
        self.global_model = model.to(device)
        self.client_datasets = client_datasets
        self.test_loader = test_loader
        self.selector = selector
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # Store test accuracy and loss for plotting
        self.test_accuracies = []
        self.test_losses = []

    def client_update(self, client_model, lr, train_loader):

        optimizer = optim.SGD(
            client_model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )

        client_model.train()
        for _ in range(self.config['local_epochs']):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

        return client_model.state_dict()

    def evaluate_local_loss(self, model, data_loader):

        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                loss = self.criterion(outputs, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
        return total_loss / total_samples if total_samples > 0 else 0.0

    def aggregate(self, client_weights):

        # Start with a copy of the first client's weights
        global_weights = copy.deepcopy(client_weights[0])
        num_clients = len(client_weights)

        # For each parameter key, sum up across all clients, then divide
        for key in global_weights.keys():
            for i in range(1, num_clients):
                global_weights[key] += client_weights[i][key]
            global_weights[key] = global_weights[key] / num_clients

        return global_weights

    def test_global_model(self):

        self.global_model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                loss = self.criterion(outputs, target)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        return accuracy, avg_loss

    def cosine_annealing_lr(self, base_lr, current_round, total_rounds):

        return base_lr * 0.5 * (1 + math.cos(math.pi * current_round / total_rounds))

    def run(self):

        rounds_list = []
        rounds = self.config['rounds']
        test_freq = self.config['test_freq']

        for rnd in tqdm(range(rounds), desc="FedAvg Training"):
            current_lr = self.cosine_annealing_lr(
                base_lr=self.config['lr'],
                current_round=rnd,
                total_rounds=rounds
            )

            # (A) Select clients
            selected_clients = self.selector.select()
            client_weights = []

            # (B) If using PoW-d, update the 'qualities' using pre-training losses
            for client_idx in selected_clients:
                train_loader = DataLoader(
                    self.client_datasets[client_idx],
                    batch_size=self.config['batch_size'],
                    shuffle=True
                )
                if hasattr(self.selector, "update_quality"):
                    pre_loss = self.evaluate_local_loss(self.global_model, train_loader)
                    self.selector.update_quality(client_idx, pre_loss)

            # (C) Local training on each selected client
            for client_idx in selected_clients:
                train_loader = DataLoader(
                    self.client_datasets[client_idx],
                    batch_size=self.config['batch_size'],
                    shuffle=True
                )
                local_model = copy.deepcopy(self.global_model).to(self.device)
                updated_weights = self.client_update(local_model, current_lr, train_loader)
                client_weights.append(updated_weights)

            # (D) Aggregate weights and update the global model
            if client_weights:
                global_weights = self.aggregate(client_weights)
                self.global_model.load_state_dict(global_weights)

            # (E) Periodically test the global model
            if (rnd + 1) % test_freq == 0 or rnd == 0:
                acc, loss = self.test_global_model()
                self.test_accuracies.append(acc)
                self.test_losses.append(loss)
                rounds_list.append(rnd + 1)
                print(f"Round {rnd + 1} --> Test Accuracy: {acc:.2f}%, Test Loss: {loss:.4f}")

        return rounds_list, self.test_accuracies


def main():
    # ---------------------------
    # Hyperparameters
    # ---------------------------
    K = 100          # number of clients
    C = 0.1          # fraction of clients per round
    N = 500           # total rounds to train
    J = 4            # local epochs per client
    lr = 0.01
    B = 64
    test_freq = 10
    d_value = 3      # for DynamicPowDClientSelector
    skew_gamma = 5

    # Ensure reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CIFAR-100
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
    ])

    trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=B, shuffle=False, num_workers=2)


    shard_type = 'non_iid'
    n_labels = 20
    sharder = DatasetSharder(dataset=trainset, K=K, shard_type=shard_type, n_labels=n_labels)
    client_datasets = sharder.shard_dataset()

    base_config = {
        'rounds': N,
        'local_epochs': J,
        'lr': lr,
        'batch_size': B,
        'test_freq': test_freq
    }

    results = {}


    print("\n--- FedAvg: Uniform Selection ---")
    # from client_selector import ClientSelector
    config_uniform = {
        'K': K,
        'C': C,
        'participation': 'uniform'
    }
    selector_uniform = ClientSelector(config_uniform)
    model = EnhancedLeNet()
    fedavg_uniform = FedAvg(base_config, model, client_datasets, test_loader, selector_uniform, device=device)
    uniform_rounds, uniform_accs = fedavg_uniform.run()
    uniform_losses = fedavg_uniform.test_losses
    results["Uniform"] = (uniform_rounds, uniform_accs, uniform_losses)


    print(f"\n--- FedAvg: Skewed (gamma={skew_gamma}) ---")
    config_skewed = {
        'K': K,
        'C': C,
        'participation': 'skewed',
        'gamma': skew_gamma
    }
    selector_skewed = ClientSelector(config_skewed)
    model = EnhancedLeNet()
    fedavg_skewed = FedAvg(base_config, model, client_datasets, test_loader, selector_skewed, device=device)
    skewed_rounds, skewed_accs = fedavg_skewed.run()
    skewed_losses = fedavg_skewed.test_losses
    results[f"Skewed (γ={skew_gamma})"] = (skewed_rounds, skewed_accs, skewed_losses)


    print(f"\n--- FedAvg: Dynamic PoW-d (d={d_value}) ---")
    config_pow_d = {
        'K': K,
        'C': C,
        'd': d_value
    }
    selector_pow_d = DynamicPowDClientSelector(config_pow_d)
    model = EnhancedLeNet()
    fedavg_pow_d = FedAvg(base_config, model, client_datasets, test_loader, selector_pow_d, device=device)
    powd_rounds, powd_accs = fedavg_pow_d.run()
    powd_losses = fedavg_pow_d.test_losses
    results[f"PoW-d(d={d_value})"] = (powd_rounds, powd_accs, powd_losses)


    plt.figure(figsize=(8, 5))
    for label, (rnds, accs, losses) in results.items():
        plt.plot(rnds, accs, label=label)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    plt.title("FedAvg - Test Accuracy Comparison")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"comparison_accuracy_{timestamp}.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    for label, (rnds, accs, losses) in results.items():
        plt.plot(rnds, losses, label=label)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Loss")
    plt.title("FedAvg - Test Loss Comparison")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"comparison_loss_{timestamp}.png", dpi=300)
    plt.show()

    # Final results
    print("\n=== Final Results ===")
    for method, (rnds, accs, losses) in results.items():
        final_round = rnds[-1] if rnds else 0
        final_acc = accs[-1] if accs else 0.0
        final_loss = losses[-1] if losses else 0.0
        print(f"{method}: round {final_round}, accuracy={final_acc:.2f}%, loss={final_loss:.4f}")


# Entry point
if __name__ == "__main__":
    main()
