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

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)
print(NUM_LETTERS)

def _one_hot(index, size): # this function returns one-hot vector with given size and value 1 at given index
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec

def letter_to_vec(letter): # this function returns one-hot representation of given letter
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)

def word_to_indices(word): # this function returns indices of given word
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

# Function to read and parse training and testing data
def read_data(train_data_path, test_data_path):
    def read_file(path):
        with open(path, 'r') as f:
            return json.load(f)

    train_data = read_file(train_data_path)
    test_data = read_file(test_data_path)

    # Extract users and their corresponding data from training and testing datasets
    train_clients = list(train_data['users'])  # List of training users
    train_groups = list(train_data['user_data'].keys())  # Groups for training data
    train_data_temp = train_data['user_data']  # Actual training data

    test_clients = list(test_data['users'])  # List of testing users
    test_groups = list(test_data['user_data'].keys())  # Groups for testing data
    test_data_temp = test_data['user_data']  # Actual testing data

    return train_clients, train_groups, train_data_temp, test_data_temp

# Custom Dataset class for the Shakespeare dataset
class Shakespeare(Dataset):
    def __init__(self, train=True, args=None):
        super(Shakespeare, self).__init__()

        # Read the training and testing data
        train_clients, train_groups, train_data_temp, test_data_temp = read_data(args.shakespeare_train_path, args.shakespeare_test_path)
        self.train = train

        if self.train:
            # For training, organize data into inputs and labels
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i, client in enumerate(train_clients):
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[client]['x']
                cur_y = train_data_temp[client]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            # For testing, organize data without user mapping
            test_data_x = []
            test_data_y = []
            for i, client in enumerate(train_clients):
                cur_x = test_data_temp[client]['x']
                cur_y = test_data_temp[client]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    # Fetches a single sample from the dataset by index.
    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        indices = torch.LongTensor(np.array(indices))
        target = torch.FloatTensor(np.array(target))
        return indices, target

    # Returns the mapping of user IDs to their data indices.
    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            raise ValueError("The test dataset does not have dic_users!")

class CharLSTM(nn.Module):
    def __init__(self):
        super(CharLSTM, self).__init__()
        embedding_dim = 8
        hidden_size = 100
        num_LSTM = 2
        input_length = 80
        self.n_cls = 80
        self.embedding = nn.Embedding(input_length, embedding_dim)
        self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
        self.fc = nn.Linear(hidden_size, self.n_cls)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output, (h_, c_) = self.stacked_LSTM(x)
        last_hidden = output[-1, :, :]
        x = self.fc(last_hidden)

        return x

# Define the model, loss function, number of epochs, and optimizer
model = CharLSTM()
criterion = nn.CrossEntropyLoss()
epochs = 10
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

class Args:
    shakespeare_train_path = '/Users/mirkomilegamberdiev/Desktop/ml/shakespeare/leaf/data/shakespeare/data/train/all_data_train_9.json'
    shakespeare_test_path = '/Users/mirkomilegamberdiev/Desktop/ml/shakespeare/leaf/data/shakespeare/data/test/all_data_test_9.json'

args = Args()
train_dataset = Shakespeare(train=True, args=args)
test_dataset = Shakespeare(train=False, args=args)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check for MPS (Metal Performance Shaders) support
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS on macOS
    print("Using MPS for acceleration")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for acceleration")

# Move the model to the selected device
model.to(device)

# Initialize the model
model = CharLSTM()
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS on macOS
    print("Using MPS for acceleration")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for acceleration")

# Move the model to the selected device
model.to(device)

# Loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=20)  # T_max = number of epochs

# Load the datasets
train_dataset = Shakespeare(train=True, args=args)
test_dataset = Shakespeare(train=False, args=args)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Lists to store test loss and accuracy for plotting
test_losses = []
test_accuracies = []

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.argmax(dim=1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Step the scheduler
    scheduler.step()

    # Evaluate on the test dataset
    model.eval()
    total_test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.argmax(dim=1))
            total_test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets.argmax(dim=1)).sum().item()
            total += targets.size(0)

    test_loss = total_test_loss / len(test_loader)
    accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)

    end_time = time.time()
    epoch_time = (end_time - start_time) / 60

    # Log epoch results
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {total_loss / len(train_loader):.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {accuracy:.2f}% - Time: {epoch_time:.2f} minutes")

print("Training complete.")

# Plotting the results
epochs = range(1, num_epochs + 1)

# Plot Test Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, test_losses, label='Test Loss', color='blue', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss Over Epochs')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Plot Test Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, test_accuracies, label='Test Accuracy', color='green', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Epochs')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
