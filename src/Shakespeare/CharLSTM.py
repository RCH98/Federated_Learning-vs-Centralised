import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

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