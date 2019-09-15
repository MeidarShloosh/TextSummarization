import torch
import numpy
from torch import autograd, nn, optim
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = (torch.zeros(self.n_layers, 1, self.hidden_size).cuda(),
                  torch.zeros(self.n_layers, 1, self.hidden_size).cuda())
        return hidden

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.loadd_state_dict(torch.load(path))