import torch

from torch import autograd, nn, optim
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_size).cuda())
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        attn_energies = torch.zeros(seq_len).cuda()

        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, 0).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        '''Aditive Attention'''
        attn_input = torch.cat((hidden, encoder_output), 1)
        energy = self.attn(attn_input)
        energy = self.v.dot(energy.view(-1))
        return energy