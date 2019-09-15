import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import Attn


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, n_layers=1):
        super(Decoder, self).__init__()

        # Keep parameters for reference
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size + hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N

        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.lstm(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(rnn_output), 1)

        # Return final output, hidden state
        return output, context, hidden

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.loadd_state_dict(torch.load(path))