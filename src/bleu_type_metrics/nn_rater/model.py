import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiGRU(nn.Module):
    def __init__(self, hidden_size, embedding, n_layer=1):
        super(BiGRU, self).__init__()
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size,
                          n_layer, bidirectional=True)

    def forward(self, input_seq, input_length, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        outputs, hidden = self.gru(packed, hidden)
        return outputs, hidden


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layer=1):
        super(MLP, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                     for _ in range(n_layer-1)])
        self.relu = nn.ReLU()
        self.n_layer = n_layer
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, source):
        h_ = self.relu(self.input(source))
        for i in range(self.n_layer-1):
            h_ = self.relu(self.hidden[i](h_))
        output = self.output(h_)
        output = self.softmax(output)
        return output
