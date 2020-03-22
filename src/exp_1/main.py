import torch as T
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(Net, self).__init__()

        # Input embedding layer
        self.fc_emb = nn.Linear(in_dim, hidden_dim)
        T.nn.init.xavier_uniform_(self.fc_emb.weight)
        self.fc_emb.bias.data.fill_(0)

        # Lstm Cell
        self.rnn_cell = nn.RNNCell(hidden_dim, hidden_dim)
        T.nn.init.orthogonal_(self.rnn_cell.weight_ih)
        T.nn.init.eye_(self.rnn_cell.weight_hh)
        self.rnn_cell.bias_ih.data.fill_(0)
        self.rnn_cell.bias_hh.data.fill_(0)

        # Lstm Full
        self.rnn_full = nn.RNN(hidden_dim, hidden_dim, 1, batch_first=True)
        T.nn.init.orthogonal_(self.rnn_full.weight_ih_l0)
        T.nn.init.eye_(self.rnn_full.weight_hh_l0)
        self.rnn_full.bias_ih_l0.data.fill_(0)
        self.rnn_full.bias_hh_l0.data.fill_(0)

        # Output linear layer
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        T.nn.init.xavier_uniform_(self.fc_out.weight)
        self.fc_emb.bias.data.fill_(0)


    def forward(self, x, h):
        x = F.relu(self.fc_emb(x))
        h = self.rnn_cell(x, h)
        x = F.relu(self.fc_out(h))
        return x, h


    def forward_full(self, x):
        x = F.relu(self.fc_emb(x))
        x, _ = self.rnn_full(x, None)
        x = F.relu(self.fc_out(x))
        return x


def examine_cell(net, in_dim, out_dim, hidden_dim, seq_len):
    seq = T.rand(1, seq_len, in_dim)

    h = None
    outputs = []
    hiddens = [h]
    for i in range(seq.shape[1]):
        x = seq[0, i:i+1, :]
        y, h = net.forward(x, h)
        outputs.append(y)
        hiddens.append(h)
    del hiddens[-1]


def examine_full(net, in_dim, out_dim, hidden_dim, seq_len):
    seq = T.rand(1, seq_len, in_dim)
    out = net.forward_full(seq)


def main():
    in_dim = 3
    out_dim = 1
    hidden_dim = 3
    seq_len = 5

    net = Net(in_dim, out_dim, hidden_dim)

    examine_cell(net, in_dim, out_dim, hidden_dim, seq_len)
    examine_full(net, in_dim, out_dim, hidden_dim, seq_len)

if __name__ == '__main__':
    main()








