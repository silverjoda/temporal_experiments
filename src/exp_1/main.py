import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, rec_dim):
        super(Net, self).__init__()

        # Input embedding layer
        self.fc_emb = nn.Linear(in_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.fc_emb.weight)
        self.fc_emb.bias.data.fill_(0)

        # Lstm Cell
        self.rnn_cell = nn.RNNCell(hidden_dim, rec_dim)
        torch.nn.init.orthogonal_(self.rnn_cell.weight_ih)
        torch.nn.init.eye_(self.rnn_cell.weight_hh)
        self.rnn_cell.bias_ih.data.fill_(0)
        self.rnn_cell.bias_hh.data.fill_(0)

        # Lstm Full
        self.rnn_full = nn.RNN(hidden_dim, rec_dim, 1, batch_first=True)
        torch.nn.init.orthogonal_(self.rnn_full.weight_ih_l0)
        torch.nn.init.eye_(self.rnn_full.weight_hh_l0)
        self.rnn_full.bias_ih_l0.data.fill_(0)
        self.rnn_full.bias_hh_l0.data.fill_(0)

        # Output linear layer
        self.fc_out = nn.Linear(rec_dim, out_dim)
        torch.nn.init.xavier_uniform_(self.fc_out.weight)
        self.fc_emb.bias.data.fill_(0)


    def forward(self, x, h):
        x = F.relu(self.fc_emb(x))
        x, h = self.rnn_cell(x)
        x = F.relu(self.fc_out(h))
        return x, h


    def forward_full(self, x):
        x = F.relu(self.fc_emb(x))
        x, _ = self.rnn_full(x, None)
        x = F.relu(self.fc_out(x))
        return x


def main():
    #
    net = Net(3, 3, 3, 3)

if __name__ == '__main__':
    main()