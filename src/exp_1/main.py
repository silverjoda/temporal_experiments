import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(Net, self).__init__()

        T.manual_seed(1337)
        # Input embedding layer
        self.fc_emb = nn.Linear(in_dim, hidden_dim)
        T.nn.init.xavier_uniform_(self.fc_emb.weight)
        self.fc_emb.bias.data.fill_(0)

        # Lstm Cell
        self.rnn_cell = nn.RNNCell(hidden_dim, hidden_dim)
        T.manual_seed(1337)
        T.nn.init.orthogonal_(self.rnn_cell.weight_ih)
        T.nn.init.eye_(self.rnn_cell.weight_hh)
        self.rnn_cell.bias_ih.data.fill_(0)
        self.rnn_cell.bias_hh.data.fill_(0)

        # Lstm Full
        self.rnn_full = nn.RNN(hidden_dim, hidden_dim, 1, batch_first=True)
        T.manual_seed(1337)
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
        x = self.fc_out(h)
        return x, h


    def forward_full(self, x):
        x = F.relu(self.fc_emb(x))
        x, _ = self.rnn_full(x, None)
        x = self.fc_out(x)
        return x



def examine_cell(net, in_dim, out_dim, hidden_dim, seq_len):
    T.manual_seed(1337)
    seq = T.rand(1, seq_len, in_dim)
    labels = T.rand(1, seq_len, out_dim)

    h = T.zeros((1,hidden_dim))
    outputs = []
    hiddens = [h]
    for i in range(seq.shape[1]):
        x = seq[0, i:i+1, :]
        y, h = net.forward(x, h)
        outputs.append(y)
        hiddens.append(h)
    del hiddens[-1]
    outputs_tensor = T.cat(outputs).unsqueeze(0)

    lossfun = nn.MSELoss()
    loss = lossfun(outputs_tensor, labels)
    loss.backward()

    print("Sequence: ", seq)
    print("Labels: ", labels)
    print("Outputs: ", outputs_tensor)

    print("fc_emb weight", net.fc_emb.weight)
    print("fc_emb weight grad", net.fc_emb.weight.grad)
    print("fc_emb bias", net.fc_emb.bias)
    print("fc_emb bias grad", net.fc_emb.bias.grad)

    # print("hh weight", net.rnn_cell.weight_hh.data)
    # print("hh weight grad", net.rnn_cell.weight_hh.grad)
    # print("hh bias", net.rnn_cell.bias_hh.data)
    # print("hh bias grad", net.rnn_cell.bias_hh.grad)

    # print("ih weight", net.rnn_cell.weight_ih.data)
    # print("ih weight grad", net.rnn_cell.weight_ih.grad)
    # print("ih bias", net.rnn_cell.bias_ih.data)
    # print("ih bias grad", net.rnn_cell.bias_ih.grad)

    print("\n")

    net.zero_grad()


def examine_cell_manual(net, in_dim, out_dim, hidden_dim, seq_len):
    T.manual_seed(1337)
    seq = T.rand(1, seq_len, in_dim)
    labels = T.rand(1, seq_len, out_dim)

    all_outputs = []
    for i in range(seq.shape[1]):
        h = T.zeros((1, hidden_dim))
        outputs = []
        hiddens = [h]
        for j in range(i + 1):
            x = seq[0, j:j+1, :]
            y, h = net.forward(x, h)
            outputs.append(y)
            hiddens.append(h)
        del hiddens[-1]
        last_output = outputs[-1]
        all_outputs.append(last_output)

        #
        lossfun = nn.MSELoss()
        loss = lossfun(last_output, labels[:, i, :])
        loss.backward()

    print("Sequence: ", seq)
    print("Labels: ", labels)
    print("Outputs: ", T.cat(all_outputs))

    print("fc_emb weight", net.fc_emb.weight)
    print("fc_emb weight grad", net.fc_emb.weight.grad / seq_len)
    print("fc_emb bias", net.fc_emb.bias)
    print("fc_emb bias grad", net.fc_emb.bias.grad / seq_len)
    print("\n")

    net.zero_grad()


def examine_full(net, in_dim, out_dim, hidden_dim, seq_len):
    T.manual_seed(1337)
    seq = T.rand(1, seq_len, in_dim)
    labels = T.rand(1, seq_len, out_dim)
    outputs_tensor = net.forward_full(seq)

    lossfun = nn.MSELoss()
    loss = lossfun(outputs_tensor, labels)
    loss.backward()

    print("Sequence: ", seq)
    print("Labels: ", labels)
    print("Outputs: ", outputs_tensor)

    # print("fc_emb weight", net.fc_emb.weight)
    # print("fc_emb weight grad", net.fc_emb.weight.grad)
    # print("fc_emb bias", net.fc_emb.bias)
    # print("fc_emb bias grad", net.fc_emb.bias.grad)

    # print("hh weight", net.rnn_full.weight_hh_l0.data)
    # print("hh weight grad", net.rnn_full.weight_hh_l0.grad)
    # print("hh bias", net.rnn_full.bias_hh_l0.data)
    # print("hh bias grad", net.rnn_full.bias_hh_l0.grad)

    print("ih weight", net.rnn_full.weight_ih_l0.data)
    print("ih weight grad", net.rnn_full.weight_ih_l0.grad)
    print("ih bias", net.rnn_full.bias_ih_l0.data)
    print("ih bias grad", net.rnn_full.bias_ih_l0.grad)

    net.zero_grad()


def main():

    in_dim = 2
    out_dim = 1
    hidden_dim = 2
    seq_len = 5

    net = Net(in_dim, out_dim, hidden_dim)

    print("============ Examine cell ============")
    examine_cell(net, in_dim, out_dim, hidden_dim, seq_len)

    print("============ Examine cell manual ============")
    examine_cell_manual(net, in_dim, out_dim, hidden_dim, seq_len)

    # print("Examine full:")
    # examine_full(net, in_dim, out_dim, hidden_dim, seq_len)

if __name__ == '__main__':
    main()








