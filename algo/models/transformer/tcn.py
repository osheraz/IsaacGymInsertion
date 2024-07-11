import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

from torch import nn


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.input_size = input_size
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.act = nn.Tanh()

    def forward(self, img, tactile, inputs, shit=None):
        """Inputs have to have dimension (N, C_in, L_in)"""

        # inputs = inputs[:, None, :].reshape(inputs.shape[0], self.input_size, -1)  # (batch, input_feature, -1)
        if inputs.dim() != 3:
            inputs = inputs[None, :, :]  # (batch, input_feature, -1)

        inputs = inputs.permute(0, 2, 1)
        
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L) batch_size, input_feature, timesteps (N 6 8)
        o = self.linear(y1[:, :, -1])
        return self.act(o)

class ACTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(ACTCN, self).__init__()
        self.tcn = TemporalConvNet(input_size - 3, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.input_size = input_size
        self.linear1 = nn.Linear(num_channels[-1], output_size)
        self.linear2 = nn.Linear(2 * output_size, 256)
        self.linear3 = nn.Linear(256, output_size)

        self.relu = nn.ReLU()
        self.act = nn.Tanh()

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""

        # inputs = inputs[:, None, :].reshape(inputs.shape[0], self.input_size, -1)  # (batch, input_feature, -1)
        if inputs.dim() != 3:
            inputs = inputs[None, :, :]  # (batch, input_feature, -1)

        a = inputs[:, 0, 6:]  # TODO: change
        inputs = inputs[:, :, :6]
        inputs = inputs.permute(0, 2, 1)

        y1 = self.tcn(inputs)  # input should have dimension (N, C, L) batch_size, input_feature, timesteps (N 6 8)
        o = self.linear1(y1[:, :, -1])
        oa = torch.cat([o, a], 1)
        o = self.relu(self.linear2(oa))
        o = self.linear3(o)

        return self.act(o)