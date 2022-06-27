import torch
from torch import nn
from torch.autograd import Variable

from DeepRiver.utils.module_finder import get_activation_fn, get_init_fn


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation_fn="selu",
        init_fn="xavier_uniform",
        weight=None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = get_activation_fn(activation_fn)()
        if weight is not None:
            self.linear.weight = nn.Parameter(weight)
        elif init_fn != "uniform":
            init = get_init_fn(init_fn)
            init(self.linear.weight, activation_fn=activation_fn)

    def forward(self, x):
        encoded = self.linear(x)
        return self.activation(encoded)

    def get_weight(self):
        return self.linear.weight


class SequentialLSTM(nn.Module):
    """Documentation needs to be added"""

    def __init__(self, input_size, hidden_size, num_layers):
        super(SequentialLSTM, self).__init__()
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size / num features
        self.hidden_size = hidden_size  # hidden state

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )  # lstm

    def forward(self, x):
        h_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # hidden state
        c_0 = Variable(
            torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        )  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next

        return hn


def init_weights(layer: nn.Module, init_fn: str = "xavier_uniform"):
    init_fn = get_init_fn(init_fn)
