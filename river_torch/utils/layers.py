import torch
from torch import nn
from torch.autograd import Variable

from river_torch.utils.params import get_activation_fn, get_init_fn


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

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x)  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next

        return hn


def find_output_layer(net: nn.Sequential) -> nn.Linear:
    """Return the output layer of a network.

    Parameters
    ----------
    net
        The network to find the output layer of.

    Returns
    -------
    nn.Linear
        The output layer of the network.
    """

    for layer in list(net.children())[::-1]:
        if isinstance(layer, nn.Linear):
            return layer
    raise ValueError("No dense layer found.")
