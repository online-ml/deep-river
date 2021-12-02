from torch import nn

from .utils import SequentialLSTM


def build_torch_mlp_regressor(n_features):
    net = nn.Sequential(
        nn.Linear(n_features, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 5),
        nn.Linear(5, 1)
    )
    return net

def build_torch_linear_regressor(n_features):
    net = nn.Sequential(
        nn.Linear(n_features,1)
    )
    return net

def build_torch_lstm_regressor(n_features, hidden_size):
    net = nn.Sequential(
        SequentialLSTM(input_size=n_features,hidden_size=hidden_size,num_layers=1),
        nn.Linear(hidden_size,20),
        nn.Linear(20,1)
    )
    return net