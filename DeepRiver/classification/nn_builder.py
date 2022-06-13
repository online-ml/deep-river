from torch import nn

from DeepRiver.utils import SequentialLSTM


def build_torch_mlp_classifier(n_features, n_classes=2):
    net = nn.Sequential(
        nn.Linear(n_features, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        # nn.Linear(5, 5),
        # nn.ReLU(),
        # nn.Linear(5, 5),
        # nn.ReLU(),
        nn.Linear(5, n_classes),
        # nn.Sigmoid()
    )
    return net


def build_torch_dynamic_mlp_classifier(n_features, n_classes=2):
    net = nn.Sequential(
        nn.Linear(n_features, n_features * 5),
        nn.LeakyReLU(),
        nn.Linear(n_features * 5, n_features * 2),
        nn.LeakyReLU(),
        nn.Linear(n_features * 2, n_features),
        nn.LeakyReLU(),
        nn.Linear(n_features, n_classes),
        # nn.Sigmoid()  #Cross entropy loss does not expect sigmoid
    )
    return net


def build_torch_lstm_classifier(n_features, hidden_size):
    net = nn.Sequential(
        SequentialLSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1),
        nn.Linear(hidden_size, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    return net


def build_torch_linear_classifier(n_features, n_classes=1):
    net = nn.Sequential(
        nn.Linear(n_features, n_classes),
        # nn.Sigmoid()
    )
    return net


def build_torch_conv1d_classifier(n_features):
    print(n_features)
    net = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
        nn.Linear(10, 1),
        # nn.ReLU(),
        # nn.Linear(10,1),
        # nn.Sigmoid(),
        # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=1),
        # nn.Linear(10, 1),
        nn.Sigmoid()
    )
    return net
