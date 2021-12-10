from torch import nn

from ..utils import get_activation_fn


def get_fc_autoencoder(
    n_features,
    dropout=0.2,
    layer_size=20,
    n_layers=3,
    activation_fn="selu",
    latent_dim=10,
):
    activation = get_activation_fn(activation_fn)

    encoder = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(n_features, layer_size),
        activation(),
        *[nn.Linear(layer_size, layer_size), activation()] * (n_layers - 2),
        nn.Linear(layer_size, latent_dim),
        activation(),
    )

    decoder = (
        nn.Linear(latent_dim, layer_size),
        activation(),
        *[nn.Linear(layer_size, layer_size), activation()] * (n_layers - 2),
        nn.Linear(layer_size, n_features),
        nn.Sigmoid(),
    )
    return encoder, decoder


def get_conv_autoencoder_28(activation_fn="selu", dropout=0.5, n_features=1):
    activation = get_activation_fn(activation_fn)

    encoder = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Conv2d(in_channels=n_features, out_channels=32, kernel_size=3, stride=2),
        activation,
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
        activation,
        nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=3),
        activation,
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=3),
        activation,
        nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
        activation,
        nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2),
    )
    return encoder, decoder


def get_conv_autoencoder_32(activation_fn="selu", dropout=0.5, n_features=3):
    activation = get_activation_fn(activation_fn)

    encoder = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Conv2d(in_channels=n_features, out_channels=64, kernel_size=3, stride=2),
        activation,
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        activation,
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
        activation,
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
        activation,
    )
    decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
        activation,
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
        activation,
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
        activation,
        nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2),
    )
    return encoder, decoder
