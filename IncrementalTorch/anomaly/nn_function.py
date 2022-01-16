from modulefinder import Module
from torch import nn
import math
from IncrementalTorch.utils import get_activation_fn, get_init_fn


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


def get_fc_autoencoder(
    n_features,
    dropout=0.1,
    layer_size=2.0,
    n_layers=1,
    activation_fn="selu",
    latent_dim=1.0,
    variational=False,
    final_activation="sigmoid",
    tied_decoder_weights=True,
    init_fn="xavier_uniform",
):
    if isinstance(latent_dim, float):
        latent_dim = math.ceil(latent_dim * n_features)
    if isinstance(layer_size, float):
        layer_size = math.ceil(layer_size * n_features)

    encoder_output_dim = latent_dim * 2 if variational else latent_dim

    layer_sizes = [n_features, *[layer_size] * (n_layers - 1), encoder_output_dim]
    encoder_activations = (
        [activation_fn] * (n_layers - 1) + ["linear"]
        if variational
        else [activation_fn] * n_layers
    )
    decoder_activations = [activation_fn] * (n_layers - 1) + [final_activation]

    encoder_layers, decoder_layers = [nn.Dropout(dropout)], []

    for layer_idx in range(len(layer_sizes) - 1):
        encoder_block = DenseBlock(
            in_features=layer_sizes[layer_idx],
            out_features=layer_sizes[layer_idx + 1],
            activation_fn=encoder_activations[layer_idx],
            init_fn=init_fn,
        )
        decoder_weight = encoder_block.get_weight().t() if tied_decoder_weights else None
        decoder_block = DenseBlock(
            in_features=layer_sizes[layer_idx + 1],
            out_features=layer_sizes[layer_idx],
            activation_fn=decoder_activations[layer_idx],
            weight=decoder_weight,
            init_fn=init_fn,
        )
        encoder_layers.append(encoder_block)
        decoder_layers.insert(0, decoder_block)

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


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
