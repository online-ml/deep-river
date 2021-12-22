from torch import nn
import math
from IncrementalTorch.utils import get_activation_fn, get_init_fn


def get_fc_autoencoder(
    n_features,
    dropout=0.1,
    layer_size=35,
    n_layers=2,
    activation_fn="selu",
    latent_dim=15,
    variational=False,
    final_activation="sigmoid",
    tied_decoder_weights=False,
    init_fn="xavier_uniform",
):
    if isinstance(latent_dim, float):
        latent_dim = math.ceil(latent_dim * n_features)
    if isinstance(layer_size, float):
        layer_size = math.ceil(layer_size * n_features)

    activation = get_activation_fn(activation_fn)
    if activation_fn == "elu":
        activation_fn = "linear"
    init_fn = get_init_fn(init_fn)

    encoder_output_dim = latent_dim * 2 if variational else latent_dim

    encoder_layers = [
        nn.Dropout(p=dropout),
        nn.Linear(n_features, layer_size),
        activation(),
        *[nn.Linear(layer_size, layer_size), activation()] * (n_layers - 2),
        nn.Linear(layer_size, encoder_output_dim),
    ]

    for idx, layer in enumerate(encoder_layers[:-1]):
        if isinstance(layer, nn.Linear):
            init_fn(layer.weight, activation_fn=activation_fn)
    if variational:
        init_fn(encoder_layers[-1].weight, activation_fn="linear")
    else:
        init_fn(encoder_layers[-1].weight, activation_fn=activation_fn)
        encoder_layers.append(activation())

    decoder_layers = [
        nn.Linear(latent_dim, layer_size),
        activation(),
        *[nn.Linear(layer_size, layer_size), activation()] * (n_layers - 2),
        nn.Linear(layer_size, n_features),
    ]

    for idx, layer in enumerate(decoder_layers):
        if isinstance(layer, nn.Linear):
            if tied_decoder_weights:
                layer.weight = nn.Parameter(encoder_layers[-idx - 2].weight.t())
            elif idx == len(decoder_layers) - 1:
                init_fn(layer.weight, activation_fn=activation_fn)
            else:
                init_fn(layer.weight, activation_fn=final_activation)

    if final_activation != "linear":
        decoder_layers.append(get_activation_fn(final_activation)())

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
