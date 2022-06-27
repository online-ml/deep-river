import math

from DeepRiver.utils import get_activation_fn, get_init_fn
from DeepRiver.utils.layers import DenseBlock
from torch import nn


def get_fc_encoder(
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

    layer_sizes = [n_features, *[layer_size] * (n_layers - 1), latent_dim]
    encoder_activations = (
        [activation_fn] * (n_layers - 1) + ["linear"]
        if variational
        else [activation_fn] * n_layers
    )
    decoder_activations = [activation_fn] * (n_layers - 1) + [final_activation]

    encoder_layers = [nn.Dropout(dropout)] if dropout > 0 else []
    decoder_layers = []

    for layer_idx in range(len(layer_sizes) - 1):
        encoder_out = layer_sizes[layer_idx + 1]
        if variational and layer_idx == len(layer_sizes) - 2:
            encoder_out *= 2
        encoder_block = DenseBlock(
            in_features=layer_sizes[layer_idx],
            out_features=encoder_out,
            activation_fn=encoder_activations[layer_idx],
            init_fn=init_fn,
        )
        encoder_layers.append(encoder_block)

    return nn.Sequential(*encoder_layers)


def get_fc_decoder(
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

    layer_sizes = [n_features, *[layer_size] * (n_layers - 1), latent_dim]
    encoder_activations = (
        [activation_fn] * (n_layers - 1) + ["linear"]
        if variational
        else [activation_fn] * n_layers
    )
    decoder_activations = [activation_fn] * (n_layers - 1) + [final_activation]

    decoder_layers = []

    for layer_idx in range(len(layer_sizes) - 1):
        encoder_out = layer_sizes[layer_idx + 1]
        if variational and layer_idx == len(layer_sizes) - 2:
            encoder_out *= 2
        encoder_block = DenseBlock(
            in_features=layer_sizes[layer_idx],
            out_features=encoder_out,
            activation_fn=encoder_activations[layer_idx],
            init_fn=init_fn,
        )
        decoder_weight = (
            encoder_block.get_weight().t() if tied_decoder_weights else None
        )
        decoder_block = DenseBlock(
            in_features=layer_sizes[layer_idx + 1],
            out_features=layer_sizes[layer_idx],
            activation_fn=decoder_activations[layer_idx],
            weight=decoder_weight,
            init_fn=init_fn,
        )
        decoder_layers.insert(0, decoder_block)

    return nn.Sequential(*decoder_layers)
