from torch import nn, optim
import torch.nn.functional as F
import torch
import pandas as pd

ACTIVATION_FNS = {
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

LOSS_FNS = {
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "smooth_mae": F.smooth_l1_loss,
    "bce": F.binary_cross_entropy,
    "kld": F.kl_div,
    "huber": F.huber_loss,
}

OPTIMIZER_FNS = {
    "adam": optim.Adam,
    "adam_w": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


def get_activation_fn(activation_fn):
    return (
        ACTIVATION_FNS.get(activation_fn)
        if isinstance(activation_fn, str)
        else activation_fn
    )


def get_loss_fn(loss_fn):
    return loss_fn if callable(loss_fn) else LOSS_FNS.get(loss_fn)


def get_optimizer_fn(optimizer_fn):
    return (
        OPTIMIZER_FNS.get(optimizer_fn)
        if isinstance(optimizer_fn, str)
        else optimizer_fn
    )



