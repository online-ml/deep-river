import torch
import torch.nn.functional as F
from torch import nn, optim
from DeepRiver.utils.optim import SGDHD


def rmse_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    return torch.sqrt(F.mse_loss(input, target, size_average, reduce, reduction))


ACTIVATION_FNS = {
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "linear": nn.Identity,
}

LOSS_FNS = {
    "mse": F.mse_loss,
    "rmse": rmse_loss,
    "mae": F.l1_loss,
    "smooth_mae": F.smooth_l1_loss,
    "bce": F.binary_cross_entropy,
    "ce": F.cross_entropy,
    "kld": F.kl_div,
    "huber": F.huber_loss,
}

OPTIMIZER_FNS = {
    "adam": optim.Adam,
    "adam_w": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop, 
    "lbfgs": optim.LBFGS, 
    "hd-sgd": SGDHD,
}

INIT_FNS = {
    "uniform": nn.init.uniform_,
    "normal": nn.init.normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
}


def get_activation_fn(activation_fn):
    return (
        ACTIVATION_FNS.get(activation_fn)
        if isinstance(activation_fn, str)
        else activation_fn
    )


def get_loss_fn(loss_fn):
    return loss_fn if callable(loss_fn) else LOSS_FNS.get(loss_fn)


# todo how to handle parameters and the nn parameters?
def get_optimizer_fn(optimizer_fn):
    return (
        OPTIMIZER_FNS.get(optimizer_fn)
        if isinstance(optimizer_fn, str)
        else optimizer_fn
    )


def get_init_fn(init_fn):
    init_fn_ = INIT_FNS.get(init_fn, "xavier_uniform")
    if init_fn.startswith("xavier"):
        result = lambda weight, activation_fn: init_fn_(
            weight, gain=nn.init.calculate_gain(activation_fn)
        )
    elif init_fn.startswith("kaiming"):
        result = lambda weight, activation_fn: init_fn_(
            weight, nonlinearity=activation_fn
        )
    elif init_fn == "uniform":
        result = lambda weight, activation_fn: 0
    else:
        result = lambda weight, activation_fn=None: init_fn_(weight)
    return result
