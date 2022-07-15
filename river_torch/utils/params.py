from typing import Callable
import torch
import torch.nn.functional as F
from torch import nn, optim
from typing import Union

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
}

INIT_FNS = {
    "uniform": nn.init.uniform_,
    "normal": nn.init.normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
}


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


BASE_PARAM_ERROR = "Unknown {}: {}. A valid string or {} is required."


def get_activation_fn(activation_fn: Union[str, type]) -> type:
    """Returns the requested activation function as a nn.Module class.

    Parameters
    ----------
    activation_fn
        The activation function to fetch. Can be a string or a nn.Module class.

    Raises:
        ValueError: If the activation function is not found.

    Returns:
        type: The class of the requested activation function.
    """
    err = ValueError(
        BASE_PARAM_ERROR.format("activation function", activation_fn, "nn.Module")
    )
    if isinstance(activation_fn, str):
        try:
            activation_fn = ACTIVATION_FNS[activation_fn]
        except KeyError:
            raise err
    elif not isinstance(activation_fn(), nn.Module):
        raise err
    return activation_fn


def get_optim_fn(optim_fn: Union[str, type]) -> type:
    """Returns the requested optimizer as a nn.Module class.

    Parameters
    ----------
    optim_fn
        The optimizer to fetch. Can be a string or a nn.Module class.

    Raises:
        ValueError: If the optimizer is not found.

    Returns:
        type: The class of the requested optimizer.
    """
    err = ValueError(BASE_PARAM_ERROR.format("optimizer", optim_fn, "nn.Module"))
    if isinstance(optim_fn, str):
        try:
            optim_fn = OPTIMIZER_FNS[optim_fn]
        except KeyError:
            raise err
    elif not isinstance(
        optim_fn(params=[torch.empty(1)], lr=1e-3), torch.optim.Optimizer
    ):
        raise err
    return optim_fn


def get_loss_fn(loss_fn: Union[str, Callable]) -> Callable:
    """Returns the requested loss function as a function.

    Parameters
    ----------
    loss_fn
        The loss function to fetch. Can be a string or a function.

    Raises:
        ValueError: If the loss function is not found.

    Returns:
        function: The function of the requested loss function.
    """
    err = ValueError(BASE_PARAM_ERROR.format("loss function", loss_fn, "function"))
    if isinstance(loss_fn, str):
        try:
            loss_fn = LOSS_FNS[loss_fn]
        except KeyError:
            raise err
    elif not callable(loss_fn):
        raise err
    return loss_fn
