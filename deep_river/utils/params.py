from typing import Callable, Union

import torch
import torch.nn.functional as F
from torch import nn, optim

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
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
    "binary_cross_entropy": F.binary_cross_entropy,
    "cross_entropy": F.cross_entropy,
    "kl_div": F.kl_div,
    "huber": F.huber_loss,
    "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits,
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
    """Returns the requested init function.

    Parameters
    ----------
    init_fn
        The init function to fetch. Must be one of ["xavier_uniform",
        "uniform", "kaiming_uniform"].

    Returns
    -------
    Callable
        The class of the requested activation function.
    """
    init_fn_ = INIT_FNS.get(init_fn, "xavier_uniform")
    if init_fn.startswith("xavier"):

        def result(weight, activation_fn):
            return init_fn_(weight, gain=nn.init.calculate_gain(activation_fn))

    elif init_fn.startswith("kaiming"):

        def result(weight, activation_fn):
            return init_fn_(weight, nonlinearity=activation_fn)

    elif init_fn == "uniform":

        def result(weight, activation_fn):
            return 0

    else:

        def result(weight, activation_fn):
            return init_fn_(weight)

    return result


BASE_PARAM_ERROR = "Unknown {}: {}. A valid string or {} is required."


def get_activation_fn(activation_fn: Union[str, Callable]) -> Callable:
    """Returns the requested activation function as a nn.Module class.

    Parameters
    ----------
    activation_fn
        The activation function to fetch. Can be a string or a nn.Module class.

    Returns
    -------
    Callable
        The class of the requested activation function.
    """
    err = ValueError(
        BASE_PARAM_ERROR.format(
            "activation function", activation_fn, "nn.Module"
        )
    )
    if isinstance(activation_fn, str):
        try:
            activation_fn = ACTIVATION_FNS[activation_fn]
        except KeyError:
            raise err
    elif not isinstance(activation_fn(), nn.Module):
        raise err
    return activation_fn


def get_optim_fn(optim_fn: Union[str, Callable]):
    """Returns the requested optimizer as a nn.Module class.

    Parameters
    ----------
    optim_fn
        The optimizer to fetch. Can be a string or a nn.Module class.


    Returns
    -------
    Callable
        The class of the requested optimizer.
    """
    err = ValueError(
        BASE_PARAM_ERROR.format("optimizer", optim_fn, "nn.Module")
    )
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


def get_loss_fn(loss_fn: Union[str, Callable]):
    """Returns the requested loss function as a function.

    Parameters
    ----------
    loss_fn
        The loss function to fetch. Can be a string or a function.

    Returns
    -------
    Callable
        The function of the requested loss function.
    """
    err = ValueError(
        BASE_PARAM_ERROR.format("loss function", loss_fn, "function")
    )
    if isinstance(loss_fn, str):
        try:
            return LOSS_FNS[loss_fn]
        except KeyError:
            raise err
    elif not callable(loss_fn):
        raise err
    return loss_fn
