"""Utility classes and functions."""

from .params import get_activation_fn, get_init_fn, get_loss_fn, get_optim_fn
from .tensor_conversion import (
    deque2rolling_tensor,
    df2tensor,
    dict2tensor,
    float2tensor,
    labels2onehot,
    output2proba,
)


def check_estimator(model):
    from .estimator_checks import check_estimator as check

    return check(model)


def __getattr__(name: str):
    if name == "estimator_checks":
        from importlib import import_module

        return import_module(f"{__name__}.estimator_checks")
    raise AttributeError(name)


__all__ = [
    "check_estimator",
    "get_activation_fn",
    "get_loss_fn",
    "get_optim_fn",
    "get_init_fn",
    "dict2tensor",
    "labels2onehot",
    "deque2rolling_tensor",
    "df2tensor",
    "float2tensor",
    "output2proba",
]
