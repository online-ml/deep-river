"""Utility classes and functions."""
from .estimator_checks import check_estimator
from .params import get_activation_fn, get_init_fn, get_loss_fn, get_optim_fn
from .tensor_conversion import (
    deque2rolling_tensor,
    df2tensor,
    dict2tensor,
    float2tensor,
    labels2onehot,
    output2proba,
)

__all__ = [
    "check_estimator",
    "get_activation_fn",
    "get_loss_fn",
    "get_optim_fn",
    "get_init_fn",
    "dict2tensor",
    "labels2onehot",
    "deque2rolling_tensor",
    "df2rolling_tensor",
    "df2tensor",
    "float2tensor",
    "output2proba",
]
