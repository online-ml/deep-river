"""Utility classes and functions."""
from .estimator_checks import check_estimator
from .layers import SequentialLSTM
from .module_finder import get_activation_fn, get_loss_fn, get_optimizer_fn, get_init_fn
from .river_compat import dict2tensor
from .incremental_stats import WindowedMeanMeter, WindowedVarianceMeter

__all__ = [
    "check_estimator",
    "SequentialLSTM",
    "get_activation_fn",
    "get_loss_fn",
    "get_optimizer_fn",
    "get_init_fn",
    "dict2tensor",
    "WindowedMeanMeter",
    "WindowedVarianceMeter",
]
