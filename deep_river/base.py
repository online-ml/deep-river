import abc
import collections
import inspect
from typing import Any, Callable, Deque, Optional, Type, Union, cast

import torch
from river import base

from deep_river.utils import get_loss_fn, get_optim_fn

try:
    from graphviz import Digraph
    from torchviz import make_dot
except ImportError as e:
    raise ValueError(
        "You have to install graphviz to use the draw method"
    ) from e


class DeepEstimator(base.Estimator):
    """
    Abstract base class that implements basic functionality of
    River-compatible PyTorch wrappers.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped.
        The Module should accept parameter `n_features` so that the returned
        model's input shape can be determined based on the number of features
        in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss
        function provided by `torch.nn.functional` or one of the following:
        'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **kwargs
        Parameters to be passed to the `Module` or the `optimizer`.
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.module_cls = module
        self.module: torch.nn.Module = cast(torch.nn.Module, None)  # cleaner
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = get_optim_fn(optimizer_fn)
        self.lr = lr
        self.device = device
        self.kwargs = kwargs
        self.seed = seed
        self.module_initialized = False
        torch.manual_seed(seed)

    @abc.abstractmethod
    def learn_one(self, x: dict, y: Optional[Any]) -> "DeepEstimator":
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        DeepEstimator
            The estimator itself.
        """
        raise NotImplementedError

    def _filter_kwargs(self, fn: Callable, override=None, **kwargs) -> dict:
        """Filters `net_params` and returns those in `fn`'s arguments.

        Parameters
        ----------
        fn
            Arbitrary function
        override
            Dictionary, values to override `torch_params`

        Returns
        -------
        dict
            Dictionary containing variables in both `sk_params` and
            `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in kwargs.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def draw(self) -> Digraph:
        """Draws the wrapped model."""
        first_parameter = next(self.module.parameters())
        input_shape = first_parameter.size()
        y_pred = self.module(torch.rand(input_shape))
        return make_dot(
            y_pred.mean(), params=dict(self.module.named_parameters())
        )

    def initialize_module(self, **kwargs):
        """
        Parameters
        ----------
        module
          The instance or class or callable to be initialized, e.g.
          ``self.module``.
        kwargs : dict
          The keyword arguments to initialize the instance or class. Can be an
          empty dict.
        Returns
        -------
        instance
          The initialized component.
        """
        if not isinstance(self.module_cls, torch.nn.Module):
            self.module = self.module_cls(
                **self._filter_kwargs(self.module_cls, kwargs)
            )

        self.module.to(self.device)
        self.optimizer = self.optimizer_fn(
            self.module.parameters(), lr=self.lr
        )
        self.module_initialized = True


class RollingDeepEstimator(DeepEstimator):
    """
    Abstract base class that implements basic functionality of
    River-compatible PyTorch wrappers including a rolling window to allow the
    model to make predictions based on multiple previous examples.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped. The Module
        should accept parameter `n_features` so that the returned model's
        input shape can be determined based on the number of features in the
        initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss
        function provided by `torch.nn.functional` or one of the following:
        'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    window_size
        Size of the rolling window used for storing previous examples.
    append_predict
        Whether to append inputs passed for prediction to the rolling window.
    **kwargs
        Parameters to be passed to the `Module` or the `optimizer`.
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            **kwargs,
        )

        self.window_size = window_size
        self.append_predict = append_predict
        self._x_window: Deque = collections.deque(maxlen=window_size)
        self._batch_i = 0
