import abc
import collections
import inspect
from typing import Callable, Union

import torch
from river import base

from river_torch.utils import get_loss_fn, get_optim_fn


class DeepEstimator(base.Estimator):
    """
    Abstract base class that implements basic functionality of River-compatible PyTorch wrappers.

    Parameters
    ----------
    module
        #todo change description
        Function that builds the PyTorch model to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss function provided by `torch.nn.functional` or one of the following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy', 'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model. Can be an optimizer class provided by `torch.optim` or one of the following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **net_params
        Parameters to be passed to the `build_fn` function aside from `n_features`.
    """

    def __init__(
        self,
        module: Union[torch.nn.Module,type[torch.nn.Module]],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        self.module = module
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = get_optim_fn(optimizer_fn)
        self.lr = lr
        self.device = device
        self.kwargs = kwargs
        self.seed = seed
        torch.manual_seed(seed)

    @classmethod
    def _unit_test_params(cls) -> dict:
        """
        Returns a dictionary of parameters to be used for unit testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the respective class.
        """

        class MyModule(torch.nn.Module):
            def __init__(self, num_units=10, nonlin=torch.nn.ReLU()):
                super(MyModule, self).__init__()

                self.dense0 = torch.nn.Linear(20, num_units)
                self.nonlin = nonlin
                self.dropout = torch.nn.Dropout(0.5)
                self.dense1 = torch.nn.Linear(num_units, num_units)
                self.output = torch.nn.Linear(num_units, 2)
                self.softmax = torch.nn.Softmax(dim=-1)

            def forward(self, X, **kwargs):
                X = self.nonlin(self.dense0(X))
                X = self.dropout(X)
                X = self.nonlin(self.dense1(X))
                X = self.softmax(self.output(X))
                return X

        yield {
            "module": MyModule,
            "loss_fn": "l1",
            "optimizer_fn": "sgd",
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.

        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    @abc.abstractmethod
    def learn_one(self, x, y) -> "DeepEstimator":
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
        return self

    def _filter_kwargs(self, fn: Callable, override=None) -> dict:
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
            Dictionary containing variables in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.kwargs.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def initialize_module(self):
        """       Parameters
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
        is_init = isinstance(self.module, torch.nn.Module)
        if not is_init:
            self.module = self.module(**self._filter_kwargs(self.module))

        self.module.to(self.device)
        self.optimizer = self.optimizer_fn(self.module.parameters(), lr=self.lr)

class RollingDeepEstimator(base.Estimator):
    """
    Abstract base class that implements basic functionality of River-compatible PyTorch wrappers including a rolling window to allow the model to make predictions based on multiple previous examples.

    Parameters
    ----------
    build_fn
        Function that builds the PyTorch model to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss function provided by `torch.nn.functional` or one of the following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy', 'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model. Can be an optimizer class provided by `torch.optim` or one of the following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
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
    **net_params
        Parameters to be passed to the `build_fn` function aside from `n_features`.
    """

    def __init__(
        self,
        build_fn: Callable,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **net_params
    ):
        self.build_fn = build_fn
        self.loss_fn = get_loss_fn(loss_fn=loss_fn)
        self.optimizer_fn = get_optim_fn(optimizer_fn)
        self.lr = lr
        self.device = device
        self.window_size = window_size
        self.net_params = net_params
        self.seed = seed
        self.append_predict = append_predict
        torch.manual_seed(seed)

        self._x_window = collections.deque(maxlen=window_size)
        self._batch_i = 0
        self.net = None

    @classmethod
    def _unit_test_params(cls) -> dict:
        """
        Returns a dictionary of parameters to be used for unit testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the respective class.
        """

        def build_torch_linear_regressor(n_features):
            net = torch.nn.Sequential(
                torch.nn.Linear(n_features, 1), torch.nn.Sigmoid()
            )
            return net

        yield {
            "build_fn": build_torch_linear_regressor,
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.

        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    @abc.abstractmethod
    def learn_one(self, x, y) -> "RollingDeepEstimator":
        """
        Performs one step of training with a sliding window of the most recent examples.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        RollingDeepEstimator
            The estimator itself.
        """
        return self

    @abc.abstractmethod
    def learn_many(self, x, y) -> "RollingDeepEstimator":
        """
        Performs one step of training with a batch of sliding windows of the most recent examples.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        RollingDeepEstimator
            The estimator itself.
        """
        return self

    def _filter_torch_params(self, fn, override=None) -> dict:
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
            Dictionary containing variables in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.net_params.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def _init_net(self, n_features) -> None:
        """
        Initializes the PyTorch model.

        Parameters
        ----------
        n_features
            Number of input features of the model to initialize.

        """
        self.net = self.build_fn(
            n_features=n_features, **self._filter_torch_params(self.build_fn)
        )
        self.net.to(self.device)
        self.optimizer = self.optimizer_fn(self.net.parameters(), self.lr)
