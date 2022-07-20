import abc
import collections
import inspect
from typing import Callable, Type, Union

import torch
from river import base

from river_torch.utils import dict2tensor, get_loss_fn, get_optim_fn
from river_torch.utils.river_compat import list2tensor, scalar2tensor


class DeepEstimator(base.Estimator):
    """
    A PyTorch to River base class that aims to provide basic supervised functionalities.

    Parameters
    ----------
    build_fn
    loss_fn
    optimizer_fn
    learning_rate
    device
    net_params
    """

    def __init__(
        self,
        build_fn: Callable,
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, Callable],
        learning_rate: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **net_params
    ):
        super().__init__()
        self.build_fn = build_fn
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = get_optim_fn(optimizer_fn)
        self.learning_rate = learning_rate
        self.device = device
        self.net_params = net_params
        self.seed = seed
        torch.manual_seed(seed)
        self.net = None

    @classmethod
    def _unit_test_params(cls):
        def build_torch_linear_regressor(n_features):
            net = torch.nn.Sequential(
                torch.nn.Linear(n_features, 1), torch.nn.Sigmoid()
            )
            return net

        yield {
            "build_fn": build_torch_linear_regressor,
            "loss_fn": "mae",
            "optimizer_fn": "sgd",
        }

    @classmethod
    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.
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
    def learn_one(self, x: dict, y: base.typing.ClfTarget):
        """
        Learn on one sample.

        Parameters
        ----------
        x
        y
        """
        return self

    def _filter_torch_params(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `torch_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.net_params.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def _init_net(self, n_features):
        self.net = self.build_fn(
            n_features=n_features, **self._filter_torch_params(self.build_fn)
        )
        self.net.to(self.device)
        self.optimizer = self.optimizer_fn(self.net.parameters(), lr=self.learning_rate)


class RollingDeepEstimator(base.Estimator):
    def __init__(
        self,
        build_fn: Callable,
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, Callable],
        learning_rate: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 1,
        append_predict: bool = True,
        **net_params
    ):
        self.build_fn = build_fn
        self.loss_fn = get_loss_fn(loss_fn=loss_fn)
        self.optimizer_fn = get_optim_fn(optimizer_fn)
        self.learning_rate = learning_rate
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
    def _unit_test_params(cls):
        def build_torch_linear_regressor(n_features):
            net = torch.nn.Sequential(
                torch.nn.Linear(n_features, 1), torch.nn.Sigmoid()
            )
            return net

        yield {
            "loss_fn": "mse",
            "build_fn": build_torch_linear_regressor,
            "optimizer_fn": "sgd",
        }

    @classmethod
    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.
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
    def learn_one(self, x: dict, y: base.typing.ClfTarget):
        """
        Learn on one sample.

        Parameters
        ----------
        x
        y
        """
        return self

    def _filter_torch_params(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `torch_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.net_params.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def _init_net(self, n_features):
        self.net = self.build_fn(
            n_features=n_features, **self._filter_torch_params(self.build_fn)
        )
        self.net.to(self.device)
        self.optimizer = self.optimizer_fn(self.net.parameters(), self.learning_rate)
