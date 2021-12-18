import copy
from typing import Type

from river import base, utils
import torch
import inspect
import numpy as np

from torch._C import device


class TorchAE(base.AnomalyDetector):
    def __init__(
            self,
            build_fn,
            loss_fn: Type[torch.nn.modules.loss._Loss],
            optimizer_fn: Type[torch.optim.Optimizer],
            learning_rate=1e-3,
            seed=42,
            device='cpu',
            **net_params):
        self.build_fn = build_fn
        self.loss_fn = loss_fn
        self.loss = loss_fn()
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.device = device
        self.net_params = net_params
        self.seed = seed
        torch.manual_seed(seed)

        self.net = None

    def score_one(self, x):
        if isinstance(x, dict):
            x = torch.Tensor([list(x.values())])

        if self.net is None:
            self._init_net(n_features=x.shape[0])

        x = x.to(self.device)
        return self._score_one(x=x)#.item() todo test if item() is necessary?

    def _score_one(self, x: dict) -> float:
        self.net.eval()
        with torch.inference_mode():
            x_pred = self.net(x)
            loss = self.loss(x_pred, x)
        return loss

    def _learn_one(self, x: torch.Tensor):
        self.net.train()
        self.net.zero_grad()

        x_pred = self.net(x)
        loss = self.loss(x_pred, x)
        loss.backward()
        self.optimizer.step()

    def learn_one(self, x):
        if isinstance(x, dict):
            x = torch.Tensor([list(x.values())])

        if self.net is None:
            self._init_net(n_features=x.shape[1])

        x = x.to(self.device)
        self._learn_one(x=x)
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
            n_features=n_features, **self._filter_torch_params(self.build_fn))
        self.net.to(self.device)
        self.optimizer = self.optimizer_fn(
            self.net.parameters(), lr=self.learning_rate)


class RollingTorchAE(base.AnomalyDetector):
    def __init__(
            self,
            build_fn,
            loss_fn: Type[torch.nn.modules.loss._Loss],
            optimizer_fn: Type[torch.optim.Optimizer],
            learning_rate=1e-3,
            window_size = 10,
            seed=42,
            device='cpu',
            **net_params):
        self.build_fn = build_fn
        self.loss_fn = loss_fn
        self.loss = loss_fn()
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.device = device
        self.net_params = net_params
        self.seed = seed
        torch.manual_seed(seed)

        self._x_window = utils.Window(window_size)
        self._batch_i = 0
        self.net = None

    def score_one(self, x) -> float:
        if isinstance(x, dict):
            x = torch.Tensor([list(x.values())])
        if self.net is None:
            self._init_net(n_features=x.shape[0])
        if len(self._x_window) == self.window_size:
            l = copy.deepcopy(self._x_window.values)
            l.append(list(x.values()))
            return self._score_one(x=l)
        else:
            return 0.0

    def _score_one(self, x) -> float:
        self.net.eval()
        with torch.inference_mode():
            x_pred = self.net(x)
            loss = self.loss(x_pred, x)
        return loss

    def _learn_batch(self, x: torch.Tensor):
        self.net.train()
        self.net.zero_grad()

        x_pred = self.net(x)
        loss = self.loss(x_pred, x)
        loss.backward()
        self.optimizer.step()

    def learn_one(self, x):
        self._x_window.append(list(x.values()))
        if isinstance(x, dict):
            x = torch.Tensor([list(x.values())])

        if self.net is None:
            self._init_net(n_features=x.shape[1])

        if len(self._x_window) == self.window_size:
            x = torch.Tensor([self._x_window.values])
            self._learn_batch(x=x)
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
            n_features=n_features, **self._filter_torch_params(self.build_fn))
        self.net.to(self.device)
        self.optimizer = self.optimizer_fn(
            self.net.parameters(), lr=self.learning_rate)


class SklearnAnomalyDetector(base.AnomalyDetector):
    def __init__(self, build_fn) -> None:
        super().__init__()
        self.model = None
        self.build_fn = build_fn

    def score_one(self, x):

        if self.model is None:
            self.model = self.build_fn()

        x = x.reshape(1, -1)
        return self._score_one(x=x)

    def _score_one(self, x: dict) -> float:
        return self.model.score_samples(x)

    def _learn_one(self, x: torch.Tensor):
        self.model.partial_fit(x)

    def learn_one(self, x):

        if self.model is None:
            self.model = self.build_fn()

        x = x.reshape(1, -1)
        self._learn_one(x=x)
        return self
