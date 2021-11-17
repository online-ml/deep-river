import inspect
from typing import Type
import torch
from river import base, utils


class PyTorch2RiverBase(base.Estimator):
    def __init__(
            self,
            build_fn,
            loss_fn: Type[torch.nn.modules.loss._Loss],
            optimizer_fn: Type[torch.optim.Optimizer],
            learning_rate=1e-3,
            seed=42,
            **net_params):
        self.build_fn = build_fn
        self.loss_fn = loss_fn
        self.loss = loss_fn()
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.net_params = net_params
        self.seed = seed
        torch.manual_seed(seed)
        self.net = None

    def _learn_one(self, x: torch.Tensor, y: torch.Tensor):
        self.net.zero_grad()
        y_pred = self.net(x)
        #depending on loss function
        try:
            loss = self.loss(y_pred, y)
        except:
            loss = self.loss(y_pred, torch.argmax(y, 1)) # TODO CHECK loss
        loss.backward()
        self.optimizer.step()


    def learn_one(self, x, y):
        if self.net is None:
            self._init_net(n_features=len(list(x.values())))

        x = torch.Tensor([list(x.values())])
        y = torch.Tensor([[y]])
        self._learn_one(x=x,y=y)
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
        self.net = self.build_fn(n_features=n_features, **self._filter_torch_params(self.build_fn))
        self.optimizer = self.optimizer_fn(self.net.parameters(), lr = self.learning_rate)


class RollingPyTorch2RiverBase(base.Estimator):
    def __init__(
            self,
            build_fn,
            loss_fn: Type[torch.nn.modules.loss._Loss],
            optimizer_fn: Type[torch.optim.Optimizer],
            learning_rate=1e-3,
            window_size=1,
            seed=42,
            **net_params):
        self.build_fn = build_fn
        self.loss_fn = loss_fn
        self.loss = loss_fn()
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.net_params = net_params
        self.seed = seed
        torch.manual_seed(seed)

        self._x_window = utils.Window(window_size)
        self._batch_i = 0
        self.net = None

    def _learn_batch(self, x: torch.Tensor, y: torch.Tensor):
        y_pred = self.net(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def learn_one(self, x, y):
        self._x_window.append(list(x.values()))
        if self.net is None:
            self._init_net(n_features=len(list(x.values())))

        if len(self._x_window) == self.window_size:
            x = torch.Tensor([self._x_window.values])
            y = torch.Tensor([[y]])
            self._learn_batch(x=x, y=y)
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
        self.net = self.build_fn(n_features=n_features, **self._filter_torch_params(self.build_fn))
        self.optimizer = self.optimizer_fn(self.net.parameters(), self.learning_rate)
