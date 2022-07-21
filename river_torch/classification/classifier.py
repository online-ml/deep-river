import collections
import copy
import math
import typing
from matplotlib.pyplot import axis

import torch
from torch.nn import init, parameter
import torch.nn.functional as F
from river import base

from river_torch.base import DeepEstimator, RollingDeepEstimator
from river_torch.utils.layers import find_output_layer
from river_torch.utils.river_compat import (
    dict2tensor,
    list2tensor,
    output2proba,
    scalar2tensor,
    target2onehot,
)


class Classifier(DeepEstimator, base.Classifier):
    """
    A river classifier that integrates neural Networks from PyTorch.
    Parameters
    ----------
    build_fn
    loss_fn
    optimizer_fn
    learning_rate
    seed
    net_params

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from torch import nn
    >>> from torch import optim
    >>> from torch import manual_seed
    >>> from river_torch.classification import Classifier
    >>> _ = manual_seed(0)
    >>> def build_torch_mlp_classifier(n_features):
    ...     net = nn.Sequential(
    ...         nn.Linear(n_features, 5),
    ...         nn.ReLU(),
    ...         nn.Linear(5, 2),
    ...         nn.Softmax(dim=-1),
    ...     )
    ...     return net
    ...
    >>> model = Classifier(build_fn=build_torch_mlp_classifier, loss_fn='bce',optimizer_fn=optim.Adam, learning_rate=1e-3)
    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()
    >>> evaluate.progressive_val_score(dataset=dataset, model=model, metric=metric)
    Accuracy: 79.82%
    """

    def __init__(
        self,
        build_fn,
        loss_fn: str = "bce",
        optimizer_fn="sgd",
        learning_rate=1e-3,
        device="cpu",
        seed=42,
        **net_params,
    ):
        self.observed_classes = []
        self.output_layer = None
        super().__init__(
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            build_fn=build_fn,
            device=device,
            learning_rate=learning_rate,
            seed=seed,
            **net_params,
        )

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        # check if model is initialized
        if self.net is None:
            self._init_net(len(x))
        x = dict2tensor(x, device=self.device)

        # check last layer
        if y not in self.observed_classes:
            self.observed_classes.append(y)

        if self.output_layer is None:
            self.output_layer = find_output_layer(self.net)

        out_features_target = (
            len(self.observed_classes) if len(self.observed_classes) > 2 else 1
        )
        n_classes_to_add = out_features_target - self.output_layer.out_features
        if n_classes_to_add > 0:
            self._add_output_dims(n_classes_to_add)

        y = target2onehot(
            y,
            self.observed_classes,
            self.output_layer.out_features,
            device=self.device,
        )
        self.net.train()
        self._learn_one(x=x, y=y)
        return self

    def _add_output_dims(self, n_classes_to_add: int):

        new_weights = torch.empty(n_classes_to_add, self.output_layer.in_features)
        init.kaiming_uniform_(new_weights, a=math.sqrt(5))
        self.output_layer.weight = parameter.Parameter(
            torch.cat([self.output_layer.weight, new_weights], axis=0)
        )

        if self.output_layer.bias is not None:
            new_bias = torch.empty(n_classes_to_add)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.output_layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(new_bias, -bound, bound)
            self.output_layer.bias = parameter.Parameter(
                torch.cat([self.output_layer.bias, new_bias], axis=0)
            )
        self.output_layer.out_features += n_classes_to_add
        self.optimizer = self.optimizer_fn(self.net.parameters(), lr=self.learning_rate)

    def _learn_one(self, x: torch.TensorType, y: torch.TensorType):
        self.optimizer.zero_grad()
        y_pred = self.net(x)[0]
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.net is None:
            self._init_net(len(x))
        x = dict2tensor(x, device=self.device)
        self.net.eval()
        y_pred = self.net(x)
        return output2proba(y_pred, self.observed_classes)


class RollingClassifier(RollingDeepEstimator, base.Classifier):
    """
    A Rolling Window PyTorch to River Regressor
    Parameters
    ----------
    build_fn
    loss_fn
    optimizer_fn
    window_size
    learning_rate
    net_params
    """

    def __init__(
        self,
        build_fn,
        loss_fn: str = "ce",
        optimizer_fn="sgd",
        window_size=1,
        device="cpu",
        learning_rate=1e-3,
        **net_params,
    ):
        """
        A Rolling Window PyTorch to River Classifier

        Parameters
        ----------
        build_fn
        loss_fn
        optimizer_fn
        window_size
        learning_rate
        net_params
        """
        self.observed_classes = []
        self.output_layer = None
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            window_size=window_size,
            device=device,
            learning_rate=learning_rate,
            **net_params,
        )

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        # check if model is initialized
        if self.net is None:
            self._init_net(len(x))

        self._x_window.append(list(x.values()))

        # check last layer
        if y not in self.observed_classes:
            self.observed_classes.append(y)

        if self.output_layer is None:
            self.output_layer = find_output_layer(self.net)

        out_features_target = (
            len(self.observed_classes) if len(self.observed_classes) > 2 else 1
        )
        n_classes_to_add = out_features_target - self.output_layer.out_features
        if n_classes_to_add > 0:
            self._add_output_dims(n_classes_to_add)

        # training process
        if len(self._x_window) == self.window_size:
            self.net.train()
            x = list2tensor(self._x_window, device=self.device)
            y = target2onehot(
                y,
                self.observed_classes,
                self.output_layer.out_features,
                device=self.device,
            )
            self._learn_window(x=x, y=y)
        return self

    def _add_output_dims(self, n_classes_to_add: int):
        new_weights = torch.empty(n_classes_to_add, self.output_layer.in_features)
        init.kaiming_uniform_(new_weights, a=math.sqrt(5))
        self.output_layer.weight = parameter.Parameter(
            torch.cat([self.output_layer.weight, new_weights], axis=0)
        )

        if self.output_layer.bias is not None:
            new_bias = torch.empty(n_classes_to_add)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.output_layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(new_bias, -bound, bound)
            self.output_layer.bias = parameter.Parameter(
                torch.cat([self.output_layer.bias, new_bias], axis=0)
            )
        self.output_layer.out_features += n_classes_to_add
        self.optimizer = self.optimizer_fn(self.net.parameters(), lr=self.learning_rate)

    def _learn_window(self, x: torch.TensorType, y: torch.TensorType):
        self.optimizer.zero_grad()
        y_pred = self.net(x)[0]
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.net is None:
            self._init_net(len(x))

        if self.append_predict:
            self._x_window.append(list(x.values()))
            x = self._x_window
        else:
            x = copy.deepcopy(self._x_window)
            x.append(list(x.values()))

        if len(self._x_window) == self.window_size:
            x = list2tensor(x, device=self.device)
            self.net.eval()
            y_pred = self.net(x)
            proba = output2proba(y_pred, self.observed_classes)
        else:
            mean_proba = 1 / len(self.observed_classes)
            proba = {c: mean_proba for c in self.observed_classes}
        return proba
