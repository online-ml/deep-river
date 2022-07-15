import collections
import copy
import typing

import torch
from river import base

from river_torch.base import DeepEstimator, RollingDeepEstimator
from river_torch.utils.river_compat import dict2tensor, list2tensor, scalar2tensor


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
    ...         nn.Linear(5, 5),
    ...         nn.Linear(5, 5),
    ...         nn.Linear(5, 5),
    ...         nn.Linear(5, 1),
    ...         nn.Sigmoid()
    ...     )
    ...     return net
    ...
    >>> model = Classifier(build_fn=build_torch_mlp_classifier,loss_fn='bce',optimizer_fn=optim.Adam,learning_rate=1e-3)
    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()
    >>> evaluate.progressive_val_score(dataset=dataset, model=model, metric=metric)
    Accuracy: 74.38%
    """

    def __init__(
        self,
        build_fn,
        loss_fn: str = "ce",
        optimizer_fn="sgd",
        learning_rate=1e-3,
        device="cpu",
        seed=42,
        **net_params,
    ):
        """

        Args:
            build_fn:
            loss_fn:
            optimizer_fn:
            learning_rate:
            seed:
            **net_params:
        """
        self.classes = collections.Counter()
        self.variable_classes = True
        self.counter = 0

        if "n_classes" in net_params:
            self.n_classes = net_params["n_classes"]
            self.variable_classes = False
        else:
            self.n_classes = 1
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
        self.counter += 1
        self.classes.update([y])
        x = dict2tensor(x, device=self.device)
        # check if model is initialized
        if self.net is None:
            self._init_net(x.shape[1])

        # check last layer
        if (len(self.classes) != self.n_classes) and (self.variable_classes):
            self.n_classes = len(self.classes)
            layers = list(self.net.children())
            i = -1
            # Get Layer to convert
            layer_to_convert = layers[i]
            while not hasattr(layer_to_convert, "weight"):
                layer_to_convert = layers[i]
                i -= 1
            if i == -1:
                i = -2

            new_net = list(self.net.children())[: i + 1]
            new_layer = torch.nn.Linear(
                in_features=layer_to_convert.in_features, out_features=self.n_classes
            )
            # copy the original weights back
            with torch.no_grad():
                new_layer.weight[:-1, :] = layer_to_convert.weight
                new_layer.weight[-1:, :] = torch.mean(layer_to_convert.weight, 0)
            # Add new layer
            new_net.append(new_layer)
            # Add non trainable layers
            if i + 1 < -1:
                for layer in layers[i + 2 :]:
                    new_net.append(layer)
            self.net = torch.nn.Sequential(*new_net)
            self.optimizer = self.optimizer_fn(
                self.net.parameters(), self.learning_rate
            )

        # training process
        if self.variable_classes:
            proba = {c: 0.0 for c in self.classes}
        else:
            proba = {c: 0.0 for c in range(self.n_classes)}
        proba[y] = 1.0

        y = dict2tensor(proba, device=self.device)

        self._learn_one(x=x, y=y)
        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        x = dict2tensor(x, device=self.device)
        if self.net is None:
            self._init_net(x.shape[1])
        yp = self.net(x).detach().numpy()[0]

        if self.variable_classes:
            proba = {c: 0.0 for c in self.classes}
            for idx, val in enumerate(self.classes):
                proba[val] = yp[idx]
        else:
            proba = {c: yp[c] for c in range(self.n_classes)}

        return proba


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
        optimizer_fn= "sgd",
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
        self.classes = collections.Counter()
        self.n_classes = 1
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
        self.classes.update([y])

        # check if model is initialized
        if self.net is None:
            self._init_net(len(list(x.values())))

        # check last layer
        if len(self.classes) != self.n_classes:
            self.n_classes = len(self.classes)
            layers = list(self.net.children())
            i = -1
            layer_to_convert = layers[i]
            while not hasattr(layer_to_convert, "weight"):
                layer_to_convert = layers[i]
                i -= 1

            removed = list(self.net.children())[: i + 1]
            new_net = removed
            new_layer = torch.nn.Linear(
                in_features=layer_to_convert.in_features, out_features=self.n_classes
            )
            # copy the original weights back
            with torch.no_grad():
                new_layer.weight[:-1, :] = layer_to_convert.weight
                new_layer.weight[-1:, :] = torch.mean(layer_to_convert.weight, 0)
            new_net.append(new_layer)
            if i + 1 < -1:
                for layer in layers[i + 2 :]:
                    new_net.append(layer)
            self.net = torch.nn.Sequential(*new_net)
            self.optimizer = self.optimizer_fn(
                self.net.parameters(), self.learning_rate
            )

        # training process
        self._x_window.append(list(x.values()))
        proba = {c: 0.0 for c in self.classes}
        proba[y] = 1.0
        y = list(proba.values())

        if len(self._x_window) == self.window_size:
            x = list2tensor(self._x_window, device=self.device)
            [y.append(0.0) for i in range(self.n_classes - len(y))]
            y = scalar2tensor(y, device=self.device)
            self._learn_batch(x=x, y=y)
        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.net is None:
            self._init_net(len(list(x.values())))
        if len(self._x_window) == self.window_size:
            if self.append_predict:
                self._x_window.append(list(x.values()))
                x = self._x_window
            else:
                x = copy.deepcopy(self._x_window)
                x.append(list(x.values()))

            x = list2tensor(x, device=self.device)
            yp = self.net(x).detach().numpy()
            proba = {c: 0.0 for c in self.classes}
            for idx, val in enumerate(self.classes):
                proba[val] = yp[0][idx]
        else:
            proba = {c: 0.0 for c in self.classes}
        return proba
