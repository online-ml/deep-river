import collections
import copy
import typing

import torch
from river import base

from IncrementalTorch.base import PyTorch2RiverBase, RollingPyTorch2RiverBase


class PyTorch2RiverClassifier(PyTorch2RiverBase, base.Classifier):
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
    >>> from river import compat
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from torch import nn
    >>> from torch import optim
    >>> from torch import manual_seed
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
    >>> model = compat.PyTorch2RiverClassifier(build_fn=build_torch_mlp_classifier,loss_fn='bce',optimizer_fn=optim.Adam,learning_rate=1e-3)
    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()
    >>> evaluate.progressive_val_score(dataset=dataset, model=model, metric=metric)
    Accuracy: 74.38%
    """
    def __init__(self,
                 build_fn,
                 loss_fn: str='ce',
                 optimizer_fn: typing.Type[torch.optim.Optimizer]=torch.optim.SGD,
                 learning_rate=1e-3,
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

        if 'n_classes' in net_params:
            self.n_classes = net_params['n_classes']
            self.variable_classes = False
        else:
            self.n_classes = 1
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            learning_rate=learning_rate,
            seed=seed,
            **net_params
        )

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> base.Classifier:
        self.counter += 1
        self.classes.update([y])

        # check if model is initialized
        if self.net is None:
            self._init_net(len(list(x.values())))

        # check last layer
        if (len(self.classes) != self.n_classes) and (self.variable_classes):
            self.n_classes = len(self.classes)
            layers = list(self.net.children())
            i = -1
            # Get Layer to convert
            layer_to_convert = layers[i]
            while not hasattr(layer_to_convert, 'weight'):
                layer_to_convert = layers[i]
                i -= 1
            if i == -1:
                i = -2

            new_net = list(self.net.children())[:i + 1]
            new_layer = torch.nn.Linear(in_features=layer_to_convert.in_features,
                                        out_features=self.n_classes)
            # copy the original weights back
            with torch.no_grad():
                new_layer.weight[:-1, :] = layer_to_convert.weight
                new_layer.weight[-1:, :] = torch.mean(layer_to_convert.weight, 0)
            # Add new layer
            new_net.append(new_layer)
            # Add non trainable layers
            if i + 1 < -1:
                for layer in layers[i + 2:]:
                    new_net.append(layer)
            self.net = torch.nn.Sequential(*new_net)
            self.optimizer = self.optimizer_fn(self.net.parameters(), self.learning_rate)

        # training process
        if self.variable_classes:
            proba = {c: 0.0 for c in self.classes}
        else:
            proba = {c: 0.0 for c in range(self.n_classes)}
        proba[y] = 1.0
        x = list(x.values())
        y = list(proba.values())

        x = torch.Tensor([x])
        y = torch.Tensor([y])
        self._learn_one(x=x, y=y)
        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.net is None:
            self._init_net(len(list(x.values())))
        x = torch.Tensor(list(x.values()))
        yp = self.net(x).detach().numpy()
        # proba = {c: 0.0 for c in self.classes} # TODO CHECK from here
        if self.variable_classes:
            proba = {c: 0.0 for c in self.classes}
            for idx, val in enumerate(self.classes):
                proba[val] = yp[idx]
        else:
            proba = {c: yp[c] for c in range(self.n_classes)}  ##NEW
            # proba = {c: 0.0 for c in range(self.n_classes)}

        # for idx, val in enumerate(self.classes):
        #    proba[val] = yp[idx]
        return proba


class RollingPyTorch2RiverClassifier(RollingPyTorch2RiverBase, base.Classifier):

    def __init__(self,
                 build_fn,
                 loss_fn: str,
                 optimizer_fn: typing.Type[torch.optim.Optimizer],
                 window_size=1,
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

        Args:
            build_fn:
            loss_fn:
            optimizer_fn:
            window_size:
            learning_rate:
            **net_params:
        """
        self.classes = collections.Counter()
        self.n_classes = 1
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            window_size=window_size,
            learning_rate=learning_rate,
            **net_params
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
            while not hasattr(layer_to_convert, 'weight'):
                layer_to_convert = layers[i]
                i -= 1

            removed = list(self.net.children())[:i + 1]
            new_net = removed
            new_layer = torch.nn.Linear(in_features=layer_to_convert.in_features,
                                        out_features=self.n_classes)
            # copy the original weights back
            with torch.no_grad():
                new_layer.weight[:-1, :] = layer_to_convert.weight
                new_layer.weight[-1:, :] = torch.mean(layer_to_convert.weight, 0)
            new_net.append(new_layer)
            if i + 1 < -1:
                for layer in layers[i + 2:]:
                    new_net.append(layer)
            self.net = torch.nn.Sequential(*new_net)
            self.optimizer = self.optimizer_fn(self.net.parameters(), self.learning_rate)

        # training process
        self._x_window.append(list(x.values()))
        proba = {c: 0.0 for c in self.classes}
        proba[y] = 1.0
        y = list(proba.values())

        if len(self._x_window) == self.window_size:
            x = torch.Tensor([self._x_window])
            [y.append(0.0) for i in range(self.n_classes - len(y))]
            y = torch.Tensor([y])
            self._learn_batch(x=x, y=y)
        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        if self.net is None:
            self._init_net(len(list(x.values())))
        if len(self._x_window) == self.window_size:
            l = copy.deepcopy(self._x_window)
            l.append(list(x.values()))
            if self.append_predict:
                self._x_window.append(list(x.values()))

            x = torch.Tensor([l])
            yp = self.net(x).detach().numpy()
            proba = {c: 0.0 for c in self.classes}
            for idx, val in enumerate(self.classes):
                proba[val] = yp[0][idx]
        else:
            proba = {c: 0.0 for c in self.classes}
        return proba
