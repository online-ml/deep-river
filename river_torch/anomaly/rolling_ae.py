import abc
import collections
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from river import anomaly

from river_torch.base import RollingDeepEstimator
from river_torch.utils.tensor_conversion import (df2rolling_tensor,
                                                 dict2rolling_tensor)


class RollingAutoencoder(RollingDeepEstimator, anomaly.base.AnomalyDetector):
    """
    Wrapper for PyTorch autoencoder models that uses the networks reconstruction error for scoring the anomalousness of a given example. The class also features a rolling window to allow the model to make predictions based on the reconstructability of multiple previous examples.

    Parameters
    ----------
    module
        Function that builds the autoencoder to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example.
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
        module: Union[torch.nn.Module, type(torch.nn.Module)],
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
        self.append_predict = append_predict
        self.window_size = window_size
        self._x_window = collections.deque(maxlen=window_size)
        self._batch_i = 0

    @classmethod
    def _unit_test_params(cls) -> dict:
        """
        Returns a dictionary of parameters to be used for unit testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the respective class.
        """

        class MyAutoEncoder(torch.nn.Module):
            def __init__(self, n_features, latent_dim=3):
                super(MyAutoEncoder, self).__init__()
                self.linear1 = nn.Linear(n_features, latent_dim)
                self.nonlin = torch.nn.LeakyReLU()
                self.linear2 = nn.Linear(latent_dim, n_features)

            def forward(self, X, **kwargs):
                X = self.linear1(X)
                X = self.nonlin(X)
                X = self.linear2(X)
                return torch.nn.functional.sigmoid(X)

        yield {
            "module": MyAutoEncoder,
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

    def _learn(self, x: torch.Tensor):
        self.module.train()
        x_pred = self.module(x)
        loss = self.loss_fn(x_pred, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_one(self, x: dict, y: None) -> "RollingAutoencoder":
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        RollingAutoencoder
            The estimator itself.
        """
        if not self.module_initialized:
            self.kwargs['n_features'] = len(x)
            self.initialize_module(**self.kwargs)

        x = dict2rolling_tensor(x, self._x_window, device=self.device)
        if x is not None:
            self._learn(x=x)
        return self

    def learn_many(self, X: pd.DataFrame, y:None) -> "RollingAutoencoder":
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        X
            Input batch of examples.

        y
            should be None

        Returns
        -------
        RollingAutoencoder
            The estimator itself.
        """
        if not self.module_initialized:
            self.kwargs['n_features'] = len(X.columns)
            self.initialize_module(**self.kwargs)

        X = df2rolling_tensor(X, self._x_window, device=self.device)
        if X is not None:
            self._learn(x=X)
        return self

    def score_one(self, x: dict) -> float:
        if not self.module_initialized:
            self.kwargs['n_features'] = len(x)
            self.initialize_module(**self.kwargs)

        x = dict2rolling_tensor(x, self._x_window, device=self.device)
        if x is not None:
            self.module.eval()
            x_pred = self.module(x)
            loss = self.loss_fn(x_pred, x)
            return loss.item()
        else:
            return 0.0

    def score_many(self, X: pd.DataFrame) -> float:
        if not self.module_initialized:
            self.kwargs['n_features'] = len(X.columns)
            self.initialize_module(**self.kwargs)

        batch = df2rolling_tensor(
            X, self._x_window, device=self.device, update_window=self.append_predict
        )

        if batch is not None:
            self.module.eval()
            x_pred = self.module(batch)
            loss = torch.mean(
                self.loss_fn(x_pred, batch, reduction="none"),
                dim=list(range(1, batch.dim())),
            )
            losses = loss.detach().numpy()
            if len(losses) < len(X):
                losses = np.pad(losses, (len(X) - len(losses), 0))
            return losses.tolist()
        else:
            return np.zeros(len(X)).tolist()
