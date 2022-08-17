import abc
import collections
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
from river import anomaly

from river_torch.base import RollingDeepEstimator
from river_torch.utils.tensor_conversion import (df2rolling_tensor,
                                                 dict2rolling_tensor)


class RollingAutoencoder(RollingDeepEstimator, anomaly.base.AnomalyDetector):
    """
    Wrapper for PyTorch autoencoder models that uses the networks reconstruction error for scoring the anomalousness of a given example. The class also features a rolling window to allow the model to make predictions based on the reconstructability of multiple previous examples.

    Parameters
    ----------
    build_fn
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
        build_fn: Callable,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **net_params,
    ):
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            **net_params,
        )
        self.append_predict = append_predict
        self.window_size = window_size
        self._x_window = collections.deque(maxlen=window_size)
        self._batch_i = 0

    def _learn(self, x: torch.Tensor):
        self.net.train()

        x_pred = self.net(x)
        loss = self.loss_fn(x_pred, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_one(self, x: dict) -> "RollingAutoencoder":
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
        if self.net is None:
            self._init_net(n_features=len(x))

        x = dict2rolling_tensor(x, self._x_window, device=self.device)
        if x is not None:
            self._learn(x=x)
        return self

    def learn_many(self, x: pd.DataFrame) -> "RollingAutoencoder":
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        x
            Input batch of examples.

        Returns
        -------
        RollingAutoencoder
            The estimator itself.
        """
        if self.net is None:
            self._init_net(n_features=len(x.columns))

        x = df2rolling_tensor(x, self._x_window, device=self.device)
        if x is not None:
            self._learn(x=x)
        return self

    def score_one(self, x: dict) -> float:
        if self.net is None:
            self._init_net(len(x))

        x = dict2rolling_tensor(x, self._x_window, device=self.device)
        if x is not None:
            self.net.eval()
            x_pred = self.net(x)
            loss = self.loss_fn(x_pred, x)
            return loss.item()
        else:
            return 0.0

    def score_many(self, x: pd.DataFrame) -> float:
        if self.net is None:
            self._init_net(n_features=len(x.columns))

        batch = df2rolling_tensor(
            x, self._x_window, device=self.device, update_window=self.append_predict
        )

        if batch is not None:
            self.net.eval()
            x_pred = self.net(batch)
            loss = torch.mean(
                self.loss_fn(x_pred, batch, reduction="none"),
                dim=list(range(1, batch.dim())),
            )
            losses = loss.detach().numpy()
            if len(losses) < len(x):
                losses = np.pad(losses, (len(x) - len(losses), 0))
            return losses.tolist()
        else:
            return np.zeros(len(x)).tolist()
