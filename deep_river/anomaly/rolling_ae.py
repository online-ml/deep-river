from typing import Any, Callable, List, Type, Union

import numpy as np
import pandas as pd
import torch
from river import anomaly
from torch import nn

from deep_river.base import RollingDeepEstimator
from deep_river.utils.tensor_conversion import deque2rolling_tensor


class _TestLSTMAutoencoder(nn.Module):
    def __init__(
        self, n_features, hidden_size=10, n_layers=1, batch_first=False
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.time_axis = 1 if batch_first else 0
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=batch_first,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            num_layers=n_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1].view(1, 1, -1)
        x_flipped = torch.flip(x[1:], dims=[self.time_axis])
        input = torch.cat((h, x_flipped), dim=self.time_axis)
        x_hat, _ = self.decoder(input)
        x_hat = torch.flip(x_hat, dims=[self.time_axis])

        return x_hat


class RollingAutoencoder(RollingDeepEstimator, anomaly.base.AnomalyDetector):
    """
    Wrapper for PyTorch autoencoder models that uses the networks
    reconstruction error for scoring the anomalousness of a given example.
    The class also features a rolling window to allow the model to make
    predictions based on the reconstructability of multiple previous examples.

    Parameters
    ----------
    module
        Torch module that builds the autoencoder to be wrapped.
        The module should accept inputs with shape
        `(window_size, batch_size, n_features)`. It should also
        feature a parameter `n_features` used to adapt the network to the
        number of features in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a
        loss function provided by `torch.nn.functional` or one of the
        following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model. Can be an
        optimizer class provided by `torch.optim` or one of the following:
        "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
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
            window_size=window_size,
            append_predict=append_predict,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        """
        Returns a dictionary of parameters to be used for unit testing
        the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing
            the respective class.
        """

        yield {
            "module": _TestLSTMAutoencoder,
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases,
        some estimators might not
        be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
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

    def learn_one(
        self, x: dict, y: Any = None, **kwargs
    ) -> "RollingAutoencoder":
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
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)

        self._x_window.append(list(x.values()))

        if len(self._x_window) == self.window_size:
            x_t = deque2rolling_tensor(self._x_window, device=self.device)
            self._learn(x=x_t)
        return self

    def learn_many(self, X: pd.DataFrame, y=None) -> "RollingAutoencoder":
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        X
            Input batch of examples.

        y
            Should be None

        Returns
        -------
        RollingAutoencoder
            The estimator itself.
        """
        if not self.module_initialized:
            self.kwargs["n_features"] = len(X.columns)
            self.initialize_module(**self.kwargs)

        self._x_window.append(X.values.tolist())
        if len(self._x_window) == self.window_size:
            X_t = deque2rolling_tensor(self._x_window, device=self.device)
            self._learn(x=X_t)
        return self

    def score_one(self, x: dict) -> float:
        res = 0.0
        if not self.module_initialized:
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)

        if len(self._x_window) == self.window_size:
            x_win = self._x_window.copy()
            x_win.append(list(x.values()))
            x_t = deque2rolling_tensor(x_win, device=self.device)
            self.module.eval()
            with torch.inference_mode():
                x_pred = self.module(x_t)
            loss = self.loss_fn(x_pred, x_t)
            res = loss.item()

        if self.append_predict:
            self._x_window.append(list(x.values()))
        return res

    def score_many(self, X: pd.DataFrame) -> List[Any]:
        if not self.module_initialized:
            self.kwargs["n_features"] = len(X.columns)
            self.initialize_module(**self.kwargs)

        x_win = self._x_window.copy()
        x_win.append(X.values.tolist())
        if self.append_predict:
            self._x_window.append(X.values.tolist())

        if len(self._x_window) == self.window_size:
            self.module.eval()
            X_t = deque2rolling_tensor(x_win, device=self.device)
            with torch.inference_mode():
                x_pred = self.module(X_t)
            loss = torch.mean(
                self.loss_fn(x_pred, x_pred, reduction="none"),
                dim=list(range(1, x_pred.dim())),
            )
            losses = loss.detach().numpy()
            if len(losses) < len(X):
                losses = np.pad(losses, (len(X) - len(losses), 0))
            return losses.tolist()
        else:
            return np.zeros(len(X)).tolist()
