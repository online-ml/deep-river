from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
import torch
from river import anomaly
from torch import nn

from deep_river.base import RollingDeepEstimator
from deep_river.utils.tensor_conversion import deque2rolling_tensor


class _TestLSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=30, n_layers=1, batch_first=False):
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

    def forward(self, x):
        output, (h, c) = self.encoder(x)
        return output


class RollingAutoencoder(RollingDeepEstimator, anomaly.base.AnomalyDetector):
    """ """

    def __init__(
        self,
        module: torch.nn.Module,
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

        yield {
            "module": _TestLSTMAutoencoder(30),
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        return {
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def learn_one(self, x: dict, y: Any = None, **kwargs) -> None:
        self._update_observed_features(x)
        self._x_window.append(list(x.values()))

        x_t = deque2rolling_tensor(self._x_window, device=self.device)
        self._learn(x=x_t)

    def learn_many(self, X: pd.DataFrame, y=None) -> None:
        self._update_observed_features(X)

        self._x_window.append(X.values.tolist())
        if len(self._x_window) == self.window_size:
            X_t = deque2rolling_tensor(self._x_window, device=self.device)
            self._learn(x=X_t)

    def score_one(self, x: dict) -> float:
        res = 0.0
        self._update_observed_features(x)
        if len(self._x_window) == self.window_size:
            x_win = self._x_window.copy()
            x_win.append(list(x.values()))
            x_t = deque2rolling_tensor(x_win, device=self.device)
            self.module.eval()
            with torch.inference_mode():
                x_pred = self.module(x_t)
            loss = self.loss_func(x_pred, x_t)
            res = loss.item()

        if self.append_predict:
            self._x_window.append(list(x.values()))
        return res

    def score_many(self, X: pd.DataFrame) -> List[Any]:

        self._update_observed_features(X)
        x_win = self._x_window.copy()
        x_win.append(X.values.tolist())
        if self.append_predict:
            self._x_window.append(X.values.tolist())

        if len(self._x_window) == self.window_size:
            X_t = deque2rolling_tensor(x_win, device=self.device)
            self.module.eval()
            with torch.inference_mode():
                x_pred = self.module(X_t)
            loss = torch.mean(
                self.loss_func(x_pred, x_pred, reduction="none"),
                dim=list(range(1, x_pred.dim())),
            )
            losses = loss.detach().numpy()
            if len(losses) < len(X):
                losses = np.pad(losses, (len(X) - len(losses), 0))
            return losses.tolist()
        else:
            return np.zeros(len(X)).tolist()
