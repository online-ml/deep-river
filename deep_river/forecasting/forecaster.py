import collections
from typing import Callable, Deque, Type, Union

import torch
from river.time_series import base as time_series_base
from torch import nn, optim

from deep_river.base import DeepEstimator


class _TestModule(nn.Module):
    def __init__(self, n_features: int = 4):
        super().__init__()
        self.n_features = n_features
        self.dense0 = nn.Linear(n_features, 8)
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(8, 1)

    def forward(self, x, **kwargs):
        return self.dense1(self.activation(self.dense0(x)))


class DeepForecaster(DeepEstimator, time_series_base.Forecaster):
    """Incremental PyTorch forecaster compatible with River's time-series API.

    ``DeepForecaster`` learns a one-step-ahead model from the recent endogenous
    target history and optional exogenous features. Multi-step forecasts are
    produced autoregressively by feeding each predicted value back into a copied
    history buffer.

    Parameters
    ----------
    module
        PyTorch module. In flat mode it receives a tensor of shape
        ``(1, window_size + n_exogenous_features)``. In sequence mode it receives
        ``(window_size, 1, 1 + n_exogenous_features)``.
    window_size
        Number of past target values used as autoregressive context.
    is_sequence_model
        Whether ``module`` expects sequence-shaped input.
    loss_fn, optimizer_fn, lr, is_feature_incremental, device, seed,
    gradient_clip_value, **kwargs
        Standard deep-river estimator configuration.
    """

    def __init__(
        self,
        module: nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 10,
        is_sequence_model: bool = False,
        gradient_clip_value: float | None = 1.0,
        **kwargs,
    ):
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        self.window_size = window_size
        self.is_sequence_model = is_sequence_model
        self._y_window: Deque[float] = collections.deque(maxlen=window_size)

        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    def learn_one(self, y: float, x: dict | None = None) -> None:
        """Update the model with one target and optional exogenous features."""
        x = x or {}
        self._update_observed_features(x)
        x_t = self._make_input_tensor(self._y_window, x)
        y_t = torch.tensor([[y]], dtype=torch.float32, device=self.device)
        self._learn(x_t, y_t)
        self._y_window.append(float(y))

    def forecast(self, horizon: int, xs: list[dict] | None = None) -> list:
        """Forecast ``horizon`` steps ahead using autoregressive rollouts."""
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        if xs is None:
            xs = [{} for _ in range(horizon)]
        if len(xs) != horizon:
            raise ValueError(
                "the length of xs should be equal to the specified horizon"
            )

        y_window = collections.deque(self._y_window, maxlen=self.window_size)
        forecasts = []

        self.module.eval()
        with torch.inference_mode():
            for x in xs:
                x = x or {}
                self._update_observed_features(x)
                x_t = self._make_input_tensor(y_window, x)
                y_pred = self.module(x_t)
                if isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.detach().view(-1)[-1].cpu().item()
                else:
                    y_pred = float(y_pred)
                forecasts.append(y_pred)
                y_window.append(float(y_pred))

        return forecasts

    def _update_observed_features(self, x):
        """Track exogenous features and expand the first layer when requested."""
        if not x:
            return False

        prev_feature_count = len(self.observed_features)
        self.observed_features.update(x.keys())

        if self.is_feature_incremental and self.input_layer:
            target_size = self._target_input_size()
            if self._get_input_size() < target_size:
                self._expand_layer(
                    self.input_layer, target_size=target_size, output=False
                )

        return len(self.observed_features) > prev_feature_count

    def _target_input_size(self) -> int:
        endogenous_size = 1 if self.is_sequence_model else self.window_size
        return endogenous_size + len(self.observed_features)

    def _n_exogenous_inputs(self) -> int:
        endogenous_size = 1 if self.is_sequence_model else self.window_size
        return max(0, self._get_input_size() - endogenous_size)

    def _lag_values(self, y_window: Deque[float]) -> list[float]:
        values = list(y_window)[-self.window_size :]
        return [0.0] * (self.window_size - len(values)) + values

    def _exogenous_values(self, x: dict) -> list[float]:
        values = [
            float(x.get(feature, 0.0) or 0.0) for feature in self.observed_features
        ]
        n_exogenous_inputs = self._n_exogenous_inputs()
        if len(values) < n_exogenous_inputs:
            values.extend([0.0] * (n_exogenous_inputs - len(values)))
        return values[:n_exogenous_inputs]

    def _make_input_tensor(self, y_window: Deque[float], x: dict) -> torch.Tensor:
        lags = self._lag_values(y_window)
        exogenous = self._exogenous_values(x)

        if self.is_sequence_model:
            rows = [[lag, *exogenous] for lag in lags]
            return torch.tensor(
                rows, dtype=torch.float32, device=self.device
            ).unsqueeze(1)

        row = [*lags, *exogenous]
        return torch.tensor([row], dtype=torch.float32, device=self.device)

    def _get_runtime_state(self):
        state = super()._get_runtime_state()
        state["target_window"] = list(self._y_window)
        return state

    def _restore_runtime_state(self, state):
        super()._restore_runtime_state(state)
        if "target_window" in state:
            self._y_window = collections.deque(
                state["target_window"], maxlen=getattr(self, "window_size", 10)
            )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "module": _TestModule(n_features=4),
            "window_size": 2,
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
        }
