from typing import Callable, Type, Union

import torch
from torch import nn, optim
from torch.nn import functional as F

from deep_river.forecasting.forecaster import DeepForecaster


class LinearForecaster(DeepForecaster):
    """Autoregressive linear forecaster with optional exogenous features."""

    class LinearModule(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.input_size = input_size
            self.dense0 = nn.Linear(input_size, 1)

        def forward(self, x, **kwargs):
            return self.dense0(x)

    def __init__(
        self,
        n_features: int = 0,
        window_size: int = 10,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = 1.0,
        **kwargs,
    ):
        self.n_features = n_features
        torch.manual_seed(seed)
        module = LinearForecaster.LinearModule(input_size=window_size + n_features)
        kwargs.pop("module", None)
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            window_size=window_size,
            is_sequence_model=False,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {"window_size": 2, "optimizer_fn": "sgd", "loss_fn": "mse"}


class MLPForecaster(DeepForecaster):
    """Autoregressive multi-layer perceptron forecaster."""

    class MLPModule(nn.Module):
        def __init__(self, input_size: int, n_width: int, n_layers: int):
            super().__init__()
            self.input_size = input_size
            self.n_width = n_width
            self.n_layers = n_layers
            layers = [nn.Linear(input_size, n_width), nn.ReLU()]
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(n_width, n_width), nn.ReLU()])
            layers.append(nn.Linear(n_width, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x, **kwargs):
            return self.net(x)

    def __init__(
        self,
        n_features: int = 0,
        window_size: int = 10,
        n_width: int = 16,
        n_layers: int = 2,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = 1.0,
        **kwargs,
    ):
        self.n_features = n_features
        self.n_width = n_width
        self.n_layers = n_layers
        torch.manual_seed(seed)
        module = MLPForecaster.MLPModule(
            input_size=window_size + n_features,
            n_width=n_width,
            n_layers=n_layers,
        )
        kwargs.pop("module", None)
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            window_size=window_size,
            is_sequence_model=False,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "window_size": 2,
            "n_width": 4,
            "n_layers": 1,
            "optimizer_fn": "adam",
            "loss_fn": "mse",
        }


class RNNForecaster(DeepForecaster):
    """Autoregressive forecaster backed by ``torch.nn.RNN``."""

    class RNNModule(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            nonlinearity: str,
            dropout: float,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.nonlinearity = nonlinearity
            self.dropout = dropout
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                dropout=0.0 if num_layers == 1 else min(dropout, 0.5),
            )
            self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x, **kwargs):
            _, hn = self.rnn(x)
            return self.head(self.post_dropout(hn[-1]))

    def __init__(
        self,
        n_features: int = 0,
        window_size: int = 10,
        hidden_size: int = 16,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        dropout: float = 0.0,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = 1.0,
        **kwargs,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        torch.manual_seed(seed)
        module = RNNForecaster.RNNModule(
            input_size=1 + n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            dropout=dropout,
        )
        kwargs.pop("module", None)
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            window_size=window_size,
            is_sequence_model=True,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {"window_size": 2, "hidden_size": 4, "optimizer_fn": "adam"}


class GRUForecaster(DeepForecaster):
    """Autoregressive forecaster backed by ``torch.nn.GRU``."""

    class GRUModule(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.0 if num_layers == 1 else min(dropout, 0.5),
            )
            self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x, **kwargs):
            _, hn = self.gru(x)
            return self.head(self.post_dropout(hn[-1]))

    def __init__(
        self,
        n_features: int = 0,
        window_size: int = 10,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = 1.0,
        **kwargs,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        torch.manual_seed(seed)
        module = GRUForecaster.GRUModule(
            input_size=1 + n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        kwargs.pop("module", None)
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            window_size=window_size,
            is_sequence_model=True,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {"window_size": 2, "hidden_size": 4, "optimizer_fn": "adam"}


class LSTMForecaster(DeepForecaster):
    """Autoregressive forecaster backed by ``torch.nn.LSTM``."""

    class LSTMModule(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=0.0 if num_layers == 1 else min(dropout, 0.5),
            )
            self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x, **kwargs):
            _, (hn, _) = self.lstm(x)
            return self.head(self.post_dropout(hn[-1]))

    def __init__(
        self,
        n_features: int = 0,
        window_size: int = 10,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = 1.0,
        **kwargs,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        torch.manual_seed(seed)
        module = LSTMForecaster.LSTMModule(
            input_size=1 + n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        kwargs.pop("module", None)
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            window_size=window_size,
            is_sequence_model=True,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {"window_size": 2, "hidden_size": 4, "optimizer_fn": "adam"}


class LiquidForecaster(DeepForecaster):
    """Autoregressive forecaster backed by closed-form liquid recurrent cells.

    The recurrent state follows a fixed-step liquid update where each hidden unit
    learns a positive time constant. ``time_delta`` is constant across all steps,
    matching regularly sampled time series.
    """

    class LiquidCell(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, time_delta: float):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.time_delta = time_delta
            combined_size = input_size + hidden_size
            self.candidate_weight = nn.Parameter(
                torch.empty(hidden_size, combined_size)
            )
            self.candidate_bias = nn.Parameter(torch.empty(hidden_size))
            self.tau_weight = nn.Parameter(torch.empty(hidden_size, combined_size))
            self.tau_bias = nn.Parameter(torch.empty(hidden_size))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.candidate_weight)
            nn.init.zeros_(self.candidate_bias)
            nn.init.xavier_uniform_(self.tau_weight)
            nn.init.constant_(self.tau_bias, 1.0)

        def forward(self, x, h):
            combined = torch.cat([x, h], dim=-1)
            candidate = torch.tanh(
                F.linear(combined, self.candidate_weight, self.candidate_bias)
            )
            tau = F.softplus(F.linear(combined, self.tau_weight, self.tau_bias)) + 1e-4
            alpha = 1.0 - torch.exp(-self.time_delta / tau)
            return h + alpha * (candidate - h)

    class LiquidModule(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            time_delta: float,
        ):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.time_delta = time_delta
            self.cells = nn.ModuleList(
                [
                    LiquidForecaster.LiquidCell(
                        input_size if i == 0 else hidden_size,
                        hidden_size,
                        time_delta,
                    )
                    for i in range(num_layers)
                ]
            )
            self.post_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x, **kwargs):
            batch_size = x.shape[1]
            states = [
                x.new_zeros(batch_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
            for x_t in x:
                layer_input = x_t
                for i, cell in enumerate(self.cells):
                    states[i] = cell(layer_input, states[i])
                    layer_input = states[i]
            return self.head(self.post_dropout(states[-1]))

    def __init__(
        self,
        n_features: int = 0,
        window_size: int = 10,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        time_delta: float = 1.0,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = 1.0,
        **kwargs,
    ):
        if is_feature_incremental:
            raise ValueError("LiquidForecaster does not support feature incrementality")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if time_delta <= 0:
            raise ValueError("time_delta must be positive")

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.time_delta = time_delta
        torch.manual_seed(seed)
        module = LiquidForecaster.LiquidModule(
            input_size=1 + n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            time_delta=time_delta,
        )
        kwargs.pop("module", None)
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            window_size=window_size,
            is_sequence_model=True,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {"window_size": 2, "hidden_size": 4, "optimizer_fn": "adam"}


class NBEATSForecaster(DeepForecaster):
    """Compact N-BEATS-style residual MLP forecaster for online point forecasts."""

    class NBEATSBlock(nn.Module):
        def __init__(
            self,
            input_size: int,
            n_width: int,
            n_layers: int,
            has_backcast: bool = True,
        ):
            super().__init__()
            layers = [nn.Linear(input_size, n_width), nn.ReLU()]
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(n_width, n_width), nn.ReLU()])
            self.layers = nn.Sequential(*layers)
            self.backcast = nn.Linear(n_width, input_size) if has_backcast else None
            self.forecast = nn.Linear(n_width, 1)

        def forward(self, x):
            h = self.layers(x)
            backcast = self.backcast(h) if self.backcast is not None else None
            return backcast, self.forecast(h)

    class NBEATSModule(nn.Module):
        def __init__(
            self,
            input_size: int,
            n_width: int,
            n_layers: int,
            n_blocks: int,
        ):
            super().__init__()
            self.input_size = input_size
            self.n_width = n_width
            self.n_layers = n_layers
            self.n_blocks = n_blocks
            self.blocks = nn.ModuleList(
                [
                    NBEATSForecaster.NBEATSBlock(
                        input_size,
                        n_width,
                        n_layers,
                        has_backcast=i < n_blocks - 1,
                    )
                    for i in range(n_blocks)
                ]
            )

        def forward(self, x, **kwargs):
            residual = x
            forecast = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            for block in self.blocks:
                backcast, block_forecast = block(residual)
                if backcast is not None:
                    residual = residual - backcast
                forecast = forecast + block_forecast
            return forecast

    def __init__(
        self,
        n_features: int = 0,
        window_size: int = 10,
        n_width: int = 32,
        n_layers: int = 2,
        n_blocks: int = 2,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = 1.0,
        **kwargs,
    ):
        if is_feature_incremental:
            raise ValueError("NBEATSForecaster does not support feature incrementality")

        self.n_features = n_features
        self.n_width = n_width
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        torch.manual_seed(seed)
        module = NBEATSForecaster.NBEATSModule(
            input_size=window_size + n_features,
            n_width=n_width,
            n_layers=n_layers,
            n_blocks=n_blocks,
        )
        kwargs.pop("module", None)
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            window_size=window_size,
            is_sequence_model=False,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "window_size": 2,
            "n_width": 4,
            "n_layers": 1,
            "n_blocks": 1,
            "optimizer_fn": "adam",
        }
