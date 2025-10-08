from typing import Callable, Type, Union

from torch import nn, optim

from deep_river.regression import Regressor
from deep_river.regression.rolling_regressor import RollingRegressor


class LinearRegressionInitialized(Regressor):
    """Incremental linear regression with optional feature growth and gradient clipping.

    A thin wrapper that instantiates a single linear layer and enables
    dynamic feature expansion when ``is_feature_incremental=True``. The model
    outputs a single continuous target value.

    Parameters
    ----------
    n_features : int, default=10
        Initial number of input features (columns). The input layer can expand
        if feature incrementality is enabled and new feature names appear.
    loss_fn : str | Callable, default='mse'
        Loss used for optimisation.
    optimizer_fn : str | type, default='sgd'
        Optimizer specification.
    lr : float, default=1e-3
        Learning rate.
    is_feature_incremental : bool, default=False
        Whether to expand the input layer when new features appear.
    device : str, default='cpu'
        Torch device.
    seed : int, default=42
        Random seed.
    gradient_clip_value : float | None, default=None
        Gradient norm clipping threshold. Disabled if ``None``.
    **kwargs
        Forwarded to :class:`~deep_river.base.DeepEstimator`.

    Examples
    --------
    Deterministisches Beispiel mit festen Gewichten (kein Training nötig)::

    >>> import torch
    >>> from torch import nn
    >>> from deep_river.regression.zoo import LinearRegressionInitialized
    >>> model = LinearRegressionInitialized(n_features=5, loss_fn='mse', optimizer_fn='sgd', lr=0.0)
    >>> with torch.no_grad():
    ...     model.module.dense0.weight[:] = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
    ...     model.module.dense0.bias[:] = torch.tensor([0.1])
    >>> x = {'f0':1,'f1':2,'f2':3,'f3':4,'f4':5}
    >>> # Erwartet: 0.2*(1+2+3+4+5)+0.1 = 3.1
    >>> round(model.predict_one(x), 2)
    3.1
    """

    class LRModule(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.dense0 = nn.Linear(in_features=n_features, out_features=1)

        def forward(self, x, **kwargs):
            return self.dense0(x)

    def __init__(
        self,
        n_features: int = 10,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = None,
        **kwargs,
    ):
        self.n_features = n_features
        module = LinearRegressionInitialized.LRModule(n_features=n_features)
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            is_feature_incremental=is_feature_incremental,
            device=device,
            lr=lr,
            seed=seed,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "gradient_clip_value": None,
        }


class MultiLayerPerceptron(Regressor):
    """Multi-layer perceptron regressor with optional feature growth.

    Stacks ``n_layers`` fully connected layers of width ``n_width`` with a
    sigmoid non-linearity (kept for backward compatibility) followed by a single
    output unit. Can expand its input layer when new feature names appear.

    Parameters
    ----------
    n_features : int, default=10
        Initial number of input features.
    n_width : int, default=5
        Hidden layer width.
    n_layers : int, default=5
        Number of hidden layers. Must be >=1.
    loss_fn, optimizer_fn, lr, is_feature_incremental, device, seed, gradient_clip_value, **kwargs
        Standard estimator configuration.

    Notes
    -----
    The use of ``sigmoid`` after each hidden layer can cause saturation; for
    deeper networks consider replacing with ReLU or GELU in a custom module.
    """

    class MLPModule(nn.Module):
        def __init__(self, n_width, n_layers, n_features):
            super().__init__()
            hidden = [nn.Linear(n_features, n_width)]
            hidden += [nn.Linear(n_width, n_width) for _ in range(n_layers - 1)]
            self.hidden = nn.ModuleList(hidden)
            self.denselast = nn.Linear(n_width, 1)

        def forward(self, x, **kwargs):
            for layer in self.hidden:
                x = layer(x)
                x = nn.functional.sigmoid(x)
            return self.denselast(x)

    def __init__(
        self,
        n_features: int = 10,
        n_width: int = 5,
        n_layers: int = 5,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = None,
        **kwargs,
    ):
        self.n_features = n_features
        self.n_width = n_width
        self.n_layers = n_layers
        module = MultiLayerPerceptron.MLPModule(
            n_features=n_features, n_layers=n_layers, n_width=n_width
        )
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            is_feature_incremental=is_feature_incremental,
            device=device,
            lr=lr,
            seed=seed,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "gradient_clip_value": None,
        }


class LSTMRegressor(RollingRegressor):
    """Rolling LSTM regressor for sequential / time-series data.

    Improves over a naïve single-unit LSTM by separating the hidden representation
    (``hidden_size``) from the 1D regression output head. Supports optional
    dropout and multiple LSTM layers. Designed to work with a rolling window
    maintained by :class:`~deep_river.base.RollingDeepEstimator`.

    Parameters
    ----------
    n_features : int, default=10
        Number of input features per timestep (may grow if feature-incremental).
    hidden_size : int, default=32
        Dimensionality of the LSTM hidden state.
    num_layers : int, default=1
        Number of stacked LSTM layers.
    dropout : float, default=0.0
        Dropout probability applied after the LSTM (and internally by PyTorch if
        ``num_layers > 1``). Capped internally for safety.
    gradient_clip_value : float | None, default=1.0
        Gradient norm clipping threshold (helps stability). ``None`` disables it.
    loss_fn, optimizer_fn, lr, is_feature_incremental, device, seed, **kwargs
        Standard configuration.

    Examples
    --------
    >>> from deep_river.regression.zoo import LSTMRegressor
    >>> lstm_reg = LSTMRegressor(n_features=6, hidden_size=16)
    """

    class LSTMModule(nn.Module):
        def __init__(
            self, n_features: int, hidden_size: int, num_layers: int, dropout: float
        ):
            super().__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=False,
                dropout=0.0 if num_layers == 1 else min(dropout, 0.5),
            )
            self.head = nn.Linear(hidden_size, 1)
            self.out_activation = (
                nn.Identity()
            )  # placeholder if Softplus etc. desired later
            self.post_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        def forward(self, X, **kwargs):  # X: (seq_len, batch=1, n_features)
            output, (hn, cn) = self.lstm(X)
            h_last = hn[-1]
            h_last = self.post_dropout(h_last)
            y = self.head(h_last)
            return self.out_activation(y)

    def __init__(
        self,
        n_features: int = 10,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        gradient_clip_value: float | None = 1.0,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gradient_clip_value = gradient_clip_value
        module = LSTMRegressor.LSTMModule(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            is_feature_incremental=is_feature_incremental,
            device=device,
            lr=lr,
            seed=seed,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "mse",
            "optimizer_fn": "adam",
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "is_feature_incremental": False,
        }


class RNNRegressor(RollingRegressor):
    """Rolling RNN regressor for sequential / time-series data.

    Uses a ``nn.RNN`` backbone and a linear head to output a single continuous
    target. Leverages the rolling window maintained by :class:`RollingRegressor`
    to feed the last ``window_size`` observations as a sequence.

    Parameters
    ----------
    n_features : int, default=10
        Number of input features per timestep.
    hidden_size : int, default=32
        Hidden state dimensionality of the RNN.
    num_layers : int, default=1
        Number of stacked RNN layers.
    nonlinearity : str, default='tanh'
        Non-linearity used inside the RNN (``'tanh'`` or ``'relu'``).
    dropout : float, default=0.0
        Dropout applied after extracting the last hidden state (no internal RNN dropout).
    gradient_clip_value : float | None, default=1.0
        Gradient norm clipping threshold. ``None`` disables clipping.
    loss_fn, optimizer_fn, lr, is_feature_incremental, device, seed, **kwargs
        Standard configuration as in other regressors.
    """

    class RNNModule(nn.Module):
        def __init__(
            self,
            n_features: int,
            hidden_size: int,
            num_layers: int,
            nonlinearity: str,
            dropout: float,
        ):
            super().__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.RNN(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
            )
            self.head = nn.Linear(hidden_size, 1)
            self.post_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        def forward(self, X, **kwargs):
            out, hn = self.rnn(X)
            h_last = hn[-1]
            h_last = self.post_dropout(h_last)
            y = self.head(h_last)
            return y

    def __init__(
        self,
        n_features: int = 10,
        hidden_size: int = 32,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        dropout: float = 0.0,
        gradient_clip_value: float | None = 1.0,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        module = RNNRegressor.RNNModule(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            dropout=dropout,
        )
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            is_feature_incremental=is_feature_incremental,
            device=device,
            lr=lr,
            seed=seed,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "mse",
            "optimizer_fn": "adam",
            "hidden_size": 8,
            "num_layers": 1,
            "is_feature_incremental": False,
            "dropout": 0.0,
        }
