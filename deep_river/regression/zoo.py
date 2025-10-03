from typing import Callable, Type, Union

from torch import nn, optim
import torch

from deep_river.regression import Regressor
from deep_river.regression.rolling_regressor import RollingRegressor


class LinearRegressionInitialized(Regressor):
    """
    Einfache lineare Regression mit optionalem Feature-Inkremental und Gradient Clipping.
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
        """
        Returns a dictionary of parameters to be used for unit testing the
        respective class.
        """

        yield {
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "gradient_clip_value": None,
        }


class MultiLayerPerceptronInitialized(Regressor):
    """
    MLP Regression mit optionalem Feature-Inkremental und Gradient Clipping.
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
        module = MultiLayerPerceptronInitialized.MLPModule(
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
        """
        Returns a dictionary of parameters to be used for unit testing the
        respective class.
        """

        yield {
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "gradient_clip_value": None,
        }


class LSTMRegressor(RollingRegressor):
    """
    LSTM Regressor für sequenzielle Regressionsaufgaben mit Rolling Window.

    Verbesserungen gegenüber der vorherigen sehr schwachen Version:
    - Separate versteckte Dimension (hidden_size) statt hidden_size=output=1
    - Separater Linear-Head (hidden -> 1) für bessere Ausdrucksstärke
    - Optional Dropout zwischen LSTM und Head
    - Konfigurierbare Anzahl LSTM-Layer (num_layers)
    - Option gradient_clipping zur Stabilisierung (Default 1.0)

    Parameter
    ---------
    n_features : int
        Anzahl der Eingangsfeatures (Start – kann bei feature-incremental wachsen).
    hidden_size : int
        LSTM Hidden-State Dimension (Standard 32).
    num_layers : int
        Anzahl LSTM-Layer (Standard 1).
    dropout : float
        Dropout nach dem LSTM (nur wenn > 0 und num_layers > 1 in LSTM selbst; wir nutzen es vorm Head).
    gradient_clip_value : float | None
        Falls gesetzt, wird der Gradienten-Norm-Clipping-Wert verwendet.
    loss_fn, optimizer_fn, lr, is_feature_incremental, device, seed
        Wie bei anderen Regressoren.
    """

    class LSTMModule(nn.Module):
        def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=False,  # Eingabe (seq_len, batch, features)
                dropout=0.0 if num_layers == 1 else min(dropout, 0.5),  # internes Dropout nur falls >1 Layer
            )
            self.head = nn.Linear(hidden_size, 1)
            self.out_activation = nn.Identity()  # Platzhalter falls später z.B. Softplus
            self.post_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        def forward(self, X, **kwargs):  # X: (seq_len, batch=1, n_features)
            output, (hn, cn) = self.lstm(X)
            # hn: (num_layers, batch, hidden_size) -> letzter Layer
            h_last = hn[-1]  # (batch, hidden_size)
            h_last = self.post_dropout(h_last)
            y = self.head(h_last)  # (batch, 1)
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
