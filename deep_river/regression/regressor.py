from typing import Callable, Type, Union

import pandas as pd
import torch
from river import base
from river.base.typing import RegTarget
from torch import nn, optim

from deep_river.base import DeepEstimator


class _TestModule(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.dense0 = torch.nn.Linear(n_features, 10)
        self.nonlin = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.dense1 = torch.nn.Linear(10, 5)
        self.output = torch.nn.Linear(5, 1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X


class Regressor(DeepEstimator, base.MiniBatchRegressor):
    """Incremental wrapper for PyTorch regression models.

    Provides feature-incremental learning (optional) by expanding the first
    trainable layer on-the-fly when unseen feature names are encountered.
    Suitable for streaming / online regression tasks using the :mod:`river` API.

    Parameters
    ----------
    module : torch.nn.Module
        PyTorch module that outputs a numeric prediction (shape (N, 1) or (N,)).
    loss_fn : str | Callable
        Loss identifier or callable (e.g. ``'mse'``).
    optimizer_fn : str | Type[torch.optim.Optimizer]
        Optimizer spec (``'adam'``, ``'sgd'`` or optimizer class).
    lr : float, default=1e-3
        Learning rate.
    is_feature_incremental : bool, default=False
        If True, expands the input layer for new feature names.
    device : str, default='cpu'
        Torch device.
    seed : int, default=42
        Random seed for reproducibility.
    **kwargs
        Extra args stored for cloning/persistence.

    Examples
    --------
    Deterministisches Beispiel mit festen Gewichten zur Überprüfung des
    Vorhersagewertes (keine Trainingsschritte nötig)::

    >>> import torch
    >>> from torch import nn
    >>> from deep_river.regression import Regressor
    >>> class FixedReg(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(4, 1)
    ...         with torch.no_grad():
    ...             # Gewichtsmatrix: alle 0.1, Bias: 0.2
    ...             self.fc.weight[:] = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    ...             self.fc.bias[:] = torch.tensor([0.2])
    ...     def forward(self, x):
    ...         return self.fc(x)
    >>> model = Regressor(module=FixedReg(), loss_fn='mse', optimizer_fn='sgd', lr=0.0)
    >>> x = {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0}
    >>> # Erwarteter Wert: 0.1*(1+2+3+4) + 0.2 = 1.2
    >>> round(model.predict_one(x), 2)
    1.2
    """

    def __init__(
        self,
        module: nn.Module,
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, Type[optim.Optimizer]],
        lr: float = 0.001,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            seed=seed,
            **kwargs,
        )

    def learn_one(self, x: dict, y: base.typing.RegTarget) -> None:
        self._update_observed_features(x)
        x_t = self._dict2tensor(x)
        self._learn(x_t, y)

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._update_observed_features(X)
        x_t = self._df2tensor(X)
        y_t = torch.tensor(y.values, dtype=torch.float32, device=self.device).view(
            -1, 1
        )
        self._learn(x_t, y_t)

    def predict_one(self, x: dict) -> RegTarget:
        """Predict target value for a single instance."""
        self._update_observed_features(x)
        x_t = self._dict2tensor(x)
        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t).item()
        return y_pred

    def predict_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict target values for multiple instances (returns single-column DataFrame)."""
        self._update_observed_features(X)
        x_t = self._df2tensor(X)
        self.module.eval()
        with torch.inference_mode():
            y_preds = self.module(x_t)
        return pd.DataFrame(y_preds if not y_preds.is_cuda else y_preds.cpu().numpy())

    @classmethod
    def _unit_test_params(cls):
        """Provides default parameters for unit testing."""
        yield {
            "module": _TestModule(10),
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
        }
