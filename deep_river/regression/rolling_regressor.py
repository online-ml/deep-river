from typing import Callable, Type, Union

import pandas as pd
import torch
from river import base
from torch import optim

from deep_river.base import RollingDeepEstimator
from deep_river.regression import Regressor


class _TestLSTM(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = 1
        self.lstm = torch.nn.LSTM(
            input_size=n_features, hidden_size=self.hidden_size, num_layers=1
        )

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)
        hn = hn.view(-1, self.hidden_size)
        return hn


class RollingRegressor(RollingDeepEstimator, Regressor):
    """Incremental regressor with a fixed-size rolling window.

    Maintains the most recent ``window_size`` observations in a deque and feeds
    them as a (sequence_length, batch=1, n_features) tensor to the wrapped
    PyTorch module. This enables simple sequence style conditioning for models
    such as RNN/LSTM/GRU without storing the full historical stream.

    Parameters
    ----------
    module : torch.nn.Module
        Wrapped regression module (expects rolling tensor input shape).
    loss_fn : str | Callable, default='mse'
        Loss used for optimisation.
    optimizer_fn : str | type, default='sgd'
        Optimizer specification.
    lr : float, default=1e-3
        Learning rate.
    is_feature_incremental : bool, default=False
        Whether to expand the first trainable layer when new feature names appear.
    device : str, default='cpu'
        Torch device.
    seed : int, default=42
        Random seed.
    window_size : int, default=10
        Number of most recent samples kept in the rolling buffer.
    append_predict : bool, default=False
        If True, predicted samples (during prediction) are appended to the window
        enabling simple autoregressive rollouts.
    **kwargs
        Forwarded to :class:`~deep_river.base.RollingDeepEstimator`.

    Examples
    --------
        Real-world regression example using the Bikes dataset from river. We keep only
        the numeric features so the rolling tensor construction succeeds. A small GRU
        is trained online and we track a running MAE. The exact value may vary across
        library versions and hardware.

    >>> import random, numpy as np
    >>> from torch import nn, manual_seed
    >>> from river import datasets, metrics
    >>> from deep_river.regression.rolling_regressor import RollingRegressor
    >>> _ = manual_seed(42)
    >>> random.seed(42)
    >>> np.random.seed(42)
    >>> first_x, _ = next(iter(datasets.Bikes()))
    >>> numeric_keys = sorted([k for k, v in first_x.items() if isinstance(v, (int, float))])
    >>> class TinySeq(nn.Module):
    ...     def __init__(self, n_features):
    ...         super().__init__()
    ...         self.rnn = nn.GRU(n_features, 8)
    ...         self.head = nn.Linear(8, 1)
    ...     def forward(self, x):
    ...         out, _ = self.rnn(x)
    ...         return self.head(out[-1])
    >>> model = RollingRegressor(module=TinySeq(len(numeric_keys)), window_size=8)
    >>> mae = metrics.MAE()
    >>> window_size = 8
    >>> for i, (x, y) in enumerate(datasets.Bikes().take(200)):
    ...     x_num = {k: x[k] for k in numeric_keys}
    ...     if i >= window_size:
    ...         y_pred = model.predict_one(x_num)
    ...         mae.update(y, y_pred)
    ...     model.learn_one(x_num, y)
    >>> assert 0.0 <= mae.get() < 15.0
    >>> print(f"MAE: {mae.get():.4f}")  # doctest: +ELLIPSIS
    MAE: ...

    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
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
            is_feature_incremental=is_feature_incremental,
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
            "module": _TestLSTM(10),
            "optimizer_fn": "sgd",
            "lr": 1e-3,
            "is_feature_incremental": False,
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However,
        in some cases, some estimators might not
        be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
            # Test fails since `sum(y_pred)` call in test produces large
            # floating point error.
            "check_predict_proba_one",
        }

    def learn_one(self, x: dict, y: base.typing.RegTarget, **kwargs) -> None:
        """Update model using a single (x, y) and current rolling window.

        Parameters
        ----------
        x : dict
            Feature mapping.
        y : float
            Target value.
        """
        self._update_observed_features(x)

        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])

        x_t = self._deque2rolling_tensor(self._x_window)

        # Convert y to tensor (ensuring proper shape for regression)
        y_t = torch.tensor([y], dtype=torch.float32, device=self.device).view(-1, 1)

        self._learn(x=x_t, y=y_t)

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """Predict a single regression target using rolling context.

        Parameters
        ----------
        x : dict
            Feature mapping.

        Returns
        -------
        float
            Predicted target value.
        """
        self._update_observed_features(x)

        x_win = self._x_window.copy()
        x_win.append([x.get(feature, 0) for feature in self.observed_features])
        if self.append_predict:
            self._x_window = x_win

        self.module.eval()
        with torch.inference_mode():
            x_t = self._deque2rolling_tensor(x_win)
            y_pred = self.module(x_t)
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().view(-1)[-1].cpu().numpy().item()
            else:
                y_pred = float(y_pred)

        return y_pred

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Batch update with multiple samples using the rolling window.

        Only performs an optimisation step once the internal window has reached
        ``window_size`` length to ensure a full sequence is available.
        """
        self._update_observed_features(X)

        X = X[list(self.observed_features)]
        self._x_window.extend(X.values.tolist())

        if len(self._x_window) == self.window_size:
            X_t = self._deque2rolling_tensor(self._x_window)

            # Convert y to tensor (ensuring proper shape for regression)
            y_t = torch.tensor(y.values, dtype=torch.float32, device=self.device).view(
                -1, 1
            )

            self._learn(x=X_t, y=y_t)

    def predict_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict targets for multiple samples (appends to a copy of the window).

        Returns a single-column DataFrame named ``'y_pred'``.
        """

        self._update_observed_features(X)
        X = X[list(self.observed_features)]
        x_win = self._x_window.copy()
        x_win.extend(X.values.tolist())
        if self.append_predict:
            self._x_window = x_win

        self.module.eval()
        with torch.inference_mode():
            x_t = self._deque2rolling_tensor(x_win)
            y_preds = self.module(x_t)
            if isinstance(y_preds, torch.Tensor):
                y_preds = y_preds.detach().cpu().view(-1).numpy().tolist()

        return pd.DataFrame({"y_pred": y_preds})
