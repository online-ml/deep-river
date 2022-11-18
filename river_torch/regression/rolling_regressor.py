from typing import Callable, List, Type, Union

import pandas as pd
import torch
from river import base
from river.base.typing import RegTarget

from river_torch.base import RollingDeepEstimator
from river_torch.utils.tensor_conversion import (df2rolling_tensor,
                                                 dict2rolling_tensor,
                                                 float2tensor)


class _TestLSTM(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.hidden_size = 1
        self.lstm = torch.nn.LSTM(
            input_size=n_features, hidden_size=self.hidden_size, num_layers=1
        )

    def forward(self, X, **kwargs):
        # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(X)
        hn = hn.view(-1, self.hidden_size)
        return hn


class RollingRegressor(RollingDeepEstimator, base.Regressor):
    """
    Wrapper that feeds a sliding window of the most recent examples to the
    wrapped PyTorch regression model.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped.
        The Module should accept parameter `n_features` so that the returned
        model's input shape can be determined based on the number of features
        in the initial training example.
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
        Number of recent examples to be fed to the wrapped model at each step.
    append_predict
        Whether to append inputs passed for prediction to the rolling window.
    **kwargs
        Parameters to be passed to the `Module` or the `optimizer`.
    """

    def __init__(
        self,
        module: Union[torch.nn.Module, Type[torch.nn.Module]],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        window_size: int = 10,
        append_predict: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            device=device,
            optimizer_fn=optimizer_fn,
            lr=lr,
            window_size=window_size,
            append_predict=append_predict,
            seed=seed,
            **kwargs
        )

    @classmethod
    def _unit_test_params(cls) -> dict:
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
            "module": _TestLSTM,
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
            "lr": 1e-3,
        }

    @classmethod
    def _unit_test_skips(self) -> set:
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
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def predict_one(self, x: dict) -> RegTarget:
        """
        Predicts the target value for the current sliding
        window of most recent examples.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        RegTarget
            Predicted target value.
        """
        if not self.module_initialized:
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)

        x = dict2rolling_tensor(x, self._x_window, device=self.device)
        if x is not None:
            self.module.eval()
            return self.module(x).detach().numpy().item()
        else:
            return 0.0

    def learn_one(self, x: dict, y: RegTarget) -> "RollingRegressor":
        """
        Performs one step of training with the sliding
        window of the most recent examples.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        RollingRegressor
            The regressor itself.
        """
        self._x_window.append(list(x.values()))
        if not self.module_initialized:
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)

        x = dict2rolling_tensor(x, self._x_window, device=self.device)
        if x is not None:
            y = float2tensor(y, device=self.device)
            self._learn(x, y)

        return self

    def _learn(self, x: torch.Tensor, y: torch.Tensor):
        self.optimizer.zero_grad()
        y_pred = self.module(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

    def learn_many(self, X: pd.DataFrame, y: List) -> "RollingDeepEstimator":
        if not self.module_initialized:
            self.kwargs["n_features"] = len(X.columns)
            self.initialize_module(**self.kwargs)

        X = df2rolling_tensor(X, self._x_window, device=self.device)
        if X is not None:
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
            self._learn(x=X, y=y)
        return self

    def predict_many(self, X: pd.DataFrame) -> List:
        if not self.module_initialized:
            self.kwargs["n_features"] = len(X.columns)
            self.initialize_module(**self.kwargs)

        batch = df2rolling_tensor(
            X,
            self._x_window,
            device=self.device,
            update_window=self.append_predict
        )
        if batch is not None:
            self.module.eval()
            y_pred = self.module(batch).detach().tolist()
            if len(y_pred) < len(batch):
                return [0.0] * (len(batch) - len(y_pred)) + y_pred
        else:
            return [0.0] * len(X)
