from typing import Callable, Type, Union

import pandas as pd
import torch
from river import base
from river.base.typing import RegTarget
from torch import nn, optim

from deep_river.base import DeepEstimator, DeepEstimatorInitialized
from deep_river.utils.tensor_conversion import df2tensor, dict2tensor, float2tensor


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
    """
    Wrapper for PyTorch regression models that enables
    compatibility with River.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped.
        The Module should accept parameter `n_features` so that the
        returned model's input shape can be determined based on the number
        of features in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model.
        Can be a loss function provided by `torch.nn.functional` or one of
        the following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **kwargs
        Parameters to be passed to the `Module` or the `optimizer`.

    Examples
    --------

    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            device=device,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            seed=seed,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        """
        Returns a dictionary of parameters to be used for unit
        testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the
            respective class.
        """

        yield {
            "module": _TestModule,
            "loss_fn": "l1",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
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
        return set()

    def learn_one(self, x: dict, y: RegTarget, **kwargs) -> None:
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        Regressor
            The regressor itself.
        """
        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.kwargs)
        self._adapt_input_dim(x)
        x_t = dict2tensor(x, features=self.observed_features, device=self.device)
        y_t = float2tensor(y, device=self.device)

        self._learn(x_t, y_t)

    def _learn(self, x: torch.Tensor, y: torch.Tensor):
        self.module.train()
        self.optimizer.zero_grad()
        y_pred = self.module(x)
        loss = self.loss_func(y_pred, y)
        loss.backward()
        self.optimizer.step()

    def predict_one(self, x: dict) -> RegTarget:
        """
        Predicts the target value for a single example.

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
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.kwargs)
        self._adapt_input_dim(x)
        x_t = dict2tensor(x, features=self.observed_features, device=self.device)

        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t).item()
        return y_pred

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        x
            Input examples.
        y
            Target values.

        Returns
        -------
        Regressor
            The regressor itself.
        """
        if not self.module_initialized:

            self._update_observed_features(X)
            self.initialize_module(x=X, **self.kwargs)

        self._adapt_input_dim(X)
        X_t = df2tensor(X, features=self.observed_features, device=self.device)
        y_t = torch.tensor(y, device=self.device, dtype=torch.float32).unsqueeze(1)

        self._learn(X_t, y_t)

    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        """
        Predicts the target value for a batch of examples.

        Parameters
        ----------
        x
            Input examples.

        Returns
        -------
        List
            Predicted target values.
        """
        if not self.module_initialized:
            self._update_observed_features(X)
            self.initialize_module(x=X, **self.kwargs)

        self._adapt_input_dim(X)
        X_t = df2tensor(X, features=self.observed_features, device=self.device)

        self.module.eval()
        with torch.inference_mode():
            y_preds = self.module(X_t).detach().squeeze().tolist()
        return y_preds


class RegressorInitialized(DeepEstimatorInitialized, base.MiniBatchRegressor):
    """
    Wrapper for PyTorch classification models that supports feature and class incremental learning.

    Parameters
    ----------
    module : torch.nn.Module
        A PyTorch model. Can be pre-initialized or uninitialized.
    loss_fn : Union[str, Callable]
        Loss function for training. Can be a string ('mse', 'cross_entropy', etc.)
        or a PyTorch function.
    optimizer_fn : Union[str, Type[torch.optim.Optimizer]]
        Optimizer for training (e.g., "adam", "sgd", or a PyTorch optimizer class).
    lr : float, default=0.001
        Learning rate of the optimizer.
    output_is_logit : bool, default=True
        If True, applies softmax/sigmoid during inference.
    is_class_incremental : bool, default=False
        If True, adds neurons when new classes appear.
    is_feature_incremental : bool, default=False
        If True, adds neurons when new features appear.
    device : str, default="cpu"
        Whether to use "cpu" or "cuda".
    seed : Optional[int], default=None
        Random seed for reproducibility.
    **kwargs
        Additional parameters for model initialization.

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
        # Note: output_is_logit removed as it's not relevant for regression

    def learn_one(self, x: dict, y: base.typing.RegTarget) -> None:
        self._update_observed_features(x)
        x_t = self._dict2tensor(x)
        self._learn(x_t, y)  # y is now correctly handled in _learn

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._update_observed_features(X)
        x_t = self._df2tensor(X)
        y_t = torch.tensor(y.values, dtype=torch.float32, device=self.device).view(
            -1, 1
        )
        self._learn(x_t, y_t)

    def predict_one(self, x: dict) -> RegTarget:
        """
        Predicts the target value for a single example.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        RegTarget
            Predicted target value.
        """
        self._update_observed_features(x)
        x_t = self._dict2tensor(x)
        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t).item()
        return y_pred

    def predict_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicts probabilities for multiple examples."""
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
