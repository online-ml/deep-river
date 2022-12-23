from typing import Any, Callable, Type, Union

import numpy as np
import pandas as pd
import torch
from river.anomaly.base import AnomalyDetector
from torch import nn

from deep_river.base import DeepEstimator
from deep_river.utils import dict2tensor
from deep_river.utils.tensor_conversion import df2tensor


class _TestAutoencoder(torch.nn.Module):
    def __init__(self, n_features, latent_dim=3):
        super().__init__()
        self.linear1 = nn.Linear(n_features, latent_dim)
        self.nonlin = torch.nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_dim, n_features)

    def forward(self, X, **kwargs):
        X = self.linear1(X)
        X = self.nonlin(X)
        X = self.linear2(X)
        return nn.functional.sigmoid(X)


class Autoencoder(DeepEstimator, AnomalyDetector):
    """
    Wrapper for PyTorch autoencoder models that uses the networks
    reconstruction error for scoring the anomalousness of a given example.

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
    **kwargs
        Parameters to be passed to the `torch.Module` class
        aside from `n_features`.

    Examples
    --------
    >>> from deep_river.anomaly import Autoencoder
    >>> from river import metrics
    >>> from river.datasets import CreditCard
    >>> from torch import nn
    >>> import math
    >>> from river.compose import Pipeline
    >>> from river.preprocessing import MinMaxScaler

    >>> dataset = CreditCard().take(5000)
    >>> metric = metrics.ROCAUC(n_thresholds=50)

    >>> class MyAutoEncoder(torch.nn.Module):
    ...     def __init__(self, n_features, latent_dim=3):
    ...         super(MyAutoEncoder, self).__init__()
    ...         self.linear1 = nn.Linear(n_features, latent_dim)
    ...         self.nonlin = torch.nn.LeakyReLU()
    ...         self.linear2 = nn.Linear(latent_dim, n_features)
    ...         self.sigmoid = nn.Sigmoid()
    ...
    ...     def forward(self, X, **kwargs):
    ...         X = self.linear1(X)
    ...         X = self.nonlin(X)
    ...         X = self.linear2(X)
    ...         return self.sigmoid(X)

    >>> ae = Autoencoder(module=MyAutoEncoder, lr=0.005)
    >>> scaler = MinMaxScaler()
    >>> model = Pipeline(scaler, ae)

    >>> for x, y in dataset:
    ...    score = model.score_one(x)
    ...    model = model.learn_one(x=x)
    ...    metric = metric.update(y, score)
    ...
    >>> print(f"ROCAUC: {metric.get():.4f}")
    ROCAUC: 0.7447
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
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
            "module": _TestAutoencoder,
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

    def learn_one(self, x: dict, y: Any = None, **kwargs) -> "Autoencoder":
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x
            Input example.

        **kwargs

        Returns
        -------
        Autoencoder
            The model itself.
        """
        if not self.module_initialized:
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)
        return self._learn(dict2tensor(x, device=self.device))

    def _learn(self, x: torch.Tensor) -> "Autoencoder":
        self.module.train()
        x_pred = self.module(x)
        loss = self.loss_fn(x_pred, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self

    def score_one(self, x: dict) -> float:
        """
        Returns an anomaly score for the provided example in the form of
        the autoencoder's reconstruction error.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        float
            Anomaly score for the given example. Larger values indicate
            more anomalous examples.

        """

        if not self.module_initialized:
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)

        x_t = dict2tensor(x, device=self.device)
        self.module.eval()
        with torch.inference_mode():
            x_pred = self.module(x_t)
        loss = self.loss_fn(x_pred, x_t).item()
        return loss

    def learn_many(self, X: pd.DataFrame) -> "Autoencoder":
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        X
            Input batch of examples.

        Returns
        -------
        Autoencoder
            The model itself.

        """
        if not self.module_initialized:
            self.kwargs["n_features"] = len(X.columns)
            self.initialize_module(**self.kwargs)
        X = df2tensor(X, device=self.device)
        return self._learn(X)

    def score_many(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns an anomaly score for the provided batch of examples in
        the form of the autoencoder's reconstruction error.

        Parameters
        ----------
        x
            Input batch of examples.

        Returns
        -------
        float
            Anomaly scores for the given batch of examples. Larger values
            indicate more anomalous examples.
        """
        if not self.module_initialized:
            self.kwargs["n_features"] = len(X.columns)
            self.initialize_module(**self.kwargs)
        X = df2tensor(X, device=self.device)

        self.module.eval()
        with torch.inference_mode():
            X_pred = self.module(X)
        loss = torch.mean(
            self.loss_fn(X_pred, X, reduction="none"),
            dim=list(range(1, X.dim())),
        )
        score = loss.cpu().detach().numpy()
        return score
