import warnings
from typing import Any, Callable, Dict, Type, Union

import numpy as np
import pandas as pd
import torch
from river.anomaly.base import AnomalyDetector
from torch import nn

from deep_river.base import DeepEstimator


class _TestAutoencoder(torch.nn.Module):
    def __init__(self, n_features, latent_dim=3):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
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
    Represents an initialized autoencoder for anomaly detection and feature learning.

    This class is built upon the DeepEstimatorInitialized and AnomalyDetector
    base classes. It provides methods for performing unsupervised learning
    through an autoencoder mechanism. The primary objective of the class is
    to train the autoencoder on input data and compute anomaly scores based
    on the reconstruction error. It supports learning on individual examples
    or entire batches of data.

    Attributes
    ----------
    is_feature_incremental : bool
        Indicates whether the model is designed to increment features dynamically.
    module : torch.nn.Module
        The PyTorch model representing the autoencoder architecture.
    loss_fn : Union[str, Callable]
        Specifies the loss function to compute the reconstruction error.
    optimizer_fn : Union[str, Callable]
        Specifies the optimizer to be used for training the autoencoder.
    lr : float
        The learning rate for optimization.
    device : str
        The device on which the model is loaded and trained (e.g., "cpu",
        "cuda").
    seed : int
        Random seed for ensuring reproducibility.
    """

    def __init__(
        self,
        module: torch.nn.Module,
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
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            **kwargs,
        )
        self.is_feature_incremental = is_feature_incremental

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
            "module": _TestAutoencoder(30, 3),
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases,
        some estimators might not
        be able to pass certain checks.
        """
        return set()

    def learn_one(self, x: dict, y: Any = None, **kwargs) -> None:
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x
            Input example.

        **kwargs
        """
        self._update_observed_features(x)
        self._learn(self._dict2tensor(x))

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

        self._update_observed_features(x)
        x_t = self._dict2tensor(x)
        self.module.eval()
        with torch.inference_mode():
            x_pred = self.module(x_t)
        loss = self.loss_func(x_pred, x_t).item()
        return loss

    def learn_many(self, X: pd.DataFrame) -> None:
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        X
            Input batch of examples.
        """

        self._update_observed_features(X)
        X_t = self._df2tensor(X)
        self._learn(X_t)

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
        self._update_observed_features(X)
        x_t = self._df2tensor(X)

        self.module.eval()
        with torch.inference_mode():
            x_pred = self.module(x_t)
        loss = torch.mean(
            self.loss_func(x_pred, x_t, reduction="none"),
            dim=list(range(1, x_t.dim())),
        )
        score = loss.cpu().detach().numpy()
        return score
