import abc
import inspect
import math
from typing import Callable, Union
import numpy as np

import pandas as pd
import torch
from river import anomaly, base
from torch import nn
from river_torch.base import DeepEstimator

from river_torch.utils import dict2tensor
from river_torch.utils.river_compat import df2tensor


class Autoencoder(DeepEstimator, anomaly.base.AnomalyDetector):
    """
    Wrapper for PyTorch autoencoder models that uses the networks reconstruction error for scoring the anomalousness of a given example.

    Parameters
    ----------
    build_fn
        Function that builds the autoencoder to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss function provided by `torch.nn.functional` or one of the following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy', 'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model. Can be an optimizer class provided by `torch.optim` or one of the following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **net_params
        Parameters to be passed to the `build_fn` function aside from `n_features`.
    """

    def __init__(
        self,
        build_fn: Callable,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **net_params
    ):
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            **net_params,
        )

    @classmethod
    def _unit_test_params(cls) -> dict:
        """
        Returns a dictionary of parameters to be used for unit testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the respective class.
        """

        def build_fn(n_features):
            latent_dim = math.ceil(n_features / 2)
            return nn.Sequential(
                nn.Linear(n_features, latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim, n_features),
            )

        yield {
            "build_fn": build_fn,
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.

        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def learn_one(self, x: dict) -> "Autoencoder":
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        Autoencoder
            The model itself.
        """
        if self.net is None:
            self._init_net(n_features=len(x))
        self.net.train()
        x = dict2tensor(x, device=self.device)
        return self._learn(x)

    def _learn(self, x: torch.TensorType) -> "Autoencoder":
        self.net.train()
        x_pred = self.net(x)
        loss = self.loss_fn(x_pred, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self

    def score_one(self, x: dict) -> float:
        """
        Returns an anomaly score for the provided example in the form of the autoencoder's reconstruction error.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        float
            Anomaly score for the given example. Larger values indicate more anomalous examples.

        """
        if self.net is None:
            self._init_net(n_features=len(x))
        x = dict2tensor(x, device=self.device)

        self.net.eval()
        with torch.inference_mode():
            x_rec = self.net(x)
        loss = self.loss_fn(x_rec, x).item()
        return loss

    def learn_many(self, x: pd.DataFrame) -> "Autoencoder":
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        x
            Input batch of examples.

        Returns
        -------
        Autoencoder
            The model itself.

        """
        x = df2tensor(x, device=self.device)
        return self._learn(x)

    def score_many(self, x: pd.DataFrame) -> np.ndarray:
        """
        Returns an anomaly score for the provided batch of examples in the form of the autoencoder's reconstruction error.

        Parameters
        ----------
        x
            Input batch of examples.

        Returns
        -------
        float
            Anomaly scores for the given batch of examples. Larger values indicate more anomalous examples.
        """
        if self.net is None:
            self._init_net(n_features=x.shape[-1])
        x = dict2tensor(x, device=self.device)

        self.eval()
        with torch.inference_mode():
            x_rec = self.net(x)
        loss = torch.mean(
            self.loss_fn(x_rec, x, reduction="none"),
            dim=list(range(1, x.dim())),
        )
        score = loss.cpu().detach().numpy()
        return score


class AnomalyScaler(base.Wrapper, anomaly.base.AnomalyDetector):
    """Wrapper around an anomaly detector that scales the output of the model to account for drift in the wrapped model's anomaly scores.

    Parameters
    ----------
    anomaly_detector
        Anomaly detector to be wrapped.
    """

    def __init__(self, anomaly_detector: anomaly.base.AnomalyDetector):
        self.anomaly_detector = anomaly_detector

    @classmethod
    def _unit_test_params(self) -> dict:
        """
        Returns a dictionary of parameters to be used for unit testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the respective class.
        """
        yield {"anomaly_detector": anomaly.HalfSpaceTrees()}

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.

        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    @property
    def _wrapped_model(self):
        return self.anomaly_detector

    @abc.abstractmethod
    def score_one(self, *args) -> float:
        """Return a scaled anomaly score based on raw score provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more anomalous examples.
        """

    def learn_one(self, *args) -> "AnomalyScaler":
        """
        Update the scaler and the underlying anomaly scaler.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        AnomalyScaler
            The model itself.
        """

        self.anomaly_detector.learn_one(*args)
        return self
