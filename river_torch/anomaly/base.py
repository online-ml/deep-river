import abc
import inspect
import math

import pandas as pd
import torch
from river import anomaly, base
from torch import nn
from river_torch.base import DeepEstimator

from river_torch.utils import dict2tensor


class Autoencoder(DeepEstimator, anomaly.base.AnomalyDetector):

    """
    Base Autoencoder

    Parameters
    ----------
    build_fn
    loss_fn
    optimizer_fn
    device
    net_params
    """

    def __init__(
        self,
        build_fn,
        loss_fn="smooth_mae",
        optimizer_fn="sgd",
        device="cpu",
        **net_params
    ):
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            **net_params,
        )

    @classmethod
    def _unit_test_params(cls):
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
    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def learn_one(self, x):
        if self.net is None:
            self._init_net(n_features=len(x))
        self.net.train()
        x = dict2tensor(x, device=self.device)
        return self._learn(x)

    def _learn(self, x):
        self.net.train()
        x_pred = self.net(x)
        loss = self.loss_fn(x_pred, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self

    def score_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.net is None:
            self._init_net(n_features=x.shape[1])

        self.net.eval()
        with torch.inference_mode():
            x_rec = self.net(x)
        loss = self.loss_fn(x_rec, x).item()
        return loss

    def learn_many(self, x: pd.DataFrame):
        return self._learn(x)

    def score_many(self, x: pd.DataFrame) -> float:

        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        self.eval()
        with torch.inference_mode():
            x_rec = self.net(x)
        loss = torch.mean(
            self.loss_fn(x_rec, x, reduction="none"),
            dim=list(range(1, x.dim())),
        )
        score = loss.cpu().detach().numpy()
        return score

    def _filter_args(self, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `torch_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in self.net_params.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res


class AnomalyScaler(base.Wrapper, anomaly.base.AnomalyDetector):
    """AnomalyScaler is a wrapper around an anomaly detector that scales the output of the model.

    Parameters
    ----------
    anomaly_detector
    """

    def __init__(self, anomaly_detector: anomaly.base.AnomalyDetector):
        self.anomaly_detector = anomaly_detector

    @classmethod
    def _unit_test_params(self):
        yield {"anomaly_detector": anomaly.HalfSpaceTrees()}

    @classmethod
    def _unit_test_skips(self):
        """Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.
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
        """Return calibrated anomaly score based on raw score provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.
        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.
        Returns
        -------
        An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.
        """

    def learn_one(self, *args) -> anomaly.base.AnomalyDetector:
        """Update the anomaly calibrator and the underlying anomaly detector.
        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.
        Returns
        -------
        self
        """

        self.anomaly_detector.learn_one(*args)
        return self
