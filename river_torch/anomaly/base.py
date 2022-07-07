import abc
import inspect
import math
from typing import Type

import pandas as pd
import torch
from river import anomaly, base, stats
from torch import nn

from river_torch.utils import dict2tensor, get_loss_fn, get_optimizer_fn


class Autoencoder(anomaly.AnomalyDetector):

    """
    Base Auto Encoder
    ----------
    encoder_fn
    decoder_fn
    loss_fn
    optimizer_fn
    device
    net_params
    """

    def __init__(
        self,
        encoder_fn,
        decoder_fn,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",
        device="cpu",
        **net_params
    ):
        super().__init__()
        self.encoder_fn = encoder_fn
        self.decoder_fn = decoder_fn
        self.encoder = None
        self.decoder = None
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = get_optimizer_fn(optimizer_fn)
        self.net_params = net_params
        self.device = device

    @classmethod
    def _unit_test_params(cls):
        def encoder_fn(n_features):
            return nn.Sequential(
                nn.Linear(n_features, math.ceil(n_features / 2)), nn.LeakyReLU()
            )

        def decoder_fn(n_features):
            return nn.Sequential(nn.Linear(math.ceil(n_features / 2), n_features))

        yield {
            "encoder_fn": encoder_fn,
            "decoder_fn": decoder_fn,
            "loss_fn": torch.nn.MSELoss,
            "optimizer_fn": torch.optim.SGD,
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
        return self._learn(x)

    def _learn(self, x):
        x = dict2tensor(x, device=self.device)

        if self.encoder is None or self.decoder is None:
            self._init_net(n_features=x.shape[1])

        self.encoder.train()
        x_pred = self.decoder(self.encoder(x))
        loss = self.loss_fn(x_pred, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self

    def score_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.encoder is None or self.decoder is None:
            self._init_net(n_features=x.shape[1])

        self.encoder.eval()
        with torch.inference_mode():
            x_rec = self.decoder(self.encoder(x))
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
            x_rec = self.decoder(self.encoder(x))
        loss = torch.mean(
            self.loss_fn(x_rec, x, reduction="none"),
            dim=list(range(1, x.dim())),
        )
        score = loss.cpu().detach().numpy()
        return score

    def _init_net(self, n_features):
        self.encoder = self.encoder_fn(
            n_features=n_features, **self._filter_args(self.encoder_fn)
        )
        self.decoder = self.decoder_fn(
            n_features=n_features, **self._filter_args(self.decoder_fn)
        )

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.optimizer = self.configure_optimizers()
        self.to_init = False

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

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(
            nn.ModuleList([self.encoder, self.decoder]).parameters(),
            **self._filter_args(self.optimizer_fn),
        )
        return optimizer


class AnomalyScaler(base.Wrapper, anomaly.AnomalyDetector):
    """AnomalyScaler is a wrapper around an anomaly detector that scales the output of the model.

    Parameters
    ----------
    anomaly_detector
    """

    def __init__(self, anomaly_detector: anomaly.AnomalyDetector):
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

    def learn_one(self, *args) -> anomaly.AnomalyDetector:
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
