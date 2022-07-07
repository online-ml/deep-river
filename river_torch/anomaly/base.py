import inspect
import math
from typing import Type

import pandas as pd
import torch
from river import anomaly, stats
from torch import nn

from river_torch.utils import dict2tensor, get_loss_fn, get_optimizer_fn


class AutoEncoder(anomaly.AnomalyDetector, nn.Module):
    """
    Base Auto Encoder
    ----------
    encoder_fn
    decoder_fn
    loss_fn
    optimizer_fn
    device
    scale_scores
    window_size
    net_params
    """
    def __init__(
        self,
        encoder_fn,
        decoder_fn,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",
        device="cpu",
        scale_scores=True,
        window_size=250,
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
        self.mean_meter = stats.RollingMean(window_size) if scale_scores else None
        self.scale_scores = scale_scores
        self.window_size = window_size

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

        self.train()
        x_pred = self(x)
        loss = self.loss_fn(x_pred, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scale_scores:
            self.mean_meter.update(loss.item())
        return self

    def score_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.encoder is None or self.decoder is None:
            self._init_net(n_features=x.shape[1])

        self.eval()
        with torch.inference_mode():
            x_rec = self(x)
        loss = self.loss_fn(x_rec, x).item()
        if self.scale_scores and self.mean_meter.get() != 0:
            loss /= self.mean_meter.get()
        return loss

    def learn_many(self, x: pd.DataFrame):
        return self._learn(x)

    def score_many(self, x: pd.DataFrame) -> float:

        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        self.eval()
        with torch.inference_mode():
            x_rec = self(x)
        loss = torch.mean(
            self.loss_fn(x_rec, x, reduction="none"),
            dim=list(range(1, x.dim())),
        )
        score = loss.cpu().detach().numpy()
        if self.scale_scores and self.mean_meter.get() != 0:
            score /= self.mean_meter.get()
        return score

    def forward(self, x):
        return self.decoder(self.encoder(x))

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