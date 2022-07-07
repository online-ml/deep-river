import collections
from typing import Type

import pandas as pd
import torch
from river import stats
from scipy.special import ndtr

from river_torch.anomaly import base
from river_torch.utils import dict2tensor

class ProbabilityWeightedAutoencoder(base.AutoEncoder):
    """
    A propability weighted auto encoder
    ----------
    encoder_fn
    decoder_fn
    loss_fn
    optimizer_f
    device
    skip_threshold
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
        skip_threshold=0.9,
        scale_scores=True,
        window_size=250,
        **net_params,
    ):
        super().__init__(
            encoder_fn=encoder_fn,
            decoder_fn=decoder_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            scale_scores=scale_scores,
            window_size=window_size,
            **net_params,
        )
        self.skip_threshold = skip_threshold
        self.var_meter = stats.RollingVar(window_size)
        self.mean_meter = stats.RollingMean(window_size)

    def learn_one(self, x):
        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        self.train()
        x_pred = self(x)
        loss = self.loss_fn(x_pred, x)
        loss_item = loss.item()
        mean = self.mean_meter.get()
        std = self.var_meter.get() if self.var_meter.get() > 0 else 1
        self.mean_meter.update(loss_item)
        self.var_meter.update(loss_item)

        loss_scaled = (loss_item - mean) / std
        prob = ndtr(loss_scaled)
        loss = (self.skip_threshold - prob) / self.skip_threshold * loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self