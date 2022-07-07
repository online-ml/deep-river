import collections
from typing import Type

import pandas as pd
import torch
from river import stats
from scipy.special import ndtr

from river_torch.anomaly import base
from river_torch.utils import dict2tensor

class NoDropoutAE(base.AutoEncoder):
    """
    No dropout auto encoder
    ----------
    encoder_fn
    decoder_fn
    loss_fn
    optimizer_fn
    device
    skip_threshold
    scale_scores
    window_size
    net_params
    """

    def score_learn_one(self, x: dict) -> float:
        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])
        x_rec = self(x)
        loss = self.loss_fn(x_rec, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        score = loss.item()
        if self.scale_scores:
            if self.stat_meter.mean != 0:
                score /= self.stat_meter.mean

            self.stat_meter.update(loss.item())

        return score
