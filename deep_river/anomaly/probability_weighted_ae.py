import math
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
import torch
from river import stats, utils
from scipy.special import ndtr

from deep_river.anomaly import ae


class ProbabilityWeightedAutoencoder(ae.Autoencoder):
    """ """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        skip_threshold: float = 0.9,
        window_size: int = 250,
        **kwargs,
    ):
        if not 0 < skip_threshold <= 1:
            raise ValueError("skip_threshold must be greater than 0 and at most 1.")
        if window_size < 1:
            raise ValueError("window_size must be positive.")
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            **kwargs,
        )
        self.window_size = window_size
        self.skip_threshold = skip_threshold
        self.rolling_mean = utils.Rolling(stats.Mean(), window_size=window_size)
        self.rolling_var = utils.Rolling(stats.Var(), window_size=window_size)

    def learn_one(self, x: dict, y: Any = None) -> None:
        """
        Performs one step of training with a single example,
        scaling the employed learning rate based on the outlier
        probability estimate of the input example.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        ProbabilityWeightedAutoencoder
            The autoencoder itself.
        """

        self._update_observed_features(x)
        x_t = self._dict2tensor(x)

        self.module.train()
        x_pred = self.module(x_t)
        loss = self.loss_func(x_pred, x_t)
        self._apply_loss(loss)

    def _apply_loss(self, loss):
        losses_numpy = np.asarray(loss.detach().cpu().tolist())
        mean = self.rolling_mean.get()
        var = self.rolling_var.get() if self.rolling_var.get() > 0 else 1
        for loss_numpy in np.atleast_1d(losses_numpy):
            self.rolling_mean.update(float(loss_numpy))
            self.rolling_var.update(float(loss_numpy))

        loss_scaled = (losses_numpy - mean) / math.sqrt(var)
        prob = ndtr(loss_scaled)
        weights = loss.new_tensor(
            np.maximum(0, (self.skip_threshold - prob) / self.skip_threshold).tolist()
        )
        self.optimizer.zero_grad()
        if not weights.numel():
            return
        loss = (weights * loss).mean()
        loss.backward()
        if not torch.any(weights > 0):
            return
        if self.gradient_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), self.gradient_clip_value
            )
        self.optimizer.step()

    def learn_many(self, X: pd.DataFrame) -> None:
        self._update_observed_features(X)
        X_t = self._df2tensor(X)

        self.module.train()
        x_pred = self.module(X_t)
        loss = torch.mean(
            self.loss_func(x_pred, X_t, reduction="none"),
            dim=list(range(1, X_t.dim())),
        )
        self._apply_loss(loss)
