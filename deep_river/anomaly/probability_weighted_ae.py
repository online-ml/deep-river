import math
import warnings
from typing import Any, Callable, Union

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
        window_size=250,
        **kwargs,
    ):
        warnings.warn(
            "This is deprecated and will be removed in future releases. "
            "Please instead use the AutoencoderInitialized class and "
            "initialize the module beforehand"
        )
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

    def learn_one(self, x: dict, y: Any = None, **kwargs) -> None:
        """
        Performs one step of training with a single example,
        scaling the employed learning rate based on the outlier
        probability estimate of the input example.

        Parameters
        ----------
        **kwargs
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
        losses_numpy = loss.detach().numpy()
        mean = self.rolling_mean.get()
        var = self.rolling_var.get() if self.rolling_var.get() > 0 else 1
        if losses_numpy.ndim == 0:
            self.rolling_mean.update(losses_numpy)
            self.rolling_var.update(losses_numpy)
        else:
            for loss_numpy in range(len(losses_numpy)):
                self.rolling_mean.update(loss_numpy)
                self.rolling_var.update(loss_numpy)

        loss_scaled = (losses_numpy - mean) / math.sqrt(var)
        prob = ndtr(loss_scaled)
        loss = torch.tensor((self.skip_threshold - prob) / self.skip_threshold) * loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_many(self, X: pd.DataFrame) -> None:
        self._update_observed_features(X)
        X_t = self._df2tensor(X)

        self.module.train()
        x_pred = self.module(X)
        loss = torch.mean(
            self.loss_func(x_pred, X_t, reduction="none"),
            dim=list(range(1, X_t.dim())),
        )
        self._apply_loss(loss)
