import math
from typing import Callable, Union

import pandas as pd
import torch
from river.stats import RollingMean, RollingVar
from scipy.special import ndtr

from river_torch.anomaly import ae
from river_torch.utils import dict2tensor


class ProbabilityWeightedAutoencoder(ae.Autoencoder):
    """
    Wrapper for PyTorch autoencoder models for anomaly detection that reduces the employed learning rate based on an outlier probability estimate of the input example as well as a threshold probability `skip_threshold`. If the outlier probability is above the threshold, the learning rate is reduced to less than 0. Given the probability estimate $p_out$, the adjusted learning rate $lr_adj$ is $lr * 1 - (\frac{p_out}{skip_threshold})$.

    Parameters
    ----------
    build_fn
        Function that builds the autoencoder to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss function provided by `torch.nn.functional` or one of the following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy', 'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model. Can be an optimizer class provided by `torch.optim` or one of the following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Base learning rate of the optimizer.
    skip_threshold
        Threshold probability to use as a reference for the reduction of the base learning rate.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **net_params
        Parameters to be passed to the `build_fn` function aside from `n_features`.

        Examples
    --------
    >>> from river_torch.anomaly import ProbabilityWeightedAutoencoder
    >>> from river import metrics
    >>> from river.datasets import CreditCard
    >>> from torch import nn, manual_seed
    >>> import math
    >>> from river.compose import Pipeline
    >>> from river.preprocessing import MinMaxScaler

    >>> _ = manual_seed(42)
    >>> dataset = CreditCard().take(5000)
    >>> metric = metrics.ROCAUC(n_thresholds=50)

    >>> def get_fc_ae(n_features):
    ...    latent_dim = math.ceil(n_features / 2)
    ...    return nn.Sequential(
    ...        nn.Linear(n_features, latent_dim),
    ...        nn.SELU(),
    ...        nn.Linear(latent_dim, n_features),
    ...        nn.Sigmoid(),
    ...    )

    >>> ae = ProbabilityWeightedAutoencoder(build_fn=get_fc_ae, lr=0.005)
    >>> scaler = MinMaxScaler()
    >>> model = Pipeline(scaler, ae)

    >>> for x, y in dataset:
    ...    score = model.score_one(x)
    ...    model = model.learn_one(x=x)
    ...    metric = metric.update(y, score)
    ...
    >>> print(f"ROCAUC: {metric.get():.4f}")
    ROCAUC: 0.8128
    """

    def __init__(
        self,
        build_fn: Callable,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        skip_threshold: float = 0.9,
        window_size=250,
        **net_params,
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
        self.window_size = window_size
        self.skip_threshold = skip_threshold
        self.rolling_mean = RollingMean(window_size=window_size)
        self.rolling_var = RollingVar(window_size=window_size)

    def learn_one(self, x: dict) -> "ProbabilityWeightedAutoencoder":
        """
        Performs one step of training with a single example, scaling the employed learning rate based on the outlier probability estimate of the input example.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        ProbabilityWeightedAutoencoder
            The autoencoder itself.
        """
        if self.net is None:
            self._init_net(n_features=len(x))
        x = dict2tensor(x, device=self.device)

        self.net.train()
        x_pred = self.net(x)
        loss = self.loss_fn(x_pred, x)
        self._apply_loss(loss)
        return self

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

    def learn_many(self, x: pd.DataFrame) -> "ProbabilityWeightedAutoencoder":
        if self.net is None:
            self._init_net(n_features=len(x.columns))
        x = dict2tensor(x, device=self.device)

        self.net.train()
        x_pred = self.net(x)
        loss = torch.mean(
            self.loss_fn(x_pred, x, reduction="none"),
            dim=list(range(1, x.dim())),
        )
        self._apply_loss(loss)
        return self
