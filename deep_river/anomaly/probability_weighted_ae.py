import math
from typing import Any, Callable, Type, Union

import pandas as pd
import torch
from river import stats, utils
from scipy.special import ndtr

from deep_river.anomaly import ae
from deep_river.utils import dict2tensor


class ProbabilityWeightedAutoencoder(ae.Autoencoder):
    """
    Wrapper for PyTorch autoencoder models for anomaly detection that
    reduces the employed learning rate based on an outlier probability
    estimate of the input example as well as a threshold probability
    `skip_threshold`. If the outlier probability is above the threshold,
    the learning rate is reduced to less than 0. Given the probability
    estimate $p_out$, the adjusted learning rate
    $lr_adj$ is $lr * 1 - (\frac{p_out}{skip_threshold})$.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped.
        The Module should accept parameter `n_features` so that the returned
        model's input shape can be determined based on the number of features
        in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model.
        Can be a loss function provided by `torch.nn.functional` or one of the
        following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Base learning rate of the optimizer.
    skip_threshold
        Threshold probability to use as a reference for the reduction
        of the base learning rate.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **kwargs
        Parameters to be passed to the `module` function
        aside from `n_features`.

        Examples
    --------
    >>> from deep_river.anomaly import ProbabilityWeightedAutoencoder
    >>> from river import metrics
    >>> from river.datasets import CreditCard
    >>> from torch import nn, manual_seed
    >>> import math
    >>> from river.compose import Pipeline
    >>> from river.preprocessing import MinMaxScaler

    >>> _ = manual_seed(42)
    >>> dataset = CreditCard().take(5000)
    >>> metric = metrics.ROCAUC(n_thresholds=50)

    >>> class MyAutoEncoder(torch.nn.Module):
    ...     def __init__(self, n_features, latent_dim=3):
    ...         super(MyAutoEncoder, self).__init__()
    ...         self.linear1 = nn.Linear(n_features, latent_dim)
    ...         self.nonlin = torch.nn.LeakyReLU()
    ...         self.linear2 = nn.Linear(latent_dim, n_features)
    ...         self.sigmoid = nn.Sigmoid()
    ...
    ...     def forward(self, X, **kwargs):
    ...         X = self.linear1(X)
    ...         X = self.nonlin(X)
    ...         X = self.linear2(X)
    ...         return self.sigmoid(X)

    >>> ae = ProbabilityWeightedAutoencoder(module=MyAutoEncoder, lr=0.005)
    >>> scaler = MinMaxScaler()
    >>> model = Pipeline(scaler, ae)

    >>> for x, y in dataset:
    ...    score = model.score_one(x)
    ...    model = model.learn_one(x=x)
    ...    metric = metric.update(y, score)
    ...
    >>> print(f"ROCAUC: {metric.get():.4f}")
    ROCAUC: 0.8599
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        skip_threshold: float = 0.9,
        window_size=250,
        **kwargs,
    ):
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
        self.rolling_mean = utils.Rolling(
            stats.Mean(), window_size=window_size
        )
        self.rolling_var = utils.Rolling(stats.Var(), window_size=window_size)

    def learn_one(
        self, x: dict, y: Any = None, **kwargs
    ) -> "ProbabilityWeightedAutoencoder":
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
        if not self.module_initialized:
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)
        x_t = dict2tensor(x, device=self.device)

        self.module.train()
        x_pred = self.module(x_t)
        loss = self.loss_fn(x_pred, x_t)
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
        loss = (
            torch.tensor((self.skip_threshold - prob) / self.skip_threshold)
            * loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_many(self, X: pd.DataFrame) -> "ProbabilityWeightedAutoencoder":
        if not self.module_initialized:
            self.kwargs["n_features"] = len(X.columns)
            self.initialize_module(**self.kwargs)
        X = dict2tensor(X.to_dict(), device=self.device)

        self.module.train()
        x_pred = self.module(X)
        loss = torch.mean(
            self.loss_fn(x_pred, X, reduction="none"),
            dim=list(range(1, X.dim())),
        )
        self._apply_loss(loss)
        return self
