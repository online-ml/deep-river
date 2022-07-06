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


class RollingWindowAutoencoder(base.AutoEncoder):
    """
    A rolling window auto encoder
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

    def __init__(
        self,
        encoder_fn,
        decoder_fn,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",
        device="cpu",
        window_size=50,
        **net_params,
    ):
        super().__init__(
            encoder_fn=encoder_fn,
            decoder_fn=decoder_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            **net_params,
        )
        self.window_size = window_size
        self._x_window = collections.deque(maxlen=window_size)
        self._batch_i = 0

    def _learn_batch(self, x: torch.Tensor):
        self.train()

        x_pred = self(x)
        loss = self.loss_fn(x_pred, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn_one(self, x):
        x = dict2tensor(x, device=self.device)

        self._x_window.append(x)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        if len(self._x_window) == self.window_size:
            x = torch.concat(list(self._x_window.values))
            self._learn_batch(x=x)
        return self


class VariationalAutoencoder(base.AutoEncoder):
    """
    A propability weighted auto encoder
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

    def __init__(
        self,
        encoder_fn,
        decoder_fn,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",
        device="cpu",
        n_reconstructions=10,
        scale_scores=True,
        beta=1,
        **net_params,
    ):
        net_params["variational"] = True
        net_params["dropout"] = 0
        net_params["tied_decoder_weights"] = False
        super().__init__(
            encoder_fn=encoder_fn,
            decoder_fn=decoder_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            scale_scores=scale_scores,
            **net_params,
        )
        self.n_reconstructions = n_reconstructions
        self.beta = beta

    def encode(self, x):
        encoded = self.encoder(x)
        mu, log_var = encoded.split(encoded.shape[1] // 2, dim=1)
        log_var = torch.clip(log_var, -3, 3)
        return mu, log_var

    def decode(self, mu, log_var, n_reconstructions=1):
        eps = torch.randn((n_reconstructions, *log_var.shape))
        z = mu + torch.exp(log_var * 0.5) * eps
        decoded = self.decoder(z)
        return decoded

    def learn_one(self, x):
        return self._learn(x)

    def _learn(self, x):
        x = dict2tensor(x, device=self.device)

        if self.to_init is True:
            self._init_net(n_features=x.shape[1])

        self.train()
        mu, log_var = self.encode(x)
        reconstructed = self.decode(mu, log_var)
        reconstructed = reconstructed.squeeze(0)

        latent_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        reconstruction_loss = self.loss_fn(reconstructed, x)
        total_loss = reconstruction_loss + self.beta * latent_loss

        if self.scaler is not None:
            self.scaler.learn_one(reconstruction_loss.item())
        self.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return self

    def score_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.to_init is True:
            self._init_net(n_features=x.shape[1])

        self.eval()
        with torch.inference_mode():
            mu, log_var = self.encode(x)
            reconstructed = self.decode(mu, log_var, self.n_reconstructions)

        loss = self.loss_fn(
            reconstructed, x.repeat((self.n_reconstructions, *[1] * x.dim()))
        ).item()
        if self.scaler is not None and self.scaler.mean is not None:
            loss /= self.scaler.mean
        return loss

    def score_learn_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.to_init is True:
            self._init_net(n_features=x.shape[1])

        self.eval()
        mu, log_var = self.encode(x)
        reconstructed = self.decode(mu, log_var, self.n_reconstructions)

        loss = self.loss_fn(
            reconstructed, x.repeat((self.n_reconstructions, *[1] * x.dim()))
        )
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.stat_meter is not None and self.stat_meter.mean > 0:
            loss /= self.stat_meter.mean
        return loss.item()

    def learn_many(self, x: pd.DataFrame):
        return super().learn_many(x)

    def score_many(self, x: pd.DataFrame) -> float:
        return super().score_many(x)
