import collections
from typing import Type

import pandas as pd
import torch
from scipy.special import ndtr

from IncrementalTorch.utils import WindowedVarianceMeter
from IncrementalTorch.utils import dict2tensor
from .. import base


class ProbabilityWeightedAutoencoder(base.AutoencoderBase):
    def __init__(
        self,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",
        build_fn=None,
        device="cpu",
        skip_threshold=0.9,
        scale_scores=True,
        window_size=250,
        **net_params,
    ):
        super().__init__(
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            build_fn=build_fn,
            device=device,
            scale_scores=scale_scores,
            window_size=window_size,
            **net_params,
        )
        self.skip_threshold = skip_threshold
        self.stat_meter = WindowedVarianceMeter(window_size)

    def learn_one(self, x):
        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        self.train()
        x_pred = self(x)
        loss = self.loss_fn(x_pred, x)
        loss_item = loss.item()
        if self.stat_meter is not None:
            mean = self.stat_meter.mean
            std = (
                self.stat_meter.sample_std if self.stat_meter.population_std > 0 else 1
            )
            self.stat_meter.update(loss_item)

        loss_scaled = (loss_item - mean) / std
        prob = ndtr(loss_scaled)
        loss = (self.skip_threshold - prob) / self.skip_threshold * loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self


class NoDropoutAE(base.AutoencoderBase):
    def __init__(
        self,
        loss_fn="smooth_mae",
        optimizer_fn="sgd",
        build_fn=None,
        device="cpu",
        scale_scores=True,
        window_size=250,
        **net_params,
    ):
        net_params["dropout"] = 0
        super().__init__(
            loss_fn,
            optimizer_fn,
            build_fn,
            device,
            scale_scores,
            window_size,
            **net_params,
        )

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
        if self.scale_scores and self.stat_meter.mean != 0:
            score /= self.stat_meter.mean
            self.stat_meter.update()

        return score


class RollingWindowAutoencoder(base.AutoencoderBase):
    def __init__(
        self,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",
        build_fn=None,
        device="cpu",
        window_size=50,
        **net_params,
    ):
        super().__init__(
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            build_fn=build_fn,
            device=device,
            **net_params,
        )
        self.window_size = window_size
        self._x_window = collections.deque(maxlen=window_size)
        self._batch_i = 0
        self.to_init = True

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


class VariationalAutoencoder(base.AutoencoderBase):
    def __init__(
        self,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",
        build_fn=None,
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
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            build_fn=build_fn,
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
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
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
