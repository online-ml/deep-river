from typing import Type

import pandas as pd
import torch
from river_torch.anomaly import base
from river_torch.utils import dict2tensor


class VariationalAutoencoder(base.Autoencoder):
    """
    A propability weighted auto encoder
    ----------
    encoder_fn
    decoder_fn
    loss_fn
    optimizer_fn
    device
    skip_threshold
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

        self.encoder.train()
        mu, log_var = self.encode(x)
        reconstructed = self.decode(mu, log_var)
        reconstructed = reconstructed.squeeze(0)

        latent_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        reconstruction_loss = self.loss_fn(reconstructed, x)
        total_loss = reconstruction_loss + self.beta * latent_loss

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
