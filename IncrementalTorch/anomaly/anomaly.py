import copy
import inspect
from typing import Type

import pandas as pd
import torch
from IncrementalTorch.utils import dict2tensor, get_loss_fn, get_optimizer_fn
from river import anomaly, utils
from torch import nn

from .nn_function import get_fc_autoencoder


class Autoencoder(anomaly.AnomalyDetector, nn.Module):
    def __init__(
        self,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",
        build_fn=None,
        device="cpu",
        **net_params,
    ):
        super().__init__()
        self.loss_fn = get_loss_fn(loss_fn)
        self.optimizer_fn = get_optimizer_fn(optimizer_fn)
        self.build_fn = build_fn
        self.net_params = net_params
        self.device = device

        self.encoder = None
        self.decoder = None
        self.to_init = True

    def learn_one(self, x):
        return self._learn(x)

    def _learn(self, x):
        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        self.train()
        x_pred = self(x)
        loss = self.loss_fn(x_pred, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self

    def score_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        self.eval()
        with torch.inference_mode():
            x_rec = self(x)
        loss = self.loss_fn(x_rec, x).item()
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
        return score

    def score_learn_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        self.train()
        x_pred = self(x)
        loss = self.loss_fn(x_pred, x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        score = loss.item()
        return score

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def _init_net(self, n_features):
        if self.build_fn is None:
            self.build_fn = get_fc_autoencoder

        self.encoder, self.decoder = self.build_fn(
            n_features=n_features, **self._filter_args(self.build_fn)
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


class AdaptiveAutoencoder(Autoencoder):
    def __init__(
        self,
        loss_fn="smooth_mae",
        optimizer_fn: Type[torch.optim.Optimizer] = "sgd",
        build_fn=None,
        beta=0.99,
        s=0.2,
        device="cpu",
        **net_params,
    ):
        super().__init__(
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            build_fn=build_fn,
            device=device,
            **net_params,
        )
        self.beta_scalar = beta
        self.dropout = None
        self.s = s

    def compute_recs(self, x, train=True):
        x_encs = []
        x_recs = []

        if train and self.dropout is not None:
            x_enc_prev = self.dropout(x)
        else:
            x_enc_prev = x

        for idx, layer in enumerate(self.encoding_layers):
            x_enc_prev = layer(x_enc_prev)
            if not isinstance(layer, nn.Linear):
                x_encs.append(x_enc_prev)
            else:
                x_encs.append(None)

        for idx, x_enc in enumerate(x_encs):
            if x_enc is not None:
                x_rec_prev = x_enc
                for layer in self.decoding_layers[-idx - 1 :]:
                    x_rec_prev = layer(x_rec_prev)
                x_recs.append(x_rec_prev)
        return torch.stack(x_recs, dim=0)

    def weight_recs(self, x_recs):
        alpha = self.alpha.view(-1, *[1] * (x_recs.dim() - 1))
        return torch.clip(torch.sum(x_recs * alpha, dim=0), min=0, max=1)

    def forward(self, x):
        x_recs = self.compute_recs(x)
        return self.weight_recs(x_recs)

    def learn_one(self, x: dict) -> anomaly.AnomalyDetector:
        x = dict2tensor(x, device=self.device)

        if self.to_init is False:
            self._init_net(x.shape[1])

        x_recs = self.compute_recs(x)
        x_rec = self.weight_recs(x_recs)

        loss = self.loss_fn(x_rec, x)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses = torch.stack([self.loss_fn(x_rec, x) for x_rec in x_recs])
        with torch.no_grad():
            self.alpha = self.alpha * torch.pow(self.beta, losses)
            self.alpha = torch.max(self.alpha, self.alpha_min)
            self.alpha = self.alpha / torch.sum(self.alpha)

        return self

    def score_learn_one(self, x: dict) -> anomaly.AnomalyDetector:
        x = dict2tensor(x, device=self.device)

        if self.to_init is False:
            self._init_net(x.shape[1])

        x_recs = self.compute_recs(x)
        x_rec = self.weight_recs(x_recs)

        loss = self.loss_fn(x_rec, x)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses = torch.stack([self.loss_fn(x_rec, x) for x_rec in x_recs])
        with torch.no_grad():
            self.alpha = self.alpha * torch.pow(self.beta, losses)
            self.alpha = torch.max(self.alpha, self.alpha_min)
            self.alpha = self.alpha / torch.sum(self.alpha)

        score = loss.item()
        return score

    def _init_net(self, n_features):
        if self.build_fn is None:
            self.build_fn = get_fc_autoencoder

        self.encoder, self.decoder = self.build_fn(
            n_features=n_features, **self._filter_args(self.build_fn)
        )
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.encoding_layers = [
            module
            for module in self.encoder.modules()
            if not isinstance(module, nn.Sequential)
        ]
        self.decoding_layers = [
            module
            for module in self.decoder.modules()
            if not isinstance(module, nn.Sequential)
        ]

        if isinstance(self.encoding_layers[0], nn.Dropout):
            self.dropout = self.encoding_layers.pop(0)

        self.optimizer = self.configure_optimizers()

        n_encoding_layers = len(
            [layer for layer in self.encoding_layers if isinstance(layer, nn.Linear)]
        )

        self.register_buffer("alpha", torch.ones(n_encoding_layers) / n_encoding_layers)
        self.register_buffer("beta", torch.ones(n_encoding_layers) * self.beta_scalar)
        self.register_buffer("alpha_min", torch.tensor(self.s / n_encoding_layers))


class RollingTorchAE(anomaly.AnomalyDetector):
    def __init__(
        self,
        build_fn,
        loss_fn: Type[torch.nn.modules.loss._Loss],
        optimizer_fn: Type[torch.optim.Optimizer],
        learning_rate=1e-3,
        window_size=10,
        seed=42,
        device="cpu",
        **net_params,
    ):
        self.build_fn = build_fn
        self.loss_fn = loss_fn
        self.loss = loss_fn()
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.window_size = window_size
        self.device = device
        self.net_params = net_params
        self.seed = seed
        torch.manual_seed(seed)

        self._x_window = utils.Window(window_size)
        self._batch_i = 0
        self.to_init = True

    def score_one(self, x) -> float:
        if isinstance(x, dict):
            x = dict2tensor(x, self.device)
        if self.to_init:
            self._init_net(n_features=x.shape[0])
        if len(self._x_window) == self.window_size:
            l = copy.deepcopy(self._x_window.values)
            l.append(list(x.values()))
            return self._score_one(x=l)
        else:
            return 0.0

    def _score_one(self, x) -> float:
        self.net.eval()
        with torch.inference_mode():
            x_pred = self.net(x)
            loss = self.loss(x_pred, x)
        return loss

    def _learn_batch(self, x: torch.Tensor):
        self.net.train()
        self.net.zero_grad()

        x_pred = self.net(x)
        loss = self.loss(x_pred, x)
        loss.backward()
        self.optimizer.step()

    def learn_one(self, x):
        self._x_window.append(list(x.values()))
        if isinstance(x, dict):
            x = dict2tensor(x, self.device)

        if self.to_init:
            self._init_net(n_features=x.shape[1])

        if len(self._x_window) == self.window_size:
            x = torch.Tensor([self._x_window.values])
            self._learn_batch(x=x)
        return self

    def _filter_torch_params(self, fn, override=None):
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

    def _init_net(self, n_features):
        self.net = self.build_fn(
            n_features=n_features, **self._filter_torch_params(self.build_fn)
        )
        self.net.to(self.device)
        self.optimizer = self.optimizer_fn(self.net.parameters(), lr=self.learning_rate)
        self.to_init = False


class SklearnAnomalyDetector(anomaly.AnomalyDetector):
    def __init__(self, build_fn) -> None:
        super().__init__()
        self.model = None
        self.build_fn = build_fn

        self.optimizer = self.configure_optimizers()

        n_encoding_layers = len(
            [layer for layer in self.encoding_layers if isinstance(layer, nn.Linear)]
        )

        self.register_buffer("alpha", torch.ones(n_encoding_layers) / n_encoding_layers)
        self.register_buffer("beta", torch.ones(n_encoding_layers) * self.beta_scalar)
        self.register_buffer("alpha_min", torch.tensor(self.s / n_encoding_layers))

    def configure_optimizers(self):
        optimizer = self.optimizer_fn(
            nn.ModuleList(self.encoding_layers + self.decoding_layers).parameters(),
            **self._filter_args(self.optimizer_fn),
        )
        return optimizer


class VariationalAutoencoder(Autoencoder):
    def __init__(
        self,
        loss_fn="mse",
        optimizer_fn: Type[torch.optim.Optimizer] = "adam_w",
        build_fn=None,
        device="cpu",
        n_reconstructions=10,
        **net_params,
    ):
        net_params["variational"] = True
        super().__init__(
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            build_fn=build_fn,
            device=device,
            **net_params,
        )
        self.n_reconstructions = n_reconstructions

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

        if self.to_init is False:
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

        self.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return self

    def score_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.to_init is False:
            self._init_net(n_features=x.shape[1])

        self.eval()
        with torch.inference_mode():
            mu, log_var = self.encode(x)
            reconstructed = self.decode(mu, log_var, self.n_reconstructions)

        loss = self.loss_fn(
            reconstructed, x.repeat((self.n_reconstructions, [1] * x.dim()))
        ).item()
        return loss

    def score_learn_one(self, x: dict):
        x = dict2tensor(x, device=self.device)

        if self.to_init is False:
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
        return loss.item()

    def learn_many(self, x: pd.DataFrame):
        return super().learn_many(x)

    def score_many(self, x: pd.DataFrame) -> float:
        return super().score_many(x)
