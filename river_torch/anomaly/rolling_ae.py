import collections
from typing import Type
import torch
from river_torch.anomaly import base
from river_torch.utils import dict2tensor


class RollingWindowAutoencoder(base.Autoencoder):
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
