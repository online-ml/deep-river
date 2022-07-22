import collections

import torch

from river_torch.anomaly import base
from river_torch.utils import dict2tensor


class RollingWindowAutoencoder(base.Autoencoder):
    """
    A rolling window auto encoder
    ----------
    build_fn
    loss_fn
    optimizer_fn
    device
    skip_threshold
    window_size
    net_params
    """

    def __init__(
        self,
        build_fn,
        loss_fn="smooth_mae",
        optimizer_fn = "sgd",
        device="cpu",
        window_size=50,
        **net_params,
    ):
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            **net_params,
        )
        self.window_size = window_size
        self._x_window = collections.deque(maxlen=window_size)
        self._batch_i = 0

    def _learn_batch(self, x: torch.Tensor):
        self.net.train()

        x_pred = self.net(x)
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
            self.net.train()
            x = torch.concat(list(self._x_window.values))
            self._learn_batch(x=x)
        return self
