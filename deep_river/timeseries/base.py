from typing import Callable, Type, Union

import pandas as pd
import river
from river import time_series as river_ts
import torch
from deep_river import base



class _TestModule(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class DeepForecaster(river_ts.base.Forecaster, base.DeepEstimatorInitialized):
    """

    """
    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, Type[torch.optim.Optimizer]],
        lr: float = 0.001,
        output_is_logit: bool = True,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            seed=seed,
            **kwargs,
        )
        self.output_is_logit = output_is_logit

    @classmethod
    def _unit_test_params(cls):
        """Provides default parameters for unit testing."""
        yield {
            "module": _TestModule(10, 1),
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "is_class_incremental": False,
        }

    def learn_one(self, y, x=None):
        if x is not None:
            self._update_observed_features(x)
        x_t = self._dict2tensor(x)
        self._learn(x_t, y)

    def forecast(self, horizon, xs=None):
        """Forecast the next `horizon` steps."""
        if xs is not None:
            self._update_observed_features(xs)
            x_t = self._df2tensor(pd.DataFrame(xs)).to(self.device)

        # Replicate the input for each step, or adaptively update x for autoregression
        preds = []
        input_tensor = x_t.clone().detach()

        for _ in range(horizon):
            with torch.no_grad():
                y_pred = self.module(input_tensor)
                preds.append(float(y_pred.squeeze().cpu().numpy()))

            # Autoregressive mode: append the last prediction to the input
            # This depends on how your model handles sequential dependencies
            # You may need to modify input_tensor here if using RNN-style input
            # For now, assuming fixed features (no time-based shift)

        return preds
