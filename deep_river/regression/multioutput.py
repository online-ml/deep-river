import typing
from typing import Callable, Union

import numpy as np
import pandas as pd
import torch
from river import base
from river.base.typing import FeatureName, RegTarget
from sortedcontainers import SortedSet

from deep_river.base import DeepEstimator


class _TestModule(torch.nn.Module):
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.dense0 = torch.nn.Linear(n_features, n_outputs)

    def forward(self, X, **kwargs):
        return self.dense0(X)


class MultiTargetRegressor(base.MultiTargetRegressor, DeepEstimator):
    """ """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        is_feature_incremental: bool = False,
        is_target_incremental: bool = False,
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            is_feature_incremental=is_feature_incremental,
            **kwargs,
        )
        self.is_target_incremental = is_target_incremental
        self.observed_targets: SortedSet = SortedSet()

    @classmethod
    def _unit_test_params(cls):
        """
        Returns a dictionary of parameters to be used for unit
        testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the
            respective class.
        """

        yield {
            "module": _TestModule(10, 3),
            "loss_fn": "l1",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
            "is_target_incremental": True,
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
        return {
            "check_shuffle_features_no_impact",
        }

    def learn_one(self, x: dict, y: dict[FeatureName, RegTarget], **kwargs) -> None:
        """Learns from a single example."""
        self._update_observed_features(x)
        self._update_observed_targets(y)
        x_t = self._dict2tensor(x)
        self._learn(x_t, y)

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Learns from a batch of examples."""
        self._update_observed_features(X)

        x_t = self._df2tensor(X)
        self._learn(x_t, y)

    def _update_observed_targets(self, y) -> bool:
        """
        Updates observed classes dynamically if new classes appear.
        Expands the output layer if is_class_incremental is True.
        """
        if isinstance(y, (base.typing.ClfTarget, np.bool_)):  # type: ignore[arg-type]
            self.observed_targets.update(y)
        else:
            self.observed_targets.update(y)

        if (self.is_target_incremental and self.output_layer) and len(
            self.observed_targets
        ) > self._get_output_size():
            self._expand_layer(
                self.output_layer,
                target_size=max(len(self.observed_targets), self._get_output_size()),
                output=True,
            )
            return True
        return False

    def predict_one(self, x: dict) -> typing.Dict[FeatureName, RegTarget]:
        """
        Predicts the target value for a single example.
        """

        self._update_observed_features(x)
        x_t = self._dict2tensor(x)
        self.module.eval()
        with torch.inference_mode():
            y_pred_t = self.module(x_t).squeeze().tolist()
            y_pred = {t: y_pred_t[i] for i, t in enumerate(self.observed_targets)}
        return y_pred
