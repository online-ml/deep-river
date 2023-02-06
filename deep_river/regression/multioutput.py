import typing
from typing import Type, Union, Callable

import torch
from ordered_set import OrderedSet
from river.base.typing import RegTarget, FeatureName

from deep_river.regression import Regressor
from river.base import MultiTargetRegressor

from deep_river.utils import dict2tensor, float2tensor


class MultiTargetRegressor(Regressor, MultiTargetRegressor):

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            device=device,
            optimizer_fn=optimizer_fn,
            lr=lr,
            seed=seed,
            **kwargs,
        )
        self.observed_targets: OrderedSet[RegTarget] = OrderedSet()

    def learn_one(self, x: dict, y: typing.Dict[FeatureName, RegTarget], **kwargs) -> "MultiTargetRegressor":
        if not self.module_initialized:
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)
        x_t = dict2tensor(x, self.device)
        self.observed_targets.update(y)
        y_t = float2tensor(y, self.device)
        self._learn(x_t, y_t)
        return self

    def predict_one(self, x: dict) -> typing.Dict[FeatureName, RegTarget]:
        """
                Predicts the target value for a single example.

                Parameters
                ----------
                x
                    Input example.

                Returns
                -------
                RegTarget
                    Predicted target value.
                """
        if not self.module_initialized:
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)
        x_t = dict2tensor(x, self.device)
        self.module.eval()
        with torch.inference_mode():
            y_pred_t = self.module(x_t).squeeze().tolist()
            y_pred = {t: y_pred_t[i] for i, t in enumerate(self.observed_targets)}
        return y_pred