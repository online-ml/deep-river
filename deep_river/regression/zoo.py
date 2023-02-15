from typing import Union, Callable
from torch import nn
from deep_river.regression import Regressor


class LinearRegression(Regressor):
    """
    This class implements a linear regression model in PyTorch.

    Parameters
    ----------

    """

    class LRModule(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.dense0 = nn.Linear(n_features, 1)

        def forward(self, X, **kwargs):
            X = self.dense0(X)
            return X

    def __init__(
            self,
            loss_fn: Union[str, Callable] = "mse",
            optimizer_fn: Union[str, Callable] = "sgd",
            lr: float = 1e-3,
            device: str = "cpu",
            seed: int = 42,
            **kwargs,
    ):
        super().__init__(
            module=LinearRegression.LRModule,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        """
        Returns a dictionary of parameters to be used for unit testing the
        respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the
            respective class.
        """

        yield {
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
        }