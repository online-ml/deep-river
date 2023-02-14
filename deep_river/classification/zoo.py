from typing import Callable, Type, Union

from torch import nn

from deep_river.classification import Classifier


class LogisticRegression(Classifier):
    """ """

    class LRModule(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.dense0 = nn.Linear(n_features, 1)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, X, **kwargs):
            X = self.dense0(X)
            return self.softmax(X)

    def __init__(
        self,
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=LogisticRegression.LRModule,
            loss_fn=loss_fn,
            output_is_logit=output_is_logit,
            is_class_incremental=is_class_incremental,
            optimizer_fn=optimizer_fn,
            device=device,
            lr=lr,
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


class MultiLayerPerceptron(Classifier):
    """ """

    class MLPModule(nn.Module):
        def __init__(self, n_width, n_depth, n_features):
            super().__init__()
            self.dense0 = nn.Linear(n_features, n_width)
            self.block = [nn.Linear(n_width, n_width) for _ in range(n_depth)]
            self.denselast = nn.Linear(n_width, 1)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, X, **kwargs):
            X = self.dense0(X)
            for layer in self.block:
                X = layer(X)
            X = self.denselast(X)
            return self.softmax(X)

    def __init__(
        self,
        n_width: int = 5,
        n_depth: int = 5,
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.n_width = n_width
        self.n_depth = n_depth
        kwargs["n_width"] = n_width
        kwargs["n_depth"] = n_depth
        super().__init__(
            module=MultiLayerPerceptron.MLPModule,
            loss_fn=loss_fn,
            output_is_logit=output_is_logit,
            is_class_incremental=is_class_incremental,
            optimizer_fn=optimizer_fn,
            device=device,
            lr=lr,
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
