from typing import Callable, Union

from torch import nn

from deep_river.classification import Classifier


class LogisticRegression(Classifier):
    """
    This class implements a logistic regression model in PyTorch.

    Parameters
    ----------
    loss_fn
            Loss function to be used for training the wrapped model. Can be a
            loss function provided by `torch.nn.functional` or one of the
            following: 'mse', 'l1', 'cross_entropy',
            'binary_cross_entropy_with_logits', 'binary_crossentropy',
            'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    output_is_logit
        Whether the module produces logits as output. If true, either
        softmax or sigmoid is applied to the outputs when predicting.
    is_class_incremental
        Whether the classifier should adapt to the appearance of
        previously unobserved classes by adding an unit to the output
        layer of the network. This works only if the last trainable
        layer is an nn.Linear layer. Note also, that output activation
        functions can not be adapted, meaning that a binary classifier
        with a sigmoid output can not be altered to perform multi-class
        predictions.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **kwargs
        Parameters to be passed to the `build_fn` function aside from
        `n_features`.

    Examples
    --------
    >>> from deep_river.classification import LogisticRegression
    >>> from river import metrics, preprocessing, compose, datasets
    >>> from torch import nn, manual_seed

    >>> _ = manual_seed(42)

    >>> model_pipeline = compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...     LogisticRegression()
    ... )

    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()

    >>> for x, y in dataset:
    ...     y_pred = model_pipeline.predict_one(x) # make a prediction
    ...     metric = metric.update(y, y_pred) # update the metric
    ...     model_pipeline = model_pipeline.learn_one(x, y) # update the model

    >>> print(f"Accuracy: {metric:.2f}")
    Accuracy: 0.44

    """

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
