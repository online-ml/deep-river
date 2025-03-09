from typing import Callable, Type, Union

from torch import nn, optim

from deep_river.regression import Regressor, RegressorInitialized
from deep_river.regression.rolling_regressor import RollingRegressorInitialized


class LinearRegression(Regressor):
    """
    This class implements a linear regression model in PyTorch.

    """

    class LRModule(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.dense0 = nn.Linear(n_features, 1)

        def forward(self, x, **kwargs):
            x = self.dense0(x)
            return x

    def __init__(
        self,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        is_feature_incremental: bool = False,
        **kwargs,
    ):
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=LinearRegression.LRModule,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            is_feature_incremental=is_feature_incremental,
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
            "loss_fn": "l1",
            "lr": 1e-4,
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
        }


class MultiLayerPerceptron(Regressor):
    """
    This class implements a logistic regression model in PyTorch.

    Parameters
    ----------
    n_width
        Number of units in each hidden layer.
    n_layers
        Number of hidden layers.
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
    is_feature_incremental
        Whether the model should adapt to the appearance of
        previously features by adding units to the input
        layer of the network.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **kwargs
        Parameters to be passed to the `build_fn` function aside from
        `n_features`.

    """

    class MLPModule(nn.Module):
        def __init__(self, n_width, n_layers, n_features):
            super().__init__()
            hidden = [nn.Linear(n_features, n_width)]
            hidden += [nn.Linear(n_width, n_width) for _ in range(n_layers - 1)]
            self.hidden = nn.ModuleList(hidden)
            self.denselast = nn.Linear(n_width, 1)

        def forward(self, x, **kwargs):
            for layer in self.hidden:
                x = layer(x)
                x = nn.functional.sigmoid(x)
            return self.denselast(x)

    def __init__(
        self,
        n_width: int = 5,
        n_layers: int = 5,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = True,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.n_width = n_width
        self.n_layers = n_layers
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=MultiLayerPerceptron.MLPModule,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            n_width=n_width,
            n_layers=n_layers,
            is_feature_incremental=is_feature_incremental,
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
            "loss_fn": "mse",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
            "n_layers": 1,
        }


class LinearRegressionInitialized(RegressorInitialized):
    """
    Linear Regression model for regression.

    Parameters
    ----------
    loss_fn : str or Callable
        Loss function to be used for training the wrapped model.
    optimizer_fn : str or Callable
        Optimizer to be used for training the wrapped model.
    lr : float
        Learning rate of the optimizer.
    output_is_logit : bool
        Whether the module produces logits as output. If true, either
        softmax or sigmoid is applied to the outputs when predicting.
    is_class_incremental : bool
        Whether the classifier should adapt to the appearance of previously unobserved classes
        by adding an unit to the output layer of the network.
    is_feature_incremental : bool
        Whether the model should adapt to the appearance of previously features by
        adding units to the input layer of the network.
    device : str
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed : int
        Random seed to be used for training the wrapped model.
    **kwargs
        Parameters to be passed to the `build_fn` function aside from `n_features`.
    """

    class LRModule(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.dense0 = nn.Linear(in_features=n_features, out_features=1)

        def forward(self, x, **kwargs):
            return self.dense0(x)

    def __init__(
        self,
        n_features: int = 10,
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.n_features = n_features
        module = LinearRegressionInitialized.LRModule(n_features=n_features)
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            output_is_logit=output_is_logit,
            is_feature_incremental=is_feature_incremental,
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
        """

        yield {
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
        }


class MultiLayerPerceptronInitialized(RegressorInitialized):
    """
    Linear Regression model for regression.

    Parameters
    ----------
    loss_fn : str or Callable
        Loss function to be used for training the wrapped model.
    optimizer_fn : str or Callable
        Optimizer to be used for training the wrapped model.
    lr : float
        Learning rate of the optimizer.
    output_is_logit : bool
        Whether the module produces logits as output. If true, either
        softmax or sigmoid is applied to the outputs when predicting.
    is_class_incremental : bool
        Whether the classifier should adapt to the appearance of previously unobserved classes
        by adding an unit to the output layer of the network.
    is_feature_incremental : bool
        Whether the model should adapt to the appearance of previously features by
        adding units to the input layer of the network.
    device : str
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed : int
        Random seed to be used for training the wrapped model.
    **kwargs
        Parameters to be passed to the `build_fn` function aside from `n_features`.
    """

    class MLPModule(nn.Module):
        def __init__(self, n_width, n_layers, n_features):
            super().__init__()
            hidden = [nn.Linear(n_features, n_width)]
            hidden += [nn.Linear(n_width, n_width) for _ in range(n_layers - 1)]
            self.hidden = nn.ModuleList(hidden)
            self.denselast = nn.Linear(n_width, 1)

        def forward(self, x, **kwargs):
            for layer in self.hidden:
                x = layer(x)
                x = nn.functional.sigmoid(x)
            return self.denselast(x)

    def __init__(
        self,
        n_features: int = 10,
        n_width: int = 5,
        n_layers: int = 5,
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.n_features = n_features
        self.n_width = n_width
        self.n_layers = n_layers
        module = MultiLayerPerceptronInitialized.MLPModule(
            n_features=n_features, n_layers=n_layers, n_width=n_width
        )
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            output_is_logit=output_is_logit,
            is_feature_incremental=is_feature_incremental,
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
        """

        yield {
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
        }


class LSTMRegressorInitialized(RollingRegressorInitialized):

    class LSTMModule(nn.Module):
        def __init__(self, n_features, output_size=1):
            super().__init__()
            self.n_features = n_features
            self.output_size = output_size
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=output_size, num_layers=1
            )

        def forward(self, X, **kwargs):
            # lstm with input, hidden, and internal state
            output, (hn, cn) = self.lstm(X)
            x = hn.view(-1, self.output_size)
            return x

    def __init__(
        self,
        n_features: int = 10,
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.n_features = n_features
        module = LSTMRegressorInitialized.LSTMModule(n_features=n_features)
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            is_feature_incremental=is_feature_incremental,
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
        """

        yield {
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
        }
