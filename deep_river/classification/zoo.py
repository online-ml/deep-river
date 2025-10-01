from typing import Callable, Type, Union

from torch import nn, optim

from deep_river.classification import Classifier
from deep_river.classification.rolling_classifier import RollingClassifierInitialized


class LogisticRegressionInitialized(Classifier):
    """
    Logistic Regression model for classification.

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
            self.n_features = n_features  # notwendig für clone Rekonstruktion
            self.dense0 = nn.Linear(in_features=n_features, out_features=1)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, **kwargs):
            x = self.dense0(x)
            return self.softmax(x)

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
        module = LogisticRegressionInitialized.LRModule(n_features=n_features)
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


class MultiLayerPerceptronInitialized(Classifier):
    """
    Logistic Regression model for classification.

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
            self.n_width = n_width      # notwendig für clone Rekonstruktion
            self.n_layers = n_layers    # notwendig für clone Rekonstruktion
            self.n_features = n_features  # notwendig für clone Rekonstruktion
            self.input_layer = nn.Linear(n_features, n_width)
            hidden = []
            hidden += [nn.Linear(n_width, n_width) for _ in range(n_layers - 1)]
            self.hidden = nn.ModuleList(hidden)
            self.denselast = nn.Linear(n_width, 1)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, **kwargs):
            x = self.input_layer(x)
            for layer in self.hidden:
                x = layer(x)
                x = nn.functional.sigmoid(x)
            x = self.denselast(x)
            return self.softmax(x)

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
            n_width=n_width, n_layers=n_layers, n_features=n_features
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


class LSTMClassifierInitialized(RollingClassifierInitialized):
    """
    A specialized LSTM-based classifier designed for handling rolling or
    incremental data classification tasks.

    This class leverages LSTM (Long Short-Term Memory) modules to process
    and classify sequential data. It is built on top of the base
    `RollingClassifierInitialized` class, inheriting its functionality for
    handling incremental learning tasks. Customization options include the
    definition of the loss function, optimizer, learning rate, and other
    hyperparameters to suit various use cases.

    Attributes
    ----------
    n_features : int
        Number of features in the input data. It defines the input dimension for the
        LSTM module.
    loss_fn : Union[str, Callable]
        Specifies the loss function to be used for model training. Can either
        be a predefined string or a callable function.
    optimizer_fn : Union[str, Type[optim.Optimizer]]
        Defines the optimizer to be utilized in training. Accepts either a
        string representing the optimizer name or the optimizer class itself.
    lr : float
        Learning rate for the chosen optimizer.
    output_is_logit : bool
        Indicates whether the model output is a raw logit (pre-sigmoid/softmax output).
    is_feature_incremental : bool
        Specifies if the model supports adding new features incrementally.
    device : str
        Designates the device for computation, e.g., 'cpu' or 'cuda'.
    seed : int
        Random seed for reproducibility of results.
    kwargs : dict
        Additional arguments passed during the initialization.
    """

    class LSTMModule(nn.Module):
        def __init__(self, n_features, output_size=1):
            super().__init__()
            self.n_features = n_features
            self.output_size = output_size
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=output_size, num_layers=1
            )
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, X, **kwargs):
            # lstm with input, hidden, and internal state
            output, (hn, cn) = self.lstm(X)
            x = hn.view(-1, self.output_size)
            return self.softmax(x)

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
        module = LSTMClassifierInitialized.LSTMModule(n_features=n_features)
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
