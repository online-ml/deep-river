from typing import Callable, Type, Union

from torch import nn, optim

from deep_river.classification import Classifier, ClassifierInitialized
from deep_river.classification.rolling_classifier import \
    RollingClassifierInitialized


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
    ...     metric.update(y, y_pred) # update the metric
    ...     model_pipeline.learn_one(x, y) # update the model

    >>> print(f"Accuracy: {metric.get():.2f}")
    Accuracy: 0.56

    """

    class LRModule(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.dense0 = nn.Linear(n_features, 1)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, **kwargs):
            x = self.dense0(x)
            return self.softmax(x)

    def __init__(
        self,
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=LogisticRegression.LRModule,
            loss_fn=loss_fn,
            output_is_logit=output_is_logit,
            is_class_incremental=is_class_incremental,
            is_feature_incremental=is_feature_incremental,
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
            "is_feature_incremental": True,
            "is_class_incremental": True,
        }


class MultiLayerPerceptron(Classifier):
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
    is_class_incremental
        Whether the classifier should adapt to the appearance of
        previously unobserved classes by adding an unit to the output
        layer of the network. This works only if the last trainable
        layer is an nn.Linear layer. Note also, that output activation
        functions can not be adapted, meaning that a binary classifier
        with a sigmoid output can not be altered to perform multi-class
        predictions.
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

    Examples
    --------
    >>> from deep_river.classification import MultiLayerPerceptron
    >>> from river import metrics, preprocessing, compose, datasets
    >>> from torch import nn, manual_seed

    >>> _ = manual_seed(42)

    >>> model_pipeline = compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...     MultiLayerPerceptron(n_width=5,n_layers=5)
    ... )

    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()

    >>> for x, y in dataset:
    ...     y_pred = model_pipeline.predict_one(x) # make a prediction
    ...     metric.update(y, y_pred) # update the metric
    ...     model_pipeline.learn_one(x, y) # update the model

    >>> print(f"Accuracy: {metric.get():.2f}")
    Accuracy: 0.44

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
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):

        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=MultiLayerPerceptron.MLPModule,
            loss_fn=loss_fn,
            output_is_logit=output_is_logit,
            is_class_incremental=is_class_incremental,
            is_feature_incremental=is_feature_incremental,
            optimizer_fn=optimizer_fn,
            device=device,
            lr=lr,
            seed=seed,
            n_width=n_width,
            n_layers=n_layers,
            **kwargs,
        )
        self.n_width = n_width
        self.n_layers = n_layers

    @classmethod
    def _unit_test_skips(cls) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite.
        However, in some cases, some estimators might not
        be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {"check_predict_proba_one"}


class LogisticRegressionInitialized(ClassifierInitialized):
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


class MultiLayerPerceptronInitialized(ClassifierInitialized):
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
        module = LSTMClassifierInitialized.LSTMModule(
            n_features=n_features
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