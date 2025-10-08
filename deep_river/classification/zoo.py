from typing import Callable, Type, Union

from torch import nn, optim

from deep_river.classification import Classifier
from deep_river.classification.rolling_classifier import RollingClassifier


class LogisticRegressionInitialized(Classifier):
    """Incremental logistic regression with optional dynamic class expansion.

    This variant outputs raw logits (no internal softmax) so that losses like
    ``cross_entropy`` can be applied directly. The output layer can grow in
    response to newly observed class labels when ``is_class_incremental=True``.

    Parameters
    ----------
    n_features : int, default=10
        Initial number of input features.
    n_init_classes : int, default=2
        Initial number of output units/classes. Expanded automatically if new
        classes appear and class incrementality is enabled.
    loss_fn : str | Callable, default='cross_entropy'
        Training loss.
    optimizer_fn : str | type, default='sgd'
        Optimizer specification.
    lr : float, default=1e-3
        Learning rate.
    output_is_logit : bool, default=True
        Indicates outputs are logits (enables proper conversion in ``predict_proba``).
    is_feature_incremental : bool, default=False
        Whether to dynamically expand the input layer when new features appear.
    is_class_incremental : bool, default=True
        Whether to expand the output layer for new class labels.
    device : str, default='cpu'
        Torch device.
    seed : int, default=42
        Random seed.
    gradient_clip_value : float | None, default=None
        Optional gradient norm clipping value.
    **kwargs
        Forwarded to the parent constructor.

    Examples
    --------
    Deterministisches Beispiel mit festen Logits für exakte Klassenvorhersage::

    >>> import torch
    >>> from deep_river.classification.zoo import LogisticRegressionInitialized
    >>> model = LogisticRegressionInitialized(n_features=3, n_init_classes=2, is_class_incremental=False, is_feature_incremental=False, lr=0.0, optimizer_fn='sgd')
    >>> # Klassen registrieren
    >>> model.learn_one({'a':0,'b':0,'c':0}, 0)
    >>> model.learn_one({'a':1,'b':1,'c':1}, 1)
    >>> def _prep_lin(m):
    ...     with torch.no_grad():
    ...         m.dense0.weight.zero_()
    ...         m.dense0.bias[:] = torch.tensor([1.5, -0.5])
    >>> _prep_lin(model.module)
    >>> model.predict_one({'a':2,'b':3,'c':4})
    0
    """

    class LRModule(nn.Module):
        def __init__(self, n_features: int, n_init_classes: int):
            super().__init__()
            self.n_features = n_features
            self.n_init_classes = n_init_classes  # kept for reconstruction
            self.dense0 = nn.Linear(in_features=n_features, out_features=n_init_classes)

        def forward(self, x, **kwargs):
            return self.dense0(x)  # raw logits

    def __init__(
        self,
        n_features: int = 10,
        n_init_classes: int = 2,
        loss_fn: Union[str, Callable] = "cross_entropy",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_feature_incremental: bool = False,
        is_class_incremental: bool = True,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = None,
        **kwargs,
    ):
        self.n_features = n_features
        self.n_init_classes = n_init_classes
        module = LogisticRegressionInitialized.LRModule(
            n_features=n_features, n_init_classes=n_init_classes
        )
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            output_is_logit=output_is_logit,
            is_feature_incremental=is_feature_incremental,
            is_class_incremental=is_class_incremental,
            device=device,
            lr=lr,
            seed=seed,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "cross_entropy",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "is_class_incremental": True,
            "gradient_clip_value": None,
        }


class MultiLayerPerceptronInitialized(Classifier):
    """Configurable multi-layer perceptron with dynamic class expansion.

    Hidden layers use ReLU activations; the output layer emits raw logits.

    Parameters
    ----------
    n_features : int, default=10
        Initial number of features.
    n_width : int, default=5
        Width (units) of each hidden layer.
    n_layers : int, default=5
        Number of hidden layers (>=1). If 1,
        only the input layer feeds the output.
    n_init_classes : int, default=2
        Initial number of classes/output units.
    loss_fn, optimizer_fn, lr, output_is_logit, is_feature_incremental,
        is_class_incremental, device, seed, gradient_clip_value, **kwargs
        See :class:`LogisticRegressionInitialized`.

    Examples
    --------
    Deterministisches Beispiel (alle Gewichte 0, Bias legt Klasse fest)::

    >>> import torch
    >>> from deep_river.classification.zoo import MultiLayerPerceptronInitialized
    >>> m = MultiLayerPerceptronInitialized(n_features=4, n_width=6, n_layers=2, n_init_classes=2, is_class_incremental=False, is_feature_incremental=False, lr=0.0, optimizer_fn='sgd')
    >>> m.learn_one({'a':0,'b':0,'c':0,'d':0}, 0)
    >>> m.learn_one({'a':1,'b':1,'c':1,'d':1}, 1)
    >>> def _prep_mlp(mod):
    ...     with torch.no_grad():
    ...         mod.input_layer.weight.zero_()
    ...         for layer in mod.hidden:
    ...             layer.weight.zero_()
    ...         mod.denselast.weight.zero_()
    ...         mod.denselast.bias[:] = torch.tensor([2.0, -1.0])
    >>> _prep_mlp(m.module)
    >>> m.predict_one({'a':5,'b':6,'c':7,'d':8})
    0
    """

    class MLPModule(nn.Module):
        def __init__(self, n_width, n_layers, n_features, n_init_classes):
            super().__init__()
            self.n_width = n_width
            self.n_layers = n_layers
            self.n_features = n_features
            self.n_init_classes = n_init_classes
            self.input_layer = nn.Linear(n_features, n_width)
            hidden = [nn.Linear(n_width, n_width) for _ in range(n_layers - 1)]
            self.hidden = nn.ModuleList(hidden)
            self.denselast = nn.Linear(n_width, n_init_classes)
            self.activation = nn.ReLU()

        def forward(self, x, **kwargs):
            x = self.activation(self.input_layer(x))
            for layer in self.hidden:
                x = self.activation(layer(x))
            x = self.denselast(x)
            return x  # raw logits

    def __init__(
        self,
        n_features: int = 10,
        n_width: int = 5,
        n_layers: int = 5,
        n_init_classes: int = 2,
        loss_fn: Union[str, Callable] = "cross_entropy",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_feature_incremental: bool = False,
        is_class_incremental: bool = True,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = None,
        **kwargs,
    ):
        self.n_features = n_features
        self.n_width = n_width
        self.n_layers = n_layers
        self.n_init_classes = n_init_classes
        module = MultiLayerPerceptronInitialized.MLPModule(
            n_width=n_width,
            n_layers=n_layers,
            n_features=n_features,
            n_init_classes=n_init_classes,
        )
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            output_is_logit=output_is_logit,
            is_feature_incremental=is_feature_incremental,
            is_class_incremental=is_class_incremental,
            device=device,
            lr=lr,
            seed=seed,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "cross_entropy",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "is_class_incremental": True,
            "gradient_clip_value": None,
        }


class LSTMClassifier(RollingClassifier):
    """Rolling LSTM classifier with dynamic class expansion.

    An LSTM backbone feeds into a linear head that produces logits. Designed for
    sequential/temporal streams processed via a rolling window (see
    :class:`RollingClassifierInitialized`). The output layer (``head``) expands
    when new classes are observed (if enabled).

    Parameters
    ----------
    n_features : int, default=10
        Number of input features per timestep.
    hidden_size : int, default=16
        Hidden state dimensionality of the LSTM.
    n_init_classes : int, default=2
        Initial number of output classes.
    loss_fn, optimizer_fn, lr, output_is_logit,
        is_feature_incremental, is_class_incremental, device, seed,
        gradient_clip_value, **kwargs
        Standard parameters as in other classifiers.

    Examples
    --------
    Deterministischer Test: RNN-Gewichte nullen, Bias steuert Klasse::

    >>> import torch
    >>> from deep_river.classification.zoo import LSTMClassifier
    >>> lstm_clf = LSTMClassifier(n_features=4, hidden_size=3, n_init_classes=2, is_class_incremental=False, is_feature_incremental=False, lr=0.0, optimizer_fn='sgd')
    >>> lstm_clf.learn_one({'f0':0,'f1':0,'f2':0,'f3':0}, 0)
    >>> lstm_clf.learn_one({'f0':1,'f1':1,'f2':1,'f3':1}, 1)
    >>> def _prep_lstm(m):
    ...     with torch.no_grad():
    ...         for p in m.lstm.parameters():
    ...             p.data.zero_()
    ...         m.head.weight.data.zero_()
    ...         m.head.bias.data[:] = torch.tensor([2.2, -0.7])
    >>> _prep_lstm(lstm_clf.module)
    >>> lstm_clf.predict_one({'f0':2,'f1':3,'f2':4,'f3':5})
    0
    """

    class LSTMModule(nn.Module):
        def __init__(self, n_features: int, hidden_size: int, n_init_classes: int):
            super().__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size
            self.n_init_classes = n_init_classes
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden_size, num_layers=1
            )
            self.head = nn.Linear(hidden_size, n_init_classes)

        def forward(self, X, **kwargs):  # X: (seq_len, batch=1, n_features)
            output, (hn, cn) = self.lstm(X)
            h_last = hn[-1]  # (batch, hidden_size)
            logits = self.head(h_last)
            return logits  # (batch, n_classes) raw logits

    def __init__(
        self,
        n_features: int = 10,
        hidden_size: int = 16,
        n_init_classes: int = 2,
        loss_fn: Union[str, Callable] = "cross_entropy",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_feature_incremental: bool = False,
        is_class_incremental: bool = True,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = None,
        **kwargs,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_init_classes = n_init_classes
        module = LSTMClassifier.LSTMModule(
            n_features=n_features,
            hidden_size=hidden_size,
            n_init_classes=n_init_classes,
        )
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            output_is_logit=output_is_logit,
            is_feature_incremental=is_feature_incremental,
            is_class_incremental=is_class_incremental,
            device=device,
            lr=lr,
            seed=seed,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "cross_entropy",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "hidden_size": 8,
            "n_init_classes": 2,
            "is_class_incremental": True,
            "gradient_clip_value": None,
        }


class RNNClassifier(RollingClassifier):
    """Rolling RNN classifier with dynamic class expansion.

    Uses a (stacked) ``nn.RNN`` backbone followed by a linear head that produces
    raw logits. Designed for streaming sequential data via a fixed-size rolling
    window handled by :class:`RollingClassifierInitialized`.

    Parameters
    ----------
    n_features : int, default=10
        Number of input features per timestep.
    hidden_size : int, default=16
        Hidden state dimensionality of the RNN.
    num_layers : int, default=1
        Number of stacked RNN layers.
    nonlinearity : str, default='tanh'
        Non-linearity used inside the RNN ('tanh' or 'relu').
    n_init_classes : int, default=2
        Initial number of classes/output units.
    loss_fn, optimizer_fn, lr, output_is_logit, is_feature_incremental,
        is_class_incremental, device, seed, gradient_clip_value, **kwargs
        Standard parameters as in the other rolling classifiers.

    Examples
    --------
    Deterministischer Test mit Null-Gewichten und Bias-Klasse::

    >>> import torch
    >>> from deep_river.classification.zoo import RNNClassifier
    >>> rnn_clf = RNNClassifier(n_features=4, hidden_size=3, n_init_classes=2, is_class_incremental=False, is_feature_incremental=False, lr=0.0, optimizer_fn='sgd')
    >>> rnn_clf.learn_one({'f0':0,'f1':0,'f2':0,'f3':0}, 0)
    >>> rnn_clf.learn_one({'f0':1,'f1':1,'f2':1,'f3':1}, 1)
    >>> def _prep_rnn(m):
    ...     with torch.no_grad():
    ...         for p in m.rnn.parameters():
    ...             p.data.zero_()
    ...         m.head.weight.data.zero_()
    ...         m.head.bias.data[:] = torch.tensor([3.0, -2.0])
    >>> _prep_rnn(rnn_clf.module)
    >>> rnn_clf.predict_one({'f0':9,'f1':8,'f2':7,'f3':6})
    0
    """

    class RNNModule(nn.Module):
        def __init__(
            self,
            n_features: int,
            hidden_size: int,
            num_layers: int,
            nonlinearity: str,
            n_init_classes: int,
        ):
            super().__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.nonlinearity = nonlinearity
            self.n_init_classes = n_init_classes
            self.rnn = nn.RNN(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
            )
            self.head = nn.Linear(hidden_size, n_init_classes)

        def forward(self, X, **kwargs):  # X: (seq_len, batch, n_features)
            out, hn = self.rnn(X)
            h_last = hn[-1]
            return self.head(h_last)  # raw logits

    def __init__(
        self,
        n_features: int = 10,
        hidden_size: int = 16,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        n_init_classes: int = 2,
        loss_fn: Union[str, Callable] = "cross_entropy",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_feature_incremental: bool = False,
        is_class_incremental: bool = True,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = None,
        **kwargs,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.n_init_classes = n_init_classes
        module = RNNClassifier.RNNModule(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            n_init_classes=n_init_classes,
        )
        if "module" in kwargs:
            del kwargs["module"]
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            output_is_logit=output_is_logit,
            is_feature_incremental=is_feature_incremental,
            is_class_incremental=is_class_incremental,
            device=device,
            lr=lr,
            seed=seed,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "cross_entropy",
            "optimizer_fn": "adam",
            "is_feature_incremental": False,
            "hidden_size": 8,
            "n_init_classes": 2,
            "is_class_incremental": True,
            "gradient_clip_value": None,
        }
