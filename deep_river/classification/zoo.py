from typing import Callable, Type, Union

from torch import nn, optim

from deep_river.classification import Classifier
from deep_river.classification.rolling_classifier import RollingClassifierInitialized


class LogisticRegressionInitialized(Classifier):
    """
    Logistic Regression Modell mit dynamischer Klassenerweiterung.

    Änderungen:
    - Softmax entfernt (CrossEntropy erwartet rohe Logits)
    - Initiale Ausgabedimension konfigurierbar (n_init_classes, default=2)
    - Ausgabeschicht wird bei neuen Klassen erweitert, falls is_class_incremental=True
    """

    class LRModule(nn.Module):
        def __init__(self, n_features: int, n_init_classes: int):
            super().__init__()
            self.n_features = n_features
            self.n_init_classes = n_init_classes  # für Rekonstruktion
            self.dense0 = nn.Linear(in_features=n_features, out_features=n_init_classes)

        def forward(self, x, **kwargs):
            return self.dense0(x)  # rohe Logits

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
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "cross_entropy",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "is_class_incremental": True,
        }


class MultiLayerPerceptronInitialized(Classifier):
    """
    Mehrschichtiges Perzeptron mit dynamischer Klassenerweiterung.

    Änderungen:
    - Softmax entfernt (CrossEntropy erwartet Logits)
    - Sigmoid durch ReLU ersetzt in Hidden-Layern (gewöhnlicher Standard)
    - Initiale Ausgabedimension konfigurierbar (n_init_classes, default=2)
    - Ausgabeschicht erweitert sich bei neuen Klassen
    """

    class MLPModule(nn.Module):
        def __init__(self, n_width, n_layers, n_features, n_init_classes):
            super().__init__()
            self.n_width = n_width
            self.n_layers = n_layers
            self.n_features = n_features
            self.n_init_classes = n_init_classes
            self.input_layer = nn.Linear(n_features, n_width)
            hidden = []
            hidden += [nn.Linear(n_width, n_width) for _ in range(n_layers - 1)]
            self.hidden = nn.ModuleList(hidden)
            self.denselast = nn.Linear(n_width, n_init_classes)
            self.activation = nn.ReLU()

        def forward(self, x, **kwargs):
            x = self.activation(self.input_layer(x))
            for layer in self.hidden:
                x = self.activation(layer(x))
            x = self.denselast(x)
            return x  # rohe Logits

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
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls):
        yield {
            "loss_fn": "cross_entropy",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "is_class_incremental": True,
        }


class LSTMClassifierInitialized(RollingClassifierInitialized):
    """
    LSTM-basierter Klassifikator (rolling) mit dynamischer Klassenerweiterung.

    Änderungen / Design:
    - LSTM verbirgt eine feste "hidden_size" (Feature-Repräsentation)
    - Separater Linear-Head (head) mappt hidden -> Klassenlogits
    - Keine Softmax im Modul (rohe Logits); `cross_entropy` als Standard-Loss
    - Dynamische Klassenerweiterung funktioniert über bestehende `_update_observed_targets` Logik,
      da der Output-Layer (head) als letzter parametrischer Layer erkannt und erweitert wird.
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

        def forward(self, X, **kwargs):
            # X shape: (seq_len, batch=1, n_features) für rolling Fenster
            output, (hn, cn) = self.lstm(X)
            # hn shape: (num_layers, batch, hidden_size) -> wir nehmen letzte Schicht
            h_last = hn[-1]  # (batch, hidden_size)
            logits = self.head(h_last)
            return logits  # (batch, n_classes) rohe Logits

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
        **kwargs,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_init_classes = n_init_classes
        module = LSTMClassifierInitialized.LSTMModule(
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
        }
