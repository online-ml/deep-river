from typing import Callable, Dict, Type, Union

import pandas as pd
import torch
from river.base.typing import ClfTarget
from sortedcontainers import SortedSet
from torch import optim

from deep_river.base import RollingDeepEstimator
from deep_river.classification import Classifier
from deep_river.utils.tensor_conversion import output2proba


class _TestLSTM(torch.nn.Module):
    def __init__(self, n_features, hidden_size=1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=1
        )
        self.linear = torch.nn.Linear(hidden_size, 2)

    def forward(self, X, **kwargs):
        # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(X)
        x = hn.view(-1, self.hidden_size)
        x = self.linear(x)
        return torch.nn.functional.softmax(x, dim=-1)


class RollingClassifierInitialized(Classifier, RollingDeepEstimator):
    """
    RollingClassifierInitialized extends both ClassifierInitialized and
    RollingDeepEstimatorInitialized,
    incorporating a rolling window mechanism for sequential learning in an
    evolving feature and class space.

    This classifier dynamically adapts to new features and classes over
    time while leveraging a rolling
    window for training. It supports single-instance and batch learning
    while maintaining adaptability.

    Attributes
    ----------
    module : torch.nn.Module
        The PyTorch model used for classification.
    loss_fn : Union[str, Callable]
        The loss function for training, defaulting to binary cross-entropy with logits.
    optimizer_fn : Union[str, Type[optim.Optimizer]]
        The optimizer function or class used for training.
    lr : float
        The learning rate for optimization.
    output_is_logit : bool
        Indicates whether model outputs logits or probabilities.
    is_class_incremental : bool
        Whether new classes should be dynamically added.
    is_feature_incremental : bool
        Whether new features should be dynamically added.
    device : str
        The computational device for training (e.g., "cpu", "cuda").
    seed : int
        The random seed for reproducibility.
    window_size : int
        The number of past instances considered in the rolling window.
    append_predict : bool
        Whether predictions should be appended to the rolling window.
    observed_classes : SortedSet
        Tracks observed class labels for incremental learning.

    Examples
    --------
    >>> from deep_river.classification import RollingClassifier
    >>> from river import metrics, preprocessing, datasets, compose
    >>> import torch

    >>> class RnnModule(torch.nn.Module):
    ... def __init__(self, n_features, hidden_size=1):
    ...     super().__init__()
    ...     self.n_features = n_features
    ...     self.rnn = torch.nn.RNN(
    ...         input_size=n_features, hidden_size=hidden_size, num_layers=1
    ...     )
    ...     self.softmax = torch.nn.Softmax(dim=-1)
    ...
    ... def forward(self, X, **kwargs):
    ...     out, hn = self.rnn(X)  # lstm with input, hidden, and internal state
    ...     hn = hn.view(-1, self.rnn.hidden_size)
    ...     return self.softmax(hn)

    >>> model_pipeline = compose.Pipeline(
    ...     preprocessing.StandardScaler,
    ...     RollingClassifierInitialized(module=RnnModule(10,1),
    ...                loss_fn="binary_cross_entropy",
    ...                optimizer_fn='adam')
    ... )

    >>> dataset = datasets.Keystroke()
    >>> metric = metrics.Accuracy()
    >>> optimizer_fn = torch.optim.SGD

    >>> model_pipeline = preprocessing.StandardScaler()
    >>> model_pipeline |= RollingClassifier(
    ...    module=RnnModule,
    ...    loss_fn="binary_cross_entropy",
    ...    optimizer_fn=torch.optim.SGD,
    ...    window_size=20,
    ...    lr=1e-2,
    ...    append_predict=True,
    ...    is_class_incremental=False,
    ... )

    >>> for x, y in dataset:
    ...     y_pred = model_pipeline.predict_one(x)  # make a prediction
    ...     metric.update(y, y_pred)  # update the metric
    ...     model_pipeline.learn_one(x, y)  # make the model learn
    >>> print(f"Accuracy: {metric.get():.2f}")
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Type[optim.Optimizer]] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            output_is_logit=output_is_logit,
            is_class_incremental=is_class_incremental,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            window_size=window_size,
            append_predict=append_predict,
            **kwargs,
        )
        self.output_is_logit = output_is_logit
        self.observed_classes: SortedSet = SortedSet()

    @classmethod
    def _unit_test_params(cls):
        yield {
            "module": _TestLSTM(10, 16),
            "optimizer_fn": "sgd",
            "lr": 1e-3,
            "is_feature_incremental": False,
            "is_class_incremental": False,
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
        return {
            "check_predict_proba_one",
        }

    def learn_one(self, x: dict, y: ClfTarget, **kwargs) -> None:
        """Learns from one example using the rolling window."""
        self._update_observed_features(x)
        self._update_observed_targets(y)
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])
        x_t = self._deque2rolling_tensor(self._x_window)
        self._learn(x=x_t, y=y)

    def predict_proba_one(self, x: dict) -> Dict[ClfTarget, float]:
        """Predicts class probabilities using the rolling window."""
        self._update_observed_features(x)
        x_win = self._x_window.copy()
        x_win.append([x.get(feature, 0) for feature in self.observed_features])
        if self.append_predict:
            self._x_window = x_win
        self.module.eval()
        with torch.inference_mode():
            x_t = self._deque2rolling_tensor(x_win)
            y_pred = self.module(x_t)
            proba = output2proba(y_pred, self.observed_classes, self.output_is_logit)
        return proba[0]

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Learns from multiple examples using the rolling window."""
        self._update_observed_targets(y)
        self._update_observed_features(X)
        X = X[list(self.observed_features)]
        self._x_window.extend(X.values.tolist())
        X_t = self._deque2rolling_tensor(self._x_window)
        self._learn(x=X_t, y=y)

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicts probabilities for many examples."""
        self._update_observed_features(X)
        X = X[list(self.observed_features)]
        x_win = self._x_window.copy()
        x_win.extend(X.values.tolist())
        if self.append_predict:
            self._x_window = x_win
        self.module.eval()
        with torch.inference_mode():
            x_t = self._deque2rolling_tensor(x_win)
            probas = self.module(x_t).detach().tolist()
        return pd.DataFrame(probas)
