from typing import Callable, Dict, Type, Union, cast

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
        output, (hn, cn) = self.lstm(X)
        x = hn.view(-1, self.hidden_size)
        x = self.linear(x)
        return torch.nn.functional.softmax(x, dim=-1)


class RollingClassifier(Classifier, RollingDeepEstimator):
    """Rolling window variant of :class:`Classifier`.

    Maintains a fixed-size deque of the most recent observations (``window_size``)
    and feeds them as a temporal slice to the underlying module. This allows
    simple sequence conditioning without full recurrent state management or
    replay buffers.

    Parameters
    ----------
    module : torch.nn.Module
        Classification module consuming a rolling tensor shaped roughly as
        (seq_len, batch=1, n_features) depending on internal conversion.
    loss_fn : str | Callable, default='binary_cross_entropy_with_logits'
        Loss identifier or callable.
    optimizer_fn : str | type, default='sgd'
        Optimizer specification.
    lr : float, default=1e-3
        Learning rate.
    output_is_logit : bool, default=True
        Whether raw logits are produced (enables post-softmax via ``output2proba``).
    is_class_incremental : bool, default=False
        Expand output layer when new class labels appear.
    is_feature_incremental : bool, default=False
        Expand input layer when new feature names appear.
    device : str, default='cpu'
        Torch device.
    seed : int, default=42
        Random seed.
    window_size : int, default=10
        Number of past samples kept.
    append_predict : bool, default=False
        If True, predictions are appended to internal window during inference
        (useful for autoregressive generation).
    gradient_clip_value : float | None, default=None
        Optional gradient clipping threshold.
    **kwargs
        Forwarded to parent constructors.

    Examples
    --------
    Deterministisches Beispiel mit festverdrahteter Klassenvorhersage::

    >>> import torch
    >>> from torch import nn
    >>> from deep_river.classification.rolling_classifier import RollingClassifier
    >>> class FixedRNN(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.rnn = nn.RNN(3, 4)
    ...         self.head = nn.Linear(4, 2)
    ...         with torch.no_grad():
    ...             # Alle Gewichte nullen, Bias setzt klare Logits
    ...             for p in self.rnn.parameters():
    ...                 p.zero_()
    ...             self.head.weight.zero_()
    ...             self.head.bias[:] = torch.tensor([2.0, -1.0])
    ...     def forward(self, x):
    ...         out, _ = self.rnn(x)
    ...         return self.head(out[-1])
    >>> rc = RollingClassifier(module=FixedRNN(), loss_fn='cross_entropy', optimizer_fn='sgd', lr=0.0, window_size=5)
    >>> # Klassen zuerst registrieren
    >>> rc.learn_one({'a':0,'b':0,'c':0}, 0)
    >>> rc.learn_one({'a':1,'b':1,'c':1}, 1)
    >>> rc.predict_one({'a':5,'b':6,'c':7})
    0
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
        gradient_clip_value: float | None = None,
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
            gradient_clip_value=gradient_clip_value,
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
        """Learn from a single (x, y) updating the rolling window."""
        self._update_observed_features(x)
        self._update_observed_targets(y)
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])
        x_t = self._deque2rolling_tensor(self._x_window)
        self._learn(x=x_t, y=y)

    def predict_proba_one(self, x: dict) -> Dict[ClfTarget, float]:
        """Return class probability mapping for one sample using rolling context."""
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
        return cast(Dict[ClfTarget, float], proba[0])

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Batch update: extend window with rows of X and perform a step."""
        self._update_observed_targets(y)
        self._update_observed_features(X)
        X = X[list(self.observed_features)]
        self._x_window.extend(X.values.tolist())
        X_t = self._deque2rolling_tensor(self._x_window)
        self._learn(x=X_t, y=y)

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return probability DataFrame for multiple samples with rolling context."""
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
