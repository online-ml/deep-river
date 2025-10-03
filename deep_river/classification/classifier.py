from typing import Callable, Union, cast

import numpy as np
import pandas as pd
import torch
from river import base
from sortedcontainers import SortedSet

from deep_river.base import DeepEstimator
from deep_river.utils.tensor_conversion import (
    output2proba,
)


class _TestModule(torch.nn.Module):
    """Small feed-forward network used in unit tests."""

    def __init__(self, n_features, n_outputs=1):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.dense0 = torch.nn.Linear(n_features, 5)
        self.nonlinear = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(5, n_outputs)
        # Note: no softmax so the module outputs raw logits for flexibility

    def forward(self, x):
        x = self.nonlinear(self.dense0(x))
        x = self.nonlinear(self.dense1(x))
        return x  # raw logits


class Classifier(DeepEstimator, base.MiniBatchClassifier):
    """Incremental PyTorch classifier with optional dynamic feature & class growth.

    This wrapper turns an arbitrary ``torch.nn.Module`` into an incremental
    classifier that follows the :mod:`river` API. It can optionally expand its
    input dimensionality when previously unseen feature names occur
    (``is_feature_incremental=True``) and expand the output layer when new class
    labels appear (``is_class_incremental=True``).

    When ``loss_fn='cross_entropy'`` targets are handled as integer class indices;
    otherwise they are converted to one-hot vectors to match the output dimension.

    Parameters
    ----------
    module : torch.nn.Module
        The underlying PyTorch model producing (logit) outputs.
    loss_fn : str | Callable
        Loss identifier (e.g. ``'cross_entropy'``, ``'mse'``) or a callable.
    optimizer_fn : str | type
        Optimizer identifier (``'adam'``, ``'sgd'``, etc.) or an optimizer class.
    lr : float, default=1e-3
        Learning rate passed to the optimizer.
    output_is_logit : bool, default=True
        If True, ``predict_proba_*`` will apply a softmax (multi-class) or sigmoid
        (binary) as needed using :func:`output2proba`.
    is_class_incremental : bool, default=False
        Whether to expand the output layer when new class labels appear.
    is_feature_incremental : bool, default=False
        Whether to expand the input layer when new feature names are observed.
    device : str, default='cpu'
        Runtime device.
    seed : int, default=42
        Random seed.
    gradient_clip_value : float | None, default=None
        Norm to clip gradients to (disabled if ``None``).
    **kwargs
        Extra parameters retained for reconstruction.

    Examples
    --------
    Basic usage with a custom module::

        >>> from river import metrics, datasets, preprocessing, compose
        >>> from deep_river.classification import Classifier
        >>> from torch import nn, manual_seed
        >>> _ = manual_seed(42)
        >>> class MyModule(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = nn.Linear(10, 5)
        ...         self.act = nn.ReLU()
        ...         self.fc2 = nn.Linear(5, 2)
        ...     def forward(self, x):
        ...         return self.fc2(self.act(self.fc1(x)))  # logits
        >>> pipeline = compose.Pipeline(
        ...     preprocessing.StandardScaler(),
        ...     Classifier(module=MyModule(), loss_fn='cross_entropy', optimizer_fn='adam')
        ... )
        >>> metric = metrics.Accuracy()
        >>> for x, y in datasets.Phishing().take(50):
        ...     y_pred = pipeline.predict_one(x)
        ...     metric.update(y, y_pred)
        ...     pipeline.learn_one(x, y)
        >>> round(metric.get(), 4)  # doctest: +SKIP
        0.70
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, type],
        lr: float = 0.001,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        gradient_clip_value: float | None = None,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            is_feature_incremental=is_feature_incremental,
            gradient_clip_value=gradient_clip_value,
            **kwargs,
        )
        self.output_is_logit = output_is_logit
        self.is_class_incremental = is_class_incremental
        self.observed_classes: SortedSet = SortedSet()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def learn_one(self, x: dict, y: base.typing.ClfTarget) -> None:
        """Learn from a single instance.

        Parameters
        ----------
        x : dict
            Feature dictionary.
        y : hashable
            Class label.
        """
        self._update_observed_features(x)
        self._update_observed_targets(y)
        x_t = self._dict2tensor(x)
        if self.loss_fn == "cross_entropy":
            self._classification_step_cross_entropy(x_t, y)
        else:
            # One-hot pathway / other losses
            self._learn(x_t, y)

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Learn from a batch of instances.

        Parameters
        ----------
        X : pandas.DataFrame
            Batch of feature rows.
        y : pandas.Series
            Corresponding labels.
        """
        self._update_observed_features(X)
        self._update_observed_targets(y)
        x_t = self._df2tensor(X)
        if self.loss_fn == "cross_entropy":
            self._classification_step_cross_entropy(x_t, y)
        else:
            self._learn(x_t, y)

    def _classification_step_cross_entropy(self, x_t: torch.Tensor, y) -> None:
        """Internal training step for cross entropy with index targets.

        Steps
        -----
        1. Convert labels to class indices.
        2. Expand the output layer if a new class index exceeds current size and
           ``is_class_incremental`` is enabled.
        3. Forward pass and backprop.
        """
        if not isinstance(y, (pd.Series, list, tuple, np.ndarray)):
            class_indices = [self.observed_classes.index(y)]
        else:
            labels_iter = y.values if isinstance(y, pd.Series) else y
            class_indices = [self.observed_classes.index(lbl) for lbl in labels_iter]

        max_idx = max(class_indices)
        current_out = self._get_output_size()
        if max_idx >= current_out:
            if self.is_class_incremental and self.output_layer is not None:
                self._expand_layer(
                    self.output_layer, target_size=max_idx + 1, output=True
                )
            else:
                raise RuntimeError(
                    f"Encountered class index {max_idx} but output layer size is "
                    f"{current_out} and expansion is disabled."
                )

        self.module.train()
        y_pred = self.module(x_t)
        y_idx = torch.tensor(class_indices, device=self.device, dtype=torch.long)
        self.optimizer.zero_grad()
        loss = self.loss_func(y_pred, y_idx)
        loss.backward()
        if getattr(self, "gradient_clip_value", None) is not None:
            clip_val = self.gradient_clip_value
            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), clip_val)
        self.optimizer.step()

    def _update_observed_targets(self, y) -> bool:
        """Update the set of observed class labels; expand output layer if needed.

        Parameters
        ----------
        y : ClfTarget | array-like
            Single label or iterable of labels.

        Returns
        -------
        bool
            True if new classes were added.
        """
        n_existing = len(self.observed_classes)
        if isinstance(y, (base.typing.ClfTarget, np.bool_)):  # type: ignore[arg-type]
            self.observed_classes.add(y)
        else:
            self.observed_classes |= set(y)

        if len(self.observed_classes) > n_existing:
            if self.is_class_incremental and self.output_layer:
                self._expand_layer(
                    self.output_layer,
                    target_size=len(self.observed_classes),
                    output=True,
                )
            return True
        return False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        """Predict class membership probabilities for one instance.

        Parameters
        ----------
        x : dict
            Feature dictionary.

        Returns
        -------
        dict
            Mapping from label -> probability.
        """
        self._update_observed_features(x)
        x_t = self._dict2tensor(x)
        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t)
        raw = output2proba(y_pred, self.observed_classes, self.output_is_logit)[0]
        return cast(dict[base.typing.ClfTarget, float], raw)

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict probabilities for a batch of instances.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        Returns
        -------
        pandas.DataFrame
            Each row sums to 1 (multi-class) or has two columns for binary.
        """
        self._update_observed_features(X)
        x_t = self._df2tensor(X)
        self.module.eval()
        with torch.inference_mode():
            y_preds = self.module(x_t)
        return pd.DataFrame(
            output2proba(y_preds, self.observed_classes, self.output_is_logit)
        )

    # ------------------------------------------------------------------
    # Test utilities
    # ------------------------------------------------------------------
    @classmethod
    def _unit_test_params(cls):
        """Provide standard parameter sets used in the test suite."""
        yield {
            "module": _TestModule(10, 1),
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "is_class_incremental": False,
        }

        yield {
            "module": _TestModule(8, 1),
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
            "is_class_incremental": False,
        }

        yield {
            "module": _TestModule(10, 1),
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "is_class_incremental": True,
        }

        yield {
            "module": _TestModule(8, 1),
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
            "is_class_incremental": True,
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
        """Return names of test checks to skip for this estimator."""
        return {
            "check_shuffle_features_no_impact",
        }
