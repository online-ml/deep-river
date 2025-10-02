from typing import Any, Callable, Dict, Union

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

    def __init__(self, n_features, n_outputs=1):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.dense0 = torch.nn.Linear(n_features, 5)
        self.nonlinear = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(5, n_outputs)
        # Softmax entfernt – Ausgaben sind jetzt rohe Logits

    def forward(self, x):
        x = self.nonlinear(self.dense0(x))
        x = self.nonlinear(self.dense1(x))
        return x  # rohe Logits


class Classifier(DeepEstimator, base.MiniBatchClassifier):
    """
    Wrapper for PyTorch classification models that automatically handles
    increases in the number of classes by adding output neurons in case
    the number of observed classes exceeds the current
    number of output neurons.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped.
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
        previously unobserved classes by adding a unit to the output
        layer of the network. This works only if the last trainable
        layer is a nn.Linear layer. Note also, that output activation
        functions can not be adapted, meaning that a binary classifier
        with a sigmoid output can not be altered to perform multi-class
        predictions.
    is_feature_incremental
        Whether the model should adapt to the appearance of
        previously features by adding units to the input
        layer of the network.
    device
        to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **kwargs
        Parameters to be passed to the `build_fn` function aside from
        `n_features`.

    Examples
    --------
    >>> from river import metrics, preprocessing, compose, datasets
    >>> from deep_river import classification
    >>> from torch import nn
    >>> from torch import manual_seed

    >>> _ = manual_seed(42)

    >>> class MyModule(nn.Module):
    ...     def __init__(self):
    ...         super(MyModule, self).__init__()
    ...         self.dense0 = nn.Linear(10,5)
    ...         self.nlin = nn.ReLU()
    ...         self.dense1 = nn.Linear(5, 2)
    ...         self.softmax = nn.Softmax(dim=-1)
    ...
    ...     def forward(self, x, **kwargs):
    ...         x = self.nlin(self.dense0(x))
    ...         x = self.nlin(self.dense1(x))
    ...         x = self.softmax(x)
    ...         return x

    >>> model_pipeline = compose.Pipeline(
    ...     preprocessing.StandardScaler,
    ...     Classifier(module=MyModule,
    ...                loss_fn="binary_cross_entropy",
    ...                optimizer_fn='adam')
    ... )


    >>> dataset = datasets.Phishing()
    >>> metric = metrics.Accuracy()

    >>> for x, y in dataset:
    ...     y_pred = model_pipeline.predict_one(x)  # make a prediction
    ...     metric.update(y, y_pred)  # update the metric
    ...     model_pipeline.learn_one(x,y)

    >>> print(f'Accuracy: {metric.get()}')
    Accuracy: 0.7264
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, type],
        lr: float = 0.001,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,  # todo needs to be tested
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
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
            **kwargs,
        )
        self.output_is_logit = output_is_logit
        self.is_class_incremental = is_class_incremental
        self.observed_classes: SortedSet = SortedSet()

    def learn_one(self, x: dict, y: base.typing.ClfTarget) -> None:
        """Learns from a single example."""
        self._update_observed_features(x)
        self._update_observed_targets(y)
        x_t = self._dict2tensor(x)
        if self.loss_fn == "cross_entropy":
            self._classification_step_cross_entropy(x_t, y)
        else:
            # Fallback: ursprüngliche Logik (One-Hot + beliebige Loss)
            self._learn(x_t, y)

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Batch-Lernen für mehrere Instanzen."""
        self._update_observed_features(X)
        self._update_observed_targets(y)
        x_t = self._df2tensor(X)
        if self.loss_fn == "cross_entropy":
            self._classification_step_cross_entropy(x_t, y)
        else:
            self._learn(x_t, y)

    def _classification_step_cross_entropy(self, x_t: torch.Tensor, y) -> None:
        """Training Schritt für cross_entropy mit Klassenindex Targets.
        Reihenfolge jetzt: Zielindizes berechnen -> ggf. Layer erweitern -> Forward -> Loss.
        """
        # Klassenindizes extrahieren (vor Forward Pass)
        if not isinstance(y, (pd.Series, list, tuple, np.ndarray)):
            class_indices = [self.observed_classes.index(y)]
        else:
            if isinstance(y, pd.Series):
                labels_iter = y.values
            else:
                labels_iter = y
            class_indices = [self.observed_classes.index(lbl) for lbl in labels_iter]

        max_idx = max(class_indices)
        current_out = self._get_output_size()
        if max_idx >= current_out:
            # Sicherheitsguard falls automatische Expansion vorher (in _update_observed_targets) nicht gegriffen hat
            if self.is_class_incremental and self.output_layer is not None:
                self._expand_layer(self.output_layer, target_size=max_idx + 1, output=True)
            else:
                raise RuntimeError(
                    f"Encountered class index {max_idx} but output layer size is {current_out} and expansion is disabled."
                )

        # Forward Pass nach sicherer Dimension
        self.module.train()
        y_pred = self.module(x_t)

        y_idx = torch.tensor(class_indices, device=self.device, dtype=torch.long)
        if y_idx.ndim == 1 and y_pred.shape[0] == 1:
            # Single sample Fall (reshape auf (1,)) ist korrekt
            pass
        # Loss & Optimierung
        self.optimizer.zero_grad()
        loss = self.loss_func(y_pred, y_idx)
        loss.backward()
        self.optimizer.step()

    def _update_observed_targets(self, y) -> bool:
        """
        Updates the set of observed classes with new classes from the provided target(s).
        If new classes are detected, the method expands the output layer when the model
        is class-incremental and an output layer exists.

        Parameters
        ----------
        y : ClfTarget or bool or array-like
            The target(s) from which new class(es) are to be observed. Can either
            be a single classification target, a boolean value, or an iterable of
            targets.

        Returns
        -------
        bool
            Returns True if new classes were detected and the set of observed
            classes was updated, otherwise False.
        """
        n_existing_classes = len(self.observed_classes)
        # Add the new class(es) from y.
        if isinstance(y, (base.typing.ClfTarget, np.bool_)):  # type: ignore[arg-type]
            self.observed_classes.add(y)
        else:
            self.observed_classes |= set(y)

        if len(self.observed_classes) > n_existing_classes:
            # Expand the output layer to match the new number of classes.
            if self.is_class_incremental and self.output_layer:
                self._expand_layer(
                    self.output_layer,
                    target_size=len(self.observed_classes),
                    output=True,
                )
            return True
        else:
            return False

    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        """Predicts probabilities for a single example."""
        self._update_observed_features(x)
        x_t = self._dict2tensor(x)
        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t)
        return output2proba(y_pred, self.observed_classes, self.output_is_logit)[0]

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicts probabilities for multiple examples."""
        self._update_observed_features(X)
        x_t = self._df2tensor(X)
        self.module.eval()
        with torch.inference_mode():
            y_preds = self.module(x_t)
        return pd.DataFrame(
            output2proba(y_preds, self.observed_classes, self.output_is_logit)
        )

    @classmethod
    def _unit_test_params(cls):
        """Provides default parameters for unit testing."""
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
        """Defines unit tests to skip."""
        return {
            "check_shuffle_features_no_impact",
        }
