from typing import Callable, Dict, Type, Union

import numpy as np
import pandas as pd
import torch
from ordered_set import OrderedSet
from river import base
from sortedcontainers import SortedSet

from deep_river.base import DeepEstimator, DeepEstimatorInitialized
from deep_river.utils.layer_adaptation import expand_layer
from deep_river.utils.tensor_conversion import (
    df2tensor,
    dict2tensor,
    labels2onehot,
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
        self.softmax = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.nonlinear(self.dense0(x))
        x = self.nonlinear(self.dense1(x))
        return self.softmax(x)


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
        The Module should accept parameter `n_features` so that the
        returned model's input shape can be determined based on the number
        of features in the initial training example.
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
    ...     def __init__(self, n_features):
    ...         super(MyModule, self).__init__()
    ...         self.dense0 = nn.Linear(n_features,5)
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
        module: Type[torch.nn.Module],
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
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            device=device,
            lr=lr,
            is_feature_incremental=is_feature_incremental,
            seed=seed,
            **kwargs,
        )
        self.observed_classes: SortedSet[base.typing.ClfTarget] = SortedSet([])
        self.output_is_logit = output_is_logit
        self.is_class_incremental = is_class_incremental

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
            "module": _TestModule,
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
            "is_class_incremental": True,
        }

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
            of checks to skip during unit testing.
        """
        return set()

    def _learn(self, x: torch.Tensor, y: Union[base.typing.ClfTarget, pd.Series]):
        self.module.train()
        self.optimizer.zero_grad()
        y_pred = self.module(x)
        n_classes = y_pred.shape[-1]
        y_onehot = labels2onehot(
            y=y,
            classes=self.observed_classes,
            n_classes=n_classes,
            device=self.device,
        )
        loss = self.loss_func(y_pred, y_onehot)
        loss.backward()
        self.optimizer.step()
        return self

    def learn_one(self, x: dict, y: base.typing.ClfTarget) -> None:
        """
        Performs one step of training with a single example.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        Classifier
            The classifier itself.
        """

        # check if model is initialized
        if not self.module_initialized:
            self._update_observed_features(x)
            self._update_observed_classes(y)
            self.initialize_module(x=x, **self.kwargs)

        # check last layer
        self._adapt_input_dim(x)
        self._adapt_output_dim(y)

        x_t = dict2tensor(x, features=self.observed_features, device=self.device)

        self._learn(x=x_t, y=y)

    def predict_proba_one(self, x: dict) -> Dict[base.typing.ClfTarget, float]:
        """
        Predict the probability of each label given the input.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        Dict[ClfTarget, float]
            Dictionary of probabilities for each label.
        """

        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.kwargs)

        self._adapt_input_dim(x)

        x_t = dict2tensor(x, features=self.observed_features, device=self.device)

        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t)
        return output2proba(y_pred, self.observed_classes, self.output_is_logit)[0]

    def _update_observed_classes(self, y) -> bool:
        n_existing_classes = len(self.observed_classes)
        if isinstance(y, Union[base.typing.ClfTarget, np.bool_]):  # type: ignore[arg-type]
            self.observed_classes.add(y)
        else:
            self.observed_classes |= y

        if len(self.observed_classes) > n_existing_classes:
            self.observed_classes = OrderedSet(sorted(self.observed_classes))
            return True
        else:
            return False

    def _adapt_output_dim(self, y: base.typing.ClfTarget | pd.Series):
        has_new_class = self._update_observed_classes(y)
        if (
            has_new_class
            and len(self.observed_classes) > 2
            and self.is_class_incremental
        ):
            expand_layer(
                self.output_layer,
                self.output_expansion_instructions,
                len(self.observed_classes),
                output=True,
            )

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        X
            Input examples.
        y
            Target values.

        Returns
        -------
        Classifier
            The classifier itself.
        """
        # check if model is initialized

        if not self.module_initialized:
            self._update_observed_features(X)
            self._update_observed_classes(y)
            self.initialize_module(x=X, **self.kwargs)

        self._adapt_input_dim(X)
        self._adapt_output_dim(y)

        x_t = df2tensor(X, features=self.observed_features, device=self.device)

        self._learn(x=x_t, y=y)

    def predict_proba_many(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the probability of each label given the input.

        Parameters
        ----------
        x
            Input examples.

        Returns
        -------
        pd.DataFrame
            of probabilities for each label.
        """
        if not self.module_initialized:
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.kwargs)

        self._adapt_input_dim(x)
        x_t = df2tensor(x, features=self.observed_features, device=self.device)
        self.module.eval()
        with torch.inference_mode():
            y_preds = self.module(x_t)
        return pd.DataFrame(
            output2proba(y_preds, self.observed_classes, self.output_is_logit)
        )


class ClassifierInitialized(DeepEstimatorInitialized, base.MiniBatchClassifier):
    """
    Wrapper for PyTorch classification models that supports feature and class incremental learning.

    Parameters
    ----------
    module : torch.nn.Module
        A PyTorch model. Can be pre-initialized or uninitialized.
    loss_fn : Union[str, Callable]
        Loss function for training. Can be a string (e.g., 'mse', 'cross_entropy', etc.)
        or a PyTorch function.
    optimizer_fn : Union[str, Type[torch.optim.Optimizer]]
        Optimizer for training (e.g., "adam", "sgd", or a PyTorch optimizer class).
    lr : float, default=0.001
        Learning rate of the optimizer.
    output_is_logit : bool, default=True
        If True, applies softmax/sigmoid during inference.
    is_class_incremental : bool, default=False
        If True, adds neurons to the output layer when new classes appear.
    is_feature_incremental : bool, default=False
        If True, adds neurons to the input layer when new features appear.
    device : str, default="cpu"
        Whether to use "cpu" or "cuda".
    seed : int, default=42
        Random seed for reproducibility.
    **kwargs
        Additional parameters for model initialization.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable],
        optimizer_fn: Union[str, type],
        lr: float = 0.001,
        output_is_logit: bool = True,
        is_class_incremental: bool = False, #todo needs to be tested
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
        self._learn(x_t, y)

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Updates the model with multiple instances for supervised learning.

        The function updates the observed features and targets based on the input
        data. It converts the data from a pandas DataFrame to a tensor format before
        learning occurs. The updates to the model are executed through an internal
        learning mechanism.

        Parameters
        ----------
        X : pd.DataFrame
            The data-frame containing instances to be learned by the model. Each
            row represents a single instance, and each column represents a feature.
        y : pd.Series
            The target values corresponding to the instances in `X`. Each entry in
            the series represents the target associated with a row in `X`.

        Returns
        -------
        None
        """
        self._update_observed_features(X)
        self._update_observed_targets(y)
        x_t = self._df2tensor(X)
        self._learn(x_t, y)

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
            "module": _TestModule(10,1),
            "loss_fn": "binary_cross_entropy_with_logits",
            "optimizer_fn": "sgd",
            "is_feature_incremental": False,
            "is_class_incremental": False,
        }

        yield {
            "module": _TestModule(8,1),
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
