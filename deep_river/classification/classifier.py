import math
import warnings
from typing import Callable, Dict, List, Type, Union, cast

import pandas as pd
import torch
from ordered_set import OrderedSet
from river import base
from river.base.typing import ClfTarget
from torch import nn
from torch.utils.hooks import RemovableHandle

from deep_river.base import DeepEstimator
from deep_river.utils.hooks import ForwardOrderTracker, apply_hooks
from deep_river.utils.tensor_conversion import (
    df2tensor,
    dict2tensor,
    labels2onehot,
    output2proba,
)


class _TestModule(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.dense0 = torch.nn.Linear(n_features, 5)
        self.nonlin = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(5, 2)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.nonlin(self.dense1(X))
        X = self.softmax(X)
        return X


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
            previously unobserved classes by adding an unit to the output
            layer of the network. This works only if the last trainable
            layer is an nn.Linear layer. Note also, that output activation
            functions can not be adapted, meaning that a binary classifier
            with a sigmoid output can not be altered to perform multi-class
            predictions.
        device
            Device to run the wrapped model on. Can be "cpu" or "cuda".
        seed
            Random seed to be used for training the wrapped model.
        **net_params
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
    ...         self.nonlin = nn.ReLU()
    ...         self.dense1 = nn.Linear(5, 2)
    ...         self.softmax = nn.Softmax(dim=-1)
    ...
    ...     def forward(self, X, **kwargs):
    ...         X = self.nonlin(self.dense0(X))
    ...         X = self.nonlin(self.dense1(X))
    ...         X = self.softmax(X)
    ...         return X

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
    ...     metric = metric.update(y, y_pred)  # update the metric
    ...     model_pipeline = model_pipeline.learn_one(x,y)

    >>> print(f'Accuracy: {metric.get()}')
    Accuracy: 0.6728
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "binary_cross_entropy_with_logits",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        output_is_logit: bool = True,
        is_class_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            module=module,
            device=device,
            lr=lr,
            seed=seed,
            **kwargs,
        )
        self.observed_classes: OrderedSet[ClfTarget] = OrderedSet()
        self.output_layer: nn.Module
        self.output_is_logit = output_is_logit
        self.is_class_incremental = is_class_incremental
        self._supported_output_layers: List[Type[nn.Module]] = [nn.Linear]

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
            Set of checks to skip during unit testing.
        """
        return {
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def _learn(self, x: torch.Tensor, y: Union[ClfTarget, List[ClfTarget]]):
        self.module.train()
        self.optimizer.zero_grad()
        y_pred = self.module(x)
        n_classes = y_pred.shape[-1]
        y = labels2onehot(
            y=y,
            classes=self.observed_classes,
            n_classes=n_classes,
            device=self.device,
        )
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return self

    def learn_one(self, x: dict, y: ClfTarget, **kwargs) -> "Classifier":
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
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)
        x_t = dict2tensor(x, device=self.device)

        # check last layer
        self.observed_classes.add(y)
        if self.is_class_incremental:
            self._adapt_output_dim()

        return self._learn(x=x_t, y=y)

    def predict_proba_one(self, x: dict) -> Dict[ClfTarget, float]:
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
            self.kwargs["n_features"] = len(x)
            self.initialize_module(**self.kwargs)
        x_t = dict2tensor(x, device=self.device)
        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t)
        return output2proba(
            y_pred, self.observed_classes, self.output_is_logit
        )[0]

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> "Classifier":
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
            self.kwargs["n_features"] = len(X.columns)
            self.initialize_module(**self.kwargs)
        X = df2tensor(X, device=self.device)

        self.observed_classes.update(y)
        if self.is_class_incremental:
            self._adapt_output_dim()

        return self._learn(x=X, y=y)

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the probability of each label given the input.

        Parameters
        ----------
        X
            Input examples.

        Returns
        -------
        pd.DataFrame
            DataFrame of probabilities for each label.
        """
        if not self.module_initialized:
            self.kwargs["n_features"] = len(X.columns)
            self.initialize_module(**self.kwargs)
        X_t = df2tensor(X, device=self.device)
        self.module.eval()
        with torch.inference_mode():
            y_preds = self.module(X_t)
        return pd.DataFrame(output2proba(y_preds, self.observed_classes))

    def _adapt_output_dim(self):
        out_features_target = (
            len(self.observed_classes) if len(self.observed_classes) > 2 else 1
        )
        n_classes_to_add = out_features_target - self.output_layer.out_features
        if n_classes_to_add > 0:
            self._add_output_features(n_classes_to_add)

    def _add_output_features(self, n_classes_to_add: int) -> None:
        """
        Adds output dimensions to the model by adding new rows of weights to
        the existing weights of the last layer.

        Parameters
        ----------
        n_classes_to_add
            Number of output dimensions to add.
        """
        new_weights = (
            torch.mean(cast(torch.Tensor, self.output_layer.weight), dim=0)
            .unsqueeze(1)
            .T
        )
        if n_classes_to_add > 1:
            new_weights = (
                new_weights.unsqueeze(1)
                .T.repeat(1, n_classes_to_add, 1)
                .squeeze()
            )
        self.output_layer.weight = nn.parameter.Parameter(
            torch.cat(
                [
                    cast(torch.Tensor, self.output_layer.weight),
                    cast(torch.Tensor, new_weights),
                ],
                dim=0,
            )
        )

        if self.output_layer.bias is not None:
            new_bias = torch.empty(n_classes_to_add)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.output_layer.weight
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(new_bias, -bound, bound)
            self.output_layer.bias = nn.parameter.Parameter(
                torch.cat(
                    [
                        cast(torch.Tensor, self.output_layer.bias),
                        cast(torch.Tensor, new_bias),
                    ],
                    dim=0,
                )
            )
        self.output_layer.out_features += torch.Tensor([n_classes_to_add])
        self.optimizer = self.optimizer_fn(
            self.module.parameters(), lr=self.lr
        )

    def find_output_layer(self, n_features: int):

        handles: List[RemovableHandle] = []
        tracker = ForwardOrderTracker()
        apply_hooks(module=self.module, hook=tracker, handles=handles)

        x_dummy = torch.empty((1, n_features), device=self.device)
        self.module(x_dummy)

        for h in handles:
            h.remove()

        if tracker.ordered_modules and isinstance(
            tracker.ordered_modules[-1], tuple(self._supported_output_layers)
        ):
            self.output_layer = tracker.ordered_modules[-1]
        else:
            warnings.warn(
                "The model will not be able to adapt its output to new "
                "classes since no linear layer output layer was found."
            )
            self.is_class_incremental = False

    def initialize_module(self, **kwargs):
        super().initialize_module(**kwargs)
        self.find_output_layer(n_features=kwargs["n_features"])
