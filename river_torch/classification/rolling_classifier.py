import math
from typing import Callable, Dict, List, Union

import pandas as pd
import torch
from river import base
from river.base.typing import ClfTarget
from torch.nn import init, parameter

from river_torch.base import RollingDeepEstimator
from river_torch.utils.layers import find_output_layer
from river_torch.utils.tensor_conversion import (
    df2rolling_tensor,
    dict2rolling_tensor,
    output2proba,
    labels2onehot,
)

class RollingClassifier(RollingDeepEstimator, base.Classifier):
    """
    Wrapper that feeds a sliding window of the most recent examples to the wrapped PyTorch classification model. The class also automatically handles increases in the number of classes by adding output neurons in case the number of observed classes exceeds the current number of output neurons.

    Parameters
    ----------
    module
        Function that builds the PyTorch classifier to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example. For the dynamic adaptation of the number of possible classes, the returned network should be a torch.nn.Sequential model with a Linear layer as the last module.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss function provided by `torch.nn.functional` or one of the following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy', 'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model. Can be an optimizer class provided by `torch.optim` or one of the following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    window_size
        Number of recent examples to be fed to the wrapped model at each step.
    append_predict
        Whether to append inputs passed for prediction to the rolling window.
    **kwargs
        Parameters to be passed to the `build_fn` function aside from `n_features`.
    """

    def __init__(
        self,
        module: Union[torch.nn.Module, type(torch.nn.Module)],
        loss_fn: Union[str, Callable] = "binary_cross_entropy",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **kwargs,
    ):
        self.observed_classes = []
        self.output_layer = None
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            window_size=window_size,
            append_predict=append_predict,
            device=device,
            lr=lr,
            seed=seed,
            **kwargs,
        )

    @classmethod
    def _unit_test_params(cls) -> dict:
        """
        Returns a dictionary of parameters to be used for unit testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the respective class.
        """

        class MyModule(torch.nn.Module):
            def __init__(self, n_features):
                super().__init__()
                self.hidden_size = 2
                self.n_features=n_features
                self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=self.hidden_size, num_layers=1)
                self.softmax = torch.nn.Softmax(dim=-1)

            def forward(self, X, **kwargs):
                output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
                hn = hn.view(-1, self.hidden_size)
                return self.softmax(hn)

        yield {
            "module": MyModule,
            "optimizer_fn": "sgd",
            "lr": 1e-3,
        }

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    def learn_one(self, x: dict, y: ClfTarget, **kwargs) -> "RollingClassifier":
        """
        Performs one step of training with the most recent training examples stored in the sliding window.

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
            self.kwargs['n_features'] = len(x)
            self.initialize_module(**self.kwargs)

        self._x_window.append(list(x.values()))

        # training process
        x = dict2rolling_tensor(x, self._x_window, device=self.device)
        if x is not None:
            self._learn(x=x, y=y)
        return self


    def _learn(self, x: torch.TensorType, y: Union[ClfTarget,List[ClfTarget]]):
        self.module.train()
        self.optimizer.zero_grad()
        y_pred = self.module(x)
        n_classes = y_pred.shape[-1]
        y = labels2onehot(y=y,classes=self.observed_classes,n_classes=n_classes, device=self.device)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

    def predict_proba_one(self, x: dict) -> Dict[ClfTarget, float]:
        """
        Predict the probability of each label given the most recent examples stored in the sliding window.

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
            self.kwargs['n_features'] = len(x)
            self.initialize_module(**self.kwargs)

        x = dict2rolling_tensor(x, self._x_window, device=self.device)
        if x is not None:
            self.module.eval()
            y_pred = self.module(x)
            proba = output2proba(y_pred, self.observed_classes)
        else:
            proba = self._get_default_proba()
        return proba

    def learn_many(self, X: pd.DataFrame, y: list) -> "RollingClassifier":
        """
        Performs one step of training with the most recent training examples stored in the sliding window.

        Parameters
        ----------
        x
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
            self.kwargs['n_features'] = len(X.columns)
            self.initialize_module(**self.kwargs)

        x = df2rolling_tensor(X, self._x_window, device=self.device)
        y = labels2onehot(
            y,
            self.observed_classes,
            self.output_layer.out_features,
            device=self.device,
        )
        if X is not None:
            self._learn(x=x, y=y)
        return self

    def predict_proba_many(self, X: pd.DataFrame) -> List:
        if not self.module_initialized:
            self.kwargs['n_features'] = len(X.columns)
            self.initialize_module(**self.kwargs)

        batch = df2rolling_tensor(
            X, self._x_window, device=self.device, update_window=self.append_predict
        )

        if batch is not None:
            self.module.eval()
            y_preds = self.module(batch)
            probas = output2proba(y_preds, self.observed_classes)
            if len(probas) < len(X):
                default_proba = self._get_default_proba()
                probas = [default_proba] * (len(X) - len(probas)) + probas
        else:
            default_proba = self._get_default_proba()
            probas = [default_proba] * len(X)
        return probas

    def _get_default_proba(self):
        if len(self.observed_classes) > 0:
            mean_proba = (
                1 / len(self.observed_classes)
                if len(self.observed_classes) != 0
                else 0.0
            )
            proba = {c: mean_proba for c in self.observed_classes}
        else:
            proba = {c: 1.0 for c in self.observed_classes}
        return proba
