from typing import Callable, Dict, Type, Union

import pandas as pd
import torch
from river.base.typing import ClfTarget

from deep_river.base import RollingDeepEstimator
from deep_river.classification import Classifier
from deep_river.utils.tensor_conversion import deque2rolling_tensor, output2proba


class _TestLSTM(torch.nn.Module):
    def __init__(self, n_features, output_size=1):
        super().__init__()
        self.n_features = n_features
        self.output_size = output_size
        self.lstm = torch.nn.LSTM(
            input_size=n_features, hidden_size=output_size, num_layers=1
        )

    def forward(self, X, **kwargs):
        # lstm with input, hidden, and internal state
        output, (hn, cn) = self.lstm(X)
        hn = hn.view(-1, self.output_size)
        return hn


class RollingClassifier(Classifier, RollingDeepEstimator):
    """
    Wrapper that feeds a sliding window of the most recent examples to the
    wrapped PyTorch classification model. The class also automatically handles
    increases in the number of classes by adding output neurons in case the
    number of observed classes exceeds the current number of output neurons.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped.
        The Module should accept parameter `n_features` so that the returned
        model's input shape can be determined based on the number of features
        in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a
        loss function provided by `torch.nn.functional` or one of the
        following: 'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model. Can be an
        optimizer class provided by `torch.optim` or one of the following:
        "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    output_is_logit
            Whether the module produces logits as output. If true, either
            softmax or sigmoid is applied to the outputs when predicting.
    is_class_incremental
        Whether the classifier should adapt to the appearance of previously
        unobserved classes by adding an unit to the output
        layer of the network. This works only if the last trainable layer
        is an nn.Linear layer. Note also, that output activation functions
        can not be adapted, meaning that a binary classifier with a sigmoid
        output can not be altered to perform multi-class predictions.
    is_feature_incremental
        Whether the model should adapt to the appearance of
        previously features by adding units to the input
        layer of the network.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    window_size
        Number of recent examples to be fed to the wrapped model at each step.
    append_predict
        Whether to append inputs passed for prediction to the rolling window.
    **kwargs
        Parameters to be passed to the `build_fn`
        function aside from `n_features`.

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
        window_size: int = 10,
        append_predict: bool = False,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            is_class_incremental=is_class_incremental,
            is_feature_incremental=is_feature_incremental,
            device=device,
            seed=seed,
            window_size=window_size,
            append_predict=append_predict,
            **kwargs,
        )
        output_is_logit = output_is_logit

    @classmethod
    def _unit_test_params(cls):
        """
        Returns a dictionary of parameters to be used for unit testing
        the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing
            the respective class.
        """

        yield {
            "module": _TestLSTM,
            "optimizer_fn": "sgd",
            "lr": 1e-3,
            "is_feature_incremental": True,
            "is_class_incremental": True,
        }

    @classmethod
    def _unit_test_skips(cls) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However,
        in some cases, some estimators might not
        be able to pass certain checks.
        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
            # Test fails since `sum(y_pred)` call in test produces large
            # floating point error.
            "check_predict_proba_one",
        }

    def learn_one(self, x: dict, y: ClfTarget, **kwargs) -> "RollingClassifier":
        """
        Performs one step of training with the most recent training examples
        stored in the sliding window.

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

        if not self.module_initialized:
            self._update_observed_classes(y)
            self._update_observed_features(x)
            self.initialize_module(x=x, **self.kwargs)

        self._adapt_input_dim(x)
        self._adapt_output_dim(y)
        self._x_window.append([x.get(feature, 0) for feature in self.observed_features])

        # training process
        if len(self._x_window) == self.window_size:
            x_t = deque2rolling_tensor(self._x_window, device=self.device)
            return self._learn(x=x_t, y=y)
        return self

    def predict_proba_one(self, x: dict) -> Dict[ClfTarget, float]:
        """
        Predict the probability of each label given the most recent examples
        stored in the sliding window.

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
        x_win = self._x_window.copy()
        x_win.append([x.get(feature, 0) for feature in self.observed_features])
        if self.append_predict:
            self._x_window = x_win

        self.module.eval()
        with torch.inference_mode():
            x_t = deque2rolling_tensor(x_win, device=self.device)
            y_pred = self.module(x_t)
            proba = output2proba(y_pred, self.observed_classes, self.output_is_logit)

        return proba[0]

    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> "RollingClassifier":
        """
        Performs one step of training with the most recent training examples
        stored in the sliding window.

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
            self._update_observed_classes(y)
            self._update_observed_features(X)
            self.initialize_module(x=X, **self.kwargs)

        self._adapt_input_dim(X)
        self._adapt_output_dim(y)
        X = X[list(self.observed_features)]
        self._x_window.extend(X.values.tolist())

        if self.is_class_incremental:
            self._adapt_output_dim(y)

        if len(self._x_window) == self.window_size:
            X_t = deque2rolling_tensor(self._x_window, device=self.device)
            self._learn(x=X_t, y=y)
        return self

    def predict_proba_many(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the probability of each label given the most recent examples

        Parameters
        ----------
        x

        Returns
        -------
        pd.DataFrame
            DataFrame of probabilities for each label.
        """
        if not self.module_initialized:

            self._update_observed_features(x)
            self.initialize_module(x=x, **self.kwargs)

        self._adapt_input_dim(x)
        x = x[list(self.observed_features)]
        x_win = self._x_window.copy()
        x_win.extend(x.values.tolist())
        if self.append_predict:
            self._x_window = x_win

        self.module.eval()
        with torch.inference_mode():
            x_t = deque2rolling_tensor(x_win, device=self.device)
            probas = self.module(x_t).detach().tolist()
        return pd.DataFrame(probas)
