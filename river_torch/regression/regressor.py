import copy
from typing import Callable, Union

import torch
from river import base
from river.base.typing import RegTarget

from river_torch.base import DeepEstimator, RollingDeepEstimator
from river_torch.utils.river_compat import dict2tensor, list2tensor, scalar2tensor


class Regressor(DeepEstimator, base.Regressor):
    """
    Wrapper for PyTorch regression models that enables compatibility with River.

    Parameters
    ----------
    build_fn
        Function that builds the PyTorch regressor to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example. For the dynamic adaptation of the number of possible classes, the returned network should be a torch.nn.Sequential model with a Linear layer as the last module.
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
    **net_params
        Parameters to be passed to the `build_fn` function aside from `n_features`.

    Examples
    --------

    >>> from river.datasets import TrumpApproval
    >>> from river import evaluate, metrics, preprocessing
    >>> from river_torch.regression import Regressor
    >>> import torch
    >>> from torch import nn, optim

    >>> _ = torch.manual_seed(0)

    >>> dataset = TrumpApproval()
    >>> def build_torch_mlp(n_features):
    ...     net = nn.Sequential(
    ...         nn.Linear(n_features, 5),
    ...         nn.ReLU(),
    ...         nn.Linear(5, 1)
    ...     )
    ...     return net
    ...


    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     Regressor(
    ...         build_fn=build_torch_mlp,
    ...         loss_fn='mse',
    ...         optimizer_fn=torch.optim.SGD,
    ...         batch_size=2
    ...     )
    ... )
    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric).get()
    1.3456

    """

    def __init__(
        self,
        build_fn: Callable,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **net_params
    ):
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            device=device,
            optimizer_fn=optimizer_fn,
            lr=lr,
            seed=seed,
            **net_params
        )

    def learn_one(self, x: dict, y: RegTarget) -> "Regressor":
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
        Regressor
            The regressor itself.
        """
        if self.net is None:
            self._init_net(len(x))
        x = dict2tensor(x, self.device)
        y = scalar2tensor(y, device=self.device)
        self.net.train()
        self._learn_one(x, y)
        return self

    def _learn_one(self, x: torch.TensorType, y: torch.TensorType):
        self.optimizer.zero_grad()
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

    def predict_one(self, x: dict) -> RegTarget:
        """
        Predicts the target value for a single example.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        RegTarget
            Predicted target value.
        """
        if self.net is None:
            self._init_net(len(x))
        x = dict2tensor(x, self.device)
        self.net.eval()
        return self.net(x).item()


class RollingRegressor(RollingDeepEstimator, base.Regressor):
    """
    Wrapper that feeds a sliding window of the most recent examples to the wrapped PyTorch regression model.

    Parameters
    ----------
    build_fn
        Function that builds the PyTorch regressor to be wrapped. The function should accept parameter `n_features` so that the returned model's input shape can be determined based on the number of features in the initial training example. For the dynamic adaptation of the number of possible classes, the returned network should be a torch.nn.Sequential model with a Linear layer as the last module.
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
    **net_params
        Parameters to be passed to the `build_fn` function aside from `n_features`.
    """

    def __init__(
        self,
        build_fn: Callable,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        window_size: int = 10,
        append_predict: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **net_params
    ):
        super().__init__(
            build_fn=build_fn,
            loss_fn=loss_fn,
            device=device,
            optimizer_fn=optimizer_fn,
            lr=lr,
            window_size=window_size,
            append_predict=append_predict,
            seed=seed,
            **net_params
        )

    def predict_one(self, x: dict) -> RegTarget:
        """
        Predicts the target value for the current sliding window of most recent examples.

        Parameters
        ----------
        x
            Input example.

        Returns
        -------
        RegTarget
            Predicted target value.
        """
        if self.net is None:
            self._init_net(len(x))
        if len(self._x_window) == self.window_size:

            if self.append_predict:
                self._x_window.append(list(x.values()))
                x = list2tensor(self._x_window, self.device)
            else:
                x = copy.deepcopy(self._x_window)
                x.append(list(x.values()))
                x = list2tensor(x, self.device)
            self.net.eval()
            return self.net(x).item()
        else:
            return 0.0

    def learn_one(self, x: dict, y: RegTarget) -> "RollingRegressor":
        """
        Performs one step of training with the sliding window of the most recent examples.

        Parameters
        ----------
        x
            Input example.
        y
            Target value.

        Returns
        -------
        RollingRegressor
            The regressor itself.
        """
        if self.net is None:
            self._init_net(len(x))

        self._x_window.append(list(x.values()))
        if len(self._x_window) == self.window_size:
            x = list2tensor(self._x_window, device=self.device)
            y = scalar2tensor(y, device=self.device)
            self.net.train()
            self._learn_window(x, y)

        return self

    def _learn_window(self, x: torch.TensorType, y: torch.TensorType):
        self.optimizer.zero_grad()
        y_pred = self.net(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
