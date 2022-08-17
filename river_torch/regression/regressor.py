from typing import Callable, List, Union

import torch
from river import base
from river.base.typing import RegTarget
import pandas as pd

from river_torch.base import DeepEstimator
from river_torch.utils.tensor_conversion import df2tensor, dict2tensor, float2tensor


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
        module: Union[torch.nn.Module, type(torch.nn.Module)],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **kwargs
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            device=device,
            optimizer_fn=optimizer_fn,
            lr=lr,
            seed=seed,
            **kwargs
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
                super(MyModule, self).__init__()

                self.dense0 = torch.nn.Linear(n_features,10)
                self.nonlin = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(0.5)
                self.dense1 = torch.nn.Linear(10, 5)
                self.output = torch.nn.Linear(5, 1)
                self.softmax = torch.nn.Softmax(dim=-1)

            def forward(self, X, **kwargs):
                X = self.nonlin(self.dense0(X))
                X = self.dropout(X)
                X = self.nonlin(self.dense1(X))
                X = self.softmax(self.output(X))
                return X

        yield {
            "module": MyModule,
            "loss_fn": "l1",
            "optimizer_fn": "sgd",
        }

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
        if not self.module_initialized:
            self.kwargs['n_features'] = len(x)
            self.initialize_module(**self.kwargs)
        x = dict2tensor(x, self.device)
        y = float2tensor(y, device=self.device)
        self._learn(x, y)
        return self

    def _learn(self, x: torch.Tensor, y: torch.Tensor):
        self.module.train()
        self.optimizer.zero_grad()
        y_pred = self.module(x)
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
        if not self.module_initialized:
            self.kwargs['n_features'] = len(x)
            self.initialize_module(**self.kwargs)
        x = dict2tensor(x, self.device)
        self.module.eval()
        return self.module(x).item()

    def learn_many(self, X: pd.DataFrame, y: List) -> "Regressor":
        """
        Performs one step of training with a batch of examples.

        Parameters
        ----------
        x
            Input examples.
        y
            Target values.

        Returns
        -------
        Regressor
            The regressor itself.
        """
        if not self.module_initialized:
            self.kwargs['n_features'] = len(X.columns)
            self.initialize_module(**self.kwargs)
        X = df2tensor(X, device=self.device)
        y = torch.tensor(y, device=self.device, dtype=torch.float32)
        self.module.train()
        self._learn(X, y)
        return self

    def predict_many(self, X: pd.DataFrame) -> List:
        """
        Predicts the target value for a batch of examples.

        Parameters
        ----------
        x
            Input examples.

        Returns
        -------
        List
            Predicted target values.
        """
        if not self.module_initialized:
            self.kwargs['n_features'] = len(X.columns)
            self.initialize_module(**self.kwargs)

        X = df2tensor(X, device=self.device)
        self.module.eval()
        return self.module(X).detach().tolist()
