import collections
import inspect
import warnings
from typing import Any, Callable, Deque, Dict, List, Type, Union, cast

import pandas as pd
import torch
from river import base
from sortedcontainers import SortedSet
from torch.utils.hooks import RemovableHandle

from deep_river.utils import df2tensor, dict2tensor, get_loss_fn, get_optim_fn
from deep_river.utils.hooks import ForwardOrderTracker, apply_hooks
from deep_river.utils.layer_adaptation import (
    SUPPORTED_LAYERS,
    expand_layer,
    load_instructions,
)

try:
    from graphviz import Digraph
    from torchviz import make_dot
except ImportError as e:
    raise ValueError("You have to install graphviz to use the draw method") from e


class DeepEstimator(base.Estimator):
    """
    Abstract base class that implements basic functionality of
    River-compatible PyTorch wrappers.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped.
        The Module should accept parameter `n_features` so that the returned
        model's input shape can be determined based on the number of features
        in the initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss
        function provided by `torch.nn.functional` or one of the following:
        'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    **kwargs
        Parameters to be passed to the `Module` or the `optimizer`.
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        is_feature_incremental: bool = False,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        self.module_cls = module
        self.module: torch.nn.Module = cast(torch.nn.Module, None)
        self.loss_func = get_loss_fn(loss_fn)
        self.loss_fn = loss_fn
        self.optimizer_func = get_optim_fn(optimizer_fn)
        self.optimizer_fn = optimizer_fn
        self.is_feature_incremental = is_feature_incremental
        self.is_class_incremental: bool = False
        self.observed_features: SortedSet[str] = SortedSet([])
        self.lr = lr
        self.device = device
        self.kwargs = kwargs
        self.seed = seed
        self.input_layer = cast(torch.nn.Module, None)
        self.input_expansion_instructions = cast(Dict, None)
        self.output_layer = cast(torch.nn.Module, None)
        self.output_expansion_instructions = cast(Dict, None)
        self.module_initialized = False
        torch.manual_seed(seed)

    def _filter_kwargs(self, fn: Callable, override=None, **kwargs) -> dict:
        """Filters `net_params` and returns those in `fn`'s arguments.

        Parameters
        ----------
        fn
            Arbitrary function
        override
            Dictionary, values to override `torch_params`

        Returns
        -------
        dict
            Dictionary containing variables in both `sk_params` and
            `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in kwargs.items():
            args = list(inspect.signature(fn).parameters)
            if name in args:
                res.update({name: value})
        res.update(override)
        return res

    def draw(self) -> Digraph:
        """Draws the wrapped model."""
        first_parameter = next(self.module.parameters())
        input_shape = first_parameter.size()
        y_pred = self.module(torch.rand(input_shape))
        return make_dot(y_pred.mean(), params=dict(self.module.named_parameters()))

    def initialize_module(self, x: dict | pd.DataFrame, **kwargs):
        """
        Parameters
        ----------
        module
          The instance or class or callable to be initialized, e.g.
          ``self.module``.
        kwargs : dict
          The keyword arguments to initialize the instance or class. Can be an
          empty dict.
        Returns
        -------
        instance
          The initialized component.
        """
        torch.manual_seed(self.seed)
        if isinstance(x, Dict):
            n_features = len(x)
        elif isinstance(x, pd.DataFrame):
            n_features = len(x.columns)

        if not isinstance(self.module_cls, torch.nn.Module):
            self.module = self.module_cls(
                n_features=n_features,
                **self._filter_kwargs(self.module_cls, kwargs),
            )

        self.module.to(self.device)
        self.optimizer = self.optimizer_func(self.module.parameters(), lr=self.lr)
        self.module_initialized = True

        self._get_input_output_layers(n_features=n_features)

    def clone(
        self,
        new_params: dict[Any, Any] | None = None,
        include_attributes=False,
    ):
        """Clones the estimator.

        Parameters
        ----------
        new_params
            New parameters to be passed to the cloned estimator.
        include_attributes
            If True, the attributes of the estimator will be copied to the
            cloned estimator. This is useful when the estimator is a
            transformer and the attributes are the learned parameters.

        Returns
        -------
        DeepEstimator
            The cloned estimator.
        """
        new_params = new_params or {}
        new_params.update(self.kwargs)
        new_params.update(self._get_params())
        new_params.update({"module": self.module_cls})

        clone = self.__class__(**new_params)
        if include_attributes:
            clone.__dict__.update(self.__dict__)
        return clone

    def _adapt_input_dim(self, x: Dict | pd.DataFrame):
        has_new_feature = self._update_observed_features(x)

        if has_new_feature and self.is_feature_incremental:
            expand_layer(
                self.input_layer,
                self.input_expansion_instructions,
                len(self.observed_features),
                output=False,
            )

    def _update_observed_features(self, x):
        n_existing_features = len(self.observed_features)
        if isinstance(x, Dict):
            self.observed_features |= x.keys()
        else:
            self.observed_features |= x.columns

        if len(self.observed_features) > n_existing_features:
            self.observed_features = SortedSet(self.observed_features)
            return True
        else:
            return False

    def _get_input_output_layers(self, n_features: int):
        handles: List[RemovableHandle] = []
        tracker = ForwardOrderTracker()
        apply_hooks(module=self.module, hook=tracker, handles=handles)

        x_dummy = torch.empty((1, n_features), device=self.device)
        self.module(x_dummy)

        for h in handles:
            h.remove()

        if self.is_class_incremental:
            if tracker.ordered_modules and isinstance(
                tracker.ordered_modules[-1], SUPPORTED_LAYERS
            ):

                self.output_layer = tracker.ordered_modules[-1]
                self.output_expansion_instructions = load_instructions(
                    self.output_layer
                )
            else:
                warnings.warn(
                    "The model will not be able to adapt its output to new "
                    "classes since no supported output layer was found."
                )
                self.is_class_incremental = False

        if self.is_feature_incremental:
            if tracker.ordered_modules and isinstance(
                tracker.ordered_modules[0], SUPPORTED_LAYERS
            ):
                self.input_layer = tracker.ordered_modules[0]
                self.input_expansion_instructions = load_instructions(self.input_layer)
            else:
                warnings.warn(
                    "The model will not be able to adapt its input layer to "
                    "new features since no supported input layer was found."
                )
                self.is_feature_incremental = False


class RollingDeepEstimator(DeepEstimator):
    """
    Abstract base class that implements basic functionality of
    River-compatible PyTorch wrappers including a rolling window to allow the
    model to make predictions based on multiple previous examples.

    Parameters
    ----------
    module
        Torch Module that builds the autoencoder to be wrapped. The Module
        should accept parameter `n_features` so that the returned model's
        input shape can be determined based on the number of features in the
        initial training example.
    loss_fn
        Loss function to be used for training the wrapped model. Can be a loss
        function provided by `torch.nn.functional` or one of the following:
        'mse', 'l1', 'cross_entropy', 'binary_crossentropy',
        'smooth_l1', 'kl_div'.
    optimizer_fn
        Optimizer to be used for training the wrapped model.
        Can be an optimizer class provided by `torch.optim` or one of the
        following: "adam", "adam_w", "sgd", "rmsprop", "lbfgs".
    lr
        Learning rate of the optimizer.
    device
        Device to run the wrapped model on. Can be "cpu" or "cuda".
    seed
        Random seed to be used for training the wrapped model.
    window_size
        Size of the rolling window used for storing previous examples.
    append_predict
        Whether to append inputs passed for prediction to the rolling window.
    **kwargs
        Parameters to be passed to the `Module` or the `optimizer`.
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
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
            device=device,
            seed=seed,
            **kwargs,
        )

        self.window_size = window_size
        self.append_predict = append_predict
        self._x_window: Deque = collections.deque(maxlen=window_size)
        self._batch_i = 0


class DeepEstimatorInitialized(base.Estimator):
    """
    A DeepEstimator that allows passing an already initialized module.

    Parameters
    ----------
    module : torch.nn.Module
        A pre-initialized Torch Module (e.g., an instance of a neural network).
    loss_fn : Union[str, Callable]
        Loss function to be used for training.
    optimizer_fn : Union[str, Callable]
        Optimizer to be used for training.
    lr : float
        Learning rate.
    device : str
        Device for training (e.g., 'cpu' or 'cuda').
    seed : int
        Random seed for reproducibility.
    is_feature_incremental : bool
        Whether to allow feature expansion dynamically.
    **kwargs
        Additional parameters.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        is_feature_incremental: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.module = module
        self.lr = lr
        self.loss_func = get_loss_fn(loss_fn)
        self.loss_fn = loss_fn
        self.optimizer = get_optim_fn(optimizer_fn)(
            self.module.parameters(), lr=self.lr
        )
        self.optimizer_fn = optimizer_fn
        self.device = device
        self.seed = seed
        self.is_feature_incremental = is_feature_incremental
        self.kwargs = kwargs

        layers = list(self.module.children())
        self.input_layer = layers[0] if layers else None
        self.output_layer = layers[-1] if layers else None
        self.observed_features: SortedSet = SortedSet()
        self.module.to(self.device)
        torch.manual_seed(seed)

    def _update_observed_features(self, x):
        """Updates observed features dynamically if new ones appear."""
        prev_feature_count = len(self.observed_features)
        new_features = x.keys() if isinstance(x, dict) else x.columns
        self.observed_features.update({f: None for f in new_features})

        return len(self.observed_features) > prev_feature_count

    def _dict2tensor(self, x: dict):
        """Converts a dictionary to a tensor, handling missing features."""
        default_value = 0.0
        tensor_data = dict2tensor(
            x,
            self.observed_features,
            default_value=default_value,
            device=self.device,
            dtype=torch.float32,
        )
        return self._pad_tensor_if_needed(tensor_data, 1)

    def _df2tensor(self, X: pd.DataFrame):
        """Converts a DataFrame to a tensor, handling missing features."""
        default_value = 0.0
        tensor_data = df2tensor(
            X,
            self.observed_features,
            default_value=default_value,
            device=self.device,
            dtype=torch.float32,
        )
        return self._pad_tensor_if_needed(tensor_data, X.shape[0])

    def _pad_tensor_if_needed(self, tensor_data, x_len, default_value=0.0):
        """Pads the tensor if fewer features are available than required."""
        len_current_features = len(self.observed_features)
        if isinstance(self.input_layer, torch.nn.Linear):
            len_module_input = self.input_layer.in_features
            if len_current_features < len_module_input:
                padding = torch.full(
                    (x_len, len_module_input - len_current_features),
                    default_value,
                    device=self.device,
                    dtype=torch.float32,
                )
                # Ensure tensor_data has the correct batch size
                if tensor_data.shape[0] != x_len:
                    tensor_data = tensor_data.expand(x_len, -1)

                tensor_data = torch.cat([tensor_data, padding], dim=1)
        return tensor_data

    def _load_instructions(
        self, layer: torch.nn.Module
    ) -> Dict[str, Union[str, dict[str, List[dict[str, int]]]]]:
        """
        Dynamically infer expansion instructions for a given layer.

        Parameters
        ----------
        layer : nn.Module
            The layer to analyze.

        Returns
        -------
        Dict
            Instructions for expanding the layer's parameters.
        """
        instructions: Dict[str, Union[str, dict[str, List[dict[str, int]]]]] = {}

        if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
            instructions["in_features"] = "input_attribute"
            instructions["out_features"] = "output_attribute"

        if hasattr(layer, "weight"):
            instructions["weight"] = {
                "input": [{"axis": 1, "n_subparams": 1}],
                "output": [{"axis": 0, "n_subparams": 1}],
            }

        if hasattr(layer, "bias") and layer.bias is not None:
            instructions["bias"] = {"output": [{"axis": 0, "n_subparams": 1}]}

        return instructions

    def _expand_layer(
        self, layer: torch.nn.Module, target_size: int, output: bool = True
    ):
        """
        Expands a layer dynamically based on inferred attributes.

        Parameters
        ----------
        layer : nn.Module
            The layer to expand.
        target_size : int
            The new target size.
        output : bool, optional
            Whether to expand the output dimension (default: True).
        """
        instructions = self._load_instructions(layer)
        target_str = "output" if output else "input"

        for param_name, instruction in instructions.items():
            if instruction == f"{target_str}_attribute":
                setattr(layer, param_name, target_size)

            elif isinstance(instruction, dict):
                for axis_info in instruction[target_str]:
                    param = getattr(layer, param_name)
                    axis = axis_info["axis"]
                    dims_to_add = target_size - param.shape[axis]
                    n_subparams = axis_info["n_subparams"]

                    # Expand weights dynamically
                    param = self._expand_weights(param, axis, dims_to_add, n_subparams)
                    setattr(layer, param_name, param)

    def _expand_weights(
        self, param: torch.Tensor, axis: int, dims_to_add: int, n_subparams: int
    ):
        """
        Expands weight tensors dynamically along a given axis.

        Parameters
        ----------
        param : torch.Tensor
            The weight tensor to expand.
        axis : int
            The axis along which to expand.
        dims_to_add : int
            The number of new dimensions to add.
        n_subparams : int
            The number of sub-parameters to initialize.

        Returns
        -------
        torch.Tensor
            The expanded weight tensor.
        """
        if dims_to_add <= 0:
            return param  # No need to expand if target size is already met

        new_weights = (
            torch.randn(
                *(param.shape[:axis] + (dims_to_add,) + param.shape[axis + 1 :]),
                device=param.device,
                dtype=param.dtype,
            )
            * 0.01
        )  # Small random initialization

        return torch.cat([param, new_weights], dim=axis)

    def clone(
        self,
        new_params: dict[Any, Any] | None = None,
        include_attributes=False,
    ):
        """Clones the estimator.

        Parameters
        ----------
        new_params
            New parameters to be passed to the cloned estimator.
        include_attributes
            If True, the attributes of the estimator will be copied to the
            cloned estimator. This is useful when the estimator is a
            transformer and the attributes are the learned parameters.

        Returns
        -------
        DeepEstimator
            The cloned estimator.
        """
        new_params = new_params or {}
        new_params.update(self.kwargs)
        new_params.update(self._get_params())

        clone = self.__class__(**new_params)
        if include_attributes:
            clone.__dict__.update(self.__dict__)
        return clone
