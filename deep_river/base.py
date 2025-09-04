import collections
import importlib
import inspect
import pickle
import warnings
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Type, Union, cast

import pandas as pd
import torch
from river import base
from sortedcontainers import SortedSet
from torch.utils.hooks import RemovableHandle

from deep_river.utils import (
    deque2rolling_tensor,
    df2tensor,
    dict2tensor,
    float2tensor,
    get_loss_fn,
    get_optim_fn,
    labels2onehot,
)
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
        self.observed_features: SortedSet = SortedSet([])
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

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        filepath : Union[str, pathlib.Path]
            Path where the model should be saved.

        Examples
        --------
        >>> from deep_river.classification import Classifier
        >>> model = Classifier(module=MyModule, loss_fn='cross_entropy')
        >>> # ... train the model ...
        >>> model.save('my_model.pkl')
        """
        save_model(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """
        Load a model from a file.

        Parameters
        ----------
        filepath : Union[str, pathlib.Path]
            Path to the saved model file.

        Returns
        -------
        estimator
            The loaded model instance.

        Examples
        --------
        >>> from deep_river.classification import Classifier
        >>> model = Classifier.load('my_model.pkl')
        """
        return load_model(filepath)


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

    # Rolling estimators inherit save/load methods from DeepEstimator


class DeepEstimatorInitialized(base.Estimator):
    """
    Enhances PyTorch modules with dynamic adaptability to evolving features.

    The class extends the functionality of a base estimator by dynamically
    updating and expanding neural network layers to handle incremental
    changes in feature space. It supports feature set discovery, input size
    adjustments, weight expansion, and varied learning procedures. This makes
    it suitable for evolving input spaces while maintaining neural network
    integrity.

    Attributes
    ----------
    module : torch.nn.Module
        The PyTorch model that serves as the backbone of this class's functionality.
    lr : float
        Learning rate for model optimization.
    loss_fn : Union[str, Callable]
        The loss function used for computing training error.
    loss_func : Callable
        The compiled loss function produced via `get_loss_fn`.
    optimizer : torch.optim.Optimizer
        The compiled optimizer used for updating model weights.
    optimizer_fn : Union[str, Callable]
        The optimizer function or class used for training.
    device : str
        The computational device (e.g., "cpu", "cuda") used for training.
    seed : int
        The random seed for ensuring reproducible operations.
    is_feature_incremental : bool
        Indicates whether the model should automatically expand based on new features.
    kwargs : dict
        Additional arguments passed to the model and utilities.
    input_layer : torch.nn.Module
        The input layer of the PyTorch model, determined dynamically.
    output_layer : torch.nn.Module
        The output layer of the PyTorch model, determined dynamically.
    observed_features : SortedSet
        Tracks all observed input features dynamically, allowing for feature incrementation.
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

        candidates = self._extract_candidate_layers(self.module)
        self.input_layer = candidates[0]
        self.output_layer = candidates[-1]

        # Set the expected input length based on the extracted input layer.
        self.module_input_len = self._get_input_size() if self.input_layer else None
        self.observed_features: SortedSet = SortedSet()
        self.module.to(self.device)
        torch.manual_seed(seed)

    @staticmethod
    def _extract_candidate_layers(module: torch.nn.Module) -> list[torch.nn.Module]:
        """
        Recursively collects candidate layers for adaptation.
        Non-parametric layers such as Softmax or LogSoftmax are filtered out.
        """
        candidates = []
        for child in module.children():
            if list(child.children()):
                candidates.extend(
                    DeepEstimatorInitialized._extract_candidate_layers(child)
                )
            else:
                if not isinstance(child, (torch.nn.Softmax, torch.nn.LogSoftmax)):
                    candidates.append(child)
        return candidates

    def _update_observed_features(self, x):
        """Updates observed features dynamically if new ones appear."""
        prev_feature_count = len(self.observed_features)
        new_features = x.keys() if isinstance(x, dict) else x.columns
        self.observed_features.update(new_features)
        if (
            self.is_feature_incremental
            and self.input_layer
            and self._get_input_size() < len(self.observed_features)
        ):
            self._expand_layer(
                self.input_layer, target_size=len(self.observed_features), output=False
            )
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

    def _get_input_size(self):
        """Dynamically determines the expected input feature size of a PyTorch layer."""
        if not hasattr(self, "input_layer") or self.output_layer is None:
            raise ValueError("No input layer found in the model.")

        if hasattr(self.input_layer, "in_features"):
            return self.input_layer.in_features
        elif hasattr(self.input_layer, "input_size"):
            return self.input_layer.input_size
        elif hasattr(self.input_layer, "in_channels"):
            return self.input_layer.in_channels
        elif (
            hasattr(self.input_layer, "weight") and self.input_layer.weight is not None
        ):
            return self.input_layer.weight.shape[1]
        else:
            raise ValueError(
                f"Cannot determine input size for layer type {type(self.input_layer)}"
            )

    def _get_output_size(self):
        """Dynamically determines the output feature size of the last layer in the module."""
        if not hasattr(self, "output_layer") or self.output_layer is None:
            raise ValueError("No output layer found in the model.")

        if hasattr(
            self.output_layer, "out_features"
        ):  # Fully Connected Layers (Linear)
            return self.output_layer.out_features
        elif hasattr(self.output_layer, "output_size"):  # Custom Layers
            return self.output_layer.output_size
        elif hasattr(self.output_layer, "out_channels"):  # Convolutional Layers
            return self.output_layer.out_channels
        elif isinstance(self.output_layer, torch.nn.LSTM):  # LSTM Handling
            return (
                self.output_layer.hidden_size
            )  # LSTMs return (hidden_state, cell_state)
        elif (
            hasattr(self.output_layer, "weight")
            and self.output_layer.weight is not None
        ):
            return self.output_layer.weight.shape[0]  # General Weight-Based Guess
        else:
            raise ValueError(
                f"Cannot determine output size for layer type {type(self.input_layer)}"
            )

    def _pad_tensor_if_needed(self, tensor_data, x_len, default_value=0.0):
        """

        Parameters
        ----------
        tensor_data
        x_len
        default_value

        Returns
        -------

        """
        len_current_features = len(self.observed_features)
        if len_current_features < self._get_input_size():
            padding_shape = None
            if isinstance(self.input_layer, torch.nn.Linear):
                padding_shape = (x_len, self._get_input_size() - len_current_features)
            elif isinstance(
                self.input_layer, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)
            ):
                if tensor_data.dim() == 3:
                    seq_len, batch_size, _ = tensor_data.shape
                    padding_shape = (
                        seq_len,
                        batch_size,
                        self._get_input_size() - len_current_features,
                    )
                elif tensor_data.dim() == 2:
                    batch_size, _ = tensor_data.shape
                    padding_shape = (
                        batch_size,
                        self._get_input_size() - len_current_features,
                    )
            if padding_shape:
                padding = torch.full(
                    padding_shape,
                    default_value,
                    device=self.device,
                    dtype=torch.float32,
                )
                tensor_data = torch.cat([tensor_data, padding], dim=-1)
        return tensor_data

    def _load_instructions(self, layer: torch.nn.Module) -> dict[str, Any]:
        instructions: dict[str, Any] = {}
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
        print("Layer:", layer, "\nInstructions:", instructions)  # Debug print
        return instructions

    def _expand_layer(
        self, layer: torch.nn.Module, target_size: int, output: bool = True
    ):
        instructions = self._load_instructions(layer)
        target_str = "output" if output else "input"

        for param_name, instruction in instructions.items():
            if instruction == f"{target_str}_attribute":
                setattr(layer, param_name, target_size)
            elif isinstance(instruction, dict):
                # Ensure the target_str key exists in instruction before accessing it
                if target_str not in instruction:
                    continue  # Skip expansion if no instructions exist for input/output

                for axis_info in instruction[target_str]:
                    param = getattr(layer, param_name)
                    axis = axis_info["axis"]
                    dims_to_add = target_size - param.shape[axis]
                    n_subparams = axis_info["n_subparams"]

                    param = self._expand_weights(param, axis, dims_to_add, n_subparams)

                    if not isinstance(param, torch.nn.Parameter):
                        param = torch.nn.Parameter(param)

                    setattr(layer, param_name, param)

    def _expand_weights(
        self, param: torch.Tensor, axis: int, dims_to_add: int, n_subparams: int
    ):
        """
        Expands weight tensors dynamically along a given axis.
        """
        if dims_to_add <= 0:
            return param

        # Create new weights to be added
        new_weights = (
            torch.randn(
                *(param.shape[:axis] + (dims_to_add,) + param.shape[axis + 1 :]),
                device=param.device,
                dtype=param.dtype,
            )
            * 0.01  # Small initialization
        )

        # Concatenate the new weights along the given axis
        expanded_param = torch.cat([param, new_weights], dim=axis)

        # Ensure the result is a torch.nn.Parameter so it's registered as a model parameter
        return torch.nn.Parameter(expanded_param)

    def _learn(self, x: torch.Tensor, y: Optional[Any] = None):
        """
        Performs a single training step.

        Supports classification, regression, and autoencoding:
        - Autoencoders: y is None, so x is used as the target.
        - Regression: y is a continuous value, converted to a tensor.
        - Classification: y is converted to one-hot encoding.
        """

        y_pred = self.module(x)

        # Autoencoder case: No explicit y, so use x as target
        if y is None:
            y = x

            # Regression case: Convert y to tensor and move to device
        elif not hasattr(self, "observed_classes"):
            if not isinstance(y, torch.Tensor):
                y = float2tensor(y, self.device)

        # Classification case: Convert y to one-hot encoding
        else:
            n_classes = y_pred.shape[-1]
            try:
                # Try to access observed_classes if it exists (for classification models)
                observed_classes = self.observed_classes  # type: ignore
            except AttributeError:
                observed_classes = SortedSet()
            y = labels2onehot(y, observed_classes, n_classes, self.device)

        self.module.train()
        loss = self.loss_func(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path where the model should be saved.
        """
        save_model(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """
        Load a model from a file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the saved model file.

        Returns
        -------
        estimator
            The loaded model instance.
        """
        return load_model(filepath)


class RollingDeepEstimatorInitialized(DeepEstimatorInitialized):
    """
    RollingDeepEstimatorInitialized class for rolling window-based deep learning
    model estimation.

    This class extends the functionality of the DeepEstimatorInitialized class to
    support training and prediction using a rolling window. It maintains a fixed-size
    deque to store a rolling window of input data. It can optionally append predictions
    to the input window to facilitate iterative prediction workflows. This class is
    designed for advanced users who need rolling window functionality in their deep
    learning estimation pipelines.

    Attributes
    ----------
    window_size : int
        The size of the rolling window used for training and prediction.
    append_predict : bool
        Flag to indicate whether to append predictions into the rolling window.
    _x_window : Deque
        A fixed-size deque object, which stores the most recent input window data.
    _batch_i : int
        The internal counter for batch index tracking during training or prediction.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        window_size: int = 10,
        append_predict: bool = False,
        **kwargs,
    ):
        self.window_size = window_size
        self.append_predict = append_predict
        self._x_window: Deque = collections.deque(maxlen=window_size)
        self._batch_i = 0
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            **kwargs,
        )

    def _deque2rolling_tensor(self, x_win: Deque):
        tensor_data = deque2rolling_tensor(x_win, device=self.device)
        return self._pad_tensor_if_needed(tensor_data, len(x_win))


# ============================================================================
# Model Persistence Utilities
# ============================================================================


def _serialize_sorted_set(sorted_set: SortedSet) -> list:
    """Convert SortedSet to list for serialization."""
    return list(sorted_set) if sorted_set is not None else []


def _deserialize_sorted_set(data: list) -> SortedSet:
    """Convert list back to SortedSet."""
    return SortedSet(data) if data is not None else SortedSet()


def save_model(estimator, filepath: Union[str, Path]) -> None:
    """
    Save a deep-river model to disk.

    This function saves the complete state of a deep-river estimator including:
    - PyTorch model state (weights, biases)
    - Optimizer state
    - Configuration parameters
    - Metadata (observed classes, features, etc.)
    - Module information for reconstruction

    Parameters
    ----------
    estimator
        A deep-river estimator instance (DeepEstimator, DeepEstimatorInitialized,
        RollingDeepEstimator, or their subclasses).
    filepath : Union[str, Path]
        Path where the model should be saved. Will be created if it doesn't exist.

    Examples
    --------
    >>> from deep_river.classification import Classifier
    >>> from deep_river.base import save_model, load_model
    >>> import torch

    >>> class SimpleNet(torch.nn.Module):
    ...     def __init__(self, n_features):
    ...         super().__init__()
    ...         self.linear = torch.nn.Linear(n_features, 2)
    ...     def forward(self, x):
    ...         return self.linear(x)

    >>> model = Classifier(module=SimpleNet, loss_fn='cross_entropy')
    >>> # ... train the model ...
    >>> save_model(model, 'my_model.pkl')
    >>> loaded_model = load_model('my_model.pkl')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine estimator type and class
    estimator_type = type(estimator).__name__
    estimator_module = type(estimator).__module__
    estimator_class = f"{estimator_module}.{estimator_type}"

    # Prepare save data structure
    save_data: Dict[str, Any] = {
        "estimator_type": estimator_type,
        "estimator_class": estimator_class,
        "deep_river_version": "0.3.0",  # Current version
    }

    # Save model state if module is initialized
    if hasattr(estimator, "module") and estimator.module is not None:
        save_data["model_state_dict"] = estimator.module.state_dict()

        # Save optimizer state if available
        if hasattr(estimator, "optimizer") and estimator.optimizer is not None:
            save_data["optimizer_state_dict"] = estimator.optimizer.state_dict()

    # Save configuration
    config: Dict[str, Any] = {
        "loss_fn": estimator.loss_fn,
        "optimizer_fn": estimator.optimizer_fn,
        "lr": estimator.lr,
        "device": estimator.device,
        "seed": estimator.seed,
    }

    # Add type-specific configuration
    if hasattr(estimator, "is_feature_incremental"):
        config["is_feature_incremental"] = estimator.is_feature_incremental
    if hasattr(estimator, "is_class_incremental"):
        config["is_class_incremental"] = estimator.is_class_incremental
    if hasattr(estimator, "output_is_logit"):
        config["output_is_logit"] = estimator.output_is_logit
    if hasattr(estimator, "window_size"):
        config["window_size"] = estimator.window_size
    if hasattr(estimator, "append_predict"):
        config["append_predict"] = estimator.append_predict

    save_data["config"] = config

    # Save metadata
    metadata: Dict[str, Any] = {}

    if hasattr(estimator, "observed_classes"):
        metadata["observed_classes"] = _serialize_sorted_set(estimator.observed_classes)
    if hasattr(estimator, "observed_features"):
        metadata["observed_features"] = _serialize_sorted_set(
            estimator.observed_features
        )
    if hasattr(estimator, "module_initialized"):
        metadata["module_initialized"] = estimator.module_initialized
    if hasattr(estimator, "window_buffer"):
        # For rolling estimators, save window buffer state
        metadata["has_window_buffer"] = hasattr(estimator, "window_buffer")
        if hasattr(estimator, "window_buffer") and len(estimator.window_buffer) > 0:
            metadata["window_buffer"] = list(estimator.window_buffer)

    save_data["metadata"] = metadata

    # Save module information for reconstruction
    module_info: Dict[str, Any] = {}
    if hasattr(estimator, "module_cls"):
        # For DeepEstimator - save module class info
        module_info["module_class"] = (
            f"{estimator.module_cls.__module__}.{estimator.module_cls.__name__}"
        )
        module_info["module_kwargs"] = estimator.kwargs
    elif hasattr(estimator, "module") and estimator.module is not None:
        # For DeepEstimatorInitialized - save module class info
        module_info["module_class"] = (
            f"{type(estimator.module).__module__}.{type(estimator.module).__name__}"
        )
        # Try to extract constructor arguments if possible
        if hasattr(estimator, "kwargs"):
            module_info["module_kwargs"] = estimator.kwargs
        else:
            module_info["module_kwargs"] = {}

    save_data["module_info"] = module_info

    # Save using pickle
    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)


def load_model(filepath: Union[str, Path]):
    """
    Load a deep-river model from disk.

    This function reconstructs a complete deep-river estimator from a saved file,
    restoring all state including model weights, optimizer state, configuration,
    and metadata.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the saved model file.

    Returns
    -------
    estimator
        A fully reconstructed deep-river estimator instance.

    Examples
    --------
    >>> from deep_river.base import load_model
    >>> model = load_model('my_model.pkl')
    >>> # Model is ready to use for prediction or continued training
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Load save data
    with open(filepath, "rb") as f:
        save_data = pickle.load(f)

    # Validate save data format
    required_keys = ["estimator_type", "estimator_class", "config"]
    for key in required_keys:
        if key not in save_data:
            raise ValueError(f"Invalid save file format: missing '{key}' key")

    # Check version compatibility (for future use)
    if "deep_river_version" in save_data:
        saved_version = save_data["deep_river_version"]
        if saved_version != "0.3.0":
            warnings.warn(
                f"Model was saved with deep-river version {saved_version}, "
                f"current version is 0.3.0. Compatibility is not guaranteed."
            )

    # Import the estimator class
    estimator_class_path = save_data["estimator_class"]
    module_path, class_name = estimator_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    estimator_class = getattr(module, class_name)

    # Reconstruct the estimator
    config = save_data["config"]
    module_info = save_data.get("module_info", {})

    if "module_class" in module_info:
        # Import the module class
        module_class_path = module_info["module_class"]
        module_module_path, module_class_name = module_class_path.rsplit(".", 1)
        module_module = importlib.import_module(module_module_path)
        module_cls = getattr(module_module, module_class_name)

        # Create estimator with module class (for DeepEstimator)
        if "Initialized" not in save_data["estimator_type"]:
            # For non-initialized estimators
            estimator = estimator_class(
                module=module_cls, **config, **module_info.get("module_kwargs", {})
            )
        else:
            # For initialized estimators, we need to create the module first
            # We'll handle this after loading the state dict
            estimator = None
    else:
        raise ValueError("Module information not found in save file")

    # Restore model state
    if "model_state_dict" in save_data and estimator is not None:
        # For non-initialized estimators, we need to create the module and set it
        if hasattr(estimator, "module") and estimator.module is None:
            # Create the module instance first
            module_kwargs = module_info.get("module_kwargs", {})

            # Try to infer n_features from model state if available
            if "model_state_dict" in save_data:
                model_state = save_data["model_state_dict"]
                # Look for the first linear layer to get n_features
                for param_name, param in model_state.items():
                    if "weight" in param_name and param.dim() == 2:
                        n_features = param.shape[1]
                        module_kwargs["n_features"] = n_features
                        break

            # Create and set the module
            module_instance = module_cls(**module_kwargs)
            estimator.module = module_instance
            estimator.module_initialized = (
                True if hasattr(estimator, "module_initialized") else None
            )

            # Load the state
            estimator.module.load_state_dict(save_data["model_state_dict"])

        elif hasattr(estimator, "module") and estimator.module is not None:
            # For already initialized modules
            estimator.module.load_state_dict(save_data["model_state_dict"])
        else:
            # For DeepEstimator, we need to initialize the module first
            # This will happen during the first learn_one call
            pass

    # Handle DeepEstimatorInitialized case
    if estimator is None and "Initialized" in save_data["estimator_type"]:
        # Create the module instance first
        module_kwargs = module_info.get("module_kwargs", {})

        # Try to infer n_features from model state if available
        if "model_state_dict" in save_data:
            model_state = save_data["model_state_dict"]
            # Look for the first linear layer to get n_features
            for param_name, param in model_state.items():
                if "weight" in param_name and param.dim() == 2:
                    n_features = param.shape[1]
                    module_kwargs["n_features"] = n_features
                    break

        # Create module instance
        module_instance = module_cls(**module_kwargs)

        # Load state dict into module
        if "model_state_dict" in save_data:
            module_instance.load_state_dict(save_data["model_state_dict"])

        # Create estimator with initialized module
        estimator_config = {
            k: v
            for k, v in config.items()
            if k not in ["window_size", "append_predict"]
        }  # Remove rolling-specific params
        estimator = estimator_class(module=module_instance, **estimator_config)

        # For initialized estimators, make sure the module is actually set
        if hasattr(estimator, "module") and module_instance is not None:
            estimator.module = module_instance
            # Mark as initialized
            if hasattr(estimator, "module_initialized"):
                estimator.module_initialized = True

            # Restore observed features if available
            if "observed_features" in save_data:
                estimator.observed_features = save_data["observed_features"]

            # Restore other saved attributes
            for attr_name, attr_value in save_data.get(
                "estimator_attributes", {}
            ).items():
                if hasattr(estimator, attr_name):
                    setattr(estimator, attr_name, attr_value)

    # Restore optimizer state
    if "optimizer_state_dict" in save_data and hasattr(estimator, "optimizer"):
        if estimator is not None and estimator.optimizer is not None:
            estimator.optimizer.load_state_dict(save_data["optimizer_state_dict"])

    # Restore metadata
    if "metadata" in save_data and estimator is not None:
        metadata = save_data["metadata"]

        if "observed_classes" in metadata:
            estimator.observed_classes = _deserialize_sorted_set(
                metadata["observed_classes"]
            )
        if "observed_features" in metadata:
            estimator.observed_features = _deserialize_sorted_set(
                metadata["observed_features"]
            )
        if "module_initialized" in metadata:
            estimator.module_initialized = metadata["module_initialized"]
        if "window_buffer" in metadata and hasattr(estimator, "window_buffer"):
            # Restore window buffer for rolling estimators
            from collections import deque

            estimator.window_buffer = deque(
                metadata["window_buffer"], maxlen=config.get("window_size")
            )

    return estimator


def get_model_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a saved model without fully loading it.

    This function reads the metadata from a saved model file and returns
    information about the model type, configuration, and training state.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the saved model file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model information including type, configuration,
        and metadata.

    Examples
    --------
    >>> from deep_river.base import get_model_info
    >>> info = get_model_info('my_model.pkl')
    >>> print(f"Model type: {info['estimator_type']}")
    >>> print(f"Device: {info['config']['device']}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with open(filepath, "rb") as f:
        save_data = pickle.load(f)

    # Extract key information
    info = {
        "estimator_type": save_data.get("estimator_type"),
        "estimator_class": save_data.get("estimator_class"),
        "deep_river_version": save_data.get("deep_river_version"),
        "config": save_data.get("config", {}),
        "metadata": save_data.get("metadata", {}),
        "has_trained_model": "model_state_dict" in save_data,
        "has_optimizer_state": "optimizer_state_dict" in save_data,
    }

    return info
