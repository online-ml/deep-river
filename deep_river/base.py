import collections
import importlib
import pickle
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Union

import pandas as pd
import torch
from river import base
from sortedcontainers import SortedSet

from deep_river.utils import (
    deque2rolling_tensor,
    df2tensor,
    dict2tensor,
    float2tensor,
    get_loss_fn,
    get_optim_fn,
    labels2onehot,
)

try:
    from graphviz import Digraph
    from torchviz import make_dot
except ImportError as e:
    raise ValueError("You have to install graphviz to use the draw method") from e


class DeepEstimator(base.Estimator):
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
                candidates.extend(DeepEstimator._extract_candidate_layers(child))
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

    def draw(self) -> Digraph:
        """Draws the wrapped model."""
        first_parameter = next(self.module.parameters())
        input_shape = first_parameter.size()
        y_pred = self.module(torch.rand(input_shape))
        return make_dot(y_pred.mean(), params=dict(self.module.named_parameters()))

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

    @staticmethod
    def _expand_weights(
        param: torch.Tensor, axis: int, dims_to_add: int, n_subparams: int
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
            # Access observed_classes if it exists, otherwise use an empty SortedSet
            observed_classes = getattr(self, "observed_classes", SortedSet())
            y = labels2onehot(y, observed_classes, n_classes, self.device)

        self.module.train()
        loss = self.loss_func(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the model to a file.

        This method saves the complete state of the estimator including:
        - PyTorch model state (weights, biases)
        - Optimizer state
        - Configuration parameters
        - Metadata (observed classes, features, etc.)
        - Module information for reconstruction

        Parameters
        ----------
        filepath : Union[str, Path]
            Path where the model should be saved. Will be created if it doesn't exist.

        Examples
        --------
        >>> from deep_river.classification import Classifier
        >>> model = Classifier(module=SimpleNet(n_features=4), loss_fn='cross_entropy')
        >>> # ... train the model ...
        >>> model.save('my_model.pkl')
        """
        self._save_model(filepath)

    def _get_save_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for saving.
        Subclasses can override this method to add their specific configurations.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary with all parameters needed for reconstruction.
        """
        config: Dict[str, Any] = {
            "loss_fn": getattr(self, "loss_fn", "mse"),
            "optimizer_fn": getattr(self, "optimizer_fn", "sgd"),
            "lr": getattr(self, "lr", 1e-3),
            "device": getattr(self, "device", "cpu"),
            "seed": getattr(self, "seed", 42),
        }

        # Add DeepEstimatorInitialized specific configuration
        if hasattr(self, "is_feature_incremental"):
            config["is_feature_incremental"] = self.is_feature_incremental

        return config

    def _get_save_metadata(self) -> Dict[str, Any]:
        """
        Get the metadata dictionary for saving.
        Subclasses can override this method to add their specific metadata.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with runtime state information.
        """
        metadata: Dict[str, Any] = {}

        # Base metadata for DeepEstimatorInitialized
        if hasattr(self, "observed_classes"):
            observed_classes = getattr(self, "observed_classes")
            metadata["observed_classes"] = self._serialize_sorted_set(observed_classes)
        if hasattr(self, "observed_features"):
            metadata["observed_features"] = self._serialize_sorted_set(
                self.observed_features
            )
        if hasattr(self, "module_initialized"):
            metadata["module_initialized"] = getattr(self, "module_initialized", True)

        return metadata

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """
        Load a model from a file.

        This method reconstructs a complete estimator from a saved file,
        restoring all state including model weights, optimizer state, configuration,
        and metadata.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to the saved model file.

        Returns
        -------
        estimator
            A fully reconstructed estimator instance.

        Examples
        --------
        >>> from deep_river.classification import Classifier
        >>> model = Classifier.load('my_model.pkl')
        >>> # Model is ready to use for prediction or continued training
        """
        return cls._load_model(filepath)

    def _save_model(self, filepath: Union[str, Path]) -> None:
        """
        Internal method to save the model state to disk.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Determine estimator type and class
        estimator_type = type(self).__name__
        estimator_module = type(self).__module__
        estimator_class = f"{estimator_module}.{estimator_type}"

        # Prepare save data structure
        save_data: Dict[str, Any] = {
            "estimator_type": estimator_type,
            "estimator_class": estimator_class,
        }

        # Save model state
        if hasattr(self, "module") and self.module is not None:
            save_data["model_state_dict"] = self.module.state_dict()

            # Save optimizer state if available
            if hasattr(self, "optimizer") and self.optimizer is not None:
                save_data["optimizer_state_dict"] = self.optimizer.state_dict()

        # Save configuration - base configuration plus type-specific
        config = self._get_save_config()

        save_data["config"] = config

        # Save metadata - base metadata plus type-specific
        metadata = self._get_save_metadata()

        save_data["metadata"] = metadata

        # Save module information for reconstruction
        module_info: Dict[str, Any] = {}
        if hasattr(self, "module") and self.module is not None:
            # Save module class info
            module_info["module_class"] = (
                f"{type(self.module).__module__}.{type(self.module).__name__}"
            )
            # Save module kwargs if available
            if hasattr(self, "kwargs"):
                module_info["module_kwargs"] = self.kwargs
            else:
                module_info["module_kwargs"] = {}

        save_data["module_info"] = module_info

        # Save using pickle
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def _load_model(cls, filepath: Union[str, Path]):
        """
        Internal method to load the model state from disk.
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
            }  # Remove rolling-specific params for base class

            # Handle rolling estimator case
            if "window_size" in config:
                estimator_config.update(
                    {
                        "window_size": config["window_size"],
                        "append_predict": config.get("append_predict", False),
                    }
                )

            estimator = estimator_class(module=module_instance, **estimator_config)

        else:
            raise ValueError("Module information not found in save file")

        # Restore optimizer state
        if "optimizer_state_dict" in save_data and hasattr(estimator, "optimizer"):
            if estimator.optimizer is not None:
                estimator.optimizer.load_state_dict(save_data["optimizer_state_dict"])

        # Restore metadata
        if "metadata" in save_data:
            metadata = save_data["metadata"]

            if "observed_classes" in metadata:
                estimator.observed_classes = cls._deserialize_sorted_set(
                    metadata["observed_classes"]
                )
            if "observed_features" in metadata:
                estimator.observed_features = cls._deserialize_sorted_set(
                    metadata["observed_features"]
                )
            if "window_buffer" in metadata and hasattr(estimator, "_x_window"):
                # Restore window buffer for rolling estimators
                from collections import deque

                estimator._x_window = deque(
                    metadata["window_buffer"], maxlen=config.get("window_size")
                )

        return estimator

    @staticmethod
    def _serialize_sorted_set(sorted_set: SortedSet) -> list:
        """Convert SortedSet to list for serialization."""
        return list(sorted_set) if sorted_set is not None else []

    @staticmethod
    def _deserialize_sorted_set(data: list) -> SortedSet:
        """Convert list back to SortedSet."""
        return SortedSet(data) if data is not None else SortedSet()


class RollingDeepEstimator(DeepEstimator):
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

    def _get_save_config(self) -> Dict[str, Any]:
        """
        Get the configuration dictionary for saving.
        Extends the base configuration with rolling-specific parameters.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary including rolling window parameters.
        """
        config = super()._get_save_config()

        # Add rolling-specific configuration
        config["window_size"] = self.window_size
        config["append_predict"] = self.append_predict

        return config

    def _get_save_metadata(self) -> Dict[str, Any]:
        """
        Get the metadata dictionary for saving.
        Extends the base metadata with rolling window state.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary including rolling window buffer state.
        """
        metadata = super()._get_save_metadata()

        # Add rolling-specific metadata
        if hasattr(self, "_x_window"):
            metadata["has_window_buffer"] = True
            if self._x_window is not None and len(self._x_window) > 0:
                metadata["window_buffer"] = list(self._x_window)

        return metadata
