import collections
import importlib
import inspect
import pickle
import copy  # added for robust module cloning fallback
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

# Entfernt hartes Erzwingen von graphviz beim Import – optional in draw()
try:
    from graphviz import Digraph  # type: ignore
    from torchviz import make_dot  # type: ignore
except Exception:  # pragma: no cover
    Digraph = None
    make_dot = None


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
        # Wähle ersten Kandidaten mit Parametern als input_layer
        for cand in candidates:
            if any(p.requires_grad for p in cand.parameters()):
                self.input_layer = cand
                break
        else:
            self.input_layer = candidates[0] if candidates else None
        # Wähle letzten parametrischen Layer als output_layer (z.B. nicht ReLU)
        self.output_layer = None
        for cand in reversed(candidates):
            if any(p.requires_grad for p in cand.parameters()):
                self.output_layer = cand
                break
        if self.output_layer is None and candidates:
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
        if Digraph is None or make_dot is None:
            raise ImportError("graphviz and torchviz must be installed to draw the model.")

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
        """Dynamically determines the output feature size of the last (parametrischen) layer."""
        if not hasattr(self, "output_layer") or self.output_layer is None:
            raise ValueError("No output layer found in the model.")
        layer = self.output_layer
        if hasattr(layer, "out_features"):
            return layer.out_features
        elif hasattr(layer, "output_size"):
            return layer.output_size
        elif hasattr(layer, "out_channels"):
            return layer.out_channels
        elif isinstance(layer, torch.nn.LSTM):
            return layer.hidden_size
        elif hasattr(layer, "weight") and getattr(layer, 'weight') is not None:
            return layer.weight.shape[0]
        else:
            # Fallback: Suche rückwärts nach einem Layer mit Gewicht
            for cand in reversed(list(self.module.modules())):
                if hasattr(cand, 'weight') and getattr(cand, 'weight') is not None:
                    return cand.weight.shape[0]
            raise ValueError(f"Cannot determine output size for layer type {type(layer)}")

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
        return instructions

    def _expand_layer(
        self, layer: torch.nn.Module, target_size: int, output: bool = True
    ):
        instructions = self._load_instructions(layer)
        target_str = "output" if output else "input"

        layer_modified = False
        for param_name, instruction in instructions.items():
            if instruction == f"{target_str}_attribute":
                if getattr(layer, param_name) != target_size:
                    setattr(layer, param_name, target_size)
                    layer_modified = True
            elif isinstance(instruction, dict):
                if target_str not in instruction:
                    continue
                for axis_info in instruction[target_str]:
                    param = getattr(layer, param_name)
                    axis = axis_info["axis"]
                    dims_to_add = target_size - param.shape[axis]
                    n_subparams = axis_info["n_subparams"]
                    if dims_to_add > 0:
                        param = self._expand_weights(param, axis, dims_to_add, n_subparams)
                        if not isinstance(param, torch.nn.Parameter):
                            param = torch.nn.Parameter(param)
                        setattr(layer, param_name, param)
                        layer_modified = True
        # Falls etwas verändert wurde: Optimizer neu aufsetzen, damit neue Parameter enthalten sind.
        if layer_modified:
            self._rebuild_optimizer()

    def _rebuild_optimizer(self):
        """Reinitialisiert den Optimizer mit allen aktuellen Modellparametern.
        Notwendig nach dynamischen Layer-Erweiterungen, weil neue Parameter sonst nicht trainiert würden."""
        optim_fn = get_optim_fn(self.optimizer_fn)
        self.optimizer = optim_fn(self.module.parameters(), lr=self.lr)

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
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "estimator_class": f"{type(self).__module__}.{type(self).__name__}",
            "init_params": self._get_all_init_params(),
            "model_state_dict": getattr(self.module, "state_dict", lambda: {})(),
            "optimizer_state_dict": getattr(self.optimizer, "state_dict", lambda: {})(),
            "runtime_state": self._get_runtime_state(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]):
        """Load model from file."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        # Reconstruct estimator with all init params
        estimator_cls = cls._import_from_path(state["estimator_class"])
        init_params = state["init_params"]

        # Rebuild module if needed
        if "module" in init_params and isinstance(init_params["module"], dict):
            module_info = init_params.pop("module")
            module_cls = cls._import_from_path(module_info["class"])
            module = module_cls(**cls._filter_kwargs(module_cls.__init__, module_info["kwargs"]))
            if state.get("model_state_dict"):
                module.load_state_dict(state["model_state_dict"])
            init_params["module"] = module

        estimator = estimator_cls(**cls._filter_kwargs(estimator_cls.__init__, init_params))

        # Restore optimizer and runtime state
        if state.get("optimizer_state_dict") and hasattr(estimator, "optimizer"):
            try: estimator.optimizer.load_state_dict(state["optimizer_state_dict"])
            except: pass

        estimator._restore_runtime_state(state.get("runtime_state", {}))
        return estimator

    def clone(self, new_params=None, include_attributes: bool = False, copy_weights: bool = False):
        """Clone estimator with optional parameter overrides."""
        new_params = new_params or {}
        copy_weights = new_params.pop("copy_weights", copy_weights)

        # Get all init parameters and apply overrides
        params = {**self._get_all_init_params(), **new_params}

        # Handle module cloning
        if "module" not in new_params:
            params["module"] = self._rebuild_module()

        new_est = self.__class__(**self._filter_kwargs(self.__class__.__init__, params))

        if copy_weights and hasattr(self.module, "state_dict"):
            new_est.module.load_state_dict(self.module.state_dict())

        if include_attributes:
            new_est._restore_runtime_state(self._get_runtime_state())

        return new_est

    def _get_all_init_params(self) -> Dict[str, Any]:
        """Get all __init__ parameters from current instance."""
        sig = inspect.signature(self.__class__.__init__)
        params = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            elif name == "module":
                # Store module info for reconstruction
                params["module"] = {
                    "class": f"{type(self.module).__module__}.{type(self.module).__name__}",
                    "kwargs": {**getattr(self, "kwargs", {}), **self._infer_module_params()}
                }
            elif hasattr(self, name):
                params[name] = getattr(self, name)
            elif name in getattr(self, "kwargs", {}):
                params[name] = self.kwargs[name]

        return params

    def _get_runtime_state(self) -> Dict[str, Any]:
        """Get runtime state (observed features, classes, window buffer, etc.)."""
        state = {}

        for attr in ["observed_features", "observed_classes"]:
            if hasattr(self, attr):
                state[attr] = list(getattr(self, attr, []))

        if hasattr(self, "_x_window") and self._x_window:
            state["window_buffer"] = list(self._x_window)

        return state

    def _restore_runtime_state(self, state: Dict[str, Any]) -> None:
        """Restore runtime state from saved data."""
        for attr, data in state.items():
            if attr.endswith(("_features", "_classes")):
                setattr(self, attr, SortedSet(data) if data else SortedSet())
            elif attr == "window_buffer" and hasattr(self, "_x_window"):
                from collections import deque
                self._x_window = deque(data, maxlen=getattr(self, "window_size", 10))

    # -------------------- Simplified helper utilities --------------------
    def _infer_module_params(self) -> dict:
        """Infer module init parameters from attributes and state."""
        params = {}
        if hasattr(self.module, "__dict__"):
            # Try to get params from module attributes
            sig = inspect.signature(self.module.__class__.__init__)
            for name in sig.parameters:
                if name != "self" and hasattr(self.module, name):
                    params[name] = getattr(self.module, name)

        # Infer n_features from first weight if needed
        if "n_features" not in params and hasattr(self.module, "named_parameters"):
            for name, param in self.module.named_parameters():
                if "weight" in name and param.dim() == 2:
                    params["n_features"] = param.shape[1]
                    break

        return params

    def _rebuild_module(self):
        """Rebuild module with same parameters but fresh weights."""
        params = self._infer_module_params()
        try:
            return self.module.__class__(**self._filter_kwargs(self.module.__class__.__init__, params))
        except:
            # Fallback to deepcopy with reset
            mod_copy = copy.deepcopy(self.module)
            for m in mod_copy.modules():
                if hasattr(m, "reset_parameters"):
                    try:
                        m.reset_parameters()
                    except:
                        pass
            return mod_copy

    @staticmethod
    def _import_from_path(path: str):
        """Import object from fully qualified path."""
        module_path, name = path.rsplit('.', 1)
        return getattr(importlib.import_module(module_path), name)

    @staticmethod
    def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs to only include parameters accepted by callable."""
        try:
            sig = inspect.signature(callable_obj)
            allowed = {p.name for p in sig.parameters.values()
                      if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
            return {k: v for k, v in kwargs.items() if k in allowed}
        except (ValueError, TypeError):
            return kwargs

    @staticmethod
    def _serialize_sorted_set(sorted_set: SortedSet) -> list:
        """Convert SortedSet to list for serialization."""
        return list(sorted_set) if sorted_set else []

    @staticmethod
    def _deserialize_sorted_set(data: list) -> SortedSet:
        """Convert list back to SortedSet."""
        return SortedSet(data) if data else SortedSet()


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
