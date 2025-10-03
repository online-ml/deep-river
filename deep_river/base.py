import collections
import copy  # fallback for robust module cloning
import importlib
import inspect
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


class DeepEstimator(base.Estimator):
    """Incremental wrapper around a PyTorch module with dynamic feature adaptation.

    This class augments a regular ``torch.nn.Module`` with utilities that make it
    compatible with the `river` incremental learning API. Beyond standard online
    optimisation it optionally supports *feature-incremental* learning: whenever
    previously unseen input feature names appear, the first trainable layer (the
    *input layer*) can be expanded on‑the‑fly so that the model seamlessly accepts
    the enlarged feature space without re‑initialisation.

    The class also provides a persistence protocol (``save``/``load``/``clone``)
    that captures both the module weights and the runtime state (observed feature
    names, rolling buffers, etc.), allowing exact round‑trips across Python
    sessions. Optimisers are transparently rebuilt after structural changes so any
    newly created parameters participate in subsequent optimisation steps.

    Typical workflow
    ----------------
    1. Instantiate with a vanilla PyTorch module (e.g. an ``nn.Sequential`` or a
       custom subclass).
    2. Feed samples via higher level task specific subclasses (e.g. classifier)
       that call ``_learn`` internally.
    3. (Optional) Enable ``is_feature_incremental=True`` for dynamic input growth.
    4. Persist with ``save`` and later restore with ``load``.

    Example
    -------
    >>> import torch
    >>> from torch import nn
    >>> from deep_river.base import DeepEstimator
    >>> class TinyNet(nn.Module):
    ...     def __init__(self, n_features=3):
    ...         super().__init__()
    ...         self.fc = nn.Linear(n_features, 2)
    ...     def forward(self, x):
    ...         return self.fc(x)
    >>> est = DeepEstimator(
    ...     module=TinyNet(3),
    ...     loss_fn='mse',
    ...     optimizer_fn='sgd',
    ...     is_feature_incremental=True,
    ... )
    >>> est._update_observed_features({'a': 1.0, 'b': 2.0, 'c': 3.0})  # internal bookkeeping
    True

    Notes
    -----
    - The class itself is task‑agnostic. Task specific behaviour (e.g. converting
      labels to one‑hot encodings) lives in subclasses such as ``Classifier`` or
      ``Regressor``.
    - Only the *first* and *last* trainable leaf modules are treated as input and
      output layers. Non‑parametric layers (e.g. ``ReLU``) are skipped.

    Parameters
    ----------
    module : torch.nn.Module
        The PyTorch model whose parameters are to be updated incrementally.
    loss_fn : str | Callable, default='mse'
        Loss identifier or callable passed to :func:`get_loss_fn`.
    optimizer_fn : str | Callable, default='sgd'
        Optimiser identifier or optimiser class / factory.
    lr : float, default=1e-3
        Learning rate.
    device : str, default='cpu'
        Device on which the module is run.
    seed : int, default=42
        Random seed (sets ``torch.manual_seed``).
    is_feature_incremental : bool, default=False
        If True, expands the input layer when new feature names are encountered.
    gradient_clip_value : float | None, default=None
        If provided, gradient norm is clipped to this value each optimisation step.
    **kwargs : dict
        Additional custom arguments retained for reconstruction on ``clone`` / ``load``.

    Attributes
    ----------
    module : torch.nn.Module
        The wrapped PyTorch module.
    loss_func : Callable
        Resolved loss function callable.
    optimizer : torch.optim.Optimizer
        Optimiser instance (rebuilt after structural changes).
    input_layer : torch.nn.Module | None
        First trainable leaf module (may be ``None`` if module has no parameters).
    output_layer : torch.nn.Module | None
        Last trainable leaf module.
    observed_features : SortedSet[str]
        Ordered set of feature names seen so far.
    module_input_len : int | None
        Cached original input size of the input layer (if identifiable).
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
        gradient_clip_value: float | None = None,
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
        self.gradient_clip_value = gradient_clip_value

        self.kwargs = kwargs

        # Explicit Optional annotations to satisfy mypy when assigning None
        self.input_layer: Optional[torch.nn.Module] = None
        self.output_layer: Optional[torch.nn.Module] = None

        candidates = self._extract_candidate_layers(self.module)
        # Pick the first parameterised layer as input_layer
        for cand in candidates:
            if any(p.requires_grad for p in cand.parameters()):
                self.input_layer = cand
                break
        else:
            self.input_layer = candidates[0] if candidates else None
        # Pick the last parameterised layer as output_layer
        for cand in reversed(candidates):
            if any(p.requires_grad for p in cand.parameters()):
                self.output_layer = cand
                break
        if self.output_layer is None and candidates:
            self.output_layer = candidates[-1]

        # Store initial expected input length
        self.module_input_len = self._get_input_size() if self.input_layer else None
        self.observed_features: SortedSet = SortedSet()
        self.module.to(self.device)
        torch.manual_seed(seed)

    @staticmethod
    def _extract_candidate_layers(module: torch.nn.Module) -> list[torch.nn.Module]:
        """Return a flat list of leaf candidate layers.

        Recursively descends into ``module`` and collects leaf modules (modules
        without children) that are potentially expandable. Non‑parametric
        distribution layers like ``Softmax`` or ``LogSoftmax`` are excluded so the
        *input* and *output* layers represent learnable transformations.
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
        """Update the set of observed feature names.

        If ``is_feature_incremental`` is True and new feature names are detected,
        the input layer is expanded in‑place to match the new dimensionality. The
        optimiser is rebuilt after expansion so that newly created weights are
        tracked.

        Parameters
        ----------
        x : dict | pandas.DataFrame
            Sample (``dict``) or batch (``DataFrame``) whose keys/columns represent
            feature names.

        Returns
        -------
        bool
            True if previously unseen feature names were added.
        """
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
        """Convert a feature dict into a dense float tensor.

        Missing previously observed features are imputed with ``0.0``. If the
        current number of observed features is *smaller* than the module's initial
        expected input size a right‑hand zero padding is applied so shapes remain
        consistent during the warm‑up phase.
        """
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
        """Convert a feature DataFrame into a dense float tensor.

        See :meth:`_dict2tensor` for handling of missing features and padding.
        """
        default_value = 0.0
        tensor_data = df2tensor(
            X,
            self.observed_features,
            default_value=default_value,
            device=self.device,
            dtype=torch.float32,
        )
        return self._pad_tensor_if_needed(tensor_data, X.shape[0])

    def draw(self):  # type: ignore[override]
        """Render a (partial) computational graph of the wrapped model.

        Imports ``graphviz`` and ``torchviz`` lazily. Raises an informative
        ImportError if the optional dependencies are not installed.
        """
        try:  # pragma: no cover
            from torchviz import make_dot  # type: ignore
        except Exception as err:  # noqa: BLE001
            raise ImportError(
                "graphviz and torchviz must be installed to draw the model."
            ) from err

        first_parameter = next(self.module.parameters())
        input_shape = first_parameter.size()
        y_pred = self.module(torch.rand(input_shape))
        return make_dot(y_pred.mean(), params=dict(self.module.named_parameters()))

    def _get_input_size(self):
        """Infer the nominal input size of the input layer.

        The method inspects common attribute names (``in_features``, ``input_size``,
        etc.) or falls back to the second dimension of the layer's weight matrix.
        """
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
        """Infer the output dimensionality of the last trainable layer."""
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
        elif hasattr(layer, "weight") and getattr(layer, "weight") is not None:
            return layer.weight.shape[0]
        else:
            # Fallback: walk backwards to the closest layer with weights
            for cand in reversed(list(self.module.modules())):
                if hasattr(cand, "weight") and getattr(cand, "weight") is not None:
                    return cand.weight.shape[0]
            raise ValueError(
                f"Cannot determine output size for layer type {type(layer)}"
            )

    def _pad_tensor_if_needed(self, tensor_data, x_len, default_value=0.0):
        """Right‑pad ``tensor_data`` to the module's nominal input size if needed.

        During the warm‑up phase (before all *initial* features have been observed)
        incoming tensors may have fewer columns than the module expects. To keep
        shapes consistent we append zero columns (or the provided ``default_value``)
        until the nominal size is reached.

        Parameters
        ----------
        tensor_data : torch.Tensor
            Prepared feature tensor.
        x_len : int
            Number of instances represented (1 for single samples, batch size or
            sequence length depending on shape).
        default_value : float, default=0.0
            Value used for padding.

        Returns
        -------
        torch.Tensor
            Possibly padded tensor.
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
        """Create a rule spec describing how to expand a layer in place.

        The returned dictionary indicates which attributes / parameter tensors can
        be resized when performing input (feature) or output (class / unit) growth.
        """
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
        # LSTM special case
        if isinstance(layer, torch.nn.LSTM):
            instructions["input_size"] = "input_attribute"
            if hasattr(layer, "weight_ih_l0"):
                instructions["weight_ih_l0"] = {
                    "input": [{"axis": 1, "n_subparams": 1}]
                }
        return instructions

    def _expand_layer(
        self, layer: torch.nn.Module, target_size: int, output: bool = True
    ):
        """In‑place expand ``layer`` to ``target_size`` along input or output dimension."""
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
                        param = self._expand_weights(
                            param, axis, dims_to_add, n_subparams
                        )
                        if not isinstance(param, torch.nn.Parameter):
                            param = torch.nn.Parameter(param)
                        setattr(layer, param_name, param)
                        layer_modified = True
        # Rebuild optimiser so new params are tracked
        if layer_modified:
            self._rebuild_optimizer()

    def _rebuild_optimizer(self):
        """Recreate the optimiser with current model parameters.

        Necessary after structural expansion so newly created parameters receive
        gradients and updates.
        """
        optim_fn = get_optim_fn(self.optimizer_fn)
        self.optimizer = optim_fn(self.module.parameters(), lr=self.lr)

    @staticmethod
    def _expand_weights(
        param: torch.Tensor, axis: int, dims_to_add: int, n_subparams: int
    ):
        """Return a new parameter tensor with additional randomly initialised slices.

        Parameters
        ----------
        param : torch.Tensor
            Original parameter tensor.
        axis : int
            Axis along which to append.
        dims_to_add : int
            Number of units to append.
        n_subparams : int
            (Reserved for future composite parameters – currently unused.)
        """
        if dims_to_add <= 0:
            return param

        new_weights = (
            torch.randn(
                *(param.shape[:axis] + (dims_to_add,) + param.shape[axis + 1 :]),
                device=param.device,
                dtype=param.dtype,
            )
            * 0.01
        )
        expanded_param = torch.cat([param, new_weights], dim=axis)
        return torch.nn.Parameter(expanded_param)

    def _learn(self, x: torch.Tensor, y: Optional[Any] = None):
        """Perform a single optimisation step.

        Behaviour depends on the task context:

        * Autoencoding: ``y`` is ``None`` – the input tensor is used as target.
        * Regression: ``y`` is a scalar/1D value converted to a tensor.
        * Classification: handled in subclasses which prepare ``y`` appropriately
          (either via one‑hot encoding or class indices for cross entropy).
        """
        y_pred = self.module(x)
        if y is None:  # Autoencoder
            y = x
        elif not hasattr(self, "observed_classes"):  # Regression
            if not isinstance(y, torch.Tensor):
                y = float2tensor(y, self.device)
        else:  # Classification (one‑hot path)
            n_classes = y_pred.shape[-1]
            observed_classes = getattr(self, "observed_classes", SortedSet())
            y = labels2onehot(y, observed_classes, n_classes, self.device)

        self.module.train()
        loss = self.loss_func(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        if getattr(self, "gradient_clip_value", None) is not None:
            clip_val = self.gradient_clip_value
            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), clip_val)
        self.optimizer.step()

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist the estimator (architecture, weights, optimiser & runtime state).

        Parameters
        ----------
        filepath : str | Path
            Destination file. Parent directories are created automatically.
        """
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
        """Load a previously saved estimator.

        The method reconstructs the estimator class, its wrapped module, optimiser
        state and runtime information (feature names, buffers, etc.).
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        estimator_cls = cls._import_from_path(state["estimator_class"])
        init_params = state["init_params"]

        # Rebuild module if needed
        if "module" in init_params and isinstance(init_params["module"], dict):
            module_info = init_params.pop("module")
            module_cls = cls._import_from_path(module_info["class"])
            module = module_cls(
                **cls._filter_kwargs(module_cls.__init__, module_info["kwargs"])
            )
            if state.get("model_state_dict"):
                module.load_state_dict(state["model_state_dict"])
            init_params["module"] = module

        estimator = estimator_cls(
            **cls._filter_kwargs(estimator_cls.__init__, init_params)
        )

        if state.get("optimizer_state_dict") and hasattr(estimator, "optimizer"):
            try:
                estimator.optimizer.load_state_dict(
                    state["optimizer_state_dict"]  # type: ignore[arg-type]
                )
            except Exception:  # noqa: E722
                pass

        estimator._restore_runtime_state(state.get("runtime_state", {}))
        return estimator

    def clone(
        self,
        new_params=None,
        include_attributes: bool = False,
        copy_weights: bool = False,
    ):
        """Return a fresh estimator instance with (optionally) copied state.

        Parameters
        ----------
        new_params : dict | None
            Parameter overrides for the cloned instance.
        include_attributes : bool, default=False
            If True, runtime state (observed features, buffers) is also copied.
        copy_weights : bool, default=False
            If True, model weights are copied (otherwise the module is re‑initialised).
        """
        new_params = new_params or {}
        copy_weights = new_params.pop("copy_weights", copy_weights)

        params = {**self._get_all_init_params(), **new_params}

        if "module" not in new_params:
            params["module"] = self._rebuild_module()

        new_est = self.__class__(**self._filter_kwargs(self.__class__.__init__, params))

        if copy_weights and hasattr(self.module, "state_dict"):
            new_est.module.load_state_dict(self.module.state_dict())

        if include_attributes:
            new_est._restore_runtime_state(self._get_runtime_state())

        return new_est

    def _get_all_init_params(self) -> Dict[str, Any]:
        """Return a serialisable mapping of constructor parameters.

        The module is represented by its fully qualified class name and a best
        effort reconstruction of its init kwargs (see :meth:`_infer_module_params`).
        """
        sig = inspect.signature(self.__class__.__init__)
        params = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            elif name == "module":
                params["module"] = {
                    "class": f"{type(self.module).__module__}.{type(self.module).__name__}",
                    "kwargs": {
                        **getattr(self, "kwargs", {}),
                        **self._infer_module_params(),
                    },
                }
            elif hasattr(self, name):
                params[name] = getattr(self, name)
            elif name in getattr(self, "kwargs", {}):
                params[name] = self.kwargs[name]

        return params

    def _get_runtime_state(self) -> Dict[str, Any]:
        """Collect runtime (non‑constructor) state for persistence."""
        state = {}

        for attr in ["observed_features", "observed_classes"]:
            if hasattr(self, attr):
                state[attr] = list(getattr(self, attr, []))

        if hasattr(self, "_x_window") and self._x_window:
            state["window_buffer"] = list(self._x_window)

        return state

    def _restore_runtime_state(self, state: Dict[str, Any]) -> None:
        """Restore runtime state (features, classes, rolling buffers)."""
        for attr, data in state.items():
            if attr.endswith(("_features", "_classes")):
                setattr(self, attr, SortedSet(data) if data else SortedSet())
            elif attr == "window_buffer" and hasattr(self, "_x_window"):
                from collections import deque

                self._x_window = deque(data, maxlen=getattr(self, "window_size", 10))

    # -------------------- Helper utilities --------------------
    def _infer_module_params(self) -> dict:
        """Best effort inference of module constructor parameters.

        Examines the module's ``__init__`` signature and pulls attribute values
        with matching names from the current instance. For linear layers it also
        infers ``n_features`` from the first 2D weight matrix encountered.
        """
        params = {}
        if hasattr(self.module, "__dict__"):
            sig = inspect.signature(self.module.__class__.__init__)
            for name in sig.parameters:
                if name != "self" and hasattr(self.module, name):
                    params[name] = getattr(self.module, name)

        if "n_features" not in params and hasattr(self.module, "named_parameters"):
            for name, param in self.module.named_parameters():
                if "weight" in name and param.dim() == 2:
                    params["n_features"] = param.shape[1]
                    break

        return params

    def _rebuild_module(self):
        """Create a fresh (re‑initialised) copy of the wrapped module."""
        params = self._infer_module_params()
        try:
            return self.module.__class__(
                **self._filter_kwargs(self.module.__class__.__init__, params)
            )
        except Exception:  # noqa: E722
            mod_copy = copy.deepcopy(self.module)
            for m in mod_copy.modules():
                if hasattr(m, "reset_parameters"):
                    try:
                        m.reset_parameters()
                    except Exception:  # noqa: E722
                        pass
            return mod_copy

    @staticmethod
    def _import_from_path(path: str):
        """Import and return an object from a fully qualified dotted path."""
        module_path, name = path.rsplit(".", 1)
        return getattr(importlib.import_module(module_path), name)

    @staticmethod
    def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a mapping to parameters accepted by ``callable_obj``."""
        try:
            sig = inspect.signature(callable_obj)
            allowed = {
                p.name
                for p in sig.parameters.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            return {k: v for k, v in kwargs.items() if k in allowed}
        except (ValueError, TypeError):
            return kwargs

    @staticmethod
    def _serialize_sorted_set(sorted_set: SortedSet) -> list:
        """Convert a ``SortedSet`` to a plain list (JSON / pickle friendly)."""
        return list(sorted_set) if sorted_set else []

    @staticmethod
    def _deserialize_sorted_set(data: list) -> SortedSet:
        """Convert a list back into a ``SortedSet``."""
        return SortedSet(data) if data else SortedSet()


class RollingDeepEstimator(DeepEstimator):
    """Extension of :class:`DeepEstimator` with a fixed-size rolling window.

    Maintains a ``collections.deque`` of the most recent ``window_size`` inputs
    enabling models (e.g. sequence learners) to condition on a short history.
    Optionally the model's own predictions can be appended to the window
    (via ``append_predict``) to facilitate iterative forecasting.
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
        """Convert the internal deque to a tensor (with padding logic)."""
        tensor_data = deque2rolling_tensor(x_win, device=self.device)
        return self._pad_tensor_if_needed(tensor_data, len(x_win))
