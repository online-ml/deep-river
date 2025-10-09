from typing import Callable, Iterable, Mapping, Sequence, SupportsFloat, Union, cast

import pandas as pd
import torch
from river import base
from river.base.typing import FeatureName, RegTarget
from sortedcontainers import SortedSet

from deep_river.base import DeepEstimator


class _TestModule(torch.nn.Module):
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.dense0 = torch.nn.Linear(n_features, n_outputs)

    def forward(self, X, **kwargs):  # noqa: D401 - simple passthrough
        return self.dense0(X)


class MultiTargetRegressor(base.MultiTargetRegressor, DeepEstimator):
    """Incremental multi-target regression wrapper for PyTorch modules.

    This estimator adapts a ``torch.nn.Module`` to the :mod:`river` streaming API
    for *multi‑target* (a.k.a. multi‑output) regression. It optionally supports
    *feature‑incremental* learning (dynamic growth of the input layer when new
    feature names appear) as provided by :class:`deep_river.base.DeepEstimator` and
    additionally (optionally) **target‑incremental** learning: if new target names
    appear during the stream, the *output layer* can be expanded on‑the‑fly so the
    model natively handles the enlarged target vector.

    Targets are tracked via an ordered :class:`~sortedcontainers.SortedSet` to
    guarantee deterministic ordering between training and prediction. Incoming
    target dictionaries / frames are converted into dense tensors with columns
    arranged according to the observed target name order. Missing targets (when
    the model has been expanded but a prior sample omits some target) are imputed
    with ``0.0``.

    Parameters
    ----------
    module : torch.nn.Module
        PyTorch module producing an output tensor of shape ``(N, T)`` where ``T``
        is the current number of target variables.
    loss_fn : str | Callable, default='mse'
        Loss identifier or custom callable passed through :func:`deep_river.utils.get_loss_fn`.
    optimizer_fn : str | Callable, default='sgd'
        Optimizer identifier (e.g. ``'adam'``, ``'sgd'``) or factory / class.
    is_feature_incremental : bool, default=False
        If True, unseen feature names trigger expansion of the first trainable
        layer (see :class:`DeepEstimator`).
    is_target_incremental : bool, default=False
        If True, unseen target names trigger expansion of the last trainable
        layer. Expansion *preserves* existing weights and initialises new units
        with small random values.
    lr : float, default=1e-3
        Learning rate.
    device : str, default='cpu'
        Torch device (e.g. ``'cuda'``).
    seed : int, default=42
        Random seed for reproducibility.
    **kwargs
        Extra arguments stored for persistence / cloning.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from deep_river.regression.multioutput import MultiTargetRegressor
    >>> class TinyMultiNet(nn.Module):
    ...     def __init__(self, n_features, n_outputs):
    ...         super().__init__()
    ...         self.net = nn.Sequential(
    ...             nn.Linear(n_features, 8),
    ...             nn.ReLU(),
    ...             nn.Linear(8, n_outputs)
    ...         )
    ...     def forward(self, x):
    ...         return self.net(x)
    >>> model = MultiTargetRegressor(
    ...     module=TinyMultiNet(3, 2),
    ...     loss_fn='mse',
    ...     optimizer_fn='sgd',
    ...     is_feature_incremental=True,
    ...     is_target_incremental=True,
    ... )
    >>> x = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    >>> y = {'y1': 10.0, 'y2': 20.0}
    >>> _ = model.learn_one(x, y)
    >>> model.predict_one(x)
    {'y1': 0.5906543731689453, 'y2': 0.6837220191955566}

    Notes
    -----
    * The module's last *trainable* leaf layer is treated as output layer for
      potential expansion. Non‑parametric terminal activations like ``Softmax``
      are skipped (mirroring the logic in :class:`DeepEstimator`).
    * If ``is_target_incremental`` is disabled, the number of outputs is fixed
      and encountering a new target name will only register it internally (the
      tensor conversion will still allocate a slot, but the model's output layer
      size will not change, possibly causing a mismatch). Therefore, enabling
      target incrementality is recommended for truly open‑world streams.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        loss_fn: Union[str, Callable] = "mse",
        optimizer_fn: Union[str, Callable] = "sgd",
        is_feature_incremental: bool = False,
        is_target_incremental: bool = False,
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            module=module,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            lr=lr,
            device=device,
            seed=seed,
            is_feature_incremental=is_feature_incremental,
            **kwargs,
        )
        self.is_target_incremental = is_target_incremental
        self.observed_targets: SortedSet[FeatureName] = SortedSet()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def learn_one(
        self,
        x: dict,
        y: dict[FeatureName, RegTarget],
        **kwargs,
    ) -> None:
        """Learn from a single multi-target instance.

        Parameters
        ----------
        x : dict[str, float]
            Feature mapping.
        y : dict[str, float]
            Mapping of target name -> target value.
        **kwargs
            Ignored (kept for signature compatibility / future hooks).
        """
        self._update_observed_features(x)
        self._update_observed_targets(y)
        x_t = self._dict2tensor(dict(x))
        y_t = self._single_target_dict_to_tensor(y)
        self._learn(x_t, y_t)

    def learn_many(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, Mapping[str, Sequence[RegTarget]]],
    ) -> None:
        """Learn from a batch of multi-target instances.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix (rows are samples, columns are feature names).
        y : pandas.DataFrame | pandas.Series | mapping
            Target matrix. Preferred is a DataFrame with one column per target.
            A Series is interpreted as *one* target. A mapping of ``name -> list``
            is converted into a DataFrame first.
        """
        self._update_observed_features(X)
        y_df = self._coerce_targets_to_frame(y)
        self._update_observed_targets(y_df)

        x_t = self._df2tensor(X)
        y_t = self._multi_target_frame_to_tensor(y_df)
        self._learn(x_t, y_t)

    def predict_one(self, x: dict) -> dict[FeatureName, RegTarget]:
        """Predict a dictionary of target values for a single instance."""
        self._update_observed_features(x)
        x_t = self._dict2tensor(dict(x))
        self.module.eval()
        with torch.inference_mode():
            y_pred_t = self.module(x_t).squeeze(0)
            if y_pred_t.dim() == 0:  # single value fallback
                y_pred_t = y_pred_t.view(1)
            if y_pred_t.is_cuda:
                y_pred_t = y_pred_t.cpu()
            y_list: list[float] = [float(v) for v in y_pred_t.tolist()]
        return {
            cast(FeatureName, t): cast(
                RegTarget, (y_list[i] if i < len(y_list) else float("nan"))
            )
            for i, t in enumerate(self.observed_targets)
        }

    def predict_many(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict target values for multiple instances.

        Returns
        -------
        pandas.DataFrame
            DataFrame whose columns follow the ordering of ``observed_targets``.
        """
        self._update_observed_features(X)
        x_t = self._df2tensor(X)
        self.module.eval()
        with torch.inference_mode():
            y_pred = self.module(x_t)
            if y_pred.is_cuda:
                y_pred = y_pred.cpu()
        # Ensure 2D
        if y_pred.dim() == 1:
            y_pred = y_pred.view(-1, 1)
        cols = list(self.observed_targets)
        # Truncate or pad columns if dimensions drift (defensive)
        if y_pred.shape[1] < len(cols):
            pad = torch.zeros(
                (y_pred.shape[0], len(cols) - y_pred.shape[1]),
                dtype=y_pred.dtype,
            )
            y_pred = torch.cat([y_pred, pad], dim=1)
        elif y_pred.shape[1] > len(cols):
            extra = [f"__extra_{i}" for i in range(y_pred.shape[1] - len(cols))]
            cols = cols + extra
        return pd.DataFrame(y_pred.numpy(), columns=cols)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _update_observed_targets(
        self, y: Union[Mapping[FeatureName, RegTarget], pd.Series, pd.DataFrame]
    ) -> bool:
        """Update the ordered set of observed target names and expand output layer.

        Parameters
        ----------
        y : mapping | pandas.Series | pandas.DataFrame
            Single-sample target mapping or batch target structure.

        Returns
        -------
        bool
            True if the output layer was expanded.
        """
        if isinstance(y, Mapping):
            new_targets: Iterable[FeatureName] = y.keys()
        elif isinstance(y, pd.Series):
            new_targets = [cast(FeatureName, y.name)] if y.name is not None else []
        elif isinstance(y, pd.DataFrame):
            # DataFrame columns can be any hashable type
            new_targets = list(cast(Iterable[FeatureName], y.columns))
        else:
            return False

        prev_n = len(self.observed_targets)
        self.observed_targets.update(new_targets)
        grew = len(self.observed_targets) > prev_n

        if (
            grew
            and self.is_target_incremental
            and self.output_layer is not None
            and len(self.observed_targets) > self._get_output_size()
        ):
            self._expand_layer(
                self.output_layer,
                target_size=len(self.observed_targets),
                output=True,
            )
            return True
        return False

    def _single_target_dict_to_tensor(
        self, y: Mapping[FeatureName, RegTarget]
    ) -> torch.Tensor:
        """Convert a single-sample target dict into a 2D tensor (shape (1, T))."""
        vector = [
            float(cast(SupportsFloat, y.get(t, 0.0)))  # type: ignore[arg-type]
            for t in self.observed_targets
        ]
        return torch.tensor([vector], dtype=torch.float32, device=self.device)

    def _coerce_targets_to_frame(
        self,
        y: Union[pd.DataFrame, pd.Series, Mapping[FeatureName, Sequence[RegTarget]]],
    ) -> pd.DataFrame:
        """Coerce assorted multi-target representations into a DataFrame."""
        if isinstance(y, pd.DataFrame):
            return y
        if isinstance(y, pd.Series):
            # Single target series -> DataFrame with its name (or 'y0').
            name = y.name or "y0"
            return y.to_frame(name)
        if isinstance(y, Mapping):
            return pd.DataFrame(y)
        raise TypeError(
            "Unsupported multi-target type. Expect DataFrame, Series or mapping of lists."
        )

    def _multi_target_frame_to_tensor(self, y_df: pd.DataFrame) -> torch.Tensor:
        """Convert a target DataFrame to a tensor with ordering by observed targets."""
        # Guarantee all observed targets present as columns (add zeros if missing)
        for t in self.observed_targets:
            if t not in y_df.columns:
                y_df[t] = 0.0
        # Reorder columns
        y_df = y_df[list(self.observed_targets)]
        return torch.tensor(y_df.values, dtype=torch.float32, device=self.device)

    def _get_runtime_state(self) -> dict:
        """Extend base runtime state with observed target names."""
        state = super()._get_runtime_state()
        state["observed_targets"] = list(self.observed_targets)
        return state

    def _restore_runtime_state(self, state: dict) -> None:  # noqa: D401
        """Restore runtime state including observed targets."""
        super()._restore_runtime_state(state)
        if "observed_targets" in state:
            self.observed_targets = SortedSet(state["observed_targets"])  # type: ignore[arg-type]

    # ---------------------------------------------------------------------
    # Testing utilities
    # ---------------------------------------------------------------------
    @classmethod
    def _unit_test_params(cls):  # noqa: D401 - part of project-wide testing pattern
        """Yield parameter dictionaries for unit tests."""
        yield {
            "module": _TestModule(10, 3),
            "loss_fn": "l1",
            "optimizer_fn": "sgd",
            "is_feature_incremental": True,
            "is_target_incremental": True,
        }

    @classmethod
    def _unit_test_skips(cls) -> set:  # noqa: D401
        """Return names of generic checks to skip for this estimator."""
        return {
            "check_shuffle_features_no_impact",
        }
