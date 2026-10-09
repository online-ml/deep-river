"""Utilities for unit testing and sanity checking estimators."""

import collections
import copy
import importlib
import inspect
import math
import pickle
import tempfile
import warnings
from collections.abc import Set
from importlib.util import find_spec
from pathlib import Path

__all__ = [
    "check_estimator",
    "iter_estimators",
    "iter_estimators_that_can_be_tested",
]

import typing

import numpy as np
import pandas as pd
import torch
from river import base
from river.base import Estimator
from river.checks import _wrapped_partial, _yield_datasets
from river.checks import yield_checks as yield_river_checks
from river.time_series.base import Forecaster

from deep_river.utils.river_compat import AnomalyDetector


def check_roc_auc(model, dataset):
    from sklearn.metrics import roc_auc_score

    scores = []
    labels = []
    for x, y in dataset:
        score = model.score_one(x)
        assert np.isfinite(score)
        scores.append(score)
        labels.append(y)
        model.learn_one(x)
    assert roc_auc_score(labels, scores) > 0.5


def yield_checks(model):
    for check in yield_river_checks(model):
        if check.__name__ == "check_roc_auc":
            yield _wrapped_partial(check_roc_auc, **check.keywords)
        else:
            yield check


def iter_estimators(submodules=None):
    if submodules is None:
        submodules = importlib.import_module("deep_river").__all__

    def is_estimator(obj):
        return inspect.isclass(obj) and issubclass(obj, Estimator)

    for submodule in submodules:
        yield from (
            obj
            for _, obj in inspect.getmembers(
                importlib.import_module(f"deep_river.{submodule}"), is_estimator
            )
        )


def iter_estimators_that_can_be_tested(submodules=None):
    ignored = ()

    def can_be_tested(estimator):
        return not inspect.isabstract(estimator) and not issubclass(estimator, ignored)

    for estimator in filter(can_be_tested, iter_estimators(submodules)):
        for params in estimator._unit_test_params():
            yield estimator(**params)


def check_deep_learn_one(model, dataset):
    import pytest

    # Simulate a crash during backward pass
    def patched_backward(self, *args, **kwargs):
        original_backward(self, *args, **kwargs)
        raise RuntimeError("Simulated exception during backward pass")

    for x, y in dataset:
        original_backward = torch.Tensor.backward
        torch.Tensor.backward = patched_backward

        try:
            # First learn_one call - will raise exception after computing gradients
            with pytest.raises(RuntimeError):
                if isinstance(model, Forecaster):
                    model.learn_one(y, x)
                elif model._supervised:
                    model.learn_one(x, y)
                else:
                    model.learn_one(x)
        finally:
            # Always restore the original function
            torch.Tensor.backward = original_backward

        for param in model.module.parameters():
            # New gradients were computed (not None)
            assert param.grad is not None, "learn_one() should compute gradients"
            # They are valid (finite values)
            assert torch.all(
                torch.isfinite(param.grad)
            ), "learn_one() should produce finite gradients"


def check_dict2tensor(model):
    x = {"a": 1, "b": 2, "c": 3}
    model._update_observed_features(x)
    input_len = model._get_input_size()
    lst = [1, 2, 3]
    lst.extend([0] * (input_len - 3))
    assert model._dict2tensor(x).tolist() == [lst]

    x2 = {"b": 2, "c": 3}
    lst = [0, 2, 3]
    lst.extend([0] * (input_len - 3))
    assert model._dict2tensor(x2).tolist() == [lst]

    x3 = {"b": 2, "a": 1, "c": 3}
    lst = [1, 2, 3]
    lst.extend([0] * (input_len - 3))
    assert model._dict2tensor(x3).tolist() == [lst]


def _assert_persisted_value(left, right, visited=None):
    """Assert that two persisted values have identical types and contents."""
    if visited is None:
        visited = set()
    pair = (id(left), id(right))
    if pair in visited:
        return
    visited.add(pair)
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right, equal_nan=True)
    elif isinstance(left, pd.DataFrame):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    elif isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right, check_exact=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_persisted_value(left[key], right[key], visited)
    elif isinstance(left, (list, tuple, collections.deque)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_persisted_value(left_item, right_item, visited)
    elif isinstance(left, Set):
        assert left == right
    elif callable(left):
        assert left == right
    elif hasattr(left, "__dict__"):
        _assert_persisted_value(vars(left), vars(right), visited)
    else:
        assert left == right


def _round_trip(model):
    """Save and reload an estimator through a temporary persistence file."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pkl"
        model.save(path)
        assert path.is_file()
        rng_state = torch.get_rng_state()
        loaded_model = type(model).load(path)
        assert torch.equal(torch.get_rng_state(), rng_state)
        return loaded_model


def _assert_persisted_estimator(model, loaded_model):
    """Assert that a loaded estimator exactly preserves model and optimizer state."""
    assert type(model) is type(loaded_model)
    assert model is not loaded_model
    assert model.module is not loaded_model.module
    assert repr(model.module) == repr(loaded_model.module)
    assert model.module.training == loaded_model.module.training
    assert model.__dict__.keys() == loaded_model.__dict__.keys()
    excluded = {"module", "optimizer", "input_layer", "output_layer"}
    for name in model.__dict__.keys() - excluded:
        _assert_persisted_value(model.__dict__[name], loaded_model.__dict__[name])
    _assert_persisted_value(model.module.state_dict(), loaded_model.module.state_dict())
    _assert_persisted_value(
        model.optimizer.state_dict(), loaded_model.optimizer.state_dict()
    )
    module_parameters = {
        id(parameter) for parameter in loaded_model.module.parameters()
    }
    optimizer_parameters = {
        id(parameter)
        for group in loaded_model.optimizer.param_groups
        for parameter in group["params"]
    }
    assert module_parameters == optimizer_parameters


def _prediction(model, x):
    """Produce the task-specific prediction used for persistence comparison."""
    if isinstance(model, Forecaster):
        return model.forecast(horizon=3, xs=[x] * 3)
    if isinstance(model, base.Classifier):
        return model.predict_proba_one(x)
    if isinstance(model, (base.MultiTargetRegressor, base.Regressor)):
        return model.predict_one(x)
    return model.score_one(x)


def check_model_persistence(model, dataset):
    """Check exact persistence after training an estimator on five samples."""
    last_x = None
    for sample_count, (x, y) in enumerate(dataset, start=1):
        if isinstance(model, Forecaster):
            model.learn_one(y, x)
        elif model._supervised:
            model.learn_one(x, y)
        else:
            model.learn_one(x)
        last_x = x
        if sample_count == 5:
            break

    assert last_x is not None
    loaded_model = _round_trip(model)
    _assert_persisted_estimator(model, loaded_model)
    _assert_persisted_value(
        _prediction(model, last_x), _prediction(loaded_model, last_x)
    )


def check_model_persistence_untrained(model):
    """Check exact persistence before an estimator has received any samples."""
    loaded_model = _round_trip(model)
    _assert_persisted_estimator(model, loaded_model)


def check_model_persistence_rejects_other_type(model):
    """Check that an estimator file cannot be loaded through an incompatible type."""
    import pytest

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pkl"
        model.save(path)
        mismatched_type = type("MismatchedEstimator", (type(model),), {})
        with pytest.raises(TypeError):
            mismatched_type.load(path)


def check_model_persistence_legacy_format(model):
    """Check that files produced by the previous persistence format remain loadable."""
    state = {
        "estimator_class": f"{type(model).__module__}.{type(model).__name__}",
        "init_params": model._get_all_init_params(),
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": model.optimizer.state_dict(),
        "runtime_state": model._get_runtime_state(),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy.pkl"
        with path.open("wb") as file:
            pickle.dump(state, file)
        rng_state = torch.get_rng_state()
        loaded_model = type(model).load(path)
        assert torch.equal(torch.get_rng_state(), rng_state)
    assert type(model) is type(loaded_model)


def check_model_persistence_after_incremental_expansion(model):
    """Check persistence after dynamic input and output layer expansion."""
    initial_input_size = model._get_input_size()
    initial_output_size = model._get_output_size()
    x = {f"f{index:04d}": float(index) for index in range(initial_input_size)}
    expanded_x = {**x, f"f{initial_input_size:04d}": 1.0}
    model.optimizer_fn = "adam"
    model._rebuild_optimizer()
    model.learn_one(x, 0)
    model.learn_one(expanded_x, 1)
    model.learn_one(expanded_x, 0)
    assert model._get_input_size() == initial_input_size + 1
    assert model._get_output_size() > initial_output_size
    loaded_model = _round_trip(model)
    _assert_persisted_estimator(model, loaded_model)
    _assert_persisted_value(
        _prediction(model, expanded_x), _prediction(loaded_model, expanded_x)
    )
    model.learn_one(expanded_x, 1)
    loaded_model.learn_one(expanded_x, 1)
    _assert_persisted_value(model.module.state_dict(), loaded_model.module.state_dict())
    _assert_persisted_value(
        model.optimizer.state_dict(), loaded_model.optimizer.state_dict()
    )


def check_seed_reproducibility(model):
    if not hasattr(model, "module") or not hasattr(model, "seed"):
        return

    with torch.random.fork_rng():
        initial_state = torch.get_rng_state()
        first = model.clone()
        assert torch.equal(torch.get_rng_state(), initial_state)

        torch.rand(10)
        advanced_state = torch.get_rng_state()
        second = model.clone()
        assert torch.equal(torch.get_rng_state(), advanced_state)

        first_parameters = tuple(first.module.parameters())
        second_parameters = tuple(second.module.parameters())
        assert len(first_parameters) == len(second_parameters)
        assert all(
            torch.equal(first_parameter, second_parameter)
            for first_parameter, second_parameter in zip(
                first_parameters, second_parameters
            )
        )

        different = model.clone({"seed": model.seed + 1})
        assert torch.equal(torch.get_rng_state(), advanced_state)
        different_parameters = tuple(different.module.parameters())
        assert len(first_parameters) == len(different_parameters)
        assert any(
            not torch.equal(first_parameter, different_parameter)
            for first_parameter, different_parameter in zip(
                first_parameters, different_parameters
            )
        )


BENCHMARK_N_FEATURES = 6
BENCHMARK_N_ONLINE = 32
BENCHMARK_N_BATCH = 64
CHECK_N_ONLINE = 12
CHECK_N_BATCH = 4


def _frame(n_samples: int, n_features: int) -> pd.DataFrame:
    values = np.arange(n_samples * n_features, dtype=np.float32).reshape(
        n_samples, n_features
    )
    scaled_values = ((values % 17) / 17).astype(np.float32)
    return pd.DataFrame(scaled_values, columns=[f"f{i}" for i in range(n_features)])


def _benchmark_frame(
    n_samples: int = BENCHMARK_N_BATCH,
    n_features: int = BENCHMARK_N_FEATURES,
) -> pd.DataFrame:
    return _frame(n_samples, n_features)


def _model_frame(model, n_samples: int) -> pd.DataFrame:
    return _frame(n_samples, model._get_input_size())


def _benchmark_rows(n_samples: int = BENCHMARK_N_ONLINE) -> list[dict[str, float]]:
    return _benchmark_frame(n_samples).to_dict(orient="records")


def _classification_targets(n_samples: int) -> pd.Series:
    return pd.Series([i % 2 for i in range(n_samples)])


def _regression_targets(n_samples: int) -> pd.Series:
    return pd.Series(np.linspace(0.0, 1.0, n_samples, dtype=np.float32))


def _multi_target_regression_targets(model, n_samples: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"y{i}": np.linspace(0.0, 1.0, n_samples, dtype=np.float32) + i
            for i in range(model._get_output_size())
        }
    )


def _expansion_stream() -> list[tuple[dict[str, float], int]]:
    return [
        ({"f0": 0.0, "f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}, 0),
        ({"f0": 0.1, "f1": 0.2, "f2": 0.3, "f3": 0.4, "f4": 0.5, "f5": 0.6}, 1),
        (
            {
                "f0": 0.6,
                "f1": 0.7,
                "f2": 0.8,
                "f3": 0.9,
                "f4": 1.0,
                "f5": 1.1,
                "f6": 1.2,
            },
            2,
        ),
    ]


def _learn_one_for_benchmark(model, x, y=None) -> None:
    if (
        isinstance(model, base.Classifier)
        or isinstance(model, base.MultiTargetRegressor)
        or isinstance(model, base.Regressor)
    ):
        model.learn_one(x, y)
    else:
        model.learn_one(x)


def _learn_many_for_benchmark(model, X, y=None) -> None:
    learn_many = getattr(model, "learn_many")
    if (
        isinstance(model, base.Classifier)
        or isinstance(model, base.MultiTargetRegressor)
        or isinstance(model, base.Regressor)
    ):
        learn_many(X, y)
    else:
        learn_many(X)


def _benchmark_targets_for(model, n_samples: int):
    if isinstance(model, base.Classifier):
        return _classification_targets(n_samples)
    if isinstance(model, base.MultiTargetRegressor):
        return _multi_target_regression_targets(model, n_samples)
    if isinstance(model, base.Regressor):
        return _regression_targets(n_samples)
    return None


def _target_rows(y):
    if isinstance(y, pd.DataFrame):
        return y.to_dict(orient="records")
    return y


def _is_rolling(model) -> bool:
    return hasattr(model, "window_size") and hasattr(model, "_x_window")


def _fit_for_many_check(model):
    n_samples = max(CHECK_N_ONLINE, getattr(model, "window_size", 0))
    X = _model_frame(model, n_samples)
    y = _benchmark_targets_for(model, len(X))
    if y is None:
        for x in X.to_dict(orient="records"):
            _learn_one_for_benchmark(model, x)
    else:
        for x, target in zip(X.to_dict(orient="records"), _target_rows(y)):
            _learn_one_for_benchmark(model, x, target)
    return model


def _fit_for_benchmark(model):
    rows = _benchmark_rows()
    y = _benchmark_targets_for(model, len(rows))
    if y is None:
        for x in rows:
            _learn_one_for_benchmark(model, x)
    else:
        for x, target in zip(rows, y):
            _learn_one_for_benchmark(model, x, target)
    return model


def _assert_learning_state(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_learning_state(left[key], right[key])
    elif isinstance(left, (list, tuple, collections.deque)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_learning_state(first, second)
    else:
        assert left == right


def check_batch_size_one_learning(model):
    if _is_rolling(model):
        _fit_for_many_check(model)
    online = copy.deepcopy(model)
    batch = copy.deepcopy(model)
    X = _model_frame(model, CHECK_N_BATCH)
    y = _benchmark_targets_for(model, len(X))
    targets = [None] * len(X) if y is None else list(_target_rows(y))
    devices = (
        [torch.device(model.device)]
        if torch.device(model.device).type == "cuda"
        else []
    )
    for index, (x, target) in enumerate(zip(X.to_dict(orient="records"), targets)):
        with torch.random.fork_rng(devices=devices):
            _learn_one_for_benchmark(online, x, target)
        with torch.random.fork_rng(devices=devices):
            _learn_many_for_benchmark(
                batch,
                X.iloc[index : index + 1],
                None if y is None else y.iloc[index : index + 1],
            )
        _assert_learning_state(online.module.state_dict(), batch.module.state_dict())
        _assert_learning_state(
            online.optimizer.state_dict(), batch.optimizer.state_dict()
        )
        if _is_rolling(model):
            _assert_learning_state(online._x_window, batch._x_window)


def check_rolling_batch_learning(model, batch_size, warmup):
    if warmup:
        X_warmup = _model_frame(model, warmup)
        y_warmup = _benchmark_targets_for(model, len(X_warmup))
        targets = (
            [None] * len(X_warmup) if y_warmup is None else list(_target_rows(y_warmup))
        )
        for x, target in zip(X_warmup.to_dict(orient="records"), targets):
            _learn_one_for_benchmark(model, x, target)

    online = copy.deepcopy(model)
    X = _model_frame(model, batch_size)
    y = _benchmark_targets_for(model, len(X))
    targets = [None] * len(X) if y is None else list(_target_rows(y))
    for x, target in zip(X.to_dict(orient="records"), targets):
        _learn_one_for_benchmark(online, x, target)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _learn_many_for_benchmark(model, X, y)
    _assert_learning_state(model.module.state_dict(), online.module.state_dict())
    _assert_learning_state(model.optimizer.state_dict(), online.optimizer.state_dict())
    _assert_learning_state(model._x_window, online._x_window)


def check_rolling_batch_rejects_misaligned_targets(model):
    import pytest

    X = _model_frame(model, 2)
    y = _benchmark_targets_for(model, 1)
    parameters = copy.deepcopy(model.module.state_dict())
    window = copy.deepcopy(model._x_window)
    with pytest.raises(ValueError, match="same number of rows"):
        _learn_many_for_benchmark(model, X, y)
    _assert_learning_state(model.module.state_dict(), parameters)
    _assert_learning_state(model._x_window, window)


def _tensor_storage_bytes(value):
    visited = {}
    storages = {}
    pending = [value]

    while pending:
        obj = pending.pop()
        if id(obj) in visited:
            continue
        visited[id(obj)] = obj
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            storages[(obj.device, storage.data_ptr())] = storage.nbytes()
            if obj.is_leaf or obj.retains_grad:
                pending.append(obj.grad)
            pending.append(obj.grad_fn)
        elif isinstance(obj, torch.autograd.graph.Node):
            pending.extend(node for node, _ in obj.next_functions)
            pending.append(getattr(obj, "variable", None))
            for name in dir(obj):
                if name == "saved_tensors" or name.startswith("_saved_"):
                    try:
                        saved = getattr(obj, name)
                    except RuntimeError as error:
                        if "after they have already been freed" not in str(error):
                            raise
                    else:
                        pending.append(saved)
        elif isinstance(obj, dict):
            pending.extend(obj.keys())
            pending.extend(obj.values())
        elif isinstance(obj, (list, tuple, collections.deque, Set)):
            pending.extend(obj)
        elif (
            not isinstance(obj, type)
            and not inspect.isroutine(obj)
            and hasattr(obj, "__dict__")
        ):
            pending.append(vars(obj))

    return sum(storages.values())


def _check_fixed_feature_memory(model, X):
    y = _benchmark_targets_for(model, len(X))
    targets = [None] * len(X) if y is None else list(_target_rows(y))
    rows = X.to_dict(orient="records")

    def run():
        for x, target in zip(rows, targets):
            _prediction(model, x)
            _learn_one_for_benchmark(model, x, target)
            if _is_rolling(model):
                assert model._x_window.maxlen == model.window_size
                assert len(model._x_window) <= model.window_size
                assert all(
                    len(row) == len(model.observed_features) for row in model._x_window
                )

    run()
    warmed_bytes = _tensor_storage_bytes(model)
    for _ in range(2):
        run()
        assert _tensor_storage_bytes(model) <= warmed_bytes
        if _is_rolling(model):
            assert len(model._x_window) == model.window_size


def check_bounded_tensor_memory(model):
    n_samples = max(32, 2 * getattr(model, "window_size", 0))
    _check_fixed_feature_memory(model, _model_frame(model, n_samples))


def check_rolling_memory_after_feature_growth(model):
    model = _prepare_rolling_feature_check(model)
    model.optimizer_fn = "adam"
    model._rebuild_optimizer()
    X = _model_frame(model, 32)
    _check_fixed_feature_memory(model, X)
    initial_bytes = _tensor_storage_bytes(model)
    X["new_feature"] = 0.5
    _check_fixed_feature_memory(model, X)
    assert model._get_input_size() == X.shape[1]
    assert _tensor_storage_bytes(model) > initial_bytes


def check_benchmark_learn_one(model, benchmark):
    rows = _benchmark_rows()
    y = _benchmark_targets_for(model, len(rows))

    def run():
        estimator = copy.deepcopy(model)
        if y is None:
            for x in rows:
                _learn_one_for_benchmark(estimator, x)
        else:
            for x, target in zip(rows, y):
                _learn_one_for_benchmark(estimator, x, target)
        return len(estimator.observed_features)

    assert benchmark(run) == len(rows[0])


def check_benchmark_predict_one(model, benchmark):
    rows = _benchmark_rows()
    estimator = _fit_for_benchmark(copy.deepcopy(model))

    def run():
        result = None
        for x in rows:
            if isinstance(estimator, base.Classifier):
                result = estimator.predict_proba_one(x)
            elif isinstance(estimator, base.Regressor):
                result = estimator.predict_one(x)
            else:
                result = estimator.score_one(x)
        return result

    result = benchmark(run)
    if isinstance(estimator, base.Classifier):
        assert result
    elif isinstance(estimator, base.Regressor):
        assert isinstance(result, float)
    else:
        assert result >= 0.0


def check_benchmark_learn_many(model, benchmark):
    X = _benchmark_frame()
    y = _benchmark_targets_for(model, len(X))

    def run():
        estimator = copy.deepcopy(model)
        _learn_many_for_benchmark(estimator, X, y)
        return len(estimator.observed_features)

    assert benchmark(run) == X.shape[1]


def check_benchmark_predict_many(model, benchmark):
    X = _benchmark_frame()
    estimator = copy.deepcopy(model)
    _learn_many_for_benchmark(estimator, X, _benchmark_targets_for(estimator, len(X)))

    def run():
        if isinstance(estimator, base.Classifier):
            return estimator.predict_proba_many(X).shape
        if isinstance(estimator, base.Regressor):
            return len(estimator.predict_many(X))
        return len(estimator.score_many(X))

    result = benchmark(run)
    if isinstance(estimator, base.Classifier):
        assert result[0] == len(X)
    else:
        assert result == len(X)


def check_benchmark_incremental_expansion(model, benchmark):
    stream = _expansion_stream()

    def run():
        estimator = copy.deepcopy(model)
        for x, y in stream:
            estimator.learn_one(x, y)
        return estimator._get_input_size(), estimator._get_output_size()

    n_features, n_outputs = benchmark(run)
    assert n_features >= 7
    assert n_outputs >= 3


def check_predict_many_output_length(model):
    if isinstance(model, Forecaster):
        return

    if not any(
        hasattr(model, method)
        for method in ("predict_proba_many", "predict_many", "score_many")
    ):
        return

    estimator = _fit_for_many_check(model)
    n_samples = 1 if _is_rolling(estimator) else CHECK_N_BATCH
    X = _model_frame(estimator, n_samples)

    if isinstance(estimator, base.Classifier):
        result = estimator.predict_proba_many(X)
    elif isinstance(estimator, base.MultiTargetRegressor) or isinstance(
        estimator, base.Regressor
    ):
        result = estimator.predict_many(X)
    else:
        result = estimator.score_many(X)

    assert len(result) == len(X)


def yield_benchmark_checks(model) -> typing.Iterator[typing.Callable]:
    if isinstance(model, base.Classifier):
        yield check_benchmark_learn_one
        yield check_benchmark_predict_one
        yield check_benchmark_learn_many
        yield check_benchmark_predict_many
        if getattr(model, "is_feature_incremental", False) and getattr(
            model, "is_class_incremental", False
        ):
            yield check_benchmark_incremental_expansion
    elif isinstance(model, base.Regressor):
        yield check_benchmark_learn_one
        yield check_benchmark_predict_one
        yield check_benchmark_learn_many
        yield check_benchmark_predict_many
    elif hasattr(model, "score_one"):
        yield check_benchmark_learn_one
        yield check_benchmark_predict_one
        yield check_benchmark_learn_many
        yield check_benchmark_predict_many


class _RollingFeatureModule(torch.nn.Module):
    def __init__(
        self,
        kind: str,
        layer_type: (
            type[torch.nn.RNN] | type[torch.nn.GRU] | type[torch.nn.LSTM]
        ) = torch.nn.GRU,
        n_features: int = 2,
        bidirectional: bool = False,
        num_layers: int = 2,
    ):
        super().__init__()
        self.kind = kind
        self.encoder = layer_type(
            n_features,
            4,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.head = torch.nn.Linear(
            4 * (2 if bidirectional else 1), 2 if kind == "classifier" else 1
        )

    def forward(self, x):
        self.last_input = x.detach().clone()
        output, _ = self.encoder(x)
        if self.kind == "autoencoder":
            return self.head(output).expand_as(x)
        return self.head(output[-1])


def _prepare_rolling_feature_check(
    model,
    layer_type=torch.nn.GRU,
    n_features=2,
    is_feature_incremental=True,
    append_predict=False,
    bidirectional=False,
    num_layers=2,
):
    from deep_river.base import RollingDeepEstimator

    estimator = copy.deepcopy(model)
    kind = (
        "classifier"
        if isinstance(model, base.Classifier)
        else "regressor" if model._supervised else "autoencoder"
    )
    RollingDeepEstimator.__init__(
        estimator,
        module=_RollingFeatureModule(
            kind,
            layer_type,
            n_features,
            bidirectional=bidirectional,
            num_layers=num_layers,
        ),
        loss_fn="mse",
        optimizer_fn="sgd",
        device=model.device,
        seed=model.seed,
        is_feature_incremental=is_feature_incremental,
        window_size=3,
        append_predict=append_predict,
    )
    if isinstance(estimator, base.Classifier):
        estimator.observed_classes.clear()
        estimator.is_class_incremental = False
        estimator.output_is_logit = True
    return estimator


def _learn_for_rolling_feature_check(model, x, many=False):
    if many:
        _learn_many_for_benchmark(model, pd.DataFrame([x]), pd.Series([1]))
    else:
        _learn_one_for_benchmark(model, x, 1)


def _predict_for_rolling_feature_check(model, x, many=False):
    if not model._supervised:
        result = model.score_many(pd.DataFrame([x]))[0] if many else model.score_one(x)
    elif isinstance(model, base.Classifier):
        result = (
            model.predict_proba_many(pd.DataFrame([x])).iloc[0].to_dict()
            if many
            else model.predict_proba_one(x)
        )
        assert all(np.isfinite(value) for value in result.values())
        return
    else:
        result = (
            model.predict_many(pd.DataFrame([x])).iloc[0]
            if many
            else model.predict_one(x)
        )
    assert np.isfinite(result)


def _rolling_feature_rows(model, records):
    return [
        [record.get(feature, 0) for feature in model.observed_features]
        for record in records[-model.window_size :]
    ]


def check_rolling_feature_growth_learn_one(model, layer_type, initial_rows):
    model = _prepare_rolling_feature_check(model, layer_type)
    records = [{"b": float(i + 1), "d": float(i + 2)} for i in range(initial_rows)]
    for record in records:
        _learn_for_rolling_feature_check(model, record)

    for record in [{"e": 5.0, "a": 1.0, "c": 3.0}, {"f": 6.0}, {"b": 7.0}]:
        records.append(record)
        _learn_for_rolling_feature_check(model, record)
        expected = _rolling_feature_rows(model, records)
        assert list(model._x_window) == expected
        assert model._x_window.maxlen == 3
        assert model.module.last_input[:, 0, :].tolist() == expected
        assert model.module.encoder.input_size == len(model.observed_features)
        for parameter in model.module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def check_rolling_feature_growth_prediction(model, append_predict, many):
    model = _prepare_rolling_feature_check(model, append_predict=append_predict)
    records = [{"b": float(i + 1), "d": float(i + 2)} for i in range(3)]
    for record in records:
        _learn_for_rolling_feature_check(model, record)

    for record in [{"a": 1.0, "c": 3.0, "e": 5.0}, {"f": 6.0}]:
        _predict_for_rolling_feature_check(model, record, many=many)
        assert model.module.last_input[:, 0, :].tolist() == _rolling_feature_rows(
            model, records + [record]
        )
        if append_predict:
            records.append(record)
        assert list(model._x_window) == _rolling_feature_rows(model, records)
        assert model._x_window.maxlen == 3

    _learn_for_rolling_feature_check(model, {"b": 8.0})
    records.append({"b": 8.0})
    assert list(model._x_window) == _rolling_feature_rows(model, records)


def check_rolling_feature_growth_learn_many(model):
    model = _prepare_rolling_feature_check(model)
    records = [{"b": 1.0, "d": 2.0}, {"d": 4.0, "b": 3.0}]
    for record in records:
        _learn_for_rolling_feature_check(model, record)

    for record in [{"e": 5.0, "a": 1.0, "c": 3.0}, {"f": 6.0}, {"b": 7.0}]:
        records.append(record)
        _learn_for_rolling_feature_check(model, record, many=True)
        assert list(model._x_window) == _rolling_feature_rows(model, records)
        assert model.module.last_input[:, 0, :].tolist() == _rolling_feature_rows(
            model, records
        )
        assert model._x_window.maxlen == 3
        _predict_for_rolling_feature_check(model, record, many=True)


def check_rolling_feature_discovery_fixed_input(model):
    model = _prepare_rolling_feature_check(
        model, n_features=5, is_feature_incremental=False
    )
    _learn_for_rolling_feature_check(model, {"b": 2.0, "d": 4.0})
    _learn_for_rolling_feature_check(model, {"a": 1.0, "c": 3.0})
    assert list(model._x_window) == [[2.0, 4.0, 0, 0], [0, 0, 1.0, 3.0]]
    assert model.module.last_input[:, 0, :].tolist() == [
        [2.0, 4.0, 0, 0, 0],
        [0, 0, 1.0, 3.0, 0],
    ]
    assert model.module.encoder.input_size == 5


def check_rolling_feature_update_preserves_window(model):
    model = _prepare_rolling_feature_check(model)
    window = model._x_window
    assert model._update_observed_features({"b": 2.0, "d": 4.0})
    assert list(window) == []
    window.append([2.0, 4.0])
    assert not model._update_observed_features({"d": 9.0, "b": 8.0})
    assert not model._update_observed_features({})
    assert list(window) == [[2.0, 4.0]]
    assert model._update_observed_features(pd.DataFrame(columns=["e", "a", "c"]))
    assert model._x_window is window
    assert list(window) == [[2.0, 4.0, 0, 0, 0]]
    assert window.maxlen == 3


def check_rolling_feature_growth_after_restore(model, use_pickle):
    model = _prepare_rolling_feature_check(model)
    _learn_for_rolling_feature_check(model, {"b": 2.0, "d": 4.0})
    restored = pickle.loads(pickle.dumps(model)) if use_pickle else copy.deepcopy(model)
    _learn_for_rolling_feature_check(restored, {"a": 1.0, "c": 3.0})
    assert list(restored._x_window) == [[2.0, 4.0, 0, 0], [0, 0, 1.0, 3.0]]
    assert list(model._x_window) == [[2.0, 4.0]]


def _incremental_autoencoder(model, one_layer=False):
    output_size = 2 if one_layer else 3
    layers = [torch.nn.Linear(2, output_size)]
    if not one_layer:
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(output_size, 2))
    kwargs = {"is_feature_incremental": True}
    if _is_rolling(model):
        kwargs["window_size"] = 2
    return type(model)(torch.nn.Sequential(*layers), **kwargs)


def check_autoencoder_reconstruction_expansion(model, many):
    estimator = _incremental_autoencoder(model)
    for width in (2, 3, 5):
        record = {str(index): float(index) / 10 for index in range(width)}
        estimator._update_observed_features(record)
        assert estimator.input_layer.in_features == width
        assert estimator.output_layer.out_features == width
        if many:
            X = pd.DataFrame([record, record])
            estimator.learn_many(X)
            assert np.isfinite(estimator.score_many(X)).all()
        else:
            estimator.learn_one(record)
            assert np.isfinite(estimator.score_one(record))
    restored = _round_trip(estimator)
    record["z"] = 1.0
    restored.learn_one(record)
    assert restored.input_layer.in_features == 6
    assert restored.output_layer.out_features == 6


def check_autoencoder_expansion_preserves_reconstruction(model):
    estimator = _incremental_autoencoder(model)
    estimator.learn_one({"a": 1.0, "b": 2.0})
    weight = estimator.output_layer.weight.detach().clone()
    bias = estimator.output_layer.bias.detach().clone()
    estimator._update_observed_features({"a": 1.0, "b": 2.0, "c": 3.0})
    torch.testing.assert_close(estimator.output_layer.weight[:2], weight)
    torch.testing.assert_close(estimator.output_layer.bias[:2], bias)


def check_autoencoder_one_layer_expansion(model):
    if _is_rolling(model):
        return
    estimator = _incremental_autoencoder(model, one_layer=True)
    estimator.learn_one({"a": 1.0, "b": 2.0})
    estimator.learn_one({"a": 1.0, "b": 2.0, "c": 3.0})
    assert estimator.module[0].weight.shape == (3, 3)
    assert np.isfinite(estimator.score_one({"a": 1.0, "b": 2.0, "c": 3.0}))


def check_optimizer_expansion_state(model, optimizer, kwargs, output):
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    estimator = type(model)(module, "mse", "sgd")
    estimator.optimizer = optimizer(module.parameters(), lr=0.01, **kwargs)
    estimator.learn_one({"a": 1.0, "b": 2.0}, 3.0)
    original_optimizer = estimator.optimizer
    layer = estimator.output_layer if output else estimator.input_layer
    old_weight = layer.weight
    old_state = copy.deepcopy(estimator.optimizer.state[old_weight])
    untouched = module[0].weight if output else module[1].weight
    untouched_state = copy.deepcopy(estimator.optimizer.state[untouched])
    estimator._expand_layer(layer, 3, output=output)
    assert estimator.optimizer is original_optimizer
    assert old_weight not in estimator.optimizer.state
    _assert_learning_state(estimator.optimizer.state[untouched], untouched_state)
    for name, value in old_state.items():
        actual = estimator.optimizer.state[layer.weight][name]
        if isinstance(value, torch.Tensor) and value.shape == old_weight.shape:
            indices = tuple(slice(0, size) for size in value.shape)
            torch.testing.assert_close(actual[indices], value)
            extra = actual[old_weight.shape[0] :] if output else actual[:, 2:]
            assert torch.count_nonzero(extra) == 0
        else:
            torch.testing.assert_close(actual, value)
    before = layer.weight.detach().clone()
    if output:
        estimator._learn(torch.tensor([[1.0, 2.0]]), torch.zeros(1, 3))
    else:
        estimator.learn_one({"a": 1.0, "b": 2.0, "c": 3.0}, 4.0)
    assert not torch.equal(before, layer.weight)


def check_optimizer_expansion_parameter_groups(model):
    module = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Linear(2, 1))
    estimator = type(model)(module, "mse", "sgd")
    estimator.optimizer = torch.optim.Adam(
        [
            {"params": module[0].parameters(), "lr": 0.02, "betas": (0.8, 0.9)},
            {"params": module[1].parameters(), "lr": 0.01, "weight_decay": 0.1},
        ]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(estimator.optimizer, step_size=1)
    estimator.learn_one({"a": 1.0}, 1.0)
    scheduler.step()
    groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in estimator.optimizer.param_groups
    ]
    estimator._expand_layer(module[0], 2, output=False)
    assert scheduler.optimizer is estimator.optimizer
    assert groups == [
        {key: value for key, value in group.items() if key != "params"}
        for group in estimator.optimizer.param_groups
    ]
    parameters = [
        parameter
        for group in estimator.optimizer.param_groups
        for parameter in group["params"]
    ]
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in module.parameters()
    }


def check_optimizer_repeated_expansion(model):
    estimator = type(model)(
        torch.nn.Sequential(torch.nn.Linear(1, 1)),
        "mse",
        "adam",
        is_feature_incremental=True,
    )
    for size in (2, 3, 5):
        estimator.learn_one({str(index): float(index) for index in range(size)}, 1.0)
        assert estimator.input_layer.weight.shape == (1, size)
        assert estimator.optimizer.state[estimator.input_layer.weight][
            "exp_avg"
        ].shape == (1, size)


def check_ordered_set_indices(model):
    import pytest

    from deep_river.utils.ordered_set import OrderedSet

    values = OrderedSet(["z", 1])
    values.update(["a", 2, "z"])
    assert list(values)[:2] == ["z", 1]
    assert values.index("z") == 0
    assert values.index(1) == 1
    for restored in (copy.deepcopy(values), pickle.loads(pickle.dumps(values))):
        assert list(restored) == list(values)
        restored.add("new")
        assert "new" not in values
    values.discard("z")
    assert values.index(1) == 0
    with pytest.raises(ValueError):
        values.index("absent")


def check_dynamic_zero_feature_mapping(model, many):
    module = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        module[0].weight.copy_(torch.tensor([[2.0, 0.0]]))
    estimator = type(model)(module, "mse", "sgd", lr=0)
    assert estimator.predict_one({"z": 3.0}) == 6.0
    if many:
        result = estimator.predict_many(pd.DataFrame({"a": [0.0], "z": [3.0]})).iloc[0]
    else:
        result = estimator.predict_one({"a": 0.0, "z": 3.0})
    assert result == 6.0
    assert list(estimator.observed_features) == ["z", "a"]


def check_dynamic_feature_order(model):
    left = type(model)(torch.nn.Sequential(torch.nn.Linear(2, 1)), "mse", "sgd")
    right = copy.deepcopy(left)
    left.predict_one({"z": 2.0, "a": 1.0})
    right.predict_one({"a": 1.0, "z": 2.0})
    assert list(left.observed_features) == list(right.observed_features)


def check_dynamic_class_mapping(model, first, second):
    module = torch.nn.Sequential(torch.nn.Linear(1, 2))
    with torch.no_grad():
        module[0].weight.zero_()
        module[0].bias.copy_(torch.tensor([3.0, -3.0]))
    estimator = type(model)(module, "cross_entropy", "sgd")
    estimator._update_observed_targets(first)
    original_index = estimator.observed_classes.index(first)
    expected = estimator.predict_proba_one({"x": 0.0})[first]
    estimator._update_observed_targets(second)
    assert estimator.observed_classes.index(first) == original_index
    assert estimator.predict_proba_one({"x": 0.0})[first] == expected


def check_boolean_class_mapping(model, first):
    from deep_river.utils.ordered_set import OrderedSet
    from deep_river.utils.tensor_conversion import labels2onehot

    classes = OrderedSet([first])
    classes.add(not first)
    assert labels2onehot(True, classes, n_classes=1).item() == 1.0
    assert labels2onehot(False, classes, n_classes=1).item() == 0.0


def check_dynamic_target_mapping(model):
    module = torch.nn.Sequential(torch.nn.Linear(1, 2))
    with torch.no_grad():
        module[0].weight.zero_()
        module[0].bias.copy_(torch.tensor([3.0, 7.0]))
    estimator = type(model)(module, "mse", "sgd")
    estimator._update_observed_targets({"z": 0.0})
    assert estimator.predict_one({"x": 0.0})["z"] == 3.0
    estimator._update_observed_targets({"a": 0.0})
    assert estimator.predict_one({"x": 0.0}) == {"z": 3.0, "a": 7.0}
    restored = _round_trip(estimator)
    assert list(restored.observed_targets) == ["z", "a"]
    assert restored.predict_one({"x": 0.0}) == estimator.predict_one({"x": 0.0})


def check_recurrent_expansion(model, layer_type, bidirectional, num_layers):
    estimator = _prepare_rolling_feature_check(
        model,
        layer_type=layer_type,
        bidirectional=bidirectional,
        num_layers=num_layers,
    )
    estimator.learn_one({"a": 1.0, "b": 2.0}, 1.0)
    recurrent = estimator.module.encoder
    old = {name: value.detach().clone() for name, value in recurrent.named_parameters()}
    estimator._update_observed_features({"a": 1.0, "b": 2.0, "c": 3.0})
    assert recurrent.input_size == 3
    for name, value in recurrent.named_parameters():
        if name.startswith("weight_ih_l0"):
            assert value.shape[1] == 3
            torch.testing.assert_close(value[:, :2], old[name])
        else:
            torch.testing.assert_close(value, old[name])
    estimator.learn_one({"a": 1.0, "b": 2.0, "c": 3.0}, 2.0)
    for parameter in recurrent.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    restored = _round_trip(estimator)
    x = {"a": 1.0, "b": 2.0, "c": 3.0}
    _assert_persisted_value(_prediction(estimator, x), _prediction(restored, x))
    restored.learn_one({**x, "d": 4.0}, 1.0)
    assert restored.module.encoder.weight_ih_l0.shape[1] == 4
    if bidirectional:
        assert restored.module.encoder.weight_ih_l0_reverse.shape[1] == 4


def _backend_frame(data, backend, index=None):
    if backend == "pandas":
        return pd.DataFrame(data, index=index)
    if backend == "polars":
        import polars as pl

        return pl.DataFrame(data)
    if backend == "pyarrow":
        import pyarrow as pa

        return pa.table(data)
    raise ValueError(backend)


def _backend_series(values, backend, index=None):
    if backend == "pandas":
        return pd.Series(values, index=index, name="target")
    if backend == "polars":
        import polars as pl

        return pl.Series("target", values)
    if backend == "pyarrow":
        import pyarrow as pa

        return pa.chunked_array([values])
    raise ValueError(backend)


def _backend_values(value):
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.to_numpy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas().to_numpy()
    return value.to_numpy()


def check_dataframe_backend(model, backend):
    reference = copy.deepcopy(model)
    candidate = copy.deepcopy(model)
    n_features = model._get_input_size()
    columns = [f"f{index}" for index in range(n_features)]
    train_data = {
        column: [float(index + 1), np.nan, float(index + 3)]
        for index, column in enumerate(columns)
    }
    train_index = pd.Index(["i0", "i1", "i2"])
    targets = (
        [False, True, False] if isinstance(model, base.Classifier) else [1.0, 0.0, 1.0]
    )
    with torch.random.fork_rng():
        state = torch.get_rng_state()
        reference.learn_many(
            _backend_frame(train_data, "pandas", train_index),
            _backend_series(targets, "pandas", train_index),
        )
        torch.set_rng_state(state)
        candidate.learn_many(
            _backend_frame(train_data, backend, train_index),
            _backend_series(targets, backend, train_index),
        )
    predict_data = {
        column: [float(index), float(index + 1)]
        for index, column in reversed(list(enumerate(columns)))
    }
    predict_index = pd.Index(["p0", "p1"])
    expected_X = _backend_frame(predict_data, "pandas", predict_index)
    actual_X = _backend_frame(predict_data, backend, predict_index)
    if isinstance(model, base.Classifier):
        expected = reference.predict_proba_many(expected_X)
        actual = candidate.predict_proba_many(actual_X)
    else:
        expected = reference.predict_many(expected_X)
        actual = candidate.predict_many(actual_X)
    assert np.allclose(_backend_values(actual), _backend_values(expected))
    if backend == "pandas":
        assert actual.index.equals(predict_index)
    elif backend == "polars":
        import polars as pl

        assert isinstance(actual, (pl.Series, pl.DataFrame))
    else:
        import pyarrow as pa

        assert isinstance(actual, (pa.ChunkedArray, pa.Table))


class _ValueDetector(AnomalyDetector):
    def __init__(self):
        self.learned = 0

    def score_one(self, x):
        return x["value"]

    def learn_one(self, x):
        self.learned += 1


def _anomaly_scaler_types():
    from deep_river.anomaly import (
        AnomalyMeanScaler,
        AnomalyMinMaxScaler,
        AnomalyStandardScaler,
    )

    class StandardScaler(AnomalyStandardScaler):
        def score_many(self, X):
            return np.array([self.score_one(x) for x in X.to_dict(orient="records")])

    class MeanScaler(AnomalyMeanScaler):
        def score_many(self, X):
            return np.array([self.score_one(x) for x in X.to_dict(orient="records")])

    class MinMaxScaler(AnomalyMinMaxScaler):
        def score_many(self, X):
            return np.array([self.score_one(x) for x in X.to_dict(orient="records")])

    return StandardScaler, MeanScaler, MinMaxScaler


def check_anomaly_scaler_finite_scores(model, scaler_type, rolling):
    estimator = scaler_type(_ValueDetector(), rolling=rolling)
    for value in (0.0, 0.0, 3.0, 3.0):
        x = {"value": value}
        assert np.isfinite(estimator.score_one(x))
        estimator.learn_one(x)
        assert np.isfinite(estimator.score_one(x))


def check_anomaly_scaler_uses_prior_statistics(model, scaler_type, expected, rolling):
    import pytest

    estimator = scaler_type(_ValueDetector(), rolling=rolling)
    for value in (1.0, 2.0, 3.0):
        estimator.learn_one({"value": value})
    state = copy.deepcopy(estimator.__dict__)
    assert estimator.score_one({"value": 4.0}) == pytest.approx(expected)
    assert estimator.score_one({"value": 4.0}) == pytest.approx(expected)
    assert estimator.anomaly_detector.learned == 3
    _assert_persisted_value(estimator.__dict__, state)


def check_anomaly_scaler_rolling_statistics(model, scaler_type, expected):
    import pytest

    estimator = scaler_type(_ValueDetector(), window_size=2)
    for value in (1.0, 2.0, 3.0):
        estimator.learn_one({"value": value})
    assert estimator.score_one({"value": 4.0}) == pytest.approx(expected)


def check_anomaly_standard_scaler_options(model, scaler_type):
    import pytest

    centered = scaler_type(_ValueDetector(), with_std=False)
    centered.learn_one({"value": 2.0})
    assert centered.score_one({"value": 5.0}) == 3.0
    stable = scaler_type(_ValueDetector(), rolling=False)
    for value in (1e12, 1e12 + 1, 1e12 + 2):
        stable.learn_one({"value": value})
    assert stable.score_one({"value": 1e12 + 3}) == pytest.approx(2 / np.sqrt(2 / 3))


def check_anomaly_scaler_batch_learning(model):
    X = pd.DataFrame({"x": [0.1, 0.4, 0.8], "y": [0.8, 0.4, 0.1]}, index=[5, 7, 9])
    original = X.copy(deep=True)
    online = copy.deepcopy(model)
    model.learn_many(X)
    for x in X.to_dict(orient="records"):
        online.learn_one(x)
    state = pickle.dumps(model)
    actual = model.score_many(X)
    np.testing.assert_allclose(
        actual, [online.score_one(x) for x in X.to_dict(orient="records")]
    )
    assert actual.shape == (3,)
    assert np.isfinite(actual).all()
    assert pickle.dumps(model) == state
    assert model.score_many(X.iloc[:0]).shape == (0,)
    model.learn_many(X.iloc[:0])
    assert pickle.dumps(model) == state
    pd.testing.assert_frame_equal(X, original)


def _probability_weighted_model(model, optimizer_fn="sgd"):
    module = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        module[0].weight.zero_()
    return type(model)(module, optimizer_fn=optimizer_fn, lr=0.01)


def check_probability_weighted_outlier_updates(model, many, optimizer_fn):
    estimator = _probability_weighted_model(model, optimizer_fn)
    estimator.learn_one({"a": 0.1})
    parameters = copy.deepcopy(estimator.module.state_dict())
    state = copy.deepcopy(estimator.optimizer.state_dict())
    before = estimator.score_one({"a": 10.0})
    if many:
        estimator.learn_many(pd.DataFrame({"a": [10.0, 10.0]}))
    else:
        estimator.learn_one({"a": 10.0})
    _assert_learning_state(estimator.module.state_dict(), parameters)
    _assert_learning_state(estimator.optimizer.state_dict(), state)
    assert estimator.score_one({"a": 10.0}) == before
    assert estimator.rolling_mean.get() > 1


def check_probability_weighted_mixed_batch(model):
    import pytest

    estimator = _probability_weighted_model(model)
    X = pd.DataFrame({"a": [0.1, 10.0]})
    probability = 0.5 * (1 + math.erf(0.01 / np.sqrt(2)))
    expected_gradient = -0.01 * (0.9 - probability) / 0.9
    estimator.learn_many(X)
    assert estimator.module[0].weight.item() == pytest.approx(-0.01 * expected_gradient)
    assert estimator.rolling_mean.get() == pytest.approx((0.01 + 100) / 2)


def check_probability_weighted_configuration(model):
    import pytest

    for threshold in (0.0, -0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="skip_threshold"):
            type(model)(
                torch.nn.Sequential(torch.nn.Linear(1, 1)),
                skip_threshold=threshold,
            )
    for window_size in (0, -1):
        with pytest.raises(ValueError, match="window_size"):
            type(model)(
                torch.nn.Sequential(torch.nn.Linear(1, 1)),
                window_size=window_size,
            )


def check_probability_weighted_gradient_clipping(model):
    module = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        module[0].weight.zero_()
    estimator = type(model)(module, gradient_clip_value=0.001)
    estimator.learn_one({"a": 0.1})
    assert 0 < module[0].weight.grad.norm().item() <= 0.001001


def yield_deep_checks(model) -> typing.Iterator[typing.Callable]:
    """Generates unit tests for a given model.

    Parameters
    ----------
    model

    """

    from deep_river.anomaly import Autoencoder
    from deep_river.anomaly.scaler import AnomalyScaler
    from deep_river.base import DeepEstimator

    if isinstance(model, AnomalyScaler):
        yield check_anomaly_scaler_batch_learning
        return

    if type(model) is Autoencoder:
        expected_prior = (2 / np.sqrt(2 / 3), 2.0, 1.5)
        expected_rolling = (3.0, 1.6, 2.0)
        for index, scaler_type in enumerate(_anomaly_scaler_types()):
            for rolling in (False, True):
                yield _wrapped_partial(
                    check_anomaly_scaler_finite_scores,
                    scaler_type=scaler_type,
                    rolling=rolling,
                )
                yield _wrapped_partial(
                    check_anomaly_scaler_uses_prior_statistics,
                    scaler_type=scaler_type,
                    expected=expected_prior[index],
                    rolling=rolling,
                )
            yield _wrapped_partial(
                check_anomaly_scaler_rolling_statistics,
                scaler_type=scaler_type,
                expected=expected_rolling[index],
            )
            if index == 0:
                yield _wrapped_partial(
                    check_anomaly_standard_scaler_options,
                    scaler_type=scaler_type,
                )

    if not isinstance(model, DeepEstimator):
        return

    dataset_checks = [check_deep_learn_one, check_model_persistence]

    from deep_river.anomaly import (
        Autoencoder,
        ProbabilityWeightedAutoencoder,
        RollingAutoencoder,
    )

    if isinstance(
        model, (Autoencoder, ProbabilityWeightedAutoencoder, RollingAutoencoder)
    ):
        for many in (False, True):
            yield _wrapped_partial(
                check_autoencoder_reconstruction_expansion, many=many
            )
        yield check_autoencoder_expansion_preserves_reconstruction
        yield check_autoencoder_one_layer_expansion

    from deep_river.classification import Classifier
    from deep_river.regression import MultiTargetRegressor, Regressor, RollingRegressor

    if type(model) is Regressor:
        for optimizer, kwargs in (
            (torch.optim.Adam, {"amsgrad": True}),
            (torch.optim.AdamW, {}),
            (torch.optim.SGD, {"momentum": 0.9}),
            (torch.optim.RMSprop, {"momentum": 0.9, "centered": True}),
        ):
            for output in (False, True):
                yield _wrapped_partial(
                    check_optimizer_expansion_state,
                    optimizer=optimizer,
                    kwargs=kwargs,
                    output=output,
                )
        yield check_optimizer_expansion_parameter_groups
        yield check_optimizer_repeated_expansion

    if type(model) is Regressor:
        yield check_ordered_set_indices
        for many in (False, True):
            yield _wrapped_partial(check_dynamic_zero_feature_mapping, many=many)
        yield check_dynamic_feature_order

    if (
        type(model) is Classifier
        and getattr(model, "is_class_incremental", False)
        and not getattr(model, "is_feature_incremental", False)
    ):
        for first, second in (("z", "a"), (True, False), (False, True)):
            yield _wrapped_partial(
                check_dynamic_class_mapping, first=first, second=second
            )
        for first in (True, False):
            yield _wrapped_partial(check_boolean_class_mapping, first=first)

    if type(model) is MultiTargetRegressor:
        yield check_dynamic_target_mapping

    if type(model) is RollingRegressor:
        for layer_type in (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM):
            for bidirectional in (False, True):
                for num_layers in (1, 2):
                    yield _wrapped_partial(
                        check_recurrent_expansion,
                        layer_type=layer_type,
                        bidirectional=bidirectional,
                        num_layers=num_layers,
                    )

    if isinstance(model, ProbabilityWeightedAutoencoder):
        for many in (False, True):
            for optimizer_fn in ("sgd", "adam"):
                yield _wrapped_partial(
                    check_probability_weighted_outlier_updates,
                    many=many,
                    optimizer_fn=optimizer_fn,
                )
        yield check_probability_weighted_mixed_batch
        yield check_probability_weighted_configuration
        yield check_probability_weighted_gradient_clipping

    # Non-dataset checks (run once per model)
    yield check_dict2tensor
    yield check_model_persistence_untrained
    yield check_model_persistence_rejects_other_type
    yield check_model_persistence_legacy_format
    yield check_seed_reproducibility
    yield check_predict_many_output_length

    if not isinstance(model, Forecaster):
        yield check_bounded_tensor_memory
        if callable(getattr(model, "learn_many", None)):
            yield check_batch_size_one_learning

    if (
        isinstance(model, base.Classifier)
        and getattr(model, "is_feature_incremental", False)
        and getattr(model, "is_class_incremental", False)
    ):
        yield check_model_persistence_after_incremental_expansion

    if _is_rolling(model):
        if callable(getattr(model, "learn_many", None)):
            for batch_size in (0, 1, 2, 3, 7):
                for warmup in (0, 2):
                    yield _wrapped_partial(
                        check_rolling_batch_learning,
                        batch_size=batch_size,
                        warmup=warmup,
                    )
            if model._supervised:
                yield check_rolling_batch_rejects_misaligned_targets
        yield check_rolling_memory_after_feature_growth
        for layer_type in (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM):
            for initial_rows in (1, 3):
                yield _wrapped_partial(
                    check_rolling_feature_growth_learn_one,
                    layer_type=layer_type,
                    initial_rows=initial_rows,
                )
        for append_predict in (False, True):
            for many in (False, True):
                yield _wrapped_partial(
                    check_rolling_feature_growth_prediction,
                    append_predict=append_predict,
                    many=many,
                )
        yield check_rolling_feature_growth_learn_many
        yield check_rolling_feature_update_preserves_window
        for use_pickle in (False, True):
            yield _wrapped_partial(
                check_rolling_feature_growth_after_restore, use_pickle=use_pickle
            )
        if model._supervised:
            yield check_rolling_feature_discovery_fixed_input

    from deep_river.classification import Classifier
    from deep_river.regression import Regressor

    if type(model) in (Classifier, Regressor):
        for backend in ("pandas", "polars", "pyarrow"):
            if find_spec(backend) is not None:
                yield _wrapped_partial(check_dataframe_backend, backend=backend)

    # Classifier checks
    if isinstance(model, base.Classifier) and not isinstance(
        model, base.MultiLabelClassifier
    ):
        yield check_dict2tensor

        if not model._multiclass:
            yield check_dict2tensor

    for dataset_check in dataset_checks:
        for dataset in _yield_datasets(model):
            yield _wrapped_partial(dataset_check, dataset=dataset)


def check_estimator(model):
    """Check if a model adheres to `river`'s conventions.
    This will run a series of unit tests. The nature of the unit tests
    depends on the type of model.
    Parameters
    ----------
    model
    """
    for check in yield_checks(model):
        if check.__name__ in model._unit_test_skips():
            continue
        check(copy.deepcopy(model))  # todo change to clone

    for check in yield_deep_checks(model):
        if check.__name__ in model._unit_test_skips():
            continue
        check(copy.deepcopy(model))  # todo change to clone
