"""Utilities for unit testing and sanity checking estimators."""

import collections
import copy
import importlib
import inspect
import pickle
import tempfile
from collections.abc import Set
from pathlib import Path

__all__ = [
    "check_estimator",
    "iter_estimators",
    "iter_estimators_that_can_be_tested",
]

import typing

import numpy as np
import pandas as pd
import pytest
import torch
from river import base
from river.base import Estimator
from river.checks import _wrapped_partial, _yield_datasets, yield_checks
from river.time_series.base import Forecaster


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
        return type(model).load(path)


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
        loaded_model = type(model).load(path)
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


def yield_deep_checks(model) -> typing.Iterator[typing.Callable]:
    """Generates unit tests for a given model.

    Parameters
    ----------
    model

    """

    dataset_checks = [check_deep_learn_one, check_model_persistence]

    # Non-dataset checks (run once per model)
    yield check_dict2tensor
    yield check_model_persistence_untrained
    yield check_model_persistence_rejects_other_type
    yield check_model_persistence_legacy_format
    yield check_predict_many_output_length

    if (
        isinstance(model, base.Classifier)
        and getattr(model, "is_feature_incremental", False)
        and getattr(model, "is_class_incremental", False)
    ):
        yield check_model_persistence_after_incremental_expansion

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
