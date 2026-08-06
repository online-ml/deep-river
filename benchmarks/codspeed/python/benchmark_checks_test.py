import copy
import importlib
import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from river import base
from river.base import Estimator

import deep_river
from deep_river.base import DeepEstimator, RollingDeepEstimator


N_ONLINE = 12
N_BATCH = 16


def iter_estimators():
    def is_estimator(obj):
        return inspect.isclass(obj) and issubclass(obj, Estimator)

    for submodule in ("anomaly", "classification", "regression"):
        yield from (
            obj
            for _, obj in inspect.getmembers(
                importlib.import_module(f"{deep_river.__name__}.{submodule}"),
                is_estimator,
            )
            if not inspect.isabstract(obj)
        )


def benchmark_estimators():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    for estimator_cls in iter_estimators():
        for params in estimator_cls._unit_test_params():
            estimator = estimator_cls(**params)
            if isinstance(estimator, DeepEstimator):
                yield estimator


def n_features(estimator):
    return estimator._get_input_size()


def frame(estimator, n_samples=N_BATCH):
    values = np.arange(n_samples * n_features(estimator), dtype=np.float32).reshape(
        n_samples, n_features(estimator)
    )
    return pd.DataFrame(
        ((values % 17) / 17).astype(np.float32),
        columns=[f"f{i}" for i in range(n_features(estimator))],
    )


def rows(estimator, n_samples=N_ONLINE):
    return frame(estimator, n_samples).to_dict(orient="records")


def targets(estimator, n_samples):
    if isinstance(estimator, base.Classifier):
        return pd.Series([i % 2 for i in range(n_samples)])
    if isinstance(estimator, base.MultiTargetRegressor):
        values = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
        return pd.DataFrame({f"y{i}": values + i for i in range(3)})
    if isinstance(estimator, base.Regressor):
        return pd.Series(np.linspace(0.0, 1.0, n_samples, dtype=np.float32))
    return None


def target_rows(y):
    if isinstance(y, pd.DataFrame):
        return y.to_dict(orient="records")
    return y


def learn_one(estimator, x, y=None):
    if y is None:
        estimator.learn_one(x)
    else:
        estimator.learn_one(x, y)


def learn_many(estimator, X, y=None):
    if y is None:
        estimator.learn_many(X)
    else:
        estimator.learn_many(X, y)


def fit_one(estimator):
    xs = rows(estimator)
    y = targets(estimator, len(xs))
    if y is None:
        for x in xs:
            learn_one(estimator, x)
    else:
        for x, target in zip(xs, target_rows(y)):
            learn_one(estimator, x, target)
    return estimator


def prefill_rolling(estimator):
    if not isinstance(estimator, RollingDeepEstimator):
        return estimator
    xs = rows(estimator, estimator.window_size - 1)
    y = targets(estimator, len(xs))
    if y is None:
        for x in xs:
            learn_one(estimator, x)
    else:
        for x, target in zip(xs, target_rows(y)):
            learn_one(estimator, x, target)
    return estimator


def batch_data(estimator):
    n_samples = 1 if isinstance(estimator, RollingDeepEstimator) else N_BATCH
    X = frame(estimator, n_samples)
    return X, targets(estimator, len(X))


def check_learn_one(estimator, benchmark):
    xs = rows(estimator)
    y = targets(estimator, len(xs))

    def run():
        model = copy.deepcopy(estimator)
        if y is None:
            for x in xs:
                learn_one(model, x)
        else:
            for x, target in zip(xs, target_rows(y)):
                learn_one(model, x, target)
        return len(model.observed_features)

    assert benchmark(run) == n_features(estimator)


def check_predict_one(estimator, benchmark):
    model = fit_one(copy.deepcopy(estimator))
    xs = rows(model)

    def run():
        result = None
        for x in xs:
            if isinstance(model, base.Classifier):
                result = model.predict_proba_one(x)
            elif isinstance(model, base.MultiTargetRegressor):
                result = model.predict_one(x)
            elif isinstance(model, base.Regressor):
                result = model.predict_one(x)
            else:
                result = model.score_one(x)
        return result

    assert benchmark(run) is not None


def check_learn_many(estimator, benchmark):
    model = prefill_rolling(copy.deepcopy(estimator))
    X, y = batch_data(model)

    def run():
        fitted = copy.deepcopy(model)
        learn_many(fitted, X, y)
        return len(fitted.observed_features)

    assert benchmark(run) == n_features(estimator)


def check_predict_many(estimator, benchmark):
    model = fit_one(prefill_rolling(copy.deepcopy(estimator)))
    X, _ = batch_data(model)

    def run():
        if isinstance(model, base.Classifier):
            return len(model.predict_proba_many(X))
        if isinstance(model, base.Regressor) or isinstance(
            model, base.MultiTargetRegressor
        ):
            return len(model.predict_many(X))
        return len(model.score_many(X))

    assert benchmark(run) == len(X)


def checks(estimator):
    yield check_learn_one
    yield check_predict_one
    if type(estimator).__name__ != "ProbabilityWeightedAutoencoder":
        yield check_learn_many
    yield check_predict_many


def benchmark_group(estimator):
    return type(estimator).__module__.split(".")[1]


@pytest.mark.parametrize(
    "estimator, check",
    [
        pytest.param(
            estimator,
            check,
            id=f"{benchmark_group(estimator)}:{type(estimator).__name__}:{check.__name__}",
            marks=pytest.mark.benchmark(group=benchmark_group(estimator)),
        )
        for estimator in benchmark_estimators()
        for check in checks(estimator)
    ],
)
def test_benchmark_check(estimator, check, benchmark):
    check(estimator, benchmark)
