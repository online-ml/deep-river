import pickle

import numpy as np
import pytest

from deep_river import anomaly

try:
    from river.base import AnomalyDetector
except ImportError:
    from river.anomaly.base import AnomalyDetector


class ValueDetector(AnomalyDetector):
    def __init__(self):
        self.learned = 0

    def score_one(self, x):
        return x["value"]

    def learn_one(self, x):
        self.learned += 1


class StandardScaler(anomaly.AnomalyStandardScaler):
    def score_many(self, X):
        return np.array([self.score_one(x) for x in X.to_dict(orient="records")])


class MeanScaler(anomaly.AnomalyMeanScaler):
    def score_many(self, X):
        return np.array([self.score_one(x) for x in X.to_dict(orient="records")])


class MinMaxScaler(anomaly.AnomalyMinMaxScaler):
    def score_many(self, X):
        return np.array([self.score_one(x) for x in X.to_dict(orient="records")])


@pytest.mark.parametrize("kind", [StandardScaler, MeanScaler, MinMaxScaler])
@pytest.mark.parametrize("rolling", [False, True])
def test_scaling_is_finite_for_cold_start_and_constant_scores(kind, rolling):
    model = kind(ValueDetector(), rolling=rolling)
    for value in (0.0, 0.0, 3.0, 3.0):
        x = {"value": value}
        assert np.isfinite(model.score_one(x))
        model.learn_one(x)
        assert np.isfinite(model.score_one(x))


@pytest.mark.parametrize(
    "kind,expected",
    [(StandardScaler, 2 / np.sqrt(2 / 3)), (MeanScaler, 2.0), (MinMaxScaler, 1.5)],
)
@pytest.mark.parametrize("rolling", [False, True])
def test_scaling_uses_prior_learning_statistics(kind, expected, rolling):
    model = kind(ValueDetector(), rolling=rolling)
    for value in (1.0, 2.0, 3.0):
        model.learn_one({"value": value})
    state = pickle.dumps(model)
    assert model.score_one({"value": 4.0}) == pytest.approx(expected)
    assert model.score_one({"value": 4.0}) == pytest.approx(expected)
    assert model.anomaly_detector.learned == 3
    assert pickle.dumps(model) == state


@pytest.mark.parametrize(
    "kind,expected", [(StandardScaler, 3.0), (MeanScaler, 1.6), (MinMaxScaler, 2.0)]
)
def test_rolling_statistics_evict_old_scores(kind, expected):
    model = kind(ValueDetector(), window_size=2)
    for value in (1.0, 2.0, 3.0):
        model.learn_one({"value": value})
    assert model.score_one({"value": 4.0}) == pytest.approx(expected)


def test_standard_scaling_can_only_center_scores():
    model = StandardScaler(ValueDetector(), with_std=False)
    model.learn_one({"value": 2.0})
    assert model.score_one({"value": 5.0}) == 3.0


def test_standard_scaling_is_stable_with_large_offsets():
    model = StandardScaler(ValueDetector(), rolling=False)
    for value in (1e12, 1e12 + 1, 1e12 + 2):
        model.learn_one({"value": value})
    assert model.score_one({"value": 1e12 + 3}) == pytest.approx(2 / np.sqrt(2 / 3))
