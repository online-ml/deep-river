import copy
import pickle

import numpy as np
import pandas as pd
import pytest
from river.anomaly import HalfSpaceTrees

from deep_river import anomaly, utils


@pytest.mark.parametrize(
    "kind",
    [
        anomaly.AnomalyStandardScaler,
        anomaly.AnomalyMeanScaler,
        anomaly.AnomalyMinMaxScaler,
    ],
)
def test_scalers_support_batch_scoring_and_learning(kind):
    model = kind(HalfSpaceTrees(seed=42, window_size=2))
    frame = pd.DataFrame({"x": [0.1, 0.4, 0.8], "y": [0.8, 0.4, 0.1]}, index=[5, 7, 9])
    original = frame.copy(deep=True)
    online = copy.deepcopy(model)
    model.learn_many(frame)
    for x in frame.to_dict(orient="records"):
        online.learn_one(x)
    state = pickle.dumps(model)
    actual = model.score_many(frame)
    np.testing.assert_allclose(
        actual, [online.score_one(x) for x in frame.to_dict(orient="records")]
    )
    assert actual.shape == (3,)
    assert np.isfinite(actual).all()
    assert pickle.dumps(model) == state
    assert model.score_many(frame.iloc[:0]).shape == (0,)
    model.learn_many(frame.iloc[:0])
    assert pickle.dumps(model) == state
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "kind",
    [
        anomaly.AnomalyStandardScaler,
        anomaly.AnomalyMeanScaler,
        anomaly.AnomalyMinMaxScaler,
    ],
)
def test_anomaly_scalers_pass_estimator_checks(kind):
    model = kind(**next(kind._unit_test_params()))
    utils.check_estimator(model)


def test_concrete_scalers_are_discovered_for_estimator_checks():
    names = {
        type(model).__name__
        for model in utils.estimator_checks.iter_estimators_that_can_be_tested()
    }
    assert {
        "AnomalyStandardScaler",
        "AnomalyMeanScaler",
        "AnomalyMinMaxScaler",
    } <= names
