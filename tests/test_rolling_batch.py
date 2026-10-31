import copy

import pandas as pd
import pytest
import torch

from deep_river import anomaly, classification, regression
from deep_river.utils.estimator_checks import _assert_learning_state


class SequenceOutput(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.linear = torch.nn.Linear(3, outputs)

    def forward(self, x):
        return self.linear(x[-1])


@pytest.mark.parametrize(
    "estimator_type",
    [
        anomaly.RollingAutoencoder,
        classification.RollingClassifier,
        regression.RollingRegressor,
    ],
)
@pytest.mark.parametrize("batch_size", [0, 1, 4])
@pytest.mark.parametrize("warmup", [0, 2])
@pytest.mark.parametrize("optimizer_fn", ["sgd", "adam"])
def test_rolling_batch_preserves_online_updates(
    estimator_type, batch_size, warmup, optimizer_fn
):
    is_autoencoder = estimator_type is anomaly.RollingAutoencoder
    is_classifier = estimator_type is classification.RollingClassifier
    module = (
        torch.nn.Linear(3, 3)
        if is_autoencoder
        else SequenceOutput(2 if is_classifier else 1)
    )
    model = estimator_type(module=module, window_size=3, optimizer_fn=optimizer_fn)
    X = pd.DataFrame(
        {
            "z": pd.Series([1, 2, 3, 4], dtype="Int64"),
            "a": [0.5, 1.5, 2.5, 3.5],
            "b": [True, False, True, False],
        },
    ).iloc[:batch_size]
    X.index = [10 + index * 2 for index in range(batch_size)]
    y = pd.Series(
        (
            ["positive", "negative", "positive", "negative"]
            if is_classifier
            else [0.25, 0.5, 0.75, 1.0]
        ),
        index=[90, 80, 70, 60],
    ).iloc[:batch_size]
    for _ in range(warmup):
        if is_autoencoder:
            model.learn_one({"a": 0.5, "b": True, "z": 1})
        else:
            model.learn_one(
                {"a": 0.5, "b": True, "z": 1},
                "positive" if is_classifier else 0.25,
            )
    online = copy.deepcopy(model)
    original = X.copy(deep=True)
    records = X.to_dict(orient="records")
    for x, target in zip(records, y):
        if is_autoencoder:
            online.learn_one(x)
        else:
            online.learn_one(x, target)
    if is_autoencoder:
        model.learn_many(X)
    else:
        model.learn_many(X, y)
    _assert_learning_state(model.module.state_dict(), online.module.state_dict())
    _assert_learning_state(model.optimizer.state_dict(), online.optimizer.state_dict())
    _assert_learning_state(model._x_window, online._x_window)
    assert list(model.observed_features) == list(online.observed_features)
    if is_classifier:
        assert list(model.observed_classes) == list(online.observed_classes)
    pd.testing.assert_frame_equal(X, original)
