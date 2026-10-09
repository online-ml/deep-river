import copy
import warnings

import pandas as pd
import pytest
import torch

from deep_river import anomaly, classification, regression


class SequenceModule(torch.nn.Module):
    def __init__(self, output_size, reconstruct=False):
        super().__init__()
        self.rnn = torch.nn.GRU(2, 4)
        self.head = torch.nn.Linear(4, output_size)
        self.reconstruct = reconstruct

    def forward(self, x):
        output = self.head(self.rnn(x)[0])
        return output if self.reconstruct else output[-1]


@pytest.mark.parametrize(
    "kind",
    [
        regression.RollingRegressor,
        classification.RollingClassifier,
        anomaly.RollingAutoencoder,
    ],
)
@pytest.mark.parametrize("batch_size", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("warmup", [0, 2])
def test_rolling_batch_matches_sequential_windows(kind, batch_size, warmup):
    classifier = kind is classification.RollingClassifier
    autoencoder = kind is anomaly.RollingAutoencoder
    model = kind(
        SequenceModule(2 if classifier or autoencoder else 1, reconstruct=autoencoder),
        window_size=3,
        optimizer_fn="adam",
        loss_fn="cross_entropy" if classifier else "mse",
    )
    for i in range(warmup):
        x = {"a": float(i), "b": 0.5}
        model.learn_one(x) if autoencoder else model.learn_one(x, i % 2)
    online = copy.deepcopy(model)
    X = pd.DataFrame(
        {"a": [float(i) for i in range(batch_size)], "b": [0.5] * batch_size},
        index=range(10, 10 + batch_size),
    )
    y = pd.Series([i % 2 for i in range(batch_size)], index=X.index, dtype=int)
    for x, target in zip(X.to_dict(orient="records"), y):
        online.learn_one(x) if autoencoder else online.learn_one(x, target)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.learn_many(X) if autoencoder else model.learn_many(X, y)
    torch.testing.assert_close(model.module.state_dict(), online.module.state_dict())
    torch.testing.assert_close(
        model.optimizer.state_dict(), online.optimizer.state_dict()
    )
    assert list(model._x_window) == list(online._x_window)


@pytest.mark.parametrize(
    "kind", [regression.RollingRegressor, classification.RollingClassifier]
)
def test_rolling_batch_rejects_misaligned_targets_without_learning(kind):
    model = kind(SequenceModule(2 if kind is classification.RollingClassifier else 1))
    weights = copy.deepcopy(model.module.state_dict())
    with pytest.raises(ValueError, match="same number of rows"):
        model.learn_many(
            pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 0.0]}), pd.Series([1])
        )
    assert not model._x_window
    torch.testing.assert_close(model.module.state_dict(), weights)
