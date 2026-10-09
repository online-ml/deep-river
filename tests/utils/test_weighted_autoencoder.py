import copy
import math

import numpy as np
import pandas as pd
import pytest
import torch

from deep_river.anomaly import ProbabilityWeightedAutoencoder


@pytest.mark.parametrize("many", [False, True])
@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_probable_outliers_skip_updates_including_optimizer_momentum(many, optimizer):
    module = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        module[0].weight.zero_()
    model = ProbabilityWeightedAutoencoder(module, optimizer_fn=optimizer, lr=0.01)
    model.learn_one({"a": 0.1})
    parameters = copy.deepcopy(module.state_dict())
    state = copy.deepcopy(model.optimizer.state_dict())
    before = model.score_one({"a": 10.0})
    if many:
        model.learn_many(pd.DataFrame({"a": [10.0, 10.0]}))
    else:
        model.learn_one({"a": 10.0})
    torch.testing.assert_close(module.state_dict(), parameters)
    torch.testing.assert_close(model.optimizer.state_dict(), state)
    assert model.score_one({"a": 10.0}) == before
    assert model.rolling_mean.get() > 1


def test_mixed_batch_ignores_outlier_gradients_and_keeps_actual_loss_statistics():
    module = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        module[0].weight.zero_()
    model = ProbabilityWeightedAutoencoder(module, lr=0.01)
    X = pd.DataFrame({"a": [0.1, 10.0]})
    probability = 0.5 * (1 + math.erf(0.01 / np.sqrt(2)))
    expected_gradient = -0.01 * (0.9 - probability) / 0.9
    model.learn_many(X)
    assert model.module[0].weight.item() == pytest.approx(-0.01 * expected_gradient)
    assert model.rolling_mean.get() == pytest.approx((0.01 + 100) / 2)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_skip_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="skip_threshold"):
        ProbabilityWeightedAutoencoder(
            torch.nn.Sequential(torch.nn.Linear(1, 1)), skip_threshold=threshold
        )


@pytest.mark.parametrize("window_size", [0, -1])
def test_invalid_window_size_is_rejected(window_size):
    with pytest.raises(ValueError, match="window_size"):
        ProbabilityWeightedAutoencoder(
            torch.nn.Sequential(torch.nn.Linear(1, 1)), window_size=window_size
        )


def test_weighted_autoencoder_clips_gradients():
    module = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        module[0].weight.zero_()
    model = ProbabilityWeightedAutoencoder(module, gradient_clip_value=0.001)
    model.learn_one({"a": 0.1})
    assert 0 < module[0].weight.grad.norm().item() <= 0.001001
