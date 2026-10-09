import numpy as np
import pandas as pd
import pytest
import torch

from deep_river import anomaly


@pytest.mark.parametrize(
    "kind",
    [
        anomaly.Autoencoder,
        anomaly.ProbabilityWeightedAutoencoder,
        anomaly.RollingAutoencoder,
    ],
)
@pytest.mark.parametrize("many", [False, True])
def test_autoencoder_grows_input_and_reconstruction(kind, many, tmp_path):
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2)
    )
    model = kind(
        module,
        is_feature_incremental=True,
        **({"window_size": 2} if kind is anomaly.RollingAutoencoder else {}),
    )
    for width in (2, 3, 5):
        record = {str(i): float(i) / 10 for i in range(width)}
        model._update_observed_features(record)
        assert model.input_layer.in_features == width
        assert model.output_layer.out_features == width
        if many:
            frame = pd.DataFrame([record, record])
            model.learn_many(frame)
            assert np.isfinite(model.score_many(frame)).all()
        else:
            model.learn_one(record)
            assert np.isfinite(model.score_one(record))
    path = tmp_path / "model.pkl"
    model.save(path)
    restored = type(model).load(path)
    record["z"] = 1.0
    restored.learn_one(record)
    assert restored.input_layer.in_features == 6
    assert restored.output_layer.out_features == 6


def test_expansion_preserves_existing_reconstruction_weights():
    model = anomaly.Autoencoder(
        torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 2)),
        is_feature_incremental=True,
    )
    model.learn_one({"a": 1.0, "b": 2.0})
    weight = model.output_layer.weight.detach().clone()
    bias = model.output_layer.bias.detach().clone()
    model._update_observed_features({"a": 1.0, "b": 2.0, "c": 3.0})
    torch.testing.assert_close(model.output_layer.weight[:2], weight)
    torch.testing.assert_close(model.output_layer.bias[:2], bias)


def test_one_layer_autoencoder_expands_both_axes():
    model = anomaly.Autoencoder(
        torch.nn.Sequential(torch.nn.Linear(2, 2)), is_feature_incremental=True
    )
    model.learn_one({"a": 1.0, "b": 2.0})
    model.learn_one({"a": 1.0, "b": 2.0, "c": 3.0})
    assert model.module[0].weight.shape == (3, 3)
    assert np.isfinite(model.score_one({"a": 1.0, "b": 2.0, "c": 3.0}))
