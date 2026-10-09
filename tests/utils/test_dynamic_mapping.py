import copy
import pickle

import pandas as pd
import pytest
import torch

from deep_river import classification, regression
from deep_river.utils.ordered_set import OrderedSet
from deep_river.utils.tensor_conversion import labels2onehot


def test_ordered_set_indices_survive_updates_and_round_trips():
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


@pytest.mark.parametrize("many", [False, True])
def test_new_zero_feature_preserves_prediction(many):
    module = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        module[0].weight.copy_(torch.tensor([[2.0, 0.0]]))
    model = regression.Regressor(module, "mse", "sgd", lr=0)
    assert model.predict_one({"z": 3.0}) == 6.0
    if many:
        result = model.predict_many(pd.DataFrame({"a": [0.0], "z": [3.0]})).iloc[0]
    else:
        result = model.predict_one({"a": 0.0, "z": 3.0})
    assert result == 6.0
    assert list(model.observed_features) == ["z", "a"]


@pytest.mark.parametrize("first,second", [("z", "a"), (True, False), (False, True)])
def test_new_class_preserves_existing_output_mapping(first, second):
    module = torch.nn.Sequential(torch.nn.Linear(1, 2))
    with torch.no_grad():
        module[0].weight.zero_()
        module[0].bias.copy_(torch.tensor([3.0, -3.0]))
    model = classification.Classifier(module, "cross_entropy", "sgd")
    model._update_observed_targets(first)
    original_index = model.observed_classes.index(first)
    expected = model.predict_proba_one({"x": 0.0})[first]
    model._update_observed_targets(second)
    assert model.observed_classes.index(first) == original_index
    assert model.predict_proba_one({"x": 0.0})[first] == expected


@pytest.mark.parametrize("first", [True, False])
def test_single_output_boolean_targets_keep_their_meaning(first):
    classes = OrderedSet([first])
    classes.add(not first)
    assert labels2onehot(True, classes, n_classes=1).item() == 1.0
    assert labels2onehot(False, classes, n_classes=1).item() == 0.0


def test_new_target_preserves_existing_prediction_and_saved_order(tmp_path):
    module = torch.nn.Sequential(torch.nn.Linear(1, 2))
    with torch.no_grad():
        module[0].weight.zero_()
        module[0].bias.copy_(torch.tensor([3.0, 7.0]))
    model = regression.MultiTargetRegressor(module, "mse", "sgd")
    model._update_observed_targets({"z": 0.0})
    assert model.predict_one({"x": 0.0})["z"] == 3.0
    model._update_observed_targets({"a": 0.0})
    assert model.predict_one({"x": 0.0}) == {"z": 3.0, "a": 7.0}
    path = tmp_path / "model.pkl"
    model.save(path)
    restored = type(model).load(path)
    assert list(restored.observed_targets) == ["z", "a"]
    assert restored.predict_one({"x": 0.0}) == model.predict_one({"x": 0.0})


def test_feature_order_is_independent_of_initial_dictionary_order():
    left = regression.Regressor(
        torch.nn.Sequential(torch.nn.Linear(2, 1)), "mse", "sgd"
    )
    right = copy.deepcopy(left)
    left.predict_one({"z": 2.0, "a": 1.0})
    right.predict_one({"a": 1.0, "z": 2.0})
    assert list(left.observed_features) == list(right.observed_features)
