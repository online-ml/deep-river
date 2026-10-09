import copy

import pytest
import torch

from deep_river.regression import Regressor


@pytest.mark.parametrize(
    "optimizer,kwargs",
    [
        (torch.optim.Adam, {"amsgrad": True}),
        (torch.optim.AdamW, {}),
        (torch.optim.SGD, {"momentum": 0.9}),
        (torch.optim.RMSprop, {"momentum": 0.9, "centered": True}),
    ],
)
@pytest.mark.parametrize("output", [False, True])
def test_expansion_preserves_state_and_continues_learning(optimizer, kwargs, output):
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model = Regressor(module, "mse", "sgd")
    model.optimizer = optimizer(module.parameters(), lr=0.01, **kwargs)
    model.learn_one({"a": 1.0, "b": 2.0}, 3.0)
    original_optimizer = model.optimizer
    layer = model.output_layer if output else model.input_layer
    old_weight = layer.weight
    old_state = copy.deepcopy(model.optimizer.state[old_weight])
    untouched = module[0].weight if output else module[1].weight
    untouched_state = copy.deepcopy(model.optimizer.state[untouched])
    model._expand_layer(layer, 3, output=output)
    assert model.optimizer is original_optimizer
    assert old_weight not in model.optimizer.state
    torch.testing.assert_close(model.optimizer.state[untouched], untouched_state)
    for name, value in old_state.items():
        actual = model.optimizer.state[layer.weight][name]
        if isinstance(value, torch.Tensor) and value.shape == old_weight.shape:
            indices = tuple(slice(0, size) for size in value.shape)
            torch.testing.assert_close(actual[indices], value)
            extra = actual[old_weight.shape[0] :] if output else actual[:, 2:]
            assert torch.count_nonzero(extra) == 0
        else:
            torch.testing.assert_close(actual, value)
    before = layer.weight.detach().clone()
    if output:
        model._learn(torch.tensor([[1.0, 2.0]]), torch.zeros(1, 3))
    else:
        model.learn_one({"a": 1.0, "b": 2.0, "c": 3.0}, 4.0)
    assert not torch.equal(before, layer.weight)


def test_expansion_preserves_parameter_groups_and_scheduler():
    module = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Linear(2, 1))
    model = Regressor(module, "mse", "sgd")
    model.optimizer = torch.optim.Adam(
        [
            {"params": module[0].parameters(), "lr": 0.02, "betas": (0.8, 0.9)},
            {"params": module[1].parameters(), "lr": 0.01, "weight_decay": 0.1},
        ]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=1)
    model.learn_one({"a": 1.0}, 1.0)
    scheduler.step()
    groups = [
        {k: v for k, v in g.items() if k != "params"}
        for g in model.optimizer.param_groups
    ]
    model._expand_layer(module[0], 2, output=False)
    assert scheduler.optimizer is model.optimizer
    assert groups == [
        {k: v for k, v in g.items() if k != "params"}
        for g in model.optimizer.param_groups
    ]
    parameters = [p for group in model.optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in parameters} == {id(p) for p in module.parameters()}


def test_expansion_before_first_update_and_repeated_expansion():
    model = Regressor(
        torch.nn.Sequential(torch.nn.Linear(1, 1)),
        "mse",
        "adam",
        is_feature_incremental=True,
    )
    for size in (2, 3, 5):
        model.learn_one({str(i): float(i) for i in range(size)}, 1.0)
        assert model.input_layer.weight.shape == (1, size)
        assert model.optimizer.state[model.input_layer.weight]["exp_avg"].shape == (
            1,
            size,
        )
