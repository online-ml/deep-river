import collections
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from river import base

from deep_river import anomaly, regression
from deep_river.utils import estimator_checks as checks


class ScoringDetector:
    def __init__(self, score):
        self.score = score
        self.learned = set()
        self.events = []

    def score_one(self, x):
        self.events.append(("score", x["value"]))
        assert x["value"] not in self.learned
        return self.score(x)

    def learn_one(self, x):
        self.events.append(("learn", x["value"]))
        self.learned.add(x["value"])


def test_anomaly_check_scores_before_learning():
    model = ScoringDetector(lambda x: x["value"])
    checks.check_roc_auc(model, [({"value": 0.0}, False), ({"value": 1.0}, True)])
    assert model.events == [
        ("score", 0.0),
        ("learn", 0.0),
        ("score", 1.0),
        ("learn", 1.0),
    ]


@pytest.mark.parametrize(
    "score", [lambda x: 0.0, lambda x: -x["value"], lambda x: np.nan, lambda x: np.inf]
)
def test_anomaly_check_rejects_invalid_or_uninformative_scores(score):
    with pytest.raises(AssertionError):
        checks.check_roc_auc(
            ScoringDetector(score), [({"value": 0.0}, False), ({"value": 1.0}, True)]
        )


def test_shared_anomaly_check_replaces_upstream_check():
    model = anomaly.Autoencoder(**next(anomaly.Autoencoder._unit_test_params()))
    roc_checks = [
        check
        for check in checks.yield_checks(model)
        if check.__name__ == "check_roc_auc"
    ]
    assert roc_checks
    for check in roc_checks:
        assert check.func is checks.check_roc_auc
        assert "dataset" in check.keywords


def test_tensor_storage_counts_shared_storage_and_gradients_once():
    weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32))
    weight.grad = torch.ones_like(weight)
    optimizer = torch.optim.Adam([weight])
    optimizer.step()
    model = SimpleNamespace(weight=weight, optimizer=optimizer, view=weight[1:3])
    model.cycle = model
    expected = (
        weight.untyped_storage().nbytes() + weight.grad.untyped_storage().nbytes()
    )
    expected += sum(
        value.untyped_storage().nbytes() for value in optimizer.state[weight].values()
    )
    assert checks._tensor_storage_bytes(model) == expected
    model.copy = weight.detach().clone()
    assert (
        checks._tensor_storage_bytes(model)
        == expected + weight.untyped_storage().nbytes()
    )


class SavedTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x, x[:2], torch.ones(17, device=x.device))
        return x.sum()

    @staticmethod
    def backward(ctx, grad):
        return grad.expand_as(ctx.saved_tensors[0])


@pytest.mark.parametrize("custom", [False, True])
def test_tensor_storage_counts_graph_tensors_once(custom):
    x = torch.ones(10, requires_grad=True)
    output = SavedTensorFunction.apply(x) if custom else x.exp().sum()
    expected = x.untyped_storage().nbytes() + output.untyped_storage().nbytes()
    expected += 17 * x.element_size() if custom else x.untyped_storage().nbytes()
    model = SimpleNamespace(output=output, alias=x[:2], leaf=x)
    assert checks._tensor_storage_bytes(model) == expected
    output.backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x) if custom else x.exp())
    assert checks._tensor_storage_bytes(model) == (
        2 * x.untyped_storage().nbytes() + output.untyped_storage().nbytes()
    )


def test_tensor_storage_counts_retained_nonleaf_gradients():
    x = torch.ones(10, requires_grad=True)
    intermediate = x * 2
    intermediate.retain_grad()
    output = intermediate.sum()
    output.backward(retain_graph=True)
    model = SimpleNamespace(intermediate=intermediate, output=output)
    expected = sum(
        tensor.untyped_storage().nbytes()
        for tensor in (x, x.grad, intermediate, intermediate.grad, output)
    )
    expected += intermediate.grad_fn._saved_other.untyped_storage().nbytes()
    assert checks._tensor_storage_bytes(model) == expected


def test_tensor_storage_traverses_long_graphs():
    output = torch.zeros((), requires_grad=True)
    for _ in range(1500):
        output = output + 1
    assert checks._tensor_storage_bytes(output) == 2 * output.element_size()


def test_tensor_storage_propagates_unexpected_saved_tensor_errors():
    def unpack(tensor):
        raise RuntimeError("Unexpected saved tensor failure")

    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, unpack):
        output = torch.ones(10, requires_grad=True).exp().sum()
    with pytest.raises(RuntimeError, match="Unexpected saved tensor failure"):
        checks._tensor_storage_bytes(output)


class GraphLeakyRegressor(regression.Regressor):
    def learn_one(self, x, y):
        super().learn_one(x, y)
        chunk = torch.ones(10000, requires_grad=True)
        self.history = getattr(self, "history", 0) + (chunk * chunk).sum()


class LeakyRegressor(regression.Regressor):
    def learn_one(self, x, y):
        super().learn_one(x, y)
        if not hasattr(self, "retained"):
            self.retained = []
        self.retained.append(torch.ones(1))


@pytest.mark.parametrize("estimator", [LeakyRegressor, GraphLeakyRegressor])
def test_tensor_memory_check_rejects_retained_tensor_growth(estimator):
    model = estimator(
        module=torch.nn.Sequential(torch.nn.Linear(2, 1)),
        loss_fn="mse",
        optimizer_fn="sgd",
    )
    with pytest.raises(AssertionError):
        checks.check_bounded_tensor_memory(model)


def test_rolling_memory_check_rejects_unbounded_buffer():
    model = regression.RollingRegressor(
        **next(regression.RollingRegressor._unit_test_params())
    )
    model._x_window = collections.deque()
    with pytest.raises(AssertionError):
        checks.check_bounded_tensor_memory(model)


class BrokenBatchRegressor(regression.Regressor):
    def learn_many(self, X, y):
        super().learn_many(X, y + 1.0)


def test_batch_size_one_check_rejects_different_updates():
    model = BrokenBatchRegressor(
        module=torch.nn.Sequential(torch.nn.Linear(2, 1)),
        loss_fn="mse",
        optimizer_fn="sgd",
    )
    with pytest.raises(AssertionError):
        checks.check_batch_size_one_learning(model)


def test_batch_size_one_check_preserves_torch_rng_with_dropout():
    model = regression.Regressor(
        module=torch.nn.Sequential(
            torch.nn.Linear(2, 3), torch.nn.Dropout(), torch.nn.Linear(3, 1)
        ),
        loss_fn="mse",
        optimizer_fn="sgd",
    )
    state = torch.get_rng_state()
    checks.check_batch_size_one_learning(model)
    assert torch.equal(state, torch.get_rng_state())


def test_batch_prediction_checks_are_retained():
    for model in checks.iter_estimators_that_can_be_tested():
        if not isinstance(model, (base.MiniBatchRegressor, base.MiniBatchClassifier)):
            continue
        names = {check.__name__ for check in checks.yield_checks(model)}
        skips = model._unit_test_skips()
        assert "check_learn_many_matches_learn_one" in skips
        assert "check_predict_many_matches_predict_one" in names - skips
        if isinstance(model, base.MiniBatchClassifier):
            assert "check_predict_proba_many_matches_predict_proba_one" in names - skips
        deep_names = {check.__name__ for check in checks.yield_deep_checks(model)}
        assert "check_batch_size_one_learning" in deep_names - skips
        assert "check_bounded_tensor_memory" in deep_names - skips


@pytest.mark.parametrize("n_samples", [1, 4])
def test_probability_weighted_batch_loss_and_statistics(n_samples):
    model = anomaly.ProbabilityWeightedAutoencoder(
        module=torch.nn.Sequential(torch.nn.Linear(2, 2))
    )
    reference = copy.deepcopy(model)
    X = pd.DataFrame({"a": [0.2] * n_samples, "b": np.linspace(0.0, 1.0, n_samples)})
    reference._update_observed_features(X)
    inputs = reference._df2tensor(X)
    losses = (reference.module(inputs) - inputs).square().mean(dim=1)
    initial_losses = losses.detach().tolist()
    probabilities = 0.5 * (1 + torch.erf(losses.detach().double() / np.sqrt(2)))
    weights = (
        (reference.skip_threshold - probabilities) / reference.skip_threshold
    ).to(losses)
    (weights * losses).mean().backward()
    reference.optimizer.step()
    model.learn_many(X)
    torch.testing.assert_close(model.module.state_dict(), reference.module.state_dict())
    assert model.rolling_mean.get() == pytest.approx(np.mean(initial_losses))
    expected_variance = np.var(initial_losses, ddof=1) if n_samples > 1 else 0.0
    assert model.rolling_var.get() == pytest.approx(expected_variance)
