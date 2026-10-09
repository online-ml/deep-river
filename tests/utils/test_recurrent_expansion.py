import pytest
import torch

from deep_river.regression import RollingRegressor


class SequenceModule(torch.nn.Module):
    def __init__(self, kind, bidirectional, layers):
        super().__init__()
        self.rnn = kind(2, 3, num_layers=layers, bidirectional=bidirectional)
        self.head = torch.nn.Linear(3 * (2 if bidirectional else 1), 1)

    def forward(self, x):
        return self.head(self.rnn(x)[0][-1])


@pytest.mark.parametrize("kind", [torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("layers", [1, 2])
def test_recurrent_expansion_preserves_all_directions(
    kind, bidirectional, layers, tmp_path
):
    model = RollingRegressor(
        SequenceModule(kind, bidirectional, layers), is_feature_incremental=True
    )
    model.learn_one({"a": 1.0, "b": 2.0}, 1.0)
    old = {
        name: value.detach().clone()
        for name, value in model.module.rnn.named_parameters()
    }
    model._update_observed_features({"a": 1.0, "b": 2.0, "c": 3.0})
    rnn = model.module.rnn
    assert rnn.input_size == 3
    for name, value in rnn.named_parameters():
        if name.startswith("weight_ih_l0"):
            assert value.shape[1] == 3
            torch.testing.assert_close(value[:, :2], old[name])
        else:
            torch.testing.assert_close(value, old[name])
    model.learn_one({"a": 1.0, "b": 2.0, "c": 3.0}, 2.0)
    for parameter in rnn.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    path = tmp_path / "model.pkl"
    model.save(path)
    restored = type(model).load(path)
    assert restored.predict_one({"a": 1.0, "b": 2.0, "c": 3.0}) == model.predict_one(
        {"a": 1.0, "b": 2.0, "c": 3.0}
    )
    restored.learn_one({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, 1.0)
    assert restored.module.rnn.weight_ih_l0.shape[1] == 4
    if bidirectional:
        assert restored.module.rnn.weight_ih_l0_reverse.shape[1] == 4
