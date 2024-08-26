import numpy as np
import torch
from torch import nn

from deep_river.utils.layer_adaptation import (
    expand_layer,
    expand_weights,
    get_expansion_instructions,
    get_in_out_axes,
    get_lstm_param_shapes,
    load_instructions,
)


def test_get_lstm_param_shapes():
    lstm0 = nn.LSTM(4, 3, num_layers=2)
    assert get_lstm_param_shapes(lstm0) == {
        "hidden_size": "O",
        "input_size": "I",
        "weight_ih_l0": "(4o,i)",
        "weight_hh_l0": "(4o,o)",
        "bias_ih_l0": "(4o)",
        "bias_hh_l0": "(4o)",
        "weight_ih_l1": "(4o,1o)",
        "weight_hh_l1": "(4o,o)",
        "bias_ih_l1": "(4o)",
        "bias_hh_l1": "(4o)",
    }
    lstm1 = nn.LSTM(4, 3, bidirectional=True, num_layers=2, proj_size=2)
    assert get_lstm_param_shapes(lstm1) == {
        "hidden_size": "O",
        "input_size": "I",
        "weight_ih_l0": "(4o,i)",
        "weight_hh_l0": "(4o,e)",
        "bias_ih_l0": "(4o)",
        "bias_hh_l0": "(4o)",
        "weight_hr_l0": "(e,o)",
        "weight_ih_l1": "(4o,2e)",
        "weight_hh_l1": "(4o,e)",
        "bias_ih_l1": "(4o)",
        "bias_hh_l1": "(4o)",
        "weight_hr_l1": "(e,o)",
        "weight_ih_l0_reverse": "(4o,i)",
        "weight_hh_l0_reverse": "(4o,e)",
        "bias_ih_l0_reverse": "(4o)",
        "bias_hh_l0_reverse": "(4o)",
        "weight_hr_l0_reverse": "(e,o)",
        "weight_ih_l1_reverse": "(4o,2e)",
        "weight_hh_l1_reverse": "(4o,e)",
        "bias_ih_l1_reverse": "(4o)",
        "bias_hh_l1_reverse": "(4o)",
        "weight_hr_l1_reverse": "(e,o)",
    }


def test_get_in_out_axes():
    expected = {
        "input": [],
        "output": [
            {"axis": 0, "n_subparams": 4},
            {"axis": 1, "n_subparams": 1},
        ],
    }
    result = get_in_out_axes("(4o,o)")
    assert result == expected
    assert get_in_out_axes("(4o,i)") == {
        "input": [{"axis": 1, "n_subparams": 1}],
        "output": [{"axis": 0, "n_subparams": 4}],
    }
    assert get_in_out_axes("(e,1o)") == {
        "input": [],
        "output": [{"axis": 1, "n_subparams": 1}],
    }


def test_get_expansion_instructions():
    shapes_lin = {
        "weight": "(o,i)",
        "bias": "(o)",
        "in_features": "I",
        "out_features": "O",
    }
    result = get_expansion_instructions(shapes_lin)
    assert result == {
        "weight": {
            "input": [{"axis": 1, "n_subparams": 1}],
            "output": [{"axis": 0, "n_subparams": 1}],
        },
        "bias": {
            "input": [],
            "output": [{"axis": 0, "n_subparams": 1}],
        },
        "in_features": "input_attribute",
        "out_features": "output_attribute",
    }


def test_expand_weights():
    rand_ints = np.random.randint(1, 10, (10, 4))
    for x, y, z, n_exp in rand_ints:
        t = torch.ones(x, y, z)
        target_shape = [x, y, z]
        axis = np.random.randint(0, 3)
        target_shape[axis] += n_exp
        t_exp = expand_weights(t, axis, n_exp, nn.init.uniform_)
        assert t_exp.shape == tuple(target_shape)

    t = torch.ones(4, 2)
    t_exp = expand_weights(
        t, axis=0, n_dims_to_add=1, init_fn=nn.init.zeros_, n_subparams=4
    )
    assert t_exp.tolist() == [
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
    ]


def test_expand_layer():
    instructions = {
        "hidden_size": "output_attribute",
        "input_size": "input_attribute",
        "weight_ih_l0": {
            "input": [{"axis": 1, "n_subparams": 1}],
            "output": [{"axis": 0, "n_subparams": 4}],
        },
        "weight_hh_l0": {
            "input": [],
            "output": [
                {"axis": 0, "n_subparams": 4},
                {"axis": 1, "n_subparams": 1},
            ],
        },
        "bias_ih_l0": {"input": [], "output": [{"axis": 0, "n_subparams": 4}]},
        "bias_hh_l0": {"input": [], "output": [{"axis": 0, "n_subparams": 4}]},
    }
    layer = nn.LSTM(4, 3)
    x = torch.zeros(2, 4)
    expand_layer(
        layer,
        output=True,
        target_size=5,
        instructions=instructions,
        init_fn=nn.init.normal_,
    )
    out, _ = layer(x)
    assert out.shape == (2, 5)


def test_load_instructions():
    instructions = {
        "hidden_size": "output_attribute",
        "input_size": "input_attribute",
        "weight_ih_l0": {
            "input": [{"axis": 1, "n_subparams": 1}],
            "output": [{"axis": 0, "n_subparams": 4}],
        },
        "weight_hh_l0": {
            "input": [],
            "output": [
                {"axis": 0, "n_subparams": 4},
                {"axis": 1, "n_subparams": 1},
            ],
        },
        "bias_ih_l0": {"input": [], "output": [{"axis": 0, "n_subparams": 4}]},
        "bias_hh_l0": {"input": [], "output": [{"axis": 0, "n_subparams": 4}]},
    }
    layer = nn.LSTM(4, 3)

    assert load_instructions(layer) == instructions


if __name__ == "__main__":
    test_expand_weights()
