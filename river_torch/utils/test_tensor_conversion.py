from collections import deque

import numpy as np
import pandas as pd
import torch

from river_torch.utils import (
    df2rolling_tensor,
    df2tensor,
    dict2rolling_tensor,
    dict2tensor,
    float2tensor,
    labels2onehot,
    output2proba,
)


def test_dict2tensor():
    x = {"a": 1, "b": 2, "c": 3}
    assert dict2tensor(x).tolist() == [[1, 2, 3]]


def test_float2tensor():
    y = 1.0
    assert float2tensor(y).tolist() == [[1.0]]


def test_dict2rolling_tensor():
    window = deque(np.ones((3, 3)).tolist(), maxlen=3)
    x = {"a": 1, "b": 2, "c": 3}

    assert dict2rolling_tensor(x, window, update_window=False).tolist() == [
        [[1, 1, 1]],
        [[1, 1, 1]],
        [[1, 2, 3]],
    ]
    assert list(window) == [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    assert dict2rolling_tensor(x, window, update_window=True).tolist() == [
        [[1, 1, 1]],
        [[1, 1, 1]],
        [[1, 2, 3]],
    ]
    assert list(window) == [
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 3],
    ]
    window = deque(np.ones((1, 3)).tolist(), maxlen=3)
    assert dict2rolling_tensor(x, window, update_window=True) is None
    assert list(window) == [
        [1, 1, 1],
        [1, 2, 3],
    ]


def test_df2rolling_tensor():
    window = deque(np.ones((3, 3)).tolist(), maxlen=3)
    x = pd.DataFrame(np.zeros((2, 3)))
    assert df2rolling_tensor(x, window, update_window=False).tolist() == [
        [
            [1, 1, 1],
            [1, 1, 1],
        ],
        [
            [1, 1, 1],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
        ],
    ]
    assert list(window) == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert df2rolling_tensor(x, window, update_window=True).tolist() == [
        [
            [1, 1, 1],
            [1, 1, 1],
        ],
        [
            [1, 1, 1],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
        ],
    ]
    assert list(window) == [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
    window = deque(np.ones((3, 3)).tolist(), maxlen=6)
    assert df2rolling_tensor(x, window, update_window=True) is None
    assert list(window) == [[1, 1, 1]] * 3 + [[0, 0, 0]] * 2


def test_df2tensor():
    x = pd.DataFrame(np.zeros((2, 3)))
    assert df2tensor(x).tolist() == [[0, 0, 0], [0, 0, 0]]


def test_labels2onehot():
    classes = ["first class", "second class", "third class"]
    y1 = "first class"
    y2 = "third class"
    assert labels2onehot(y1, classes).tolist() == [[1, 0, 0]]
    assert labels2onehot(y2, classes).tolist() == [[0, 0, 1]]
    classes = ["first class"]
    n_classes = 3
    assert labels2onehot(y1, classes, n_classes).tolist() == [[1, 0, 0]]

    classes = ["first class", "second class", "third class"]
    y1 = ["first class", "third class"]
    assert labels2onehot(y1, classes).tolist() == [[1, 0, 0], [0, 0, 1]]
    assert labels2onehot(y1, classes, n_classes=4).tolist() == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ]


def test_output2proba():
    def assert_dicts_almost_equal(d1, d2):
        for k in d1:
            assert np.isclose(d1[k], d2[k])

    y = torch.tensor([[0.1, 0.2, 0.7]])
    classes = ["first class", "second class", "third class"]
    assert_dicts_almost_equal(
        output2proba(y, classes),
        dict(zip(classes, np.array([0.1, 0.2, 0.7], dtype=np.float32))),
    )
    y = torch.tensor([[0.6]])
    classes = ["first class"]
    assert_dicts_almost_equal(
        output2proba(y, classes),
        dict(
            zip(
                ["first class", "unobserved 0"],
                np.array([0.6, 0.4], dtype=np.float32),
            )
        ),
    )
    y = torch.tensor([[0.6, 0.4, 0.0]])
    assert_dicts_almost_equal(
        output2proba(y, classes),
        dict(
            zip(
                ["first class", "unobserved 0", "unobserved 1"],
                np.array([0.6, 0.4, 0.0], dtype=np.float32),
            )
        ),
    )
