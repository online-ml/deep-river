from collections import deque

import numpy as np
import pandas as pd
import torch
from ordered_set import OrderedSet

from deep_river.utils import (
    deque2rolling_tensor,
    df2tensor,
    dict2tensor,
    float2tensor,
    labels2onehot,
    output2proba,
)


def test_dict2tensor():
    x = {"a": 1, "b": 2, "c": 3}
    assert dict2tensor(x, features=OrderedSet(x.keys())).tolist() == [[1, 2, 3]]
    # Test dissapearing features
    x2 = {"b": 2, "c": 3}
    assert dict2tensor(x2, features=OrderedSet(x.keys())).tolist() == [[0, 2, 3]]
    # Test shuffled features
    x3 = {"b": 2, "a": 1, "c": 3}
    assert dict2tensor(x3, features=OrderedSet(x.keys())).tolist() == [[1, 2, 3]]


def test_float2tensor():
    y = 1.0
    assert float2tensor(y).tolist() == [[1.0]]


def test_deque2rolling_tensor():
    window = deque(np.ones((2, 3)).tolist(), maxlen=3)

    assert deque2rolling_tensor(window).tolist() == [
        [[1, 1, 1]],
        [[1, 1, 1]],
    ]
    assert list(window) == [[1, 1, 1], [1, 1, 1]]
    window.append([1, 2, 3])

    assert list(window) == [
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 3],
    ]
    assert deque2rolling_tensor(window).tolist() == [
        [[1, 1, 1]],
        [[1, 1, 1]],
        [[1, 2, 3]],
    ]
    assert list(window) == [
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 3],
    ]


def test_df2tensor():
    features = ["a", "b", "c"]
    x = pd.DataFrame(np.zeros((2, 3)), columns=features)
    assert df2tensor(x, features=features).tolist() == [[0, 0, 0], [0, 0, 0]]
    x2 = pd.DataFrame(np.zeros((2, 2)), columns=["b", "c"])
    assert df2tensor(x2, features=features).tolist() == [[0, 0, 0], [0, 0, 0]]


def test_labels2onehot():
    classes = OrderedSet(["first class", "second class", "third class"])
    y1 = "first class"
    y2 = "third class"
    assert labels2onehot(y1, classes).tolist() == [[1, 0, 0]]
    assert labels2onehot(y2, classes).tolist() == [[0, 0, 1]]
    classes = OrderedSet(["first class"])
    n_classes = 3
    assert labels2onehot(y1, classes, n_classes).tolist() == [[1, 0, 0]]

    classes = OrderedSet(["first class", "second class", "third class"])
    y1 = pd.Series(["first class", "third class"])
    assert labels2onehot(y1, classes).tolist() == [[1, 0, 0], [0, 0, 1]]
    assert labels2onehot(y1, classes, n_classes=4).tolist() == [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ]


def dicts_are_close(d1, d2):
    keys_match = list(d1.keys()) == list(d2.keys())
    values_match = np.allclose(list(d1.values()), list(d2.values()))
    return keys_match and values_match


def test_output2proba():
    preds = torch.tensor([[2.0, 1.0, 0.1]])
    classes = OrderedSet(["class1", "class2", "class3"])
    expected_output = {"class1": 2.0, "class2": 1.0, "class3": 0.1}
    assert dicts_are_close(
        output2proba(preds, classes, output_is_logit=False)[0], expected_output
    )

    preds = torch.tensor([[2.0, 1.0, 0.1]])
    classes = OrderedSet(["class1", "class2", "class3"])
    softmaxed_preds = torch.softmax(preds, dim=-1).numpy().flatten()
    expected_output = {
        "class1": softmaxed_preds[0],
        "class2": softmaxed_preds[1],
        "class3": softmaxed_preds[2],
    }

    assert dicts_are_close(
        output2proba(preds, classes, output_is_logit=True)[0], expected_output
    )

    preds = torch.tensor([[0.8]])
    classes = OrderedSet(["positive"])
    sigmoid_pred = torch.sigmoid(preds).numpy().flatten()
    expected_output = {
        "positive": sigmoid_pred[0],
        "Unobserved0": 1 - sigmoid_pred[0],
    }

    assert dicts_are_close(
        output2proba(preds, classes, output_is_logit=True)[0], expected_output
    )

    preds = torch.tensor([[0.7]])
    classes = OrderedSet([])
    expected_output = {True: 0.7, False: 0.3}
    assert dicts_are_close(
        output2proba(preds, classes, output_is_logit=False)[0], expected_output
    )

    preds = torch.tensor([[0.2, 0.5, 0.3]])
    classes = OrderedSet(["class1"])
    expected_output = {"class1": 0.2, "Unobserved0": 0.5, "Unobserved1": 0.3}

    assert dicts_are_close(
        output2proba(preds, classes, output_is_logit=False)[0], expected_output
    )

    preds = torch.tensor([[2.0, 1.0], [0.1, 0.9]])
    classes = OrderedSet(["class1", "class2"])
    expected_output = [
        {"class1": 2.0, "class2": 1.0},
        {"class1": 0.1, "class2": 0.9},
    ]
    output = output2proba(preds, classes, output_is_logit=False)
    for output_i, expected_output_i in zip(output, expected_output):
        assert dicts_are_close(output_i, expected_output_i)
