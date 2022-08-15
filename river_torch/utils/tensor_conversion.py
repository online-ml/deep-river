from typing import Deque, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from river import base


def dict2tensor(x: dict, device="cpu", dtype=torch.float32) -> torch.TensorType:
    x = torch.tensor([list(x.values())], device=device, dtype=dtype)
    return x


def float2tensor(
    y: Union[float, int], device="cpu", dtype=torch.float32
) -> torch.TensorType:
    y = torch.tensor([y], device=device, dtype=dtype)
    return y


def dict2rolling_tensor(
    x: Dict, window: Deque, device="cpu", dtype=torch.float32, update_window=True
) -> torch.TensorType:
    output = None
    excess_len = len(window) + 1 - window.maxlen
    if update_window:
        window.append(list(x.values()))
        if excess_len >= 0:
            output = torch.tensor(window, device=device, dtype=dtype)
    else:
        if excess_len >= 0:
            window_copy = list(window)[excess_len:] + [list(x.values())]
            output = torch.tensor(window_copy, device=device, dtype=dtype)

    return output


def df2tensor(x: pd.DataFrame, device="cpu", dtype=torch.float32) -> torch.TensorType:
    x = torch.tensor(x.values, device=device, dtype=dtype)
    return x


def df2rolling_tensor(
    x: pd.DataFrame,
    window: Deque,
    device="cpu",
    dtype=torch.float32,
    update_window=True,
) -> torch.TensorType:
    x_old = list(window)
    if len(window) >= window.maxlen:
        x_old = x_old[1:]
    x_new = x.values.tolist()
    x = x_old + x_new
    if len(x) >= window.maxlen:
        x = [x[i : i + window.maxlen] for i in range(len(x) - window.maxlen + 1)]
        x = torch.tensor(x, device=device, dtype=dtype)
    else:
        x = None
    if update_window:
        window.extend(x_new)
    return x


def labels2onehot(
    y: Union[base.typing.ClfTarget, List],
    classes: list,
    n_classes: int = None,
    device="cpu",
) -> torch.TensorType:
    if n_classes is None:
        n_classes = len(classes)
    if not isinstance(y, list):
        y = [y]
    onehot = torch.zeros(len(y), n_classes, device=device)
    pos_idcs = [classes.index(y_i) for y_i in y]
    for i, pos_idx in enumerate(pos_idcs):
        if pos_idx < n_classes:
            onehot[i, pos_idx] = 1
    return onehot


def output2proba(preds: torch.TensorType, classes: List) -> List:
    preds = preds.detach().numpy()
    if preds.shape[1] == 1:
        preds = np.hstack((preds, 1 - preds))
    n_unobserved_classes = preds.shape[1] - len(classes)
    if n_unobserved_classes > 0:
        classes = classes + [f"unobserved {i}" for i in range(n_unobserved_classes)]
    probas = (
        dict(zip(classes, preds[0]))
        if preds.shape[0] == 1
        else [dict(zip(classes, pred)) for pred in preds]
    )
    return probas
