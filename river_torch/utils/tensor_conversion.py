from typing import Deque, Dict, List, Type, Union

import numpy as np
import pandas as pd
import torch
from ordered_set import OrderedSet
from river import base
from river.base.typing import RegTarget


def dict2tensor(x: dict,
                device: str = "cpu",
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    x = torch.tensor([list(x.values())], device=device, dtype=dtype)
    return x


def float2tensor(
        y: Union[float, int, RegTarget], device="cpu", dtype=torch.float32
) -> torch.Tensor:
    y = torch.tensor([[y]], device=device, dtype=dtype)
    return y


def dict2rolling_tensor(
        x: Dict,
        window: Deque,
        device="cpu",
        dtype=torch.float32,
        update_window=True,
) -> torch.Tensor:
    output = None
    excess_len = len(window) + 1 - window.maxlen
    if update_window:
        window.append(list(x.values()))
        new_window = window

    if excess_len >= 0:
        if not update_window:
            new_window = list(window)[excess_len:] + [list(x.values())]
        output = torch.tensor(new_window, device=device, dtype=dtype)
        output = torch.unsqueeze(output, 1)
    return output


def df2tensor(
        x: pd.DataFrame, device="cpu", dtype=torch.float32
) -> torch.Tensor:
    x = torch.tensor(x.values, device=device, dtype=dtype)
    return x


def df2rolling_tensor(
        x: pd.DataFrame,
        window: Deque,
        device="cpu",
        dtype=torch.float32,
        update_window=True,
) -> torch.Tensor:
    x_old = list(window)
    if len(window) >= window.maxlen:
        x_old = x_old[1:]
    x_new = x.values.tolist()
    x = x_old + x_new
    if len(x) >= window.maxlen:
        x = [
            x[i: i + len(x) - window.maxlen + 1] for i in range(window.maxlen)
        ]
        x = torch.tensor(x, device=device, dtype=dtype)
    else:
        x = None
    if update_window:
        window.extend(x_new)
    return x


def labels2onehot(
        y: Union[base.typing.ClfTarget, List],
        classes: Type[OrderedSet],
        n_classes: int = None,
        device="cpu",
        dtype=torch.float32,
) -> torch.Tensor:
    if n_classes is None:
        n_classes = len(classes)
    if isinstance(y, list):
        onehot = torch.zeros(len(y), n_classes, device=device, dtype=dtype)
        pos_idcs = [classes.index(y_i) for y_i in y]
        for i, pos_idx in enumerate(pos_idcs):
            if pos_idx < n_classes:
                onehot[i, pos_idx] = 1
    else:
        onehot = torch.zeros(1, n_classes, device=device, dtype=dtype)
        pos_idx = classes.index(y)
        if pos_idx < n_classes:
            onehot[0, pos_idx] = 1

    return onehot


def output2proba(
        preds: torch.Tensor, classes: Type[OrderedSet], with_logits=False
) -> List:
    if with_logits:
        if preds.shape[-1] >= 1:
            preds = torch.softmax(preds, dim=-1)
        else:
            preds = torch.sigmoid(preds)

    preds = preds.detach().numpy()
    if preds.shape[1] == 1:
        preds = np.hstack((preds, 1 - preds))
    n_unobserved_classes = preds.shape[1] - len(classes)
    if n_unobserved_classes > 0:
        classes = list(classes)
        classes.extend(
            [f"unobserved {i}" for i in range(n_unobserved_classes)]
        )
    probas = (
        dict(zip(classes, preds[0]))
        if preds.shape[0] == 1
        else [dict(zip(classes, pred)) for pred in preds]
    )
    return probas
