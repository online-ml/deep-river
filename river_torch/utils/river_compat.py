from typing import Union

import pandas as pd
import torch
from river import base


def dict2tensor(x: dict, device="cpu", dtype=torch.float32):
    x = torch.tensor([list(x.values())], device=device, dtype=dtype)
    return x


def scalar2tensor(x: Union[float, int], device="cpu", dtype=torch.float32):
    x = torch.tensor([x], device=device, dtype=dtype)
    return x


def pandas2tensor(x: pd.DataFrame, device="cpu", dtype=torch.float32):
    x = torch.tensor(x.values, device=device, dtype=dtype)
    return x


def list2tensor(x: list, device="cpu", dtype=torch.float32):
    x = torch.tensor(x, device=device, dtype=dtype)
    return x


def target2onehot(
    y: base.typing.ClfTarget, classes: list, n_classes: int, device="cpu"
) -> torch.Tensor:
    onehot = torch.zeros(n_classes, device=device)
    pos_idx = classes.index(y)
    if pos_idx < n_classes:
        onehot[classes.index(y)] = 1
    return onehot


def output2proba(pred: torch.TensorType, classes: list):
    pred = pred[0].detach().numpy()
    if len(pred) == 1:
        pred = (pred[0], 1 - pred[0])
        proba = dict(zip(classes, pred))
    else:
        proba = dict(zip(classes, pred))
    return proba
