from typing import Union

import pandas as pd
import torch


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
