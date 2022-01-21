import pandas as pd
import torch


def dict2tensor(x, device):
    if isinstance(x, dict):
        x = torch.Tensor([list(x.values())], device=device)
    elif isinstance(x, pd.DataFrame):
        x = torch.Tensor(x.values, device=device)
    return x
