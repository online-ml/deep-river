from typing import Deque, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from ordered_set import OrderedSet
from river import base
from river.base.typing import ClfTarget, RegTarget


def dict2tensor(
    x: dict,
    features: OrderedSet,
    default_value: float = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert a dictionary to a tensor.

    Parameters
    ----------
    x
        Dictionary.
    features:
        Set of possible features.
    default_value:
        Value to use for features not present in x.
    device
        Device.
    dtype
        Dtype.

    Returns
    -------
        torch.Tensor
    """
    return torch.tensor(
        [[x.get(feature, default_value) for feature in features]],
        device=device,
        dtype=dtype,
    )


def float2tensor(
    y: Union[float, int, RegTarget, dict], device="cpu", dtype=torch.float32
) -> torch.Tensor:
    """
    Convert a float to a tensor.

    Parameters
    ----------
    y
        Float.
    device
        Device.
    dtype
        Dtype.

    Returns
    -------
        torch.Tensor
    """
    if isinstance(y, dict):
        return torch.tensor([list(y.values())], device=device, dtype=dtype)
    else:
        return torch.tensor([[y]], device=device, dtype=dtype)


def deque2rolling_tensor(
    window: Deque,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Convert a dictionary to a rolling tensor.

    Parameters
    ----------
    window
        Rolling window.
    device
        Device.
    dtype
        Dtype.

    Returns
    -------
        torch.Tensor
    """
    output = torch.tensor(window, device=device, dtype=dtype)
    return torch.unsqueeze(output, 1)


def df2tensor(
    X: pd.DataFrame,
    features: OrderedSet,
    default_value: float = 0,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Convert a dataframe to a tensor.
    Parameters
    ----------
    X
        Dataframe.
    features:
        Set of possible features.
    default_value:
        Value to use for features not present in x.
    device
        Device.
    dtype
        Dtype.

    Returns
    -------
        torch.Tensor
    """
    for feature in features:
        if feature not in X.columns:
            X[feature] = default_value
    return torch.tensor(X[list(features)].values, device=device, dtype=dtype)


def labels2onehot(
    y: Union[base.typing.ClfTarget, pd.Series],
    classes: OrderedSet[base.typing.ClfTarget],
    n_classes: Optional[int] = None,
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Convert a label or a list of labels to a one-hot encoded tensor.

    Parameters
    ----------
    y
        Label or list of labels.
    classes
        Classes.
    n_classes
        Number of classes.
    device
        Device.
    dtype
        Dtype.

    Returns
    -------
        torch.Tensor
    """

    def get_class_index(label):
        """Retrieve class index with type checking and conversion."""
        if isinstance(label, float):  # Handle float case
            if label.is_integer():  # Convert to int if it's an integer-like float
                label = int(label)
            else:
                raise ValueError(
                    f"Label {label} is a float and cannot be mapped to a class index."
                )
        return classes.index(label)

    if n_classes is None:
        n_classes = len(classes)
    if isinstance(y, pd.Series):
        onehot = torch.zeros(len(y), n_classes, device=device, dtype=dtype)
        pos_idcs = [get_class_index(y_i) for y_i in y]
        for i, pos_idx in enumerate(pos_idcs):
            if isinstance(pos_idx, int) and pos_idx < n_classes:
                onehot[i, pos_idx] = 1
    else:
        onehot = torch.zeros(1, n_classes, device=device, dtype=dtype)
        pos_idx = classes.index(y)
        if isinstance(pos_idx, int) and pos_idx < n_classes:
            onehot[0, pos_idx] = 1

    return onehot


def output2proba(
    preds: torch.Tensor, classes: OrderedSet, output_is_logit=True
) -> List[Dict[ClfTarget, float]]:
    if output_is_logit:
        if preds.shape[-1] > 1:
            preds = torch.softmax(preds, dim=-1)
        else:
            preds = torch.sigmoid(preds)

    preds_np = preds.numpy(force=True)
    if preds_np.shape[-1] == 1:
        preds_np = np.hstack((preds_np, 1 - preds_np))

    # Assume a binary problem if no classes are known yet
    if len(classes) == 0:
        all_classes = OrderedSet([True, False])
    # Add opposite class if first label seems to be binary
    elif classes == [True] or classes == [False]:
        all_classes = classes.union([not classes[0]])
    # Add `unobserved` classes if pred shape is greater than number of
    # observed classes
    else:
        all_classes = classes.union(
            [f"Unobserved{i}" for i in range(preds_np.shape[1] - len(classes))]
        )

    return [dict(zip(all_classes, pred)) for pred in preds_np]
