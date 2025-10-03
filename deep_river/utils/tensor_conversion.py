from typing import Deque, Dict, Hashable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from river import base
from river.base.typing import RegTarget
from sortedcontainers import SortedSet


def dict2tensor(
    x: dict,
    features: SortedSet,
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
    features: SortedSet,
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
    classes: SortedSet[base.typing.ClfTarget],
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
    preds: torch.Tensor, classes: SortedSet, output_is_logit: bool = True
) -> List[Dict[Hashable, float]]:
    is_probabilistic = output_is_logit
    if output_is_logit:
        if preds.shape[-1] > 1:
            preds = torch.softmax(preds, dim=-1)
        else:
            preds = torch.sigmoid(preds)

    preds_np = preds.detach().cpu().numpy()
    n_outputs = preds_np.shape[-1]
    n_classes = len(classes)

    def renorm_rows(arr):
        sums = arr.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return arr / sums

    # Boolean mode (binary classification) – always {False, True}
    boolean_mode = (
        all(c in (True, False) for c in classes) and n_outputs in (1, 2)
    ) or (n_classes == 0 and n_outputs in (1, 2))
    if boolean_mode:
        if n_outputs == 1:
            p_true = preds_np[:, 0].astype("float64")
            p_false = (1.0 - p_true).astype("float64")
            probs = np.stack([p_true, p_false], axis=1)
            probs = renorm_rows(probs)
            return [dict(zip([True, False], row.astype("float64"))) for row in probs]
        else:  # n_outputs == 2
            probs = preds_np.astype("float64")
            if is_probabilistic:
                probs = renorm_rows(probs)
            return [dict(zip([False, True], row.astype("float64"))) for row in probs]

    # Single-output (non-boolean) -> observed class + Unobserved0
    if n_outputs == 1:
        p_obs = preds_np[:, 0].astype("float64")
        p_un = (1.0 - p_obs).astype("float64")
        probs = np.stack([p_obs, p_un], axis=1)
        if is_probabilistic:
            probs = renorm_rows(probs)
        if n_classes == 0:
            labels: List[Hashable] = [0, 1]
        else:
            primary = list(classes)[0]
            labels = [primary, "Unobserved0"]  # mixed types intentional
        return [dict(zip(labels, row.astype("float64"))) for row in probs]

    # Multi-output handling (n_outputs > 1, non-boolean)
    if n_classes == 0:
        labels2: List[Hashable] = list(range(n_outputs))
        rows = preds_np
        if is_probabilistic:
            rows = renorm_rows(rows.astype("float64")).astype(rows.dtype)
        return [dict(zip(labels2, row)) for row in rows]

    labels3: List[Hashable] = list(classes)
    if len(labels3) < n_outputs:
        for i in range(n_outputs - len(labels3)):
            labels3.append(f"Unobserved{i}")  # type: ignore[list-item]
    else:
        labels3 = labels3[:n_outputs]

    rows = preds_np
    if is_probabilistic:
        rows = renorm_rows(rows.astype("float64")).astype(rows.dtype)
    return [dict(zip(labels3, row)) for row in rows]
