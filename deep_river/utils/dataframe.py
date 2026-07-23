from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import narwhals.stable.v2 as nw
import numpy as np


def into_frame(X: Any) -> nw.DataFrame[Any]:
    return cast(nw.DataFrame[Any], nw.from_native(X, eager_only=True))


def frame_columns(X: Any) -> list[Any]:
    return list(into_frame(X).columns)


def frame_to_numpy(
    X: Any,
    columns: Sequence[Any],
    default_value: float = 0.0,
    dtype: np.dtype[Any] | type = np.float32,
) -> np.ndarray[Any, Any]:
    frame = into_frame(X)
    n_rows = frame.shape[0]
    frame_columns = set(frame.columns)
    arrays = []
    for column in columns:
        if column in frame_columns:
            values = np.asarray(frame.select(nw.col(column)).to_numpy()).reshape(n_rows)
            values = np.asarray(values, dtype=dtype)
            values = np.where(np.isnan(values), default_value, values)
        else:
            values = np.full(n_rows, default_value, dtype=dtype)
        arrays.append(values)
    if not arrays:
        return np.empty((n_rows, 0), dtype=dtype)
    return np.column_stack(arrays)


def values_list(data: Any) -> list[Any] | None:
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (list, tuple)):
        return list(data)
    try:
        return nw.from_native(data, series_only=True).to_list()
    except TypeError:
        pass
    if hasattr(data, "to_pylist"):
        return data.to_pylist()
    return None


def values_to_numpy(
    data: Any, dtype: np.dtype[Any] | type = np.float32
) -> np.ndarray[Any, Any]:
    values = values_list(data)
    if values is None:
        values = [data]
    return np.asarray(values, dtype=dtype)


def to_native_series(values: Sequence[Any], *, name: str | None, like: Any) -> Any:
    frame = into_frame(like)
    series = nw.new_series(
        name=name if name is not None else "",
        values=values,
        backend=nw.get_native_namespace(frame),
    ).to_native()
    if name is None and frame.implementation.is_pandas_like():
        series.name = None
    if (index := nw.maybe_get_index(frame)) is not None:
        series.index = index
    return series


def to_native_frame(
    data: Mapping[Any, Sequence[Any]] | np.ndarray[Any, Any],
    *,
    like: Any,
    columns: Sequence[Any] | None = None,
) -> Any:
    frame_like = into_frame(like)
    implementation = frame_like.implementation
    if isinstance(data, np.ndarray):
        if columns is None:
            raise ValueError("`columns` must be provided when `data` is a numpy array.")
        names = (
            list(columns)
            if implementation.is_pandas_like()
            else [str(c) for c in columns]
        )
        frame = nw.from_numpy(
            np.ascontiguousarray(data),
            schema=names,
            backend=implementation,
        ).to_native()
    else:
        frame_data = data
        if not implementation.is_pandas_like():
            frame_data = {str(key): value for key, value in data.items()}
        frame = nw.from_dict(
            cast(Mapping[Any, Sequence[Any]], frame_data), backend=implementation
        ).to_native()
    if (index := nw.maybe_get_index(frame_like)) is not None:
        frame.index = index
    return frame
