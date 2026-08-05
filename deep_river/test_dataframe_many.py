import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from deep_river.classification import Classifier
from deep_river.regression import Regressor


class RegressionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.1, -0.2, 0.3, 0.4]]))

    def forward(self, x):
        return self.linear(x)


class ClassificationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[0.2, -0.1, 0.4, 0.3], [-0.3, 0.5, -0.2, 0.1]])
            )

    def forward(self, x):
        return self.linear(x)


def make_frame(data, backend, index=None):
    if backend == "pandas":
        return pd.DataFrame(data, index=index)
    if backend == "polars":
        return pl.DataFrame(data)
    if backend == "pyarrow":
        return pa.table(data)
    raise ValueError(backend)


def make_series(values, backend, index=None, name="target"):
    if backend == "pandas":
        return pd.Series(values, index=index, name=name)
    if backend == "polars":
        return pl.Series(name, values)
    if backend == "pyarrow":
        return pa.chunked_array([values])
    raise ValueError(backend)


def series_values(series):
    if isinstance(series, pd.Series):
        return series.to_numpy()
    if isinstance(series, pl.Series):
        return series.to_numpy()
    if isinstance(series, pa.ChunkedArray):
        return np.asarray(series.to_pylist())
    raise TypeError(type(series))


def frame_values(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.to_numpy()
    if isinstance(frame, pl.DataFrame):
        return frame.to_numpy()
    if isinstance(frame, pa.Table):
        return frame.to_pandas().to_numpy()
    raise TypeError(type(frame))


def assert_series_backend(series, backend, index):
    if backend == "pandas":
        assert isinstance(series, pd.Series)
        assert series.index.equals(index)
    elif backend == "polars":
        assert isinstance(series, pl.Series)
    else:
        assert isinstance(series, pa.ChunkedArray)


def assert_frame_backend(frame, backend, index):
    if backend == "pandas":
        assert isinstance(frame, pd.DataFrame)
        assert frame.index.equals(index)
    elif backend == "polars":
        assert isinstance(frame, pl.DataFrame)
    else:
        assert isinstance(frame, pa.Table)


def train_regressor(backend):
    model = Regressor(
        module=RegressionModule(),
        loss_fn="mse",
        optimizer_fn="sgd",
        lr=0.01,
    )
    model.learn_many(
        make_frame(
            {"a": [1.0, 2.0, None], "b": [0.5, np.nan, 1.5], "c": [2.0, 1.0, 0.0]},
            backend,
            index=pd.Index(["i0", "i1", "i2"]),
        ),
        make_series([1.0, 0.0, 1.0], backend, index=pd.Index(["i0", "i1", "i2"])),
    )
    model.learn_many(
        make_frame(
            {"d": [3.0, 1.0], "b": [2.0, None]},
            backend,
            index=pd.Index(["i3", "i4"]),
        ),
        make_series([0.5, 1.5], backend, index=pd.Index(["i3", "i4"])),
    )
    return model


def train_classifier(backend):
    model = Classifier(
        module=ClassificationModule(),
        loss_fn="cross_entropy",
        optimizer_fn="sgd",
        lr=0.01,
        output_is_logit=True,
    )
    model.learn_many(
        make_frame(
            {"a": [1.0, 2.0, None], "b": [0.5, np.nan, 1.5], "c": [2.0, 1.0, 0.0]},
            backend,
            index=pd.Index(["i0", "i1", "i2"]),
        ),
        make_series(
            ["good", "bad", "good"], backend, index=pd.Index(["i0", "i1", "i2"])
        ),
    )
    model.learn_many(
        make_frame(
            {"d": [3.0, 1.0], "b": [2.0, None]},
            backend,
            index=pd.Index(["i3", "i4"]),
        ),
        make_series(["bad", "good"], backend, index=pd.Index(["i3", "i4"])),
    )
    return model


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_regressor_many_backends_match_pandas(backend):
    index = pd.Index(["p0", "p1"])
    X = make_frame(
        {"d": [1.0, 0.0], "c": [np.nan, 2.0], "b": [3.0, None], "a": [4.0, 5.0]},
        backend,
        index=index,
    )
    expected = train_regressor("pandas").predict_many(
        make_frame(
            {"d": [1.0, 0.0], "c": [np.nan, 2.0], "b": [3.0, None], "a": [4.0, 5.0]},
            "pandas",
            index=index,
        )
    )
    actual = train_regressor(backend).predict_many(X)
    assert_series_backend(actual, backend, index)
    assert np.allclose(series_values(actual), series_values(expected))


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_classifier_many_backends_match_pandas(backend):
    index = pd.Index(["p0", "p1"])
    X = make_frame(
        {"d": [1.0, 0.0], "c": [np.nan, 2.0], "b": [3.0, None], "a": [4.0, 5.0]},
        backend,
        index=index,
    )
    expected = train_classifier("pandas").predict_proba_many(
        make_frame(
            {"d": [1.0, 0.0], "c": [np.nan, 2.0], "b": [3.0, None], "a": [4.0, 5.0]},
            "pandas",
            index=index,
        )
    )
    actual = train_classifier(backend).predict_proba_many(X)
    assert_frame_backend(actual, backend, index)
    assert np.allclose(frame_values(actual), frame_values(expected))
