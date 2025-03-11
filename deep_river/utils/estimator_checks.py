"""Utilities for unit testing and sanity checking estimators."""

import copy

__all__ = ["check_estimator"]

import typing

from river.checks import yield_checks
from river.utils import inspect as river_inspect

from deep_river.utils.inspect import isdeepestimator_initialized


def check_dict2tensor(model):
    x = {"a": 1, "b": 2, "c": 3}
    model._update_observed_features(x)
    input_len = model._get_input_size()
    lst = [1, 2, 3]
    lst.extend([0] * (input_len - 3))
    print(f"input_len: {input_len}")
    print(model._dict2tensor(x).tolist())
    assert model._dict2tensor(x).tolist() == [lst]

    x2 = {"b": 2, "c": 3}
    lst = [0, 2, 3]
    lst.extend([0] * (input_len - 3))
    assert model._dict2tensor(x2).tolist() == [lst]

    x3 = {"b": 2, "a": 1, "c": 3}
    lst = [1, 2, 3]
    lst.extend([0] * (input_len - 3))
    assert model._dict2tensor(x3).tolist() == [lst]


def yield_deep_checks(model) -> typing.Iterator[typing.Callable]:
    """Generates unit tests for a given model.

    Parameters
    ----------
    model

    """
    if isdeepestimator_initialized(model):
        yield check_dict2tensor
        # Classifier checks
        if river_inspect.isclassifier(model) and not river_inspect.ismoclassifier(
            model
        ):
            yield check_dict2tensor

            if not model._multiclass:
                yield check_dict2tensor


def check_estimator(model):
    """Check if a model adheres to `river`'s conventions.
    This will run a series of unit tests. The nature of the unit tests
    depends on the type of model.
    Parameters
    ----------
    model
    """
    for check in yield_checks(model):
        if check.__name__ in model._unit_test_skips():
            continue
        check(copy.deepcopy(model))  # todo change to clone

    for check in yield_deep_checks(model):
        if check.__name__ in model._unit_test_skips():
            continue
        check(copy.deepcopy(model))  # todo change to clone
