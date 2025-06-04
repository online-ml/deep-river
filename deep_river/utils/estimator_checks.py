"""Utilities for unit testing and sanity checking estimators."""

import copy

__all__ = ["check_estimator"]

import typing

from river.checks import yield_checks, _yield_datasets, _wrapped_partial
from river.utils import inspect as river_inspect
import torch
import pytest
from deep_river.utils.inspect import isdeepestimator_initialized

def check_deep_learn_one(model, dataset):

    # Simulate a crash during backward pass
    def patched_backward(self, *args, **kwargs):
        original_backward(self, *args, **kwargs)
        raise RuntimeError("Simulated exception during backward pass")


    for x, y in dataset:
        original_backward = torch.Tensor.backward
        torch.Tensor.backward = patched_backward
        
        try:
        # First learn_one call - will raise exception after computing gradients
            with pytest.raises(RuntimeError):
                if model._supervised:
                    model.learn_one(x, y)
                else:
                    model.learn_one(x)
        finally:
            # Always restore the original function
            torch.Tensor.backward = original_backward
        
        for param in model.module.parameters():
            # New gradients were computed (not None)
            assert param.grad is not None, "learn_one() should compute gradients" 
            # They are valid (finite values)
            assert torch.all(torch.isfinite(param.grad)), "learn_one() should produce finite gradients"


def check_dict2tensor(model):
    x = {"a": 1, "b": 2, "c": 3}
    model._update_observed_features(x)
    input_len = model._get_input_size()
    lst = [1, 2, 3]
    lst.extend([0] * (input_len - 3))
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
    if isdeepestimator_initialized(model): #todo remove after refactoring for initilized modules
        dataset_checks = [
            check_deep_learn_one
        ]

        yield check_dict2tensor
        # Classifier checks
        if river_inspect.isclassifier(model) and not river_inspect.ismoclassifier(
            model
        ):
            yield check_dict2tensor

            if not model._multiclass:
                yield check_dict2tensor
        
        for dataset_check in dataset_checks:
            for dataset in _yield_datasets(model):
                yield _wrapped_partial(dataset_check, dataset=dataset)


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
