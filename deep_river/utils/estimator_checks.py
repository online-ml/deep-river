"""Utilities for unit testing and sanity checking estimators."""
import copy

__all__ = ["check_estimator"]

from river.checks import yield_checks


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
        check(copy.deepcopy(model))
