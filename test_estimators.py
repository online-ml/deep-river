"""General tests that all estimators need to pass."""
import copy
import importlib
import inspect
import river
import pytest

from IncrementalTorch import base, classification, regression, anomaly, utils


def iter_estimators():
    for submodule in importlib.import_module("IncrementalTorch").__all__:

        def is_estimator(obj):
            return inspect.isclass(obj) and issubclass(obj, river.base.Estimator)

        for _, obj in inspect.getmembers(
            importlib.import_module(f"IncrementalTorch.{submodule}"), is_estimator
        ):
            yield obj


def iter_estimators_which_can_be_tested():

    ignored = (

    )

    def can_be_tested(estimator):
        return not inspect.isabstract(estimator) and not issubclass(estimator, ignored)

    for estimator in filter(can_be_tested, iter_estimators()):
        for params in estimator._unit_test_params():
            yield estimator(**params)


@pytest.mark.parametrize(
    "estimator, check",
    [
        pytest.param(estimator, check, id=f"{estimator}:{check.__name__}")
        for estimator in list(iter_estimators_which_can_be_tested())

        for check in utils.estimator_checks.yield_checks(estimator)
        if check.__name__ not in estimator._unit_test_skips()
    ],
)
def test_check_estimator(estimator, check):
    check(copy.deepcopy(estimator))