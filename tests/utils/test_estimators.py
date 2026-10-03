"""General tests that all estimators need to pass."""

import copy
import itertools

import pytest

from deep_river import utils


@pytest.mark.parametrize(
    "estimator, check",
    [
        pytest.param(estimator, check, id=f"{estimator}:{check.__name__}")
        for estimator in list(
            utils.estimator_checks.iter_estimators_that_can_be_tested()
        )
        for check in itertools.chain(
            utils.estimator_checks.yield_checks(estimator),
            utils.estimator_checks.yield_deep_checks(estimator),
        )
        if check.__name__ not in estimator._unit_test_skips()
    ],
)
def test_check_estimator(estimator, check):
    check(copy.deepcopy(estimator))
