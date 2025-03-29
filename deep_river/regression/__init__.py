from deep_river.regression.multioutput import (
    MultiTargetRegressor,
    MultiTargetRegressorInitialized,
)
from deep_river.regression.regressor import Regressor, RegressorInitialized
from deep_river.regression.rolling_regressor import (
    RollingRegressor,
    RollingRegressorInitialized,
)

"""
This module contains the regressors for the deep_river package.
"""
__all__ = [
    "Regressor",
    "RegressorInitialized",
    "RollingRegressor",
    "RollingRegressorInitialized",
    "MultiTargetRegressor",
    "MultiTargetRegressorInitialized",
]
