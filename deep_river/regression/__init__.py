from deep_river.regression.regressor import Regressor, RegressorInitialized
from deep_river.regression.rolling_regressor import (
    RollingRegressor,
    RollingRegressorInitialized,
)

# isort: split
from deep_river.regression.multioutput import MultiTargetRegressor
from deep_river.regression.zoo import LinearRegression, MultiLayerPerceptron

"""
This module contains the regressors for the deep_river package.
"""
__all__ = [
    "Regressor",
    "RegressorInitialized",
    "RollingRegressor",
    "RollingRegressorInitialized",
    "MultiTargetRegressor",
    "LinearRegression",
    "MultiLayerPerceptron",
]
