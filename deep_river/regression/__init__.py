from deep_river.regression.multioutput import (
    MultiTargetRegressor,
)
from deep_river.regression.regressor import Regressor
from deep_river.regression.rolling_regressor import (
    RollingRegressor,
)
from deep_river.regression.zoo import (
    LinearRegression,
    LSTMRegressor,
    MultiLayerPerceptron,
    RNNRegressor
)

"""
This module contains the regressors for the deep_river package.
"""
__all__ = [
    "Regressor",
    "RollingRegressor",
    "MultiTargetRegressor",
    "LinearRegression",
    "LSTMRegressor",
    "MultiLayerPerceptron",
    "RNNRegressor",
]
