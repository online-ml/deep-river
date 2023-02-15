from deep_river.regression.regressor import Regressor
from deep_river.regression.rolling_regressor import RollingRegressor

# isort: split
from deep_river.regression.multioutput import MultiTargetRegressor
from deep_river.regression.zoo import LinearRegression, MultiLayerPerceptron

"""
This module contains the regressors for the deep_river package.
"""
__all__ = [
    "Regressor",
    "RollingRegressor",
    "MultiTargetRegressor",
    "LinearRegression",
    "MultiLayerPerceptron",
]
