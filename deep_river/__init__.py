from importlib.metadata import version as _version

from . import anomaly, classification, forecasting, regression, utils

__version__ = _version("deep-river")

__all__ = ["anomaly", "classification", "forecasting", "regression", "utils"]
