from importlib import import_module
from typing import TYPE_CHECKING, Any

from river import base

if TYPE_CHECKING:
    AnomalyDetector = Any
else:
    AnomalyDetector = getattr(base, "AnomalyDetector", None)
    if AnomalyDetector is None:
        AnomalyDetector = import_module("river.anomaly.base").AnomalyDetector
