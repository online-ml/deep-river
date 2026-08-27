"""
This module contains the classifiers for the deep_river package.
"""

from deep_river.classification.classifier import Classifier
from deep_river.classification.rolling_classifier import (
    RollingClassifier,
)
from deep_river.classification.zoo import (
    LogisticRegression,
    LSTMClassifier,
    MultiLayerPerceptron,
    RNNClassifier,
)

__all__ = [
    "Classifier",
    "RollingClassifier",
    "LogisticRegression",
    "MultiLayerPerceptron",
    "LSTMClassifier",
    "RNNClassifier",
]
