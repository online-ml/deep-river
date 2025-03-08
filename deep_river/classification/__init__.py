from deep_river.classification.classifier import Classifier, ClassifierInitialized
from deep_river.classification.rolling_classifier import RollingClassifier
from deep_river.classification.zoo import (
    LogisticRegression,
    LogisticRegressionInitialized,
    LSTMClassifierInitialized,
    MultiLayerPerceptron,
    MultiLayerPerceptronInitialized,
)

"""
This module contains the classifiers for the deep_river package.
"""
__all__ = [
    "Classifier",
    "ClassifierInitialized",
    "RollingClassifier",
    "MultiLayerPerceptron",
    "LogisticRegression",
    "LogisticRegressionInitialized",
    "MultiLayerPerceptronInitialized",
    "LSTMClassifierInitialized",
]
