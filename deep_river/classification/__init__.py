from deep_river.classification.classifier import Classifier
from deep_river.classification.rolling_classifier import RollingClassifier
from deep_river.classification.zoo import LogisticRegression, MultiLayerPerceptron

"""
This module contains the classifiers for the deep_river package.
"""
__all__ = [
    "Classifier",
    "RollingClassifier",
    "MultiLayerPerceptron",
    "LogisticRegression",
]
