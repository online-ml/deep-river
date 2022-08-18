import abc

import numpy as np
from river import base
from river.anomaly import HalfSpaceTrees
from river.anomaly.base import AnomalyDetector
from river.stats import (EWMean, Max, Mean, Min, RollingMax, RollingMean,
                         RollingMin)

METRICS = {
    "mean": {"incremental": Mean, "rolling": RollingMean, "adaptive": EWMean},
    "max": {"incremental": Max, "rolling": RollingMax},
    "min": {"incremental": Min, "rolling": RollingMin},
}


def _get_metric(
        metric: str, metric_type: str, window_size: int = 250, alpha: float = 0.3
):
    """
    Returns the metric class for the given metric and metric type.

    Parameters
    ----------
    metric
        The metric to use.
    metric_type
        The type of metric to use.
    window_size
        The window size to use for the rolling metrics.
    alpha
        The alpha to use for the EWMean metric.

    Returns
    -------
    Callable
        The metric class.
    """

    assert metric in METRICS.keys(), f"Invalid metric: {metric}"
    assert (
            metric_type in METRICS[metric].keys()
    ), f"Invalid metric type {metric_type} for metric: {metric}"

    if metric_type == "incremental":
        return METRICS[metric][metric_type]()
    elif metric_type == "rolling":
        return METRICS[metric][metric_type](window_size=window_size)
    else:
        return METRICS[metric][metric_type](alpha=alpha)


class AnomalyScaler(base.Wrapper, AnomalyDetector):
    """Wrapper around an anomaly detector that scales the output of the model to account for drift in the wrapped model's anomaly scores.

    Parameters
    ----------
    anomaly_detector
        Anomaly detector to be wrapped.
    """

    def __init__(self, anomaly_detector: AnomalyDetector):
        self.anomaly_detector = anomaly_detector

    @classmethod
    def _unit_test_params(self) -> dict:
        """
        Returns a dictionary of parameters to be used for unit testing the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the respective class.
        """
        yield {"anomaly_detector": HalfSpaceTrees()}

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases, some estimators might not
        be able to pass certain checks.

        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
            "check_pickling",
            "check_shuffle_features_no_impact",
            "check_emerging_features",
            "check_disappearing_features",
            "check_predict_proba_one",
            "check_predict_proba_one_binary",
        }

    @property
    def _wrapped_model(self):
        return self.anomaly_detector

    @abc.abstractmethod
    def score_one(self, *args) -> float:
        """Return a scaled anomaly score based on raw score provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more anomalous examples.
        """

    def learn_one(self, *args) -> "AnomalyScaler":
        """
        Update the scaler and the underlying anomaly scaler.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        AnomalyScaler
            The model itself.
        """

        self.anomaly_detector.learn_one(*args)
        return self

    @abc.abstractmethod
    def score_many(self, *args) -> np.ndarray:
        """Return scaled anomaly scores based on raw score provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        Scaled anomaly scores. Larger values indicate more anomalous examples.
        """

    def learn_many(self, *args) -> "AnomalyScaler":
        """
        Update the scaler and the underlying anomaly scaler to a batch of examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        AnomalyScaler
            The model itself.
        """

        self.anomaly_detector.learn_many(*args)
        return self


class AnomalyStandardScaler(AnomalyScaler):
    """
    Wrapper around an anomaly detector that standardizes the model's output using incremental mean and variance metrics.

    Parameters
    ----------
    anomaly_detector
        The anomaly detector to wrap.
    with_std
        Whether to use standard deviation for scaling.
    metric_type
        The type of metric to use.
    window_size
        The window size used for the metrics if metric_type=="rolling".
    alpha
        The alpha used for the metrics if metric_type=="adaptive".
    """

    def __init__(
            self,
            anomaly_detector: AnomalyDetector,
            with_std: bool = True,
            metric_type: str = "rolling",
            window_size: int = 250,
            alpha: float = 0.3,
    ):
        super().__init__(anomaly_detector)
        self.metric_type = metric_type
        self.alpha = alpha
        self.window_size = window_size
        self.mean = _get_metric("mean", metric_type, window_size, alpha)
        self.sq_mean = _get_metric("mean", metric_type, window_size, alpha)
        self.with_std = with_std

    def score_one(self, *args):
        """
        Return a scaled anomaly score based on raw score provided by the wrapped anomaly detector. Larger values indicate more anomalous examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more anomalous examples.
        """
        raw_score = self.anomaly_detector.score_one(*args)
        mean = self.mean.update(raw_score).get()
        if self.with_std:
            var = self.sq_mean.update(raw_score ** 2).get() - mean ** 2
            score = (raw_score - mean) / var ** 0.5
        else:
            score = raw_score - mean

        return score

    def score_many(self, *args):
        """
        Return scaled anomaly scores based on raw scores provided by the wrapped anomaly detector. Larger values indicate more anomalous examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more anomalous examples.
        """
        raw_scores = self.anomaly_detector.score_many(*args)
        mean = self.mean.update_many(raw_scores).get()
        if self.with_std:
            var = self.sq_mean.update(raw_scores ** 2).get() - mean ** 2
            scores = (raw_scores - mean) / var ** 0.5
        else:
            scores = raw_scores - mean

        return scores


class AnomalyMeanScaler(AnomalyScaler):
    """Wrapper around an anomaly detector that scales the model's output by the incremental mean of previous scores.

    Parameters
    ----------
    anomaly_detector
        The anomaly detector to wrap.
    metric_type
        The type of metric to use.
    window_size
        The window size used for mean computation if metric_type=="rolling".
    alpha
        The alpha used for mean computation if metric_type=="adaptive".
    """

    def __init__(
            self,
            anomaly_detector: AnomalyDetector,
            metric_type: str = "rolling",
            window_size: int = 250,
            alpha: float = 0.3,
    ):
        super().__init__(anomaly_detector=anomaly_detector)
        self.metric_type = metric_type
        self.alpha = alpha
        self.window_size = window_size
        self.mean = _get_metric("mean", metric_type, window_size, alpha)

    def score_one(self, *args):
        """
        Return a scaled anomaly score based on raw score provided by the wrapped anomaly detector. Larger values indicate more anomalous examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more anomalous examples.
        """
        raw_score = self.anomaly_detector.score_one(*args)
        mean = self.mean.update(raw_score).get()
        score = raw_score / mean

        return score

    def score_many(self, *args):
        """
        Return scaled anomaly scores based on raw scores provided by the wrapped anomaly detector. Larger values indicate more anomalous examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more anomalous examples.
        """
        raw_score = self.anomaly_detector.score_many(*args)
        mean = self.mean.update_many(raw_score).get()
        score = raw_score / mean

        return score


class AnomalyMinMaxScaler(AnomalyScaler):
    """Wrapper around an anomaly detector that scales the model's output to $[0, 1]$ using rolling min and max metrics.

    Parameters
    ----------
    anomaly_detector
        The anomaly detector to wrap.
    metric_type
        The type of metric to use.
    window_size
        The window size used for the metrics if metric_type=="rolling".
    alpha
        The alpha used for the metrics if metric_type=="adaptive".
    """

    def __init__(
            self,
            anomaly_detector: AnomalyDetector,
            metric_type: str = "rolling",
            window_size: int = 250,
            alpha: float = 0.3,
    ):
        super().__init__(anomaly_detector)
        self.metric_type = metric_type
        self.alpha = alpha
        self.window_size = window_size
        self.min = _get_metric("min", metric_type, window_size, alpha)
        self.max = _get_metric("max", metric_type, window_size, alpha)

    def score_one(self, *args):
        """
        Return a scaled anomaly score based on raw score provided by the wrapped anomaly detector. Larger values indicate more anomalous examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more anomalous examples.
        """
        raw_score = self.anomaly_detector.score_one(*args)
        min = self.min.update(raw_score).get()
        max = self.max.update(raw_score).get()
        score = (raw_score - min) / (max - min)

        return score

    def score_many(self, *args):
        """
        Return scaled anomaly scores based on raw scores provided by the wrapped anomaly detector. Larger values indicate more anomalous examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more anomalous examples.
        """
        raw_scores = self.anomaly_detector.score_many(*args)
        for raw_score in raw_scores:
            self.min.update(raw_score)
            self.max.update(raw_score)

        min = self.min.get()
        max = self.max.get()
        score = (raw_score - min) / (max - min)

        return score
