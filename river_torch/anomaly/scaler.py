import abc

import numpy as np
from river import base, utils
from river.anomaly import HalfSpaceTrees
from river.anomaly.base import AnomalyDetector
from river.stats import Mean, Min


class AnomalyScaler(base.Wrapper, AnomalyDetector):
    """Wrapper around an anomaly detector that scales the output of the model
    to account for drift in the wrapped model's anomaly scores.

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
        Returns a dictionary of parameters to be used for unit testing
        the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the
            respective class.
        """
        return {"anomaly_detector": HalfSpaceTrees()}

    @classmethod
    def _unit_test_skips(self) -> set:
        """
        Indicates which checks to skip during unit testing.
        Most estimators pass the full test suite. However, in some cases,
        some estimators might not
        be able to pass certain checks.

        Returns
        -------
        set
            Set of checks to skip during unit testing.
        """
        return {
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
        """Return a scaled anomaly score based on raw score provided by
        the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds
        to a normal observation.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector
            is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate
        more anomalous examples.
        """

    def learn_one(self, *args) -> "AnomalyScaler":
        """
        Update the scaler and the underlying anomaly scaler.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector
            is supervised or not.

        Returns
        -------
        AnomalyScaler
            The model itself.
        """

        self.anomaly_detector.learn_one(*args)
        return self

    @abc.abstractmethod
    def score_many(self, *args) -> np.ndarray:
        """Return scaled anomaly scores based on raw score provided by
        the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds
        to a normal observation.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is
            supervised or not.

        Returns
        -------
        Scaled anomaly scores. Larger values indicate more anomalous examples.
        """


class AnomalyStandardScaler(AnomalyScaler):
    """
    Wrapper around an anomaly detector that standardizes the model's output
    using incremental mean and variance metrics.

    Parameters
    ----------
    anomaly_detector
        The anomaly detector to wrap.
    with_std
        Whether to use standard deviation for scaling.
    rolling
        Choose whether the metrics are rolling metrics or not.
    window_size
        The window size used for the metrics if rolling==True.
    """

    def __init__(
        self,
        anomaly_detector: AnomalyDetector,
        with_std: bool = True,
        rolling: bool = True,
        window_size: int = 250,
    ):
        super().__init__(anomaly_detector)
        self.rolling = rolling
        self.window_size = window_size
        self.mean = (
            utils.Rolling(Mean(), self.window_size) if self.rolling else Mean()
        )
        self.sq_mean = (
            utils.Rolling(Mean(), self.window_size) if self.rolling else Mean()
        )
        self.with_std = with_std

    def score_one(self, *args):
        """
        Return a scaled anomaly score based on raw score provided by the
        wrapped anomaly detector. Larger values indicate more
        anomalous examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector
            is supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more
        anomalous examples.
        """
        raw_score = self.anomaly_detector.score_one(*args)
        mean = self.mean.update(raw_score).get()
        if self.with_std:
            var = (
                self.sq_mean.update(raw_score**2).get() - mean**2
            )  # todo is this correct?
            score = (raw_score - mean) / var**0.5
        else:
            score = raw_score - mean

        return score


class AnomalyMeanScaler(AnomalyScaler):
    """Wrapper around an anomaly detector that scales the model's output
    by the incremental mean of previous scores.

    Parameters
    ----------
    anomaly_detector
        The anomaly detector to wrap.
    metric_type
        The type of metric to use.
    rolling
        Choose whether the metrics are rolling metrics or not.
    window_size
        The window size used for mean computation if rolling==True.
    """

    def __init__(
        self,
        anomaly_detector: AnomalyDetector,
        rolling: bool = True,
        window_size=250,
    ):
        super().__init__(anomaly_detector=anomaly_detector)
        self.rolling = rolling
        self.window_size = window_size
        self.mean = (
            utils.Rolling(Mean(), self.window_size) if self.rolling else Mean()
        )

    def score_one(self, *args):
        """
        Return a scaled anomaly score based on raw score provided by the
        wrapped anomaly detector. Larger values indicate more
        anomalous examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is
            supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more
        anomalous examples.
        """
        raw_score = self.anomaly_detector.score_one(*args)
        mean = self.mean.update(raw_score).get()
        score = raw_score / mean

        return score


class AnomalyMinMaxScaler(AnomalyScaler):
    """Wrapper around an anomaly detector that scales the model's output to
    $[0, 1]$ using rolling min and max metrics.

    Parameters
    ----------
    anomaly_detector
        The anomaly detector to wrap.
    rolling
        Choose whether the metrics are rolling metrics or not.
    window_size
        The window size used for the metrics if rolling==True
    """

    def __init__(
        self,
        anomaly_detector: AnomalyDetector,
        rolling: bool = True,
        window_size: int = 250,
    ):
        super().__init__(anomaly_detector)
        self.rolling = rolling
        self.window_size = window_size
        self.min = (
            utils.Rolling(Min(), self.window_size) if self.rolling else Min()
        )
        self.max = (
            utils.Rolling(Min(), self.window_size) if self.rolling else Min()
        )

    def score_one(self, *args):
        """
        Return a scaled anomaly score based on raw score provided by the
        wrapped anomaly detector. Larger values indicate more
        anomalous examples.

        Parameters
        ----------
        *args
            Depends on whether the underlying anomaly detector is
            supervised or not.

        Returns
        -------
        An scaled anomaly score. Larger values indicate more
        anomalous examples.
        """
        raw_score = self.anomaly_detector.score_one(*args)
        min = self.min.update(raw_score).get()
        max = self.max.update(raw_score).get()
        score = (raw_score - min) / (max - min)

        return score
