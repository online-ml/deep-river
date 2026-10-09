import abc

import numpy as np
from river import base, utils
from river.anomaly import HalfSpaceTrees
from river.base import AnomalyDetector
from river.stats import Max, Mean, Min, RollingMax, RollingMin, Var


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
    def _unit_test_params(cls):
        """
        Returns a dictionary of parameters to be used for unit testing
        the respective class.

        Yields
        -------
        dict
            Dictionary of parameters to be used for unit testing the
            respective class.
        """
        yield {"anomaly_detector": HalfSpaceTrees()}

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
    def score_one(self, *args, **kwargs) -> float:
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

    def learn_one(self, *args, **kwargs) -> None:
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

        self._update_score(self.anomaly_detector.score_one(*args, **kwargs))
        self.anomaly_detector.learn_one(*args, **kwargs)

    @abc.abstractmethod
    def _update_score(self, score: float) -> None:
        pass

    @abc.abstractmethod
    def score_many(self, *args, **kwargs) -> np.ndarray:
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
        self.mean = utils.Rolling(Mean(), self.window_size) if self.rolling else Mean()
        self.var = (
            utils.Rolling(Var(ddof=0), self.window_size)
            if self.rolling
            else Var(ddof=0)
        )
        self.with_std = with_std

    def score_one(self, *args, **kwargs):
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
        raw_score = self.anomaly_detector.score_one(*args, **kwargs)
        mean = self.mean.get()
        if not self.with_std:
            return raw_score - mean
        var = self.var.get()
        return (raw_score - mean) / var**0.5 if var > 0 else 0.0

    def _update_score(self, score: float) -> None:
        self.mean.update(score)
        self.var.update(score)


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
        window_size: int = 250,
    ):
        super().__init__(anomaly_detector=anomaly_detector)
        self.rolling = rolling
        self.window_size = window_size
        self.mean = utils.Rolling(Mean(), self.window_size) if self.rolling else Mean()

    def score_one(self, *args, **kwargs):
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
        raw_score = self.anomaly_detector.score_one(*args, **kwargs)
        mean = self.mean.get()
        return raw_score / mean if mean else 0.0

    def _update_score(self, score: float) -> None:
        self.mean.update(score)


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
        self.min = RollingMin(self.window_size) if self.rolling else Min()
        self.max = RollingMax(self.window_size) if self.rolling else Max()

    def score_one(self, *args, **kwargs):
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
        raw_score = self.anomaly_detector.score_one(*args, **kwargs)
        minimum = self.min.get()
        maximum = self.max.get()
        if minimum is None or maximum is None:
            return 0.0
        return (raw_score - minimum) / (maximum - minimum) if maximum > minimum else 0.0

    def _update_score(self, score: float) -> None:
        self.min.update(score)
        self.max.update(score)
