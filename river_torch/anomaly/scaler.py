from .base import AnomalyScaler
from river.stats import EWMean, RollingMean, RollingMin, RollingMax, Mean, Min, Max


class StandardScaler(AnomalyScaler):
    def __init__(self, anomaly_detector, with_std=True):
        super().__init__(anomaly_detector)
        self.mean = Mean()
        self.sq_mean = Mean()
        self.with_std = with_std

    def score_one(self, *args) -> float:
        """Return calibrated anomaly score based on raw score provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.
        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.
        Returns
        -------
        An anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.
        """
        raw_score = self.anomaly_detector.score_one(*args)
        mean = self.mean.update(raw_score).get()
        if self.with_std:
            var = self.sq_mean.update(raw_score**2).get() - mean**2
            score = (raw_score - mean) / var**0.5
        else:
            score = raw_score - mean

        return score

    def score_many(self, *args):
        """Return calibrated anomaly scores based on raw scores provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.
        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.
        Returns
        -------
        Anomaly scores. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.
        """
        raw_scores = self.anomaly_detector.score_many(*args)
        mean = self.mean.update_many(raw_scores).get()
        if self.with_std:
            var = self.sq_mean.update(raw_scores**2).get() - mean**2
            scores = (raw_scores - mean) / var**0.5
        else:
            scores = raw_scores - mean

        return scores


class MeanScaler(AnomalyScaler):
    def __init__(self, anomaly_detector):
        super().__init__(anomaly_detector=anomaly_detector)
        self.mean = Mean()

    def score_one(self, *args) -> float:
        """Return calibrated anomaly score based on raw score provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.
        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.
        Returns
        -------
        Anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.
        """
        raw_score = self.anomaly_detector.score_one(*args)
        mean = self.mean.update(raw_score).get()
        score = raw_score / mean

        return score

    def score_many(self, *args) -> float:
        """Return calibrated anomaly scores based on raw scores provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.
        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.
        Returns
        -------
        Anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.
        """
        raw_score = self.anomaly_detector.score_many(*args)
        mean = self.mean.update_many(raw_score).get()
        score = raw_score / mean

        return score


class MinMaxScaler(AnomalyScaler):
    def __init__(self, anomaly_detector):
        super().__init__(anomaly_detector)
        self.min = Min()
        self.max = Max()

    def score_one(self, *args) -> float:
        """Return calibrated anomaly score based on raw score provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.
        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.
        Returns
        -------
        Anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.
        """
        raw_score = self.anomaly_detector.score_one(*args)
        min = self.min.update(raw_score).get()
        max = self.max.update(raw_score).get()
        score = (raw_score - min) / (max - min)

        return score

    def score_many(self, *args) -> float:
        """Return calibrated anomaly score based on raw score provided by the wrapped anomaly detector.

        A high score is indicative of an anomaly. A low score corresponds to a normal observation.
        Parameters
        ----------
        args
            Depends on whether the underlying anomaly detector is supervised or not.
        Returns
        -------
        Anomaly score. A high score is indicative of an anomaly. A low score corresponds a
        normal observation.
        """
        raw_scores = self.anomaly_detector.score_many(*args)
        for raw_score in raw_scores:
            self.min.update(raw_score)
            self.max.update(raw_score)

        min = self.min.get()
        max = self.max.get()
        score = (raw_score - min) / (max - min)

        return score


class RollingStandardScaler(StandardScaler):
    def __init__(self, anomaly_detector, window_size=250, with_std=True):
        super().__init__(anomaly_detector=anomaly_detector)
        self.window_size = window_size
        self.mean = RollingMean(window_size=window_size)
        self.sq_mean = RollingMean(window_size=window_size) if with_std else None
        self.with_std = with_std


class AdaptiveStandardScaler(StandardScaler):
    def __init__(self, anomaly_detector, alpha=0.3, with_std=True):
        super().__init__(anomaly_detector=anomaly_detector)
        self.apha = alpha
        self.mean = EWMean(alpha=alpha)
        self.sq_mean = EWMean(alpha=alpha) if with_std else None
        self.with_std = with_std


class RollingMinMaxScaler(MinMaxScaler):
    def __init__(self, anomaly_detector, window_size=250):
        super().__init__(anomaly_detector=anomaly_detector)
        self.window_size = window_size
        self.min = RollingMin(window_size=window_size)
        self.max = RollingMax(window_size=window_size)


class RollingMeanScaler(MeanScaler):
    def __init__(self, anomaly_detector, window_size=250):
        super().__init__(anomaly_detector)
        self.window_size = window_size
        self.mean = RollingMean(window_size=window_size)


class AdaptiveMeanScaler(MeanScaler):
    def __init__(self, anomaly_detector, alpha=0.3):
        super().__init__(anomaly_detector)
        self.alpha = alpha
        self.mean = EWMean(alpha=alpha)
