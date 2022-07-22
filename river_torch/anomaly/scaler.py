from river.stats import EWMean, Max, Mean, Min, RollingMax, RollingMean, RollingMin
from river.anomaly.base import AnomalyDetector

from .base import AnomalyScaler

METRICS = {
    "mean": {"incremental": Mean, "rolling": RollingMean, "adaptive": EWMean},
    "max": {"incremental": Max, "rolling": RollingMax},
    "min": {"incremental": Min, "rolling": RollingMin},
}


def get_metric(
    metric: str, metric_type: str, window_size: int = 250, alpha: float = 0.3
):
    """Get the metric class for the given metric and metric type.

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

    Returns:
    --------
    metric_class: river.stats.Statistic
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


class StandardScaler(AnomalyScaler):
    """Wrapper around an anomaly detector that standardizes the model's output using incremental mean and variance metrics.

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
        self.mean = get_metric("mean", metric_type, window_size, alpha)
        self.sq_mean = get_metric("mean", metric_type, window_size, alpha)
        self.with_std = with_std

    def score_one(self, *args):
        raw_score = self.anomaly_detector.score_one(*args)
        mean = self.mean.update(raw_score).get()
        if self.with_std:
            var = self.sq_mean.update(raw_score**2).get() - mean**2
            score = (raw_score - mean) / var**0.5
        else:
            score = raw_score - mean

        return score

    def score_many(self, *args):
        raw_scores = self.anomaly_detector.score_many(*args)
        mean = self.mean.update_many(raw_scores).get()
        if self.with_std:
            var = self.sq_mean.update(raw_scores**2).get() - mean**2
            scores = (raw_scores - mean) / var**0.5
        else:
            scores = raw_scores - mean

        return scores


class MeanScaler(AnomalyScaler):
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
        self.mean = get_metric("mean", metric_type, window_size, alpha)

    def score_one(self, *args):
        raw_score = self.anomaly_detector.score_one(*args)
        mean = self.mean.update(raw_score).get()
        score = raw_score / mean

        return score

    def score_many(self, *args):
        raw_score = self.anomaly_detector.score_many(*args)
        mean = self.mean.update_many(raw_score).get()
        score = raw_score / mean

        return score


class MinMaxScaler(AnomalyScaler):
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
        self.min = get_metric("min", metric_type, window_size, alpha)
        self.max = get_metric("max", metric_type, window_size, alpha)

    def score_one(self, *args):
        raw_score = self.anomaly_detector.score_one(*args)
        min = self.min.update(raw_score).get()
        max = self.max.update(raw_score).get()
        score = (raw_score - min) / (max - min)

        return score

    def score_many(self, *args):
        raw_scores = self.anomaly_detector.score_many(*args)
        for raw_score in raw_scores:
            self.min.update(raw_score)
            self.max.update(raw_score)

        min = self.min.get()
        max = self.max.get()
        score = (raw_score - min) / (max - min)

        return score
