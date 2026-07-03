import datetime as dt
import itertools
import statistics
import time

from river import datasets, metrics, time_series
from river.evaluate import Track


class LimitedDataset:
    """Wrapper um einen (ggf. unendlichen) river Dataset, der nur die ersten n Samples liefert.

    Delegiert alle Attribute an das Original-Dataset, überschreibt nur __iter__.
    Stellt zusätzlich ein *n_samples* Attribut bereit, sodass river.evaluate.Track.run
    eine feste Länge erkennt (verhindert Fehler bei None // int).
    """

    def __init__(self, base, n):
        self._base = base
        self._n = int(n)
        # Name wie im Original, aber markiert als limitiert
        self.dataset_name = f"{base.__class__.__name__} (limited {self._n})"
        # Einige river Datasets haben n_samples=None (endlos). Wir setzen hier die Kappung.
        self.n_samples = self._n

    def __iter__(self):
        return itertools.islice(self._base, self._n)

    def __len__(self):  # optional, falls irgendwo len(dataset) genutzt wird
        return self._n

    def __getattr__(self, item):
        # Attribute des Basis-Datasets durchreichen (z.B. task, n_features, etc.)
        return getattr(self._base, item)

    def __repr__(self):
        return f"{self._base.__class__.__name__}(limited n={self._n})"


def limit_dataset(dataset, n):
    return LimitedDataset(dataset, n)


class TargetOnlyDataset:
    """Wrapper that exposes a dataset as a univariate forecasting stream."""

    def __init__(self, base):
        self._base = base
        self.dataset_name = f"{base.__class__.__name__} (target only)"
        self.n_samples = base.n_samples

    def __iter__(self):
        for _, y in self._base:
            yield {}, y

    def __len__(self):
        return self.n_samples

    def __getattr__(self, item):
        return getattr(self._base, item)

    def __repr__(self):
        return (
            f"{repr(self._base)}\n\n"
            "Forecasting benchmark variant: only the historical target values are used."
        )


def _iter_with_horizon(dataset, horizon: int):
    """Yield current observations together with the next horizon of targets."""
    x_horizon = []
    y_horizon = []
    stream = iter(dataset)

    for _ in range(horizon):
        x, y = next(stream)
        x_horizon.append(x)
        y_horizon.append(y)

    for x, y in stream:
        x_now = x_horizon.pop(0)
        y_now = y_horizon.pop(0)
        x_horizon.append(x)
        y_horizon.append(y)
        yield x_now, y_now, list(x_horizon), list(y_horizon)


class BinaryClassificationTrack(Track):
    """This track evaluates a model's performance on binary classification tasks.
    These do not include synthetic datasets.

    Parameters
    ----------
    n_samples
        The number of samples to use for each dataset.

    """

    def __init__(self):
        super().__init__(
            name="Binary classification",
            datasets=[datasets.Bananas(), datasets.Elec2(), datasets.Phishing()],
            metric=metrics.Accuracy() + metrics.F1(),
        )


class MultiClassClassificationTrack(Track):
    """This track evaluates a model's performance on multi-class classification tasks.
    These do not include synthetic datasets.

    Parameters
    ----------
    n_samples
        The number of samples to use for each dataset.

    """

    def __init__(self, n_samples: int = 5000):
        self.n_samples = int(n_samples)
        datasets_limited = [
            limit_dataset(
                datasets.synth.Hyperplane(seed=42, n_features=30), self.n_samples
            ),
            limit_dataset(
                datasets.synth.LED(
                    seed=112, noise_percentage=0.28, irrelevant_features=False
                ),
                self.n_samples,
            ),
            limit_dataset(
                datasets.synth.RandomRBF(
                    seed_model=42,
                    seed_sample=42,
                    n_classes=4,
                    n_features=4,
                    n_centroids=20,
                ),
                self.n_samples,
            ),
        ]
        super().__init__(
            name="Multiclass classification",
            datasets=datasets_limited,
            metric=metrics.Accuracy() + metrics.MicroF1() + metrics.MacroF1(),
        )

        # Sicherheitsnetz: Falls Track.run intern n_samples nutzt und step=0 herauskäme,
        # könnte das Probleme machen. Wir stellen sicher, dass n_samples >= Anzahl Checkpoints.
        for ds in self.datasets:
            if getattr(ds, "n_samples", self.n_samples) < 1:
                ds.n_samples = self.n_samples


class RegressionTrack(Track):
    """This track evaluates a model's performance on regression tasks.
    These do not include synthetic datasets.

    Parameters
    ----------
    n_samples
        The number of samples to use for each dataset.

    """

    def __init__(self):
        super().__init__(
            "Regression",
            datasets=[
                datasets.ChickWeights(),
                datasets.TrumpApproval(),
            ],
            metric=metrics.MAE() + metrics.RMSE() + metrics.R2(),
        )


class ForecastingTrack(Track):
    """This track evaluates forecasters on sequential time-series tasks."""

    def __init__(self, horizon: int = 3, grace_period: int = 12):
        self.horizon = horizon
        self.grace_period = grace_period
        self.forecasting_metrics = [metrics.MAE(), metrics.RMSE(), metrics.SMAPE()]
        super().__init__(
            "Forecasting",
            datasets=[
                TargetOnlyDataset(datasets.AirlinePassengers()),
                TargetOnlyDataset(datasets.TrumpApproval()),
            ],
            metric=metrics.MAE(),
        )

    def run(self, model, dataset, n_checkpoints=10):
        model = model.clone()
        horizon_metrics = {
            metric.__class__.__name__: time_series.HorizonAggMetric(
                metric.clone(), statistics.mean
            )
            for metric in self.forecasting_metrics
        }

        steps = _iter_with_horizon(dataset, self.horizon)
        for _ in range(self.grace_period):
            x, y, _, _ = next(steps)
            model.learn_one(y=y, x=x)

        step = max(1, dataset.n_samples // n_checkpoints)
        next_checkpoint = step
        previous_checkpoint = None
        n_evaluations = 0
        start = time.perf_counter()

        def report():
            state = dict(horizon_metrics)
            state["Step"] = n_evaluations
            state["Time"] = dt.timedelta(seconds=time.perf_counter() - start)
            state["Memory"] = model._raw_memory_usage
            return state

        for x, y, x_horizon, y_horizon in steps:
            y_pred = model.forecast(self.horizon, xs=x_horizon)
            for horizon_metric in horizon_metrics.values():
                horizon_metric.update(y_horizon, y_pred)
            model.learn_one(y=y, x=x)

            n_evaluations += 1
            if n_evaluations == next_checkpoint:
                yield report()
                previous_checkpoint = next_checkpoint
                next_checkpoint += step

        if previous_checkpoint and n_evaluations != previous_checkpoint:
            yield report()
