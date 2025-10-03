from river.evaluate import Track
from river import datasets, metrics
import itertools

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
            datasets=[
                datasets.Bananas(),
                datasets.Elec2(),
                datasets.Phishing()
            ],
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
            limit_dataset(datasets.synth.Hyperplane(seed=42, n_features=30), self.n_samples),
            limit_dataset(datasets.synth.LED(seed=112, noise_percentage=0.28, irrelevant_features=False), self.n_samples),
            limit_dataset(datasets.synth.RandomRBF(seed_model=42, seed_sample=42, n_classes=4, n_features=4, n_centroids=20), self.n_samples)
        ]
        super().__init__(
            name="Multiclass classification",
            datasets=datasets_limited,
            metric=metrics.Accuracy() + metrics.MicroF1() + metrics.MacroF1(),
        )

        # Sicherheitsnetz: Falls Track.run intern n_samples nutzt und step=0 herauskäme,
        # könnte das Probleme machen. Wir stellen sicher, dass n_samples >= Anzahl Checkpoints.
        for ds in self.datasets:
            if getattr(ds, 'n_samples', self.n_samples) < 1:
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