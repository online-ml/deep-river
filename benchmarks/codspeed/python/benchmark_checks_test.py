import pytest
import torch
from torch import nn

from deep_river.anomaly import Autoencoder
from deep_river.classification import Classifier
from deep_river.regression import Regressor
from deep_river.utils.estimator_checks import yield_benchmark_checks


class ClassifierBenchmarkModule(nn.Module):
    def __init__(self, n_features: int = 6, n_outputs: int = 2):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.hidden = nn.Linear(n_features, 8)
        self.output = nn.Linear(8, n_outputs)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))


class RegressorBenchmarkModule(nn.Module):
    def __init__(self, n_features: int = 6):
        super().__init__()
        self.n_features = n_features
        self.hidden = nn.Linear(n_features, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))


class AutoencoderBenchmarkModule(nn.Module):
    def __init__(self, n_features: int = 6, latent_dim: int = 3):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(n_features, latent_dim)
        self.decoder = nn.Linear(latent_dim, n_features)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))


def benchmark_estimators() -> dict[str, object]:
    torch.manual_seed(42)
    torch.set_num_threads(1)
    return {
        "classification": Classifier(
            module=ClassifierBenchmarkModule(),
            loss_fn="cross_entropy",
            optimizer_fn="sgd",
            lr=0.01,
            is_class_incremental=True,
            is_feature_incremental=True,
            seed=42,
        ),
        "regression": Regressor(
            module=RegressorBenchmarkModule(),
            loss_fn="mse",
            optimizer_fn="sgd",
            lr=0.01,
            is_feature_incremental=True,
            seed=42,
        ),
        "anomaly": Autoencoder(
            module=AutoencoderBenchmarkModule(),
            loss_fn="mse",
            optimizer_fn="sgd",
            lr=0.01,
            is_feature_incremental=True,
            seed=42,
        ),
    }


@pytest.mark.parametrize(
    "estimator, check",
    [
        pytest.param(
            estimator,
            check,
            id=f"{group}:{check.__name__}",
            marks=pytest.mark.benchmark(group=group),
        )
        for group, estimator in benchmark_estimators().items()
        for check in yield_benchmark_checks(estimator)
    ],
)
def test_benchmark_check(estimator, check, benchmark):
    check(estimator, benchmark)
