from river import (dummy, linear_model, neural_net, optim,
                   preprocessing, stats, compose)

from deep_river.classification.zoo import LogisticRegressionInitialized, MultiLayerPerceptronInitialized as ClassificationMLP
from deep_river.regression.zoo import LinearRegressionInitialized, MultiLayerPerceptronInitialized as RegressionMLP
from tracks import BinaryClassificationTrack, MultiClassClassificationTrack, RegressionTrack

N_CHECKPOINTS = 50

LEARNING_RATE = 0.005

TRACKS = [
    BinaryClassificationTrack(),
    MultiClassClassificationTrack(),
    RegressionTrack(),
]

# Models configuration for different tracks
MODELS = {
    "Binary classification": {
        "Logistic regression": (
            preprocessing.StandardScaler()
            | linear_model.LogisticRegression(
                optimizer=optim.SGD(LEARNING_RATE)
            )
        ),
        "Deep River Logistic": (
            preprocessing.StandardScaler()
            | LogisticRegressionInitialized(
                loss_fn="cross_entropy",
                optimizer_fn="sgd",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=LEARNING_RATE
            )
        ),
        "Deep River MLP": (
            preprocessing.StandardScaler()
            | ClassificationMLP(
                loss_fn="cross_entropy",
                optimizer_fn="sgd",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=LEARNING_RATE
            )
        ),
        "[baseline] Prior class": dummy.PriorClassifier(),
    },
    "Multiclass classification": {
        "Logistic regression": (
            preprocessing.StandardScaler()
            | linear_model.LogisticRegression(
                optimizer=optim.SGD(LEARNING_RATE)
            )
        ),
        "Deep River Logistic": (
            preprocessing.StandardScaler()
            | LogisticRegressionInitialized(
                loss_fn="cross_entropy",
                optimizer_fn="sgd",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=LEARNING_RATE
            )
        ),
        "Deep River MLP": (
            preprocessing.StandardScaler()
            | ClassificationMLP(
                loss_fn="cross_entropy",
                optimizer_fn="sgd",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=LEARNING_RATE
            )
        ),
        "[baseline] Last Class": dummy.NoChangeClassifier(),
        "[baseline] Prior Class": dummy.PriorClassifier(),
    },
    "Regression": {
        "Linear regression": (
            preprocessing.StandardScaler()
            | linear_model.LinearRegression(
                optimizer=optim.SGD(LEARNING_RATE)
            )
        ),
        "Deep River Linear": (
            preprocessing.StandardScaler()
            | LinearRegressionInitialized(
                loss_fn="mse",
                optimizer_fn="sgd",
                lr=LEARNING_RATE,
                is_feature_incremental=True,
            )
        ),
        "Deep River MLP": (
            preprocessing.StandardScaler()
            | RegressionMLP(
                loss_fn="mse",
                optimizer_fn="sgd",
                lr=LEARNING_RATE,
                is_feature_incremental=True,
            )
        ),
        "River MLP": preprocessing.StandardScaler()
        | neural_net.MLPRegressor(
            hidden_dims=(10,),
            activations=(
                neural_net.activations.ReLU,
                neural_net.activations.ReLU,
                neural_net.activations.Identity,
            ),
            optimizer=optim.SGD(LEARNING_RATE),
            seed=42,
        ),
        "[baseline] Mean predictor": dummy.StatisticRegressor(stats.Mean()),
    },
}
