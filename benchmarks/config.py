from river import dummy, linear_model, neural_net, optim, preprocessing, stats
from river import time_series

from deep_river.classification.zoo import (
    LogisticRegression,
    MultiLayerPerceptron as ClassificationMLP,
    LSTMClassifier,
    RNNClassifier,
)
from deep_river.forecasting.zoo import (
    GRUForecaster,
    LinearForecaster as ForecastingLinear,
    LiquidForecaster,
    LSTMForecaster,
    MLPForecaster,
    NBEATSForecaster,
    RNNForecaster,
)
from deep_river.regression.zoo import (
    LinearRegression,
    MultiLayerPerceptron as RegressionMLP,
    LSTMRegressor,
    RNNRegressor,
)
from tracks import (
    BinaryClassificationTrack,
    ForecastingTrack,
    MultiClassClassificationTrack,
    RegressionTrack,
)

N_CHECKPOINTS = 50

LEARNING_RATE = 0.005

TRACKS = [
    BinaryClassificationTrack(),
    MultiClassClassificationTrack(),
    RegressionTrack(),
    ForecastingTrack(),
]

# Models configuration for different tracks
MODELS = {
    "Binary classification": {
        "Logistic regression": (
            preprocessing.StandardScaler()
            | linear_model.LogisticRegression(optimizer=optim.SGD(LEARNING_RATE))
        ),
        "Deep River Logistic": (
            preprocessing.StandardScaler()
            | LogisticRegression(
                loss_fn="cross_entropy",
                optimizer_fn="sgd",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=LEARNING_RATE,
            )
        ),
        "Deep River MLP": (
            preprocessing.StandardScaler()
            | ClassificationMLP(
                loss_fn="cross_entropy",
                optimizer_fn="sgd",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=LEARNING_RATE,
            )
        ),
        "Deep River LSTM": (
            preprocessing.StandardScaler()
            | LSTMClassifier(
                loss_fn="cross_entropy",
                optimizer_fn="adam",  # Adam meist stabiler für RNNs
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=1e-3,
                hidden_size=32,
                # window_size optional via kwargs (RollingClassifierInitialized nimmt window_size)
                window_size=30,
            )
        ),
        "Deep River RNN": (
            preprocessing.StandardScaler()
            | RNNClassifier(
                loss_fn="cross_entropy",
                optimizer_fn="adam",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=1e-3,
                hidden_size=32,
                num_layers=1,
                window_size=30,
            )
        ),
        "[baseline] Prior class": dummy.PriorClassifier(),
    },
    "Multiclass classification": {
        "Logistic regression": (
            preprocessing.StandardScaler()
            | linear_model.LogisticRegression(optimizer=optim.SGD(LEARNING_RATE))
        ),
        "Deep River Logistic": (
            preprocessing.StandardScaler()
            | LogisticRegression(
                loss_fn="cross_entropy",
                optimizer_fn="sgd",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=LEARNING_RATE,
            )
        ),
        "Deep River MLP": (
            preprocessing.StandardScaler()
            | ClassificationMLP(
                loss_fn="cross_entropy",
                optimizer_fn="sgd",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=LEARNING_RATE,
            )
        ),
        "Deep River LSTM": (
            preprocessing.StandardScaler()
            | LSTMClassifier(
                loss_fn="cross_entropy",
                optimizer_fn="adam",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=1e-3,
                hidden_size=32,
                window_size=30,
            )
        ),
        "Deep River RNN": (
            preprocessing.StandardScaler()
            | RNNClassifier(
                loss_fn="cross_entropy",
                optimizer_fn="adam",
                is_class_incremental=True,
                is_feature_incremental=True,
                lr=1e-3,
                hidden_size=32,
                num_layers=1,
                window_size=30,
            )
        ),
        "[baseline] Last Class": dummy.NoChangeClassifier(),
        "[baseline] Prior Class": dummy.PriorClassifier(),
    },
    "Regression": {
        "Linear regression": (
            preprocessing.StandardScaler()
            | linear_model.LinearRegression(optimizer=optim.SGD(LEARNING_RATE))
        ),
        "Deep River Linear": (
            preprocessing.StandardScaler()
            | LinearRegression(
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
        "Deep River LSTM": (
            preprocessing.StandardScaler()
            | LSTMRegressor(
                loss_fn="mse",
                optimizer_fn="adam",  # Wichtiger Wechsel für LSTM
                lr=1e-3,  # Kleinerer Lernrate für Stabilität
                hidden_size=64,  # Größere Kapazität
                num_layers=1,  # Einfach starten
                dropout=0.1,  # Leichtes Dropout zur Regularisierung
                gradient_clip_value=1.0,
                window_size=30,  # Längeres Kontextfenster
                is_feature_incremental=True,
            )
        ),
        "Deep River RNN": (
            preprocessing.StandardScaler()
            | RNNRegressor(
                loss_fn="mse",
                optimizer_fn="adam",
                lr=1e-3,
                hidden_size=64,
                num_layers=1,
                dropout=0.1,
                gradient_clip_value=1.0,
                window_size=30,
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
    "Forecasting": {
        "Holt-Winters": time_series.HoltWinters(
            alpha=0.3,
            beta=0.1,
            gamma=0.1,
            seasonality=12,
        ),
        "SNARIMAX": time_series.SNARIMAX(
            p=1,
            d=1,
            q=0,
            m=12,
            sp=1,
            sd=0,
            sq=0,
        ),
        "Deep River Linear": ForecastingLinear(
            window_size=12,
            loss_fn="mse",
            optimizer_fn="adam",
            lr=3e-3,
        ),
        "Deep River MLP": MLPForecaster(
            window_size=12,
            loss_fn="mse",
            optimizer_fn="adam",
            lr=5e-3,
        ),
        "Deep River RNN": RNNForecaster(
            window_size=12,
            hidden_size=32,
            loss_fn="mse",
            optimizer_fn="adam",
            lr=3e-3,
        ),
        "Deep River GRU": GRUForecaster(
            window_size=12,
            hidden_size=32,
            loss_fn="mse",
            optimizer_fn="adam",
            lr=5e-3,
        ),
        "Deep River LSTM": LSTMForecaster(
            window_size=12,
            hidden_size=32,
            loss_fn="mse",
            optimizer_fn="adam",
            lr=3e-3,
        ),
        "Deep River Liquid": LiquidForecaster(
            window_size=12,
            hidden_size=32,
            loss_fn="mse",
            optimizer_fn="adam",
            lr=3e-3,
        ),
        "Deep River N-BEATS": NBEATSForecaster(
            window_size=12,
            n_width=64,
            n_layers=2,
            n_blocks=2,
            loss_fn="mse",
            optimizer_fn="adam",
            lr=1e-3,
        ),
    },
}
