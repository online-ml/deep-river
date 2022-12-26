from deep_river.classification import Classifier as TorchClassifier
from deep_river.classification import RollingClassifier as TorchRollingClassifier
from deep_river.regression import Regressor as TorchRegressor
from deep_river.regression import RollingRegressor as TorchRollingRegressor
from model_zoo.torch import TorchMLPClassifier, TorchMLPRegressor, TorchLogisticRegression, \
    TorchLinearRegression, TorchLSTMClassifier, TorchLSTMRegressor
from river import preprocessing, linear_model, neural_net, dummy
from river import optim, evaluate, stats

N_CHECKPOINTS = 50

LEARNING_RATE = 0.005

TRACKS = [
    evaluate.BinaryClassificationTrack(),
    evaluate.MultiClassClassificationTrack(),
    evaluate.RegressionTrack(),
]

MODELS = {
    "Binary classification": {
        "Logistic regression": (
            preprocessing.StandardScaler()
            | linear_model.LogisticRegression(optimizer=optim.SGD(LEARNING_RATE))
        )
    },
    "Multiclass classification": {
        "Torch MLP": (
            preprocessing.StandardScaler()
            | TorchClassifier(
                module=TorchMLPClassifier,
                loss_fn="binary_cross_entropy",
                optimizer="adam",
                is_class_incremental=True,
                lr=LEARNING_RATE
            )
        ),
        "[baseline] Last Class": dummy.NoChangeClassifier(),
    },
    "Regression": {
    },
}