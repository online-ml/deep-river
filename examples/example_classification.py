from pathlib import Path

from river import preprocessing, tree
from torch import nn, optim

from IncrementalDL.OnlineKeras.classifier import Keras2RiverClassifier
from IncrementalDL.OnlineKeras.nn_functions.classification import build_keras_mlp_classifier
from IncrementalDL.OnlineTorch.classifier import PyTorch2RiverClassifier
from IncrementalDL.OnlineTorch.nn_functions.classification import build_torch_mlp_classifier
from IncrementalDL.config import CLASSIFICATION_TRACKS, N_SAMPLES, N_CHECKPOINTS
from IncrementalDL.utils import plot_track

if __name__ == '__main__':
    track_name, track = CLASSIFICATION_TRACKS[0]
    fig = plot_track(
        track=track,
        metric_name="Accuracy",
        models={
            'Keras MLP Classifier': (
                        preprocessing.StandardScaler() | Keras2RiverClassifier(build_fn=build_keras_mlp_classifier,
                                                                               width=4,
                                                                               )),
            # 'Torch Binary MLP Classifier' : (preprocessing.StandardScaler() | PyTorch2RiverBinaryClassifier(build_fn=build_torch_mlp_classifier,
            #                                                                                         loss_fn=nn.BCELoss(),
            #                                                                                         optimizer=optim.SGD,
            #                                                                                         batch_size = 1,
            #                                                                                         n_features=200,
            #                                                                                         )),
            'Torch MLP Classifier': (preprocessing.StandardScaler() | PyTorch2RiverClassifier(
                build_fn=build_torch_mlp_classifier,
                loss_fn=nn.BCELoss,
                optimizer_fn=optim.Adam,
                learning_rate=1e-3,
            )),
            'HTC': (preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()),
        },
        n_samples=N_SAMPLES,
        n_checkpoints=N_CHECKPOINTS,
        result_path=Path(f'./results/classification/example_classification/{track_name}'),
        verbose=2
    )
