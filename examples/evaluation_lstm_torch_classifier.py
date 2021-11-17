import itertools
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import river
from river import preprocessing, tree, datasets, metrics, compose, linear_model
from river.evaluate import Track
from torch import nn, optim
from IncrementalDL.OnlineTorch.classifier import PyTorch2RiverClassifier, RollingPyTorch2RiverClassifer
from IncrementalDL.OnlineTorch.nn_functions.classification import build_torch_mlp_classifier, build_torch_lstm_classifier
from IncrementalDL.config import CLASSIFICATION_TRACKS, N_SAMPLES, N_CHECKPOINTS, LSTM_CLASSIFICATION_TRACKS, SEED
from IncrementalDL.utils import plot_track

if __name__ == '__main__':

    for track_tuple in LSTM_CLASSIFICATION_TRACKS:
        track = track_tuple[1]
        plot_track(
            track=track,
            metric_name="Accuracy",
            models={
                'Torch LSTM Classifer': (
                        preprocessing.StandardScaler()
                        | RollingPyTorch2RiverClassifer(
                            build_fn=build_torch_lstm_classifier,
                            loss_fn=nn.BCELoss,
                            optimizer_fn=optim.Adam,
                            window_size=25,
                            learning_rate=1e-3,
                            hidden_size=10,
                            seed=SEED)
                ),
                'Torch MLP Classifier': (
                        preprocessing.StandardScaler()
                        | PyTorch2RiverClassifier(
                            build_fn=build_torch_mlp_classifier,
                            loss_fn=nn.BCELoss,
                            optimizer_fn=optim.Adam,
                            batch_size=1,
                            learning_rate=1e-3,
                            seed=SEED)
                ),
                'HTC': (
                        preprocessing.StandardScaler()
                        | linear_model.LinearRegression(optimizer=river.optim.SGD(0.001))
                ),
            },
            n_samples=N_SAMPLES,
            n_checkpoints=N_CHECKPOINTS,
            result_path=Path(f'../results/classification/lstm/'),
            verbose=2
        )






