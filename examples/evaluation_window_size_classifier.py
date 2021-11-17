import itertools
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from river import preprocessing, tree, datasets, metrics, compose
from river.evaluate import Track
from torch import nn, optim
from IncrementalDL.OnlineTorch.classifier import PyTorch2RiverClassifier, RollingPyTorch2RiverClassifer
from IncrementalDL.OnlineTorch.nn_functions.classification import build_torch_mlp_classifier, build_torch_lstm_classifier
from IncrementalDL.config import CLASSIFICATION_TRACKS, N_SAMPLES, N_CHECKPOINTS, LSTM_CLASSIFICATION_TRACKS, SEED
from IncrementalDL.utils import plot_track

def evaluate_window_size(track_tuple,window_size):
    track = track_tuple[1]
    track_name = track_tuple[0]

    df = plot_track(
        track=track,
        metric_name="Accuracy",
        models={
            'Torch LSTM Classifer': (
                    preprocessing.StandardScaler()
                    | RollingPyTorch2RiverClassifer(
                build_fn=build_torch_lstm_classifier,
                loss_fn=nn.BCELoss,
                optimizer_fn=optim.Adam,
                window_size=window_size,
                learning_rate=1e-3,
                hidden_size=10,
                seed=SEED

            )),
            'Torch MLP Classifier': (
                    preprocessing.StandardScaler()
                    | PyTorch2RiverClassifier(
                build_fn=build_torch_mlp_classifier,
                loss_fn=nn.BCELoss,
                optimizer_fn=optim.Adam,
                batch_size=1,
                learning_rate=1e-3,
                seed=SEED
            )),
            'HTC': (
                    preprocessing.StandardScaler()
                    | tree.HoeffdingTreeClassifier()),
        },
        n_samples=N_SAMPLES,
        n_checkpoints=N_CHECKPOINTS,
        result_path=Path(f'./results/classification/example_classification/lstm/{track_name}_{window_size}'),
        verbose=2
    )
    df['window_size'] = window_size
    df['track'] = track_name
    return df



if __name__ == '__main__':

    window_sizes = [1,2,5,10,15,25,50,100,250]
    testing_configurations = list(itertools.product(LSTM_CLASSIFICATION_TRACKS, window_sizes))

    pool = Pool(20)  # Create a multiprocessing Pool
    output = pool.starmap(evaluate_window_size, testing_configurations)
    result_data = pd.concat(output)

    # t = evaluate_sampling_rate(250,EVO_CLASSIFICATION_TRACKS[0])
    # result_data = t

    result_path = Path(f'./results/lstm')
    result_path.mkdir(parents=True, exist_ok=True)
    result_path = result_path / f'lstm_window_size.xlsx'
    result_data.to_excel(str(result_path))
