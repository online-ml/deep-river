from pathlib import Path

import river
import torch
from river import preprocessing, tree, datasets, compose, linear_model
from river.evaluate import Track
from river.metrics import MSE, MAE
from torch import nn, optim
from IncrementalDL.OnlineTorch.classifier import RollingPyTorch2RiverClassifer
from IncrementalDL.OnlineTorch.nn_functions.regression import build_torch_mlp_regressor, build_torch_lstm_regressor
from IncrementalDL.OnlineTorch.regressor import PyTorch2RiverRegressor, RollingPyTorch2RiverRegressor
from IncrementalDL.config import CLASSIFICATION_TRACKS, N_SAMPLES, N_CHECKPOINTS
from IncrementalDL.utils import plot_track

if __name__ == '__main__':
    track_name = 'Bikes'
    n_samples = 10000
    n_checkpoints = 1000


    def track_fn(n_samples=n_samples, seed=42):
        dataset = datasets.Bikes().take(n_samples)
        metric = MSE()
        track = Track(track_name, dataset, metric, n_samples)
        return track


    fig = plot_track(
        track=track_fn,
        metric_name="MSE",
        models={
            'Torch LSTM Regressor': (
                    compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
                    | preprocessing.StandardScaler()
                    | RollingPyTorch2RiverRegressor(
                        build_fn=build_torch_lstm_regressor,
                        loss_fn=torch.nn.MSELoss,
                        optimizer_fn=optim.Adam,
                        window_size=25,
                        learning_rate=1e-3,
                        hidden_size = 2,
                        num_layers = 1,

                    )),
            'Torch MLP Regressor': (
                compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
                | preprocessing.StandardScaler()
                | PyTorch2RiverRegressor(
                    build_fn=build_torch_mlp_regressor,
                    loss_fn=torch.nn.MSELoss,
                    optimizer_fn=optim.Adam,
                    learning_rate=1e-3,
                )),
            'Linear Regression': (
                compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
                | preprocessing.StandardScaler()
                | linear_model.LinearRegression(optimizer=river.optim.SGD(0.001))
            ),
        },
        n_samples=n_samples,
        n_checkpoints=n_checkpoints,
        result_path=Path(f'./results/regression/example_regression/{track_name}'),
        verbose=2
    )
