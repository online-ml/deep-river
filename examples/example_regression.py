from pathlib import Path

import torch
from river import preprocessing, neural_net, optim
from tqdm import tqdm

from IncrementalDL.OnlineKeras.nn_functions.regression import build_mlp
from IncrementalDL.OnlineKeras.regressor import Keras2RiverRegressor
from IncrementalDL.OnlineTorch.nn_functions.regression import build_torch_mlp_regressor
from IncrementalDL.OnlineTorch.regressor import PyTorch2RiverRegressor
from IncrementalDL.config import REGRESSION_TRACKS, N_SAMPLES, N_CHECKPOINTS
from IncrementalDL.utils import plot_track

if __name__ == '__main__':

    torch_net = build_torch_mlp_regressor(n_features=6)
    for track_name, track in tqdm(REGRESSION_TRACKS):
        fig = plot_track(
            track=track,
            metric_name="MSE",
            models={
                'KerasRegressor': (preprocessing.StandardScaler() | Keras2RiverRegressor(build_fn=build_mlp)),
                # 'HTR': (preprocessing.StandardScaler() | tree.HoeffdingTreeRegressor()),
                'RiverMLP': (preprocessing.StandardScaler() | neural_net.MLPRegressor(
                    hidden_dims=(5, 5, 5),
                    activations=(
                        neural_net.activations.ReLU,
                        neural_net.activations.ReLU,
                        neural_net.activations.ReLU,
                        neural_net.activations.ReLU,
                        neural_net.activations.Identity
                    ),
                    loss=optim.losses.Squared(),
                    optimizer=optim.SGD(1e-2),
                    seed=42)),
                'PytorchRegressor': (preprocessing.StandardScaler() |
                                     PyTorch2RiverRegressor(
                                         build_fn=build_torch_mlp_regressor,
                                         loss_fn=torch.nn.MSELoss,
                                         optimizer_fn=torch.optim.SGD,
                                         learning_rate=1e-3)
                                     ),
            },
            n_samples=N_SAMPLES,
            n_checkpoints=N_CHECKPOINTS,
            result_path=Path(f'./results/regression/example_regression/{track_name}'),
            verbose=2
        )
