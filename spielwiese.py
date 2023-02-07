from river import evaluate, compose
from river import metrics
from river import preprocessing
from river import stream
from sklearn import datasets
from torch import nn
from deep_river.regression.multioutput import MultiTargetRegressor

class MyModule(nn.Module):
    def __init__(self, n_features):
        super(MyModule, self).__init__()
        self.dense0 = nn.Linear(n_features, 3)

    def forward(self, X, **kwargs):
        X = self.dense0(X)
        return X

dataset = stream.iter_sklearn_dataset(
        dataset=datasets.load_linnerud(),
        shuffle=True,
        seed=42
    )
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    MultiTargetRegressor(
        module=MyModule,
        loss_fn='mse',
        lr=0.3,
        optimizer_fn='sgd',
    ))
metric = metrics.multioutput.MicroAverage(metrics.MAE())
print(evaluate.progressive_val_score(dataset, model, metric))

