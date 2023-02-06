

from river import evaluate, compose
from river import linear_model
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

if __name__ == '__main__':
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
            optimizer_fn='sgd',
        ))
    metric = metrics.multioutput.MicroAverage(metrics.MAE())
    evaluate.progressive_val_score(dataset, model, metric)
    print(metric)
