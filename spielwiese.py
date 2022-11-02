from river_torch.classification import RollingClassifier
from river import metrics, compose, preprocessing, datasets
import torch

class MyModule(torch.nn.Module):

    def __init__(self, n_features, hidden_size=1):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=False, num_layers=1, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.lstm.hidden_size)
        return self.softmax(hn)

if __name__ == '__main__':
    dataset = datasets.Bikes()
    metric = metrics.Accuracy()
    optimizer_fn = torch.optim.SGD

    model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
    model_pipeline |= preprocessing.StandardScaler()
    model_pipeline |= RollingClassifier(
        module=MyModule,
        loss_fn="binary_cross_entropy",
        optimizer_fn=torch.optim.SGD,
        window_size=20,
        lr=1e-2,
        append_predict=True,
        is_class_incremental=False
    )

    for x, y in dataset.take(5000):
        y_pred = model_pipeline.predict_one(x)  # make a prediction
        metric = metric.update(y, y_pred)  # update the metric
        model = model_pipeline.learn_one(x, y)  # make the model learn
    print(f'Accuracy: {metric.get()}')