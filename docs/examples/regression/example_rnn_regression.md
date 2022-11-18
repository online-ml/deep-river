# LSTM Regression Model


```python
from river_torch.regression import RollingRegressor
from river import metrics, compose, preprocessing, datasets, stats, feature_extraction
from torch import nn
from tqdm import tqdm
```


```python
def get_hour(x):
    x['hour'] = x['moment'].hour
    return x
```

## Simple RNN Regression Model


```python
class RnnModule(nn.Module):

    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.n_features=n_features
        self.rnn = nn.RNN(input_size=n_features, hidden_size=hidden_size, num_layers=1)
        self.fc = nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, X, **kwargs):
        output, hn  = self.rnn(X)  # lstm with input, hidden, and internal state
        return self.fc(output[-1, :])
```


```python
dataset = datasets.Bikes()
metric = metrics.MAE()

model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model_pipeline += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model_pipeline |= preprocessing.StandardScaler()
model_pipeline |= RollingRegressor(
    module=RnnModule,
    loss_fn='mse',
    optimizer_fn='sgd',
    window_size=20,
    lr=1e-2,
    hidden_size=32,  # parameters of MyModule can be overwritten
    append_predict=True,
)
model_pipeline
```


```python
for x, y in tqdm(dataset.take(5000)):
    y_pred = model_pipeline.predict_one(x)
    metric = metric.update(y_true=y, y_pred=y_pred)
    model_pipeline = model_pipeline.learn_one(x=x, y=y)
print(f'MAE: {metric.get():.2f}')
```

## LSTM Regression Model


```python
class LstmModule(nn.Module):

    def __init__(self, n_features, hidden_size=1):
        super().__init__()
        self.n_features=n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(in_features=hidden_size,out_features=1)

    def forward(self, X, **kwargs):
        output, (hn, cn) = self.lstm(X)  # lstm with input, hidden, and internal state
        return self.fc(output[-1, :])

```


```python
dataset = datasets.Bikes()
metric = metrics.MAE()

model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model_pipeline += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model_pipeline |= preprocessing.StandardScaler()
model_pipeline |= RollingRegressor(
    module=LstmModule,
    loss_fn='mse',
    optimizer_fn='sgd',
    window_size=20,
    lr=1e-2,
    hidden_size=32,  # parameters of MyModule can be overwritten
    append_predict=True,
)
model_pipeline
```


```python
for x, y in tqdm(dataset.take(5000)):
    y_pred = model_pipeline.predict_one(x)
    metric = metric.update(y_true=y, y_pred=y_pred)
    model_pipeline = model_pipeline.learn_one(x=x, y=y)
print(f'MAE: {metric.get():.2f}')
```
