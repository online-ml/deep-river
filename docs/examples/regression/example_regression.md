# Simple Regression Model


```python
from river import metrics, compose, preprocessing, datasets, stats, feature_extraction
from river_torch.regression import Regressor
from torch import nn
from pprint import pprint
from tqdm import tqdm
```


```python
dataset = datasets.Bikes()

for x, y in dataset:
    pprint(x)
    print(f'Number of available bikes: {y}')
    break
```


```python
class MyModule(nn.Module):
    def __init__(self, n_features):
        super(MyModule, self).__init__()
        self.dense0 = nn.Linear(n_features,5)
        self.nonlin = nn.ReLU()
        self.dense1 = nn.Linear(5, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.nonlin(self.dense1(X))
        X = self.softmax(X)
        return X

def get_hour(x):
    x['hour'] = x['moment'].hour
    return x
```


```python
metric = metrics.MAE()

model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model_pipeline += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model_pipeline |= preprocessing.StandardScaler()
model_pipeline |= Regressor(module=MyModule, loss_fn="mse", optimizer_fn='sgd')
model_pipeline
```


```python
for x, y in tqdm(dataset.take(5000)):
    y_pred = model_pipeline.predict_one(x)
    metric.update(y_true=y, y_pred=y_pred)
    model_pipeline.learn_one(x=x, y=y)
print(f'MAE: {metric.get():.2f}')
```
