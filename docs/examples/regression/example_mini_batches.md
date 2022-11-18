```python
import pandas as pd
from river import datasets
from river_torch import regression
from torch import nn
from river import compose
from river import preprocessing
from itertools import islice
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

def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch
```


```python
dataset = datasets.Bikes()

model_pipeline = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model_pipeline |= preprocessing.StandardScaler()
model_pipeline |= regression.Regressor(module=MyModule, loss_fn="mse", optimizer_fn="sgd")
model_pipeline
```


```python
for batch in batcher(dataset.take(5000),5):
    x,y = zip(*batch)
    x = pd.DataFrame(x)
    y_pred = model_pipeline.predict_many(X=x)
    model_pipeline.learn_many(X=x, y=y)
```
