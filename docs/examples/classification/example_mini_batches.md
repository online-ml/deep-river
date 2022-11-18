# Mini Batches
Iterate over a data stream in mini batches


```python
import pandas as pd
from river import datasets
from river_torch import classification
from torch import nn
from river import compose
from river import preprocessing
from itertools import islice
```


```python
dataset = datasets.Phishing()
```


```python
class MyModule(nn.Module):
    def __init__(self, n_features):
        super(MyModule, self).__init__()
        self.dense0 = nn.Linear(n_features,5)
        self.nonlin = nn.ReLU()
        self.dense1 = nn.Linear(5, 2)
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
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    classification.Classifier(module=MyModule,loss_fn="binary_cross_entropy",optimizer_fn="sgd")
)
model
```


```python
for batch in batcher(dataset,5):
    x,y = zip(*batch)
    x = pd.DataFrame(x)
    y = list(y)
    y_pred = model.predict_proba_many(X=x)
    model = model.learn_many(x, y)    # make the model learn
```
