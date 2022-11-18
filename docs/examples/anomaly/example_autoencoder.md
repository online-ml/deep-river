# Simple Fully Connected Autoencoder


```python
from river import compose, preprocessing, metrics, datasets

from river_torch.anomaly import Autoencoder
from torch import nn, manual_seed
```


```python
_ = manual_seed(42)
dataset = datasets.CreditCard().take(5000)
metric = metrics.ROCAUC(n_thresholds=50)

class MyAutoEncoder(nn.Module):
    def __init__(self, n_features, latent_dim=3):
        super(MyAutoEncoder, self).__init__()
        self.linear1 = nn.Linear(n_features, latent_dim)
        self.nonlin = nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_dim, n_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, **kwargs):
        X = self.linear1(X)
        X = self.nonlin(X)
        X = self.linear2(X)
        return self.sigmoid(X)

model_pipeline = compose.Pipeline(
    preprocessing.MinMaxScaler(),
    Autoencoder(module=MyAutoEncoder, lr=0.005)
)
model_pipeline
```


```python
for x, y in dataset:
    score = model_pipeline.score_one(x)
    metric.update(y_true=y, y_pred=score)
    model_pipeline.learn_one(x=x)
print(f"ROCAUC: {metric.get():.4f}")
```


```python
for x, y in dataset:
    score = model_pipeline.score_one(x)
    metric.update(y_true=y, y_pred=score)
    model_pipeline.learn_one(x=x)
print(f"ROCAUC: {metric.get():.4f}")
```


```python
for x, y in dataset:
    score = model_pipeline.score_one(x)
    metric.update(y_true=y, y_pred=score)
    model_pipeline.learn_one(x=x)
print(f"ROCAUC: {metric.get():.4f}")
```


```python
for x, y in dataset:
    score = model_pipeline.score_one(x)
    metric.update(y_true=y, y_pred=score)
    model_pipeline.learn_one(x=x)
print(f"ROCAUC: {metric.get():.4f}")
```
