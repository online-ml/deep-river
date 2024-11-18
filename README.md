<p align="center">
  <img height="150px" src="https://raw.githubusercontent.com/online-ml/deep-river/master/docs/img/logo.png" alt="incremental dl logo">
</p>
<p align="center">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/deep-river">
    <a href="https://codecov.io/gh/online-ml/deep-river" > 
        <img src="https://codecov.io/gh/online-ml/deep-river/branch/master/graph/badge.svg?token=ZKUIISZAYA"/> 
    </a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/deep-river">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/deep-river">
    <img alt="GitHub" src="https://img.shields.io/github/license/online-ml/deep-river">
    <a href="https://joss.theoj.org/papers/6a76784f55e8b041d71a7fa776eb386a"><img src="https://joss.theoj.org/papers/6a76784f55e8b041d71a7fa776eb386a/status.svg"></a>
</p>
<p align="center">
    deep-river is a Python library for online deep learning.
    deep-river's ambition is to enable <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a> for neural networks.
    It combines the <a href="https://www.riverml.xyz">river</a> API with the capabilities of designing neural networks based on <a href="https://pytorch.org">PyTorch</a>.
</p>

## 📚 [Documentation](https://online-ml.github.io/deep-river/)
The [documentation](https://online-ml.github.io/deep-river/) contains an overview of all features of this repository as well as the repository's full features list. 
In each of these, the git repo reference is listed in a section that shows [examples](https://github.com/online-ml/deep-river/blob/master/docs/examples) of the features and functionality.
As we are always looking for further use cases and examples, feel free to contribute to the documentation or the repository itself via a pull request

## 💈 Installation

```shell
pip install deep-river
```
or
```shell
pip install "river[deep]"
```
You can install the latest development version from GitHub as so:

```shell
pip install https://github.com/online-ml/deep-river/archive/refs/heads/master.zip
```

## 🍫 Quickstart

We build the development of neural networks on top of the <a href="https://www.riverml.xyz">river API</a> and refer to the rivers design principles.
The following example creates a simple MLP architecture based on PyTorch and incrementally predicts and trains on the website phishing dataset.
For further examples check out the <a href="https://online-ml.github.io/deep-river">Documentation</a>.

### Classification

```python
>>> from river import metrics, datasets, preprocessing, compose
>>> from deep_river import classification
>>> from torch import nn
>>> from torch import optim
>>> from torch import manual_seed

>>> _ = manual_seed(42)

>>> class MyModule(nn.Module):
...     def __init__(self, n_features):
...         super(MyModule, self).__init__()
...         self.dense0 = nn.Linear(n_features, 5)
...         self.nonlin = nn.ReLU()
...         self.dense1 = nn.Linear(5, 2)
...         self.softmax = nn.Softmax(dim=-1)
...
...     def forward(self, X, **kwargs):
...         X = self.nonlin(self.dense0(X))
...         X = self.nonlin(self.dense1(X))
...         X = self.softmax(X)
...         return X

>>> model_pipeline = compose.Pipeline(
...     preprocessing.StandardScaler(),
...     classification.Classifier(module=MyModule, loss_fn='binary_cross_entropy', optimizer_fn='adam')
... )

>>> dataset = datasets.Phishing()
>>> metric = metrics.Accuracy()

>>> for x, y in dataset:
...     y_pred = model_pipeline.predict_one(x)  # make a prediction
...     metric.update(y, y_pred)  # update the metric
...     model_pipeline.learn_one(x, y)  # make the model learn
>>> print(f"Accuracy: {metric.get():.4f}")
Accuracy: 0.7264

```
### Multi Target Regression 
```python
>>> from river import evaluate, compose
>>> from river import metrics
>>> from river import preprocessing
>>> from river import stream
>>> from sklearn import datasets
>>> from torch import nn
>>> from deep_river.regression.multioutput import MultiTargetRegressor

>>> class MyModule(nn.Module):
...     def __init__(self, n_features):
...         super(MyModule, self).__init__()
...         self.dense0 = nn.Linear(n_features, 3)
...
...     def forward(self, X, **kwargs):
...         X = self.dense0(X)
...         return X

>>> dataset = stream.iter_sklearn_dataset(
...         dataset=datasets.load_linnerud(),
...         shuffle=True,
...         seed=42
...     )
>>> model = compose.Pipeline(
...     preprocessing.StandardScaler(),
...     MultiTargetRegressor(
...         module=MyModule,
...         loss_fn='mse',
...         lr=0.3,
...         optimizer_fn='sgd',
...     ))
>>> metric = metrics.multioutput.MicroAverage(metrics.MAE())
>>> ev = evaluate.progressive_val_score(dataset, model, metric)
>>> print(f"MicroAverage(MAE): {metric.get():.2f}")
MicroAverage(MAE): 34.31

```

### Anomaly Detection

```python
>>> from deep_river.anomaly import Autoencoder
>>> from river import metrics
>>> from river.datasets import CreditCard
>>> from torch import nn
>>> import math
>>> from river.compose import Pipeline
>>> from river.preprocessing import MinMaxScaler

>>> dataset = CreditCard().take(5000)
>>> metric = metrics.RollingROCAUC(window_size=5000)

>>> class MyAutoEncoder(nn.Module):
...     def __init__(self, n_features, latent_dim=3):
...         super(MyAutoEncoder, self).__init__()
...         self.linear1 = nn.Linear(n_features, latent_dim)
...         self.nonlin = nn.LeakyReLU()
...         self.linear2 = nn.Linear(latent_dim, n_features)
...         self.sigmoid = nn.Sigmoid()
...
...     def forward(self, X, **kwargs):
...         X = self.linear1(X)
...         X = self.nonlin(X)
...         X = self.linear2(X)
...         return self.sigmoid(X)

>>> ae = Autoencoder(module=MyAutoEncoder, lr=0.005)
>>> scaler = MinMaxScaler()
>>> model = Pipeline(scaler, ae)

>>> for x, y in dataset:
...     score = model.score_one(x)
...     model.learn_one(x=x)
...     metric.update(y, score)
...
>>> print(f"Rolling ROCAUC: {metric.get():.4f}")
Rolling ROCAUC: 0.8901

```

## 🏫 Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>

<p align="center">
    <img src="https://lieferbotnet.de/wp-content/uploads/2022/09/LieferBotNet-Logo.png?raw=true" alt="Lieferbot net" height="200"/>
</p>
