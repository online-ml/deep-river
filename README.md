<p align="center">
  <img height="150px" src="https://raw.githubusercontent.com/online-ml/deep-river/master/docs/img/logo.png" alt="incremental dl logo">
</p>
<p align="center">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/deep-river">
    <!-- Single Coverage Badge (Shields.io) -->
    <a href="https://app.codecov.io/gh/online-ml/deep-river" >
        <img alt="Coverage" src="https://img.shields.io/codecov/c/github/online-ml/deep-river?branch=master&logo=codecov&label=coverage" />
    </a>
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/deep-river">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/deep-river">
    <img alt="GitHub" src="https://img.shields.io/github/license/online-ml/deep-river">
    <a href="https://joss.theoj.org/papers/6a76784f55e8b041d71a7fa776eb386a"><img src="https://joss.theoj.org/papers/6a76784f55e8b041d71a7fa776eb386a/status.svg"></a>
    <a href="https://doi.org/10.5281/zenodo.14601980"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.14601980.svg" alt="DOI"></a>
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

### Development Environment

For contributing to deep-river, we recommend using [uv](https://docs.astral.sh/uv/) for fast dependency management and environment setup:

```shell
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/online-ml/deep-river.git
cd deep-river

# Install all dependencies (including dev dependencies)
uv sync --extra dev

# Run tests
make test

# Format code
make format

# Build documentation
make doc
```

## 🍫 Quickstart

We build the development of neural networks on top of the <a href="https://www.riverml.xyz">river API</a> and refer to the rivers design principles.
The following example creates a simple MLP architecture based on PyTorch and incrementally predicts and trains on the website phishing dataset.
For further examples check out the <a href="https://online-ml.github.io/deep-river">Documentation</a>.


### Classification

```python
>>> import random, numpy as np
>>> from river import metrics, datasets, preprocessing, compose
>>> from deep_river.classification import Classifier
>>> from torch import nn, manual_seed
>>> _ = manual_seed(42); random.seed(42); np.random.seed(42)
>>> first_x, _ = next(iter(datasets.Phishing()))
>>> n_features = len(first_x)
>>> class MyModule(nn.Module):
...     def __init__(self, n_features):
...         super().__init__()
...         self.net = nn.Sequential(
...             nn.Linear(n_features, 16),
...             nn.ReLU(),
...             nn.Linear(16, 2)
...         )
...     def forward(self, x):
...         return self.net(x)
>>> model = compose.Pipeline(
...     preprocessing.StandardScaler(),
...     Classifier(
...         module=MyModule(n_features),
...         loss_fn='cross_entropy',
...         optimizer_fn='adam',
...         lr=1e-3,
...         is_class_incremental=True
...     )
... )
>>> metric = metrics.Accuracy()
>>> for i, (x, y) in enumerate(datasets.Phishing().take(80)):
...     if i > 0:
...         y_pred = model.predict_one(x)
...         metric.update(y, y_pred)
...     model.learn_one(x, y)
>>> print(f"Accuracy: {metric.get():.4f}")
Accuracy: 0.8101

```

### Multi Target Regression

```python
>>> import random, numpy as np
>>> from river import stream, metrics
>>> from sklearn import datasets as sk_datasets
>>> from deep_river.regression import MultiTargetRegressor
>>> from torch import nn, manual_seed
>>> _ = manual_seed(42); random.seed(42); np.random.seed(42)
>>> linnerud_stream = stream.iter_sklearn_dataset(sk_datasets.load_linnerud(), shuffle=True, seed=42)
>>> first_x, first_y = next(iter(linnerud_stream))
>>> n_features, n_outputs = len(first_x), len(first_y)
>>> class TinyNet(nn.Module):
...     def __init__(self, n_features, n_outputs):
...         super().__init__()
...         self.net = nn.Sequential(
...             nn.Linear(n_features, 8),
...             nn.ReLU(),
...             nn.Linear(8, n_outputs)
...         )
...     def forward(self, x):
...         return self.net(x)
>>> model = MultiTargetRegressor(
...     module=TinyNet(n_features, n_outputs),
...     loss_fn='mse', optimizer_fn='sgd', lr=1e-3,
...     is_feature_incremental=False, is_target_incremental=False
... )
>>> metric = metrics.multioutput.MicroAverage(metrics.MAE())
>>> linnerud_stream = stream.iter_sklearn_dataset(sk_datasets.load_linnerud(), shuffle=True, seed=42)
>>> for i, (x, y_dict) in enumerate(linnerud_stream):
...     if i > 0:
...         y_pred = model.predict_one(x)
...         metric.update(y_dict, y_pred)
...     model.learn_one(x, y_dict)
>>> print(f"MAE: {metric.get():.4f}")
MAE: 1410.5710

```

### Anomaly Detection

```python
>>> import random, numpy as np
>>> from torch import nn, manual_seed
>>> from river import metrics
>>> from river.datasets import CreditCard
>>> from river.compose import Pipeline
>>> from river.preprocessing import MinMaxScaler
>>> from deep_river.anomaly import Autoencoder
>>> _ = manual_seed(42); random.seed(42); np.random.seed(42)
>>> first_x, _ = next(iter(CreditCard()))
>>> n_features = len(first_x)
>>> class MyAutoEncoder(nn.Module):
...     def __init__(self, n_features, latent_dim=5):
...         super().__init__()
...         self.encoder = nn.Sequential(
...             nn.Linear(n_features, latent_dim),
...             nn.LeakyReLU()
...         )
...         self.decoder = nn.Sequential(
...             nn.Linear(latent_dim, n_features),
...             nn.Sigmoid()
...         )
...     def forward(self, x):
...         z = self.encoder(x)
...         return self.decoder(z)
>>> ae = Autoencoder(module=MyAutoEncoder(n_features), lr=5e-3, optimizer_fn='adam')
>>> model = Pipeline(MinMaxScaler(), ae)
>>> metric = metrics.RollingROCAUC(window_size=500)
>>> for i, (x, y) in enumerate(CreditCard().take(300)):
...     score = model.score_one(x)
...     model.learn_one(x)
...     metric.update(y, score)
>>> print(f"ROCAUC(500): {metric.get():.4f}")


```

## 💬 Citation

To acknowledge the use of the `DeepRiver` library in your research, please refer to our [paper](https://joss.theoj.org/papers/10.21105/joss.07226) published on Journal of Open Source Software (JOSS):

```bibtex
@article{Kulbach2025, 
    doi = {10.21105/joss.07226}, 
    url = {https://doi.org/10.21105/joss.07226}, 
    year = {2025}, 
    publisher = {The Open Journal}, 
    volume = {10}, 
    number = {105}, 
    pages = {7226}, 
    author = {Cedric Kulbach and Lucas Cazzonelli and Hoang-Anh Ngo and Max Halford and Saulo Martiello Mastelini}, 
    title = {DeepRiver: A Deep Learning Library for Data Streams}, 
    journal = {Journal of Open Source Software} 
}
```

## 🏫 Affiliations

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/de/thumb/4/44/Fzi_logo.svg/1200px-Fzi_logo.svg.png?raw=true" alt="FZI Logo" height="200"/>
</p>

<p align="center">
    <img src="https://lieferbotnet.de/wp-content/uploads/2022/09/LieferBotNet-Logo.png?raw=true" alt="Lieferbot net" height="200"/>
</p>
