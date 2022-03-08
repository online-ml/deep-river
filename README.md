<p align="center">
  <img height="150px" src="docs/img/logo.png" alt="incremental dl logo">
</p>

<p align="center">
    Incremental DL is a Python library for incremental deep learning.
    Incremental DL ambition is to enable <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a> for neural networks. 
</p>

## Quickstart
As a quick example, we'll train a simple MLP classifier to classifiy the



## Installation
```shell
pip install IncrementalTorch
```
There are 


### base
pip install -e .

### docs

For doing work on the documentation:

```console
pip install -e ".[docs]"
```

### all

I don't care about memory and want to have all packages that are used within incremental deep learning:

```console
pip install -e ".[all]"
```

## Docs
Run
```console
mkdocs serve
```