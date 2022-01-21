<p align="center">
  <img height="150px" src="docs/img/logo.png" alt="incremental dl logo">
</p>

<p align="center">
    Incremental DL is a Python library for incremental deep learning.
    Incremental DL ambition is to enable <a href="https://www.wikiwand.com/en/Online_machine_learning">online machine learning</a> for neural networks. 
</p>

## Installation

There are different versions, that can be installed

### base

pip install -e .

### docs

For doing work on the documentation:

```python
pip install -e ".[docs]"
```

### all

I don't care about memory and want to have all packages that are used within incremental deep learning:

```python
pip install -e ".[all]"
```