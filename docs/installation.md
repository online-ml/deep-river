# Installation

There are different versions, that can be installed

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