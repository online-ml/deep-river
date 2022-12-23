import os

import setuptools

# Package meta-data.
NAME = "deep_river"
DESCRIPTION = "Online Deep Learning for river"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://online-ml.github.io/deep-river/"
EMAIL = "cedric.kulbach@googlemail.com"
AUTHOR = "Cedric Kulbach"
REQUIRES_PYTHON = ">=3.6.0"

# Package requirements.
base_packages = [
    "scikit-learn~=1.0.2",
    "torch~=1.13.0",
    "pandas~=1.3.2",
    "numpy~=1.24.0",
    "river~=0.14.0",
    "tqdm~=4.61.2",
    "ordered-set~=4.1.0",
    "torchviz~=0.0.2",
]

dev_packages = base_packages + [
    "graphviz>=0.10.1",
    "matplotlib>=3.0.2",
    "mypy>=0.990",
    "pre-commit>=2.20.0",
    "pytest>=7.2.0",
    "pytest-cov>=4.0.0",
    "scikit-learn>=0.22.1",
    "black>=22.10.0",
    "flake8>=5.0.4",
    "isort>=5.10.1",
    "jupyter>=1.0.0",
    "pyupgrade==3.2.0"
]

docs_packages = [
    "flask>=2.0.2",
    "ipykernel>=6.9.0",
    "mike>=0.5.3",
    "mkdocs>=1.2.3",
    "mkdocs-awesome-pages-plugin>=2.7.0",
    "mkdocs-gen-files>=0.3.5",
    "mkdocs-literate-nav>=0.4.1",
    "mkdocs-material>=8.1.11",
    "mkdocstrings[python]>=0.19.0",
    "pytkdocs[numpy-style]>=0.5.0",
    "ipython_genutils>=0.1.0",
    "mkdocs-jupyter>=0.20.0",
    "nbconvert>=6.4.2",
    "numpydoc>=1.2",
    "spacy>=3.2.2",
    "jinja2>=3.0.3",
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

# Where the magic happens:
setuptools.setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
        "test": dev_packages,
        "docs": docs_packages,
        "all": dev_packages + docs_packages,
        ":python_version == '3.6'": ["dataclasses"],
    },
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[],
)
