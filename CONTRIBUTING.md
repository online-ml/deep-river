# Contribution guidelines

## What to work on?

Take a look at our [GitHub issues](https://github.com/online-ml/deep-river/issues). The labelling scheme should be self-explanatory. You're welcome to pick anything that takes your fancy and that you deem important. We encourage you to discuss with us your approach before you start the implementation, to avoid wasting time on out-of-scope work.

We do not assign issues to people. If you want to indicate you're working on something, just start a draft pull request, indicating the issue you're targeting.

Of course, you're welcome to propose and contribute new ideas. We encourage you to [open a discussion](https://github.com/online-ml/deep-river/discussions/new) so that we can have a chat and align.

## Rules for coding agents

We are not against coding agents. But deep-river was made by humans who enjoy working with each other, and we want to preserve that human touch. Here are our rules:

> 1. Coding agents can write code, but not comments.

We have a codebase that is of good quality, with enough examples for coding agents to write idiomatic code. Therefore, AI generated code is not a problem per se. But using an AI to write comments is worrying, because it's a sign we did not put in the effort to understand the generated code.

> 2. Prose is written by humans. This covers issues, pull request descriptions, commit messages, docstrings, release notes, and any kind of discussion.

We don't want coding agents to do the high-level thinking for us. Therefore, we should force ourselves to write all our discussions with our own words. AI generated prose almost always reads like slop, and too much of it is off-putting. We believe using our own words is more polite, friendly, and enjoyable for everyone. Docstrings and release notes count too: they're how we talk to our users, so they deserve the same care.

Of course, you can use a coding agent to run a benchmark and produce a summary table. But you should editorialize and insert it into a message you've written yourself.

> 3. Code written by agents should be disclosed as such.

We should not deceive each other by asking an AI to generate code, and merging it into the codebase without indicating its source. We want to be able to differentiate between the two. A `Co-authored-by:` trailer on the commit is a simple way to do this.

> 4. Be thorough on tests.

Good tests usually span more lines than implementations themselves. They can be tedious to write. Access to coding agents means there is no more excuse for not writing tests.

> 5. Align before you build.

Don't let an agent open a drive-by pull request. As above, discuss your approach with us first, and start from a draft pull request. This matters all the more when an agent makes it cheap to produce a lot of code quickly.

> 6. You are accountable for what your agent submits.

An agent acting on your behalf is still you. You own its output, and the project's contribution standards apply to it just as they do to anything you write yourself.

> 7. Any infringement of the rules above allows the maintainers to close any associated discussion or pull request.

*These rules are enforced in `AGENTS.md`.*

## Fork/clone/pull

The typical workflow for contributing to deep-river is:

1. Fork the `main` branch from the [GitHub repository](https://github.com/online-ml/deep-river/).
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original `main` branch.

## Local setup

Start by cloning the repository:

```sh
git clone https://github.com/online-ml/deep-river
cd deep-river
```

Next, install uv and a supported Python version:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
```

Now you're set to install deep-river and its development dependencies:

```sh
uv sync --extra dev
```

Finally, install the [prek](https://prek.j178.dev/) push hooks. This will run some code quality checks every time you push to GitHub.

```sh
prek install --hook-type pre-push --overwrite
```

You can optionally run `prek` at any time as so:

```sh
prek run --all-files
```

## Making changes

You're now ready to make some changes. We strongly recommend that you check out deep-river's source code for inspiration before getting into the thick of it. How you make the changes is up to you of course. However we can give you some pointers as to how to test your changes. Here is an example workflow that works for most cases:

- Create and open a Jupyter notebook at the root of the directory.
- Add the following in the code cell:

```py
%load_ext autoreload
%autoreload 2
```

- The previous code will automatically reimport deep-river for you whenever you make changes.
- For instance, if a change is made to `regression.Regressor`, then rerunning the following code doesn't require rebooting the notebook:

```py
from deep_river.regression import Regressor
from torch import nn

class MyModule(nn.Module):
    def __init__(self, n_features):
        super(MyModule, self).__init__()

    def forward(self, X, **kwargs):
        # your transformation here
        return X

model = Regressor(module=MyModule)
```

## Creating a new estimator

1. Pick a base class from the `base.py` file, which can either be `DeepEstimator` or `RollingDeepEstimator`.
2. Check if any of the mixin classes from the `base` module apply to your implementation.
3. Make sure you've implemented the required methods, with the following exceptions:
   1. Stateless transformers do not require a `learn_one` method.
   2. In case of a classifier, the `predict_one` is implemented by default, but can be overridden.
4. Add type hints to the parameters of the `__init__` method.
5. If possible provide a default value for each parameter. If, for whatever reason, no good default exists, then implement the `_unit_test_params` method. This is a private method that is meant to be used for testing.
6. Write a comprehensive docstring with example usage. Try to have empathy for new users when you do this.
7. Check that the class you have implemented is imported in the `__init__.py` file of the module it belongs to.
8. When you're done, run the `utils.check_estimator` function on your class and check that no exceptions are raised.

## Documenting your change

If you're adding a class or a function, then you'll need to add a docstring. We follow the [Google docstring convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html), so please do too.

To build the documentation, install the development dependencies:

```sh
uv sync --extra dev
```

From the root of the repository, you can then run the `make livedoc` command to take a look at the documentation in your browser. This will run the benchmark renderer and API reference generator before starting the Zensical preview server.

## Testing

**Unit tests**

These tests absolutely have to pass.

```sh
uv run pytest
```

**Static typing**

These tests absolutely have to pass.

```sh
uv run mypy deep_river
```

**Notebook tests**

You don't have to worry too much about these, as we only check them before each release. If you break them because you changed some code, then it's probably because the notebooks have to be modified, not the other way around.

```sh
make execute-notebooks
```

## Making a new release

1. Checkout `main`
2. Run `make execute-notebooks` just to be safe
3. Run the [benchmarks](benchmarks)
4. Bump the version in `deep_river/__version__.py`
5. Bump the version in `pyproject.toml`
6. Commit and push
7. Wait for CI to [run the unit tests](https://github.com/online-ml/deep-river/actions/workflows/unit-tests.yml)
8. Push the release tag:

```sh
DEEP_RIVER_VERSION=$(python -c "import deep_river; print(deep_river.__version__)")
echo $DEEP_RIVER_VERSION
```

```sh
git tag "v$DEEP_RIVER_VERSION"
git push origin "v$DEEP_RIVER_VERSION"
```

9. Wait for CI to [ship to PyPI](https://github.com/online-ml/deep-river/actions/workflows/pypi-publish.yml) and [publish the new docs](https://github.com/online-ml/deep-river/actions/workflows/docs.yml)

## Versioned docs

The documentation site publishes one entry per git tag and keeps `dev` up to date from `main`.

- Tagged releases are published under the exact tag name, for example `v0.3.2`.
- The newest stable release is also copied to the `latest` alias.
- The site root redirects to `latest`.
- `dev` is rebuilt from `main` on every docs deployment.

If you need to rebuild all historical documentation versions, run the manual [Backfill Documentation Versions](https://github.com/online-ml/deep-river/actions/workflows/docs-backfill.yml) workflow.
