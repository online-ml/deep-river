# Benchmarks

Run benchmark commands with `uv` from the repository root.

## Usage

The `run.py` script executes benchmarks and creates the CSV inputs for rendering:

```sh
uv run python benchmarks/run.py
```

The `render.py` script renders benchmark pages into `docs/benchmarks`:

```sh
uv run python benchmarks/render.py
```

Or generate everything via:

```sh
make doc
```
