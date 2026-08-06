# Benchmarks

## Installation 
```sh
pip install ".[benchmarks]"
```

## Usage
The `run.py` executes the benchmarks and creates the necessary .csv files for rendering the plots.
```sh
cd benchmarks
python run.py
```
The `render.py` renders the plots from the .csv files and moves them to the `docs/benchmarks` folder.
```sh
python render.py
```

## CodSpeed

CodSpeed runs a small pytest benchmark suite from `benchmarks/codspeed/python/` on pull requests and pushes to `main`. These benchmarks target deterministic CPU workloads for representative online and mini-batch deep-river operations.

Run them locally from the repository root:

```sh
make benchmark
make benchmark K=classifier
```

Local runs use wall-clock timing and are useful as smoke tests. CI uses CodSpeed simulation mode for pull-request comparisons.

Add new CodSpeed benchmark checks in `deep_river/utils/estimator_checks.py` and expose them through `benchmarks/codspeed/python/`. Keep inputs deterministic, seed PyTorch before model construction, keep data creation outside the measured callable, use CPU-only estimators, and keep benchmark names stable after merging so CodSpeed history is preserved.
