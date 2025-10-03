import itertools
import json
import logging
import multiprocessing
import copy
from typing import List, Dict, Any

import pandas as pd
from config import MODELS, N_CHECKPOINTS, TRACKS
from river import metrics
from tqdm import tqdm

# Neu: limit_dataset import für Fallback
try:
    from tracks import limit_dataset
except Exception:
    limit_dataset = None  # Falls nicht verfügbar, ignorieren wir den Fallback

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def run_dataset(model_str: str, no_dataset: int, no_track: int) -> List[dict]:
    """Run a single model on a single dataset from a track.

    Args:
        model_str: Name of the model to run
        no_dataset: Index of the dataset in the track
        no_track: Index of the track

    Returns:
        List of result dictionaries for each checkpoint
    """
    model_name = model_str
    track = TRACKS[no_track]
    dataset = track.datasets[no_dataset]

    # Fallback: Falls dataset.n_samples fehlt oder None ist (z.B. endloser Stream), kappen.
    n_s = getattr(dataset, 'n_samples', None)
    if (n_s is None or not isinstance(n_s, int)) and limit_dataset is not None:
        # Standardlimit analog MultiClassClassificationTrack
        dataset = limit_dataset(dataset, 5000)
        # Ersetze im Track-Datasets-Array, damit spätere Ausgaben konsistent sind
        track.datasets[no_dataset] = dataset

    ds_name = getattr(dataset, 'dataset_name', dataset.__class__.__name__)

    # Get a fresh model instance for this run
    model = MODELS[track.name][model_name].clone()
    print(f"Processing {model_str} on {ds_name}")

    results = []
    time = 0.0

    try:
        # Use the track's run method directly without deep copying to avoid issues
        for i in tqdm(
            track.run(model, dataset, n_checkpoints=N_CHECKPOINTS),
            total=N_CHECKPOINTS,
            desc=f"{model_name} on {ds_name}",
        ):
            time += i["Time"].total_seconds()
            res = {
                "step": i["Step"],
                "track": track.name,
                "model": model_name,
                "dataset": ds_name,
            }

            # Extract metric values
            for k, v in i.items():
                if isinstance(v, metrics.base.Metric):
                    res[k] = v.get()

            res["Memory in Mb"] = i["Memory"] / 1024**2
            res["Time in s"] = time
            results.append(res)

            # Break if running too long (1 hour limit)
            if time > 3600:
                break

    except Exception as e:
        print(f"Error running {model_name} on {ds_name}: {e}")
        logger.exception(f"Full error for {model_name} on {ds_name}")
        # Return empty results on error but continue with other models
        return []

    return results


def run_track(models: List[str], no_track: int, n_workers: int = 1) -> None:
    """Run all models on all datasets for a specific track.

    Args:
        models: List of model names to run
        no_track: Index of the track to run
        n_workers: Number of parallel workers to use
    """
    track = TRACKS[no_track]
    runs = list(
        itertools.product(models, range(len(track.datasets)), [no_track])
    )
    results = []

    if n_workers > 1:
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(processes=n_workers) as pool:
            for val in pool.starmap(run_dataset, runs):
                if val:  # Only extend if val is not empty
                    results.extend(val)
    else:
        for args in runs:
            val = run_dataset(*args)
            if val:  # Only extend if val is not empty
                results.extend(val)

    # Save results to CSV
    if results:  # Only save if we have results
        csv_name = track.name.replace(" ", "_").lower()
        df = pd.DataFrame(results)
        csv_path = f"./{csv_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    else:
        print(f"No results to save for track: {track.name}")


def create_details_json() -> None:
    """Create a details.json file with dataset and model information."""
    details = {}
    for track in TRACKS:
        details[track.name] = {"Dataset": {}, "Model": {}}
        for dataset in track.datasets:
            ds_name = getattr(dataset, 'dataset_name', dataset.__class__.__name__)
            details[track.name]["Dataset"][ds_name] = repr(dataset)
        for model_name, model in MODELS[track.name].items():
            details[track.name]["Model"][model_name] = repr(model)
    with open("details.json", "w") as f:
        json.dump(details, f, indent=2)
    print("Details saved to details.json")


def main() -> None:
    """Main function to run all benchmarks."""
    print("Starting benchmarks...")

    # Create details file
    create_details_json()

    # Run benchmarks for each track
    for i, track in enumerate(TRACKS):
        print(f"\nRunning track: {track.name}")
        available_models = list(MODELS[track.name].keys())
        print(f"Available models: {available_models}")

        # Filter out problematic models if needed
        safe_models = []
        for model_name in available_models:
            try:
                # Test if model can be cloned successfully
                test_model = MODELS[track.name][model_name].clone()
                safe_models.append(model_name)
            except Exception as e:
                print(f"Skipping problematic model {model_name}: {e}")
                continue

        print(f"Running safe models: {safe_models}")
        if safe_models:
            run_track(models=safe_models, no_track=i, n_workers=8)

    print("\nAll benchmarks completed!")


if __name__ == "__main__":
    main()
