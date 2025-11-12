import json
import os
import sys
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dominate.tags import pre
from watermark import watermark


# ---------- Paths & helpers ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_BENCH_DIR = REPO_ROOT / "docs" / "benchmarks"


def slugify(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("_", "-")
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy(str(src), str(dst))


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# ---------- Plot rendering ----------
def _infer_measures(df: pd.DataFrame) -> List[str]:
    """Infer metric columns from a benchmark dataframe.

    Prefer numeric columns excluding the common identifier columns.
    Fallback to columns[4:] if needed to remain backward-compatible.
    """
    id_cols = {"dataset", "model", "step"}
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in {"step"}]
    measures = [c for c in numeric_cols if c not in id_cols]
    if not measures:
        # fallback to the previous behavior
        measures = list(df.columns)[4:]
    return [c for c in measures if c in df.columns]


def _palette(models: List[str]) -> Dict[str, str]:
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]
    return {m: colors[i % len(colors)] for i, m in enumerate(models)}


def render_df_blocks(df_path: Path, id_prefix: str | None = None) -> List[tuple[str, str]]:
    """Render the benchmark CSV into responsive Plotly HTML blocks per dataset.

    - Uses Markdown for dataset headings outside this function.
    - No Plotly figure title to avoid duplicate headings.
    Returns: list of (dataset_name, html_block)
    """
    df = pd.read_csv(str(df_path))

    required = {"dataset", "model", "step"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        warn_html = f"<div class='admonition warning'>Missing required columns: {missing}</div>"
        return [("", warn_html)]

    # Stable ordering
    unique_datasets = sorted(df["dataset"].unique())
    unique_models = sorted(df["model"].unique())
    measures = _infer_measures(df)

    palette = _palette(unique_models)
    blocks: List[tuple[str, str]] = []
    first_block = True

    for dataset in unique_datasets:
        dataset_df = df[df["dataset"] == dataset]
        if dataset_df.empty:
            continue

        nrows = max(1, len(measures))
        fig = make_subplots(
            rows=nrows,
            cols=1,
            subplot_titles=[m.replace("_", " ").title() for m in measures] if measures else None,
            shared_xaxes=True,
            vertical_spacing=0.06,
        )

        for model in unique_models:
            model_df = dataset_df[dataset_df["model"] == model]
            if model_df.empty:
                continue
            # Ensure sorted by step for nice lines
            model_df = model_df.sort_values("step")

            if measures:
                for i, measure in enumerate(measures):
                    y = model_df[measure]
                    if y.isna().all():
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=model_df["step"],
                            y=y,
                            name=str(model),
                            mode="lines",
                            line=dict(color=palette[model], width=2),
                            legendgroup=str(model),
                            showlegend=(i == 0),  # one legend entry per model
                            hovertemplate=(
                                f"Model: {model}<br>Step: %{{x}}<br>{measure.replace('_',' ').title()}: %{{y:.4f}}<extra></extra>"
                            ),
                        ),
                        row=(i + 1),
                        col=1,
                    )

        # Layout
        fig.update_layout(
            height=max(420, 180 * nrows),
            showlegend=bool(measures),
            # No Plotly title; the page shows a Markdown heading instead
            template="plotly_white",
            margin=dict(l=56, r=36, t=36, b=48),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, -apple-system, BlinkMacSystemFont, system-ui, sans-serif"),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        )

        # Axes and empty-state annotation
        if measures:
            fig.update_xaxes(title_text="Instance", row=nrows, col=1)
            for i, measure in enumerate(measures):
                fig.update_yaxes(title_text=measure.replace("_", " ").title(), row=i + 1, col=1)
        else:
            fig.add_annotation(
                text="No numeric metrics found in CSV.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            )

        # HTML conversion
        dataset_slug = slugify(dataset)
        prefix = f"{slugify(id_prefix)}-" if id_prefix else ""
        fig_div_id = f"plot-{prefix}{dataset_slug}"
        config = {
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d", "autoScale2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{dataset_slug}_benchmark",
                "height": 600,
                "width": 1000,
                "scale": 2,
            },
            "scrollZoom": False,
            "staticPlot": False,
        }

        # Only include plotly.js in the first block to avoid loading it multiple times
        include_plotlyjs = "cdn" if first_block else False
        first_block = False

        html = fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            config=config,
            div_id=fig_div_id,
            validate=False,
        )

        blocks.append((dataset,html))

    return blocks

if __name__ == '__main__':

    print("Starting benchmarks render...")
    print(f"CWD: {os.getcwd()}")

    ensure_dir(DOCS_BENCH_DIR)

    # Locate and ensure details.json is in docs/benchmarks
    details_src = find_first_existing([Path.cwd() / "details.json", DOCS_BENCH_DIR / "details.json"])
    if details_src is None:
        print("ERROR: details.json not found in current directory or docs/benchmarks/")
    if details_src.parent != DOCS_BENCH_DIR:
        print("Moving details.json into docs/benchmarks/")
        safe_copy(details_src, DOCS_BENCH_DIR / "details.json")

    details_path = DOCS_BENCH_DIR / "details.json"
    try:
        details: Dict = json.loads(details_path.read_text())
    except Exception as e:
        print(f"ERROR: Failed to load details.json: {e}")

    index_md_path = DOCS_BENCH_DIR / "index.md"
    with index_md_path.open("w", encoding="utf-8") as f:
        def print_(x: str) -> None:
            print(x, file=f, end="\n\n")

        # Tracks
        for track_name, track_details in details.items():
            print_(f"# {track_name}")
            df_path = DOCS_BENCH_DIR / (track_name.replace(" ", "_").lower() + ".csv")

            df_md = (
                pd.read_csv(str(df_path))
                .groupby(["model", "dataset"])
                .last()
                .drop(columns=["track", "step"])
                .reset_index()
                .rename(columns={"model": "Model", "dataset": "Dataset"})
                .to_markdown(index=False)
            )
            print_(df_md)

            for dataset_name, html_block in render_df_blocks(df_path,id_prefix=slugify(track_name)):
                print_(f"### {dataset_name}")
                print_(html_block)

            print_("### Datasets")
            for dataset_name, dataset_details in track_details.get(
                    "Dataset", {}).items():
                print_("<details class=\"bench-details\">")
                print_(f"<summary class=\"bench-summary\">{dataset_name}</summary>")
                print_(str(pre(dataset_details,
                                   _class="bench-pre dataset-pre")))
                print_("</details>")

            print_("### Models")
            for model_name, model_details in track_details.get("Model",
                                                               {}).items():
                print_("<details class=\"bench-details\">")
                print_(f"<summary class=\"bench-summary\">{model_name}</summary>")
                print_(str(pre(model_details, _class="bench-pre model-pre")))
                print_("</details>")

        # Environment
        print_("# Environment")
        print_(
            str(
                pre(
                    watermark(
                        python=True, packages="river,numpy,scikit-learn,pandas,scipy,plotly", machine=True
                    )
                )
            )
        )

    print(f"Wrote {index_md_path.relative_to(REPO_ROOT)}")

