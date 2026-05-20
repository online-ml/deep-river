from __future__ import annotations

from pathlib import Path
import re
import shutil

import nbformat
from nbconvert.exporters import MarkdownExporter
from nbconvert.writers import FilesWriter


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
EXAMPLES_DIR = DOCS_DIR / "examples"
GENERATED_DIR = DOCS_DIR / "generated" / "examples"
REPO_URL = "https://github.com/online-ml/deep-river"
DEFAULT_BRANCH = "main"
HEADING_RE = re.compile(r"^(#{1,6})(\s+.+)$")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*#*\s*$")


def notebook_title(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() for word in stem.split())


def find_notebooks() -> list[Path]:
    return sorted(EXAMPLES_DIR.rglob("*.ipynb"))


def colab_link(notebook_path: Path) -> str:
    rel_path = notebook_path.relative_to(ROOT).as_posix()
    return f"https://colab.research.google.com/github/online-ml/deep-river/blob/{DEFAULT_BRANCH}/{rel_path}"


def binder_link(notebook_path: Path) -> str:
    rel_path = notebook_path.relative_to(ROOT).as_posix()
    return (
        "https://mybinder.org/v2/gh/online-ml/deep-river/"
        f"{DEFAULT_BRANCH}?filepath={rel_path}"
    )


def split_title(source: str) -> tuple[str | None, str]:
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue

        match = TITLE_RE.match(line)
        if match:
            return match.group(1), "\n".join([*lines[:index], *lines[index + 1 :]])

        break

    return None, source


def demote_headings(source: str) -> str:
    lines: list[str] = []
    in_fence = False

    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence

        if in_fence:
            lines.append(line)
            continue

        match = HEADING_RE.match(line)
        if match and len(match.group(1)) < 6:
            line = f"#{match.group(1)}{match.group(2)}"

        lines.append(line)

    return "\n".join(lines)


def prepare_notebook(notebook: nbformat.NotebookNode, fallback_title: str) -> str:
    title = fallback_title
    found_title = False
    cells = []

    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            cells.append(cell)
            continue

        source = cell.source
        if not found_title:
            notebook_title, source = split_title(source)
            if notebook_title:
                title = notebook_title
                found_title = True

        cell.source = demote_headings(source)
        if cell.source.strip():
            cells.append(cell)

    notebook.cells = cells
    return title


def write_markdown(notebook_path: Path) -> str:
    rel_path = notebook_path.relative_to(EXAMPLES_DIR)
    output_dir = GENERATED_DIR / rel_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files_dir = f"{notebook_path.stem}_files"

    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    title = prepare_notebook(notebook, notebook_title(notebook_path))
    exporter = MarkdownExporter()
    body, resources = exporter.from_notebook_node(
        notebook,
        resources={"output_files_dir": output_files_dir},
    )

    links = (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_link(notebook_path)})"
        f" "
        f"[![Binder](https://mybinder.org/badge_logo.svg)]"
        f"({binder_link(notebook_path)})"
    )
    content = f"# {title}\n\n{links}\n\n{body.strip()}\n"
    FilesWriter(build_directory=str(output_dir)).write(
        content,
        resources,
        notebook_name="index",
    )
    return title


def category_label(category: str) -> str:
    return " ".join(word.capitalize() for word in category.replace("_", " ").split())


def write_category_index(category: str, notebooks: list[tuple[Path, str]]) -> None:
    index_path = EXAMPLES_DIR / category / "index.md"
    links = [
        f"- [{title}](../../generated/examples/{notebook.relative_to(EXAMPLES_DIR).with_suffix('').as_posix()}/)"
        for notebook, title in notebooks
        if notebook.relative_to(EXAMPLES_DIR).parts[0] == category
    ]
    title = category_label(category)
    intro = (
        "Hands-on notebooks that walk through core workflows and real datasets, "
        "with ready-to-run code and model outputs."
    )
    content = "\n".join([f"# {title}", "", intro, "", *links, ""])
    index_path.write_text(content, encoding="utf-8")


def write_examples_index(categories: list[str]) -> None:
    index_path = EXAMPLES_DIR / "index.md"
    preferred_order = [
        "classification",
        "regression",
        "anomaly",
        "catastrophic_forgetting",
        "model_persistence",
    ]
    known = [category for category in preferred_order if category in categories]
    unknown = sorted(
        category for category in categories if category not in preferred_order
    )
    ordered_categories = [*known, *unknown]
    links = [
        f"- [{category_label(category)}]({category}/index.md)"
        for category in ordered_categories
    ]
    intro = (
        "Explore runnable notebooks that demonstrate how deep-river behaves in real streaming settings. "
        "The collection is organized from foundational tasks to advanced continual learning topics."
    )
    content = "\n".join(["# Examples", "", intro, "", *links, ""])
    index_path.write_text(content, encoding="utf-8")


def main() -> None:
    notebooks = find_notebooks()

    if GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    titles = [(notebook, write_markdown(notebook)) for notebook in notebooks]
    categories = sorted(
        {notebook.relative_to(EXAMPLES_DIR).parts[0] for notebook in notebooks}
    )

    for category in categories:
        write_category_index(category, titles)

    write_examples_index(categories)


if __name__ == "__main__":
    main()
