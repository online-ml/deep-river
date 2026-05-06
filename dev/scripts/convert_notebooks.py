from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import nbformat
from nbconvert.exporters import MarkdownExporter


@dataclass(frozen=True)
class NotebookPage:
    source_path: Path
    output_path: Path
    title: str
    category: str


ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
EXAMPLES_DIR = DOCS_DIR / "examples"
GENERATED_DIR = DOCS_DIR / "generated" / "examples"
REPO_URL = "https://github.com/online-ml/deep-river"
DEFAULT_BRANCH = "master"


def notebook_title(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() for word in stem.split())


def find_notebooks() -> list[NotebookPage]:
    pages: list[NotebookPage] = []
    for notebook_path in sorted(EXAMPLES_DIR.rglob("*.ipynb")):
        rel_path = notebook_path.relative_to(EXAMPLES_DIR)
        category = rel_path.parts[0] if len(rel_path.parts) > 1 else "examples"
        output_path = GENERATED_DIR / rel_path.with_suffix("") / "index.md"
        pages.append(
            NotebookPage(
                source_path=notebook_path,
                output_path=output_path,
                title=notebook_title(notebook_path),
                category=category,
            )
        )
    return pages


def colab_link(notebook_path: Path) -> str:
    rel_path = notebook_path.relative_to(ROOT).as_posix()
    return f"https://colab.research.google.com/github/online-ml/deep-river/blob/{DEFAULT_BRANCH}/{rel_path}"


def binder_link(notebook_path: Path) -> str:
    rel_path = notebook_path.relative_to(ROOT).as_posix()
    return (
        "https://mybinder.org/v2/gh/online-ml/deep-river/"
        f"{DEFAULT_BRANCH}?filepath={rel_path}"
    )


def write_markdown(page: NotebookPage) -> None:
    page.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_files_dir = f"{page.source_path.stem}_files"

    with page.source_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    exporter = MarkdownExporter()
    body, resources = exporter.from_notebook_node(
        notebook,
        resources={"output_files_dir": output_files_dir},
    )

    links = (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_link(page.source_path)})"
        f" "
        f"[![Binder](https://mybinder.org/badge_logo.svg)]"
        f"({binder_link(page.source_path)})"
    )
    header = f"# {page.title}\n\n{links}\n\n"
    content = f'{header}<div class="notebook-content">\n\n{body}\n\n</div>\n'

    page.output_path.write_text(content, encoding="utf-8")

    if resources.get("outputs"):
        for name, data in resources["outputs"].items():
            target = page.output_path.parent / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(data, str):
                target.write_text(data, encoding="utf-8")
            else:
                target.write_bytes(data)


def category_label(category: str) -> str:
    return " ".join(word.capitalize() for word in category.replace("_", " ").split())


def write_category_index(category: str, pages: list[NotebookPage]) -> None:
    index_path = EXAMPLES_DIR / category / "index.md"
    links = [
        f"- [{page.title}](../../generated/examples/{page.source_path.relative_to(EXAMPLES_DIR).with_suffix('').as_posix()}/)"
        for page in pages
        if page.category == category
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
    unknown = sorted(category for category in categories if category not in preferred_order)
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
    pages = find_notebooks()

    if GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    categories = sorted({page.category for page in pages})

    for page in pages:
        write_markdown(page)

    for category in categories:
        write_category_index(category, pages)

    write_examples_index(categories)


if __name__ == "__main__":
    main()
