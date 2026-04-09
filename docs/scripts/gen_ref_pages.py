"""Backward-compatible entrypoint for reference generation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_docs_assets import generate_reference_pages


if __name__ == "__main__":
    generate_reference_pages()
