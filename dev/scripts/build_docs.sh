#!/usr/bin/env bash

set -euo pipefail

repo_root="${1:-.}"

if [[ ! -d "$repo_root" ]]; then
  printf 'Repository path does not exist: %s\n' "$repo_root" >&2
  exit 1
fi

if [[ -f "$repo_root/benchmarks/render.py" ]]; then
  (
    cd "$repo_root/benchmarks"
    uv run python render.py
  )
fi

if [[ -f "$repo_root/docs/scripts/convert_notebooks.py" ]]; then
  (
    cd "$repo_root"
    uv run python docs/scripts/convert_notebooks.py
  )
fi

if [[ -f "$repo_root/docs/scripts/gen_ref_pages.py" ]]; then
  (
    cd "$repo_root"
    uv run python docs/scripts/gen_ref_pages.py
  )
fi

if [[ -f "$repo_root/zensical.toml" ]]; then
  (
    cd "$repo_root"
    uv run zensical build --clean
  )
  exit 0
fi

if [[ -f "$repo_root/mkdocs.yml" || -f "$repo_root/mkdocs.yaml" ]]; then
  (
    cd "$repo_root"
    uv run mkdocs build
  )
  exit 0
fi

printf 'No supported docs configuration found in %s\n' "$repo_root" >&2
exit 1
