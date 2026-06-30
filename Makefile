COMMIT_HASH := $(shell eval git rev-parse HEAD)

# Test that uv is installed
check-uv:
	@which uv > /dev/null || (echo "uv is not installed. Please install it from https://github.com/astral-sh/uv" && exit 1)

install: check-uv
	uv sync --extra dev

format: check-uv
	uv run prek run --all-files

test: check-uv
	uv run pytest

execute-notebooks: check-uv
	uv run jupyter nbconvert --execute --to notebook --inplace docs/*/*/*.ipynb --ExecutePreprocessor.timeout=-1

doc: check-uv
	(cd benchmarks && uv run python render.py)
	uv run python docs/scripts/convert_notebooks.py
	uv run python docs/scripts/gen_ref_pages.py
	uv run zensical build --clean

livedoc: check-uv
	(cd benchmarks && uv run python render.py)
	uv run python docs/scripts/convert_notebooks.py
	uv run python docs/scripts/gen_ref_pages.py
	uv run zensical serve

rebase:
	git fetch && git rebase origin/main

clean:
	 rm -rf .venv/
	 rm -f uv.lock
	 rm -rf htmlcov .coverage coverage.xml
