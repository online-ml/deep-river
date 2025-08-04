COMMIT_HASH := $(shell eval git rev-parse HEAD)

# Test that uv is installed
check-uv:
	@which uv > /dev/null || (echo "uv is not installed. Please install it from https://github.com/astral-sh/uv" && exit 1)

install: check-uv
	uv sync --extra dev

format: check-uv
	uv run pre-commit run --all-files

test: check-uv
	uv run pytest

execute-notebooks: check-uv
	uv run jupyter nbconvert --execute --to notebook --inplace docs/*/*/*.ipynb --ExecutePreprocessor.timeout=-1

doc: check-uv
	(cd benchmarks && uv run python render.py)
	uv run mkdocs build

livedoc: doc
	uv run mkdocs serve --dirtyreload

rebase:
	git fetch && git rebase origin/master

clean:
	rm -rf .venv/
	rm -f uv.lock