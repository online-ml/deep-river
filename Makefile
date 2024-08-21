COMMIT_HASH := $(shell eval git rev-parse HEAD)

format:
	pre-commit run --all-files

test:
	pytest

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace docs/*/*/*.ipynb --ExecutePreprocessor.timeout=-1

doc:
	(cd benchmarks && python render.py)
	mkdocs build

livedoc: doc
	mkdocs serve --dirtyreload

rebase:
	git fetch && git rebase origin/master