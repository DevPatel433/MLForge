.PHONY: install test cov lint format docs docs-site dashboard mutation gallery notebooks benchmark benchmark-quick build clean all

install:            ## editable install with all dev dependencies
	pip install -e ".[dev]"

test:               ## run the test-suite
	pytest

cov:                ## tests + coverage report (fails under 95%)
	pytest --cov=mlforge --cov-report=term-missing --cov-fail-under=95

lint:               ## static checks
	ruff check .

format:             ## auto-fix what ruff can fix
	ruff check --fix .

docs:               ## regenerate the API reference
	python docs/generate_api.py

docs-site:          ## build the documentation site (strict: broken links fail)
	python docs/generate_api.py && mkdocs build --strict

dashboard:          ## run the suite under coverage and rebuild docs/correctness.md + badges
	python tools/build_dashboard.py

mutation:           ## mutation testing of the numerical core (tens of minutes)
	python tools/mutation_test.py

gallery:            ## regenerate README / docs images (about a minute)
	python examples/make_gallery.py

notebooks:          ## rebuild + execute the example notebooks
	python notebooks/build_notebooks.py --execute

benchmark:          ## full MLForge-vs-scikit-learn benchmark -> benchmarks/results/
	python -m mlforge benchmark

benchmark-quick:    ## smoke benchmark on smaller data
	python -m mlforge benchmark --quick --out /tmp/mlforge-bench

build:              ## sdist + wheel
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

all: lint cov docs
