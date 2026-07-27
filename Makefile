.DEFAULT_GOAL := help
.PHONY: help setup fmt lint type test cov check example build clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install git hooks
	uv sync
	uv run pre-commit install

fmt:  ## Format and autofix
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint without writing
	uv run ruff check .
	uv run ruff format --check .

type:  ## Type check
	uv run mypy

test:  ## Run the test suite
	uv run pytest

cov:  ## Run the test suite with coverage
	uv run pytest --cov --cov-report=term-missing

check: lint type test  ## Everything CI runs

example:  ## Train a tokenizer and print what it learned
	uv run python examples/train_and_inspect.py

build:  ## Build sdist and wheel
	uv build

clean:  ## Remove build and tool caches
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
