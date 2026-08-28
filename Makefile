.PHONY: sync test lint check

sync:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .

check: lint test
