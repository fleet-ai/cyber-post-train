.PHONY: sync doctor test lint check

sync:
	uv sync --extra dev

doctor:
	uv run cyber-post-train doctor

test:
	uv run pytest

lint:
	uv run ruff check .

check: lint test
