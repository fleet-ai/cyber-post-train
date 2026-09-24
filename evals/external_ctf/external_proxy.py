"""Null-seed entrypoint for the existing fixed inference proxy."""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any


def null_seed_overrides(
    validate: Callable[[dict[str, Any]], dict[str, Any]], value: dict[str, Any]
) -> dict[str, Any]:
    """Reuse the fixed policy validator while preserving an explicit null seed."""
    if value.get("seed", object()) is not None:
        raise ValueError("external CTF attempts require a null seed")
    checked = validate({**value, "seed": 0})
    return {**checked, "seed": None}


def proxy_namespace() -> dict[str, Any]:
    source = Path(__file__).with_name("fixed_proxy.py")
    if not source.is_file():
        source = Path(__file__).parents[1] / "fleet/fixed_proxy.py"
    namespace = runpy.run_path(str(source))
    runtime = namespace["Handler"]._forward.__globals__  # noqa: SLF001
    validate = runtime["completion_overrides"]
    runtime["completion_overrides"] = lambda value: null_seed_overrides(validate, value)
    return namespace


def main() -> None:
    namespace = proxy_namespace()
    namespace["ThreadingHTTPServer"](("0.0.0.0", 8877), namespace["Handler"]).serve_forever()


if __name__ == "__main__":
    main()
