"""Fail-closed Python import-closure validation for ConfigMap runtimes."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable, Mapping


BINDING = re.compile(r"([A-Za-z0-9_-]+\.py):([A-Za-z0-9_-]+\.py)")
PREFIX = "evals.fleet."


def installed_modules(data: Mapping[str, str], runner_key: str = "run.sh") -> dict[str, str]:
    runner = data.get(runner_key)
    if not runner:
        raise ValueError("packaged runner is absent")
    installed: dict[str, str] = {}
    for source, destination in BINDING.findall(runner):
        if source not in data or not data[source]:
            raise ValueError(f"runner source is absent: {source}")
        module = Path(destination).stem
        if module in installed and installed[module] != data[source]:
            raise ValueError(f"multiple bytes install as {module}")
        installed[module] = data[source]
    return installed


def direct_fleet_imports(source: str) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == "evals.fleet":
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PREFIX):
                    result.add(alias.name.removeprefix(PREFIX).split(".", 1)[0])
    return result


def validate(
    data: Mapping[str, str], *, dynamic_import_allowlist: Iterable[str] = ()
) -> dict[str, tuple[str, ...]]:
    installed = installed_modules(data)
    allowlist = set(dynamic_import_allowlist)
    missing: dict[str, list[str]] = {}
    graph: dict[str, tuple[str, ...]] = {}
    for module, source in sorted(installed.items()):
        imports = direct_fleet_imports(source)
        graph[module] = tuple(sorted(imports))
        for dependency in imports:
            if dependency not in installed and dependency not in allowlist:
                missing.setdefault(dependency, []).append(module)
    if missing:
        detail = ", ".join(
            f"{dependency} imported by {','.join(sorted(importers))}"
            for dependency, importers in sorted(missing.items())
        )
        raise ValueError(f"packaged Python import closure is incomplete: {detail}")
    return graph

