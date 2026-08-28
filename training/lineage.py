"""Immutable lineage extraction and leakage-safe grouping."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

BLACKBOX_SUFFIX = re.compile(r"(?:__|_)(?:blackbox_ctf_v\d+)(?:_.+)?$")
ENV_PREFIX = re.compile(r"^cysec\d+(?:-\d+)?-[^_]+(?:-gen)?_")


def _find_mapping(value: Any, names: set[str]) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, Mapping):
                return candidate
        for candidate in value.values():
            found = _find_mapping(candidate, names)
            if found:
                return found
    elif isinstance(value, list):
        for candidate in value:
            found = _find_mapping(candidate, names)
            if found:
                return found
    return None


def _locator(mapping: Mapping[str, Any] | None) -> tuple[str, int] | None:
    if not mapping:
        return None
    key = mapping.get("artifact_key") or mapping.get("key")
    version = mapping.get("version_index")
    if isinstance(key, str) and key.startswith("cyber/task-graphs/") and isinstance(version, int):
        return key, version
    return None


def task_family(task_key: str) -> str:
    family = ENV_PREFIX.sub("", task_key)
    family = BLACKBOX_SUFFIX.sub("", family)
    return family or task_key


def infer_app(task: Mapping[str, Any], instance: Mapping[str, Any]) -> str:
    for value in (instance.get("env_key"), task.get("env_id"), task.get("key")):
        if not isinstance(value, str):
            continue
        match = re.search(r"cysec\d+(?:-\d+)?-([a-z0-9]+)", value)
        if match:
            return match.group(1)
    return "unknown"


def extract_lineage(task: Mapping[str, Any], instance: Mapping[str, Any]) -> dict[str, Any]:
    task_key = str(task.get("key") or "unknown-task")
    locator = _locator(_find_mapping(task, {"task_graph_locator", "source_locator"}))
    if locator:
        artifact_key, version_index = locator
        lineage_key = f"registry:{artifact_key}@{version_index}"
        quality = "immutable_registry"
    elif isinstance(task.get("eval_task_version_id"), str) and task["eval_task_version_id"]:
        lineage_key = f"fleet-task-version:{task['eval_task_version_id']}"
        quality = "fleet_task_version_id"
    else:
        lineage_key = f"task-key:{task_key}"
        quality = "task_key_only"
    prompt = str(task.get("prompt") or "")
    prompt_hash = "sha256:" + hashlib.sha256(prompt.encode()).hexdigest()
    verifier_revision = task.get("verifier_sha") or task.get("verifier_version")
    bindings = {
        "eval_task_version_id": task.get("eval_task_version_id"),
        "task_revision": task.get("version"),
        "prompt_sha256": prompt_hash,
        "verifier_id": task.get("verifier_id"),
        "verifier_revision": verifier_revision,
        "environment_version": instance.get("version"),
        "environment_profile_id": instance.get("profile_id"),
        "environment_data_version": instance.get("data_version") or task.get("data_version"),
    }
    return {
        "lineage_key": lineage_key,
        "lineage_quality": quality,
        "task_key": task_key,
        "task_family": task_family(task_key),
        "application": infer_app(task, instance),
        "task_version": task.get("version"),
        "eval_task_version_id": task.get("eval_task_version_id"),
        "environment_version": instance.get("version"),
        "environment_data_key": instance.get("data_key") or task.get("data_id"),
        "environment_data_version": instance.get("data_version") or task.get("data_version"),
        "prompt_sha256": prompt_hash,
        "bindings": bindings,
        "missing_exact_bindings": [name for name, value in bindings.items() if value in {None, ""}],
    }


def leakage_group(lineage: Mapping[str, Any]) -> str:
    """Group correlated variants so no split can cross the family boundary."""
    basis = {
        "application": lineage.get("application"),
        "family": lineage.get("task_family"),
        "registry": (
            lineage.get("lineage_key")
            if lineage.get("lineage_quality") == "immutable_registry"
            else None
        ),
    }
    return (
        "leakage:"
        + hashlib.sha256(
            json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
