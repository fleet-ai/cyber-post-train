"""Metadata-only atom-family lineage shared by source and holdout planners.

Historical qualification reports were retired; this module does not certify
runtime health or authorize a rollout.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict

PRIVATE_FIELDS = {"prompt", "task_prompt", "transcript", "messages", "flag",
                  "answer", "score", "trajectory", "tool_outputs", "grader_source"}


def load(path: str) -> dict:
    if path == "-":
        value = json.load(sys.stdin)
    else:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def metadata_only(row: object) -> None:
    if isinstance(row, dict):
        if PRIVATE_FIELDS & set(row):
            raise ValueError("private task or trace field in metadata input")
        for value in row.values():
            metadata_only(value)
    elif isinstance(row, list):
        for value in row:
            metadata_only(value)


def identity(row: dict) -> tuple[str, str]:
    key = row.get("task_key")
    version = row.get("task_version_id", row.get("eval_task_version_id"))
    if not isinstance(key, str) or not key or not isinstance(version, str) or not version:
        raise ValueError("exact task key and version are required")
    return key, version


def atom(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("cyber/atoms/"):
        raise ValueError("reviewed atom artifact key required for lineage")
    return value.split("@", 1)[0]


def lineage_rows(document: dict):
    if "training_task_keys" in document:
        for task in document["training_task_keys"]:
            metadata_only(task)
            for version in task["versions"]:
                metadata_only(version)
                yield (task["task_key"], version["task_version_id"]), version["atom_lineages"]
    else:
        for row in document.get("task_versions", []):
            metadata_only(row)
            values = row.get("atom_artifact_keys")
            if values is None:
                family = row.get("lineage", {}).get("task_family", "")
                values = [family] if family.startswith("cyber/atoms/") else []
            yield identity(row), values


def components(*documents: dict) -> tuple[dict[tuple[str, str], str], set[tuple[str, str]]]:
    """Union exact versions, same-key aliases, and transitive shared atoms."""
    atoms_by_version: dict[tuple[str, str], tuple[str, ...]] = {}
    for document in documents:
        for task, values in lineage_rows(document):
            normalized = tuple(sorted({atom(value) for value in values}))
            if not normalized:
                continue
            if task in atoms_by_version and atoms_by_version[task] != normalized:
                raise ValueError("conflicting atom lineage for exact task version")
            atoms_by_version[task] = normalized
    parent = {task: task for task in atoms_by_version}

    def root(task):
        while parent[task] != task:
            parent[task] = parent[parent[task]]
            task = parent[task]
        return task

    def join(left, right):
        parent[root(right)] = root(left)

    by_key, by_atom = {}, {}
    for task, atoms in atoms_by_version.items():
        key = task[0]
        if key in by_key:
            join(task, by_key[key])
        by_key[key] = task
        for name in atoms:
            if name in by_atom:
                join(task, by_atom[name])
            by_atom[name] = task
    groups = defaultdict(list)
    for task in parent:
        groups[root(task)].append(task)
    labels = {}
    for members in groups.values():
        label = "sha256:" + hashlib.sha256(json.dumps(sorted(members), separators=(",", ":")).encode()).hexdigest()
        labels.update({task: label for task in members})
    return labels, set(atoms_by_version)
