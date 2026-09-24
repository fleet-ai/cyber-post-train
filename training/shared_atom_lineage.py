"""Build deterministic task components from shared immutable atom identities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from training.task_family_split import canonical_digest


def atom_artifact_key(value: object) -> str:
    """Return the stable artifact key from a bare key or exact source locator."""
    if not isinstance(value, str) or not value:
        raise ValueError("atom identity must be a nonempty string")
    key = value
    if value.endswith(":atom_source"):
        locator = value.removesuffix(":atom_source")
        try:
            key, version = locator.rsplit("@", 1)
        except ValueError as error:
            raise ValueError("atom source locator must contain a version") from error
        if not version.isdigit():
            raise ValueError("atom source locator version must be a nonnegative integer")
    elif "@" in value or ":" in value:
        raise ValueError("atom identity must be a bare key or exact atom_source locator")
    if "@" in key or ":" in key:
        raise ValueError("atom artifact key contains a reserved locator delimiter")
    parts = key.removeprefix("cyber/atoms/").split("/")
    if not key.startswith("cyber/atoms/") or len(parts) != 2 or not all(parts):
        raise ValueError("atom artifact key must be cyber/atoms/<application>/<atom>")
    return key


def task_atom_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize exact task rows while keeping content out of the result."""
    normalized = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("lineage census contains a non-object task row")
        task_key = raw.get("task_key")
        task_version_id = raw.get("task_version_id")
        sources = raw.get("atom_artifact_keys")
        if not isinstance(task_key, str) or not task_key:
            raise ValueError("task_key is required")
        if not isinstance(task_version_id, str) or not task_version_id:
            raise ValueError("task_version_id is required")
        identity = (task_key, task_version_id)
        if identity in seen:
            raise ValueError("lineage census duplicates an exact task version")
        seen.add(identity)
        if not isinstance(sources, list) or not sources:
            raise ValueError("every task version needs reviewed atom lineage")
        atoms = sorted({atom_artifact_key(source) for source in sources})
        if len(atoms) != len(sources):
            raise ValueError("task atom lineage contains duplicate artifact keys")
        normalized.append(
            {
                "task_key": task_key,
                "task_version_id": task_version_id,
                "atom_artifact_keys": atoms,
            }
        )
    return sorted(normalized, key=lambda row: (row["task_key"], row["task_version_id"]))


def build_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group the transitive closure of shared atoms and logical task versions.

    Atom version indices are deliberately removed before grouping, so successor
    versions of one atom stay together. Composite tasks union all of their atom
    keys; sharing any constituent atom joins the complete connected component.
    Exact versions sharing a task key also stay together even if their source
    metadata changes.
    """
    tasks = task_atom_rows(rows)
    parents = list(range(len(tasks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    by_atom: dict[str, int] = {}
    by_task_key: dict[str, int] = {}
    for index, row in enumerate(tasks):
        for atom in row["atom_artifact_keys"]:
            if atom in by_atom:
                union(index, by_atom[atom])
            else:
                by_atom[atom] = index
        if row["task_key"] in by_task_key:
            union(index, by_task_key[row["task_key"]])
        else:
            by_task_key[row["task_key"]] = index

    members: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(tasks):
        members[find(index)].append(row)

    components = []
    task_versions = []
    for rows_in_component in members.values():
        atoms = sorted({atom for row in rows_in_component for atom in row["atom_artifact_keys"]})
        component_id = canonical_digest({"atom_artifact_keys": atoms})
        identities = sorted(
            (
                {"task_key": row["task_key"], "task_version_id": row["task_version_id"]}
                for row in rows_in_component
            ),
            key=lambda row: (row["task_key"], row["task_version_id"]),
        )
        components.append(
            {
                "component_id": component_id,
                "atom_artifact_keys": atoms,
                "task_versions": identities,
            }
        )
        task_versions.extend({**identity, "component_id": component_id} for identity in identities)
    components.sort(key=lambda row: row["component_id"])
    task_versions.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    return {"components": components, "task_versions": task_versions}


def component_roles(
    components: dict[str, Any], role_by_identity: dict[tuple[str, str], str]
) -> dict[str, str]:
    """Return inherited component roles, rejecting any train/heldout bridge."""
    roles = {}
    for component in components["components"]:
        inherited = {
            role_by_identity[(row["task_key"], row["task_version_id"])]
            for row in component["task_versions"]
            if (row["task_key"], row["task_version_id"]) in role_by_identity
        }
        if len(inherited) > 1:
            raise ValueError("one shared-atom component crosses frozen split roles")
        if inherited:
            roles[component["component_id"]] = inherited.pop()
    return roles
