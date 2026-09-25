"""Read-only, metadata-only qualification of Fleet blackbox SFT and eval inputs.

The inputs are *evidence projections*, not prompts or trajectories. A positive
result is valid only for the exact versions and observation time in those inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict

SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
SUCCESS = ("sft_eligible", "infrastructure_valid", "authoritative_success",
           "finite_outcome", "verifier_completed")
SFT = ("trace_intact", "retained_successful_report", "tool_contract_proven")
RUNTIME = ("environment_started", "tools_reachable", "verifier_executed",
           "finite_outcome", "cleanup_complete")
ROLES = {"train", "teacher_validation", "dev", "final_test"}
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


def metadata_only(row: dict) -> None:
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


def prove(row: dict, fields: tuple[str, ...], digests: tuple[str, ...]) -> bool:
    return all(row.get(name) is True for name in fields) and all(
        isinstance(row.get(name), str) and SHA.fullmatch(row[name]) for name in digests
    )


def report(catalog: dict, lineages: list[dict], split: dict | None = None,
           teacher_lineage: dict | None = None, sessions: dict | None = None,
           runtime: dict | None = None, include_rosters: bool = False) -> dict:
    if "tasks" not in catalog or not isinstance(catalog["tasks"], list):
        raise ValueError("catalog needs tasks[] from a metadata-only current-version census")
    metadata_only(catalog)
    catalog_rows = {}
    for row in catalog["tasks"]:
        metadata_only(row)
        key, version = identity(row)
        if row.get("task_shape") != "blackbox" and "blackbox" not in key.lower():
            continue
        if (key, version) in catalog_rows:
            raise ValueError("duplicate exact current task version")
        status = row.get("qa_status", row.get("status"))
        if status not in {"broken_task", "agent_failure", "clean", "not_analyzed"}:
            raise ValueError("unknown QA status")
        catalog_rows[key, version] = row
    if "task_count" in catalog and catalog["task_count"] != len(catalog_rows):
        raise ValueError("catalog declared task count does not match exact blackbox roster")

    labels, known = components(*lineages, *([teacher_lineage] if teacher_lineage else []))
    prior = set()
    for document in lineages:
        for row in document.get("task_versions", []):
            certificate = row.get("provenance", {}).get("certification", {})
            if certificate.get("status") == "accepted" and isinstance(
                    certificate.get("receipt_sha256"), str) and SHA.fullmatch(certificate["receipt_sha256"]):
                prior.add(identity(row))
    teacher = set()
    if teacher_lineage:
        teacher = {task for task, _ in lineage_rows(teacher_lineage)}
        if teacher - known:
            raise ValueError("teacher training version lacks reviewed atom lineage")
    exposed = {labels[task] for task in teacher}

    roles = {}
    for row in (split or {}).get("tasks", []):
        task = identity(row)
        role = row.get("split")
        if role not in ROLES or task not in known:
            raise ValueError("split needs a reviewed exact-version lineage and known role")
        if task in roles and roles[task] != role:
            raise ValueError("conflicting exact-version roles")
        roles[task] = role
    component_roles = defaultdict(set)
    for task, role in roles.items():
        component_roles[labels[task]].add(role)
    role_conflicts = sum(len(values) > 1 for values in component_roles.values())

    runtime_rows = {}
    for row in (runtime or {}).get("task_versions", []):
        metadata_only(row)
        task = identity(row)
        if task in runtime_rows:
            raise ValueError("duplicate current runtime receipt")
        runtime_rows[task] = row
    current_proven = set()
    needs = defaultdict(list)
    for task, row in catalog_rows.items():
        if row.get("qa_status", row.get("status")) == "broken_task":
            needs["known_broken"].append(task)
            continue
        if task not in known:
            needs["review_atom_lineage"].append(task)
        if row.get("lifecycle_status") != "production" or row.get("verifier_attached") is not True:
            needs["read_current_catalog_status"].append(task)
        if not prove(runtime_rows.get(task, {}), RUNTIME, ("receipt_sha256",)):
            needs["run_model_free_runtime_qualification"].append(task)
        if (task in known and row.get("lifecycle_status") == "production"
                and row.get("verifier_attached") is True
                and prove(runtime_rows.get(task, {}), RUNTIME, ("receipt_sha256",))):
            current_proven.add(task)

    qualified_sessions, sft_ready = set(), set()
    seen_digests = {name: set() for name in ("session_sha256", "receipt_sha256", "trace_sha256")}
    for row in (sessions or {}).get("sessions", []):
        metadata_only(row)
        task = identity(row)
        digest = row.get("session_sha256")
        for name, seen in seen_digests.items():
            value = row.get(name)
            if not isinstance(value, str) or not SHA.fullmatch(value) or value in seen:
                raise ValueError(f"missing or duplicate opaque {name}")
            seen.add(value)
        if task in known and prove(row, SUCCESS, ("receipt_sha256", "trace_sha256")):
            qualified_sessions.add(digest)
            if prove(row, SFT, ()) and roles.get(task) == "train":
                sft_ready.add(digest)

    clean_heldout = {task for task in current_proven if roles.get(task) in {"dev", "final_test"}
                     and labels[task] not in exposed and len(component_roles[labels[task]]) == 1}
    qa = Counter(row.get("qa_status", row.get("status")) for row in catalog_rows.values())
    unreviewed = {task for task, row in catalog_rows.items()
                  if row.get("qa_status", row.get("status")) == "not_analyzed" and task not in prior}
    needs["qualify_unreviewed_pool"] = sorted(unreviewed)
    result = {
        "schema": "fleet_blackbox_qualification_report_v1",
        "scope": "read-only point-in-time metadata; no task content or capability outcomes",
        "counts": {
            "current_blackbox_versions": len(catalog_rows),
            "qa": dict(sorted(qa.items())),
            "reviewed_lineage_current_versions": len(set(catalog_rows) & known),
            "prior_exact_receipt_current_versions": len(set(catalog_rows) & prior),
            "historical_success_receipt_complete_sessions": len(qualified_sessions),
            "historical_sft_ready_train_sessions": len(sft_ready),
            "current_runtime_receipt_complete_versions": len(current_proven),
            "lineage_clean_live_heldout_versions": len(clean_heldout),
            "unreviewed_not_analyzed_versions": len(set(needs["review_atom_lineage"]) &
                                                      {task for task, row in catalog_rows.items()
                                                       if row.get("qa_status", row.get("status")) == "not_analyzed"}),
            "not_analyzed_without_prior_exact_receipt": len(unreviewed),
            "split_component_role_conflicts": role_conflicts,
            "teacher_exposed_split_heldout_versions": sum(
                roles.get(task) in {"dev", "final_test"} and labels[task] in exposed
                for task in roles
            ),
        },
        "next_qualification": {name: len(rows) for name, rows in sorted(needs.items())},
        "launch_authorized": False,
        "split_safe": split is not None and role_conflicts == 0 and not any(
            roles.get(task) in {"dev", "final_test"} and labels[task] in exposed for task in roles
        ),
    }
    if include_rosters:
        result["rosters"] = {
            name: [{"task_key": key, "task_version_id": version} for key, version in sorted(rows)]
            for name, rows in sorted(needs.items())
        }
        result["rosters"]["lineage_clean_live_heldout"] = [
            {"task_key": key, "task_version_id": version} for key, version in sorted(clean_heldout)
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="metadata-only current task roster JSON; '-' reads stdin")
    parser.add_argument("--lineage", action="append", required=True, help="reviewed exact-version atom lineage JSON; repeatable")
    parser.add_argument("--split", help="frozen exact-version role JSON")
    parser.add_argument("--teacher-lineage", help="reviewed source corpus lineage map JSON")
    parser.add_argument("--sessions", help="metadata-only historical-success receipt projection JSON")
    parser.add_argument("--runtime", help="exact-version model-free runtime receipt projection JSON")
    parser.add_argument("--rosters", action="store_true", help="include exact key/version rosters; never task content")
    args = parser.parse_args()
    value = report(load(args.catalog), [load(path) for path in args.lineage],
                   load(args.split) if args.split else None,
                   load(args.teacher_lineage) if args.teacher_lineage else None,
                   load(args.sessions) if args.sessions else None,
                   load(args.runtime) if args.runtime else None, args.rosters)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
