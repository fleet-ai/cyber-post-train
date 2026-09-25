"""Read-only Fleet census for a model-free blackbox qualification wave.

Only exact IDs, source locators, and safe metadata leave this process. A task is
not called runnable until a separate runtime/tools/verifier/cleanup probe passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx

TEAM = "a1025f0b-ad67-49fc-a023-51800ab43e84"
PROJECT = "63d6fda8-48c4-4726-9ec3-d1028f2c47f5"
PUBLIC = "https://orchestrator.fleetai.com"
PRIVATE = "https://api.internal.fleet-platform.fleetai.com"


def read(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def bare_atom(value: str) -> str:
    return value.split("@", 1)[0]


def identity(row: dict, version_field: str) -> tuple[str, str]:
    return row["task_key"], row[version_field]


def candidates(qa: list[dict], teacher: dict, coverage: dict,
               heldout: dict, pending: dict, fetch) -> dict:
    teacher_rows = teacher["training_task_keys"]
    teacher_keys = {row["task_key"] for row in teacher_rows}
    teacher_atoms = {bare_atom(atom) for row in teacher_rows
                     for version in row["versions"] for atom in version["atom_lineages"]}
    protected = {bare_atom(row["reviewed_task_family"])
                 for row in heldout["selection"]["tasks"]}
    protected |= {row["atom_artifact_key"] for row in pending["roster"]}
    protected_keys = {row["task_key"] for row in heldout["selection"]["tasks"]}
    protected_keys |= {row["task_key"] for row in pending["roster"]}
    prior = {identity(row, "task_version_id")
             for row in coverage["exact_receipt_proven_task_versions"]}
    pool = [row for row in qa if "blackbox" in row["task_key"]
            and row["status"] == "not_analyzed"
            and identity(row, "eval_task_version_id") not in prior]
    unseen = [row for row in pool if row["task_key"] not in teacher_keys | protected_keys]
    with ThreadPoolExecutor(max_workers=8) as workers:
        metadata = list(workers.map(fetch, unseen))
    if len(metadata) != len(unseen) or any(row.get("error") for row in metadata):
        raise ValueError("incomplete exact-version metadata census")
    valid = []
    for qa_row, row in zip(unseen, metadata, strict=True):
        if identity(row, "task_version_id") != identity(qa_row, "eval_task_version_id"):
            raise ValueError("metadata exact-version mismatch")
        if row["task_id"] != qa_row["eval_task_id"]:
            raise ValueError("metadata exact task ID mismatch")
        if row["lifecycle"] != "production" or not row["verifier_attached"]:
            continue
        atoms = row["atom_artifact_keys"]
        if not atoms or not row["environment_version_id"]:
            continue
        valid.append(row)
    teacher_unexposed = [row for row in valid
                         if not set(row["atom_artifact_keys"]) & teacher_atoms]
    unexposed = [row for row in teacher_unexposed
                 if not set(row["atom_artifact_keys"]) & protected]
    # Transitive components, not one task per graph or one task per atom.
    parent = list(range(len(unexposed)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owner = {}
    for index, row in enumerate(unexposed):
        for atom in row["atom_artifact_keys"]:
            if atom in owner:
                parent[root(index)] = root(owner[atom])
            owner[atom] = index
    groups = defaultdict(list)
    for index, row in enumerate(unexposed):
        groups[root(index)].append(row)
    representatives = [min(rows, key=lambda row: (len(row["atom_artifact_keys"]),
                                                    row["difficulty"] is None,
                                                    row["task_key"]))
                       for rows in groups.values()]
    by_app = defaultdict(list)
    for row in representatives:
        by_app[row["applications"][0]].append(row)
    for rows in by_app.values():
        rows.sort(key=lambda row: (row["difficulty"] is None,
                                   len(row["atom_artifact_keys"]), row["task_key"]))
    wave = []
    while any(by_app.values()):
        for app in sorted(by_app, key=lambda app: (-len(by_app[app]), app)):
            if by_app[app]:
                wave.append(by_app[app].pop(0))
    return {
        "schema": "fleet_blackbox_readonly_qualification_wave_v1",
        "observed_at": datetime.now(UTC).isoformat(),
        "counts": {
            "blackbox": sum("blackbox" in row["task_key"] for row in qa),
            "not_analyzed_without_prior_receipt": len(pool),
            "unseen_task_keys": len(unseen),
            "exact_metadata_valid": len(valid),
            "teacher_unexposed_versions": len(teacher_unexposed),
            "teacher_unexposed_apps": len({app for row in teacher_unexposed
                                           for app in row["applications"]}),
            "teacher_and_protected_unexposed_versions": len(unexposed),
            "teacher_and_protected_unexposed_apps": len({app for row in unexposed
                                                         for app in row["applications"]}),
            "independent_candidate_families": len(groups),
        },
        "wave": wave,
        "launch_authorized": False,
    }


def safe_task(row: dict, client: httpx.Client) -> dict:
    key, version = identity(row, "eval_task_version_id")
    response = client.get(f"{PUBLIC}/v1/tasks/{quote(key, safe='')}",
                          params={"version_id": version})
    response.raise_for_status()
    task = response.json()  # Response contains private fields; never print or persist it.
    metadata = task.get("metadata") or {}
    subject = metadata.get("cyber_subject") or {}
    sources = subject.get("atom_sources") or []
    atoms = []
    for source in sources:
        atom = source.get("artifact_key")
        index = source.get("version_index")
        if (not isinstance(atom, str) or not atom.startswith("cyber/atoms/")
                or type(index) is not int or index < 0
                or source.get("locator") != f"{atom}@{index}:atom_source"):
            raise ValueError("incomplete exact atom-source binding")
        atoms.append(atom)
    if len(atoms) != len(set(atoms)):
        raise ValueError("duplicate atom-source binding")
    return {
        "task_key": task.get("key"),
        "task_version_id": task.get("eval_task_version_id"),
        "task_id": row["eval_task_id"],
        "atom_artifact_keys": sorted(atoms),
        "applications": sorted({atom.split("/")[2] for atom in atoms}),
        "environment_version_id": task.get("environment_version_id"),
        "difficulty": metadata.get("task_graph_band") or metadata.get("expected_difficulty"),
        "lifecycle": task.get("task_lifecycle_status"),
        "verifier_attached": bool(task.get("verifier_id")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "coverage", "heldout", "pending"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output", help="write one immutable local JSON plan")
    parser.add_argument("--counts-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("limit must be 1..100")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        parser.error("FLEET_API_KEY is unavailable")
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=25,
                      limits=httpx.Limits(max_connections=8)) as client:
        account = client.get(PUBLIC + "/v1/account")
        account.raise_for_status()
        if account.json().get("team_id") != TEAM:
            raise ValueError("Fleet team identity mismatch")
        route = f"{PRIVATE}/v1/qa/projects/{PROJECT}/task-quality"
        first = client.get(route); first.raise_for_status()
        qa = first.json()["tasks"]
        result = candidates(qa, read(args.teacher), read(args.coverage),
                            read(args.heldout), read(args.pending),
                            lambda row: safe_task(row, client))
        second = client.get(route); second.raise_for_status()
        projection = lambda rows: sorted((row["task_key"], row["eval_task_version_id"],
                                          row["status"]) for row in rows)
        if projection(qa) != projection(second.json()["tasks"]):
            raise ValueError("live QA roster drifted during metadata census")
    result["wave"] = result["wave"][:args.limit]
    body = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    result["sha256"] = "sha256:" + hashlib.sha256(body.encode()).hexdigest()
    text = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if args.counts_only:
        print(json.dumps(result["counts"], sort_keys=True))
    elif args.output:
        with Path(args.output).open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
