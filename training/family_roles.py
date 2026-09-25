"""Freeze a teacher-only validation split by transitive task/atom family.

Input is an exact-version, metadata-only lineage map. No session text is read.
The selection is outcome-blind; it never consults loss or task scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict

from training.qualify import components, load, metadata_only


def roles(source: dict, *, validation_families: int = 37,
          seed: str = "teacher-validation-20260925-v1") -> dict:
    metadata_only(source)
    rows = source.get("training_task_keys")
    if not isinstance(rows, list) or not rows:
        raise ValueError("exact teacher lineage map is required")
    labels, known = components(source)
    identities = []
    families = defaultdict(lambda: {"tokens": 0, "sessions": 0, "apps": set()})
    for task in rows:
        for version in task["versions"]:
            identity = task["task_key"], version["task_version_id"]
            if identity not in known:
                raise ValueError("source version lacks reviewed atom lineage")
            family = labels[identity]
            facts = families[family]
            facts["tokens"] += version["supervised_tokens"]
            facts["sessions"] += version["source_sessions"]
            facts["apps"].update(atom.split("/")[2] for atom in version["atom_lineages"])
            identities.append((identity, family))
    if len(identities) != len(set(identity for identity, _ in identities)):
        raise ValueError("duplicate exact source version")
    if not 0 < validation_families < len(families):
        raise ValueError("invalid validation-family count")
    ranked = sorted(families, key=lambda family: (
        hashlib.sha256(f"{seed}:{family}".encode()).hexdigest(), family))
    validation = set(ranked[:validation_families])
    all_apps = {app for facts in families.values() for app in facts["apps"]}
    val_apps = {app for family in validation for app in families[family]["apps"]}
    if val_apps != all_apps:
        raise ValueError("validation split misses a source application")
    result = {
        "schema": "fleet_teacher_family_roles_v1",
        "identities": [
            {"task_key": key, "task_version_id": version,
             "family_id": family,
             "split": "dev" if family in validation else "train"}
            for ((key, version), family) in sorted(identities)
        ],
    }
    body = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    result["sha256"] = "sha256:" + hashlib.sha256(body).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-map", required=True)
    parser.add_argument("--validation-families", type=int, default=37)
    args = parser.parse_args()
    print(json.dumps(roles(load(args.teacher_map), validation_families=args.validation_families),
                     sort_keys=True, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
