"""Freeze a teacher-only validation split by transitive task/atom family.

Input is an exact-version, metadata-only lineage map. No session text is read.
The selection is outcome-blind; it never consults loss or task scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from training.qualify import components, load, metadata_only


def protected_atoms(split: dict, receipts: dict) -> set[str]:
    proven = {(row["task_key"], row["task_version_id"]): row
              for row in receipts["task_versions"]}
    atoms = set()
    for row in split["tasks"]:
        if row["split"] not in {"dev", "final_test"}:
            continue
        exact = row["task_key"], row["task_version_id"]
        if exact not in proven:
            raise ValueError("protected Fleet version lacks exact receipt")
        atom = proven[exact]["lineage"]["task_family"].split("@", 1)[0]
        if not atom.startswith("cyber/atoms/"):
            raise ValueError("protected Fleet version lacks reviewed atom family")
        atoms.add(atom)
    if not atoms:
        raise ValueError("protected Fleet split is missing")
    return atoms


def roles(source: dict, *, protected_split: dict, protected_receipts: dict,
          validation_families: int = 37,
          seed: str = "teacher-validation-20260925-v1") -> dict:
    metadata_only(source)
    metadata_only(protected_split)
    metadata_only(protected_receipts)
    protected = protected_atoms(protected_split, protected_receipts)
    rows = source.get("training_task_keys")
    if not isinstance(rows, list) or not rows:
        raise ValueError("exact teacher lineage map is required")
    labels, known = components(source)
    identities = []
    families = defaultdict(lambda: {"tokens": 0, "sessions": 0,
                                    "apps": set(), "atoms": set()})
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
            facts["atoms"].update(atom.split("@", 1)[0] for atom in version["atom_lineages"])
            identities.append((identity, family))
    if len(identities) != len(set(identity for identity, _ in identities)):
        raise ValueError("duplicate exact source version")
    quarantined = {family for family, facts in families.items()
                   if facts["atoms"] & protected}
    eligible = set(families) - quarantined
    if not 0 < validation_families < len(eligible):
        raise ValueError("invalid validation-family count")
    ranked = sorted(eligible, key=lambda family: (
        hashlib.sha256(f"{seed}:{family}".encode()).hexdigest(), family))
    validation = set(ranked[:validation_families])
    all_apps = {app for family in eligible for app in families[family]["apps"]}
    val_apps = {app for family in validation for app in families[family]["apps"]}
    if val_apps != all_apps:
        raise ValueError("validation split misses a source application")
    result = {
        "schema": "fleet_teacher_family_roles_v1",
        "identities": [
            {"task_key": key, "task_version_id": version,
             "family_id": family,
             "split": ("test" if family in quarantined else
                       "dev" if family in validation else "train")}
            for ((key, version), family) in sorted(identities)
        ],
    }
    body = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    result["sha256"] = "sha256:" + hashlib.sha256(body).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-map", required=True)
    parser.add_argument("--protected-split", required=True)
    parser.add_argument("--protected-receipts", required=True)
    parser.add_argument("--validation-families", type=int, default=37)
    parser.add_argument("--output", type=Path, help="create one immutable metadata-only roster")
    args = parser.parse_args()
    text = json.dumps(roles(load(args.teacher_map),
                            protected_split=load(args.protected_split),
                            protected_receipts=load(args.protected_receipts),
                            validation_families=args.validation_families),
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
