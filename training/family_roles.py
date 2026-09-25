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

ROOT_ID = "fleet-q38-teacher3k-transitive-roles-20260925-v1"


def seal(body: dict) -> dict:
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {**body, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}


def checked_sha(value: dict) -> str:
    expected = value.get("sha256")
    if expected != seal({key: item for key, item in value.items() if key != "sha256"})["sha256"]:
        raise ValueError("reviewed input digest mismatch")
    return expected


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
    return seal(result)


def legacy_projection(source: dict, split: dict, receipts: dict, reviewed: dict,
                      validation_families: int = 37) -> tuple[dict, dict]:
    """Bind the historical packer to the corrected, reviewed transitive roles."""
    inputs = {"teacher_lineage_map": checked_sha(source),
              "protected_split": checked_sha(split),
              "protected_receipts": checked_sha(receipts),
              "family_roles": checked_sha(reviewed)}
    if reviewed != roles(source, protected_split=split, protected_receipts=receipts,
                        validation_families=validation_families):
        raise ValueError("reviewed family roles differ from exact source inputs")
    mapped = [{"task_key": row["task_key"], "task_version_id": row["task_version_id"],
               "group_id": row["family_id"],
               "split": "final_test" if row["split"] == "test" else row["split"]}
              for row in reviewed["identities"]]
    if len({row["task_version_id"] for row in mapped}) != len(mapped):
        raise ValueError("task version ID is shared by multiple task keys")
    groups = {row["group_id"] for row in mapped}
    heldout = sorted({row["group_id"] for row in mapped if row["split"] != "train"})
    anchor = seal({"schema": "fleet_teacher_transitive_role_anchor_v1",
                   "root_role_anchor_id": ROOT_ID,
                   "method": {"family": "task-key+shared-base-atom-transitive",
                              "protected": "heldout-quarantine",
                              "validation_seed": "teacher-validation-20260925-v1",
                              "validation_families": validation_families},
                   "inputs_sha256": inputs,
                   "counts": {"versions": len(mapped), "families": len(groups),
                              "nontraining_families": len(heldout)}})
    roster = seal({"schema": "cyber_exact_task_family_role_roster_v1",
                   "root_role_anchor_id": ROOT_ID,
                   "family_role_anchor_sha256": anchor["sha256"],
                   "heldout_group_ids": heldout, "identities": mapped})
    return anchor, roster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-map", required=True)
    parser.add_argument("--protected-split", required=True)
    parser.add_argument("--protected-receipts", required=True)
    parser.add_argument("--validation-families", type=int, default=37)
    parser.add_argument("--output", type=Path, help="create one immutable metadata-only roster")
    parser.add_argument("--reviewed-roles", type=Path)
    parser.add_argument("--anchor-output", type=Path)
    parser.add_argument("--legacy-output", type=Path)
    args = parser.parse_args()
    source, split, receipts = (load(args.teacher_map), load(args.protected_split),
                               load(args.protected_receipts))
    if args.reviewed_roles:
        if not args.anchor_output or not args.legacy_output or args.output:
            parser.error("projection requires --anchor-output and --legacy-output only")
        anchor, roster = legacy_projection(source, split, receipts,
                                           load(args.reviewed_roles), args.validation_families)
        for path, value in ((args.anchor_output, anchor), (args.legacy_output, roster)):
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False) + "\n")
        return
    text = json.dumps(roles(source, protected_split=split, protected_receipts=receipts,
                            validation_families=args.validation_families),
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
