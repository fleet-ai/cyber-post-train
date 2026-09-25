"""Freeze an outcome-blind, exact-version Fleet qualification queue.

This only ranks candidates. It never creates an environment or admits a task.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HINTS = {
    "injection": ("sql", "inject", "ssti", "template", "command", "shell"),
    "identity": ("auth", "login", "session", "jwt", "token", "oauth", "spoof"),
    "access": ("idor", "tenant", "privilege", "permission", "role", "acl"),
    "file": ("path", "traversal", "upload", "file", "archive"),
    "network": ("ssrf", "redirect", "webhook", "host-header", "proxy"),
    "browser": ("xss", "csrf", "html", "script"),
    "logic": ("race", "workflow", "payment", "business-logic"),
}
FIELDS = {"task_key", "task_version_id", "task_id", "atom_artifact_keys", "applications",
          "environment_version_id", "difficulty", "source_project", "source_repo",
          "lifecycle", "verifier_attached"}


def seal(value: dict) -> dict:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return value | {"sha256": "sha256:" + hashlib.sha256(body).hexdigest()}


def checked(value: dict) -> dict:
    digest = value["sha256"]
    if seal({key: item for key, item in value.items() if key != "sha256"})["sha256"] != digest:
        raise ValueError("input digest mismatch")
    return value


def atom_hint(atoms: set[str]) -> str:
    slugs = [atom.rsplit("/", 1)[-1] for atom in atoms]
    matches = [kind for kind, words in HINTS.items()
               if any(any(re.search(rf"(^|-)({word})(-|$)", slug) for word in words)
                      for slug in slugs)]
    return "mixed" if len(matches) > 1 else matches[0] if matches else "unclassified"


def plan(rows: list[dict], *, dev_count: int = 60) -> dict:
    if not rows or len({(r["task_key"], r["task_version_id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate or empty exact-version roster")
    if any(set(r) != FIELDS or r["lifecycle"] != "production" or not r["verifier_attached"]
           or not r["atom_artifact_keys"] or not r["environment_version_id"] for r in rows):
        raise ValueError("unsafe or incomplete candidate metadata")
    parents = list(range(len(rows)))
    owners = {}

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, row in enumerate(rows):
        for key in ["task:" + row["task_key"], *("atom:" + a for a in row["atom_artifact_keys"])]:
            if key in owners:
                parents[root(i)] = root(owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    families = []
    for members in groups.values():
        members.sort(key=lambda r: (r["task_key"], r["task_version_id"]))
        atoms = {a for r in members for a in r["atom_artifact_keys"]}
        apps = sorted({a for r in members for a in r["applications"]})
        difficulty = sorted({str(r["difficulty"] or "unknown") for r in members})
        provenance = sorted({f'{r["source_repo"]}:{r["source_project"]}' for r in members})
        identities = [[r["task_key"], r["task_version_id"]] for r in members]
        representative = min(members, key=lambda r: (len(r["atom_artifact_keys"]),
                                                     r["difficulty"] is None,
                                                     r["task_key"], r["task_version_id"]))
        families.append({"family_id": seal({"identities": identities})["sha256"],
                         "versions": identities,
                         "representative": [representative["task_key"], representative["task_version_id"]],
                         "strata": {"app": apps[0], "apps": apps,
                                    "difficulty": difficulty[0] if len(difficulty) == 1 else "mixed",
                                    "atom_shape": "single" if len(atoms) == 1 else "multi",
                                    "atom_name_hint": atom_hint(atoms),
                                    "provenance": provenance[0] if len(provenance) == 1 else "mixed"}})
    by_app = defaultdict(list)
    for family in families:
        by_app[family["strata"]["app"]].append(family)
    totals = {app: len(group) for app, group in by_app.items()}
    seen = Counter()
    ordered = []
    while any(by_app.values()):
        for app in sorted(totals, key=lambda name: (-totals[name], name)):
            if not by_app[app]:
                continue
            def rank(family: dict) -> tuple:
                s = family["strata"]
                fields = (s["atom_name_hint"], s["difficulty"], s["provenance"], s["atom_shape"])
                return (sum(seen[field] for field in fields),
                        hashlib.sha256(family["family_id"].encode()).hexdigest())
            chosen = min(by_app[app], key=rank)
            by_app[app].remove(chosen)
            ordered.append(chosen)
            seen.update((chosen["strata"][field] for field in
                         ("atom_name_hint", "difficulty", "provenance", "atom_shape")))
    if not 0 < dev_count < len(ordered):
        raise ValueError("invalid development target")
    fields = ("app", "difficulty", "atom_shape", "atom_name_hint", "provenance")
    indices = {app: [i for i, f in enumerate(ordered) if f["strata"]["app"] == app]
               for app in totals}
    quota = {app: min(len(group) - 1, max(1, dev_count * len(group) // len(ordered)))
             for app, group in indices.items()}
    if sum(quota.values()) > dev_count:
        raise ValueError("development target too small for app coverage")
    while sum(quota.values()) < dev_count:
        app = max((a for a in indices if quota[a] < len(indices[a]) - 1),
                  key=lambda a: (dev_count * len(indices[a]) / len(ordered) - quota[a], a))
        quota[app] += 1
    counts = Counter((field, f["strata"][field]) for f in ordered for field in fields)
    eligible = {app: [i for i in group if all(counts[field, ordered[i]["strata"][field]] > 1
                                           for field in fields)] for app, group in indices.items()}
    if any(len(eligible[app]) < quota[app] for app in quota):
        raise ValueError("cannot preserve rare final stratum")
    dev = {group[(2 * j + 1) * len(group) // (2 * quota[app])]
           for app, group in eligible.items() for j in range(quota[app])}
    for i, family in enumerate(ordered):
        family["qualification_rank"] = i + 1
        family["reserved_role"] = "dev" if i in dev else "final_test"
    strata = {field: dict(Counter(f["strata"][field] for f in ordered)) for field in fields}
    role_strata = {role: {field: dict(Counter(f["strata"][field] for f in ordered
                                     if f["reserved_role"] == role)) for field in fields}
                   for role in ("dev", "final_test")}
    return seal({"schema": "fleet_blackbox_qualification_order_v1",
                 "candidate_versions": sorted(rows, key=lambda r: (r["task_key"], r["task_version_id"])),
                 "ordered_families": ordered, "strata": strata, "role_strata": role_strata,
                 "counts": {"versions": len(rows), "families": len(ordered),
                            "dev_reserved": len(dev), "final_reserved": len(ordered) - len(dev),
                            "final_qualified_floor": 100},
                 "quality_gate": {"runtime": "unproven", "positive_control": "unproven",
                                  "model_outcomes_used": False, "launch_authorized": False}})


def main() -> None:
    frozen = checked(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    repeated = plan(frozen["candidate_versions"])
    for field in ("counts", "ordered_families", "role_strata", "strata", "quality_gate"):
        if frozen[field] != repeated[field]:
            raise ValueError(f"frozen qualification plan drifted: {field}")
    print(json.dumps({"sha256": frozen["sha256"], "counts": frozen["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
