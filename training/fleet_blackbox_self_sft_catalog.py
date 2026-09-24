"""Build the sanitized Fleet blackbox catalog for future Qwen self-SFT.

The catalog is a metadata-only join over exact task versions.  A row without
reviewed transitive atom lineage is never admitted to training.  Live refresh
queries are read-only and retain no prompt, trajectory, score, or session ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from training import task_family_split

SCHEMA = "qwen38_self_sft_blackbox_task_catalog_v1"
OUTPUT = "configs/data/qwen38-self-sft-blackbox-task-catalog-20260924-v1.json"

INVENTORY = "configs/data/fleet-blackbox-current-production-20260924-v1.json"
PROVEN = "configs/data/fleet-blackbox-receipt-proven-20260924-v1.json"
CANDIDATES = "configs/data/fleet-blackbox-qa-candidates-20260924-v1.json"
LIVE_LINEAGE = "configs/data/fleet-blackbox-qa33-live-lineage-20260924-v1.json"
LINEAGE_CENSUS = "configs/data/fleet-blackbox-shared-atom-lineage-census-20260924-v1.json"
SPLIT = "configs/data/fleet-blackbox-lineage-safe-split-20260924-v2.json"
STRICT_RECEIPTS = "configs/data/qwen-blackbox-eligible-v1.json"
CANARY = "configs/qualification/fleet-blackbox-qa33-zero-model-canary-20260924-v1.json"

SOURCE_DIGESTS = {
    INVENTORY: (
        "sha256:3de1271d11769989a4d80b0e329a53744642402a2754544eec1851db7322efe7",
        "sha256:71564b9abfa4d3bd3c47a31e7cc5a4a7acf6406120772478f5e85bc83bde4d30",
    ),
    PROVEN: (
        "sha256:23cfc25af183ff474e914443682e4707162110ddc2aa16103a13d984fe05b83c",
        "sha256:6c211259a29376de195411e5cc182bb6f26c5b8027ab86fec855b44f064b0fd0",
    ),
    CANDIDATES: (
        "sha256:4f61fd78b4d92c933030f026e0137abe199e0157fd72b2f959df0354df489be8",
        "sha256:b3d551863655bc8ad92c76bb1b2b0f87c8bcec812b5c42987fc9c0c28c1a5b2a",
    ),
    LIVE_LINEAGE: (
        "sha256:7622ce3870f74a3a9896575d1e42a122826743b6cf15307f1d411694135bea6b",
        "sha256:8f5a0e8f08a8b193ef785bd18741f46f9355fa24b49694b49e9f7ca07c58373a",
    ),
    LINEAGE_CENSUS: (
        "sha256:70bb11654272dd0adfa9519522df028a9d5a7f1fea42d98b2b47bfdb5271a07c",
        "sha256:976fc1d4f6c8765bd03fecd30280668097998af84ad9806d43e7355459b9845a",
    ),
    SPLIT: (
        "sha256:76e0169c189dd5a0ec2c9858f7172a9d958357cef88c86a8c596b8c8da5663a1",
        "sha256:24b7e98a98b97aca8cc2c63eccbeb28a2963a7cd459078b2ddcb3f386fce1323",
    ),
    CANARY: (
        "sha256:cba6aa26f51576ca1417bb4d15556f409651a0f98572efadec67bcbdb0d1faab",
        "sha256:d28ffb863452325120b70bc5c9921794a68234a09a37a3f7c602c25d997c37d2",
    ),
}

STRICT_RECEIPTS_FILE_SHA256 = (
    "sha256:b12a755c7e65bc04694eab85d9b0c21fadc222207d7bcc6ed8f0c788e032589b"
)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    key = row.get("task_key")
    version = row.get("task_version_id")
    if not isinstance(key, str) or not key or not isinstance(version, str) or not version:
        raise ValueError("task identity must bind an exact key and version")
    return key, version


def _index(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {_identity(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{label} contains duplicate exact task versions")
    return indexed


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for relative, (logical_digest, file_digest) in SOURCE_DIGESTS.items():
        path = root / relative
        value = json.loads(path.read_text())
        unsigned = {key: item for key, item in value.items() if key != "sha256"}
        if (
            value.get("sha256") != logical_digest
            or task_family_split.canonical_digest(unsigned) != logical_digest
            or _file_digest(path) != file_digest
        ):
            raise ValueError(f"{relative} drifted from its exact reviewed bytes")
        values[relative] = value
    strict_path = root / STRICT_RECEIPTS
    if _file_digest(strict_path) != STRICT_RECEIPTS_FILE_SHA256:
        raise ValueError(f"{STRICT_RECEIPTS} drifted from its exact reviewed bytes")
    values[STRICT_RECEIPTS] = json.loads(strict_path.read_text())
    return values


def _validate_live_recheck(live: dict[str, Any], values: dict[str, dict[str, Any]]) -> None:
    inventory = values[INVENTORY]
    lineage = values[LIVE_LINEAGE]
    expected_qa = Counter(row["qa_status"] for row in inventory["tasks"])
    catalog = live.get("catalog")
    sessions = live.get("session_store")
    exact_lineage = live.get("exact_source_lineage")
    safety = live.get("safety")
    if (
        live.get("authority_team_id") != "a1025f0b-ad67-49fc-a023-51800ab43e84"
        or not isinstance(live.get("observed_from"), str)
        or not isinstance(live.get("observed_through"), str)
        or not isinstance(catalog, dict)
        or catalog.get("exact_inventory_sha256") != inventory["sha256"]
        or catalog.get("exact_inventory_match") is not True
        or catalog.get("blackbox_task_versions") != len(inventory["tasks"])
        or catalog.get("verifier_attached_blackbox_task_versions") != len(inventory["tasks"])
        or catalog.get("qa_status") != dict(sorted(expected_qa.items()))
        or not isinstance(catalog.get("status_rows_sha256"), str)
        or not isinstance(sessions, dict)
        or sessions.get("exact_task_versions_checked") != 75
        or sessions.get("outcomes") != {"proven_model_failure": 33, "proven_success": 42}
        or not isinstance(sessions.get("receipt_binding_rows_sha256"), str)
        or not isinstance(exact_lineage, dict)
        or exact_lineage.get("exact_task_versions_checked") != 33
        or exact_lineage.get("exact_match") is not True
        or exact_lineage.get("lineage_rows_sha256")
        != task_family_split.canonical_digest(lineage["task_versions"])
        or safety
        != {
            "artifact_aliases_used": False,
            "external_mutations": 0,
            "model_calls": 0,
            "task_content_persisted": False,
            "session_content_persisted": False,
        }
    ):
        raise ValueError("live authority recheck is incomplete or inconsistent")


def build(root: Path, live_recheck: dict[str, Any]) -> dict[str, Any]:
    """Join the frozen exact authorities into one mutually exclusive catalog."""
    values = _load_sources(root)
    _validate_live_recheck(live_recheck, values)

    inventory_rows = values[INVENTORY]["tasks"]
    inventory = _index(inventory_rows, "current inventory")
    proven = _index(values[PROVEN]["task_versions"], "receipt-proven inventory")
    candidates = _index(values[CANDIDATES]["tasks"], "QA candidates")
    live_lineage = _index(values[LIVE_LINEAGE]["task_versions"], "live lineage")
    split = _index(values[SPLIT]["tasks"], "lineage-safe split")
    if set(proven) != set(split) or set(candidates) != set(live_lineage):
        raise ValueError("receipt, split, candidate, and lineage identities do not join exactly")
    if set(proven) & set(candidates):
        raise ValueError("receipt-proven and QA-candidate identities overlap")

    census = values[LINEAGE_CENSUS]
    census_candidates = _index(census["candidate_task_versions"], "lineage census candidates")
    if set(census_candidates) != set(candidates):
        raise ValueError("lineage census does not exactly cover the QA candidates")
    component_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for component in census["components"]:
        for row in component["task_versions"]:
            identity = _identity(row)
            if identity in component_by_identity:
                raise ValueError("lineage census repeats a task version")
            component_by_identity[identity] = component
    if set(component_by_identity) != set(proven) | set(candidates):
        raise ValueError("transitive lineage census has an unexpected identity set")

    rows = []
    for identity, row in inventory.items():
        projected: dict[str, Any] = {
            "task_id": row["task_id"],
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "qa_status": row["qa_status"],
        }
        if identity in split:
            assignment = split[identity]
            role = assignment["split"]
            classification = (
                "immediately_train_eligible" if role == "train" else "heldout_family_overlap"
            )
            component = component_by_identity[identity]
            projected.update(
                {
                    "classification": classification,
                    "transitive_family": {
                        "status": "resolved",
                        "component_id": assignment["shared_atom_component_id"],
                        "group_id": assignment["group_id"],
                        "role": role,
                        "atom_artifact_keys": component["atom_artifact_keys"],
                    },
                    "admission_evidence": "complete_exact_receipts_and_frozen_role",
                }
            )
        elif row["qa_status"] == "broken_task":
            projected.update(
                {
                    "classification": "known_broken",
                    "transitive_family": {
                        "status": "unresolved_blocked",
                        "role": None,
                    },
                    "admission_evidence": "cross_session_QA_broken_task_exclusion",
                }
            )
        elif identity in candidates:
            candidate = census_candidates[identity]
            live = live_lineage[identity]
            projected.update(
                {
                    "classification": "qa_needed",
                    "transitive_family": {
                        "status": "resolved_unqualified",
                        "component_id": candidate["component_id"],
                        "role": None,
                        "atom_artifact_keys": live["atom_artifact_keys"],
                    },
                    "qualification": {
                        "runtime_receipt_qualified": False,
                        "teacher3k_exact_version_exposed": candidate[
                            "teacher3k_exact_version_exposed"
                        ],
                        "teacher3k_shared_atom_exposed": candidate["teacher3k_shared_atom_exposed"],
                    },
                    "admission_evidence": "lineage_bound_missing_complete_runtime_receipts",
                }
            )
        elif row["qa_status"] == "not_analyzed":
            projected.update(
                {
                    "classification": "unanalyzed",
                    "transitive_family": {
                        "status": "unresolved_blocked",
                        "role": None,
                    },
                    "admission_evidence": "missing_exact_receipt_and_reviewed_lineage",
                }
            )
        else:
            raise ValueError("current inventory row has no conservative classification")
        rows.append(projected)
    rows.sort(key=lambda row: (row["task_key"], row["task_version_id"]))

    counts = Counter(row["classification"] for row in rows)
    expected_counts = {
        "heldout_family_overlap": 25,
        "immediately_train_eligible": 50,
        "known_broken": 74,
        "qa_needed": 33,
        "unanalyzed": 1035,
    }
    if dict(sorted(counts.items())) != expected_counts or len(rows) != 1217:
        raise ValueError("self-SFT classification does not partition the current catalog")

    train_components = {
        row["transitive_family"]["component_id"]
        for row in rows
        if row["classification"] == "immediately_train_eligible"
    }
    heldout_components = {
        row["transitive_family"]["component_id"]
        for row in rows
        if row["classification"] == "heldout_family_overlap"
    }
    qa_components = {
        row["transitive_family"]["component_id"]
        for row in rows
        if row["classification"] == "qa_needed"
    }
    if (
        train_components & heldout_components
        or qa_components & (train_components | heldout_components)
        or len(train_components) != 50
        or len(heldout_components) != 25
        or len(qa_components) != 26
    ):
        raise ValueError("transitive task families are not leakage-safe")

    canary = values[CANARY]
    canary_identity = (
        canary["candidate"]["task_key"],
        canary["candidate"]["task_version_id"],
    )
    if (
        canary_identity not in candidates
        or census_candidates[canary_identity]["component_id"]
        != canary["candidate"]["shared_atom_component_id"]
        or canary["no_launch_receipt"]["launch_authorized"] is not False
    ):
        raise ValueError("next qualification canary is not bound to the candidate census")

    source_receipts = {
        path: {"logical_sha256": digests[0], "file_sha256": digests[1]}
        for path, digests in SOURCE_DIGESTS.items()
    }
    source_receipts[STRICT_RECEIPTS] = {
        "file_sha256": STRICT_RECEIPTS_FILE_SHA256,
        "logical_sha256": values[STRICT_RECEIPTS].get("sha256"),
    }
    body = {
        "schema": SCHEMA,
        "purpose": (
            "sanitized exact-version eligibility inventory for future Qwen pass@8 "
            "blackbox self-SFT collection"
        ),
        "authority_recheck": live_recheck,
        "sources": source_receipts,
        "classification_order": [
            "complete exact receipt plus frozen transitive-family role",
            "known broken exclusion",
            "lineage-bound QA candidate missing complete runtime receipts",
            "unanalysed and missing exact receipt plus reviewed lineage",
        ],
        "counts": {
            "current_blackbox_task_versions": len(rows),
            **expected_counts,
            "resolved_transitive_components": {
                "train": len(train_components),
                "heldout": len(heldout_components),
                "qa_needed": len(qa_components),
            },
        },
        "maximum_safe_train_only": {
            "exact_task_versions": 50,
            "transitive_components": 50,
            "condition": (
                "future collection trains only on the exact frozen train components and "
                "performs a just-in-time exact task/runtime freshness check before rollout"
            ),
            "launch_ready_now": False,
            "reason_launch_not_ready": (
                "this static catalog does not replace a fresh runtime binding and availability "
                "check, and it authorizes no rollout"
            ),
        },
        "next_qualification_wave": {
            "status": canary["no_launch_receipt"]["status"],
            "model_calls": 0,
            "task_key": canary_identity[0],
            "task_version_id": canary_identity[1],
            "component_id": canary["candidate"]["shared_atom_component_id"],
            "atom_source_locators": canary["candidate"]["atom_source_locators"],
            "task_graph_source_locator": canary["candidate"]["task_graph_source_locator"],
            "remaining_candidate_components_after_canary": 25,
            "remaining_candidate_task_versions_after_canary": 32,
            "followup_rule": (
                "after a clean canary and separate authorization, preregister one exact "
                "representative per remaining transitive component; never count aliases as "
                "independent families"
            ),
            "launch_authorized": False,
        },
        "task_versions": rows,
        "evidence_boundary": {
            "catalog_level_train_eligibility_is_not_session_admission": True,
            "future_pass8_sessions_require_authoritative_success_and_corpus_admission": True,
            "unresolved_lineage_can_never_enter_training": True,
            "heldout_components_can_never_enter_training": True,
        },
        "privacy": {
            "private_content_persisted": False,
            "session_ids_persisted": False,
            "numeric_scores_persisted": False,
            "task_prompts_or_trajectories_persisted": False,
        },
        "safety": {
            "external_mutations": 0,
            "rollouts_launched": 0,
            "registry_writes": 0,
            "kubernetes_objects_created": 0,
        },
    }
    body["sha256"] = task_family_split.canonical_digest(body)
    return body


def live_recheck(root: Path, *, workers: int = 24) -> dict[str, Any]:
    """Reconcile the three read-only authorities without retaining private data."""
    from evals.fleet import opencode_self_hosted as self_hosted
    from evals.fleet import task_quality_qualification as qualification
    from evals.fleet.qa_candidate_lineage_census import project_lineage
    from scripts import refresh_fleet_task_inventory as refresh

    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise ValueError("FLEET_API_KEY is required for a live read-only recheck")
    values = _load_sources(root)
    observed_from = refresh.utc_now()

    client = refresh.ReadClient(api_key)
    account = client.get(refresh.PUBLIC_API, "/v1/account")
    if (account.get("team_id"), account.get("team_name")) != (
        refresh.FLEET_TEAM_ID,
        "fleet",
    ):
        raise ValueError("FLEET_API_KEY does not resolve to the Fleet team")
    membership_path = f"/v1/projects/{refresh.OTS_CYBER_PROJECT_ID}/tasks"
    task_ids = client.get(refresh.PRIVATE_API, membership_path).get("taskIds")
    if not isinstance(task_ids, list) or len(task_ids) != len(set(task_ids)):
        raise ValueError("OTS Cyber membership is malformed")
    statuses = refresh.parallel_map(
        lambda task_id: refresh.catalog_status(client, task_id), task_ids, workers
    )
    if task_ids != client.get(refresh.PRIVATE_API, membership_path).get("taskIds"):
        raise ValueError("OTS Cyber membership changed during the recheck")
    statuses.sort(key=lambda row: (row["task_id"], row["current_version_id"]))
    quality = refresh.task_quality(client)
    production = {
        row["task_id"]: row for row in statuses if row["lifecycle_status"] == "production"
    }
    if set(production) != {row["task_id"] for row in quality} or any(
        production[row["task_id"]]["current_version_id"] != row["task_version_id"]
        for row in quality
    ):
        raise ValueError("task-quality and exact production bindings disagree")
    blackbox = [row for row in quality if row["task_shape"] == "blackbox"]
    blackbox_statuses = [production[row["task_id"]] for row in blackbox]
    if any(row["verifier_attached"] is not True for row in blackbox_statuses):
        raise ValueError("a current production blackbox task has no attached verifier")
    current_selection = refresh.selection(
        schema="fleet_current_production_blackbox_inventory_v1",
        purpose="all current production task keys explicitly identified as blackbox",
        tasks=blackbox,
        evidence=[
            "current OTS Cyber project membership",
            "current production lifecycle status",
            "current exact version and attached verifier",
            "cross-session QA bucket; not a validity certificate",
        ],
    )
    if current_selection != values[INVENTORY]:
        raise ValueError("live exact blackbox inventory differs from the reviewed snapshot")

    strict = refresh.load_rows_by_version(root / STRICT_RECEIPTS, "task_versions")
    proven = values[PROVEN]["task_versions"]

    def check_session(row: dict[str, Any]) -> dict[str, Any]:
        evidence = strict[row["task_version_id"]]["execution_evidence"]
        outcome = refresh.canonical_execution_outcome(
            client,
            task_key=row["task_key"],
            session_id=evidence["session_id"],
            verifier_execution_id=evidence["verifier_execution_id"],
        )
        return {
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "outcome": outcome,
            "receipt_sha256": evidence["receipt_sha256"],
            "cleanup_sha256": evidence["cleanup_sha256"],
            "ingest_sha256": evidence["ingest_sha256"],
        }

    session_rows = sorted(
        refresh.parallel_map(check_session, proven, min(workers, 8)),
        key=lambda row: (row["task_key"], row["task_version_id"]),
    )

    candidate_rows = values[CANDIDATES]["tasks"]
    exact_client = qualification._client(api_key)  # noqa: SLF001
    try:
        qualification._account(exact_client)  # noqa: SLF001
        projected = []
        for selected in candidate_rows:
            task = self_hosted._request(  # noqa: SLF001
                exact_client,
                "GET",
                f"/v1/tasks/{selected['task_key']}",
                params={"version_id": selected["task_version_id"]},
            )
            projected.append(project_lineage(task, selected))
    finally:
        exact_client.close()
    projected.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    if projected != values[LIVE_LINEAGE]["task_versions"]:
        raise ValueError("exact task-version source lineage changed")

    recheck = {
        "observed_from": observed_from,
        "observed_through": refresh.utc_now(),
        "repository_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "authority_team_id": account["team_id"],
        "catalog": {
            "project_id": refresh.OTS_CYBER_PROJECT_ID,
            "project_task_ids": len(task_ids),
            "production_task_versions": len(quality),
            "blackbox_task_versions": len(blackbox),
            "verifier_attached_blackbox_task_versions": len(blackbox_statuses),
            "qa_status": dict(sorted(Counter(row["qa_status"] for row in blackbox).items())),
            "exact_inventory_sha256": current_selection["sha256"],
            "exact_inventory_match": True,
            "status_rows_sha256": refresh.digest(statuses),
        },
        "session_store": {
            "exact_task_versions_checked": len(session_rows),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in session_rows).items())),
            "receipt_binding_rows_sha256": refresh.digest(session_rows),
            "session_ids_persisted": False,
            "numeric_scores_persisted": False,
        },
        "exact_source_lineage": {
            "authority": "read-only exact Fleet task-version GET",
            "exact_task_versions_checked": len(projected),
            "lineage_rows_sha256": task_family_split.canonical_digest(projected),
            "exact_match": True,
            "artifact_aliases_used": False,
        },
        "safety": {
            "artifact_aliases_used": False,
            "external_mutations": 0,
            "model_calls": 0,
            "task_content_persisted": False,
            "session_content_persisted": False,
        },
    }
    _validate_live_recheck(recheck, values)
    return recheck


def _encoded(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_once_or_equal(path: Path, value: dict[str, Any]) -> None:
    encoded = _encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_text() != encoded:
            raise ValueError(f"refusing to replace immutable catalog {path}") from None


def check(root: Path) -> None:
    path = root / OUTPUT
    actual = json.loads(path.read_text())
    expected = build(root, actual["authority_recheck"])
    if actual != expected:
        raise ValueError(f"{OUTPUT} reproduction drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--refresh-live", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise SystemExit("--workers must be in 1..32")
    if args.refresh_live:
        evidence = live_recheck(args.root, workers=args.workers)
        _write_once_or_equal(args.root / OUTPUT, build(args.root, evidence))
    else:
        check(args.root)


if __name__ == "__main__":
    main()
