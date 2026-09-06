"""Build held, model-blind supersession receipts for infrastructure-invalid cells.

The receipts produced here never authorize a launch.  They preserve the original
execution and any session it created, mint a fresh execution identity for the
same statistical cell, and describe the separate live gates a later release
must satisfy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_blocked_disposition_v1 as disposition
from evals.fleet import self_hosted

CAMPAIGN_PATH = Path(
    "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
)
SELECTION_PATH = Path("evals/fleet/configs/opencode-easiest-train100-selection-v2.json")
DISPOSITION_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-06-exact-pass4-blocked-cell-disposition-v2.json"
)
HELD_SCHEMA = "fleet-infrastructure-invalid-generation-supersession-held-v1"
MANIFEST_SCHEMA = "fleet-infrastructure-invalid-generation-supersession-manifest-v1"
STATUS = "HELD_NO_LAUNCH"

def _sha(value: str) -> str:
    return f"sha256:{value}"


RECEIPT_PATHS = {
    ("zai-org/GLM-5.3", 1, 4): (
        Path("docs/evidence/glm53-study")
        / "2026-09-06-glm53-r001-a4-infrastructure-supersession-held-v1.json"
    ),
    ("Qwen/Qwen3.8-27B", 3, 2): (
        Path("docs/evidence/qwen38-study")
        / "2026-09-06-qwen38-r003-a2-infrastructure-supersession-held-v1.json"
    ),
    ("Qwen/Qwen3.8-27B", 4, 2): (
        Path("docs/evidence/qwen38-study")
        / "2026-09-06-qwen38-r004-a2-infrastructure-supersession-held-v1.json"
    ),
    ("Qwen/Qwen3.8-27B", 97, 1): (
        Path("docs/evidence/qwen38-study")
        / "2026-09-06-qwen38-r097-a1-infrastructure-supersession-held-v1.json"
    ),
}
PRIOR_EXECUTION_AUTHORITY = {
    _sha("f3921b5927bcf73a0df58ff991841db52f9b23f04cae5d482c5f7e8630308250"): {
        "execution_generation": 1,
        "claim_receipt_sha256": _sha(
            "91721355308d2c552dc3c08686582e4bdb3641fe99e48b1d32bcb60311c9323d"
        ),
        "claim_file_sha256": _sha(
            "13b0911f3f1ec3e46b353a54611fb50600b3cb23c4a57e56daec03bcffaa4e39"
        ),
    },
    _sha("6edac950ae44ff62c07074a775afdb65cc9b394047883e0b57fcad84e74d2fa7"): {
        "execution_generation": 22,
        "claim_receipt_sha256": _sha(
            "23df72dccca4736949fe3711bd3d619c5517780031f7506f32509ebc6b4371fa"
        ),
        "claim_file_sha256": _sha(
            "36961bfb58715de4f6d6e37a880194c9517069058357b200ab84f9fbd06cd956"
        ),
    },
    _sha("f5dced06656b0bf069930628978ce31b6ff682f437c86d55c6b241b2f83f4f48"): {
        "execution_generation": 22,
        "claim_receipt_sha256": _sha(
            "18219263f3d95ea63465dafed3d1eb7dbeb9a3669dd8925ae93cef6ef1aee3b9"
        ),
        "claim_file_sha256": _sha(
            "8de97e7d3383fd5ac770f3ec6c695975842921e22be19839e9a0340d54421b0c"
        ),
    },
    _sha("59df171f31662465c8a9a98844b254a3fcf6bb242e53347501248a8e499bb068"): {
        "execution_generation": 19,
        "claim_receipt_sha256": _sha(
            "22c93654e7ed24cf69fecf3e5c775dd3901a98c2b9dccded751773858c990f46"
        ),
        "claim_file_sha256": _sha(
            "b03396839f0191f82cd76217a49aedfd543c2fcc9a2a950b82c6e30163031819"
        ),
    },
}


class SupersessionError(ValueError):
    """The held supersession authority is incomplete or has drifted."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SupersessionError(f"expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source(root: Path, relative: Path) -> dict[str, str]:
    path = root / relative
    value = _read(path)
    result = {"path": str(relative), "file_sha256": _file_sha256(path)}
    if "receipt_sha256" in value:
        if value["receipt_sha256"] != self_hosted.digest_without(
            value, "receipt_sha256"
        ):
            raise SupersessionError(f"source receipt self digest drifted: {relative}")
        result["receipt_sha256"] = value["receipt_sha256"]
    return result


def _selection_by_rank(root: Path) -> dict[int, dict[str, Any]]:
    selection = _read(root / SELECTION_PATH)
    rows = selection.get("tasks")
    if not isinstance(rows, list) or len(rows) != 100:
        raise SupersessionError("exact selection must contain 100 tasks")
    result = {row.get("rank"): row for row in rows}
    if set(result) != set(range(1, 101)):
        raise SupersessionError("exact selection ranks drifted")
    return result


def classify_infrastructure_invalid(
    row: dict[str, Any], *, proof: dict[str, bool]
) -> str:
    """Classify only authority state; model identity and outcome content are ignored."""
    session = row.get("session_state", {})
    exact_matches = session.get("exact_matches", session.get("exact_run_id_matches"))
    if (
        proof == {"authoritative_outcome_absence_proven": True}
        and exact_matches == 0
        and row.get("acceptance_state") == "absent"
        and row.get("verifier_state") == "absent"
        and row.get("cleanup_state")
        in {"instance_closed_and_containers_removed", "complete"}
    ):
        return "no_authoritative_outcome"
    if (
        proof == {"bound_endpoint_identity_discontinuity_proven": True}
        and exact_matches == 1
        and session.get("session_ingest_present") is True
        and row.get("acceptance_state") == "local_receipt_present_but_not_credited"
        and row.get("verifier_state", "").startswith("present")
        and row.get("cleanup_state") == "complete"
    ):
        return "bound_endpoint_identity_discontinuity"
    raise SupersessionError("blocked cell does not match a reviewed infrastructure class")


def _exact_infrastructure_proof(root: Path, row: dict[str, Any]) -> dict[str, bool]:
    evidence = row.get("evidence", [])
    if not evidence:
        raise SupersessionError("blocked cell lacks immutable evidence")
    authority = _read(root / evidence[0]["path"])
    schema = authority.get("schema_version")
    if schema == "fleet-hosted-controller-abort-classification-v1":
        if (
            authority.get("status") != "INFRASTRUCTURE_BLOCKED_UNRESOLVED"
            or authority.get("retry_authorized") is not False
            or authority.get("scoring_interpretation") != "NOT_A_SCORE"
        ):
            raise SupersessionError("hosted abort authority drifted")
        return {"authoritative_outcome_absence_proven": True}
    if schema == "fleet-qwen38-rank3-a2-terminal-reconciliation-v1":
        reconciliation = authority.get("fresh_authoritative_reconciliation", {})
        if (
            authority.get("status") != "NO_AUTHORITATIVE_SESSION_OR_VERIFIER"
            or reconciliation.get("matching_session_count") != 0
            or reconciliation.get("verifier_execution_presence") is not False
            or authority.get("retry_allowed") is not False
        ):
            raise SupersessionError("post-model absence authority drifted")
        return {"authoritative_outcome_absence_proven": True}
    if schema == "fleet-qwen38-dedicated-dp8-preemption-incident-v1":
        if (
            authority.get("preemption", {}).get("exact_endpoint_identity_rotated")
            is not True
            or authority.get("immutable_timing", {}).get(
                "all_model_interaction_proven_complete_before_preemption"
            )
            is not False
            or authority.get("disposition", {}).get("credited") is not False
        ):
            raise SupersessionError("DP8 endpoint-discontinuity authority drifted")
        return {"bound_endpoint_identity_discontinuity_proven": True}
    if schema == "fleet-qwen38-dedicated-rank97-v2-interrupted-terminal-v1":
        if (
            authority.get("server_release", {}).get("http_status") != 204
            or authority.get("server_release", {}).get("exact_get_after_release") != 404
            or authority.get("timing_boundary", {}).get(
                "stream_finalization_before_release_proven"
            )
            is not False
            or authority.get("attempts", {}).get("1", {}).get("credited") is not False
        ):
            raise SupersessionError("rank97 endpoint-discontinuity authority drifted")
        return {"bound_endpoint_identity_discontinuity_proven": True}
    raise SupersessionError("unsupported infrastructure-invalid authority")


def _successor_execution_id(
    *,
    campaign_id: str,
    model: str,
    task_key: str,
    task_version_id: str,
    selection_rank: int,
    attempt: int,
    cell_id: str,
    prior_execution_id: str,
    prior_generation: int,
    successor_generation: int,
) -> str:
    payload = {
        "schema_version": HELD_SCHEMA,
        "campaign_id": campaign_id,
        "model": model,
        "task_key": task_key,
        "task_version_id": task_version_id,
        "selection_rank": selection_rank,
        "attempt": attempt,
        "cell_id": cell_id,
        "prior_execution_id": prior_execution_id,
        "prior_execution_generation": prior_generation,
        "successor_execution_generation": successor_generation,
    }
    return _digest(payload)


def build_held_receipts(root: Path, *, observed_at_utc: str) -> list[dict[str, Any]]:
    campaign = _read(root / CAMPAIGN_PATH)
    if campaign.get("campaign_id") != "chris-cyber-q38-glm53-exact-easiest100-pass4-v1":
        raise SupersessionError("campaign identity drifted")
    source_disposition = disposition.validate(root / DISPOSITION_PATH, root=root)
    tasks = _selection_by_rank(root)
    blocked = [
        row
        for row in source_disposition["cells"]
        if row["disposition"] == "irrecoverable_under_current_reviewed_rules"
    ]
    if len(blocked) != 4:
        raise SupersessionError("expected exactly four currently fenced cells")

    result: list[dict[str, Any]] = []
    for row in blocked:
        key = (row["model"], row["selection_rank"], row["attempt"])
        if key not in RECEIPT_PATHS:
            raise SupersessionError("unexpected blocked-cell identity")
        task = tasks[row["selection_rank"]]
        prior_authority = PRIOR_EXECUTION_AUTHORITY.get(row["cell_id"])
        if prior_authority is None:
            raise SupersessionError("prior execution authority is not bound")
        prior_generation = prior_authority["execution_generation"]
        if row.get("execution_generation", prior_generation) != prior_generation:
            raise SupersessionError("prior execution generation drifted")
        successor_generation = prior_generation + 1
        successor_id = _successor_execution_id(
            campaign_id=campaign["campaign_id"],
            model=row["model"],
            task_key=task["task_key"],
            task_version_id=task["task_version_id"],
            selection_rank=row["selection_rank"],
            attempt=row["attempt"],
            cell_id=row["cell_id"],
            prior_execution_id=row["execution_id"],
            prior_generation=prior_generation,
            successor_generation=successor_generation,
        )
        infrastructure_proof = _exact_infrastructure_proof(root, row)
        invalid_class = classify_infrastructure_invalid(
            row, proof=infrastructure_proof
        )
        matching_sessions = row["session_state"].get(
            "exact_matches", row["session_state"].get("exact_run_id_matches")
        )
        if matching_sessions not in {0, 1}:
            raise SupersessionError("authoritative matching-session count drifted")
        value: dict[str, Any] = {
            "schema_version": HELD_SCHEMA,
            "status": STATUS,
            "observed_at_utc": observed_at_utc,
            "campaign_id": campaign["campaign_id"],
            "model": row["model"],
            "task": {
                "selection_rank": row["selection_rank"],
                "attempt": row["attempt"],
                "key": task["task_key"],
                "version_id": task["task_version_id"],
            },
            "statistical_cell": {
                "cell_id": row["cell_id"],
                "one_cell_per_model_task_attempt": True,
                "maximum_accepted_valid_generations": 1,
            },
            "prior_execution": {
                "execution_id": row["execution_id"],
                "execution_generation": prior_generation,
                "claim_and_incident_preserved": True,
                "matching_authoritative_sessions": matching_sessions,
                "session_preservation": (
                    "none_present"
                    if matching_sessions == 0
                    else "retained_excluded_as_protocol_invalid"
                ),
                "acceptance_state": row["acceptance_state"],
                "verifier_state": row["verifier_state"],
                "cleanup_state": row["cleanup_state"],
                "process": row["process"],
                "claim": {
                    "path": (
                        "/mnt/sfs/cell-execution-claims/"
                        f"opencode11827-autocontinue-v1/{row['execution_id'][7:]}.json"
                    ),
                    "receipt_sha256": prior_authority["claim_receipt_sha256"],
                    "file_sha256": prior_authority["claim_file_sha256"],
                },
            },
            "infrastructure_invalid_class": invalid_class,
            "scientific_basis": {
                "classification_uses_scores": False,
                "classification_uses_prompts_traces_or_flags": False,
                "classification_uses_model_output": False,
                "classification_is_model_blind": True,
                "prior_generation_remains_excluded_from_valid_outcomes": True,
                "infrastructure_proof": infrastructure_proof,
            },
            "successor_execution": {
                "execution_id": successor_id,
                "execution_generation": successor_generation,
                "claim_namespace": "opencode11827-autocontinue-v1",
                "same_model_task_version_attempt_and_cell_required": True,
                "fresh_claim_required": True,
            },
            "future_release_requirements": {
                "fresh_get_only_session_and_verifier_reconciliation": True,
                "fresh_accepted_active_and_claim_collision_scan": True,
                "exact_successor_claim_and_output_roots_absent": True,
                "exact_harness_model_context_tools_and_compaction_binding": True,
                "serving_block_identity_and_non_scored_parity_required": True,
                "launch_time_capacity_and_lease_gate_required": True,
                "separate_self_digesting_release_required": True,
            },
            "authorization": {
                "launch_authorized": False,
                "score_credit_authorized": False,
                "claim_creation_authorized": False,
                "workload_mutation_authorized": False,
            },
            "sources": {
                "campaign": _source(root, CAMPAIGN_PATH),
                "selection": _source(root, SELECTION_PATH),
                "blocked_disposition": _source(root, DISPOSITION_PATH),
                "cell_evidence": row["evidence"],
            },
            "privacy": {
                "prompts_read": False,
                "traces_read": False,
                "flags_read": False,
                "scores_read": False,
                "credentials_included": False,
            },
        }
        value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
        result.append(value)
    return result


def validate_held_receipt(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != HELD_SCHEMA or value.get("status") != STATUS:
        raise SupersessionError("held supersession schema or status drifted")
    if value.get("receipt_sha256") != self_hosted.digest_without(
        value, "receipt_sha256"
    ):
        raise SupersessionError("held supersession self digest drifted")
    if value.get("authorization") != {
        "launch_authorized": False,
        "score_credit_authorized": False,
        "claim_creation_authorized": False,
        "workload_mutation_authorized": False,
    }:
        raise SupersessionError("held receipt gained launch authority")
    if any(value.get("scientific_basis", {}).get(field) is not False for field in (
        "classification_uses_scores",
        "classification_uses_prompts_traces_or_flags",
        "classification_uses_model_output",
    )):
        raise SupersessionError("held classification consumed protected outcome data")
    if value.get("scientific_basis", {}).get("classification_is_model_blind") is not True:
        raise SupersessionError("held classification is not model blind")

    expected = {
        item["statistical_cell"]["cell_id"]: item
        for item in build_held_receipts(root, observed_at_utc=value["observed_at_utc"])
    }
    cell_id = value.get("statistical_cell", {}).get("cell_id")
    if cell_id not in expected or value != expected[cell_id]:
        raise SupersessionError("held supersession content drifted from source authority")


def validate_launch_time_clearance(
    held: dict[str, Any], observation: dict[str, Any]
) -> None:
    """Validate a fresh score-blind duplicate scan; this still grants no launch."""
    required = {
        "cell_id": held["statistical_cell"]["cell_id"],
        "successor_execution_id": held["successor_execution"]["execution_id"],
        "prior_claim_receipt_sha256": held["prior_execution"]["claim"][
            "receipt_sha256"
        ],
        "prior_matching_authoritative_sessions": held["prior_execution"][
            "matching_authoritative_sessions"
        ],
        "successor_matching_authoritative_sessions": 0,
        "valid_accepted_generation_count": 0,
        "active_valid_generation_count": 0,
        "successor_claim_count": 0,
        "successor_output_root_count": 0,
        "api_mutations": 0,
        "scores_or_protected_content_read": False,
    }
    if observation != required:
        raise SupersessionError("launch-time clearance is incomplete or drifted")


def validate_generation_precedence(
    held: dict[str, Any], generations: list[dict[str, Any]]
) -> None:
    """Allow at most one valid acceptance and only from the held successor."""
    prior_id = held["prior_execution"]["execution_id"]
    successor_id = held["successor_execution"]["execution_id"]
    by_id = {row.get("execution_id"): row for row in generations}
    if len(by_id) != len(generations) or prior_id not in by_id:
        raise SupersessionError("generation precedence lacks unique prior authority")
    if by_id[prior_id] != {
        "execution_id": prior_id,
        "execution_generation": held["prior_execution"]["execution_generation"],
        "state": "infrastructure_invalid_preserved",
        "valid_accepted": False,
    }:
        raise SupersessionError("prior infrastructure-invalid generation was rewritten")
    accepted = [row for row in generations if row.get("valid_accepted") is True]
    if len(accepted) > 1:
        raise SupersessionError("more than one valid generation was accepted")
    if accepted and accepted[0] != {
        "execution_id": successor_id,
        "execution_generation": held["successor_execution"]["execution_generation"],
        "state": "accepted_valid",
        "valid_accepted": True,
    }:
        raise SupersessionError("acceptance did not come from the authorized successor")


def build_manifest(
    root: Path,
    *,
    observed_at_utc: str,
    receipt_paths: dict[str, Path],
) -> dict[str, Any]:
    held = build_held_receipts(root, observed_at_utc=observed_at_utc)
    rows = []
    for value in held:
        cell_id = value["statistical_cell"]["cell_id"]
        path = receipt_paths[cell_id]
        rows.append(
            {
                "model": value["model"],
                "selection_rank": value["task"]["selection_rank"],
                "attempt": value["task"]["attempt"],
                "cell_id": cell_id,
                "prior_execution_id": value["prior_execution"]["execution_id"],
                "successor_execution_id": value["successor_execution"]["execution_id"],
                "infrastructure_invalid_class": value["infrastructure_invalid_class"],
                "path": str(path),
                "receipt_sha256": value["receipt_sha256"],
                "file_sha256": _file_sha256(root / path),
            }
        )
    value = {
        "schema_version": MANIFEST_SCHEMA,
        "status": STATUS,
        "observed_at_utc": observed_at_utc,
        "campaign_id": held[0]["campaign_id"],
        "counts": {
            "held_cells": 4,
            "no_authoritative_outcome": 2,
            "bound_endpoint_identity_discontinuity": 2,
            "launch_authorized": 0,
        },
        "cells": rows,
        "precedence": {
            "old_claims_sessions_and_incidents_remain_immutable": True,
            "fresh_generation_may_supersede_only_as_valid_execution_authority": True,
            "maximum_accepted_valid_generations_per_statistical_cell": 1,
            "ledger_must_retain_invalid_generations_as_operational_evidence": True,
        },
        "authorization": {
            "launch_authorized": False,
            "score_credit_authorized": False,
            "claim_creation_authorized": False,
            "workload_mutation_authorized": False,
        },
        "privacy": {
            "prompts_read": False,
            "traces_read": False,
            "flags_read": False,
            "scores_read": False,
            "credentials_included": False,
        },
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def validate_manifest(value: dict[str, Any], root: Path) -> None:
    if value.get("schema_version") != MANIFEST_SCHEMA or value.get("status") != STATUS:
        raise SupersessionError("supersession manifest schema or status drifted")
    if value.get("receipt_sha256") != self_hosted.digest_without(
        value, "receipt_sha256"
    ):
        raise SupersessionError("supersession manifest self digest drifted")
    if value.get("counts") != {
        "held_cells": 4,
        "no_authoritative_outcome": 2,
        "bound_endpoint_identity_discontinuity": 2,
        "launch_authorized": 0,
    }:
        raise SupersessionError("supersession manifest counts drifted")
    if value.get("authorization") != {
        "launch_authorized": False,
        "score_credit_authorized": False,
        "claim_creation_authorized": False,
        "workload_mutation_authorized": False,
    }:
        raise SupersessionError("supersession manifest gained launch authority")
    if value.get("precedence") != {
        "old_claims_sessions_and_incidents_remain_immutable": True,
        "fresh_generation_may_supersede_only_as_valid_execution_authority": True,
        "maximum_accepted_valid_generations_per_statistical_cell": 1,
        "ledger_must_retain_invalid_generations_as_operational_evidence": True,
    }:
        raise SupersessionError("supersession precedence drifted")
    cells = value.get("cells", [])
    if (
        not isinstance(cells, list)
        or len(cells) != 4
        or {row.get("cell_id") for row in cells} != set(PRIOR_EXECUTION_AUTHORITY)
    ):
        raise SupersessionError("supersession manifest cell set drifted")
    if value.get("privacy") != {
        "prompts_read": False,
        "traces_read": False,
        "flags_read": False,
        "scores_read": False,
        "credentials_included": False,
    }:
        raise SupersessionError("supersession manifest privacy contract drifted")
    for row in cells:
        path = root / row["path"]
        held = _read(path)
        validate_held_receipt(held, root)
        if (
            row["receipt_sha256"] != held["receipt_sha256"]
            or row["file_sha256"] != _file_sha256(path)
        ):
            raise SupersessionError("manifest-to-held-receipt binding drifted")
