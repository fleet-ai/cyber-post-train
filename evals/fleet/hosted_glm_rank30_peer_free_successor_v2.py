"""Held peer-free rank-30 hosted GLM whole-task successor.

The four statistical cells are unchanged.  Execution, run, Job, ConfigMap,
output, and release identities are fresh.  A score-blind release observer must
prove both hosted endpoint slots are simultaneously free immediately before
the create-once Job is submitted.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import hosted_glm_whole_task_successor_v1 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank30-peer-free-plan-v2"
HELD_SCHEMA = "fleet-hosted-glm-rank30-peer-free-held-v2"
RELEASE_SCHEMA = "fleet-hosted-glm-rank30-peer-free-release-v2"
EXECUTION_GENERATION = 2
CONTROLLER = "glm-hosted-r30-whole-task"
JOB_NAME = "chris-glm53-exact100-hosted-r030-whole-task-g2-v2"
CONFIGMAP_NAME = "chris-glm53-exact100-hosted-r030-whole-task-g2-package-v2"
SFS_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
RELEASE_PATH = Path("/bootstrap/release.json")
RELEASE_MAX_AGE_SECONDS = 600

LEDGER_AUTHORITY = {
    "path": ("docs/evidence/qwen38-study/2026-09-05-exact-pass4-ledger-evidence-snapshot-v48.json"),
    "receipt_sha256": ("sha256:1275d84f1b6aad8d02bc916be55f3a1f3ea99c6817f465905ec0791b0f66cd7a"),
    "file_sha256": ("sha256:332c5ceae480e9c73cf1e391aeb59994049b69b240a8bbc106aefc35a9a01f25"),
}
LIVE_LEDGER_VALIDATION = {
    "path": "docs/evidence/glm53-study/2026-09-06-glm53-ledger-v48-live-validation.json",
    "receipt_sha256": ("sha256:7b77ed00959f6bb219c5c20724ed6958ee85b56d01771f9f158a575cd7301c96"),
    "file_sha256": ("sha256:c3c527a68217190555af394b94b7996901dd46e1e52505c9843d02afe66c7738"),
    "glm_tally": {"accepted": 24, "active": 0, "blocked": 4, "unstarted": 372},
}
DIAGNOSTIC_V2 = {
    "path": "/mnt/sfs/jobs/chris-glm53-r030-preclaim-phase-observer-v2/DIAGNOSTIC.json",
    "receipt_sha256": ("sha256:769a2b5fb758d6f39d73d9a1624eb822283abc7a4465288b01af4f2046f0679c"),
    "file_sha256": ("sha256:6211d4c0727b951252ac435186a71cb957167d6efa04ebf826f8e49ddea5f817"),
    "job_uid": "9a1f90a2-e0f4-41b5-a5ed-77169eec1622",
    "pod_uid": "81a3f651-0d5e-46eb-b216-31c39da082f9",
    "failed_phase": "07-strict-current-peer",
    "failure_code": "CURRENT_PEER_INVARIANT_FAILED",
    "provider_session_model_boundary_crossed": False,
    "api_calls": 0,
    "scores_read": False,
    "prompts_traces_flags_read": False,
}
SUPERSEDED_IDENTITIES = {
    "job_name": "chris-glm53-exact100-hosted-r030-whole-task-g1-v1",
    "configmap_name": "chris-glm53-exact100-hosted-r030-whole-task-g1-package-v1",
    "sfs_root": "/mnt/sfs/jobs/chris-glm53-exact100-hosted-r030-whole-task-g1-v1",
    "retry_authorized": False,
}
SELECTION_AUTHORITY = prior.SELECTION_AUTHORITY
CONTROLLERS = {
    CONTROLLER: {
        "rank": 30,
        "task_version_id": "a0cacaaf-480b-4a4c-9ed7-6b6192bb6783",
        "job_name": JOB_NAME,
        "configmap_name": CONFIGMAP_NAME,
    }
}
SHA256_RE = prior.SHA256_RE
COMMIT_RE = prior.COMMIT_RE
CANARY_GATE_SCHEMA = prior.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = prior.RECONCILIATION_GATE_SCHEMA
validate_inventory_gate = prior.validate_inventory_gate
load = prior.load
AtomicWholeTaskClaims = prior.AtomicWholeTaskClaims


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def _rebind_attempt(row: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(row)
    execution = exact.execution_for(item["cell_id"], EXECUTION_GENERATION)
    run_id = f"chris-glm53-ac-bulk-a-r030-a{item['attempt']}-g2-{execution['execution_id'][7:15]}"
    item.update(
        execution_generation=EXECUTION_GENERATION,
        execution_id=execution["execution_id"],
        run_id=run_id,
        network=run_id.removeprefix("chris-")[:63],
    )
    return item


def _transform(plan: dict[str, Any], *, runtime: bool) -> dict[str, Any]:
    predecessor = prior._transform(plan, CONTROLLER, runtime=runtime)  # noqa: SLF001
    body = copy.deepcopy(predecessor)
    body.pop("plan_sha256")
    body.update(
        schema_version=SCHEMA,
        campaign_id=JOB_NAME,
        source_job_id=JOB_NAME,
        job_name=JOB_NAME,
        configmap_name=CONFIGMAP_NAME,
        sfs_root=SFS_ROOT,
        attempts=[_rebind_attempt(row) for row in predecessor["attempts"]],
        predecessor_plan_sha256=predecessor["plan_sha256"],
        launch_authorized=runtime,
        release_required=True,
        execution_rollforward={
            "statistical_cell_identity_unchanged": True,
            "predecessor_execution_generation": 1,
            "execution_generation": EXECUTION_GENERATION,
            "predecessor_object_and_execution_retry_forbidden": True,
            "fresh_object_and_execution_identities_required": True,
        },
    )
    body["partition"] = {
        "whole_task_rank": 30,
        "attempts": [1, 2, 3, 4],
        "all_cells_previously_unstarted_required": True,
        "current_peer_required": False,
        "both_endpoint_slots_free_required": True,
        "other_ranks_excluded": True,
    }
    body["plan_sha256"] = self_hosted.digest_without(body, "plan_sha256")
    return body


def build_plan(root: Path) -> dict[str, Any]:
    source_plan = prior.source.validate_all(root)[prior.SOURCE_CONTROLLER]
    return _transform(source_plan, runtime=False)


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown peer-free rank30 controller")
    source_plan = prior.source.build_runtime_plan(prior.SOURCE_CONTROLLER, inventory_receipt, root)
    return _transform(source_plan, runtime=True)


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plan = build_plan(root)
    predecessor = prior.build_plan(CONTROLLER, root)
    attempts = plan.get("attempts") or []
    if any(
        (
            plan.get("schema_version") != SCHEMA,
            plan.get("controller") != CONTROLLER,
            plan.get("job_name") != JOB_NAME,
            plan.get("configmap_name") != CONFIGMAP_NAME,
            plan.get("sfs_root") != SFS_ROOT,
            plan.get("launch_authorized") is not False,
            plan.get("release_required") is not True,
            plan.get("model") != predecessor.get("model"),
            plan.get("harness") != predecessor.get("harness"),
            plan.get("execution") != predecessor.get("execution"),
            len(attempts) != 4,
            [row.get("attempt") for row in attempts] != [1, 2, 3, 4],
            plan.get("partition", {}).get("current_peer_required") is not False,
            plan.get("partition", {}).get("both_endpoint_slots_free_required") is not True,
            plan.get("plan_sha256") != self_hosted.digest_without(plan, "plan_sha256"),
        )
    ):
        raise ValueError("peer-free rank30 whole-task plan drifted")
    for item, old in zip(attempts, predecessor["attempts"], strict=True):
        stable = {
            key: value
            for key, value in item.items()
            if key not in {"execution_generation", "execution_id", "run_id", "network"}
        }
        old_stable = {
            key: value
            for key, value in old.items()
            if key not in {"execution_generation", "execution_id", "run_id", "network"}
        }
        expected = exact.execution_for(item["cell_id"], EXECUTION_GENERATION)
        if any(
            (
                stable != old_stable,
                item.get("execution_generation") != EXECUTION_GENERATION,
                item.get("execution_id") != expected["execution_id"],
                item.get("execution_id") == old.get("execution_id"),
                item.get("run_id") == old.get("run_id"),
                item.get("selection_rank") != 30,
                item.get("task_version_id") != CONTROLLERS[CONTROLLER]["task_version_id"],
            )
        ):
            raise ValueError("peer-free rank30 execution rollforward drifted")
    return {CONTROLLER: plan}


def release_projection(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "controller": CONTROLLER,
        "selection_rank": 30,
        "task_version_id": CONTROLLERS[CONTROLLER]["task_version_id"],
        "job_name": JOB_NAME,
        "configmap_name": CONFIGMAP_NAME,
        "sfs_root": SFS_ROOT,
        "plan_sha256": plan["plan_sha256"],
        "cells": [
            {
                "attempt": row["attempt"],
                "cell_id": row["cell_id"],
                "execution_generation": row["execution_generation"],
                "execution_id": row["execution_id"],
                "run_id": row["run_id"],
            }
            for row in plan["attempts"]
        ],
    }


def expected_held(plan: dict[str, Any], source_package_sha256: str) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": HELD_SCHEMA,
            "status": "HELD_PENDING_FRESH_SCORE_BLIND_RELEASE",
            "launch_authorized": False,
            "scoring_authorized": False,
            "controller": release_projection(plan),
            "source_package_sha256": source_package_sha256,
            "ledger_authority": LEDGER_AUTHORITY,
            "live_ledger_validation": LIVE_LEDGER_VALIDATION,
            "diagnostic_v2": DIAGNOSTIC_V2,
            "superseded_identities": SUPERSEDED_IDENTITIES,
            "required_release": {
                "fresh_uid_bound_observer": True,
                "all_four_rank30_cells_clear_all_generations": True,
                "canonical_claim_session_accepted_output_collisions_zero": True,
                "archived_sessions_included": True,
                "stable_keyset_session_snapshot_required": True,
                "session_identity_projection_required": "/v1/sessions/identities",
                "fresh_job_configmap_sfs_collisions_zero": True,
                "active_hosted_controllers": 0,
                "both_endpoint_slots_simultaneously_free": True,
                "current_peer_required": False,
                "create_once": True,
                "max_age_seconds": RELEASE_MAX_AGE_SECONDS,
            },
            "privacy": {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
        }
    )


def validate_release(
    release: dict[str, Any], plan: dict[str, Any], source_package_sha256: str
) -> None:
    collision = release.get("fresh_collision_reconciliation") or {}
    if any(
        (
            set(release)
            != {
                "schema_version",
                "status",
                "checked_at_utc",
                "launch_authorized",
                "scoring_authorized",
                "controller",
                "source_package_sha256",
                "ledger_authority",
                "live_ledger_validation",
                "diagnostic_v2",
                "superseded_identities",
                "fresh_collision_reconciliation",
                "privacy",
                "receipt_sha256",
            },
            release.get("schema_version") != RELEASE_SCHEMA,
            release.get("status") != "CLEAR_PEER_FREE",
            release.get("launch_authorized") is not True,
            release.get("scoring_authorized") is not True,
            release.get("controller") != release_projection(plan),
            release.get("source_package_sha256") != source_package_sha256,
            SHA256_RE.fullmatch(str(source_package_sha256)) is None,
            release.get("ledger_authority") != LEDGER_AUTHORITY,
            release.get("live_ledger_validation") != LIVE_LEDGER_VALIDATION,
            release.get("diagnostic_v2") != DIAGNOSTIC_V2,
            release.get("superseded_identities") != SUPERSEDED_IDENTITIES,
            set(collision)
            != {
                "checked_immediately_before_create",
                "observer_job_uid",
                "observer_pod_uid",
                "observed_cells",
                "all_generation_claim_collisions",
                "authoritative_session_collisions",
                "accepted_evidence_collisions",
                "output_root_collisions",
                "new_job_collisions",
                "new_pod_collisions",
                "new_configmap_collisions",
                "active_hosted_controllers",
                "endpoint_lease_slots_available",
                "both_endpoint_lease_slots_simultaneously_free",
                "api_mutations",
                "session_inventory_scans",
                "archived_sessions_included",
                "stable_session_snapshot",
                "session_snapshot_sha256",
                "session_identity_projection",
                "accepted_authority_snapshot_sha256",
            },
            collision.get("checked_immediately_before_create") is not True,
            prior.engine.UUID_RE.fullmatch(str(collision.get("observer_job_uid"))) is None,
            prior.engine.UUID_RE.fullmatch(str(collision.get("observer_pod_uid"))) is None,
            collision.get("observed_cells") != 4,
            any(
                collision.get(key) != 0
                for key in (
                    "all_generation_claim_collisions",
                    "authoritative_session_collisions",
                    "accepted_evidence_collisions",
                    "output_root_collisions",
                    "new_job_collisions",
                    "new_pod_collisions",
                    "new_configmap_collisions",
                    "active_hosted_controllers",
                    "api_mutations",
                )
            ),
            collision.get("endpoint_lease_slots_available") != 2,
            collision.get("both_endpoint_lease_slots_simultaneously_free") is not True,
            collision.get("session_inventory_scans") != 1,
            collision.get("archived_sessions_included") is not True,
            collision.get("stable_session_snapshot") is not True,
            collision.get("session_identity_projection") != "/v1/sessions/identities",
            SHA256_RE.fullmatch(str(collision.get("session_snapshot_sha256"))) is None,
            SHA256_RE.fullmatch(str(collision.get("accepted_authority_snapshot_sha256"))) is None,
            release.get("privacy")
            != {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
            release.get("receipt_sha256") != self_hosted.digest_without(release, "receipt_sha256"),
        )
    ):
        raise RuntimeError("peer-free rank30 release drifted")
    try:
        checked_at = datetime.fromisoformat(str(release["checked_at_utc"]).replace("Z", "+00:00"))
        age = (datetime.now(UTC) - checked_at).total_seconds()
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("peer-free rank30 release timestamp drifted") from None
    if age < -60 or age > RELEASE_MAX_AGE_SECONDS:
        raise RuntimeError("peer-free rank30 release is stale")


def load_runtime_release(plan: dict[str, Any], source_package_sha256: str) -> dict[str, Any]:
    release = load(RELEASE_PATH)
    validate_release(release, plan, source_package_sha256)
    return release
