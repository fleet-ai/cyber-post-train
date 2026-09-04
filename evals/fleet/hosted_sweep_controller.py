"""Create and run duplicate-safe hosted OpenCode task-boundary shards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import opencode_train_sweep_runner as legacy
from evals.fleet import self_hosted

PLAN_SCHEMA = "fleet-hosted-opencode-task-boundary-shard-v1"
SOURCE_SCHEMA = "opencode-hosted-successor-source-v1"
REMAINDER_SOURCE_SCHEMA = "hosted-opencode-remainder-source-v1"
DEDICATED_SCORING_RELEASE_SCHEMA = "fleet-cyber-glm53-dedicated-scoring-release-v1"
HOSTED_REPLACEMENT_SCORING_RELEASE_SCHEMA = (
    "fleet-glm53-hosted-replacement-scoring-release-v1"
)
QWEN_HTTP500_SCORING_RELEASE_SCHEMA = (
    "fleet-qwen38-http500-successor-scoring-release-v1"
)
GLM_HTTP500_SCORING_RELEASE_SCHEMA = (
    "fleet-glm53-http500-hosted-primary-scoring-release-v1"
)
GLM_DEDICATED_B_V5_SCORING_RELEASE_SCHEMA = (
    "fleet-glm53-dedicated-b-v5-scoring-release-v1"
)
QWEN_HTTP500_AUTH_STATEMENT = (
    "I authorize the create-once scored launch of Qwen successor plan "
    "sha256:8d6df6af63b308d648fb0c7ea9115f90b16de1d7e57b0968dda8fc8fd4a20c81 "
    "after append-only scoring-recovery, hydration, exact-treatment, duplicate, and "
    "overlap gates pass. It contains exactly the remaining 49 complete primary "
    "tasks/196 cells, uses fleet-train-high, retains prior completed source4 for "
    "50/200, and must not repeat any fenced or accepted cell."
)
GLM_HTTP500_AUTH_STATEMENT = (
    "I authorize the create-once scored launch of hosted GLM plan "
    "sha256:8b0deafa9f51b75a0715be56b417454e96665b5342d4c346d5cda32dc24c8279 "
    "after append-only scoring-recovery, hydration, exact-treatment, duplicate, and "
    "overlap gates pass. It contains exactly 46 complete primary tasks/184 cells, "
    "uses fleet-train-high, is disjoint from dedicated A and B, and must not repeat "
    "any fenced or accepted cell."
)
GLM_DEDICATED_B_V5_AUTH_STATEMENT = (
    "I authorize the create-once scored launch of dedicated GLM B v5 successor "
    "plan sha256:b3436ff09a6f382dac12029dbd9bea2270ccf52ab7c945f07464e715590305b8 "
    "after exact parity, scoring-recovery, hydration, corrected attrition, duplicate, "
    "and overlap gates pass. It contains exactly 27 complete primary tasks/108 cells, "
    "uses fleet-train-high, binds B v5 runtime identities, excludes fenced source54, "
    "reuses source56 only under its proven zero-execution tombstone, and must not "
    "repeat any scored cell."
)
CAMPAIGNS = {
    "qwen38": "chris-cyber-q38-opencode11827-hosted-complete49-p4-v5",
    "glm53": "chris-cyber-glm53-opencode11827-hosted-complete99-p4-v5",
    "glm53_clean": "chris-cyber-glm53-opencode11827-hosted-complete98-p4-v6",
    "glm53_hosted_odd": "chris-cyber-glm53-opencode11827-hosted-odd49-p4-v7",
}
SOURCE_MODEL_KEYS = {
    "qwen38": "qwen38",
    "glm53": "glm53",
    "glm53_clean": "glm53",
    "glm53_hosted_odd": "glm53",
}
EXPECTED_SOURCE_TASK_COUNTS = {
    "qwen38": 50,
    "glm53": 100,
    "glm53_clean": 100,
    "glm53_hosted_odd": 100,
}
EXPECTED_INCLUDED_TASK_COUNTS = {
    "qwen38": 49,
    "glm53": 99,
    "glm53_clean": 98,
    "glm53_hosted_odd": 49,
}
ADDITIONAL_DEFERRED_SOURCE_RANKS = {
    "qwen38": set(),
    "glm53": set(),
    "glm53_clean": {1},
    "glm53_hosted_odd": {1},
}
RESERVED_SOURCE_RANKS = {
    "qwen38": set(),
    "glm53": set(),
    "glm53_clean": set(),
    "glm53_hosted_odd": set(range(4, 101, 2)),
}
REMAINDER_CAMPAIGNS = {
    "qwen38_remainder": "chris-cyber-q38-opencode11827-hosted-complete48-p4-v6",
    "glm53_remainder": "chris-cyber-glm53-opencode11827-hosted-odd47-p4-v8",
    "qwen38_remainder2": "chris-cyber-q38-opencode11827-hosted-complete47-p4-v7",
    "glm53_remainder2": "chris-cyber-glm53-opencode11827-hosted-odd46-p4-v9",
    "glm53_remainder3": "chris-cyber-glm53-opencode11827-hosted-odd45-p4-v10",
}
DEDICATED_CAMPAIGNS = {
    "glm53_dedicated_a": "chris-cyber-glm53-opencode11827-dedicated-a-even27-p4-v1",
    "glm53_dedicated_b": "chris-cyber-glm53-opencode11827-dedicated-b-even27-p4-v1",
}
QWEN_REPLACEMENT_CAMPAIGN = (
    "chris-cyber-q38-opencode11827-hosted-replacements3-p4-v1"
)
GLM_HOSTED_REPLACEMENT_CAMPAIGN = (
    "chris-cyber-glm53-opencode11827-hosted-replacement-r106-p4-v1"
)
GLM_HOSTED_REASSIGNED_B_CAMPAIGN = (
    "chris-cyber-glm53-opencode11827-hosted-reassigned-b27-p4-v1"
)
QWEN_HTTP500_SUCCESSOR_CAMPAIGN = (
    "chris-cyber-q38-opencode11827-hosted-successor49-p4-v8"
)
GLM_HTTP500_SUCCESSOR_CAMPAIGN = (
    "chris-cyber-glm53-opencode11827-hosted-successor73-p4-v11"
)
GLM_HTTP500_HOSTED_PRIMARY_CAMPAIGN = (
    "chris-cyber-glm53-opencode11827-hosted-primary46-p4-v12"
)
GLM_DEDICATED_B_V5_CAMPAIGN = (
    "chris-cyber-glm53-opencode11827-dedicated-b-v5-successor27-p4-v2"
)
EXPECTED_INCLUDED_TASK_COUNTS.update(
    {
        "qwen38_remainder": 48,
        "glm53_remainder": 47,
        "qwen38_remainder2": 47,
        "glm53_remainder2": 46,
        "glm53_remainder3": 45,
    }
)
EXPECTED_INCLUDED_TASK_COUNTS.update(
    {"glm53_dedicated_a": 27, "glm53_dedicated_b": 27}
)
EXPECTED_INCLUDED_TASK_COUNTS.update({"qwen38_replacements": 3})
EXPECTED_INCLUDED_TASK_COUNTS.update({"glm53_hosted_replacement": 1})
EXPECTED_INCLUDED_TASK_COUNTS.update({"glm53_hosted_reassigned_b": 27})
EXPECTED_INCLUDED_TASK_COUNTS.update(
    {
        "qwen38_http500_successor": 49,
        "glm53_http500_successor": 73,
        "glm53_http500_hosted_primary": 46,
        "glm53_dedicated_b_v5": 27,
    }
)
SCHEDULE = [
    {
        "accepted_outcomes_at_least": 0,
        "workers": 1,
        "headroom_gate": "fixed_at_launch_no_automatic_widening",
    },
]


def _emit(event: str, **fields: Any) -> None:
    """Emit only score- and content-blind controller progress."""
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def digest_without(value: dict[str, Any], field: str) -> str:
    return self_hosted.digest_without(value, field)


def validate_source(value: dict[str, Any]) -> None:
    if value.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("unsupported hosted successor source schema")
    if value.get("receipt_sha256") != digest_without(value, "receipt_sha256"):
        raise ValueError("hosted successor source digest mismatch")
    if value.get("scores_read") is not False or value.get("prompts_or_traces_read") is not False:
        raise ValueError("hosted successor source crossed the sealed-content boundary")


def validate_remainder_source(value: dict[str, Any]) -> None:
    if value.get("schema_version") != REMAINDER_SOURCE_SCHEMA:
        raise ValueError("unsupported hosted remainder source schema")
    if value.get("receipt_sha256") != digest_without(value, "receipt_sha256"):
        raise ValueError("hosted remainder source digest mismatch")
    if value.get("scores_read") is not False or value.get("prompts_or_traces_read") is not False:
        raise ValueError("hosted remainder source crossed the sealed-content boundary")


def _dedicated_replica(plan: dict[str, Any]) -> str | None:
    campaigns = {**CAMPAIGNS, **REMAINDER_CAMPAIGNS, **DEDICATED_CAMPAIGNS}
    campaign_to_shard = {value: key for key, value in campaigns.items()}
    shard_key = plan.get("shard_key") or campaign_to_shard.get(plan.get("campaign_id"))
    if shard_key == "glm53_dedicated_a":
        return "A"
    if shard_key == "glm53_dedicated_b":
        return "B"
    return None


def validate_dedicated_scoring_release(
    plan: dict[str, Any], release: dict[str, Any] | None
) -> None:
    """Require the append-only, score-blind release for dedicated GLM scoring."""
    replica = _dedicated_replica(plan)
    if replica is None:
        return
    if not isinstance(release, dict):
        raise ValueError("dedicated scoring release receipt is required")
    if (
        release.get("schema_version") != DEDICATED_SCORING_RELEASE_SCHEMA
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or release.get("append_only") is not True
        or release.get("supersedes") is not None
    ):
        raise ValueError("dedicated scoring release receipt is invalid")

    source = plan.get("source") or {}
    selection = release.get("selection_lock") or {}
    hydration = release.get("hydration") or {}
    replicas = release.get("replicas") or {}
    row = replicas.get(replica) or {}
    plan_evidence = row.get("plan") or {}
    parity = row.get("parity") or {}
    preflight = row.get("preflight") or {}
    launch = row.get("scored_launch") or {}
    authorization = release.get("authorization") or {}
    selected = [int(task["source_rank"]) for task in plan.get("tasks") or []]
    if (
        selection.get("receipt_sha256") != source.get("assignment_receipt_sha256")
        or selection.get("rewritten") is not False
        or hydration.get("receipt_sha256") != source.get("hydration_receipt_sha256")
        or hydration.get("passed") is not True
        or plan_evidence.get("plan_sha256") != plan.get("plan_sha256")
        or plan_evidence.get("task_count") != plan.get("task_count")
        or plan_evidence.get("session_count") != plan.get("total_session_count")
        or plan_evidence.get("source_ranks") != selected
        or parity.get("receipt_sha256") != source.get("canary_evidence_sha256")
        or parity.get("passed") is not True
        or preflight.get("plan_sha256") != plan.get("plan_sha256")
        or preflight.get("passed") is not True
        or preflight.get("exit_code") != 0
        or preflight.get("restart_count") != 0
        or not str(preflight.get("receipt_sha256") or "").startswith("sha256:")
        or launch.get("create_once") is not True
        or launch.get("must_not_repeat") is not True
        or authorization.get("timestamp_utc") != release.get("created_at")
        or release.get("created_at") != "2026-09-04T03:53:09Z"
        or authorization.get("author") != "/root"
        or plan.get("plan_sha256") not in (authorization.get("authorized_plan_sha256") or [])
        or plan.get("plan_sha256") not in str(authorization.get("statement") or "")
        or "must not be repeated" not in str(authorization.get("statement") or "")
    ):
        raise ValueError("dedicated scoring release does not bind this plan")
    for evidence in (preflight, launch):
        for field in ("job_uid", "pod_uid"):
            try:
                uuid.UUID(str(evidence.get(field)))
            except ValueError as exc:
                raise ValueError("dedicated scoring release has an invalid UID") from exc

    a = set((replicas.get("A") or {}).get("plan", {}).get("source_ranks") or [])
    b = set((replicas.get("B") or {}).get("plan", {}).get("source_ranks") or [])
    hosted = set(range(9, 100, 2))
    primary = release.get("primary_estimator") or {}
    conditions = release.get("release_conditions") or {}
    privacy = release.get("privacy") or {}
    if (
        a != {*range(4, 53, 2), 102, 104}
        or b != {*range(54, 101, 2), 101, 103, 105}
        or a & b
        or a & hosted
        or b & hosted
        or len(a | b | hosted) != 100
        or primary.get("unique_task_count") != 100
        or primary.get("pass_k") != 4
        or primary.get("session_count") != 400
        or primary.get("pairwise_disjoint") is not True
        or conditions.get("release_granted") is not True
        or not all(
            conditions.get(field) is True
            for field in (
                "hydration_passed",
                "both_parity_gates_passed",
                "both_preflights_passed",
                "current_plan_identities_absent_at_preflight",
                "serving_blocks_explicit",
                "task_blocks_pairwise_disjoint",
                "primary_estimator_is_exactly_100_tasks_pass4",
                "scored_launches_are_create_once",
            )
        )
        or any(
            privacy.get(field) is not False
            for field in (
                "scores_read",
                "prompts_included",
                "traces_included",
                "flags_included",
                "credentials_included",
            )
        )
    ):
        raise ValueError("dedicated scoring release arithmetic or privacy drifted")


def validate_hosted_replacement_scoring_release(
    plan: dict[str, Any], release: dict[str, Any] | None
) -> None:
    """Require an append-only hydration and scoring release for hosted r106."""
    if plan.get("shard_key") != "glm53_hosted_replacement":
        return
    if not isinstance(release, dict):
        raise ValueError("hosted replacement scoring release receipt is required")
    source = plan.get("source") or {}
    supplement = release.get("selection_supplement") or {}
    hydration = release.get("hydration") or {}
    plan_evidence = release.get("plan") or {}
    treatment = release.get("treatment") or {}
    estimator = release.get("primary_estimator") or {}
    authorization = release.get("authorization") or {}
    privacy = release.get("privacy") or {}
    if (
        release.get("schema_version") != HOSTED_REPLACEMENT_SCORING_RELEASE_SCHEMA
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or release.get("append_only") is not True
        or release.get("supersedes") is not None
        or supplement.get("receipt_sha256")
        != source.get("selection_supplement_receipt_sha256")
        or supplement.get("rewritten") is not False
        or hydration.get("receipt_sha256") != source.get("hydration_receipt_sha256")
        or hydration.get("passed") is not True
        or hydration.get("exit_code") != 0
        or hydration.get("restart_count") != 0
        or hydration.get("model_or_scoring_calls") is not False
        or plan_evidence.get("plan_sha256") != plan["plan_sha256"]
        or plan_evidence.get("source_ranks") != [106]
        or plan_evidence.get("task_count") != 1
        or plan_evidence.get("pass_k") != 4
        or plan_evidence.get("cell_count") != 4
        or plan_evidence.get("workers") != 1
        or treatment.get("kind") != "hosted_inference_endpoint_v1"
        or treatment.get("endpoint_origin") != plan["model"]["endpoint_origin"]
        or treatment.get("served_id") != plan["model"]["served_id"]
        or treatment.get("model_revision") != plan["model"]["revision"]
        or treatment.get("required_task_tools")
        != plan["execution"]["required_task_tools"]
        or treatment.get("required_task_tool_catalog_sha256")
        != plan["execution"]["required_task_tool_catalog_sha256"]
        or estimator.get("unique_task_count") != 100
        or estimator.get("pass_k") != 4
        or estimator.get("cell_count") != 400
        or estimator.get("pairwise_disjoint") is not True
        or authorization.get("author") != "/root"
        or authorization.get("timestamp_utc") != release.get("created_at")
        or authorization.get("scored_launch_authorized") is not True
        or authorization.get("create_once") is not True
        or authorization.get("must_not_repeat") is not True
        or plan["plan_sha256"] not in str(authorization.get("statement") or "")
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("hosted replacement scoring release does not bind this plan")
    for field in ("job_uid", "pod_uid"):
        try:
            uuid.UUID(str(hydration.get(field)))
        except ValueError as exc:
            raise ValueError("hosted replacement hydration has an invalid UID") from exc


def validate_qwen_http500_scoring_release(
    plan: dict[str, Any], release: dict[str, Any] | None
) -> None:
    """Fail closed until the exact recovery-bound Qwen release is supplied."""
    if plan.get("shard_key") != "qwen38_http500_successor":
        return
    if not isinstance(release, dict):
        raise ValueError("Qwen HTTP500 successor scoring release is required")
    source = plan.get("source") or {}
    plan_evidence = release.get("plan") or {}
    gates = release.get("gates") or {}
    scheduling = release.get("scheduling") or {}
    authorization = release.get("authorization") or {}
    privacy = release.get("privacy") or {}
    if (
        release.get("schema_version") != QWEN_HTTP500_SCORING_RELEASE_SCHEMA
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or plan_evidence.get("plan_sha256") != plan["plan_sha256"]
        or plan_evidence.get("new_task_count") != 49
        or plan_evidence.get("new_cell_count") != 196
        or plan_evidence.get("retained_complete_task_count") != 1
        or plan_evidence.get("primary_task_count") != 50
        or plan_evidence.get("primary_cell_count") != 200
        or gates.get("incident_receipt_sha256")
        != source.get("incident_receipt_sha256")
        or gates.get("selection_supplement_receipt_sha256")
        != source.get("selection_supplement_receipt_sha256")
        or gates.get("hydration_receipt_sha256")
        != source.get("hydration_receipt_sha256")
        or gates.get("hydration_execution_receipt_sha256")
        != "sha256:00660b55b4aa3c5b02f796b29e424acb45caaf4e6da432b203d237b01710f1ec"
        or gates.get("scoring_recovery_receipt_sha256")
        != "sha256:708d635968e073d27153d3d9f13ae5fa7b4784349cc9e8e3928f0fbafeade132"
        or not all(
            gates.get(field) is True
            for field in (
                "exact_treatment_gate_required",
                "duplicate_gate_required",
                "overlap_gate_required",
                "fresh_create_once_identity_required",
            )
        )
        or scheduling.get("required_priority_class") != "fleet-train-high"
        or scheduling.get("true_non_preemptible_available") is not False
        or scheduling.get("priority_class_is_not_preemption_immunity") is not True
        or authorization.get("timestamp_utc") != "2026-09-04T05:46:25Z"
        or authorization.get("author") != "/root"
        or authorization.get("statement") != QWEN_HTTP500_AUTH_STATEMENT
        or authorization.get("scored_launch_authorized") is not True
        or authorization.get("must_not_repeat") is not True
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("Qwen HTTP500 scoring release does not bind this plan")


def validate_glm_http500_scoring_release(
    plan: dict[str, Any], release: dict[str, Any] | None
) -> None:
    """Fail closed until the exact recovery-bound hosted GLM release is supplied."""
    if plan.get("shard_key") != "glm53_http500_hosted_primary":
        return
    if not isinstance(release, dict):
        raise ValueError("GLM HTTP500 hosted scoring release is required")
    source = plan.get("source") or {}
    plan_evidence = release.get("plan") or {}
    gates = release.get("gates") or {}
    scheduling = release.get("scheduling") or {}
    authorization = release.get("authorization") or {}
    privacy = release.get("privacy") or {}
    if (
        release.get("schema_version") != GLM_HTTP500_SCORING_RELEASE_SCHEMA
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or plan_evidence.get("plan_sha256") != plan["plan_sha256"]
        or plan_evidence.get("hosted_task_count") != 46
        or plan_evidence.get("hosted_cell_count") != 184
        or plan_evidence.get("dedicated_a_task_count") != 27
        or plan_evidence.get("dedicated_b_task_count") != 27
        or plan_evidence.get("primary_task_count") != 100
        or plan_evidence.get("primary_cell_count") != 400
        or gates.get("incident_receipt_sha256")
        != source.get("incident_receipt_sha256")
        or gates.get("selection_supplement_receipt_sha256")
        != source.get("selection_supplement_receipt_sha256")
        or gates.get("hydration_receipt_sha256")
        != source.get("hydration_receipt_sha256")
        or gates.get("hydration_execution_receipt_sha256")
        != "sha256:00660b55b4aa3c5b02f796b29e424acb45caaf4e6da432b203d237b01710f1ec"
        or gates.get("scoring_recovery_receipt_sha256")
        != "sha256:708d635968e073d27153d3d9f13ae5fa7b4784349cc9e8e3928f0fbafeade132"
        or not all(
            gates.get(field) is True
            for field in (
                "exact_treatment_gate_required",
                "duplicate_gate_required",
                "overlap_gate_required",
                "fresh_create_once_identity_required",
            )
        )
        or scheduling.get("required_priority_class") != "fleet-train-high"
        or scheduling.get("true_non_preemptible_available") is not False
        or scheduling.get("priority_class_is_not_preemption_immunity") is not True
        or authorization.get("timestamp_utc") != "2026-09-04T05:53:30Z"
        or authorization.get("author") != "/root"
        or authorization.get("statement") != GLM_HTTP500_AUTH_STATEMENT
        or authorization.get("scored_launch_authorized") is not True
        or authorization.get("must_not_repeat") is not True
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("GLM HTTP500 scoring release does not bind this plan")


def validate_glm_dedicated_b_v5_scoring_release(
    plan: dict[str, Any], release: dict[str, Any] | None
) -> None:
    """Fail closed until the exact B v5 release and root authorization are supplied."""
    if plan.get("shard_key") != "glm53_dedicated_b_v5":
        return
    if not isinstance(release, dict):
        raise ValueError("GLM dedicated B v5 scoring release is required")
    source = plan.get("source") or {}
    treatment = plan.get("treatment_block") or {}
    plan_evidence = release.get("plan") or {}
    gates = release.get("gates") or {}
    release_treatment = release.get("treatment") or {}
    scheduling = release.get("scheduling") or {}
    authorization = release.get("authorization") or {}
    privacy = release.get("privacy") or {}
    if (
        release.get("schema_version") != GLM_DEDICATED_B_V5_SCORING_RELEASE_SCHEMA
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or release.get("append_only") is not True
        or release.get("supersedes") is not None
        or plan_evidence.get("plan_sha256") != plan["plan_sha256"]
        or plan_evidence.get("task_count") != 27
        or plan_evidence.get("pass_k") != 4
        or plan_evidence.get("cell_count") != 108
        or plan_evidence.get("source_ranks")
        != [int(row["source_rank"]) for row in plan["tasks"]]
        or gates.get("forced_stop_tombstone_receipt_sha256")
        != source.get("forced_stop_tombstone_receipt_sha256")
        or gates.get("selection_supplement_receipt_sha256")
        != source.get("selection_supplement_receipt_sha256")
        or gates.get("hydration_receipt_sha256")
        != source.get("hydration_receipt_sha256")
        or gates.get("parity_receipt_sha256")
        != source.get("parity_receipt_sha256")
        or gates.get("scoring_recovery_receipt_sha256")
        != "sha256:708d635968e073d27153d3d9f13ae5fa7b4784349cc9e8e3928f0fbafeade132"
        or gates.get("source54_fenced") is not True
        or gates.get("source56_zero_execution_reuse_proven") is not True
        or not all(
            gates.get(field) is True
            for field in (
                "exact_treatment_gate_required",
                "duplicate_gate_required",
                "overlap_gate_required",
                "fresh_create_once_identity_required",
            )
        )
        or any(
            release_treatment.get(field) != treatment.get(field)
            for field in (
                "kind",
                "replica",
                "serving_generation",
                "service_uid",
                "ray_job_uid",
                "ray_cluster_uid",
                "head_pod_uid",
                "model_revision",
            )
        )
        or scheduling.get("required_priority_class") != "fleet-train-high"
        or scheduling.get("true_non_preemptible_available") is not False
        or scheduling.get("priority_class_is_not_preemption_immunity") is not True
        or scheduling.get("workers") != 1
        or authorization.get("timestamp_utc") != "2026-09-04T06:04:38Z"
        or authorization.get("author") != "/root"
        or authorization.get("statement") != GLM_DEDICATED_B_V5_AUTH_STATEMENT
        or authorization.get("scored_launch_authorized") is not True
        or authorization.get("must_not_repeat") is not True
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("GLM dedicated B v5 scoring release does not bind this plan")


def validate_dedicated_a_stop_tombstone(receipt: dict[str, Any]) -> None:
    """Reject a zero-execution classification when a nonempty agent stream exists."""
    stopped = receipt.get("stopped_controller") or {}
    attempt = receipt.get("source4_attempt4") or {}
    fence = receipt.get("source4_task_fence") or {}
    later = receipt.get("later_task_claim_audit") or {}
    if (
        receipt.get("schema_version")
        != "fleet-glm53-dedicated-a-controller-stop-tombstone-v2"
        or receipt.get("receipt_sha256") != digest_without(receipt, "receipt_sha256")
        or (receipt.get("supersedes") or {}).get("receipt_sha256")
        != "sha256:9ef664c9a2abcdb99137d4b1a648b39b5134c3985ec48897716b92fb107e4e1d"
        or stopped.get("job_absent_after_delete") is not True
        or stopped.get("pod_absent_after_delete") is not True
        or stopped.get("sfs_preserved") is not True
        or attempt.get("opencode_stream_present") is not True
        or int(attempt.get("opencode_stream_bytes") or 0) <= 0
        or attempt.get("agent_execution_started") is not True
        or attempt.get("agent_execution_completed") is not False
        or attempt.get("scientific_classification")
        != "infrastructure_incomplete_mid_agent_pre_scoring_session_verifier"
        or any(
            attempt.get(field) is not False
            for field in (
                "result_present",
                "reward_result_present",
                "verifier_execution_present",
                "session_ingest_present",
                "accepted_or_noncreditable_receipt_present",
            )
        )
        or fence.get("whole_task_fenced") is not True
        or fence.get("must_not_repeat_any_source4_cell") is not True
        or later.get("source6_claim_present") is not False
        or later.get("later_task_claimed") is not False
    ):
        raise ValueError("dedicated A stop tombstone classification drifted")


def build_remainder_plan(
    predecessor: dict[str, Any], source: dict[str, Any], model_key: str
) -> dict[str, Any]:
    validate_plan(predecessor)
    validate_remainder_source(source)
    if model_key not in REMAINDER_CAMPAIGNS:
        raise ValueError("unsupported hosted remainder shard")
    source_key = model_key.split("_", 1)[0]
    evidence = source["runs"][source_key]
    if (
        evidence.get("predecessor_plan_sha256") != predecessor["plan_sha256"]
        or evidence.get("active_attempts") != 0
        or not evidence.get("job_uid")
        or not evidence.get("pod_uid")
    ):
        raise ValueError("hosted remainder predecessor is not terminal-bound")
    excluded_source_ranks = {int(rank) for rank in evidence["excluded_source_ranks"]}
    expected_excluded_by_model = {
        "qwen38_remainder": {2},
        "glm53_remainder": {3, 5},
        "qwen38_remainder2": {3},
        "glm53_remainder2": {7},
        "glm53_remainder3": {9},
    }
    expected_excluded = expected_excluded_by_model[model_key]
    if excluded_source_ranks != expected_excluded:
        raise ValueError("hosted remainder excluded-task boundary drifted")
    for outcome in evidence["outcomes"]:
        if int(outcome["source_rank"]) not in excluded_source_ranks:
            raise ValueError("hosted remainder outcome escaped excluded task")
        if outcome.get("retry_allowed") is not False:
            raise ValueError("hosted remainder outcome became retryable")

    tasks = []
    for source_task in predecessor["tasks"]:
        source_rank = int(source_task["source_rank"])
        if source_rank in excluded_source_ranks:
            continue
        task = copy.deepcopy(source_task)
        task["rank"] = len(tasks) + 1
        tasks.append(task)
    if len(tasks) != EXPECTED_INCLUDED_TASK_COUNTS[model_key]:
        raise ValueError("hosted remainder included-task count drifted")

    campaign = REMAINDER_CAMPAIGNS[model_key]
    attempts = []
    for task in tasks:
        rank = int(task["rank"])
        source_rank = int(task["source_rank"])
        key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][:8]
        for attempt_number in range(1, 5):
            attempts.append(
                {
                    "ordinal": len(attempts) + 1,
                    "rank": rank,
                    "source_rank": source_rank,
                    "attempt": attempt_number,
                    "run_id": f"{campaign}-sr{source_rank:03d}-a{attempt_number}-{key_digest}",
                    "network": f"{model_key}-sr{source_rank:03d}-a{attempt_number}-{key_digest}",
                }
            )
    predecessor_tasks = {int(row["source_rank"]): row for row in predecessor["tasks"]}
    excluded_tasks = [
        {
            "source_rank": source_rank,
            "task_key": predecessor_tasks[source_rank]["task"]["key"],
            "task_version_id": predecessor_tasks[source_rank]["task"]["version_id"],
            "reason": "predecessor_task_partially_touched_and_fenced",
            "retry_allowed": False,
            "credited": False,
        }
        for source_rank in sorted(excluded_source_ranks)
    ]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": model_key,
        "campaign_id": campaign,
        "source_job_id": predecessor["source_job_id"],
        "source": {
            "predecessor_plan_sha256": predecessor["plan_sha256"],
            "source_state_receipt_sha256": source["receipt_sha256"],
            "source_job_name": evidence["job_name"],
            "source_job_uid": evidence["job_uid"],
            "source_pod_uid": evidence["pod_uid"],
        },
        "treatment_block": predecessor["treatment_block"],
        "model": predecessor["model"],
        "harness": predecessor["harness"],
        "authority": predecessor["authority"],
        "task_count": len(tasks),
        "pass_k": 4,
        "total_session_count": len(tasks) * 4,
        "credited_sessions": [],
        "new_session_count": len(attempts),
        "upstream_excluded_tasks": [
            *(predecessor.get("upstream_excluded_tasks") or []),
            *predecessor["excluded_tasks"],
        ],
        "excluded_tasks": excluded_tasks,
        "execution": {
            **predecessor["execution"],
            "inventory_policy": (
                "plan_identity_plus_authoritative_receipt_v1"
                if model_key.startswith("qwen38_")
                else "conservative_no_same_model_session_for_task_key_v1"
            ),
        },
        "tasks": tasks,
        "attempts": attempts,
        "privacy": predecessor["privacy"],
    }
    if predecessor.get("reserved_tasks"):
        plan["reserved_tasks"] = predecessor["reserved_tasks"]
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def hydrate_glm53_replacements(
    assignment: dict[str, Any], key: str
) -> dict[str, Any]:
    """Hydrate locked GLM replacements without retaining task content."""
    if assignment.get("receipt_sha256") != digest_without(assignment, "receipt_sha256"):
        raise ValueError("replacement assignment receipt digest mismatch")
    supplement_schema = assignment.get("schema_version")
    if supplement_schema == "fleet-qwen38-replacement-selection-supplement-v2":
        rows = assignment.get("replacements") or []
        expected_ranks = [54, 55]
        if (
            assignment.get("append_only") is not True
            or (assignment.get("hydration_gate") or {}).get("status")
            != "required_not_satisfied"
            or any(row.get("serving_block") != "hosted_qwen_successor" for row in rows)
        ):
            raise ValueError("Qwen replacement supplement drifted")
        receipt_schema = "fleet-qwen38-hosted-replacement-hydration-v2"
        receipt_source_field = "selection_supplement_receipt_sha256"
    elif supplement_schema == "fleet-opencode-replacement-selection-supplement-v3":
        rows = assignment.get("replacements") or []
        expected_ranks = [108, 109]
        if (
            assignment.get("append_only") is not True
            or (assignment.get("hydration_gate") or {}).get("status")
            != "required_not_satisfied"
            or any(row.get("serving_block") != "hosted_glm_successor" for row in rows)
            or (assignment.get("prior_supplement") or {}).get("receipt_sha256")
            != "sha256:731ba0583f43a7cc56f16cbf8c71ab98ca39642ff7091e21763f8561bf9d587a"
        ):
            raise ValueError("GLM HTTP500 replacement supplement drifted")
        receipt_schema = "fleet-glm53-hosted-replacement-hydration-v2"
        receipt_source_field = "selection_supplement_receipt_sha256"
    elif supplement_schema == "fleet-opencode-replacement-selection-supplement-v4":
        row = assignment.get("replacement") or {}
        rows = [row]
        expected_ranks = [110]
        stop = assignment.get("controller_stop_evidence") or {}
        if (
            assignment.get("append_only") is not True
            or row.get("serving_block") != "dedicated_a_successor"
            or row.get("scored_launch_authorized") is not False
            or (row.get("hydration_gate") or {}).get("status")
            != "required_not_satisfied"
            or stop.get("tombstone_receipt_sha256")
            != "sha256:9bf6683d80c2dd517e371709a1aa2db0a13e1f4089a09a8296941c6df7054b79"
            or stop.get("corrected_tombstone_binds_nonempty_agent_stream") is not True
        ):
            raise ValueError("dedicated A replacement supplement drifted")
        receipt_schema = "fleet-glm53-dedicated-a-replacement-hydration-v1"
        receipt_source_field = "selection_supplement_receipt_sha256"
    elif supplement_schema in {
        "fleet-opencode-replacement-selection-supplement-v1",
        "fleet-opencode-replacement-selection-supplement-v2",
    }:
        row = assignment.get("replacement") or {}
        rows = [row]
        expected_rank = 106 if supplement_schema.endswith("v1") else 107
        expected_serving_block = (
            "hosted" if expected_rank == 106 else "dedicated_b_successor"
        )
        expected_ranks = [expected_rank]
        if (
            assignment.get("append_only") is not True
            or row.get("serving_block") != expected_serving_block
            or row.get("scored_launch_authorized") is not False
            or (row.get("hydration_gate") or {}).get("status")
            != "required_not_satisfied"
        ):
            raise ValueError("replacement supplement drifted")
        if expected_rank == 107 and (
            (assignment.get("forced_stop_tombstone") or {}).get("receipt_sha256")
            != "sha256:7cf5483e07e4053d7807d599b738e0555ade3b728ea20ae2356186592bd7c355"
        ):
            raise ValueError("dedicated B stop tombstone drifted")
        receipt_schema = (
            "fleet-glm53-hosted-replacement-hydration-v1"
            if expected_rank == 106
            else "fleet-glm53-dedicated-b-replacement-hydration-v1"
        )
        receipt_source_field = "selection_supplement_receipt_sha256"
    else:
        glm = assignment.get("glm53") or {}
        rows = glm.get("tasks") or []
        expected_ranks = list(range(101, 106))
        receipt_schema = "fleet-glm53-replacement-hydration-v1"
        receipt_source_field = "assignment_receipt_sha256"
    if [int(row.get("replacement_rank") or 0) for row in rows] != expected_ranks:
        raise ValueError("replacement assignment ranks drifted")
    with _client(key) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
        hydrated = []
        for row in rows:
            task = self_hosted._request(
                client,
                "GET",
                f"/v1/tasks/{row['task_key']}",
                params={"version_id": row["task_version_id"]},
            )
            metadata = task.get("metadata") or {}
            verifier = task.get("verifier") or {}
            runtime_seed = metadata.get("runtime_seed_manifest") or {}
            actual_binding = {
                "task_key": task.get("key"),
                "env_key": task.get("environment_id"),
                "env_version": task.get("version"),
                "data_key": task.get("data_id"),
                "data_version": task.get("data_version"),
            }
            expected_binding = {field: row[field] for field in actual_binding}
            if actual_binding != expected_binding:
                raise RuntimeError("replacement exact task binding drifted")
            cyber_contract = metadata.get("cyber_contract")
            if cyber_contract != {
                "evidence_schema": "1.0.0",
                "submission_protocol": "2.0.0",
                "verifier_contract": "3.0.0",
            }:
                raise RuntimeError("replacement cyber contract drifted")
            verifier_receipt = {
                "id": task.get("verifier_id"),
                "version_id": verifier.get("verifier_version_id"),
                "version": verifier.get("version"),
                "sha256": verifier.get("sha256"),
                "function_name": verifier.get("function_name") or "verify",
            }
            if any(value in (None, "") for value in verifier_receipt.values()):
                raise RuntimeError("replacement verifier receipt is incomplete")
            if not runtime_seed.get("content_sha256") or not runtime_seed.get("files"):
                raise RuntimeError("replacement runtime seed manifest is incomplete")
            hydrated.append(
                {
                    "replacement_rank": int(row["replacement_rank"]),
                    "task": {
                        "key": row["task_key"],
                        "version_id": row["task_version_id"],
                        "prompt_sha256": self_hosted.sha256(
                            (task.get("prompt") or "").encode()
                        ),
                        "env_variables_sha256": self_hosted.sha256(
                            self_hosted.canonical_json(task.get("env_variables") or {})
                        ),
                        "output_json_schema_sha256": self_hosted.sha256(
                            self_hosted.canonical_json(task.get("output_json_schema"))
                        ),
                        "cyber_contract": cyber_contract,
                    },
                    "environment": {
                        "id": row["env_key"],
                        "version": row["env_version"],
                        "version_id": row["environment_version_id"],
                        "data_id": row["data_key"],
                        "data_version": row["data_version"],
                        "runtime_seed_content_sha256": runtime_seed["content_sha256"],
                        "ttl_seconds": 32400,
                    },
                    "verifier": verifier_receipt,
                    "runtime_seed_file_count": len(runtime_seed["files"]),
                }
            )
    receipt = {
        "schema_version": receipt_schema,
        receipt_source_field: assignment["receipt_sha256"],
        "fleet_team_id": self_hosted.FLEET_TEAM_ID,
        "tasks_hydrated": len(hydrated),
        "tasks": hydrated,
        "scores_read": False,
        "task_content_retained": False,
        "prompts_or_traces_included": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def build_dedicated_a_plan(
    source_plan: dict[str, Any],
    assignment: dict[str, Any],
    hydration: dict[str, Any],
    canary: dict[str, Any],
    replica: str = "A",
) -> dict[str, Any]:
    """Build one dedicated replica's immutable complete-task shard."""
    if replica not in {"A", "B"}:
        raise ValueError("unsupported dedicated replica")
    replica_key = replica.lower()
    shard_key = f"glm53_dedicated_{replica_key}"
    _validate_source_plan(source_plan)
    if int(source_plan.get("task_count") or 0) != 100:
        raise ValueError("dedicated source plan task count drifted")
    if assignment.get("receipt_sha256") != digest_without(assignment, "receipt_sha256"):
        raise ValueError("replacement assignment receipt digest mismatch")
    if (
        hydration.get("receipt_sha256")
        != digest_without(hydration, "receipt_sha256")
        or hydration.get("assignment_receipt_sha256") != assignment["receipt_sha256"]
        or hydration.get("tasks_hydrated") != 5
    ):
        raise ValueError("replacement hydration receipt drifted")
    assigned_rows = {
        int(row["replacement_rank"]): row for row in assignment["glm53"]["tasks"]
    }
    hydrated_rows = {
        int(row["replacement_rank"]): row for row in hydration["tasks"]
    }
    if set(assigned_rows) != set(range(101, 106)) or set(hydrated_rows) != set(
        range(101, 106)
    ):
        raise ValueError("replacement hydration ranks drifted")
    for rank, row in hydrated_rows.items():
        assigned = assigned_rows[rank]
        if (
            row["task"].get("key") != assigned["task_key"]
            or row["task"].get("version_id") != assigned["task_version_id"]
            or row["environment"].get("id") != assigned["env_key"]
            or row["environment"].get("version") != assigned["env_version"]
            or row["environment"].get("version_id")
            != assigned["environment_version_id"]
            or row["environment"].get("data_id") != assigned["data_key"]
            or row["environment"].get("data_version") != assigned["data_version"]
            or int(row.get("runtime_seed_file_count") or 0) < 1
        ):
            raise ValueError("replacement hydrated task binding drifted")
    blocks = assignment["glm53"]["dedicated_block_assignment"]
    if (
        blocks["dedicated_a"].get("replacement_ranks") != [102, 104]
        or blocks["dedicated_b"].get("replacement_ranks") != [101, 103, 105]
        or blocks["hosted"].get("plan_sha256")
        != "sha256:7fa9cb527083defd0197caccf7fb87634cfd2f22932afcf0d51a33f97be703f1"
    ):
        raise ValueError("dedicated replacement block assignment drifted")
    if (
        canary.get("classification") != "operational-gate-passed"
        or (canary.get("outcome_integrity") or {}).get("scored_sessions") != 0
        or (canary.get("network") or {}).get("health_http_status") != 200
        or (canary.get("model") or {}).get("revision")
        != source_plan["model"]["revision"]
        or (canary.get("controllers") or {}).get("ray_job_uid")
        != blocks[f"dedicated_{replica_key}"]["serving_ray_job_uid"]
        or (canary.get("immutable_config") or {}).get("sha256")
        != blocks[f"dedicated_{replica_key}"]["serving_config_sha256"]
    ):
        raise ValueError("dedicated replica A canary is not admissible")

    original_by_rank = {int(row["rank"]): row for row in source_plan["tasks"]}
    hydrated_by_rank = {
        int(row["replacement_rank"]): row for row in hydration["tasks"]
    }
    selected_source_ranks = (
        [*range(4, 53, 2), 102, 104]
        if replica == "A"
        else [*range(54, 101, 2), 101, 103, 105]
    )
    tasks = []
    for source_rank in selected_source_ranks:
        if source_rank <= 100:
            task = copy.deepcopy(original_by_rank[source_rank])
        else:
            row = hydrated_by_rank[source_rank]
            task = {
                "task": copy.deepcopy(row["task"]),
                "environment": copy.deepcopy(row["environment"]),
                "verifier": copy.deepcopy(row["verifier"]),
            }
        task["rank"] = len(tasks) + 1
        task["source_rank"] = source_rank
        task.pop("baseline_session_ids", None)
        tasks.append(task)

    campaign = DEDICATED_CAMPAIGNS[shard_key]
    attempts = []
    for task in tasks:
        source_rank = int(task["source_rank"])
        key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][:8]
        for attempt in range(1, 5):
            attempts.append(
                {
                    "ordinal": len(attempts) + 1,
                    "rank": int(task["rank"]),
                    "source_rank": source_rank,
                    "attempt": attempt,
                    "run_id": f"{campaign}-sr{source_rank:03d}-a{attempt}-{key_digest}",
                    "network": (
                        f"glm53-dedicated-{replica_key}-"
                        f"sr{source_rank:03d}-a{attempt}-{key_digest}"
                    ),
                }
            )

    model = copy.deepcopy(source_plan["model"])
    service_name = canary["network"]["service_name"]
    model["endpoint_origin"] = (
        f"http://{service_name}.fleet-train-jobs.svc.cluster.local:8000"
    )
    treatment = {
        "kind": "dedicated_inference_endpoint_v1",
        "replica": replica,
        "endpoint_origin": model["endpoint_origin"],
        "service_name": service_name,
        "service_uid": canary["network"]["service_uid"],
        "api_run_id": canary["api_run"]["run_id"],
        "ray_job_uid": canary["controllers"]["ray_job_uid"],
        "ray_cluster_uid": canary["controllers"]["ray_cluster_uid"],
        "head_pod_uid": canary["runtime"]["head_pod_uid"],
        "runtime_image_digest": canary["image"]["resolved_image_id"],
        "serving_config_sha256": canary["immutable_config"]["sha256"],
        "assignment_receipt_sha256": assignment["receipt_sha256"],
        "hydration_receipt_sha256": hydration["receipt_sha256"],
        "parity_receipt_sha256": self_hosted.sha256(
            self_hosted.canonical_json(canary)
        ),
        "served_id": model["served_id"],
        "model_revision": model["revision"],
        "session_model": model["session_model"],
        "harness": source_plan["harness"],
        "required_task_tools": source_plan["execution"]["required_task_tools"],
        "required_task_tool_catalog_sha256": source_plan["execution"][
            "required_task_tool_catalog_sha256"
        ],
    }
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": shard_key,
        "campaign_id": campaign,
        "source_job_id": source_plan["source_job_id"],
        "source": {
            "source_plan_sha256": source_plan["plan_sha256"],
            "assignment_receipt_sha256": assignment["receipt_sha256"],
            "hydration_receipt_sha256": hydration["receipt_sha256"],
            "canary_evidence_sha256": self_hosted.sha256(
                self_hosted.canonical_json(canary)
            ),
        },
        "treatment_block": treatment,
        "model": model,
        "harness": source_plan["harness"],
        "authority": source_plan["authority"],
        "task_count": len(tasks),
        "pass_k": 4,
        "total_session_count": len(tasks) * 4,
        "credited_sessions": [],
        "new_session_count": len(attempts),
        "fenced_source_ranks": [1, 2, 3, 5, 7],
        "hosted_source_ranks": list(range(9, 100, 2)),
        "reserved_source_ranks": (
            [*range(54, 101, 2), 101, 103, 105]
            if replica == "A"
            else [*range(4, 53, 2), 102, 104]
        ),
        "execution": {
            "task_partition": "complete_task_boundary",
            "same_task_max_inflight": 1,
            "retry_policy": "never_repeat_any_verifier_backed_outcome",
            "future_nonzero_exit_policy": "fence_task_and_continue_other_tasks",
            "training_data_eligible": True,
            "required_task_tools": source_plan["execution"]["required_task_tools"],
            "required_task_tool_catalog_sha256": source_plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
            "inventory_policy": "immutable_plan_claim_and_endpoint_uid_v1",
            "score_blind_concurrency_schedule": SCHEDULE,
        },
        "tasks": tasks,
        "attempts": attempts,
        "privacy": {
            "scores_included": False,
            "prompts_included": False,
            "transcripts_included": False,
            "credentials_included": False,
        },
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def build_qwen_replacement_plan(
    source_plan: dict[str, Any],
    hosted_plan: dict[str, Any],
    assignment: dict[str, Any],
) -> dict[str, Any]:
    """Build Qwen's three locked complete-task replacement cells."""
    _validate_source_plan(source_plan)
    validate_plan(hosted_plan)
    if (
        source_plan.get("plan_sha256")
        != "sha256:63bf008198170f4e55153c2307560560befe231f2fa11186acebf3e01e3ac50f"
        or hosted_plan.get("plan_sha256")
        != "sha256:f367d5148b035e29f57e5a29633609c4dbc8876d94c2795a79ff1ff25dad2ee2"
        or assignment.get("receipt_sha256")
        != digest_without(assignment, "receipt_sha256")
    ):
        raise ValueError("Qwen replacement source identity drifted")
    qwen = assignment.get("qwen38") or {}
    if qwen.get("hydration_gate") != {
        "status": "satisfied_by_existing_exact_task_receipts",
        "source_plan_sha256": source_plan["plan_sha256"],
    }:
        raise ValueError("Qwen replacement hydration gate is not satisfied")
    locked = {int(row["replacement_rank"]): row for row in qwen.get("tasks") or []}
    if set(locked) != {51, 52, 53}:
        raise ValueError("Qwen replacement ranks drifted")
    source_tasks = {int(row["rank"]): row for row in source_plan["tasks"]}
    tasks = []
    for source_rank in (51, 52, 53):
        task = copy.deepcopy(source_tasks[source_rank])
        expected = locked[source_rank]
        if (
            task["task"].get("key") != expected["task_key"]
            or task["task"].get("version_id") != expected["task_version_id"]
            or task["task"].get("cyber_contract") != expected["cyber_contract"]
            or task["environment"].get("id") != expected["env_key"]
            or task["environment"].get("version") != expected["env_version"]
            or task["environment"].get("version_id")
            != expected["environment_version_id"]
            or task["environment"].get("data_id") != expected["data_key"]
            or task["environment"].get("data_version") != expected["data_version"]
            or task["environment"].get("runtime_seed_content_sha256")
            != expected["runtime_seed_content_sha256"]
        ):
            raise ValueError("Qwen replacement exact task receipt drifted")
        task["rank"] = len(tasks) + 1
        task["source_rank"] = source_rank
        task.pop("baseline_session_ids", None)
        tasks.append(task)
    attempts = []
    for task in tasks:
        source_rank = int(task["source_rank"])
        key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][:8]
        for attempt in range(1, 5):
            attempts.append(
                {
                    "ordinal": len(attempts) + 1,
                    "rank": int(task["rank"]),
                    "source_rank": source_rank,
                    "attempt": attempt,
                    "run_id": (
                        f"{QWEN_REPLACEMENT_CAMPAIGN}-"
                        f"sr{source_rank:03d}-a{attempt}-{key_digest}"
                    ),
                    "network": (
                        f"qwen38-hosted-replacement-"
                        f"sr{source_rank:03d}-a{attempt}-{key_digest}"
                    ),
                }
            )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": "qwen38_replacements",
        "campaign_id": QWEN_REPLACEMENT_CAMPAIGN,
        "source_job_id": source_plan["source_job_id"],
        "source": {
            "source_plan_sha256": source_plan["plan_sha256"],
            "hosted_plan_sha256": hosted_plan["plan_sha256"],
            "assignment_receipt_sha256": assignment["receipt_sha256"],
        },
        "treatment_block": copy.deepcopy(hosted_plan["treatment_block"]),
        "model": copy.deepcopy(hosted_plan["model"]),
        "harness": copy.deepcopy(hosted_plan["harness"]),
        "authority": copy.deepcopy(hosted_plan["authority"]),
        "task_count": 3,
        "pass_k": 4,
        "total_session_count": 12,
        "credited_sessions": [],
        "new_session_count": 12,
        "fenced_source_ranks": [1, 2, 3],
        "existing_hosted_source_ranks": list(range(4, 51)),
        "execution": {
            **copy.deepcopy(hosted_plan["execution"]),
            "inventory_policy": "plan_identity_plus_authoritative_receipt_v1",
        },
        "tasks": tasks,
        "attempts": attempts,
        "privacy": copy.deepcopy(hosted_plan["privacy"]),
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def build_glm53_hosted_replacement_plan(
    hosted_plan: dict[str, Any],
    supplement: dict[str, Any],
    hydration: dict[str, Any],
) -> dict[str, Any]:
    """Build the separately locked hosted r106 pass@4 treatment block."""
    validate_plan(hosted_plan)
    replacement = supplement.get("replacement") or {}
    if (
        hosted_plan.get("shard_key") != "glm53_remainder3"
        or [int(row["source_rank"]) for row in hosted_plan["tasks"]]
        != list(range(11, 100, 2))
        or supplement.get("schema_version")
        != "fleet-opencode-replacement-selection-supplement-v1"
        or supplement.get("receipt_sha256")
        != digest_without(supplement, "receipt_sha256")
        or replacement.get("replacement_rank") != 106
        or replacement.get("serving_block") != "hosted"
        or replacement.get("model_revision") != hosted_plan["model"]["revision"]
        or hydration.get("schema_version")
        != "fleet-glm53-hosted-replacement-hydration-v1"
        or hydration.get("receipt_sha256")
        != digest_without(hydration, "receipt_sha256")
        or hydration.get("selection_supplement_receipt_sha256")
        != supplement["receipt_sha256"]
        or hydration.get("fleet_team_id") != self_hosted.FLEET_TEAM_ID
        or hydration.get("tasks_hydrated") != 1
        or hydration.get("scores_read") is not False
        or hydration.get("task_content_retained") is not False
        or hydration.get("prompts_or_traces_included") is not False
    ):
        raise ValueError("hosted r106 source or hydration drifted")
    hydrated = (hydration.get("tasks") or [None])[0]
    if not isinstance(hydrated, dict) or int(hydrated.get("replacement_rank") or 0) != 106:
        raise ValueError("hosted r106 hydration task drifted")
    task = {
        "rank": 1,
        "source_rank": 106,
        "task": copy.deepcopy(hydrated["task"]),
        "environment": copy.deepcopy(hydrated["environment"]),
        "verifier": copy.deepcopy(hydrated["verifier"]),
    }
    expected_binding = {
        "task_key": task["task"].get("key"),
        "task_version_id": task["task"].get("version_id"),
        "env_key": task["environment"].get("id"),
        "env_version": task["environment"].get("version"),
        "environment_version_id": task["environment"].get("version_id"),
        "data_key": task["environment"].get("data_id"),
        "data_version": task["environment"].get("data_version"),
    }
    if expected_binding != {field: replacement[field] for field in expected_binding}:
        raise ValueError("hosted r106 exact task binding drifted")
    key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][:8]
    attempts = [
        {
            "ordinal": attempt,
            "rank": 1,
            "source_rank": 106,
            "attempt": attempt,
            "run_id": (
                f"{GLM_HOSTED_REPLACEMENT_CAMPAIGN}-sr106-a{attempt}-{key_digest}"
            ),
            "network": f"glm53-hosted-replacement-sr106-a{attempt}-{key_digest}",
        }
        for attempt in range(1, 5)
    ]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": "glm53_hosted_replacement",
        "campaign_id": GLM_HOSTED_REPLACEMENT_CAMPAIGN,
        "source_job_id": hosted_plan["source_job_id"],
        "source": {
            "hosted_plan_sha256": hosted_plan["plan_sha256"],
            "selection_supplement_receipt_sha256": supplement["receipt_sha256"],
            "hydration_receipt_sha256": hydration["receipt_sha256"],
        },
        "treatment_block": copy.deepcopy(hosted_plan["treatment_block"]),
        "model": copy.deepcopy(hosted_plan["model"]),
        "harness": copy.deepcopy(hosted_plan["harness"]),
        "authority": copy.deepcopy(hosted_plan["authority"]),
        "task_count": 1,
        "pass_k": 4,
        "total_session_count": 4,
        "credited_sessions": [],
        "new_session_count": 4,
        "fenced_source_ranks": [1, 2, 3, 5, 7, 9],
        "existing_hosted_source_ranks": list(range(11, 100, 2)),
        "reserved_source_ranks": [*range(4, 101, 2), *range(101, 106)],
        "execution": {
            **copy.deepcopy(hosted_plan["execution"]),
            "inventory_policy": "conservative_no_same_model_session_for_task_key_v1",
        },
        "tasks": [task],
        "attempts": attempts,
        "privacy": copy.deepcopy(hosted_plan["privacy"]),
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def build_glm53_hosted_reassigned_b_plan(
    dedicated_plan: dict[str, Any],
    hosted_plan: dict[str, Any],
    tombstone: dict[str, Any],
    supplement: dict[str, Any],
    hydration: dict[str, Any],
) -> dict[str, Any]:
    """Build preview-only B reassignment on the exact hosted treatment."""
    validate_plan(dedicated_plan)
    validate_plan(hosted_plan)
    replacement = supplement.get("replacement") or {}
    stopped = tombstone.get("stopped_controller") or {}
    aborted = tombstone.get("source56_aborted_before_execution") or {}
    if (
        dedicated_plan.get("shard_key") != "glm53_dedicated_b"
        or dedicated_plan.get("plan_sha256")
        != "sha256:008386c1bbc6d82229f2afdb85e074a0d5e717853dc7cab5b720ef13271e9cb5"
        or hosted_plan.get("shard_key") != "glm53_remainder3"
        or tombstone.get("receipt_sha256") != digest_without(tombstone, "receipt_sha256")
        or tombstone.get("receipt_sha256")
        != "sha256:7cf5483e07e4053d7807d599b738e0555ade3b728ea20ae2356186592bd7c355"
        or stopped.get("job_absent_after_delete") is not True
        or stopped.get("pod_absent_after_delete") is not True
        or aborted.get("scored_or_terminal_cell") is not False
        or aborted.get("eligible_under_fresh_plan") is not True
        or aborted.get("old_run_ids_reusable") is not False
        or supplement.get("schema_version")
        != "fleet-opencode-replacement-selection-supplement-v2"
        or supplement.get("receipt_sha256")
        != digest_without(supplement, "receipt_sha256")
        or (supplement.get("forced_stop_tombstone") or {}).get("receipt_sha256")
        != tombstone["receipt_sha256"]
        or replacement.get("replacement_rank") != 107
        or replacement.get("serving_block") != "dedicated_b_successor"
        or replacement.get("scored_launch_authorized") is not False
        or hydration.get("schema_version")
        != "fleet-glm53-dedicated-b-replacement-hydration-v1"
        or hydration.get("receipt_sha256")
        != digest_without(hydration, "receipt_sha256")
        or hydration.get("selection_supplement_receipt_sha256")
        != supplement["receipt_sha256"]
        or hydration.get("fleet_team_id") != self_hosted.FLEET_TEAM_ID
        or hydration.get("tasks_hydrated") != 1
    ):
        raise ValueError("hosted B reassignment source evidence drifted")

    source_tasks = {int(row["source_rank"]): row for row in dedicated_plan["tasks"]}
    selected_ranks = [*range(56, 101, 2), 101, 103, 105]
    if set(source_tasks) != {54, *selected_ranks}:
        raise ValueError("dedicated B predecessor partition drifted")
    tasks = []
    for source_rank in selected_ranks:
        task = copy.deepcopy(source_tasks[source_rank])
        task["rank"] = len(tasks) + 1
        task["source_rank"] = source_rank
        tasks.append(task)
    hydrated = (hydration.get("tasks") or [None])[0]
    if not isinstance(hydrated, dict) or int(hydrated.get("replacement_rank") or 0) != 107:
        raise ValueError("hosted B r107 hydration task drifted")
    replacement_task = {
        "rank": len(tasks) + 1,
        "source_rank": 107,
        "task": copy.deepcopy(hydrated["task"]),
        "environment": copy.deepcopy(hydrated["environment"]),
        "verifier": copy.deepcopy(hydrated["verifier"]),
    }
    expected_binding = {
        "task_key": replacement_task["task"].get("key"),
        "task_version_id": replacement_task["task"].get("version_id"),
        "env_key": replacement_task["environment"].get("id"),
        "env_version": replacement_task["environment"].get("version"),
        "environment_version_id": replacement_task["environment"].get("version_id"),
        "data_key": replacement_task["environment"].get("data_id"),
        "data_version": replacement_task["environment"].get("data_version"),
    }
    if expected_binding != {field: replacement[field] for field in expected_binding}:
        raise ValueError("hosted B r107 exact task binding drifted")
    tasks.append(replacement_task)

    attempts = []
    for task in tasks:
        source_rank = int(task["source_rank"])
        key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][:8]
        for attempt in range(1, 5):
            attempts.append(
                {
                    "ordinal": len(attempts) + 1,
                    "rank": int(task["rank"]),
                    "source_rank": source_rank,
                    "attempt": attempt,
                    "run_id": (
                        f"{GLM_HOSTED_REASSIGNED_B_CAMPAIGN}-"
                        f"sr{source_rank:03d}-a{attempt}-{key_digest}"
                    ),
                    "network": (
                        f"glm53-hosted-reassigned-b-"
                        f"sr{source_rank:03d}-a{attempt}-{key_digest}"
                    ),
                }
            )
    treatment = copy.deepcopy(hosted_plan["treatment_block"])
    treatment["serving_block"] = "hosted-reassigned-after-dedicated-preemption"
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": "glm53_hosted_reassigned_b",
        "campaign_id": GLM_HOSTED_REASSIGNED_B_CAMPAIGN,
        "source_job_id": hosted_plan["source_job_id"],
        "source": {
            "dedicated_predecessor_plan_sha256": dedicated_plan["plan_sha256"],
            "hosted_treatment_plan_sha256": hosted_plan["plan_sha256"],
            "forced_stop_tombstone_receipt_sha256": tombstone["receipt_sha256"],
            "selection_supplement_receipt_sha256": supplement["receipt_sha256"],
            "hydration_receipt_sha256": hydration["receipt_sha256"],
        },
        "treatment_block": treatment,
        "model": copy.deepcopy(hosted_plan["model"]),
        "harness": copy.deepcopy(hosted_plan["harness"]),
        "authority": copy.deepcopy(hosted_plan["authority"]),
        "task_count": 27,
        "pass_k": 4,
        "total_session_count": 108,
        "credited_sessions": [],
        "new_session_count": 108,
        "fenced_source_ranks": [1, 2, 3, 5, 7, 9, 54],
        "existing_hosted_source_ranks": [*range(11, 100, 2), 106],
        "dedicated_a_source_ranks": [*range(4, 53, 2), 102, 104],
        "source56_tombstone_attempt_claim_sha256": aborted["attempt_claim_sha256"],
        "execution": {
            **copy.deepcopy(hosted_plan["execution"]),
            "inventory_policy": "conservative_no_same_model_session_for_task_key_v1",
            "launch_authorized": False,
            "required_concurrency_gate": "hosted_glm_concurrency2_health_overlap_v1",
        },
        "tasks": tasks,
        "attempts": attempts,
        "privacy": copy.deepcopy(hosted_plan["privacy"]),
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def _hydrated_replacement_tasks(
    supplement: dict[str, Any], hydration: dict[str, Any], first_rank: int
) -> list[dict[str, Any]]:
    replacement_rows = supplement.get("replacements") or [supplement["replacement"]]
    assigned = {int(row["replacement_rank"]): row for row in replacement_rows}
    hydrated = {
        int(row["replacement_rank"]): row for row in hydration["tasks"]
    }
    if set(assigned) != set(hydrated):
        raise ValueError("HTTP500 replacement hydration ranks drifted")
    tasks = []
    for rank in sorted(assigned):
        source = assigned[rank]
        row = hydrated[rank]
        binding = {
            "task_key": row["task"].get("key"),
            "task_version_id": row["task"].get("version_id"),
            "env_key": row["environment"].get("id"),
            "env_version": row["environment"].get("version"),
            "environment_version_id": row["environment"].get("version_id"),
            "data_key": row["environment"].get("data_id"),
            "data_version": row["environment"].get("data_version"),
        }
        if binding != {field: source[field] for field in binding}:
            raise ValueError("HTTP500 replacement exact binding drifted")
        tasks.append(
            {
                "rank": first_rank + len(tasks),
                "source_rank": rank,
                "task": copy.deepcopy(row["task"]),
                "environment": copy.deepcopy(row["environment"]),
                "verifier": copy.deepcopy(row["verifier"]),
            }
        )
    return tasks


def _fresh_attempts(
    tasks: list[dict[str, Any]], campaign: str, network: str
) -> list[dict[str, Any]]:
    attempts = []
    for task in tasks:
        source_rank = int(task["source_rank"])
        key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][
            :8
        ]
        for attempt in range(1, 5):
            attempts.append(
                {
                    "ordinal": len(attempts) + 1,
                    "rank": int(task["rank"]),
                    "source_rank": source_rank,
                    "attempt": attempt,
                    "run_id": (
                        f"{campaign}-sr{source_rank:03d}-a{attempt}-{key_digest}"
                    ),
                    "network": (
                        f"{network}-sr{source_rank:03d}-a{attempt}-{key_digest}"
                    ),
                }
            )
    return attempts


def build_qwen_http500_successor_plan(
    original_plan: dict[str, Any],
    replacement_plan: dict[str, Any],
    incident: dict[str, Any],
    supplement: dict[str, Any],
    hydration: dict[str, Any],
) -> dict[str, Any]:
    """Build the non-launchable Qwen post-HTTP500 complete-task successor."""
    validate_plan(original_plan)
    validate_plan(replacement_plan)
    if (
        original_plan.get("plan_sha256")
        != "sha256:f367d5148b035e29f57e5a29633609c4dbc8876d94c2795a79ff1ff25dad2ee2"
        or replacement_plan.get("plan_sha256")
        != "sha256:9f879c054e770c120f2d4b77750475f402ba067873fbb93d0e356b062168841d"
        or incident.get("receipt_sha256") != digest_without(incident, "receipt_sha256")
        or incident.get("receipt_sha256")
        != "sha256:043c01e05e9118a080bd642a5da6f7c22c279781bc0c97803330091e55fdb27e"
        or supplement.get("schema_version")
        != "fleet-qwen38-replacement-selection-supplement-v2"
        or supplement.get("receipt_sha256")
        != "sha256:7ad740c7b8a13178a2e090623d3f0c4786d52d29143e740fd3649392acbf5276"
        or supplement.get("receipt_sha256")
        != digest_without(supplement, "receipt_sha256")
        or hydration.get("schema_version")
        != "fleet-qwen38-hosted-replacement-hydration-v2"
        or hydration.get("receipt_sha256") != digest_without(hydration, "receipt_sha256")
        or hydration.get("selection_supplement_receipt_sha256")
        != supplement["receipt_sha256"]
        or hydration.get("tasks_hydrated") != 2
    ):
        raise ValueError("Qwen HTTP500 successor evidence drifted")
    original = [
        copy.deepcopy(row)
        for row in original_plan["tasks"]
        if 6 <= int(row["source_rank"]) <= 50
    ]
    replacements = [
        copy.deepcopy(row)
        for row in replacement_plan["tasks"]
        if int(row["source_rank"]) in {52, 53}
    ]
    tasks = [*original, *replacements]
    for index, row in enumerate(tasks, 1):
        row["rank"] = index
    tasks.extend(_hydrated_replacement_tasks(supplement, hydration, len(tasks) + 1))
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": "qwen38_http500_successor",
        "campaign_id": QWEN_HTTP500_SUCCESSOR_CAMPAIGN,
        "source_job_id": original_plan["source_job_id"],
        "source": {
            "original_plan_sha256": original_plan["plan_sha256"],
            "replacement_plan_sha256": replacement_plan["plan_sha256"],
            "incident_receipt_sha256": incident["receipt_sha256"],
            "selection_supplement_receipt_sha256": supplement["receipt_sha256"],
            "hydration_receipt_sha256": hydration["receipt_sha256"],
        },
        "treatment_block": copy.deepcopy(original_plan["treatment_block"]),
        "model": copy.deepcopy(original_plan["model"]),
        "harness": copy.deepcopy(original_plan["harness"]),
        "authority": copy.deepcopy(original_plan["authority"]),
        "task_count": 49,
        "pass_k": 4,
        "total_session_count": 196,
        "credited_sessions": [],
        "new_session_count": 196,
        "prior_complete_source_ranks": [4],
        "prior_accepted_session_count": 4,
        "primary_estimator_task_count": 50,
        "primary_estimator_cell_count": 200,
        "fenced_source_ranks": [1, 2, 3, 5, 51],
        "execution": {
            **copy.deepcopy(original_plan["execution"]),
            "inventory_policy": "plan_identity_plus_authoritative_receipt_v1",
            "launch_authorized": False,
            "required_priority_class": "fleet-train-high",
            "required_recovery_gate": "fleet_scoring_api_http500_recovery_v1",
            "third_hosted_stream_authorized": False,
        },
        "tasks": tasks,
        "attempts": _fresh_attempts(
            tasks, QWEN_HTTP500_SUCCESSOR_CAMPAIGN, "qwen38-hosted-successor"
        ),
        "privacy": copy.deepcopy(original_plan["privacy"]),
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def build_glm_http500_successor_plan(
    original_plan: dict[str, Any],
    reassigned_b_plan: dict[str, Any],
    incident: dict[str, Any],
    supplement: dict[str, Any],
    hydration: dict[str, Any],
) -> dict[str, Any]:
    """Build the non-launchable 73-task hosted GLM successor."""
    validate_plan(original_plan)
    validate_plan(reassigned_b_plan)
    if (
        original_plan.get("plan_sha256")
        != "sha256:ee31a410a8dcf955e97a3f5ee2d26717e0cdda1d07257e30760d62a25666df8e"
        or reassigned_b_plan.get("plan_sha256")
        != "sha256:729f63116093c40626bb5446bbcea23ea4eb4d3417a441c4dff472ae8f3fe0ed"
        or incident.get("receipt_sha256") != digest_without(incident, "receipt_sha256")
        or incident.get("receipt_sha256")
        != "sha256:043c01e05e9118a080bd642a5da6f7c22c279781bc0c97803330091e55fdb27e"
        or supplement.get("schema_version")
        != "fleet-opencode-replacement-selection-supplement-v3"
        or supplement.get("receipt_sha256")
        != "sha256:a08c2c41f782f9b41e95575a75071e2a3c2fe8916a5db13bc095614dacb92b63"
        or supplement.get("receipt_sha256")
        != digest_without(supplement, "receipt_sha256")
        or hydration.get("schema_version")
        != "fleet-glm53-hosted-replacement-hydration-v2"
        or hydration.get("receipt_sha256") != digest_without(hydration, "receipt_sha256")
        or hydration.get("selection_supplement_receipt_sha256")
        != supplement["receipt_sha256"]
        or hydration.get("tasks_hydrated") != 2
        or original_plan["model"] != reassigned_b_plan["model"]
        or original_plan["harness"] != reassigned_b_plan["harness"]
    ):
        raise ValueError("GLM HTTP500 successor evidence drifted")
    original = [
        copy.deepcopy(row)
        for row in original_plan["tasks"]
        if 13 <= int(row["source_rank"]) <= 99
    ]
    tasks = [*original, *copy.deepcopy(reassigned_b_plan["tasks"])]
    for index, row in enumerate(tasks, 1):
        row["rank"] = index
    tasks.extend(_hydrated_replacement_tasks(supplement, hydration, len(tasks) + 1))
    treatment = copy.deepcopy(original_plan["treatment_block"])
    treatment["serving_block"] = "hosted-successor-after-scoring-outage-v1"
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": "glm53_http500_successor",
        "campaign_id": GLM_HTTP500_SUCCESSOR_CAMPAIGN,
        "source_job_id": original_plan["source_job_id"],
        "source": {
            "original_plan_sha256": original_plan["plan_sha256"],
            "hosted_reassigned_b_plan_sha256": reassigned_b_plan["plan_sha256"],
            "incident_receipt_sha256": incident["receipt_sha256"],
            "selection_supplement_receipt_sha256": supplement["receipt_sha256"],
            "hydration_receipt_sha256": hydration["receipt_sha256"],
        },
        "treatment_block": treatment,
        "model": copy.deepcopy(original_plan["model"]),
        "harness": copy.deepcopy(original_plan["harness"]),
        "authority": copy.deepcopy(original_plan["authority"]),
        "task_count": 73,
        "pass_k": 4,
        "total_session_count": 292,
        "credited_sessions": [],
        "new_session_count": 292,
        "dedicated_a_task_count": 27,
        "primary_estimator_task_count": 100,
        "primary_estimator_cell_count": 400,
        "fenced_source_ranks": [1, 2, 3, 5, 7, 9, 11, 54, 106],
        "dedicated_a_source_ranks": [*range(4, 53, 2), 102, 104],
        "execution": {
            **copy.deepcopy(original_plan["execution"]),
            "inventory_policy": "conservative_no_same_model_session_for_task_key_v1",
            "launch_authorized": False,
            "required_priority_class": "fleet-train-high",
            "required_recovery_gate": "fleet_scoring_api_http500_recovery_v1",
            "third_hosted_stream_authorized": False,
        },
        "tasks": tasks,
        "attempts": _fresh_attempts(
            tasks, GLM_HTTP500_SUCCESSOR_CAMPAIGN, "glm53-hosted-successor"
        ),
        "privacy": copy.deepcopy(original_plan["privacy"]),
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def build_glm_http500_hosted_primary_plan(
    combined_hosted_plan: dict[str, Any],
) -> dict[str, Any]:
    """Split the hosted primary stream from the separately reserved B block."""
    validate_plan(combined_hosted_plan)
    if (
        combined_hosted_plan.get("shard_key") != "glm53_http500_successor"
        or combined_hosted_plan.get("plan_sha256")
        != "sha256:d5f0d14e88f45b339a91a03b263c769a4f0be56fb7b700b353ee04effbcb5822"
    ):
        raise ValueError("GLM combined hosted successor plan drifted")
    selected = {*range(13, 100, 2), 108, 109}
    tasks = [
        copy.deepcopy(row)
        for row in combined_hosted_plan["tasks"]
        if int(row["source_rank"]) in selected
    ]
    for index, row in enumerate(tasks, 1):
        row["rank"] = index
    treatment = copy.deepcopy(combined_hosted_plan["treatment_block"])
    treatment["serving_block"] = "hosted-primary-after-scoring-outage-v1"
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": "glm53_http500_hosted_primary",
        "campaign_id": GLM_HTTP500_HOSTED_PRIMARY_CAMPAIGN,
        "source_job_id": combined_hosted_plan["source_job_id"],
        "source": {
            "combined_hosted_plan_sha256": combined_hosted_plan["plan_sha256"],
            **copy.deepcopy(combined_hosted_plan["source"]),
        },
        "treatment_block": treatment,
        "model": copy.deepcopy(combined_hosted_plan["model"]),
        "harness": copy.deepcopy(combined_hosted_plan["harness"]),
        "authority": copy.deepcopy(combined_hosted_plan["authority"]),
        "task_count": 46,
        "pass_k": 4,
        "total_session_count": 184,
        "credited_sessions": [],
        "new_session_count": 184,
        "dedicated_a_task_count": 27,
        "dedicated_b_task_count": 27,
        "primary_estimator_task_count": 100,
        "primary_estimator_cell_count": 400,
        "fenced_source_ranks": [1, 2, 3, 5, 7, 9, 11, 54, 106],
        "dedicated_a_source_ranks": [*range(4, 53, 2), 102, 104],
        "dedicated_b_source_ranks": [
            *range(56, 101, 2),
            101,
            103,
            105,
            107,
        ],
        "execution": {
            **copy.deepcopy(combined_hosted_plan["execution"]),
            "launch_authorized": False,
            "required_priority_class": "fleet-train-high",
            "required_recovery_gate": "fleet_scoring_api_http500_recovery_v1",
            "third_hosted_stream_authorized": False,
            "dedicated_b_route_authorization_required": True,
        },
        "tasks": tasks,
        "attempts": _fresh_attempts(
            tasks,
            GLM_HTTP500_HOSTED_PRIMARY_CAMPAIGN,
            "glm53-hosted-primary",
        ),
        "privacy": copy.deepcopy(combined_hosted_plan["privacy"]),
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def build_glm_dedicated_b_v5_plan(
    predecessor_plan: dict[str, Any],
    tombstone: dict[str, Any],
    supplement: dict[str, Any],
    hydration: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    """Build the non-launchable dedicated B successor on serving generation v5."""
    validate_plan(predecessor_plan)
    if (
        predecessor_plan.get("shard_key") != "glm53_dedicated_b"
        or predecessor_plan.get("plan_sha256")
        != "sha256:008386c1bbc6d82229f2afdb85e074a0d5e717853dc7cab5b720ef13271e9cb5"
        or tombstone.get("receipt_sha256") != digest_without(tombstone, "receipt_sha256")
        or tombstone.get("receipt_sha256")
        != "sha256:7cf5483e07e4053d7807d599b738e0555ade3b728ea20ae2356186592bd7c355"
        or supplement.get("receipt_sha256")
        != "sha256:731ba0583f43a7cc56f16cbf8c71ab98ca39642ff7091e21763f8561bf9d587a"
        or supplement.get("receipt_sha256")
        != digest_without(supplement, "receipt_sha256")
        or hydration.get("receipt_sha256")
        != "sha256:4397e1a849245ac13535d4afe9682148afcdecd8e525130a59b7147637746b46"
        or hydration.get("receipt_sha256") != digest_without(hydration, "receipt_sha256")
        or hydration.get("selection_supplement_receipt_sha256")
        != supplement["receipt_sha256"]
        or parity.get("receipt_sha256")
        != "sha256:4eba7bbbe3ce13bbce61b13b8a2898278407e518611e25625e3688d4219f722d"
        or parity.get("receipt_sha256") != digest_without(parity, "receipt_sha256")
        or parity.get("classification") != "operational-gate-passed"
        or (parity.get("outcome_integrity") or {}).get("scored_sessions") != 0
        or (parity.get("content_blind_probe") or {}).get("health") is not True
        or (parity.get("runtime") or {}).get("priority_class") != "fleet-train-high"
    ):
        raise ValueError("dedicated B v5 source evidence drifted")
    source_tasks = {
        int(row["source_rank"]): copy.deepcopy(row)
        for row in predecessor_plan["tasks"]
    }
    selected = [*range(56, 101, 2), 101, 103, 105]
    tasks = []
    for source_rank in selected:
        row = source_tasks[source_rank]
        row["rank"] = len(tasks) + 1
        tasks.append(row)
    tasks.extend(_hydrated_replacement_tasks(supplement, hydration, len(tasks) + 1))
    model = copy.deepcopy(predecessor_plan["model"])
    service_name = parity["network"]["service_name"]
    model["endpoint_origin"] = (
        f"http://{service_name}.fleet-train-jobs.svc.cluster.local:8000"
    )
    treatment = {
        "kind": "dedicated_inference_endpoint_v1",
        "replica": "B",
        "serving_generation": "v5",
        "endpoint_origin": model["endpoint_origin"],
        "service_name": service_name,
        "service_uid": parity["network"]["service_uid"],
        "api_run_id": parity["api_run"]["run_id"],
        "ray_job_uid": parity["controllers"]["ray_job_uid"],
        "ray_cluster_uid": parity["controllers"]["ray_cluster_uid"],
        "head_pod_uid": parity["runtime"]["head_pod_uid"],
        "runtime_image_digest": parity["image"]["resolved_image_id"],
        "serving_config_sha256": parity["immutable_config"]["sha256"],
        "forced_stop_tombstone_receipt_sha256": tombstone["receipt_sha256"],
        "selection_supplement_receipt_sha256": supplement["receipt_sha256"],
        "hydration_receipt_sha256": hydration["receipt_sha256"],
        "parity_receipt_sha256": parity["receipt_sha256"],
        "served_id": model["served_id"],
        "model_revision": model["revision"],
        "session_model": model["session_model"],
        "harness": copy.deepcopy(predecessor_plan["harness"]),
        "required_task_tools": predecessor_plan["execution"]["required_task_tools"],
        "required_task_tool_catalog_sha256": predecessor_plan["execution"][
            "required_task_tool_catalog_sha256"
        ],
    }
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": "glm53_dedicated_b_v5",
        "campaign_id": GLM_DEDICATED_B_V5_CAMPAIGN,
        "source_job_id": predecessor_plan["source_job_id"],
        "source": {
            "predecessor_plan_sha256": predecessor_plan["plan_sha256"],
            "forced_stop_tombstone_receipt_sha256": tombstone["receipt_sha256"],
            "selection_supplement_receipt_sha256": supplement["receipt_sha256"],
            "hydration_receipt_sha256": hydration["receipt_sha256"],
            "parity_receipt_sha256": parity["receipt_sha256"],
        },
        "treatment_block": treatment,
        "model": model,
        "harness": copy.deepcopy(predecessor_plan["harness"]),
        "authority": copy.deepcopy(predecessor_plan["authority"]),
        "task_count": 27,
        "pass_k": 4,
        "total_session_count": 108,
        "credited_sessions": [],
        "new_session_count": 108,
        "fenced_source_ranks": [1, 2, 3, 5, 7, 9, 11, 54, 106],
        "hosted_source_ranks": [*range(13, 100, 2), 108, 109],
        "dedicated_a_reserved_source_ranks": [*range(6, 53, 2), 102, 104],
        "dedicated_a_pending_replacement_ranks": [110],
        "execution": {
            **copy.deepcopy(predecessor_plan["execution"]),
            "launch_authorized": False,
            "required_priority_class": "fleet-train-high",
            "required_recovery_gate": "fleet_scoring_api_http500_recovery_v1",
        },
        "tasks": tasks,
        "attempts": _fresh_attempts(
            tasks, GLM_DEDICATED_B_V5_CAMPAIGN, "glm53-dedicated-b-v5"
        ),
        "privacy": copy.deepcopy(predecessor_plan["privacy"]),
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def _validate_source_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") == legacy.PLAN_SCHEMA:
        legacy.validate_plan(plan)
    elif plan.get("schema_version") == legacy.PARALLEL_PLAN_SCHEMA:
        legacy.validate_parallel_plan(plan)
    else:
        raise ValueError("unsupported hosted predecessor plan")


def build_plan(
    source_plan: dict[str, Any], source: dict[str, Any], model_key: str
) -> dict[str, Any]:
    validate_source(source)
    _validate_source_plan(source_plan)
    if model_key not in CAMPAIGNS:
        raise ValueError("unsupported hosted model shard")
    evidence = source["runs"][SOURCE_MODEL_KEYS[model_key]]
    if (
        evidence.get("source_plan_sha256") != source_plan["plan_sha256"]
        or evidence.get("source_terminal") is not True
        or evidence.get("source_active_attempts") != 0
        or len(evidence.get("fenced_noncreditable") or []) != 1
    ):
        raise ValueError("hosted predecessor is not exactly fenced")
    if int(source_plan["task_count"]) != EXPECTED_SOURCE_TASK_COUNTS[model_key]:
        raise ValueError("hosted predecessor task count drifted")
    evidence_excluded = {int(rank) for rank in evidence["excluded_source_ranks"]}
    blocked = evidence["fenced_noncreditable"][0]
    if evidence_excluded != {int(blocked["source_rank"])}:
        raise ValueError("fenced cell must exclude its complete source task")
    deferred = ADDITIONAL_DEFERRED_SOURCE_RANKS[model_key]
    if deferred & evidence_excluded:
        raise ValueError("deferred source tasks overlap fenced source tasks")
    reserved = RESERVED_SOURCE_RANKS[model_key]
    if reserved & (evidence_excluded | deferred):
        raise ValueError("reserved source tasks overlap fenced source tasks")
    excluded = evidence_excluded | deferred | reserved
    if (
        blocked.get("agent_exit_code") != 1
        or blocked.get("session_ingest_completed") is not True
        or blocked.get("cleanup_completed") is not True
        or blocked.get("retry_allowed") is not False
        or blocked.get("credited") is not False
    ):
        raise ValueError("fenced noncreditable outcome policy drifted")

    source_tasks = {int(row["rank"]): row for row in source_plan["tasks"]}
    source_attempts = {row["run_id"]: row for row in source_plan["attempts"]}
    accepted = evidence.get("accepted") or []
    for row in accepted:
        attempt = source_attempts.get(row["run_id"])
        if attempt is None or (int(attempt["rank"]), int(attempt["attempt"])) != (
            int(row["source_rank"]),
            int(row["attempt"]),
        ):
            raise ValueError("accepted predecessor outcome is not plan-bound")

    credit_rows = [evidence["credited_smoke"], *accepted]
    credits_by_source_rank: dict[int, list[dict[str, Any]]] = {}
    for row in credit_rows:
        credits_by_source_rank.setdefault(int(row["source_rank"]), []).append(row)

    tasks: list[dict[str, Any]] = []
    credits: list[dict[str, Any]] = []
    source_rank_to_rank: dict[int, int] = {}
    for source_rank in sorted(source_tasks):
        if source_rank in excluded:
            continue
        rank = len(tasks) + 1
        source_rank_to_rank[source_rank] = rank
        task = copy.deepcopy(source_tasks[source_rank])
        task["rank"] = rank
        task["source_rank"] = source_rank
        task.pop("baseline_session_ids", None)
        tasks.append(task)
        for row in credits_by_source_rank.get(source_rank, []):
            credits.append(
                {
                    "rank": rank,
                    "source_rank": source_rank,
                    "attempt": int(row["attempt"]),
                    "session_id": row["session_id"],
                    "verifier_execution_id": row["verifier_execution_id"],
                    "source_run_id": row.get("run_id", "credited-smoke"),
                }
            )
    if len(tasks) != EXPECTED_INCLUDED_TASK_COUNTS[model_key]:
        raise ValueError("hosted included task count drifted")

    credited_cells = {(int(row["rank"]), int(row["attempt"])) for row in credits}
    attempts: list[dict[str, Any]] = []
    campaign = CAMPAIGNS[model_key]
    for task in tasks:
        rank = int(task["rank"])
        source_rank = int(task["source_rank"])
        key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][:8]
        for attempt_number in range(1, 5):
            if (rank, attempt_number) in credited_cells:
                continue
            attempts.append(
                {
                    "ordinal": len(attempts) + 1,
                    "rank": rank,
                    "source_rank": source_rank,
                    "attempt": attempt_number,
                    "run_id": (
                        f"{campaign}-sr{source_rank:03d}-a{attempt_number}-{key_digest}"
                    ),
                    "network": (
                        f"{model_key}-hosted-"
                        f"{'v7' if model_key == 'glm53_hosted_odd' else 'v5'}-"
                        f"sr{source_rank:03d}-a{attempt_number}-{key_digest}"
                    ),
                }
            )

    blocked_task = source_tasks[int(blocked["source_rank"])]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": campaign,
        "source_job_id": source_plan["source_job_id"],
        "source": {
            "source_plan_sha256": source_plan["plan_sha256"],
            "source_state_receipt_sha256": source["receipt_sha256"],
            "source_job_name": evidence["source_job_name"],
            "source_job_uid": evidence["source_job_uid"],
            "source_pod_uid": evidence["source_pod_uid"],
        },
        "treatment_block": {
            "kind": "hosted_inference_endpoint_v1",
            "endpoint_origin": source_plan["model"]["endpoint_origin"],
            "served_id": source_plan["model"]["served_id"],
            "model_revision": source_plan["model"]["revision"],
            "session_model": source_plan["model"]["session_model"],
            "harness": source_plan["harness"],
            "required_task_tools": source_plan["execution"]["required_task_tools"],
            "required_task_tool_catalog_sha256": source_plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
        },
        "model": source_plan["model"],
        "harness": source_plan["harness"],
        "authority": source_plan["authority"],
        "task_count": len(tasks),
        "pass_k": 4,
        "total_session_count": len(tasks) * 4,
        "credited_sessions": credits,
        "new_session_count": len(attempts),
        "excluded_tasks": [
            {
                "source_rank": int(blocked["source_rank"]),
                "task_key": blocked_task["task"]["key"],
                "task_version_id": blocked_task["task"]["version_id"],
                "reason": "fenced_verifier_backed_ingested_cleaned_opencode_exit_1",
                "run_id": blocked["run_id"],
                "session_id": blocked["session_id"],
                "verifier_execution_id": blocked["verifier_execution_id"],
                "retry_allowed": False,
                "credited": False,
            }
        ],
        "execution": {
            "task_partition": "complete_task_boundary",
            "same_task_max_inflight": 1,
            "retry_policy": "never_repeat_any_verifier_backed_outcome",
            "future_nonzero_exit_policy": "fence_task_and_continue_other_tasks",
            "training_data_eligible": True,
            "required_task_tools": source_plan["execution"]["required_task_tools"],
            "required_task_tool_catalog_sha256": source_plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
            "score_blind_concurrency_schedule": SCHEDULE,
        },
        "tasks": tasks,
        "attempts": attempts,
        "privacy": {
            "scores_included": False,
            "prompts_included": False,
            "transcripts_included": False,
            "credentials_included": False,
        },
    }
    if model_key in {"glm53_clean", "glm53_hosted_odd"}:
        plan["shard_key"] = model_key
        plan["execution"]["inventory_policy"] = (
            "conservative_no_same_model_session_for_task_key_v1"
        )
    for source_rank in sorted(deferred):
        task = source_tasks[source_rank]
        plan["excluded_tasks"].append(
            {
                "source_rank": source_rank,
                "task_key": task["task"]["key"],
                "task_version_id": task["task"]["version_id"],
                "reason": "retained_history_requires_authoritative_metadata_projection",
                "retry_allowed": False,
                "credited": False,
            }
        )
    if reserved:
        plan["reserved_tasks"] = [
            {
                "source_rank": source_rank,
                "task_key": source_tasks[source_rank]["task"]["key"],
                "task_version_id": source_tasks[source_rank]["task"]["version_id"],
                "reserved_treatment_block": "dedicated_inference_endpoint_v1",
            }
            for source_rank in sorted(reserved)
        ]
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported hosted shard plan schema")
    if plan.get("plan_sha256") != digest_without(plan, "plan_sha256"):
        raise ValueError("hosted shard plan digest mismatch")
    campaign_to_shard = {
        value: key for key, value in {**CAMPAIGNS, **REMAINDER_CAMPAIGNS}.items()
    }
    shard_key = plan.get("shard_key") or campaign_to_shard.get(plan.get("campaign_id"))
    task_count = int(plan.get("task_count") or 0)
    if (
        shard_key not in EXPECTED_INCLUDED_TASK_COUNTS
        or task_count != EXPECTED_INCLUDED_TASK_COUNTS[shard_key]
        or plan.get("pass_k") != 4
    ):
        raise ValueError("hosted shard task/pass shape drifted")
    tasks = plan.get("tasks") or []
    if [row.get("rank") for row in tasks] != list(range(1, task_count + 1)):
        raise ValueError("hosted shard task ranks drifted")
    credits = plan.get("credited_sessions") or []
    attempts = plan.get("attempts") or []
    cells = [
        *[(int(row["rank"]), int(row["attempt"])) for row in credits],
        *[(int(row["rank"]), int(row["attempt"])) for row in attempts],
    ]
    expected = {(rank, attempt) for rank in range(1, task_count + 1) for attempt in range(1, 5)}
    if len(cells) != len(set(cells)) or set(cells) != expected:
        raise ValueError("hosted shard Cartesian cells drifted")
    if len(attempts) != int(plan["new_session_count"]):
        raise ValueError("hosted shard new-session count drifted")
    if int(plan["total_session_count"]) != task_count * 4:
        raise ValueError("hosted shard total-session count drifted")
    if [row.get("ordinal") for row in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError("hosted shard attempt ordinals drifted")
    if len({row["run_id"] for row in attempts}) != len(attempts):
        raise ValueError("hosted shard run identities drifted")
    execution = plan.get("execution") or {}
    expected_inventory_policy = (
        "immutable_plan_claim_and_endpoint_uid_v1"
        if shard_key
        in {"glm53_dedicated_a", "glm53_dedicated_b", "glm53_dedicated_b_v5"}
        else "conservative_no_same_model_session_for_task_key_v1"
        if shard_key in {
            "glm53_clean",
            "glm53_hosted_odd",
            "glm53_remainder",
            "glm53_remainder2",
            "glm53_remainder3",
            "glm53_hosted_replacement",
            "glm53_hosted_reassigned_b",
            "glm53_http500_successor",
            "glm53_http500_hosted_primary",
        }
        else (
            "plan_identity_plus_authoritative_receipt_v1"
            if shard_key
            in {
                "qwen38_remainder",
                "qwen38_remainder2",
                "qwen38_replacements",
                "qwen38_http500_successor",
            }
            else None
        )
    )
    if (
        execution.get("task_partition") != "complete_task_boundary"
        or execution.get("same_task_max_inflight") != 1
        or execution.get("score_blind_concurrency_schedule") != SCHEDULE
        or execution.get("required_task_tools") != ["bash", "submit_report"]
        or execution.get("inventory_policy") != expected_inventory_policy
    ):
        raise ValueError("hosted shard execution policy drifted")
    if shard_key == "glm53_dedicated_b_v5":
        treatment = plan.get("treatment_block") or {}
        source = plan.get("source") or {}
        selected = {int(row["source_rank"]) for row in tasks}
        if (
            selected != {*range(56, 101, 2), 101, 103, 105, 107}
            or set(plan.get("fenced_source_ranks") or [])
            != {1, 2, 3, 5, 7, 9, 11, 54, 106}
            or set(plan.get("hosted_source_ranks") or [])
            != {*range(13, 100, 2), 108, 109}
            or set(plan.get("dedicated_a_reserved_source_ranks") or [])
            != {*range(6, 53, 2), 102, 104}
            or plan.get("dedicated_a_pending_replacement_ranks") != [110]
            or treatment.get("kind") != "dedicated_inference_endpoint_v1"
            or treatment.get("replica") != "B"
            or treatment.get("serving_generation") != "v5"
            or treatment.get("model_revision") != plan["model"].get("revision")
            or treatment.get("endpoint_origin") != plan["model"].get("endpoint_origin")
            or treatment.get("forced_stop_tombstone_receipt_sha256")
            != source.get("forced_stop_tombstone_receipt_sha256")
            or treatment.get("selection_supplement_receipt_sha256")
            != source.get("selection_supplement_receipt_sha256")
            or treatment.get("hydration_receipt_sha256")
            != source.get("hydration_receipt_sha256")
            or treatment.get("parity_receipt_sha256")
            != source.get("parity_receipt_sha256")
            or not treatment.get("service_uid")
            or not treatment.get("ray_job_uid")
            or not treatment.get("ray_cluster_uid")
            or not treatment.get("head_pod_uid")
            or execution.get("launch_authorized") is not False
            or execution.get("required_priority_class") != "fleet-train-high"
            or execution.get("required_recovery_gate")
            != "fleet_scoring_api_http500_recovery_v1"
        ):
            raise ValueError("dedicated B v5 partition or binding drifted")
        return

    if shard_key in {"glm53_dedicated_a", "glm53_dedicated_b"}:
        treatment = plan.get("treatment_block") or {}
        replica = "A" if shard_key.endswith("_a") else "B"
        expected_selected = (
            [*range(4, 53, 2), 102, 104]
            if replica == "A"
            else [*range(54, 101, 2), 101, 103, 105]
        )
        expected_reserved = (
            [*range(54, 101, 2), 101, 103, 105]
            if replica == "A"
            else [*range(4, 53, 2), 102, 104]
        )
        observed_selected = [int(row["source_rank"]) for row in tasks]
        if (
            observed_selected != expected_selected
            or plan.get("fenced_source_ranks") != [1, 2, 3, 5, 7]
            or plan.get("hosted_source_ranks") != list(range(9, 100, 2))
            or plan.get("reserved_source_ranks")
            != expected_reserved
            or treatment.get("kind") != "dedicated_inference_endpoint_v1"
            or treatment.get("replica") != replica
            or not treatment.get("service_uid")
            or not treatment.get("ray_cluster_uid")
            or not treatment.get("head_pod_uid")
            or not str(treatment.get("runtime_image_digest") or "").startswith(
                "ghcr.io/fleet-ai/cyber-post-train-glm53-runtime@sha256:"
            )
            or treatment.get("assignment_receipt_sha256")
            != plan["source"].get("assignment_receipt_sha256")
            or treatment.get("hydration_receipt_sha256")
            != plan["source"].get("hydration_receipt_sha256")
            or treatment.get("parity_receipt_sha256")
            != plan["source"].get("canary_evidence_sha256")
            or treatment.get("endpoint_origin") != plan["model"].get("endpoint_origin")
            or treatment.get("model_revision") != plan["model"].get("revision")
            or set(expected_selected)
            & (
                set(plan["fenced_source_ranks"])
                | set(plan["hosted_source_ranks"])
                | set(plan["reserved_source_ranks"])
            )
        ):
            raise ValueError("dedicated replica partition or binding drifted")
        return

    if shard_key == "qwen38_replacements":
        if (
            [int(row["source_rank"]) for row in tasks] != [51, 52, 53]
            or plan.get("fenced_source_ranks") != [1, 2, 3]
            or plan.get("existing_hosted_source_ranks") != list(range(4, 51))
            or (plan.get("treatment_block") or {}).get("kind")
            != "hosted_inference_endpoint_v1"
            or (plan.get("treatment_block") or {}).get("model_revision")
            != plan["model"].get("revision")
        ):
            raise ValueError("Qwen replacement partition or binding drifted")
        return

    if shard_key == "glm53_hosted_replacement":
        treatment = plan.get("treatment_block") or {}
        selected = {int(row["source_rank"]) for row in tasks}
        fenced = set(plan.get("fenced_source_ranks") or [])
        hosted = set(plan.get("existing_hosted_source_ranks") or [])
        reserved = set(plan.get("reserved_source_ranks") or [])
        if (
            selected != {106}
            or fenced != {1, 2, 3, 5, 7, 9}
            or hosted != set(range(11, 100, 2))
            or reserved != {*range(4, 101, 2), *range(101, 106)}
            or selected & (fenced | hosted | reserved)
            or len(selected | fenced | hosted | reserved) != 106
            or treatment.get("kind") != "hosted_inference_endpoint_v1"
            or treatment.get("model_revision") != plan["model"].get("revision")
            or (plan.get("source") or {}).get("hydration_receipt_sha256") is None
            or (plan.get("source") or {}).get("selection_supplement_receipt_sha256")
            is None
        ):
            raise ValueError("GLM hosted replacement partition or binding drifted")
        return

    if shard_key == "glm53_hosted_reassigned_b":
        treatment = plan.get("treatment_block") or {}
        source = plan.get("source") or {}
        selected = {int(row["source_rank"]) for row in tasks}
        fenced = set(plan.get("fenced_source_ranks") or [])
        hosted = set(plan.get("existing_hosted_source_ranks") or [])
        dedicated_a = set(plan.get("dedicated_a_source_ranks") or [])
        expected_selected = {*range(56, 101, 2), 101, 103, 105, 107}
        expected_fenced = {1, 2, 3, 5, 7, 9, 54}
        expected_hosted = {*range(11, 100, 2), 106}
        expected_dedicated_a = {*range(4, 53, 2), 102, 104}
        if (
            selected != expected_selected
            or fenced != expected_fenced
            or hosted != expected_hosted
            or dedicated_a != expected_dedicated_a
            or selected & (fenced | hosted | dedicated_a)
            or len(selected | fenced | hosted | dedicated_a) != 107
            or treatment.get("kind") != "hosted_inference_endpoint_v1"
            or treatment.get("serving_block")
            != "hosted-reassigned-after-dedicated-preemption"
            or treatment.get("model_revision") != plan["model"].get("revision")
            or source.get("forced_stop_tombstone_receipt_sha256") is None
            or source.get("selection_supplement_receipt_sha256") is None
            or source.get("hydration_receipt_sha256") is None
            or plan.get("source56_tombstone_attempt_claim_sha256") is None
            or execution.get("launch_authorized") is not False
            or execution.get("required_concurrency_gate")
            != "hosted_glm_concurrency2_health_overlap_v1"
        ):
            raise ValueError("GLM hosted B reassignment partition or binding drifted")
        return

    if shard_key == "qwen38_http500_successor":
        treatment = plan.get("treatment_block") or {}
        selected = {int(row["source_rank"]) for row in tasks}
        expected_selected = {*range(6, 51), 52, 53, 54, 55}
        if (
            selected != expected_selected
            or plan.get("prior_complete_source_ranks") != [4]
            or plan.get("prior_accepted_session_count") != 4
            or plan.get("primary_estimator_task_count") != 50
            or plan.get("primary_estimator_cell_count") != 200
            or set(plan.get("fenced_source_ranks") or []) != {1, 2, 3, 5, 51}
            or treatment.get("kind") != "hosted_inference_endpoint_v1"
            or treatment.get("model_revision") != plan["model"].get("revision")
            or execution.get("launch_authorized") is not False
            or execution.get("required_priority_class") != "fleet-train-high"
            or execution.get("required_recovery_gate")
            != "fleet_scoring_api_http500_recovery_v1"
            or execution.get("third_hosted_stream_authorized") is not False
        ):
            raise ValueError("Qwen HTTP500 successor partition or binding drifted")
        return

    if shard_key == "glm53_http500_successor":
        treatment = plan.get("treatment_block") or {}
        selected = {int(row["source_rank"]) for row in tasks}
        dedicated_a = set(plan.get("dedicated_a_source_ranks") or [])
        expected_selected = {
            *range(13, 100, 2),
            *range(56, 101, 2),
            101,
            103,
            105,
            107,
            108,
            109,
        }
        if (
            selected != expected_selected
            or dedicated_a != {*range(4, 53, 2), 102, 104}
            or selected & dedicated_a
            or len(selected | dedicated_a) != 100
            or plan.get("dedicated_a_task_count") != 27
            or plan.get("primary_estimator_task_count") != 100
            or plan.get("primary_estimator_cell_count") != 400
            or set(plan.get("fenced_source_ranks") or [])
            != {1, 2, 3, 5, 7, 9, 11, 54, 106}
            or treatment.get("kind") != "hosted_inference_endpoint_v1"
            or treatment.get("serving_block")
            != "hosted-successor-after-scoring-outage-v1"
            or treatment.get("model_revision") != plan["model"].get("revision")
            or execution.get("launch_authorized") is not False
            or execution.get("required_priority_class") != "fleet-train-high"
            or execution.get("required_recovery_gate")
            != "fleet_scoring_api_http500_recovery_v1"
            or execution.get("third_hosted_stream_authorized") is not False
        ):
            raise ValueError("GLM HTTP500 successor partition or binding drifted")
        return

    if shard_key == "glm53_http500_hosted_primary":
        treatment = plan.get("treatment_block") or {}
        selected = {int(row["source_rank"]) for row in tasks}
        dedicated_a = set(plan.get("dedicated_a_source_ranks") or [])
        dedicated_b = set(plan.get("dedicated_b_source_ranks") or [])
        if (
            selected != {*range(13, 100, 2), 108, 109}
            or dedicated_a != {*range(4, 53, 2), 102, 104}
            or dedicated_b != {*range(56, 101, 2), 101, 103, 105, 107}
            or selected & (dedicated_a | dedicated_b)
            or dedicated_a & dedicated_b
            or len(selected | dedicated_a | dedicated_b) != 100
            or plan.get("dedicated_a_task_count") != 27
            or plan.get("dedicated_b_task_count") != 27
            or plan.get("primary_estimator_task_count") != 100
            or plan.get("primary_estimator_cell_count") != 400
            or treatment.get("kind") != "hosted_inference_endpoint_v1"
            or treatment.get("serving_block")
            != "hosted-primary-after-scoring-outage-v1"
            or treatment.get("model_revision") != plan["model"].get("revision")
            or execution.get("launch_authorized") is not False
            or execution.get("required_priority_class") != "fleet-train-high"
            or execution.get("required_recovery_gate")
            != "fleet_scoring_api_http500_recovery_v1"
            or execution.get("third_hosted_stream_authorized") is not False
            or execution.get("dedicated_b_route_authorization_required") is not True
        ):
            raise ValueError("GLM hosted primary partition or binding drifted")
        return

    excluded = plan.get("excluded_tasks") or []
    expected_excluded = (
        2
        if shard_key in {"glm53_clean", "glm53_hosted_odd", "glm53_remainder"}
        else 1
    )
    if len(excluded) != expected_excluded or any(
        row.get("retry_allowed") is not False for row in excluded
    ):
        raise ValueError("hosted shard fenced-task policy drifted")
    included_keys = {row["task"]["key"] for row in tasks}
    if any(row["task_key"] in included_keys for row in excluded):
        raise ValueError("fenced task leaked into hosted shard")
    reserved = plan.get("reserved_tasks") or []
    expected_reserved = (
        49
        if shard_key
        in {
            "glm53_hosted_odd",
            "glm53_remainder",
            "glm53_remainder2",
            "glm53_remainder3",
        }
        else 0
    )
    if (
        len(reserved) != expected_reserved
        or any(row["task_key"] in included_keys for row in reserved)
        or any(
            row.get("reserved_treatment_block") != "dedicated_inference_endpoint_v1"
            for row in reserved
        )
    ):
        raise ValueError("hosted shard reserved-task policy drifted")


def _client(key: str) -> httpx.Client:
    return httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    )


def _exact_treatment_sessions(
    client: httpx.Client, plan: dict[str, Any], task_key: str
) -> list[dict[str, Any]]:
    model = self_hosted.persisted_session_model_identity(plan)
    harness = f"opencode-{plan['harness']['version']}"
    tool_digest = plan["execution"]["required_task_tool_catalog_sha256"]
    exact = []
    for row in self_hosted._task_sessions(client, task_key):
        metadata = row.get("metadata") or {}
        if (
            row.get("model") == model
            and metadata.get("self_hosted_harness") == harness
            and metadata.get("tool_catalog_sha256") == tool_digest
        ):
            exact.append(row)
    return exact


def _credits_for_rank(plan: dict[str, Any], rank: int) -> list[dict[str, Any]]:
    return [row for row in plan["credited_sessions"] if int(row["rank"]) == rank]


def _accepted(plan: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    paths = list((root / "attempts").glob("*/ACCEPTED.json"))
    if paths:
        persisted_plan = load_object(root / "PLAN.json")
        if persisted_plan.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError("hosted accepted root is not plan-bound")
    plan_attempts = {row["run_id"]: row for row in plan["attempts"]}
    for path in paths:
        row = load_object(path)
        if row.get("receipt_sha256") != digest_without(row, "receipt_sha256"):
            raise RuntimeError("hosted accepted receipt digest mismatch")
        claim = load_object(root / "claims" / f"{row['run_id']}.json")
        planned = plan_attempts.get(row["run_id"])
        if (
            planned is None
            or claim.get("claim_sha256") != digest_without(claim, "claim_sha256")
            or claim.get("plan_sha256") != plan["plan_sha256"]
            or claim.get("run_id") != row["run_id"]
            or claim.get("claim_sha256") != row.get("claim_sha256")
            or claim.get("config_sha256") != row.get("config_sha256")
            or int(claim.get("rank") or 0) != int(planned["rank"])
            or int(claim.get("attempt") or 0) != int(planned["attempt"])
        ):
            raise RuntimeError("hosted accepted receipt is not claim-bound")
        if row["run_id"] in rows:
            raise RuntimeError("hosted accepted run identity duplicated")
        rows[row["run_id"]] = row
    return rows


def _allowed_sessions(plan: dict[str, Any], root: Path, rank: int) -> set[str]:
    allowed = {row["session_id"] for row in _credits_for_rank(plan, rank)}
    attempt_rank = {row["run_id"]: int(row["rank"]) for row in plan["attempts"]}
    for run_id, row in _accepted(plan, root).items():
        if attempt_rank.get(run_id) == rank:
            allowed.add(row["session_id"])
    return allowed


def _validate_inventory_for_task(
    plan: dict[str, Any], root: Path, task: dict[str, Any], key: str
) -> int:
    with _client(key) as client:
        all_rows = self_hosted._task_sessions(client, task["task"]["key"])
    rank = int(task["rank"])
    planned_run_ids = {
        row["run_id"] for row in plan["attempts"] if int(row["rank"]) == rank
    }
    accepted_by_run = {
        run_id: row
        for run_id, row in _accepted(plan, root).items()
        if run_id in planned_run_ids
    }
    for row in all_rows:
        run_id = (row.get("metadata") or {}).get("run_id")
        accepted = accepted_by_run.get(run_id)
        if run_id in planned_run_ids and (
            accepted is None or row.get("session_id") != accepted["session_id"]
        ):
            raise RuntimeError("hosted current-plan API run identity already exists")
    policy = plan["execution"].get(
        "inventory_policy", "exact_hosted_treatment_metadata_v1"
    )
    allowed = _allowed_sessions(plan, root, rank)
    if policy in {
        "exact_hosted_treatment_metadata_v1",
        "plan_identity_plus_authoritative_receipt_v1",
    }:
        model = self_hosted.persisted_session_model_identity(plan)
        harness = f"opencode-{plan['harness']['version']}"
        tool_digest = plan["execution"]["required_task_tool_catalog_sha256"]
        rows = [
            row
            for row in all_rows
            if row.get("model") == model
            and (row.get("metadata") or {}).get("self_hosted_harness") == harness
            and (row.get("metadata") or {}).get("tool_catalog_sha256") == tool_digest
        ]
        if policy == "plan_identity_plus_authoritative_receipt_v1":
            exact_ids = {row.get("session_id") for row in rows}
            rows.extend(
                row
                for row in all_rows
                if row.get("session_id") in allowed
                and row.get("session_id") not in exact_ids
            )
    elif policy == "immutable_plan_claim_and_endpoint_uid_v1":
        # The public session projection omits endpoint UID/treatment metadata.
        # For a canary-bound dedicated endpoint, only a local, digest-valid,
        # plan/claim-bound receipt can authorize an exact cell. Unbound public
        # rows remain visible to the run-id collision check above but cannot be
        # classified as this treatment.
        rows = [row for row in all_rows if row.get("session_id") in allowed]
    elif policy == "conservative_no_same_model_session_for_task_key_v1":
        model = self_hosted.persisted_session_model_identity(plan)
        rows = [row for row in all_rows if row.get("model") == model]
    else:
        raise RuntimeError("hosted inventory policy drifted")
    observed = {row.get("session_id") for row in rows if isinstance(row.get("session_id"), str)}
    if observed != allowed:
        raise RuntimeError("hosted exact-treatment session inventory drifted")
    by_id = {row.get("session_id"): row for row in rows}
    credits = [*_credits_for_rank(plan, rank)]
    local_accepted = _accepted(plan, root)
    receipts = [*credits]
    attempt_rank = {row["run_id"]: int(row["rank"]) for row in plan["attempts"]}
    receipts.extend(
        row
        for run_id, row in local_accepted.items()
        if attempt_rank.get(run_id) == rank
    )
    for receipt in receipts:
        row = by_id.get(receipt["session_id"])
        verifier = (row or {}).get("verifier_execution") or {}
        local_receipt = receipt.get("run_id") in local_accepted
        if (
            row is None
            or row.get("status") != "completed"
            or verifier.get("id") != receipt["verifier_execution_id"]
            or (
                not local_receipt
                and (
                    row.get("eval_task_version_id") or row.get("task_version_id")
                )
                != task["task"]["version_id"]
            )
        ):
            raise RuntimeError("hosted credited session is not authoritative")
    return len(rows)


def _attempt_config(
    plan: dict[str, Any], task: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    config = {
        "schema_version": "fleet-hosted-opencode-task-boundary-attempt-v1",
        "run_id": item["run_id"],
        "campaign_id": plan["campaign_id"],
        "source_job_id": plan["source_job_id"],
        "task": task["task"],
        "environment": task["environment"],
        "verifier": task["verifier"],
        "authority": plan["authority"],
        "model": plan["model"],
        "harness": plan["harness"],
        "execution": {
            "pass_k": 1,
            "planned_full_pass_k": 4,
            "max_concurrent": 1,
            "network": item["network"],
            "training_data_eligible": plan["execution"]["training_data_eligible"],
            "required_task_tools": plan["execution"]["required_task_tools"],
            "required_task_tool_catalog_sha256": plan["execution"][
                "required_task_tool_catalog_sha256"
            ],
        },
    }
    config["config_sha256"] = digest_without(config, "config_sha256")
    return config


def _classify_result(
    out_dir: Path, config: dict[str, Any], item: dict[str, Any], claim_sha256: str
) -> dict[str, Any]:
    result = load_object(out_dir / "result.json")
    ingest = load_object(out_dir / "session-ingest.json")
    cleanup = load_object(out_dir / "cleanup.json")
    verifier_id = result.get("verifier_execution_id")
    try:
        uuid.UUID(str(verifier_id))
        uuid.UUID(str(result.get("session_id")))
    except ValueError as exc:
        raise RuntimeError("hosted attempt lacks authoritative UUID evidence") from exc
    operationally_terminal = all(
        (
            result.get("run_id") == config["run_id"],
            result.get("agent_termination") == "completed",
            result.get("session_ingest_status") == "completed",
            ingest.get("status") == "completed",
            cleanup == {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            },
        )
    )
    if not operationally_terminal:
        raise RuntimeError("hosted attempt is infrastructure-incomplete")
    accepted = result.get("agent_exit_code") == 0
    row = {
        "schema_version": (
            "fleet-hosted-opencode-attempt-accepted-v1"
            if accepted
            else "fleet-hosted-opencode-attempt-noncreditable-v1"
        ),
        "accepted": accepted,
        "credited": accepted,
        "retry_allowed": False,
        "run_id": config["run_id"],
        "rank": int(item["rank"]),
        "source_rank": int(item["source_rank"]),
        "attempt": int(item["attempt"]),
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "session_id": result["session_id"],
        "verifier_execution_id": verifier_id,
        "agent_exit_code": result.get("agent_exit_code"),
        "config_sha256": config["config_sha256"],
        "claim_sha256": claim_sha256,
        "cleanup_completed": True,
        "session_ingest_completed": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    row["receipt_sha256"] = digest_without(row, "receipt_sha256")
    name = "ACCEPTED.json" if accepted else "NONCREDITABLE.json"
    self_hosted.write_json_once(out_dir / name, row)
    return row


def _run_task(
    plan: dict[str, Any],
    task: dict[str, Any],
    items: list[dict[str, Any]],
    root: Path,
    proxy: Path,
    key: str,
) -> dict[str, Any]:
    rank = int(task["rank"])
    task_claim = {
        "schema_version": "fleet-hosted-opencode-task-claim-v1",
        "plan_sha256": plan["plan_sha256"],
        "rank": rank,
        "source_rank": int(task["source_rank"]),
        "task_key": task["task"]["key"],
        "task_version_id": task["task"]["version_id"],
        "run_ids": [item["run_id"] for item in items],
    }
    task_claim["claim_sha256"] = digest_without(task_claim, "claim_sha256")
    self_hosted.write_json_once(root / "task-claims" / f"rank-{rank:03d}.json", task_claim)
    _emit(
        "task_claimed",
        plan_sha256=plan["plan_sha256"],
        rank=rank,
        source_rank=int(task["source_rank"]),
        attempts=[int(item["attempt"]) for item in items],
        claim_sha256=task_claim["claim_sha256"],
    )
    accepted_count = 0
    for item in items:
        _validate_inventory_for_task(plan, root, task, key)
        config = _attempt_config(plan, task, item)
        claim = {
            "schema_version": "fleet-hosted-opencode-attempt-claim-v1",
            "plan_sha256": plan["plan_sha256"],
            "task_claim_sha256": task_claim["claim_sha256"],
            "run_id": item["run_id"],
            "rank": rank,
            "source_rank": int(item["source_rank"]),
            "attempt": int(item["attempt"]),
            "network": item["network"],
            "config_sha256": config["config_sha256"],
        }
        claim["claim_sha256"] = digest_without(claim, "claim_sha256")
        self_hosted.write_json_once(root / "claims" / f"{item['run_id']}.json", claim)
        out_dir = root / "attempts" / item["run_id"]
        self_hosted.run(config, out_dir, proxy)
        outcome = _classify_result(out_dir, config, item, claim["claim_sha256"])
        if not outcome["accepted"]:
            task_receipt = {
                "schema_version": "fleet-hosted-opencode-task-fenced-v1",
                "plan_sha256": plan["plan_sha256"],
                "rank": rank,
                "source_rank": int(task["source_rank"]),
                "accepted_new_outcomes": accepted_count,
                "noncreditable_receipt_sha256": outcome["receipt_sha256"],
                "remaining_cells_not_started": [
                    next_item["attempt"]
                    for next_item in items
                    if int(next_item["attempt"]) > int(item["attempt"])
                ],
                "retry_allowed": False,
                "scores_included": False,
            }
            task_receipt["receipt_sha256"] = digest_without(task_receipt, "receipt_sha256")
            self_hosted.write_json_once(
                root / "task-results" / f"rank-{rank:03d}.json", task_receipt
            )
            _emit(
                "task_fenced_noncreditable",
                rank=rank,
                source_rank=int(task["source_rank"]),
                attempt=int(item["attempt"]),
                receipt_sha256=outcome["receipt_sha256"],
            )
            return {"complete": False, "accepted": accepted_count}
        accepted_count += 1
        _emit(
            "attempt_accepted",
            rank=rank,
            source_rank=int(item["source_rank"]),
            attempt=int(item["attempt"]),
            receipt_sha256=outcome["receipt_sha256"],
        )
    task_receipt = {
        "schema_version": "fleet-hosted-opencode-task-pass4-accepted-v1",
        "plan_sha256": plan["plan_sha256"],
        "rank": rank,
        "source_rank": int(task["source_rank"]),
        "credited_outcomes": len(_credits_for_rank(plan, rank)),
        "accepted_new_outcomes": accepted_count,
        "total_outcomes": len(_credits_for_rank(plan, rank)) + accepted_count,
        "scores_included": False,
    }
    if task_receipt["total_outcomes"] != 4:
        raise RuntimeError("hosted task did not reach exact pass@4")
    task_receipt["receipt_sha256"] = digest_without(task_receipt, "receipt_sha256")
    self_hosted.write_json_once(root / "task-results" / f"rank-{rank:03d}.json", task_receipt)
    _emit(
        "task_pass4_complete",
        rank=rank,
        source_rank=int(task["source_rank"]),
        receipt_sha256=task_receipt["receipt_sha256"],
    )
    return {"complete": True, "accepted": accepted_count}


def _worker_cap(accepted: int) -> int:
    cap = 1
    for stage in SCHEDULE:
        if accepted >= int(stage["accepted_outcomes_at_least"]):
            cap = int(stage["workers"])
    return cap


def _validate_plan_identity_absence(plan: dict[str, Any], root: Path) -> int:
    """Prove no current-plan run identity or claim exists anywhere on shared storage."""
    if root.exists():
        raise RuntimeError("hosted shard output root already exists")
    parent = root.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("hosted shared job root is not an authoritative directory")
    planned = {row["run_id"] for row in plan["attempts"]}
    roots_reconciled = 0
    for job_root in parent.iterdir():
        if job_root.is_symlink():
            raise RuntimeError("hosted shared job root contains a symlink")
        if not job_root.is_dir():
            continue
        roots_reconciled += 1
        for run_id in planned:
            if (job_root / "claims" / f"{run_id}.json").exists() or (
                job_root / "attempts" / run_id
            ).exists():
                raise RuntimeError("hosted current-plan run identity already exists")
        for claim_path in (job_root / "task-claims").glob("*.json"):
            claim = load_object(claim_path)
            run_ids = claim.get("run_ids") or []
            if not isinstance(run_ids, list) or any(run_id in planned for run_id in run_ids):
                raise RuntimeError("hosted current-plan task claim already exists")
    return roots_reconciled


def preflight_plan(
    plan: dict[str, Any],
    root: Path,
    key: str,
    release: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_plan(plan)
    validate_dedicated_scoring_release(plan, release)
    validate_hosted_replacement_scoring_release(plan, release)
    validate_qwen_http500_scoring_release(plan, release)
    validate_glm_http500_scoring_release(plan, release)
    validate_glm_dedicated_b_v5_scoring_release(plan, release)
    roots_reconciled = _validate_plan_identity_absence(plan, root)
    with _client(key) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
    if (
        account.get("team_name") != "fleet"
        or account.get("team_id") != self_hosted.FLEET_TEAM_ID
    ):
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
    # A temporary empty root lets the same exact-inventory function prove that
    # no plan-owned new receipt is being silently credited before launch.
    scratch = Path("/tmp/hosted-preflight-empty")
    if scratch.exists():
        raise RuntimeError("hosted preflight scratch unexpectedly exists")
    (scratch / "attempts").mkdir(parents=True)
    inventory_counts: dict[int, int] = {}
    lock = threading.Lock()

    def inspect(task: dict[str, Any]) -> None:
        count = _validate_inventory_for_task(plan, scratch, task, key)
        with lock:
            inventory_counts[int(task["rank"])] = count

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(inspect, plan["tasks"]))
    finally:
        (scratch / "attempts").rmdir()
        scratch.rmdir()
    receipt = {
        "schema_version": "fleet-hosted-opencode-preflight-v1",
        "plan_sha256": plan["plan_sha256"],
        "fleet_team_id": self_hosted.FLEET_TEAM_ID,
        "tasks_reconciled": len(inventory_counts),
        "exact_treatment_sessions_reconciled": sum(inventory_counts.values()),
        "active_source_attempts": 0,
        "output_root_absent": True,
        "sfs_job_roots_reconciled": roots_reconciled,
        "current_plan_run_and_claim_identities_absent": True,
        "public_list_treatment_metadata_authoritative": False,
        "empty_exact_treatment_set_used_as_sole_proof": False,
        "scores_read": False,
        "prompts_or_traces_read": False,
    }
    receipt["receipt_sha256"] = digest_without(receipt, "receipt_sha256")
    return receipt


def run_plan(
    plan: dict[str, Any],
    root: Path,
    proxy: Path,
    release: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_plan(plan)
    validate_dedicated_scoring_release(plan, release)
    validate_hosted_replacement_scoring_release(plan, release)
    validate_qwen_http500_scoring_release(plan, release)
    validate_glm_http500_scoring_release(plan, release)
    validate_glm_dedicated_b_v5_scoring_release(plan, release)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    roots_reconciled = _validate_plan_identity_absence(plan, root)
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    for name in ("attempts", "claims", "task-claims", "task-results", "ramps"):
        (root / name).mkdir(mode=0o700)
    self_hosted.write_json_once(root / "PLAN.json", plan)
    with _client(key) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") != self_hosted.FLEET_TEAM_ID:
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")

    inventory_counts: dict[int, int] = {}
    lock = threading.Lock()
    def inspect(task: dict[str, Any]) -> None:
        count = _validate_inventory_for_task(plan, root, task, key)
        with lock:
            inventory_counts[int(task["rank"])] = count
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(inspect, plan["tasks"]))
    preflight = {
        "schema_version": "fleet-hosted-opencode-preflight-v1",
        "plan_sha256": plan["plan_sha256"],
        "fleet_team_id": self_hosted.FLEET_TEAM_ID,
        "tasks_reconciled": len(inventory_counts),
        "exact_treatment_sessions_reconciled": sum(inventory_counts.values()),
        "active_source_attempts": 0,
        "sfs_job_roots_reconciled": roots_reconciled,
        "current_plan_run_and_claim_identities_absent": True,
        "public_list_treatment_metadata_authoritative": False,
        "empty_exact_treatment_set_used_as_sole_proof": False,
        "scores_read": False,
        "prompts_or_traces_read": False,
    }
    preflight["receipt_sha256"] = digest_without(preflight, "receipt_sha256")
    self_hosted.write_json_once(root / "PREFLIGHT.json", preflight)

    tasks = {int(row["rank"]): row for row in plan["tasks"]}
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for rank in sorted(tasks):
        items = sorted(
            (row for row in plan["attempts"] if int(row["rank"]) == rank),
            key=lambda row: int(row["attempt"]),
        )
        groups.append((tasks[rank], items))

    pending = iter(groups)
    active: dict[Future[dict[str, Any]], int] = {}
    accepted_count = 0
    complete_tasks = 0
    fenced_tasks = 0
    last_cap = 1
    infrastructure_failure: Exception | None = None
    with ThreadPoolExecutor(max_workers=max(stage["workers"] for stage in SCHEDULE)) as pool:
        def submit_until_cap() -> None:
            nonlocal last_cap
            cap = _worker_cap(accepted_count)
            if cap != last_cap:
                ramp = {
                    "schema_version": "fleet-hosted-opencode-score-blind-ramp-v1",
                    "plan_sha256": plan["plan_sha256"],
                    "from_workers": last_cap,
                    "to_workers": cap,
                    "accepted_operational_outcomes": accepted_count,
                    "scores_read": False,
                    "decision_inputs": [
                        "accepted",
                        "session_ingest_completed",
                        "verifier_uuid",
                        "cleanup_completed",
                    ],
                }
                ramp["receipt_sha256"] = digest_without(ramp, "receipt_sha256")
                self_hosted.write_json_once(root / "ramps" / f"workers-{cap}.json", ramp)
                last_cap = cap
            while len(active) < cap and infrastructure_failure is None:
                try:
                    task, items = next(pending)
                except StopIteration:
                    return
                future = pool.submit(_run_task, plan, task, items, root, proxy, key)
                active[future] = int(task["rank"])

        submit_until_cap()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                try:
                    result = future.result()
                    accepted_count += int(result["accepted"])
                    if result["complete"]:
                        complete_tasks += 1
                    else:
                        fenced_tasks += 1
                except Exception as exc:  # noqa: BLE001
                    infrastructure_failure = exc
            submit_until_cap()
    if infrastructure_failure is not None:
        raise infrastructure_failure

    terminal = {
        "schema_version": "fleet-hosted-opencode-collection-terminal-v1",
        "plan_sha256": plan["plan_sha256"],
        "planned_tasks": int(plan["task_count"]),
        "pass4_complete_tasks": complete_tasks,
        "fenced_noncreditable_tasks": fenced_tasks,
        "accepted_new_outcomes": accepted_count,
        "credited_prior_outcomes": len(plan["credited_sessions"]),
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if complete_tasks + fenced_tasks != int(plan["task_count"]):
        raise RuntimeError("hosted task terminal accounting drifted")
    terminal["receipt_sha256"] = digest_without(terminal, "receipt_sha256")
    self_hosted.write_json_once(root / "TERMINAL.json", terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source-plan", type=Path, required=True)
    build.add_argument("--source-state", type=Path, required=True)
    build.add_argument("--model", choices=sorted(CAMPAIGNS), required=True)
    build.add_argument("--output", type=Path, required=True)
    remainder = sub.add_parser("build-remainder")
    remainder.add_argument("--predecessor-plan", type=Path, required=True)
    remainder.add_argument("--source-state", type=Path, required=True)
    remainder.add_argument("--model", choices=sorted(REMAINDER_CAMPAIGNS), required=True)
    remainder.add_argument("--output", type=Path, required=True)
    hydrate = sub.add_parser("hydrate-replacements")
    hydrate.add_argument("--assignment", type=Path, required=True)
    hydrate.add_argument("--output", type=Path, required=True)
    hydrate_hosted = sub.add_parser("hydrate-hosted-replacement")
    hydrate_hosted.add_argument("--supplement", type=Path, required=True)
    hydrate_hosted.add_argument("--output", type=Path, required=True)
    dedicated = sub.add_parser("build-dedicated-a")
    dedicated.add_argument("--source-plan", type=Path, required=True)
    dedicated.add_argument("--assignment", type=Path, required=True)
    dedicated.add_argument("--hydration", type=Path, required=True)
    dedicated.add_argument("--canary", type=Path, required=True)
    dedicated.add_argument("--output", type=Path, required=True)
    dedicated_b = sub.add_parser("build-dedicated-b")
    dedicated_b.add_argument("--source-plan", type=Path, required=True)
    dedicated_b.add_argument("--assignment", type=Path, required=True)
    dedicated_b.add_argument("--hydration", type=Path, required=True)
    dedicated_b.add_argument("--canary", type=Path, required=True)
    dedicated_b.add_argument("--output", type=Path, required=True)
    dedicated_b_v5 = sub.add_parser("build-dedicated-b-v5")
    dedicated_b_v5.add_argument("--predecessor-plan", type=Path, required=True)
    dedicated_b_v5.add_argument("--tombstone", type=Path, required=True)
    dedicated_b_v5.add_argument("--supplement", type=Path, required=True)
    dedicated_b_v5.add_argument("--hydration", type=Path, required=True)
    dedicated_b_v5.add_argument("--parity", type=Path, required=True)
    dedicated_b_v5.add_argument("--output", type=Path, required=True)
    qwen_replacements = sub.add_parser("build-qwen-replacements")
    qwen_replacements.add_argument("--source-plan", type=Path, required=True)
    qwen_replacements.add_argument("--hosted-plan", type=Path, required=True)
    qwen_replacements.add_argument("--assignment", type=Path, required=True)
    qwen_replacements.add_argument("--output", type=Path, required=True)
    glm_hosted_replacement = sub.add_parser("build-glm-hosted-replacement")
    glm_hosted_replacement.add_argument("--hosted-plan", type=Path, required=True)
    glm_hosted_replacement.add_argument("--supplement", type=Path, required=True)
    glm_hosted_replacement.add_argument("--hydration", type=Path, required=True)
    glm_hosted_replacement.add_argument("--output", type=Path, required=True)
    glm_hosted_reassigned_b = sub.add_parser("build-glm-hosted-reassigned-b")
    glm_hosted_reassigned_b.add_argument("--dedicated-plan", type=Path, required=True)
    glm_hosted_reassigned_b.add_argument("--hosted-plan", type=Path, required=True)
    glm_hosted_reassigned_b.add_argument("--tombstone", type=Path, required=True)
    glm_hosted_reassigned_b.add_argument("--supplement", type=Path, required=True)
    glm_hosted_reassigned_b.add_argument("--hydration", type=Path, required=True)
    glm_hosted_reassigned_b.add_argument("--output", type=Path, required=True)
    qwen_http500 = sub.add_parser("build-qwen-http500-successor")
    qwen_http500.add_argument("--original-plan", type=Path, required=True)
    qwen_http500.add_argument("--replacement-plan", type=Path, required=True)
    qwen_http500.add_argument("--incident", type=Path, required=True)
    qwen_http500.add_argument("--supplement", type=Path, required=True)
    qwen_http500.add_argument("--hydration", type=Path, required=True)
    qwen_http500.add_argument("--output", type=Path, required=True)
    glm_http500 = sub.add_parser("build-glm-http500-successor")
    glm_http500.add_argument("--original-plan", type=Path, required=True)
    glm_http500.add_argument("--reassigned-b-plan", type=Path, required=True)
    glm_http500.add_argument("--incident", type=Path, required=True)
    glm_http500.add_argument("--supplement", type=Path, required=True)
    glm_http500.add_argument("--hydration", type=Path, required=True)
    glm_http500.add_argument("--output", type=Path, required=True)
    glm_http500_primary = sub.add_parser("build-glm-http500-hosted-primary")
    glm_http500_primary.add_argument("--combined-plan", type=Path, required=True)
    glm_http500_primary.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--plan", type=Path, required=True)
    preflight.add_argument("--out-dir", type=Path, required=True)
    preflight.add_argument("--release-receipt", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--proxy-script", type=Path, required=True)
    run.add_argument("--release-receipt", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        value = build_plan(
            load_object(args.source_plan), load_object(args.source_state), args.model
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "validate":
        validate_plan(load_object(args.plan))
        return 0
    if args.command == "hydrate-replacements":
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        value = hydrate_glm53_replacements(load_object(args.assignment), key)
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "hydrate-hosted-replacement":
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        value = hydrate_glm53_replacements(load_object(args.supplement), key)
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-dedicated-a":
        value = build_dedicated_a_plan(
            load_object(args.source_plan),
            load_object(args.assignment),
            load_object(args.hydration),
            load_object(args.canary),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-dedicated-b":
        value = build_dedicated_a_plan(
            load_object(args.source_plan),
            load_object(args.assignment),
            load_object(args.hydration),
            load_object(args.canary),
            replica="B",
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-dedicated-b-v5":
        value = build_glm_dedicated_b_v5_plan(
            load_object(args.predecessor_plan),
            load_object(args.tombstone),
            load_object(args.supplement),
            load_object(args.hydration),
            load_object(args.parity),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-qwen-replacements":
        value = build_qwen_replacement_plan(
            load_object(args.source_plan),
            load_object(args.hosted_plan),
            load_object(args.assignment),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-glm-hosted-replacement":
        value = build_glm53_hosted_replacement_plan(
            load_object(args.hosted_plan),
            load_object(args.supplement),
            load_object(args.hydration),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-glm-hosted-reassigned-b":
        value = build_glm53_hosted_reassigned_b_plan(
            load_object(args.dedicated_plan),
            load_object(args.hosted_plan),
            load_object(args.tombstone),
            load_object(args.supplement),
            load_object(args.hydration),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-qwen-http500-successor":
        value = build_qwen_http500_successor_plan(
            load_object(args.original_plan),
            load_object(args.replacement_plan),
            load_object(args.incident),
            load_object(args.supplement),
            load_object(args.hydration),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-glm-http500-successor":
        value = build_glm_http500_successor_plan(
            load_object(args.original_plan),
            load_object(args.reassigned_b_plan),
            load_object(args.incident),
            load_object(args.supplement),
            load_object(args.hydration),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-glm-http500-hosted-primary":
        value = build_glm_http500_hosted_primary_plan(
            load_object(args.combined_plan)
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-remainder":
        value = build_remainder_plan(
            load_object(args.predecessor_plan),
            load_object(args.source_state),
            args.model,
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "preflight":
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        release = load_object(args.release_receipt) if args.release_receipt else None
        result = preflight_plan(load_object(args.plan), args.out_dir, key, release)
        print(json.dumps(result, sort_keys=True))
        return 0
    release = load_object(args.release_receipt) if args.release_receipt else None
    result = run_plan(load_object(args.plan), args.out_dir, args.proxy_script, release)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
