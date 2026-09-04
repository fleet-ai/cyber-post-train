"""Create and run duplicate-safe hosted OpenCode task-boundary shards."""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
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
GLM_DEDICATED_A_V5_SCORING_RELEASE_SCHEMA = (
    "fleet-glm53-dedicated-a-v5-scoring-release-v1"
)
COMPLETED_EXIT1_GAP_SCORING_RELEASE_SCHEMA = (
    "fleet-completed-exit1-gap-scoring-release-v1"
)
DEDICATED_A_COMPLETED_EXIT1_RECONCILIATION_SCHEMA = (
    "fleet-dedicated-a-completed-exit1-reconciliation-v1"
)
DEDICATED_A_COMPLETED_EXIT1_RECONCILIATION_DIGEST = (
    "sha256:f53f54a0c77cb213f5245442e96b5a8e41fe2095855176fda204b7c440eb7027"
)
DEDICATED_A_COMPLETED_EXIT1_GAP_CAMPAIGN = (
    "chris-cyber-glm53-opencode11827-dedicated-a-v5-gap-s6a4-p4-v1"
)
DEDICATED_A_COMPLETED_EXIT1_GAP_RELEASE_SCHEMA = (
    "fleet-dedicated-a-completed-exit1-gap-scoring-release-v1"
)
QWEN_POST_PARTIAL_TAIL_RELEASE_SCHEMA = (
    "fleet-qwen38-post-partial-tail-scoring-release-v1"
)
QWEN_POST_PARTIAL_TAIL_AUTH_STATEMENT = (
    "I authorize create-once launch of the independently audited Qwen3.8 hosted "
    "untouched-tail shards with exact rebuilt plans A "
    "sha256:ce973c3842be767e46c79422dce8395932d81626b29e6e34c4b9bf05d89eda9e "
    "(sources11..32, 88 cells) and B "
    "sha256:410324cc8a68c8d45746814c3882dc1c07e9182ce0c89a626eb5dfb045370131 "
    "(sources33..50 plus52..55, 88 cells), only after exact releases/manifests "
    "are committed, each create-once preflight exclusively succeeds, and fresh "
    "exhaustive API/Kubernetes/SFS checks prove no planned run/cell is accepted, "
    "claimed, active, or present. Preserve the exact v8 Qwen revision, "
    "OpenCode1.18.27 frozen no-autocontinue bytes, task/environment/verifier/tool "
    "identities, corrected verifier-backed exit1 credit policy, and "
    "fleet-train-high. Exclude source10 and r56. Use at most two Qwen hosted "
    "streams; never repeat any claimed/scored/ingested/accepted/reconciled/active "
    "cell."
)
DEDICATED_A_COMPLETED_EXIT1_GAP_AUTH_STATEMENT = (
    "I authorize the create-once scored launch of dedicated GLM A v5 gap plan "
    "sha256:14b799389571e82765ce83373a5d88fc37adb8c4a36b414366f699b75ef4da27 "
    "after exact reconciliation, treatment, duplicate, overlap, and preflight "
    "gates pass. It runs only source6 attempt4, uses fleet-train-high, credits "
    "source6 attempts1-3 without repeating them, and must not repeat any scored "
    "cell."
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
GLM_DEDICATED_A_V5_AUTH_STATEMENT = (
    "I authorize the create-once scored launch of dedicated GLM A v5 successor "
    "plan sha256:e1f37476700beb16ccb3978f61a162a59e16501334d316417cd0718a94c2ef68 "
    "after exact parity, scoring-recovery, hydration, corrected attrition, duplicate, "
    "and overlap gates pass. It contains exactly 27 complete primary tasks/108 cells, "
    "uses fleet-train-high, binds A v5 runtime identities, excludes fenced source4, "
    "and must not repeat any scored cell."
)
COMPLETED_EXIT1_GAP_AUTH_STATEMENT = (
    "I authorize create-once scored launches of Qwen gap plan "
    "sha256:43ef5b8e649ccd88b2f77216f9ca4ec6dea73edddff972321f89ba6d8a48c4b1 "
    "and hosted GLM gap plan "
    "sha256:f67e1b02040bd795d336d27a42c8f3ea8752d66c1618f623ebe0ba87e152a0ac "
    "after exact reconciliation, treatment, duplicate, overlap, and preflight "
    "gates pass. This raises hosted concurrency from one to two streams per model "
    "endpoint (four hosted streams total; six scored controllers including dedicated "
    "A/B), not three on either endpoint. Prior Qwen concurrency2 was healthy; prior "
    "GLM common failure was control-plane HTTP500, not inference pressure. Use "
    "fleet-train-high, monitor score-blind latency/5xx/transport/restarts, and stop "
    "adding concurrency on any regression. Run only the five never-started cells per "
    "model; never repeat credited cells or launch r56/r111."
)
COMPLETED_EXIT1_GAP_V2_AUTH_STATEMENT = (
    "I authorize create-once scored launches of Qwen gap plan "
    "sha256:2a86babdce4914e85bd2261b15cc5e2b476bf2f34dd224c2d76b9b8af5caeabc "
    "and hosted GLM gap plan "
    "sha256:6cc8e9cb7b7f6fbb7fd7a0ca97b9d434e410e2d6cffcc6d9cc9a41cd586532c0 "
    "after exact reconciliation, treatment, sealed-credit, duplicate, overlap, "
    "and preflight gates pass. These fresh v2 campaigns run only five never-started "
    "cells per model, use fleet-train-high, raise concurrency only to two streams "
    "per model endpoint, preserve Q50/200 and G100/400, must not repeat any credited "
    "cell, and must not launch r56/r111."
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
GLM_DEDICATED_A_V5_CAMPAIGN = (
    "chris-cyber-glm53-opencode11827-dedicated-a-v5-successor27-p4-v2"
)
QWEN_ATTRITION_REPLACEMENT_CAMPAIGN = (
    "chris-cyber-q38-opencode11827-hosted-replacement-r56-p4-v1"
)
GLM_ATTRITION_REPLACEMENT_CAMPAIGN = (
    "chris-cyber-glm53-opencode11827-hosted-replacement-r111-p4-v1"
)
QWEN_COMPLETED_EXIT1_GAP_CAMPAIGN = (
    "chris-cyber-q38-opencode11827-hosted-gap-q6q7-p4-v2"
)
QWEN_POST_PARTIAL_TAIL_CAMPAIGNS = {
    "qwen38_post_partial_tail_a": (
        "chris-cyber-q38-opencode11827-hosted-tail-a22-p4-v9"
    ),
    "qwen38_post_partial_tail_b": (
        "chris-cyber-q38-opencode11827-hosted-tail-b22-p4-v9"
    ),
}
QWEN_POST_PARTIAL_INCIDENT_DIGEST = (
    "sha256:bf34f05d3dff0de55b0f7aeff948b3b6c793dfa64adb298d4ae7e16d42666056"
)
GLM_COMPLETED_EXIT1_GAP_CAMPAIGN = (
    "chris-cyber-glm53-opencode11827-hosted-gap-g13g15-p4-v2"
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
        "glm53_dedicated_a_v5": 27,
        "qwen38_attrition_replacement": 1,
        "glm53_attrition_replacement": 1,
        "qwen38_completed_exit1_gap": 2,
        "glm53_completed_exit1_gap": 2,
        "glm53_dedicated_a_completed_exit1_gap": 1,
        "qwen38_post_partial_tail_a": 22,
        "qwen38_post_partial_tail_b": 22,
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


def validate_qwen_post_partial_tail_release(
    plan: dict[str, Any], release: dict[str, Any] | None
) -> None:
    """Bind both disjoint Qwen tail shards to one exact create-once release."""
    if plan.get("shard_key") not in QWEN_POST_PARTIAL_TAIL_CAMPAIGNS:
        return
    if not isinstance(release, dict):
        raise ValueError("Qwen post-partial tail scoring release is required")
    plans = release.get("plans") or {}
    gates = release.get("gates") or {}
    renderer = release.get("frozen_no_autocontinue_renderer") or {}
    scheduling = release.get("scheduling") or {}
    authorization = release.get("authorization") or {}
    privacy = release.get("privacy") or {}
    expected_plans = {
        "qwen38_post_partial_tail_a": {
            "plan_sha256": (
                "sha256:ce973c3842be767e46c79422dce8395932d81626b29e6e34c4b9bf05d89eda9e"
            ),
            "source_ranks": list(range(11, 33)),
            "task_count": 22,
            "cell_count": 88,
        },
        "qwen38_post_partial_tail_b": {
            "plan_sha256": (
                "sha256:410324cc8a68c8d45746814c3882dc1c07e9182ce0c89a626eb5dfb045370131"
            ),
            "source_ranks": [*range(33, 51), 52, 53, 54, 55],
            "task_count": 22,
            "cell_count": 88,
        },
    }
    if (
        release.get("schema_version") != QWEN_POST_PARTIAL_TAIL_RELEASE_SCHEMA
        or release.get("append_only") is not True
        or release.get("authorized_at") != "2026-09-04T17:39:39Z"
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or plans != expected_plans
        or plans.get(plan["shard_key"], {}).get("plan_sha256")
        != plan["plan_sha256"]
        or gates.get("partial_ingest_incident_receipt_sha256")
        != QWEN_POST_PARTIAL_INCIDENT_DIGEST
        or gates.get("pairwise_task_and_cell_overlap") != 0
        or gates.get("source10_and_r56_excluded") is not True
        or gates.get("each_preflight_must_succeed_exclusively") is not True
        or gates.get("fresh_api_kubernetes_sfs_duplicate_check_required") is not True
        or gates.get("current_plan_identity_absence_required") is not True
        or gates.get("corrected_exit1_credit_policy_required")
        != "credit_only_if_reward_ingest_cleanup_and_authoritative_session_match"
        or renderer.get("context_management")
        != self_hosted.OPENCODE_NO_AUTOCONTINUE_CONTEXT_MANAGEMENT
        or renderer.get("settings_canonical_sha256")
        != "sha256:fa7464a2a043b1e02febbabc70b1fce4278ec31a4d0968e4318677cae1f36e24"
        or renderer.get("settings_file_sha256_with_newline")
        != "sha256:2af2db821b685da8029f8d5765e1ce34c92c87df4a9f14315f56fb22ef6a90ed"
        or renderer.get("plugin_sha256")
        != "sha256:3542f8fe30d270bec6ee8e832081da8169fd78b667647ecd119425b1961a7a28"
        or renderer.get("matches_predecessor_v8_runtime_bytes") is not True
        or scheduling.get("required_priority_class") != "fleet-train-high"
        or scheduling.get("workers_per_shard") != 1
        or scheduling.get("maximum_qwen_hosted_streams") != 2
        or scheduling.get("true_non_preemptible_available") is not False
        or scheduling.get("priority_class_is_not_preemption_immunity") is not True
        or authorization.get("timestamp_utc") != "2026-09-04T17:39:39Z"
        or authorization.get("author") != "/root"
        or authorization.get("statement") != QWEN_POST_PARTIAL_TAIL_AUTH_STATEMENT
        or authorization.get("scored_launch_authorized") is not True
        or authorization.get("create_once") is not True
        or authorization.get("must_not_repeat") is not True
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("Qwen post-partial tail release does not bind this plan")


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


def validate_glm_dedicated_a_v5_scoring_release(
    plan: dict[str, Any], release: dict[str, Any] | None
) -> None:
    """Fail closed until the exact A v5 release and root authorization are supplied."""
    if plan.get("shard_key") != "glm53_dedicated_a_v5":
        return
    if not isinstance(release, dict):
        raise ValueError("GLM dedicated A v5 scoring release is required")
    source = plan.get("source") or {}
    treatment = plan.get("treatment_block") or {}
    plan_evidence = release.get("plan") or {}
    gates = release.get("gates") or {}
    release_treatment = release.get("treatment") or {}
    scheduling = release.get("scheduling") or {}
    authorization = release.get("authorization") or {}
    privacy = release.get("privacy") or {}
    if (
        release.get("schema_version") != GLM_DEDICATED_A_V5_SCORING_RELEASE_SCHEMA
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or release.get("append_only") is not True
        or release.get("supersedes") is not None
        or plan_evidence.get("plan_sha256") != plan["plan_sha256"]
        or plan_evidence.get("task_count") != 27
        or plan_evidence.get("pass_k") != 4
        or plan_evidence.get("cell_count") != 108
        or plan_evidence.get("source_ranks")
        != [int(row["source_rank"]) for row in plan["tasks"]]
        or gates.get("corrected_stop_tombstone_receipt_sha256")
        != source.get("corrected_stop_tombstone_receipt_sha256")
        or gates.get("selection_supplement_receipt_sha256")
        != source.get("selection_supplement_receipt_sha256")
        or gates.get("hydration_receipt_sha256")
        != source.get("hydration_receipt_sha256")
        or gates.get("parity_receipt_sha256")
        != source.get("parity_receipt_sha256")
        or gates.get("scoring_recovery_receipt_sha256")
        != "sha256:708d635968e073d27153d3d9f13ae5fa7b4784349cc9e8e3928f0fbafeade132"
        or gates.get("source4_fenced") is not True
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
        or authorization.get("timestamp_utc") != "2026-09-04T06:48:59Z"
        or authorization.get("author") != "/root"
        or authorization.get("statement") != GLM_DEDICATED_A_V5_AUTH_STATEMENT
        or authorization.get("scored_launch_authorized") is not True
        or authorization.get("must_not_repeat") is not True
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("GLM dedicated A v5 scoring release does not bind this plan")


def validate_completed_exit1_gap_scoring_release(
    plan: dict[str, Any], release: dict[str, Any] | None
) -> None:
    """Fail closed until exact gap cells and concurrency are authorized."""
    if plan.get("shard_key") not in {
        "qwen38_completed_exit1_gap",
        "glm53_completed_exit1_gap",
    }:
        return
    if not isinstance(release, dict):
        raise ValueError("completed exit-1 gap scoring release is required")
    plan_evidence = release.get("plan") or {}
    gates = release.get("gates") or {}
    concurrency = release.get("concurrency") or {}
    treatment = release.get("treatment") or {}
    scheduling = release.get("scheduling") or {}
    authorization = release.get("authorization") or {}
    privacy = release.get("privacy") or {}
    source = plan.get("source") or {}
    is_qwen = plan["shard_key"].startswith("qwen38")
    is_v2 = plan.get("campaign_id") in {
        QWEN_COMPLETED_EXIT1_GAP_CAMPAIGN,
        GLM_COMPLETED_EXIT1_GAP_CAMPAIGN,
    }
    expected_timestamp = (
        "2026-09-04T07:38:44Z" if is_v2 else "2026-09-04T07:30:30Z"
    )
    expected_statement = (
        COMPLETED_EXIT1_GAP_V2_AUTH_STATEMENT
        if is_v2
        else COMPLETED_EXIT1_GAP_AUTH_STATEMENT
    )
    expected_attempts = (
        [[6, 3], [6, 4], [7, 2], [7, 3], [7, 4]]
        if is_qwen
        else [[13, 3], [13, 4], [15, 2], [15, 3], [15, 4]]
    )
    if (
        release.get("schema_version") != COMPLETED_EXIT1_GAP_SCORING_RELEASE_SCHEMA
        or release.get("receipt_sha256") != digest_without(release, "receipt_sha256")
        or release.get("append_only") is not True
        or release.get("supersedes") is not None
        or plan_evidence.get("plan_sha256") != plan["plan_sha256"]
        or plan_evidence.get("task_count") != 2
        or plan_evidence.get("credited_cell_count") != 3
        or plan_evidence.get("new_cell_count") != 5
        or plan_evidence.get("pass_k") != 4
        or plan_evidence.get("attempt_cells") != expected_attempts
        or treatment != plan.get("treatment_block")
        or gates.get("reconciliation_receipt_sha256")
        != source.get("reconciliation_receipt_sha256")
        or gates.get("gap_source_receipt_sha256")
        != source.get("gap_source_receipt_sha256")
        or gates.get("held_unused_replacement_rank")
        != source.get("held_unused_replacement_rank")
        or (is_v2 and gates.get("sealed_credit_gate_required") is not True)
        or (
            is_v2
            and is_qwen
            and gates.get("preflight_v1_failure_receipt_sha256")
            != source.get("preflight_v1_failure_receipt_sha256")
        )
        or not all(
            gates.get(field) is True
            for field in (
                "exact_treatment_gate_required",
                "duplicate_gate_required",
                "overlap_gate_required",
                "fresh_create_once_identity_required",
                "credited_cells_must_not_repeat",
                "primary_denominator_completion_gate_required",
            )
        )
        or concurrency.get("authorized_hosted_streams_per_model_endpoint") != 2
        or concurrency.get("authorized_hosted_streams_total") != 4
        or concurrency.get("authorized_scored_controllers_total") != 6
        or concurrency.get("third_hosted_stream_per_endpoint_authorized") is not False
        or concurrency.get("stop_adding_on_regression") is not True
        or scheduling.get("required_priority_class") != "fleet-train-high"
        or scheduling.get("workers") != 1
        or scheduling.get("true_non_preemptible_available") is not False
        or scheduling.get("priority_class_is_not_preemption_immunity") is not True
        or authorization.get("timestamp_utc") != expected_timestamp
        or authorization.get("author") != "/root"
        or authorization.get("statement") != expected_statement
        or authorization.get("scored_launch_authorized") is not True
        or authorization.get("create_once") is not True
        or authorization.get("must_not_repeat") is not True
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("completed exit-1 gap scoring release does not bind this plan")


def validate_dedicated_a_completed_exit1_gap_release(
    plan: dict[str, Any], release: dict[str, Any] | None
) -> None:
    """Require exact root authorization for the source6/a4-only A-v5 gap."""
    if plan.get("shard_key") != "glm53_dedicated_a_completed_exit1_gap":
        return
    if not isinstance(release, dict):
        raise ValueError("dedicated A completed exit-1 gap release is required")
    plan_evidence = release.get("plan") or {}
    gates = release.get("gates") or {}
    scheduling = release.get("scheduling") or {}
    authorization = release.get("authorization") or {}
    privacy = release.get("privacy") or {}
    if (
        release.get("schema_version")
        != DEDICATED_A_COMPLETED_EXIT1_GAP_RELEASE_SCHEMA
        or release.get("receipt_sha256")
        != digest_without(release, "receipt_sha256")
        or release.get("append_only") is not True
        or release.get("supersedes") is not None
        or plan_evidence.get("plan_sha256") != plan["plan_sha256"]
        or plan_evidence.get("task_count") != 1
        or plan_evidence.get("credited_cell_count") != 3
        or plan_evidence.get("new_cell_count") != 1
        or plan_evidence.get("pass_k") != 4
        or plan_evidence.get("attempt_cells") != [[6, 4]]
        or release.get("treatment") != plan.get("treatment_block")
        or gates.get("reconciliation_receipt_sha256")
        != DEDICATED_A_COMPLETED_EXIT1_RECONCILIATION_DIGEST
        or gates.get("predecessor_job_uid")
        != "87e2ca85-d684-4d87-9f05-f4399b7906d0"
        or gates.get("predecessor_pod_uid")
        != "083a29e3-7d0e-4cbb-9447-62373772080d"
        or not all(
            gates.get(field) is True
            for field in (
                "exact_treatment_gate_required",
                "duplicate_gate_required",
                "overlap_gate_required",
                "authoritative_session_requery_required",
                "source6_attempt4_zero_history_required",
                "credited_cells_must_not_repeat",
                "fresh_create_once_identity_required",
                "primary_denominator_completion_gate_required",
            )
        )
        or scheduling.get("required_priority_class") != "fleet-train-high"
        or scheduling.get("workers") != 1
        or scheduling.get("true_non_preemptible_available") is not False
        or scheduling.get("priority_class_is_not_preemption_immunity") is not True
        or authorization.get("timestamp_utc") != "2026-09-04T08:45:29Z"
        or authorization.get("author") != "/root"
        or authorization.get("statement")
        != DEDICATED_A_COMPLETED_EXIT1_GAP_AUTH_STATEMENT
        or authorization.get("scored_launch_authorized") is not True
        or authorization.get("create_once") is not True
        or authorization.get("must_not_repeat") is not True
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("dedicated A completed exit-1 gap release does not bind plan")


def validate_task_boundary_concurrency_cutover_proposal(
    proposal: dict[str, Any],
) -> None:
    """Validate the no-launch design for future whole-task concurrency cutovers."""
    finding = proposal.get("current_controller_finding") or {}
    gates = proposal.get("required_cutover_gates") or {}
    successor = proposal.get("successor_partition_contract") or {}
    implementation = proposal.get("future_controller_change") or {}
    release = proposal.get("release") or {}
    privacy = proposal.get("privacy") or {}
    if (
        proposal.get("schema_version")
        != "fleet-task-boundary-concurrency-cutover-proposal-v1"
        or proposal.get("append_only") is not True
        or proposal.get("receipt_sha256")
        != digest_without(proposal, "receipt_sha256")
        or finding.get("task_claim_scope") != "whole_task"
        or finding.get("same_task_max_inflight") != 1
        or finding.get("next_task_claim_is_immediate_after_terminal") is not True
        or finding.get("stop_after_current_task_barrier_supported") is not False
        or finding.get("cross_plan_lease_supported") is not False
        or finding.get("safe_live_tail_shard_while_predecessor_active") is not False
        or finding.get("temporal_distance_to_tail_is_safety_evidence") is not False
        or gates.get("predecessor_terminal_or_sealed_stop_tombstone_required")
        is not True
        or gates.get("stopped_pod_absence_required") is not True
        or gates.get("last_claimed_task_exactly_enumerated") is not True
        or gates.get("next_task_zero_execution_required") is not True
        or gates.get("all_prior_claims_and_sessions_reconciled") is not True
        or gates.get("api_sfs_k8s_duplicate_inventory_required") is not True
        or successor.get("complete_task_boundaries_only") is not True
        or successor.get("pairwise_task_disjointness_required") is not True
        or successor.get("pairwise_cell_disjointness_required") is not True
        or successor.get("strictly_after_last_claimed_source_task") is not True
        or successor.get("exact_predecessor_treatment_required") is not True
        or successor.get("fresh_plan_run_and_root_identities_required") is not True
        or successor.get("create_once") is not True
        or successor.get("workers_per_shard") != 1
        or successor.get("required_priority_class") != "fleet-train-high"
        or successor.get("priority_class_is_not_preemption_immunity") is not True
        or implementation.get("new_plans_only") is not True
        or implementation.get("active_jobs_must_not_be_mutated") is not True
        or implementation.get("barrier_receipt_written_before_next_task_claim")
        is not True
        or implementation.get("barrier_receipt_binds_plan_job_pod_and_last_task")
        is not True
        or implementation.get("successor_preflight_requires_barrier_and_pod_absence")
        is not True
        or release.get("concrete_successor_tasks_selected") is not False
        or release.get("cluster_objects_created") is not False
        or release.get("scored_launch_authorized") is not False
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("task-boundary concurrency cutover proposal drifted")


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
        "fleet-qwen38-replacement-selection-supplement-v3",
        "fleet-opencode-replacement-selection-supplement-v5",
    }:
        row = assignment.get("replacement") or {}
        rows = [row]
        is_qwen = supplement_schema.startswith("fleet-qwen38")
        expected_rank = 56 if is_qwen else 111
        expected_ranks = [expected_rank]
        fence = assignment.get("source6_fence" if is_qwen else "source13_fence") or {}
        if (
            assignment.get("append_only") is not True
            or row.get("serving_block")
            != (
                "hosted_qwen_successor_replacement"
                if is_qwen
                else "hosted_glm_successor_replacement"
            )
            or row.get("scored_launch_authorized") is not False
            or (row.get("hydration_gate") or {}).get("status")
            != "required_not_satisfied"
            or fence.get("whole_task_fenced") is not True
            or (fence.get("noncreditable_attempt") or {}).get("retry_allowed")
            is not False
        ):
            raise ValueError("hosted noncreditable replacement supplement drifted")
        receipt_schema = (
            "fleet-qwen38-hosted-replacement-hydration-v3"
            if is_qwen
            else "fleet-glm53-hosted-replacement-hydration-v3"
        )
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


def build_glm_dedicated_a_v5_plan(
    predecessor_plan: dict[str, Any],
    tombstone: dict[str, Any],
    supplement: dict[str, Any],
    hydration: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    """Build the non-launchable dedicated A successor on serving generation v5."""
    validate_plan(predecessor_plan)
    if (
        predecessor_plan.get("shard_key") != "glm53_dedicated_a"
        or predecessor_plan.get("plan_sha256")
        != "sha256:4f8d4fcf50af8abccf3b9d272a18755f9be0bc862a66fe25bba691bee8a22208"
        or tombstone.get("receipt_sha256")
        != "sha256:9bf6683d80c2dd517e371709a1aa2db0a13e1f4089a09a8296941c6df7054b79"
        or tombstone.get("receipt_sha256") != digest_without(tombstone, "receipt_sha256")
        or supplement.get("receipt_sha256")
        != "sha256:3ccf7fdace01bc2a3e6245d26f65d6069fa5327d4a5ac7515640c7d59092e534"
        or supplement.get("receipt_sha256")
        != digest_without(supplement, "receipt_sha256")
        or hydration.get("receipt_sha256")
        != "sha256:5b52c2f54865434685ab30b10246b83a738b6fb9288cfcfbce0dcab9805377f7"
        or hydration.get("receipt_sha256") != digest_without(hydration, "receipt_sha256")
        or hydration.get("selection_supplement_receipt_sha256")
        != supplement["receipt_sha256"]
        or parity.get("receipt_sha256")
        != "sha256:9aa92ac211f102939dc24e03af5c93766b94b0ccf5ce887df1ac80415adf73bc"
        or parity.get("receipt_sha256") != digest_without(parity, "receipt_sha256")
        or parity.get("classification") != "operational-gate-passed"
        or (parity.get("outcome_integrity") or {}).get("scored_sessions") != 0
        or (parity.get("content_blind_probe") or {}).get("health") is not True
        or (parity.get("runtime") or {}).get("priority_class") != "fleet-train-high"
        or (parity.get("corrected_source4_fence") or {}).get("receipt_sha256")
        != tombstone["receipt_sha256"]
    ):
        raise ValueError("dedicated A v5 source evidence drifted")
    source_tasks = {
        int(row["source_rank"]): copy.deepcopy(row)
        for row in predecessor_plan["tasks"]
    }
    selected = [*range(6, 53, 2), 102, 104]
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
        "replica": "A",
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
        "corrected_stop_tombstone_receipt_sha256": tombstone["receipt_sha256"],
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
        "shard_key": "glm53_dedicated_a_v5",
        "campaign_id": GLM_DEDICATED_A_V5_CAMPAIGN,
        "source_job_id": predecessor_plan["source_job_id"],
        "source": {
            "predecessor_plan_sha256": predecessor_plan["plan_sha256"],
            "corrected_stop_tombstone_receipt_sha256": tombstone["receipt_sha256"],
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
        "fenced_source_ranks": [1, 2, 3, 4, 5, 7, 9, 11, 54, 106],
        "hosted_source_ranks": [*range(13, 100, 2), 108, 109],
        "dedicated_b_source_ranks": [*range(56, 101, 2), 101, 103, 105, 107],
        "execution": {
            **copy.deepcopy(predecessor_plan["execution"]),
            "launch_authorized": False,
            "required_priority_class": "fleet-train-high",
            "required_recovery_gate": "fleet_scoring_api_http500_recovery_v1",
        },
        "tasks": tasks,
        "attempts": _fresh_attempts(
            tasks, GLM_DEDICATED_A_V5_CAMPAIGN, "glm53-dedicated-a-v5"
        ),
        "privacy": copy.deepcopy(predecessor_plan["privacy"]),
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def build_hosted_attrition_replacement_plan(
    predecessor_plan: dict[str, Any],
    supplement: dict[str, Any],
    hydration: dict[str, Any],
) -> dict[str, Any]:
    """Build a preview-only one-task replacement for a noncreditable task fence."""
    validate_plan(predecessor_plan)
    is_qwen = supplement.get("schema_version") == (
        "fleet-qwen38-replacement-selection-supplement-v3"
    )
    expected = {
        True: {
            "predecessor": (
                "sha256:8d6df6af63b308d648fb0c7ea9115f90"
                "b16de1d7e57b0968dda8fc8fd4a20c81"
            ),
            "supplement": "sha256:56406bb32541babc0189e089f339d6a7668b24cb86b1fd8587b60a3cfc57b24c",
            "hydration": "sha256:4d059348e1c00eed04aef8b6a700a0b44921cfe0f6e6be0e8e55e08316886f79",
            "rank": 56,
            "fenced": 6,
            "fence_key": "source6_fence",
            "shard": "qwen38_attrition_replacement",
            "campaign": QWEN_ATTRITION_REPLACEMENT_CAMPAIGN,
            "network_prefix": "qwen38-hosted-r56",
            "primary_tasks": 50,
            "primary_cells": 200,
        },
        False: {
            "predecessor": (
                "sha256:8b0deafa9f51b75a0715be56b417454e"
                "96665b5342d4c346d5cda32dc24c8279"
            ),
            "supplement": "sha256:e8b6c3f993c4e5b76f823acc0bbda4c210a7dbd4526c1adb093c0535bbb4c00d",
            "hydration": "sha256:ba6327ce5ca006c48b4cec33ed93179c0fc9c32a0b2bda8ec0a023272ac1758d",
            "rank": 111,
            "fenced": 13,
            "fence_key": "source13_fence",
            "shard": "glm53_attrition_replacement",
            "campaign": GLM_ATTRITION_REPLACEMENT_CAMPAIGN,
            "network_prefix": "glm53-hosted-r111",
            "primary_tasks": 100,
            "primary_cells": 400,
        },
    }[is_qwen]
    fence = supplement.get(expected["fence_key"]) or {}
    row = supplement.get("replacement") or {}
    if (
        predecessor_plan.get("plan_sha256") != expected["predecessor"]
        or supplement.get("receipt_sha256") != expected["supplement"]
        or supplement.get("receipt_sha256")
        != digest_without(supplement, "receipt_sha256")
        or hydration.get("receipt_sha256") != expected["hydration"]
        or hydration.get("receipt_sha256") != digest_without(hydration, "receipt_sha256")
        or hydration.get("selection_supplement_receipt_sha256")
        != supplement["receipt_sha256"]
        or int(row.get("replacement_rank") or 0) != expected["rank"]
        or int(fence.get("source_rank") or 0) != expected["fenced"]
        or fence.get("plan_sha256") != predecessor_plan["plan_sha256"]
        or fence.get("whole_task_fenced") is not True
        or (fence.get("noncreditable_attempt") or {}).get("retry_allowed") is not False
    ):
        raise ValueError("hosted attrition replacement evidence drifted")
    tasks = _hydrated_replacement_tasks(supplement, hydration, 1)
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": expected["shard"],
        "campaign_id": expected["campaign"],
        "source_job_id": predecessor_plan["source_job_id"],
        "source": {
            "predecessor_plan_sha256": predecessor_plan["plan_sha256"],
            "selection_supplement_receipt_sha256": supplement["receipt_sha256"],
            "hydration_receipt_sha256": hydration["receipt_sha256"],
            "fenced_source_rank": expected["fenced"],
            "accepted_attrition_receipt_sha256": fence["accepted_attempt"][
                "receipt_sha256"
            ],
            "noncreditable_receipt_sha256": fence["noncreditable_attempt"][
                "receipt_sha256"
            ],
        },
        "treatment_block": copy.deepcopy(predecessor_plan["treatment_block"]),
        "model": copy.deepcopy(predecessor_plan["model"]),
        "harness": copy.deepcopy(predecessor_plan["harness"]),
        "authority": copy.deepcopy(predecessor_plan["authority"]),
        "task_count": 1,
        "pass_k": 4,
        "total_session_count": 4,
        "credited_sessions": [],
        "new_session_count": 4,
        "fenced_source_ranks": sorted(
            {*predecessor_plan.get("fenced_source_ranks", []), expected["fenced"]}
        ),
        "primary_estimator_task_count": expected["primary_tasks"],
        "primary_estimator_cell_count": expected["primary_cells"],
        "execution": {
            **copy.deepcopy(predecessor_plan["execution"]),
            "launch_authorized": False,
            "required_priority_class": "fleet-train-high",
        },
        "tasks": tasks,
        "attempts": _fresh_attempts(
            tasks, expected["campaign"], expected["network_prefix"]
        ),
        "privacy": copy.deepcopy(predecessor_plan["privacy"]),
    }
    plan["plan_sha256"] = digest_without(plan, "plan_sha256")
    validate_plan(plan)
    return plan


def validate_completed_exit1_reconciliation(
    receipt: dict[str, Any],
    qwen_plan: dict[str, Any],
    glm_plan: dict[str, Any],
) -> None:
    """Validate the reviewed, score-blind classification of completed exit-1 cells."""
    validate_plan(qwen_plan)
    validate_plan(glm_plan)
    if (
        receipt.get("schema_version")
        != "fleet-completed-exit1-reconciliation-v1"
        or receipt.get("append_only") is not True
        or receipt.get("receipt_sha256")
        != digest_without(receipt, "receipt_sha256")
    ):
        raise ValueError("completed exit-1 reconciliation receipt drifted")

    methodology = receipt.get("methodology") or {}
    required = set(methodology.get("required_all") or [])
    excluded = set(methodology.get("exclude_if_any") or [])
    if (
        methodology.get("outcome_unit") != "authoritative_scored_session"
        or methodology.get("process_exit_role")
        != "diagnostic_only_after_all_required_scoring_evidence_exists"
        or methodology.get("selection_bias_control")
        != "retain_original_task_and_complete_only_unstarted_attempt_indices"
        or methodology.get("original_noncreditable_receipts_mutated") is not False
        or required
        != {
            "immutable_plan_claim_and_config_binding",
            "agent_termination_completed",
            "reward_result_present",
            "reward_result_verifier_matches_result",
            "authoritative_session_unique_completed",
            "authoritative_session_verifier_matches_result",
            "session_ingest_completed",
            "instance_cleanup_completed",
            "no_stderr_or_proxy_transport_error_signal",
            "no_independent_uid_time_overlap_infrastructure_incident",
        }
        or excluded
        != {
            "reward_result_missing",
            "verifier_execution_missing",
            "authoritative_session_missing_or_incomplete",
            "session_ingest_incomplete",
            "cleanup_incomplete",
            "scoring_http_failure",
            "endpoint_or_control_plane_preemption_overlaps_attempt",
            "model_or_harness_transport_failure",
        }
    ):
        raise ValueError("completed exit-1 methodology drifted")

    plans = {
        "qwen_hosted_v8": qwen_plan,
        "glm_hosted_v12": glm_plan,
    }
    expected_cells = {
        ("qwen_hosted_v8", 6, 2),
        ("qwen_hosted_v8", 7, 1),
        ("glm_hosted_v12", 13, 2),
        ("glm_hosted_v12", 15, 1),
    }
    cells = receipt.get("reconciled_cells") or []
    observed_cells = {
        (
            row.get("model_block"),
            int(row.get("source_rank") or 0),
            int(row.get("attempt") or 0),
        )
        for row in cells
    }
    if len(cells) != 4 or observed_cells != expected_cells:
        raise ValueError("completed exit-1 cell set drifted")

    run_ids: set[str] = set()
    session_ids: set[str] = set()
    verifier_ids: set[str] = set()
    for row in cells:
        plan = plans[row["model_block"]]
        if row.get("plan_sha256") != plan["plan_sha256"]:
            raise ValueError("reconciled cell plan binding drifted")
        planned = next(
            (
                item
                for item in plan["attempts"]
                if int(item["source_rank"]) == int(row["source_rank"])
                and int(item["attempt"]) == int(row["attempt"])
            ),
            None,
        )
        task = next(
            (
                item
                for item in plan["tasks"]
                if int(item["source_rank"]) == int(row["source_rank"])
            ),
            None,
        )
        if (
            planned is None
            or task is None
            or row.get("run_id") != planned["run_id"]
            or int(row.get("rank") or 0) != int(planned["rank"])
            or row.get("task_key") != task["task"]["key"]
            or row.get("task_version_id") != task["task"]["version_id"]
        ):
            raise ValueError("reconciled cell is not plan-bound")
        evidence = row.get("evidence") or {}
        if (
            evidence.get("agent_exit_code") != 1
            or evidence.get("agent_termination_completed") is not True
            or evidence.get("reward_result_present") is not True
            or evidence.get("reward_result_verifier_matches") is not True
            or evidence.get("authoritative_session_match_count") != 1
            or evidence.get("authoritative_session_status") != "completed"
            or evidence.get("authoritative_session_verifier_matches") is not True
            or evidence.get("session_ingest_completed") is not True
            or evidence.get("cleanup_completed") is not True
            or evidence.get("stderr_file_count") != 0
            or evidence.get("proxy_http_5xx_token_count") != 0
            or evidence.get("proxy_transport_error_token_count") != 0
            or evidence.get("independent_infrastructure_incident") is not False
            or evidence.get("agent_exit_code_was_only_prior_rejection") is not True
            or row.get("reconciled_outcome") != "RECONCILED_ACCEPTED"
            or row.get("counts_as_primary_cell") is not True
        ):
            raise ValueError("reconciled cell is not fully scored and operationally complete")
        digest_fields = [
            "claim_sha256",
            "config_sha256",
            "original_noncreditable_receipt_sha256",
            "original_noncreditable_file_sha256",
            "original_task_fence_receipt_sha256",
            "original_task_fence_file_sha256",
            "result_file_sha256",
            "reward_result_file_sha256",
            "session_ingest_file_sha256",
            "cleanup_file_sha256",
        ]
        if any(
            not isinstance(row.get(field), str)
            or not row[field].startswith("sha256:")
            or len(row[field]) != 71
            for field in digest_fields
        ):
            raise ValueError("reconciled cell evidence digest drifted")
        try:
            uuid.UUID(str(row.get("session_id")))
            uuid.UUID(str(row.get("verifier_execution_id")))
            uuid.UUID(str(row.get("job_uid")))
            uuid.UUID(str(row.get("pod_uid")))
        except ValueError as exc:
            raise ValueError("reconciled cell UID evidence drifted") from exc
        run_ids.add(row["run_id"])
        session_ids.add(row["session_id"])
        verifier_ids.add(row["verifier_execution_id"])
    if len(run_ids) != 4 or len(session_ids) != 4 or len(verifier_ids) != 4:
        raise ValueError("reconciled cell identity duplicated")

    gaps = receipt.get("same_task_gap_completion") or []
    expected_gaps = {
        ("qwen_hosted_v8", 6): ([1], [2], [3, 4]),
        ("qwen_hosted_v8", 7): ([], [1], [2, 3, 4]),
        ("glm_hosted_v12", 13): ([1], [2], [3, 4]),
        ("glm_hosted_v12", 15): ([], [1], [2, 3, 4]),
    }
    observed_gaps = {
        (row.get("model_block"), int(row.get("source_rank") or 0)): (
            row.get("already_accepted_attempts"),
            row.get("reconciled_accepted_attempts"),
            row.get("missing_attempts"),
        )
        for row in gaps
    }
    if observed_gaps != expected_gaps:
        raise ValueError("completed exit-1 gap set drifted")

    comparators = receipt.get("comparators") or {}
    if (
        (comparators.get("hosted_http500_incident") or {}).get("receipt_sha256")
        != "sha256:043c01e05e9118a080bd642a5da6f7c22c279781bc0c97803330091e55fdb27e"
        or (comparators.get("hosted_http500_incident") or {}).get(
            "classification"
        )
        != "EXCLUDED_INFRASTRUCTURE_INCOMPLETE"
        or (comparators.get("dedicated_a_preemption") or {}).get("receipt_sha256")
        != "sha256:9bf6683d80c2dd517e371709a1aa2db0a13e1f4089a09a8296941c6df7054b79"
        or (comparators.get("dedicated_a_preemption") or {}).get(
            "classification"
        )
        != "EXCLUDED_INFRASTRUCTURE_INCOMPLETE"
        or (comparators.get("dedicated_b_source54_attempt3") or {}).get(
            "receipt_sha256"
        )
        != "sha256:7cf5483e07e4053d7807d599b738e0555ade3b728ea20ae2356186592bd7c355"
        or (comparators.get("dedicated_b_source54_attempt3") or {}).get(
            "classification"
        )
        != "TERMINAL_ATTRITION_ONLY_NOT_PRIMARY_TASK"
    ):
        raise ValueError("completed exit-1 comparator classification drifted")
    replacements = receipt.get("replacement_artifacts") or []
    if len(replacements) != 2 or any(
        row.get("status") != "HELD_UNUSED_SUPERSEDED_IF_RECONCILIATION_RATIFIED"
        or row.get("scored_claims") != 0
        for row in replacements
    ):
        raise ValueError("completed exit-1 replacement hold drifted")
    denominators = receipt.get("primary_denominators_after_gap_completion") or {}
    if denominators.get("qwen") != {
        "retained_complete_source4_tasks": 1,
        "original_v8_tasks": 49,
        "replacement_tasks": 0,
        "tasks": 50,
        "cells": 200,
        "affected_missing_cells": 5,
    } or denominators.get("glm") != {
        "hosted_original_v12_tasks": 46,
        "dedicated_a_tasks": 27,
        "dedicated_b_tasks": 27,
        "replacement_tasks_for_current_exit1_cells": 0,
        "tasks": 100,
        "cells": 400,
        "affected_missing_cells": 5,
    }:
        raise ValueError("completed exit-1 denominator drifted")
    release = receipt.get("release_gate") or {}
    if (
        release.get("review_status")
        != "methodology_reviewed_pending_root_scoring_release"
        or release.get("scored_launch_authorized") is not False
        or release.get("future_gap_plan_must_credit_exact_reconciled_sessions")
        is not True
        or release.get("future_gap_plan_must_run_only_missing_attempt_indices")
        is not True
        or release.get("future_gap_plan_must_preserve_exact_predecessor_treatment")
        is not True
        or release.get("replacement_addons_must_not_launch") is not True
    ):
        raise ValueError("completed exit-1 release gate drifted")
    privacy = receipt.get("privacy") or {}
    if any(
        privacy.get(field) is not False
        for field in (
            "scores_read",
            "prompts_or_traces_read",
            "task_content_included",
            "credentials_included",
        )
    ):
        raise ValueError("completed exit-1 reconciliation privacy drifted")


def validate_dedicated_a_completed_exit1_reconciliation(
    receipt: dict[str, Any], predecessor: dict[str, Any]
) -> None:
    """Validate the exact fully scored dedicated-A source6/a3 correction."""
    validate_plan(predecessor)
    source = receipt.get("source") or {}
    result = receipt.get("authoritative_result") or {}
    api = receipt.get("authoritative_api") or {}
    exclusions = receipt.get("infrastructure_exclusions") or {}
    decision = receipt.get("decision") or {}
    privacy = receipt.get("privacy") or {}
    accepted = receipt.get("already_accepted") or []
    task = next(
        (row for row in predecessor["tasks"] if int(row["source_rank"]) == 6),
        None,
    )
    attempt = next(
        (
            row
            for row in predecessor["attempts"]
            if int(row["source_rank"]) == 6 and int(row["attempt"]) == 3
        ),
        None,
    )
    expected_accepted = {
        (
            1,
            "chris-cyber-glm53-opencode11827-dedicated-a-v5-successor27-p4-v2-sr006-a1-51fc6fa0",
            "c8c012fa-e92b-45bd-9151-f35c16eea629",
            "b5a92b99-3042-46ff-ad7e-141b66246c82",
            "sha256:76dbba0ccfd6fe4190c532712d25df48b76107067b3b01096589cc18a48ad537",
        ),
        (
            2,
            "chris-cyber-glm53-opencode11827-dedicated-a-v5-successor27-p4-v2-sr006-a2-51fc6fa0",
            "f77b673a-9a5f-4ca9-8ecd-4860b0ed9387",
            "3a81ee58-95b4-4eab-904c-5f7cbd5ce06c",
            "sha256:5bcf23b2f9ec5625cc089acba25d89d3bc0590d5e4321f489031c0b217290790",
        ),
    }
    observed_accepted = {
        (
            int(row.get("attempt") or 0),
            row.get("run_id"),
            row.get("session_id"),
            row.get("verifier_execution_id"),
            row.get("receipt_sha256"),
        )
        for row in accepted
        if int(row.get("source_rank") or 0) == 6
    }
    digest_fields = (
        "claim_sha256",
        "claim_file_sha256",
        "config_sha256",
        "original_noncreditable_receipt_sha256",
        "original_noncreditable_file_sha256",
        "original_task_fence_receipt_sha256",
        "original_task_fence_file_sha256",
    )
    result_digest_fields = (
        "result_file_sha256",
        "reward_result_file_sha256",
        "session_ingest_file_sha256",
        "cleanup_file_sha256",
    )
    if (
        task is None
        or attempt is None
        or receipt.get("schema_version")
        != DEDICATED_A_COMPLETED_EXIT1_RECONCILIATION_SCHEMA
        or receipt.get("append_only") is not True
        or receipt.get("receipt_sha256")
        != digest_without(receipt, "receipt_sha256")
        or receipt.get("receipt_sha256")
        != DEDICATED_A_COMPLETED_EXIT1_RECONCILIATION_DIGEST
        or receipt.get("classification") != "RECONCILED_ACCEPTED"
        or receipt.get("counts_as_primary_cell") is not True
        or source.get("plan_sha256") != predecessor["plan_sha256"]
        or source.get("job_uid")
        != "87e2ca85-d684-4d87-9f05-f4399b7906d0"
        or source.get("pod_uid")
        != "083a29e3-7d0e-4cbb-9447-62373772080d"
        or (
            int(source.get("rank") or 0),
            int(source.get("source_rank") or 0),
            int(source.get("attempt") or 0),
        )
        != (1, 6, 3)
        or source.get("run_id") != attempt["run_id"]
        or source.get("task_key") != task["task"]["key"]
        or source.get("task_version_id") != task["task"]["version_id"]
        or source.get("claim_sha256")
        != "sha256:602dd3290a2b6f6b89cde41271d4c9f9aec0489851eb581e5a00220440235f8c"
        or source.get("original_noncreditable_receipt_sha256")
        != "sha256:570a9440d8c54752c66de32345b85bc475bd6e3a0826816ce36c66adfd8717dc"
        or source.get("original_task_fence_receipt_sha256")
        != "sha256:258b88f6e78c9b8ce7f0aa52326454cea0ed54bb8ba827612a1372b38e83c065"
        or any(
            not isinstance(source.get(field), str)
            or not source[field].startswith("sha256:")
            or len(source[field]) != 71
            for field in digest_fields
        )
        or any(
            not isinstance(result.get(field), str)
            or not result[field].startswith("sha256:")
            or len(result[field]) != 71
            for field in result_digest_fields
        )
        or result.get("session_id")
        != "bdfa0c8b-6b40-4820-8d34-69c20fc2e686"
        or result.get("verifier_execution_id")
        != "b461aa17-3c60-44b4-bfc0-43b308b674f6"
        or result.get("agent_exit_code") != 1
        or result.get("agent_termination") != "completed"
        or result.get("reward_numeric_present") is not True
        or result.get("reward_result_verifier_matches_result") is not True
        or result.get("reward_result_task_version_matches_result") is not True
        or result.get("session_ingest_status") != "completed"
        or result.get("session_ingest_chunks_complete") is not True
        or result.get("cleanup_completed") is not True
        or api.get("session_match_count") != 1
        or api.get("status") != "completed"
        or api.get("model") != "glm-5.3"
        or api.get("verifier_execution_id") != result.get("verifier_execution_id")
        or api.get("task_version_projection_omitted") is not True
        or api.get("run_id_projection_omitted") is not True
        or api.get("present_projection_contradiction") is not False
        or exclusions
        != {
            "stderr_bytes": 0,
            "proxy_http_5xx_token_count": 0,
            "proxy_transport_error_token_count": 0,
            "independent_infrastructure_incident": False,
            "endpoint_or_controller_preemption_overlap": False,
        }
        or decision.get("agent_exit_code_was_only_prior_rejection") is not True
        or decision.get("original_receipts_mutated") is not False
        or decision.get("source6_attempt3_must_not_repeat") is not True
        or decision.get("source6_attempt4_was_never_started") is not True
        or decision.get("gap_plan_may_run_only_source6_attempt4") is not True
        or decision.get("same_dedicated_a_v5_treatment_required") is not True
        or decision.get("fresh_create_once_identity_required") is not True
        or decision.get("scored_gap_launch_authorized") is not False
        or observed_accepted != expected_accepted
        or len(accepted) != 2
        or any(value is not False for value in privacy.values())
    ):
        raise ValueError("dedicated A completed exit-1 reconciliation drifted")


def build_dedicated_a_completed_exit1_gap_plan(
    predecessor: dict[str, Any], reconciliation: dict[str, Any]
) -> dict[str, Any]:
    """Build the exact source6/a4-only continuation under dedicated-A v5."""
    validate_dedicated_a_completed_exit1_reconciliation(reconciliation, predecessor)
    task = copy.deepcopy(
        next(row for row in predecessor["tasks"] if int(row["source_rank"]) == 6)
    )
    task["rank"] = 1
    accepted = sorted(reconciliation["already_accepted"], key=lambda row: row["attempt"])
    credits = [
        {
            "rank": 1,
            "source_rank": 6,
            "attempt": int(row["attempt"]),
            "session_id": row["session_id"],
            "verifier_execution_id": row["verifier_execution_id"],
            "source_run_id": row["run_id"],
            "source_receipt_sha256": row["receipt_sha256"],
            "classification": "ACCEPTED",
        }
        for row in accepted
    ]
    source = reconciliation["source"]
    result = reconciliation["authoritative_result"]
    credits.append(
        {
            "rank": 1,
            "source_rank": 6,
            "attempt": 3,
            "session_id": result["session_id"],
            "verifier_execution_id": result["verifier_execution_id"],
            "source_run_id": source["run_id"],
            "source_receipt_sha256": reconciliation["receipt_sha256"],
            "classification": "RECONCILED_ACCEPTED",
        }
    )
    key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[1][:8]
    attempt = {
        "ordinal": 1,
        "rank": 1,
        "source_rank": 6,
        "attempt": 4,
        "run_id": f"{DEDICATED_A_COMPLETED_EXIT1_GAP_CAMPAIGN}-sr006-a4-{key_digest}",
        "network": f"glm53-dedicated-a-v5-gap-sr006-a4-{key_digest}",
    }
    execution = copy.deepcopy(predecessor["execution"])
    execution.update(
        {
            "inventory_policy": "immutable_plan_claim_and_endpoint_uid_v1",
            "retry_policy": "never_repeat_any_authoritative_scored_outcome",
            "future_nonzero_exit_policy": (
                "credit_only_if_reward_ingest_cleanup_and_authoritative_session_match"
            ),
            "required_priority_class": "fleet-train-high",
            "launch_authorized": False,
        }
    )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": "glm53_dedicated_a_completed_exit1_gap",
        "campaign_id": DEDICATED_A_COMPLETED_EXIT1_GAP_CAMPAIGN,
        "source_job_id": predecessor["source_job_id"],
        "source": {
            "predecessor_plan_sha256": predecessor["plan_sha256"],
            "predecessor_job_uid": source["job_uid"],
            "predecessor_pod_uid": source["pod_uid"],
            "reconciliation_receipt_sha256": reconciliation["receipt_sha256"],
            "original_noncreditable_receipt_sha256": source[
                "original_noncreditable_receipt_sha256"
            ],
            "original_task_fence_receipt_sha256": source[
                "original_task_fence_receipt_sha256"
            ],
        },
        "treatment_block": copy.deepcopy(predecessor["treatment_block"]),
        "model": copy.deepcopy(predecessor["model"]),
        "harness": copy.deepcopy(predecessor["harness"]),
        "authority": copy.deepcopy(predecessor["authority"]),
        "task_count": 1,
        "pass_k": 4,
        "total_session_count": 4,
        "credited_sessions": credits,
        "new_session_count": 1,
        "primary_denominator": {"tasks": 100, "cells": 400},
        "primary_denominator_restored_only_after_gap_task_pass4": True,
        "execution": execution,
        "tasks": [task],
        "attempts": [attempt],
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


def build_completed_exit1_gap_plan(
    qwen_plan: dict[str, Any],
    glm_plan: dict[str, Any],
    reconciliation: dict[str, Any],
    source: dict[str, Any],
    model_block: str,
) -> dict[str, Any]:
    """Build a create-once, same-task plan for only skipped exit-1 gaps."""
    validate_completed_exit1_reconciliation(reconciliation, qwen_plan, glm_plan)
    if (
        source.get("schema_version") != "fleet-completed-exit1-gap-source-v1"
        or source.get("receipt_sha256") != digest_without(source, "receipt_sha256")
        or source.get("reconciliation_receipt_sha256")
        != reconciliation["receipt_sha256"]
        or source.get("scores_read") is not False
        or source.get("prompts_or_traces_read") is not False
        or source.get("replacement_addons_scored_claims") != 0
        or source.get("primary_denominators")
        != {"qwen": {"tasks": 50, "cells": 200}, "glm": {"tasks": 100, "cells": 400}}
        or source.get("primary_denominator_restored_only_after_all_gap_tasks_pass4")
        is not True
    ):
        raise ValueError("completed exit-1 gap source drifted")
    settings = {
        "qwen_hosted_v8": {
            "plan": qwen_plan,
            "shard": "qwen38_completed_exit1_gap",
            "campaign": QWEN_COMPLETED_EXIT1_GAP_CAMPAIGN,
            "source_ranks": [6, 7],
            "network": "qwen38-hosted-gap-v2",
            "accepted_cell": (6, 1),
            "accepted_receipt_sha256": (
                "sha256:c6b1dea26588f2511d8dd5b7463e9be23de1f3723f7586f8a692e262ebc2c76a"
            ),
            "held_replacement": 56,
            "primary_denominator": {"tasks": 50, "cells": 200},
        },
        "glm_hosted_v12": {
            "plan": glm_plan,
            "shard": "glm53_completed_exit1_gap",
            "campaign": GLM_COMPLETED_EXIT1_GAP_CAMPAIGN,
            "source_ranks": [13, 15],
            "network": "glm53-hosted-gap-v2",
            "accepted_cell": (13, 1),
            "accepted_receipt_sha256": (
                "sha256:1c6e404b9edca1a8ab78303c8137a4cac55eb3bfa6c94db1a473ac9512425637"
            ),
            "held_replacement": 111,
            "primary_denominator": {"tasks": 100, "cells": 400},
        },
    }
    if model_block not in settings:
        raise ValueError("unsupported completed exit-1 model block")
    selected = settings[model_block]
    predecessor = selected["plan"]
    source_job = (source.get("source_jobs") or {}).get(model_block) or {}
    if (
        source_job.get("plan_sha256") != predecessor["plan_sha256"]
        or not source_job.get("job_uid")
        or not source_job.get("pod_uid")
    ):
        raise ValueError("completed exit-1 source job binding drifted")
    try:
        uuid.UUID(source_job["job_uid"])
        uuid.UUID(source_job["pod_uid"])
    except ValueError as exc:
        raise ValueError("completed exit-1 source job UID drifted") from exc

    task_by_source = {
        int(task["source_rank"]): task for task in predecessor["tasks"]
    }
    tasks: list[dict[str, Any]] = []
    for source_rank in selected["source_ranks"]:
        task = copy.deepcopy(task_by_source[source_rank])
        task["rank"] = len(tasks) + 1
        tasks.append(task)

    accepted_rows = [
        row
        for row in source.get("accepted_cells") or []
        if row.get("model_block") == model_block
    ]
    if len(accepted_rows) != 1:
        raise ValueError("completed exit-1 accepted source cell drifted")
    accepted = accepted_rows[0]
    accepted_receipt = accepted.get("receipt") or {}
    source_rank, attempt = selected["accepted_cell"]
    predecessor_attempt = next(
        row
        for row in predecessor["attempts"]
        if int(row["source_rank"]) == source_rank
        and int(row["attempt"]) == attempt
    )
    if (
        accepted.get("source_rank") != source_rank
        or accepted.get("attempt") != attempt
        or accepted_receipt.get("receipt_sha256")
        != digest_without(accepted_receipt, "receipt_sha256")
        or accepted_receipt.get("receipt_sha256")
        != selected["accepted_receipt_sha256"]
        or accepted_receipt.get("run_id") != predecessor_attempt["run_id"]
        or accepted_receipt.get("source_rank") != source_rank
        or accepted_receipt.get("attempt") != attempt
        or accepted_receipt.get("task_version_id")
        != task_by_source[source_rank]["task"]["version_id"]
        or accepted_receipt.get("accepted") is not True
        or accepted_receipt.get("credited") is not True
    ):
        raise ValueError("completed exit-1 accepted receipt drifted")
    try:
        uuid.UUID(accepted_receipt["session_id"])
        uuid.UUID(accepted_receipt["verifier_execution_id"])
    except (KeyError, ValueError) as exc:
        raise ValueError("completed exit-1 accepted receipt UUID drifted") from exc

    reconciled_rows = [
        row
        for row in reconciliation["reconciled_cells"]
        if row["model_block"] == model_block
    ]
    rank_by_source = {int(task["source_rank"]): int(task["rank"]) for task in tasks}
    credits = [
        {
            "rank": rank_by_source[source_rank],
            "source_rank": source_rank,
            "attempt": attempt,
            "session_id": accepted_receipt["session_id"],
            "verifier_execution_id": accepted_receipt["verifier_execution_id"],
            "source_run_id": accepted_receipt["run_id"],
            "source_receipt_sha256": accepted_receipt["receipt_sha256"],
            "classification": "ACCEPTED",
        }
    ]
    for row in reconciled_rows:
        credits.append(
            {
                "rank": rank_by_source[int(row["source_rank"])],
                "source_rank": int(row["source_rank"]),
                "attempt": int(row["attempt"]),
                "session_id": row["session_id"],
                "verifier_execution_id": row["verifier_execution_id"],
                "source_run_id": row["run_id"],
                "source_receipt_sha256": reconciliation["receipt_sha256"],
                "classification": "RECONCILED_ACCEPTED",
            }
        )
    credits.sort(key=lambda row: (int(row["rank"]), int(row["attempt"])))
    credited_cells = {(int(row["rank"]), int(row["attempt"])) for row in credits}
    attempts: list[dict[str, Any]] = []
    for task in tasks:
        key_digest = self_hosted.sha256(task["task"]["key"].encode()).split(":", 1)[
            1
        ][:8]
        for gap_attempt in range(1, 5):
            if (int(task["rank"]), gap_attempt) in credited_cells:
                continue
            attempts.append(
                {
                    "ordinal": len(attempts) + 1,
                    "rank": int(task["rank"]),
                    "source_rank": int(task["source_rank"]),
                    "attempt": gap_attempt,
                    "run_id": (
                        f"{selected['campaign']}-sr{int(task['source_rank']):03d}"
                        f"-a{gap_attempt}-{key_digest}"
                    ),
                    "network": (
                        f"{selected['network']}-sr{int(task['source_rank']):03d}"
                        f"-a{gap_attempt}-{key_digest}"
                    ),
                }
            )
    plan = {
        "schema_version": PLAN_SCHEMA,
        "shard_key": selected["shard"],
        "campaign_id": selected["campaign"],
        "source_job_id": predecessor["source_job_id"],
        "source": {
            "predecessor_plan_sha256": predecessor["plan_sha256"],
            "reconciliation_receipt_sha256": reconciliation["receipt_sha256"],
            "gap_source_receipt_sha256": source["receipt_sha256"],
            "source_job_name": source_job["job_name"],
            "source_job_uid": source_job["job_uid"],
            "source_pod_uid": source_job["pod_uid"],
            "held_unused_replacement_rank": selected["held_replacement"],
        },
        "treatment_block": copy.deepcopy(predecessor["treatment_block"]),
        "model": copy.deepcopy(predecessor["model"]),
        "harness": copy.deepcopy(predecessor["harness"]),
        "authority": copy.deepcopy(predecessor["authority"]),
        "task_count": 2,
        "pass_k": 4,
        "total_session_count": 8,
        "credited_sessions": credits,
        "new_session_count": len(attempts),
        "primary_denominator": selected["primary_denominator"],
        "primary_denominator_restored_only_after_all_gap_tasks_pass4": True,
        "excluded_tasks": [],
        "execution": {
            **copy.deepcopy(predecessor["execution"]),
            "inventory_policy": (
                "plan_identity_plus_authoritative_receipt_v1"
                if model_block == "qwen_hosted_v8"
                else "conservative_no_same_model_session_for_task_key_v1"
            ),
            "retry_policy": "never_repeat_any_authoritative_scored_outcome",
            "future_nonzero_exit_policy": (
                "credit_only_if_reward_ingest_cleanup_and_authoritative_session_match"
            ),
            "required_priority_class": "fleet-train-high",
            "launch_authorized": False,
        },
        "tasks": tasks,
        "attempts": attempts,
        "privacy": copy.deepcopy(predecessor["privacy"]),
    }
    if model_block == "qwen_hosted_v8":
        plan["source"]["preflight_v1_failure_receipt_sha256"] = (
            "sha256:9b294a1f55059f3ba12670b060caf86ed42d33a84eb4b462e29f05f521aa3b8b"
        )
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


def build_qwen_post_partial_tail_plans(
    predecessor: dict[str, Any], incident: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Split only the untouched Qwen v8 tail into two disjoint whole-task shards."""
    validate_plan(predecessor)
    if (
        predecessor.get("shard_key") != "qwen38_http500_successor"
        or predecessor.get("plan_sha256")
        != "sha256:8d6df6af63b308d648fb0c7ea9115f90b16de1d7e57b0968dda8fc8fd4a20c81"
        or incident.get("schema_version")
        != "fleet-qwen38-v8-partial-ingest-terminal-v1"
        or incident.get("receipt_sha256") != QWEN_POST_PARTIAL_INCIDENT_DIGEST
        or incident.get("receipt_sha256") != digest_without(incident, "receipt_sha256")
        or (incident.get("controller") or {}).get("job_uid")
        != "84727b82-5187-425b-ba4b-9598f46691c4"
        or (incident.get("controller") or {}).get("pod_uid")
        != "42df9b32-60c2-45aa-8ce3-ba850b82b909"
        or (incident.get("classification") or {}).get(
            "same_session_suffix_resume_pending"
        )
        is not True
        or (incident.get("classification") or {}).get("replacement_selection_allowed")
        is not False
    ):
        raise ValueError("Qwen post-partial predecessor evidence drifted")
    predecessor_tasks = {
        int(row["source_rank"]): row for row in predecessor.get("tasks") or []
    }
    predecessor_attempts = {
        (int(row["source_rank"]), int(row["attempt"])): row
        for row in predecessor.get("attempts") or []
    }
    partitions = {
        "qwen38_post_partial_tail_a": list(range(11, 33)),
        "qwen38_post_partial_tail_b": [*range(33, 51), 52, 53, 54, 55],
    }
    plans: dict[str, dict[str, Any]] = {}
    for shard_key, source_ranks in partitions.items():
        campaign = QWEN_POST_PARTIAL_TAIL_CAMPAIGNS[shard_key]
        tasks: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        ordinal = 0
        for rank, source_rank in enumerate(source_ranks, 1):
            source_task = copy.deepcopy(predecessor_tasks[source_rank])
            source_task["rank"] = rank
            tasks.append(source_task)
            for attempt in range(1, 5):
                ordinal += 1
                old = predecessor_attempts[(source_rank, attempt)]
                suffix = str(old["run_id"]).rsplit("-", 1)[-1]
                attempts.append(
                    {
                        "ordinal": ordinal,
                        "rank": rank,
                        "source_rank": source_rank,
                        "attempt": attempt,
                        "run_id": (
                            f"{campaign}-sr{source_rank:03d}-a{attempt}-{suffix}"
                        ),
                        "network": (
                            f"q38-tail-{shard_key[-1]}-sr{source_rank:03d}"
                            f"-a{attempt}-{suffix}"
                        ),
                    }
                )
        execution = copy.deepcopy(predecessor["execution"])
        execution.update(
            {
                "inventory_policy": "plan_identity_plus_authoritative_receipt_v1",
                "launch_authorized": False,
                "required_priority_class": "fleet-train-high",
                "max_hosted_streams_for_model": 2,
                "pairwise_tail_task_and_cell_disjointness_required": True,
                "source10_recovery_and_attempt4_gap_required": True,
                "future_nonzero_exit_policy": (
                    "credit_only_if_reward_ingest_cleanup_and_authoritative_session_match"
                ),
            }
        )
        plan = {
            "schema_version": PLAN_SCHEMA,
            "shard_key": shard_key,
            "campaign_id": campaign,
            "source_job_id": predecessor["source_job_id"],
            "pass_k": 4,
            "task_count": 22,
            "new_session_count": 88,
            "total_session_count": 88,
            "source": {
                "predecessor_plan_sha256": predecessor["plan_sha256"],
                "predecessor_job_uid": incident["controller"]["job_uid"],
                "predecessor_pod_uid": incident["controller"]["pod_uid"],
                "predecessor_terminal_at": incident["controller"]["finished_at"],
                "partial_ingest_incident_receipt_sha256": incident["receipt_sha256"],
                "source10_attempt3_session_id": incident["attempt_evidence"][
                    "session_id"
                ],
                "source10_attempt3_verifier_execution_id": incident[
                    "attempt_evidence"
                ]["verifier_execution_id"],
                "source10_attempt4_unstarted": True,
                "peer_tail_shard": (
                    "qwen38_post_partial_tail_b"
                    if shard_key.endswith("_a")
                    else "qwen38_post_partial_tail_a"
                ),
            },
            "authority": copy.deepcopy(predecessor["authority"]),
            "model": copy.deepcopy(predecessor["model"]),
            "harness": copy.deepcopy(predecessor["harness"]),
            "execution": execution,
            "treatment_block": copy.deepcopy(predecessor["treatment_block"]),
            "privacy": copy.deepcopy(predecessor["privacy"]),
            "tasks": tasks,
            "attempts": attempts,
            "credited_sessions": [],
            "deferred_source_ranks": [10],
            "held_unused_replacement_ranks": [56],
            "primary_denominator": {"tasks": 50, "cells": 200},
            "primary_denominator_restored_only_after_source10_recovery_and_gap": True,
        }
        plan["plan_sha256"] = digest_without(plan, "plan_sha256")
        validate_plan(plan)
        plans[shard_key] = plan
    left = {
        (int(row["source_rank"]), int(row["attempt"]))
        for row in plans["qwen38_post_partial_tail_a"]["attempts"]
    }
    right = {
        (int(row["source_rank"]), int(row["attempt"]))
        for row in plans["qwen38_post_partial_tail_b"]["attempts"]
    }
    if left & right or len(left | right) != 176:
        raise ValueError("Qwen post-partial tail shards overlap")
    return plans


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
        in {
            "glm53_dedicated_a",
            "glm53_dedicated_b",
            "glm53_dedicated_b_v5",
            "glm53_dedicated_a_v5",
            "glm53_dedicated_a_completed_exit1_gap",
        }
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
            "glm53_attrition_replacement",
            "glm53_completed_exit1_gap",
        }
        else (
            "plan_identity_plus_authoritative_receipt_v1"
            if shard_key
            in {
                "qwen38_remainder",
                "qwen38_remainder2",
                "qwen38_replacements",
                "qwen38_http500_successor",
                "qwen38_attrition_replacement",
                "qwen38_completed_exit1_gap",
                "qwen38_post_partial_tail_a",
                "qwen38_post_partial_tail_b",
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
    if shard_key in QWEN_POST_PARTIAL_TAIL_CAMPAIGNS:
        is_a = shard_key.endswith("_a")
        expected_sources = list(range(11, 33)) if is_a else [*range(33, 51), 52, 53, 54, 55]
        source = plan.get("source") or {}
        treatment = plan.get("treatment_block") or {}
        if (
            plan.get("campaign_id") != QWEN_POST_PARTIAL_TAIL_CAMPAIGNS[shard_key]
            or [int(row["source_rank"]) for row in tasks] != expected_sources
            or credits
            or len(attempts) != 88
            or source.get("predecessor_plan_sha256")
            != "sha256:8d6df6af63b308d648fb0c7ea9115f90b16de1d7e57b0968dda8fc8fd4a20c81"
            or source.get("predecessor_job_uid")
            != "84727b82-5187-425b-ba4b-9598f46691c4"
            or source.get("predecessor_pod_uid")
            != "42df9b32-60c2-45aa-8ce3-ba850b82b909"
            or source.get("partial_ingest_incident_receipt_sha256")
            != QWEN_POST_PARTIAL_INCIDENT_DIGEST
            or source.get("source10_attempt3_session_id")
            != "579cc47e-3b00-4868-894b-8def06761b1e"
            or source.get("peer_tail_shard")
            != ("qwen38_post_partial_tail_b" if is_a else "qwen38_post_partial_tail_a")
            or plan.get("deferred_source_ranks") != [10]
            or plan.get("held_unused_replacement_ranks") != [56]
            or treatment.get("kind") != "hosted_inference_endpoint_v1"
            or treatment.get("model_revision") != plan["model"]["revision"]
            or treatment.get("endpoint_origin") != plan["model"]["endpoint_origin"]
            or treatment.get("served_id") != plan["model"]["served_id"]
            or treatment.get("session_model") != plan["model"]["session_model"]
            or treatment.get("harness") != plan["harness"]
            or treatment.get("required_task_tools") != ["bash", "submit_report"]
            or treatment.get("required_task_tool_catalog_sha256")
            != execution["required_task_tool_catalog_sha256"]
            or execution.get("required_priority_class") != "fleet-train-high"
            or execution.get("launch_authorized") is not False
            or execution.get("max_hosted_streams_for_model") != 2
            or execution.get("future_nonzero_exit_policy")
            != "credit_only_if_reward_ingest_cleanup_and_authoritative_session_match"
            or plan.get("primary_denominator") != {"tasks": 50, "cells": 200}
            or plan.get(
                "primary_denominator_restored_only_after_source10_recovery_and_gap"
            )
            is not True
        ):
            raise ValueError("Qwen post-partial untouched tail binding drifted")
        return
    if shard_key in {"qwen38_completed_exit1_gap", "glm53_completed_exit1_gap"}:
        is_qwen = shard_key.startswith("qwen38")
        expected_source_ranks = [6, 7] if is_qwen else [13, 15]
        expected_new_cells = (
            {(6, 3), (6, 4), (7, 2), (7, 3), (7, 4)}
            if is_qwen
            else {(13, 3), (13, 4), (15, 2), (15, 3), (15, 4)}
        )
        expected_credit_cells = (
            {(6, 1), (6, 2), (7, 1)}
            if is_qwen
            else {(13, 1), (13, 2), (15, 1)}
        )
        expected_accepted_receipt = (
            "sha256:c6b1dea26588f2511d8dd5b7463e9be23de1f3723f7586f8a692e262ebc2c76a"
            if is_qwen
            else "sha256:1c6e404b9edca1a8ab78303c8137a4cac55eb3bfa6c94db1a473ac9512425637"
        )
        expected_primary = {"tasks": 50, "cells": 200} if is_qwen else {
            "tasks": 100,
            "cells": 400,
        }
        source = plan.get("source") or {}
        if (
            [int(row["source_rank"]) for row in tasks] != expected_source_ranks
            or {(int(row["source_rank"]), int(row["attempt"])) for row in attempts}
            != expected_new_cells
            or {(int(row["source_rank"]), int(row["attempt"])) for row in credits}
            != expected_credit_cells
            or len(attempts) != 5
            or len(credits) != 3
            or plan.get("campaign_id")
            not in (
                {
                    QWEN_COMPLETED_EXIT1_GAP_CAMPAIGN,
                    "chris-cyber-q38-opencode11827-hosted-gap-q6q7-p4-v1",
                }
                if is_qwen
                else {
                    GLM_COMPLETED_EXIT1_GAP_CAMPAIGN,
                    "chris-cyber-glm53-opencode11827-hosted-gap-g13g15-p4-v1",
                }
            )
            or len(
                [
                    row
                    for row in credits
                    if row.get("classification") == "ACCEPTED"
                    and row.get("source_receipt_sha256")
                    == expected_accepted_receipt
                ]
            )
            != 1
            or len(
                [
                    row
                    for row in credits
                    if row.get("classification") == "RECONCILED_ACCEPTED"
                    and row.get("source_receipt_sha256")
                    == "sha256:5435c0a25b3be1872c1aae09fd88858c5251272e164ab8eb49ab66b4c5a4ca0e"
                ]
            )
            != 2
            or source.get("reconciliation_receipt_sha256")
            != "sha256:5435c0a25b3be1872c1aae09fd88858c5251272e164ab8eb49ab66b4c5a4ca0e"
            or source.get("held_unused_replacement_rank") != (56 if is_qwen else 111)
            or (
                is_qwen
                and plan.get("campaign_id") == QWEN_COMPLETED_EXIT1_GAP_CAMPAIGN
                and source.get("preflight_v1_failure_receipt_sha256")
                != "sha256:9b294a1f55059f3ba12670b060caf86ed42d33a84eb4b462e29f05f521aa3b8b"
            )
            or plan.get("primary_denominator") != expected_primary
            or plan.get("primary_denominator_restored_only_after_all_gap_tasks_pass4")
            is not True
            or execution.get("required_priority_class") != "fleet-train-high"
            or execution.get("launch_authorized") is not False
            or execution.get("retry_policy")
            != "never_repeat_any_authoritative_scored_outcome"
            or execution.get("future_nonzero_exit_policy")
            != "credit_only_if_reward_ingest_cleanup_and_authoritative_session_match"
            or plan.get("treatment_block")
            != (
                plan.get("treatment_block")
                | {
                    "kind": "hosted_inference_endpoint_v1",
                    "model_revision": plan["model"]["revision"],
                    "endpoint_origin": plan["model"]["endpoint_origin"],
                    "served_id": plan["model"]["served_id"],
                    "session_model": plan["model"]["session_model"],
                    "harness": plan["harness"],
                    "required_task_tools": ["bash", "submit_report"],
                    "required_task_tool_catalog_sha256": execution[
                        "required_task_tool_catalog_sha256"
                    ],
                }
            )
        ):
            raise ValueError("completed exit-1 gap binding drifted")
        return
    if shard_key == "glm53_dedicated_a_completed_exit1_gap":
        source = plan.get("source") or {}
        treatment = plan.get("treatment_block") or {}
        expected_credits = {
            (
                6,
                1,
                "ACCEPTED",
                "sha256:76dbba0ccfd6fe4190c532712d25df48b76107067b3b01096589cc18a48ad537",
            ),
            (
                6,
                2,
                "ACCEPTED",
                "sha256:5bcf23b2f9ec5625cc089acba25d89d3bc0590d5e4321f489031c0b217290790",
            ),
            (
                6,
                3,
                "RECONCILED_ACCEPTED",
                DEDICATED_A_COMPLETED_EXIT1_RECONCILIATION_DIGEST,
            ),
        }
        observed_credits = {
            (
                int(row["source_rank"]),
                int(row["attempt"]),
                row.get("classification"),
                row.get("source_receipt_sha256"),
            )
            for row in credits
        }
        if (
            plan.get("campaign_id") != DEDICATED_A_COMPLETED_EXIT1_GAP_CAMPAIGN
            or [int(row["source_rank"]) for row in tasks] != [6]
            or {(int(row["source_rank"]), int(row["attempt"])) for row in attempts}
            != {(6, 4)}
            or observed_credits != expected_credits
            or len(credits) != 3
            or len(attempts) != 1
            or source.get("predecessor_plan_sha256")
            != "sha256:e1f37476700beb16ccb3978f61a162a59e16501334d316417cd0718a94c2ef68"
            or source.get("predecessor_job_uid")
            != "87e2ca85-d684-4d87-9f05-f4399b7906d0"
            or source.get("predecessor_pod_uid")
            != "083a29e3-7d0e-4cbb-9447-62373772080d"
            or source.get("reconciliation_receipt_sha256")
            != DEDICATED_A_COMPLETED_EXIT1_RECONCILIATION_DIGEST
            or source.get("original_noncreditable_receipt_sha256")
            != "sha256:570a9440d8c54752c66de32345b85bc475bd6e3a0826816ce36c66adfd8717dc"
            or source.get("original_task_fence_receipt_sha256")
            != "sha256:258b88f6e78c9b8ce7f0aa52326454cea0ed54bb8ba827612a1372b38e83c065"
            or treatment.get("kind") != "dedicated_inference_endpoint_v1"
            or treatment.get("replica") != "A"
            or treatment.get("serving_generation") != "v5"
            or treatment.get("ray_job_uid")
            != "ef7cb0f2-84d4-4017-ae31-bf34ebb70d0d"
            or treatment.get("ray_cluster_uid")
            != "5107d72e-ae6a-4582-a490-55b349421d06"
            or treatment.get("service_uid")
            != "2c0e64de-c4a0-4f70-ae07-d15b1adad0b3"
            or treatment.get("head_pod_uid")
            != "3f37ab91-4afa-4e4d-8f29-a3eeda70a774"
            or treatment.get("endpoint_origin") != plan["model"]["endpoint_origin"]
            or treatment.get("model_revision") != plan["model"]["revision"]
            or treatment.get("served_id") != plan["model"]["served_id"]
            or treatment.get("session_model") != plan["model"]["session_model"]
            or treatment.get("harness") != plan["harness"]
            or treatment.get("required_task_tools") != ["bash", "submit_report"]
            or plan.get("primary_denominator") != {"tasks": 100, "cells": 400}
            or plan.get("primary_denominator_restored_only_after_gap_task_pass4")
            is not True
            or execution.get("required_priority_class") != "fleet-train-high"
            or execution.get("launch_authorized") is not False
            or execution.get("retry_policy")
            != "never_repeat_any_authoritative_scored_outcome"
            or execution.get("future_nonzero_exit_policy")
            != "credit_only_if_reward_ingest_cleanup_and_authoritative_session_match"
        ):
            raise ValueError("dedicated A completed exit-1 gap binding drifted")
        return
    if shard_key in {"qwen38_attrition_replacement", "glm53_attrition_replacement"}:
        is_qwen = shard_key.startswith("qwen38")
        source = plan.get("source") or {}
        treatment = plan.get("treatment_block") or {}
        expected_rank = 56 if is_qwen else 111
        expected_fence = 6 if is_qwen else 13
        expected_revision = (
            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
            if is_qwen
            else "30333038ada1f1dacb294a93270305a890b50c14"
        )
        if (
            [int(row["source_rank"]) for row in tasks] != [expected_rank]
            or source.get("fenced_source_rank") != expected_fence
            or not source.get("accepted_attrition_receipt_sha256")
            or not source.get("noncreditable_receipt_sha256")
            or treatment.get("kind") != "hosted_inference_endpoint_v1"
            or treatment.get("endpoint_origin") != "https://inference.flt.build"
            or treatment.get("model_revision") != expected_revision
            or treatment.get("model_revision") != plan["model"].get("revision")
            or treatment.get("harness") != plan.get("harness")
            or treatment.get("required_task_tools") != ["bash", "submit_report"]
            or plan.get("primary_estimator_task_count") != (50 if is_qwen else 100)
            or plan.get("primary_estimator_cell_count") != (200 if is_qwen else 400)
            or execution.get("launch_authorized") is not False
            or execution.get("required_priority_class") != "fleet-train-high"
        ):
            raise ValueError("hosted attrition replacement binding drifted")
        return
    if shard_key in {"glm53_dedicated_a_v5", "glm53_dedicated_b_v5"}:
        treatment = plan.get("treatment_block") or {}
        source = plan.get("source") or {}
        selected = {int(row["source_rank"]) for row in tasks}
        is_a = shard_key == "glm53_dedicated_a_v5"
        expected_selected = (
            {*range(6, 53, 2), 102, 104, 110}
            if is_a
            else {*range(56, 101, 2), 101, 103, 105, 107}
        )
        expected_fenced = (
            {1, 2, 3, 4, 5, 7, 9, 11, 54, 106}
            if is_a
            else {1, 2, 3, 5, 7, 9, 11, 54, 106}
        )
        stop_field = (
            "corrected_stop_tombstone_receipt_sha256"
            if is_a
            else "forced_stop_tombstone_receipt_sha256"
        )
        if (
            selected != expected_selected
            or set(plan.get("fenced_source_ranks") or [])
            != expected_fenced
            or set(plan.get("hosted_source_ranks") or [])
            != {*range(13, 100, 2), 108, 109}
            or (
                set(plan.get("dedicated_b_source_ranks") or [])
                != {*range(56, 101, 2), 101, 103, 105, 107}
                if is_a
                else set(plan.get("dedicated_a_reserved_source_ranks") or [])
                != {*range(6, 53, 2), 102, 104}
            )
            or (not is_a and plan.get("dedicated_a_pending_replacement_ranks") != [110])
            or treatment.get("kind") != "dedicated_inference_endpoint_v1"
            or treatment.get("replica") != ("A" if is_a else "B")
            or treatment.get("serving_generation") != "v5"
            or treatment.get("model_revision") != plan["model"].get("revision")
            or treatment.get("endpoint_origin") != plan["model"].get("endpoint_origin")
            or treatment.get(stop_field) != source.get(stop_field)
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
            raise ValueError("dedicated v5 partition or binding drifted")
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


def _sealed_gap_credit(plan: dict[str, Any], receipt: dict[str, Any]) -> bool:
    shard = plan.get("shard_key")
    cell = (int(receipt.get("source_rank") or 0), int(receipt.get("attempt") or 0))
    sealed: dict[str, dict[tuple[int, int], tuple[str, str]]] = {
        "qwen38_completed_exit1_gap": {
            (6, 1): (
                "ACCEPTED",
                "sha256:c6b1dea26588f2511d8dd5b7463e9be23de1f3723f7586f8a692e262ebc2c76a",
            ),
            (6, 2): (
                "RECONCILED_ACCEPTED",
                "sha256:5435c0a25b3be1872c1aae09fd88858c5251272e164ab8eb49ab66b4c5a4ca0e",
            ),
            (7, 1): (
                "RECONCILED_ACCEPTED",
                "sha256:5435c0a25b3be1872c1aae09fd88858c5251272e164ab8eb49ab66b4c5a4ca0e",
            ),
        },
        "glm53_completed_exit1_gap": {
            (13, 1): (
                "ACCEPTED",
                "sha256:1c6e404b9edca1a8ab78303c8137a4cac55eb3bfa6c94db1a473ac9512425637",
            ),
            (13, 2): (
                "RECONCILED_ACCEPTED",
                "sha256:5435c0a25b3be1872c1aae09fd88858c5251272e164ab8eb49ab66b4c5a4ca0e",
            ),
            (15, 1): (
                "RECONCILED_ACCEPTED",
                "sha256:5435c0a25b3be1872c1aae09fd88858c5251272e164ab8eb49ab66b4c5a4ca0e",
            ),
        },
        "glm53_dedicated_a_completed_exit1_gap": {
            (6, 1): (
                "ACCEPTED",
                "sha256:76dbba0ccfd6fe4190c532712d25df48b76107067b3b01096589cc18a48ad537",
            ),
            (6, 2): (
                "ACCEPTED",
                "sha256:5bcf23b2f9ec5625cc089acba25d89d3bc0590d5e4321f489031c0b217290790",
            ),
            (6, 3): (
                "RECONCILED_ACCEPTED",
                DEDICATED_A_COMPLETED_EXIT1_RECONCILIATION_DIGEST,
            ),
        },
    }
    expected = sealed.get(str(shard), {}).get(cell)
    return bool(
        expected is not None
        and receipt.get("classification") == expected[0]
        and receipt.get("source_receipt_sha256") == expected[1]
    )


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
        matching_rows = [
            row for row in rows if row.get("session_id") == receipt["session_id"]
        ]
        row = matching_rows[0] if len(matching_rows) == 1 else None
        verifier = (row or {}).get("verifier_execution") or {}
        local_receipt = receipt.get("run_id") in local_accepted
        sealed_gap_credit = _sealed_gap_credit(plan, receipt)
        projected_version = (row or {}).get("eval_task_version_id") or (
            row or {}
        ).get("task_version_id")
        projected_run_id = ((row or {}).get("metadata") or {}).get("run_id")
        if (
            row is None
            or row.get("status") != "completed"
            or verifier.get("id") != receipt["verifier_execution_id"]
            or (
                sealed_gap_credit
                and row.get("model")
                != self_hosted.persisted_session_model_identity(plan)
            )
            or (
                not local_receipt
                and not sealed_gap_credit
                and projected_version != task["task"]["version_id"]
            )
            or (
                sealed_gap_credit
                and projected_version is not None
                and projected_version != task["task"]["version_id"]
            )
            or (
                sealed_gap_credit
                and projected_run_id is not None
                and projected_run_id != receipt["source_run_id"]
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
    out_dir: Path,
    config: dict[str, Any],
    item: dict[str, Any],
    claim_sha256: str,
    key: str,
) -> dict[str, Any]:
    result = load_object(out_dir / "result.json")
    reward = load_object(out_dir / "reward-result.json")
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
            result.get("task_key") == config["task"]["key"],
            result.get("task_version_id") == config["task"]["version_id"],
            result.get("agent_termination") == "completed",
            result.get("session_ingest_status") == "completed",
            ingest.get("status") == "completed",
            ingest.get("session_id") == result.get("session_id"),
            reward.get("task_key") == config["task"]["key"],
            reward.get("task_version_id") == config["task"]["version_id"],
            reward.get("verifier_execution_id") == verifier_id,
            reward.get("instance_id") == result.get("instance_id"),
            isinstance(reward.get("reward"), (int, float)),
            not isinstance(reward.get("reward"), bool),
            cleanup == {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            },
            not any(
                path.is_file()
                and path.suffix == ".json"
                and (
                    "infrastructure" in path.name.lower()
                    or "incident" in path.name.lower()
                )
                for path in out_dir.iterdir()
            ),
        )
    )
    if not operationally_terminal:
        raise RuntimeError("hosted attempt is infrastructure-incomplete")
    authoritative_rows: list[dict[str, Any]] = []
    for attempt in range(12):
        with _client(key) as client:
            rows = self_hosted._task_sessions(client, config["task"]["key"])
        authoritative_rows = [
            row for row in rows if row.get("session_id") == result["session_id"]
        ]
        if authoritative_rows:
            break
        if attempt < 11:
            time.sleep(5)
    authoritative = authoritative_rows[0] if len(authoritative_rows) == 1 else None
    authoritative_verifier = (authoritative or {}).get("verifier_execution") or {}
    projected_version = (authoritative or {}).get("eval_task_version_id") or (
        authoritative or {}
    ).get("task_version_id")
    projected_run_id = ((authoritative or {}).get("metadata") or {}).get("run_id")
    if (
        authoritative is None
        or authoritative.get("status") != "completed"
        or authoritative.get("model")
        != self_hosted.persisted_session_model_identity(config)
        or authoritative_verifier.get("id") != verifier_id
        or (
            projected_version is not None
            and projected_version != config["task"]["version_id"]
        )
        or (projected_run_id is not None and projected_run_id != config["run_id"])
    ):
        raise RuntimeError("hosted attempt lacks authoritative scored session")
    # A completed, ingested verifier execution is an observed eval outcome.
    # The OpenCode process exit code describes the agent runtime, not whether
    # the scored session belongs in the pass@k denominator.  Conditioning
    # creditability on exit code would discard valid failures and bias the
    # estimator toward successful agent processes.
    accepted = True
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
        "agent_process_exit_success": result.get("agent_exit_code") == 0,
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
        outcome = _classify_result(
            out_dir, config, item, claim["claim_sha256"], key
        )
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
    validate_qwen_post_partial_tail_release(plan, release)
    validate_glm_http500_scoring_release(plan, release)
    validate_glm_dedicated_b_v5_scoring_release(plan, release)
    validate_glm_dedicated_a_v5_scoring_release(plan, release)
    validate_completed_exit1_gap_scoring_release(plan, release)
    validate_dedicated_a_completed_exit1_gap_release(plan, release)
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
    validate_qwen_post_partial_tail_release(plan, release)
    validate_glm_http500_scoring_release(plan, release)
    validate_glm_dedicated_b_v5_scoring_release(plan, release)
    validate_glm_dedicated_a_v5_scoring_release(plan, release)
    validate_completed_exit1_gap_scoring_release(plan, release)
    validate_dedicated_a_completed_exit1_gap_release(plan, release)
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
    dedicated_a_v5 = sub.add_parser("build-dedicated-a-v5")
    dedicated_a_v5.add_argument("--predecessor-plan", type=Path, required=True)
    dedicated_a_v5.add_argument("--tombstone", type=Path, required=True)
    dedicated_a_v5.add_argument("--supplement", type=Path, required=True)
    dedicated_a_v5.add_argument("--hydration", type=Path, required=True)
    dedicated_a_v5.add_argument("--parity", type=Path, required=True)
    dedicated_a_v5.add_argument("--output", type=Path, required=True)
    dedicated_a_gap = sub.add_parser("build-dedicated-a-exit1-gap")
    dedicated_a_gap.add_argument("--predecessor-plan", type=Path, required=True)
    dedicated_a_gap.add_argument("--reconciliation", type=Path, required=True)
    dedicated_a_gap.add_argument("--output", type=Path, required=True)
    attrition_replacement = sub.add_parser("build-hosted-attrition-replacement")
    attrition_replacement.add_argument("--predecessor-plan", type=Path, required=True)
    attrition_replacement.add_argument("--supplement", type=Path, required=True)
    attrition_replacement.add_argument("--hydration", type=Path, required=True)
    attrition_replacement.add_argument("--output", type=Path, required=True)
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
    qwen_post_partial = sub.add_parser("build-qwen-post-partial-tails")
    qwen_post_partial.add_argument("--predecessor-plan", type=Path, required=True)
    qwen_post_partial.add_argument("--incident", type=Path, required=True)
    qwen_post_partial.add_argument("--output-a", type=Path, required=True)
    qwen_post_partial.add_argument("--output-b", type=Path, required=True)
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
    if args.command == "build-dedicated-a-v5":
        value = build_glm_dedicated_a_v5_plan(
            load_object(args.predecessor_plan),
            load_object(args.tombstone),
            load_object(args.supplement),
            load_object(args.hydration),
            load_object(args.parity),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-dedicated-a-exit1-gap":
        value = build_dedicated_a_completed_exit1_gap_plan(
            load_object(args.predecessor_plan),
            load_object(args.reconciliation),
        )
        self_hosted.write_json_once(args.output, value)
        return 0
    if args.command == "build-hosted-attrition-replacement":
        value = build_hosted_attrition_replacement_plan(
            load_object(args.predecessor_plan),
            load_object(args.supplement),
            load_object(args.hydration),
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
    if args.command == "build-qwen-post-partial-tails":
        value = build_qwen_post_partial_tail_plans(
            load_object(args.predecessor_plan), load_object(args.incident)
        )
        self_hosted.write_json_once(
            args.output_a, value["qwen38_post_partial_tail_a"]
        )
        self_hosted.write_json_once(
            args.output_b, value["qwen38_post_partial_tail_b"]
        )
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
