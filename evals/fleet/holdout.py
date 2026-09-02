"""Sequential exact-version Qwen Code evaluation on the sealed Fleet test split."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx

from evals.fleet import self_hosted

HOLDOUT_SCHEMA = "fleet-qwen-code-test20-plan-v1"
RANKED50_SCHEMA = "fleet-qwen-code-ranked50-plan-v1"
RECEIPT_SCHEMA = "fleet-qwen-code-test20-receipt-v1"
DUPLICATE_PROOF_SCHEMA = "fleet-qwen38-live-session-duplicate-proof-v1"
TERMINAL_ACCEPTANCE_SCHEMA = "fleet-qwen-code-ranked50-run-acceptance-v1"
EXPECTED_SPLIT_SCHEMA = "fleet_rl_task_split_v1"
EXPECTED_TASK_COUNT = 20
RANKED50_TASK_COUNT = 50
EXPECTED_RANKED50_CYBER_CONTRACT = {
    "submission_protocol": "2.0.0",
    "evidence_schema": "1.0.0",
    "verifier_contract": "3.0.0",
}
BASE_EXECUTION_CONTROLS = {
    "pass_k": 1,
    "training_data_eligible": False,
}
OPTIONAL_EXECUTION_CONTROLS = {
    "max_concurrent",
    "required_task_tools",
    "required_task_tool_catalog_sha256",
}
SANITIZED_TASK_FILES = {
    "binding.json",
    "cleanup.json",
    "provisioning-intent.json",
    "result.json",
    "reward-result.json",
    "resource-plan.json",
    "runtime-binding.json",
    "scoring-intent.json",
    "session-ingest.json",
}


def _selected(value: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: value.get(field) for field in fields}


def _write_or_validate_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if load_json(path) != value:
            raise RuntimeError("durable sanitized receipt drifted")
        return
    self_hosted.write_json_once(path, value)


def _write_provisioning_intent(task_out: Path, config: dict[str, Any]) -> None:
    payload: dict[str, Any] = {
        "schema_version": "fleet-qwen-provisioning-intent-v1",
        "run_id": config["run_id"],
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "request_id": self_hosted.provisioning_request_id(config),
        "request_payload_sha256": self_hosted.sha256(self_hosted.canonical_json({})),
    }
    payload["intent_sha256"] = _digest_without(payload, "intent_sha256")
    self_hosted.write_json_once(task_out / "provisioning-intent.json", payload)


def _persist_sanitized_task_artifacts(
    scratch: Path, task_out: Path, config: dict[str, Any], result: dict[str, Any]
) -> None:
    raw = {
        name: load_json(scratch / name)
        for name in SANITIZED_TASK_FILES - {"provisioning-intent.json"}
    }
    binding = {
        "schema_version": config["schema_version"],
        **_selected(
            config,
            (
                "run_id",
                "source_job_id",
                "task",
                "environment",
                "verifier",
                "authority",
                "model",
                "harness",
            ),
        ),
    }
    cleanup = _selected(
        raw["cleanup.json"],
        ("instance_created", "instance_closed", "containers_removed", "terminated_at"),
    )
    resource = _selected(
        raw["resource-plan.json"],
        ("schema_version", "run_id", "instance_id", "evidence_run_id"),
    )
    resource["resource_plan_sha256"] = _digest_without(resource, "resource_plan_sha256")
    runtime = _selected(
        raw["runtime-binding.json"],
        (
            "instance_id",
            "evidence_run_id",
            "env_key",
            "environment_version",
            "environment_version_id",
            "data_key",
            "data_version",
            "tool_names",
            "tool_catalog_sha256",
        ),
    )
    scoring = _selected(
        raw["scoring-intent.json"],
        (
            "schema_version",
            "run_id",
            "task_key",
            "task_version_id",
            "instance_id",
            "evidence_run_id",
            "scoring_payload_mode",
            "request_keys",
            "request_sha256",
            "scoring_intent_sha256",
        ),
    )
    reward = _selected(
        raw["reward-result.json"],
        (
            "instance_id",
            "task_key",
            "task_version_id",
            "reward",
            "verifier_execution_id",
            "direct_authority_attestation",
        ),
    )
    session = _selected(
        raw["session-ingest.json"],
        (
            "status",
            "mode",
            "success",
            "evidence_only",
            "trace_persisted",
            "created_new_session",
            "session_id",
            "message_count",
            "chunks_completed",
            "chunk_count",
            "score",
            "model",
            "verifier_execution_id",
            "task_key",
            "task_version_id",
            "instance_id",
            "evidence_run_id",
        ),
    )
    minimized_result = _selected(
        result,
        (
            "run_id",
            "task_key",
            "task_version_id",
            "instance_id",
            "evidence_run_id",
            "session_id",
            "session_ingest_status",
            "score",
            "verifier_execution_id",
            "qwen_exit_code",
            "agent_termination",
            "elapsed_seconds",
        ),
    )
    receipts = {
        "binding.json": binding,
        "cleanup.json": cleanup,
        "result.json": minimized_result,
        "reward-result.json": reward,
        "resource-plan.json": resource,
        "runtime-binding.json": runtime,
        "scoring-intent.json": scoring,
        "session-ingest.json": session,
    }
    for name, value in receipts.items():
        _write_or_validate_once(task_out / name, value)


def _expected_task_count(schema_version: Any) -> int:
    if schema_version == HOLDOUT_SCHEMA:
        return EXPECTED_TASK_COUNT
    if schema_version == RANKED50_SCHEMA:
        return RANKED50_TASK_COUNT
    raise ValueError("unsupported holdout plan schema")


def _validate_first_task_gate(plan: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    gate = plan.get("first_task_release_gate")
    if plan.get("schema_version") != RANKED50_SCHEMA:
        if gate is not None:
            raise ValueError("first-task release gate is only supported by the ranked-50 plan")
        return
    expected = {
        "task_index": 1,
        "task_key": rows[0]["task_key"],
        "task_version_id": rows[0]["task_version_id"],
        "required_outcome_status": "model_outcome",
        "required_verifier_execution_id": "nonzero_uuid",
        "required_cleanup_verified": True,
        "required_qwen_exit_code": 0,
        "required_session_ingest_status": "completed",
        "required_session_id": "nonzero_uuid",
        "required_instance_id": "nonzero_uuid",
        "required_evidence_run_id": "nonzero_uuid",
        "required_session_matches_evidence_run_id": True,
        "accept_zero": True,
    }
    if gate != expected:
        raise ValueError("ranked-50 first-task release gate drifted")
    credential_gate = plan.get("credential_gate")
    if credential_gate != {
        "schema_version": "fleet-credential-rotation-receipt-v1",
        "secret_namespace": "fleet-train-jobs",
        "secret_name": "fleet-api",
        "secret_key": "FLEET_API_KEY",
        "rotation_not_before": "2026-09-01T23:39:05Z",
        "secret_value_must_not_be_decoded_by_submitter": True,
    }:
        raise ValueError("ranked-50 credential rotation gate drifted")


def _first_task_gate_satisfied(plan: dict[str, Any], outcome: dict[str, Any]) -> bool:
    gate = plan.get("first_task_release_gate")
    if gate is None:
        return True
    if outcome.get("index") != gate["task_index"]:
        return False
    if outcome.get("task_version_id") != gate["task_version_id"]:
        return False
    if outcome.get("status") != gate["required_outcome_status"]:
        return False
    if outcome.get("cleanup_verified") is not True:
        return False
    try:
        _validate_authoritative_model_result(
            outcome,
            {
                "task_key": gate["task_key"],
                "task_version_id": gate["task_version_id"],
            },
        )
    except RuntimeError:
        return False
    return True


def _validate_authoritative_model_result(
    result: Any, expected_task: dict[str, Any] | None = None
) -> None:
    if not isinstance(result, dict):
        raise RuntimeError("task result is not an object")
    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RuntimeError("authoritative task score is not numeric")
    if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
        raise RuntimeError("authoritative task score is outside [0, 1]")
    try:
        verifier_id = UUID(str(result.get("verifier_execution_id")))
    except ValueError as exc:
        raise RuntimeError("authoritative task verifier execution ID is invalid") from exc
    if verifier_id.int == 0:
        raise RuntimeError("authoritative task verifier execution ID is zero")
    if result.get("qwen_exit_code") != 0:
        raise RuntimeError("authoritative task Qwen process did not exit cleanly")
    if result.get("session_ingest_status") != "completed":
        raise RuntimeError("authoritative task session ingestion is incomplete")
    if expected_task is not None and (
        result.get("task_key") != expected_task["task_key"]
        or result.get("task_version_id") != expected_task["task_version_id"]
    ):
        raise RuntimeError("authoritative task identity cross-link drifted")
    try:
        instance_id = UUID(str(result.get("instance_id")))
        evidence_run_id = UUID(str(result.get("evidence_run_id")))
    except ValueError as exc:
        raise RuntimeError("authoritative runtime identity is invalid") from exc
    if instance_id.int == 0 or evidence_run_id.int == 0:
        raise RuntimeError("authoritative runtime identity is zero")
    try:
        session_id = UUID(str(result.get("session_id")))
    except ValueError as exc:
        raise RuntimeError("authoritative task session ID is invalid") from exc
    if session_id.int == 0:
        raise RuntimeError("authoritative task session ID is zero")
    if session_id != evidence_run_id:
        raise RuntimeError("authoritative task session is not bound to its evidence run")


def _validate_direct_authority_attestation(
    attestation: Any,
    *,
    config: dict[str, Any],
    instance_id: str,
    evidence_run_id: str,
    verifier_execution_id: str,
    score: float,
) -> None:
    expected = {
        "schema_version": self_hosted.DIRECT_AUTHORITY_ATTESTATION_SCHEMA,
        "context": {
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "verifier_version_id": config["verifier"]["version_id"],
            "scoring_payload_mode": self_hosted.RUNTIME_EVIDENCE_ONLY_V3,
        },
        "activity": {
            "result_schema_version": "cyber_verification_result_v3",
            "reward": float(score),
            "task_version_id": config["task"]["version_id"],
            "verifier_execution_id": verifier_execution_id,
        },
        "shadow": {
            "mode": "authoritative",
            "status": "authoritative",
            "match": True,
            "production_execution_id": verifier_execution_id,
            "direct_verifier": {
                "status": "authoritative",
                "match": True,
                "execution_id": verifier_execution_id,
                "verifier_contract_version": EXPECTED_RANKED50_CYBER_CONTRACT["verifier_contract"],
                "context_schema_version": "cyber_verification_context_v1",
            },
        },
        "data_minimization": {
            "components_included": False,
            "diagnostics_included": False,
            "evidence_payloads_included": False,
            "prompts_included": False,
            "traces_included": False,
            "flags_included": False,
        },
    }
    if attestation != expected:
        raise RuntimeError("accepted direct-authority attestation drifted")


def _validate_persisted_direct_evidence_task(
    task_dir: Path,
    config: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Fail closed on one sanitized v3 task before releasing another task."""
    artifacts = {
        name: load_json(task_dir / name)
        for name in (
            "runtime-binding.json",
            "scoring-intent.json",
            "reward-result.json",
            "result.json",
            "session-ingest.json",
        )
    }
    runtime = artifacts["runtime-binding.json"]
    scoring = artifacts["scoring-intent.json"]
    reward = artifacts["reward-result.json"]
    persisted_result = artifacts["result.json"]
    session = artifacts["session-ingest.json"]
    instance_id = runtime.get("instance_id")
    evidence_run_id = runtime.get("evidence_run_id")
    verifier_execution_id = result.get("verifier_execution_id")
    score = result.get("score")

    def score_matches(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        try:
            observed = float(value)
            expected = float(score)
        except (TypeError, ValueError):
            return False
        return math.isfinite(observed) and observed == expected

    for value, label in (
        (instance_id, "persisted instance ID"),
        (evidence_run_id, "persisted evidence-run ID"),
        (verifier_execution_id, "persisted verifier execution ID"),
    ):
        try:
            parsed = UUID(str(value))
        except ValueError as exc:
            raise RuntimeError(f"{label} is invalid") from exc
        if parsed.int == 0:
            raise RuntimeError(f"{label} is zero")
    expected_payload = {
        "instance_id": instance_id,
        "multi_app_aggregation_mode": config["authority"]["multi_app_aggregation_mode"],
        "scoring_mode": config["authority"]["scoring_mode"],
    }
    if (
        scoring.get("schema_version") != "fleet-selfhosted-scoring-intent-v1"
        or scoring.get("scoring_intent_sha256") != _digest_without(scoring, "scoring_intent_sha256")
        or scoring.get("run_id") != config["run_id"]
        or scoring.get("task_key") != config["task"]["key"]
        or scoring.get("task_version_id") != config["task"]["version_id"]
        or scoring.get("instance_id") != instance_id
        or scoring.get("evidence_run_id") != evidence_run_id
        or scoring.get("scoring_payload_mode") != self_hosted.RUNTIME_EVIDENCE_ONLY_V3
        or scoring.get("request_keys") != list(self_hosted.RUNTIME_EVIDENCE_ONLY_V3_SCORING_KEYS)
        or scoring.get("request_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(expected_payload))
    ):
        raise RuntimeError("persisted scoring request contract drifted")
    exact_identity = {
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
    }
    if any(reward.get(key) != value for key, value in exact_identity.items()) or (
        reward.get("verifier_execution_id") != verifier_execution_id
        or not score_matches(reward.get("reward"))
    ):
        raise RuntimeError("persisted authoritative reward binding drifted")
    _validate_direct_authority_attestation(
        reward.get("direct_authority_attestation"),
        config=config,
        instance_id=instance_id,
        evidence_run_id=evidence_run_id,
        verifier_execution_id=verifier_execution_id,
        score=float(score),
    )
    expected_result = {
        "run_id": config["run_id"],
        **exact_identity,
        "evidence_run_id": evidence_run_id,
        "session_id": evidence_run_id,
        "session_ingest_status": "completed",
        "verifier_execution_id": verifier_execution_id,
        "qwen_exit_code": 0,
    }
    if any(
        persisted_result.get(key) != value for key, value in expected_result.items()
    ) or not score_matches(persisted_result.get("score")):
        raise RuntimeError("persisted task result binding drifted")
    expected_session = {
        "status": "completed",
        "mode": "metadata_only_runtime_evidence_v1",
        "success": True,
        "evidence_only": True,
        "trace_persisted": False,
        "session_id": evidence_run_id,
        "evidence_run_id": evidence_run_id,
        "message_count": 0,
        "chunks_completed": 1,
        "chunk_count": 1,
        "model": f"qwen/{config['model']['served_id']}",
        "verifier_execution_id": verifier_execution_id,
        **exact_identity,
    }
    if (
        any(session.get(key) != value for key, value in expected_session.items())
        or not score_matches(session.get("score"))
        or session.get("success") is not True
        or session.get("evidence_only") is not True
        or session.get("trace_persisted") is not False
        or type(session.get("message_count")) is not int
        or type(session.get("chunks_completed")) is not int
        or type(session.get("chunk_count")) is not int
        or not isinstance(session.get("created_new_session"), bool)
    ):
        raise RuntimeError("persisted metadata-only session binding drifted")


def _safe_artifact_manifest(
    out_dir: Path,
    outcomes: list[dict[str, Any]],
    plan: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = (
        "binding.json",
        "cleanup.json",
        "provisioning-intent.json",
        "result.json",
        "reward-result.json",
        "resource-plan.json",
        "runtime-binding.json",
        "scoring-intent.json",
        "session-ingest.json",
    )
    files: list[dict[str, Any]] = []
    accepted_outcomes: list[dict[str, Any]] = []
    frozen_by_index = {row["index"]: row for row in receipt["tasks"]}
    for outcome in outcomes:
        matches = list(out_dir.glob(f"task-{outcome['index']:02d}-*"))
        if len(matches) != 1:
            raise RuntimeError("accepted task artifact directory is missing or duplicated")
        task_dir = matches[0]
        artifacts: dict[str, dict[str, Any]] = {}
        for name in required:
            path = task_dir / name
            if not path.is_file():
                raise RuntimeError("accepted task artifact manifest is incomplete")
            artifacts[name] = load_json(path)
            files.append(
                {
                    "path": str(path.relative_to(out_dir)),
                    "sha256": self_hosted.sha256(path.read_bytes()),
                }
            )
        binding = artifacts["binding.json"]
        runtime = artifacts["runtime-binding.json"]
        resource = artifacts["resource-plan.json"]
        scoring = artifacts["scoring-intent.json"]
        reward = artifacts["reward-result.json"]
        result = artifacts["result.json"]
        session = artifacts["session-ingest.json"]
        cleanup = artifacts["cleanup.json"]
        provisioning = artifacts["provisioning-intent.json"]
        frozen = frozen_by_index.get(outcome["index"])
        if frozen is None:
            raise RuntimeError("accepted task is absent from the frozen receipt")
        expected_config = task_config(plan, frozen)
        _validate_persisted_direct_evidence_task(task_dir, expected_config, outcome)
        if {path.name for path in task_dir.iterdir()} != set(required):
            raise RuntimeError("accepted task directory contains a non-sanitized artifact")
        if (
            provisioning.get("intent_sha256") != _digest_without(provisioning, "intent_sha256")
            or provisioning.get("run_id") != expected_config["run_id"]
            or provisioning.get("task_key") != expected_config["task"]["key"]
            or provisioning.get("task_version_id") != expected_config["task"]["version_id"]
            or provisioning.get("request_id")
            != self_hosted.provisioning_request_id(expected_config)
        ):
            raise RuntimeError("accepted provisioning intent is invalid")
        if resource.get("resource_plan_sha256") != _digest_without(
            resource, "resource_plan_sha256"
        ):
            raise RuntimeError("accepted resource plan digest is invalid")
        if scoring.get("scoring_intent_sha256") != _digest_without(
            scoring, "scoring_intent_sha256"
        ):
            raise RuntimeError("accepted scoring intent digest is invalid")
        run_id = result.get("run_id")
        instance_id = runtime.get("instance_id")
        evidence_run_id = runtime.get("evidence_run_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise RuntimeError("accepted instance ID is missing")
        for value, label in (
            (evidence_run_id, "accepted evidence-run ID"),
            (outcome.get("verifier_execution_id"), "accepted verifier execution ID"),
        ):
            try:
                parsed = UUID(str(value))
            except ValueError as exc:
                raise RuntimeError(f"{label} is invalid") from exc
            if parsed.int == 0:
                raise RuntimeError(f"{label} is zero")
        if (
            binding.get("run_id") != run_id
            or run_id != expected_config["run_id"]
            or binding.get("source_job_id") != expected_config["source_job_id"]
            or binding.get("task") != expected_config["task"]
            or binding.get("environment") != expected_config["environment"]
            or binding.get("verifier") != expected_config["verifier"]
            or binding.get("authority") != expected_config["authority"]
            or binding.get("model") != expected_config["model"]
            or binding.get("harness") != expected_config["harness"]
            or resource.get("run_id") != run_id
            or scoring.get("run_id") != run_id
            or scoring.get("task_key") != expected_config["task"]["key"]
            or scoring.get("task_version_id") != expected_config["task"]["version_id"]
            or resource.get("instance_id") != instance_id
            or scoring.get("instance_id") != instance_id
            or resource.get("evidence_run_id") != evidence_run_id
            or scoring.get("evidence_run_id") != evidence_run_id
            or result.get("task_key") != expected_config["task"]["key"]
            or result.get("task_version_id") != expected_config["task"]["version_id"]
            or result.get("instance_id") != instance_id
            or result.get("evidence_run_id") != evidence_run_id
        ):
            raise RuntimeError("accepted task artifact identity cross-link drifted")
        if (
            runtime.get("env_key") != expected_config["environment"]["id"]
            or runtime.get("environment_version") != expected_config["environment"]["version"]
            or runtime.get("environment_version_id") != expected_config["environment"]["version_id"]
            or runtime.get("data_key") != expected_config["environment"]["data_id"]
            or runtime.get("data_version") != expected_config["environment"]["data_version"]
            or reward.get("instance_id") != instance_id
            or reward.get("task_key") != expected_config["task"]["key"]
            or reward.get("task_version_id") != expected_config["task"]["version_id"]
            or float(reward.get("reward")) != float(outcome["score"])
            or float(result.get("score")) != float(outcome["score"])
            or reward.get("verifier_execution_id") != outcome["verifier_execution_id"]
            or result.get("verifier_execution_id") != outcome["verifier_execution_id"]
        ):
            raise RuntimeError("accepted authoritative result cross-link drifted")
        if (
            runtime.get("tool_names") != ["bash", "submit_report"]
            or runtime.get("tool_catalog_sha256")
            != plan["execution"]["required_task_tool_catalog_sha256"]
            or cleanup.get("instance_closed") is not True
            or cleanup.get("containers_removed") is not True
            or result.get("qwen_exit_code") != 0
        ):
            raise RuntimeError("accepted runtime or cleanup evidence is incomplete")
        accepted_outcomes.append(
            {
                "index": outcome["index"],
                "task_key": outcome["task_key"],
                "task_version_id": outcome["task_version_id"],
                "run_id": run_id,
                "instance_id": instance_id,
                "evidence_run_id": evidence_run_id,
                "score": outcome["score"],
                "verifier_execution_id": outcome["verifier_execution_id"],
                "session_id": result.get("session_id"),
                "session_ingest_status": session["status"],
                "tool_catalog_sha256": runtime["tool_catalog_sha256"],
                "cleanup_verified": True,
            }
        )
    manifest = {"files": files}
    for name in ("campaign-state.json", "summary.json"):
        path = out_dir / name
        if not path.is_file():
            raise RuntimeError("accepted campaign state is incomplete")
        files.append({"path": name, "sha256": self_hosted.sha256(path.read_bytes())})
    expected_root = {
        "campaign-state.json",
        "summary.json",
        *(path.parent.name for path in out_dir.glob("task-*/binding.json")),
    }
    if {path.name for path in out_dir.iterdir()} != expected_root:
        raise RuntimeError("accepted campaign root contains a non-sanitized artifact")
    manifest["manifest_sha256"] = self_hosted.sha256(self_hosted.canonical_json(manifest))
    return manifest, accepted_outcomes


def build_terminal_acceptance(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    summary: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any] | None:
    if plan.get("schema_version") != RANKED50_SCHEMA:
        return None
    if (
        summary.get("model_outcomes") != RANKED50_TASK_COUNT
        or summary.get("authoritative_verifier_backed_outcomes") != RANKED50_TASK_COUNT
        or summary.get("infrastructure_errors") != 0
        or summary.get("wave_identity_error") is not None
    ):
        return None
    outcomes = summary.get("outcomes") or []
    if len(outcomes) != RANKED50_TASK_COUNT:
        return None
    for outcome in outcomes:
        if outcome.get("status") != "model_outcome" or outcome.get("cleanup_verified") is not True:
            return None
        try:
            _validate_authoritative_model_result(outcome)
        except RuntimeError:
            return None
    observations = summary.get("wave_identity_observations") or []
    expected_waves = list(range(18))
    if [row.get("wave") for row in observations] != expected_waves:
        return None
    if any(
        row.get("observation_sha256") != _digest_without(row, "observation_sha256")
        for row in observations
    ):
        return None
    runtime_images_path = os.environ.get("FLEET_EVAL_RUNTIME_IMAGES_RECEIPT")
    if not runtime_images_path:
        raise RuntimeError("ranked-50 runtime image receipt path is required")
    runtime_images = load_json(Path(runtime_images_path))
    if runtime_images.get("schema_version") != "fleet-eval-runtime-images-v1":
        raise RuntimeError("ranked-50 runtime image receipt schema is unsupported")
    if runtime_images.get("receipt_sha256") != _digest_without(runtime_images, "receipt_sha256"):
        raise RuntimeError("ranked-50 runtime image receipt digest mismatch")
    images = runtime_images.get("images") or {}
    expected_image_refs = {
        "qwen_code": os.environ.get("QWEN_CODE_IMAGE"),
        "fixed_proxy": os.environ.get("FIXED_PROXY_IMAGE"),
    }
    if set(images) != set(expected_image_refs):
        raise RuntimeError("ranked-50 runtime image receipt entries drifted")
    for name, requested_ref in expected_image_refs.items():
        image = images[name]
        if image.get("requested_ref") != requested_ref or not requested_ref:
            raise RuntimeError("ranked-50 requested image reference drifted")
        if image.get("architecture") != "amd64":
            raise RuntimeError("ranked-50 runtime image architecture drifted")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("runtime_id"))):
            raise RuntimeError("ranked-50 runtime image identity is invalid")
    pod_uid = os.environ.get("FLEET_EVAL_POD_UID")
    try:
        parsed_pod_uid = UUID(str(pod_uid))
    except ValueError as exc:
        raise RuntimeError("ranked-50 Pod UID is invalid") from exc
    if parsed_pod_uid.int == 0:
        raise RuntimeError("ranked-50 Pod UID is zero")
    pod_identity = {
        "job_name": os.environ.get("FLEET_EVAL_JOB_NAME"),
        "pod_name": os.environ.get("FLEET_EVAL_POD_NAME"),
        "pod_uid": pod_uid,
    }
    if not all(isinstance(value, str) and value for value in pod_identity.values()):
        raise RuntimeError("ranked-50 workload identity is incomplete")
    artifact_manifest, accepted_outcomes = _safe_artifact_manifest(out_dir, outcomes, plan, receipt)
    accepted = {
        "schema_version": TERMINAL_ACCEPTANCE_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "accepted": True,
        "job_uid_binding": "requires_post_complete_observer",
        "pod_identity": pod_identity,
        "plan_sha256": self_hosted.sha256(self_hosted.canonical_json(plan)),
        "frozen_receipt_sha256": receipt["receipt_sha256"],
        "fleet_account": receipt["fleet_account"],
        "selection_sha256": plan["selection"]["selection_sha256"],
        "prior_attempt_exclusions_sha256": plan["prior_attempt_exclusions"]["receipt_sha256"],
        "duplicate_preflight_sha256": receipt["duplicate_preflight"]["proof_sha256"],
        "wave_identity_observation_sha256s": [row["observation_sha256"] for row in observations],
        "runtime_images": runtime_images,
        "artifact_manifest": artifact_manifest,
        "outcomes": accepted_outcomes,
        "summary_sha256": self_hosted.sha256((out_dir / "summary.json").read_bytes()),
        "data_minimization": {
            "prompts_included": False,
            "transcripts_included": False,
            "tool_content_included": False,
            "verifier_content_included": False,
            "flags_included": False,
            "credentials_included": False,
        },
    }
    accepted["acceptance_sha256"] = self_hosted.sha256(self_hosted.canonical_json(accepted))
    return accepted


def _digest_without(value: dict[str, Any], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return self_hosted.sha256(self_hosted.canonical_json(unsigned))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _qwen38_model_aliases(model: dict[str, Any]) -> set[str]:
    served_id = model["served_id"]
    return {
        served_id.lower(),
        f"qwen/{served_id}".lower(),
        str(model.get("repository") or "").lower(),
    }


def live_duplicate_preflight(
    client: httpx.Client,
    rows: list[dict[str, Any]],
    model: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed if Fleet already has a Qwen3.8 session on a selected task key."""

    served_id = model.get("served_id")
    if not isinstance(served_id, str) or not served_id:
        raise ValueError("duplicate preflight model identity is missing")
    aliases = _qwen38_model_aliases(model)
    queries: list[dict[str, Any]] = []
    for row in rows:
        offset = 0
        pages = 0
        returned = 0
        matches = 0
        while True:
            payload = self_hosted._request(
                client,
                "GET",
                "/v1/sessions",
                params={
                    "task_key": row["task_key"],
                    "limit": 500,
                    "offset": offset,
                },
            )
            sessions = payload.get("sessions")
            if not isinstance(sessions, list):
                raise RuntimeError("Fleet session inventory omitted sessions[]")
            if payload.get("has_more") not in {True, False}:
                raise RuntimeError("Fleet session inventory omitted has_more")
            pages += 1
            returned += len(sessions)
            for session in sessions:
                if not isinstance(session, dict):
                    raise RuntimeError("Fleet session inventory contains a malformed row")
                session_model = session.get("model")
                if not isinstance(session_model, str) or not session_model:
                    raise RuntimeError("Fleet session inventory row omits model identity")
                matches += session_model.lower() in aliases
            if payload["has_more"] is False:
                break
            if not sessions:
                raise RuntimeError("Fleet session inventory pagination made no progress")
            offset += len(sessions)
            if offset > 10_000:
                raise RuntimeError("Fleet session inventory exceeded the bounded scan")
        if matches:
            raise RuntimeError("equivalent Qwen3.8 Fleet session already exists")
        queries.append(
            {
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "pages": pages,
                "returned_sessions": returned,
                "equivalent_qwen38_sessions": 0,
            }
        )
    proof = {
        "schema_version": DUPLICATE_PROOF_SCHEMA,
        "route": "/v1/sessions?task_key=<task-lineage-key>",
        "inventory_scope": "live_non_archived_sessions",
        "served_model": served_id,
        "queried_task_count": len(rows),
        "queries": queries,
        "transcripts_read": False,
        "session_identifiers_persisted": False,
    }
    proof["proof_sha256"] = self_hosted.sha256(self_hosted.canonical_json(proof))
    return proof


def validate_duplicate_preflight(
    proof: dict[str, Any], rows: list[dict[str, Any]], model: dict[str, Any]
) -> None:
    if proof.get("schema_version") != DUPLICATE_PROOF_SCHEMA:
        raise ValueError("unsupported Fleet duplicate-preflight proof schema")
    if proof.get("proof_sha256") != _digest_without(proof, "proof_sha256"):
        raise ValueError("Fleet duplicate-preflight proof digest mismatch")
    if proof.get("served_model") != model.get("served_id"):
        raise ValueError("Fleet duplicate-preflight model identity drifted")
    if proof.get("route") != "/v1/sessions?task_key=<task-lineage-key>":
        raise ValueError("Fleet duplicate-preflight route drifted")
    if proof.get("inventory_scope") != "live_non_archived_sessions":
        raise ValueError("Fleet duplicate-preflight inventory scope drifted")
    if proof.get("queried_task_count") != len(rows):
        raise ValueError("Fleet duplicate-preflight task count drifted")
    queries = proof.get("queries") or []
    if len(queries) != len(rows):
        raise ValueError("Fleet duplicate-preflight query count drifted")
    for query in queries:
        if not isinstance(query.get("pages"), int) or query["pages"] <= 0:
            raise ValueError("Fleet duplicate-preflight page count is invalid")
        if not isinstance(query.get("returned_sessions"), int) or query["returned_sessions"] < 0:
            raise ValueError("Fleet duplicate-preflight session count is invalid")
    expected_queries = [
        {
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "pages": query.get("pages"),
            "returned_sessions": query.get("returned_sessions"),
            "equivalent_qwen38_sessions": 0,
        }
        for row, query in zip(rows, queries, strict=True)
    ]
    if proof.get("queries") != expected_queries:
        raise ValueError("Fleet duplicate-preflight query evidence drifted")
    if proof.get("transcripts_read") is not False:
        raise ValueError("Fleet duplicate preflight must not read transcripts")
    if proof.get("session_identifiers_persisted") is not False:
        raise ValueError("Fleet duplicate preflight must not persist session IDs")


def validate_plan(plan: dict[str, Any], split: dict[str, Any]) -> list[dict[str, Any]]:
    task_count = _expected_task_count(plan.get("schema_version"))
    if split.get("schema") != EXPECTED_SPLIT_SCHEMA:
        raise ValueError("unsupported task split schema")
    if split.get("manifest_digest") != _digest_without(split, "manifest_digest"):
        raise ValueError("task split manifest digest mismatch")
    if plan.get("split_manifest_digest") != split.get("manifest_digest"):
        raise ValueError("holdout plan does not pin this task split")
    if plan.get("task_count") != task_count:
        raise ValueError(f"holdout plan must declare exactly {task_count} tasks")
    if plan.get("source_job_id") != (split.get("source") or {}).get("job_id"):
        raise ValueError("holdout plan source job does not match split provenance")
    execution = plan.get("execution") or {}
    if any(execution.get(key) != value for key, value in BASE_EXECUTION_CONTROLS.items()):
        raise ValueError("holdout execution controls drifted")
    if set(execution) - (set(BASE_EXECUTION_CONTROLS) | OPTIONAL_EXECUTION_CONTROLS):
        raise ValueError("holdout execution controls contain unsupported fields")
    expected_concurrency = 3 if plan.get("schema_version") == RANKED50_SCHEMA else 1
    if execution.get("max_concurrent") != expected_concurrency:
        raise ValueError(f"holdout max concurrency must be {expected_concurrency}")
    required_tools = execution.get("required_task_tools")
    required_tool_digest = execution.get("required_task_tool_catalog_sha256")
    if (required_tools is None) != (required_tool_digest is None):
        raise ValueError("holdout exact tool controls must be provided together")
    if required_tools is not None:
        if required_tools != ["bash", "submit_report"]:
            raise ValueError("holdout exact tool surface drifted")
        if not isinstance(required_tool_digest, str) or not required_tool_digest.startswith(
            "sha256:"
        ):
            raise ValueError("holdout exact tool catalog digest is invalid")
    resource_prefix = plan.get("resource_prefix", "q36qcode-test20-b1")
    if not isinstance(resource_prefix, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,39}", resource_prefix
    ):
        raise ValueError("holdout resource prefix is invalid")

    if plan.get("schema_version") == RANKED50_SCHEMA:
        from evals.fleet import qwen38_fleet50

        exclusions = plan.get("prior_attempt_exclusions")
        selection = plan.get("selection")
        if not isinstance(exclusions, dict) or not isinstance(selection, dict):
            raise ValueError("ranked-50 plan lacks frozen selection evidence")
        if exclusions.get("treatment_identity") != qwen38_fleet50.treatment_identity(plan):
            raise ValueError("ranked-50 plan treatment differs from prior-attempt exclusions")
        if (plan.get("authority") or {}).get(
            "required_cyber_contract"
        ) != EXPECTED_RANKED50_CYBER_CONTRACT:
            raise ValueError("ranked-50 required cyber contract drifted")
        rows = qwen38_fleet50.validate_selection(selection, split, exclusions)
        privacy_gate = plan.get("privacy_gate") or {}
        metadata_authority = privacy_gate.get("metadata_only_session_ingestion_authority")
        if (
            privacy_gate.get("full_trace_session_ingestion_forbidden") is not True
            or privacy_gate.get("status")
            != "blocked_pending_deployed_metadata_only_behavioral_probe"
            or metadata_authority
            != {
                "route": "/v1/sessions/ingest",
                "contract_source_pr": "https://github.com/fleet-ai/theseus/pull/28932",
                "contract_source_commit": "8fe26474ac64a0c1a30f60c23ffa1fc5577c7a28",
                "request_message_count": 0,
                "required_response_bindings": [
                    "success",
                    "session_id",
                    "message_count",
                    "created_new_session",
                    "evidence_only",
                    "trace_persisted",
                    "score",
                    "verifier_execution_id",
                    "task_key",
                    "eval_task_version_id",
                    "instance_id",
                    "model",
                ],
                "deployed_openapi_sha256": None,
                "behavioral_probe_receipt_sha256": None,
            }
        ):
            raise ValueError("ranked-50 private-trace gate drifted")
        if plan.get("duplicate_gate") != {
            "known_local_attempts_excluded": 5,
            "live_inventory_scope": "live_non_archived_sessions",
            "authoritative_all_history_duplicate_authority": None,
            "status": "blocked_pending_authoritative_all_history_inventory",
        }:
            raise ValueError("ranked-50 all-history duplicate gate drifted")
    else:
        rows = [row for row in split.get("tasks", []) if row.get("split") == "test"]
        if len(rows) != EXPECTED_TASK_COUNT:
            raise ValueError("task split does not contain exactly 20 test tasks")
    required = {
        "task_key",
        "task_version_id",
        "task_version",
        "environment_version_id",
        "env_key",
        "env_version",
        "data_key",
        "data_version",
    }
    for row in rows:
        if any(not row.get(field) for field in required):
            raise ValueError("test task binding is incomplete")
    if len({row["task_key"] for row in rows}) != task_count:
        raise ValueError("test task keys are not unique")
    if len({row["task_version_id"] for row in rows}) != task_count:
        raise ValueError("test task version ids are not unique")
    if plan.get("schema_version") == HOLDOUT_SCHEMA:
        non_test = [row for row in split.get("tasks", []) if row.get("split") != "test"]
        if {row["task_key"] for row in rows} & {row["task_key"] for row in non_test}:
            raise ValueError("test task lineage overlaps another split")
        rows = sorted(rows, key=lambda row: row["task_key"])
    _validate_first_task_gate(plan, rows)
    return rows


def _gateway_request(
    client: httpx.Client, method: str, path: str, *, model: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    headers = dict(kwargs.pop("headers", {}))
    if model:
        headers["X-Fleet-Model"] = model
    response = client.request(method, path, headers=headers, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(
            f"inference gateway {method} {path} failed with HTTP {response.status_code}"
        )
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("inference gateway returned a non-object response")
    return value


def live_model_identity(
    api_key: str,
    model: dict[str, Any],
    *,
    include_structured_tool_probe: bool = True,
) -> dict[str, Any] | None:
    """Validate an optional exact self-hosted serving contract without exposing model output."""

    contract = model.get("live_identity")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ValueError("holdout live model identity contract is malformed")
    served_id = model["served_id"]
    origin = model["endpoint_origin"].rstrip("/")
    with httpx.Client(
        base_url=origin,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    ) as client:
        catalog = _gateway_request(client, "GET", "/fleet/v1/model-catalog")
        matches = [row for row in catalog.get("data", []) if row.get("id") == served_id]
        if len(matches) != 1:
            raise RuntimeError("holdout served model is missing or duplicated in the catalog")
        catalog_row = matches[0]
        expected_catalog = contract.get("catalog") or {}
        actual_catalog = {field: catalog_row.get(field) for field in expected_catalog}
        if actual_catalog != expected_catalog or int(catalog_row.get("ready_replicas") or 0) < 1:
            raise RuntimeError("holdout catalog model identity drifted")
        if "tool_calling" not in (catalog_row.get("capabilities") or []):
            raise RuntimeError("holdout served model no longer declares tool calling")

        model_info = _gateway_request(client, "GET", "/model_info", model=served_id)
        server_info = _gateway_request(client, "GET", "/server_info", model=served_id)
        expected_model_info = contract.get("model_info") or {}
        expected_server_info = contract.get("server_info") or {}
        actual_model_info = {field: model_info.get(field) for field in expected_model_info}
        actual_server_info = {field: server_info.get(field) for field in expected_server_info}
        if actual_model_info != expected_model_info:
            raise RuntimeError("holdout model_info identity drifted")
        if actual_server_info != expected_server_info:
            raise RuntimeError("holdout server_info identity drifted")

        probe = None
        if include_structured_tool_probe:
            probe = _gateway_request(
                client,
                "POST",
                "/v1/chat/completions",
                json={
                    "model": served_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "For this protocol probe, call submit_report exactly once with "
                                "an empty flags list and explanation identity probe."
                            ),
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "description": (
                                    "Run a shell command in an isolated challenge sandbox."
                                ),
                                "parameters": {
                                    "type": "object",
                                    "properties": {"command": {"type": "string"}},
                                    "required": ["command"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "submit_report",
                                "description": "Submit recovered flags and an explanation.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "flags": {"type": "array", "items": {"type": "string"}},
                                        "explanation": {"type": "string"},
                                    },
                                    "required": ["flags", "explanation"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                    ],
                    "tool_choice": "auto",
                    "temperature": 0,
                    "max_tokens": 256,
                },
            )
    identity = {
        "catalog": actual_catalog,
        "model_info": actual_model_info,
        "server_info": actual_server_info,
    }
    if probe is None:
        return identity
    choices = probe.get("choices") or []
    calls = (choices[0].get("message") or {}).get("tool_calls") or [] if len(choices) == 1 else []
    call_names = [call.get("function", {}).get("name") for call in calls]
    if (
        len(choices) != 1
        or probe.get("model") != served_id
        or choices[0].get("finish_reason") != "tool_calls"
        or call_names != ["submit_report"]
    ):
        raise RuntimeError("holdout structured-tool identity probe failed")
    identity["structured_tool_probe"] = {
        "response_model": probe.get("model"),
        "finish_reason": choices[0].get("finish_reason"),
        "tool_names": call_names,
    }
    return identity


def _wave_model_identity_observation(plan: dict[str, Any], wave: int) -> dict[str, Any]:
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise RuntimeError("FLEET_API_KEY is required for wave model-identity validation")
    identity = live_model_identity(
        api_key,
        plan["model"],
        include_structured_tool_probe=False,
    )
    if identity is None:
        raise RuntimeError("ranked-50 wave model identity was not validated")
    observation = {
        "wave": wave,
        "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "identity": identity,
    }
    observation["observation_sha256"] = self_hosted.sha256(self_hosted.canonical_json(observation))
    return observation


def _task_receipt(client: httpx.Client, row: dict[str, Any], index: int) -> dict[str, Any]:
    task = self_hosted._request(
        client,
        "GET",
        f"/v1/tasks/{row['task_key']}",
        params={"version_id": row["task_version_id"]},
    )
    actual_binding = {
        "task_key": task.get("key"),
        "env_key": task.get("environment_id"),
        "env_version": task.get("version"),
        "data_key": task.get("data_id"),
        "data_version": task.get("data_version"),
    }
    expected_binding = {field: row[field] for field in actual_binding}
    if actual_binding != expected_binding:
        raise RuntimeError(f"exact task binding drifted at test index {index}")
    verifier = task.get("verifier") or {}
    verifier_receipt = {
        "id": task.get("verifier_id"),
        "version_id": verifier.get("verifier_version_id"),
        "version": verifier.get("version"),
        "sha256": verifier.get("sha256"),
        "function_name": verifier.get("function_name") or "verify",
    }
    if any(value in (None, "") for value in verifier_receipt.values()):
        raise RuntimeError(f"verifier receipt is incomplete at test index {index}")
    metadata = task.get("metadata") or {}
    runtime_seed = metadata.get("runtime_seed_manifest") or {}
    cyber_contract = metadata.get("cyber_contract")
    return {
        "index": index,
        "task_key": row["task_key"],
        "task_version_id": row["task_version_id"],
        "task_version": row["task_version"],
        "environment_version_id": row["environment_version_id"],
        "env_key": row["env_key"],
        "env_version": row["env_version"],
        "data_key": row["data_key"],
        "data_version": row["data_version"],
        "prompt_sha256": self_hosted.sha256((task.get("prompt") or "").encode()),
        "env_variables_sha256": self_hosted.sha256(
            self_hosted.canonical_json(task.get("env_variables") or {})
        ),
        "output_json_schema_sha256": self_hosted.sha256(
            self_hosted.canonical_json(task.get("output_json_schema"))
        ),
        "runtime_seed_content_sha256": runtime_seed.get("content_sha256"),
        "runtime_seed_file_count": len(runtime_seed.get("files") or []),
        "cyber_contract": cyber_contract,
        "verifier": verifier_receipt,
    }


def build_live_receipt(
    client: httpx.Client,
    plan: dict[str, Any],
    split: dict[str, Any],
    api_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = validate_plan(plan, split)
    account = self_hosted._request(client, "GET", "/v1/account")
    if account.get("team_name") != "fleet" or account.get("team_id") != self_hosted.FLEET_TEAM_ID:
        raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
    authority_gate = self_hosted.assert_authoritative_routes_deployed(client, plan)
    model_identity = live_model_identity(api_key, plan["model"]) if api_key else None
    if plan["model"].get("live_identity") is not None and model_identity is None:
        raise RuntimeError("holdout live model identity was not validated")
    duplicate_preflight = (
        live_duplicate_preflight(client, rows, plan["model"])
        if plan.get("schema_version") == RANKED50_SCHEMA
        else None
    )
    tasks = [_task_receipt(client, row, index) for index, row in enumerate(rows, 1)]
    required_contract = plan["authority"].get("required_cyber_contract")
    if plan.get("schema_version") == RANKED50_SCHEMA and (
        required_contract != EXPECTED_RANKED50_CYBER_CONTRACT
        or plan["authority"].get("scoring_payload_mode") != self_hosted.RUNTIME_EVIDENCE_ONLY_V3
        or any(row.get("cyber_contract") != required_contract for row in tasks)
    ):
        raise RuntimeError("ranked-50 task is not exact Verifier Contract v3")
    if plan.get("schema_version") == RANKED50_SCHEMA:
        raise RuntimeError(
            "ranked-50 launch is blocked until deployed OpenAPI and a behavioral probe prove "
            "zero-message Fleet session ingestion"
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "plan_schema_version": plan["schema_version"],
        "campaign_id": plan["campaign_id"],
        "source_job_id": plan["source_job_id"],
        "split_manifest_digest": split["manifest_digest"],
        "task_count": len(tasks),
        "planned_sessions": len(tasks) * plan["execution"]["pass_k"],
        "model": copy.deepcopy(plan["model"]),
        "harness": copy.deepcopy(plan["harness"]),
        "authority": copy.deepcopy(plan["authority"]),
        "execution": copy.deepcopy(plan["execution"]),
        "fleet_account": {
            "team_name": account["team_name"],
            "team_id": account["team_id"],
        },
        "tasks": tasks,
    }
    if duplicate_preflight is not None:
        receipt["duplicate_preflight"] = duplicate_preflight
    if model_identity is not None:
        receipt["live_model_identity"] = model_identity
    if "resource_prefix" in plan:
        receipt["resource_prefix"] = plan["resource_prefix"]
    if "launch_gate" in plan:
        receipt["launch_gate"] = copy.deepcopy(plan["launch_gate"])
    receipt["receipt_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
    return receipt, authority_gate


def validate_frozen_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported frozen receipt schema")
    if receipt.get("receipt_sha256") != _digest_without(receipt, "receipt_sha256"):
        raise ValueError("frozen receipt digest mismatch")
    plan_schema = receipt.get("plan_schema_version", HOLDOUT_SCHEMA)
    expected_count = _expected_task_count(plan_schema)
    if receipt.get("task_count") != expected_count:
        raise ValueError(f"frozen receipt must contain exactly {expected_count} tasks")
    if receipt.get("planned_sessions") != expected_count:
        raise ValueError(f"frozen receipt must plan exactly {expected_count} pass@1 sessions")
    tasks = receipt.get("tasks") or []
    if len(tasks) != expected_count:
        raise ValueError("frozen receipt task rows are incomplete")
    if len({row["task_version_id"] for row in tasks}) != expected_count:
        raise ValueError("frozen receipt task versions are not unique")
    if plan_schema == RANKED50_SCHEMA and receipt.get("fleet_account") != {
        "team_name": "fleet",
        "team_id": self_hosted.FLEET_TEAM_ID,
    }:
        raise ValueError("frozen receipt does not prove the exact Fleet team")
    if plan_schema == RANKED50_SCHEMA:
        required_contract = receipt.get("authority", {}).get("required_cyber_contract")
        if (
            required_contract != EXPECTED_RANKED50_CYBER_CONTRACT
            or receipt.get("authority", {}).get("scoring_payload_mode")
            != self_hosted.RUNTIME_EVIDENCE_ONLY_V3
            or any(row.get("cyber_contract") != required_contract for row in tasks)
        ):
            raise ValueError("frozen ranked-50 receipt is not exact Verifier Contract v3")
        validate_duplicate_preflight(
            receipt.get("duplicate_preflight") or {},
            tasks,
            receipt.get("model") or {},
        )


def task_config(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    key_digest = self_hosted.sha256(row["task_key"].encode()).split(":", 1)[1][:8]
    resource_prefix = plan.get("resource_prefix", "q36qcode-test20-b1")
    return {
        "schema_version": "fleet-selfhosted-qwen-code-protocol-v1",
        "run_id": f"{plan['campaign_id']}-t{row['index']:02d}-{key_digest}",
        "source_job_id": plan["source_job_id"],
        "task": {
            "key": row["task_key"],
            "version_id": row["task_version_id"],
            "prompt_sha256": row["prompt_sha256"],
            "env_variables_sha256": row["env_variables_sha256"],
            "output_json_schema_sha256": row["output_json_schema_sha256"],
            "cyber_contract": row.get("cyber_contract"),
        },
        "environment": {
            "id": row["env_key"],
            "version": row["env_version"],
            "version_id": row.get("environment_version_id"),
            "data_id": row["data_key"],
            "data_version": row["data_version"],
            "runtime_seed_content_sha256": row["runtime_seed_content_sha256"],
            "ttl_seconds": plan["environment_ttl_seconds"],
        },
        "verifier": copy.deepcopy(row["verifier"]),
        "authority": copy.deepcopy(plan["authority"]),
        "model": copy.deepcopy(plan["model"]),
        "harness": copy.deepcopy(plan["harness"]),
        "execution": {
            **copy.deepcopy(plan["execution"]),
            "network": f"{resource_prefix}-{row['index']:02d}-{key_digest}",
        },
    }


def run_campaign(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    out_dir: Path,
    proxy_script: Path,
) -> dict[str, Any]:
    validate_frozen_receipt(receipt)
    out_dir.mkdir(parents=True, exist_ok=False)
    out_dir.chmod(0o700)
    outcomes: list[dict[str, Any]] = []
    wave_identity_observations: list[dict[str, Any]] = []
    wave_identity_error: str | None = None

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        config = task_config(plan, row)
        task_out = out_dir / f"task-{row['index']:02d}-{config['run_id'].rsplit('-', 1)[-1]}"
        task_out.mkdir(mode=0o700)
        _write_provisioning_intent(task_out, config)
        scratch_value = os.environ.get("FLEET_EVAL_SCRATCH_ROOT")
        if not scratch_value:
            raise RuntimeError("FLEET_EVAL_SCRATCH_ROOT is required")
        scratch_root = Path(scratch_value)
        scratch_root.mkdir(parents=True, exist_ok=True)
        durable_path = out_dir.resolve()
        scratch_path = scratch_root.resolve()
        if (
            durable_path == scratch_path
            or durable_path in scratch_path.parents
            or scratch_path in durable_path.parents
        ):
            raise RuntimeError("Fleet task scratch must not contain durable output")
        scratch_parent = Path(
            tempfile.mkdtemp(prefix=f"q38-fleet-{row['index']:02d}-", dir=scratch_root)
        )
        scratch_parent.chmod(0o700)
        scratch = scratch_parent / "task"
        result = None
        cleanup_verified = False

        def persist_scoring_intent(intent: dict[str, Any]) -> None:
            safe = _selected(
                intent,
                (
                    "schema_version",
                    "run_id",
                    "task_key",
                    "task_version_id",
                    "instance_id",
                    "evidence_run_id",
                    "scoring_payload_mode",
                    "request_keys",
                    "request_sha256",
                    "scoring_intent_sha256",
                ),
            )
            if safe != intent:
                raise RuntimeError("scoring intent contains unsupported durable fields")
            _write_or_validate_once(task_out / "scoring-intent.json", safe)

        try:
            result = self_hosted.run(
                config,
                scratch,
                proxy_script,
                safe_scoring_intent_sink=persist_scoring_intent,
            )
            cleanup = load_json(scratch / "cleanup.json")
            cleanup_verified = cleanup.get("containers_removed") is True and (
                cleanup.get("instance_created") is not True
                or cleanup.get("instance_closed") is True
            )
            if not cleanup_verified:
                raise RuntimeError("task cleanup receipt is incomplete")
            _validate_authoritative_model_result(
                result,
                {
                    "task_key": row["task_key"],
                    "task_version_id": row["task_version_id"],
                },
            )
            _persist_sanitized_task_artifacts(scratch, task_out, config, result)
            if plan.get("schema_version") == RANKED50_SCHEMA:
                _validate_persisted_direct_evidence_task(task_out, config, result)
            return {
                "index": row["index"],
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "status": "model_outcome",
                "cleanup_verified": True,
                **result,
            }
        except Exception as exc:  # noqa: BLE001
            cleanup_path = scratch / "cleanup.json"
            if cleanup_path.is_file() and not (task_out / "cleanup.json").exists():
                cleanup = _selected(
                    load_json(cleanup_path),
                    (
                        "instance_created",
                        "instance_closed",
                        "containers_removed",
                        "terminated_at",
                    ),
                )
                self_hosted.write_json_once(task_out / "cleanup.json", cleanup)
            outcome: dict[str, Any] = {
                "index": row["index"],
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
            }
            if result is None:
                outcome.update(status="infrastructure_error", error_type=type(exc).__name__)
            else:
                outcome.update(
                    status="infrastructure_error",
                    **result,
                    cleanup_verified=cleanup_verified,
                    infrastructure_error=type(exc).__name__,
                )
            return outcome
        finally:
            shutil.rmtree(scratch_parent, ignore_errors=True)

    def write_state() -> None:
        state = {
            "campaign_id": plan["campaign_id"],
            "planned_sessions": receipt["planned_sessions"],
            "completed_attempts": len(outcomes),
            "wave_identity_observations": wave_identity_observations,
            "wave_identity_error": wave_identity_error,
            "outcomes": outcomes,
        }
        (out_dir / "campaign-state.json").write_bytes(self_hosted.canonical_json(state) + b"\n")

    try:
        wave_identity_observations.append(_wave_model_identity_observation(plan, 0))
    except Exception as exc:  # noqa: BLE001
        wave_identity_error = type(exc).__name__
    if wave_identity_error is None:
        first = run_one(receipt["tasks"][0])
        outcomes.append(first)
        first_gate_passed = _first_task_gate_satisfied(plan, first)
    else:
        first_gate_passed = False
    write_state()
    if first_gate_passed:
        remaining = receipt["tasks"][1:]
        max_workers = plan["execution"]["max_concurrent"]
        for wave_number, offset in enumerate(range(0, len(remaining), max_workers), 1):
            wave = remaining[offset : offset + max_workers]
            try:
                wave_identity_observations.append(
                    _wave_model_identity_observation(plan, wave_number)
                )
            except Exception as exc:  # noqa: BLE001
                wave_identity_error = type(exc).__name__
                write_state()
                break
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                wave_outcomes = list(executor.map(run_one, wave))
            outcomes.extend(wave_outcomes)
            write_state()
            should_stop = any(row["status"] == "infrastructure_error" for row in wave_outcomes)
            for row in wave:
                task_out = out_dir / (
                    f"task-{row['index']:02d}-{task_config(plan, row)['run_id'].rsplit('-', 1)[-1]}"
                )
                cleanup_path = task_out / "cleanup.json"
                if cleanup_path.exists():
                    cleanup = load_json(cleanup_path)
                    should_stop = should_stop or (
                        cleanup.get("instance_created") is True
                        and cleanup.get("instance_closed") is not True
                    )
            if should_stop:
                break
    model_outcomes = [row for row in outcomes if row["status"] == "model_outcome"]
    infrastructure_errors = sum(
        row["status"] == "infrastructure_error" or "infrastructure_error" in row for row in outcomes
    )
    infrastructure_errors += wave_identity_error is not None
    summary = {
        "campaign_id": plan["campaign_id"],
        "planned_sessions": receipt["planned_sessions"],
        "model_outcomes": len(model_outcomes),
        "authoritative_verifier_backed_outcomes": len(model_outcomes),
        "infrastructure_errors": infrastructure_errors,
        "score_sum": sum(float(row["score"]) for row in model_outcomes),
        "first_task_release_gate_satisfied": (first_gate_passed),
        "wave_identity_observations": wave_identity_observations,
        "wave_identity_error": wave_identity_error,
        "outcomes": outcomes,
    }
    self_hosted.write_json_once(out_dir / "summary.json", summary)
    accepted = build_terminal_acceptance(plan, receipt, summary, out_dir)
    if accepted is not None:
        self_hosted.write_json_once(out_dir / "ACCEPTED.json", accepted)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--preflight-out", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy-script", type=Path)
    args = parser.parse_args()
    plan = load_json(args.config)
    split = load_json(args.split)
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise RuntimeError("FLEET_API_KEY is required")
    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=180,
    ) as client:
        live_receipt, authority_gate = build_live_receipt(client, plan, split, api_key)
    if args.receipt_out:
        self_hosted.write_json_once(args.receipt_out, live_receipt)
    if args.receipt:
        frozen = load_json(args.receipt)
        validate_frozen_receipt(frozen)
        if frozen != live_receipt:
            raise RuntimeError("live exact-task receipt does not match frozen receipt")
    if args.command == "preflight":
        preflight = {
            "ok": True,
            "campaign_id": plan["campaign_id"],
            "planned_sessions": live_receipt["planned_sessions"],
            "receipt_sha256": live_receipt["receipt_sha256"],
            "authority_gate": authority_gate,
            "model_revision": plan["model"]["revision"],
            "harness_version": plan["harness"]["version"],
            "max_concurrent": plan["execution"]["max_concurrent"],
            "training_data_eligible": plan["execution"]["training_data_eligible"],
        }
        if args.preflight_out:
            self_hosted.write_json_once(args.preflight_out, preflight)
        print(json.dumps(preflight, sort_keys=True))
        return 0
    if not args.receipt or not args.out_dir or not args.proxy_script:
        raise ValueError("run requires --receipt, --out-dir, and --proxy-script")
    summary = run_campaign(plan, live_receipt, args.out_dir, args.proxy_script)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["infrastructure_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
