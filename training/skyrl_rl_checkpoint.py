"""Seal and validate an exact native SkyRL RL checkpoint.

This is deliberately separate from :mod:`training.skyrl_training`: adding the
reload gate must not change the immutable runtime digest of an already prepared
reward canary.  Checkpoint metadata is native PyTorch pickle and is therefore
read only from the exact trusted run named by the source plan.

The GPU validator creates policy workers only.  It never creates an inference
engine, reads task rows, runs a rollout, calls an optimizer method, writes a new
checkpoint, initializes W&B, or mutates the source checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
import platform
import re
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path, PurePosixPath

from cyber_post_train.jobs import API_URLS, JobsError, bundled_request, digest, quantity

from . import skyrl
from .checkpoints import checkpoint_files as native_checkpoint_files
from .miles_conversion import _hash, _write
from .rl_runtime import hard_deadline
from .skyrl_training import (
    IMAGE,
    REWARD_CANARY_ARGUMENTS,
    REWARD_CANARY_DATA_CONTRACT,
    REWARD_CANARY_MODEL_IDENTITY,
    REWARD_CANARY_RESOURCES,
    REWARD_CANARY_RUNTIME_USER,
    _add_identity,
    _bounded_ray_get,
    _cleanup_engine_diagnostic,
    _credential_names,
    _DiagnosticOwnership,
    _ray_environment,
    _ray_job_id_hex,
    _scrubbed_actor_environment,
    check_inputs,
    digest_template,
    is_reward_canary,
    native_source,
)
from .skyrl_training import (
    SCHEMA as TRAINING_SCHEMA,
)
from .skyrl_training import (
    job_request as training_request,
)

CHECKPOINT_SCHEMA = "cyber_skyrl_rl_checkpoint_manifest_v1"
CHECKPOINT_CANDIDATE_SCHEMA = "cyber_skyrl_rl_checkpoint_candidate_v1"
CHECKPOINT_SEALER_AUDIT_SCHEMA = "cyber_skyrl_rl_checkpoint_sealer_audit_v1"
CHECKPOINT_SEALER_RELEASE_SCHEMA = "cyber_skyrl_rl_checkpoint_sealer_release_v1"
RELOAD_CONFIG_SCHEMA = "cyber_skyrl_rl_reload_config_v1"
RELOAD_SCHEMA = "cyber_skyrl_rl_reload_v1"
PREFLIGHT_SCHEMA = "cyber_skyrl_rl_reload_cpu_preflight_v1"
RELOAD_STARTED_SCHEMA = "cyber_skyrl_rl_reload_started_v1"
RELOAD_RESULT_SCHEMA = "cyber_skyrl_rl_reload_validated_v1"
RELOAD_ACCEPTED_SCHEMA = "cyber_skyrl_rl_reload_accepted_v1"
RELEASE_SCHEMA = "cyber_skyrl_external_release_v1"
CONTROLLER_AUDIT_SCHEMA = "cyber_skyrl_controller_audit_v1"
RELOAD_REJECTED_SCHEMA = "cyber_skyrl_rl_reload_rejected_v1"
REWARD_TERMINAL_SCHEMA = "cyber_skyrl_rl_reward_canary_terminal_v1"
MODULE = "training.skyrl_rl_checkpoint"
WORLD_SIZE = 8
SOURCE_RUN_NAME = "chris-q38-rlreward-dev1"
SOURCE_OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-rlreward-dev1"
SOURCE_MODEL_ROOT = "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
SOURCE_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_DATA_ROOT = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-dev1/data"
SEALER_RUN_NAME = "chris-q38-rlreward-checkpoint-seal-dev1"
SEALER_OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-rlreward-checkpoint-seal-dev1"
DEV_KUBE_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
NAMESPACE = "fleet-train-jobs"
NAMESPACE_UID = "10394b76-e1d4-40b1-a8e2-7575e95df216"
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_SHA256 = re.compile(r"(?:sha256:)?[a-f0-9]{64}")
_RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "64",
    "memory_request": "512Gi",
    "memory_limit": "768Gi",
}
_LIFECYCLE = {
    "startup_timeout_seconds": 1800,
    "cleanup_timeout_seconds": 300,
}


def _fields(names: str) -> frozenset[str]:
    return frozenset(names.split())


_MANIFEST_FIELDS = _fields(
    "schema source_plan_sha256 source_plan sealer completion release source_stat_sha256 "
    "checkpoint sha256"
)
_CANDIDATE_FIELDS = _fields(
    "schema source_plan_sha256 source_plan runtime completion release source_stat_sha256 "
    "checkpoint created_at sha256"
)
_CANDIDATE_RUNTIME_FIELDS = _fields(
    "expected_image runtime_sha256 native_sources_sha256 module_sha256 python packages uid gid "
    "cuda_available"
)
_SEALER_FIELDS = _fields(
    "image runtime_sha256 native_sources_sha256 module_sha256 python packages uid gid "
    "candidate controller_audit release"
)
_SEALER_AUDIT_FIELDS = _fields(
    "schema cluster kube_context namespace namespace_uid run_name source_plan_sha256 "
    "source_request_sha256 checkpoint_global_step candidate_path candidate_file_sha256 "
    "candidate_self_sha256 request_sha256 job_name job_uid pod_name pod_uid pod_owner_job_uid "
    "runtime_image_id effective_security_context container_restarts gpus "
    "kubernetes_observation observed_at sha256"
)
_SEALER_RELEASE_FIELDS = _fields(
    "schema status cluster kube_context namespace namespace_uid run_name source_plan_sha256 "
    "checkpoint_global_step candidate_path candidate_file_sha256 candidate_self_sha256 "
    "controller_audit_path controller_audit_file_sha256 controller_audit_self_sha256 "
    "job_name job_uid pod_name pod_uid job_complete job_failed pod_succeeded "
    "container_exit_code terminated_at job_present pod_present active_gpus "
    "terminal_observation absence_observation observed_at sha256"
)
_SEALER_RUNNING_OBSERVATION_FIELDS = _fields(
    "authority verb kube_context namespace namespace_uid job pod observed_at"
)
_SEALER_JOB_OBSERVATION_FIELDS = _fields(
    "api_version kind name uid resource_version complete failed"
)
_SEALER_POD_OBSERVATION_FIELDS = _fields(
    "api_version kind name uid resource_version owner_job_uid phase container_name "
    "runtime_image_id security_context restart_count gpu_limit exit_code"
)
_SEALER_ABSENCE_OBSERVATION_FIELDS = _fields(
    "authority verb kube_context namespace namespace_uid job_name job_http_status "
    "pod_name pod_http_status active_gpu_pod_uids observed_at"
)
_CONTROLLER_AUDIT_FIELDS = _fields(
    "schema cluster kube_context namespace namespace_uid api_base_url run_name request_sha256 "
    "api_run_name post_response_job_id api_job_id runtime_run_id rayjob_name rayjob_uid "
    "workload_name workload_uid workload_owner_rayjob_uid raycluster_name raycluster_uid "
    "raycluster_owner_rayjob_uid pods effective_priority automatic_requeue workers "
    "gpus_per_worker total_gpus observed_at sha256"
)
_POD_OBSERVATION_FIELDS = _fields(
    "name uid owner_raycluster_uid runtime_image_id runtime_uid runtime_gid container_restarts gpus"
)
_POD_TERMINAL_OBSERVATION_FIELDS = _fields(
    "name uid owner_raycluster_uid phase exit_code termination_reason terminated_at "
    "runtime_image_id runtime_uid runtime_gid container_restarts gpus"
)
_EVIDENCE_FIELDS = _fields("path file_sha256 receipt")
_CHECKPOINT_FIELDS = _fields(
    "path step world_size files total_bytes latest_pointer native_config_sha256 "
    "sampler_batches_in_epoch ranks"
)
_INVENTORY_FIELDS = _fields("bytes sha256")
_RANK_STATE_FIELDS = _fields(
    "rank policy_state_sha256 optimizer_state_sha256 scheduler_state_sha256 rng_state_sha256 "
    "optimizer_states optimizer_step_states optimizer_step scheduler_last_epoch rng_present"
)
_INITIALIZED_RANK_FIELDS = _fields(
    "rank world_size base_policy_sha256 initial_optimizer_states runtime_uid runtime_gid "
    "native_sources_sha256"
)
_REJECTED_RANK_FIELDS = _fields(
    "rank world_size rejected runtime_uid runtime_gid native_sources_sha256"
)
_RELOAD_RANK_FIELDS = _fields(
    "rank world_size base_policy_sha256 loaded_policy_sha256 post_forward_policy_sha256 "
    "optimizer_states optimizer_step_states optimizer_step loaded_optimizer_sha256 "
    "post_forward_optimizer_sha256 loaded_scheduler_sha256 post_forward_scheduler_sha256 "
    "rng_state_sha256 post_forward_rng_sha256 forward_finite forward_batch forward_tokens "
    "gradients_created optimizer_updates"
    " runtime_uid runtime_gid native_sources_sha256"
)
_RESULT_FIELDS = _fields(
    "schema status plan_sha256 request_sha256 runtime_run_id rayjob_name "
    "source_manifest_sha256 source_manifest_file_sha256 "
    "checkpoint_global_step world_size ranks sampler policy_changed_from_base "
    "base_model_and_tokenizer_verified source_checkpoint_unchanged optimizer_updates_executed "
    "rollouts_executed verifier_calls_executed new_checkpoints_created wandb_initialized "
    "internal_ray_resources_released internal_cleanup external_job_gpu_release_verified "
    "external_release_evidence_required_after_process_exit source driver_runtime_identity "
    "completed_at sha256"
)
_RELOAD_ACCEPTED_FIELDS = _fields(
    "schema status plan plan_sha256 reload_result reload_result_file_sha256 "
    "external_release external_release_file_sha256 source_manifest_sha256 source "
    "acceptance_runtime_identity reward_terminal_receipt_verified "
    "source_checkpoint_unchanged_after_release optimizer_updates_executed "
    "rollouts_executed verifier_calls_executed internal_ray_resources_released "
    "external_job_gpu_release_verified production_promotion_requires_this_accepted_receipt sha256"
)
_RUNTIME_IDENTITY_FIELDS = _fields("uid gid native_sources_sha256")
_SOURCE_FIELDS = _fields(
    "run_name plan_sha256 request_sha256 checkpoint_global_step checkpoint_manifest_path "
    "checkpoint_manifest_file_sha256 checkpoint_manifest_self_sha256 "
    "reward_terminal_receipt_path reward_terminal_receipt_file_sha256 "
    "reward_terminal_receipt_self_sha256 source_external_release_path "
    "source_external_release_file_sha256 source_external_release_self_sha256"
)
_REWARD_TERMINAL_FIELDS = _fields(
    "schema status source_run_name source_plan_sha256 source_request_sha256 "
    "runtime_user native_completion episode_audit optimizer_update_proof "
    "wandb_scalar_history checkpoint_manifest source_controller_audit "
    "source_external_release sha256"
)
_REWARD_REFERENCE_FIELDS = _fields("path file_sha256 receipt_self_sha256")
_REWARD_WANDB_FIELDS = _fields(
    "schema source_plan_sha256 identity state remote_scalar_history "
    "remote_scalar_history_sha256 remote_history_rows local_metrics_path "
    "local_metrics_file_sha256 local_scalar_history_sha256 local_history_rows "
    "local_remote_step_history_sha256 required_step_1_scalars episode_audit_sha256 "
    "optimizer_update_proof_sha256 logged_artifact_count rich_payload_count sha256"
)
_CLEANUP_FIELDS = _fields(
    "active_owned_actors active_owned_placement_groups tracked_ray_actors "
    "tracked_engine_actors tracked_placement_groups tracked_routers graceful_actor_shutdown "
    "cleanup_proven"
)
_RELOAD_PLAN_FIELDS = _fields(
    "schema run_name output_root source_manifest_path source_manifest_file_sha256 "
    "source_manifest source runtime_sha256 lifecycle execution"
)


class _ReloadContractRejected(ValueError):
    """A recognized post-cleanup model/state mismatch, not an infrastructure failure."""

    def __init__(self, reason_code: str):
        if reason_code not in {
            "base_initialization_contract",
            "native_state_restore_contract",
        }:
            raise ValueError("unknown reload rejection code")
        super().__init__(reason_code)
        self.reason_code = reason_code


def _runtime() -> dict[str, str]:
    from .skyrl_training import _runtime as training_runtime

    files = training_runtime()
    files["training/skyrl_rl_checkpoint.py"] = Path(__file__).read_text()
    return files


def _unsigned(value: Mapping) -> dict:
    return {key: item for key, item in value.items() if key != "sha256"}


def _sealed(value: object, schema: str) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or not isinstance(value.get("sha256"), str)
        or value["sha256"].removeprefix("sha256:") != digest(_unsigned(value))
    ):
        raise ValueError("receipt schema or self-digest mismatch")
    return value


def _json(path: Path) -> dict:
    return _json_snapshot(path)[0]


def _file_snapshot(path: Path) -> tuple[bytes, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or indirect file input")
    before = _stat(path)
    payload = path.read_bytes()
    if _stat(path) != before:
        raise ValueError("input changed while reading")
    return payload, hashlib.sha256(payload).hexdigest()


def _json_snapshot(path: Path) -> tuple[dict, str]:
    payload, file_sha256 = _file_snapshot(path)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value, file_sha256


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _exact_integer(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _canonical_sfs(value: str, *, jobs: bool = False) -> str:
    path = PurePosixPath(value)
    prefix = ("/", "mnt", "sfs", "jobs") if jobs else ("/", "mnt", "sfs")
    if (
        not isinstance(value, str)
        or path.parts[: len(prefix)] != prefix
        or len(path.parts) <= len(prefix)
        or str(path) != value
        or ".." in path.parts
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("expected a canonical SFS path")
    return value


def _source_plan(plan: dict) -> dict:
    request = training_request(plan)
    args = skyrl.SkyRLConfig(**plan["arguments"])
    data = plan["data"]
    expected_args = skyrl.SkyRLConfig(**REWARD_CANARY_ARGUMENTS)
    files = data.get("files", {})
    if (
        plan.get("schema") != TRAINING_SCHEMA
        or is_reward_canary(plan) is not True
        or plan.get("run_name") != SOURCE_RUN_NAME
        or plan.get("output_root") != SOURCE_OUTPUT_ROOT
        or args != expected_args
        or args.model_root != plan["model"].get("root")
        or args.model_root != SOURCE_MODEL_ROOT
        or any(
            plan["model"].get(key) != value for key, value in REWARD_CANARY_MODEL_IDENTITY.items()
        )
        or any(data["tokenizer"].get(key) != plan["model"].get(key) for key in ("repo", "revision"))
        or any(
            data.get(key) != REWARD_CANARY_DATA_CONTRACT[key]
            for key in ("selection_sha256", "split_sha256", "tool_catalog_sha256", "limits")
        )
        or set(files) != {"train", "dev"}
        or any(
            files[split].get("path") != f"{split}.jsonl" or files[split].get("rows") != rows
            for split, rows in REWARD_CANARY_DATA_CONTRACT["rows"].items()
        )
        or plan["execution"].get("cluster_target") != "dev"
        or plan["execution"].get("priority") != "c1"
        or plan["execution"].get("resources") != REWARD_CANARY_RESOURCES
        or plan["execution"].get("runtime_user") != REWARD_CANARY_RUNTIME_USER
        or "engine_diagnostic_prerequisite" not in plan["execution"]
        or "reward_canary_source" not in plan["execution"]
    ):
        raise ValueError("RL reload source must be the exact qualified Qwen3.8 reward canary")
    return request


def _package_identity() -> dict[str, str]:
    packages = {}
    for name in ("ray", "torch", "transformers"):
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"required native package identity is absent: {name}") from exc
        if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
            raise ValueError("native package version is invalid")
        packages[name] = value
    return packages


def checkpoint_sealer_contract(plan: dict, step: int) -> dict:
    """Return the immutable zero-GPU Kubernetes Job contract audited after exit."""
    source_request = _source_plan(plan)
    if step != 1:
        raise ValueError("reward-canary checkpoint sealer is fixed to step 1")
    return {
        "schema": "cyber_skyrl_rl_checkpoint_sealer_job_v1",
        "run_name": SEALER_RUN_NAME,
        "output_root": SEALER_OUTPUT_ROOT,
        "source_plan_sha256": digest(plan),
        "source_request_sha256": digest(source_request),
        "checkpoint_global_step": 1,
        "candidate_path": SEALER_OUTPUT_ROOT + "/RL_CHECKPOINT_STEP_1_CANDIDATE.json",
        "image": IMAGE,
        "kube_context": DEV_KUBE_CONTEXT,
        "namespace": NAMESPACE,
        "namespace_uid": NAMESPACE_UID,
        "security_context": {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True},
        "restart_policy": "Never",
        "backoff_limit": 0,
        "gpus": 0,
    }


def _observed_epoch(value: str) -> float:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("release evidence needs an exact UTC observation time")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("release evidence observation time is invalid") from exc
    return parsed.timestamp()


def _request(plan: dict) -> dict:
    if plan.get("schema") == TRAINING_SCHEMA:
        return training_request(plan)
    if plan.get("schema") == RELOAD_SCHEMA:
        return reload_request(plan)
    raise ValueError("release evidence names an unsupported plan")


def _submission(plan: dict, release: dict) -> tuple[dict, str]:
    suffix = (
        "RELOAD_SUBMISSION.jsonl"
        if plan.get("schema") == RELOAD_SCHEMA
        else "SOURCE_SUBMISSION.jsonl"
    )
    expected_path = Path(plan["output_root"]) / suffix
    if release.get("submission_journal_path") != str(expected_path):
        raise ValueError("release evidence submission journal path changed")
    payload, file_sha256 = _file_snapshot(expected_path)
    if release.get("submission_journal_file_sha256") != file_sha256:
        raise ValueError("release evidence submission journal digest changed")
    try:
        rows = [json.loads(line) for line in payload.splitlines()]
    except (TypeError, ValueError) as exc:
        raise ValueError("submission journal is not valid JSONL") from exc
    if len(rows) != 2 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("submission journal must contain one intent and one response")
    intent, response = rows
    request_sha256 = digest(_request(plan))
    if (
        intent.get("state") != "POST_INTENT_DO_NOT_RETRY"
        or intent.get("api_base_url") != API_URLS["dev"]
        or intent.get("request_sha256") != request_sha256
        or response.get("state") != "POST_RESPONSE"
        or response.get("name") != release.get("api_run_name")
        or response.get("job_id") != release.get("post_response_job_id")
        or response.get("run_dir") not in (None, plan["output_root"])
    ):
        raise ValueError("release evidence differs from the create-once Jobs API journal")
    return {"request_sha256": request_sha256, "journal": rows}, file_sha256


def _controller_audit(plan: dict, release: dict) -> tuple[dict, str]:
    """Validate the independent live API/Kubernetes ownership observation."""
    suffix = (
        "RELOAD_CONTROLLER_AUDIT.json"
        if plan.get("schema") == RELOAD_SCHEMA
        else "SOURCE_CONTROLLER_AUDIT.json"
    )
    expected_path = Path(plan["output_root"]) / suffix
    audit, file_sha256 = _json_snapshot(expected_path)
    _sealed(audit, CONTROLLER_AUDIT_SCHEMA)
    if (
        set(audit) != _CONTROLLER_AUDIT_FIELDS
        or release.get("controller_audit_file_sha256") != file_sha256
        or release.get("controller_audit_sha256") != audit.get("sha256")
    ):
        raise ValueError("release evidence controller-audit digest changed")
    pods = audit.get("pods")
    if (
        not isinstance(pods, list)
        or len(pods) != 1
        or any(not isinstance(item, dict) or set(item) != _POD_OBSERVATION_FIELDS for item in pods)
        or [item["uid"] for item in pods] != sorted(item["uid"] for item in pods)
        or any(
            not isinstance(item["name"], str)
            or not item["name"]
            or _UUID.fullmatch(str(item["uid"])) is None
            or item["owner_raycluster_uid"] != audit.get("raycluster_uid")
            or item["runtime_image_id"] != release.get("runtime_image_id")
            or _image_digest(item["runtime_image_id"])
            != plan["execution"]["image"].rsplit("@sha256:", 1)[-1]
            or not _exact_integer(item["runtime_uid"], 1000)
            or not _exact_integer(item["runtime_gid"], 100)
            or not _exact_integer(item["container_restarts"], 0)
            or not _exact_integer(item["gpus"], WORLD_SIZE)
            for item in pods
        )
    ):
        raise ValueError("controller audit Pod ownership or runtime identity changed")
    if (
        audit.get("cluster") != "dev"
        or audit.get("kube_context") != DEV_KUBE_CONTEXT
        or audit.get("namespace") != NAMESPACE
        or audit.get("namespace_uid") != NAMESPACE_UID
        or audit.get("api_base_url") != API_URLS["dev"]
        or audit.get("run_name") != plan["run_name"]
        or audit.get("request_sha256") != release.get("request_sha256")
        or audit.get("api_run_name") != release.get("api_run_name")
        or audit.get("post_response_job_id") != release.get("post_response_job_id")
        or audit.get("api_job_id") != release.get("api_job_id")
        or audit.get("runtime_run_id") != release.get("runtime_run_id")
        or audit.get("rayjob_name") != release.get("rayjob_name")
        or audit.get("rayjob_name") != audit.get("api_run_name")
        or audit.get("rayjob_uid") != release.get("rayjob_uid")
        or audit.get("workload_name") != release.get("workload_name")
        or audit.get("workload_uid") != release.get("workload_uid")
        or audit.get("workload_owner_rayjob_uid") != audit.get("rayjob_uid")
        or audit.get("raycluster_name") != release.get("raycluster_name")
        or audit.get("raycluster_uid") != release.get("raycluster_uid")
        or audit.get("raycluster_owner_rayjob_uid") != audit.get("rayjob_uid")
        or [item["uid"] for item in pods] != release.get("pod_uids")
        or len(
            {
                audit.get("runtime_run_id"),
                audit.get("rayjob_uid"),
                audit.get("workload_uid"),
                audit.get("raycluster_uid"),
                *(item["uid"] for item in pods),
            }
        )
        != 4 + len(pods)
        or not _exact_integer(audit.get("effective_priority"), 10000)
        or audit.get("automatic_requeue") is not False
        or not all(
            _exact_integer(audit.get(key), expected)
            for key, expected in (
                ("workers", 1),
                ("gpus_per_worker", WORLD_SIZE),
                ("total_gpus", WORLD_SIZE),
            )
        )
        or _observed_epoch(audit.get("observed_at")) > _observed_epoch(release.get("observed_at"))
    ):
        raise ValueError("release evidence differs from the bound live controller audit")
    return audit, file_sha256


def _image_digest(image_id: object) -> str:
    if not isinstance(image_id, str):
        raise ValueError("runtime imageID is absent")
    match = re.fullmatch(
        r"(?:containerd|docker-pullable)://(?:[^@\s]+@)?sha256:([a-f0-9]{64})", image_id
    )
    if match is None:
        raise ValueError("runtime imageID is not an immutable observed digest")
    return match.group(1)


def _release(
    value: dict,
    plan: dict,
    *,
    not_before: float | None = None,
    check_submission: bool = True,
) -> None:
    _sealed(value, RELEASE_SCHEMA)
    required = _fields(
        "schema status cluster kube_context namespace namespace_uid run_name plan_sha256 "
        "request_sha256 api_base_url "
        "submission_journal_path submission_journal_file_sha256 controller_audit_path "
        "controller_audit_file_sha256 controller_audit_sha256 api_run_name "
        "post_response_job_id api_job_id "
        "runtime_run_id rayjob_name "
        "rayjob_uid workload_name workload_uid raycluster_name raycluster_uid pod_uids "
        "pod_terminal_observations "
        "api_status controller_status "
        "effective_priority automatic_requeue workers gpus_per_worker total_gpus "
        "runtime_image_id container_restarts raycluster_present gpu_pods_present active_gpus "
        "gpu_release_proven observed_at sha256"
    )
    is_reload = plan.get("schema") == RELOAD_SCHEMA
    if not is_reload:
        required |= _fields(
            "source_pod_uid source_runtime_image_id source_runtime_uid source_runtime_gid"
        )
    if set(value) != required:
        raise ValueError("release evidence fields changed")
    suffix = (
        "RELOAD_SUBMISSION.jsonl"
        if plan.get("schema") == RELOAD_SCHEMA
        else "SOURCE_SUBMISSION.jsonl"
    )
    audit_suffix = (
        "RELOAD_CONTROLLER_AUDIT.json"
        if plan.get("schema") == RELOAD_SCHEMA
        else "SOURCE_CONTROLLER_AUDIT.json"
    )
    if value.get("submission_journal_path") != str(Path(plan["output_root"]) / suffix):
        raise ValueError("release evidence submission journal path changed")
    if value.get("controller_audit_path") != str(Path(plan["output_root"]) / audit_suffix):
        raise ValueError("release evidence controller-audit path changed")
    submission = _submission(plan, value)[0] if check_submission else None
    audit = _controller_audit(plan, value)[0] if check_submission else None
    identities = [
        value[key] for key in ("runtime_run_id", "rayjob_uid", "workload_uid", "raycluster_uid")
    ]
    terminal_pods = value.get("pod_terminal_observations")
    expected_image = plan["execution"]["image"].rsplit("@sha256:", 1)[-1]
    rayjob_name = value.get("rayjob_name")
    if (
        value["status"] != "released"
        or value["cluster"] != "dev"
        or value["kube_context"] != DEV_KUBE_CONTEXT
        or value["namespace"] != NAMESPACE
        or value["namespace_uid"] != NAMESPACE_UID
        or value["run_name"] != plan["run_name"]
        or value["plan_sha256"] != digest(plan)
        or not _is_sha256(value["request_sha256"])
        or (submission is not None and value["request_sha256"] != submission["request_sha256"])
        or value["api_base_url"] != API_URLS["dev"]
        or not _is_sha256(value["submission_journal_file_sha256"])
        or not _is_sha256(value["controller_audit_file_sha256"])
        or not _is_sha256(value["controller_audit_sha256"])
        or not isinstance(value["api_run_name"], str)
        or re.fullmatch(re.escape(plan["run_name"]) + r"-[a-f0-9]{8}", value["api_run_name"])
        is None
        or (
            value["post_response_job_id"] is not None
            and (
                not isinstance(value["post_response_job_id"], str)
                or re.fullmatch(
                    re.escape(value["api_run_name"]) + r"-[a-z0-9]{5}",
                    value["post_response_job_id"],
                )
                is None
            )
        )
        or not isinstance(value["api_job_id"], str)
        or re.fullmatch(re.escape(value["api_run_name"]) + r"-[a-z0-9]{5}", value["api_job_id"])
        is None
        or not isinstance(rayjob_name, str)
        or rayjob_name != value["api_run_name"]
        or any(not isinstance(item, str) or _UUID.fullmatch(item) is None for item in identities)
        or not isinstance(value["workload_name"], str)
        or not value["workload_name"]
        or not isinstance(value["raycluster_name"], str)
        or not value["raycluster_name"]
        or not isinstance(value["pod_uids"], list)
        or len(value["pod_uids"]) != 1
        or any(
            not isinstance(item, str) or _UUID.fullmatch(item) is None for item in value["pod_uids"]
        )
        or not isinstance(terminal_pods, list)
        or len(terminal_pods) != len(value["pod_uids"])
        or any(
            not isinstance(item, dict)
            or set(item) != _POD_TERMINAL_OBSERVATION_FIELDS
            or item.get("uid") != value["pod_uids"][index]
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or item.get("owner_raycluster_uid") != value["raycluster_uid"]
            or item.get("phase") != "Succeeded"
            or not _exact_integer(item.get("exit_code"), 0)
            or item.get("termination_reason") != "Completed"
            or _image_digest(item.get("runtime_image_id")) != expected_image
            or not _exact_integer(item.get("runtime_uid"), 1000)
            or not _exact_integer(item.get("runtime_gid"), 100)
            or not _exact_integer(item.get("container_restarts"), 0)
            or not _exact_integer(item.get("gpus"), WORLD_SIZE)
            or _observed_epoch(item.get("terminated_at"))
            > _observed_epoch(value.get("observed_at"))
            for index, item in enumerate(terminal_pods)
        )
        or len({*identities, *value["pod_uids"]}) != 4 + len(value["pod_uids"])
        or (
            value["post_response_job_id"] is not None
            and value["api_job_id"] is not None
            and value["post_response_job_id"] != value["api_job_id"]
        )
        or value["api_status"] != "SUCCEEDED"
        or value["controller_status"] != "SUCCEEDED"
        or not _exact_integer(value["effective_priority"], 10000)
        or value["automatic_requeue"] is not False
        or not all(
            _exact_integer(value[key], expected)
            for key, expected in (
                ("workers", 1),
                ("gpus_per_worker", WORLD_SIZE),
                ("total_gpus", WORLD_SIZE),
            )
        )
        or _image_digest(value["runtime_image_id"]) != expected_image
        or not _exact_integer(value["container_restarts"], 0)
        or value["raycluster_present"] is not False
        or value["gpu_pods_present"] is not False
        or not _exact_integer(value["active_gpus"], 0)
        or value["gpu_release_proven"] is not True
        or (audit is not None and audit.get("sha256") != value["controller_audit_sha256"])
        or (
            not is_reload
            and (
                value.get("source_pod_uid") != value["pod_uids"][0]
                or value.get("source_runtime_image_id") != value["runtime_image_id"]
                or not _exact_integer(value.get("source_runtime_uid"), 1000)
                or not _exact_integer(value.get("source_runtime_gid"), 100)
                or (
                    audit is not None
                    and (
                        audit["pods"][0]["uid"] != value["source_pod_uid"]
                        or audit["pods"][0]["runtime_image_id"] != value["source_runtime_image_id"]
                        or audit["pods"][0]["runtime_uid"] != value["source_runtime_uid"]
                        or audit["pods"][0]["runtime_gid"] != value["source_runtime_gid"]
                    )
                )
            )
        )
    ):
        raise ValueError("resource release is not exact, terminal and proven")
    observed = _observed_epoch(value["observed_at"])
    if not_before is not None and observed < not_before:
        raise ValueError("resource release observation predates terminal evidence")


def _required_checkpoint_files(world_size: int = WORLD_SIZE) -> set[str]:
    return {"data.pt", "trainer_state.pt", "policy/fsdp_config.json"} | {
        f"policy/{kind}_world_size_{world_size}_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state")
        for rank in range(world_size)
    }


def _checkpoint_files(root: Path, world_size: int = WORLD_SIZE) -> dict[str, Path]:
    for path in root.rglob("*") if root.is_dir() and not root.is_symlink() else ():
        if not path.is_dir() and not path.is_file():
            raise ValueError("checkpoint contains an indirect or special entry")
    files = native_checkpoint_files(root, world_size)
    if "policy/huggingface/tokenizer_config.json" not in files:
        raise ValueError("incomplete or unexpected native RL checkpoint layout")
    return files


def _stat(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _inventory(files: Mapping[str, Path]) -> tuple[dict, dict]:
    before = {name: _stat(path) for name, path in files.items()}
    result = {
        name: {
            "bytes": state[2],
            "sha256": _hash(files[name]),
        }
        for name, state in before.items()
    }
    if any(_stat(files[name]) != state for name, state in before.items()):
        raise ValueError("checkpoint changed while hashing")
    return result, before


def _source_stat_snapshot(manifest: dict) -> tuple[dict[str, tuple[int, int, int, int]], str]:
    """Capture the sealed files cheaply enough to repeat while GPUs are allocated."""
    source, checkpoint = manifest["source_plan"], manifest["checkpoint"]
    root = Path(checkpoint["path"])
    paths = {
        **{f"checkpoint/{name}": root / name for name in checkpoint["files"]},
        **{
            f"model/{item['path']}": Path(source["model"]["root"]) / item["path"]
            for item in source["model"]["files"]
        },
        "checkpoint/latest": Path(checkpoint["latest_pointer"]["path"]),
        "terminal/completion": Path(manifest["completion"]["path"]),
        "terminal/release": Path(manifest["release"]["path"]),
        "terminal/submission": Path(manifest["release"]["receipt"]["submission_journal_path"]),
        "terminal/controller_audit": Path(manifest["release"]["receipt"]["controller_audit_path"]),
    }
    if len(paths) != len(set(paths.values())) or any(
        path.is_symlink() or not path.is_file() for path in paths.values()
    ):
        raise ValueError("sealed source file topology changed")
    # NFS device numbers are client-local; inode/size/mtime/ctime are stable
    # across the CPU sealer and a later GPU node.
    states = {name: _stat(path)[1:] for name, path in paths.items()}
    for name, item in checkpoint["files"].items():
        if states[f"checkpoint/{name}"][1] != item["bytes"]:
            raise ValueError("sealed checkpoint file size changed")
    value = digest({name: list(state) for name, state in sorted(states.items())})
    return states, value


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:  # pinned torch supports mmap; retained for focused CPU fakes
        return torch.load(path, map_location="cpu", weights_only=False)


def _number(value) -> float:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError("optimizer counter is not a finite number")
    return float(value)


def _native_config_dict(plan: dict) -> dict:
    cfg = skyrl.native_config(skyrl.SkyRLConfig(**plan["arguments"]))
    if not dataclasses.is_dataclass(cfg):
        raise ValueError("native SkyRL configuration representation changed")
    return dataclasses.asdict(cfg)


def _checkpoint_state(plan: dict, files: Mapping[str, Path], step: int) -> dict:
    trainer = _torch_load(files["trainer_state.pt"])
    sampler = _torch_load(files["data.pt"])
    native = _native_config_dict(plan)
    if (
        not isinstance(trainer, dict)
        or set(trainer) != {"global_step", "config"}
        or trainer["global_step"] != step
        or trainer["config"] != native
        or not isinstance(sampler, dict)
    ):
        raise ValueError("trainer configuration, step or sampler state changed")
    prompt_batches = plan["arguments"]["train_rows"] // plan["arguments"]["groups"]
    cursor = (step - 1) % prompt_batches + 1
    if sampler.get("_num_yielded") != cursor:
        raise ValueError("saved sampler cursor differs from the optimizer step")
    ranks = []
    for rank in range(WORLD_SIZE):
        policy = _torch_load(files[f"policy/model_world_size_{WORLD_SIZE}_rank_{rank}.pt"])
        optimizer = _torch_load(files[f"policy/optim_world_size_{WORLD_SIZE}_rank_{rank}.pt"])
        extra = _torch_load(files[f"policy/extra_state_world_size_{WORLD_SIZE}_rank_{rank}.pt"])
        if (
            not isinstance(optimizer, dict)
            or not isinstance(optimizer.get("state"), dict)
            or not optimizer["state"]
            or not isinstance(optimizer.get("param_groups"), list)
            or not optimizer["param_groups"]
            or not isinstance(extra, dict)
            or extra.get("rank") != rank
            or extra.get("world_size") != WORLD_SIZE
            or extra.get("fsdp_strategy") != "fsdp"
            or not isinstance(extra.get("lr_scheduler"), dict)
            or not extra["lr_scheduler"]
            or "rng" not in extra
        ):
            raise ValueError("optimizer/scheduler/RNG rank state is incomplete")
        steps = [
            _number(state["step"])
            for state in optimizer["state"].values()
            if isinstance(state, dict) and "step" in state
        ]
        if len(steps) != len(optimizer["state"]) or any(value != step for value in steps):
            raise ValueError("optimizer counters differ from the checkpoint step")
        if extra["lr_scheduler"].get("last_epoch") != step:
            raise ValueError("scheduler counter differs from the checkpoint step")
        ranks.append(
            {
                "rank": rank,
                "policy_state_sha256": _policy_state_digest(policy),
                "optimizer_state_sha256": _state_digest(optimizer),
                "scheduler_state_sha256": _state_digest(extra["lr_scheduler"]),
                "rng_state_sha256": _state_digest(extra["rng"]),
                "optimizer_states": len(optimizer["state"]),
                "optimizer_step_states": len(steps),
                "optimizer_step": step,
                "scheduler_last_epoch": step,
                "rng_present": True,
            }
        )
    return {
        "native_config_sha256": digest(native),
        "sampler_batches_in_epoch": cursor,
        "ranks": ranks,
    }


def _completion(plan: dict, step: int) -> tuple[dict, str]:
    root = Path(plan["output_root"])
    value, file_sha256 = _json_snapshot(root / "NATIVE_TRAINING_COMPLETE.json")
    _validate_completion(value, plan, step)
    if any((root / name).exists() for name in ("FAILED.json", "REJECTED.json")):
        raise ValueError("source run has conflicting terminal evidence")
    return value, file_sha256


def checkpoint_candidate(plan: dict, step: int, output: Path, release_evidence: Path) -> dict:
    """Inspect trusted native state inside the pinned zero-GPU sealer Job."""
    import torch

    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("RL checkpoint candidate must use pinned image user 1000:100")
    if torch.cuda.is_available():
        raise ValueError("RL checkpoint candidate is CPU-only")
    _source_plan(plan)
    native_source()
    check_inputs(plan)
    _tokenizer_check(plan)
    if type(step) is not int or not 0 < step <= plan["arguments"]["steps"]:
        raise ValueError("checkpoint step lies outside the source plan")
    if output.exists() or output.is_symlink():
        raise FileExistsError("checkpoint manifest destination already exists")
    _canonical_sfs(str(output), jobs=True)
    _canonical_sfs(str(release_evidence), jobs=True)
    root = Path(plan["output_root"])
    if release_evidence != root / "SOURCE_RELEASE.json":
        raise ValueError("source release evidence must use its create-once bound path")
    expected_output = Path(SEALER_OUTPUT_ROOT) / f"RL_CHECKPOINT_STEP_{step}_CANDIDATE.json"
    if output != expected_output or os.environ.get("RUN_DIR") != SEALER_OUTPUT_ROOT:
        raise ValueError("checkpoint candidate must use its exact sealer Job output")
    completion, completion_file_sha256 = _completion(plan, step)
    release, release_file_sha256 = _json_snapshot(release_evidence)
    _release(release, plan, not_before=float(completion["completed_at"]))
    pointer = root / "checkpoints/latest_ckpt_global_step.txt"
    checkpoint = root / f"checkpoints/global_step_{step}"
    files = _checkpoint_files(checkpoint)
    inventory, before = _inventory(files)
    state = _checkpoint_state(plan, files, step)
    pointer_payload, pointer_sha256 = _file_snapshot(pointer)
    if pointer_payload.decode().strip() != str(step):
        raise ValueError("latest checkpoint pointer differs from the requested step")
    completion_path = root / "NATIVE_TRAINING_COMPLETE.json"
    completion_after, completion_sha256_after = _json_snapshot(completion_path)
    release_after, release_sha256_after = _json_snapshot(release_evidence)
    pointer_after = _file_snapshot(pointer)
    if (
        any(_stat(files[name]) != value for name, value in before.items())
        or (completion_after, completion_sha256_after) != (completion, completion_file_sha256)
        or (release_after, release_sha256_after) != (release, release_file_sha256)
        or pointer_after != (pointer_payload, pointer_sha256)
    ):
        raise ValueError("source changed while sealing")
    result = {
        "schema": CHECKPOINT_CANDIDATE_SCHEMA,
        "source_plan_sha256": digest(plan),
        "source_plan": plan,
        "runtime": {
            "expected_image": IMAGE,
            "runtime_sha256": digest(_runtime()),
            "native_sources_sha256": digest(plan["native_sources"]),
            "module_sha256": _hash(Path(__file__)),
            "python": platform.python_version(),
            "packages": _package_identity(),
            "uid": os.geteuid(),
            "gid": os.getegid(),
            "cuda_available": False,
        },
        "completion": {
            "path": str(completion_path),
            "file_sha256": completion_file_sha256,
            "receipt": completion,
        },
        "release": {
            "path": str(release_evidence),
            "file_sha256": release_file_sha256,
            "receipt": release,
        },
        "source_stat_sha256": "",
        "checkpoint": {
            "path": str(checkpoint),
            "step": step,
            "world_size": WORLD_SIZE,
            "files": inventory,
            "total_bytes": sum(item["bytes"] for item in inventory.values()),
            "latest_pointer": {"path": str(pointer), "file_sha256": pointer_sha256},
            **state,
        },
        "created_at": time.time(),
    }
    _, result["source_stat_sha256"] = _source_stat_snapshot(result)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return _write(output, result)


def _bound_receipt(path: Path, expected: Path, schema: str) -> tuple[dict, str]:
    if path != expected:
        raise ValueError("checkpoint sealer evidence path changed")
    _canonical_sfs(str(path), jobs=True)
    value, file_sha256 = _json_snapshot(path)
    _sealed(value, schema)
    return value, file_sha256


def _validate_sealer_kubernetes_observation(
    observation: object,
    audit: dict,
    *,
    terminal: bool,
) -> None:
    """Validate a sanitized direct Kubernetes API GET, not an author-entered verdict."""
    if not isinstance(observation, dict) or set(observation) != _SEALER_RUNNING_OBSERVATION_FIELDS:
        raise ValueError("checkpoint sealer Kubernetes observation fields changed")
    job, pod = observation.get("job"), observation.get("pod")
    context = audit.get("effective_security_context")
    if (
        observation.get("authority") != "kubernetes_api"
        or observation.get("verb") != "GET"
        or observation.get("kube_context") != DEV_KUBE_CONTEXT
        or observation.get("namespace") != NAMESPACE
        or observation.get("namespace_uid") != NAMESPACE_UID
        or not isinstance(job, dict)
        or set(job) != _SEALER_JOB_OBSERVATION_FIELDS
        or job.get("api_version") != "batch/v1"
        or job.get("kind") != "Job"
        or job.get("name") != audit.get("job_name")
        or job.get("uid") != audit.get("job_uid")
        or re.fullmatch(r"[1-9][0-9]*", str(job.get("resource_version", ""))) is None
        or job.get("complete") is not terminal
        or job.get("failed") is not False
        or not isinstance(pod, dict)
        or set(pod) != _SEALER_POD_OBSERVATION_FIELDS
        or pod.get("api_version") != "v1"
        or pod.get("kind") != "Pod"
        or pod.get("name") != audit.get("pod_name")
        or pod.get("uid") != audit.get("pod_uid")
        or re.fullmatch(r"[1-9][0-9]*", str(pod.get("resource_version", ""))) is None
        or pod.get("owner_job_uid") != audit.get("job_uid")
        or pod.get("phase") != ("Succeeded" if terminal else "Running")
        or pod.get("container_name") != "training"
        or pod.get("runtime_image_id") != audit.get("runtime_image_id")
        or pod.get("security_context") != context
        or not _exact_integer(pod.get("restart_count"), 0)
        or not _exact_integer(pod.get("gpu_limit"), 0)
        or pod.get("exit_code") != (0 if terminal else None)
        or observation.get("observed_at")
        != (audit.get("terminated_at") if terminal else audit.get("observed_at"))
    ):
        raise ValueError("checkpoint sealer Kubernetes observation is not authoritative and exact")
    _observed_epoch(observation["observed_at"])


def _validate_sealer_absence_observation(observation: object, release: dict) -> None:
    if (
        not isinstance(observation, dict)
        or set(observation) != _SEALER_ABSENCE_OBSERVATION_FIELDS
        or observation.get("authority") != "kubernetes_api"
        or observation.get("verb") != "GET"
        or observation.get("kube_context") != DEV_KUBE_CONTEXT
        or observation.get("namespace") != NAMESPACE
        or observation.get("namespace_uid") != NAMESPACE_UID
        or observation.get("job_name") != release.get("job_name")
        or observation.get("job_http_status") != 404
        or observation.get("pod_name") != release.get("pod_name")
        or observation.get("pod_http_status") != 404
        or observation.get("active_gpu_pod_uids") != []
        or observation.get("observed_at") != release.get("observed_at")
    ):
        raise ValueError("checkpoint sealer post-exit absence observation changed")
    _observed_epoch(observation["observed_at"])


def _validate_sealer_audit(
    audit: dict, candidate: dict, candidate_path: Path, candidate_file_sha256: str
) -> None:
    contract = checkpoint_sealer_contract(candidate["source_plan"], candidate["checkpoint"]["step"])
    context = audit.get("effective_security_context")
    if (
        set(audit) != _SEALER_AUDIT_FIELDS
        or audit.get("cluster") != "dev"
        or audit.get("kube_context") != DEV_KUBE_CONTEXT
        or audit.get("namespace") != NAMESPACE
        or audit.get("namespace_uid") != NAMESPACE_UID
        or audit.get("run_name") != SEALER_RUN_NAME
        or audit.get("source_plan_sha256") != candidate["source_plan_sha256"]
        or audit.get("source_request_sha256") != contract["source_request_sha256"]
        or audit.get("checkpoint_global_step") != 1
        or audit.get("candidate_path") != str(candidate_path)
        or audit.get("candidate_file_sha256") != candidate_file_sha256
        or audit.get("candidate_self_sha256") != candidate["sha256"].removeprefix("sha256:")
        or audit.get("request_sha256") != digest(contract)
        or audit.get("job_name") != SEALER_RUN_NAME
        or not isinstance(audit.get("pod_name"), str)
        or not audit["pod_name"]
        or any(
            _UUID.fullmatch(str(audit.get(field))) is None
            for field in ("job_uid", "pod_uid", "pod_owner_job_uid")
        )
        or audit.get("pod_owner_job_uid") != audit.get("job_uid")
        or audit.get("pod_uid") == audit.get("job_uid")
        or _image_digest(audit.get("runtime_image_id")) != IMAGE.rsplit("@sha256:", 1)[-1]
        or context != {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}
        or not _exact_integer(audit.get("container_restarts"), 0)
        or not _exact_integer(audit.get("gpus"), 0)
    ):
        raise ValueError("checkpoint sealer controller audit is incomplete or changed")
    _validate_sealer_kubernetes_observation(
        audit.get("kubernetes_observation"), audit, terminal=False
    )
    _observed_epoch(audit.get("observed_at"))


def _validate_sealer_release(
    release: dict,
    candidate: dict,
    candidate_path: Path,
    candidate_file_sha256: str,
    audit: dict,
    audit_path: Path,
    audit_file_sha256: str,
) -> None:
    if (
        set(release) != _SEALER_RELEASE_FIELDS
        or release.get("status") != "released"
        or release.get("cluster") != "dev"
        or release.get("kube_context") != DEV_KUBE_CONTEXT
        or release.get("namespace") != NAMESPACE
        or release.get("namespace_uid") != NAMESPACE_UID
        or release.get("run_name") != SEALER_RUN_NAME
        or release.get("source_plan_sha256") != candidate["source_plan_sha256"]
        or release.get("checkpoint_global_step") != 1
        or release.get("candidate_path") != str(candidate_path)
        or release.get("candidate_file_sha256") != candidate_file_sha256
        or release.get("candidate_self_sha256") != candidate["sha256"].removeprefix("sha256:")
        or release.get("controller_audit_path") != str(audit_path)
        or release.get("controller_audit_file_sha256") != audit_file_sha256
        or release.get("controller_audit_self_sha256") != audit["sha256"].removeprefix("sha256:")
        or any(
            release.get(key) != audit.get(key)
            for key in ("job_name", "job_uid", "pod_name", "pod_uid")
        )
        or release.get("job_complete") is not True
        or release.get("job_failed") is not False
        or release.get("pod_succeeded") is not True
        or not _exact_integer(release.get("container_exit_code"), 0)
        or release.get("job_present") is not False
        or release.get("pod_present") is not False
        or not _exact_integer(release.get("active_gpus"), 0)
        or _observed_epoch(release.get("terminated_at")) < _observed_epoch(audit.get("observed_at"))
        or _observed_epoch(release.get("terminated_at")) < candidate["created_at"]
        or _observed_epoch(release.get("observed_at"))
        < _observed_epoch(release.get("terminated_at"))
    ):
        raise ValueError("checkpoint sealer release is not exact, terminal and proven")
    terminal_audit = {
        **audit,
        "terminated_at": release["terminated_at"],
    }
    _validate_sealer_kubernetes_observation(
        release.get("terminal_observation"), terminal_audit, terminal=True
    )
    _validate_sealer_absence_observation(release.get("absence_observation"), release)


def seal(
    plan: dict,
    step: int,
    output: Path,
    candidate_path: Path,
    controller_audit_path: Path,
    sealer_release_path: Path,
) -> dict:
    """Finalize one usable manifest only after independent sealer Pod release."""
    import torch

    if torch.cuda.is_available():
        raise ValueError("RL checkpoint manifest finalization is CPU-only")
    _source_plan(plan)
    root = Path(plan["output_root"])
    if step != 1 or output != root / "RL_CHECKPOINT_STEP_1_MANIFEST.json":
        raise ValueError("checkpoint manifest must use the exact step-1 create-once path")
    if output.exists() or output.is_symlink():
        raise FileExistsError("checkpoint manifest destination already exists")
    candidate, candidate_file_sha256 = _bound_receipt(
        candidate_path,
        Path(SEALER_OUTPUT_ROOT) / "RL_CHECKPOINT_STEP_1_CANDIDATE.json",
        CHECKPOINT_CANDIDATE_SCHEMA,
    )
    verify_candidate(candidate, check_files=True)
    if candidate["source_plan"] != plan or candidate["checkpoint"]["step"] != step:
        raise ValueError("checkpoint candidate source identity changed")
    audit, audit_file_sha256 = _bound_receipt(
        controller_audit_path,
        Path(SEALER_OUTPUT_ROOT) / "CHECKPOINT_SEALER_CONTROLLER_AUDIT.json",
        CHECKPOINT_SEALER_AUDIT_SCHEMA,
    )
    _validate_sealer_audit(audit, candidate, candidate_path, candidate_file_sha256)
    sealer_release, sealer_release_file_sha256 = _bound_receipt(
        sealer_release_path,
        Path(SEALER_OUTPUT_ROOT) / "CHECKPOINT_SEALER_RELEASE.json",
        CHECKPOINT_SEALER_RELEASE_SCHEMA,
    )
    _validate_sealer_release(
        sealer_release,
        candidate,
        candidate_path,
        candidate_file_sha256,
        audit,
        controller_audit_path,
        audit_file_sha256,
    )
    for terminal in ("FAILED.json", "REJECTED.json"):
        if (Path(SEALER_OUTPUT_ROOT) / terminal).exists() or (root / terminal).exists():
            raise ValueError("checkpoint source or sealer has conflicting terminal evidence")
    runtime = candidate["runtime"]
    result = {
        key: candidate[key]
        for key in (
            "source_plan_sha256",
            "source_plan",
            "completion",
            "release",
            "source_stat_sha256",
            "checkpoint",
        )
    }
    result.update(
        schema=CHECKPOINT_SCHEMA,
        sealer={
            "image": IMAGE,
            **{
                key: runtime[key]
                for key in (
                    "runtime_sha256",
                    "native_sources_sha256",
                    "module_sha256",
                    "python",
                    "packages",
                    "uid",
                    "gid",
                )
            },
            "candidate": {
                "path": str(candidate_path),
                "file_sha256": candidate_file_sha256,
                "receipt": candidate,
            },
            "controller_audit": {
                "path": str(controller_audit_path),
                "file_sha256": audit_file_sha256,
                "receipt": audit,
            },
            "release": {
                "path": str(sealer_release_path),
                "file_sha256": sealer_release_file_sha256,
                "receipt": sealer_release,
            },
        },
    )
    if _source_stat_snapshot(result)[1] != result["source_stat_sha256"].removeprefix("sha256:"):
        raise ValueError("source changed before checkpoint manifest finalization")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return _write(output, result)


def _validate_inventory(manifest: dict) -> None:
    files = manifest.get("checkpoint", {}).get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("checkpoint manifest has no files")
    for name, item in files.items():
        path = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or not isinstance(item, dict)
            or set(item) != _INVENTORY_FIELDS
            or type(item["bytes"]) is not int
            or item["bytes"] <= 0
            or not _is_sha256(item["sha256"])
        ):
            raise ValueError("invalid checkpoint inventory entry")
    required = _required_checkpoint_files()
    names = set(files)
    extras = names - required
    if (
        not required <= names
        or extras != {name for name in extras if name.startswith("policy/huggingface/")}
        or "policy/huggingface/config.json" not in names
        or "policy/huggingface/tokenizer_config.json" not in names
    ):
        raise ValueError("checkpoint manifest file topology changed")


def _evidence(value: object) -> tuple[Path, dict]:
    if not isinstance(value, dict) or set(value) != _EVIDENCE_FIELDS:
        raise ValueError("source evidence fields changed")
    path = Path(value["path"])
    _canonical_sfs(str(path), jobs=True)
    if not _is_sha256(value["file_sha256"]) or not isinstance(value["receipt"], dict):
        raise ValueError("source evidence binding is incomplete")
    return path, value["receipt"]


def _validate_completion(value: dict, source: dict, step: int) -> None:
    if (
        set(value)
        != _fields(
            "status plan_sha256 checkpoint_global_step completed_batches completed_at "
            "optimizer_update_independently_verified checkpoint_reload_verified sha256"
        )
        or value.get("sha256", "").removeprefix("sha256:") != digest(_unsigned(value))
        or value.get("status") != "native_loop_returned"
        or value.get("plan_sha256") != digest(source)
        or value.get("checkpoint_global_step") != step
        or type(value.get("completed_batches")) is not int
        or value["completed_batches"] < 1
        or not isinstance(value.get("completed_at"), numbers.Real)
        or not math.isfinite(value["completed_at"])
        or value.get("optimizer_update_independently_verified") is not False
        or value.get("checkpoint_reload_verified") is not False
    ):
        raise ValueError("source run lacks an exact native completion receipt")


def _validate_final_sealer(sealer: object, manifest: dict, *, check_files: bool) -> None:
    if not isinstance(sealer, dict) or set(sealer) != _SEALER_FIELDS:
        raise ValueError("checkpoint sealer evidence fields changed")
    candidate_path, candidate = _evidence(sealer["candidate"])
    audit_path, audit = _evidence(sealer["controller_audit"])
    release_path, release = _evidence(sealer["release"])
    candidate_file_sha256 = sealer["candidate"]["file_sha256"].removeprefix("sha256:")
    audit_file_sha256 = sealer["controller_audit"]["file_sha256"].removeprefix("sha256:")
    _sealed(candidate, CHECKPOINT_CANDIDATE_SCHEMA)
    _sealed(audit, CHECKPOINT_SEALER_AUDIT_SCHEMA)
    _sealed(release, CHECKPOINT_SEALER_RELEASE_SCHEMA)
    runtime = candidate.get("runtime", {})
    if (
        sealer.get("image") != IMAGE
        or any(
            sealer.get(key) != runtime.get(key)
            for key in (
                "runtime_sha256",
                "native_sources_sha256",
                "module_sha256",
                "python",
                "packages",
                "uid",
                "gid",
            )
        )
        or any(
            manifest.get(key) != candidate.get(key)
            for key in (
                "source_plan_sha256",
                "source_plan",
                "completion",
                "release",
                "source_stat_sha256",
                "checkpoint",
            )
        )
    ):
        raise ValueError("checkpoint manifest differs from its audited candidate")
    _verify_checkpoint(candidate, check_files=check_files, candidate=True)
    _validate_sealer_audit(audit, candidate, candidate_path, candidate_file_sha256)
    _validate_sealer_release(
        release,
        candidate,
        candidate_path,
        candidate_file_sha256,
        audit,
        audit_path,
        audit_file_sha256,
    )
    if check_files:
        for path, binding in (
            (candidate_path, sealer["candidate"]),
            (audit_path, sealer["controller_audit"]),
            (release_path, sealer["release"]),
        ):
            current, file_sha256 = _json_snapshot(path)
            if current != binding["receipt"] or file_sha256 != binding["file_sha256"].removeprefix(
                "sha256:"
            ):
                raise ValueError("checkpoint sealer evidence changed after finalization")


def _verify_checkpoint(manifest: dict, *, check_files: bool, candidate: bool) -> None:
    _sealed(manifest, CHECKPOINT_CANDIDATE_SCHEMA if candidate else CHECKPOINT_SCHEMA)
    if set(manifest) != (_CANDIDATE_FIELDS if candidate else _MANIFEST_FIELDS):
        raise ValueError("checkpoint receipt fields changed")
    source = manifest.get("source_plan")
    _source_plan(source)
    if candidate:
        runtime = manifest.get("runtime")
        packages = runtime.get("packages") if isinstance(runtime, dict) else None
        if (
            not isinstance(runtime, dict)
            or set(runtime) != _CANDIDATE_RUNTIME_FIELDS
            or runtime.get("expected_image") != IMAGE
            or runtime.get("runtime_sha256") != digest(_runtime())
            or runtime.get("native_sources_sha256") != digest(source["native_sources"])
            or runtime.get("module_sha256") != _hash(Path(__file__))
            or not isinstance(runtime.get("python"), str)
            or not runtime["python"]
            or not isinstance(packages, dict)
            or set(packages) != {"ray", "torch", "transformers"}
            or any(not isinstance(value, str) or not value for value in packages.values())
            or (runtime.get("uid"), runtime.get("gid")) != (1000, 100)
            or runtime.get("cuda_available") is not False
            or not isinstance(manifest.get("created_at"), numbers.Real)
            or not math.isfinite(manifest["created_at"])
        ):
            raise ValueError("checkpoint candidate runtime identity changed")
    else:
        _validate_final_sealer(manifest.get("sealer"), manifest, check_files=check_files)
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_FIELDS:
        raise ValueError("checkpoint fields changed")
    step = checkpoint.get("step")
    checkpoint = Path(source["output_root"]) / f"checkpoints/global_step_{step}"
    if (
        manifest.get("source_plan_sha256") != digest(source)
        or not _is_sha256(manifest.get("source_stat_sha256"))
        or type(step) is not int
        or not 0 < step <= source["arguments"]["steps"]
        or manifest["checkpoint"].get("path") != str(checkpoint)
        or manifest["checkpoint"].get("world_size") != WORLD_SIZE
    ):
        raise ValueError("checkpoint manifest source bindings disagree")
    completion_path, completion = _evidence(manifest["completion"])
    release_path, release = _evidence(manifest["release"])
    _validate_completion(completion, source, step)
    _release(
        release,
        source,
        not_before=float(completion["completed_at"]),
        check_submission=check_files,
    )
    if (
        completion_path != Path(source["output_root"]) / "NATIVE_TRAINING_COMPLETE.json"
        or release_path != Path(source["output_root"]) / "SOURCE_RELEASE.json"
    ):
        raise ValueError("checkpoint manifest source evidence path changed")
    _validate_inventory(manifest)
    checkpoint_value = manifest["checkpoint"]
    if checkpoint_value.get("total_bytes") != sum(
        item["bytes"] for item in checkpoint_value["files"].values()
    ):
        raise ValueError("checkpoint manifest byte total disagrees")
    ranks = checkpoint_value.get("ranks")
    if (
        not _is_sha256(checkpoint_value.get("native_config_sha256"))
        or checkpoint_value.get("sampler_batches_in_epoch")
        != (step - 1) % (source["arguments"]["train_rows"] // source["arguments"]["groups"]) + 1
        or not isinstance(ranks, list)
        or [item.get("rank") for item in ranks] != list(range(WORLD_SIZE))
        or any(
            not isinstance(item, dict)
            or set(item) != _RANK_STATE_FIELDS
            or any(
                not _is_sha256(item.get(key))
                for key in (
                    "policy_state_sha256",
                    "optimizer_state_sha256",
                    "scheduler_state_sha256",
                    "rng_state_sha256",
                )
            )
            or item.get("optimizer_step") != step
            or item.get("scheduler_last_epoch") != step
            or item.get("rng_present") is not True
            or type(item.get("optimizer_states")) is not int
            or item["optimizer_states"] < 1
            or type(item.get("optimizer_step_states")) is not int
            or item["optimizer_step_states"] < 1
            for item in ranks
        )
    ):
        raise ValueError("checkpoint manifest state proof is incomplete")
    pointer = checkpoint_value.get("latest_pointer")
    if (
        not isinstance(pointer, dict)
        or set(pointer) != {"path", "file_sha256"}
        or pointer.get("path")
        != str(Path(source["output_root"]) / "checkpoints/latest_ckpt_global_step.txt")
        or not _is_sha256(pointer.get("file_sha256"))
    ):
        raise ValueError("checkpoint pointer binding is incomplete")
    if check_files:
        source_before, source_stat_sha256 = _source_stat_snapshot(manifest)
        if source_stat_sha256 != manifest["source_stat_sha256"].removeprefix("sha256:"):
            raise ValueError("sealed source stat identity changed")
        files = _checkpoint_files(checkpoint)
        current, _ = _inventory(files)
        expected = {
            name: {**item, "sha256": item["sha256"].removeprefix("sha256:")}
            for name, item in checkpoint_value["files"].items()
        }
        if current != expected:
            raise ValueError("checkpoint inventory changed after sealing")
        if _checkpoint_state(source, files, step) != {
            key: checkpoint_value[key]
            for key in ("native_config_sha256", "sampler_batches_in_epoch", "ranks")
        }:
            raise ValueError("checkpoint optimizer/trainer/sampler state changed")
        path = Path(pointer["path"])
        payload, file_sha256 = _file_snapshot(path)
        if (
            path != Path(source["output_root"]) / "checkpoints/latest_ckpt_global_step.txt"
            or file_sha256 != pointer["file_sha256"].removeprefix("sha256:")
            or payload.decode().strip() != str(step)
        ):
            raise ValueError("latest checkpoint pointer changed")
        if _json_snapshot(completion_path) != (
            completion,
            manifest["completion"]["file_sha256"].removeprefix("sha256:"),
        ) or _json_snapshot(release_path) != (
            release,
            manifest["release"]["file_sha256"].removeprefix("sha256:"),
        ):
            raise ValueError("source terminal or release evidence changed")
        source_after, source_stat_sha256 = _source_stat_snapshot(manifest)
        if source_after != source_before or source_stat_sha256 != manifest[
            "source_stat_sha256"
        ].removeprefix("sha256:"):
            raise ValueError("source changed during full manifest verification")


def verify_candidate(candidate: dict, *, check_files: bool) -> None:
    _verify_checkpoint(candidate, check_files=check_files, candidate=True)


def verify_manifest(manifest: dict, *, check_files: bool) -> None:
    _verify_checkpoint(manifest, check_files=check_files, candidate=False)


def _reward_reference(
    value: object,
    expected: Path,
    schema: str | None,
    *,
    extra_fields: frozenset[str] = frozenset(),
) -> tuple[dict, str]:
    """Reopen one exact producer artifact; a reference is never evidence itself."""
    if not isinstance(value, dict) or set(value) != (_REWARD_REFERENCE_FIELDS | extra_fields):
        raise ValueError("reward-canary evidence reference fields changed")
    path = Path(value.get("path", ""))
    if path != expected:
        raise ValueError("reward-canary evidence path changed")
    _canonical_sfs(str(path), jobs=True)
    receipt, file_sha256 = _json_snapshot(path)
    self_sha256 = receipt.get("sha256") if isinstance(receipt, dict) else None
    if (
        not _is_sha256(value.get("file_sha256"))
        or file_sha256 != str(value["file_sha256"]).removeprefix("sha256:")
        or not _is_sha256(value.get("receipt_self_sha256"))
        or not isinstance(self_sha256, str)
        or self_sha256.removeprefix("sha256:")
        != str(value["receipt_self_sha256"]).removeprefix("sha256:")
        or self_sha256.removeprefix("sha256:") != digest(_unsigned(receipt))
    ):
        raise ValueError("reward-canary evidence file or self digest changed")
    if schema is not None:
        _sealed(receipt, schema)
    return receipt, file_sha256


def _validate_reward_wandb(value: dict, source: dict, episode: dict, update: dict) -> None:
    """Recompute the scalar-only W&B/local-history cross-binding without trusting flags."""
    from .skyrl_training import (
        REWARD_CANARY_WANDB_SCHEMA,
        _finite_scalar_rows,
        _scalar_step_view,
    )

    _sealed(value, REWARD_CANARY_WANDB_SCHEMA)
    args = source["arguments"]
    identity = value.get("identity")
    remote = _finite_scalar_rows(value.get("remote_scalar_history"), label="sealed W&B")
    local_path = Path(value.get("local_metrics_path", ""))
    expected_local_path = Path(source["output_root"]) / "metrics.jsonl"
    if local_path != expected_local_path:
        raise ValueError("reward-canary local W&B history path changed")
    payload, local_file_sha256 = _file_snapshot(local_path)
    try:
        local_raw = [json.loads(line) for line in payload.splitlines()]
    except (TypeError, ValueError) as exc:
        raise ValueError("reward-canary local scalar history is invalid JSONL") from exc
    local = _finite_scalar_rows(local_raw, label="sealed local")
    remote_by_step = _scalar_step_view(remote, remote=True)
    local_by_step = _scalar_step_view(local, remote=False)
    step = remote_by_step.get(1, {})
    required_policy = {"policy/policy_loss", "policy/policy_lr", "policy/grad_norm"}
    expected_scalars = {
        "trainer/global_step": 1.0,
        "reward/avg_raw_reward": float(episode.get("train_reward_mean", math.nan)),
        "cyber/train/episodes": 8.0,
        "cyber/train/reward_mean": float(episode.get("train_reward_mean", math.nan)),
        "cyber/train/reward_nonzero_count": float(episode.get("train_nonzero_count", math.nan)),
        "cyber/train/reward_population_variance": float(
            episode.get("train_reward_population_variance", math.nan)
        ),
        "cyber/optimizer_step": float(update.get("final_optimizer_step", math.nan)),
    }
    required_scalars = value.get("required_step_1_scalars")
    if not isinstance(required_scalars, dict) or set(required_scalars) != (
        set(expected_scalars) | required_policy
    ):
        raise ValueError("reward-canary W&B required scalar set changed")
    expected_scalars.update({key: required_scalars[key] for key in required_policy})
    if (
        set(value) != _REWARD_WANDB_FIELDS
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(source)
        or identity
        != {
            "entity": args["wandb_entity"],
            "project": args["wandb_project"],
            "run_id": args["wandb_run_id"],
            "name": args["name"],
        }
        or value.get("state") != "finished"
        or remote != value.get("remote_scalar_history")
        or len(remote) > 10_000
        or value.get("remote_scalar_history_sha256", "").removeprefix("sha256:") != digest(remote)
        or value.get("remote_history_rows") != len(remote)
        or local_file_sha256
        != str(value.get("local_metrics_file_sha256", "")).removeprefix("sha256:")
        or value.get("local_scalar_history_sha256", "").removeprefix("sha256:") != digest(local)
        or value.get("local_history_rows") != len(local)
        or remote_by_step != local_by_step
        or set(remote_by_step) != {0, 1}
        or value.get("local_remote_step_history_sha256", "").removeprefix("sha256:")
        != digest(local_by_step)
        or any(not math.isfinite(item) for item in expected_scalars.values())
        or any(step.get(key) != item for key, item in expected_scalars.items())
        or required_scalars != expected_scalars
        or value.get("episode_audit_sha256", "").removeprefix("sha256:")
        != episode["sha256"].removeprefix("sha256:")
        or value.get("optimizer_update_proof_sha256", "").removeprefix("sha256:")
        != update["sha256"].removeprefix("sha256:")
        or not _exact_integer(value.get("logged_artifact_count"), 0)
        or not _exact_integer(value.get("rich_payload_count"), 0)
    ):
        raise ValueError("reward-canary W&B evidence is incomplete or mismatched")


def _validate_reward_update(value: dict, source: dict, manifest: dict, manifest_path: Path) -> None:
    """Reopen eight pre-update rank states and recompute the step-1 policy delta."""
    from .skyrl_training import (
        REWARD_CANARY_BASE_POLICY_RANK_SCHEMA,
        REWARD_CANARY_UPDATE_PROOF_SCHEMA,
    )

    _sealed(value, REWARD_CANARY_UPDATE_PROOF_SCHEMA)
    expected_fields = _fields(
        "schema source_plan_sha256 checkpoint_manifest base_policy_ranks "
        "initial_optimizer_step final_optimizer_step optimizer_updates "
        "changed_optimizer_ranks delta_basis changed_policy_ranks policy_delta_basis ranks sha256"
    )
    checkpoint_receipt, checkpoint_file_sha256 = _reward_reference(
        value.get("checkpoint_manifest"), manifest_path, CHECKPOINT_SCHEMA
    )
    if checkpoint_receipt != manifest or checkpoint_file_sha256 != str(
        value["checkpoint_manifest"]["file_sha256"]
    ).removeprefix("sha256:"):
        raise ValueError("reward-canary optimizer proof checkpoint changed")
    base_rows, compared = value.get("base_policy_ranks"), value.get("ranks")
    checkpoint_ranks = manifest["checkpoint"]["ranks"]
    rank_fields = _fields(
        "rank policy_state_sha256 optimizer_state_sha256 optimizer_states "
        "optimizer_step_states optimizer_step scheduler_last_epoch base_policy_state_sha256 "
        "checkpoint_policy_state_sha256 policy_changed"
    )
    base_fields = _fields(
        "schema source_plan_sha256 model_root model_revision weight_manifest_sha256 "
        "rank world_size initial_optimizer_states policy_state_sha256 sha256"
    )
    if (
        set(value) != expected_fields
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(source)
        or not isinstance(base_rows, list)
        or not isinstance(compared, list)
        or len(base_rows) != WORLD_SIZE
        or len(compared) != WORLD_SIZE
        or value.get("initial_optimizer_step") != 0
        or value.get("final_optimizer_step") != 1
        or value.get("optimizer_updates") != 1
        or value.get("changed_optimizer_ranks") != WORLD_SIZE
        or value.get("delta_basis")
        != "decoded_step1_optimizer_state_after_exact_no_resume_initialization"
        or value.get("policy_delta_basis")
        != "native_rank_state_digest_before_first_optim_step_vs_sealed_step1_checkpoint"
    ):
        raise ValueError("reward-canary optimizer proof shape changed")
    changes = 0
    for rank, (base_ref, row, checkpoint_row) in enumerate(
        zip(base_rows, compared, checkpoint_ranks, strict=True)
    ):
        expected_path = (
            Path(source["output_root"]) / "reward_canary_base_policy" / f"rank-{rank}.json"
        )
        base, _ = _reward_reference(
            base_ref,
            expected_path,
            REWARD_CANARY_BASE_POLICY_RANK_SCHEMA,
            extra_fields=frozenset({"rank", "policy_state_sha256"}),
        )
        checkpoint_subset = {
            key: checkpoint_row[key]
            for key in (
                "rank",
                "policy_state_sha256",
                "optimizer_state_sha256",
                "optimizer_states",
                "optimizer_step_states",
                "optimizer_step",
                "scheduler_last_epoch",
            )
        }
        base_digest = str(base.get("policy_state_sha256", ""))
        checkpoint_digest = str(checkpoint_row.get("policy_state_sha256", ""))
        changed = base_digest.removeprefix("sha256:") != checkpoint_digest.removeprefix("sha256:")
        expected_row = {
            **checkpoint_subset,
            "base_policy_state_sha256": base_digest,
            "checkpoint_policy_state_sha256": checkpoint_digest,
            "policy_changed": changed,
        }
        if (
            set(base_ref) != (_REWARD_REFERENCE_FIELDS | {"rank", "policy_state_sha256"})
            or base_ref.get("rank") != rank
            or base_ref.get("policy_state_sha256") != base_digest
            or set(base) != base_fields
            or base.get("source_plan_sha256", "").removeprefix("sha256:") != digest(source)
            or base.get("model_root") != source["model"]["root"]
            or base.get("model_revision") != source["model"]["revision"]
            or base.get("weight_manifest_sha256") != source["model"]["weight_manifest_sha256"]
            or base.get("rank") != rank
            or base.get("world_size") != WORLD_SIZE
            or base.get("initial_optimizer_states") != 0
            or not _is_sha256(base_digest)
            or set(row) != rank_fields
            or row != expected_row
            or checkpoint_subset["optimizer_step"] != 1
            or checkpoint_subset["scheduler_last_epoch"] != 1
            or checkpoint_subset["optimizer_states"] < 1
            or checkpoint_subset["optimizer_step_states"] < 1
        ):
            raise ValueError("reward-canary rank update proof is incomplete or mismatched")
        changes += int(changed)
    if changes < 1 or value.get("changed_policy_ranks") != changes:
        raise ValueError("reward-canary step-1 checkpoint lacks an independent policy delta")


def _validate_reward_terminal(
    terminal: dict,
    source: dict,
    manifest: dict,
    manifest_path: Path,
    manifest_file_sha256: str,
) -> None:
    """Reopen and rederive every reward-canary terminal prerequisite."""
    from .skyrl_training import (
        REWARD_CANARY_EPISODE_AUDIT_SCHEMA,
        REWARD_CANARY_UPDATE_PROOF_SCHEMA,
        REWARD_CANARY_WANDB_SCHEMA,
        reward_canary_episode_audit,
    )

    root = Path(source["output_root"])
    request_sha256 = digest(training_request(source))
    if (
        set(terminal) != _REWARD_TERMINAL_FIELDS
        or terminal.get("status") != "accepted"
        or terminal.get("source_run_name") != SOURCE_RUN_NAME
        or terminal.get("source_plan_sha256", "").removeprefix("sha256:") != digest(source)
        or terminal.get("source_request_sha256", "").removeprefix("sha256:") != request_sha256
        or terminal.get("runtime_user") != {"uid": 1000, "gid": 100}
    ):
        raise ValueError("reward-canary terminal identity changed")

    completion, _ = _reward_reference(
        terminal["native_completion"], root / "NATIVE_TRAINING_COMPLETE.json", None
    )
    _validate_completion(completion, source, 1)
    if completion != manifest["completion"]["receipt"]:
        raise ValueError("reward-canary completion differs from the checkpoint seal")

    terminal_manifest, terminal_manifest_file_sha256 = _reward_reference(
        terminal["checkpoint_manifest"], manifest_path, CHECKPOINT_SCHEMA
    )
    if terminal_manifest != manifest or terminal_manifest_file_sha256 != manifest_file_sha256:
        raise ValueError("reward-canary terminal checkpoint binding changed")

    audit, audit_file_sha256 = _reward_reference(
        terminal["source_controller_audit"],
        root / "SOURCE_CONTROLLER_AUDIT.json",
        CONTROLLER_AUDIT_SCHEMA,
    )
    release, release_file_sha256 = _reward_reference(
        terminal["source_external_release"], root / "SOURCE_RELEASE.json", RELEASE_SCHEMA
    )
    manifest_release = manifest["release"]
    if (
        release != manifest_release["receipt"]
        or release_file_sha256 != str(manifest_release["file_sha256"]).removeprefix("sha256:")
        or audit_file_sha256
        != str(release.get("controller_audit_file_sha256", "")).removeprefix("sha256:")
        or audit.get("sha256", "").removeprefix("sha256:")
        != str(release.get("controller_audit_sha256", "")).removeprefix("sha256:")
    ):
        raise ValueError("reward-canary controller/release linkage changed")
    _release(release, source, not_before=float(completion["completed_at"]), check_submission=True)
    observed_audit, observed_audit_file_sha256 = _controller_audit(source, release)
    if observed_audit != audit or observed_audit_file_sha256 != audit_file_sha256:
        raise ValueError("reward-canary controller audit changed while validating")
    pods = audit.get("pods")
    if (
        not isinstance(pods, list)
        or len(pods) != 1
        or (pods[0].get("runtime_uid"), pods[0].get("runtime_gid")) != (1000, 100)
        or (release.get("source_runtime_uid"), release.get("source_runtime_gid")) != (1000, 100)
    ):
        raise ValueError("reward-canary source process identity was not independently observed")

    episode, _ = _reward_reference(
        terminal["episode_audit"],
        root / "REWARD_CANARY_EPISODE_AUDIT.json",
        REWARD_CANARY_EPISODE_AUDIT_SCHEMA,
    )
    if _unsigned(episode) != reward_canary_episode_audit(source):
        raise ValueError("reward-canary episode evidence is not reproducible from raw receipts")

    update, _ = _reward_reference(
        terminal["optimizer_update_proof"],
        root / "REWARD_CANARY_UPDATE_PROOF.json",
        REWARD_CANARY_UPDATE_PROOF_SCHEMA,
    )
    _validate_reward_update(update, source, manifest, manifest_path)

    wandb, _ = _reward_reference(
        terminal["wandb_scalar_history"],
        root / "WANDB_SCALAR_HISTORY.json",
        REWARD_CANARY_WANDB_SCHEMA,
    )
    _validate_reward_wandb(wandb, source, episode, update)


def _source_binding(config: object, manifest: dict, manifest_path: Path, file_sha256: str) -> dict:
    """Bind the reload to one independently accepted reward-canary source run."""
    if not isinstance(config, dict) or set(config) != _SOURCE_FIELDS:
        raise ValueError("reward-canary source binding fields changed")
    source = manifest["source_plan"]
    release = manifest["release"]
    if manifest["checkpoint"]["step"] != 1:
        raise ValueError("reward-canary reload requires the exact step-1 checkpoint")
    request_sha256 = digest(training_request(source))
    expected = {
        "run_name": SOURCE_RUN_NAME,
        "plan_sha256": digest(source),
        "request_sha256": request_sha256,
        "checkpoint_global_step": 1,
        "checkpoint_manifest_path": str(manifest_path),
        "checkpoint_manifest_file_sha256": file_sha256,
        "checkpoint_manifest_self_sha256": manifest["sha256"].removeprefix("sha256:"),
        "source_external_release_path": release["path"],
        "source_external_release_file_sha256": release["file_sha256"].removeprefix("sha256:"),
        "source_external_release_self_sha256": release["receipt"]["sha256"].removeprefix("sha256:"),
    }
    for key, value in expected.items():
        supplied = config.get(key)
        if key.endswith("sha256") and isinstance(supplied, str):
            supplied = supplied.removeprefix("sha256:")
        if supplied != value:
            raise ValueError("reward-canary source identity or checkpoint linkage changed")

    terminal_path = Path(config.get("reward_terminal_receipt_path", ""))
    if terminal_path != Path(SOURCE_OUTPUT_ROOT) / "REWARD_CANARY_TERMINAL.json":
        raise ValueError("reward-canary terminal receipt path changed")
    _canonical_sfs(str(terminal_path), jobs=True)
    terminal, terminal_file_sha256 = _json_snapshot(terminal_path)
    _sealed(terminal, REWARD_TERMINAL_SCHEMA)
    terminal_self_sha256 = terminal["sha256"].removeprefix("sha256:")
    if terminal_file_sha256 != str(
        config.get("reward_terminal_receipt_file_sha256", "")
    ).removeprefix("sha256:") or terminal_self_sha256 != str(
        config.get("reward_terminal_receipt_self_sha256", "")
    ).removeprefix("sha256:"):
        raise ValueError("reward-canary terminal receipt digest changed")
    _validate_reward_terminal(terminal, source, manifest, manifest_path, file_sha256)
    return {
        **expected,
        "reward_terminal_receipt_path": str(terminal_path),
        "reward_terminal_receipt_file_sha256": terminal_file_sha256,
        "reward_terminal_receipt_self_sha256": terminal_self_sha256,
    }


def compile_reload(config: dict, *, relative_to: Path) -> dict:
    from .sft import _known, _sfs_root

    _known(
        config,
        {"schema", "name", "output_root", "source", "cluster", "lifecycle"},
        "SkyRL RL reload",
    )
    if config.get("schema") != RELOAD_CONFIG_SCHEMA:
        raise ValueError("RL reload configuration schema mismatch")
    source_config = config["source"]
    cluster = config["cluster"]
    lifecycle = config.get("lifecycle", dict(_LIFECYCLE))
    _known(cluster, {"target", "priority", "resources"}, "cluster")
    _known(lifecycle, set(_LIFECYCLE), "lifecycle")
    path = Path(source_config["checkpoint_manifest_path"])
    if not path.is_absolute():
        path = (relative_to / path).resolve()
    _canonical_sfs(str(path), jobs=True)
    expected = str(source_config["checkpoint_manifest_file_sha256"]).removeprefix("sha256:")
    manifest, file_sha256 = _json_snapshot(path)
    if not _is_sha256(source_config["checkpoint_manifest_file_sha256"]) or file_sha256 != expected:
        raise ValueError("checkpoint manifest file digest mismatch")
    verify_manifest(manifest, check_files=False)
    source = _source_binding(source_config, manifest, path, file_sha256)
    output = _sfs_root(config["output_root"], "reload output root")
    if (
        cluster.get("target") != "dev"
        or cluster.get("priority") != "c1"
        or cluster.get("resources", _RESOURCES) != _RESOURCES
        or lifecycle != _LIFECYCLE
        or config["name"] == manifest["source_plan"]["run_name"]
        or output == manifest["source_plan"]["output_root"]
        or Path(output).is_relative_to(Path(manifest["source_plan"]["output_root"]))
        or Path(manifest["source_plan"]["output_root"]).is_relative_to(Path(output))
    ):
        raise ValueError("reload must be a distinct dev c1 one-node operation")
    plan = {
        "schema": RELOAD_SCHEMA,
        "run_name": config["name"],
        "output_root": output,
        "source_manifest_path": str(path),
        "source_manifest_file_sha256": expected,
        "source_manifest": manifest,
        "source": source,
        "runtime_sha256": digest(_runtime()),
        "lifecycle": lifecycle,
        "execution": {
            "image": IMAGE,
            "cluster_target": "dev",
            "priority": "c1",
            "resources": dict(_RESOURCES),
            "workers": 1,
            "gpus_per_worker": WORLD_SIZE,
            "requeue_if_preempted": False,
        },
    }
    reload_request(plan)
    return plan


def _manifest_file(plan: dict) -> None:
    path = Path(plan["source_manifest_path"])
    _canonical_sfs(str(path), jobs=True)
    value, file_sha256 = _json_snapshot(path)
    if file_sha256 != plan["source_manifest_file_sha256"] or value != plan["source_manifest"]:
        raise ValueError("bound checkpoint manifest file changed")


def _reload_runtime_files(plan: dict) -> dict[str, str]:
    files = _runtime()
    files.update(
        {
            path + "/__init__.py": ""
            for path in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return files


def reload_request(plan: dict) -> dict:
    manifest = plan.get("source_manifest")
    verify_manifest(manifest, check_files=False)
    output = Path(plan.get("output_root", ""))
    source_output = Path(manifest["source_plan"]["output_root"])
    manifest_path = Path(plan.get("source_manifest_path", ""))
    expected_manifest_path = source_output / (
        f"RL_CHECKPOINT_STEP_{manifest['checkpoint']['step']}_MANIFEST.json"
    )
    if (
        set(plan) != _RELOAD_PLAN_FIELDS
        or plan.get("schema") != RELOAD_SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("lifecycle") != _LIFECYCLE
        or plan.get("execution")
        != {
            "image": IMAGE,
            "cluster_target": "dev",
            "priority": "c1",
            "resources": _RESOURCES,
            "workers": 1,
            "gpus_per_worker": WORLD_SIZE,
            "requeue_if_preempted": False,
        }
        or not isinstance(plan.get("run_name"), str)
        or not plan["run_name"]
        or _canonical_sfs(str(output), jobs=True) != str(output)
        or output.parent != Path("/mnt/sfs/jobs")
        or output.name != plan["run_name"]
        or output == source_output
        or output.is_relative_to(source_output)
        or source_output.is_relative_to(output)
        or manifest_path != expected_manifest_path
        or _canonical_sfs(str(manifest_path), jobs=True) != str(manifest_path)
        or not _is_sha256(plan.get("source_manifest_file_sha256"))
    ):
        raise ValueError("RL reload plan/runtime/output binding drift")
    _manifest_file(plan)
    if plan["source"] != _source_binding(
        plan["source"], manifest, manifest_path, plan["source_manifest_file_sha256"]
    ):
        raise ValueError("RL reload source provenance changed")
    resources = plan["execution"]["resources"]
    if quantity(resources["cpu_request"]) < 64 or quantity(resources["memory_request"]) < quantity(
        "512Gi"
    ):
        raise ValueError("Qwen RL reload requires its reviewed loading reservation")
    return bundled_request(
        {
            "name": plan["run_name"],
            "title": plan["run_name"] + " zero-update SkyRL RL reload",
            "run_dir": plan["output_root"],
            "image": IMAGE,
            "workers": 1,
            "gpus_per_worker": WORLD_SIZE,
            "resources": resources,
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "WANDB_MODE": "disabled",
            },
        },
        _reload_runtime_files(plan),
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan), "--run"],
    )


def _reload_native_config(source: dict):
    cfg = skyrl.diagnostic_native_config(skyrl.SkyRLConfig(**source["arguments"]))
    if cfg.trainer.policy.inference_only_init is not False or cfg.trainer.strategy != "fsdp":
        raise ValueError("native reload configuration changed")
    return cfg


def _runtime_identity_proof(source: dict) -> tuple[dict, dict]:
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("native RL reload runtime must use pinned image user 1000:100")
    modules = native_source()
    return (
        {
            "uid": os.geteuid(),
            "gid": os.getegid(),
            "native_sources_sha256": digest(source["native_sources"]),
        },
        modules,
    )


def validate_reload_preview(request: dict, preview: dict) -> dict:
    """Require every rendered reload Pod to run as the pinned native image user."""
    import yaml

    try:
        obj = yaml.safe_load(preview["manifest_yaml"])
        cluster = obj["spec"]["rayClusterSpec"]
        groups = [(1, cluster["headGroupSpec"]["template"])] + [
            (group["replicas"], group["template"]) for group in cluster.get("workerGroupSpecs", [])
        ]
        observed = 0
        for replicas, template in groups:
            if type(replicas) is not int or replicas < 0:
                raise JobsError("invalid reload preview replica count")
            if replicas == 0:
                continue
            pod = template["spec"]
            containers = pod["containers"]
            if len(containers) != 1:
                raise JobsError("reload preview must have one container per Pod")
            pod_context = pod.get("securityContext", {})
            container_context = containers[0].get("securityContext", {})
            effective = {
                key: container_context.get(key, pod_context.get(key))
                for key in ("runAsUser", "runAsGroup", "runAsNonRoot")
            }
            if effective != {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}:
                raise JobsError("reload preview runtime user differs from required 1000:100")
            observed += replicas
        if (
            obj["metadata"]["namespace"] != NAMESPACE
            or observed != 1
            or request.get("workers") != 1
            or request.get("gpus_per_worker") != WORLD_SIZE
            or request.get("priority_class") != "c1"
            or request.get("requeueIfPreempted") is not False
        ):
            raise JobsError("reload preview topology or queue contract changed")
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("malformed reload Jobs API preview") from exc
    return {"runtime_user": {"uid": 1000, "gid": 100}, "pods": observed}


def _state_digest(value) -> str:
    """Hash nested local state without retaining a second optimizer copy."""
    import numpy as np
    import torch

    hasher = hashlib.sha256()

    def update(item) -> None:
        if hasattr(item, "to_local"):
            item = item.to_local()
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            metadata = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            hasher.update(b"tensor")
            hasher.update(len(metadata).to_bytes(8, "big"))
            hasher.update(metadata)
            raw = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
            hasher.update(len(raw).to_bytes(8, "big"))
            hasher.update(raw)
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            metadata = json.dumps(
                {"dtype": str(array.dtype), "shape": list(array.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            hasher.update(b"numpy")
            hasher.update(len(metadata).to_bytes(8, "big"))
            hasher.update(metadata)
            raw = memoryview(array).cast("B")
            hasher.update(len(raw).to_bytes(8, "big"))
            hasher.update(raw)
        elif isinstance(item, dict):
            if any(type(key) not in {str, int} for key in item):
                raise ValueError("state dictionary key type changed")
            hasher.update(b"dict")
            ordered = sorted(item, key=lambda key: (type(key).__name__, str(key)))
            hasher.update(len(ordered).to_bytes(8, "big"))
            for key in ordered:
                update(key)
                update(item[key])
        elif isinstance(item, tuple):
            hasher.update(b"tuple")
            hasher.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
        elif isinstance(item, list):
            hasher.update(b"list")
            hasher.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
        elif item is None or type(item) in {bool, int, float, str}:
            payload = json.dumps(
                {"type": type(item).__name__, "value": item},
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            hasher.update(b"scalar")
            hasher.update(len(payload).to_bytes(8, "big"))
            hasher.update(payload)
        else:
            raise ValueError("unsupported state value in reload proof")

    update(value)
    return hasher.hexdigest()


def _sampler_restore(plan: dict, cfg) -> dict:
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader

    source = plan["source_manifest"]["source_plan"]
    checkpoint = plan["source_manifest"]["checkpoint"]
    state = _torch_load(Path(checkpoint["path"]) / "data.pt")
    generator = torch.Generator()
    generator.manual_seed(cfg.trainer.seed)
    loader = StatefulDataLoader(
        list(range(source["arguments"]["train_rows"])),
        batch_size=cfg.trainer.train_batch_size,
        shuffle=True,
        num_workers=cfg.data.dataloader.num_workers,
        persistent_workers=cfg.data.dataloader.persistent_workers,
        drop_last=True,
        generator=generator,
        multiprocessing_context="spawn" if cfg.data.dataloader.num_workers > 0 else None,
    )
    loader.load_state_dict(state)
    if _state_digest(loader.state_dict()) != _state_digest(state):
        raise ValueError("native StatefulDataLoader did not restore the exact sampler state")
    return {
        "sampler_restored": True,
        "sampler_batches_in_epoch": checkpoint["sampler_batches_in_epoch"],
    }


def _tokenizer_check(source: dict) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source["model"]["root"], trust_remote_code=False, local_files_only=True
    )
    if (
        not isinstance(tokenizer.chat_template, str)
        or "sha256:" + digest_template(tokenizer) != source["data"]["template_sha256"]
    ):
        raise ValueError("reload tokenizer/chat-template binding changed")


def preflight(plan: dict) -> dict:
    import torch

    source = plan["source_manifest"]["source_plan"]
    runtime_identity, _ = _runtime_identity_proof(source)
    if torch.cuda.is_available():
        raise ValueError("RL reload preflight is CPU-only")
    request = reload_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("reload output already exists")
    verify_manifest(plan["source_manifest"], check_files=True)
    check_inputs(source)
    cfg = _reload_native_config(source)
    _tokenizer_check(source)
    sampler = _sampler_restore(plan, cfg)
    _, source_stat_sha256 = _source_stat_snapshot(plan["source_manifest"])
    if source_stat_sha256 != plan["source_manifest"]["source_stat_sha256"].removeprefix("sha256:"):
        raise ValueError("source changed during CPU preflight")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": runtime_identity["uid"], "gid": runtime_identity["gid"]},
        "native_sources_sha256": runtime_identity["native_sources_sha256"],
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "checkpoint_manifest_sha256": plan["source_manifest"]["sha256"],
        "source": plan["source"],
        "source_stat_sha256": source_stat_sha256,
        "checkpoint_files": len(plan["source_manifest"]["checkpoint"]["files"]),
        "rank_count": WORLD_SIZE,
        "optimizer_updates": 0,
        "rollouts": 0,
        "wandb": False,
        "base_model_and_tokenizer_verified": True,
        **sampler,
        "gpu_reload_qualified": False,
    }


def _policy_state_digest(state: object) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("policy state must be a nonempty mapping")
    return _state_digest(dict(state))


def _reload_worker(plan: dict):
    import torch
    import torch.distributed as dist
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    manifest = plan["source_manifest"]
    checkpoint = Path(manifest["checkpoint"]["path"]) / "policy"
    step = manifest["checkpoint"]["step"]
    model_root = manifest["source_plan"]["model"]["root"]

    class ReloadWorker(FSDPPolicyWorkerBase):
        def init_model(self, model_path, num_training_steps=None):
            identity, _ = _runtime_identity_proof(manifest["source_plan"])
            self._runtime_reply = {
                "runtime_uid": identity["uid"],
                "runtime_gid": identity["gid"],
                "native_sources_sha256": identity["native_sources_sha256"],
            }
            if (
                model_path != model_root
                or num_training_steps != manifest["source_plan"]["arguments"]["steps"]
            ):
                raise ValueError("reload worker model or optimizer horizon changed")
            super().init_model(model_path, num_training_steps=num_training_steps)
            if self.optimizer is None or self.scheduler is None or self.optimizer.state:
                return {
                    "rank": dist.get_rank(),
                    "world_size": dist.get_world_size(),
                    "rejected": "base_initialization_contract",
                    **self._runtime_reply,
                }
            self._base_policy_sha256 = _policy_state_digest(self.model.model.state_dict())
            return {
                "rank": dist.get_rank(),
                "world_size": dist.get_world_size(),
                "base_policy_sha256": self._base_policy_sha256,
                "initial_optimizer_states": len(self.optimizer.state),
                **self._runtime_reply,
            }

        def validate_reload(self, ckpt_dir):
            if str(Path(ckpt_dir)) != str(checkpoint):
                raise ValueError("reload worker checkpoint path changed")
            rank = dist.get_rank()
            expected = manifest["checkpoint"]["ranks"][rank]
            states = super().load_checkpoint(
                str(checkpoint),
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            optimizer_steps = [
                _number(item["step"])
                for item in self.optimizer.state.values()
                if isinstance(item, dict) and "step" in item
            ]
            scheduler_loaded = _state_digest(self.scheduler.state_dict())
            rng_loaded = _state_digest(self.strategy.get_rng_state())
            loaded = _policy_state_digest(self.model.model.state_dict())
            optimizer_loaded = _state_digest(self.optimizer.state_dict())
            if (
                states.get("rank") != rank
                or states.get("world_size") != WORLD_SIZE
                or states.get("fsdp_strategy") != "fsdp"
                or dist.get_world_size() != WORLD_SIZE
                or not optimizer_steps
                or any(value != step for value in optimizer_steps)
                or loaded != expected["policy_state_sha256"].removeprefix("sha256:")
                or optimizer_loaded != expected["optimizer_state_sha256"].removeprefix("sha256:")
                or scheduler_loaded != expected["scheduler_state_sha256"].removeprefix("sha256:")
                or rng_loaded != expected["rng_state_sha256"].removeprefix("sha256:")
            ):
                return {
                    "rank": rank,
                    "world_size": WORLD_SIZE,
                    "rejected": "native_state_restore_contract",
                    **self._runtime_reply,
                }
            self.model.eval()
            token_ids = torch.tensor([[1, 2, 3, 4]], device=torch.cuda.current_device())
            mask = torch.ones_like(token_ids)
            with torch.inference_mode():
                output = self.model.model(
                    input_ids=token_ids,
                    attention_mask=mask,
                    use_cache=False,
                    return_dict=True,
                )
            logits = output.logits
            policy_after_forward = _policy_state_digest(self.model.model.state_dict())
            optimizer_after_forward = _state_digest(self.optimizer.state_dict())
            scheduler_after_forward = _state_digest(self.scheduler.state_dict())
            rng_after_forward = _state_digest(self.strategy.get_rng_state())
            if (
                logits.ndim != 3
                or tuple(logits.shape[:2]) != (1, 4)
                or not bool(torch.isfinite(logits).all().item())
                or any(parameter.grad is not None for parameter in self.model.parameters())
                or policy_after_forward != loaded
                or optimizer_after_forward != optimizer_loaded
                or scheduler_after_forward != scheduler_loaded
                or rng_after_forward != rng_loaded
            ):
                return {
                    "rank": rank,
                    "world_size": WORLD_SIZE,
                    "rejected": "native_state_restore_contract",
                    **self._runtime_reply,
                }
            return {
                "rank": rank,
                "world_size": WORLD_SIZE,
                "base_policy_sha256": self._base_policy_sha256,
                "loaded_policy_sha256": loaded,
                "post_forward_policy_sha256": policy_after_forward,
                "optimizer_states": len(self.optimizer.state),
                "optimizer_step_states": len(optimizer_steps),
                "optimizer_step": step,
                "loaded_optimizer_sha256": optimizer_loaded,
                "post_forward_optimizer_sha256": optimizer_after_forward,
                "loaded_scheduler_sha256": scheduler_loaded,
                "post_forward_scheduler_sha256": scheduler_after_forward,
                "rng_state_sha256": rng_loaded,
                "post_forward_rng_sha256": rng_after_forward,
                "forward_finite": True,
                "forward_batch": 1,
                "forward_tokens": 4,
                "gradients_created": 0,
                "optimizer_updates": 0,
                **self._runtime_reply,
            }

        def optim_step(self, *args, **kwargs):
            raise RuntimeError("zero-update reload must never call the optimizer")

        def shutdown(self):
            if dist.is_initialized():
                dist.destroy_process_group()
            self.model = self.optimizer = self.scheduler = None
            return True

    return ReloadWorker


def _validate_rank_replies(
    replies: object, checkpoint: dict, native_sources_sha256: str
) -> list[dict]:
    step = checkpoint["step"]
    expected = {item["rank"]: item for item in checkpoint["ranks"]}
    if not isinstance(replies, list) or len(replies) != WORLD_SIZE:
        raise ValueError("reload result must cover exactly eight ranks")
    if any(
        not isinstance(item, dict) or type(item.get("rank")) is not int for item in replies
    ) or sorted(item["rank"] for item in replies) != list(range(WORLD_SIZE)):
        raise ValueError("reload rank coverage changed")
    rejected = False
    for item in replies:
        if set(item) == _REJECTED_RANK_FIELDS:
            if (
                not _exact_integer(item["world_size"], WORLD_SIZE)
                or item["rejected"] != "native_state_restore_contract"
                or not _exact_integer(item["runtime_uid"], 1000)
                or not _exact_integer(item["runtime_gid"], 100)
                or item["native_sources_sha256"] != native_sources_sha256
            ):
                raise ValueError("malformed native-state rejection receipt")
            rejected = True
            continue
        rank = item["rank"]
        sealed_rank = expected.get(rank, {})
        if (
            set(item) != _RELOAD_RANK_FIELDS
            or not _exact_integer(item["world_size"], WORLD_SIZE)
            or not _exact_integer(item["runtime_uid"], 1000)
            or not _exact_integer(item["runtime_gid"], 100)
            or item["native_sources_sha256"] != native_sources_sha256
            or not _exact_integer(item["optimizer_step"], step)
            or item["optimizer_states"] != expected[item["rank"]]["optimizer_states"]
            or item["optimizer_step_states"] != expected[item["rank"]]["optimizer_step_states"]
            or item["loaded_policy_sha256"]
            != sealed_rank.get("policy_state_sha256", "").removeprefix("sha256:")
            or item["loaded_optimizer_sha256"]
            != sealed_rank.get("optimizer_state_sha256", "").removeprefix("sha256:")
            or item["loaded_scheduler_sha256"]
            != sealed_rank.get("scheduler_state_sha256", "").removeprefix("sha256:")
            or item["rng_state_sha256"]
            != sealed_rank.get("rng_state_sha256", "").removeprefix("sha256:")
            or not _exact_integer(item["optimizer_updates"], 0)
            or not _exact_integer(item["gradients_created"], 0)
            or not _exact_integer(item["forward_batch"], 1)
            or not _exact_integer(item["forward_tokens"], 4)
            or item["forward_finite"] is not True
            or type(item["optimizer_states"]) is not int
            or item["optimizer_states"] < 1
            or type(item["optimizer_step_states"]) is not int
            or item["optimizer_step_states"] < 1
            or any(
                not isinstance(item[key], str) or re.fullmatch(r"[a-f0-9]{64}", item[key]) is None
                for key in (
                    "base_policy_sha256",
                    "loaded_policy_sha256",
                    "post_forward_policy_sha256",
                    "loaded_optimizer_sha256",
                    "post_forward_optimizer_sha256",
                    "loaded_scheduler_sha256",
                    "post_forward_scheduler_sha256",
                    "rng_state_sha256",
                    "post_forward_rng_sha256",
                )
            )
            or item["loaded_policy_sha256"] != item["post_forward_policy_sha256"]
            or item["loaded_optimizer_sha256"] != item["post_forward_optimizer_sha256"]
            or item["loaded_scheduler_sha256"] != item["post_forward_scheduler_sha256"]
            or item["rng_state_sha256"] != item["post_forward_rng_sha256"]
        ):
            raise ValueError("zero-update all-rank reload proof is incomplete")
    if rejected:
        raise _ReloadContractRejected("native_state_restore_contract")
    if not any(item["loaded_policy_sha256"] != item["base_policy_sha256"] for item in replies):
        raise ValueError("checkpoint policy is indistinguishable from the exact base")
    return replies


def _validate_initialization_replies(replies: object, native_sources_sha256: str) -> None:
    if (
        not isinstance(replies, list)
        or len(replies) != WORLD_SIZE
        or any(not isinstance(item, dict) or type(item.get("rank")) is not int for item in replies)
        or sorted(item["rank"] for item in replies) != list(range(WORLD_SIZE))
    ):
        raise ValueError("base policy initialization rank coverage changed")
    rejected = False
    for item in replies:
        if set(item) == _REJECTED_RANK_FIELDS:
            if (
                not _exact_integer(item["world_size"], WORLD_SIZE)
                or item["rejected"] != "base_initialization_contract"
                or not _exact_integer(item["runtime_uid"], 1000)
                or not _exact_integer(item["runtime_gid"], 100)
                or item["native_sources_sha256"] != native_sources_sha256
            ):
                raise ValueError("malformed base-initialization rejection receipt")
            rejected = True
        elif (
            set(item) != _INITIALIZED_RANK_FIELDS
            or not _exact_integer(item["world_size"], WORLD_SIZE)
            or not _exact_integer(item["initial_optimizer_states"], 0)
            or not _exact_integer(item["runtime_uid"], 1000)
            or not _exact_integer(item["runtime_gid"], 100)
            or item["native_sources_sha256"] != native_sources_sha256
            or not isinstance(item["base_policy_sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", item["base_policy_sha256"]) is None
        ):
            raise ValueError("base policy did not initialize on all eight ranks")
    if rejected:
        raise _ReloadContractRejected("base_initialization_contract")


def _validate_cleanup(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != _CLEANUP_FIELDS
        or value.get("active_owned_actors") != 0
        or value.get("active_owned_placement_groups") != 0
        # One short-lived CPU credential-hygiene probe plus eight policy ranks.
        or value.get("tracked_ray_actors") != WORLD_SIZE + 1
        or value.get("tracked_engine_actors") != 0
        or value.get("tracked_placement_groups") != 1
        or value.get("tracked_routers") != 0
        or type(value.get("graceful_actor_shutdown")) is not bool
        or value.get("cleanup_proven") is not True
    ):
        raise ValueError("internal all-rank Ray cleanup proof is incomplete")


def _owned_output_root(plan: dict) -> Path:
    reload_request(plan)
    root = Path(plan["output_root"])
    if (
        os.environ.get("RUN_DIR") != str(root)
        or not root.is_dir()
        or root.is_symlink()
        or {path.name for path in root.iterdir()} != {".runtime"}
    ):
        raise ValueError("reload output is not a fresh Jobs API-owned directory")
    return root


def _runtime_identity(plan: dict) -> tuple[str, str]:
    runtime_run_id, rayjob_name = (
        os.environ.get("FLEET_RUN_ID", ""),
        os.environ.get("FLEET_RUN_NAME", ""),
    )
    if (
        _UUID.fullmatch(runtime_run_id) is None
        or re.fullmatch(re.escape(plan["run_name"]) + r"-[a-f0-9]{8}", rayjob_name) is None
    ):
        raise ValueError("Jobs API runtime identity is absent or changed")
    return runtime_run_id, rayjob_name


def _output_evidence(
    plan: dict,
    root: Path,
    *,
    added: set[str] | None = None,
    identity: tuple[str, str] | None = None,
) -> None:
    """Validate every allowed runtime/output entry without reading private logs."""
    expected_top = {".runtime", "private-native-logs", "RELOAD_STARTED.json"} | (added or set())
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != expected_top
    ):
        raise ValueError("zero-update reload output topology changed")

    runtime = root / ".runtime"
    expected = _reload_runtime_files(plan)
    actual_files, actual_directories = {}, set()
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError("reload runtime bundle is not a regular directory")
    for path in runtime.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("reload runtime bundle contains an unsafe entry")
        relative = str(path.relative_to(runtime))
        if path.is_dir():
            actual_directories.add(relative)
        else:
            actual_files[relative] = path
    expected_directories = {
        str(parent) for name in expected for parent in Path(name).parents if str(parent) != "."
    }
    if (
        set(actual_files) != set(expected)
        or actual_directories != expected_directories
        or any(
            _file_snapshot(actual_files[name])[0].decode() != value
            for name, value in expected.items()
        )
    ):
        raise ValueError("reload runtime bundle changed")

    logs = root / "private-native-logs"
    if (
        logs.is_symlink()
        or not logs.is_dir()
        or logs.stat().st_mode & 0o777 != 0o700
        or {path.name for path in logs.iterdir()} != {"infra.log"}
    ):
        raise ValueError("reload private-log topology changed")
    infra = logs / "infra.log"
    if infra.is_symlink() or not infra.is_file() or infra.stat().st_mode & 0o777 != 0o600:
        raise ValueError("reload private infrastructure log changed")

    started = _json(root / "RELOAD_STARTED.json")
    _sealed(started, RELOAD_STARTED_SCHEMA)
    runtime_run_id, rayjob_name = _runtime_identity(plan) if identity is None else identity
    if (
        set(started)
        != _fields(
            "schema status plan_sha256 request_sha256 runtime_run_id rayjob_name "
            "checkpoint_manifest_sha256 source driver_runtime_identity optimizer_updates "
            "rollouts sha256"
        )
        or started.get("status") != "started"
        or started.get("plan_sha256") != digest(plan)
        or started.get("request_sha256") != digest(reload_request(plan))
        or (started.get("runtime_run_id"), started.get("rayjob_name"))
        != (runtime_run_id, rayjob_name)
        or started.get("checkpoint_manifest_sha256") != plan["source_manifest"]["sha256"]
        or started.get("source") != plan["source"]
        or started.get("driver_runtime_identity")
        != {
            "uid": 1000,
            "gid": 100,
            "native_sources_sha256": digest(
                plan["source_manifest"]["source_plan"]["native_sources"]
            ),
        }
        or started.get("optimizer_updates") != 0
        or started.get("rollouts") != 0
    ):
        raise ValueError("reload start receipt changed")


def _run_reload(plan: dict, *, root: Path | None = None) -> dict:
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from .skyrl_training import _probe_worker_credentials

    root = _owned_output_root(plan) if root is None else root
    _manifest_file(plan)
    manifest = plan["source_manifest"]
    verify_manifest(manifest, check_files=False)
    source = manifest["source_plan"]
    driver_runtime_identity, modules = _runtime_identity_proof(source)
    runtime_run_id, rayjob_name = _runtime_identity(plan)
    request_sha256 = digest(reload_request(plan))
    _write(
        root / "RELOAD_STARTED.json",
        {
            "schema": RELOAD_STARTED_SCHEMA,
            "status": "started",
            "plan_sha256": digest(plan),
            "request_sha256": request_sha256,
            "runtime_run_id": runtime_run_id,
            "rayjob_name": rayjob_name,
            "checkpoint_manifest_sha256": manifest["sha256"],
            "source": plan["source"],
            "driver_runtime_identity": driver_runtime_identity,
            "optimizer_updates": 0,
            "rollouts": 0,
        },
    )
    cfg = _reload_native_config(source)
    environment, _, scrubbed = _ray_environment(plan, cfg, modules, diagnostic=True)
    actor_environment = _scrubbed_actor_environment(environment, scrubbed)
    source_before, source_stat_sha256 = _source_stat_snapshot(manifest)
    terminal_path = Path(plan["source"]["reward_terminal_receipt_path"])
    terminal_before = _stat(terminal_path)
    if source_stat_sha256 != manifest["source_stat_sha256"].removeprefix("sha256:"):
        raise RuntimeError("source changed before reload")
    sampler = _sampler_restore(plan, cfg)
    ownership = _DiagnosticOwnership()
    ray_initialization_attempted = False
    ray_started = False
    error = None
    replies = None
    try:
        with (
            hard_deadline(
                plan["lifecycle"]["startup_timeout_seconds"], "SkyRL RL reload"
            ) as deadline,
            _bounded_ray_get(ray, deadline),
        ):
            ray_initialization_attempted = True
            ray.init(
                address="auto",
                log_to_driver=False,
                runtime_env={"env_vars": actor_environment},
            )
            ray_started = True
            ownership.job_id = _ray_job_id_hex(ray.get_runtime_context().get_job_id())
            _probe_worker_credentials(
                ray,
                ownership,
                actor_environment,
                _credential_names(os.environ, actor_environment),
                expected_nodes=1,
                expected_gpus_per_node=WORLD_SIZE,
            )
            raw_pg = placement_group(
                [{"GPU": 1, "CPU": 1} for _ in range(WORLD_SIZE)], strategy="STRICT_PACK"
            )
            _add_identity(ownership.placement_groups, raw_pg)
            ray.get(raw_pg.ready(), timeout=deadline.remaining())
            Worker = ray.remote(num_gpus=1, max_restarts=0)(_reload_worker(plan))
            handles = []
            master_addr = master_port = None
            for rank in range(WORLD_SIZE):
                strategy = PlacementGroupSchedulingStrategy(
                    placement_group=raw_pg, placement_group_bundle_index=rank
                )
                actor = Worker.options(
                    num_cpus=1,
                    num_gpus=1,
                    runtime_env={"env_vars": actor_environment},
                    scheduling_strategy=strategy,
                ).remote(
                    cfg=cfg.trainer,
                    world_size=WORLD_SIZE,
                    rank=rank,
                    local_rank=rank,
                    master_addr=master_addr,
                    master_port=master_port,
                    sequence_parallel_size=1,
                    record_memory=False,
                )
                _add_identity(ownership.actors, actor)
                handles.append(actor)
                if rank == 0:
                    master_addr, master_port = ray.get(actor.get_master_addr_port.remote())
            ray.get([actor.init_worker_process_group.remote() for actor in handles])
            initialized = ray.get(
                [
                    actor.init_model.remote(
                        source["model"]["root"],
                        num_training_steps=source["arguments"]["steps"],
                    )
                    for actor in handles
                ]
            )
            native_sources_sha256 = digest(source["native_sources"])
            _validate_initialization_replies(initialized, native_sources_sha256)
            replies = ray.get(
                [
                    actor.validate_reload.remote(
                        str(Path(manifest["checkpoint"]["path"]) / "policy")
                    )
                    for actor in handles
                ]
            )
            _validate_rank_replies(replies, manifest["checkpoint"], native_sources_sha256)
    except BaseException as exc:
        error = exc
    ray_connected = ray_started or (
        ray_initialization_attempted
        and callable(getattr(ray, "is_initialized", None))
        and ray.is_initialized()
    )
    if ray_connected:
        try:
            cleanup = _cleanup_engine_diagnostic(
                None, ray, ownership, plan["lifecycle"]["cleanup_timeout_seconds"]
            )
        except BaseException as cleanup_error:
            if error is not None:
                raise BaseExceptionGroup(
                    "RL reload and owned-resource cleanup both failed", [error, cleanup_error]
                ) from None
            raise
    elif error is not None:
        with suppress(Exception):
            ray.shutdown()
        raise error
    else:
        raise RuntimeError("RL reload did not initialize Ray")
    _validate_cleanup(cleanup)
    _manifest_file(plan)
    source_after, source_stat_sha256 = _source_stat_snapshot(manifest)
    source_changed = (
        source_after != source_before
        or source_stat_sha256 != manifest["source_stat_sha256"].removeprefix("sha256:")
        or _stat(terminal_path) != terminal_before
    )
    if error is not None:
        if source_changed:
            raise BaseExceptionGroup(
                "RL reload failed and its sealed source also changed",
                [error, ValueError("source changed during reload")],
            ) from None
        raise error
    if source_changed:
        raise RuntimeError("source changed during reload")
    _source_binding(
        plan["source"],
        manifest,
        Path(plan["source_manifest_path"]),
        plan["source_manifest_file_sha256"],
    )
    if replies is None:
        raise RuntimeError("RL reload result or internal cleanup proof is absent")
    _output_evidence(plan, root)
    result = {
        "schema": RELOAD_RESULT_SCHEMA,
        "status": "reload_validated",
        "plan_sha256": digest(plan),
        "request_sha256": request_sha256,
        "runtime_run_id": runtime_run_id,
        "rayjob_name": rayjob_name,
        "source_manifest_sha256": manifest["sha256"],
        "source_manifest_file_sha256": plan["source_manifest_file_sha256"],
        "checkpoint_global_step": manifest["checkpoint"]["step"],
        "world_size": WORLD_SIZE,
        "ranks": replies,
        "sampler": sampler,
        "policy_changed_from_base": True,
        "base_model_and_tokenizer_verified": True,
        "source_checkpoint_unchanged": True,
        "optimizer_updates_executed": 0,
        "rollouts_executed": 0,
        "verifier_calls_executed": 0,
        "new_checkpoints_created": 0,
        "wandb_initialized": False,
        "internal_ray_resources_released": True,
        "internal_cleanup": cleanup,
        "external_job_gpu_release_verified": False,
        "external_release_evidence_required_after_process_exit": True,
        "source": plan["source"],
        "driver_runtime_identity": driver_runtime_identity,
        "completed_at": time.time(),
    }
    return _write(root / "RELOAD_VALIDATED.json", result)


def validate_reload_accepted(value: dict, *, check_files: bool) -> dict:
    """Recompute the accepted zero-update reload contract for promotion consumers."""
    _sealed(value, RELOAD_ACCEPTED_SCHEMA)
    if not isinstance(value, dict) or set(value) != _RELOAD_ACCEPTED_FIELDS:
        raise ValueError("accepted reload receipt fields changed")
    plan = value.get("plan")
    request = reload_request(plan)
    result, release = value.get("reload_result"), value.get("external_release")
    _sealed(result, RELOAD_RESULT_SCHEMA)
    _sealed(release, RELEASE_SCHEMA)
    completed_at = result.get("completed_at")
    expected_runtime_identity = {
        "uid": 1000,
        "gid": 100,
        "native_sources_sha256": digest(plan["source_manifest"]["source_plan"]["native_sources"]),
    }
    if (
        result.get("status") != "reload_validated"
        or result.get("plan_sha256") != digest(plan)
        or result.get("request_sha256") != digest(request)
        or (result.get("runtime_run_id"), result.get("rayjob_name"))
        != (release.get("runtime_run_id"), release.get("rayjob_name"))
        or result.get("source_manifest_sha256") != plan["source_manifest"]["sha256"]
        or result.get("source_manifest_file_sha256") != plan["source_manifest_file_sha256"]
        or result.get("checkpoint_global_step") != 1
        or result.get("source") != plan["source"]
        or result.get("driver_runtime_identity") != expected_runtime_identity
        or result.get("world_size") != WORLD_SIZE
        or result.get("optimizer_updates_executed") != 0
        or result.get("rollouts_executed") != 0
        or result.get("verifier_calls_executed") != 0
        or result.get("new_checkpoints_created") != 0
        or result.get("wandb_initialized") is not False
        or result.get("internal_ray_resources_released") is not True
        or result.get("source_checkpoint_unchanged") is not True
        or result.get("base_model_and_tokenizer_verified") is not True
        or result.get("policy_changed_from_base") is not True
        or result.get("external_job_gpu_release_verified") is not False
        or result.get("external_release_evidence_required_after_process_exit") is not True
        or not isinstance(completed_at, numbers.Real)
        or isinstance(completed_at, bool)
        or not math.isfinite(completed_at)
    ):
        raise ValueError("accepted reload result is incomplete or mismatched")
    _validate_cleanup(result.get("internal_cleanup"))
    _validate_rank_replies(
        result.get("ranks"),
        plan["source_manifest"]["checkpoint"],
        digest(plan["source_manifest"]["source_plan"]["native_sources"]),
    )
    _release(release, plan, not_before=float(completed_at), check_submission=check_files)
    if (
        value.get("status") != "accepted"
        or value.get("plan") != plan
        or value.get("plan_sha256") != digest(plan)
        or value.get("source_manifest_sha256") != plan["source_manifest"]["sha256"]
        or value.get("source") != plan["source"]
        or value.get("acceptance_runtime_identity") != expected_runtime_identity
        or value.get("reward_terminal_receipt_verified") is not True
        or value.get("source_checkpoint_unchanged_after_release") is not True
        or value.get("optimizer_updates_executed") != 0
        or value.get("rollouts_executed") != 0
        or value.get("verifier_calls_executed") != 0
        or value.get("internal_ray_resources_released") is not True
        or value.get("external_job_gpu_release_verified") is not True
        or value.get("production_promotion_requires_this_accepted_receipt") is not True
    ):
        raise ValueError("accepted reload promotion binding is incomplete")
    if check_files:
        root = Path(plan["output_root"])
        for path, expected, expected_sha256 in (
            (root / "RELOAD_VALIDATED.json", result, value["reload_result_file_sha256"]),
            (root / "RELOAD_RELEASE.json", release, value["external_release_file_sha256"]),
        ):
            observed, file_sha256 = _json_snapshot(path)
            if observed != expected or file_sha256 != str(expected_sha256).removeprefix("sha256:"):
                raise ValueError("accepted reload input changed after acceptance")
        verify_manifest(plan["source_manifest"], check_files=True)
    return {
        "plan_sha256": digest(plan),
        "source_manifest_sha256": plan["source_manifest"]["sha256"],
        "reward_terminal_receipt_self_sha256": plan["source"][
            "reward_terminal_receipt_self_sha256"
        ],
        "reload_result_sha256": result["sha256"],
        "release_sha256": release["sha256"],
    }


def accept_reload(plan: dict, release_evidence: Path, output: Path) -> dict:
    """Seal external Job/Pod release after the validator process has exited."""
    import torch

    if torch.cuda.is_available():
        raise ValueError("RL reload acceptance is CPU-only")
    acceptance_runtime_identity, _ = _runtime_identity_proof(plan["source_manifest"]["source_plan"])
    reload_request(plan)
    expected_output = Path(plan["output_root"]) / "RELOAD_ACCEPTED.json"
    expected_release = Path(plan["output_root"]) / "RELOAD_RELEASE.json"
    if output != expected_output:
        raise ValueError("reload acceptance must use the bound reload output root")
    if release_evidence != expected_release:
        raise ValueError("reload release evidence must use the bound reload output root")
    if output.exists() or output.is_symlink():
        raise FileExistsError("reload acceptance destination already exists")
    _canonical_sfs(str(release_evidence), jobs=True)
    _canonical_sfs(str(output), jobs=True)
    result, result_file_sha256 = _json_snapshot(Path(plan["output_root"]) / "RELOAD_VALIDATED.json")
    _sealed(result, RELOAD_RESULT_SCHEMA)
    release, release_file_sha256 = _json_snapshot(release_evidence)
    if set(result) != _RESULT_FIELDS:
        raise ValueError("reload validation result fields changed")
    completed_at = result.get("completed_at")
    if not isinstance(completed_at, numbers.Real) or not math.isfinite(completed_at):
        raise ValueError("reload validation completion time is invalid")
    _output_evidence(
        plan,
        Path(plan["output_root"]),
        added={
            "RELOAD_VALIDATED.json",
            "RELOAD_RELEASE.json",
            "RELOAD_SUBMISSION.jsonl",
            "RELOAD_CONTROLLER_AUDIT.json",
        },
        identity=(result.get("runtime_run_id", ""), result.get("rayjob_name", "")),
    )
    _release(release, plan, not_before=float(completed_at))
    _validate_rank_replies(
        result.get("ranks"),
        plan["source_manifest"]["checkpoint"],
        digest(plan["source_manifest"]["source_plan"]["native_sources"]),
    )
    cleanup = result.get("internal_cleanup")
    if (
        result.get("status") != "reload_validated"
        or result.get("plan_sha256") != digest(plan)
        or result.get("request_sha256") != digest(reload_request(plan))
        or (result.get("runtime_run_id"), result.get("rayjob_name"))
        != (release.get("runtime_run_id"), release.get("rayjob_name"))
        or result.get("source_manifest_sha256") != plan["source_manifest"]["sha256"]
        or result.get("source_manifest_file_sha256") != plan["source_manifest_file_sha256"]
        or result.get("checkpoint_global_step") != plan["source_manifest"]["checkpoint"]["step"]
        or result.get("source") != plan["source"]
        or result.get("driver_runtime_identity") != acceptance_runtime_identity
        or result.get("world_size") != WORLD_SIZE
        or result.get("optimizer_updates_executed") != 0
        or result.get("rollouts_executed") != 0
        or result.get("verifier_calls_executed") != 0
        or result.get("new_checkpoints_created") != 0
        or result.get("wandb_initialized") is not False
        or result.get("internal_ray_resources_released") is not True
        or result.get("source_checkpoint_unchanged") is not True
        or result.get("base_model_and_tokenizer_verified") is not True
        or result.get("policy_changed_from_base") is not True
        or result.get("sampler")
        != {
            "sampler_restored": True,
            "sampler_batches_in_epoch": plan["source_manifest"]["checkpoint"][
                "sampler_batches_in_epoch"
            ],
        }
        or result.get("external_job_gpu_release_verified") is not False
        or result.get("external_release_evidence_required_after_process_exit") is not True
    ):
        raise ValueError("reload validation result is incomplete or conflicting")
    _validate_cleanup(cleanup)
    _manifest_file(plan)
    verify_manifest(plan["source_manifest"], check_files=True)
    if any(
        (Path(plan["output_root"]) / name).exists()
        for name in ("FAILED.json", "RELOAD_REJECTED.json")
    ):
        raise ValueError("reload output contains conflicting terminal evidence")
    _output_evidence(
        plan,
        Path(plan["output_root"]),
        added={
            "RELOAD_VALIDATED.json",
            "RELOAD_RELEASE.json",
            "RELOAD_SUBMISSION.jsonl",
            "RELOAD_CONTROLLER_AUDIT.json",
        },
        identity=(result["runtime_run_id"], result["rayjob_name"]),
    )
    accepted = {
        "schema": RELOAD_ACCEPTED_SCHEMA,
        "status": "accepted",
        "plan": plan,
        "plan_sha256": digest(plan),
        "reload_result": result,
        "reload_result_file_sha256": result_file_sha256,
        "external_release": release,
        "external_release_file_sha256": release_file_sha256,
        "source_manifest_sha256": plan["source_manifest"]["sha256"],
        "source": plan["source"],
        "acceptance_runtime_identity": acceptance_runtime_identity,
        "reward_terminal_receipt_verified": True,
        "source_checkpoint_unchanged_after_release": True,
        "optimizer_updates_executed": 0,
        "rollouts_executed": 0,
        "verifier_calls_executed": 0,
        "internal_ray_resources_released": True,
        "external_job_gpu_release_verified": True,
        "production_promotion_requires_this_accepted_receipt": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return _write(output, accepted)


def _record_failure(root: Path | None, plan: object, error: BaseException) -> None:
    with suppress(Exception):
        terminals = ("FAILED.json", "RELOAD_VALIDATED.json", "RELOAD_REJECTED.json")
        if root is not None and not any((root / name).exists() for name in terminals):
            _write(
                root / "FAILED.json",
                {
                    "schema": "cyber_skyrl_rl_reload_failure_v1",
                    "status": "failed",
                    "plan_sha256": digest(plan) if isinstance(plan, dict) else None,
                    "error_class": type(error).__name__,
                    "automatic_retry": False,
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    failure_root = None
    try:
        plan = _json(args.plan)
        if digest(plan) != args.sha256 or not args.run:
            raise ValueError("RL reload plan digest or mode mismatch")
        failure_root = _owned_output_root(plan)
        result = _run_reload(plan, root=failure_root)
        print(json.dumps({key: result[key] for key in ("status", "sha256")}))
    except _ReloadContractRejected as exc:
        try:
            if failure_root is None:
                raise RuntimeError("reload rejection lacks an owned output root")
            _output_evidence(plan, failure_root)
            rejected = _write(
                failure_root / "RELOAD_REJECTED.json",
                {
                    "schema": RELOAD_REJECTED_SCHEMA,
                    "status": "rejected",
                    "plan_sha256": digest(plan),
                    "request_sha256": digest(reload_request(plan)),
                    "source": plan["source"],
                    "reason_code": exc.reason_code,
                    "optimizer_updates_executed": 0,
                    "rollouts_executed": 0,
                    "internal_ray_resources_released": True,
                    "automatic_retry": False,
                    "external_release_evidence_required": True,
                },
            )
        except BaseException as rejection_error:
            _record_failure(failure_root, locals().get("plan"), rejection_error)
            print(json.dumps({"status": "failed", "error_class": type(rejection_error).__name__}))
            raise SystemExit(1) from None
        print(json.dumps({key: rejected[key] for key in ("status", "reason_code", "sha256")}))
        return
    except BaseException as exc:
        _record_failure(failure_root, locals().get("plan"), exc)
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
