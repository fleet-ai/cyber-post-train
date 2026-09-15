"""Fail-closed scientific acceptance for a one-update Miles reward canary.

The native driver deliberately emits only operational completion.  This module
reopens the immutable plan, batches, private episode receipts, checkpoint seal,
scalar-only W&B observation, and UID-bound controller/release observations
before creating one sanitized acceptance receipt.  Reward values and task
content are inspected only to validate the private evidence and are never
copied into the acceptance receipt.
"""

from __future__ import annotations

import base64
import binascii
import copy
import gzip
import hashlib
import json
import math
import numbers
import re
import shlex
import subprocess
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

from .miles_cluster import cluster_profile, plan_cluster_target
from .miles_conversion import _write
from .miles_reload import CHECKPOINT_SCHEMA, _verify_checkpoint
from .miles_training import LONG_CONTEXT_SCHEMA, RUNTIME_FILES, job_request
from .miles_training import SCHEMA as TRAINING_SCHEMA
from .rl_episode import _validate as validate_episode_config
from .rl_runtime import sealed

TERMINAL_SCHEMA = "cyber_miles_reward_canary_terminal_v1"
SUBMISSION_SCHEMA = "cyber_miles_submitted_execution_binding_v1"
EPISODE_AUDIT_SCHEMA = "cyber_miles_reward_canary_episode_audit_v1"
WANDB_SCHEMA = "cyber_miles_wandb_scalar_observation_v1"
POLICY_DELTA_SCHEMA = "cyber_miles_policy_tensor_delta_observation_v2"
CONTROLLER_SCHEMA = "cyber_miles_controller_terminal_observation_v1"
RELEASE_SCHEMA = "cyber_miles_external_release_v1"
NAMESPACE = cluster_profile("dev").namespace
# Backwards-compatible name used by historical dev evidence and tests. New
# evidence derives its immutable namespace UID from the plan's cluster target.
NAMESPACE_UID = cluster_profile("dev").namespace_uid
WORLD_SIZE = 8

_SUBMITTED_RUNTIME_FILES = RUNTIME_FILES
_SUBMITTED_ENV = {
    "PYTHONPATH": "/root/Megatron-LM",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MILES_USE_LEGACY_ROLLOUT_V1": "0",
    "WANDB_MODE": "online",
    "WANDB_DISABLE_CODE": "true",
    "WANDB_CONSOLE": "off",
    "PYTHONUNBUFFERED": "1",
}

_MILES_UPDATE_METRICS = frozenset(
    {
        "train/step",
        "train/grad_norm",
        "train/loss",
        "train/lr-pg_0",
    }
)


def _sensitive_wandb_metric(name: str) -> bool:
    """Return whether a metric can disclose a reward or model outcome.

    Native Miles logs rollout rewards under ``rollout/`` and held-out outcomes
    under ``eval/<dataset>``.  Keep this deliberately broader than those two
    current spellings so a renamed score/pass/success field fails private and
    never enters a public receipt or one of its digests.
    """

    lowered = name.lower()
    return lowered.startswith(("rollout/", "eval/")) or any(
        token in lowered for token in ("reward", "score", "success", "pass_rate", "outcome")
    )


_SHA = re.compile(r"(?:sha256:)?[a-f0-9]{64}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_EPISODE_FILES = frozenset(
    {
        "binding.json",
        "create-intent.json",
        "instance.json",
        "conversation.json",
        "score-intent.json",
        "reward.json",
        "cleanup.json",
        "recording.json",
    }
)
_REFERENCE_FIELDS = frozenset({"path", "file_sha256", "receipt_sha256"})
_EPISODE_AUDIT_FIELDS = frozenset(
    {
        "schema",
        "source_plan_sha256",
        "accepted_episode_count",
        "train_episode_count",
        "dev_episode_count",
        "infrastructure_invalid_count",
        "truncated_count",
        "unique_authoritative_verifier_execution_count",
        "authoritative_verifier_execution_ids_sha256",
        "authoritative_verifier_bindings_sha256",
        "unique_challenge_instance_count",
        "challenge_instance_ids_sha256",
        "unique_evidence_run_count",
        "evidence_run_ids_sha256",
        "train_nonzero_reward_present",
        "train_within_group_reward_variance_present",
        "reward_values_included",
        "assistant_message_count",
        "tool_message_count",
        "report_submitted_count",
        "task_content_included",
    }
)
_UPDATE_PROOF_FIELDS = frozenset(
    {
        "optimizer_updates",
        "basis",
        "policy_tensor_payload_changed_from_base",
        "changed_policy_ranks",
        "counter_only_claim",
    }
)
_TERMINAL_FIELDS = frozenset(
    {
        "schema",
        "status",
        "source_run_name",
        "source_plan_sha256",
        "source_request_sha256",
        "submission_binding",
        "native_completion",
        "episode_audit",
        "optimizer_update_proof",
        "wandb_scalar_observation",
        "checkpoint_manifest",
        "policy_delta_observation",
        "controller_observation",
        "external_release",
        "external_gpu_release_verified",
        "reward_values_included",
        "task_content_included",
        "production_promotion_requires_reload_acceptance",
        "sha256",
    }
)
_SUBMISSION_FIELDS = frozenset(
    {
        "schema",
        "source_commit",
        "source_plan_path",
        "source_plan_sha256",
        "source_plan_file_sha256",
        "source_request_path",
        "source_request_sha256",
        "source_request_file_sha256",
        "runtime_bundle_sha256",
        "runtime_source_sha256",
        "runtime_source_commit_match",
        "api",
        "request",
        "jobs_api_post_count",
        "submitted_at",
        "secret_values_included",
        "task_content_included",
        "private_payload_included",
        "sha256",
    }
)
_SUBMISSION_REQUEST_FIELDS = frozenset(
    {
        "name",
        "run_dir",
        "image",
        "workers",
        "gpus_per_worker",
        "resources",
        "priority_class",
        "automatic_requeue",
        "secret_names",
        "environment_names",
        "runtime_module",
        "runtime_argv",
    }
)


def _unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "sha256"}


_STABLE_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def _json_snapshot(path: Path) -> tuple[Any, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("acceptance input must be a regular file")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if any(getattr(before, key) != getattr(after, key) for key in _STABLE_STAT_FIELDS):
        raise ValueError("acceptance input changed while reading")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("acceptance input is not valid JSON") from error
    return value, fleet.sha256(payload).removeprefix("sha256:")


def _read(path: Path, *, schema: str | None = None) -> tuple[dict[str, Any], str]:
    value, file_sha256 = _json_snapshot(path)
    if not isinstance(value, dict):
        raise ValueError("acceptance input must be a JSON object")
    if schema is None:
        if value.get("sha256", "").removeprefix("sha256:") != digest(_unsigned(value)):
            raise ValueError("acceptance input self-digest mismatch")
    else:
        sealed(value, schema)
    return value, file_sha256


def _reference(path: Path, value: Mapping[str, Any], file_sha256: str) -> dict[str, str]:
    return {
        "path": str(path),
        "file_sha256": "sha256:" + file_sha256.removeprefix("sha256:"),
        "receipt_sha256": "sha256:" + str(value["sha256"]).removeprefix("sha256:"),
    }


def _uuid(value: object, label: str) -> str:
    try:
        result = str(uuid.UUID(str(value)))
    except ValueError as error:
        raise ValueError(f"{label} is not a UUID") from error
    if result == "00000000-0000-0000-0000-000000000000":
        raise ValueError(f"{label} is a zero UUID")
    return result


def _time(value: object, label: str) -> float:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as error:
            raise ValueError(f"{label} is not an ISO timestamp") from error
    else:
        raise ValueError(f"{label} is not a timestamp")
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} is not a positive finite timestamp")
    return result


def _image_digest(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("runtime image ID is absent")
    match = re.fullmatch(
        r"(?:containerd|docker-pullable)://(?:[^@\s]+@)?sha256:([a-f0-9]{64})",
        value,
    )
    if match is None:
        raise ValueError("runtime image ID is not immutable")
    return match.group(1)


def _canary(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate scientific plan shape without reconstructing its old request.

    A submitted plan remains valid evidence after launch code evolves.  Its
    exact request and runtime bundle are checked through ``SUBMISSION_SCHEMA``;
    calling today's ``job_request`` here would incorrectly judge history using
    different source bytes.
    """

    args = plan.get("arguments", {})
    execution = plan.get("execution", {})
    profile = cluster_profile(plan_cluster_target(plan))
    image = execution.get("image") if isinstance(execution, dict) else None
    checkpoint = plan.get("checkpoint", {})
    long_horizon = plan.get("schema") == LONG_CONTEXT_SCHEMA
    nodes = 4 if long_horizon else 1
    gpus_per_node = 8
    if (
        plan.get("schema") not in {TRAINING_SCHEMA, LONG_CONTEXT_SCHEMA}
        or plan.get("run_name") != args.get("name")
        or plan.get("output_root") != args.get("output_root")
        or plan.get("model", {}).get("repo") != "Qwen/Qwen3.8-27B"
        or args.get("model") != "Qwen/Qwen3.8-27B"
        or args.get("nodes") != nodes
        or args.get("gpus_per_node") != gpus_per_node
        or args.get("steps") != 1
        or args.get("groups") != 1
        or args.get("samples_per_prompt") != WORLD_SIZE
        or args.get("eval_interval") != 1
        or args.get("checkpoint_interval") != 1
        or not isinstance(image, str)
        or re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image) is None
        or checkpoint.get("image") != image
        or execution.get("priority") != "c1"
        or profile.namespace != NAMESPACE
        or _SHA.fullmatch(str(plan.get("runtime_sha256"))) is None
        or _SHA.fullmatch(str(plan.get("native_driver_sha256"))) is None
        or (
            long_horizon
            and (
                args.get("harness") != "opencode"
                or args.get("native_profile") != "qwen3.8-27b-256k"
                or args.get("context_tokens") != 262_144
                or args.get("response_tokens") != 245_760
                or args.get("tokens_per_turn") != 32_768
                or args.get("max_tokens_per_gpu") != 65_536
                or args.get("session_node_cap") != 4_096
            )
        )
    ):
        raise ValueError("terminal acceptance requires the exact one-update Miles canary")
    files = plan.get("data", {}).get("files", {})
    if (
        set(files) != {"train", "dev"}
        or files["train"].get("rows") != 1
        or files["dev"].get("rows") != 1
    ):
        raise ValueError("reward canary requires one frozen train and dev task")
    return args


def _topology(plan: dict[str, Any]) -> tuple[int, int, int]:
    """Return the already-validated worker, per-worker GPU and rank counts."""

    args = _canary(plan)
    nodes = args["nodes"]
    gpus = args["gpus_per_node"]
    return nodes, gpus, nodes * gpus


def reconstruct_current_request(plan: dict[str, Any]) -> dict[str, Any]:
    """Render today's request for a future launch, never for historical proof."""

    _canary(plan)
    return job_request(plan)


def _transport_blob(request: Mapping[str, Any]) -> bytes:
    env = request.get("env")
    if not isinstance(env, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
    ):
        raise ValueError("submitted request environment is invalid")
    direct = env.get("CYBER_RUNTIME_BUNDLE")
    chunks = sorted(
        (
            (int(key.removeprefix("CYBER_RUNTIME_BUNDLE_")), value)
            for key, value in env.items()
            if re.fullmatch(r"CYBER_RUNTIME_BUNDLE_\d+", key)
        ),
        key=lambda item: item[0],
    )
    if direct is not None:
        if chunks:
            raise ValueError("submitted request has ambiguous runtime transport")
        encoded = direct
    else:
        if not chunks or [index for index, _ in chunks] != list(range(len(chunks))):
            raise ValueError("submitted runtime transport is absent or non-contiguous")
        encoded = "".join(value for _, value in chunks)
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("submitted runtime transport is not valid base64") from error


def _git_source(commit: str, path: str, repo_root: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        raise ValueError("submitted source commit or runtime file is unavailable")
    return result.stdout


def _submitted_request(
    plan: dict[str, Any], request: dict[str, Any], source_commit: str, repo_root: Path
) -> tuple[str, dict[str, Any]]:
    """Reopen the exact historical request without invoking today's compiler."""

    _canary(plan)
    if re.fullmatch(r"[a-f0-9]{40}", source_commit) is None:
        raise ValueError("submitted source commit is not a full Git SHA")
    expected_keys = {
        "name",
        "title",
        "run_dir",
        "image",
        "workers",
        "gpus_per_worker",
        "resources",
        "priority_class",
        "requeueIfPreempted",
        "secrets",
        "env",
        "command",
    }
    args = plan["arguments"]
    env = request.get("env")
    if not isinstance(env, dict):
        raise ValueError("submitted request environment is absent")
    transport_names = {
        key
        for key in env
        if key == "CYBER_RUNTIME_BUNDLE" or re.fullmatch(r"CYBER_RUNTIME_BUNDLE_\d+", key)
    }
    ordinary_env = {key: value for key, value in env.items() if key not in transport_names}
    expected_env = {**_SUBMITTED_ENV, "WANDB_RUN_ID": args["wandb_run_id"]}
    nodes, gpus_per_node, _ = _topology(plan)
    if (
        set(request) != expected_keys
        or request.get("name") != plan["run_name"]
        or request.get("title") != plan["run_name"] + " native Miles RL"
        or request.get("run_dir") != plan["output_root"]
        or request.get("image") != plan["execution"]["image"]
        or request.get("workers") != nodes
        or request.get("gpus_per_worker") != gpus_per_node
        or request.get("resources") != plan["execution"]["resources"]
        or request.get("priority_class") != "c1"
        or request.get("requeueIfPreempted") is not False
        or request.get("secrets") != ["fleet-api", "wandb-api"]
        or ordinary_env != expected_env
    ):
        raise ValueError("submitted request differs from the reviewed one-update contract")
    blob = _transport_blob(request)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    command = request.get("command")
    try:
        command_parts = shlex.split(command) if isinstance(command, str) else []
    except ValueError as error:
        raise ValueError("submitted request command is malformed") from error
    if (
        len(command_parts) != 3
        or command_parts[:2] != ["python", "-c"]
        or blob_sha256 not in command_parts[2]
    ):
        raise ValueError("submitted request command is not bound to its runtime bundle")
    try:
        payload = json.loads(gzip.decompress(blob))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("submitted runtime bundle is invalid") from error
    if not isinstance(payload, dict) or set(payload) != {"files", "module", "argv"}:
        raise ValueError("submitted runtime bundle fields changed")
    files = payload.get("files")
    required_runtime_files = set(_SUBMITTED_RUNTIME_FILES)
    if plan.get("schema") != LONG_CONTEXT_SCHEMA:
        required_runtime_files.discard("training/miles_opencode.py")
    if (
        not isinstance(files, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in files.items()
        )
        or payload.get("module") != "training.miles_training"
        or payload.get("argv") != ["--plan", "plan.json", "--sha256", digest(plan)]
        or files.get("plan.json") != json.dumps(plan, sort_keys=True, separators=(",", ":"))
        or not required_runtime_files.issubset(files)
    ):
        raise ValueError("submitted runtime bundle is not bound to the exact Miles plan")
    source_files = {path: files[path] for path in _SUBMITTED_RUNTIME_FILES}
    if digest(source_files) != plan["runtime_sha256"]:
        raise ValueError("submitted runtime files differ from the exact plan")
    for path, text in source_files.items():
        if _git_source(source_commit, path, repo_root) != text.encode():
            raise ValueError("submitted runtime bundle differs from its source commit")
    return blob_sha256, {
        "name": request["name"],
        "run_dir": request["run_dir"],
        "image": request["image"],
        "workers": request["workers"],
        "gpus_per_worker": request["gpus_per_worker"],
        "resources": request["resources"],
        "priority_class": request["priority_class"],
        "automatic_requeue": request["requeueIfPreempted"],
        "secret_names": request["secrets"],
        "environment_names": sorted(ordinary_env),
        "runtime_module": payload["module"],
        "runtime_argv": payload["argv"],
    }


def compile_submission_binding(
    *,
    plan_path: Path,
    request_path: Path,
    source_commit: str,
    api_run_id: str,
    submitted_at: object,
    output: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Seal exact request/source provenance from immutable pre-submit files."""

    plan, plan_file_sha256 = _json_snapshot(plan_path)
    request, request_file_sha256 = _json_snapshot(request_path)
    if not isinstance(plan, dict) or not isinstance(request, dict):
        raise ValueError("submitted plan and request must be JSON objects")
    bundle_sha256, projection = _submitted_request(
        plan,
        request,
        source_commit,
        repo_root or Path(__file__).resolve().parents[1],
    )
    run_id = _uuid(api_run_id, "API run ID")
    profile = cluster_profile(plan_cluster_target(plan))
    _time(submitted_at, "submission time")
    return _write(
        output,
        {
            "schema": SUBMISSION_SCHEMA,
            "source_commit": source_commit,
            "source_plan_path": str(plan_path),
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_plan_file_sha256": "sha256:" + plan_file_sha256,
            "source_request_path": str(request_path),
            "source_request_sha256": "sha256:" + digest(request),
            "source_request_file_sha256": "sha256:" + request_file_sha256,
            "runtime_bundle_sha256": "sha256:" + bundle_sha256,
            "runtime_source_sha256": "sha256:" + plan["runtime_sha256"],
            "runtime_source_commit_match": True,
            "api": {
                "base_url": profile.api_base_url,
                "run_id": run_id,
                "run_name": plan["run_name"] + "-" + run_id[:8],
            },
            "request": projection,
            "jobs_api_post_count": 1,
            "submitted_at": submitted_at,
            "secret_values_included": False,
            "task_content_included": False,
            "private_payload_included": False,
        },
    )


def validate_submission_binding(
    value: dict[str, Any], plan: dict[str, Any], *, check_files: bool = True
) -> dict[str, Any]:
    """Validate and optionally reopen the exact historical request/bundle."""

    _canary(plan)
    sealed(value, SUBMISSION_SCHEMA)
    projection = value.get("request")
    api = value.get("api")
    profile = cluster_profile(plan_cluster_target(plan))
    nodes, gpus_per_node, _ = _topology(plan)
    if (
        set(value) != _SUBMISSION_FIELDS
        or re.fullmatch(r"[a-f0-9]{40}", str(value.get("source_commit"))) is None
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("runtime_source_sha256", "").removeprefix("sha256:")
        != plan.get("runtime_sha256")
        or value.get("runtime_source_commit_match") is not True
        or not isinstance(projection, dict)
        or set(projection) != _SUBMISSION_REQUEST_FIELDS
        or projection.get("name") != plan["run_name"]
        or projection.get("run_dir") != plan["output_root"]
        or projection.get("image") != plan["execution"]["image"]
        or projection.get("workers") != nodes
        or projection.get("gpus_per_worker") != gpus_per_node
        or projection.get("resources") != plan["execution"]["resources"]
        or projection.get("priority_class") != "c1"
        or projection.get("automatic_requeue") is not False
        or projection.get("secret_names") != ["fleet-api", "wandb-api"]
        or projection.get("environment_names") != sorted({*_SUBMITTED_ENV, "WANDB_RUN_ID"})
        or projection.get("runtime_module") != "training.miles_training"
        or projection.get("runtime_argv") != ["--plan", "plan.json", "--sha256", digest(plan)]
        or not isinstance(api, dict)
        or set(api) != {"base_url", "run_id", "run_name"}
        or api.get("base_url") != profile.api_base_url
        or _UUID.fullmatch(str(api.get("run_id"))) is None
        or api.get("run_name") != plan["run_name"] + "-" + str(api.get("run_id"))[:8]
        or value.get("jobs_api_post_count") != 1
        or value.get("secret_values_included") is not False
        or value.get("task_content_included") is not False
        or value.get("private_payload_included") is not False
        or not isinstance(value.get("source_plan_path"), str)
        or not value["source_plan_path"]
        or not isinstance(value.get("source_request_path"), str)
        or not value["source_request_path"]
        or _SHA.fullmatch(str(value.get("source_plan_file_sha256"))) is None
        or _SHA.fullmatch(str(value.get("source_request_sha256"))) is None
        or _SHA.fullmatch(str(value.get("source_request_file_sha256"))) is None
        or _SHA.fullmatch(str(value.get("runtime_bundle_sha256"))) is None
    ):
        raise ValueError("submitted execution binding is incomplete or mismatched")
    _time(value.get("submitted_at"), "submission time")
    if check_files:
        reopened_plan, plan_file_sha256 = _json_snapshot(Path(value["source_plan_path"]))
        reopened_request, request_file_sha256 = _json_snapshot(Path(value["source_request_path"]))
        if (
            reopened_plan != plan
            or value["source_plan_file_sha256"].removeprefix("sha256:") != plan_file_sha256
            or not isinstance(reopened_request, dict)
            or value["source_request_sha256"].removeprefix("sha256:") != digest(reopened_request)
            or value["source_request_file_sha256"].removeprefix("sha256:") != request_file_sha256
        ):
            raise ValueError("submitted plan or request file changed")
        bundle_sha256, reopened_projection = _submitted_request(
            plan,
            reopened_request,
            value["source_commit"],
            Path(__file__).resolve().parents[1],
        )
        if (
            value["runtime_bundle_sha256"].removeprefix("sha256:") != bundle_sha256
            or reopened_projection != projection
        ):
            raise ValueError("submitted request bundle changed")
    return {"request_sha256": value["source_request_sha256"], "api": api}


def _source_configs(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    policy_identity_root = plan["arguments"].get("policy_identity_root") or plan[
        "arguments"
    ]["model_root"]
    manifest_path = Path(plan["arguments"]["data_manifest"])
    data_schema = (
        "cyber_miles_data_v2"
        if plan.get("schema") == LONG_CONTEXT_SCHEMA
        else "cyber_miles_data_v1"
    )
    manifest, _ = _read(manifest_path, schema=data_schema)
    if manifest != plan["data"]:
        raise ValueError("staged Miles data manifest differs from the plan")
    for split in ("train", "dev"):
        item = manifest["files"][split]
        path = manifest_path.parent / item["path"]
        values, file_sha256 = _jsonl_snapshot(path)
        if path != Path(plan["arguments"][split + "_data"]) or file_sha256 != str(
            item["sha256"]
        ).removeprefix("sha256:"):
            raise ValueError("staged Miles task row differs from the plan")
        if len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError("reward canary task row cardinality changed")
        row = values[0]
        metadata = row.get("metadata")
        config = metadata.get("cyber_config") if isinstance(metadata, dict) else None
        if (
            not isinstance(config, dict)
            or row.get("input") is None
            or metadata.get("split") != split
            or config.get("run_id") != plan["run_name"]
            or config.get("model", {}).get("repo") != plan["model"]["repo"]
            or config.get("model", {}).get("revision") != plan["model"]["revision"]
            or config.get("model", {}).get("root") != policy_identity_root
            or config.get("initial_prompt_sha256") != fleet.sha256(row["input"].encode())
            or config.get("rl")
            != {key: value for key, value in manifest["limits"].items() if key != "response_tokens"}
            or config.get("execution", {}).get("required_task_tool_catalog_sha256")
            != manifest["tool_catalog_sha256"]
        ):
            raise ValueError("reward canary task binding changed")
        validate_episode_config(config)
        result[split] = config
    return result


def _jsonl_snapshot(path: Path) -> tuple[list[Any], str]:
    """Read one JSONL file once without treating access-time changes as mutation."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("acceptance input must be a regular file")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if any(getattr(before, key) != getattr(after, key) for key in _STABLE_STAT_FIELDS):
        raise ValueError("acceptance input changed while reading")
    try:
        rows = [json.loads(line) for line in payload.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("acceptance JSONL input is invalid") from error
    return rows, fleet.sha256(payload).removeprefix("sha256:")


def _dynamic_config(binding: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(binding)
    result.pop("native_batch")
    result.pop("sampling")
    result["run_id"] = (
        result["run_id"]
        .split("-dev-baseline-r0-s", 1)[0]
        .split("-dev-after-r0-s", 1)[0]
        .split("-train-r0-s", 1)[0]
    )
    result["config_sha256"] = fleet.digest_without(result, "config_sha256")
    return result


def _reward(config: dict[str, Any], value: object) -> tuple[float, str, str]:
    if not isinstance(value, dict) or set(value) != {
        "task_key",
        "task_version_id",
        "instance_id",
        "reward",
        "verifier_execution_id",
        "direct_authority_attestation",
    }:
        raise ValueError("episode reward is not a direct-authority receipt")
    score = value.get("reward")
    execution_id = _uuid(value.get("verifier_execution_id"), "verifier execution ID")
    if (
        isinstance(score, bool)
        or not isinstance(score, numbers.Real)
        or not math.isfinite(score)
        or not 0 <= score <= 1
        or value.get("task_key") != config["task"]["key"]
        or value.get("task_version_id") != config["task"]["version_id"]
    ):
        raise ValueError("authoritative reward identity or value changed")
    attestation = value["direct_authority_attestation"]
    context = attestation.get("context") if isinstance(attestation, dict) else None
    activity = attestation.get("activity") if isinstance(attestation, dict) else None
    shadow = attestation.get("shadow") if isinstance(attestation, dict) else None
    direct = shadow.get("direct_verifier") if isinstance(shadow, dict) else None
    evidence_id = _uuid(context.get("evidence_run_id"), "evidence run ID") if context else None
    if (
        attestation.get("schema_version") != fleet.DIRECT_AUTHORITY_ATTESTATION_SCHEMA
        or context
        != {
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": value["instance_id"],
            "evidence_run_id": evidence_id,
            "verifier_version_id": config["verifier"]["version_id"],
            "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
        }
        or activity
        != {
            "result_schema_version": "cyber_verification_result_v3",
            "reward": float(score),
            "task_version_id": config["task"]["version_id"],
            "verifier_execution_id": execution_id,
        }
        or shadow
        != {
            "mode": "authoritative",
            "status": "authoritative",
            "match": True,
            "production_execution_id": execution_id,
            "direct_verifier": direct,
        }
        or direct
        != {
            "status": "authoritative",
            "match": True,
            "execution_id": execution_id,
            "verifier_contract_version": config["authority"]["required_cyber_contract"][
                "verifier_contract"
            ],
            "context_schema_version": "cyber_verification_context_v1",
        }
        or attestation.get("data_minimization")
        != {
            "components_included": False,
            "diagnostics_included": False,
            "evidence_payloads_included": False,
            "prompts_included": False,
            "traces_included": False,
            "flags_included": False,
        }
    ):
        raise ValueError("authoritative verifier attestation changed")
    return float(score), execution_id, evidence_id


def _episode(path: Path, source: dict[str, Any], kind: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_dir()
        or {item.name for item in path.iterdir()} != (_EPISODE_FILES | {"ACCEPTED.json"})
    ):
        raise ValueError("Miles episode package is incomplete or ambiguous")
    accepted, _ = _read(path / "ACCEPTED.json")
    if (
        set(accepted)
        != {
            "task_version_id",
            "instance_id",
            "verifier_execution_id",
            "done_reason",
            "config_sha256",
            "sample_count",
            "files",
            "sha256",
        }
        or set(accepted.get("files", {})) != _EPISODE_FILES
    ):
        raise ValueError("Miles episode acceptance fields changed")
    values = {}
    for name in _EPISODE_FILES:
        file_path = path / name
        item, item_sha256 = _json_snapshot(file_path)
        if "sha256:" + item_sha256 != accepted["files"][name]:
            raise ValueError("Miles episode payload differs from its acceptance receipt")
        values[name] = item
    binding = values["binding.json"]
    validate_episode_config(binding)
    if (
        path.name != binding.get("run_id")
        or accepted.get("config_sha256") != binding.get("config_sha256")
        or binding.get("config_sha256") != fleet.digest_without(binding, "config_sha256")
        or _dynamic_config(binding) != source
        or binding.get("native_batch") != {"kind": kind, "rollout_id": 0}
        or not path.name.startswith(source["run_id"] + f"-{kind}-r0-s")
    ):
        raise ValueError("Miles episode/source/batch identity changed")
    instance = values["instance.json"]
    cleanup = values["cleanup.json"]
    instance_id = accepted.get("instance_id")
    if (
        not isinstance(instance_id, str)
        or not instance_id
        or set(instance) != {"instance_id", "evidence_run_id"}
        or instance.get("instance_id") != instance_id
        or cleanup
        != {
            "create_attempted": True,
            "instance_created": True,
            "instance_id": instance_id,
            "instance_closed": True,
            "possible_instance_leak": False,
        }
    ):
        raise ValueError("Miles episode environment release is incomplete")
    score, execution_id, evidence_id = _reward(binding, values["reward.json"])
    if (
        instance["evidence_run_id"] != evidence_id
        or accepted.get("task_version_id") != binding["task"]["version_id"]
        or accepted.get("verifier_execution_id") != execution_id
        or values["create-intent.json"] != {"run_id": binding["run_id"]}
        or values["score-intent.json"]
        != {
            "instance_id": instance_id,
            "scoring_mode": binding["authority"]["scoring_mode"],
            "multi_app_aggregation_mode": binding["authority"]["multi_app_aggregation_mode"],
        }
    ):
        raise ValueError("Miles episode scoring identity changed")
    conversation = values["conversation.json"].get("messages")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("Miles episode has no recorded task interaction")
    assistant = sum(
        isinstance(message, dict) and message.get("role") == "assistant" for message in conversation
    )
    tools = sum(
        isinstance(message, dict) and message.get("role") == "tool" for message in conversation
    )
    if (
        assistant < 1
        or tools < 1
        or (score > 0 and accepted.get("done_reason") != "report_submitted")
    ):
        raise ValueError("Miles episode lacks genuine task interaction or terminal report")
    samples = values["recording.json"].get("samples")
    if (
        type(accepted.get("sample_count")) is not int
        or accepted["sample_count"] < 1
        or not isinstance(samples, list)
        or len(samples) != accepted["sample_count"]
    ):
        raise ValueError("Miles episode token recording is incomplete")
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {
            "tokens",
            "response_length",
            "loss_mask",
            "rollout_log_probs",
        }:
            raise ValueError("Miles episode token recording fields changed")
        tokens, response, mask, probabilities = (
            sample["tokens"],
            sample["response_length"],
            sample["loss_mask"],
            sample["rollout_log_probs"],
        )
        if (
            not isinstance(tokens, list)
            or any(type(value) is not int or value < 0 for value in tokens)
            or type(response) is not int
            or not 0 < response < len(tokens)
            or not isinstance(mask, list)
            or not isinstance(probabilities, list)
            or len(mask) != response
            or len(probabilities) != response
            or not any(value == 1 for value in mask)
            or any(type(value) is not int or value not in (0, 1) for value in mask)
            or any(
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
                or value > 1e-6
                for value in probabilities
            )
            or any(
                probability != 0
                for probability, enabled in zip(probabilities, mask, strict=True)
                if not enabled
            )
        ):
            raise ValueError("Miles episode token recording is invalid")
    verifier = binding["verifier"]
    if (
        set(verifier)
        not in (
            {"id", "version_id", "version", "sha256", "function_name"},
            {"id", "version_id", "version", "sha256", "code_sha256", "function_name"},
        )
        or _UUID.fullmatch(str(verifier.get("id"))) is None
        or _UUID.fullmatch(str(verifier.get("version_id"))) is None
        or _SHA.fullmatch(str(verifier.get("sha256"))) is None
        or verifier.get("function_name") != "verify"
    ):
        raise ValueError("Miles episode verifier identity is incomplete")
    return {
        "kind": kind,
        "sample_index": int(path.name.rsplit("-s", 1)[1]),
        "score": score,
        "execution_id": execution_id,
        "instance_id": instance_id,
        "evidence_id": evidence_id,
        "assistant_messages": assistant,
        "tool_messages": tools,
        "report_submitted": accepted.get("done_reason") == "report_submitted",
        "verifier": {
            "task_key": binding["task"]["key"],
            "task_version_id": binding["task"]["version_id"],
            "verifier_id": verifier["id"],
            "verifier_version_id": verifier["version_id"],
            "verifier_sha256": verifier["sha256"],
        },
    }


def episode_audit(plan: dict[str, Any]) -> dict[str, Any]:
    """Reopen all ten private episodes and return value-free scientific facts."""
    _canary(plan)
    sources = _source_configs(plan)
    root = Path(plan["output_root"]) / "episodes"
    batch_root = root / "batches"
    if root.is_symlink() or not root.is_dir() or batch_root.is_symlink() or not batch_root.is_dir():
        raise ValueError("Miles episode root is missing or indirect")
    expected_batches = {
        "dev-baseline-r0": ("dev-baseline", [0]),
        "train-r0": ("train", list(range(WORLD_SIZE))),
        "dev-after-r0": ("dev-after", [0]),
    }
    if {path.name for path in batch_root.iterdir()} != set(expected_batches):
        raise ValueError("Miles canary batch set changed")
    for name, (kind, indices) in expected_batches.items():
        directory = batch_root / name
        if directory.is_symlink() or {path.name for path in directory.iterdir()} != {
            "STARTED.json",
            "COLLECTED.json",
        }:
            raise ValueError("Miles batch evidence is incomplete or ambiguous")
        started = json.loads((directory / "STARTED.json").read_bytes())
        collected, _ = _read(directory / "COLLECTED.json", schema="cyber_miles_batch_v1")
        if (
            _unsigned(collected) != started
            or started.get("batch_id") != name
            or started.get("data_sha256") != plan["data"]["sha256"]
            or started.get("episode_indices") != indices
            or started.get("native_rollout_id") != 0
            or started.get("evaluation") is not (kind != "train")
            or started.get("optimizer_step_verified") is not False
        ):
            raise ValueError("Miles batch evidence differs from the immutable plan")
    episode_paths = [path for path in root.iterdir() if path.name != "batches"]
    if len(episode_paths) != 10:
        raise ValueError("Miles reward canary requires exactly ten episode packages")
    rows = []
    for path in episode_paths:
        matches = [
            kind for kind in ("dev-baseline", "train", "dev-after") if f"-{kind}-r0-s" in path.name
        ]
        if len(matches) != 1:
            raise ValueError("Miles episode name does not identify one batch")
        kind = matches[0]
        rows.append(_episode(path, sources["train" if kind == "train" else "dev"], kind))
    by_kind = {
        kind: sorted(row["sample_index"] for row in rows if row["kind"] == kind)
        for kind in ("dev-baseline", "train", "dev-after")
    }
    if by_kind != {
        "dev-baseline": [0],
        "train": list(range(WORLD_SIZE)),
        "dev-after": [0],
    }:
        raise ValueError("Miles episode sample identities changed")
    execution_ids = [row["execution_id"] for row in rows]
    instance_ids = [row["instance_id"] for row in rows]
    evidence_ids = [row["evidence_id"] for row in rows]
    train = [row for row in rows if row["kind"] == "train"]
    scores = [row["score"] for row in train]
    mean = math.fsum(scores) / len(scores)
    variance = math.fsum((score - mean) ** 2 for score in scores) / len(scores)
    if (
        len(set(execution_ids)) != 10
        or len(set(instance_ids)) != 10
        or len(set(evidence_ids)) != 10
        or set(evidence_ids) & set(execution_ids)
        or not any(score > 0 for score in scores)
        or variance <= 0
    ):
        raise ValueError("Miles canary lacks unique authoritative reward variance")
    verifier_bindings = sorted(
        {json.dumps(row["verifier"], sort_keys=True, separators=(",", ":")) for row in rows}
    )
    return {
        "schema": EPISODE_AUDIT_SCHEMA,
        "source_plan_sha256": "sha256:" + digest(plan),
        "accepted_episode_count": 10,
        "train_episode_count": WORLD_SIZE,
        "dev_episode_count": 2,
        "infrastructure_invalid_count": 0,
        "truncated_count": 0,
        "unique_authoritative_verifier_execution_count": 10,
        "authoritative_verifier_execution_ids_sha256": "sha256:" + digest(sorted(execution_ids)),
        "authoritative_verifier_bindings_sha256": "sha256:" + digest(verifier_bindings),
        "unique_challenge_instance_count": 10,
        "challenge_instance_ids_sha256": "sha256:" + digest(sorted(instance_ids)),
        "unique_evidence_run_count": 10,
        "evidence_run_ids_sha256": "sha256:" + digest(sorted(evidence_ids)),
        "train_nonzero_reward_present": True,
        "train_within_group_reward_variance_present": True,
        "reward_values_included": False,
        "assistant_message_count": sum(row["assistant_messages"] for row in rows),
        "tool_message_count": sum(row["tool_messages"] for row in rows),
        "report_submitted_count": sum(row["report_submitted"] for row in rows),
        "task_content_included": False,
    }


def validate_wandb_observation(
    value: dict[str, Any], plan: dict[str, Any], submission: dict[str, Any]
) -> None:
    sealed(value, WANDB_SCHEMA)
    submitted = validate_submission_binding(submission, plan, check_files=False)
    args = plan["arguments"]
    expected_keys = {
        "schema",
        "source_plan_sha256",
        "source_request_sha256",
        "identity",
        "state",
        "history_rows",
        "remote_non_sensitive_metric_schema",
        "remote_metric_schema_sha256",
        "observed_train_steps",
        "optimizer_update_count",
        "update_zero_metric_names",
        "positive_finite_gradient_observed",
        "finite_train_loss_observed",
        "learning_rate_matches_plan",
        "sensitive_metric_values_redacted",
        "logged_artifact_count",
        "rich_payload_count",
        "reward_values_included",
        "sha256",
    }
    identity = value.get("identity")
    metrics = value.get("update_zero_metric_names")
    remote_schema = value.get("remote_non_sensitive_metric_schema")
    if (
        set(value) != expected_keys
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("source_request_sha256") != submitted["request_sha256"]
        or identity
        != {
            "entity": args["wandb_entity"],
            "project": args["wandb_project"],
            "run_id": args["wandb_run_id"],
            "name": args["name"],
        }
        or value.get("state") != "finished"
        or type(value.get("history_rows")) is not int
        or not 1 <= value["history_rows"] <= 10000
        or not isinstance(remote_schema, list)
        or len(remote_schema) != value["history_rows"]
        or any(
            not isinstance(row, list)
            or row != sorted(set(row))
            or any(
                not isinstance(name, str) or name.startswith("_") or _sensitive_wandb_metric(name)
                for name in row
            )
            for row in remote_schema
        )
        or _SHA.fullmatch(str(value.get("remote_metric_schema_sha256"))) is None
        or value["remote_metric_schema_sha256"].removeprefix("sha256:") != digest(remote_schema)
        or value.get("observed_train_steps") != [0]
        or value.get("optimizer_update_count") != 1
        or not isinstance(metrics, list)
        or metrics != sorted(_MILES_UPDATE_METRICS)
        or value.get("positive_finite_gradient_observed") is not True
        or value.get("finite_train_loss_observed") is not True
        or value.get("learning_rate_matches_plan") is not True
        or value.get("sensitive_metric_values_redacted") is not True
        or value.get("logged_artifact_count") != 0
        or value.get("rich_payload_count") != 0
        or value.get("reward_values_included") is not False
    ):
        raise ValueError("W&B observation does not prove the exact value-free Miles update")


def observe_wandb(
    plan: dict[str, Any], submission: dict[str, Any], *, api: Any = None
) -> dict[str, Any]:
    """Read one exact W&B run and return a reward-value-free observation.

    The public receipt records only identity, metric names/presence, and boolean
    predicates computed from the private scalar stream.  It never copies or
    hashes scalar values: even a digest of reward history is a low-entropy
    reward commitment and is therefore not safe public evidence.
    """
    from .skyrl_training import _finite_scalar_rows, _wandb_has_rich_payloads

    _canary(plan)
    submitted = validate_submission_binding(submission, plan, check_files=False)
    args = plan["arguments"]
    if api is None:
        import wandb

        api = wandb.Api()
    run = api.run(f"{args['wandb_entity']}/{args['wandb_project']}/{args['wandb_run_id']}")
    if (
        (run.entity, run.project, run.id, run.name)
        != (
            args["wandb_entity"],
            args["wandb_project"],
            args["wandb_run_id"],
            args["name"],
        )
        or run.state != "finished"
        or _wandb_has_rich_payloads(run)
    ):
        raise ValueError("W&B run identity, state, or scalar-only surface changed")
    rows = _finite_scalar_rows(run.scan_history(), label="W&B")
    if len(rows) > 10_000:
        raise ValueError("W&B scalar history exceeds its reviewed bound")
    by_train_step: dict[int, dict[str, float]] = {}
    public_schema_rows: list[list[str]] = []
    for row in rows:
        names = sorted(
            key for key in row if not key.startswith("_") and not _sensitive_wandb_metric(key)
        )
        public_schema_rows.append(names)
        raw_step = row.get("train/step")
        if raw_step is None:
            continue
        if not float(raw_step).is_integer() or raw_step < 0:
            raise ValueError("W&B train/step is not a nonnegative integer")
        step = int(raw_step)
        target = by_train_step.setdefault(step, {})
        for key, item in row.items():
            if key.startswith("_") or _sensitive_wandb_metric(key):
                continue
            if key in target and target[key] != item:
                raise ValueError("W&B rewrites a metric within one train step")
            target[key] = item
    if set(by_train_step) != {0}:
        raise ValueError("W&B does not prove exactly one optimizer update")
    update = by_train_step[0]
    if (
        not set(update) >= _MILES_UPDATE_METRICS
        or update["train/step"] != 0.0
        or not math.isfinite(update["train/grad_norm"])
        or update["train/grad_norm"] <= 0
        or not math.isfinite(update["train/loss"])
        or update["train/lr-pg_0"] != args["lr"]
    ):
        raise ValueError("W&B update zero lacks finite native Miles telemetry")
    value = {
        "schema": WANDB_SCHEMA,
        "source_plan_sha256": "sha256:" + digest(plan),
        "source_request_sha256": submitted["request_sha256"],
        "identity": {
            "entity": run.entity,
            "project": run.project,
            "run_id": run.id,
            "name": run.name,
        },
        "state": run.state,
        "history_rows": len(rows),
        # Names and their per-row presence are safe to bind; values are not.
        "remote_non_sensitive_metric_schema": public_schema_rows,
        "remote_metric_schema_sha256": "sha256:" + digest(public_schema_rows),
        "observed_train_steps": [0],
        "optimizer_update_count": 1,
        "update_zero_metric_names": sorted(_MILES_UPDATE_METRICS),
        "positive_finite_gradient_observed": True,
        "finite_train_loss_observed": True,
        "learning_rate_matches_plan": True,
        "sensitive_metric_values_redacted": True,
        "logged_artifact_count": 0,
        "rich_payload_count": 0,
        "reward_values_included": False,
    }
    value["sha256"] = "sha256:" + digest(value)
    validate_wandb_observation(value, plan, submission)
    return value


def validate_policy_delta_observation(
    value: dict[str, Any], plan: dict[str, Any], checkpoint: dict[str, Any]
) -> int:
    """Validate a value-level policy comparison that excludes optimizer state.

    Distributed-checkpoint files mix model tensors with optimizer, scheduler,
    RNG, and metadata payloads.  A new filename or file digest therefore does
    not prove a policy update.  After the run, the independent all-rank
    observer reopens the immutable sealed checkpoint: it hashes named policy
    tensors before/after the update and seals the trained optimizer, scheduler,
    and RNG commitments used by the later independent reload.  The training
    process itself need not have emitted these observations.
    """

    from .miles_policy_observer import validate_policy_evidence

    return validate_policy_evidence(value, plan, checkpoint)


def validate_controller_observation(
    value: dict[str, Any], plan: dict[str, Any], submission: dict[str, Any]
) -> None:
    sealed(value, CONTROLLER_SCHEMA)
    _canary(plan)
    submitted = validate_submission_binding(submission, plan, check_files=False)
    api = value.get("api")
    kube = value.get("kubernetes")
    execution = value.get("execution")
    pods = kube.get("pods") if isinstance(kube, dict) else None
    rayjob = kube.get("rayjob") if isinstance(kube, dict) else None
    workload = kube.get("workload") if isinstance(kube, dict) else None
    raycluster = kube.get("raycluster") if isinstance(kube, dict) else None
    api_name = api.get("run_name") if isinstance(api, dict) else None
    target = plan_cluster_target(plan)
    profile = cluster_profile(target)
    expected_image = plan["execution"]["image"].rsplit("@sha256:", 1)[1]
    nodes, gpus_per_node, world_size = _topology(plan)
    if (
        set(value)
        != {
            "schema",
            "status",
            "source_plan_sha256",
            "source_request_sha256",
            "api",
            "kubernetes",
            "execution",
            "observed_at",
            "sha256",
        }
        or value.get("status") != "succeeded"
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("source_request_sha256") != submitted["request_sha256"]
        or not isinstance(api, dict)
        or set(api) != {"base_url", "run_id", "run_name", "status"}
        or api.get("base_url") != submitted["api"]["base_url"]
        or _UUID.fullmatch(str(api.get("run_id"))) is None
        or not isinstance(api_name, str)
        or api.get("run_id") != submitted["api"]["run_id"]
        or api_name != submitted["api"]["run_name"]
        or api.get("status") != "SUCCEEDED"
        or not isinstance(kube, dict)
        or set(kube)
        != {"cluster", "namespace", "namespace_uid", "rayjob", "workload", "raycluster", "pods"}
        or kube.get("cluster") != target
        or kube.get("namespace") != profile.namespace
        or kube.get("namespace_uid") != profile.namespace_uid
        or not isinstance(rayjob, dict)
        or set(rayjob) != {"name", "uid", "status"}
        or rayjob.get("name") != api_name
        or _uuid(rayjob.get("uid"), "RayJob UID") != rayjob.get("uid")
        or rayjob.get("status") != "SUCCEEDED"
        or not isinstance(workload, dict)
        or set(workload) != {"name", "uid", "owner_rayjob_uid"}
        or _uuid(workload.get("uid"), "Workload UID") != workload.get("uid")
        or workload.get("owner_rayjob_uid") != rayjob.get("uid")
        or not isinstance(workload.get("name"), str)
        or not isinstance(raycluster, dict)
        or set(raycluster) != {"name", "uid", "owner_rayjob_uid"}
        or _uuid(raycluster.get("uid"), "RayCluster UID") != raycluster.get("uid")
        or raycluster.get("owner_rayjob_uid") != rayjob.get("uid")
        or not isinstance(raycluster.get("name"), str)
        or not isinstance(pods, list)
        or len(pods) != nodes
        or not isinstance(execution, dict)
        or execution
        != {
            "requested_image": plan["execution"]["image"],
            "priority_class": "c1",
            "effective_priority": 10000,
            "automatic_requeue": False,
            "workers": nodes,
            "gpus_per_worker": gpus_per_node,
            "total_gpus": world_size,
        }
    ):
        raise ValueError("controller observation differs from the exact canary cluster")
    pod_fields = {
        "name",
        "uid",
        "owner_raycluster_uid",
        "phase",
        "exit_code",
        "termination_reason",
        "runtime_image_id",
        "container_restarts",
        "gpus",
    }
    pod_uids: set[str] = set()
    for pod in pods:
        uid = pod.get("uid") if isinstance(pod, dict) else None
        if (
            not isinstance(pod, dict)
            or set(pod) != pod_fields
            or _uuid(uid, "Pod UID") != uid
            or uid in pod_uids
            or pod.get("owner_raycluster_uid") != raycluster["uid"]
            or pod.get("phase") != "Succeeded"
            or pod.get("exit_code") != 0
            or pod.get("termination_reason") != "Completed"
            or _image_digest(pod.get("runtime_image_id")) != expected_image
            or pod.get("container_restarts") != 0
            or pod.get("gpus") != gpus_per_node
        ):
            raise ValueError("controller Pod observation is incomplete or mismatched")
        pod_uids.add(uid)
    _time(value.get("observed_at"), "controller observation")


def validate_release_observation(
    value: dict[str, Any],
    plan: dict[str, Any],
    controller: dict[str, Any],
    controller_file_sha256: str,
    submission: dict[str, Any],
    *,
    not_before: float,
) -> None:
    sealed(value, RELEASE_SCHEMA)
    submitted = validate_submission_binding(submission, plan, check_files=False)
    api = controller["api"]
    kube = controller["kubernetes"]
    identities = value.get("identities")
    if (
        set(value)
        != {
            "schema",
            "status",
            "source_plan_sha256",
            "source_request_sha256",
            "controller_observation_sha256",
            "controller_observation_file_sha256",
            "api_status",
            "controller_status",
            "identities",
            "raycluster_present",
            "rayjob_present",
            "workload_present",
            "quota_reservation_present",
            "gpu_pods_present",
            "active_gpus",
            "gpu_release_proven",
            "observed_at",
            "sha256",
        }
        or value.get("status") != "released"
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("source_request_sha256") != submitted["request_sha256"]
        or value.get("controller_observation_sha256", "").removeprefix("sha256:")
        != controller["sha256"].removeprefix("sha256:")
        or value.get("controller_observation_file_sha256", "").removeprefix("sha256:")
        != controller_file_sha256.removeprefix("sha256:")
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or identities
        != {
            "api_run_id": api["run_id"],
            "api_run_name": api["run_name"],
            "rayjob_uid": kube["rayjob"]["uid"],
            "workload_uid": kube["workload"]["uid"],
            "raycluster_uid": kube["raycluster"]["uid"],
            "pod_uids": sorted(pod["uid"] for pod in kube["pods"]),
        }
        or value.get("raycluster_present") is not False
        or value.get("rayjob_present") is not False
        or value.get("workload_present") is not False
        or value.get("quota_reservation_present") is not False
        or value.get("gpu_pods_present") is not False
        or value.get("active_gpus") != 0
        or value.get("gpu_release_proven") is not True
        or _time(value.get("observed_at"), "release observation")
        < max(not_before, _time(controller["observed_at"], "controller observation"))
    ):
        raise ValueError("external release observation is incomplete or mismatched")


def _checkpoint(plan: dict[str, Any], path: Path) -> tuple[dict[str, Any], str]:
    manifest, file_sha256 = _read(path, schema=CHECKPOINT_SCHEMA)
    _verify_checkpoint(manifest, hashes=True)
    source = manifest.get("source", {})
    completion, _ = _read(Path(plan["output_root"]) / "NATIVE_TRAINING_COMPLETE.json")
    nodes, gpus_per_node, world_size = _topology(plan)
    if (
        manifest.get("rollout_index") != 0
        or manifest.get("next_rollout_id") != 1
        or manifest.get("world_size") != world_size
        or manifest.get("topology") != {"nodes": nodes, "gpus_per_node": gpus_per_node}
        or manifest.get("model") != plan["model"]
        or source.get("run_name") != plan["run_name"]
        or source.get("output_root") != plan["output_root"]
        or source.get("plan_sha256") != digest(plan)
        or source.get("completion_sha256") != completion["sha256"].removeprefix("sha256:")
        or source.get("arguments") != plan["arguments"]
        or source.get("execution") != plan["execution"]
        or source.get("native_driver_sha256") != plan["native_driver_sha256"]
        or manifest.get("source_optimizer_update_claimed") is not False
        or manifest.get("gpu_reload_verified") is not False
        or manifest.get("optimizer_update_during_reload") is not False
    ):
        raise ValueError("Miles checkpoint seal differs from the terminal source")
    return manifest, file_sha256


def _compile(
    plan: dict[str, Any],
    *,
    submission_binding_path: Path,
    checkpoint_manifest_path: Path,
    policy_delta_observation_path: Path,
    controller_observation_path: Path,
    release_observation_path: Path,
    wandb_observation_path: Path,
) -> dict[str, Any]:
    _canary(plan)
    submission, submission_file_sha256 = _read(submission_binding_path, schema=SUBMISSION_SCHEMA)
    submitted = validate_submission_binding(submission, plan, check_files=True)
    root = Path(plan["output_root"])
    completion_path = root / "NATIVE_TRAINING_COMPLETE.json"
    completion, completion_file_sha256 = _read(completion_path)
    if (
        set(completion)
        != {
            "status",
            "plan_sha256",
            "checkpoint_rollout_index",
            "completed_batches",
            "completed_at",
            "optimizer_update_independently_verified",
            "checkpoint_reload_verified",
            "sha256",
        }
        or completion.get("status") != "native_loop_returned"
        or completion.get("plan_sha256") != digest(plan)
        or completion.get("checkpoint_rollout_index") != 0
        or completion.get("completed_batches") != 3
        or completion.get("optimizer_update_independently_verified") is not False
        or completion.get("checkpoint_reload_verified") is not False
        or any(
            (root / name).exists()
            for name in (
                "FAILED.json",
                "REJECTED.json",
                "NATIVE_FAILURE.json",
                "NATIVE_REJECTED.json",
                "ACCEPTED.json",
            )
        )
    ):
        raise ValueError("Miles native completion is conflicting or incomplete")
    completed_at = _time(completion.get("completed_at"), "native completion")
    episodes = episode_audit(plan)
    checkpoint, checkpoint_file_sha256 = _checkpoint(plan, checkpoint_manifest_path)
    policy_delta, policy_delta_file_sha256 = _read(
        policy_delta_observation_path, schema=POLICY_DELTA_SCHEMA
    )
    changed_policy_ranks = validate_policy_delta_observation(policy_delta, plan, checkpoint)
    wandb, wandb_file_sha256 = _read(wandb_observation_path, schema=WANDB_SCHEMA)
    validate_wandb_observation(wandb, plan, submission)
    controller, controller_file_sha256 = _read(
        controller_observation_path, schema=CONTROLLER_SCHEMA
    )
    validate_controller_observation(controller, plan, submission)
    release, release_file_sha256 = _read(release_observation_path, schema=RELEASE_SCHEMA)
    validate_release_observation(
        release,
        plan,
        controller,
        controller_file_sha256,
        submission,
        not_before=completed_at,
    )
    return {
        "schema": TERMINAL_SCHEMA,
        "status": "accepted",
        "source_run_name": plan["run_name"],
        "source_plan_sha256": "sha256:" + digest(plan),
        "source_request_sha256": submitted["request_sha256"],
        "submission_binding": _reference(
            submission_binding_path, submission, submission_file_sha256
        ),
        "native_completion": _reference(completion_path, completion, completion_file_sha256),
        "episode_audit": episodes,
        "optimizer_update_proof": {
            "optimizer_updates": 1,
            "basis": [
                "one_native_rollout_with_one_step_per_rollout",
                "eight_accepted_train_episodes_with_positive_reward_variance",
                "positive_finite_gradient_and_finite_train_loss_at_update_zero",
                "all_rank_named_policy_tensor_value_delta_from_exact_base",
            ],
            "policy_tensor_payload_changed_from_base": True,
            "changed_policy_ranks": changed_policy_ranks,
            "counter_only_claim": False,
        },
        "wandb_scalar_observation": _reference(wandb_observation_path, wandb, wandb_file_sha256),
        "checkpoint_manifest": _reference(
            checkpoint_manifest_path, checkpoint, checkpoint_file_sha256
        ),
        "policy_delta_observation": _reference(
            policy_delta_observation_path, policy_delta, policy_delta_file_sha256
        ),
        "controller_observation": _reference(
            controller_observation_path, controller, controller_file_sha256
        ),
        "external_release": _reference(release_observation_path, release, release_file_sha256),
        "external_gpu_release_verified": True,
        "reward_values_included": False,
        "task_content_included": False,
        "production_promotion_requires_reload_acceptance": True,
    }


def accept_terminal(
    plan: dict[str, Any],
    *,
    submission_binding_path: Path,
    checkpoint_manifest_path: Path,
    policy_delta_observation_path: Path,
    controller_observation_path: Path,
    release_observation_path: Path,
    wandb_observation_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Create one sanitized terminal acceptance after all source gates pass."""
    root = Path(plan["output_root"])
    if output != root / "MILES_TERMINAL_ACCEPTED.json":
        raise ValueError("Miles terminal acceptance output path is not plan-bound")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Miles terminal acceptance already exists")
    value = _compile(
        plan,
        submission_binding_path=submission_binding_path,
        checkpoint_manifest_path=checkpoint_manifest_path,
        policy_delta_observation_path=policy_delta_observation_path,
        controller_observation_path=controller_observation_path,
        release_observation_path=release_observation_path,
        wandb_observation_path=wandb_observation_path,
    )
    return _write(output, value)


def validate_terminal(value: dict[str, Any], *, check_files: bool = True) -> dict[str, str]:
    """Reopen every terminal prerequisite for a later production promoter."""
    if not check_files:
        raise ValueError("Miles terminal promotion requires reopening all referenced evidence")
    sealed(value, TERMINAL_SCHEMA)
    episode = value.get("episode_audit")
    update = value.get("optimizer_update_proof")
    references = tuple(
        value.get(key)
        for key in (
            "submission_binding",
            "native_completion",
            "wandb_scalar_observation",
            "checkpoint_manifest",
            "policy_delta_observation",
            "controller_observation",
            "external_release",
        )
    )
    if (
        set(value) != _TERMINAL_FIELDS
        or value.get("status") != "accepted"
        or value.get("production_promotion_requires_reload_acceptance") is not True
        or value.get("external_gpu_release_verified") is not True
        or value.get("reward_values_included") is not False
        or value.get("task_content_included") is not False
        or not isinstance(episode, dict)
        or set(episode) != _EPISODE_AUDIT_FIELDS
        or episode.get("schema") != EPISODE_AUDIT_SCHEMA
        or episode.get("accepted_episode_count") != 10
        or episode.get("train_episode_count") != WORLD_SIZE
        or episode.get("dev_episode_count") != 2
        or episode.get("infrastructure_invalid_count") != 0
        or episode.get("truncated_count") != 0
        or episode.get("unique_authoritative_verifier_execution_count") != 10
        or episode.get("unique_challenge_instance_count") != 10
        or episode.get("unique_evidence_run_count") != 10
        or episode.get("train_nonzero_reward_present") is not True
        or episode.get("train_within_group_reward_variance_present") is not True
        or episode.get("reward_values_included") is not False
        or episode.get("task_content_included") is not False
        or not isinstance(update, dict)
        or set(update) != _UPDATE_PROOF_FIELDS
        or update.get("optimizer_updates") != 1
        or update.get("policy_tensor_payload_changed_from_base") is not True
        or type(update.get("changed_policy_ranks")) is not int
        or update["changed_policy_ranks"] < 1
        or update.get("counter_only_claim") is not False
        or not all(
            isinstance(reference, dict) and set(reference) == _REFERENCE_FIELDS
            for reference in references
        )
    ):
        raise ValueError("Miles terminal acceptance is not promotion-safe")
    checkpoint = value["checkpoint_manifest"]
    submission = value["submission_binding"]
    policy_delta = value["policy_delta_observation"]
    controller = value["controller_observation"]
    release = value["external_release"]
    wandb = value["wandb_scalar_observation"]
    submission_value, _ = _read(Path(submission["path"]), schema=SUBMISSION_SCHEMA)
    plan, _ = _json_snapshot(Path(submission_value["source_plan_path"]))
    if not isinstance(plan, dict):
        raise ValueError("Miles terminal source plan is not an object")
    _, _, world_size = _topology(plan)
    if update["changed_policy_ranks"] > world_size:
        raise ValueError("Miles terminal changed-rank count exceeds checkpoint topology")
    expected = _compile(
        plan,
        submission_binding_path=Path(submission["path"]),
        checkpoint_manifest_path=Path(checkpoint["path"]),
        policy_delta_observation_path=Path(policy_delta["path"]),
        controller_observation_path=Path(controller["path"]),
        release_observation_path=Path(release["path"]),
        wandb_observation_path=Path(wandb["path"]),
    )
    if _unsigned(value) != expected:
        raise ValueError("Miles terminal acceptance differs from rederived evidence")
    return {
        "source_plan_sha256": value["source_plan_sha256"],
        "terminal_receipt_sha256": value["sha256"],
    }
