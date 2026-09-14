"""Post-hoc, zero-update policy-state observation for one Miles canary.

The training process is not trusted to attest its own update.  This module
loads the exact base once and the CPU-sealed trained distributed checkpoint
twice in three isolated all-rank Miles actor groups, compares named policy
tensor values, and seals independently reproduced optimizer, scheduler, and
RNG commitments needed by the later reload gate.  It creates no rollout
engine and performs one fixed, task-free forward prediction probe, zero
backward passes or optimizer/scheduler updates, and no saves, evaluations, or
W&B calls.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import copy
import gzip
import hashlib
import json
import math
import numbers
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, bundled_request, digest, quantity

from . import miles
from .miles_conversion import _hash, _write
from .miles_conversion import inventory as base_inventory
from .miles_reload import (
    CHECKPOINT_SCHEMA,
    NATIVE_DRIVER_SHA256,
    RANK_STATE_COMMITMENT_METHOD,
    _model_probe,
    _optimizer_probe,
    _rng_probe,
    _scheduler_probe,
    _verify_checkpoint,
    native_source,
    reload_arguments,
)
from .rl_runtime import sealed

CONFIG_SCHEMA = "cyber_miles_policy_observer_config_v2"
PLAN_SCHEMA = "cyber_miles_policy_observer_plan_v2"
RESULT_SCHEMA = "cyber_miles_policy_observer_result_v2"
SUBMISSION_SCHEMA = "cyber_miles_policy_observer_submission_v2"
CAPTURE_INTENT_SCHEMA = "cyber_miles_policy_observer_capture_intent_v1"
CONTROLLER_SCHEMA = "cyber_miles_policy_observer_controller_v2"
RELEASE_QUERY_SCHEMA = "cyber_miles_policy_observer_release_query_v1"
RELEASE_SCHEMA = "cyber_miles_policy_observer_release_v2"
POLICY_SCHEMA = "cyber_miles_policy_tensor_delta_observation_v2"
RELOAD_ACCEPTED_SCHEMA = "cyber_miles_policy_observer_reload_accepted_v2"
WORLD_SIZE = 8
DEADLINE_SECONDS = 1800
COMPARISON_METHOD = "all_rank_named_policy_tensor_value_sha256_v2"
RESTORE_METHOD = "independent_preload_sentinel_overwrite_and_reload_sha256_v1"
PREDICTION_SCHEMA = "cyber_miles_non_task_prediction_probe_v1"
PREDICTION_PROBE_ID = "qwen38-fixed-token-topk-v1"
PREDICTION_INPUT_IDS = tuple(range(1, 33))
PREDICTION_TOP_K = 16
PREDICTION_MARGIN = 0.01
BASE_SENTINEL = 101
REFERENCE_SENTINEL = 202
RELOAD_SENTINEL = 303
RUNTIME_FILES = (
    "training/miles_policy_observer.py",
    "training/miles_reload.py",
    "training/miles.py",
    "training/miles_conversion.py",
    "training/rl_runtime.py",
    "cyber_post_train/jobs.py",
)
_SHA = re.compile(r"(?:sha256:)?[a-f0-9]{64}")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_REFERENCE_FIELDS = {"path", "file_sha256", "receipt_sha256"}
_PLAN_FIELDS = {
    "schema",
    "run_name",
    "output_root",
    "source_plan",
    "source_plan_sha256",
    "source_plan_file_sha256",
    "source_plan_path",
    "source_submission",
    "trained_checkpoint",
    "trained_checkpoint_reference",
    "base_checkpoint_receipt_sha256",
    "world_size",
    "comparison_method",
    "rank_state_commitment_method",
    "work_authorized",
    "runtime_sha256",
    "native_driver_sha256",
    "deadline_seconds",
    "restore_method",
    "sentinel_markers",
    "execution",
}
_RESULT_FIELDS = {
    "schema",
    "status",
    "observer_plan_sha256",
    "source_plan_sha256",
    "base_checkpoint_receipt_sha256",
    "trained_checkpoint_receipt_sha256",
    "world_size",
    "comparison_method",
    "ranks",
    "changed_policy_ranks",
    "restored_next_rollout_id",
    "checkpoint_state_commitment_method",
    "restore_method",
    "trained_restore_count",
    "prediction_probe",
    "state_stable_across_zero_updates",
    "work_executed",
    "base_checkpoint_unchanged",
    "trained_checkpoint_unchanged",
    "completed_at",
    "reward_values_included",
    "task_content_included",
    "tensor_values_included",
    "sha256",
}
_RELOAD_ACCEPTED_FIELDS = {
    "schema",
    "status",
    "source_manifest_sha256",
    "source_terminal_acceptance_sha256",
    "source_policy_delta_observation_sha256",
    "terminal_acceptance",
    "checkpoint_manifest",
    "policy_delta_observation",
    "observer_result",
    "observer_controller",
    "observer_release",
    "world_size",
    "ranks",
    "restored_rollout_index",
    "restored_next_rollout_id",
    "rank_state_commitment_method",
    "rank_state_commitments_sha256",
    "prediction_probe",
    "all_rank_model_loaded",
    "all_rank_optimizer_loaded",
    "all_rank_scheduler_loaded",
    "all_rank_rng_loaded",
    "state_stable_across_zero_updates",
    "exact_checkpoint_payload_reopened_after_release",
    "source_checkpoint_unchanged_after_release",
    "work_executed",
    "observer_gpu_jobs",
    "additional_reload_gpu_jobs",
    "external_gpu_release_verified",
    "production_promotion_requires_this_receipt",
    "reward_values_included",
    "task_content_included",
    "tensor_values_included",
    "sha256",
}
_RANK_FIELDS = {
    "rank",
    "policy_tensor_count",
    "local_policy_numel",
    "base",
    "trained_reference",
    "trained_reload",
    "policy_changed",
}
_STATE_FIELDS = {"model", "optimizer", "scheduler", "rng_sha256"}
_LOAD_FIELDS = {"marker", "load_calls", "preload", "restored", "live"}
_SUBMISSION_FIELDS = {
    "schema",
    "source_commit",
    "observer_plan_path",
    "observer_plan_sha256",
    "observer_plan_file_sha256",
    "observer_request_path",
    "observer_request_sha256",
    "observer_request_file_sha256",
    "submission_journal_path",
    "submission_journal_file_sha256",
    "runtime_bundle_sha256",
    "api",
    "request",
    "jobs_api_post_count",
    "submitted_at",
    "secret_values_included",
    "task_content_included",
    "metric_values_included",
    "sha256",
}
_CAPTURE_INTENT_FIELDS = {
    "schema",
    "status",
    "observer_plan_sha256",
    "observer_request_sha256",
    "requested_name",
    "api_base_url",
    "kube_context",
    "namespace",
    "namespace_uid",
    "started_at",
    "private_logs_included",
    "metric_values_included",
    "task_content_included",
    "sha256",
}
_RELEASE_QUERY_FIELDS = {
    "schema",
    "status",
    "observer_plan_sha256",
    "observer_request_sha256",
    "controller_observation_sha256",
    "api",
    "kubernetes",
    "active_gpus",
    "observed_at",
    "private_logs_included",
    "metric_values_included",
    "task_content_included",
    "sha256",
}
_SUBMISSION_REQUEST_FIELDS = {
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
_ZERO_WORK = {
    "rollouts": 0,
    "verifier_calls": 0,
    "forward_passes": 0,
    "backward_passes": 0,
    "optimizer_updates": 0,
    "scheduler_updates": 0,
    "checkpoint_writes": 0,
    "wandb_events": 0,
}
_OBSERVER_WORK = {**_ZERO_WORK, "forward_passes": 1}
_ENV = {
    "PYTHONPATH": "/root/Megatron-LM",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MILES_USE_LEGACY_ROLLOUT_V1": "0",
    "MILES_EXPERIMENTAL_FT_TRAINER": "0",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "PYTHONUNBUFFERED": "1",
}


def _runtime() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def _sha(value: object, label: str) -> str:
    result = str(value).removeprefix("sha256:")
    if re.fullmatch(r"[a-f0-9]{64}", result) is None:
        raise ValueError(f"{label} is not an exact SHA-256")
    return result


def _reference(path: Path, value: Mapping[str, Any], file_sha256: str) -> dict[str, str]:
    return {
        "path": str(path),
        "file_sha256": "sha256:" + file_sha256.removeprefix("sha256:"),
        "receipt_sha256": "sha256:" + _sha(value.get("sha256"), "receipt digest"),
    }


def _json_snapshot(path: Path) -> tuple[Any, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("policy observer input must be a regular file")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ValueError("policy observer input changed while reading")
    try:
        return json.loads(payload), hashlib.sha256(payload).hexdigest()
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("policy observer input is not valid JSON") from error


def _time(value: object, label: str) -> float:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as error:
            raise ValueError(f"{label} is not an RFC3339 timestamp") from error
    else:
        raise ValueError(f"{label} is not a timestamp")
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} is not a positive finite timestamp")
    return result


def _read(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    value, file_sha256 = _json_snapshot(path)
    if not isinstance(value, dict):
        raise ValueError("observer evidence must be a JSON object")
    if schema is None:
        if _sha(value.get("sha256"), "receipt digest") != digest(
            {key: item for key, item in value.items() if key != "sha256"}
        ):
            raise ValueError("observer evidence self-digest mismatch")
    else:
        sealed(value, schema)
    return value, file_sha256


def _source_canary(plan: Mapping[str, Any]) -> None:
    args = plan.get("arguments")
    execution = plan.get("execution")
    data = plan.get("data")
    model = plan.get("model")
    files = data.get("files") if isinstance(data, Mapping) else None
    if (
        plan.get("schema") != "cyber_miles_training_v1"
        or not isinstance(args, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(model, Mapping)
        or plan.get("run_name") != args.get("name")
        or plan.get("output_root") != args.get("output_root")
        or model.get("repo") != "Qwen/Qwen3.8-27B"
        or args.get("model") != "Qwen/Qwen3.8-27B"
        or args.get("nodes") != 1
        or args.get("gpus_per_node") != WORLD_SIZE
        or args.get("steps") != 1
        or args.get("groups") != 1
        or args.get("samples_per_prompt") != WORLD_SIZE
        or args.get("eval_interval") != 1
        or args.get("checkpoint_interval") != 1
        or execution.get("image") != miles.IMAGE
        or execution.get("priority") != "c1"
        or not isinstance(files, Mapping)
        or set(files) != {"train", "dev"}
        or files["train"].get("rows") != 1
        or files["dev"].get("rows") != 1
    ):
        raise ValueError("policy observer source is not the exact one-update Miles canary")


def _base_checkpoint(plan: Mapping[str, Any], *, hashes: bool) -> list[dict[str, Any]]:
    checkpoint = plan.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("source plan has no exact base checkpoint")
    sealed(checkpoint, "cyber_miles_checkpoint_v1")
    root = Path(str(checkpoint.get("root")))
    observed = base_inventory(root)
    expected = [{key: row[key] for key in ("path", "size")} for row in checkpoint["files"]]
    if (
        checkpoint.get("image") != miles.IMAGE
        or checkpoint.get("optimizer_steps") != 0
        or checkpoint.get("model") != plan.get("model")
        or observed != expected
    ):
        raise ValueError("base checkpoint differs from the exact source plan")
    if hashes:
        for row in checkpoint["files"]:
            if _hash(root / row["path"]) != _sha(row["sha256"], "base checkpoint file digest"):
                raise ValueError("base checkpoint payload changed")
    return observed


def _source_bindings(
    source_plan_path: Path, submission_path: Path, checkpoint_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    from . import miles_acceptance

    source_plan, source_plan_file_sha256 = miles_acceptance._json_snapshot(source_plan_path)
    if not isinstance(source_plan, dict):
        raise ValueError("source Miles plan is not a JSON object")
    miles_acceptance._canary(source_plan)
    submission, submission_file_sha256 = _read(submission_path, miles_acceptance.SUBMISSION_SCHEMA)
    miles_acceptance.validate_submission_binding(submission, source_plan, check_files=True)
    if Path(submission["source_plan_path"]) != source_plan_path:
        raise ValueError("source submission does not bind the selected plan path")
    trained, trained_file_sha256 = miles_acceptance._checkpoint(source_plan, checkpoint_path)
    _base_checkpoint(source_plan, hashes=True)
    refs = {
        "source_plan_path": str(source_plan_path),
        "source_plan_file_sha256": source_plan_file_sha256,
        "source_submission_path": str(submission_path),
        "source_submission_file_sha256": submission_file_sha256,
        "trained_checkpoint_path": str(checkpoint_path),
        "trained_checkpoint_file_sha256": trained_file_sha256,
    }
    return source_plan, submission, trained, refs


def compile_observer(config: dict[str, Any], *, relative_to: Path) -> dict[str, Any]:
    from .sft import _known, _sfs_root

    _known(config, {"schema", "name", "output_root", "source", "cluster"}, "Miles observer")
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Miles observer configuration schema mismatch")
    source = config["source"]
    cluster = config["cluster"]
    _known(source, {"plan", "submission_binding", "trained_checkpoint_manifest"}, "source")
    _known(cluster, {"target", "priority", "resources"}, "cluster")
    if cluster.get("target") != "dev" or cluster.get("priority") != "c1":
        raise ValueError("policy observation is a dev-only c1 qualification")
    source_plan_path = (relative_to / source["plan"]).resolve()
    submission_path = (relative_to / source["submission_binding"]).resolve()
    checkpoint_path = (relative_to / source["trained_checkpoint_manifest"]).resolve()
    source_plan, submission, trained, refs = _source_bindings(
        source_plan_path, submission_path, checkpoint_path
    )
    output = _sfs_root(config["output_root"], "observer output root")
    output_path = Path(output)
    protected = {
        Path(source_plan["output_root"]),
        Path(source_plan["checkpoint"]["root"]),
        Path(trained["root"]),
    }
    if any(
        output_path == path or output_path in path.parents or path in output_path.parents
        for path in protected
    ):
        raise ValueError("observer output must be disjoint from immutable checkpoints and source")
    resources = {**source_plan["execution"]["resources"], **cluster.get("resources", {})}
    if any(
        quantity(resources[key]) < quantity(source_plan["execution"]["resources"][key])
        for key in ("cpu_request", "cpu_limit", "memory_request", "memory_limit")
    ):
        raise ValueError("policy observer cannot underreserve the successful source topology")
    plan = {
        "schema": PLAN_SCHEMA,
        "run_name": config["name"],
        "output_root": output,
        "source_plan": source_plan,
        "source_plan_sha256": "sha256:" + digest(source_plan),
        "source_plan_file_sha256": "sha256:" + refs["source_plan_file_sha256"],
        "source_plan_path": refs["source_plan_path"],
        "source_submission": _reference(
            submission_path, submission, refs["source_submission_file_sha256"]
        ),
        "trained_checkpoint": trained,
        "trained_checkpoint_reference": _reference(
            checkpoint_path, trained, refs["trained_checkpoint_file_sha256"]
        ),
        "base_checkpoint_receipt_sha256": source_plan["checkpoint"]["sha256"],
        "world_size": WORLD_SIZE,
        "comparison_method": COMPARISON_METHOD,
        "rank_state_commitment_method": RANK_STATE_COMMITMENT_METHOD,
        "work_authorized": dict(_OBSERVER_WORK),
        "runtime_sha256": digest(_runtime()),
        "native_driver_sha256": NATIVE_DRIVER_SHA256,
        "deadline_seconds": DEADLINE_SECONDS,
        "restore_method": RESTORE_METHOD,
        "sentinel_markers": {
            "base": BASE_SENTINEL,
            "trained_reference": REFERENCE_SENTINEL,
            "trained_reload": RELOAD_SENTINEL,
        },
        "execution": {
            "cluster_target": "dev",
            "image": miles.IMAGE,
            "priority": "c1",
            "resources": resources,
        },
    }
    job_request(plan)
    return plan


def _validate_plan(
    plan: Mapping[str, Any], *, check_files: bool, require_current_runtime: bool
) -> None:
    source = plan.get("source_plan")
    trained = plan.get("trained_checkpoint")
    submission_ref = plan.get("source_submission")
    checkpoint_ref = plan.get("trained_checkpoint_reference")
    if (
        set(plan) != _PLAN_FIELDS
        or plan.get("schema") != PLAN_SCHEMA
        or not isinstance(source, dict)
        or plan.get("source_plan_sha256", "").removeprefix("sha256:") != digest(source)
        or _SHA.fullmatch(str(plan.get("source_plan_file_sha256"))) is None
        or not Path(str(plan.get("source_plan_path", ""))).is_absolute()
        or not Path(str(plan.get("output_root", ""))).is_absolute()
        or plan.get("base_checkpoint_receipt_sha256") != source.get("checkpoint", {}).get("sha256")
        or plan.get("world_size") != WORLD_SIZE
        or plan.get("comparison_method") != COMPARISON_METHOD
        or plan.get("rank_state_commitment_method") != RANK_STATE_COMMITMENT_METHOD
        or plan.get("work_authorized") != _OBSERVER_WORK
        or _SHA.fullmatch(str(plan.get("runtime_sha256"))) is None
        or plan.get("native_driver_sha256") != NATIVE_DRIVER_SHA256
        or plan.get("deadline_seconds") != DEADLINE_SECONDS
        or plan.get("restore_method") != RESTORE_METHOD
        or plan.get("sentinel_markers")
        != {
            "base": BASE_SENTINEL,
            "trained_reference": REFERENCE_SENTINEL,
            "trained_reload": RELOAD_SENTINEL,
        }
        or plan.get("execution", {}).get("cluster_target") != "dev"
        or plan.get("execution", {}).get("image") != miles.IMAGE
        or plan.get("execution", {}).get("priority") != "c1"
        or not isinstance(submission_ref, dict)
        or set(submission_ref) != _REFERENCE_FIELDS
        or not isinstance(checkpoint_ref, dict)
        or set(checkpoint_ref) != _REFERENCE_FIELDS
    ):
        raise ValueError("Miles policy observer plan drift")
    if require_current_runtime and plan["runtime_sha256"] != digest(_runtime()):
        raise ValueError("Miles policy observer runtime differs from the compiled plan")
    _source_canary(source)
    sealed(trained, CHECKPOINT_SCHEMA)
    if (
        trained.get("world_size") != WORLD_SIZE
        or trained.get("topology") != {"nodes": 1, "gpus_per_node": WORLD_SIZE}
        or trained.get("source", {}).get("plan_sha256") != digest(source)
        or Path(str(submission_ref["path"])) == Path(str(checkpoint_ref["path"]))
        or not Path(str(submission_ref["path"])).is_absolute()
        or not Path(str(checkpoint_ref["path"])).is_absolute()
        or any(
            _SHA.fullmatch(str(reference[key])) is None
            for reference in (submission_ref, checkpoint_ref)
            for key in ("file_sha256", "receipt_sha256")
        )
        or checkpoint_ref["receipt_sha256"].removeprefix("sha256:")
        != _sha(trained.get("sha256"), "trained checkpoint digest")
    ):
        raise ValueError("Miles policy observer trained checkpoint drift")
    if check_files:
        source_path = Path(str(plan["source_plan_path"]))
        reopened_source, source_file_sha256 = _json_snapshot(source_path)
        if reopened_source != source or source_file_sha256 != plan[
            "source_plan_file_sha256"
        ].removeprefix("sha256:"):
            raise ValueError("source Miles plan file changed")
        submission, submission_file_sha256 = _read(
            Path(submission_ref["path"]), "cyber_miles_submitted_execution_binding_v1"
        )
        if submission_file_sha256 != submission_ref["file_sha256"].removeprefix("sha256:") or _sha(
            submission["sha256"], "source submission digest"
        ) != submission_ref["receipt_sha256"].removeprefix("sha256:"):
            raise ValueError("source submission evidence changed")
        if (
            submission.get("source_plan_sha256", "").removeprefix("sha256:") != digest(source)
            or Path(str(submission.get("source_plan_path"))) != source_path
        ):
            raise ValueError("source submission does not bind the selected Miles plan")
        checkpoint, checkpoint_file_sha256 = _read(Path(checkpoint_ref["path"]), CHECKPOINT_SCHEMA)
        if checkpoint != trained or checkpoint_file_sha256 != checkpoint_ref[
            "file_sha256"
        ].removeprefix("sha256:"):
            raise ValueError("trained checkpoint manifest changed")
        _verify_checkpoint(trained, hashes=True)
        _base_checkpoint(source, hashes=True)


def job_request(plan: dict[str, Any]) -> dict[str, Any]:
    _validate_plan(plan, check_files=False, require_current_runtime=True)
    files = _runtime()
    files.update({"training/__init__.py": "", "cyber_post_train/__init__.py": ""})
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    topology = plan["trained_checkpoint"]["topology"]
    return bundled_request(
        {
            "name": plan["run_name"],
            "title": plan["run_name"] + " zero-update policy observer",
            "run_dir": plan["output_root"],
            "image": miles.IMAGE,
            "workers": topology["nodes"],
            "gpus_per_worker": topology["gpus_per_node"],
            "resources": plan["execution"]["resources"],
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "secrets": [],
            "env": dict(_ENV),
        },
        files,
        "training.miles_policy_observer",
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def _transport_blob(request: Mapping[str, Any]) -> bytes:
    env = request.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("observer request environment is absent")
    direct = env.get("CYBER_RUNTIME_BUNDLE")
    chunks = sorted(
        (
            (int(key.removeprefix("CYBER_RUNTIME_BUNDLE_")), value)
            for key, value in env.items()
            if re.fullmatch(r"CYBER_RUNTIME_BUNDLE_\d+", str(key))
        ),
        key=lambda item: item[0],
    )
    if direct is not None:
        if chunks:
            raise ValueError("observer runtime transport is ambiguous")
        encoded = direct
    else:
        if not chunks or [index for index, _ in chunks] != list(range(len(chunks))):
            raise ValueError("observer runtime transport is absent or non-contiguous")
        encoded = "".join(value for _, value in chunks)
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("observer runtime transport is not valid base64") from error


def _request_projection(
    plan: dict[str, Any], request: dict[str, Any], source_commit: str, repo_root: Path
) -> tuple[str, dict[str, Any]]:
    if re.fullmatch(r"[a-f0-9]{40}", source_commit) is None:
        raise ValueError("observer source commit is not a full Git SHA")
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
    env = request.get("env")
    if not isinstance(env, dict):
        raise ValueError("observer request environment is absent")
    transport = {
        key
        for key in env
        if key == "CYBER_RUNTIME_BUNDLE" or re.fullmatch(r"CYBER_RUNTIME_BUNDLE_\d+", key)
    }
    ordinary_env = {key: value for key, value in env.items() if key not in transport}
    if (
        set(request) != expected_keys
        or request.get("name") != plan["run_name"]
        or request.get("title") != plan["run_name"] + " zero-update policy observer"
        or request.get("run_dir") != plan["output_root"]
        or request.get("image") != miles.IMAGE
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != WORLD_SIZE
        or request.get("resources") != plan["execution"]["resources"]
        or request.get("priority_class") != "c1"
        or request.get("requeueIfPreempted") is not False
        or request.get("secrets") != []
        or ordinary_env != _ENV
    ):
        raise ValueError("observer request differs from its exact plan")
    blob = _transport_blob(request)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    try:
        payload = json.loads(gzip.decompress(blob))
        command = shlex.split(request["command"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("observer runtime bundle or command is invalid") from error
    if (
        len(command) != 3
        or command[:2] != ["python", "-c"]
        or blob_sha256 not in command[2]
        or not isinstance(payload, dict)
        or set(payload) != {"files", "module", "argv"}
        or payload.get("module") != "training.miles_policy_observer"
        or payload.get("argv") != ["--plan", "plan.json", "--sha256", digest(plan)]
        or payload.get("files", {}).get("plan.json")
        != json.dumps(plan, sort_keys=True, separators=(",", ":"))
        or not set(RUNTIME_FILES).issubset(payload.get("files", {}))
    ):
        raise ValueError("observer runtime bundle is not plan-bound")
    source_files = {path: payload["files"][path] for path in RUNTIME_FILES}
    if digest(source_files) != plan["runtime_sha256"]:
        raise ValueError("observer runtime source digest changed")
    for path, text in source_files.items():
        process = subprocess.run(
            ["git", "show", f"{source_commit}:{path}"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.returncode or process.stdout != text.encode():
            raise ValueError("observer runtime differs from its source commit")
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
    submission_journal_path: Path,
    output: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    from .miles_acceptance import _json_snapshot, _time, _uuid

    plan, plan_file_sha256 = _json_snapshot(plan_path)
    request, request_file_sha256 = _json_snapshot(request_path)
    if not isinstance(plan, dict) or not isinstance(request, dict):
        raise ValueError("observer plan and request must be JSON objects")
    _validate_plan(plan, check_files=True, require_current_runtime=True)
    bundle_sha256, projection = _request_projection(
        plan, request, source_commit, repo_root or Path(__file__).resolve().parents[1]
    )
    journal, journal_file_sha256 = _submission_journal(submission_journal_path, request)
    run_id = _uuid(journal[1]["job_id"], "observer API run ID")
    submitted_at = journal[1]["created_at"]
    _time(submitted_at, "observer submission time")
    return _write(
        output,
        {
            "schema": SUBMISSION_SCHEMA,
            "source_commit": source_commit,
            "observer_plan_path": str(plan_path),
            "observer_plan_sha256": "sha256:" + digest(plan),
            "observer_plan_file_sha256": "sha256:" + plan_file_sha256,
            "observer_request_path": str(request_path),
            "observer_request_sha256": "sha256:" + digest(request),
            "observer_request_file_sha256": "sha256:" + request_file_sha256,
            "submission_journal_path": str(submission_journal_path),
            "submission_journal_file_sha256": "sha256:" + journal_file_sha256,
            "runtime_bundle_sha256": "sha256:" + bundle_sha256,
            "api": {
                "base_url": API_URLS["dev"],
                "run_id": run_id,
                "run_name": plan["run_name"] + "-" + run_id[:8],
            },
            "request": projection,
            "jobs_api_post_count": 1,
            "submitted_at": submitted_at,
            "secret_values_included": False,
            "task_content_included": False,
            "metric_values_included": False,
        },
    )


def _submission_journal(path: Path, request: Mapping[str, Any]) -> tuple[list[dict], str]:
    from .miles_acceptance import _time, _uuid

    if path.is_symlink() or not path.is_file():
        raise ValueError("observer Jobs API submission journal is missing or indirect")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError("observer Jobs API submission journal changed while reading")
    try:
        rows = [json.loads(line) for line in payload.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("observer Jobs API submission journal is invalid JSONL") from error
    if len(rows) != 2 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("observer Jobs API journal must contain one intent and one response")
    intent, response = rows
    if (
        set(intent)
        != {
            "state",
            "api_base_url",
            "request_sha256",
            "manifest_sha256",
            "nodes",
            "gpus",
            "image",
        }
        or intent.get("state") != "POST_INTENT_DO_NOT_RETRY"
        or intent.get("api_base_url") != API_URLS["dev"]
        or intent.get("request_sha256") != digest(request)
        or _SHA.fullmatch(str(intent.get("manifest_sha256"))) is None
        or intent.get("nodes") != 1
        or intent.get("gpus") != WORLD_SIZE
        or intent.get("image") != miles.IMAGE
        or set(response)
        != {"state", "name", "job_id", "run_dir", "status", "created_at", "finished_at"}
        or response.get("state") != "POST_RESPONSE"
        or response.get("name") != str(request["name"]) + "-" + str(response.get("job_id", ""))[:8]
        or response.get("run_dir") != request["run_dir"]
        or response.get("status")
        not in {"queued", "PENDING", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "STOPPED"}
        or (
            response.get("status") in {"queued", "PENDING", "QUEUED", "RUNNING"}
            and response.get("finished_at") is not None
        )
        or (
            response.get("status") in {"SUCCEEDED", "FAILED", "STOPPED"}
            and response.get("finished_at") is None
        )
    ):
        raise ValueError("observer Jobs API submission journal is incomplete or mismatched")
    _uuid(response.get("job_id"), "observer API run ID")
    created_at = _time(response.get("created_at"), "observer submission time")
    if (
        response.get("finished_at") is not None
        and _time(response["finished_at"], "observer recovered terminal time") < created_at
    ):
        raise ValueError("observer recovered terminal time predates submission")
    return rows, hashlib.sha256(payload).hexdigest()


def validate_submission_binding(
    value: dict[str, Any], plan: dict[str, Any], *, check_files: bool
) -> dict[str, Any]:
    from .miles_acceptance import _json_snapshot, _time

    _validate_plan(plan, check_files=False, require_current_runtime=False)
    sealed(value, SUBMISSION_SCHEMA)
    api = value.get("api")
    request = value.get("request")
    if (
        set(value) != _SUBMISSION_FIELDS
        or re.fullmatch(r"[a-f0-9]{40}", str(value.get("source_commit"))) is None
        or value.get("observer_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or _SHA.fullmatch(str(value.get("observer_plan_file_sha256"))) is None
        or _SHA.fullmatch(str(value.get("observer_request_sha256"))) is None
        or _SHA.fullmatch(str(value.get("observer_request_file_sha256"))) is None
        or not Path(str(value.get("submission_journal_path", ""))).is_absolute()
        or _SHA.fullmatch(str(value.get("submission_journal_file_sha256"))) is None
        or _SHA.fullmatch(str(value.get("runtime_bundle_sha256"))) is None
        or not isinstance(api, dict)
        or set(api) != {"base_url", "run_id", "run_name"}
        or api.get("base_url") != API_URLS["dev"]
        or _UUID.fullmatch(str(api.get("run_id"))) is None
        or api.get("run_name") != plan["run_name"] + "-" + str(api.get("run_id"))[:8]
        or not isinstance(request, dict)
        or set(request) != _SUBMISSION_REQUEST_FIELDS
        or request.get("name") != plan["run_name"]
        or request.get("run_dir") != plan["output_root"]
        or request.get("image") != miles.IMAGE
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != WORLD_SIZE
        or request.get("resources") != plan["execution"]["resources"]
        or request.get("priority_class") != "c1"
        or request.get("automatic_requeue") is not False
        or request.get("secret_names") != []
        or request.get("environment_names") != sorted(_ENV)
        or request.get("runtime_module") != "training.miles_policy_observer"
        or request.get("runtime_argv") != ["--plan", "plan.json", "--sha256", digest(plan)]
        or value.get("jobs_api_post_count") != 1
        or value.get("secret_values_included") is not False
        or value.get("task_content_included") is not False
        or value.get("metric_values_included") is not False
    ):
        raise ValueError("observer submission binding is incomplete or mismatched")
    _time(value.get("submitted_at"), "observer submission time")
    if check_files:
        reopened_plan, plan_file_sha256 = _json_snapshot(Path(value["observer_plan_path"]))
        reopened_request, request_file_sha256 = _json_snapshot(Path(value["observer_request_path"]))
        if (
            reopened_plan != plan
            or value["observer_plan_file_sha256"].removeprefix("sha256:") != plan_file_sha256
            or not isinstance(reopened_request, dict)
            or value["observer_request_sha256"].removeprefix("sha256:") != digest(reopened_request)
            or value["observer_request_file_sha256"].removeprefix("sha256:") != request_file_sha256
        ):
            raise ValueError("observer plan or request file changed")
        bundle, projection = _request_projection(
            plan, reopened_request, value["source_commit"], Path(__file__).resolve().parents[1]
        )
        journal, journal_file_sha256 = _submission_journal(
            Path(value["submission_journal_path"]), reopened_request
        )
        if (
            value["runtime_bundle_sha256"].removeprefix("sha256:") != bundle
            or projection != request
            or value["submission_journal_file_sha256"].removeprefix("sha256:")
            != journal_file_sha256
            or journal[1]["job_id"] != api["run_id"]
            or journal[1]["name"] != api["run_name"]
            or journal[1]["created_at"] != value["submitted_at"]
        ):
            raise ValueError("observer runtime bundle changed")
    return {"request_sha256": value["observer_request_sha256"], "api": api}


def start_capture_intent(
    plan: dict[str, Any],
    request: dict[str, Any],
    *,
    namespace_uid: str,
    started_at: object,
    directory: Path,
    kube_context: str,
) -> dict[str, Any]:
    """Commit the exact watch target before the single Jobs API POST."""
    from . import miles_event_evidence as events
    from .miles_acceptance import NAMESPACE, NAMESPACE_UID, _time, _uuid

    _validate_plan(plan, check_files=False, require_current_runtime=False)
    if (
        request != job_request(plan)
        or kube_context != events.DEV_KUBE_CONTEXT
        or namespace_uid != NAMESPACE_UID
    ):
        raise ValueError("observer capture intent differs from exact dev3 request")
    _time(started_at, "observer capture-intent start")
    namespace = _uuid(namespace_uid, "observer namespace UID")
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    return _write(
        directory / "CAPTURE_INTENT.json",
        {
            "schema": CAPTURE_INTENT_SCHEMA,
            "status": "watch_starting_before_post",
            "observer_plan_sha256": "sha256:" + digest(plan),
            "observer_request_sha256": "sha256:" + digest(request),
            "requested_name": plan["run_name"],
            "api_base_url": API_URLS["dev"],
            "kube_context": kube_context,
            "namespace": NAMESPACE,
            "namespace_uid": namespace,
            "started_at": started_at,
            "private_logs_included": False,
            "metric_values_included": False,
            "task_content_included": False,
        },
    )


def start_capture(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    intent_path: Path,
) -> dict[str, Any]:
    """Bind an already-running pre-POST watcher to the returned API identity."""
    from . import miles_event_evidence as events
    from .miles_acceptance import _time

    _validate_plan(plan, check_files=False, require_current_runtime=False)
    submitted = validate_submission_binding(submission, plan, check_files=False)
    intent, intent_file_sha256 = _read(intent_path, CAPTURE_INTENT_SCHEMA)
    directory = intent_path.parent
    if (
        set(intent) != _CAPTURE_INTENT_FIELDS
        or intent.get("status") != "watch_starting_before_post"
        or intent.get("observer_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or intent.get("observer_request_sha256") != submitted["request_sha256"]
        or intent.get("requested_name") != plan["run_name"]
        or intent.get("api_base_url") != submitted["api"]["base_url"]
        or intent.get("kube_context") != events.DEV_KUBE_CONTEXT
        or intent.get("namespace") != "fleet-train-jobs"
        or intent.get("private_logs_included") is not False
        or intent.get("metric_values_included") is not False
        or intent.get("task_content_included") is not False
        or _time(intent.get("started_at"), "observer capture start")
        > _time(submission.get("submitted_at"), "observer submission time")
    ):
        raise ValueError("observer watcher was not committed before its Jobs API POST")
    return _write(
        directory / "STARTED.json",
        {
            "schema": events.START_SCHEMA,
            "status": "watching",
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_request_sha256": submitted["request_sha256"],
            "api_base_url": submitted["api"]["base_url"],
            "api_run_id": submitted["api"]["run_id"],
            "api_run_name": submitted["api"]["run_name"],
            "cluster": "dev",
            "kube_context": intent["kube_context"],
            "namespace": intent["namespace"],
            "namespace_uid": intent["namespace_uid"],
            "started_at": intent["started_at"],
            "capture_intent": _reference(intent_path, intent, intent_file_sha256),
            "private_logs_included": False,
            "metric_values_included": False,
            "task_content_included": False,
        },
    )


def _controller_body(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    directory: Path,
    api_status: str,
    observed_at: object,
) -> dict[str, Any]:
    """Derive terminal observer identity from a pre-admission event journal."""
    from . import miles_event_evidence as events

    submitted = validate_submission_binding(submission, plan, check_files=True)
    start_path = directory / "STARTED.json"
    start, start_file_sha256 = events._read(start_path, events.START_SCHEMA)
    intent_ref = start.get("capture_intent")
    if not isinstance(intent_ref, dict) or set(intent_ref) != _REFERENCE_FIELDS:
        raise ValueError("policy observer event capture omitted its pre-POST intent")
    intent, intent_file_sha256 = _read(Path(intent_ref["path"]), CAPTURE_INTENT_SCHEMA)
    if (
        start["source_plan_sha256"].removeprefix("sha256:") != digest(plan)
        or start["source_request_sha256"] != submitted["request_sha256"]
        or start["api_run_id"] != submitted["api"]["run_id"]
        or start["api_run_name"] != submitted["api"]["run_name"]
        or start.get("kube_context") != intent.get("kube_context")
        or start.get("started_at") != intent.get("started_at")
        or intent_file_sha256 != intent_ref["file_sha256"].removeprefix("sha256:")
        or intent["sha256"].removeprefix("sha256:")
        != intent_ref["receipt_sha256"].removeprefix("sha256:")
        or api_status != "SUCCEEDED"
    ):
        raise ValueError("policy observer event capture differs from its API run")
    event_paths = events._event_files(directory)
    event_records = [events._read(path, events.EVENT_SCHEMA) for path in event_paths]
    journal = [value for value, _ in event_records]
    if [row["sequence"] for row in journal] != list(range(len(journal))):
        raise ValueError("policy observer event sequence changed")
    for row in journal:
        if (
            row["source_plan_sha256"] != start["source_plan_sha256"]
            or row["source_request_sha256"] != start["source_request_sha256"]
            or row["api_run_id"] != start["api_run_id"]
            or row["api_run_name"] != start["api_run_name"]
            or row["private_logs_included"] is not False
            or row["metric_values_included"] is not False
            or row["task_content_included"] is not False
        ):
            raise ValueError("policy observer event identity or minimization changed")
    by_kind = {
        kind: [row for row in journal if row["kind"] == kind]
        for kind in ("RayJob", "Workload", "RayCluster", "Pod")
    }
    if any(not rows for rows in by_kind.values()):
        raise ValueError("policy observer journal missed a lifecycle object")
    identities: dict[str, tuple[str, str]] = {}
    for kind, rows in by_kind.items():
        values = {(row["name"], row["uid"]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"policy observer journal contains multiple {kind} identities")
        identities[kind] = next(iter(values))
    rayjob_name, rayjob_uid = identities["RayJob"]
    workload_name, workload_uid = identities["Workload"]
    raycluster_name, raycluster_uid = identities["RayCluster"]
    pod_name, pod_uid = identities["Pod"]
    if (
        rayjob_name != start["api_run_name"]
        or any(
            row["owner"] != {"kind": "RayJob", "name": rayjob_name, "uid": rayjob_uid}
            for kind in ("Workload", "RayCluster")
            for row in by_kind[kind]
        )
        or any(
            row["owner"] != {"kind": "RayCluster", "name": raycluster_name, "uid": raycluster_uid}
            for row in by_kind["Pod"]
        )
    ):
        raise ValueError("policy observer Kubernetes ownership changed")
    admissions = [row["admitted_at"] for row in by_kind["Workload"] if row["admitted_at"]]
    if not admissions:
        raise ValueError("policy observer journal missed Workload admission")
    admitted_at = min(events._time(value, "observer Workload admission") for value in admissions)
    if events._time(start["started_at"], "observer capture start") > admitted_at:
        raise ValueError("policy observer event capture began after admission")
    terminal_jobs = [row for row in by_kind["RayJob"] if row["controller_status"] == "SUCCEEDED"]
    terminal_pods = [
        row
        for row in by_kind["Pod"]
        if isinstance(row["pod"], dict)
        and row["pod"]["phase"] == "Succeeded"
        and row["pod"]["exit_code"] == 0
        and row["pod"]["termination_reason"] == "Completed"
    ]
    if not terminal_jobs or not terminal_pods:
        raise ValueError("policy observer journal missed terminal success")
    terminal_job, terminal_pod = terminal_jobs[-1], terminal_pods[-1]
    pod = terminal_pod["pod"]
    if (
        terminal_job["shutdown_after_job_finishes"] is not True
        or terminal_job["ttl_seconds_after_finished"] != 0
        or terminal_job["priority_class"] != "c1"
        or terminal_job["raycluster_name"] != raycluster_name
        or pod["container_restarts"] != 0
        or pod["gpus"] != WORLD_SIZE
        or not isinstance(pod["runtime_image_id"], str)
        or not pod["runtime_image_id"].endswith(plan["execution"]["image"].rsplit("@", 1)[1])
    ):
        raise ValueError("policy observer terminal execution differs from its plan")
    events._time(pod["terminated_at"], "policy observer Pod termination")
    terminal_observed = max(
        events._time(observed_at, "policy observer API terminal observation"),
        events._time(terminal_job["observed_at"], "policy observer RayJob terminal event"),
        events._time(terminal_pod["observed_at"], "policy observer Pod terminal event"),
    )
    return {
        "schema": CONTROLLER_SCHEMA,
        "status": "succeeded",
        "observer_plan_sha256": "sha256:" + digest(plan),
        "observer_request_sha256": submitted["request_sha256"],
        "api": {
            "base_url": start["api_base_url"],
            "run_id": start["api_run_id"],
            "run_name": start["api_run_name"],
            "status": api_status,
        },
        "kubernetes": {
            "cluster": start["cluster"],
            "namespace": start["namespace"],
            "namespace_uid": start["namespace_uid"],
            "rayjob": {"name": rayjob_name, "uid": rayjob_uid, "status": "SUCCEEDED"},
            "workload": {
                "name": workload_name,
                "uid": workload_uid,
                "owner_rayjob_uid": rayjob_uid,
            },
            "raycluster": {
                "name": raycluster_name,
                "uid": raycluster_uid,
                "owner_rayjob_uid": rayjob_uid,
            },
            "pods": [
                {
                    "name": pod_name,
                    "uid": pod_uid,
                    "owner_raycluster_uid": raycluster_uid,
                    "phase": pod["phase"],
                    "exit_code": pod["exit_code"],
                    "termination_reason": pod["termination_reason"],
                    "runtime_image_id": pod["runtime_image_id"],
                    "container_restarts": pod["container_restarts"],
                    "gpus": pod["gpus"],
                }
            ],
        },
        "execution": {
            "requested_image": plan["execution"]["image"],
            "priority_class": "c1",
            "effective_priority": 10000,
            "automatic_requeue": False,
            "workers": 1,
            "gpus_per_worker": WORLD_SIZE,
            "total_gpus": WORLD_SIZE,
        },
        "event_journal": {
            "intent": intent_ref,
            "start": _reference(start_path, start, start_file_sha256),
            "events": [
                _reference(path, value, file_sha256)
                for path, (value, file_sha256) in zip(event_paths, event_records, strict=True)
            ],
        },
        "observed_at": terminal_observed,
        "private_logs_included": False,
        "metric_values_included": False,
        "task_content_included": False,
    }


def validate_controller_observation(
    value: dict[str, Any],
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    check_files: bool = True,
) -> None:
    from .miles_acceptance import NAMESPACE, NAMESPACE_UID, _image_digest, _time, _uuid

    sealed(value, CONTROLLER_SCHEMA)
    submitted = validate_submission_binding(submission, plan, check_files=False)
    api, kube, execution = value.get("api"), value.get("kubernetes"), value.get("execution")
    journal = value.get("event_journal")
    pods = kube.get("pods") if isinstance(kube, dict) else None
    rayjob = kube.get("rayjob") if isinstance(kube, dict) else None
    workload = kube.get("workload") if isinstance(kube, dict) else None
    raycluster = kube.get("raycluster") if isinstance(kube, dict) else None
    expected_image = plan["execution"]["image"].rsplit("@sha256:", 1)[1]
    if (
        set(value)
        != {
            "schema",
            "status",
            "observer_plan_sha256",
            "observer_request_sha256",
            "api",
            "kubernetes",
            "execution",
            "event_journal",
            "observed_at",
            "private_logs_included",
            "metric_values_included",
            "task_content_included",
            "sha256",
        }
        or value.get("status") != "succeeded"
        or value.get("observer_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("observer_request_sha256") != submitted["request_sha256"]
        or not isinstance(api, dict)
        or api
        != {
            "base_url": submitted["api"]["base_url"],
            "run_id": submitted["api"]["run_id"],
            "run_name": submitted["api"]["run_name"],
            "status": "SUCCEEDED",
        }
        or not isinstance(kube, dict)
        or set(kube)
        != {"cluster", "namespace", "namespace_uid", "rayjob", "workload", "raycluster", "pods"}
        or kube.get("cluster") != "dev"
        or kube.get("namespace") != NAMESPACE
        or kube.get("namespace_uid") != NAMESPACE_UID
        or not isinstance(rayjob, dict)
        or set(rayjob) != {"name", "uid", "status"}
        or rayjob.get("name") != submitted["api"]["run_name"]
        or _uuid(rayjob.get("uid"), "observer RayJob UID") != rayjob.get("uid")
        or rayjob.get("status") != "SUCCEEDED"
        or not isinstance(workload, dict)
        or set(workload) != {"name", "uid", "owner_rayjob_uid"}
        or _uuid(workload.get("uid"), "observer Workload UID") != workload.get("uid")
        or workload.get("owner_rayjob_uid") != rayjob.get("uid")
        or not isinstance(raycluster, dict)
        or set(raycluster) != {"name", "uid", "owner_rayjob_uid"}
        or _uuid(raycluster.get("uid"), "observer RayCluster UID") != raycluster.get("uid")
        or raycluster.get("owner_rayjob_uid") != rayjob.get("uid")
        or not isinstance(pods, list)
        or len(pods) != 1
        or execution
        != {
            "requested_image": plan["execution"]["image"],
            "priority_class": "c1",
            "effective_priority": 10000,
            "automatic_requeue": False,
            "workers": 1,
            "gpus_per_worker": WORLD_SIZE,
            "total_gpus": WORLD_SIZE,
        }
        or not isinstance(journal, dict)
        or set(journal) != {"intent", "start", "events"}
        or not isinstance(journal.get("intent"), dict)
        or set(journal["intent"]) != _REFERENCE_FIELDS
        or not isinstance(journal.get("start"), dict)
        or set(journal["start"]) != _REFERENCE_FIELDS
        or not isinstance(journal.get("events"), list)
        or not journal["events"]
        or not all(
            isinstance(reference, dict) and set(reference) == _REFERENCE_FIELDS
            for reference in journal["events"]
        )
        or value.get("private_logs_included") is not False
        or value.get("metric_values_included") is not False
        or value.get("task_content_included") is not False
    ):
        raise ValueError("policy observer controller evidence is incomplete")
    pod = pods[0]
    if (
        not isinstance(pod, dict)
        or set(pod)
        != {
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
        or _uuid(pod.get("uid"), "observer Pod UID") != pod.get("uid")
        or pod.get("owner_raycluster_uid") != raycluster["uid"]
        or pod.get("phase") != "Succeeded"
        or pod.get("exit_code") != 0
        or pod.get("termination_reason") != "Completed"
        or _image_digest(pod.get("runtime_image_id")) != expected_image
        or pod.get("container_restarts") != 0
        or pod.get("gpus") != WORLD_SIZE
    ):
        raise ValueError("policy observer Pod evidence is incomplete")
    _time(value.get("observed_at"), "policy observer controller observation")
    if check_files:
        start_path = Path(journal["start"]["path"])
        if start_path.name != "STARTED.json":
            raise ValueError("policy observer event-journal start path changed")
        expected = _controller_body(
            plan,
            submission,
            directory=start_path.parent,
            api_status="SUCCEEDED",
            observed_at=value["observed_at"],
        )
        if {key: item for key, item in value.items() if key != "sha256"} != expected:
            raise ValueError("policy observer controller differs from its event journal")


def compile_controller(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    directory: Path,
    api_status: str,
    observed_at: object,
    output: Path,
) -> dict[str, Any]:
    body = _controller_body(
        plan,
        submission,
        directory=directory,
        api_status=api_status,
        observed_at=observed_at,
    )
    validate_controller_observation({**body, "sha256": digest(body)}, plan, submission)
    return _write(output, body)


def collect_release_query(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    controller_path: Path,
    jobs: Any,
    output: Path,
    run_command: Any = subprocess.run,
    clock: Any = time.time,
) -> dict[str, Any]:
    """Read exact dev/API identities after TTL cleanup; never trust booleans."""
    from . import miles_event_evidence as events
    from .miles_acceptance import NAMESPACE, NAMESPACE_UID

    controller, _ = _read(controller_path, CONTROLLER_SCHEMA)
    validate_controller_observation(controller, plan, submission)
    submitted = validate_submission_binding(submission, plan, check_files=True)

    def get(resource: str, name: str, *, namespace: bool = True) -> dict[str, Any] | None:
        command = ["kubectl", "--context", events.DEV_KUBE_CONTEXT, "get", resource, name]
        if namespace:
            command.extend(("--namespace", NAMESPACE, "--ignore-not-found"))
        command.extend(("--output", "json"))
        process = run_command(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.returncode:
            raise ValueError("exact dev Kubernetes release query failed")
        payload = process.stdout
        if not payload:
            return None
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("exact dev Kubernetes release query returned invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("exact dev Kubernetes release query returned a non-object")
        return value

    namespace = get("namespace", NAMESPACE, namespace=False)
    if (namespace or {}).get("metadata", {}).get("uid") != NAMESPACE_UID:
        raise ValueError("release query is not bound to the exact dev3 namespace UID")
    kube = controller["kubernetes"]
    targets = {
        "rayjob": ("rayjobs.ray.io", kube["rayjob"]["name"], kube["rayjob"]["uid"]),
        "workload": (
            "workloads.kueue.x-k8s.io",
            kube["workload"]["name"],
            kube["workload"]["uid"],
        ),
        "raycluster": (
            "rayclusters.ray.io",
            kube["raycluster"]["name"],
            kube["raycluster"]["uid"],
        ),
        "pod": ("pods", kube["pods"][0]["name"], kube["pods"][0]["uid"]),
    }
    objects: dict[str, Any] = {}
    for key, (resource, name, uid) in targets.items():
        observed = get(resource, name)
        metadata = observed.get("metadata", {}) if observed else {}
        observed_uid = metadata.get("uid") if observed else None
        if observed and (
            metadata.get("namespace") != NAMESPACE
            or metadata.get("name") != name
            or observed_uid != uid
        ):
            raise ValueError("release query found a reused or mismatched lifecycle identity")
        objects[key] = {
            "resource": resource,
            "name": name,
            "expected_uid": uid,
            "present": observed is not None,
            "observed_uid": observed_uid,
        }
    api = jobs.status(submitted["api"]["run_name"])
    if (
        not isinstance(api, dict)
        or api.get("name") != submitted["api"]["run_name"]
        or api.get("job_id") != submitted["api"]["run_id"]
        or api.get("status") != "SUCCEEDED"
    ):
        raise ValueError("Jobs API does not report exact observer success at release query")
    if any(row["present"] for row in objects.values()):
        raise ValueError("observer lifecycle allocation is still present")
    return _write(
        output,
        {
            "schema": RELEASE_QUERY_SCHEMA,
            "status": "released",
            "observer_plan_sha256": "sha256:" + digest(plan),
            "observer_request_sha256": submitted["request_sha256"],
            "controller_observation_sha256": controller["sha256"],
            "api": {
                "base_url": submitted["api"]["base_url"],
                "run_id": api["job_id"],
                "run_name": api["name"],
                "status": api["status"],
            },
            "kubernetes": {
                "context": events.DEV_KUBE_CONTEXT,
                "namespace": NAMESPACE,
                "namespace_uid": NAMESPACE_UID,
                "objects": objects,
                "quota_reservation_present": False,
            },
            "active_gpus": 0,
            "observed_at": clock(),
            "private_logs_included": False,
            "metric_values_included": False,
            "task_content_included": False,
        },
    )


def validate_release_query(
    value: dict[str, Any],
    plan: dict[str, Any],
    submission: dict[str, Any],
    controller: dict[str, Any],
) -> None:
    from . import miles_event_evidence as events
    from .miles_acceptance import NAMESPACE, NAMESPACE_UID, _time

    sealed(value, RELEASE_QUERY_SCHEMA)
    submitted = validate_submission_binding(submission, plan, check_files=False)
    kube = value.get("kubernetes")
    objects = kube.get("objects") if isinstance(kube, dict) else None
    controller_kube = controller["kubernetes"]
    expected = {
        "rayjob": ("rayjobs.ray.io", controller_kube["rayjob"]),
        "workload": ("workloads.kueue.x-k8s.io", controller_kube["workload"]),
        "raycluster": ("rayclusters.ray.io", controller_kube["raycluster"]),
        "pod": ("pods", controller_kube["pods"][0]),
    }
    if (
        set(value) != _RELEASE_QUERY_FIELDS
        or value.get("status") != "released"
        or value.get("observer_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("observer_request_sha256") != submitted["request_sha256"]
        or value.get("controller_observation_sha256") != controller["sha256"]
        or value.get("api") != {**submitted["api"], "status": "SUCCEEDED"}
        or not isinstance(kube, dict)
        or set(kube)
        != {"context", "namespace", "namespace_uid", "objects", "quota_reservation_present"}
        or kube.get("context") != events.DEV_KUBE_CONTEXT
        or kube.get("namespace") != NAMESPACE
        or kube.get("namespace_uid") != NAMESPACE_UID
        or kube.get("quota_reservation_present") is not False
        or not isinstance(objects, dict)
        or set(objects) != set(expected)
        or value.get("active_gpus") != 0
        or value.get("private_logs_included") is not False
        or value.get("metric_values_included") is not False
        or value.get("task_content_included") is not False
        or _time(value.get("observed_at"), "policy observer release query")
        < _time(controller["observed_at"], "policy observer terminal evidence")
    ):
        raise ValueError("policy observer release query is incomplete")
    for key, (resource, identity) in expected.items():
        if objects[key] != {
            "resource": resource,
            "name": identity["name"],
            "expected_uid": identity["uid"],
            "present": False,
            "observed_uid": None,
        }:
            raise ValueError("policy observer release query did not prove exact absence")


def validate_release_observation(
    value: dict[str, Any],
    plan: dict[str, Any],
    submission: dict[str, Any],
    controller: dict[str, Any],
    controller_file_sha256: str,
) -> None:
    from .miles_acceptance import _time

    sealed(value, RELEASE_SCHEMA)
    validate_controller_observation(controller, plan, submission)
    submitted = validate_submission_binding(submission, plan, check_files=False)
    kube, api = controller["kubernetes"], controller["api"]
    if (
        set(value)
        != {
            "schema",
            "status",
            "observer_plan_sha256",
            "observer_request_sha256",
            "controller_observation_sha256",
            "controller_observation_file_sha256",
            "release_query",
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
        or value.get("observer_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("observer_request_sha256") != submitted["request_sha256"]
        or value.get("controller_observation_sha256") != controller["sha256"]
        or value.get("controller_observation_file_sha256", "").removeprefix("sha256:")
        != controller_file_sha256.removeprefix("sha256:")
        or not isinstance(value.get("release_query"), dict)
        or set(value["release_query"]) != _REFERENCE_FIELDS
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or value.get("identities")
        != {
            "api_run_id": api["run_id"],
            "api_run_name": api["run_name"],
            "rayjob_uid": kube["rayjob"]["uid"],
            "workload_uid": kube["workload"]["uid"],
            "raycluster_uid": kube["raycluster"]["uid"],
            "pod_uids": [kube["pods"][0]["uid"]],
        }
        or value.get("raycluster_present") is not False
        or value.get("rayjob_present") is not False
        or value.get("workload_present") is not False
        or value.get("quota_reservation_present") is not False
        or value.get("gpu_pods_present") is not False
        or value.get("active_gpus") != 0
        or value.get("gpu_release_proven") is not True
        or _time(value.get("observed_at"), "policy observer release")
        < _time(controller["observed_at"], "policy observer terminal evidence")
    ):
        raise ValueError("policy observer external release is incomplete")
    query, query_file_sha256 = _read(Path(value["release_query"]["path"]), RELEASE_QUERY_SCHEMA)
    if query_file_sha256 != value["release_query"]["file_sha256"].removeprefix("sha256:") or query[
        "sha256"
    ].removeprefix("sha256:") != value["release_query"]["receipt_sha256"].removeprefix("sha256:"):
        raise ValueError("policy observer release query reference changed")
    validate_release_query(query, plan, submission, controller)


def compile_release(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    controller_path: Path,
    release_query_path: Path,
    output: Path,
) -> dict[str, Any]:
    controller, controller_file_sha256 = _read(controller_path, CONTROLLER_SCHEMA)
    query, query_file_sha256 = _read(release_query_path, RELEASE_QUERY_SCHEMA)
    validate_release_query(query, plan, submission, controller)
    kube, api = controller["kubernetes"], controller["api"]
    body = {
        "schema": RELEASE_SCHEMA,
        "status": "released",
        "observer_plan_sha256": "sha256:" + digest(plan),
        "observer_request_sha256": validate_submission_binding(submission, plan, check_files=False)[
            "request_sha256"
        ],
        "controller_observation_sha256": controller["sha256"],
        "controller_observation_file_sha256": "sha256:" + controller_file_sha256,
        "release_query": _reference(release_query_path, query, query_file_sha256),
        "api_status": "SUCCEEDED",
        "controller_status": "SUCCEEDED",
        "identities": {
            "api_run_id": api["run_id"],
            "api_run_name": api["run_name"],
            "rayjob_uid": kube["rayjob"]["uid"],
            "workload_uid": kube["workload"]["uid"],
            "raycluster_uid": kube["raycluster"]["uid"],
            "pod_uids": [kube["pods"][0]["uid"]],
        },
        "raycluster_present": False,
        "rayjob_present": False,
        "workload_present": False,
        "quota_reservation_present": False,
        "gpu_pods_present": False,
        "active_gpus": 0,
        "gpu_release_proven": True,
        "observed_at": query["observed_at"],
    }
    validate_release_observation(
        {**body, "sha256": digest(body)},
        plan,
        submission,
        controller,
        controller_file_sha256,
    )
    return _write(output, body)


def _parse_args(source: miles.MilesConfig, checkpoint_root: str, *, trained: bool):
    from miles.utils.arguments import parse_args

    argv = reload_arguments(source, checkpoint_root)
    if not trained:
        argv = [token for token in argv if token != "--use-checkpoint-opt-param-scheduler"]
        argv.extend(("--no-load-optim", "--no-load-rng", "--finetune"))
    previous = sys.argv
    try:
        sys.argv = [str(native_source()), *argv]
        args = parse_args()
    finally:
        sys.argv = previous
    expected = {
        "load": checkpoint_root,
        "save": None,
        "debug_train_only": True,
        "no_load_optim": not trained,
        "no_load_rng": not trained,
        "finetune": not trained,
        "use_checkpoint_opt_param_scheduler": trained,
        "use_wandb": False,
        "colocate": False,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": WORLD_SIZE,
    }
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError("native parser changed policy observer load semantics")
    return args


def native_args(plan: dict[str, Any], *, trained: bool):
    _validate_plan(plan, check_files=False, require_current_runtime=True)
    source = miles.MilesConfig(**plan["source_plan"]["arguments"])
    root = (
        plan["trained_checkpoint"]["root"] if trained else plan["source_plan"]["checkpoint"]["root"]
    )
    return _parse_args(source, root, trained=trained)


def _state_probe(model: Any, optimizer: Any, scheduler: Any) -> dict[str, Any]:
    return {
        "model": _model_probe(model),
        "optimizer": _optimizer_probe(optimizer),
        "scheduler": _scheduler_probe(scheduler),
        "rng_sha256": _rng_probe(),
    }


def _fill_tensors(value: Any, scalar: float, seen: set[int] | None = None) -> int:
    """Overwrite tensor leaves in an optimizer state without changing its shape."""
    import torch

    seen = seen if seen is not None else set()
    if torch.is_tensor(value):
        with torch.no_grad():
            value.fill_(scalar)
        return 1
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return 0
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, Mapping):
        return sum(_fill_tensors(item, scalar, seen) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_fill_tensors(item, scalar, seen) for item in value)
    return 0


def _optimizer_sentinel(optimizer: Any, marker: int) -> None:
    pending, seen, groups = [optimizer], set(), 0
    scalar = marker / 1000.0
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        _fill_tensors(getattr(current, "state", None), scalar)
        param_groups = getattr(current, "param_groups", None)
        if isinstance(param_groups, Sequence):
            for group in param_groups:
                if isinstance(group, dict) and "lr" in group:
                    group["lr"] = scalar
                    groups += 1
        for attr in ("optimizer", "optimizers", "chained_optimizers"):
            child = getattr(current, attr, None)
            if isinstance(child, Sequence):
                pending.extend(child)
            elif child is not None:
                pending.append(child)
    if groups < 1:
        raise ValueError("optimizer exposes no mutable parameter-group sentinel")


def _scheduler_sentinel(scheduler: Any, marker: int) -> None:
    state = copy.deepcopy(scheduler.state_dict())
    changed = 0

    def replace(value: Any, key: str = "") -> Any:
        nonlocal changed
        if isinstance(value, Mapping):
            return {name: replace(item, str(name)) for name, item in value.items()}
        if isinstance(value, list):
            return [replace(item, key) for item in value]
        if (
            isinstance(value, numbers.Real)
            and not isinstance(value, bool)
            and any(token in key.lower() for token in ("step", "sample", "consum"))
        ):
            changed += 1
            return type(value)(marker)
        return value

    scheduler.load_state_dict(replace(state))
    if changed < 1:
        raise ValueError("scheduler exposes no progress-counter sentinel")


def _rng_sentinel(marker: int) -> None:
    import torch

    random.seed(marker)
    with suppress(ImportError):
        import numpy

        numpy.random.seed(marker)
    torch.manual_seed(marker)
    torch.cuda.manual_seed_all(marker)


def _install_sentinel(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    marker: int,
    *,
    full_state: bool,
) -> dict[str, Any]:
    import torch

    if full_state:
        _scheduler_sentinel(scheduler, marker)
        _optimizer_sentinel(optimizer, marker)
    with torch.no_grad():
        tensors = 0
        for chunk in model:
            for parameter in chunk.parameters():
                parameter.fill_(marker / 1000.0)
                tensors += 1
    if tensors < 1:
        raise ValueError("model exposes no policy tensor sentinel")
    if full_state:
        _rng_sentinel(marker)
    return _state_probe(model, optimizer, scheduler)


def _instrumented_init(self, *args: Any, **kwargs: Any) -> Any:
    """Capture commitments immediately around Miles' actual checkpoint loader."""
    from miles.backends.megatron_utils import model as model_module

    marker = int(type(self)._cyber_sentinel_marker)
    original_load = model_module.load_checkpoint
    observations: list[dict[str, Any]] = []

    def observed_load(model: Any, optimizer: Any, scheduler: Any, *a: Any, **kw: Any) -> Any:
        if observations:
            raise ValueError("Miles invoked checkpoint load more than once")
        preload = _install_sentinel(
            model,
            optimizer,
            scheduler,
            marker,
            full_state=marker != BASE_SENTINEL,
        )
        result = original_load(model, optimizer, scheduler, *a, **kw)
        observations.append(
            {
                "marker": marker,
                "load_calls": 1,
                "preload": preload,
                "restored": _state_probe(model, optimizer, scheduler),
            }
        )
        return result

    model_module.load_checkpoint = observed_load
    try:
        result = super(type(self), self).init(*args, **kwargs)
    finally:
        model_module.load_checkpoint = original_load
    if len(observations) != 1:
        raise ValueError("Miles did not invoke exactly one instrumented checkpoint load")
    self._cyber_restore_observation = observations[0]
    return result


def _native_prediction_forward(model: Any, tokens: Any, packed: Any) -> Any:
    """Call the bound native GPTModel signature without wrapper-only options."""
    return model(
        input_ids=tokens,
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=packed,
        loss_mask=None,
    )


def _prediction_probe(self) -> dict[str, Any]:
    """One fixed, task-free greedy-order probe over the restored native policy."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.mappings import (
        gather_from_tensor_model_parallel_region,
    )
    from miles.backends.megatron_utils.parallel import get_packed_seq_params
    from miles.backends.training_utils.cp_utils import slice_with_cp

    if len(self.model) != 1:
        raise ValueError("fixed prediction probe requires one local model chunk")
    model = self.model[0]
    previous_training = bool(model.training)
    model.eval()
    full_tokens = torch.tensor(
        PREDICTION_INPUT_IDS,
        dtype=torch.long,
        device=torch.cuda.current_device(),
    )
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    if cp_size != 2 or self.args.qkv_format != "thd":
        raise ValueError("fixed prediction probe requires exact CP2/THD topology")
    local_tokens = slice_with_cp(full_tokens, 0, self.args.qkv_format)
    tokens = local_tokens.unsqueeze(0)
    batch: dict[str, Any] = {
        "cu_seqlens": torch.tensor(
            [0, len(PREDICTION_INPUT_IDS)],
            dtype=torch.int,
            device=torch.cuda.current_device(),
        ),
        "max_seqlen": len(PREDICTION_INPUT_IDS),
    }
    packed = get_packed_seq_params(batch, self.args)
    try:
        with torch.no_grad():
            output = _native_prediction_forward(model, tokens, packed)
            if not torch.is_tensor(output) or output.ndim != 3:
                raise ValueError("native fixed prediction probe returned an unexpected shape")
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                output = gather_from_tensor_model_parallel_region(output)
            if output.shape[0] != 1 or output.shape[1] != local_tokens.numel():
                raise ValueError("native fixed prediction probe batch/sequence shape changed")
            vocabulary_size = int(self.hf_config.vocab_size)
            if not PREDICTION_TOP_K < vocabulary_size <= output.shape[-1]:
                raise ValueError("native fixed prediction probe vocabulary changed")
            prediction_ids = torch.full(
                (PREDICTION_TOP_K,),
                -1,
                dtype=torch.long,
                device=output.device,
            )
            margin = torch.zeros(1, dtype=torch.float32, device=output.device)
            if cp_rank == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
                if dist.get_rank() != 0:
                    raise ValueError("fixed prediction probe source-rank mapping changed")
                # CP2 uses zig-zag chunks.  For an exact 32-token sequence, CP rank
                # zero owns tokens 0..7 and 24..31, so local index 15 is the
                # next-token prediction corresponding to the full sequence.
                target = len(PREDICTION_INPUT_IDS) // cp_size - 1
                values, indices = torch.topk(
                    output[0, target, :vocabulary_size].float(),
                    k=PREDICTION_TOP_K + 1,
                    largest=True,
                    sorted=True,
                )
                prediction_ids.copy_(indices[:PREDICTION_TOP_K])
                margin[0] = values[PREDICTION_TOP_K - 1] - values[PREDICTION_TOP_K]
            dist.broadcast(prediction_ids, src=0)
            dist.broadcast(margin, src=0)
            margin_satisfied = bool(margin.item() >= PREDICTION_MARGIN)
            predictions = [int(item) for item in prediction_ids.cpu().tolist()]
    finally:
        model.train(previous_training)
    if not margin_satisfied:
        raise ValueError("native fixed prediction top-k boundary is not numerically robust")
    return {
        "schema": PREDICTION_SCHEMA,
        "probe_id": PREDICTION_PROBE_ID,
        "input_ids_sha256": "sha256:" + digest(list(PREDICTION_INPUT_IDS)),
        "sequence_length": len(PREDICTION_INPUT_IDS),
        "top_k": PREDICTION_TOP_K,
        "selection_margin_threshold": PREDICTION_MARGIN,
        "selection_margin_satisfied": True,
        "prediction_sha256": "sha256:" + digest(predictions),
        "logits_included": False,
        "task_content_included": False,
        "benchmark_content_included": False,
    }


def validate_prediction_probe(value: object) -> dict[str, Any]:
    fields = {
        "schema",
        "probe_id",
        "input_ids_sha256",
        "sequence_length",
        "top_k",
        "selection_margin_threshold",
        "selection_margin_satisfied",
        "prediction_sha256",
        "logits_included",
        "task_content_included",
        "benchmark_content_included",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value.get("schema") != PREDICTION_SCHEMA
        or value.get("probe_id") != PREDICTION_PROBE_ID
        or value.get("input_ids_sha256") != "sha256:" + digest(list(PREDICTION_INPUT_IDS))
        or value.get("sequence_length") != len(PREDICTION_INPUT_IDS)
        or value.get("top_k") != PREDICTION_TOP_K
        or value.get("selection_margin_threshold") != PREDICTION_MARGIN
        or value.get("selection_margin_satisfied") is not True
        or _SHA.fullmatch(str(value.get("prediction_sha256"))) is None
        or value.get("logits_included") is not False
        or value.get("task_content_included") is not False
        or value.get("benchmark_content_included") is not False
    ):
        raise ValueError("fixed non-task prediction probe is incomplete")
    return value


def _restore_probe(self) -> dict[str, Any]:
    import torch.distributed as dist

    dist.barrier()
    first = _state_probe(self.model, self.optimizer, self.opt_param_scheduler)
    second = _state_probe(self.model, self.optimizer, self.opt_param_scheduler)
    if first != second:
        raise ValueError("state changed during the zero-update observer")
    prediction = None
    if self._cyber_restore_observation["marker"] == RELOAD_SENTINEL:
        prediction = _prediction_probe(self)
        if first != _state_probe(self.model, self.optimizer, self.opt_param_scheduler):
            raise ValueError("fixed prediction probe changed restored training state")
    observation = {**self._cyber_restore_observation, "live": first}
    result = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "load": observation,
        "prediction_probe": prediction,
    }
    dist.barrier()
    return result


def _remove_group(ray: Any, placement_group: Any) -> None:
    ray.util.remove_placement_group(placement_group)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = ray.util.placement_group_table(placement_group)
        if not state or state.get("state") == "REMOVED":
            return
        time.sleep(0.25)
    raise TimeoutError("Miles placement group did not release before the next restore")


async def _probe_group(args, marker: int) -> tuple[list[Any], list[dict[str, Any]]]:
    import ray
    from miles.backends.megatron_utils import actor as actor_module
    from miles.ray.placement_group import allocate_train_group, create_placement_groups

    original = actor_module.MegatronTrainRayActor
    actor = type(
        "CyberMilesPolicyObserverActor",
        (original,),
        {
            "init": _instrumented_init,
            "cyber_restore_probe": _restore_probe,
            "_cyber_sentinel_marker": marker,
        },
    )
    pgs = group = None
    try:
        actor_module.MegatronTrainRayActor = actor
        pgs = create_placement_groups(args)
        group = allocate_train_group(
            args=args,
            num_nodes=1,
            num_gpus_per_node=WORLD_SIZE,
            pg=pgs["actor"],
            role="actor",
            with_ref=False,
            rollout_manager=None,
        )
        actor_module.MegatronTrainRayActor = original
        start_ids = await group.init()
        rows = await group._broadcast("cyber_restore_probe")
        return list(start_ids), list(rows)
    finally:
        actor_module.MegatronTrainRayActor = original
        if group is not None:
            for handle in group._actor_handles:
                with suppress(Exception):
                    ray.kill(handle, no_restart=True)
        if pgs is not None and pgs["actor"][0] is not None:
            _remove_group(ray, pgs["actor"][0])


def _rank_rows(
    base: Sequence[dict[str, Any]],
    trained_reference: Sequence[dict[str, Any]],
    trained_reload: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if any(len(rows) != WORLD_SIZE for rows in (base, trained_reference, trained_reload)):
        raise ValueError("policy observer did not receive all ranks")
    base = sorted(base, key=lambda row: row["rank"])
    trained_reference = sorted(trained_reference, key=lambda row: row["rank"])
    trained_reload = sorted(trained_reload, key=lambda row: row["rank"])
    if any(
        [row["rank"] for row in rows] != list(range(WORLD_SIZE))
        for rows in (base, trained_reference, trained_reload)
    ):
        raise ValueError("policy observer rank identity mismatch")
    result = []
    predictions = []
    for rank, (before, reference, reloaded) in enumerate(
        zip(base, trained_reference, trained_reload, strict=True)
    ):
        base_load, reference_load, reload_load = (
            before["load"],
            reference["load"],
            reloaded["load"],
        )
        if (
            before.get("prediction_probe") is not None
            or reference.get("prediction_probe") is not None
        ):
            raise ValueError("fixed prediction probe ran on more than one restore")
        predictions.append(validate_prediction_probe(reloaded.get("prediction_probe")))
        base_model = base_load["restored"]["model"]
        trained_model = reference_load["restored"]["model"]
        if (
            before["world_size"] != WORLD_SIZE
            or reference["world_size"] != WORLD_SIZE
            or reloaded["world_size"] != WORLD_SIZE
            or base_model["tensors"] < 1
            or base_model["local_numel"] < 1
            or trained_model["tensors"] != base_model["tensors"]
            or trained_model["local_numel"] != base_model["local_numel"]
            or trained_model["structure_sha256"] != base_model["structure_sha256"]
            or reference_load["restored"]["optimizer"]["state_entries"] < 1
            or reference_load["restored"]["optimizer"]["parameter_groups"] < 1
            or reference_load["restored"]["scheduler"]["positive_progress_counters"] < 1
        ):
            raise ValueError("policy observer found incomplete or structurally different state")
        changed = base_model["value_sha256"] != trained_model["value_sha256"]
        result.append(
            {
                "rank": rank,
                "policy_tensor_count": base_model["tensors"],
                "local_policy_numel": base_model["local_numel"],
                "base": base_load,
                "trained_reference": reference_load,
                "trained_reload": reload_load,
                "policy_changed": changed,
            }
        )
    if any(value != predictions[0] for value in predictions[1:]):
        raise ValueError("fixed prediction probe differs across native ranks")
    return result, predictions[0]


def validate_result(plan: dict[str, Any], value: dict[str, Any]) -> list[int]:
    _validate_plan(plan, check_files=False, require_current_runtime=False)
    sealed(value, RESULT_SCHEMA)
    rows = value.get("ranks")
    if (
        set(value) != _RESULT_FIELDS
        or value.get("status") != "observed"
        or value.get("observer_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("source_plan_sha256") != plan["source_plan_sha256"]
        or value.get("base_checkpoint_receipt_sha256") != plan["base_checkpoint_receipt_sha256"]
        or value.get("trained_checkpoint_receipt_sha256") != plan["trained_checkpoint"]["sha256"]
        or value.get("world_size") != WORLD_SIZE
        or value.get("comparison_method") != COMPARISON_METHOD
        or value.get("restored_next_rollout_id") != plan["trained_checkpoint"]["next_rollout_id"]
        or value.get("checkpoint_state_commitment_method") != RANK_STATE_COMMITMENT_METHOD
        or value.get("restore_method") != RESTORE_METHOD
        or value.get("trained_restore_count") != 2
        or validate_prediction_probe(value.get("prediction_probe")) != value.get("prediction_probe")
        or value.get("state_stable_across_zero_updates") is not True
        or value.get("work_executed") != _OBSERVER_WORK
        or value.get("base_checkpoint_unchanged") is not True
        or value.get("trained_checkpoint_unchanged") is not True
        or _time(value.get("completed_at"), "policy observer completion") <= 0
        or value.get("reward_values_included") is not False
        or value.get("task_content_included") is not False
        or value.get("tensor_values_included") is not False
        or not isinstance(rows, list)
        or len(rows) != WORLD_SIZE
    ):
        raise ValueError("policy observer result is incomplete or mismatched")

    def state(commitment: object, label: str) -> dict[str, Any]:
        if not isinstance(commitment, dict) or set(commitment) != _STATE_FIELDS:
            raise ValueError(f"{label} state fields changed")
        model = commitment.get("model")
        optimizer = commitment.get("optimizer")
        scheduler = commitment.get("scheduler")
        if (
            not isinstance(model, dict)
            or set(model) != {"tensors", "local_numel", "structure_sha256", "value_sha256"}
            or type(model.get("tensors")) is not int
            or model["tensors"] < 1
            or type(model.get("local_numel")) is not int
            or model["local_numel"] < 1
            or not isinstance(optimizer, dict)
            or set(optimizer)
            != {
                "objects",
                "state_entries",
                "parameter_groups",
                "structure_sha256",
                "value_sha256",
            }
            or any(
                type(optimizer.get(key)) is not int or optimizer[key] < 0
                for key in ("objects", "state_entries", "parameter_groups")
            )
            or not isinstance(scheduler, dict)
            or set(scheduler) != {"positive_progress_counters", "structure_sha256", "value_sha256"}
            or type(scheduler.get("positive_progress_counters")) is not int
            or scheduler["positive_progress_counters"] < 0
            or any(
                _SHA.fullmatch(str(item)) is None
                for item in (
                    model.get("structure_sha256"),
                    model.get("value_sha256"),
                    optimizer.get("structure_sha256"),
                    optimizer.get("value_sha256"),
                    scheduler.get("structure_sha256"),
                    scheduler.get("value_sha256"),
                    commitment.get("rng_sha256"),
                )
            )
        ):
            raise ValueError(f"{label} state commitment is incomplete")
        return commitment

    def load(value: object, marker: int, label: str, *, all_components: bool) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or set(value) != _LOAD_FIELDS
            or value.get("marker") != marker
            or value.get("load_calls") != 1
        ):
            raise ValueError(f"{label} load evidence is incomplete")
        preload = state(value.get("preload"), label + " preload")
        restored = state(value.get("restored"), label + " restored")
        live = state(value.get("live"), label + " live")
        keys = ("model", "optimizer", "scheduler") if all_components else ("model",)
        if any(preload[key]["value_sha256"] == restored[key]["value_sha256"] for key in keys):
            raise ValueError(f"{label} did not overwrite its preload sentinel")
        if all_components and preload["rng_sha256"] == restored["rng_sha256"]:
            raise ValueError(f"{label} did not restore RNG over its preload sentinel")
        if live != restored:
            raise ValueError(f"{label} changed after its zero-update restore")
        if not all_components and (
            any(preload[key] != restored[key] for key in ("optimizer", "scheduler"))
            or preload["rng_sha256"] != restored["rng_sha256"]
        ):
            raise ValueError("base no-load state changed outside the policy checkpoint")
        return value

    def values(commitment: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            commitment["model"]["value_sha256"],
            commitment["optimizer"]["value_sha256"],
            commitment["scheduler"]["value_sha256"],
            commitment["rng_sha256"],
        )

    changed = []
    for rank, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != _RANK_FIELDS
            or row.get("rank") != rank
            or type(row.get("policy_tensor_count")) is not int
            or row["policy_tensor_count"] < 1
            or type(row.get("local_policy_numel")) is not int
            or row["local_policy_numel"] < 1
        ):
            raise ValueError("policy observer rank result is invalid")
        base = load(row["base"], BASE_SENTINEL, "base", all_components=False)
        reference = load(
            row["trained_reference"],
            REFERENCE_SENTINEL,
            "trained reference",
            all_components=True,
        )
        reloaded = load(
            row["trained_reload"],
            RELOAD_SENTINEL,
            "trained reload",
            all_components=True,
        )
        base_restored = base["restored"]
        reference_preload = reference["preload"]
        reference_restored = reference["restored"]
        reload_preload = reloaded["preload"]
        reload_restored = reloaded["restored"]
        preload_values = (
            values(base["preload"]),
            values(reference_preload),
            values(reload_preload),
        )
        if (
            any(
                first[index] == second[index]
                for first, second in (
                    (preload_values[0], preload_values[1]),
                    (preload_values[0], preload_values[2]),
                    (preload_values[1], preload_values[2]),
                )
                for index in range(len(first))
            )
            or reference_restored != reload_restored
            or base_restored["model"]["structure_sha256"]
            != reference_restored["model"]["structure_sha256"]
            or reference_restored["model"]["tensors"] != row["policy_tensor_count"]
            or reference_restored["model"]["local_numel"] != row["local_policy_numel"]
            or reference_restored["optimizer"]["state_entries"] < 1
            or reference_restored["optimizer"]["parameter_groups"] < 1
            or reference_restored["scheduler"]["positive_progress_counters"] < 1
            # This exact one-update recipe guarantees optimizer and scheduler
            # progress.  A legitimate no/partial policy delta is still an
            # observed scientific result and is rejected only by promotion.
            # The no-dropout recipe also permits byte-identical RNG state; RNG
            # is still sentinel-overwrite checked and independently reproduced.
            or any(
                values(base_restored)[index] == values(reference_restored)[index]
                for index in (1, 2)
            )
        ):
            raise ValueError("policy observer restore commitments are circular or unchanged")
        policy_changed = (
            base_restored["model"]["value_sha256"] != reference_restored["model"]["value_sha256"]
        )
        if row.get("policy_changed") is not policy_changed:
            raise ValueError("policy observer policy-delta marker is invalid")
        if row["policy_changed"]:
            changed.append(rank)
    if value.get("changed_policy_ranks") != changed:
        raise ValueError("policy observer changed-rank summary mismatch")
    return changed


def _policy_body(
    plan: dict[str, Any],
    *,
    submission_path: Path,
    result_path: Path,
    controller_path: Path,
    release_path: Path,
) -> dict[str, Any]:
    """Reopen one complete observer run and derive its public policy proof."""
    from .miles_acceptance import _time

    _validate_plan(plan, check_files=True, require_current_runtime=False)
    submission, submission_file_sha256 = _read(submission_path, SUBMISSION_SCHEMA)
    validate_submission_binding(submission, plan, check_files=True)
    result, result_file_sha256 = _read(result_path, RESULT_SCHEMA)
    changed = validate_result(plan, result)
    if changed != list(range(WORLD_SIZE)):
        raise ValueError(
            "trained checkpoint lacks an all-rank independently observed policy tensor delta"
        )
    controller, controller_file_sha256 = _read(controller_path, CONTROLLER_SCHEMA)
    validate_controller_observation(controller, plan, submission)
    release, release_file_sha256 = _read(release_path, RELEASE_SCHEMA)
    validate_release_observation(
        release,
        plan,
        submission,
        controller,
        controller_file_sha256,
    )
    if (
        Path(result_path) != Path(plan["output_root"]) / "POLICY_OBSERVER_RESULT.json"
        or Path(submission["observer_plan_path"]) == Path(plan["source_plan_path"])
        or _time(controller["observed_at"], "policy observer controller")
        < _time(result["completed_at"], "policy observer completion")
    ):
        raise ValueError("policy observer result or execution identity is not independent")
    from . import miles_acceptance

    source_submission, _ = _read(
        Path(plan["source_submission"]["path"]),
        miles_acceptance.SUBMISSION_SCHEMA,
    )
    miles_acceptance.validate_submission_binding(
        source_submission,
        plan["source_plan"],
        check_files=True,
    )
    if submission["api"]["run_id"] == source_submission["api"]["run_id"]:
        raise ValueError("policy observer reused the source training API run")
    return {
        "schema": POLICY_SCHEMA,
        "source_plan_sha256": plan["source_plan_sha256"],
        "base_checkpoint_receipt_sha256": plan["base_checkpoint_receipt_sha256"],
        "trained_checkpoint_receipt_sha256": plan["trained_checkpoint"]["sha256"],
        "observer_plan_sha256": "sha256:" + digest(plan),
        "observer_submission": _reference(submission_path, submission, submission_file_sha256),
        "observer_result": _reference(result_path, result, result_file_sha256),
        "observer_controller": _reference(controller_path, controller, controller_file_sha256),
        "observer_release": _reference(release_path, release, release_file_sha256),
        "world_size": WORLD_SIZE,
        "comparison_method": COMPARISON_METHOD,
        "prediction_probe": result["prediction_probe"],
        "ranks": result["ranks"],
        "changed_policy_ranks": changed,
        "policy_structure_matches": True,
        "optimizer_state_used_for_delta": False,
        "scheduler_state_used_for_delta": False,
        "rng_state_used_for_delta": False,
        "metadata_used_for_delta": False,
        "checkpoint_state_commitment_method": RANK_STATE_COMMITMENT_METHOD,
        "observer_work": dict(_OBSERVER_WORK),
        "source_checkpoints_unchanged": True,
        "observer_external_gpu_release_verified": True,
        "reward_values_included": False,
        "task_content_included": False,
        "tensor_values_included": False,
    }


def accept_policy_delta(
    plan: dict[str, Any],
    *,
    submission_path: Path,
    result_path: Path,
    controller_path: Path,
    release_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Create the only policy-delta receipt consumed by Miles acceptance."""
    expected = Path(plan["output_root"]) / "POLICY_DELTA.json"
    if output != expected:
        raise ValueError("policy-delta output is not bound to the observer run")
    if output.exists() or output.is_symlink():
        raise FileExistsError("policy-delta acceptance already exists")
    body = _policy_body(
        plan,
        submission_path=submission_path,
        result_path=result_path,
        controller_path=controller_path,
        release_path=release_path,
    )
    validate_policy_evidence(
        {**body, "sha256": digest(body)},
        plan["source_plan"],
        plan["trained_checkpoint"],
    )
    return _write(output, body)


def validate_policy_evidence(
    value: dict[str, Any], source_plan: dict[str, Any], checkpoint: dict[str, Any]
) -> int:
    """Deeply reopen observer provenance for terminal acceptance and reload."""
    sealed(value, POLICY_SCHEMA)
    references = tuple(
        value.get(key)
        for key in (
            "observer_submission",
            "observer_result",
            "observer_controller",
            "observer_release",
        )
    )
    expected_fields = {
        "schema",
        "source_plan_sha256",
        "base_checkpoint_receipt_sha256",
        "trained_checkpoint_receipt_sha256",
        "observer_plan_sha256",
        "observer_submission",
        "observer_result",
        "observer_controller",
        "observer_release",
        "world_size",
        "comparison_method",
        "ranks",
        "prediction_probe",
        "changed_policy_ranks",
        "policy_structure_matches",
        "optimizer_state_used_for_delta",
        "scheduler_state_used_for_delta",
        "rng_state_used_for_delta",
        "metadata_used_for_delta",
        "checkpoint_state_commitment_method",
        "observer_work",
        "source_checkpoints_unchanged",
        "observer_external_gpu_release_verified",
        "reward_values_included",
        "task_content_included",
        "tensor_values_included",
        "sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("source_plan_sha256", "").removeprefix("sha256:") != digest(source_plan)
        or value.get("base_checkpoint_receipt_sha256")
        != source_plan.get("checkpoint", {}).get("sha256")
        or value.get("trained_checkpoint_receipt_sha256") != checkpoint.get("sha256")
        or value.get("world_size") != WORLD_SIZE
        or value.get("comparison_method") != COMPARISON_METHOD
        or validate_prediction_probe(value.get("prediction_probe")) != value.get("prediction_probe")
        or value.get("policy_structure_matches") is not True
        or value.get("optimizer_state_used_for_delta") is not False
        or value.get("scheduler_state_used_for_delta") is not False
        or value.get("rng_state_used_for_delta") is not False
        or value.get("metadata_used_for_delta") is not False
        or value.get("checkpoint_state_commitment_method") != RANK_STATE_COMMITMENT_METHOD
        or value.get("observer_work") != _OBSERVER_WORK
        or value.get("source_checkpoints_unchanged") is not True
        or value.get("observer_external_gpu_release_verified") is not True
        or value.get("reward_values_included") is not False
        or value.get("task_content_included") is not False
        or value.get("tensor_values_included") is not False
        or not all(
            isinstance(reference, dict) and set(reference) == _REFERENCE_FIELDS
            for reference in references
        )
    ):
        raise ValueError("policy delta is not bound to a complete independent observer")
    submission_ref = value["observer_submission"]
    submission, submission_file_sha256 = _read(Path(submission_ref["path"]), SUBMISSION_SCHEMA)
    plan_path = Path(submission["observer_plan_path"])
    from .miles_acceptance import _json_snapshot

    observer_plan, observer_plan_file_sha256 = _json_snapshot(plan_path)
    if not isinstance(observer_plan, dict):
        raise ValueError("policy observer plan is not a JSON object")
    _validate_plan(observer_plan, check_files=True, require_current_runtime=False)
    if (
        observer_plan["source_plan"] != source_plan
        or observer_plan["trained_checkpoint"] != checkpoint
        or value["observer_plan_sha256"].removeprefix("sha256:") != digest(observer_plan)
        or submission["observer_plan_file_sha256"].removeprefix("sha256:")
        != observer_plan_file_sha256
        or submission_file_sha256 != submission_ref["file_sha256"].removeprefix("sha256:")
        or submission["sha256"].removeprefix("sha256:")
        != submission_ref["receipt_sha256"].removeprefix("sha256:")
    ):
        raise ValueError("policy observer plan or submission reference changed")
    validate_submission_binding(submission, observer_plan, check_files=True)
    expected = _policy_body(
        observer_plan,
        submission_path=Path(submission_ref["path"]),
        result_path=Path(value["observer_result"]["path"]),
        controller_path=Path(value["observer_controller"]["path"]),
        release_path=Path(value["observer_release"]["path"]),
    )
    if {key: item for key, item in value.items() if key != "sha256"} != expected:
        raise ValueError("policy delta differs from rederived observer evidence")
    return len(value["changed_policy_ranks"])


def _reopen_reference(reference: Mapping[str, Any], schema: str) -> tuple[dict[str, Any], str]:
    if set(reference) != _REFERENCE_FIELDS:
        raise ValueError("observer-backed reload reference fields changed")
    value, file_sha256 = _read(Path(str(reference["path"])), schema)
    if file_sha256 != str(reference["file_sha256"]).removeprefix("sha256:") or _sha(
        value["sha256"], "observer-backed reload receipt digest"
    ) != str(reference["receipt_sha256"]).removeprefix("sha256:"):
        raise ValueError("observer-backed reload reference changed")
    return value, file_sha256


def _trained_state_commitments(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for rank, row in enumerate(rows):
        restored = row["trained_reload"]["restored"]
        result.append(
            {
                "rank": rank,
                "model_tensor_count": row["policy_tensor_count"],
                "model_local_numel": row["local_policy_numel"],
                "model_structure_sha256": restored["model"]["structure_sha256"],
                "model_value_sha256": restored["model"]["value_sha256"],
                "optimizer_value_sha256": restored["optimizer"]["value_sha256"],
                "scheduler_value_sha256": restored["scheduler"]["value_sha256"],
                "rng_value_sha256": restored["rng_sha256"],
            }
        )
    return result


def _observer_reload_body(terminal_path: Path) -> dict[str, Any]:
    """Derive reload acceptance from the same released all-rank observer run."""
    from . import miles_acceptance

    terminal, terminal_file_sha256 = _read(
        terminal_path,
        miles_acceptance.TERMINAL_SCHEMA,
    )
    miles_acceptance.validate_terminal(terminal, check_files=True)
    training_submission, _ = _reopen_reference(
        terminal["submission_binding"],
        miles_acceptance.SUBMISSION_SCHEMA,
    )
    source_plan, _ = _json_snapshot(Path(training_submission["source_plan_path"]))
    if not isinstance(source_plan, dict):
        raise ValueError("observer-backed reload source plan is not an object")
    checkpoint, checkpoint_file_sha256 = _reopen_reference(
        terminal["checkpoint_manifest"],
        CHECKPOINT_SCHEMA,
    )
    policy, policy_file_sha256 = _reopen_reference(
        terminal["policy_delta_observation"],
        POLICY_SCHEMA,
    )
    validate_policy_evidence(policy, source_plan, checkpoint)

    observer_submission, _ = _reopen_reference(
        policy["observer_submission"],
        SUBMISSION_SCHEMA,
    )
    observer_plan, observer_plan_file_sha256 = _json_snapshot(
        Path(observer_submission["observer_plan_path"])
    )
    if not isinstance(observer_plan, dict):
        raise ValueError("observer-backed reload plan is not an object")
    _validate_plan(observer_plan, check_files=True, require_current_runtime=False)
    if (
        observer_plan_file_sha256
        != observer_submission["observer_plan_file_sha256"].removeprefix("sha256:")
        or observer_plan["source_plan"] != source_plan
        or observer_plan["trained_checkpoint"] != checkpoint
    ):
        raise ValueError("observer-backed reload plan changed")
    result, _ = _reopen_reference(policy["observer_result"], RESULT_SCHEMA)
    validate_result(observer_plan, result)
    controller, controller_file_sha256 = _reopen_reference(
        policy["observer_controller"],
        CONTROLLER_SCHEMA,
    )
    validate_controller_observation(
        controller,
        observer_plan,
        observer_submission,
        check_files=True,
    )
    release, _ = _reopen_reference(policy["observer_release"], RELEASE_SCHEMA)
    validate_release_observation(
        release,
        observer_plan,
        observer_submission,
        controller,
        controller_file_sha256,
    )
    _verify_checkpoint(checkpoint, hashes=True)
    commitments = _trained_state_commitments(result["ranks"])
    return {
        "schema": RELOAD_ACCEPTED_SCHEMA,
        "status": "accepted",
        "source_manifest_sha256": checkpoint["sha256"],
        "source_terminal_acceptance_sha256": terminal["sha256"],
        "source_policy_delta_observation_sha256": policy["sha256"],
        "terminal_acceptance": _reference(
            terminal_path,
            terminal,
            terminal_file_sha256,
        ),
        "checkpoint_manifest": _reference(
            Path(terminal["checkpoint_manifest"]["path"]),
            checkpoint,
            checkpoint_file_sha256,
        ),
        "policy_delta_observation": _reference(
            Path(terminal["policy_delta_observation"]["path"]),
            policy,
            policy_file_sha256,
        ),
        "observer_result": policy["observer_result"],
        "observer_controller": policy["observer_controller"],
        "observer_release": policy["observer_release"],
        "world_size": WORLD_SIZE,
        "ranks": list(range(WORLD_SIZE)),
        "restored_rollout_index": checkpoint["rollout_index"],
        "restored_next_rollout_id": result["restored_next_rollout_id"],
        "rank_state_commitment_method": RANK_STATE_COMMITMENT_METHOD,
        "rank_state_commitments_sha256": "sha256:" + digest(commitments),
        "prediction_probe": result["prediction_probe"],
        "all_rank_model_loaded": True,
        "all_rank_optimizer_loaded": True,
        "all_rank_scheduler_loaded": True,
        "all_rank_rng_loaded": True,
        "state_stable_across_zero_updates": True,
        "exact_checkpoint_payload_reopened_after_release": True,
        "source_checkpoint_unchanged_after_release": True,
        "work_executed": dict(_OBSERVER_WORK),
        "observer_gpu_jobs": 1,
        "additional_reload_gpu_jobs": 0,
        "external_gpu_release_verified": True,
        "production_promotion_requires_this_receipt": True,
        "reward_values_included": False,
        "task_content_included": False,
        "tensor_values_included": False,
    }


def accept_observer_reload(*, terminal_path: Path, output: Path) -> dict[str, Any]:
    """Accept the observer's trained leg as the zero-update native reload gate."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("observer-backed reload acceptance already exists")
    body = _observer_reload_body(terminal_path)
    expected = Path(body["observer_result"]["path"]).parent / "RELOAD_ACCEPTED.json"
    if output != expected:
        raise ValueError("observer-backed reload output is not observer-run-bound")
    validate_observer_reload_accepted(
        {**body, "sha256": "sha256:" + digest(body)},
        check_files=True,
    )
    return _write(output, body)


def validate_observer_reload_accepted(
    value: dict[str, Any], *, check_files: bool = True
) -> dict[str, str]:
    """Reopen the single-run observer reload chain for production promotion."""
    if not check_files:
        raise ValueError("observer-backed reload requires reopening every referenced file")
    sealed(value, RELOAD_ACCEPTED_SCHEMA)
    if (
        set(value) != _RELOAD_ACCEPTED_FIELDS
        or value.get("status") != "accepted"
        or value.get("world_size") != WORLD_SIZE
        or value.get("ranks") != list(range(WORLD_SIZE))
        or value.get("rank_state_commitment_method") != RANK_STATE_COMMITMENT_METHOD
        or validate_prediction_probe(value.get("prediction_probe")) != value.get("prediction_probe")
        or value.get("all_rank_model_loaded") is not True
        or value.get("all_rank_optimizer_loaded") is not True
        or value.get("all_rank_scheduler_loaded") is not True
        or value.get("all_rank_rng_loaded") is not True
        or value.get("state_stable_across_zero_updates") is not True
        or value.get("exact_checkpoint_payload_reopened_after_release") is not True
        or value.get("source_checkpoint_unchanged_after_release") is not True
        or value.get("work_executed") != _OBSERVER_WORK
        or value.get("observer_gpu_jobs") != 1
        or value.get("additional_reload_gpu_jobs") != 0
        or value.get("external_gpu_release_verified") is not True
        or value.get("production_promotion_requires_this_receipt") is not True
        or value.get("reward_values_included") is not False
        or value.get("task_content_included") is not False
        or value.get("tensor_values_included") is not False
        or any(
            not isinstance(value.get(name), dict) or set(value[name]) != _REFERENCE_FIELDS
            for name in (
                "terminal_acceptance",
                "checkpoint_manifest",
                "policy_delta_observation",
                "observer_result",
                "observer_controller",
                "observer_release",
            )
        )
    ):
        raise ValueError("observer-backed reload acceptance is incomplete")
    expected = _observer_reload_body(Path(value["terminal_acceptance"]["path"]))
    if {key: item for key, item in value.items() if key != "sha256"} != expected:
        raise ValueError("observer-backed reload differs from rederived evidence")
    return {
        "source_manifest_sha256": value["source_manifest_sha256"],
        "source_terminal_acceptance_sha256": value["source_terminal_acceptance_sha256"],
        "rank_state_commitments_sha256": value["rank_state_commitments_sha256"],
        "prediction_probe": value["prediction_probe"],
    }


def _native(plan: dict[str, Any]) -> dict[str, Any]:
    import ray

    trained_before = _verify_checkpoint(plan["trained_checkpoint"], hashes=True)
    base_before = _base_checkpoint(plan["source_plan"], hashes=True)
    ray.init(
        address="auto",
        log_to_driver=False,
        runtime_env={"env_vars": dict(_ENV)},
    )
    try:
        base_start, base = asyncio.run(
            _probe_group(native_args(plan, trained=False), BASE_SENTINEL)
        )
        reference_start, trained_reference = asyncio.run(
            _probe_group(native_args(plan, trained=True), REFERENCE_SENTINEL)
        )
        reload_start, trained_reload = asyncio.run(
            _probe_group(native_args(plan, trained=True), RELOAD_SENTINEL)
        )
    finally:
        ray.shutdown()
    if set(base_start) != {1}:
        raise ValueError("base observer ranks restored the wrong rollout index")
    if any(
        set(start_ids) != {plan["trained_checkpoint"]["next_rollout_id"]}
        for start_ids in (reference_start, reload_start)
    ):
        raise ValueError("trained observer ranks restored the wrong rollout index")
    if (
        _verify_checkpoint(plan["trained_checkpoint"], hashes=True) != trained_before
        or _base_checkpoint(plan["source_plan"], hashes=True) != base_before
    ):
        raise ValueError("a source checkpoint changed during policy observation")
    rows, prediction_probe = _rank_rows(base, trained_reference, trained_reload)
    return {
        "schema": RESULT_SCHEMA,
        "status": "observed",
        "observer_plan_sha256": "sha256:" + digest(plan),
        "source_plan_sha256": plan["source_plan_sha256"],
        "base_checkpoint_receipt_sha256": plan["base_checkpoint_receipt_sha256"],
        "trained_checkpoint_receipt_sha256": plan["trained_checkpoint"]["sha256"],
        "world_size": WORLD_SIZE,
        "comparison_method": COMPARISON_METHOD,
        "ranks": rows,
        "changed_policy_ranks": [row["rank"] for row in rows if row["policy_changed"]],
        "restored_next_rollout_id": plan["trained_checkpoint"]["next_rollout_id"],
        "checkpoint_state_commitment_method": RANK_STATE_COMMITMENT_METHOD,
        "restore_method": RESTORE_METHOD,
        "trained_restore_count": 2,
        "prediction_probe": prediction_probe,
        "state_stable_across_zero_updates": True,
        "work_executed": dict(_OBSERVER_WORK),
        "base_checkpoint_unchanged": True,
        "trained_checkpoint_unchanged": True,
        "completed_at": time.time(),
        "reward_values_included": False,
        "task_content_included": False,
        "tensor_values_included": False,
    }


def run(plan: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("RUN_DIR") != plan["output_root"]:
        raise ValueError("Jobs API observer output binding drift")
    _validate_plan(plan, check_files=True, require_current_runtime=True)
    root = Path(plan["output_root"])
    if any((root / name).exists() for name in ("POLICY_OBSERVER_RESULT.json", "FAILED.json")):
        raise FileExistsError("policy observer output already exists")
    _write(
        root / "STARTED.json",
        {
            "schema": PLAN_SCHEMA,
            "observer_plan_sha256": "sha256:" + digest(plan),
            "work_authorized": dict(_OBSERVER_WORK),
            "started_at": time.time(),
        },
    )

    def timeout(*_: object) -> None:
        raise TimeoutError

    previous = {
        number: signal.signal(number, timeout) for number in (signal.SIGALRM, signal.SIGTERM)
    }
    signal.alarm(DEADLINE_SECONDS)
    try:
        value = _native(plan)
        value["sha256"] = "sha256:" + digest(value)
        validate_result(plan, value)
        return _write(
            root / "POLICY_OBSERVER_RESULT.json",
            {key: item for key, item in value.items() if key != "sha256"},
        )
    except BaseException as error:
        with suppress(Exception):
            _write(
                root / "FAILED.json",
                {
                    "schema": PLAN_SCHEMA,
                    "observer_plan_sha256": "sha256:" + digest(plan),
                    "error_class": type(error).__name__,
                },
            )
        raise RuntimeError("policy observation failed; preserve output, never auto-retry") from None
    finally:
        signal.alarm(0)
        for number, handler in previous.items():
            signal.signal(number, handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        if not isinstance(plan, dict) or digest(plan) != args.sha256:
            raise ValueError("observer plan digest mismatch")
        result = run(plan)
        print(json.dumps({key: result[key] for key in ("status", "sha256")}, sort_keys=True))
    except BaseException as error:
        print(json.dumps({"status": "failed", "error_class": type(error).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
