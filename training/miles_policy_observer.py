"""Post-hoc, zero-update policy-state observation for one Miles canary.

The training process is not trusted to attest its own update.  This module
loads the exact base and CPU-sealed trained distributed checkpoints in two
isolated all-rank Miles actor groups, compares named policy tensor values, and
seals the trained optimizer, scheduler, and RNG commitments needed by the
later reload gate.  It never creates rollout engines or calls forward,
backward, optimizer, scheduler, save, evaluation, or W&B methods.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import gzip
import hashlib
import json
import math
import numbers
import os
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

CONFIG_SCHEMA = "cyber_miles_policy_observer_config_v1"
PLAN_SCHEMA = "cyber_miles_policy_observer_plan_v1"
RESULT_SCHEMA = "cyber_miles_policy_observer_result_v1"
SUBMISSION_SCHEMA = "cyber_miles_policy_observer_submission_v1"
CONTROLLER_SCHEMA = "cyber_miles_policy_observer_controller_v1"
RELEASE_SCHEMA = "cyber_miles_policy_observer_release_v1"
POLICY_SCHEMA = "cyber_miles_policy_tensor_delta_observation_v1"
WORLD_SIZE = 8
DEADLINE_SECONDS = 1800
COMPARISON_METHOD = "all_rank_named_policy_tensor_value_sha256_v1"
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
    "checkpoint_state_commitment_method",
    "work_executed",
    "base_checkpoint_unchanged",
    "trained_checkpoint_unchanged",
    "completed_at",
    "reward_values_included",
    "task_content_included",
    "tensor_values_included",
    "sha256",
}
_RANK_FIELDS = {
    "rank",
    "policy_tensor_count",
    "local_policy_numel",
    "base_policy_structure_sha256",
    "trained_policy_structure_sha256",
    "base_policy_value_sha256",
    "trained_policy_value_sha256",
    "trained_optimizer_value_sha256",
    "trained_scheduler_value_sha256",
    "trained_rng_value_sha256",
    "policy_changed",
}
_SUBMISSION_FIELDS = {
    "schema",
    "source_commit",
    "observer_plan_path",
    "observer_plan_sha256",
    "observer_plan_file_sha256",
    "observer_request_path",
    "observer_request_sha256",
    "observer_request_file_sha256",
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
    submission, submission_file_sha256 = _read(
        submission_path, miles_acceptance.SUBMISSION_SCHEMA
    )
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
        "work_authorized": dict(_ZERO_WORK),
        "runtime_sha256": digest(_runtime()),
        "native_driver_sha256": NATIVE_DRIVER_SHA256,
        "deadline_seconds": DEADLINE_SECONDS,
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
        or plan.get("work_authorized") != _ZERO_WORK
        or _SHA.fullmatch(str(plan.get("runtime_sha256"))) is None
        or plan.get("native_driver_sha256") != NATIVE_DRIVER_SHA256
        or plan.get("deadline_seconds") != DEADLINE_SECONDS
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
        if (
            reopened_source != source
            or source_file_sha256 != plan["source_plan_file_sha256"].removeprefix("sha256:")
        ):
            raise ValueError("source Miles plan file changed")
        submission, submission_file_sha256 = _read(
            Path(submission_ref["path"]), "cyber_miles_submitted_execution_binding_v1"
        )
        if (
            submission_file_sha256 != submission_ref["file_sha256"].removeprefix("sha256:")
            or _sha(submission["sha256"], "source submission digest")
            != submission_ref["receipt_sha256"].removeprefix("sha256:")
        ):
            raise ValueError("source submission evidence changed")
        if (
            submission.get("source_plan_sha256", "").removeprefix("sha256:")
            != digest(source)
            or Path(str(submission.get("source_plan_path"))) != source_path
        ):
            raise ValueError("source submission does not bind the selected Miles plan")
        checkpoint, checkpoint_file_sha256 = _read(
            Path(checkpoint_ref["path"]), CHECKPOINT_SCHEMA
        )
        if (
            checkpoint != trained
            or checkpoint_file_sha256 != checkpoint_ref["file_sha256"].removeprefix("sha256:")
        ):
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
        "name", "title", "run_dir", "image", "workers", "gpus_per_worker", "resources",
        "priority_class", "requeueIfPreempted", "secrets", "env", "command",
    }
    env = request.get("env")
    if not isinstance(env, dict):
        raise ValueError("observer request environment is absent")
    transport = {
        key for key in env
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
    api_run_id: str,
    submitted_at: object,
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
    run_id = _uuid(api_run_id, "observer API run ID")
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
        if (
            value["runtime_bundle_sha256"].removeprefix("sha256:") != bundle
            or projection != request
        ):
            raise ValueError("observer runtime bundle changed")
    return {"request_sha256": value["observer_request_sha256"], "api": api}


def start_capture(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    namespace_uid: str,
    started_at: object,
    directory: Path,
) -> dict[str, Any]:
    """Start the UID event journal before observer admission."""
    from . import miles_event_evidence as events
    from .miles_acceptance import _time, _uuid

    _validate_plan(plan, check_files=False, require_current_runtime=False)
    submitted = validate_submission_binding(submission, plan, check_files=False)
    _time(started_at, "observer capture start")
    namespace = _uuid(namespace_uid, "observer namespace UID")
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
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
            "namespace": "fleet-train-jobs",
            "namespace_uid": namespace,
            "started_at": started_at,
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
    if (
        start["source_plan_sha256"].removeprefix("sha256:") != digest(plan)
        or start["source_request_sha256"] != submitted["request_sha256"]
        or start["api_run_id"] != submitted["api"]["run_id"]
        or start["api_run_name"] != submitted["api"]["run_name"]
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
            row["owner"]
            != {"kind": "RayCluster", "name": raycluster_name, "uid": raycluster_uid}
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
            "start": _reference(start_path, start, start_file_sha256),
            "events": [
                _reference(path, value, file_sha256)
                for path, (value, file_sha256) in zip(
                    event_paths, event_records, strict=True
                )
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
            "schema", "status", "observer_plan_sha256", "observer_request_sha256",
            "api", "kubernetes", "execution", "event_journal", "observed_at",
            "private_logs_included",
            "metric_values_included", "task_content_included", "sha256",
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
        or set(journal) != {"start", "events"}
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
            "name", "uid", "owner_raycluster_uid", "phase", "exit_code",
            "termination_reason", "runtime_image_id", "container_restarts", "gpus",
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
            "schema", "status", "observer_plan_sha256", "observer_request_sha256",
            "controller_observation_sha256", "controller_observation_file_sha256",
            "api_status", "controller_status", "identities", "raycluster_present",
            "rayjob_present", "workload_present", "quota_reservation_present",
            "gpu_pods_present", "active_gpus", "gpu_release_proven", "observed_at", "sha256",
        }
        or value.get("status") != "released"
        or value.get("observer_plan_sha256", "").removeprefix("sha256:") != digest(plan)
        or value.get("observer_request_sha256") != submitted["request_sha256"]
        or value.get("controller_observation_sha256") != controller["sha256"]
        or value.get("controller_observation_file_sha256", "").removeprefix("sha256:")
        != controller_file_sha256.removeprefix("sha256:")
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


def compile_release(
    plan: dict[str, Any],
    submission: dict[str, Any],
    *,
    controller_path: Path,
    absence: Mapping[str, Any],
    observed_at: object,
    output: Path,
) -> dict[str, Any]:
    controller, controller_file_sha256 = _read(controller_path, CONTROLLER_SCHEMA)
    expected_absence = {
        "raycluster_present": False,
        "rayjob_present": False,
        "workload_present": False,
        "quota_reservation_present": False,
        "gpu_pods_present": False,
        "active_gpus": 0,
    }
    if dict(absence) != expected_absence:
        raise ValueError("policy observer Kubernetes absence is incomplete")
    kube, api = controller["kubernetes"], controller["api"]
    body = {
        "schema": RELEASE_SCHEMA,
        "status": "released",
        "observer_plan_sha256": "sha256:" + digest(plan),
        "observer_request_sha256": validate_submission_binding(
            submission, plan, check_files=False
        )["request_sha256"],
        "controller_observation_sha256": controller["sha256"],
        "controller_observation_file_sha256": "sha256:" + controller_file_sha256,
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
        **expected_absence,
        "gpu_release_proven": True,
        "observed_at": observed_at,
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
        plan["trained_checkpoint"]["root"]
        if trained
        else plan["source_plan"]["checkpoint"]["root"]
    )
    return _parse_args(source, root, trained=trained)


def _base_probe(self) -> dict[str, Any]:
    import torch.distributed as dist

    dist.barrier()
    result = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "model": _model_probe(self.model),
    }
    dist.barrier()
    return result


def _trained_probe(self) -> dict[str, Any]:
    import torch.distributed as dist

    dist.barrier()
    result = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "model": _model_probe(self.model),
        "optimizer": _optimizer_probe(self.optimizer),
        "scheduler": _scheduler_probe(self.opt_param_scheduler),
        "rng_sha256": _rng_probe(),
    }
    dist.barrier()
    return result


async def _probe_group(args, method_name: str, method) -> tuple[list[Any], list[dict[str, Any]]]:
    import ray
    from miles.backends.megatron_utils import actor as actor_module
    from miles.ray.placement_group import allocate_train_group, create_placement_groups

    original = actor_module.MegatronTrainRayActor
    actor = type("CyberMilesPolicyObserverActor", (original,), {method_name: method})
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
        rows = await group._broadcast(method_name)
        return list(start_ids), list(rows)
    finally:
        actor_module.MegatronTrainRayActor = original
        if group is not None:
            for handle in group._actor_handles:
                with suppress(Exception):
                    ray.kill(handle, no_restart=True)
        if pgs is not None and pgs["actor"][0] is not None:
            with suppress(Exception):
                ray.util.remove_placement_group(pgs["actor"][0])


def _rank_rows(
    base: Sequence[dict[str, Any]], trained: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(base) != WORLD_SIZE or len(trained) != WORLD_SIZE:
        raise ValueError("policy observer did not receive all ranks")
    base = sorted(base, key=lambda row: row["rank"])
    trained = sorted(trained, key=lambda row: row["rank"])
    if [row["rank"] for row in base] != list(range(WORLD_SIZE)) or [
        row["rank"] for row in trained
    ] != list(range(WORLD_SIZE)):
        raise ValueError("policy observer rank identity mismatch")
    result = []
    for rank, (before, after) in enumerate(zip(base, trained, strict=True)):
        base_model, trained_model = before["model"], after["model"]
        if (
            before["world_size"] != WORLD_SIZE
            or after["world_size"] != WORLD_SIZE
            or base_model["tensors"] < 1
            or base_model["local_numel"] < 1
            or trained_model["tensors"] != base_model["tensors"]
            or trained_model["local_numel"] != base_model["local_numel"]
            or trained_model["structure_sha256"] != base_model["structure_sha256"]
            or after["optimizer"]["state_entries"] < 1
            or after["optimizer"]["parameter_groups"] < 1
            or after["scheduler"]["positive_progress_counters"] < 1
        ):
            raise ValueError("policy observer found incomplete or structurally different state")
        changed = base_model["value_sha256"] != trained_model["value_sha256"]
        result.append(
            {
                "rank": rank,
                "policy_tensor_count": base_model["tensors"],
                "local_policy_numel": base_model["local_numel"],
                "base_policy_structure_sha256": "sha256:" + base_model["structure_sha256"],
                "trained_policy_structure_sha256": "sha256:" + trained_model["structure_sha256"],
                "base_policy_value_sha256": "sha256:" + base_model["value_sha256"],
                "trained_policy_value_sha256": "sha256:" + trained_model["value_sha256"],
                "trained_optimizer_value_sha256": "sha256:" + after["optimizer"]["value_sha256"],
                "trained_scheduler_value_sha256": "sha256:" + after["scheduler"]["value_sha256"],
                "trained_rng_value_sha256": "sha256:" + after["rng_sha256"],
                "policy_changed": changed,
            }
        )
    return result


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
        or value.get("trained_checkpoint_receipt_sha256")
        != plan["trained_checkpoint"]["sha256"]
        or value.get("world_size") != WORLD_SIZE
        or value.get("comparison_method") != COMPARISON_METHOD
        or value.get("checkpoint_state_commitment_method") != RANK_STATE_COMMITMENT_METHOD
        or value.get("work_executed") != _ZERO_WORK
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
            or any(
                _SHA.fullmatch(str(row.get(key))) is None
                for key in (
                    "base_policy_structure_sha256", "trained_policy_structure_sha256",
                    "base_policy_value_sha256", "trained_policy_value_sha256",
                    "trained_optimizer_value_sha256", "trained_scheduler_value_sha256",
                    "trained_rng_value_sha256",
                )
            )
            or row["base_policy_structure_sha256"] != row["trained_policy_structure_sha256"]
            or row.get("policy_changed")
            is not (row["base_policy_value_sha256"] != row["trained_policy_value_sha256"])
        ):
            raise ValueError("policy observer rank result is invalid")
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
    if not changed:
        raise ValueError("trained checkpoint has no independently observed policy tensor delta")
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
        "observer_submission": _reference(
            submission_path, submission, submission_file_sha256
        ),
        "observer_result": _reference(result_path, result, result_file_sha256),
        "observer_controller": _reference(
            controller_path, controller, controller_file_sha256
        ),
        "observer_release": _reference(release_path, release, release_file_sha256),
        "world_size": WORLD_SIZE,
        "comparison_method": COMPARISON_METHOD,
        "ranks": result["ranks"],
        "changed_policy_ranks": changed,
        "policy_structure_matches": True,
        "optimizer_state_used_for_delta": False,
        "scheduler_state_used_for_delta": False,
        "rng_state_used_for_delta": False,
        "metadata_used_for_delta": False,
        "checkpoint_state_commitment_method": RANK_STATE_COMMITMENT_METHOD,
        "observer_work": dict(_ZERO_WORK),
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
        "schema", "source_plan_sha256", "base_checkpoint_receipt_sha256",
        "trained_checkpoint_receipt_sha256", "observer_plan_sha256",
        "observer_submission", "observer_result", "observer_controller",
        "observer_release", "world_size", "comparison_method", "ranks",
        "changed_policy_ranks", "policy_structure_matches",
        "optimizer_state_used_for_delta", "scheduler_state_used_for_delta",
        "rng_state_used_for_delta", "metadata_used_for_delta",
        "checkpoint_state_commitment_method", "observer_work",
        "source_checkpoints_unchanged", "observer_external_gpu_release_verified",
        "reward_values_included", "task_content_included", "tensor_values_included",
        "sha256",
    }
    if (
        set(value) != expected_fields
        or value.get("source_plan_sha256", "").removeprefix("sha256:")
        != digest(source_plan)
        or value.get("base_checkpoint_receipt_sha256")
        != source_plan.get("checkpoint", {}).get("sha256")
        or value.get("trained_checkpoint_receipt_sha256") != checkpoint.get("sha256")
        or value.get("world_size") != WORLD_SIZE
        or value.get("comparison_method") != COMPARISON_METHOD
        or value.get("policy_structure_matches") is not True
        or value.get("optimizer_state_used_for_delta") is not False
        or value.get("scheduler_state_used_for_delta") is not False
        or value.get("rng_state_used_for_delta") is not False
        or value.get("metadata_used_for_delta") is not False
        or value.get("checkpoint_state_commitment_method")
        != RANK_STATE_COMMITMENT_METHOD
        or value.get("observer_work") != _ZERO_WORK
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
    submission, submission_file_sha256 = _read(
        Path(submission_ref["path"]), SUBMISSION_SCHEMA
    )
    plan_path = Path(submission["observer_plan_path"])
    from .miles_acceptance import _json_snapshot

    observer_plan, observer_plan_file_sha256 = _json_snapshot(plan_path)
    if not isinstance(observer_plan, dict):
        raise ValueError("policy observer plan is not a JSON object")
    _validate_plan(observer_plan, check_files=True, require_current_runtime=False)
    if (
        observer_plan["source_plan"] != source_plan
        or observer_plan["trained_checkpoint"] != checkpoint
        or value["observer_plan_sha256"].removeprefix("sha256:")
        != digest(observer_plan)
        or submission["observer_plan_file_sha256"].removeprefix("sha256:")
        != observer_plan_file_sha256
        or submission_file_sha256
        != submission_ref["file_sha256"].removeprefix("sha256:")
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
        _, base = asyncio.run(
            _probe_group(native_args(plan, trained=False), "cyber_base_probe", _base_probe)
        )
        trained_start, trained = asyncio.run(
            _probe_group(native_args(plan, trained=True), "cyber_trained_probe", _trained_probe)
        )
    finally:
        ray.shutdown()
    if set(trained_start) != {plan["trained_checkpoint"]["next_rollout_id"]}:
        raise ValueError("trained observer ranks restored the wrong rollout index")
    if (
        _verify_checkpoint(plan["trained_checkpoint"], hashes=True) != trained_before
        or _base_checkpoint(plan["source_plan"], hashes=True) != base_before
    ):
        raise ValueError("a source checkpoint changed during policy observation")
    rows = _rank_rows(base, trained)
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
        "checkpoint_state_commitment_method": RANK_STATE_COMMITMENT_METHOD,
        "work_executed": dict(_ZERO_WORK),
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
            "work_authorized": dict(_ZERO_WORK),
            "started_at": time.time(),
        },
    )
    def timeout(*_: object) -> None:
        raise TimeoutError

    previous = {
        number: signal.signal(number, timeout)
        for number in (signal.SIGALRM, signal.SIGTERM)
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
