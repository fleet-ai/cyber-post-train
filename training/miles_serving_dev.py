"""Bounded dev-only SGLang qualification for one accepted Miles HF export.

This module never registers a persistent endpoint.  It compiles one generic
Jobs API request, starts the exact production SGLang command on localhost,
performs non-task probes, stops the server, and accepts a qualification only
after separate UID-bound terminal/release evidence is supplied.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, bundled_request, digest

from .io import canonical_json, digest_json, file_sha256

CONFIG_SCHEMA = "cyber_miles_serving_dev_config_v1"
PLAN_SCHEMA = "cyber_miles_serving_dev_plan_v1"
RESULT_SCHEMA = "cyber_miles_serving_dev_result_v1"
EXTERNAL_SCHEMA = "cyber_miles_serving_dev_external_v1"
PREFLIGHT_SCHEMA = "cyber_miles_serving_dev_cpu_preflight_v1"
DEV_QUALIFICATION_SCHEMA = "cyber_serving_dev_qualification_v1"
DEV_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
NAMESPACE = "fleet-train-jobs"
MODEL_REPO = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
STARTUP_TIMEOUT_SECONDS = 1800
REQUEST_TIMEOUT_SECONDS = 300
SHUTDOWN_TIMEOUT_SECONDS = 60
DEV_CHECKS = (
    "exact_model_identity",
    "full_model_reload",
    "finite_forward",
    "structured_tool_call",
    "context_continuation",
    "cleanup_verified",
)
RESULT_FIELDS = {
    "schema",
    "status",
    "plan_sha256",
    "request_sha256",
    "api_run_name",
    "api_run_id",
    "execution_contract_sha256",
    "export_receipt_sha256",
    "reload_acceptance_receipt_sha256",
    "source_update_identity_sha256",
    "staged_manifest_sha256",
    "requested_runtime_image",
    "checks",
    "probes",
    "server_pid",
    "server_exit_code",
    "server_process_group_stopped",
    "source_export_unchanged",
    "optimizer_updates",
    "rollouts",
    "verifier_calls",
    "benchmark_attempts",
    "completed_at",
    "sha256",
}
PROBE_FIELDS = {
    "model_list_exact",
    "forward_completion_tokens",
    "continuation_completion_tokens",
    "tool_name",
    "tool_argument_keys",
    "response_content_recorded",
    "task_content_included",
    "benchmark_content_included",
    "scores_observed",
}
EXTERNAL_FIELDS = {
    "schema",
    "status",
    "cluster",
    "api_base_url",
    "kube_context",
    "namespace",
    "namespace_uid",
    "run_name",
    "api_run_id",
    "request_sha256",
    "effective_priority",
    "automatic_requeue",
    "controller",
    "pod",
    "release",
    "observed_at",
}
CONTROLLER_FIELDS = {"kind", "name", "uid", "status", "workload_uid", "raycluster_uid"}
POD_FIELDS = {
    "name",
    "uid",
    "owner_raycluster_uid",
    "phase",
    "exit_code",
    "termination_reason",
    "terminated_at",
    "container_restarts",
    "gpus",
    "runtime_image_id",
}
RELEASE_FIELDS = {
    "api_status",
    "raycluster_present",
    "workload_present",
    "gpu_pods_present",
    "active_gpus",
    "observed_at",
}
RUNTIME_FILES = (
    "training/miles_serving_dev.py",
    "training/io.py",
    "cyber_post_train/jobs.py",
)
_SHA = re.compile(r"(?:sha256:)?[0-9a-f]{64}")
_PREFIX = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?")
_API_RUN = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,52}[a-z0-9])?-[0-9a-f]{8}")
_RUNTIME_IMAGE = re.compile(
    r"(?:(?:containerd|docker-pullable)://(?:[^@\s]+@)?|[^@\s]+@)"
    r"sha256:([a-f0-9]{64})"
)


def _sha(value: object) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("expected exact SHA-256")
    return "sha256:" + value.removeprefix("sha256:")


def _read(path: Path, expected: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("evidence must be a regular nonsymlink file")
    if expected is not None and file_sha256(path) != _sha(expected):
        raise ValueError("evidence file SHA-256 mismatch")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _sealed(value: dict[str, Any], schema: str) -> dict[str, Any]:
    if value.get("schema") != schema:
        raise ValueError(f"expected {schema}")
    actual = digest({key: item for key, item in value.items() if key != "sha256"})
    if value.get("sha256", "").removeprefix("sha256:") != actual:
        raise ValueError(f"{schema} self-digest mismatch")
    return value


def _sign(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "sha256": digest(value)}


def _once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (canonical_json(value) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _absolute(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or len(path.parts) < 5:
        raise ValueError(f"{label} must be an exact absolute path")
    return path


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _reference(value: object, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "file_sha256"}:
        raise ValueError(f"{label} must contain path and file_sha256")
    path = _absolute(value["path"], label)
    if path.resolve() != path or path.is_symlink():
        raise ValueError(f"{label} path must not traverse a symlink")
    return path, _sha(value["file_sha256"])


def _runtime_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    sources = {name: (root / name).read_text() for name in RUNTIME_FILES}
    sources.update({"training/__init__.py": "", "cyber_post_train/__init__.py": ""})
    return sources


def _source_update_identity(export: dict[str, Any]) -> dict[str, str]:
    source = export.get("source")
    checkpoint = source.get("checkpoint") if isinstance(source, dict) else None
    if not isinstance(checkpoint, dict):
        raise ValueError("Miles export lacks source checkpoint identity")
    return {
        "schema": "cyber_miles_source_update_identity_v1",
        "source_plan_sha256": _sha(source.get("source_plan_sha256")),
        "source_checkpoint_receipt_sha256": _sha(checkpoint.get("receipt_sha256")),
        "export_tensor_inventory_sha256": _sha(export.get("tensor_inventory_sha256")),
    }


def _artifacts(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    from . import miles_hf_export

    export_path, export_file_sha = _reference(config["export"], "export")
    reload_path, reload_file_sha = _reference(config["reload_acceptance"], "reload acceptance")
    export, _ = miles_hf_export.inspect_export(export_path, export_file_sha)
    reload_receipt = _read(reload_path, reload_file_sha)
    validation = miles_hf_export.validate_reload_accepted(reload_receipt, check_files=True)
    export_receipt_sha = _sha(export.get("sha256"))
    model = export.get("model")
    if (
        not isinstance(model, dict)
        or model
        != {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "base_root": model.get("base_root"),
            "native_parallelism": {"tensor": 4, "context": 2, "world_size": 8},
        }
        or export.get("dtype") != "BF16"
        or export.get("optimizer_updates_executed") != 0
        or reload_receipt.get("serving_qualified") is not False
        or reload_receipt.get("export_path") != str(export_path)
        or _sha(reload_receipt.get("export_file_sha256")) != export_file_sha
        or _sha(reload_receipt.get("export_receipt_sha256")) != export_receipt_sha
        or _sha(validation.get("export_receipt_sha256")) != export_receipt_sha
        or _sha(reload_receipt.get("export_tensor_inventory_sha256"))
        != _sha(export.get("tensor_inventory_sha256"))
    ):
        raise ValueError("accepted Miles export/reload identity mismatch")
    return export, reload_receipt, {key: _sha(value) for key, value in validation.items()}


def _replace(args: list[str], flag: str, value: str) -> list[str]:
    if args.count(flag) != 1:
        raise ValueError(f"exact serving command requires one {flag}")
    result = list(args)
    index = result.index(flag) + 1
    if index == len(result) or result[index].startswith("--"):
        raise ValueError(f"exact serving command has no {flag} value")
    result[index] = value
    return result


def _serving_image(registration: object) -> str:
    if not isinstance(registration, dict):
        raise ValueError("serving registration is malformed")
    spec = registration.get("spec")
    runtime = spec.get("runtime") if isinstance(spec, dict) else None
    image = runtime.get("image") if isinstance(runtime, dict) else None
    if not isinstance(image, dict) or not isinstance(image.get("repository"), str):
        raise ValueError("serving registration image is malformed")
    return image["repository"] + "@" + _sha(image.get("digest"))


def _runtime_image_digest(value: object) -> str:
    if not isinstance(value, str) or (match := _RUNTIME_IMAGE.fullmatch(value)) is None:
        raise ValueError("runtime imageID is not an immutable observed digest")
    return match.group(1)


def compile_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Validate accepted artifacts and freeze one dev-only one-GPU request."""
    from . import serving_registration

    if (
        set(config)
        != {
            "schema",
            "name",
            "output_root",
            "base_registration",
            "export",
            "reload_acceptance",
            "cluster",
        }
        or config.get("schema") != CONFIG_SCHEMA
    ):
        raise ValueError("unexpected Miles serving dev configuration")
    if not _PREFIX.fullmatch(str(config.get("name", ""))):
        raise ValueError("dev canary name must be a Jobs API DNS prefix")
    output = _absolute(config.get("output_root"), "output root")
    if output.parts[:4] != ("/", "mnt", "sfs", "jobs"):
        raise ValueError("dev canary output must be under /mnt/sfs/jobs")
    if config.get("cluster") != {"target": "dev", "priority": "c1"}:
        raise ValueError("serving qualification is one dev c1 canary")

    base_path, base_sha = _reference(config["base_registration"], "base registration")
    base = serving_registration._registration(_read(base_path, base_sha))
    export, reload_receipt, reload_validation = _artifacts(config)
    if base["spec"]["model"].get("revision") != MODEL_REVISION:
        raise ValueError("base serving registration is not exact Qwen3.8-27B")
    model_root = _absolute(export["output_root"], "Miles export root")
    if model_root != Path(config["export"]["path"]).parent:
        raise ValueError("Miles export root and receipt path differ")
    if _overlap(output, model_root):
        raise ValueError("dev evidence root must not overlap the immutable export")

    dev_registration = copy.deepcopy(base)
    runtime = dev_registration["spec"]["runtime"]
    model = dev_registration["spec"]["model"]
    args = _replace(runtime["args"], "--model-path", str(model_root))
    args = _replace(args, "--served-model-name", config["name"])
    dev_registration["id"] = config["name"]
    model.update(
        sourcePath=str(model_root),
        path=str(model_root),
        revision=_sha(export["sha256"]),
    )
    runtime["args"] = args
    serving_registration._registration(dev_registration)
    contract = digest_json(serving_registration.execution_contract(dev_registration))
    if contract != digest_json(serving_registration.execution_contract(base)):
        raise ValueError("dev canary differs from production serving execution contract")
    image = _serving_image(dev_registration)
    resources = dev_registration["spec"]["resources"]
    if resources["limits"].get("nvidia.com/gpu") != 1:
        raise ValueError("bounded dev qualification requires exact one-GPU serving shape")
    update = _source_update_identity(export)
    plan = {
        "schema": PLAN_SCHEMA,
        "run_name": config["name"],
        "output_root": str(output),
        "cluster_target": "dev",
        "priority": "c1",
        "base_registration": {
            "path": str(base_path),
            "file_sha256": base_sha,
            "id": base["id"],
            "revision": MODEL_REVISION,
        },
        "artifact": {
            "export_path": config["export"]["path"],
            "export_file_sha256": _sha(config["export"]["file_sha256"]),
            "export_receipt_sha256": _sha(export["sha256"]),
            "reload_acceptance_path": config["reload_acceptance"]["path"],
            "reload_acceptance_file_sha256": _sha(config["reload_acceptance"]["file_sha256"]),
            "reload_acceptance_receipt_sha256": _sha(reload_receipt["sha256"]),
            "reload_validation": reload_validation,
            "source_update_identity": update,
            "source_update_identity_sha256": digest_json(update),
            "staged_manifest_sha256": digest_json(export["files"]),
            "files": export["files"],
        },
        "serving": {
            "registration": dev_registration,
            "execution_contract_sha256": contract,
            "image": image,
            "command": runtime["command"],
            "args": args,
            "model_root": str(model_root),
            "resources": {
                "cpu_request": resources["requests"]["cpu"],
                "cpu_limit": resources["limits"]["cpu"],
                "memory_request": resources["requests"]["memory"],
                "memory_limit": resources["limits"]["memory"],
            },
            "gpus": 1,
        },
        "probe": {
            "startup_timeout_seconds": STARTUP_TIMEOUT_SECONDS,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "shutdown_timeout_seconds": SHUTDOWN_TIMEOUT_SECONDS,
            "task_content_included": False,
            "benchmark_content_included": False,
            "scores_observed": False,
            "webexploitbench_eligible_for_selection": False,
        },
        "runtime_sha256": digest(_runtime_sources()),
    }
    return _sign(plan)


def validate_plan(plan: dict[str, Any], *, check_files: bool) -> dict[str, Any]:
    from . import serving_registration

    _sealed(plan, PLAN_SCHEMA)
    serving = plan.get("serving")
    artifact = plan.get("artifact")
    base_reference = plan.get("base_registration")
    registration = serving.get("registration") if isinstance(serving, dict) else None
    if (
        set(plan)
        != {
            "schema",
            "run_name",
            "output_root",
            "cluster_target",
            "priority",
            "base_registration",
            "artifact",
            "serving",
            "probe",
            "runtime_sha256",
            "sha256",
        }
        or not _PREFIX.fullmatch(str(plan.get("run_name", "")))
        or plan.get("cluster_target") != "dev"
        or plan.get("priority") != "c1"
        or not isinstance(serving, dict)
        or not isinstance(artifact, dict)
        or not isinstance(base_reference, dict)
        or set(base_reference) != {"path", "file_sha256", "id", "revision"}
        or set(artifact)
        != {
            "export_path",
            "export_file_sha256",
            "export_receipt_sha256",
            "reload_acceptance_path",
            "reload_acceptance_file_sha256",
            "reload_acceptance_receipt_sha256",
            "reload_validation",
            "source_update_identity",
            "source_update_identity_sha256",
            "staged_manifest_sha256",
            "files",
        }
        or set(serving)
        != {
            "registration",
            "execution_contract_sha256",
            "image",
            "command",
            "args",
            "model_root",
            "resources",
            "gpus",
        }
        or serving.get("gpus") != 1
        or serving.get("image") != _serving_image(registration)
        or plan.get("probe")
        != {
            "startup_timeout_seconds": STARTUP_TIMEOUT_SECONDS,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "shutdown_timeout_seconds": SHUTDOWN_TIMEOUT_SECONDS,
            "task_content_included": False,
            "benchmark_content_included": False,
            "scores_observed": False,
            "webexploitbench_eligible_for_selection": False,
        }
        or plan.get("runtime_sha256") != digest(_runtime_sources())
    ):
        raise ValueError("Miles serving dev plan changed")
    registered = serving_registration._registration(registration)
    runtime = registered["spec"]["runtime"]
    resources = registered["spec"]["resources"]
    expected_resources = {
        "cpu_request": resources["requests"]["cpu"],
        "cpu_limit": resources["limits"]["cpu"],
        "memory_request": resources["requests"]["memory"],
        "memory_limit": resources["limits"]["memory"],
    }
    expected_contract = digest_json(serving_registration.execution_contract(registered))
    if (
        registered["id"] != plan["run_name"]
        or serving.get("command") != runtime["command"]
        or serving.get("args") != runtime["args"]
        or serving.get("model_root") != registered["spec"]["model"]["path"]
        or serving.get("resources") != expected_resources
        or serving.get("execution_contract_sha256") != expected_contract
        or registered["spec"]["resources"]["limits"].get("nvidia.com/gpu") != 1
    ):
        raise ValueError("Miles serving registration/request contract changed")
    _sha(serving.get("execution_contract_sha256"))
    for key in (
        "export_file_sha256",
        "export_receipt_sha256",
        "reload_acceptance_file_sha256",
        "reload_acceptance_receipt_sha256",
        "source_update_identity_sha256",
        "staged_manifest_sha256",
    ):
        _sha(artifact.get(key))
    files = artifact.get("files")
    reload_validation = artifact.get("reload_validation")
    update_identity = artifact.get("source_update_identity")
    if (
        not isinstance(files, dict)
        or not files
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(info, dict)
            or set(info) != {"bytes", "sha256"}
            or type(info["bytes"]) is not int
            or info["bytes"] < 0
            or _SHA.fullmatch(str(info["sha256"])) is None
            for name, info in files.items()
        )
        or digest_json(files) != artifact["staged_manifest_sha256"]
        or not isinstance(reload_validation, dict)
        or set(reload_validation)
        != {
            "export_receipt_sha256",
            "prediction_sha256",
            "reload_result_sha256",
            "controller_terminal_sha256",
            "external_release_sha256",
        }
        or any(_SHA.fullmatch(str(value)) is None for value in reload_validation.values())
        or not isinstance(update_identity, dict)
        or set(update_identity)
        != {
            "schema",
            "source_plan_sha256",
            "source_checkpoint_receipt_sha256",
            "export_tensor_inventory_sha256",
        }
        or _absolute(artifact.get("export_path"), "export path")
        != Path(serving["model_root"]) / "EXPORT.json"
    ):
        raise ValueError("Miles serving artifact manifest changed")
    _absolute(artifact.get("reload_acceptance_path"), "reload acceptance path")
    if (
        digest_json(update_identity) != artifact["source_update_identity_sha256"]
        or update_identity.get("schema") != "cyber_miles_source_update_identity_v1"
    ):
        raise ValueError("Miles source-update identity changed")
    if check_files:
        base_path, base_sha = _reference(
            {
                "path": base_reference["path"],
                "file_sha256": base_reference["file_sha256"],
            },
            "base registration",
        )
        base = serving_registration._registration(_read(base_path, base_sha))
        if (
            base["id"] != base_reference["id"]
            or base["spec"]["model"].get("revision") != base_reference["revision"]
            or serving_registration.execution_contract(base)
            != serving_registration.execution_contract(registered)
        ):
            raise ValueError("base serving registration changed after preparation")
        config = {
            "export": {
                "path": artifact["export_path"],
                "file_sha256": artifact["export_file_sha256"],
            },
            "reload_acceptance": {
                "path": artifact["reload_acceptance_path"],
                "file_sha256": artifact["reload_acceptance_file_sha256"],
            },
        }
        export, reload_receipt, validation = _artifacts(config)
        if (
            _sha(export["sha256"]) != artifact["export_receipt_sha256"]
            or _sha(reload_receipt["sha256"]) != artifact["reload_acceptance_receipt_sha256"]
            or validation != artifact["reload_validation"]
            or export["files"] != artifact["files"]
            or _source_update_identity(export) != artifact["source_update_identity"]
        ):
            raise ValueError("Miles serving artifact changed after preparation")
    return plan


def job_request(plan: dict[str, Any], *, check_files: bool = True) -> dict[str, Any]:
    """Render the one request; runtime may re-render without recursive evidence reads."""
    validate_plan(plan, check_files=check_files)
    serving = plan["serving"]
    request = {
        "name": plan["run_name"],
        "title": plan["run_name"] + " bounded Miles SGLang dev qualification",
        "run_dir": plan["output_root"],
        "image": serving["image"],
        "workers": 1,
        "gpus_per_worker": 1,
        "resources": serving["resources"],
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        },
    }
    files = _runtime_sources()
    files["plan.json"] = canonical_json(plan)
    return bundled_request(
        request,
        files,
        "training.miles_serving_dev",
        ["run", "--plan", "plan.json", "--sha256", _sha(plan["sha256"])],
    )


def preflight(plan: dict[str, Any]) -> dict[str, Any]:
    validate_plan(plan, check_files=True)
    request = job_request(plan)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "export_receipt_sha256": plan["artifact"]["export_receipt_sha256"],
        "reload_acceptance_receipt_sha256": plan["artifact"]["reload_acceptance_receipt_sha256"],
        "execution_contract_sha256": plan["serving"]["execution_contract_sha256"],
        "submitted": False,
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("local SGLang redirect refused")


def _request(path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = "http://127.0.0.1:8000" + path
    if not url.startswith("http://127.0.0.1:8000/"):
        raise ValueError("only the local canary server may be probed")
    request = urllib.request.Request(
        url,
        method="POST" if body is not None else "GET",
        data=None if body is None else canonical_json(body).encode(),
        headers={} if body is None else {"Content-Type": "application/json"},
    )
    with urllib.request.build_opener(_NoRedirect).open(
        request, timeout=REQUEST_TIMEOUT_SECONDS
    ) as response:
        if response.status != 200:
            raise RuntimeError("local SGLang probe returned non-200")
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("local SGLang probe returned a non-object")
    return value


def _choice(value: dict[str, Any], model_id: str) -> dict[str, Any]:
    choices = value.get("choices")
    if value.get("model") != model_id or not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("chat completion model/choice identity mismatch")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("chat completion is malformed")
    usage = value.get("usage")
    if not isinstance(usage, dict) or type(usage.get("completion_tokens")) is not int:
        raise ValueError("chat completion lacks finite generation usage")
    if usage["completion_tokens"] < 1:
        raise ValueError("chat completion produced no token")
    for number in usage.values():
        if isinstance(number, float) and not math.isfinite(number):
            raise ValueError("chat completion usage is non-finite")
    return choice["message"]


def _probes(model_id: str) -> dict[str, Any]:
    models = _request("/v1/models")
    data = models.get("data")
    if not isinstance(data, list) or [row.get("id") for row in data if isinstance(row, dict)] != [
        model_id
    ]:
        raise ValueError("local model catalog differs from exact served identity")
    common = {
        "model": model_id,
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    forward = _request(
        "/v1/chat/completions",
        {**common, "messages": [{"role": "user", "content": "Reply briefly."}]},
    )
    _choice(forward, model_id)
    continuation = _request(
        "/v1/chat/completions",
        {
            **common,
            "messages": [
                {"role": "user", "content": "Reply briefly."},
                {"role": "assistant", "content": "OK"},
                {"role": "user", "content": "Continue briefly."},
            ],
        },
    )
    _choice(continuation, model_id)
    tool = _request(
        "/v1/chat/completions",
        {
            **common,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": "Call identity exactly once with value parity.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "identity",
                        "description": "Return the supplied public test value.",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "identity"}},
        },
    )
    message = _choice(tool, model_id)
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("forced structured tool call is absent")
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict) or function.get("name") != "identity":
        raise ValueError("forced structured tool name differs")
    arguments = json.loads(function.get("arguments", ""))
    if arguments != {"value": "parity"}:
        raise ValueError("forced structured tool arguments differ")
    return {
        "model_list_exact": True,
        "forward_completion_tokens": forward["usage"]["completion_tokens"],
        "continuation_completion_tokens": continuation["usage"]["completion_tokens"],
        "tool_name": "identity",
        "tool_argument_keys": ["value"],
        "response_content_recorded": False,
        "task_content_included": False,
        "benchmark_content_included": False,
        "scores_observed": False,
    }


def _wait_ready(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("SGLang exited before readiness")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(5)
    raise TimeoutError("SGLang startup allowance expired")


def _stop(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is not None:
        raise RuntimeError("SGLang exited before controlled shutdown")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    if process.poll() is None:
        raise RuntimeError("SGLang process group did not terminate")
    return process.returncode


def _verify_runtime_payload(plan: dict[str, Any]) -> None:
    root = Path(plan["serving"]["model_root"])
    files = plan["artifact"]["files"]
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != set(files) | {"EXPORT.json"}
    ):
        raise ValueError("runtime Miles export inventory differs")
    if file_sha256(root / "EXPORT.json") != plan["artifact"]["export_file_sha256"]:
        raise ValueError("runtime Miles export receipt changed")
    for name, expected in files.items():
        path = root / name
        if (
            Path(name).name != name
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or file_sha256(path) != _sha(expected["sha256"])
        ):
            raise ValueError("runtime Miles export payload changed")


def run(plan_path: Path, plan_sha256: str) -> dict[str, Any]:
    plan = _read(plan_path)
    validate_plan(plan, check_files=False)
    if _sha(plan["sha256"]) != _sha(plan_sha256):
        raise ValueError("runtime plan digest mismatch")
    root = Path(plan["output_root"])
    if os.environ.get("RUN_DIR") != str(root):
        raise ValueError("Jobs API run directory differs from plan")
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != {".runtime"}
    ):
        raise ValueError("dev canary output is not a fresh Jobs API-owned directory")
    api_run_name = os.environ.get("FLEET_RUN_NAME", "")
    api_run_id = os.environ.get("FLEET_RUN_ID", "")
    if not _API_RUN.fullmatch(api_run_name) or str(uuid.UUID(api_run_id)) != api_run_id:
        raise ValueError("Jobs API runtime identity is malformed")
    command = [*plan["serving"]["command"], *plan["serving"]["args"]]
    log_path = root / "private-sglang.log"
    process: subprocess.Popen[bytes] | None = None
    try:
        _verify_runtime_payload(plan)
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_ready(process)
            probes = _probes(plan["run_name"])
        server_exit_code = _stop(process)
        _verify_runtime_payload(plan)
        result = _sign(
            {
                "schema": RESULT_SCHEMA,
                "status": "passed",
                "plan_sha256": _sha(plan["sha256"]),
                "request_sha256": digest_json(job_request(plan, check_files=False)),
                "api_run_name": api_run_name,
                "api_run_id": api_run_id,
                "execution_contract_sha256": plan["serving"]["execution_contract_sha256"],
                "export_receipt_sha256": plan["artifact"]["export_receipt_sha256"],
                "reload_acceptance_receipt_sha256": plan["artifact"][
                    "reload_acceptance_receipt_sha256"
                ],
                "source_update_identity_sha256": plan["artifact"]["source_update_identity_sha256"],
                "staged_manifest_sha256": plan["artifact"]["staged_manifest_sha256"],
                "requested_runtime_image": plan["serving"]["image"],
                "checks": {key: True for key in DEV_CHECKS},
                "probes": probes,
                "server_pid": process.pid,
                "server_exit_code": server_exit_code,
                "server_process_group_stopped": True,
                "source_export_unchanged": True,
                "optimizer_updates": 0,
                "rollouts": 0,
                "verifier_calls": 0,
                "benchmark_attempts": 0,
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        _once(root / "DEV_SERVING_RESULT.json", result)
        return result
    except Exception as error:
        if process is not None:
            with suppress(Exception):
                _stop(process)
        failure = _sign(
            {
                "schema": "cyber_miles_serving_dev_failure_v1",
                "status": "failed",
                "plan_sha256": _sha(plan["sha256"]),
                "error_type": type(error).__name__,
                "private_log_path": str(log_path),
                "task_content_included": False,
                "benchmark_content_included": False,
                "scores_observed": False,
                "failed_at": datetime.now(UTC).isoformat(),
            }
        )
        _once(root / "DEV_SERVING_FAILED.json", failure)
        raise


def accept(
    plan: dict[str, Any],
    result_path: Path,
    result_file_sha256: str,
    external_path: Path,
    external_file_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Create the receipt consumed by production registration, without a POST."""
    validate_plan(plan, check_files=True)
    result = _sealed(_read(result_path, result_file_sha256), RESULT_SCHEMA)
    external = _read(external_path, external_file_sha256)
    controller = external.get("controller")
    pod = external.get("pod")
    release = external.get("release")
    probes = result.get("probes")
    expected_image = plan["serving"]["image"]
    if (
        set(result) != RESULT_FIELDS
        or set(external) != EXTERNAL_FIELDS
        or external.get("schema") != EXTERNAL_SCHEMA
        or result.get("status") != "passed"
        or result.get("plan_sha256") != _sha(plan["sha256"])
        or result.get("request_sha256") != digest_json(job_request(plan))
        or result.get("execution_contract_sha256") != plan["serving"]["execution_contract_sha256"]
        or result.get("export_receipt_sha256") != plan["artifact"]["export_receipt_sha256"]
        or result.get("reload_acceptance_receipt_sha256")
        != plan["artifact"]["reload_acceptance_receipt_sha256"]
        or result.get("source_update_identity_sha256")
        != plan["artifact"]["source_update_identity_sha256"]
        or result.get("staged_manifest_sha256") != plan["artifact"]["staged_manifest_sha256"]
        or result.get("checks") != {key: True for key in DEV_CHECKS}
        or result.get("optimizer_updates") != 0
        or result.get("rollouts") != 0
        or result.get("verifier_calls") != 0
        or result.get("benchmark_attempts") != 0
        or result.get("server_process_group_stopped") is not True
        or result.get("source_export_unchanged") is not True
        or type(result.get("server_pid")) is not int
        or result["server_pid"] < 1
        or result.get("server_exit_code") not in {0, -signal.SIGTERM, -signal.SIGKILL}
        or result.get("requested_runtime_image") != expected_image
        or not isinstance(probes, dict)
        or set(probes) != PROBE_FIELDS
        or probes.get("model_list_exact") is not True
        or type(probes.get("forward_completion_tokens")) is not int
        or probes["forward_completion_tokens"] < 1
        or type(probes.get("continuation_completion_tokens")) is not int
        or probes["continuation_completion_tokens"] < 1
        or probes.get("tool_name") != "identity"
        or probes.get("tool_argument_keys") != ["value"]
        or any(
            probes.get(key) is not False
            for key in (
                "response_content_recorded",
                "task_content_included",
                "benchmark_content_included",
                "scores_observed",
            )
        )
        or not isinstance(controller, dict)
        or not isinstance(pod, dict)
        or not isinstance(release, dict)
        or set(controller) != CONTROLLER_FIELDS
        or set(pod) != POD_FIELDS
        or set(release) != RELEASE_FIELDS
        or external.get("status") != "succeeded_released"
        or external.get("cluster") != "dev"
        or external.get("api_base_url") != API_URLS["dev"]
        or external.get("kube_context") != DEV_CONTEXT
        or external.get("namespace") != NAMESPACE
        or external.get("run_name") != result["api_run_name"]
        or external.get("api_run_id") != result["api_run_id"]
        or external.get("request_sha256") != result["request_sha256"]
        or external.get("effective_priority") != 10000
        or external.get("automatic_requeue") is not False
        or controller.get("kind") != "RayJob"
        or controller.get("name") != result["api_run_name"]
        or controller.get("status") != "SUCCEEDED"
        or not isinstance(pod.get("name"), str)
        or not pod["name"]
        or pod.get("phase") != "Succeeded"
        or pod.get("exit_code") != 0
        or pod.get("container_restarts") != 0
        or pod.get("gpus") != 1
        or pod.get("owner_raycluster_uid") != controller.get("raycluster_uid")
        or pod.get("termination_reason") != "Completed"
        or _runtime_image_digest(pod.get("runtime_image_id"))
        != expected_image.rsplit("@sha256:", 1)[-1]
        or release
        != {
            "api_status": "SUCCEEDED",
            "raycluster_present": False,
            "workload_present": False,
            "gpu_pods_present": False,
            "active_gpus": 0,
            "observed_at": release.get("observed_at"),
        }
        or external.get("observed_at") != release.get("observed_at")
    ):
        raise ValueError("Miles serving dev result/release evidence is incomplete")
    for value in (
        external.get("namespace_uid"),
        result["api_run_id"],
        controller.get("uid"),
        controller.get("workload_uid"),
        controller.get("raycluster_uid"),
        pod.get("uid"),
    ):
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            raise ValueError("Miles serving dev UID is not canonical")
    if not _API_RUN.fullmatch(str(result.get("api_run_name", ""))):
        raise ValueError("Miles serving API run name is malformed")
    timestamps = []
    for value in (
        result.get("completed_at"),
        pod.get("terminated_at"),
        external.get("observed_at"),
        release.get("observed_at"),
    ):
        if (
            not isinstance(value, str)
            or datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None
        ):
            raise ValueError("Miles serving observation lacks timezone")
        timestamps.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
    if timestamps[1] < timestamps[0] or timestamps[2] < timestamps[1]:
        raise ValueError("release observation predates runtime completion")
    evidence_manifest = {
        "plan_sha256": _sha(plan["sha256"]),
        "runtime_result_file_sha256": _sha(result_file_sha256),
        "runtime_result_receipt_sha256": _sha(result["sha256"]),
        "external_file_sha256": _sha(external_file_sha256),
    }
    qualification = {
        "schema": DEV_QUALIFICATION_SCHEMA,
        "status": "passed",
        "cluster": "dev",
        "api_base_url": API_URLS["dev"],
        "execution_contract_sha256": plan["serving"]["execution_contract_sha256"],
        "export_receipt_sha256": plan["artifact"]["export_receipt_sha256"],
        "source_update_identity_sha256": plan["artifact"]["source_update_identity_sha256"],
        "reload_acceptance_receipt_sha256": plan["artifact"]["reload_acceptance_receipt_sha256"],
        "staged_manifest_sha256": plan["artifact"]["staged_manifest_sha256"],
        "checks": {key: True for key in DEV_CHECKS},
        "controller_uid": controller["uid"],
        "pod_uid": pod["uid"],
        "observed_at": external["observed_at"],
        "runtime_image_id": pod["runtime_image_id"],
        "evidence_manifest": evidence_manifest,
        "evidence_manifest_sha256": digest_json(evidence_manifest),
        "optimizer_updates": 0,
        "rollouts": 0,
        "verifier_calls": 0,
        "benchmark_attempts": 0,
        "gpu_release_verified": True,
        "serving_ready": False,
        "production_registration_executed": False,
    }
    signed = {**qualification, "receipt_sha256": digest_json(qualification)}
    _once(output, signed)
    return signed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run(args.plan, args.sha256)
        else:  # pragma: no cover - argparse owns this boundary
            raise ValueError("unknown command")
        print(canonical_json({"status": result["status"], "sha256": result["sha256"]}))
        return 0
    except Exception as error:
        print(canonical_json({"error_type": type(error).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
