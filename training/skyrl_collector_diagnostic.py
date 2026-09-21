"""One bounded SkyRL collector diagnostic, deliberately outside the trainer.

This module is a repair instrument for a failed RL collection, not an RL
recipe.  It starts the exact pinned image-native inference client, runs one
frozen episode through ``rl_episode.collect``, and records only a small safe
outcome enum.  It never calls SkyRL's trainer, optimizer, checkpoint writer or
W&B.  The diagnostic is dev-only and has no preview or create operation: a
separate caller must obtain a real Jobs API preview and arm the exact UID-bound
release observer before a one-shot generic Jobs API create can be considered.

``collect`` deliberately writes private trajectory artifacts, so this wrapper
uses a local ephemeral directory and removes it before writing its single
sanitized SFS receipt.  Do not replace that directory with the run output.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import hashlib
import importlib
import json
import os
import re
import signal
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import yaml

from cyber_post_train.jobs import (
    API_URLS,
    FAILURE_ALERT_ANNOTATION,
    JobsError,
    bundled_request,
    digest,
    quantity,
)
from evals.fleet import opencode_self_hosted as fleet

from . import rl_episode, skyrl, skyrl_episode, skyrl_training

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cyber_skyrl_collector_diagnostic_v1"
CONFIG_SCHEMA = "cyber_skyrl_collector_diagnostic_config_v1"
OFFLINE_PREVIEW_SCHEMA = "cyber_skyrl_collector_diagnostic_offline_preview_v1"
SERVER_PREVIEW_SCHEMA = "cyber_skyrl_collector_diagnostic_server_preview_v1"
OBSERVER_CONTRACT_SCHEMA = "cyber_skyrl_collector_diagnostic_release_contract_v1"
RECEIPT_SCHEMA = "cyber_skyrl_collection_diagnostic_v1"

MODULE = "training.skyrl_collector_diagnostic"
IMAGE = skyrl_training.IMAGE
MAXIMUM_SECONDS = 30 * 60
# The Kubernetes observer owns the full 30-minute allocation.  Keep a full
# minute outside this process for the observer's terminal transition and give
# the wrapper explicit setup, collection, and local-engine-cleanup envelopes.
# These are not training settings: they only bound this one diagnostic attempt.
SETUP_SECONDS = 10 * 60
EPISODE_SECONDS = 14 * 60
CLEANUP_SECONDS = 5 * 60
PROCESS_SECONDS = SETUP_SECONDS + EPISODE_SECONDS + CLEANUP_SECONDS
RUNTIME_UID, RUNTIME_GID = 1000, 100
RECEIPT_NAME = "RL_DIAGNOSTIC.json"
STARTED_NAME = "STARTED.json"
STARTED_SCHEMA = "cyber_skyrl_collector_diagnostic_started_v1"
ALLOWED_WANDB_ENV = frozenset(
    {"WANDB_MODE", "WANDB_DISABLED", "WANDB_DISABLE_CODE", "WANDB_CONSOLE"}
)

# This is intentionally no broader than the pre-existing safe receipt enum in
# rl_episode._failure.  A new code needs an explicit review/change here; never
# persist arbitrary exception arguments merely because they happen to be text.
ALLOWED_LEAF_REASONS = frozenset(
    {
        "generation_incomplete",
        "generation_incomplete_length",
        "generation_incomplete_context_full",
        "generation_incomplete_aborted",
        "generation_incomplete_nontext",
        "generation_transport_failure",
        "generation_invalid_json",
        "generation_finish_invalid",
        "tool_parser_contract_invalid",
        "non_text_tool_result",
        "tool_error_status_missing",
        "tool_result_exceeds_budget",
        "bash_timeout_maximum_unresolved",
        "tool_timeout_below_advertised_budget",
        "turn_response_budget_exhausted",
        "turn_budget_exhausted",
        "response_budget_exhausted",
        "instance_release_unconfirmed",
    }
)

SETUP_MODULE = "skyrl.backends.skyrl_train.inference_servers.setup"
SETUP_BINDING = "setup"
REMOTE_CLIENT_BINDING = "remote_inference_client"
GENERATOR_HELPER_BINDING = "generator_utils"
CONFIG_BINDING = "native_config"
UTILS_BINDING = "native_utils"


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _sealed(value: object, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema or value != _seal(value):
        raise JobsError("collector diagnostic receipt digest changed")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise JobsError(f"collector diagnostic {field} must be a sha256 digest")
    return value


def _path(value: object, *, field: str, root: str | None = None) -> str:
    if not isinstance(value, str):
        raise JobsError(f"collector diagnostic {field} is invalid")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or any(ord(char) < 32 for char in value)
        or (root is not None and not path.is_relative_to(PurePosixPath(root)))
    ):
        raise JobsError(f"collector diagnostic {field} is not a canonical path")
    return value


def _runtime_files() -> dict[str, str]:
    """Return the diagnostic wrapper plus unchanged legacy source bytes."""
    files = dict(skyrl_training._runtime())
    path = ROOT / "training/skyrl_collector_diagnostic.py"
    files["training/skyrl_collector_diagnostic.py"] = path.read_text()
    return files


def _legacy_runtime() -> dict[str, str]:
    return dict(skyrl_training._runtime())


def _module_sha256(module: object) -> str:
    path = getattr(module, "__file__", None)
    if not isinstance(path, str):
        raise rl_episode.InvalidEpisode("native_module_file_unavailable")
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise rl_episode.InvalidEpisode("native_module_file_unavailable") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _bound_module(plan: Mapping[str, Any], binding: str, module_name: str) -> Any:
    expected = plan["image_native_sources"][binding]
    module = importlib.import_module(module_name)
    if _module_sha256(module) != expected:
        raise rl_episode.InvalidEpisode("native_image_source_drift")
    return module


def _source_plan(config: Mapping[str, Any], relative_to: Path) -> dict[str, Any]:
    path_value = config.get("source_plan")
    if not isinstance(path_value, str) or not path_value:
        raise JobsError("collector diagnostic source_plan is required")
    raw = relative_to / path_value
    if raw.is_symlink():
        raise JobsError("collector diagnostic source_plan escapes its config directory")
    candidate = raw.resolve()
    if not candidate.is_relative_to(relative_to.resolve()):
        raise JobsError("collector diagnostic source_plan escapes its config directory")
    try:
        value = json.loads(candidate.read_bytes())
    except (OSError, ValueError) as exc:
        raise JobsError("collector diagnostic source_plan is unreadable") from exc
    if not isinstance(value, dict):
        raise JobsError("collector diagnostic source_plan must be an object")
    # This is a local source/runtime contract check only.  It does not contact
    # Fleet, start Ray, create an environment, preview or submit a workload.
    skyrl_training.job_request(value)
    return value


def _known(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    if set(value) != allowed:
        raise JobsError(f"collector diagnostic {label} fields changed")


def _image_native_sources(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise JobsError("collector diagnostic image native sources are missing")
    expected = {
        SETUP_BINDING,
        REMOTE_CLIENT_BINDING,
        GENERATOR_HELPER_BINDING,
        CONFIG_BINDING,
        UTILS_BINDING,
    }
    _known(value, expected, "image native source")
    return {
        name: _sha256(item, field=f"image native source {name}") for name, item in value.items()
    }


def _source_image_native_sources(source: Mapping[str, Any], value: object) -> dict[str, str]:
    """Bind wrapper-owned image modules to the frozen training plan.

    The separate setup module is not part of the old training-plan schema, so
    its digest is supplied explicitly and checked again inside the image.  The
    existing config, utility, recorder, and client digests must agree with the
    byte-pinned legacy sources; this wrapper cannot silently substitute them.
    """
    result = _image_native_sources(value)
    native = source.get("native_sources")
    if not isinstance(native, Mapping):
        raise JobsError("collector diagnostic source native bindings are missing")
    expected = {
        CONFIG_BINDING: native.get("skyrl.train.config.config"),
        UTILS_BINDING: native.get("skyrl.train.utils.utils"),
        GENERATOR_HELPER_BINDING: native.get("skyrl.train.generators.utils"),
        REMOTE_CLIENT_BINDING: skyrl_episode.CLIENT_SHA256,
    }
    if any(
        not isinstance(source_digest, str) or result[binding] != "sha256:" + source_digest
        for binding, source_digest in expected.items()
    ):
        raise JobsError("collector diagnostic image bindings differ from the pinned runtime")
    return result


def _selection(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobsError("collector diagnostic selection is missing")
    _known(
        value,
        {"split", "row_index", "line_sha256", "source_config_sha256"},
        "selection",
    )
    if value["split"] != "train" or type(value["row_index"]) is not int or value["row_index"] < 0:
        raise JobsError("collector diagnostic must select one nonnegative train row")
    return {
        "split": "train",
        "row_index": value["row_index"],
        "line_sha256": _sha256(value["line_sha256"], field="selection line"),
        "source_config_sha256": _sha256(
            value["source_config_sha256"], field="selection source config"
        ),
    }


def _diagnostic_arguments(
    source: Mapping[str, Any], *, name: str, output_root: str
) -> dict[str, Any]:
    original = skyrl.SkyRLConfig(**source["arguments"])
    original.validate()
    # The native config object requires identifiers even though this route never
    # initializes W&B.  These fixed placeholders prevent the source run's W&B
    # identity from being carried into the diagnostic plan or environment.
    values = dataclasses.asdict(original)
    values.update(
        {
            "name": name,
            "output_root": output_root,
            "nodes": 1,
            "steps": 1,
            "wandb_entity": "disabled",
            "wandb_project": "disabled",
            "wandb_run_id": "disabled",
        }
    )
    result = skyrl.SkyRLConfig(**values)
    result.validate()
    return dataclasses.asdict(result)


def compile_diagnostic(config: Mapping[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Freeze a dev-only, one-episode collector diagnostic without execution."""
    if not isinstance(config, Mapping) or config.get("schema") != CONFIG_SCHEMA:
        raise JobsError("collector diagnostic config schema changed")
    _known(
        config,
        {"schema", "source_plan", "name", "output_root", "selection", "image_native_sources"},
        "config",
    )
    name, output_root = config["name"], config["output_root"]
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", name):
        raise JobsError("collector diagnostic name is invalid")
    output_root = _path(output_root, field="output root", root="/mnt/sfs/jobs")
    source = _source_plan(config, relative_to)
    if (
        source.get("schema") != skyrl_training.SCHEMA
        or source["execution"].get("image") != IMAGE
        or source["execution"].get("priority") != "c1"
        or source.get("runtime_sha256") != digest(_legacy_runtime())
    ):
        raise JobsError("collector diagnostic source training plan is not the pinned SkyRL runtime")
    if output_root == source.get("output_root") or name == source.get("run_name"):
        raise JobsError("collector diagnostic must use a new output and run identity")
    args = _diagnostic_arguments(source, name=name, output_root=output_root)
    if args["model"] != "Qwen/Qwen3.8-27B":
        raise JobsError("collector diagnostic model changed")
    resources = source["execution"].get("resources")
    if not isinstance(resources, dict):
        raise JobsError("collector diagnostic resources are missing")
    plan = {
        "schema": SCHEMA,
        "run_name": name,
        "output_root": output_root,
        "source_training_plan_sha256": "sha256:" + digest(source),
        "legacy_runtime_sha256": "sha256:" + digest(_legacy_runtime()),
        "diagnostic_runtime_sha256": "sha256:" + digest(_runtime_files()),
        "legacy_native_sources": copy.deepcopy(source["native_sources"]),
        "model": copy.deepcopy(source["model"]),
        "data": {
            "manifest_sha256": _sha256(source["data"]["sha256"], field="data manifest"),
            "template_sha256": _sha256(source["data"]["template_sha256"], field="data template"),
            "manifest_path": args["data_manifest"],
            "train_path": args["train_data"],
        },
        "arguments": args,
        "selection": _selection(config["selection"]),
        "sampling_sha256": "sha256:"
        + digest(skyrl.overrides(skyrl.SkyRLConfig(**args))["generator.sampling_params"]),
        "image_native_sources": _source_image_native_sources(
            source, config["image_native_sources"]
        ),
        "execution": {
            "cluster_target": "dev",
            "jobs_api_base_url": API_URLS["dev"],
            "image": IMAGE,
            "priority": "c1",
            "resources": copy.deepcopy(resources),
            "maximum_seconds": MAXIMUM_SECONDS,
        },
        "deadlines": {
            "setup_seconds": SETUP_SECONDS,
            "episode_seconds": EPISODE_SECONDS,
            "cleanup_seconds": CLEANUP_SECONDS,
            "process_seconds": PROCESS_SECONDS,
        },
        "scientific_work": {
            "task_rows": 1,
            "rollout_episodes": 1,
            "optimizer_steps": 0,
            "checkpoints": 0,
            "wandb_events": 0,
        },
        "qualification": {
            "submission_gate": {
                "preview_authorized": False,
                "submission_authorized": False,
                "blockers": [
                    "fresh_server_preview_receipt_required",
                    "uid_bound_30_minute_observer_required",
                    "separate_one_shot_generic_jobs_api_creator_required",
                ],
            }
        },
    }
    _validate_plan(plan)
    job_request(plan)
    return plan


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping) or plan.get("schema") != SCHEMA:
        raise JobsError("collector diagnostic plan schema changed")
    _known(
        plan,
        {
            "schema",
            "run_name",
            "output_root",
            "source_training_plan_sha256",
            "legacy_runtime_sha256",
            "diagnostic_runtime_sha256",
            "legacy_native_sources",
            "model",
            "data",
            "arguments",
            "selection",
            "sampling_sha256",
            "image_native_sources",
            "execution",
            "deadlines",
            "scientific_work",
            "qualification",
        },
        "plan",
    )
    if (
        not isinstance(plan["run_name"], str)
        or not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", plan["run_name"])
        or _path(plan["output_root"], field="output root", root="/mnt/sfs/jobs")
        != plan["output_root"]
    ):
        raise JobsError("collector diagnostic plan identity is invalid")
    for field in (
        "source_training_plan_sha256",
        "legacy_runtime_sha256",
        "diagnostic_runtime_sha256",
        "sampling_sha256",
    ):
        _sha256(plan[field], field=field)
    if plan["legacy_runtime_sha256"] != "sha256:" + digest(_legacy_runtime()) or plan[
        "diagnostic_runtime_sha256"
    ] != "sha256:" + digest(_runtime_files()):
        raise JobsError("collector diagnostic runtime bytes drifted")
    arguments = skyrl.SkyRLConfig(**plan["arguments"])
    arguments.validate()
    if (
        arguments.name != plan["run_name"]
        or arguments.output_root != plan["output_root"]
        or arguments.nodes != 1
        or arguments.wandb_entity != "disabled"
        or arguments.wandb_project != "disabled"
        or arguments.wandb_run_id != "disabled"
    ):
        raise JobsError("collector diagnostic arguments changed")
    data = plan["data"]
    if not isinstance(data, Mapping) or set(data) != {
        "manifest_sha256",
        "template_sha256",
        "manifest_path",
        "train_path",
    }:
        raise JobsError("collector diagnostic data binding changed")
    for field in ("manifest_sha256", "template_sha256"):
        _sha256(data[field], field=f"data {field}")
    if (
        data["manifest_path"] != arguments.data_manifest
        or data["train_path"] != arguments.train_data
    ):
        raise JobsError("collector diagnostic data path drift")
    _selection(plan["selection"])
    if plan["legacy_native_sources"] != skyrl_training.NATIVE:
        raise JobsError("collector diagnostic legacy native source bindings changed")
    image_sources = _image_native_sources(plan["image_native_sources"])
    expected_image_sources = {
        CONFIG_BINDING: skyrl_training.NATIVE["skyrl.train.config.config"],
        UTILS_BINDING: skyrl_training.NATIVE["skyrl.train.utils.utils"],
        GENERATOR_HELPER_BINDING: skyrl_training.NATIVE["skyrl.train.generators.utils"],
        REMOTE_CLIENT_BINDING: skyrl_episode.CLIENT_SHA256,
    }
    if any(
        image_sources[binding] != "sha256:" + expected
        for binding, expected in expected_image_sources.items()
    ):
        raise JobsError("collector diagnostic image source bindings changed")
    execution = plan["execution"]
    if not isinstance(execution, Mapping) or execution != {
        "cluster_target": "dev",
        "jobs_api_base_url": API_URLS["dev"],
        "image": IMAGE,
        "priority": "c1",
        "resources": execution.get("resources"),
        "maximum_seconds": MAXIMUM_SECONDS,
    }:
        raise JobsError(
            "collector diagnostic must remain dev-only c1 with a 30-minute observer limit"
        )
    if plan["deadlines"] != {
        "setup_seconds": SETUP_SECONDS,
        "episode_seconds": EPISODE_SECONDS,
        "cleanup_seconds": CLEANUP_SECONDS,
        "process_seconds": PROCESS_SECONDS,
    }:
        raise JobsError("collector diagnostic process deadlines changed")
    scientific = plan["scientific_work"]
    if scientific != {
        "task_rows": 1,
        "rollout_episodes": 1,
        "optimizer_steps": 0,
        "checkpoints": 0,
        "wandb_events": 0,
    }:
        raise JobsError("collector diagnostic scientific boundary changed")
    gate = (
        plan["qualification"].get("submission_gate")
        if isinstance(plan["qualification"], Mapping)
        else None
    )
    if (
        not isinstance(gate, Mapping)
        or gate.get("preview_authorized") is not False
        or gate.get("submission_authorized") is not False
        or gate.get("blockers")
        != [
            "fresh_server_preview_receipt_required",
            "uid_bound_30_minute_observer_required",
            "separate_one_shot_generic_jobs_api_creator_required",
        ]
    ):
        raise JobsError("collector diagnostic launch gates changed")


def _bundle_files(plan: Mapping[str, Any]) -> dict[str, str]:
    # Never mutate a caller's frozen runtime inventory.  The validator hashes
    # that inventory again before preview/create, so adding bundle-only
    # package markers or plan.json in place would turn a valid immutable plan
    # into apparent source drift.
    files = dict(_runtime_files())
    files.update(
        {
            path + "/__init__.py": ""
            for path in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return files


def job_request(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build a generic dev request, but never preview or create it."""
    _validate_plan(plan)
    request = bundled_request(
        {
            "name": plan["run_name"],
            "title": plan["run_name"] + " collector-only diagnostic",
            "run_dir": plan["output_root"],
            "image": IMAGE,
            "workers": 1,
            "gpus_per_worker": 8,
            "resources": plan["execution"]["resources"],
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "failureAlerts": False,
            "secrets": ["fleet-api"],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "WANDB_MODE": "disabled",
                "WANDB_DISABLED": "true",
                "WANDB_DISABLE_CODE": "true",
                "WANDB_CONSOLE": "off",
                # The runtime directory is a sealed bundle.  Never let
                # Python add bytecode artifacts that would make it impossible
                # to distinguish code drift from normal import side effects.
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "CYBER_EXPECTED_RUNTIME_UID": str(RUNTIME_UID),
                "CYBER_EXPECTED_RUNTIME_GID": str(RUNTIME_GID),
                "CYBER_COLLECTOR_DIAGNOSTIC": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
            },
        },
        _bundle_files(plan),
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )
    if "wandb-api" in request["secrets"] or any(
        key.startswith("WANDB_")
        and key not in {"WANDB_MODE", "WANDB_DISABLED", "WANDB_DISABLE_CODE", "WANDB_CONSOLE"}
        for key in request["env"]
    ):
        raise JobsError("collector diagnostic must not receive a W&B identity or secret")
    return request


def offline_preview(plan: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """Record local intent only; it is explicitly not a server preview."""
    if job_request(plan) != request:
        raise JobsError("collector diagnostic request changed")
    return _seal(
        {
            "schema": OFFLINE_PREVIEW_SCHEMA,
            "status": "locally_rendered_not_server_previewed",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "cluster_target": "dev",
            "submitted": False,
            "preview_authorized": False,
            "create_authorized": False,
            "external_reads": 0,
            "external_mutations": 0,
        }
    )


def validate_server_preview(
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    preview: Mapping[str, Any],
    *,
    api_base_url: str,
) -> dict[str, Any]:
    """Validate a *caller-supplied real* Jobs API server render.

    This function deliberately makes no HTTP request.  The only supported caller
    is a one-shot creator that obtained ``preview`` from ``Jobs.preview`` in the
    same controlled launch transaction.  The returned proof is necessary, not
    sufficient, for a create.
    """
    if job_request(plan) != request or api_base_url != plan["execution"]["jobs_api_base_url"]:
        raise JobsError("collector diagnostic request changed before server preview")
    result = skyrl_training.validate_gpu_runtime_preview(dict(request), dict(preview))
    try:
        rendered = yaml.safe_load(preview["manifest_yaml"])
        metadata, spec = rendered["metadata"], rendered["spec"]
        annotations = metadata["annotations"]
        if (
            not isinstance(rendered, Mapping)
            or not isinstance(metadata, Mapping)
            or not isinstance(spec, Mapping)
            or not isinstance(annotations, Mapping)
            or rendered["kind"] != "RayJob"
            or annotations.get(FAILURE_ALERT_ANNOTATION) != "off"
            or spec.get("shutdownAfterJobFinishes") is not True
            or spec.get("backoffLimit") != 0
            or spec.get("activeDeadlineSeconds") not in (None, MAXIMUM_SECONDS)
        ):
            raise JobsError("collector diagnostic server preview is not safe")
        cluster = spec["rayClusterSpec"]
        templates = [cluster["headGroupSpec"]["template"]] + [
            group["template"]
            for group in cluster.get("workerGroupSpecs", [])
            if group.get("replicas", 0)
        ]
        containers = []
        for template in templates:
            pod = template["spec"]
            containers.extend(pod.get("initContainers", []))
            containers.extend(pod["containers"])
        if not containers:
            raise JobsError("collector diagnostic server preview has unexpected containers")
        for container in containers:
            raw_entries = container.get("env", [])
            raw_env_from = container.get("envFrom", [])
            if (
                not isinstance(raw_entries, list)
                or not isinstance(raw_env_from, list)
                or any(
                    not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str)
                    for entry in raw_entries
                )
                or any(not isinstance(row, Mapping) for row in raw_env_from)
            ):
                raise JobsError("collector diagnostic preview has malformed environment")
            entries = {entry["name"]: entry.get("value") for entry in raw_entries}
            secret_names = []
            for row in raw_env_from:
                secret = row.get("secretRef")
                if (
                    not isinstance(secret, Mapping)
                    or secret.get("name") != "fleet-api"
                    or secret.get("optional", False) is not False
                    or row.get("prefix", "") != ""
                ):
                    raise JobsError("collector diagnostic preview exposes an unapproved Secret")
                secret_names.append(secret["name"])
            if (
                len(entries) != len(raw_entries)
                or len(set(secret_names)) != len(secret_names)
                or any(
                    key.startswith("WANDB_")
                    and (key not in ALLOWED_WANDB_ENV or entries[key] != request["env"].get(key))
                    for key in entries
                )
            ):
                raise JobsError("collector diagnostic preview exposes W&B")
        head_containers = cluster["headGroupSpec"]["template"]["spec"]["containers"]
        head_gpu = [
            container
            for container in head_containers
            if quantity(container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
            > 0
        ]
        if len(head_gpu) != 1:
            raise JobsError("collector diagnostic server preview has no unique head GPU container")
        head_entries = {entry["name"]: entry.get("value") for entry in head_gpu[0].get("env", [])}
        if (
            head_entries.get("WANDB_MODE") != "disabled"
            or head_entries.get("WANDB_DISABLED") != "true"
            or head_entries.get("WANDB_DISABLE_CODE") != "true"
            or head_entries.get("WANDB_CONSOLE") != "off"
        ):
            raise JobsError("collector diagnostic preview does not disable W&B")
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("collector diagnostic server preview is malformed") from exc
    return _seal(
        {
            "schema": SERVER_PREVIEW_SCHEMA,
            "status": "server_preview_validated_not_create_authorized",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "jobs_api_base_url": api_base_url,
            "manifest_sha256": "sha256:" + digest(rendered),
            "failure_alert_annotation": "off",
            "shutdown_after_job_finishes": True,
            "preview_authorized": False,
            "create_authorized": False,
            "gpus": result["gpus"],
            "runtime_user": result["runtime_user"],
        }
    )


def release_observer_contract(
    plan: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe the only permitted pre/post-create release sequence.

    This is a contract, not an observer and not permission to POST.  The
    concrete observer lives at the Kubernetes boundary because only it can
    bind the API-returned name to a real RayJob UID.
    """
    if job_request(plan) != request:
        raise JobsError("collector diagnostic request changed")
    return _seal(
        {
            "schema": OBSERVER_CONTRACT_SCHEMA,
            "status": "contract_only_not_armed",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "cluster_target": "dev",
            "jobs_api_base_url": plan["execution"]["jobs_api_base_url"],
            "kubernetes_context": "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
            "namespace": "fleet-train-jobs",
            "run_name_prefix": request["name"],
            "server_run_name_pattern": "^" + re.escape(request["name"]) + r"-[a-f0-9]{8}$",
            "run_dir": request["run_dir"],
            "expected_gpus": 8,
            "maximum_seconds": MAXIMUM_SECONDS,
            "arm_before_jobs_post": True,
            "pre_post_prefix_guard_schema": "cyber_jobs_api_prefix_guard_armed_v1",
            "post_post_exact_binding_schema": "cyber_jobs_api_exact_rayjob_binding_v1",
            "exact_observer_schema": "cyber_jobs_api_exact_uid_observer_result_v1",
            "creator_identity_fields": ["jobs_api_run_name", "jobs_api_run_id", "run_dir"],
            "release_acceptance": {
                "terminal_root_observed": True,
                "bound_root_uid_absent": True,
                "observed_child_uids_absent": True,
                "peak_gpus_at_most": 8,
                "raw_delete_requires_separate_creator_contract": True,
            },
        }
    )


def _validate_runtime_user() -> None:
    if (
        os.environ.get("CYBER_EXPECTED_RUNTIME_UID") != str(RUNTIME_UID)
        or os.environ.get("CYBER_EXPECTED_RUNTIME_GID") != str(RUNTIME_GID)
        or (os.geteuid(), os.getegid()) != (RUNTIME_UID, RUNTIME_GID)
    ):
        raise rl_episode.InvalidEpisode("runtime_user_mismatch")


def _validate_destination(plan: Mapping[str, Any]) -> Path:
    root = Path(plan["output_root"])
    if os.environ.get("RUN_DIR") != str(root) or root.is_symlink() or not root.is_dir():
        raise rl_episode.InvalidEpisode("output_root_binding_mismatch")
    runtime = root / ".runtime"
    expected = _bundle_files(plan)
    files = {str(path.relative_to(runtime)): path for path in runtime.rglob("*") if path.is_file()}
    if (
        runtime.is_symlink()
        or not runtime.is_dir()
        or set(root.iterdir()) != {runtime}
        or set(files) != set(expected)
        or any(
            path.is_symlink() or path.read_text() != expected[name] for name, path in files.items()
        )
    ):
        raise rl_episode.InvalidEpisode("diagnostic_bundle_drift")
    return root


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_receipt(root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _seal(_validate_receipt(value))
    _write_once(root / RECEIPT_NAME, receipt)
    return receipt


def _validate_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the deliberately tiny terminal receipt vocabulary."""
    base = {
        "schema",
        "status",
        "diagnostic_plan_sha256",
        "legacy_runtime_sha256",
        "diagnostic_runtime_sha256",
        "image_digest",
        "source_row_sha256",
        "sampling_sha256",
        "episode_attempted",
        "collection_completed",
        "optimizer_steps",
        "checkpoints",
        "wandb_events",
        "fleet_instance_release_confirmed",
        "engine_cleanup_confirmed",
        "external_job_cleanup_required",
        "leaf_reason_count",
    }
    status = value.get("status")
    has_leaf = status == "allowlisted_invalid_episode"
    allowed = base | ({"allowlisted_leaf_reason"} if has_leaf else set())
    if set(value) != allowed or value.get("schema") != RECEIPT_SCHEMA:
        raise JobsError("collector diagnostic receipt fields changed")
    if status not in {
        "collection_completed_no_training",
        "allowlisted_invalid_episode",
        "unclassified_failure",
    }:
        raise JobsError("collector diagnostic receipt status changed")
    for field in (
        "diagnostic_plan_sha256",
        "legacy_runtime_sha256",
        "diagnostic_runtime_sha256",
        "image_digest",
        "source_row_sha256",
        "sampling_sha256",
    ):
        _sha256(value.get(field), field=f"receipt {field}")
    if any(
        type(value.get(field)) is not bool
        for field in (
            "episode_attempted",
            "collection_completed",
            "fleet_instance_release_confirmed",
            "engine_cleanup_confirmed",
            "external_job_cleanup_required",
        )
    ):
        raise JobsError("collector diagnostic receipt boolean changed")
    if (
        value.get("optimizer_steps") != 0
        or value.get("checkpoints") != 0
        or value.get("wandb_events") != 0
        or value.get("external_job_cleanup_required") is not True
    ):
        raise JobsError("collector diagnostic receipt scientific boundary changed")
    if has_leaf:
        if (
            value.get("allowlisted_leaf_reason") not in ALLOWED_LEAF_REASONS
            or value.get("leaf_reason_count") != 1
            or value.get("collection_completed") is not False
            or value.get("fleet_instance_release_confirmed") is not False
        ):
            raise JobsError("collector diagnostic receipt leaf cause changed")
    elif (
        value.get("leaf_reason_count") != 0
        or value.get("collection_completed") is not (status == "collection_completed_no_training")
        or value.get("fleet_instance_release_confirmed")
        is not (status == "collection_completed_no_training")
    ):
        raise JobsError("collector diagnostic receipt terminal state changed")
    if (
        status == "collection_completed_no_training"
        and value.get("engine_cleanup_confirmed") is not True
    ):
        raise JobsError("collector diagnostic success lacks engine cleanup")
    return dict(value)


def _load_selected_config(plan: Mapping[str, Any]) -> dict[str, Any]:
    data, selection = plan["data"], plan["selection"]
    try:
        manifest_bytes = Path(data["manifest_path"]).read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        raise rl_episode.InvalidEpisode("diagnostic_data_unavailable") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("sha256") != data["manifest_sha256"]
        or fleet.digest_without(manifest, "sha256") != data["manifest_sha256"]
        or manifest.get("template_sha256") != data["template_sha256"]
        or manifest.get("files", {}).get("train", {}).get("path") != Path(data["train_path"]).name
    ):
        raise rl_episode.InvalidEpisode("diagnostic_data_manifest_drift")
    try:
        raw = Path(data["train_path"]).read_bytes()
        lines = raw.splitlines()
        line = lines[selection["row_index"]]
        row = json.loads(line)
        source = json.loads(row["cyber_config_json"])
    except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
        raise rl_episode.InvalidEpisode("diagnostic_selected_row_unavailable") from exc
    if (
        "sha256:" + hashlib.sha256(line).hexdigest() != selection["line_sha256"]
        or row.get("split") != "train"
        or source.get("config_sha256") != selection["source_config_sha256"]
        or fleet.digest_without(source, "config_sha256") != selection["source_config_sha256"]
    ):
        raise rl_episode.InvalidEpisode("diagnostic_selected_row_drift")
    source = copy.deepcopy(source)
    source["run_id"] = plan["run_name"] + "-episode"
    # This is a diagnostic-specific upper bound, not a training-data edit.  A
    # source row may permit a much longer environment lease, but the dev
    # collector must leave time for deterministic local and cluster cleanup.
    source["rl"]["episode_seconds"] = min(
        source["rl"]["episode_seconds"], plan["deadlines"]["episode_seconds"]
    )
    source["config_sha256"] = fleet.digest_without(source, "config_sha256")
    rl_episode._validate(source)
    return source


def _allowlisted_leaf_reason(error: BaseException) -> str | None:
    """Return one known leaf only when the entire causal tree is known.

    A generic wrapper beside a known ``InvalidEpisode`` is not evidence for a
    diagnosis.  Likewise, repeated reason codes still represent multiple
    leaves.  This deliberately favors an unclassified receipt over a tempting
    but false explanation.
    """
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    values: list[str] = []
    unknown = False
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            unknown = True
            continue
        seen.add(id(current))
        if len(seen) > 64:
            unknown = True
            break
        children: list[BaseException] = []
        if isinstance(current, BaseExceptionGroup):
            children.extend(
                child for child in current.exceptions if isinstance(child, BaseException)
            )
        for chained in (current.__cause__, current.__context__):
            if isinstance(chained, BaseException):
                children.append(chained)
        if children:
            # A generic exception wrapper is not an allowlisted failure
            # contract.  ExceptionGroup is the one structural wrapper we
            # accept, and only for one child; a group with multiple children
            # would make the stored leaf ambiguous even when their text
            # happens to match.
            if not isinstance(current, BaseExceptionGroup) or len(children) != 1:
                unknown = True
            pending.extend(children)
            continue
        if isinstance(current, rl_episode.InvalidEpisode):
            argument = current.args[0] if current.args else None
            if isinstance(argument, str) and argument in ALLOWED_LEAF_REASONS:
                values.append(argument)
                continue
        unknown = True
    return values[0] if not unknown and len(values) == 1 else None


def _native_config_equivalent(plan: Mapping[str, Any]) -> Any:
    _bound_module(plan, CONFIG_BINDING, "skyrl.train.config.config")
    _bound_module(plan, UTILS_BINDING, "skyrl.train.utils.utils")
    args = skyrl.SkyRLConfig(**plan["arguments"])
    training_cfg = skyrl.native_config(args)
    diagnostic_cfg = skyrl.diagnostic_native_config(args)
    if training_cfg.generator.inference_engine != diagnostic_cfg.generator.inference_engine:
        raise rl_episode.InvalidEpisode("diagnostic_engine_config_drift")
    return diagnostic_cfg


def _local_paths(config: Any, scratch: Path) -> None:
    """Force every known diagnostic log/output route into the ephemeral volume."""
    trainer = getattr(config, "trainer", None)
    if trainer is None:
        raise rl_episode.InvalidEpisode("diagnostic_native_config_invalid")
    for name in ("log_path", "ckpt_path", "export_path"):
        if not hasattr(trainer, name):
            raise rl_episode.InvalidEpisode("diagnostic_native_config_invalid")
        setattr(trainer, name, str(scratch / name))
    if getattr(trainer, "logger", None) != "console":
        raise rl_episode.InvalidEpisode("diagnostic_native_config_invalid")


class _Deadline:
    """A non-nestable, main-thread process deadline for one safe phase."""

    def __init__(self, seconds: int, phase: str):
        self.seconds = seconds
        self.phase = phase

    def __enter__(self) -> _Deadline:
        if (
            self.seconds < 1
            or threading.current_thread() is not threading.main_thread()
            or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0)
            or signal.getsignal(signal.SIGALRM) is signal.SIG_IGN
        ):
            raise rl_episode.InvalidEpisode("diagnostic_deadline_unavailable")
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def _expired(self, *_: object) -> None:
        raise rl_episode.InvalidEpisode("diagnostic_deadline_elapsed")

    def __exit__(self, *_: object) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)


class _EngineSetupRejected(rl_episode.InvalidEpisode):
    """A rejected post-start topology with its teardown result kept private."""

    def __init__(self, reason: str, *, cleanup_confirmed: bool) -> None:
        super().__init__(reason)
        self.cleanup_confirmed = cleanup_confirmed


def _stop_setup(setup: Any, ray: Any) -> bool:
    """Bounded image-engine teardown; no exception detail enters the receipt."""
    if setup is None:
        return True
    clean = True
    router = getattr(setup, "router", None)
    if router is None:
        clean = False
    else:
        try:
            router.shutdown()
        except Exception:
            clean = False
    actors, placements = [], []
    for group in reversed(tuple(getattr(setup, "server_groups", ()) or ())):
        try:
            actors.extend(tuple(group.get_actors()))
        except Exception:
            clean = False
        for name in ("_internal_pg", "_external_pg"):
            value = getattr(group, name, None)
            if value is not None:
                placements.append(getattr(value, "pg", value))
    refs = []
    for actor in actors:
        try:
            refs.append(actor.shutdown.remote())
        except Exception:
            clean = False
    if refs:
        try:
            ray.get(refs, timeout=30)
        except Exception:
            clean = False
    for actor in actors:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            clean = False
    if placements:
        try:
            from ray.util.placement_group import remove_placement_group

            for placement in placements:
                try:
                    remove_placement_group(placement)
                except Exception:
                    clean = False
        except Exception:
            clean = False
    return clean


def _shutdown_ray(ray: Any) -> bool:
    """Stop Ray without turning a failed shutdown into a cleanup claim."""
    try:
        ray.shutdown()
    except Exception:
        return False
    return True


def _start_engine(plan: Mapping[str, Any], cfg: Any, tokenizer: Any, ray: Any) -> tuple[Any, Any]:
    setup_module = _bound_module(plan, SETUP_BINDING, SETUP_MODULE)
    remote_module = _bound_module(
        plan,
        REMOTE_CLIENT_BINDING,
        "skyrl.backends.skyrl_train.inference_servers.remote_inference_client",
    )
    if not callable(getattr(setup_module, "build_new_inference_client", None)):
        raise rl_episode.InvalidEpisode("native_engine_builder_unavailable")
    engine, setup = setup_module.build_new_inference_client(cfg, tokenizer)
    if type(engine) is not getattr(remote_module, "RemoteInferenceClient", None):
        raise _EngineSetupRejected(
            "native_engine_client_drift", cleanup_confirmed=_stop_setup(setup, ray)
        )
    groups = tuple(getattr(setup, "server_groups", ()) or ())
    if len(groups) != 2 or len(tuple(getattr(setup, "server_urls", ()) or ())) != 2:
        raise _EngineSetupRejected(
            "diagnostic_engine_topology_drift", cleanup_confirmed=_stop_setup(setup, ray)
        )
    return engine, setup


async def _collect_once(
    plan: Mapping[str, Any], tokenizer: Any, engine: Any, scratch: Path
) -> None:
    config = _load_selected_config(plan)
    helper = _bound_module(plan, GENERATOR_HELPER_BINDING, "skyrl.train.generators.utils")
    if _module_sha256(helper) != plan["image_native_sources"][GENERATOR_HELPER_BINDING]:
        raise rl_episode.InvalidEpisode("native_image_source_drift")
    sampling = skyrl.overrides(skyrl.SkyRLConfig(**plan["arguments"]))["generator.sampling_params"]
    if "sha256:" + digest(sampling) != plan["sampling_sha256"]:
        raise rl_episode.InvalidEpisode("diagnostic_sampling_drift")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise rl_episode.InvalidEpisode("missing_fleet_auth")
    async with (
        httpx.AsyncClient(
            headers={"Authorization": "Bearer " + key},
            timeout=120,
            transport=httpx.AsyncHTTPTransport(retries=0),
            follow_redirects=False,
        ) as client,
        skyrl_episode.single_attempt_engine(
            engine, tokenizer, config["rl"]["episode_seconds"]
        ) as single_attempt,
    ):
        # Directly call the shared collector.  Generator.generate is prohibited
        # because it deliberately erases child InvalidEpisode causes.  The
        # recorder must receive the yielded one-attempt proxy, not the native
        # client, or native HTTP retries would still be possible.
        recorder = skyrl_episode.Recorder(
            config,
            tokenizer,
            single_attempt,
            sampling,
            plan["arguments"]["response_tokens"],
            Path(helper.__file__),
        )
        await rl_episode.collect(
            config, scratch / "episode", recorder, skyrl_episode.parse, client=client
        )


def run(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Perform one collector attempt and persist only its enum-only receipt."""
    _validate_plan(plan)
    _validate_runtime_user()
    root = _validate_destination(plan)
    _write_once(
        root / STARTED_NAME,
        _seal(
            {
                "schema": STARTED_SCHEMA,
                "status": "started",
                "diagnostic_plan_sha256": "sha256:" + digest(plan),
            }
        ),
    )
    started = time.monotonic()
    engine_setup = None
    ray = None
    engine_cleanup_confirmed: bool | None = None
    collection_completed = False
    episode_attempted = False
    status, leaf = "unclassified_failure", None
    try:
        import ray as ray_module
        from transformers import AutoTokenizer

        ray = ray_module
        with tempfile.TemporaryDirectory(prefix="skyrl-collector-", dir="/tmp") as temporary:
            scratch = Path(temporary)
            with _Deadline(plan["deadlines"]["setup_seconds"], "engine_setup"):
                cfg = _native_config_equivalent(plan)
                _local_paths(cfg, scratch)
                ray.init(address="auto", log_to_driver=False)
                tokenizer = AutoTokenizer.from_pretrained(
                    plan["model"]["root"], trust_remote_code=False, local_files_only=True
                )
                if (
                    "sha256:" + hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
                    != plan["data"]["template_sha256"]
                ):
                    raise rl_episode.InvalidEpisode("diagnostic_template_drift")
                engine, engine_setup = _start_engine(plan, cfg, tokenizer, ray)
            episode_attempted = True
            with _Deadline(plan["deadlines"]["episode_seconds"], "episode_collection"):
                asyncio.run(_collect_once(plan, tokenizer, engine, scratch))
            collection_completed, status = True, "collection_completed_no_training"
    except BaseException as error:
        if isinstance(error, _EngineSetupRejected):
            engine_cleanup_confirmed = error.cleanup_confirmed
        leaf = _allowlisted_leaf_reason(error)
        if leaf is not None:
            status = "allowlisted_invalid_episode"
    finally:
        if ray is not None:
            if engine_setup is not None:
                engine_cleanup_confirmed = _stop_setup(engine_setup, ray)
            elif engine_cleanup_confirmed is None:
                # ``build_new_inference_client`` may have failed after
                # allocating opaque native state.  Without an owned setup
                # object it cannot be truthfully certified as clean.
                engine_cleanup_confirmed = False
            if not _shutdown_ray(ray):
                # A successful actor teardown is not enough to claim local
                # cleanup when the Ray runtime itself declined to stop.
                engine_cleanup_confirmed = False
        elif engine_cleanup_confirmed is None:
            engine_cleanup_confirmed = False
    if (
        time.monotonic() - started > plan["deadlines"]["process_seconds"]
        or engine_cleanup_confirmed is not True
    ):
        status, leaf, collection_completed = "unclassified_failure", None, False
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "diagnostic_plan_sha256": "sha256:" + digest(plan),
        "legacy_runtime_sha256": plan["legacy_runtime_sha256"],
        "diagnostic_runtime_sha256": plan["diagnostic_runtime_sha256"],
        "image_digest": plan["execution"]["image"].split("@", 1)[1],
        "source_row_sha256": plan["selection"]["line_sha256"],
        "sampling_sha256": plan["sampling_sha256"],
        "episode_attempted": episode_attempted,
        "collection_completed": collection_completed,
        "optimizer_steps": 0,
        "checkpoints": 0,
        "wandb_events": 0,
        # ``collect`` returns only after a successful exact release.  For a
        # rejected/unknown episode we deliberately do not infer this fact from
        # a cause chain; the external UID-bound workload observer remains the
        # release authority for the GPU allocation.
        "fleet_instance_release_confirmed": collection_completed,
        "engine_cleanup_confirmed": engine_cleanup_confirmed,
        "external_job_cleanup_required": True,
    }
    if leaf is not None:
        receipt["allowlisted_leaf_reason"] = leaf
        receipt["leaf_reason_count"] = 1
    else:
        receipt["leaf_reason_count"] = 0
    return _write_receipt(root, _validate_receipt(receipt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    arguments = parser.parse_args()
    try:
        plan = json.loads(arguments.plan.read_bytes())
        if not isinstance(plan, dict) or digest(plan) != arguments.sha256:
            raise JobsError("collector diagnostic plan digest mismatch")
        receipt = run(plan)
        print(json.dumps({key: receipt[key] for key in ("status", "sha256")}))
        if receipt["status"] == "unclassified_failure" or not receipt["engine_cleanup_confirmed"]:
            raise SystemExit(1)
    except SystemExit:
        raise
    except BaseException:
        # Do not stringify exceptions: image/Fleet errors can contain private task
        # material.  A terminal receipt is written only after the output binding
        # is proven by run().
        print(json.dumps({"status": "failed"}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
