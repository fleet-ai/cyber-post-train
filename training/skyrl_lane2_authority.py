"""Fail-closed authority for the second one-step SkyRL canary.

Lane 2 deliberately uses only members of the sealed 89-task production
*train* partition.  Its ``dev`` row is an internal trainer diagnostic, not a
protected capability holdout.  Historical outcomes choose a task with a
non-saturated prior; only live Fleet verifier rewards may train the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from cyber_post_train.jobs import digest

from . import rl_reward_canary

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cyber_qwen38_skyrl_lane2_authority_v1"
PLAN_SCHEMA = "cyber_qwen38_skyrl_lane2_plan_binding_v1"
AUTHORITY_FILE_SHA256 = "sha256:04eeda6a3a43082cf61dacc2a3668cf5274d806fd1dca0fd739d3aa37bd49f45"
AUTHORITY_SELF_SHA256 = "sha256:c86f892244da06c90badeb0752492c3bfdf1203cedec0e4f2f2e64cb0c249a5c"
TASK_SET_SELF_SHA256 = "sha256:261c08c51d33baa88b7c94396059c3ad98bb98ded708d8b7f1658557f35f2563"
SPLIT_SELF_SHA256 = "sha256:594b6a6c9aebb69422ece630fb9571f581f4756ce9933021c5c8ebd6e8e66cbc"
OPTIMIZER_TASK_VERSION_ID = "dd8dd22e-75c0-4b93-8f8e-ea8a292d92bb"
DIAGNOSTIC_TASK_VERSION_ID = "54425601-6fd2-43d8-8cb9-e565b767676a"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)


def _read(path: Path, expected_file_sha256: str | None = None) -> dict:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError("lane2 authority path escapes or is unavailable")
    payload = resolved.read_bytes()
    if expected_file_sha256 is not None and (
        "sha256:" + hashlib.sha256(payload).hexdigest() != expected_file_sha256
    ):
        raise ValueError("lane2 authority file digest changed")
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("lane2 authority input is not JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("lane2 authority input is not an object")
    return value


def _sealed(value: dict, schema: str, expected: str) -> None:
    body = {key: item for key, item in value.items() if key != "sha256"}
    if (
        value.get("schema") != schema
        or value.get("sha256") != expected
        or value["sha256"] != "sha256:" + digest(body)
    ):
        raise ValueError("lane2 sealed input changed")


def _bound(base: Path, binding: dict) -> dict:
    if not {"path", "file_sha256"} <= set(binding):
        raise ValueError("lane2 authority binding is incomplete")
    return _read(base / binding["path"], binding["file_sha256"])


def validate_authority(path: Path) -> dict:
    """Reopen exact task, split, verifier and historical-selection evidence."""
    authority = _read(path, AUTHORITY_FILE_SHA256)
    _sealed(authority, SCHEMA, AUTHORITY_SELF_SHA256)
    base = path.parent
    production_tasks = _bound(base, authority["sealed_production_authority"]["task_set"])
    production_split = _bound(base, authority["sealed_production_authority"]["split"])
    lane_tasks = _bound(base, authority["lane2_shard"]["task_set"])
    lane_split = _bound(base, authority["lane2_shard"]["split"])
    eligible = _bound(base, authority["exact_binding_evidence"])
    historical = _bound(base, authority["historical_selection_prior"])
    _sealed(
        production_tasks,
        "cyber_rl_task_set_v1",
        authority["sealed_production_authority"]["task_set"]["self_sha256"],
    )
    _sealed(
        production_split,
        "cyber_task_split_v1",
        authority["sealed_production_authority"]["split"]["self_sha256"],
    )
    _sealed(lane_tasks, "cyber_rl_task_set_v1", TASK_SET_SELF_SHA256)
    _sealed(lane_split, "cyber_task_split_v1", SPLIT_SELF_SHA256)
    if (
        len(production_tasks.get("tasks", [])) != 89
        or len(production_split.get("tasks", [])) != 89
        or eligible.get("sha256") != authority["exact_binding_evidence"]["self_sha256"]
    ):
        raise ValueError("lane2 production authority changed")

    from .rl_data import selection

    selected = selection(lane_tasks, lane_split)
    by_role = {row["split"]: row for row in selected}
    if (
        set(by_role) != {"train", "dev"}
        or by_role["train"]["task_version_id"] != OPTIMIZER_TASK_VERSION_ID
        or by_role["dev"]["task_version_id"] != DIAGNOSTIC_TASK_VERSION_ID
    ):
        raise ValueError("lane2 trainer roles changed")
    production_roles = {
        (row["task_key"], row["task_version_id"]): row["split"] for row in production_split["tasks"]
    }
    if any(
        production_roles.get((row["task_key"], row["task_version_id"])) != "train"
        for row in selected
    ):
        raise ValueError("lane2 selected a protected holdout row")
    production_rows = {
        (row["task_key"], row["task_version_id"]): row for row in production_tasks["tasks"]
    }
    if any(
        production_rows.get((row["task_key"], row["task_version_id"]))
        != {key: value for key, value in row.items() if key != "split"}
        for row in selected
    ):
        raise ValueError("lane2 exact production task tuple changed")

    eligible_rows = {row["task_version_id"]: row for row in eligible["task_versions"]}
    for role, version_id in (
        ("optimizer_task", OPTIMIZER_TASK_VERSION_ID),
        ("internal_diagnostic_task", DIAGNOSTIC_TASK_VERSION_ID),
    ):
        expected, row = authority["lane2_shard"][role], eligible_rows.get(version_id, {})
        if (
            row.get("split") != "train"
            or row.get("task_key") != expected["task_key"]
            or row.get("environment", {}).get("version_id") != expected["environment_version_id"]
            or row.get("verifier", {}).get("id") != expected["verifier_id"]
            or row.get("verifier", {}).get("version_id") != expected["verifier_version_id"]
            or row.get("verifier", {}).get("sha256") != expected["verifier_sha256"]
            or row.get("current_binding_sha256") != expected["current_binding_sha256"]
            or row.get("execution_evidence", {}).get("exact_current_environment_and_verifier_match")
            is not True
        ):
            raise ValueError("lane2 exact environment/verifier binding changed")

    ease = next(
        (
            row
            for row in historical.get("tasks", [])
            if row.get("task_version_id") == OPTIMIZER_TASK_VERSION_ID
        ),
        None,
    )
    prior = authority["historical_selection_prior"]
    if ease is None or ease.get("historical_ease") != {
        "exact_task_version": prior["optimizer_exact_task_version"],
        "pass_rate": prior["optimizer_pass_rate"],
        "passes": prior["optimizer_passes"],
        "sessions": prior["optimizer_sessions"],
    }:
        raise ValueError("lane2 mixed-outcome selection prior changed")
    return authority


def validate_run_config(
    config: dict, metadata: dict, bound_model: dict, *, relative_to: Path
) -> dict:
    """Bind the one-node recipe and fresh identity to the lane2 authority."""
    authority = validate_authority(relative_to / config["qualification"])
    output, data_root = PurePosixPath(config["output_root"]), PurePosixPath(config["data"]["root"])
    if (
        metadata.get("selection_sha256") != TASK_SET_SELF_SHA256
        or metadata.get("split_sha256") != SPLIT_SELF_SHA256
        or metadata.get("tool_catalog_sha256") != rl_reward_canary.TOOL_CATALOG_SHA256
        or metadata.get("limits") != rl_reward_canary.LIMITS
        or {key: item.get("rows") for key, item in metadata.get("files", {}).items()}
        != {"train": 1, "dev": 1}
        or config["recipe"] != rl_reward_canary.RECIPE
        or config["cluster"]
        != {"priority": "c1", "resources": rl_reward_canary.RESOURCES, "target": "prod"}
        or any(bound_model.get(key) != value for key, value in rl_reward_canary.MODEL.items())
        or config["name"] != "chris-q38-rlreward-lane2-v1"
        or config["name"] != metadata.get("name")
        or config["name"] != config["wandb"]["run_id"]
        or config["wandb"]
        != {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": "chris-q38-rlreward-lane2-v1",
        }
        or str(output) != "/mnt/sfs/jobs/chris-q38-rlreward-lane2-v1"
        or str(data_root)
        != "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-lane2-v1/data"
        or config["data"]["manifest"] != str(data_root / "manifest.json")
        or output == data_root
        or output in data_root.parents
        or data_root in output.parents
    ):
        raise ValueError("lane2 model, data, recipe, resource, or identity changed")
    execution = authority["execution"]
    result = {
        "schema": PLAN_SCHEMA,
        "authority_file_sha256": AUTHORITY_FILE_SHA256,
        "authority_self_sha256": AUTHORITY_SELF_SHA256,
        "optimizer_task_version_id": OPTIMIZER_TASK_VERSION_ID,
        "diagnostic_task_version_id": DIAGNOSTIC_TASK_VERSION_ID,
        "image": IMAGE,
        "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "cluster_target": execution["cluster_target"],
        "jobs_api_base_url": "https://api.ft.flt.build",
        "preflight_name": execution["preflight_name"],
        "data_stage_name": execution["data_stage_name"],
        "submission_gate": authority["submission_gate"],
    }
    return {**result, "sha256": "sha256:" + digest(result)}


def validate_plan_binding(binding: object, metadata: dict, arguments: dict) -> dict:
    """Validate a bundled plan without reading mutable source-tree files."""
    if not isinstance(binding, dict):
        raise ValueError("lane2 plan lacks its authority binding")
    body = {key: item for key, item in binding.items() if key != "sha256"}
    if (
        binding.get("schema") != PLAN_SCHEMA
        or binding.get("sha256") != "sha256:" + digest(body)
        or binding.get("authority_file_sha256") != AUTHORITY_FILE_SHA256
        or binding.get("authority_self_sha256") != AUTHORITY_SELF_SHA256
        or binding.get("optimizer_task_version_id") != OPTIMIZER_TASK_VERSION_ID
        or binding.get("diagnostic_task_version_id") != DIAGNOSTIC_TASK_VERSION_ID
        or binding.get("image") != IMAGE
        or binding.get("environment") != {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
        or binding.get("cluster_target") != "prod"
        or binding.get("jobs_api_base_url") != "https://api.ft.flt.build"
        or metadata.get("selection_sha256") != TASK_SET_SELF_SHA256
        or metadata.get("split_sha256") != SPLIT_SELF_SHA256
        or metadata.get("limits") != rl_reward_canary.LIMITS
        or {key: arguments[key] for key in rl_reward_canary.RECIPE} != rl_reward_canary.RECIPE
    ):
        raise ValueError("lane2 bundled authority changed")
    return binding
