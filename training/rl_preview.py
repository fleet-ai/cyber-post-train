"""Read-only checks for historical typed RL previews; never a submission client."""

from __future__ import annotations

import shlex
from typing import Any

import yaml

from .io import digest_json

RL_TRUE_STEP_OVERRIDE = "trainer.max_training_steps"
RL_TOOL_EVIDENCE_SCHEMA = "fleet_rl_task_tool_allowlists_v1"
CYBER_RL_TASK_TOOLS = ("bash", "submit_report")


def _assignment_values(words: list[str], key: str) -> list[str]:
    prefix = f"{key}="
    return [word[len(prefix) :] for word in words if word.startswith(prefix)]


def _rl_task_version_ids(config: dict[str, Any]) -> tuple[list[str], str | None]:
    ids: list[str] = []
    for section in ("tasks", "eval"):
        spec = config.get(section)
        if not isinstance(spec, dict):
            return [], f"RL request requires an object at {section}"
        versions = spec.get("task_versions")
        if not isinstance(versions, list):
            return [], (
                f"paid RL requires exact {section}.task_versions; mutable task keys "
                "cannot bind authoritative tool evidence"
            )
        if section == "tasks" and not versions:
            return [], "paid RL requires at least one exact tasks.task_versions binding"
        if section == "eval" and not versions:
            task_keys = spec.get("task_keys")
            if task_keys:
                return [], "eval task_keys cannot substitute for exact task-version tool evidence"
            continue
        for index, row in enumerate(versions):
            task_version_id = row.get("task_version_id") if isinstance(row, dict) else None
            if not isinstance(task_version_id, str) or not task_version_id:
                return [], f"{section}.task_versions[{index}] has no task_version_id"
            ids.append(task_version_id)
    if len(ids) != len(set(ids)):
        return [], "train and eval task_version_id bindings must be unique and disjoint"
    return ids, None


def _preview_entrypoint(preview: dict[str, Any]) -> tuple[list[str], str | None]:
    manifest_yaml = preview.get("manifest_yaml")
    if not isinstance(manifest_yaml, str) or not manifest_yaml.strip():
        return [], "Train API preview omitted manifest_yaml"
    try:
        manifest = yaml.safe_load(manifest_yaml)
    except yaml.YAMLError:
        return [], "Train API preview returned invalid manifest_yaml"
    spec = manifest.get("spec") if isinstance(manifest, dict) else None
    entrypoint = spec.get("entrypoint") if isinstance(spec, dict) else None
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        return [], "Train API preview manifest omitted spec.entrypoint"
    try:
        return shlex.split(entrypoint), None
    except ValueError:
        return [], "Train API preview entrypoint is not valid shell-word syntax"


def rl_paid_launch_blockers(config: dict[str, Any], preview: dict[str, Any]) -> list[str]:
    """Return every fail-closed scientific blocker for a paid native RL launch.

    The config check is deliberately separate from server validation. The current Train API maps
    ``grpo.max_steps`` to SkyRL ``trainer.epochs``; an epoch is a complete dataloader pass, not a
    step. The explicit SkyRL cap is therefore required in both the request and the rendered preview.

    Tool evidence must come back from the server preview, not from a caller-authored assertion. The
    running full Fleet job proved that task bindings alone can resolve to Task objects whose
    ``metadata.tools`` is empty, exposing every environment tool and consuming the context window.
    """
    if config.get("kind", "rl") != "rl" or "grpo" not in config or "tasks" not in config:
        return ["expected a typed RL request with grpo and tasks"]

    blockers: list[str] = []
    grpo = config.get("grpo")
    max_steps = grpo.get("max_steps") if isinstance(grpo, dict) else None
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        blockers.append("RL request requires a positive integer grpo.max_steps")
        expected_cap = None
    else:
        expected_cap = str(max_steps)

    trainer = config.get("trainer")
    args = trainer.get("args") if isinstance(trainer, dict) else None
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        blockers.append("RL request requires trainer.args as a list of strings")
        request_caps: list[str] = []
    else:
        request_caps = _assignment_values(args, RL_TRUE_STEP_OVERRIDE)
    if expected_cap is not None and request_caps != [expected_cap]:
        blockers.append(
            f"request must contain exactly one {RL_TRUE_STEP_OVERRIDE}={expected_cap}; "
            "grpo.max_steps currently renders as trainer.epochs and does not bound steps"
        )

    entrypoint, entrypoint_error = _preview_entrypoint(preview)
    if entrypoint_error:
        blockers.append(entrypoint_error)
    elif expected_cap is not None:
        rendered_caps = _assignment_values(entrypoint, RL_TRUE_STEP_OVERRIDE)
        if rendered_caps != [expected_cap]:
            rendered_epochs = _assignment_values(entrypoint, "trainer.epochs")
            detail = f" (preview trainer.epochs={rendered_epochs})" if rendered_epochs else ""
            blockers.append(
                f"preview must render exactly one {RL_TRUE_STEP_OVERRIDE}={expected_cap}{detail}"
            )

    expected_ids, id_error = _rl_task_version_ids(config)
    if id_error:
        blockers.append(id_error)
        return blockers

    evidence = preview.get("task_tool_allowlist_evidence")
    if not isinstance(evidence, dict):
        blockers.append(
            "Train API preview lacks authoritative task_tool_allowlist_evidence; exact "
            "task-version metadata.tools must be populated and version-bound before paid RL"
        )
        return blockers
    if evidence.get("schema") != RL_TOOL_EVIDENCE_SCHEMA:
        blockers.append(f"unsupported task tool evidence schema {evidence.get('schema')!r}")
        return blockers
    if (
        evidence.get("source") != "authoritative_task_version_metadata"
        or evidence.get("source_field") != "metadata.tools"
    ):
        blockers.append("task tool evidence is not sourced from authoritative metadata.tools")

    bindings = evidence.get("bindings")
    if not isinstance(bindings, list):
        blockers.append("task tool evidence bindings must be a list")
        return blockers
    if evidence.get("bindings_sha256") != digest_json(bindings):
        blockers.append("task tool evidence bindings_sha256 mismatch")

    observed_ids: list[str] = []
    for index, row in enumerate(bindings):
        if not isinstance(row, dict):
            blockers.append(f"task tool evidence binding {index} is not an object")
            continue
        task_version_id = row.get("task_version_id")
        tools = row.get("tools")
        if not isinstance(task_version_id, str) or not task_version_id:
            blockers.append(f"task tool evidence binding {index} has no task_version_id")
            continue
        observed_ids.append(task_version_id)
        if (
            not isinstance(tools, list)
            or not tools
            or not all(isinstance(tool, str) and tool for tool in tools)
            or len(tools) != len(set(tools))
        ):
            blockers.append(
                f"task {task_version_id} needs a non-empty, duplicate-free metadata.tools list"
            )
        elif tuple(tools) != CYBER_RL_TASK_TOOLS:
            blockers.append(
                f"task {task_version_id} metadata.tools must expose exactly ordered "
                f"{list(CYBER_RL_TASK_TOOLS)!r}, got {tools!r}"
            )
    if observed_ids != expected_ids:
        blockers.append(
            "task tool evidence does not exactly match the ordered train+eval task_version_id set"
        )
    return blockers
