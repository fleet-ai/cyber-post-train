"""Typed, duplicate-safe client for Fleet's Nebius training Jobs API."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import httpx
import yaml

from .io import digest_json
from .rl_config import read_json

DEFAULT_BASE_URL = "https://api.ft.flt.build"
SUPPORTED_KINDS = {"rl", "sft"}
RL_TRUE_STEP_OVERRIDE = "trainer.max_training_steps"
RL_TOOL_EVIDENCE_SCHEMA = "fleet_rl_task_tool_allowlists_v1"
CYBER_RL_TASK_TOOLS = ("bash", "submit_report")


class JobsAPIError(ValueError):
    """A request or response violated the safe launch contract."""


def run_kind(config: dict[str, Any]) -> str:
    explicit = config.get("kind")
    if explicit in SUPPORTED_KINDS:
        return str(explicit)
    if explicit is not None:
        raise JobsAPIError(f"run config kind must be one of {sorted(SUPPORTED_KINDS)}")
    if "grpo" in config and "tasks" in config:
        return "rl"
    if "sft" in config and "data" in config:
        return "sft"
    raise JobsAPIError("could not infer RL or SFT run kind from config schema")


def load_run_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    run_kind(config)
    title = config.get("title")
    if not isinstance(title, str) or not title.strip():
        raise JobsAPIError("run config requires a non-empty title")
    return config


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
    entrypoint = manifest.get("spec", {}).get("entrypoint") if isinstance(manifest, dict) else None
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
    if run_kind(config) != "rl":
        return []

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
    if evidence.get("source") != "authoritative_task_version_metadata" or evidence.get(
        "source_field"
    ) != "metadata.tools":
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


class TrainingJobsClient:
    """Small API boundary; bearer credentials are held only in request headers."""

    def __init__(
        self,
        bearer: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not bearer:
            raise JobsAPIError("FLEET_TRAINING_API_TOKEN must be non-empty")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {bearer}"},
            timeout=60,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TrainingJobsClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(method, path, **kwargs)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise JobsAPIError(
                f"Jobs API {method} {path} returned HTTP {response.status_code}"
            ) from None
        try:
            return response.json()
        except ValueError as exc:
            raise JobsAPIError(f"Jobs API {method} {path} returned non-JSON") from exc

    def preview(self, config: dict[str, Any]) -> dict[str, Any]:
        kind = run_kind(config)
        value = self._json("POST", f"/v1/{kind}/run/preview", json=config)
        if not isinstance(value, dict):
            raise JobsAPIError("Jobs API preview returned a non-object")
        return value

    def list_runs(self, *, limit: int = 200) -> list[dict[str, Any]]:
        value = self._json("GET", "/v1/rl/run", params={"limit": limit})
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise JobsAPIError("Jobs API run list returned an invalid payload")
        return value

    def runs_with_title(self, title: str) -> list[dict[str, Any]]:
        return [row for row in self.list_runs() if row.get("title") == title]

    def submit(self, config: dict[str, Any]) -> dict[str, Any]:
        kind = run_kind(config)
        title = str(config.get("title") or "")
        duplicates = self.runs_with_title(title)
        if duplicates:
            names = ", ".join(str(row.get("name")) for row in duplicates)
            raise JobsAPIError(f"refusing duplicate run title {title!r}; existing: {names}")
        preview = self.preview(config)
        errors = preview.get("errors") or []
        if errors:
            raise JobsAPIError(f"refusing preview with errors: {errors}")
        blockers = rl_paid_launch_blockers(config, preview)
        if blockers:
            raise JobsAPIError(
                "refusing scientifically invalid paid launch: " + "; ".join(blockers)
            )
        value = self._json("POST", f"/v1/{kind}/run", json=config)
        if not isinstance(value, dict) or not value.get("name"):
            raise JobsAPIError("Jobs API submit returned an invalid receipt")
        return value

    def status(self, name: str) -> dict[str, Any]:
        value = self._json("GET", f"/v1/rl/run/{name}")
        if not isinstance(value, dict):
            raise JobsAPIError("Jobs API status returned a non-object")
        return value


def concise_status(run: dict[str, Any]) -> dict[str, Any]:
    """Return operational evidence without copying manifests or task payloads."""
    sessions = run.get("sessions") if isinstance(run.get("sessions"), dict) else {}
    return {
        "name": run.get("name"),
        "kind": run.get("kind"),
        "title": run.get("title"),
        "status": run.get("status"),
        "trainer_version_id": run.get("trainer_version_id"),
        "ray_job_id": run.get("ray_job_id"),
        "steps": len(run.get("steps") or []),
        "step_metrics": len(run.get("step_metrics") or []),
        "checkpoints": len(run.get("checkpoints") or []),
        "training_sessions": len(sessions.get("training") or []),
        "eval_sessions": len(sessions.get("eval") or []),
    }
