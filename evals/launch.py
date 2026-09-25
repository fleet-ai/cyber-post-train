"""Fail-closed Fleet Jobs API launcher for a sealed OpenCode pass@4 comparison.

The native API creates a Fleet eval/Temporal workflow, not a Kubernetes Job.
There is no server preview endpoint or caller-controlled seed/sampling setting.
This module never pretends otherwise: preview is local and read-only, and the
independent route/qualification adapters must prove facts the API cannot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from evals.fleet import validate_protocol

SCHEMA = "fleet_native_paired_launch_v1"
TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
UUID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z")


class LaunchError(ValueError):
    """A pre-submit gate failed; uncertain POSTs must never be replayed."""


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _fields(value: Any, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise LaunchError(f"{label} has missing or unknown fields")
    return value


def _sha(value: Any) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _identity(protocol: dict) -> str:
    """Display-name and duplicate task-group changes cannot evade identity."""
    return digest({key: protocol[key] for key in (
        "role", "tasks", "common", "arms", "final_selection_sha256",
    )})


def _families(plan: dict) -> None:
    roles = _fields(plan["family_roles"], {"train", "dev", "final"}, "family roles")
    seen: set[tuple[str, str]] = set()
    for role, rows in roles.items():
        if not isinstance(rows, list) or (role != "train" and not rows):
            raise LaunchError("frozen development and final family rosters are required")
        for row in rows:
            if (not isinstance(row, list) or len(row) != 2
                    or any(not isinstance(x, str) or not x for x in row)):
                raise LaunchError("invalid reviewed family identity")
            family = tuple(row)
            if family in seen:
                raise LaunchError("train/development/final family overlap or duplicate")
            seen.add(family)
    role = plan["protocol"]["role"]
    selected = {(x["application"], x["family_id"]) for x in plan["protocol"]["tasks"]}
    if selected != {tuple(x) for x in roles[role]}:
        raise LaunchError("protocol tasks differ from frozen family roster")


def validate_plan(plan: dict) -> str:
    _fields(plan, {"schema", "protocol", "task_group_id", "family_roles",
                   "task_response_sha256", "task_qualification_sha256",
                   "readiness_sha256", "routes"}, "native launch plan")
    if plan["schema"] != SCHEMA:
        raise LaunchError("unsupported native launch schema")
    try:
        protocol = validate_protocol(plan["protocol"])
    except ValueError as exc:
        raise LaunchError(str(exc)) from exc
    common = protocol["common"]
    if (common["seed_policy"] != {"mode": "server_assigned_unobserved"}
            or common["temperature"] is not None or common["top_p"] is not None):
        raise LaunchError("native Fleet Jobs API requires unobserved server seed and sampling defaults")
    if common["retry_limit"] != 0:
        raise LaunchError("native attempt accounting requires zero planned retries")
    _families(plan)
    if not isinstance(plan["task_group_id"], str) or not UUID.fullmatch(plan["task_group_id"]):
        raise LaunchError("exact existing task-group UUID is required")
    versions = {task["task_version_id"] for task in protocol["tasks"]}
    if not all(map(_sha, _fields(plan["task_response_sha256"], versions,
                                  "task API digests").values())):
        raise LaunchError("exact-version task GET digests are required")
    if not all(map(_sha, _fields(plan["task_qualification_sha256"], versions,
                                  "task qualification receipts").values())):
        raise LaunchError("independent current-runtime task qualification is required")
    receipts = _fields(plan["readiness_sha256"], {
        "candidate_export", "candidate_reload", "base_registration",
        "candidate_registration", "live_parity",
    }, "readiness receipts")
    if not all(map(_sha, receipts.values())):
        raise LaunchError("export, reload, registration and live parity receipts are required")
    routes = _fields(plan["routes"], {"base", "candidate"}, "native model routes")
    if (routes["base"] == routes["candidate"] or any(
        not isinstance(route, str) or not route.startswith("fleet-qwen/")
        or len(route) <= len("fleet-qwen/") or any(c.isspace() for c in route)
        for route in routes.values()
    )):
        raise LaunchError("distinct catalog-compatible Fleet Qwen routes are required")
    return _identity(protocol)


def job_payload(plan: dict, arm: str) -> dict:
    identity = validate_plan(plan)
    if arm not in ("base", "candidate"):
        raise LaunchError("unknown comparison arm")
    common = plan["protocol"]["common"]
    return {
        "name": f"chris-q38-fleet-{identity[7:19]}-{arm}",
        "models": [plan["routes"][arm]],
        "pass_k": 4,
        "task_group_id": plan["task_group_id"],
        "agent_runtime": True,
        "harness": "opencode",
        "mode": "tool-use",
        "tools": [],  # Challenge bash comes from Fleet MCP, never local AgentRuntime bash.
        "max_steps": common["max_steps"],
        "max_duration_minutes": common["max_duration_minutes"],
    }


def _readiness_expected(plan: dict) -> dict:
    protocol = plan["protocol"]
    return {
        "protocol_sha256": protocol["sha256"],
        "common_sha256": digest(protocol["common"]),
        "receipts": plan["readiness_sha256"],
        "routes": {
            arm: {
                "model_id": plan["routes"][arm],
                "model_revision": protocol["arms"][arm]["model_revision"],
                "weights_sha256": protocol["arms"][arm]["weights_sha256"],
                "checkpoint_sha256": protocol["arms"][arm].get("checkpoint_sha256"),
                "catalog_open_code": True,
                "gateway_routed": True,
            } for arm in ("base", "candidate")
        },
        "mcp_tool_schema_sha256": protocol["common"]["tool_schema_sha256"],
    }


def _check_task(task: dict, live: dict, expected_digest: str) -> None:
    verifier = live.get("verifier") or {}
    metadata = live.get("metadata") or {}
    if (live.get("team_id") != TEAM_ID
            or live.get("key") != task["task_key"]
            or live.get("eval_task_version_id") != task["task_version_id"]
            or live.get("environment_version_id") != task["environment_version_id"]
            or live.get("data_version") != task["data_version"]
            or "sha256:" + str(verifier.get("sha256")) != task["verifier_sha256"]
            or not verifier.get("verifier_version_id")
            or not (metadata.get("runtime_seed_manifest") or {}).get("content_sha256")
            or metadata.get("projection_id") != "blackbox_ctf_v1"
            or live.get("task_lifecycle_status") != "production"
            or not live.get("seed_config")
            or digest(live) != expected_digest):
        raise LaunchError("live exact-version task/runtime/verifier binding changed")


def preview(plan: dict, arm: str, *,
            account_get: Callable[[], dict],
            group_get: Callable[[str], dict],
            task_get: Callable[[str, str], dict],
            qualification_get: Callable[[], dict],
            readiness_get: Callable[[], dict],
            budget_check: Callable[[int], bool]) -> dict:
    """Read-only local request preflight; the Fleet API has no server preview."""
    payload = job_payload(plan, arm)
    protocol = plan["protocol"]
    account = account_get()
    if account.get("team_id") != TEAM_ID or account.get("team_name") != "fleet":
        raise LaunchError("Fleet-team account identity is required")
    group = group_get(plan["task_group_id"])
    members = group.get("members")
    expected_versions = {x["task_version_id"] for x in protocol["tasks"]}
    if (group.get("id") != plan["task_group_id"] or group.get("team_id") != TEAM_ID
            or not isinstance(members, list)
            or len(members) != len(expected_versions)
            or {x.get("eval_task_version_id") for x in members if isinstance(x, dict)}
            != expected_versions):
        raise LaunchError("existing task group differs from exact frozen versions")
    if qualification_get() != plan["task_qualification_sha256"]:
        raise LaunchError("independent task qualification proof did not match")
    if readiness_get() != _readiness_expected(plan):
        raise LaunchError("independent checkpoint/route/harness proof did not match")
    for task in protocol["tasks"]:
        _check_task(task, task_get(task["task_key"], task["task_version_id"]),
                    plan["task_response_sha256"][task["task_version_id"]])
    sessions = 4 * len(protocol["tasks"])
    if budget_check(sessions) is not True:
        raise LaunchError("shared Fleet session budget gate did not pass")
    return {
        "preview_kind": "local_read_only_no_server_preview",
        "payload": payload,
        "payload_sha256": digest(payload),
        "protocol_sha256": protocol["sha256"],
        "scientific_identity_sha256": _identity(protocol),
        "planned_sessions": sessions,
        "sampling": "server defaults; seeds, temperature and top_p are not observed or asserted",
        "server_retry_policy": "not caller-configurable; duplicate attempts are invalid in accounting",
    }


def launch_once(plan: dict, arm: str, *, post_job: Callable[[dict], dict],
                journal_dir: Path, **preflight: Any) -> dict:
    """One native POST after two matching preflights; uncertain means stop."""
    identity = validate_plan(plan)
    journal = journal_dir / f"{identity[7:]}-{arm}.jsonl"
    if journal.exists() or journal.is_symlink():
        raise LaunchError("create intent already exists; reconcile, never retry")
    first = preview(plan, arm, **preflight)
    second = preview(plan, arm, **preflight)
    if first != second:
        raise LaunchError("pre-submit preflights changed")
    if not journal_dir.is_dir() or journal_dir.is_symlink():
        raise LaunchError("durable journal directory is unavailable")
    try:
        fd = os.open(journal, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise LaunchError("concurrent create intent already exists") from exc
    with os.fdopen(fd, "w") as stream:
        json.dump({"state": "POST_INTENT_DO_NOT_RETRY",
                   "plan_sha256": digest(plan),
                   "payload_sha256": first["payload_sha256"]}, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    dir_fd = os.open(journal_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    try:
        created = post_job(first["payload"])
        if (not isinstance(created, dict)
                or created.get("name") != first["payload"]["name"]
                or not created.get("job_id")):
            raise LaunchError("native POST response lacks exact job identity")
        result = {"job_id": created["job_id"], "name": created["name"],
                  "status": created.get("status"), "arm": arm,
                  "protocol_sha256": plan["protocol"]["sha256"]}
        with journal.open("a") as stream:
            json.dump({"state": "CREATED", **result}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return result
    except Exception as exc:
        raise LaunchError("native POST outcome uncertain; inspect exact name/job ID and never retry") from exc


class FleetClient:
    """Small live boundary. Never logs credentials, task contents, or POST bodies."""

    def _request(self, method: str, path: str, *, query: dict | None = None,
                 body: dict | None = None) -> dict:
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise LaunchError("Fleet API credential unavailable")
        url = "https://orchestrator.fleetai.com" + path
        if query:
            url += "?" + urlencode(query)
        payload = None if body is None else json.dumps(body).encode()
        request = Request(url, data=payload, method=method, headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    def account_get(self) -> dict:
        return self._request("GET", "/v1/account")

    def group_get(self, group_id: str) -> dict:
        return self._request("GET", "/v1/task-groups/" + quote(group_id, safe=""))

    def task_get(self, task_key: str, version_id: str) -> dict:
        return self._request("GET", "/v1/tasks/" + quote(task_key, safe=""),
                             query={"version_id": version_id})

    def post_job(self, payload: dict) -> dict:
        return self._request("POST", "/v1/jobs", body=payload)
