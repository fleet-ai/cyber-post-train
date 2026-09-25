"""Create-once, model-free Fleet blackbox runtime qualification.

This is a direct API client, not a Kubernetes Job and not a model rollout. It
uses one environment at a time, probes only bash/submit_report, checks the real
verifier with a fixed no-flag control, and always attempts exact-ID cleanup.
Only content-free receipts are saved. Run `recover` after an interrupted cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx

from training.qualify_live import PRIVATE, PROJECT, PUBLIC, TEAM
from training.qualification_order import checked

CONTRACT = {"evidence_schema": "1.0.0", "submission_protocol": "2.0.0",
            "verifier_contract": "3.0.0"}
# Only the first reviewed cell is authorized; widen after its cleanup audit.
RUN_AUTHORIZED = {0}
ORDER_PATH = Path(__file__).resolve().parents[1] / "configs/data/fleet-blackbox-qualification-order-20260925-v1.json"
PROBES = {"bash": {"script": "printf task-quality-runtime-ok"},
          "submit_report": {"verdict": "no_flag", "explanation":
                            "Automated runtime check; exploitability was not tested."}}


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def seal(value: dict) -> dict:
    return {**value, "sha256": "sha256:" + hashlib.sha256(canonical(value)).hexdigest()}


def write_once(path: Path, value: dict) -> None:
    data = canonical(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def request(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    response = client.request(method, PUBLIC + path, **kwargs)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Fleet returned a non-object response")
    return value


def route(row: dict, instances: bool = False) -> str:
    key = quote(row["task_key"], safe="")
    version = row["task_version_id"]
    suffix = "/instances" if instances else ""
    return f"/v1/rollout-rewards/{key}/versions/{version}{suffix}"


def create_instance(client: httpx.Client, row: dict, request_id: str) -> dict:
    provision = request(client, "POST", route(row, True),
                        json={"create_request_id": request_id, "ttl_seconds": 900})
    if (provision.get("task_key") != row["task_key"]
            or provision.get("task_version_id") != row["task_version_id"]
            or provision.get("create_request_id") != request_id
            or not provision.get("instance_id") or not provision.get("evidence_run_id")):
        raise ValueError("provision response identity mismatch")
    return provision


def load_cell(wave_path: Path, index: int) -> tuple[dict, dict, str, str]:
    wave = load(wave_path)
    sha = wave.pop("sha256", None)
    rows = wave.get("wave")
    order = checked(load(ORDER_PATH))
    first = sorted(order["ordered_families"], key=lambda family: family["qualification_rank"])[:16]
    identities = [tuple(family["representative"]) for family in first]
    if (sha != seal(wave)["sha256"]
            or wave.get("schema") != "fleet_blackbox_readonly_qualification_wave_v1"
            or wave.get("launch_authorized") is not False
            or wave.get("source_order_sha256") != order["sha256"]
            or wave.get("rank_range") != [1, 16]
            or not isinstance(rows, list) or len(rows) != 16 or not 0 <= index < 16
            or [(row["task_key"], row["task_version_id"]) for row in rows] != identities
            or len({row["task_version_id"] for row in rows}) != len(rows)):
        raise ValueError("frozen qualification wave is invalid")
    row = rows[index]
    request_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                f"fleet-model-free:{sha}:{index}:{row['task_version_id']}"))
    return wave, row, sha, request_id


def preview(wave_path: Path, index: int, root: Path) -> dict:
    _, row, sha, request_id = load_cell(wave_path, index)
    if (root / f"cell-{index:02d}").exists():
        raise ValueError("this exact cell was already attempted")
    key = os.environ.get("FLEET_API_KEY", "")
    if not key:
        raise ValueError("Fleet key unavailable")
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=45) as client:
        binding = preflight(client, row, request_id)
    return seal({"schema": "fleet_model_free_preview_v1", "wave_sha256": sha,
                 "cell_index": index, "task_version_id": row["task_version_id"],
                 "request_id": request_id, "binding_sha256": binding["sha256"],
                 "exact_binding": True, "claim_unclaimed": True,
                 "create_authorized": index in RUN_AUTHORIZED,
                 "root_alert_annotation_required": False,
                 "reason": "first cell only; widen after cleanup audit"})


def preflight(client: httpx.Client, row: dict, request_id: str) -> dict:
    account = request(client, "GET", "/v1/account")
    if account.get("team_id") != TEAM or account.get("team_name") != "fleet":
        raise ValueError("Fleet account is not the expected team")
    if account.get("instance_count", 0) >= account.get("instance_limit", 0):
        raise ValueError("Fleet instance capacity is exhausted")
    api = request(client, "GET", "/openapi.json")
    paths = api.get("paths") or {}
    claim_route = paths.get("/v1/env/instances/create-requests/{request_id}") or {}
    instance_route = paths.get("/v1/env/instances/{instance_id}") or {}
    ttl_route = paths.get("/v1/env/instances/{instance_id}/ttl") or {}
    if (not {"get", "delete"} <= set(claim_route)
            or not {"get", "delete"} <= set(instance_route)
            or "post" not in ttl_route
            or ttl_route["post"].get("requestBody", {}).get("content", {}).get(
                "application/json", {}).get("schema", {}).get("$ref")
            != "#/components/schemas/SetTTLRequest"):
        raise ValueError("exact claim, TTL, or cleanup routes are not deployed")
    capability = request(client, "GET", "/v1/rollout-rewards/capabilities")
    if (capability.get("version_scoped_durable_create_claim") != "v1"
            or capability.get("create_request_field") != "create_request_id"
            or capability.get("claim_route") != "/v1/env/instances/create-requests/{request_id}"
            or capability.get("ttl_seconds_range") != [60, 3600]):
        raise ValueError("version-scoped durable create capability mismatch")
    for path in (route(row), route(row, True)):
        with client.stream("GET", PUBLIC + path) as response:
            if response.status_code != 405:
                raise ValueError("version-scoped verifier route is not deployed")
    claim = client.get(PUBLIC + f"/v1/env/instances/create-requests/{request_id}")
    if (claim.status_code != 404 or claim.json().get("detail", {}).get("error")
            != "durable_create_request_not_found"):
        raise ValueError("create request ID is already claimed or cannot be checked")
    response = client.get(PRIVATE + f"/v1/pipeline/tasks/{row['task_id']}/status")
    response.raise_for_status()
    status = response.json()
    if (status.get("eval_task_id") != row["task_id"]
            or status.get("current_version_id") != row["task_version_id"]
            or status.get("lifecycle_status") != "production"
            or status.get("verifier_attached") is not True):
        raise ValueError("exact current task version is not production/verifier-bound")
    response = client.get(PRIVATE + f"/v1/qa/projects/{PROJECT}/task-quality")
    response.raise_for_status()
    matches = [item for item in response.json().get("tasks", [])
               if item.get("task_key") == row["task_key"]
               and item.get("eval_task_version_id") == row["task_version_id"]]
    if len(matches) != 1 or matches[0].get("status") != "not_analyzed":
        raise ValueError("exact version is no longer in the unreviewed QA pool")
    task = request(client, "GET", f"/v1/tasks/{quote(row['task_key'], safe='')}",
                   params={"version_id": row["task_version_id"]})
    metadata = task.get("metadata") or {}
    subject = metadata.get("cyber_subject") or {}
    atoms = subject.get("atom_sources") or []
    atom_keys = []
    for source in atoms:
        key, index = source.get("artifact_key"), source.get("version_index")
        if (not isinstance(key, str) or type(index) is not int or index < 0
                or source.get("locator") != f"{key}@{index}:atom_source"):
            raise ValueError("exact atom-source binding drifted")
        atom_keys.append(key)
    atom_keys.sort()
    if (task.get("key") != row["task_key"]
            or task.get("eval_task_version_id") != row["task_version_id"]
            or task.get("environment_version_id") != row["environment_version_id"]
            or atom_keys != row["atom_artifact_keys"]
            or metadata.get("projection_id") != "blackbox_ctf_v1"
            or metadata.get("cyber_contract") != CONTRACT
            or metadata.get("tools") not in (None, ["bash", "submit_report"])
            or not task.get("verifier_id") or not task.get("verifier")):
        raise ValueError("exact task, source, verifier, or tool contract drifted")
    safe = {"task_key": row["task_key"], "task_version_id": row["task_version_id"],
            "task_id": row["task_id"], "environment_id": task.get("environment_id"),
            "environment_version": task.get("version"),
            "environment_version_id": task.get("environment_version_id"),
            "verifier_id": task["verifier_id"],
            "verifier_version_id": task["verifier"].get("verifier_version_id"),
            "atom_artifact_keys": atom_keys}
    if not all(safe.values()):
        raise ValueError("incomplete exact runtime binding")
    return seal(safe)


def mcp_json(response: httpx.Response) -> dict:
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise ValueError("MCP returned no JSON event")
    return response.json()


def probe_tools(root_url: str, auth_header: str, token: str) -> str:
    endpoint = root_url.rstrip("/") + "/mcp"
    headers = {auth_header: token, "Accept": "application/json, text/event-stream",
               "Content-Type": "application/json"}
    with httpx.Client(timeout=120) as client:
        session = None
        try:
            init_response = client.post(endpoint, headers=headers, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                    "protocolVersion": "2025-03-26", "capabilities": {},
                    "clientInfo": {"name": "model-free-qualification", "version": "1"}}})
            init = mcp_json(init_response)
            if init.get("id") != 1 or not isinstance(init.get("result"), dict):
                raise ValueError("MCP initialize failed")
            session = init_response.headers.get("mcp-session-id")
            if session:
                headers["Mcp-Session-Id"] = session
            ready = client.post(endpoint, headers=headers, json={
                "jsonrpc": "2.0", "method": "notifications/initialized"})
            ready.raise_for_status()
            listed = mcp_json(client.post(endpoint, headers=headers, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}))
            tools = listed.get("result", {}).get("tools")
            if listed.get("id") != 2 or not isinstance(tools, list):
                raise ValueError("MCP tool listing failed")
            if any(not isinstance(tool, dict) or not isinstance(tool.get("name"), str)
                   for tool in tools):
                raise ValueError("MCP tool catalog is malformed")
            names = sorted(tool["name"] for tool in tools)
            if names != ["bash", "submit_report"]:
                raise ValueError("MCP tool catalog differs from expected tools")
            for call_id, name in enumerate(names, 3):
                answer = mcp_json(client.post(endpoint, headers=headers, json={
                    "jsonrpc": "2.0", "id": call_id, "method": "tools/call",
                    "params": {"name": name, "arguments": PROBES[name]}}))
                if (answer.get("id") != call_id or not isinstance(answer.get("result"), dict)
                        or answer["result"].get("isError") is True):
                    raise ValueError("MCP tool probe failed")
            return "sha256:" + hashlib.sha256(canonical({"tools": tools})).hexdigest()
        finally:
            if session:
                closed = client.delete(endpoint, headers=headers)
                closed.raise_for_status()


def cleanup(client: httpx.Client, intent: dict, binding: dict,
            known_instance: str | None = None) -> dict:
    claim_path = f"/v1/env/instances/create-requests/{intent['request_id']}"
    claim = None
    for _ in range(5):
        response = client.get(PUBLIC + claim_path)
        if response.status_code != 404:
            response.raise_for_status(); claim = response.json(); break
        time.sleep(2)
    if claim is None and known_instance is None:
        raise ValueError("create claim unresolved; possible environment leak")
    if claim is not None:
        if claim.get("request_id") != intent["request_id"] or claim.get("team_id") != TEAM:
            raise ValueError("create claim ownership mismatch; possible environment leak")
        instance_id = claim.get("instance_id")
        if known_instance and instance_id and known_instance != instance_id:
            raise ValueError("create claim instance mismatch; possible environment leak")
        if not instance_id and claim.get("state") == "accepted":
            cancelled = request(client, "DELETE", claim_path)
            if cancelled.get("state") not in {"cancelled", "failed"}:
                raise ValueError("create claim did not cancel; possible environment leak")
            instance_id = cancelled.get("instance_id")
        if not instance_id and claim.get("state") in {"failed", "cancelled"}:
            return seal({"claim_resolved": True, "instance_absent": True,
                         "cleanup_complete": True})
    else:
        instance_id = known_instance
    if not instance_id:
        raise ValueError("create claim has no resolved instance")
    path = f"/v1/env/instances/{instance_id}"
    observed = client.get(PUBLIC + path)
    if observed.status_code == 404:
        return seal({"instance_id": instance_id, "instance_absent": True,
                     "cleanup_complete": True})
    observed.raise_for_status(); instance = observed.json()
    if (instance.get("instance_id") != instance_id or instance.get("team_id") != TEAM
            or instance.get("env_key") != binding["environment_id"]
            or instance.get("version") != binding["environment_version"]):
        raise ValueError("instance ownership mismatch; possible environment leak")
    if not instance.get("terminated_at"):
        deleted = request(client, "DELETE", path)
        if not deleted.get("terminated_at"):
            raise ValueError("instance delete unconfirmed; possible environment leak")
    return seal({"instance_id": instance_id, "terminated": True,
                 "cleanup_complete": True})


def clamp_ttl(client: httpx.Client, instance_id: str) -> dict:
    path = f"/v1/env/instances/{instance_id}"
    updated = request(client, "POST", path + "/ttl", json={"ttl_seconds": 900})
    observed = request(client, "GET", path)
    if (updated.get("instance_id") != instance_id
            or observed.get("instance_id") != instance_id
            or observed.get("status") != "running"):
        raise ValueError("TTL clamp instance identity/status mismatch")
    try:
        expiry = datetime.fromisoformat(observed["expires_at"].replace("Z", "+00:00")).timestamp()
        echoed = datetime.fromisoformat(updated["expires_at"].replace("Z", "+00:00")).timestamp()
    except (KeyError, AttributeError, ValueError, TypeError) as error:
        raise ValueError("TTL clamp has no valid expiry readback") from error
    if abs(expiry - echoed) > 1 or not 0 < expiry - time.time() <= 930:
        raise ValueError("TTL clamp expiry was not bounded and persisted")
    return seal({"instance_id": instance_id, "ttl_clamped": True,
                 "expires_at": observed["expires_at"]})


def run_cell(wave_path: Path, index: int, root: Path) -> dict:
    if index not in RUN_AUTHORIZED:
        raise ValueError("qualification create is not authorized; no new provision")
    _, row, sha, request_id = load_cell(wave_path, index)
    key = os.environ.get("FLEET_API_KEY", "")
    if not key:
        raise ValueError("Fleet key unavailable")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    cell = root / f"cell-{index:02d}"
    cell.mkdir(mode=0o700)
    intent = seal({"schema": "fleet_model_free_create_intent_v1", "wave_sha256": sha,
                   "cell_index": index, "request_id": request_id,
                   "task_key": row["task_key"],
                   "task_version_id": row["task_version_id"],
                   "create_attempts_maximum": 1})
    binding = None
    instance_id = None
    passed = False
    error_type = None
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=180) as client:
        try:
            binding = preflight(client, row, request_id)
            write_once(cell / "BINDING.json", binding)
            write_once(cell / "INTENT.json", intent)
            provision = create_instance(client, row, request_id)
            instance_id = provision["instance_id"]
            write_once(cell / "STARTED.json", seal({
                "instance_id": instance_id,
                "evidence_run_id": provision["evidence_run_id"],
                "task_version_id": row["task_version_id"], "environment_started": True}))
            claim = request(client, "GET",
                            f"/v1/env/instances/create-requests/{request_id}")
            if (claim.get("request_id") != request_id or claim.get("team_id") != TEAM
                    or claim.get("state") != "materialized"
                    or claim.get("instance_id") != instance_id):
                raise ValueError("materialized create claim mismatch")
            instance = request(client, "GET", f"/v1/env/instances/{instance_id}")
            if (instance.get("instance_id") != instance_id or instance.get("team_id") != TEAM
                    or instance.get("env_key") != binding["environment_id"]
                    or instance.get("version") != binding["environment_version"]
                    or instance.get("status") != "running"):
                raise ValueError("running instance identity mismatch")
            write_once(cell / "TTL.json", clamp_ttl(client, instance_id))
            auth = request(client, "GET", "/v1/runner-auth/token")
            if not auth.get("header") or not auth.get("token"):
                raise ValueError("runner authorization unavailable")
            tool_digest = probe_tools(instance["urls"]["root"],
                                      auth["header"], auth["token"])
            write_once(cell / "TOOLS.json", seal({"task_version_id": row["task_version_id"],
                                                   "bash_reachable": True,
                                                   "submit_report_reachable": True,
                                                   "tool_catalog_sha256": tool_digest}))
            score = request(client, "POST", route(row), json={
                "instance_id": instance_id, "scoring_mode": "partial",
                "multi_app_aggregation_mode": "fractional"})
            shadow = score.get("cyber_evidence") or {}
            direct = shadow.get("direct_verifier") or {}
            if (score.get("task_key") != row["task_key"]
                    or score.get("task_version_id") != row["task_version_id"]
                    or score.get("instance_id") != instance_id
                    or score.get("reward") != 0
                    or not score.get("verifier_execution_id")
                    or score.get("cyber_verification_result", {}).get("schema_version")
                    != "cyber_verification_result_v3"
                    or score["cyber_verification_result"].get("reward") != 0
                    or score["cyber_verification_result"].get("bindings", {}).get(
                        "task_version_id") != row["task_version_id"]
                    or shadow.get("mode") != "authoritative"
                    or shadow.get("status") != "authoritative"
                    or shadow.get("match") is not True
                    or shadow.get("production_execution_id") != score["verifier_execution_id"]
                    or direct.get("status") != "authoritative"
                    or direct.get("match") is not True
                    or direct.get("execution_id") != score["verifier_execution_id"]
                    or direct.get("verifier_contract_version") != CONTRACT["verifier_contract"]
                    or direct.get("context_schema_version") != "cyber_verification_context_v1"):
                raise ValueError("negative-control verifier contract failed")
            write_once(cell / "VERIFIER.json", seal({
                "task_version_id": row["task_version_id"],
                "verifier_execution_id": score["verifier_execution_id"],
                "negative_control_zero": True, "authoritative": True,
                "numeric_score_persisted": False}))
            passed = True
        except Exception as error:
            error_type = type(error).__name__
        finally:
            if (cell / "INTENT.json").exists() and binding is not None:
                try:
                    write_once(cell / "CLEANUP.json",
                               cleanup(client, intent, binding, instance_id))
                except Exception as error:
                    error_type = "Cleanup" + type(error).__name__
                    passed = False
    cleanup_complete = (cell / "CLEANUP.json").exists()
    if not (cell / "INTENT.json").exists():
        write_once(cell / "CLEANUP.json", seal({"cleanup_complete": True,
                                                "no_provision_attempt": True}))
        cleanup_complete = True
    terminal = seal({"schema": "fleet_model_free_terminal_v1", "wave_sha256": sha,
                     "cell_index": index, "task_version_id": row["task_version_id"],
                     "runtime_checked": bool(passed and cleanup_complete),
                     "qualified": False,  # Positive solvability/grading still unproven.
                     "cleanup_complete": cleanup_complete,
                     "error_type": error_type,
                     "private_content_persisted": False})
    write_once(cell / "TERMINAL.json", terminal)
    return terminal


def recover_cell(root: Path, index: int) -> dict:
    cell = root / f"cell-{index:02d}"
    if (cell / "CLEANUP.json").exists():
        receipt = load(cell / "CLEANUP.json")
        sha = receipt.pop("sha256", None)
        if sha != seal(receipt)["sha256"] or receipt.get("cleanup_complete") is not True:
            raise ValueError("stored cleanup receipt is invalid")
        return {**receipt, "sha256": sha}
    intent = load(cell / "INTENT.json")
    binding = load(cell / "BINDING.json")
    started = load(cell / "STARTED.json") if (cell / "STARTED.json").exists() else {}
    key = os.environ.get("FLEET_API_KEY", "")
    if not key:
        raise ValueError("Fleet key unavailable")
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=180) as client:
        receipt = cleanup(client, intent, binding, started.get("instance_id"))
    if not (cell / "CLEANUP.json").exists():
        write_once(cell / "CLEANUP.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preview", "run", "recover"])
    parser.add_argument("--wave", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command in {"preview", "run"} and args.wave is None:
        parser.error("--wave is required for preview/run")
    result = (preview(args.wave, args.index, args.evidence_dir)
              if args.command == "preview" else
              run_cell(args.wave, args.index, args.evidence_dir)
              if args.command == "run" else recover_cell(args.evidence_dir, args.index))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
