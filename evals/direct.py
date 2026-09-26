"""One-shot version-pinned Fleet/OpenCode eval; no retries or private evidence retention."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from evals.fleet import validate_protocol

TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


class LaunchError(ValueError):
    """A pre-submit gate failed; uncertain creates must never be replayed."""


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _families(plan: dict) -> None:
    roles = plan["family_roles"]
    if set(roles) != {"train", "dev", "final"}:
        raise LaunchError("frozen family roles are required")
    seen = set()
    for role, rows in roles.items():
        if not isinstance(rows, list) or (role != "train" and not rows):
            raise LaunchError("frozen development and final family rosters are required")
        for row in rows:
            if (not isinstance(row, list) or len(row) != 2
                    or any(not isinstance(x, str) or not x for x in row)
                    or tuple(row) in seen):
                raise LaunchError("train/development/final family overlap or duplicate")
            seen.add(tuple(row))
    selected = {(x["application"], x["family_id"]) for x in plan["protocol"]["tasks"]}
    if selected != {tuple(x) for x in roles[plan["protocol"]["role"]]}:
        raise LaunchError("protocol tasks differ from frozen family roster")


def _check_task(task: dict, live: dict, expected_digest: str) -> None:
    verifier, metadata = live.get("verifier") or {}, live.get("metadata") or {}
    manifest = metadata.get("runtime_seed_manifest") or {}
    files = manifest.get("files") if isinstance(manifest, dict) else None
    seed_bound = (isinstance(files, list) and bool(files)
                  and type(manifest.get("version")) is int and manifest["version"] > 0
                  and isinstance(manifest.get("data_root"), str) and bool(manifest["data_root"])
                  and all(isinstance(row, dict) and _sha("sha256:" + str(row.get("sha256")))
                          and isinstance(row.get("target_path"), str) and bool(row["target_path"])
                          and type(row.get("size_bytes")) is int and row["size_bytes"] > 0
                          for row in files)
                  and ("content_sha256" not in manifest
                       or _sha("sha256:" + str(manifest["content_sha256"]))))
    if (live.get("team_id") != TEAM_ID or live.get("key") != task["task_key"]
        or live.get("eval_task_version_id") != task["task_version_id"]
        or live.get("environment_version_id") != task["environment_version_id"]
        or live.get("data_version") != task["data_version"]
        or "sha256:" + str(verifier.get("sha256")) != task["verifier_sha256"]
        or not verifier.get("verifier_version_id") or not seed_bound
        or metadata.get("projection_id") != "blackbox_ctf_v1"
        or live.get("task_lifecycle_status") != "production"
        or not isinstance(live.get("seed_config"), dict)
        or not isinstance(live.get("multi_app_seed_bindings"), dict)
        or not isinstance(live.get("multi_app_seed_versions"), dict)
        or digest(live) != expected_digest):
        raise LaunchError("live exact-version task/runtime/verifier binding changed")


class FleetClient:
    """Small authenticated Fleet boundary; never logs private content."""

    def _request(self, method: str, path: str, *, query: dict | None = None,
                 body: dict | None = None) -> dict:
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise LaunchError("Fleet API credential unavailable")
        url = "https://orchestrator.fleetai.com" + path
        if query:
            url += "?" + urlencode(query)
        request = Request(url, data=None if body is None else json.dumps(body).encode(),
                          method=method, headers={"Authorization": "Bearer " + key,
                                                  "Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    def account_get(self) -> dict:
        return self._request("GET", "/v1/account")

    def task_get(self, task_key: str, version_id: str) -> dict:
        return self._request("GET", "/v1/tasks/" + quote(task_key, safe=""),
                             query={"version_id": version_id})

SCHEMA = "fleet_direct_opencode_v1"
CAPABILITY = {"version_scoped_durable_create_claim": "v1", "create_request_field": "create_request_id", "claim_route": "/v1/env/instances/create-requests/{request_id}", "ttl_seconds_range": [60, 3600], "exact_instance_runtime_readback": "v1"}
MODEL_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
HEADER = re.compile(r"[A-Za-z0-9-]+\Z")
CONTEXT, OUTPUT = 98_304, 16_384
PASS = "cyber_ctf_full_solve_v1"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, url):
        return None


def _http(method: str, url: str, headers: dict, body: dict | None = None):
    payload = None if body is None else json.dumps(body).encode()
    request = Request(url, data=payload, method=method, headers=headers)
    with build_opener(_NoRedirect).open(request, timeout=60) as response:
        return response.read(2 << 20).decode(), response.headers


def _url(task: dict, suffix: str) -> str:
    return f"/v1/rollout-rewards/{quote(task['task_key'], safe='')}/versions/{quote(task['task_version_id'], safe='')}{suffix}"


def _append(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _instance_id(value: object) -> str:
    if not isinstance(value, str) or not MODEL_ID.fullmatch(value):
        raise LaunchError("invalid Fleet instance ID")
    return value


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise LaunchError("missing instance lifetime")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise LaunchError("unqualified instance lifetime")
    return parsed.astimezone(timezone.utc)


def validate_plan(plan: dict) -> dict:
    keys = {"schema", "protocol", "family_roles", "routes", "task_response_sha256", "task_qualification_sha256", "readiness_sha256"}
    if not isinstance(plan, dict) or set(plan) != keys or plan["schema"] != SCHEMA:
        raise LaunchError("invalid direct eval plan")
    try:
        protocol = validate_protocol(plan["protocol"])
    except ValueError as exc:
        raise LaunchError(str(exc)) from exc
    common = protocol["common"]
    if (common["harness_version"] != "1.18.27" or common["context_window_tokens"] != CONTEXT
            or common["max_output_tokens"] != OUTPUT or common["context_policy"] != "native_compaction_reserved_32768"
            or common["seed_policy"] != {"mode": "server_assigned_unobserved"}
            or common["temperature"] is not None or common["top_p"] is not None
            or common["retry_limit"] != 0 or common["scoring_mode"] != "partial"
            or common["tools"] != ["fleet_bash", "fleet_submit_report"]
            or common["pass_criterion"] != PASS):
        raise LaunchError("direct OpenCode requires exact matched 96K profile, scoring and unobserved sampling")
    _families(plan)
    if (set(plan["routes"]) != {"base", "candidate"} or plan["routes"]["base"] == plan["routes"]["candidate"]
            or any(not isinstance(x, str) or not MODEL_ID.fullmatch(x) for x in plan["routes"].values())):
        raise LaunchError("direct routes need distinct exact Fleet served IDs")
    versions = {task["task_version_id"] for task in protocol["tasks"]}
    if any(set(plan[key]) != versions or not all(map(_sha, plan[key].values()))
           for key in ("task_response_sha256", "task_qualification_sha256")):
        raise LaunchError("exact task and qualification digests required")
    readiness = {"candidate_export", "candidate_reload", "base_route", "candidate_route", "tool_parity"}
    if set(plan["readiness_sha256"]) != readiness or not all(map(_sha, plan["readiness_sha256"].values())):
        raise LaunchError("independent export/reload/route/tool receipts required")
    return protocol


def _expected_readiness(plan: dict) -> dict:
    protocol = plan["protocol"]
    return {"protocol_sha256": protocol["sha256"], "receipts": plan["readiness_sha256"],
            "routes": {arm: {"served_id": plan["routes"][arm], "model_revision": protocol["arms"][arm]["model_revision"],
                             "weights_sha256": protocol["arms"][arm]["weights_sha256"],
                             "checkpoint_sha256": protocol["arms"][arm].get("checkpoint_sha256"),
                             "gateway_routed": True, "opencode_profile_verified": True}
                       for arm in ("base", "candidate")}}


def preview(plan: dict, arm: str, version: str, attempt: int, *, api: FleetClient,
            qualification_get: Callable[[], dict], readiness_get: Callable[[], dict],
            budget_check: Callable[[int], bool], model_get: Callable[[str], dict],
            harness_get: Callable[[str], str]) -> dict:
    protocol = validate_plan(plan)
    tasks = {task["task_version_id"]: task for task in protocol["tasks"]}
    if arm not in ("base", "candidate") or version not in tasks or type(attempt) is not int or attempt not in (1, 2, 3, 4):
        raise LaunchError("attempt outside sealed pass@4 roster")
    if api._request("GET", "/v1/rollout-rewards/capabilities") != CAPABILITY:
        raise LaunchError("durable create and exact runtime readback not deployed")
    account = api.account_get()
    if account.get("team_id") != TEAM_ID or account.get("team_name") != "fleet":
        raise LaunchError("Fleet-team account required")
    if qualification_get() != plan["task_qualification_sha256"] or readiness_get() != _expected_readiness(plan):
        raise LaunchError("task quality or checkpoint/route parity not independently proven")
    task = tasks[version]
    live = api.task_get(task["task_key"], version)
    _check_task(task, live, plan["task_response_sha256"][version])
    if not isinstance(live.get("prompt"), str) or not live["prompt"]:
        raise LaunchError("exact task prompt unavailable")
    profile = {"served_model_name": plan["routes"][arm], "model_type": "qwen3_5", "reasoning_parser": "qwen3",
               "tool_call_parser": "qwen3_coder", "context_window_tokens": CONTEXT, "max_output_tokens": OUTPUT}
    if model_get(plan["routes"][arm]) != profile:
        raise LaunchError("live exact served Qwen/OpenCode profile changed")
    if harness_get(protocol["common"]["harness_image"]) != "1.18.27":
        raise LaunchError("digest-pinned OpenCode image is not version 1.18.27")
    if budget_check(1) is not True:
        raise LaunchError("Fleet attempt budget denied")
    claim = str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join((protocol["sha256"], arm, version, str(attempt)))))
    return {"protocol_sha256": protocol["sha256"], "arm": arm, "task_version_id": version, "attempt": attempt,
            "create_request_id": claim, "served_id": plan["routes"][arm],
            "task_response_sha256": plan["task_response_sha256"][version], "scoring_mode": "partial",
            "pass_criterion": PASS, "seed_policy": "server_assigned_unobserved", "planned_sessions": 1,
            "preview_kind": "read_only_local_no_server_preview"}


def live_model_profile(served_id: str) -> dict:
    """Authenticated gateway metadata; never infer a served checkpoint from an alias."""
    key = os.environ.get("FLEET_QWEN_API_KEY")
    if not key:
        raise LaunchError("Fleet Qwen inference credential unavailable")
    headers = {"Authorization": "Bearer " + key, "X-Fleet-Model": served_id}
    info = json.loads(_http("GET", "https://inference.flt.build/model_info", headers)[0])
    runtime = json.loads(_http("GET", "https://inference.flt.build/server_info", headers)[0])
    length = runtime.get("context_length")
    if (runtime.get("served_model_name") != served_id or type(length) is not int
            or (length not in (CONTEXT, 262_144) if served_id == "chris-q38-base-pass4-v1" else length != CONTEXT)):
        raise LaunchError("served model/context metadata mismatch")
    return {**{key: info.get(key) for key in ("served_model_name", "model_type", "reasoning_parser", "tool_call_parser")},
            "context_window_tokens": CONTEXT, "max_output_tokens": OUTPUT}


def harness_version(image: str) -> str:
    check = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "opencode",
                            image, "--version"], capture_output=True, text=True, timeout=90, check=False)
    match = re.search(r"\b1\.18\.27\b", check.stdout) if check.returncode == 0 else None
    return "1.18.27" if match else "unavailable"


def _mcp_tools(url: str, header: str, token: str) -> list[dict]:
    headers = {header: token, "Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    _, init_headers = _http("POST", url, headers, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "q38-direct-eval", "version": "1"}}})
    if init_headers.get("Mcp-Session-Id"):
        headers["Mcp-Session-Id"] = init_headers["Mcp-Session-Id"]
    _http("POST", url, headers, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    body, _ = _http("POST", url, headers, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    messages = ([body] if body.lstrip().startswith("{") else
                [line[5:].strip() for line in body.splitlines() if line.startswith("data:")])
    tools = None
    for raw in messages:
        try:
            candidate = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("result"), dict):
            tools = candidate["result"].get("tools")
            if isinstance(tools, list):
                break
    if not isinstance(tools, list) or not all(isinstance(x, dict) and isinstance(x.get("name"), str) for x in tools):
        raise LaunchError("Fleet MCP tool catalog unavailable")
    return tools


def _opencode(prompt: str, image: str, served_id: str, mcp_url: str, auth_header: str,
              token: str, max_minutes: int, max_steps: int, instance_id: str,
              lease: Callable[[], None]) -> dict:
    """Isolated pinned image; native tools denied and private trace destroyed on exit."""
    if not HEADER.fullmatch(auth_header) or not os.environ.get("FLEET_QWEN_API_KEY"):
        raise LaunchError("model/MCP authentication unavailable")
    with tempfile.TemporaryDirectory(prefix="q38-direct-") as temp:
        root = Path(temp)
        root.chmod(0o700)
        home, output = root / "home", root / "output"
        home.mkdir(mode=0o700)
        output.mkdir(mode=0o700)
        config_dir = home / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        settings = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {"fleet-qwen": {"npm": "@ai-sdk/openai-compatible", "name": "Fleet Qwen",
                        "options": {"baseURL": "https://inference.flt.build/v1", "apiKey": "{env:FLEET_QWEN_API_KEY}", "headers": {"X-Fleet-Model": served_id}, "timeout": False, "chunkTimeout": 300000},
                        "models": {served_id: {"name": served_id, "reasoning": True, "tool_call": True, "interleaved": "reasoning_content", "limit": {"context": CONTEXT, "output": OUTPUT, "input": CONTEXT - OUTPUT}}}}},
            "mcp": {"fleet": {"type": "remote", "url": mcp_url, "enabled": True, "oauth": False, "headers": {auth_header: "{env:FLEET_MCP_TOKEN}"}}},
            "permission": {"*": "deny", "fleet_*": "allow"},
            "tools": {name: False for name in ("bash", "edit", "read", "glob", "grep", "list", "task", "webfetch", "websearch", "skill")},
            "compaction": {"auto": True, "reserved": 32768}}
        (config_dir / "opencode.json").write_text(json.dumps(settings))
        (root / "prompt.txt").write_text(prompt)
        name = "q38-eval-" + hashlib.sha256(instance_id.encode()).hexdigest()[:16]
        command = ('opencode run --format json --thinking --model fleet-qwen/' + served_id
                   + ' --dir /workspace --auto -- "$(</input/prompt.txt)" > /output/trace.jsonl 2> /output/stderr.log')
        args = ["docker", "run", "--rm", "--name", name, "--network", "bridge", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/home/node",
                "-e", "FLEET_QWEN_API_KEY", "-e", "FLEET_MCP_TOKEN", "-v", f"{home}:/home/node", "-v", f"{output}:/output",
                "-v", f"{root / 'prompt.txt'}:/input/prompt.txt:ro", image, "bash", "-lc", command]
        env = {**os.environ, "FLEET_MCP_TOKEN": token}
        deadline = time.monotonic() + max_minutes * 60
        process = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            next_lease = time.monotonic() + 300
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    raise LaunchError("OpenCode wall deadline; result is infrastructure-invalid until reviewed")
                if now >= next_lease:
                    lease()
                    next_lease = now + 300
                time.sleep(5)
            if process.returncode != 0:
                raise LaunchError("OpenCode process failed")
            trace = output / "trace.jsonl"
            if not trace.is_file() or not trace.stat().st_size:
                raise LaunchError("OpenCode produced no JSON trace")
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            if not rows or any(not isinstance(row, dict) or row.get("type") == "error" for row in rows):
                raise LaunchError("OpenCode emitted malformed or error events")
            finishes = [(i, row["part"].get("reason")) for i, row in enumerate(rows)
                        if row.get("type") == "step_finish" and isinstance(row.get("part"), dict)]
            steps = sum(row.get("type") == "step_start" for row in rows)
            if (not finishes or finishes[-1][1] != "stop" or steps > max_steps
                    or any(row.get("type") == "step_start" for row in rows[finishes[-1][0] + 1:])):
                raise LaunchError("OpenCode did not finish naturally within the frozen step budget")
            return {"exit_code": 0, "trace_sha256": "sha256:" + hashlib.sha256(trace.read_bytes()).hexdigest(), "steps": steps}
        finally:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30, check=False)
            if process.poll() is None:
                process.wait(timeout=30)


def run_once(plan: dict, arm: str, version: str, attempt: int, journal_dir: Path, *,
             api: FleetClient, qualification_get: Callable[[], dict], readiness_get: Callable[[], dict],
             budget_check: Callable[[int], bool], model_get: Callable[[str], dict] = live_model_profile,
             harness_get: Callable[[str], str] = harness_version, runner: Callable = _opencode) -> dict:
    """One claimed attempt; unknown POST or failed cleanup never becomes a valid zero."""
    def preflight():
        return preview(plan, arm, version, attempt, api=api, qualification_get=qualification_get,
                       readiness_get=readiness_get, budget_check=budget_check, model_get=model_get, harness_get=harness_get)

    first = preflight()
    if first != preflight():
        raise LaunchError("direct preflight changed")
    if not journal_dir.is_dir() or journal_dir.is_symlink():
        raise LaunchError("durable journal directory unavailable")
    cell = (first["protocol_sha256"], arm, version, str(attempt))
    path = journal_dir / (hashlib.sha256("/".join(cell).encode()).hexdigest() + ".jsonl")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise LaunchError("attempt already claimed; reconcile instead of replaying") from exc
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps({"state": "CREATE_INTENT_DO_NOT_RETRY", **first}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    instance_id = None
    try:
        task = next(t for t in plan["protocol"]["tasks"] if t["task_version_id"] == version)
        live = api.task_get(task["task_key"], version)
        _check_task(task, live, plan["task_response_sha256"][version])
        created = api._request("POST", _url(task, "/instances"), body={
            "create_request_id": first["create_request_id"], "ttl_seconds": 3600})
        if isinstance(created, dict) and created.get("instance_id") is not None:
            instance_id = _instance_id(created["instance_id"])
        if (created.get("task_key") != task["task_key"] or created.get("task_version_id") != version
                or created.get("create_request_id") != first["create_request_id"]
                or not created.get("evidence_run_id")):
            raise LaunchError("instance create response lacks exact claim/task binding")
        _append(path, {"state": "INSTANCE_CREATED", "instance_id": instance_id,
                       "evidence_run_id": created["evidence_run_id"]})
        instance = api._request("GET", "/v1/env/instances/" + quote(instance_id, safe=""))
        bound = {"instance_id": instance_id, "team_id": TEAM_ID, "status": "running", "terminated_at": None,
                 "env_key": live.get("environment_id"), "version": live.get("version"),
                 "data_key": live.get("data_id"), "data_version": live.get("data_version"),
                 "eval_task_version_id": version, "environment_version_id": live["environment_version_id"],
                 "seed_config": live["seed_config"], "multi_app_seed_bindings": live["multi_app_seed_bindings"],
                 "multi_app_seed_versions": live["multi_app_seed_versions"]}
        if any(instance.get(key) != value for key, value in bound.items()):
            raise LaunchError("created instance differs from pinned task/runtime/seed")
        expiry = _time(instance.get("expires_at"))
        if expiry <= datetime.now(timezone.utc) + timedelta(minutes=5):
            raise LaunchError("instance TTL insufficient for safe execution")
        auth = api._request("GET", "/v1/runner-auth/token")
        header, token = auth.get("header"), auth.get("token")
        root = (instance.get("urls") or {}).get("root")
        if (not isinstance(header, str) or not HEADER.fullmatch(header) or not isinstance(token, str)
                or len(token) < 16 or not isinstance(root, str) or not root.startswith("https://")):
            raise LaunchError("Fleet MCP authentication/URL unavailable")
        mcp_url = root.rstrip("/") + "/mcp"
        tools = _mcp_tools(mcp_url, header, token)
        if ([tool["name"] for tool in tools] != ["bash", "submit_report"]
                or digest(tools) != plan["protocol"]["common"]["tool_schema_sha256"]):
            raise LaunchError("live challenge MCP tool schema changed")
        _append(path, {"state": "RUNTIME_BOUND", "instance_id": instance_id, "task_response_sha256": first["task_response_sha256"],
                       "tool_schema_sha256": digest(tools), "expiry": expiry.isoformat()})

        def lease() -> None:
            nonlocal expiry
            if expiry > datetime.now(timezone.utc) + timedelta(minutes=15):
                return
            target = datetime.now(timezone.utc) + timedelta(minutes=min(55, plan["protocol"]["common"]["max_duration_minutes"] + 5))
            changed = api._request("POST", "/v1/env/instances/" + quote(instance_id, safe="") + "/extend_ttl",
                                   body={"absolute_expires_at": target.isoformat()})
            if changed.get("instance_id") != instance_id or changed.get("status") != "running":
                raise LaunchError("exact instance TTL extension not confirmed")
            expiry = _time(changed.get("expires_at"))
            if expiry < target:
                raise LaunchError("instance TTL readback below bounded target")
            _append(path, {"state": "LEASE_EXTENDED", "instance_id": instance_id, "expiry": expiry.isoformat()})

        common = plan["protocol"]["common"]
        run = runner(live["prompt"], common["harness_image"], first["served_id"], mcp_url, header, token,
                     common["max_duration_minutes"], common["max_steps"], instance_id, lease)
        if (not isinstance(run, dict) or run.get("exit_code") != 0 or not _sha(run.get("trace_sha256"))
                or type(run.get("steps")) is not int or not 0 < run["steps"] <= common["max_steps"]):
            raise LaunchError("OpenCode did not finish with a valid bounded trace")
        _append(path, {"state": "OPENCODE_COMPLETED", "instance_id": instance_id, "trace_sha256": run["trace_sha256"], "steps": run["steps"]})
        score = api._request("POST", _url(task, ""), body={"instance_id": instance_id,
                             "scoring_mode": "partial", "multi_app_aggregation_mode": "fractional"})
        result = score.get("cyber_verification_result") or {}
        shadow = score.get("cyber_evidence") or {}
        direct = shadow.get("direct_verifier") or {}
        ctf = (result.get("components") or {}).get("ctf") or {}
        value = ctf.get("score")
        reward = score.get("reward")
        execution = score.get("verifier_execution_id")
        expected = ((score, {"task_key": task["task_key"], "task_version_id": version, "instance_id": instance_id}),
                    (result, {"schema_version": "cyber_verification_result_v3", "reward": reward}),
                    (result.get("bindings") or {}, {"task_version_id": version}),
                    (shadow, {"mode": "authoritative", "status": "authoritative", "match": True,
                              "production_execution_id": execution}),
                    (direct, {"status": "authoritative", "match": True, "execution_id": execution,
                              "verifier_contract_version": "3.0.0", "context_schema_version": "cyber_verification_context_v1"}))
        if (not execution or any(actual.get(key) != target for actual, fields in expected for key, target in fields.items())
                or shadow.get("match") is not True or direct.get("match") is not True
                or any(isinstance(number, bool) or not isinstance(number, (int, float))
                       or not math.isfinite(number) or not 0 <= number <= 1 for number in (value, reward))):
            raise LaunchError("authoritative verifier result lacks exact v3 full-CTF binding")
        arm_info = plan["protocol"]["arms"][arm]
        event = {"protocol_sha256": first["protocol_sha256"], "arm": arm, "task_version_id": version,
                 "attempt": attempt, "seed": None, "model_revision": arm_info["model_revision"],
                 "weights_sha256": arm_info["weights_sha256"], "checkpoint_sha256": arm_info.get("checkpoint_sha256"),
                 "process_exit_code": 0, "termination": "completed", "budget_evidence_sha256": None,
                 "verifier_sha256": task["verifier_sha256"], "verifier_status": "completed", "verifier_execution_id": execution,
                 "verifier_result_schema": result["schema_version"], "verifier_result_sha256": digest(result),
                 "verifier_result_task_version_id": version, "ctf_score": float(value), "success": value >= 0.999}
        deleted = api._request("DELETE", "/v1/env/instances/" + quote(instance_id, safe=""))
        if not deleted.get("terminated_at"):
            raise LaunchError("instance termination unconfirmed")
        _append(path, {"state": "INSTANCE_RELEASED", "instance_id": instance_id})
        instance_id = None
        _append(path, {"state": "SCORED", "event": event})
        return event
    except BaseException as exc:
        _append(path, {"state": "ATTEMPT_INVALID_OR_UNCERTAIN", "error_class": type(exc).__name__,
                       "instance_id": instance_id, "possible_resource_leak": instance_id is None})
        raise
    finally:
        if instance_id is not None:
            try:
                deleted = api._request("DELETE", "/v1/env/instances/" + quote(instance_id, safe=""))
                if not deleted.get("terminated_at"):
                    raise LaunchError("instance termination unconfirmed")
                _append(path, {"state": "INSTANCE_RELEASED", "instance_id": instance_id})
            except Exception as exc:
                _append(path, {"state": "RELEASE_UNCONFIRMED", "instance_id": instance_id,
                               "error_class": type(exc).__name__})
                raise LaunchError("Fleet instance release unconfirmed; possible resource leak") from exc
