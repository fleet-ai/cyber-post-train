"""Exact OpenCode 1.18.27 long-horizon agent for Miles session server v2.

The model-facing history is owned by OpenCode.  Miles records ordinary build
agent calls in one tree session and compaction summaries in a separate session.
That separation makes the optimization policy explicit: summaries influence
later behavior, but their synthetic tokens are not themselves optimized.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import math
import os
import re
import signal
import time
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from evals.fleet import opencode_self_hosted as fleet

from .rl_episode import InvalidEpisode, _lifetime, _mcp, _release, _request, validate_tool_budget

OPENCODE_VERSION = "1.18.27"
OPENCODE_BINARY = "/usr/local/bin/opencode"
OPENCODE_BINARY_SHA256 = "bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256"
CONTEXT_MANAGEMENT = "opencode_1.18.27_native_compaction_autocontinue_segmented_v1"
SUMMARY_TOKEN_TREATMENT = "excluded_separate_session_v1"
CONTEXT_TOKENS = 262_144
TOTAL_RESPONSE_TOKENS = 245_760
MAX_TOKENS_PER_TURN = 32_768
COMPACTION_BUFFER_TOKENS = 20_000
COMPACTION_RESERVED_TOKENS = MAX_TOKENS_PER_TURN + COMPACTION_BUFFER_TOKENS
COMPACTION_THRESHOLD_TOKENS = CONTEXT_TOKENS - MAX_TOKENS_PER_TURN - COMPACTION_RESERVED_TOKENS
SUMMARY_MAX_TOKENS = 4_096
PRESERVE_RECENT_TOKENS = 15_000
MAX_MODEL_REQUESTS = 2_048
EPISODE_SECONDS = 28_800
INSTANCE_TTL_SECONDS = 32_400
JOB_HARD_SECONDS = 32_400
SESSION_NODE_CAP = 4_096
TITO_FAMILY = "qwen35"
TEMPLATE_SHA256 = "38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe"
_CHILD_ENV_KEYS = {
    "LANG",
    "LC_ALL",
    "NO_PROXY",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TZ",
}


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", value):
        raise InvalidEpisode(f"invalid_{label}")
    return value


def harness_contract() -> dict[str, Any]:
    """Return the complete immutable OpenCode/session context contract."""
    return {
        "name": "opencode",
        "version": OPENCODE_VERSION,
        "context_management": CONTEXT_MANAGEMENT,
        "context_window_size": CONTEXT_TOKENS,
        "max_output_tokens": MAX_TOKENS_PER_TURN,
        "max_model_requests": MAX_MODEL_REQUESTS,
        "job_hard_seconds": JOB_HARD_SECONDS,
        "compaction_buffer_tokens": COMPACTION_BUFFER_TOKENS,
        "compaction_reserved_tokens": COMPACTION_RESERVED_TOKENS,
        "compaction_threshold_tokens": COMPACTION_THRESHOLD_TOKENS,
        "summary_max_tokens": SUMMARY_MAX_TOKENS,
        "preserve_recent_tokens": PRESERVE_RECENT_TOKENS,
        "summary_token_treatment": SUMMARY_TOKEN_TREATMENT,
        "session_node_cap": SESSION_NODE_CAP,
        "binary_sha256": "sha256:" + OPENCODE_BINARY_SHA256,
    }


def validate_episode(config: dict[str, Any]) -> None:
    """Validate the complete long-horizon contract without touching a resource."""
    harness = config.get("harness")
    limits = config.get("rl")
    model = config.get("model")
    if not all(isinstance(value, dict) for value in (harness, limits, model)):
        raise InvalidEpisode("long_horizon_contract_missing")
    expected_limits = {
        "max_turns": MAX_MODEL_REQUESTS,
        "episode_seconds": EPISODE_SECONDS,
        "tool_seconds": limits.get("tool_seconds"),
        "max_tokens_per_turn": MAX_TOKENS_PER_TURN,
        "context_tokens": CONTEXT_TOKENS,
        "response_tokens": TOTAL_RESPONSE_TOKENS,
    }
    if set(limits) != set(expected_limits) or any(
        type(value) is not int or value <= 0 for value in limits.values()
    ):
        raise InvalidEpisode("long_horizon_limits_invalid")
    if any(limits[key] != value for key, value in expected_limits.items() if key != "tool_seconds"):
        raise InvalidEpisode("long_horizon_limits_changed")
    if harness != harness_contract():
        raise InvalidEpisode("long_horizon_harness_changed")
    if (
        model.get("repo") != "Qwen/Qwen3.8-27B"
        or model.get("tito_family") != TITO_FAMILY
        or model.get("runtime_chat_template_sha256") != "sha256:" + TEMPLATE_SHA256
        or not isinstance(model.get("served_id"), str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", model["served_id"])
    ):
        raise InvalidEpisode("long_horizon_model_binding_changed")
    if config.get("environment", {}).get("ttl_seconds") != INSTANCE_TTL_SECONDS:
        raise InvalidEpisode("long_horizon_ttl_changed")


def settings(
    config: dict[str, Any],
    *,
    primary_base_url: str,
    summary_base_url: str,
    mcp_url: str,
    runner_header: str,
) -> dict[str, Any]:
    """Render exact OpenCode settings for distinct policy and summary sessions."""
    validate_episode(config)
    if not all(
        isinstance(value, str) and value.startswith("http://")
        for value in (primary_base_url, summary_base_url, mcp_url)
    ):
        raise InvalidEpisode("session_base_url_invalid")
    if not isinstance(runner_header, str) or not re.fullmatch(
        r"[A-Za-z0-9-]{1,100}", runner_header
    ):
        raise InvalidEpisode("runner_header_invalid")
    model_id = config["model"]["served_id"]

    def provider(base_url: str, output: int) -> dict[str, Any]:
        return {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Miles session",
            "options": {
                "baseURL": base_url.rstrip("/") + "/v1",
                "apiKey": "session-only",
                "timeout": False,
                "chunkTimeout": 300_000,
            },
            "models": {
                model_id: {
                    "name": model_id,
                    "reasoning": True,
                    "tool_call": True,
                    "interleaved": "reasoning_content",
                    "limit": {
                        "context": CONTEXT_TOKENS,
                        "input": CONTEXT_TOKENS - MAX_TOKENS_PER_TURN,
                        "output": output,
                    },
                }
            },
        }

    return {
        "$schema": "https://opencode.ai/config.json",
        "model": f"fleet-primary/{model_id}",
        "default_agent": "build",
        "provider": {
            "fleet-primary": provider(primary_base_url, MAX_TOKENS_PER_TURN),
            "fleet-summary": provider(summary_base_url, SUMMARY_MAX_TOKENS),
        },
        "agent": {
            "build": {"steps": MAX_MODEL_REQUESTS},
            "compaction": {"model": f"fleet-summary/{model_id}"},
        },
        "compaction": {
            "auto": True,
            "reserved": COMPACTION_RESERVED_TOKENS,
            "preserve_recent_tokens": PRESERVE_RECENT_TOKENS,
        },
        "mcp": {
            "fleet": {
                "type": "remote",
                "url": mcp_url,
                "headers": {runner_header: "{env:CYBER_RUNNER_AUTH_TOKEN}"},
                "oauth": False,
                "enabled": True,
            }
        },
        "permission": {"*": "deny", "fleet_*": "allow"},
        "tools": {
            name: False
            for name in (
                "bash",
                "edit",
                "read",
                "glob",
                "grep",
                "list",
                "task",
                "webfetch",
                "websearch",
                "skill",
            )
        },
    }


def _tree_contract(
    session_metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict], list[dict]]:
    tree = session_metadata.get("tree")
    agent = session_metadata.get("agent")
    if not isinstance(tree, dict) or not isinstance(agent, dict):
        raise ValueError("compaction metadata is absent")
    contract = agent.get("cyber_compaction")
    nodes, leaves = tree.get("nodes"), tree.get("leaves")
    if (
        not isinstance(contract, dict)
        or not isinstance(nodes, list)
        or not isinstance(leaves, list)
    ):
        raise ValueError("compaction tree is malformed")
    if (
        contract.get("schema") != "cyber_miles_opencode_compaction_v1"
        or contract.get("summary_token_treatment") != SUMMARY_TOKEN_TREATMENT
        or contract.get("session_node_cap") != SESSION_NODE_CAP
        or contract.get("expected_segments") != contract.get("summary_calls", -1) + 1
    ):
        raise ValueError("compaction treatment changed")
    return contract, nodes, leaves


def pick_compaction_segments(
    leaf_samples: list[Any], session_metadata: dict[str, Any]
) -> list[Any]:
    """Keep every disjoint OpenCode segment and reject retries or silent branches."""
    contract, nodes, leaves = _tree_contract(session_metadata)
    if not nodes or len(nodes) > SESSION_NODE_CAP:
        raise ValueError("primary session node count is outside its bound")
    if any(not isinstance(row, dict) or type(row.get("id")) is not int for row in nodes):
        raise ValueError("primary session tree contains drift or truncation")
    by_id = {row["id"]: row for row in nodes}
    if set(by_id) != set(range(len(nodes))):
        raise ValueError("primary session tree contains drift or truncation")
    for node_id, row in by_id.items():
        parent = row.get("parent")
        if (
            row.get("seq") != node_id
            or type(row.get("truncated")) is not bool
            or row["truncated"]
            or (parent is not None and (type(parent) is not int or not 0 <= parent < node_id))
        ):
            raise ValueError("primary session tree contains drift or truncation")
    roots = [row["id"] for row in nodes if row["parent"] is None]
    if (
        len(roots) != contract["expected_segments"]
        or len(leaves) != len(roots)
        or contract.get("primary_model_calls") != len(nodes)
        or len(leaf_samples) != len(leaves)
    ):
        raise ValueError("compaction segment count differs from the session tree")
    leaf_by_id = {row.get("node_id"): row for row in leaves}
    sample_by_id = {
        sample.metadata.get("leaf", {}).get("node_id"): sample for sample in leaf_samples
    }
    if len(leaf_by_id) != len(leaves) or set(sample_by_id) != set(leaf_by_id):
        raise ValueError("leaf samples do not bind every segment")
    covered: set[int] = set()
    for leaf_id, row in leaf_by_id.items():
        path = row.get("path_node_ids")
        if (
            not isinstance(path, list)
            or not path
            or path[-1] != leaf_id
            or any(type(node_id) is not int or node_id not in by_id for node_id in path)
        ):
            raise ValueError("segment path is malformed")
        if covered.intersection(path):
            raise ValueError("segments share a prefix; retry/branch policy is unqualified")
        if by_id[path[0]]["parent"] is not None or any(
            by_id[node_id]["parent"] != parent
            for parent, node_id in zip(path, path[1:], strict=False)
        ):
            raise ValueError("segment is not one linear root-to-leaf path")
        covered.update(path)
    if covered != set(by_id):
        raise ValueError("session contains nodes outside retained compaction segments")
    return [sample_by_id[leaf_id] for leaf_id in sorted(leaf_by_id)]


def postprocess_compaction_segments(
    leaf_samples: list[Any], session_metadata: dict[str, Any]
) -> list[Any]:
    """Apply Miles' once-only masking, then enforce reward and token invariants."""
    contract, _, _ = _tree_contract(session_metadata)
    from miles.rollout.session.v2.postprocessor_hub.default_postprocess import (
        default_postprocess,
    )

    samples = default_postprocess(leaf_samples, session_metadata)
    reward = (session_metadata.get("agent") or {}).get("reward")
    if (
        type(reward) not in (int, float)
        or not math.isfinite(reward)
        or not 0 <= reward <= 1
        or len(samples) != contract["expected_segments"]
    ):
        raise ValueError("authoritative episode reward or segment count is invalid")
    for index, sample in enumerate(samples):
        if (
            sample.reward != reward
            or not sample.loss_mask
            or not any(sample.loss_mask)
            or len(sample.loss_mask) != sample.response_length
        ):
            raise ValueError("compaction segment has no valid once-only policy objective")
        sample.metadata["cyber_segment"] = {
            "index": index,
            "count": len(samples),
            "summary_token_treatment": SUMMARY_TOKEN_TREATMENT,
        }
    return samples


async def _session_create(origin: str) -> tuple[str, str]:
    response = await _request_absolute("POST", origin + "/sessions", json={})
    session_id = response.get("session_id")
    if not isinstance(session_id, str) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise InvalidEpisode("summary_session_identity_invalid")
    return session_id, f"{origin}/sessions/{session_id}"


async def _request_absolute(method: str, url: str, **kwargs: Any) -> Any:
    async with httpx.AsyncClient(
        timeout=120, transport=httpx.AsyncHTTPTransport(retries=0), follow_redirects=False
    ) as client:
        response = await client.request(method, url, **kwargs)
    if not 200 <= response.status_code < 300:
        raise InvalidEpisode(f"session_{method.lower()}_failed")
    return response.json() if response.content else {}


async def _delete_session(url: str) -> bool:
    await _request_absolute("DELETE", url)
    async with httpx.AsyncClient(
        timeout=30, transport=httpx.AsyncHTTPTransport(retries=0), follow_redirects=False
    ) as client:
        response = await client.get(url)
    return response.status_code == 404


def _session_origin(base_url: str, metadata: dict[str, Any]) -> str:
    parsed = urlsplit(base_url)
    server_id = metadata.get("session_server_id")
    if (
        parsed.scheme != "http"
        or parsed.query
        or parsed.fragment
        or not isinstance(server_id, str)
        or parsed.netloc != server_id
        or not re.fullmatch(r"/sessions/[a-f0-9]{32}", parsed.path)
    ):
        raise InvalidEpisode("primary_session_identity_invalid")
    return f"http://{parsed.netloc}"


def _binary_gate() -> None:
    path = Path(OPENCODE_BINARY)
    if not path.is_file():
        raise InvalidEpisode("opencode_binary_changed")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != OPENCODE_BINARY_SHA256:
        raise InvalidEpisode("opencode_binary_changed")


def _child_env(
    *, home: Path, prompt_path: Path, workspace: Path, model: str, runner_token: str
) -> dict[str, str]:
    """Give OpenCode only the local runtime values it needs, never parent credentials."""
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key in _CHILD_ENV_KEYS and isinstance(value, str)
    }
    return {
        **inherited,
        "HOME": str(home),
        "CYBER_PROMPT_FILE": str(prompt_path),
        "CYBER_OPENCODE_WORKSPACE": str(workspace),
        "CYBER_OPENCODE_MODEL": model,
        "CYBER_RUNNER_AUTH_TOKEN": runner_token,
        "OPENAI_API_KEY": "session-only",
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "OPENCODE_DISABLE_SHARE": "true",
        "OPENCODE_DISABLE_TELEMETRY": "true",
        "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
        "NO_COLOR": "1",
    }


def _validate_summary_nodes(nodes: list[Any], primary_calls: int) -> None:
    """Reject an impossible, malformed, or silently truncated summary recording."""
    if len(nodes) >= primary_calls or len(nodes) >= MAX_MODEL_REQUESTS:
        raise InvalidEpisode("summary_model_request_count_invalid")
    for node_id, row in enumerate(nodes):
        parent = row.get("parent") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("id") != node_id
            or row.get("seq") != node_id
            or row.get("truncated") is not False
            or (parent is not None and (type(parent) is not int or not 0 <= parent < node_id))
        ):
            raise InvalidEpisode("summary_session_tree_invalid")


async def _run_opencode(config: dict[str, Any], directory: Path, env: dict[str, str]) -> None:
    trace, stderr = directory / "private-opencode.jsonl", directory / "private-opencode.stderr"
    with (
        trace.open("wb") as out,
        stderr.open("wb") as err,
        Path(env["CYBER_PROMPT_FILE"]).open("rb") as prompt,
    ):
        process = await asyncio.create_subprocess_exec(
            OPENCODE_BINARY,
            "run",
            "--format",
            "json",
            "--thinking",
            "--model",
            env["CYBER_OPENCODE_MODEL"],
            "--dir",
            env["CYBER_OPENCODE_WORKSPACE"],
            "--auto",
            stdin=prompt,
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,
        )
        timed_out = False
        try:
            async with asyncio.timeout(config["rl"]["episode_seconds"]):
                code = await process.wait()
        except TimeoutError:
            timed_out = True
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with suppress(TimeoutError):
                async with asyncio.timeout(30):
                    await process.wait()
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            code = process.returncode
    events, malformed = fleet.load_opencode_trace(trace)
    if (
        fleet.opencode_termination(
            events, malformed_lines=malformed, exit_code=code, timed_out=timed_out
        )
        != "completed"
    ):
        raise InvalidEpisode("opencode_episode_incomplete")


async def run(
    *,
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Miles custom-agent function. Raw prompts/traces stay in the private claim."""
    del prompt, request_kwargs
    base_config = copy.deepcopy(metadata.get("cyber_config"))
    if not isinstance(base_config, dict):
        raise InvalidEpisode("episode_config_missing")
    from .rl_episode import _validate

    _validate(base_config)
    attempt = metadata.get("cyber_attempt")
    if not isinstance(attempt, dict) or set(attempt) != {"run_id", "output_root"}:
        raise InvalidEpisode("native_attempt_identity_missing")
    run_id = _safe_id(attempt["run_id"], "attempt_id")
    batch = metadata.get("cyber_batch")
    if (
        not isinstance(batch, dict)
        or set(batch) != {"kind", "rollout_id"}
        or batch.get("kind") not in {"train", "dev-baseline", "dev-after"}
        or type(batch.get("rollout_id")) is not int
        or batch["rollout_id"] < 0
    ):
        raise InvalidEpisode("native_batch_identity_missing")
    root = Path(attempt["output_root"])
    if not root.is_absolute() or root.parts[:4] != ("/", "mnt", "sfs", "jobs"):
        raise InvalidEpisode("attempt_output_root_invalid")
    config = copy.deepcopy(base_config)
    config["run_id"] = run_id
    config["native_batch"] = copy.deepcopy(batch)
    config["config_sha256"] = fleet.digest_without(config, "config_sha256")
    _validate(config)
    directory = root / run_id
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    fleet.write_json_once(directory / "binding.json", config)
    started = time.monotonic()
    instance_id = summary_url = None
    owned = summary_closed = False
    cleanup = {
        "create_attempted": False,
        "summary_session_create_attempted": False,
        "instance_closed": False,
        "summary_session_closed": False,
    }
    fleet_key: str | None = None
    compaction: dict[str, Any] | None = None
    reward: dict[str, Any] | None = None
    try:
        _binary_gate()
        origin = _session_origin(base_url, metadata)
        fleet_key = os.environ.get("FLEET_API_KEY")
        if not fleet_key:
            raise InvalidEpisode("missing_fleet_auth")
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {fleet_key}"},
            timeout=1200,
            transport=httpx.AsyncHTTPTransport(retries=0),
            follow_redirects=False,
        ) as client:
            account = await _request(client, "GET", "/v1/account")
            if account.get("team_id") != fleet.FLEET_TEAM_ID or account.get("team_name") != "fleet":
                raise InvalidEpisode("wrong_fleet_team")
            task = await _request(
                client,
                "GET",
                f"/v1/tasks/{config['task']['key']}",
                params={"version_id": config["task"]["version_id"]},
            )
            fleet.verify_task(config, task)
            cleanup["create_attempted"] = True
            created = await _request(
                client,
                "POST",
                fleet.authoritative_route(config, "provisioning"),
                headers={"X-Request-ID": fleet.provisioning_request_id(config)},
                json={},
                timeout=1200,
            )
            if isinstance(created, dict):
                instance_id = fleet._instance_identifier(created.get("instance_id"))
            instance_id, evidence_id = fleet.validate_rollout_instance_response(config, created)
            owned = True
            instance = await _request(client, "GET", f"/v1/env/instances/{instance_id}")
            instance = await _lifetime(client, config, instance, instance_id)
            auth = await _request(client, "GET", "/v1/runner-auth/token")
            async with _mcp(instance["urls"]["root"], auth, config["rl"]["tool_seconds"]) as mcp:
                catalog = (await mcp.list_tools()).tools
                raw = [
                    item.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for item in catalog
                ]
                fleet.assert_required_task_tools(
                    config,
                    sorted(item.name for item in catalog),
                    fleet.sha256(fleet.canonical_json(raw)),
                )
                validate_tool_budget(raw, config["rl"]["tool_seconds"])
            cleanup["summary_session_create_attempted"] = True
            _, summary_url = await _session_create(origin)
            rendered = settings(
                config,
                primary_base_url=base_url,
                summary_base_url=summary_url,
                mcp_url=instance["urls"]["root"].rstrip("/") + "/mcp",
                runner_header=auth["header"],
            )
            home = directory / "opencode-home"
            workspace = directory / "workspace"
            config_dir = home / ".config" / "opencode"
            config_dir.mkdir(parents=True, mode=0o700)
            workspace.mkdir(mode=0o700)
            fleet.write_json_once(config_dir / "opencode.json", rendered)
            task_prompt = task.get("prompt")
            if not isinstance(task_prompt, str) or not task_prompt.strip():
                raise InvalidEpisode("task_prompt_invalid")
            prompt_path = directory / "private-prompt.txt"
            prompt_path.write_text(task_prompt)
            prompt_path.chmod(0o400)
            env = _child_env(
                home=home,
                prompt_path=prompt_path,
                workspace=workspace,
                model=rendered["model"],
                runner_token=auth["token"],
            )
            await _run_opencode(config, directory, env)
            primary = await _request_absolute("GET", base_url)
            summary = await _request_absolute("GET", summary_url)
            primary_tree = (primary.get("metadata") or {}).get("tree") or {}
            summary_tree = (summary.get("metadata") or {}).get("tree") or {}
            primary_nodes = primary_tree.get("nodes")
            summary_nodes = summary_tree.get("nodes")
            if not isinstance(primary_nodes, list) or not isinstance(summary_nodes, list):
                raise InvalidEpisode("session_tree_evidence_missing")
            primary_calls = len(primary_nodes)
            if not 0 < primary_calls <= MAX_MODEL_REQUESTS:
                raise InvalidEpisode("primary_model_request_count_invalid")
            _validate_summary_nodes(summary_nodes, primary_calls)
            compaction = {
                "schema": "cyber_miles_opencode_compaction_v1",
                "primary_model_calls": primary_calls,
                "summary_calls": len(summary_nodes),
                "expected_segments": len(summary_nodes) + 1,
                "summary_token_treatment": SUMMARY_TOKEN_TREATMENT,
                "session_node_cap": SESSION_NODE_CAP,
                "context_tokens": CONTEXT_TOKENS,
                "compaction_threshold_tokens": COMPACTION_THRESHOLD_TOKENS,
            }
            fleet.write_json_once(directory / "compaction.json", compaction)
            payload = fleet.build_scoring_payload(
                config, instance_id=instance_id, final_answer="", messages=[]
            )
            response = await _request(
                client,
                "POST",
                fleet.authoritative_route(config, "scoring"),
                json=payload,
                timeout=120,
            )
            reward = fleet.sanitize_authoritative_reward_response(
                config, response, instance_id=instance_id, evidence_run_id=evidence_id
            )
            fleet.write_json_once(directory / "reward.json", reward)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        with suppress(FileExistsError):
            fleet.write_json_once(
                directory / "failure.json",
                fleet.sanitized_failure_receipt(
                    exc, run_id=run_id, elapsed_seconds=time.monotonic() - started
                ),
            )
        if isinstance(exc, InvalidEpisode):
            raise
        raise InvalidEpisode("opencode_agent_failure_" + type(exc).__name__) from None
    finally:
        if summary_url is not None:
            with suppress(Exception):
                summary_closed = await _delete_session(summary_url)
        cleanup["summary_session_closed"] = summary_closed
        if instance_id is not None and owned and fleet_key is not None:
            with suppress(Exception):
                async with httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {fleet_key}"},
                    timeout=120,
                    transport=httpx.AsyncHTTPTransport(retries=0),
                    follow_redirects=False,
                ) as cleanup_client:
                    cleanup["instance_closed"] = await _release(cleanup_client, instance_id)
        cleanup["possible_resource_leak"] = (
            cleanup["create_attempted"] and not cleanup["instance_closed"]
        ) or (cleanup["summary_session_create_attempted"] and not cleanup["summary_session_closed"])
        with suppress(FileExistsError):
            fleet.write_json_once(directory / "cleanup.json", cleanup)
        if cleanup["possible_resource_leak"]:
            raise InvalidEpisode("resource_release_unconfirmed") from None
    if not isinstance(reward, dict) or not isinstance(compaction, dict):
        raise InvalidEpisode("terminal_evidence_missing")
    accepted = {
        "run_id": run_id,
        "config_sha256": config["config_sha256"],
        "compaction_sha256": fleet.sha256((directory / "compaction.json").read_bytes()),
        "reward_sha256": fleet.sha256((directory / "reward.json").read_bytes()),
        "cleanup_sha256": fleet.sha256((directory / "cleanup.json").read_bytes()),
        "raw_trace_private": True,
    }
    fleet.write_json_once(
        directory / "ACCEPTED.json",
        {**accepted, "sha256": fleet.sha256(fleet.canonical_json(accepted))},
    )
    return {
        "reward": reward["reward"],
        "cyber_compaction": compaction,
    }


async def generate(input: Any) -> Any:
    """Parser-visible wrapper around Miles' session-v2 agentic generator."""
    from miles.rollout.generate_hub.agentic_tool_call import generate as native_generate

    return await native_generate(input)


def _add_arguments(parser: Any) -> None:
    from miles.rollout.generate_hub.agentic_tool_call import _add_arguments as native_arguments

    native_arguments(parser)
    parser.add_argument("--cyber-run-id", required=True)
    parser.add_argument("--cyber-output-root", required=True)
    parser.add_argument("--cyber-data-manifest", required=True)
    parser.add_argument("--fleet-policy-identity-root", required=True)
    parser.add_argument("--fleet-tito-model", required=True)
    parser.add_argument("--fleet-max-tokens-per-turn", required=True, type=int)
    parser.add_argument("--fleet-session-node-cap", required=True, type=int)


generate.add_arguments = _add_arguments
