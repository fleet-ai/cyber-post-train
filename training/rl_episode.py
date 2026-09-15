"""One exact V1 cyber episode for native token recorders; no optimizer or retries.

Internal integration boundary, not a launch command. The caller must have frozen
train-only task selection and native model/tokenizer/recorder identities. All
outputs here are private. An uncertain create, score, or cleanup raises; it never
becomes a zero-reward sample or an automatically replaced episode.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import math
import os
import re
import time
import traceback
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from evals.fleet import opencode_self_hosted as fleet

# Admission control: at most this many Fleet environments in flight per rollout
# process, however large a wave the native trainer submits. Bound to the running
# loop on first use and never resized, so every episode in a process observes the
# same reviewed ceiling.
_episode_slots: tuple[int, asyncio.Semaphore] | None = None


class InvalidEpisode(RuntimeError):
    """Safe reason code only: underlying SDK exceptions may contain task data."""


class EpisodeBudgetExceeded(InvalidEpisode):
    """A declared horizon ended; no reward or optimizer input may be fabricated."""

    def __init__(self, reason):
        super().__init__(reason)
        # Ray's dual exception replaces .args but delegates custom attributes.
        self.reason = reason


BUDGET_STOPS = {
    "generation_incomplete_length",
    "generation_incomplete_context_full",
    "turn_budget_exhausted",
    "response_budget_exhausted",
}


def episode_slots(limit) -> asyncio.Semaphore:
    """Return this process's episode admission gate, created on first use."""
    global _episode_slots
    if type(limit) is not int or not 1 <= limit <= 256:
        raise InvalidEpisode("invalid_episode_admission_limit")
    if _episode_slots is None:
        _episode_slots = (limit, asyncio.Semaphore(limit))
    if _episode_slots[0] != limit:
        raise InvalidEpisode("episode_admission_limit_changed")
    return _episode_slots[1]


def budget_stop(error):
    """Only explicitly typed budget stops qualify, never an arbitrary cause chain."""
    if isinstance(error, EpisodeBudgetExceeded):
        reason = error.reason
        return reason if isinstance(reason, str) and reason in BUDGET_STOPS else None
    if isinstance(error, BaseExceptionGroup):
        reasons = [budget_stop(child) for child in error.exceptions]
        return reasons[0] if all(reasons) else None
    return None


def _failure(error, *, run_id, elapsed_seconds, phase):
    """Retain nested MCP failure locations without serializing exception messages."""
    result = fleet.sanitized_failure_receipt(error, run_id=run_id, elapsed_seconds=elapsed_seconds)
    result["phase"] = phase
    pending, seen, causes = [error], set(), []
    while pending and len(causes) < 16:
        item = pending.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        cause = {
            "error_type": type(item).__name__,
            "frames": [
                {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                for f in traceback.extract_tb(item.__traceback__)[-10:]
            ],
        }
        if (
            isinstance(item, InvalidEpisode)
            and item.args
            and isinstance(item.args[0], str)
            and item.args[0]
            in {
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
                "turn_budget_exhausted",
                "response_budget_exhausted",
                "instance_release_unconfirmed",
            }
        ):
            cause["reason"] = item.args[0]
        causes.append(cause)
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
        chained = item.__cause__ or item.__context__
        if chained is not None:
            pending.append(chained)
    result["causes"] = causes
    return result


async def _release(client, instance_id):
    """Delete once; observe the same instance through asynchronous shutdown."""
    async with asyncio.timeout(120):
        await _request(client, "DELETE", f"/v1/env/instances/{instance_id}")
        while True:
            response = await client.get(fleet.ORCHESTRATOR + f"/v1/env/instances/{instance_id}")
            if response.status_code == 404:
                return True
            if response.status_code != 200:
                return False
            instance = response.json()
            if instance.get("instance_id") != instance_id:
                return False
            if instance.get("status") in {"stopped", "terminated", "closed"}:
                return True
            await asyncio.sleep(2)


async def _request(client: httpx.AsyncClient, method, path, **kwargs):
    kwargs.setdefault("timeout", 120)
    response = await client.request(method, fleet.ORCHESTRATOR + path, **kwargs)
    if not 200 <= response.status_code < 300:
        raise fleet.FleetRequestError(
            method, path, response.status_code, diagnostic=fleet.request_diagnostic(response)
        )
    return response.json() if response.content else {}


@asynccontextmanager
async def _mcp(root, auth, timeout):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    version = importlib.metadata.version("mcp")
    if version == "2.1.1":
        import httpx2 as transport

        stream_count, read_timeout = 2, timeout
    elif version == "1.28.0":
        transport = httpx
        stream_count, read_timeout = 3, timedelta(seconds=timeout)
    else:
        raise InvalidEpisode("unqualified_mcp_version")
    async with (
        transport.AsyncClient(
            headers={auth["header"]: auth["token"]},
            timeout=timeout,
            follow_redirects=False,
            transport=transport.AsyncHTTPTransport(retries=0),
        ) as client,
        streamable_http_client(root.rstrip("/") + "/mcp", http_client=client) as streams,
    ):
        if len(streams) != stream_count:
            raise InvalidEpisode("mcp_transport_contract_drift")
        async with ClientSession(*streams[:2], read_timeout_seconds=read_timeout) as session:
            await session.initialize()
            yield session


def _live_instance(config, instance, instance_id):
    expected = {
        "instance_id": instance_id,
        "team_id": fleet.FLEET_TEAM_ID,
        "status": "running",
        "env_key": config["environment"]["id"],
        "version": config["environment"]["version"],
        "data_key": config["environment"]["data_id"],
        "data_version": config["environment"]["data_version"],
        "terminated_at": None,
    }
    if any(instance.get(k) != v for k, v in expected.items()):
        raise InvalidEpisode("instance_binding_drift")


async def _lifetime(client, config, instance, instance_id):
    _live_instance(config, instance, instance_id)
    created = fleet._instance_time(instance.get("created_at"))
    now = datetime.now(UTC)
    expires = fleet._instance_time(instance.get("expires_at"))
    target = created + timedelta(seconds=config["environment"]["ttl_seconds"])
    if created > now + timedelta(minutes=5) or expires <= now:
        raise InvalidEpisode("instance_lifetime_invalid")
    if target < now + timedelta(seconds=config["rl"]["episode_seconds"] + 240):
        raise InvalidEpisode("instance_budget_insufficient")
    changed = expires < target
    if changed:
        await _request(
            client,
            "POST",
            f"/v1/env/instances/{instance_id}/extend_ttl",
            json={"absolute_expires_at": target.isoformat()},
        )
    result = await _request(client, "GET", f"/v1/env/instances/{instance_id}")
    _live_instance(config, result, instance_id)
    actual = fleet._instance_time(result.get("expires_at"))
    if (
        fleet._instance_time(result.get("created_at")) != created
        or actual < target
        or (changed and actual != target)
    ):
        raise InvalidEpisode("instance_lifetime_readback_mismatch")
    return result


def _validate(config):
    if config.get("config_sha256") != fleet.digest_without(config, "config_sha256"):
        raise InvalidEpisode("config_digest_mismatch")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", config["run_id"]):
        raise InvalidEpisode("unsafe_episode_id")
    authority = config["authority"]
    prefix = "/v1/rollout-rewards/{task_key}/versions/{task_version_id}"
    if (
        authority["provisioning_route_template"] != prefix + "/instances"
        or authority["scoring_route_template"] != prefix
        or authority["scoring_payload_mode"] != fleet.RUNTIME_EVIDENCE_ONLY_V3
        or config["execution"]["required_task_tools"] != ["bash", "submit_report"]
    ):
        raise InvalidEpisode("unsupported_cyber_contract")
    tool_sha = config["execution"]["required_task_tool_catalog_sha256"]
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", tool_sha):
        raise InvalidEpisode("unpinned_tool_catalog")
    limits = config["rl"]
    if set(limits) != {
        "max_turns",
        "episode_seconds",
        "tool_seconds",
        "tool_result_chars",
        "max_tokens_per_turn",
        "context_tokens",
    }:
        raise InvalidEpisode("incomplete_episode_limits")
    if any(type(v) is not int or v <= 0 for v in limits.values()):
        raise InvalidEpisode("invalid_episode_limits")
    if limits["max_tokens_per_turn"] >= limits["context_tokens"]:
        raise InvalidEpisode("generation_budget_exceeds_context")
    ttl = config["environment"]["ttl_seconds"]
    if type(ttl) is not int or not limits["episode_seconds"] + 240 < ttl <= 32400:
        raise InvalidEpisode("invalid_instance_ttl")


def validate_samples(samples):
    """Reject token/mask/log-probability drift before native optimization sees it."""
    if not samples:
        raise InvalidEpisode("empty_recording")
    for sample in samples:
        n = sample.response_length
        if (
            type(n) is not int
            or not 0 < n < len(sample.tokens)
            or len(sample.loss_mask) != n
            or len(sample.rollout_log_probs) != n
            or not any(sample.loss_mask)
            or getattr(getattr(sample, "status", None), "name", None) != "COMPLETED"
            or any(type(t) is not int or t < 0 for t in sample.tokens)
            or any(type(m) is not int or m not in (0, 1) for m in sample.loss_mask)
            or any(
                type(p) not in (int, float) or not math.isfinite(p) or p > 1e-6
                for p in sample.rollout_log_probs
            )
            or any(
                p != 0
                for p, m in zip(sample.rollout_log_probs, sample.loss_mask, strict=True)
                if not m
            )
        ):
            raise InvalidEpisode("token_recording_invalid")


def validate_tool_budget(catalog, seconds):
    """Allow every advertised bash timeout plus transport time; do not edit tools."""
    bash = next(tool for tool in catalog if tool["name"] == "bash")
    properties = bash["inputSchema"].get("properties", {})
    if "timeoutMs" not in properties:
        return
    maximum = properties["timeoutMs"].get("maximum")
    if type(maximum) is not int or maximum <= 0:
        raise InvalidEpisode("bash_timeout_maximum_unresolved")
    if seconds * 1000 <= maximum:
        raise InvalidEpisode("tool_timeout_below_advertised_budget")


async def _agent(recorder, session, messages, tools, limits, parse):
    recorder.begin_segment(messages, tools)
    env_time = 0.0
    for index in range(limits["max_turns"]):
        turn = await recorder.sample()
        if turn.finish != "ok" or not isinstance(turn.text, str):
            reason = {
                "length": "generation_incomplete_length",
                "context_full": "generation_incomplete_context_full",
                "aborted": "generation_incomplete_aborted",
                "ok": "generation_incomplete_nontext",
            }.get(turn.finish if isinstance(turn.finish, str) else None, "generation_incomplete")
            if turn.finish == "length" and not isinstance(turn.text, str):
                reason = "generation_incomplete_nontext"
            raise (EpisodeBudgetExceeded if reason in BUDGET_STOPS else InvalidEpisode)(reason)
        call = parse(turn.text)
        if call is not None and (
            not isinstance(call, dict)
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), dict)
        ):
            raise InvalidEpisode("tool_parser_contract_invalid")
        messages.append(recorder.append_assistant(turn.text, call, index))
        if call is None:
            return messages, "model_stop", env_time
        start = time.monotonic()
        if call["name"] not in {"bash", "submit_report"}:
            text, error = "Tool unavailable: only bash and submit_report are permitted.", True
        else:
            async with asyncio.timeout(limits["tool_seconds"]):
                result = await session.call_tool(call["name"], arguments=call["arguments"])
            if any(block.type != "text" for block in result.content):
                raise InvalidEpisode("non_text_tool_result")
            text = "\n".join(block.text for block in result.content)
            error = getattr(result, "is_error", getattr(result, "isError", None))
            if type(error) is not bool:
                raise InvalidEpisode("tool_error_status_missing")
        env_time += time.monotonic() - start
        if len(text) > limits["tool_result_chars"]:
            raise InvalidEpisode("tool_result_exceeds_budget")
        message = {
            "role": "tool",
            "tool_call_id": f"call_{index:06d}",
            "name": call["name"],
            "content": text,
        }
        messages.append(recorder.append_observation(message, []))
        if call["name"] == "submit_report" and not error:
            return messages, "report_submitted", env_time
    raise EpisodeBudgetExceeded("turn_budget_exhausted")


async def collect(config, directory: Path, recorder, parse, *, client):
    """Return native samples only after authoritative grading AND confirmed release.

    `client` is an authenticated httpx.AsyncClient with no transport retries.
    `directory` must be a unique, durable episode claim on shared private storage.
    """
    _validate(config)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    fleet.write_json_once(directory / "binding.json", config)
    instance_id = None
    owned = False
    messages = []
    reward = None
    cleanup = {"create_attempted": False, "instance_created": False, "instance_closed": False}
    started, phase = time.monotonic(), "account"
    try:
        account = await _request(client, "GET", "/v1/account")
        if account.get("team_id") != fleet.FLEET_TEAM_ID or account.get("team_name") != "fleet":
            raise InvalidEpisode("wrong_fleet_team")
        phase = "task_binding"
        task = await _request(
            client,
            "GET",
            f"/v1/tasks/{config['task']['key']}",
            params={"version_id": config["task"]["version_id"]},
        )
        fleet.verify_task(config, task)
        fleet.write_json_once(directory / "create-intent.json", {"run_id": config["run_id"]})
        cleanup["create_attempted"] = True
        phase = "provisioning"
        created = await _request(
            client,
            "POST",
            fleet.authoritative_route(config, "provisioning"),
            headers={"X-Request-ID": fleet.provisioning_request_id(config)},
            json={},
            timeout=1200,
        )
        # Retain the ID for reconciliation; a mismatched response must not
        # authorize deletion of a potentially unrelated instance.
        if isinstance(created, dict):
            instance_id = fleet._instance_identifier(created.get("instance_id"))
            cleanup["instance_created"] = True
            cleanup["instance_id"] = instance_id
        instance_id, evidence_id = fleet.validate_rollout_instance_response(config, created)
        owned = True
        cleanup["instance_created"] = True
        cleanup["instance_id"] = instance_id
        fleet.write_json_once(
            directory / "instance.json",
            {
                "instance_id": instance_id,
                "evidence_run_id": evidence_id,
            },
        )
        phase = "instance_lifetime"
        instance = await _request(client, "GET", f"/v1/env/instances/{instance_id}")
        instance = await _lifetime(client, config, instance, instance_id)
        auth = await _request(client, "GET", "/v1/runner-auth/token")
        async with asyncio.timeout(config["rl"]["episode_seconds"]):
            phase = "mcp_initialization"
            async with _mcp(instance["urls"]["root"], auth, config["rl"]["tool_seconds"]) as mcp:
                catalog = (await mcp.list_tools()).tools
                raw = [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in catalog]
                fleet.assert_required_task_tools(
                    config,
                    sorted(t.name for t in catalog),
                    fleet.sha256(fleet.canonical_json(raw)),
                )
                validate_tool_budget(raw, config["rl"]["tool_seconds"])
                by_name = {t["name"]: t for t in raw}
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": by_name[name].get("description", ""),
                            "parameters": by_name[name]["inputSchema"],
                        },
                    }
                    for name in config["execution"]["required_task_tools"]
                ]
                messages.append({"role": "user", "content": task["prompt"]})
                phase = "agent_interaction"
                messages, reason, env_time = await _agent(
                    recorder,
                    mcp,
                    messages,
                    tools,
                    config["rl"],
                    parse,
                )
        fleet.write_json_once(directory / "conversation.json", {"messages": messages})
        phase = "scoring"
        payload = fleet.build_scoring_payload(
            config, instance_id=instance_id, final_answer="", messages=[]
        )
        fleet.write_json_once(directory / "score-intent.json", payload)
        response = await _request(
            client, "POST", fleet.authoritative_route(config, "scoring"), json=payload, timeout=120
        )
        reward = fleet.sanitize_authoritative_reward_response(
            config,
            response,
            instance_id=instance_id,
            evidence_run_id=evidence_id,
        )
        fleet.write_json_once(directory / "reward.json", reward)
    except BaseException as exc:
        fleet.write_json_once(
            directory / "failure.json",
            _failure(
                exc,
                run_id=config["run_id"],
                elapsed_seconds=time.monotonic() - started,
                phase=phase,
            ),
        )
        raise
    finally:
        if instance_id is not None and owned:
            try:
                cleanup["instance_closed"] = await _release(client, instance_id)
            except BaseException as exc:
                cleanup["error_type"] = type(exc).__name__
        cleanup["possible_instance_leak"] = (
            cleanup["create_attempted"] and not cleanup["instance_closed"]
        )
        fleet.write_json_once(directory / "cleanup.json", cleanup)
        if messages and not (directory / "conversation.json").exists():
            fleet.write_json_once(directory / "conversation.json", {"messages": messages})
        if cleanup["possible_instance_leak"]:
            # Also fail during exception/cancellation unwinding. A cancelled
            # sibling with an uncertain create/release must not disappear from
            # a TaskGroup and let another episode look cleanly budget-limited.
            raise InvalidEpisode("instance_release_unconfirmed") from None
    if not cleanup["instance_closed"]:
        raise InvalidEpisode("instance_release_unconfirmed")
    meta = {
        "task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
        "verifier_execution_id": reward["verifier_execution_id"],
        "done_reason": reason,
    }
    samples = recorder.finalize(reward["reward"], meta, env_time)
    validate_samples(samples)
    recording = {
        "samples": [
            {
                "tokens": s.tokens,
                "response_length": s.response_length,
                "loss_mask": s.loss_mask,
                "rollout_log_probs": s.rollout_log_probs,
            }
            for s in samples
        ]
    }
    fleet.write_json_once(directory / "recording.json", recording)
    receipt = {
        **meta,
        "config_sha256": config["config_sha256"],
        "sample_count": len(samples),
        "files": {
            name: fleet.sha256((directory / name).read_bytes())
            for name in (
                "binding.json",
                "instance.json",
                "conversation.json",
                "reward.json",
                "cleanup.json",
                "recording.json",
            )
        },
    }
    fleet.write_json_once(
        directory / "ACCEPTED.json",
        {
            **receipt,
            "sha256": fleet.sha256(fleet.canonical_json(receipt)),
        },
    )
    return samples


def _make_recorder(args, state, sample, sampling, engine_client):
    from fti.trainers.miles import recording

    class SingleAttemptRecorder(recording.Recorder):
        # Keep native TITO, payload construction and sample updates. Only the
        # HTTP boundary differs: Miles' default post retries 60 times and logs
        # response bodies. Never patch its module globals across concurrent tasks.
        async def sample(self):
            if self.state.aborted:
                return recording.Turn(text=None, finish="aborted")
            segment = self.segments[-1]
            params = dict(self.sampling_params)
            per_turn = self.args.fleet_max_tokens_per_turn
            params["max_new_tokens"] = min(per_turn, params.get("max_new_tokens") or per_turn)
            payload, halt = recording.compute_request_payload(
                self.args,
                segment.sample.tokens,
                params,
                multimodal_inputs=segment.sample.multimodal_inputs,
            )
            if payload is None:
                segment.sample.status = halt
                return recording.Turn(text=None, finish="context_full")
            try:
                response = await engine_client.post(
                    self.url,
                    json=payload,
                    headers=recording.compute_routing_headers(self.args, segment.sample),
                )
            except httpx.HTTPError:
                raise InvalidEpisode("generation_transport_failure") from None
            if not 200 <= response.status_code < 300:
                raise InvalidEpisode(f"generation_http_{response.status_code}")
            try:
                output = response.json()
            except ValueError:
                raise InvalidEpisode("generation_invalid_json") from None
            finish = output["meta_info"]["finish_reason"]["type"]
            if finish not in {"stop", "length", "abort"}:
                raise InvalidEpisode("generation_finish_invalid")
            await recording.update_sample_from_response(
                self.args, segment.sample, payload=payload, output=output, update_loss_mask=True
            )
            if finish == "abort":
                return recording.Turn(text=None, finish="aborted")
            return recording.Turn(
                text=output["text"], finish="length" if finish == "length" else "ok"
            )

    return SingleAttemptRecorder(args, state, sample, sampling)


async def generate(input):
    """Miles custom-generate hook; reuse FTI's native token recorder and parser.

    No oversampling/replacement policy is implemented here. The bounded trainer
    request must stop on invalid groups, not install check_no_aborted's retry loop.
    Dataset split and exact config bytes must pass CPU preflight before launch.
    """
    from fti.trainers.miles.parser import parse_tool_call
    from miles.rollout.base_types import GenerateFnOutput

    args, sample = input.args, input.sample
    if getattr(args, "partial_rollout", False) or input.state.aborted:
        raise InvalidEpisode("partial_or_aborted_rollout")
    if sample.metadata.get("split") != ("dev" if input.evaluation else "train"):
        raise InvalidEpisode("task_split_mismatch")
    if any(type(x) is not int or x < 0 for x in (sample.rollout_id, sample.index)) or (
        sample.rollout_id != sample.index
    ):
        raise InvalidEpisode("missing_native_attempt_identity")
    batch = sample.metadata.get("cyber_batch", {})
    kinds = {"dev-baseline", "dev-after"} if input.evaluation else {"train"}
    if (
        set(batch) != {"kind", "rollout_id"}
        or batch["kind"] not in kinds
        or (type(batch["rollout_id"]) is not int or batch["rollout_id"] < 0)
    ):
        raise InvalidEpisode("missing_native_batch_identity")
    config = copy.deepcopy(sample.metadata["cyber_config"])
    _validate(config)
    if config["run_id"] != args.cyber_run_id:
        raise InvalidEpisode("run_identity_mismatch")
    if "initial_prompt_sha256" in config and (
        not isinstance(sample.prompt, str)
        or fleet.sha256(sample.prompt.encode()) != config["initial_prompt_sha256"]
    ):
        raise InvalidEpisode("native_input_prompt_drift")
    if (
        args.hf_checkpoint != config["model"]["root"]
        or args.fleet_tito_model != config["model"]["tito_family"]
        or args.fleet_max_tokens_per_turn != config["rl"]["max_tokens_per_turn"]
        or args.rollout_max_context_len != config["rl"]["context_tokens"]
    ):
        raise InvalidEpisode("native_model_or_budget_drift")
    template = input.state.tokenizer.chat_template
    if (
        not isinstance(template, str)
        or fleet.sha256(template.encode()) != config["model"]["runtime_chat_template_sha256"]
    ):
        raise InvalidEpisode("native_chat_template_drift")
    config["run_id"] += f"-{batch['kind']}-r{batch['rollout_id']}-s{sample.index}"
    config["native_batch"] = copy.deepcopy(batch)
    config["sampling"] = copy.deepcopy(input.sampling_params)
    config["config_sha256"] = fleet.digest_without(config, "config_sha256")
    root = Path(args.cyber_output_root)
    if not root.is_absolute():
        raise InvalidEpisode("output_root_must_be_absolute")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise InvalidEpisode("missing_fleet_auth")
    # Hold a slot across provisioning, not only the turn loop: an environment's
    # TTL starts when it is created, so a queued episode must not own one yet.
    slots = episode_slots(args.cyber_max_concurrent_episodes)
    try:
        async with (
            slots,
            httpx.AsyncClient(
                headers={"Authorization": f"Bearer {key}"},
                timeout=120,
                transport=httpx.AsyncHTTPTransport(retries=0),
                follow_redirects=False,
            ) as client,
            httpx.AsyncClient(
                timeout=config["rl"]["episode_seconds"],
                transport=httpx.AsyncHTTPTransport(retries=0),
                follow_redirects=False,
            ) as engine_client,
        ):
            recorder = _make_recorder(
                args, input.state, copy.deepcopy(sample), input.sampling_params, engine_client
            )
            samples = await collect(
                config, root / config["run_id"], recorder, parse_tool_call, client=client
            )
    except asyncio.CancelledError:
        raise
    except InvalidEpisode:
        raise
    except Exception as exc:
        if reason := budget_stop(exc):
            raise EpisodeBudgetExceeded(reason) from None
        # Native trainer/Ray logs must not render MCP/HTTP exception bodies.
        raise InvalidEpisode("episode_failure_" + type(exc).__name__) from None
    return GenerateFnOutput(samples=samples[0] if len(samples) == 1 else samples)


def _add_arguments(parser):
    parser.add_argument("--cyber-run-id", required=True)
    parser.add_argument("--cyber-output-root", required=True)
    parser.add_argument("--cyber-data-manifest", required=True)
    parser.add_argument("--fleet-tito-model", required=True)
    parser.add_argument("--fleet-max-tokens-per-turn", required=True, type=int)
    parser.add_argument("--cyber-max-concurrent-episodes", required=True, type=int)


generate.add_arguments = _add_arguments
