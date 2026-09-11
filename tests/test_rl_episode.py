"""Synthetic real-HTTP-boundary tests; no Fleet environments or sampled models."""

import asyncio
import copy
import json
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import ModuleType
from types import SimpleNamespace as NS

import httpx
import pytest

from evals.fleet import opencode_self_hosted as fleet
from training import rl_episode as rl

TASK = "11111111-1111-4111-8111-111111111111"
EVIDENCE = "22222222-2222-4222-8222-222222222222"
EXECUTION = "33333333-3333-4333-8333-333333333333"
CATALOG = [
    {"name": name, "description": "synthetic tool", "inputSchema": {"type": "object"}}
    for name in ("bash", "submit_report")
]


def seal(config):
    config["config_sha256"] = fleet.digest_without(config, "config_sha256")
    return config


@pytest.fixture
def fixture(monkeypatch):
    prefix = "/v1/rollout-rewards/{task_key}/versions/{task_version_id}"
    config = seal(
        {
            "run_id": "synthetic-episode-001",
            "task": {"key": "synthetic", "version_id": TASK},
            "model": {
                "root": "/model",
                "tito_family": "qwen35",
                "runtime_chat_template_sha256": fleet.sha256(b"synthetic-template"),
            },
            "environment": {
                "id": "env",
                "version": "v1",
                "data_id": "data",
                "data_version": "v1",
                "ttl_seconds": 3600,
            },
            "verifier": {"version_id": TASK},
            "authority": {
                "provisioning_route_template": prefix + "/instances",
                "scoring_route_template": prefix,
                "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
                "scoring_mode": "full",
                "multi_app_aggregation_mode": "binary",
                "required_cyber_contract": {"verifier_contract": "v3"},
            },
            "execution": {
                "required_task_tools": ["bash", "submit_report"],
                "required_task_tool_catalog_sha256": fleet.sha256(fleet.canonical_json(CATALOG)),
            },
            "rl": {
                "max_turns": 4,
                "episode_seconds": 60,
                "tool_seconds": 5,
                "tool_result_chars": 100,
                "max_tokens_per_turn": 4096,
                "context_tokens": 32768,
            },
        }
    )
    created = datetime.now(UTC) - timedelta(seconds=5)
    instance = {
        "instance_id": "synthetic-instance",
        "team_id": fleet.FLEET_TEAM_ID,
        "status": "running",
        "terminated_at": None,
        "env_key": "env",
        "version": "v1",
        "data_key": "data",
        "data_version": "v1",
        "created_at": created.isoformat(),
        "expires_at": (created + timedelta(seconds=3600)).isoformat(),
        "urls": {"root": "https://synthetic.invalid"},
    }
    reward = {
        "task_key": "synthetic",
        "task_version_id": TASK,
        "instance_id": "synthetic-instance",
        "verifier_execution_id": EXECUTION,
        "reward": 0.5,
        "cyber_verification_result": {
            "schema_version": "cyber_verification_result_v3",
            "reward": 0.5,
            "bindings": {"task_version_id": TASK},
        },
        "cyber_evidence": {
            "mode": "authoritative",
            "status": "authoritative",
            "match": True,
            "production_execution_id": EXECUTION,
            "direct_verifier": {
                "status": "authoritative",
                "match": True,
                "execution_id": EXECUTION,
                "verifier_contract_version": "v3",
                "context_schema_version": "cyber_verification_context_v1",
            },
        },
    }
    state = NS(
        config=config,
        instance=instance,
        reward=reward,
        calls=[],
        tool_calls=[],
        overrides={},
        deleted=False,
        cleanup_status=404,
        tool_error=False,
        tool_text="synthetic observation",
        catalog=copy.deepcopy(CATALOG),
    )

    def handler(request):
        state.calls.append((request.method, request.url.path))
        override = state.overrides.get((request.method, request.url.path))
        if override is not None:
            if isinstance(override, BaseException):
                raise override
            return override
        path = request.url.path
        if path == "/v1/account":
            return httpx.Response(200, json={"team_name": "fleet", "team_id": fleet.FLEET_TEAM_ID})
        if path == "/v1/tasks/synthetic":
            assert request.url.params["version_id"] == TASK
            return httpx.Response(200, json={"prompt": "synthetic task"})
        if path.endswith("/instances"):
            assert request.method == "POST" and json.loads(request.content) == {}
            return httpx.Response(
                200,
                json={
                    "instance_id": "synthetic-instance",
                    "task_key": "synthetic",
                    "task_version_id": TASK,
                    "evidence_run_id": EVIDENCE,
                },
            )
        if path.endswith("/extend_ttl"):
            state.instance["expires_at"] = json.loads(request.content)["absolute_expires_at"]
            return httpx.Response(200, json=state.instance)
        if path == "/v1/env/instances/synthetic-instance":
            if request.method == "DELETE":
                state.deleted = True
                return httpx.Response(204)
            return httpx.Response(
                state.cleanup_status if state.deleted else 200, json=state.instance
            )
        if path == "/v1/runner-auth/token":
            return httpx.Response(200, json={"header": "x-runner-auth", "token": "private-token"})
        if path == f"/v1/rollout-rewards/synthetic/versions/{TASK}":
            assert set(json.loads(request.content)) == {
                "instance_id",
                "scoring_mode",
                "multi_app_aggregation_mode",
            }
            return httpx.Response(200, json=state.reward)
        raise AssertionError("unexpected request")

    class MCP:
        async def list_tools(self):
            return NS(
                tools=[NS(name=t["name"], model_dump=lambda _t=t, **kw: _t) for t in state.catalog]
            )

        async def call_tool(self, name, arguments):
            state.tool_calls.append((name, arguments))
            if state.tool_text == "timeout":
                raise TimeoutError("private tool error")
            return NS(content=[NS(type="text", text=state.tool_text)], is_error=state.tool_error)

    @asynccontextmanager
    async def mcp(root, auth, timeout):
        assert auth["token"] == "private-token" and timeout == 5
        yield MCP()

    monkeypatch.setattr(rl, "_mcp", mcp)
    # The existing exact task verifier has its own cross-field tests; isolate
    # this lifecycle test's API fixture, not grading, cleanup or token checks.
    monkeypatch.setattr(fleet, "verify_task", lambda config, task: task)
    state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return state


class Recorder:
    def __init__(self, turns=None):
        self.turns = iter(
            turns or [NS(text="bash", finish="ok"), NS(text="submit_report", finish="ok")]
        )
        self.finalized = False
        self.sample_value = NS(
            tokens=[1, 2, 3], response_length=2, loss_mask=[1, 0], rollout_log_probs=[-0.2, 0.0]
        )

    def begin_segment(self, messages, tools):
        assert messages == [{"role": "user", "content": "synthetic task"}]
        assert [t["function"]["name"] for t in tools] == ["bash", "submit_report"]

    async def sample(self):
        return next(self.turns)

    def append_assistant(self, text, call, turn):
        return {"role": "assistant", "content": text}

    def append_observation(self, message, images):
        assert images == []
        return message

    def finalize(self, reward, meta, env_time):
        self.finalized = True
        self.sample_value.status = NS(name="COMPLETED")
        self.sample_value.reward = reward
        self.sample_value.metadata = meta
        return [self.sample_value]


def parse(text):
    return None if text == "done" else {"name": text, "arguments": {}}


async def collect(fixture, tmp_path, recorder=None):
    async with fixture.client:
        return await rl.collect(
            fixture.config,
            tmp_path / "episode",
            recorder or Recorder(),
            parse,
            client=fixture.client,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [0.0, 0.5, 1.0])
async def test_complete_valid_reward_including_genuine_zero(fixture, tmp_path, reward):
    fixture.reward["reward"] = fixture.reward["cyber_verification_result"]["reward"] = reward
    samples = await collect(fixture, tmp_path)
    assert samples[0].reward == reward
    assert samples[0].metadata["verifier_execution_id"] == EXECUTION
    assert fixture.deleted and fixture.tool_calls == [("bash", {}), ("submit_report", {})]
    receipt = json.loads((tmp_path / "episode/ACCEPTED.json").read_text())
    assert receipt["sha256"] == fleet.sha256(
        fleet.canonical_json({k: v for k, v in receipt.items() if k != "sha256"})
    )
    for name, sha in receipt["files"].items():
        assert fleet.sha256((tmp_path / "episode" / name).read_bytes()) == sha
    assert "private-token" not in "".join(p.read_text() for p in (tmp_path / "episode").iterdir())
    with pytest.raises(FileExistsError):
        await rl.collect(
            fixture.config, tmp_path / "episode", Recorder(), parse, client=fixture.client
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["create", "score", "delete"])
@pytest.mark.parametrize("error", ["http", "timeout"])
async def test_mutations_once_never_become_zero(fixture, tmp_path, phase, error):
    prefix = f"/v1/rollout-rewards/synthetic/versions/{TASK}"
    method, path = {
        "create": ("POST", prefix + "/instances"),
        "score": ("POST", prefix),
        "delete": ("DELETE", "/v1/env/instances/synthetic-instance"),
    }[phase]
    fixture.overrides[method, path] = (
        httpx.Response(503, json={"detail": "private server error"})
        if error == "http"
        else httpx.ReadTimeout("private server error")
    )
    recorder = Recorder()
    with pytest.raises((rl.InvalidEpisode, fleet.FleetRequestError, httpx.ReadTimeout)):
        await collect(fixture, tmp_path, recorder)
    assert fixture.calls.count((method, path)) == 1
    assert not recorder.finalized and not (tmp_path / "episode/ACCEPTED.json").exists()
    assert "private server error" not in "".join(
        p.read_text() for p in (tmp_path / "episode").iterdir()
    )
    if phase == "score":
        assert fixture.deleted


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "context_full", "aborted", "unexpected"])
async def test_incomplete_generation_is_not_scored(fixture, tmp_path, finish):
    with pytest.raises(rl.InvalidEpisode, match="generation_incomplete"):
        await collect(fixture, tmp_path, Recorder([NS(text="submit_report", finish=finish)]))
    assert fixture.deleted
    assert not (tmp_path / "episode/score-intent.json").exists()


@pytest.mark.asyncio
async def test_unavailable_tool_never_executes(fixture, tmp_path):
    recorder = Recorder([NS(text="read_secret", finish="ok"), NS(text="done", finish="ok")])
    samples = await collect(fixture, tmp_path, recorder)
    assert fixture.tool_calls == []
    assert samples[0].metadata["done_reason"] == "model_stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing_id", "nan", "crosslink", "cleanup", "tool_schema"])
async def test_invalid_evidence_never_reaches_trainer(fixture, tmp_path, fault):
    if fault == "missing_id":
        fixture.reward.pop("verifier_execution_id")
    elif fault == "nan":
        fixture.reward["reward"] = "not numeric"
    elif fault == "crosslink":
        fixture.reward["cyber_evidence"]["direct_verifier"]["execution_id"] = EVIDENCE
    elif fault == "cleanup":
        fixture.cleanup_status = 200  # Still running despite DELETE returning 204.
    else:
        fixture.catalog[0]["description"] = "drifted schema"
    recorder = Recorder()
    with pytest.raises((RuntimeError, ValueError)):
        await collect(fixture, tmp_path, recorder)
    assert fixture.deleted and not recorder.finalized
    assert not (tmp_path / "episode/ACCEPTED.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["ttl", "instance", "tool_size", "tool_timeout", "turn_limit"])
async def test_budgets_and_runtime_bindings(fixture, tmp_path, fault):
    if fault == "ttl":
        fixture.instance["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    elif fault == "instance":
        fixture.instance["data_version"] = "other"
    elif fault == "tool_size":
        fixture.tool_text = "x" * 101
    elif fault == "tool_timeout":
        fixture.tool_text = "timeout"
    else:
        fixture.config["rl"]["max_turns"] = 1
        seal(fixture.config)
    with pytest.raises((rl.InvalidEpisode, TimeoutError)):
        await collect(fixture, tmp_path)
    assert fixture.deleted and not (tmp_path / "episode/ACCEPTED.json").exists()


@pytest.mark.asyncio
async def test_extends_original_ttl_once_and_checks_readback(fixture, tmp_path):
    fixture.instance["expires_at"] = (datetime.now(UTC) + timedelta(seconds=10)).isoformat()
    await collect(fixture, tmp_path)
    assert fixture.calls.count(("POST", "/v1/env/instances/synthetic-instance/extend_ttl")) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens", [1, True, 3]),
        ("response_length", 0),
        ("loss_mask", [0, 0]),
        ("loss_mask", [True, 0]),
        ("loss_mask", [1]),
        ("rollout_log_probs", [-0.2]),
        ("rollout_log_probs", [float("nan"), 0]),
        ("rollout_log_probs", [0.1, 0]),
        ("rollout_log_probs", [-0.2, -0.1]),
    ],
)
def test_token_evidence_rejected(field, value):
    sample = Recorder().sample_value
    sample.status = NS(name="COMPLETED")
    setattr(sample, field, value)
    with pytest.raises(rl.InvalidEpisode, match="token_recording_invalid"):
        rl.validate_samples([sample])


@pytest.mark.asyncio
async def test_cancellation_cleans_known_instance(fixture, tmp_path):
    recorder = Recorder()

    async def cancelled():
        raise asyncio.CancelledError()

    recorder.sample = cancelled
    with pytest.raises(asyncio.CancelledError):
        await collect(fixture, tmp_path, recorder)
    assert fixture.deleted and not recorder.finalized


@pytest.mark.asyncio
async def test_digest_rejection_precedes_claim_and_network(fixture, tmp_path):
    fixture.config["run_id"] = "tampered"
    with pytest.raises(rl.InvalidEpisode, match="config_digest_mismatch"):
        await collect(fixture, tmp_path)
    assert fixture.calls == [] and not (tmp_path / "episode").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value,code",
    [
        (("run_id",), "../escape", "unsafe_episode_id"),
        (("authority", "scoring_payload_mode"), "legacy", "unsupported_cyber_contract"),
        (("execution", "required_task_tool_catalog_sha256"), "current", "unpinned_tool_catalog"),
        (("rl",), {}, "incomplete_episode_limits"),
        (("rl", "tool_seconds"), True, "invalid_episode_limits"),
        (("environment", "ttl_seconds"), 1, "invalid_instance_ttl"),
    ],
)
async def test_config_failures_precede_network(fixture, tmp_path, field, value, code):
    target = fixture.config if len(field) == 1 else fixture.config[field[0]]
    target[field[-1]] = value
    seal(fixture.config)
    with pytest.raises(rl.InvalidEpisode, match=code):
        await collect(fixture, tmp_path)
    assert fixture.calls == [] and not (tmp_path / "episode").exists()


@pytest.mark.asyncio
async def test_wrong_team_never_creates(fixture, tmp_path):
    fixture.overrides["GET", "/v1/account"] = httpx.Response(200, json={"team_name": "other"})
    with pytest.raises(rl.InvalidEpisode, match="wrong_fleet_team"):
        await collect(fixture, tmp_path)
    assert len(fixture.calls) == 1


@pytest.mark.asyncio
async def test_mismatched_create_does_not_delete_potential_peer(fixture, tmp_path):
    fixture.overrides["POST", f"/v1/rollout-rewards/synthetic/versions/{TASK}/instances"] = (
        httpx.Response(
            200,
            json={
                "instance_id": "possible-peer",
                "task_key": "other",
                "task_version_id": TASK,
                "evidence_run_id": EVIDENCE,
            },
        )
    )
    with pytest.raises(RuntimeError, match="binding drifted"):
        await collect(fixture, tmp_path)
    assert not any(method == "DELETE" for method, _ in fixture.calls)
    cleanup = json.loads((tmp_path / "episode/cleanup.json").read_text())
    assert cleanup["possible_instance_leak"] and cleanup["instance_id"] == "possible-peer"


@pytest.mark.asyncio
async def test_insufficient_remaining_ttl(fixture, tmp_path):
    fixture.instance["created_at"] = (datetime.now(UTC) - timedelta(seconds=3350)).isoformat()
    with pytest.raises(rl.InvalidEpisode, match="instance_budget_insufficient"):
        await collect(fixture, tmp_path)
    assert fixture.deleted


@pytest.mark.asyncio
async def test_extension_must_read_back(fixture, tmp_path):
    fixture.instance["expires_at"] = (datetime.now(UTC) + timedelta(seconds=10)).isoformat()
    fixture.overrides["POST", "/v1/env/instances/synthetic-instance/extend_ttl"] = httpx.Response(
        200, json={}
    )
    with pytest.raises(rl.InvalidEpisode, match="instance_lifetime_readback_mismatch"):
        await collect(fixture, tmp_path)
    assert fixture.deleted


@pytest.mark.asyncio
async def test_tool_error_can_be_corrected(fixture, tmp_path):
    fixture.tool_error = True
    turns = [NS(text="submit_report", finish="ok"), NS(text="done", finish="ok")]
    samples = await collect(fixture, tmp_path, Recorder(turns))
    assert samples[0].metadata["done_reason"] == "model_stop"


@pytest.mark.asyncio
async def test_bad_parser_contract_is_not_a_zero(fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys.modules[__name__], "parse", lambda text: {"name": "bash", "arguments": []}
    )
    with pytest.raises(rl.InvalidEpisode, match="tool_parser_contract_invalid"):
        await collect(fixture, tmp_path)
    assert fixture.deleted and not fixture.tool_calls


def test_empty_recording():
    with pytest.raises(rl.InvalidEpisode, match="empty_recording"):
        rl.validate_samples([])


@pytest.mark.asyncio
async def test_native_mcp_connection_contract(monkeypatch):
    events = []

    @asynccontextmanager
    async def client(**kwargs):
        assert kwargs == {
            "headers": {"x-auth": "fixture"},
            "timeout": 5,
            "follow_redirects": False,
            "transport": 0,
        }
        yield "client"
        events.append("client_closed")

    @asynccontextmanager
    async def stream(url, **kwargs):
        assert url == "https://fixture.invalid/mcp"
        assert kwargs == {"http_client": "client"}
        yield "reader", "writer"
        events.append("stream_closed")

    class Session:
        def __init__(self, reader, writer, *, read_timeout_seconds):
            assert (reader, writer) == ("reader", "writer")
            assert read_timeout_seconds == 5

        async def __aenter__(self):
            return self

        async def initialize(self):
            events.append("initialized")

        async def __aexit__(self, *args):
            events.append("session_closed")

    for name in ("httpx2", "mcp", "mcp.client", "mcp.client.streamable_http"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["httpx2"].AsyncClient = client
    sys.modules["httpx2"].AsyncHTTPTransport = lambda *, retries: retries
    sys.modules["mcp"].ClientSession = Session
    sys.modules["mcp.client.streamable_http"].streamable_http_client = stream
    async with rl._mcp("https://fixture.invalid/", {"header": "x-auth", "token": "fixture"}, 5):
        assert events == ["initialized"]
    assert events == ["initialized", "session_closed", "stream_closed", "client_closed"]


@pytest.fixture
def native(fixture, tmp_path, monkeypatch):
    for name in (
        "fti",
        "fti.trainers",
        "fti.trainers.miles",
        "fti.trainers.miles.parser",
        "fti.trainers.miles.recording",
        "miles",
        "miles.rollout",
        "miles.rollout.base_types",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["fti.trainers.miles.parser"].parse_tool_call = parse

    def recorder(*args):
        engine = args[-1]
        assert "Authorization" not in engine.headers and not engine.follow_redirects
        return Recorder()

    monkeypatch.setattr(rl, "_make_recorder", recorder)
    sys.modules["miles.rollout.base_types"].GenerateFnOutput = NS
    input = NS(
        args=NS(
            cyber_run_id=fixture.config["run_id"],
            cyber_output_root=str(tmp_path),
            partial_rollout=False,
            hf_checkpoint="/model",
            fleet_tito_model="qwen35",
            fleet_max_tokens_per_turn=4096,
            rollout_max_context_len=32768,
        ),
        sample=NS(
            metadata={"cyber_config": fixture.config, "split": "train"}, rollout_id=2, index=3
        ),
        state=NS(aborted=False, tokenizer=NS(chat_template="synthetic-template")),
        sampling_params={},
        evaluation=False,
    )
    monkeypatch.setenv("FLEET_API_KEY", "synthetic-key")
    fixture.inputs = []

    async def collect(config, directory, recorder, parser, *, client):
        fixture.inputs.append((config, directory))
        assert client.headers["Authorization"] == "Bearer synthetic-key"
        assert parser is parse and client.follow_redirects is False
        return ["native-sample"]

    monkeypatch.setattr(rl, "collect", collect)
    return input


@pytest.mark.asyncio
@pytest.mark.parametrize("dev", [False, True])
async def test_native_hook_exact_attempt_and_split(native, fixture, dev):
    native.evaluation = dev
    native.sample.metadata["split"] = "dev" if dev else "train"
    output = await rl.generate(native)
    assert output.samples == "native-sample"
    config, directory = fixture.inputs[0]
    assert config["run_id"].endswith(f"{'dev' if dev else 'train'}-r2-s3")
    assert config["config_sha256"] == fleet.digest_without(config, "config_sha256")
    assert directory.name == config["run_id"]
    assert fixture.config["run_id"] == "synthetic-episode-001"  # Deep copy, no template mutation.


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["synthetic rendered prompt", "drift", []])
async def test_frozen_native_input_prompt_is_verified(native, fixture, prompt):
    fixture.config["initial_prompt_sha256"] = fleet.sha256(b"synthetic rendered prompt")
    seal(fixture.config)
    native.sample.prompt = prompt
    if prompt == "synthetic rendered prompt":
        assert (await rl.generate(native)).samples == "native-sample"
    else:
        with pytest.raises(rl.InvalidEpisode, match="native_input_prompt_drift"):
            await rl.generate(native)
        assert fixture.inputs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["partial", "split", "identity", "run", "path", "auth", "model", "template", "dict"]
)
async def test_native_preflight_stops_before_environment(native, fixture, fault, monkeypatch):
    if fault == "partial":
        native.args.partial_rollout = True
    elif fault == "split":
        native.evaluation = True
    elif fault == "identity":
        native.sample.index = None
    elif fault == "run":
        native.args.cyber_run_id = "other"
    elif fault == "path":
        native.args.cyber_output_root = "relative"
    elif fault == "model":
        native.args.hf_checkpoint = "/other"
    elif fault == "template":
        native.state.tokenizer.chat_template = "changed"
    elif fault == "dict":
        native.state.tokenizer.chat_template = {"default": "synthetic-template"}
    else:
        monkeypatch.delenv("FLEET_API_KEY")
    with pytest.raises(rl.InvalidEpisode):
        await rl.generate(native)
    assert fixture.inputs == []


@pytest.mark.asyncio
async def test_budget_must_leave_prompt_context(fixture, tmp_path):
    fixture.config["rl"]["max_tokens_per_turn"] = fixture.config["rl"]["context_tokens"]
    seal(fixture.config)
    with pytest.raises(rl.InvalidEpisode, match="generation_budget_exceeds_context"):
        await collect(fixture, tmp_path)
    assert not fixture.calls


@pytest.mark.asyncio
async def test_native_recorder_errors_are_sanitized(native, monkeypatch):
    def fail(*args):
        raise ValueError("private prompt data")

    monkeypatch.setattr(rl, "_make_recorder", fail)
    with pytest.raises(rl.InvalidEpisode) as caught:
        await rl.generate(native)
    assert str(caught.value) == "episode_failure_ValueError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ValueError("private token"), rl.InvalidEpisode("safe_reason"), asyncio.CancelledError()],
)
async def test_native_exception_boundary_never_exposes_sdk_text(native, monkeypatch, error):
    async def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(rl, "collect", fail)
    with pytest.raises((rl.InvalidEpisode, asyncio.CancelledError)) as caught:
        await rl.generate(native)
    assert "private token" not in str(caught.value)


def test_native_flags_are_required_and_typed():
    import argparse

    parser = argparse.ArgumentParser()
    rl.generate.add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--cyber-run-id",
            "fixture",
            "--cyber-output-root",
            "/private/fixture",
            "--fleet-tito-model",
            "qwen35",
            "--fleet-max-tokens-per-turn",
            "4096",
        ]
    )
    assert args.fleet_max_tokens_per_turn == 4096 and args.fleet_tito_model == "qwen35"


@pytest.mark.asyncio
async def test_nontext_observation_is_rejected(fixture, tmp_path, monkeypatch):
    @asynccontextmanager
    async def mcp(*args):
        class Session:
            async def list_tools(self):
                return NS(
                    tools=[NS(name=t["name"], model_dump=lambda _t=t, **kw: _t) for t in CATALOG]
                )

            async def call_tool(self, *args, **kwargs):
                return NS(content=[NS(type="image")], isError=False)

        yield Session()

    monkeypatch.setattr(rl, "_mcp", mcp)
    with pytest.raises(rl.InvalidEpisode, match="non_text_tool_result"):
        await collect(fixture, tmp_path)
    assert fixture.deleted


@pytest.mark.asyncio
async def test_nonobject_create_has_unknown_resource_state(fixture, tmp_path):
    fixture.overrides["POST", f"/v1/rollout-rewards/synthetic/versions/{TASK}/instances"] = (
        httpx.Response(200, json=[])
    )
    with pytest.raises(RuntimeError, match="not an object"):
        await collect(fixture, tmp_path)
    cleanup = json.loads((tmp_path / "episode/cleanup.json").read_text())
    assert cleanup["possible_instance_leak"] and not cleanup["instance_closed"]
