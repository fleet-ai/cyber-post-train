"""Fast3 generation retry tests; synthetic HTTP only and no provider calls."""

from __future__ import annotations

import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from training import rl_episode, skyrl_fast3_rollout, skyrl_prod9_rollout
from training import skyrl_fast3_retry as retry


class NativeClient:
    def __init__(
        self, *, proxy_url, server_urls, data_parallel_size, model_name, tokenizer
    ) -> None:
        self.proxy_url = proxy_url
        self.server_urls = server_urls
        self.data_parallel_size = data_parallel_size
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.uses_lora_weight_sync = False
        self.enable_return_routed_experts = False


class Response:
    def __init__(self, status: int, value=None, *, invalid=False) -> None:
        self.status_code = status
        self.value = value
        self.invalid = invalid
        self.json_calls = 0

    def json(self):
        self.json_calls += 1
        if self.invalid:
            raise ValueError("private malformed response")
        return self.value


class AsyncClient:
    instances = []
    outcomes = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls = []
        self.wire_bytes = []
        self.outcomes = list(type(self).outcomes)
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, *, json, headers):
        self.calls.append((url, copy.deepcopy(json), copy.deepcopy(headers)))
        self.wire_bytes.append(httpx.Request("POST", url, json=json, headers=headers).content)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def engine(monkeypatch):
    AsyncClient.instances = []
    monkeypatch.setattr(
        retry.skyrl_episode,
        "_module",
        lambda *args: SimpleNamespace(RemoteInferenceClient=NativeClient),
    )
    monkeypatch.setattr(retry.httpx, "AsyncClient", AsyncClient)
    return NativeClient(
        proxy_url="https://generation.invalid",
        server_urls=["http://worker.invalid"],
        data_parallel_size=1,
        model_name="/mnt/sfs/models/qwen",
        tokenizer=object(),
    )


async def _post(engine, outcomes, *, policy=None):
    AsyncClient.outcomes = outcomes
    context = retry.bounded_retry_engine(
        engine, object(), 30, policy=policy or retry.expected_policy()
    )
    async with context as client:
        return await client._post(
            "https://generation.invalid/inference/v1/generate",
            {"prompt_token_ids": [1, 2, 3]},
            {"Content-Type": "application/json"},
        )


def test_shared_v8_episode_collectors_remain_byte_identical() -> None:
    root = Path(__file__).resolve().parents[1]
    import hashlib

    assert hashlib.sha256((root / "training/rl_episode.py").read_bytes()).hexdigest() == (
        "00b6f38ebc33cead6dcb5ebb468c04d8192314333094b5767147e733cecc4ff8"
    )
    assert hashlib.sha256((root / "training/skyrl_episode.py").read_bytes()).hexdigest() == (
        "1642ffe9a51425bdc77e5ea9624efcfeecfd1c005387359df75249ebab2e421d"
    )


@pytest.mark.asyncio
async def test_retryable_statuses_reuse_request_and_admit_only_success_json(
    engine, monkeypatch
) -> None:
    private = Response(503, {"tokens": [999], "private": "do-not-admit"})
    second = Response(503, {"text": "still private"})
    success = Response(200, {"text": "accepted", "tokens": [7]})
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    result = await _post(engine, [private, second, success])

    client = AsyncClient.instances[-1]
    assert result == {"text": "accepted", "tokens": [7]}
    assert len(client.calls) == 3
    assert client.calls[0] == client.calls[1] == client.calls[2]
    assert client.wire_bytes[0] == client.wire_bytes[1] == client.wire_bytes[2]
    assert sleeps == [1, 2]
    assert private.json_calls == second.json_calls == 0
    assert success.json_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [300, 302, 400, 418, 428, 430, 499])
async def test_nonretryable_3xx_and_other_4xx_make_one_call(engine, monkeypatch, status) -> None:
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    with pytest.raises(retry.GenerationHTTPFailure) as caught:
        await _post(engine, [Response(status, {"private": "never parsed"})])

    assert len(AsyncClient.instances[-1].calls) == 1
    assert sleeps == []
    assert caught.value.status_code == status
    assert caught.value.diagnostic == {"attempts": 1}


@pytest.mark.asyncio
async def test_transport_and_invalid_json_each_make_one_call(engine, monkeypatch) -> None:
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    request = httpx.Request("POST", "https://generation.invalid/inference/v1/generate")
    with pytest.raises(rl_episode.InvalidEpisode, match="generation_transport_failure"):
        await _post(engine, [httpx.ConnectError("private transport", request=request)])
    assert len(AsyncClient.instances[-1].calls) == 1
    with pytest.raises(rl_episode.InvalidEpisode, match="generation_invalid_json"):
        await _post(engine, [Response(200, invalid=True)])
    assert len(AsyncClient.instances[-1].calls) == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_429_then_2xx_retries_once_and_admits_success(engine, monkeypatch) -> None:
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    failed = Response(429, {"tokens": [123], "private": "ignored"})
    assert await _post(engine, [failed, Response(200, {"tokens": [7]})]) == {"tokens": [7]}
    assert len(AsyncClient.instances[-1].calls) == 2
    assert sleeps == [1]
    assert failed.json_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 599])
async def test_terminal_retry_evidence_is_numeric_and_payload_free(
    engine, monkeypatch, status
) -> None:
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    outcomes = [Response(status, {"token": "SECRET"}) for _ in range(3)]
    with pytest.raises(retry.GenerationHTTPFailure) as caught:
        await _post(engine, outcomes)
    evidence = rl_episode._failure(
        caught.value, run_id="synthetic-fast3", elapsed_seconds=1.25, phase="agent_interaction"
    )
    sealed = retry.failure_receipt(caught.value, retry.expected_policy())

    assert len(AsyncClient.instances[-1].calls) == 3
    assert sleeps == [1, 2]
    assert isinstance(caught.value, retry.fleet.FleetRequestError)
    assert isinstance(caught.value, rl_episode.InvalidEpisode)
    assert evidence["http_status"] == status
    assert evidence["attempts"] == 3
    assert sealed["http_status"] == status and sealed["attempts"] == 3
    assert set(evidence) == {
        "error_type",
        "elapsed_seconds",
        "run_id",
        "method",
        "route",
        "http_status",
        "attempts",
        "phase",
        "causes",
    }
    assert set(sealed) == {"schema", "http_status", "attempts", "policy_sha256", "sha256"}
    assert "response_sha256" not in evidence and "response_sha256" not in sealed
    assert "SECRET" not in json.dumps([evidence, sealed])
    assert all(response.json_calls == 0 for response in outcomes)


def test_policy_is_type_exact_and_returned_as_a_trusted_copy() -> None:
    policy = retry.expected_policy()
    checked = retry.validate_policy(policy)
    policy["backoff_seconds"][0] = 99
    assert checked == retry.expected_policy()

    numeric_drift = retry.expected_policy()
    numeric_drift["max_http_attempts"] = 3.0
    with pytest.raises(rl_episode.InvalidEpisode, match="generation_retry_policy_invalid"):
        retry.validate_policy(numeric_drift)

    container_drift = retry.expected_policy()
    container_drift["backoff_seconds"] = (1, 2)
    with pytest.raises(rl_episode.InvalidEpisode, match="generation_retry_policy_invalid"):
        retry.validate_policy(container_drift)


def test_generator_rejects_invalid_policy_before_runtime_or_http(monkeypatch) -> None:
    AsyncClient.instances = []
    monkeypatch.setattr(
        skyrl_fast3_rollout.HistoricalGenerator,
        "__init__",
        lambda self, *args, **kwargs: None,
    )
    invalid = retry.expected_policy()
    invalid["retryable_http_statuses"]["exact"] = [429.0]
    with pytest.raises(rl_episode.InvalidEpisode, match="generation_retry_policy_invalid"):
        skyrl_fast3_rollout.Generator(generation_retry_policy=invalid)
    assert AsyncClient.instances == []


@pytest.mark.asyncio
async def test_retryable_status_then_invalid_json_does_not_make_third_call(
    engine, monkeypatch
) -> None:
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    with pytest.raises(rl_episode.InvalidEpisode, match="generation_invalid_json"):
        await _post(engine, [Response(500, {"private": "ignored"}), Response(200, invalid=True)])
    assert len(AsyncClient.instances[-1].calls) == 2
    assert sleeps == [1]


def _synthetic_generator(tmp_path: Path, *, policy: dict | None):
    generator = object.__new__(skyrl_fast3_rollout.Generator)
    generator.busy = generator.failed = False
    generator.root = tmp_path
    generator.manifest = {"sha256": "sha256:synthetic"}
    generator.tokenizer = object()
    generator.engine = object()
    generator.response_tokens = 8
    generator.concurrency = 1
    if policy is not None:
        generator.generation_retry_policy = policy
    config = {"run_id": "synthetic", "rl": {"episode_seconds": 1}}
    generator._inputs = lambda _batch: (
        {"phase": "train", "global_step": 0, "trajectory_ids": [("0", 0)]},
        "synthetic-batch",
        [config],
    )
    return generator


def _patch_generator_collection(monkeypatch, calls: list) -> None:
    class MarkerRecorder:
        def __init__(self, *args):
            calls.append(("recorder", args))

    async def collect(_config, _directory, recorder, _parse, *, client):
        assert isinstance(recorder, MarkerRecorder)
        assert client is not None
        return [
            SimpleNamespace(
                reward=1.0,
                tokens=[1, 2],
                response_length=1,
                loss_mask=[1],
                rollout_log_probs=[-0.1],
                metadata={"done_reason": "report_submitted"},
            )
        ]

    monkeypatch.setenv("FLEET_API_KEY", "synthetic")
    monkeypatch.setattr(skyrl_fast3_rollout.hardening, "Recorder", MarkerRecorder)
    monkeypatch.setattr(
        skyrl_fast3_rollout.skyrl_episode,
        "_module",
        lambda *_args: SimpleNamespace(__file__="/tmp/helper"),
    )
    monkeypatch.setattr(skyrl_fast3_rollout.rl_episode, "collect", collect)
    monkeypatch.setattr(skyrl_fast3_rollout.rl_episode, "validate_samples", lambda _samples: None)


@pytest.mark.asyncio
async def test_exact_fast3_policy_selects_bounded_generator_engine(tmp_path, monkeypatch) -> None:
    calls = []

    @asynccontextmanager
    async def bounded(engine, _tokenizer, timeout, *, policy):
        calls.append(("bounded", engine, timeout, copy.deepcopy(policy)))
        yield engine

    def forbidden(*_args, **_kwargs):
        raise AssertionError("single-attempt engine selected for exact Fast3 policy")

    _patch_generator_collection(monkeypatch, calls)
    monkeypatch.setattr(skyrl_fast3_rollout.skyrl_fast3_retry, "bounded_retry_engine", bounded)
    monkeypatch.setattr(skyrl_fast3_rollout.skyrl_episode, "single_attempt_engine", forbidden)
    generator = _synthetic_generator(tmp_path, policy=retry.expected_policy())
    result = await generator.generate(
        {
            "sampling_params": {},
            "trajectory_ids": [SimpleNamespace(instance_id="0", repetition_id=0)],
        }
    )

    bounded_call = next(call for call in calls if call[0] == "bounded")
    assert bounded_call[2:] == (1, retry.expected_policy())
    assert result["rewards"] == [1.0]
    collected = json.loads((tmp_path / "batches/synthetic-batch/COLLECTED.json").read_text())
    assert collected["recorder_implementation"] == skyrl_fast3_rollout.RECORDER_IMPLEMENTATION


@pytest.mark.asyncio
async def test_missing_policy_preserves_single_attempt_generator_path(
    tmp_path, monkeypatch
) -> None:
    calls = []

    @asynccontextmanager
    async def single(engine, _tokenizer, timeout):
        calls.append(("single", engine, timeout))
        yield engine

    def forbidden(*_args, **_kwargs):
        raise AssertionError("bounded engine selected without the exact Fast3 policy")

    _patch_generator_collection(monkeypatch, calls)
    monkeypatch.setattr(skyrl_fast3_rollout.skyrl_episode, "single_attempt_engine", single)
    monkeypatch.setattr(skyrl_fast3_rollout.skyrl_fast3_retry, "bounded_retry_engine", forbidden)
    generator = _synthetic_generator(tmp_path, policy=None)
    result = await generator.generate(
        {
            "sampling_params": {},
            "trajectory_ids": [SimpleNamespace(instance_id="0", repetition_id=0)],
        }
    )

    assert next(call for call in calls if call[0] == "single")[2] == 1
    assert result["rewards"] == [1.0]
    assert not hasattr(generator, "generation_retry_policy")


def test_historical_prod9_generator_has_no_retry_policy_surface() -> None:
    assert "generation_retry_policy" not in skyrl_prod9_rollout.Generator.__dict__
    assert "bounded_retry_engine" not in skyrl_prod9_rollout.__dict__
