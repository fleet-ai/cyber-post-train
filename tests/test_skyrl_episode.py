"""Synthetic episodes and real native interfaces; never contact Fleet or a model."""

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from test_rl_episode import fixture as fleet_fixture  # noqa: F401
from test_rl_episode import seal

from evals.fleet import opencode_self_hosted as fleet
from training import rl_episode
from training import skyrl_episode as sky


class Tokenizer:
    chat_template = "synthetic-template"
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(
            tools=[], tokenize=True, return_dict=False, add_generation_prompt=True
        )
        return [1, 2]

    def encode(self, text, **kwargs):
        assert text == "\n" and kwargs == {"add_special_tokens": False}
        return [10]


class Engine:
    model_name = "/model"

    def __init__(self):
        self.requests = []
        self.reply = {
            "responses": ['<tool_call>{"name":"submit_report","arguments":{}}</tool_call>'],
            "response_ids": [[77, 9]],
            "response_logprobs": [[-0.3, -0.2]],
            "stop_reasons": ["stop"],
        }

    async def generate(self, request):
        self.requests.append(copy.deepcopy(request))
        if isinstance(self.reply, Exception):
            raise self.reply
        return copy.deepcopy(self.reply)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    helpers = NS(
        __globals__={
            "get_generation_prompt_ids": lambda tokenizer: [12],
            "encode_messages_subset": lambda messages, tokenizer: [11],
        }
    )
    monkeypatch.setattr(sky, "native_helper", lambda path: helpers)
    config = {
        "model": {
            "repo": "Qwen/Qwen3.8-27B",
            "root": "/model",
            "runtime_chat_template_sha256": fleet.sha256(b"synthetic-template"),
        },
        "rl": {"context_tokens": 20, "max_tokens_per_turn": 8},
        "initial_prompt_tokens_sha256": fleet.sha256(fleet.canonical_json([1, 2])),
    }
    return NS(
        config=config,
        tokenizer=Tokenizer(),
        engine=Engine(),
        sampling={"temperature": 0.7},
        helper=tmp_path / "helper",
        response_tokens=18,
    )


def recorder(setup):
    return sky.Recorder(
        setup.config,
        setup.tokenizer,
        setup.engine,
        setup.sampling,
        setup.response_tokens,
        setup.helper,
    )


@pytest.mark.asyncio
async def test_recording_keeps_sampled_ids_and_masks_only_observations(setup):
    value = recorder(setup)
    value.begin_segment([], [])
    turn = await value.sample()
    assert value.append_assistant(turn.text, sky.parse(turn.text), 0)["content"] == turn.text
    observation = {"role": "tool", "content": "synthetic", "name": "submit_report"}
    assert value.append_observation(observation, []) == observation
    record = value.finalize(0.0, {"verifier_execution_id": "fixture"}, 0)[0]
    assert record.tokens == [1, 2, 77, 9, 10, 11, 12]
    assert record.loss_mask == [1, 1, 0, 0, 0]
    assert record.rollout_log_probs == [-0.3, -0.2, 0, 0, 0]
    assert record.reward == 0.0 and record.response_length == 5
    assert setup.engine.requests == [
        {
            "prompt_token_ids": [[1, 2]],
            "sampling_params": {"temperature": 0.7, "logprobs": 0, "max_tokens": 8},
        }
    ]
    for operation in (
        lambda: value.finalize(1, {}, 0),
        lambda: value.append_observation(observation, []),
        lambda: value.begin_segment([], []),
        lambda: value.append_assistant("changed", None, 1),
    ):
        with pytest.raises(rl_episode.InvalidEpisode):
            operation()
    with pytest.raises(rl_episode.InvalidEpisode):
        await value.sample()


@pytest.mark.parametrize(
    "field,value", [("repo", "other"), ("root", "other"), ("runtime_chat_template_sha256", "bad")]
)
def test_model_drift(setup, field, value):
    setup.config["model"][field] = value
    with pytest.raises(rl_episode.InvalidEpisode, match="model_or_budget"):
        recorder(setup)


@pytest.mark.parametrize("tokens", [True, 0, 20])
def test_response_budget(setup, tokens):
    setup.response_tokens = tokens
    with pytest.raises(rl_episode.InvalidEpisode, match="budget"):
        recorder(setup)


@pytest.mark.parametrize(
    "params",
    [{"n": 2}, {"ignore_eos": True}, {"stop": ["x"]}, {"stop_token_ids": [5]}, {"logprobs": 3}],
)
def test_sampling_rejects_unreviewed_stops(setup, params):
    setup.sampling.update(params)
    with pytest.raises(rl_episode.InvalidEpisode, match="sampling"):
        recorder(setup)


@pytest.mark.parametrize("tokens", [[], [True], [-1], [4], list(range(19))])
def test_prompt_drift_or_overflow(setup, tokens, monkeypatch):
    monkeypatch.setattr(setup.tokenizer, "apply_chat_template", lambda *a, **k: tokens)
    with pytest.raises(rl_episode.InvalidEpisode, match="initial_prompt"):
        recorder(setup).begin_segment([], [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("responses", []),
        ("response_ids", None),
        ("response_logprobs", []),
        ("stop_reasons", []),
        ("responses", [None]),
        ("response_ids", [[]]),
        ("response_ids", [[1, 2]]),
        ("response_ids", [[True, 9]]),
        ("response_ids", [[-1, 9]]),
        ("response_ids", [[1] * 8 + [9]]),
        ("response_logprobs", [None]),
        ("response_logprobs", [[-0.1]]),
        ("response_logprobs", [[float("nan"), 0.0]]),
        ("response_logprobs", [[True, 0.0]]),
        ("response_logprobs", [[0.2, 0.0]]),
        ("stop_reasons", ["length"]),
        ("stop_reasons", ["abort"]),
    ],
)
async def test_invalid_generation_never_becomes_reward_zero(setup, field, value):
    setup.engine.reply[field] = value
    value = recorder(setup)
    value.begin_segment([], [])
    with pytest.raises(rl_episode.InvalidEpisode):
        await value.sample()
    assert value.recording.response_length == 0 and len(setup.engine.requests) == 1


@pytest.mark.asyncio
async def test_transport_error_sanitized_without_retry(setup):
    setup.engine.reply = ValueError("private response body")
    value = recorder(setup)
    value.begin_segment([], [])
    with pytest.raises(rl_episode.InvalidEpisode, match="^generation_failure_ValueError$"):
        await value.sample()
    assert len(setup.engine.requests) == 1


@pytest.mark.asyncio
async def test_budgets_no_truncation_and_order(setup):
    value = recorder(setup)
    with pytest.raises(rl_episode.InvalidEpisode):
        await value.sample()
    with pytest.raises(rl_episode.InvalidEpisode):
        value.finalize(1, {}, 0)
    with pytest.raises(rl_episode.InvalidEpisode):
        value.append_observation({}, [])
    value.begin_segment([], [])
    with pytest.raises(rl_episode.InvalidEpisode):
        value.append_observation({}, [1])
    value.response_tokens = 2
    await value.sample()
    assert setup.engine.requests[-1]["sampling_params"]["max_tokens"] == 2
    with pytest.raises(rl_episode.InvalidEpisode, match="exhausted"):
        await value.sample()
    with pytest.raises(rl_episode.InvalidEpisode, match="exhausted"):
        value.append_observation({}, [])
    assert value.recording.tokens == [1, 2, 77, 9]


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [False, 0, -1])
async def test_bad_turn_cap(setup, cap):
    setup.sampling["max_tokens"] = cap
    value = recorder(setup)
    value.begin_segment([], [])
    with pytest.raises(rl_episode.InvalidEpisode, match="turn_budget"):
        await value.sample()
    assert not setup.engine.requests


@pytest.mark.parametrize(
    "body,expected",
    [
        (
            '{"name":"bash","arguments":{"script":"fixture"}}',
            {"name": "bash", "arguments": {"script": "fixture"}},
        ),
        (
            "<function=bash><parameter=script>fixture</parameter><parameter=timeoutMs>12</parameter></function>",
            {"name": "bash", "arguments": {"script": "fixture", "timeoutMs": 12}},
        ),
        (
            '<function=submit_report><parameter=flags>["synthetic"]</parameter><parameter=explanation></parameter></function>',
            {"name": "submit_report", "arguments": {"flags": ["synthetic"], "explanation": ""}},
        ),
        ('{"name":"bash",broken}', {"name": "bash", "arguments": {}}),
        ("broken", {"name": "", "arguments": {}}),
        ("[]", {"name": "", "arguments": {}}),
    ],
)
def test_native_parser_forms(body, expected):
    assert sky.parse("<tool_call>" + body + "</tool_call>") == expected


def test_parser_never_silently_drops_multiple_calls():
    assert sky.parse("no call") is None
    with pytest.raises(rl_episode.InvalidEpisode, match="multiple_calls"):
        sky.parse('<tool_call>{"name":"bash"}</tool_call>' * 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [0.0, 1.0])
async def test_shared_lifecycle_grades_once_and_confirms_release(
    setup,
    fleet_fixture,
    tmp_path,
    score,  # noqa: F811
):
    state = fleet_fixture
    state.config["model"] = setup.config["model"]
    state.config["initial_prompt_tokens_sha256"] = setup.config["initial_prompt_tokens_sha256"]
    seal(state.config)
    setup.config = state.config
    # This fixture's prompt token IDs deliberately stay fixed to isolate the
    # real lifecycle; the native-tokenizer fixture below checks text rendering.
    setup.tokenizer.apply_chat_template = lambda *a, **kw: [1, 2]
    state.reward["reward"] = state.reward["cyber_verification_result"]["reward"] = score
    async with state.client:
        samples = await rl_episode.collect(
            state.config, tmp_path / "episode", recorder(setup), sky.parse, client=state.client
        )
    assert state.deleted and samples[0].reward == score
    assert samples[0].metadata["verifier_execution_id"] == state.reward["verifier_execution_id"]
    receipt = json.loads((tmp_path / "episode/ACCEPTED.json").read_text())
    assert receipt["sample_count"] == 1
    assert sum(m == "POST" and p.endswith("/instances") for m, p in state.calls) == 1


@dataclass
class NativeClient:
    proxy_url: str = "http://synthetic.invalid"
    server_urls: tuple = ("http://synthetic.invalid",)
    data_parallel_size: int = 1
    model_name: str = "/model"
    tokenizer: object = None
    uses_lora_weight_sync: bool = False
    enable_return_routed_experts: bool = False


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(sky, "_module", lambda *args: NS(RemoteInferenceClient=NativeClient))
    return NativeClient()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"ok": True}),
        httpx.Response(503, text="private"),
        httpx.Response(200, text="private"),
        httpx.ConnectError("private"),
    ],
)
async def test_single_attempt_http_boundary(native, monkeypatch, response):
    seen = []

    def handler(request):
        seen.append(request)
        assert "authorization" not in request.headers
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda *, retries: httpx.MockTransport(handler)
    )
    async with sky.single_attempt_engine(native, Tokenizer(), 3) as engine:
        if getattr(response, "status_code", None) == 200 and response.content.startswith(b"{"):
            assert await engine._post(native.proxy_url + "/inference/v1/generate", {}) == {
                "ok": True
            }
        else:
            with pytest.raises(rl_episode.InvalidEpisode) as error:
                await engine._post(native.proxy_url + "/inference/v1/generate", {})
            assert "private" not in str(error.value)
        for url, headers in (
            ("http://other.invalid", {}),
            (native.proxy_url + "/inference/v1/generate", {"Authorization": "no"}),
        ):
            with pytest.raises(rl_episode.InvalidEpisode, match="unexpected"):
                await engine._post(url, {}, headers)
    assert len(seen) == 1 and type(native) is NativeClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "ftp://synthetic.invalid",
        "http:/missing",
        "http://name:secret@synthetic.invalid",
        "http://synthetic.invalid/?x",
        "http://synthetic.invalid/#x",
    ],
)
async def test_unsafe_engine_urls(native, url):
    native.proxy_url = url
    with pytest.raises(rl_episode.InvalidEpisode, match="unsupported"):
        async with sky.single_attempt_engine(native, Tokenizer(), 3):
            pytest.fail("invalid engine accepted")


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["uses_lora_weight_sync", "enable_return_routed_experts"])
async def test_unqualified_engine_modes(native, field):
    setattr(native, field, True)
    with pytest.raises(rl_episode.InvalidEpisode, match="unsupported"):
        async with sky.single_attempt_engine(native, Tokenizer(), 3):
            pytest.fail("invalid engine accepted")


def test_native_source_hash(tmp_path, monkeypatch):
    path = tmp_path / "native.py"
    path.write_text("# synthetic\n")
    module = NS(__file__=str(path))
    monkeypatch.setattr(sky.importlib, "import_module", lambda name: module)
    assert sky._module("fixture", hashlib.sha256(path.read_bytes()).hexdigest()) is module
    with pytest.raises(rl_episode.InvalidEpisode, match="source_drift"):
        sky._module("fixture", "wrong")


@pytest.mark.asyncio
async def test_real_skyrl_client_tokenizer_and_helpers(monkeypatch, tmp_path):
    native = pytest.importorskip(sky.CLIENT_MODULE).RemoteInferenceClient
    root = os.environ.get("CYBER_TEST_QWEN_ROOT")
    if not root:
        pytest.skip("pinned SkyRL image plus exact local Qwen tokenizer required")
    from training.corpus import local_tokenizer

    lock = json.loads(
        (Path(__file__).parents[1] / "configs/models/qwen38-27b-1d4bf0f2.lock.json").read_text()
    )
    tokenizer, _ = local_tokenizer(lock, Path(root))
    import skyrl

    helper = Path(skyrl.__file__).parent / "train/generators/utils.py"
    messages = [{"role": "user", "content": "synthetic task"}]
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=True, return_dict=False
    )
    config = {
        "model": {
            "repo": lock["repo"],
            "root": root,
            "runtime_chat_template_sha256": fleet.sha256(tokenizer.chat_template.encode()),
        },
        "rl": {"context_tokens": 4096, "max_tokens_per_turn": 1024},
        "initial_prompt_tokens_sha256": fleet.sha256(fleet.canonical_json(prompt)),
    }
    # Noncanonical sampled segmentation is intentional: re-encoding the
    # decoded text would change IDs and corrupt their log-probability mapping.
    text = (
        "<tool_call><function=bash><parameter=script>echo synthetic</parameter>"
        "</function></tool_call><|im_end|>"
    )
    ids = [
        t for c in "hello" for t in tokenizer.encode(c, add_special_tokens=False)
    ] + tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.encode(tokenizer.decode(ids), add_special_tokens=False) != ids
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "token_ids": ids,
                        "finish_reason": "stop",
                        "logprobs": {"content": [{"logprob": -0.1} for _ in ids]},
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda *, retries: httpx.MockTransport(handler)
    )
    original = native(
        proxy_url="http://synthetic.invalid",
        server_urls=["http://synthetic.invalid"],
        data_parallel_size=1,
        model_name=root,
    )
    async with sky.single_attempt_engine(original, tokenizer, 5) as engine:
        record = sky.Recorder(config, tokenizer, engine, {"temperature": 0.8}, 2048, helper)
        record.begin_segment(messages, tools)
        turn = await record.sample()
        call = sky.parse(turn.text)
        assert call == {"name": "bash", "arguments": {"script": "echo synthetic"}}
        record.append_assistant(turn.text, call, 0)
        observation = {
            "role": "tool",
            "tool_call_id": "call_000000",
            "name": "bash",
            "content": "synthetic output",
        }
        record.append_observation(observation, [])
        first = list(record.recording.tokens)
        await record.sample()
        result = record.finalize(0, {}, 0)[0]
    assert requests[0]["token_ids"] == prompt and requests[1]["token_ids"] == first
    assert result.tokens[len(prompt) : len(prompt) + len(ids)] == ids
    assert sum(result.loss_mask) == 2 * len(ids)
    # Verify the template-owned suffix, including the EOS newline; no assistant
    # history retokenization participates in the actual sampled trajectory.
    tail = tokenizer.decode(first[len(prompt) + len(ids) :])
    assert tail.startswith("\n<|im_start|>user\n<tool_response>\nsynthetic output")
    assert tail.endswith("<|im_start|>assistant\n") or tail.endswith(
        "<|im_start|>assistant\n<think>\n"
    )
    assert original._session is None
