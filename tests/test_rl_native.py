"""Native-image integration tests; all task/engine replies here are synthetic."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from training import rl_episode as rl


@pytest.mark.asyncio
async def test_real_mcp_transport_initializes_calls_and_closes(monkeypatch):
    pytest.importorskip("mcp")
    http = pytest.importorskip("httpx2")
    calls = []

    def handler(request):
        assert request.headers["x-fixture-auth"] == "synthetic"
        if request.method == "GET":
            return http.Response(405)
        if request.method == "DELETE":
            calls.append("DELETE")
            return http.Response(204)
        body = json.loads(request.content)
        method = body["method"]
        calls.append(method)
        if method == "notifications/initialized":
            return http.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "synthetic", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{"name": "submit_report", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            assert body["params"]["name"] == "submit_report"
            assert body["params"]["arguments"] == {}
            result = {"content": [{"type": "text", "text": "synthetic result"}], "isError": False}
        else:
            raise AssertionError(f"unexpected synthetic method: {method}")
        return http.Response(
            200,
            headers={"mcp-session-id": "synthetic-session"},
            json={"jsonrpc": "2.0", "id": body["id"], "result": result},
        )

    def transport(*, retries):
        assert retries == 0
        return http.MockTransport(handler)

    monkeypatch.setattr(http, "AsyncHTTPTransport", transport)
    async with asyncio.timeout(10):
        async with rl._mcp(
            "https://synthetic.invalid", {"header": "x-fixture-auth", "token": "synthetic"}, 3
        ) as session:
            assert [t.name for t in (await session.list_tools()).tools] == ["submit_report"]
            result = await session.call_tool("submit_report", arguments={})
            assert result.content[0].text == "synthetic result" and not result.is_error
    assert calls == [
        "initialize", "notifications/initialized", "tools/list", "tools/call", "DELETE"
    ]


@pytest.fixture
def native_recorder(monkeypatch):
    recording = pytest.importorskip("fti.trainers.miles.recording")
    from miles.rollout.base_types import GenerateFnInput
    from miles.utils.types import Sample
    from transformers import AutoTokenizer

    root = os.environ.get("CYBER_TEST_QWEN_ROOT")
    if not root:
        pytest.skip("native image plus staged exact Qwen tokenizer required")
    lock = json.loads(
        (Path(__file__).parents[1] / "configs/models/qwen38-27b-1d4bf0f2.lock.json").read_text()
    )
    for item in lock["tokenizer"]["files"]:
        actual = hashlib.sha256((Path(root) / item["path"]).read_bytes()).hexdigest()
        assert actual == item["sha256"]
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    # FTI's recipe binds this override on BOTH native training and sampling.
    # The exact HF template rejects Miles' synthetic system/assistant prefix.
    template = (Path(recording.__file__).parent / "templates/qwen3.8_fixed.jinja").read_bytes()
    assert hashlib.sha256(template).hexdigest() == (
        "38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe"
    )
    tokenizer.chat_template = template.decode()
    args = NS(
        fleet_tito_model="qwen35",  # Exact FTI Qwen3.8-27B recipe's text family.
        fleet_max_tokens_per_turn=4096,
        rollout_max_context_len=32768,
        rollout_max_response_len=8192,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=9,
        sglang_router_policy="round_robin",
        sglang_speculative_algorithm=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
    )
    state = NS(args=args, tokenizer=tokenizer, processor=None, aborted=False)
    sample = Sample(group_index=0, index=0, rollout_id=0)
    input = GenerateFnInput(state, sample, {"temperature": 1.0}, evaluation=False)
    assert input.args is args  # Native property, not a permissive Namespace mock.
    recording._TITO_CACHE.clear()
    recorder = recording.Recorder(args, state, sample, input.sampling_params)
    fixture = NS(recorder=recorder, tokens=[], calls=0, finish="stop", state=state)

    async def post(url, payload, *, headers):
        assert url == "http://127.0.0.1:9/generate" and headers is None
        assert payload["return_logprob"] is True
        assert payload["sampling_params"]["max_new_tokens"] == 4096
        function = (
            "<function=bash>\n<parameter=command>\necho fixture\n</parameter>\n</function>"
            if fixture.calls == 0
            else "<function=submit_report>\n</function>"
        )
        text = f"<tool_call>\n{function}\n</tool_call><|im_end|>"
        ids = tokenizer.encode(text, add_special_tokens=False)
        fixture.tokens.extend(ids)
        fixture.calls += 1
        return {
            "text": text,
            "meta_info": {
                "finish_reason": {"type": fixture.finish},
                "output_token_logprobs": [[-0.1, token, None] for token in ids],
                "weight_version": "synthetic-weight-0",
                "completion_tokens": len(ids),
            },
        }

    monkeypatch.setattr(recording, "post", post)  # Only the synthetic engine, not token assembly.
    yield fixture
    recording._TITO_CACHE.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [0.0, 1.0])
async def test_real_recorder_preserves_sampled_tokens_and_masks_tools(native_recorder, reward):
    from fti.trainers.miles.parser import parse_tool_call
    from mcp.types import CallToolResult, TextContent

    calls = []

    class Session:
        async def call_tool(self, name, arguments):
            calls.append(name)
            return CallToolResult(content=[TextContent(type="text", text="synthetic observation")])

    fixture = native_recorder
    messages = [{"role": "user", "content": "Synthetic tool-protocol fixture; no real task."}]
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
        for name in ("bash", "submit_report")
    ]
    messages, reason, duration = await rl._agent(
        fixture.recorder,
        Session(),
        messages,
        tools,
        {"max_turns": 4, "tool_seconds": 2, "tool_result_chars": 100},
        parse_tool_call,
    )
    assert reason == "report_submitted" and calls == ["bash", "submit_report"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant", "tool"]
    samples = fixture.recorder.finalize(reward, {"synthetic": True}, duration)
    rl.validate_samples(samples)
    assert len(samples) == 1 and fixture.calls == 2
    sample = samples[0]
    sample.validate()
    assert sample.reward == reward and sample.loss_mask.count(0) > 0
    response = sample.tokens[-sample.response_length :]
    assert [t for t, mask in zip(response, sample.loss_mask, strict=True) if mask] == fixture.tokens
    assert all(
        p == -0.1 if m else p == 0
        for p, m in zip(sample.rollout_log_probs, sample.loss_mask, strict=True)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "abort"])
async def test_real_recorder_rejects_incomplete_generation(native_recorder, finish):
    fixture = native_recorder
    fixture.finish = finish
    with pytest.raises(rl.InvalidEpisode, match="generation_incomplete"):
        await rl._agent(
            fixture.recorder,
            None,
            [{"role": "user", "content": "Synthetic incomplete response."}],
            [],
            {"max_turns": 2},
            lambda text: None,
        )
    assert fixture.calls == 1
