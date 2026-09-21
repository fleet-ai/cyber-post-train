"""SkyRL token recording for the shared, single-attempt Fleet episode lifecycle.

Internal integration, not a qualified RL launcher. SkyRL owns generation and
optimization. Sampled token IDs are never reconstructed from text. Each
assistant turn (including working-memory compaction) remains a separate policy
step with its exact prompt, sampled IDs and log probabilities. No fallback
reward, silent truncation or resampling is allowed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx

from evals.fleet import opencode_self_hosted as fleet

from .dense import native_helper
from .qwen_tools import parse_tool_calls
from .rl_episode import EpisodeBudgetExceeded, InvalidEpisode, validate_samples

CLIENT_MODULE = "skyrl.backends.skyrl_train.inference_servers.remote_inference_client"
CLIENT_SHA256 = "7a798659decf8a49b9ab28c5fc299cdb0f3b1eedaa1db791e7a023f467bda174"
COMPACTION_PROMPT = """Create a compact working memory for continuing this cyber task.
Keep the original objective, confirmed facts, discovered services and vulnerabilities,
credentials or tokens that are safe to retain inside this isolated task, commands and
results that changed the investigation, failed hypotheses, and the most useful next
actions. Do not invent facts. Do not call tools or submit a report. Return only the
working memory."""
CONTINUATION_PROMPT = """Original task:
{task}

Continue solving the original task using the compact working
memory below. Treat it as a lossy memory, verify important facts when needed, use only
the available task tools, and submit the report only when it is ready.

Compact working memory:
{summary}"""


def _module(name, expected):
    module = importlib.import_module(name)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
        raise InvalidEpisode("native_skyrl_source_drift")
    return module


def parse(text):
    """Keep every parsed call from one sampled turn in its original order."""
    calls = parse_tool_calls(text)
    if not calls:
        return None
    parsed = []
    for call in calls:
        function = call["function"]
        parsed.append({"name": function["name"], "arguments": json.loads(function["arguments"])})
    return parsed[0] if len(parsed) == 1 else parsed


@asynccontextmanager
async def single_attempt_engine(engine, tokenizer, timeout):
    """Clone only data-plane routing; never alter the trainer's weight-sync client.

    The pinned native client retries disconnected generation POSTs. Its payload
    builder/decoder stay native; this subclass replaces only that HTTP boundary.
    No session ID is sent, so no sticky-router session lease is created.
    """
    native = _module(CLIENT_MODULE, CLIENT_SHA256).RemoteInferenceClient
    url = urlsplit(engine.proxy_url)
    if (
        type(engine) is not native
        or url.scheme not in {"http", "https"}
        or not url.netloc
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or engine.uses_lora_weight_sync
        or engine.enable_return_routed_experts
        or type(timeout) is not int
        or timeout <= 0
    ):
        raise InvalidEpisode("unsupported_skyrl_engine")
    endpoint = engine.proxy_url.rstrip("/")
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=False, transport=httpx.AsyncHTTPTransport(retries=0)
    ) as client:

        class SingleAttemptClient(native):
            async def _post(self, url, json, headers=None):
                if url != endpoint + "/inference/v1/generate" or set(headers or {}) - {
                    "Content-Type"
                }:
                    raise InvalidEpisode("unexpected_skyrl_data_request")
                try:
                    response = await client.post(url, json=json, headers=headers)
                except httpx.HTTPError:
                    raise InvalidEpisode("generation_transport_failure") from None
                if not 200 <= response.status_code < 300:
                    raise InvalidEpisode(f"generation_http_{response.status_code}")
                try:
                    return response.json()
                except ValueError:
                    raise InvalidEpisode("generation_invalid_json") from None

        yield SingleAttemptClient(
            proxy_url=endpoint,
            server_urls=list(engine.server_urls),
            data_parallel_size=engine.data_parallel_size,
            model_name=engine.model_name,
            tokenizer=tokenizer,
        )


class Recorder:
    """Qwen ChatML recorder accepted by ``training.rl_episode.collect``.

    The supplied helper is the pinned SkyRL ``generators/utils.py`` file. Fresh
    data must bind this template and initial token sequence independently from
    Miles' different native template/recorder contract. A length stop is only a
    generation-chunk boundary: the exact sampled IDs are appended and generation
    continues until EOS or the declared per-turn cap. If history is compacted,
    the changed prompt begins a new policy step instead of pretending the prompt
    was append-only.
    """

    def __init__(self, config, tokenizer, engine, sampling, response_tokens, helper: Path):
        self.config, self.tokenizer, self.engine = config, tokenizer, engine
        if (
            config["model"]["repo"] != "Qwen/Qwen3.8-27B"
            or config["model"]["root"] != engine.model_name
            or fleet.sha256(tokenizer.chat_template.encode())
            != config["model"]["runtime_chat_template_sha256"]
            or type(response_tokens) is not int
            or response_tokens <= 0
        ):
            raise InvalidEpisode("skyrl_model_or_budget_drift")
        if (
            sampling.get("n", 1) != 1
            or sampling.get("ignore_eos", False)
            or sampling.get("stop")
            or sampling.get("stop_token_ids") not in (None, [], [tokenizer.eos_token_id])
            or sampling.get("logprobs") not in (None, 0)
        ):
            raise InvalidEpisode("unsupported_skyrl_sampling")
        self.sampling = dict(sampling, logprobs=0)
        # Loading the helper pins this integration to the reviewed native SkyRL
        # source. Step-wise prompts are rendered directly by the tokenizer so a
        # compacted prompt never masquerades as an append-only continuation.
        native_helper(helper)
        limits = config["rl"]
        if set(limits) - {
            "context_tokens",
            "max_tokens_per_turn",
            "generation_chunk_tokens",
            "compaction_trigger_tokens",
            "compaction_summary_tokens",
            "max_turns",
            "episode_seconds",
            "tool_seconds",
            "tool_result_chars",
        } or any(
            type(limits.get(key)) is not int or limits[key] <= 0
            for key in (
                "generation_chunk_tokens",
                "compaction_trigger_tokens",
                "compaction_summary_tokens",
            )
        ):
            raise InvalidEpisode("skyrl_compaction_contract_drift")
        self.response_tokens = response_tokens
        self.messages = self.tools = self.initial_task = None
        self.steps = []
        self.response_length = 0
        self.compactions = 0
        self.prompt = self.recording = None
        self.last_text = None
        self.finalized = False

    def begin_segment(self, messages, tools):
        if self.prompt is not None:
            raise InvalidEpisode("recording_already_started")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or messages[0].get("role") != "user"
            or not isinstance(messages[0].get("content"), str)
            or not isinstance(tools, list)
        ):
            raise InvalidEpisode("skyrl_initial_messages_invalid")
        tokens = self._render(messages, tools)
        if (
            len(tokens) + self.config["rl"]["max_tokens_per_turn"]
            > self.config["rl"]["context_tokens"]
            or fleet.sha256(fleet.canonical_json(tokens))
            != self.config["initial_prompt_tokens_sha256"]
        ):
            raise InvalidEpisode("skyrl_initial_prompt_drift")
        self.messages = messages
        self.tools = tools
        self.initial_task = copy.deepcopy(messages[0])
        self.prompt = tokens

    def _render(self, messages, tools):
        tokens = list(
            self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
            )
        )
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise InvalidEpisode("skyrl_prompt_tokens_invalid")
        return tokens

    def _start_step(self, prompt, kind):
        if self.recording is not None:
            raise InvalidEpisode("skyrl_step_already_active")
        if (
            kind not in {"action", "compaction"}
            or not prompt
            or len(prompt) >= self.config["rl"]["context_tokens"]
        ):
            raise InvalidEpisode("skyrl_step_prompt_invalid")
        self.recording = SimpleNamespace(
            tokens=list(prompt),
            loss_mask=[],
            rollout_log_probs=[],
            response_length=0,
            step_kind=kind,
        )

    async def _generate_complete(self, prompt, *, turn_tokens, kind):
        if type(turn_tokens) is not int or turn_tokens <= 0:
            raise InvalidEpisode("invalid_turn_budget")
        self._start_step(prompt, kind)
        record = self.recording
        texts = []
        while True:
            episode_remaining = self.response_tokens - self.response_length
            turn_remaining = turn_tokens - record.response_length
            context_remaining = self.config["rl"]["context_tokens"] - len(record.tokens)
            if episode_remaining <= 0:
                raise EpisodeBudgetExceeded("response_budget_exhausted")
            if turn_remaining <= 0:
                raise EpisodeBudgetExceeded("turn_response_budget_exhausted")
            if context_remaining <= 0:
                raise EpisodeBudgetExceeded("generation_incomplete_context_full")
            cap = min(
                self.config["rl"]["generation_chunk_tokens"],
                episode_remaining,
                turn_remaining,
                context_remaining,
            )
            params = dict(self.sampling, max_tokens=min(self.sampling.get("max_tokens", cap), cap))
            if type(params["max_tokens"]) is not int or params["max_tokens"] <= 0:
                raise InvalidEpisode("invalid_turn_budget")
            try:
                out = await self.engine.generate(
                    {"prompt_token_ids": [list(record.tokens)], "sampling_params": params}
                )
                if any(
                    len(out.get(key) or []) != 1
                    for key in ("responses", "response_ids", "response_logprobs", "stop_reasons")
                ):
                    raise InvalidEpisode("generation_batch_invalid")
                text, ids, probabilities, stop = (
                    out[key][0]
                    for key in ("responses", "response_ids", "response_logprobs", "stop_reasons")
                )
                if (
                    not isinstance(text, str)
                    or not ids
                    or len(ids) > params["max_tokens"]
                    or not isinstance(probabilities, list)
                    or len(probabilities) != len(ids)
                    or any(type(token) is not int or token < 0 for token in ids)
                    or any(
                        type(probability) not in (float, int)
                        or not math.isfinite(probability)
                        or probability > 1e-6
                        for probability in probabilities
                    )
                ):
                    raise InvalidEpisode("generation_tokens_or_stop_invalid")
                if stop == "length":
                    if len(ids) != params["max_tokens"] or ids[-1] == self.tokenizer.eos_token_id:
                        raise InvalidEpisode("generation_tokens_or_stop_invalid")
                    self._append(ids, probabilities)
                    texts.append(text)
                    continue
                if stop != "stop" or ids[-1] != self.tokenizer.eos_token_id:
                    raise InvalidEpisode("generation_tokens_or_stop_invalid")
            except InvalidEpisode:
                raise
            except Exception as exc:
                raise InvalidEpisode("generation_failure_" + type(exc).__name__) from None
            self._append(ids, probabilities)
            texts.append(text)
            self.steps.append(record)
            self.recording = None
            return "".join(texts)

    def _append(self, ids, probabilities):
        if self.response_length + len(ids) > self.response_tokens:
            raise EpisodeBudgetExceeded("response_budget_exhausted")
        self.recording.tokens.extend(ids)
        self.recording.loss_mask.extend([1] * len(ids))
        self.recording.rollout_log_probs.extend(probabilities)
        self.recording.response_length += len(ids)
        self.response_length += len(ids)

    async def _compact(self):
        summary_messages = [
            *copy.deepcopy(self.messages),
            {"role": "user", "content": COMPACTION_PROMPT},
        ]
        prompt = self._render(summary_messages, [])
        summary_budget = self.config["rl"]["compaction_summary_tokens"]
        if len(prompt) + summary_budget > self.config["rl"]["context_tokens"]:
            raise EpisodeBudgetExceeded("generation_incomplete_context_full")
        summary = await self._generate_complete(
            prompt, turn_tokens=summary_budget, kind="compaction"
        )
        if not summary.strip():
            raise InvalidEpisode("empty_compaction_summary")
        self.messages[:] = [
            {
                "role": "user",
                "content": CONTINUATION_PROMPT.format(
                    task=self.initial_task["content"], summary=summary
                ),
            }
        ]
        self.compactions += 1

    async def sample(self):
        if self.messages is None or self.recording is not None or self.finalized:
            raise InvalidEpisode("recording_not_active")
        prompt = self._render(self.messages, self.tools)
        if (
            len(prompt) + self.config["rl"]["max_tokens_per_turn"]
            > self.config["rl"]["compaction_trigger_tokens"]
        ):
            await self._compact()
            prompt = self._render(self.messages, self.tools)
        if (
            len(prompt) + self.config["rl"]["max_tokens_per_turn"]
            > self.config["rl"]["context_tokens"]
        ):
            raise EpisodeBudgetExceeded("generation_incomplete_context_full")
        text = await self._generate_complete(
            prompt,
            turn_tokens=self.config["rl"]["max_tokens_per_turn"],
            kind="action",
        )
        self.last_text = text
        return SimpleNamespace(text=text, finish="ok")

    def append_assistant(self, text, call, turn):
        if self.last_text is None or text != self.last_text:
            raise InvalidEpisode("sampled_text_changed")
        self.last_text = None
        return {"role": "assistant", "content": text}

    def append_observation(self, message, images):
        if images or self.messages is None or self.recording is not None or self.finalized:
            raise InvalidEpisode("unsupported_observation")
        return message

    def append_observations(self, messages, images):
        """Encode one ordered parallel-tool result group and one next-turn header."""
        if (
            images
            or self.messages is None
            or self.recording is not None
            or self.finalized
            or not isinstance(messages, list)
            or len(messages) < 2
        ):
            raise InvalidEpisode("unsupported_observation_group")
        return messages

    def finalize(self, reward, metadata, env_time):
        if self.recording is not None or not self.steps or self.finalized:
            raise InvalidEpisode("recording_not_active")
        self.finalized = True
        for index, step in enumerate(self.steps):
            step.reward = reward if index == len(self.steps) - 1 else 0.0
            step.metadata = {
                **metadata,
                "step_kind": step.step_kind,
                "compactions": self.compactions,
                "environment_seconds": env_time,
            }
            step.status = SimpleNamespace(name="COMPLETED")
        validate_samples(self.steps)
        return self.steps


async def offline_long_horizon_probe(model, tokenizer, helper: Path):
    """Exercise chunk continuation, compaction and ordered tools without GPUs.

    This intentionally uses the production recorder and the installed model's
    tokenizer, but a deterministic in-memory engine/session. It proves the old
    ``stop=length`` boundary continues into a valid ordered bash/report turn and
    that a non-append-only compacted prompt stays a separate SkyRL step.
    """
    task = "Synthetic offline continuation and compaction contract probe."
    tools = []
    messages = [{"role": "user", "content": task}]
    initial = list(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
    )
    if not initial or any(type(token) is not int or token < 0 for token in initial):
        raise InvalidEpisode("skyrl_offline_probe_prompt_invalid")
    context = max(8192, len(initial) + 4096)
    limits = {
        "context_tokens": context,
        "max_tokens_per_turn": 512,
        "generation_chunk_tokens": 8,
        "compaction_trigger_tokens": len(initial) + 511,
        "compaction_summary_tokens": 32,
        "max_turns": 4,
        "episode_seconds": 60,
        "tool_seconds": 10,
        "tool_result_chars": 512,
    }
    if (
        limits["compaction_trigger_tokens"]
        + limits["max_tokens_per_turn"]
        + limits["tool_result_chars"]
        + limits["compaction_summary_tokens"]
        >= context
    ):
        raise InvalidEpisode("skyrl_offline_probe_budget_invalid")
    config = {
        "model": {
            "repo": model["repo"],
            "root": model["root"],
            "runtime_chat_template_sha256": fleet.sha256(tokenizer.chat_template.encode()),
        },
        "rl": limits,
        "initial_prompt_tokens_sha256": fleet.sha256(fleet.canonical_json(initial)),
    }
    complete = (
        '<tool_call>{"name":"bash","arguments":{"script":"true"}}</tool_call>'
        '<tool_call>{"name":"submit_report","arguments":'
        '{"flags":[],"explanation":""}}</tool_call>'
    )
    action_ids = list(tokenizer.encode(complete, add_special_tokens=False))
    if (
        not 8 < len(action_ids) < limits["max_tokens_per_turn"]
        or any(type(token) is not int or token < 0 for token in action_ids)
    ):
        raise InvalidEpisode("skyrl_offline_probe_response_invalid")

    class Engine:
        model_name = model["root"]

        def __init__(self):
            self.requests = []
            self.summary = True
            self.remaining = list(action_ids)

        async def generate(self, request):
            self.requests.append(copy.deepcopy(request))
            cap = request["sampling_params"]["max_tokens"]
            if self.summary:
                self.summary = False
                return {
                    "responses": ["synthetic working memory"],
                    "response_ids": [[101, tokenizer.eos_token_id]],
                    "response_logprobs": [[-0.2, -0.1]],
                    "stop_reasons": ["stop"],
                }
            if len(self.remaining) + 1 > cap:
                ids, self.remaining = self.remaining[:cap], self.remaining[cap:]
                return {
                    "responses": [""],
                    "response_ids": [ids],
                    "response_logprobs": [[-0.2] * len(ids)],
                    "stop_reasons": ["length"],
                }
            ids = [*self.remaining, tokenizer.eos_token_id]
            self.remaining = []
            return {
                "responses": [complete],
                "response_ids": [ids],
                "response_logprobs": [[-0.2] * len(ids)],
                "stop_reasons": ["stop"],
            }

    class Session:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append((name, copy.deepcopy(arguments)))
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="synthetic result")],
                is_error=False,
            )

    # Import here to avoid a module cycle at import time.
    from . import rl_episode

    engine, session = Engine(), Session()
    recorder = Recorder(
        config,
        tokenizer,
        engine,
        {"temperature": 0.0, "logprobs": 0},
        1024,
        helper,
    )
    _, reason, _ = await rl_episode._agent(
        recorder, session, messages, tools, limits, parse
    )
    samples = recorder.finalize(1.0, {"verifier_execution_id": "offline-probe"}, 0.0)
    action_requests = engine.requests[1:]
    if (
        reason != "report_submitted"
        or session.calls
        != [("bash", {"script": "true"}), ("submit_report", {"flags": [], "explanation": ""})]
        or len(action_requests) < 2
        or [sample.metadata["step_kind"] for sample in samples]
        != ["compaction", "action"]
        or [sample.reward for sample in samples] != [0.0, 1.0]
        or samples[0].tokens[: -samples[0].response_length]
        == samples[1].tokens[: -samples[1].response_length]
    ):
        raise InvalidEpisode("skyrl_offline_long_horizon_probe_failed")
    # The continuation assertion above needs the exact sampled first chunk,
    # not a synthetic repeated token.
    first_prompt = action_requests[0]["prompt_token_ids"][0]
    second_prompt = action_requests[1]["prompt_token_ids"][0]
    if second_prompt != first_prompt + action_ids[: limits["generation_chunk_tokens"]]:
        raise InvalidEpisode("skyrl_offline_chunk_continuation_failed")
    return {
        "chunk_continuation_checked": True,
        "compaction_checked": True,
        "stepwise_prompt_checked": True,
        "ordered_multi_tool_execution_checked": True,
        "samples": len(samples),
        "generation_requests": len(engine.requests),
    }
