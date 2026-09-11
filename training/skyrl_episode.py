"""SkyRL token recording for the shared, single-attempt Fleet episode lifecycle.

Internal integration, not a qualified RL launcher. SkyRL owns generation and
optimization. Sampled token IDs are never reconstructed from text; observations
are encoded once and masked. No fallback reward, truncation or resampling.
"""

from __future__ import annotations

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
from .rl_episode import InvalidEpisode, validate_samples

CLIENT_MODULE = "skyrl.backends.skyrl_train.inference_servers.remote_inference_client"
CLIENT_SHA256 = "7a798659decf8a49b9ab28c5fc299cdb0f3b1eedaa1db791e7a023f467bda174"


def _module(name, expected):
    module = importlib.import_module(name)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
        raise InvalidEpisode("native_skyrl_source_drift")
    return module


def parse(text):
    """Keep the native parser; this initial adapter permits one call per turn."""
    calls = parse_tool_calls(text)
    if len(calls) > 1:
        raise InvalidEpisode("multiple_calls_not_qualified")
    if not calls:
        return None
    function = calls[0]["function"]
    return {"name": function["name"], "arguments": json.loads(function["arguments"])}


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

    The supplied helper is the pinned SkyRL ``generators/utils.py`` file. Its
    fixed-base encoder handles observations, not sampled assistant messages.
    Fresh data must bind this template and initial token sequence, independently
    from Miles' different native template/recorder contract.
    """

    def __init__(self, config, tokenizer, engine, sampling, response_tokens, helper: Path):
        self.config, self.tokenizer, self.engine = config, tokenizer, engine
        if (
            config["model"]["repo"] != "Qwen/Qwen3.8-27B"
            or config["model"]["root"] != engine.model_name
            or fleet.sha256(tokenizer.chat_template.encode())
            != config["model"]["runtime_chat_template_sha256"]
            or type(response_tokens) is not int
            or not 0 < response_tokens < config["rl"]["context_tokens"]
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
        scope = native_helper(helper).__globals__
        self.encode = scope["encode_messages_subset"]
        self.header = list(scope["get_generation_prompt_ids"](tokenizer))
        self.response_tokens = response_tokens
        self.prompt = self.recording = None
        self.last_text = None
        self.finalized = False

    def begin_segment(self, messages, tools):
        if self.prompt is not None:
            raise InvalidEpisode("recording_already_started")
        tokens = list(
            self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=True, return_dict=False, add_generation_prompt=True
            )
        )
        if (
            not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
            or len(tokens) + self.response_tokens > self.config["rl"]["context_tokens"]
            or fleet.sha256(fleet.canonical_json(tokens))
            != self.config["initial_prompt_tokens_sha256"]
        ):
            raise InvalidEpisode("skyrl_initial_prompt_drift")
        self.prompt = tokens
        self.recording = SimpleNamespace(
            tokens=list(tokens), loss_mask=[], rollout_log_probs=[], response_length=0
        )

    async def sample(self):
        if self.recording is None or self.finalized:
            raise InvalidEpisode("recording_not_active")
        record = self.recording
        remaining = self.response_tokens - record.response_length
        if remaining <= 0:
            raise InvalidEpisode("response_budget_exhausted")
        cap = min(self.config["rl"]["max_tokens_per_turn"], remaining)
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
                stop != "stop"
                or not isinstance(text, str)
                or not ids
                or len(ids) > params["max_tokens"]
                or ids[-1] != self.tokenizer.eos_token_id
                or not isinstance(probabilities, list)
                or len(probabilities) != len(ids)
                or any(type(t) is not int or t < 0 for t in ids)
                or any(
                    type(p) not in (float, int) or not math.isfinite(p) or p > 1e-6
                    for p in probabilities
                )
            ):
                raise InvalidEpisode("generation_tokens_or_stop_invalid")
        except InvalidEpisode:
            raise
        except Exception as exc:
            raise InvalidEpisode("generation_failure_" + type(exc).__name__) from None
        self._append(ids, [1] * len(ids), probabilities)
        self.last_text = text
        return SimpleNamespace(text=text, finish="ok")

    def _append(self, ids, mask, probabilities):
        if self.recording.response_length + len(ids) > self.response_tokens:
            raise InvalidEpisode("response_budget_exhausted")
        self.recording.tokens.extend(ids)
        self.recording.loss_mask.extend(mask)
        self.recording.rollout_log_probs.extend(probabilities)
        self.recording.response_length += len(ids)

    def append_assistant(self, text, call, turn):
        if self.last_text is None or text != self.last_text:
            raise InvalidEpisode("sampled_text_changed")
        self.last_text = None
        return {"role": "assistant", "content": text}

    def append_observation(self, message, images):
        if images or self.recording is None or self.finalized:
            raise InvalidEpisode("unsupported_observation")
        # Sampling ends AT EOS; ChatML's following newline is template-owned.
        # Preserve sampled IDs and encode only the masked newline/tool/header.
        ids = (
            self.tokenizer.encode("\n", add_special_tokens=False)
            + list(self.encode([message], self.tokenizer))
            + self.header
        )
        self._append(ids, [0] * len(ids), [0.0] * len(ids))
        return message

    def finalize(self, reward, metadata, env_time):
        if self.recording is None or self.finalized:
            raise InvalidEpisode("recording_not_active")
        self.finalized = True
        self.recording.reward = reward
        self.recording.metadata = metadata
        self.recording.status = SimpleNamespace(name="COMPLETED")
        validate_samples([self.recording])
        return [self.recording]
