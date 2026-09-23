"""Fast3-only bounded HTTP retry adapter for native SkyRL generation.

The historical single-attempt adapter remains untouched.  This module accepts
one exact, self-sealed policy and retries only a generation POST that receives
429 or 5xx.  Transport errors, redirects, invalid JSON, and whole episodes are
never retried; failed response bodies are never decoded or returned.
"""

from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

from . import rl_episode, skyrl_episode

POLICY_SCHEMA = "cyber_skyrl_generation_http_retry_policy_v1"
FAILURE_SCHEMA = "cyber_skyrl_generation_http_failure_v1"
POLICY_BODY = {
    "schema": POLICY_SCHEMA,
    "max_http_attempts": 3,
    "retryable_http_statuses": {"exact": [429], "inclusive_ranges": [[500, 599]]},
    "backoff_seconds": [1, 2],
    "transport_retry": False,
    "follow_redirects": False,
    "whole_episode_retry": False,
    "failed_response_body_admitted": False,
    "failed_response_tokens_admitted": False,
    "admitted_response": "first_2xx_json_only",
}


def expected_policy() -> dict:
    """Return the only retry policy this source admits."""
    body = copy.deepcopy(POLICY_BODY)
    return {**body, "sha256": "sha256:" + digest(body)}


def _same_json_types(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _same_json_types(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _same_json_types(item, wanted) for item, wanted in zip(value, expected, strict=True)
        )
    return True


def validate_policy(value: object) -> dict:
    expected = expected_policy()
    if not _same_json_types(value, expected) or fleet.canonical_json(value) != fleet.canonical_json(
        expected
    ):
        raise rl_episode.InvalidEpisode("generation_retry_policy_invalid")
    return expected


class GenerationHTTPFailure(fleet.FleetRequestError, rl_episode.InvalidEpisode):
    """Payload-free terminal generation status with bounded attempt evidence."""

    def __init__(self, status: int, attempts: int) -> None:
        if type(status) is not int or not 300 <= status <= 599:
            raise ValueError("generation HTTP status is invalid")
        if type(attempts) is not int or not 1 <= attempts <= POLICY_BODY["max_http_attempts"]:
            raise ValueError("generation HTTP attempt count is invalid")
        fleet.FleetRequestError.__init__(
            self,
            "POST",
            "/inference/v1/generate",
            status,
            diagnostic={"attempts": attempts},
        )


def failure_receipt(error: GenerationHTTPFailure, policy: object) -> dict:
    """Project only numeric status/attempt evidence from a failed response."""
    checked = validate_policy(policy)
    if not isinstance(error, GenerationHTTPFailure):
        raise ValueError("generation failure evidence is not typed")
    body = {
        "schema": FAILURE_SCHEMA,
        "http_status": error.status_code,
        "attempts": error.diagnostic["attempts"],
        "policy_sha256": checked["sha256"],
    }
    return {**body, "sha256": "sha256:" + digest(body)}


@asynccontextmanager
async def bounded_retry_engine(engine, tokenizer, timeout, *, policy):
    """Clone SkyRL's data-plane client with the exact Fast3 retry policy."""
    checked = validate_policy(policy)
    native = skyrl_episode._module(
        skyrl_episode.CLIENT_MODULE, skyrl_episode.CLIENT_SHA256
    ).RemoteInferenceClient
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
        raise rl_episode.InvalidEpisode("unsupported_skyrl_engine")
    endpoint = engine.proxy_url.rstrip("/")
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        transport=httpx.AsyncHTTPTransport(retries=0),
    ) as client:

        class Fast3RetryClient(native):
            async def _post(self, url, json, headers=None):
                if url != endpoint + "/inference/v1/generate" or set(headers or {}) - {
                    "Content-Type"
                }:
                    raise rl_episode.InvalidEpisode("unexpected_skyrl_data_request")
                for attempt in range(1, checked["max_http_attempts"] + 1):
                    try:
                        response = await client.post(url, json=json, headers=headers)
                    except httpx.HTTPError:
                        raise rl_episode.InvalidEpisode("generation_transport_failure") from None
                    status = response.status_code
                    if 200 <= status < 300:
                        try:
                            return response.json()
                        except ValueError:
                            raise rl_episode.InvalidEpisode("generation_invalid_json") from None
                    retryable = status == 429 or 500 <= status <= 599
                    if not 300 <= status <= 599:
                        raise rl_episode.InvalidEpisode("generation_http_status_invalid")
                    if not retryable or attempt == checked["max_http_attempts"]:
                        raise GenerationHTTPFailure(status, attempt)
                    await asyncio.sleep(checked["backoff_seconds"][attempt - 1])
                raise AssertionError("unreachable generation retry state")

        yield Fast3RetryClient(
            proxy_url=endpoint,
            server_urls=list(engine.server_urls),
            data_parallel_size=engine.data_parallel_size,
            model_name=engine.model_name,
            tokenizer=tokenizer,
        )
