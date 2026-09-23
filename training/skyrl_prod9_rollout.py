"""Fresh prod9 rollout binding for the long-horizon SkyRL canary.

The historical SkyRL rollout module is part of prior sealed source closures.
This module deliberately owns only the one changed dependency: every prod9
episode constructs :class:`skyrl_prod9_hardening.Recorder` directly.  It does
not monkeypatch, shadow, or otherwise alter the historical implementation.
"""

from __future__ import annotations

import asyncio
import copy
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from evals.fleet import opencode_self_hosted as fleet

from . import rl_episode, skyrl_episode
from . import skyrl_prod9_hardening as hardening
from .skyrl_rollout import Generator as HistoricalGenerator

RECORDER_IMPLEMENTATION = "training.skyrl_prod9_hardening.Recorder"


def runtime_binding() -> dict[str, str]:
    """Return the exact fresh classes that a prod9 bundle executes."""
    return {
        "generator": "training.skyrl_prod9_rollout.Generator",
        "recorder": RECORDER_IMPLEMENTATION,
    }


class Generator(HistoricalGenerator):
    """Historical input validation with an explicitly fresh recorder.

    ``HistoricalGenerator`` keeps the reviewed immutable-data parsing and
    batch identity checks.  Its ``generate`` method cannot be inherited: it
    constructs the historical recorder lexically.  This small owned copy
    changes that one construction point and stamps the collected receipt with
    the implementation identity, so later acceptance can prove what ran.
    """

    def __init__(self, *args, generation_retry_policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_retry_policy = generation_retry_policy

    async def generate(self, input_batch):
        if self.busy or self.failed:
            raise rl_episode.InvalidEpisode("skyrl_batch_reentry_or_previous_failure")
        self.busy = True
        owned, directory = False, None
        try:
            binding, identifier, configs = self._inputs(input_batch)
            directory = self.root / "batches" / identifier
            directory.mkdir(parents=True, mode=0o700, exist_ok=False)
            owned = True
            intent = {
                "schema": "cyber_skyrl_batch_v1",
                **binding,
                "data_sha256": self.manifest["sha256"],
                "optimizer_step_verified": False,
                "recorder_implementation": RECORDER_IMPLEMENTATION,
            }
            fleet.write_json_once(directory / "STARTED.json", intent)
            key = os.environ.get("FLEET_API_KEY")
            if not key:
                raise rl_episode.InvalidEpisode("missing_fleet_auth")
            timeout = max(config["rl"]["episode_seconds"] for config in configs)
            semaphore = asyncio.Semaphore(self.concurrency)
            retry_policy = getattr(self, "generation_retry_policy", None)
            engine_context = (
                skyrl_episode.single_attempt_engine(
                    self.engine,
                    self.tokenizer,
                    timeout,
                    generation_retry_policy=retry_policy,
                )
                if retry_policy is not None
                else skyrl_episode.single_attempt_engine(self.engine, self.tokenizer, timeout)
            )
            async with (
                httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=120,
                    transport=httpx.AsyncHTTPTransport(retries=0),
                    follow_redirects=False,
                ) as client,
                engine_context as engine,
            ):

                async def episode(index, source):
                    async with semaphore:
                        started = time.monotonic()
                        config = copy.deepcopy(source)
                        config["run_id"] += f"-{identifier}-{index}"
                        config["native_batch"] = binding
                        config["sampling"] = copy.deepcopy(input_batch["sampling_params"] or {})
                        config["config_sha256"] = fleet.digest_without(config, "config_sha256")
                        helper = skyrl_episode._module(
                            "skyrl.train.generators.utils",
                            "55c15b660067749febda00d4fb1c2110ff436717bbd4b73bf66055a73d0b87d5",
                        )
                        recorder = hardening.Recorder(
                            config,
                            self.tokenizer,
                            engine,
                            config["sampling"],
                            self.response_tokens,
                            Path(helper.__file__),
                        )
                        samples = await rl_episode.collect(
                            config,
                            directory / f"episode-{index}",
                            recorder,
                            skyrl_episode.parse,
                            client=client,
                        )
                        rl_episode.validate_samples(samples)
                        if (
                            not samples
                            or any(
                                type(sample.reward) not in (float, int)
                                or not math.isfinite(sample.reward)
                                or not 0 <= sample.reward <= 1
                                for sample in samples
                            )
                            or any(sample.reward != 0 for sample in samples[:-1])
                        ):
                            raise rl_episode.InvalidEpisode("skyrl_episode_result_invalid")
                        return samples, time.monotonic() - started

                async with asyncio.TaskGroup() as group:
                    tasks = [
                        group.create_task(episode(i, config)) for i, config in enumerate(configs)
                    ]
            trajectories, durations = zip(*(task.result() for task in tasks), strict=True)
            fleet.write_json_once(
                directory / "COLLECTED.json",
                {
                    **intent,
                    "sha256": fleet.sha256(fleet.canonical_json(intent)),
                },
            )
            steps = [sample for samples in trajectories for sample in samples]
            trajectory_ids = [
                identity
                for identity, samples in zip(
                    input_batch["trajectory_ids"], trajectories, strict=True
                )
                for _ in samples
            ]
            is_last_step = [
                index == len(samples) - 1
                for samples in trajectories
                for index in range(len(samples))
            ]
            stop_reasons = [
                (
                    "length"
                    if last
                    and sample.metadata.get("done_reason") == "turn_response_budget_exhausted"
                    else "stop"
                )
                for samples in trajectories
                for sample, last in zip(
                    samples,
                    [index == len(samples) - 1 for index in range(len(samples))],
                    strict=True,
                )
            ]
            generation_times = [
                duration
                for duration, samples in zip(durations, trajectories, strict=True)
                for _ in samples
            ]
            return {
                "prompt_token_ids": [step.tokens[: -step.response_length] for step in steps],
                "response_ids": [step.tokens[-step.response_length :] for step in steps],
                "rewards": [step.reward for step in steps],
                "loss_masks": [step.loss_mask for step in steps],
                "rollout_logprobs": [step.rollout_log_probs for step in steps],
                "stop_reasons": stop_reasons,
                "trajectory_ids": trajectory_ids,
                "trajectory_generation_times": generation_times,
                "is_last_step": is_last_step,
                "rollout_metrics": {
                    "cyber/episodes": len(trajectories),
                    "cyber/steps": len(steps),
                    "cyber/compactions": sum(
                        int(sample.metadata.get("step_kind") == "compaction")
                        for samples in trajectories
                        for sample in samples
                    ),
                },
                "rollout_expert_indices": None,
                "env_metrics": None,
                "pixel_values": None,
                "image_grid_thw": None,
            }
        except asyncio.CancelledError:
            self.failed = True
            if owned:
                fleet.write_json_once(directory / "FAILED.json", {"reason": "cancelled"})
            raise
        except Exception as exc:
            self.failed = True
            if owned and (reason := rl_episode.budget_stop(exc)):
                fleet.write_json_once(directory / "REJECTED.json", {"reason": reason})
                raise rl_episode.EpisodeBudgetExceeded(reason) from None
            if owned:
                fleet.write_json_once(directory / "FAILED.json", {"error_type": type(exc).__name__})
            raise rl_episode.InvalidEpisode("skyrl_batch_failed_no_replacement") from None
        finally:
            self.busy = False


async def offline_token_safe_tool_probe(model, tokenizer, helper: Path) -> dict[str, object]:
    """Exercise the fresh recorder with the exact tokenizer and no GPU/Fleet.

    This proves the bundled class can render a real tool result through the
    Qwen template.  The adversarial expansion case belongs in a regression
    test; a CPU preflight must never manufacture private task data to test it.
    """
    task = "Synthetic token-safe prod9 tool-result probe."
    messages, tools = [{"role": "user", "content": task}], []
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
        raise rl_episode.InvalidEpisode("prod9_offline_probe_prompt_invalid")
    context = max(8192, len(initial) + 4096)
    limits = {
        "context_tokens": context,
        "max_tokens_per_turn": 512,
        "generation_chunk_tokens": 512,
        "compaction_trigger_tokens": context - 1024,
        "compaction_summary_tokens": 32,
        "max_turns": 4,
        "episode_seconds": 60,
        "tool_seconds": 10,
        "tool_result_chars": 512,
    }
    hardening.validate_episode_limits(limits)
    config = {
        "model": {
            "repo": model["repo"],
            "root": model["root"],
            "runtime_chat_template_sha256": fleet.sha256(tokenizer.chat_template.encode()),
        },
        "rl": limits,
        "initial_prompt_tokens_sha256": fleet.sha256(fleet.canonical_json(initial)),
    }

    class Engine:
        model_name = model["root"]

        def __init__(self):
            self.requests = []
            self.replies = [
                '<tool_call>{"name":"bash","arguments":{}}</tool_call>',
                '<tool_call>{"name":"submit_report","arguments":{}}</tool_call>',
            ]

        async def generate(self, request):
            self.requests.append(copy.deepcopy(request))
            text = self.replies.pop(0)
            return {
                "responses": [text],
                "response_ids": [[101, tokenizer.eos_token_id]],
                "response_logprobs": [[-0.2, -0.1]],
                "stop_reasons": ["stop"],
            }

    class Session:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append((name, copy.deepcopy(arguments)))
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="synthetic tool result")],
                is_error=False,
            )

    engine, session = Engine(), Session()
    recorder = hardening.Recorder(
        config,
        tokenizer,
        engine,
        {"temperature": 0.0, "logprobs": 0},
        1024,
        helper,
    )
    _, reason, _ = await rl_episode._agent(
        recorder, session, messages, tools, limits, skyrl_episode.parse
    )
    if (
        reason != "report_submitted"
        or session.calls != [("bash", {}), ("submit_report", {})]
        or len(engine.requests) != 2
        or not all(request["prompt_token_ids"][0] for request in engine.requests)
    ):
        raise rl_episode.InvalidEpisode("prod9_offline_token_safe_probe_failed")
    return {
        "fresh_recorder_checked": True,
        "recorder_implementation": RECORDER_IMPLEMENTATION,
        "tool_result_token_safe": True,
    }
