"""Strict native SkyRL batches over the shared Fleet episode lifecycle.

No optimizer, dataset cursor, retry, replacement sample or reward filtering.
The native trainer supplies immutable row/repetition IDs; one invalid episode
stops the batch after every sibling has completed its environment cleanup.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import time
from pathlib import Path

import httpx

from evals.fleet import opencode_self_hosted as fleet

from . import rl_episode, skyrl_episode


class Generator:
    """Implements native ``GeneratorInterface.generate`` without changing SkyRL."""

    def __init__(
        self,
        manifest_path,
        manifest_sha256,
        tokenizer,
        engine,
        output_root,
        *,
        response_tokens,
        repetitions,
        concurrency,
    ):
        path = Path(manifest_path)
        manifest = json.loads(path.read_bytes())
        if (
            manifest.get("schema") != "cyber_skyrl_data_v1"
            or manifest.get("sha256") != manifest_sha256
            or manifest_sha256 != fleet.digest_without(manifest, "sha256")
            or manifest["template_sha256"] != fleet.sha256(tokenizer.chat_template.encode())
            or set(repetitions) != {"train", "eval"}
            or any(type(n) is not int or n < 1 for n in repetitions.values())
            or repetitions["train"] < 2
            or type(concurrency) is not int
            or concurrency < 1
            or type(response_tokens) is not int
            or response_tokens < 1
        ):
            raise rl_episode.InvalidEpisode("skyrl_batch_manifest_or_policy_mismatch")
        self.rows = {}
        for phase, split in (("train", "train"), ("eval", "dev")):
            item = manifest["files"][split]
            if item["path"] != f"{split}.jsonl":
                raise rl_episode.InvalidEpisode("skyrl_batch_data_path_mismatch")
            raw = (path.parent / item["path"]).read_bytes()
            if fleet.sha256(raw) != item["sha256"]:
                raise rl_episode.InvalidEpisode("skyrl_batch_data_changed")
            rows = [json.loads(line) for line in raw.splitlines()]
            if len(rows) != item["rows"] or not rows:
                raise rl_episode.InvalidEpisode("skyrl_batch_data_changed")
            for row in rows:
                config = json.loads(row["cyber_config_json"])
                rl_episode._validate(config)
                if row["split"] != split or config["run_id"] != manifest["name"]:
                    raise rl_episode.InvalidEpisode("skyrl_batch_split_or_run_mismatch")
            self.rows[phase] = rows
        self.root = Path(output_root)
        if not self.root.is_absolute():
            raise rl_episode.InvalidEpisode("output_root_must_be_absolute")
        self.manifest = manifest
        self.tokenizer, self.engine = tokenizer, engine
        self.response_tokens, self.repetitions = response_tokens, dict(repetitions)
        self.concurrency, self.busy, self.failed = concurrency, False, False

    def _inputs(self, batch):
        metadata = batch["batch_metadata"]
        phase, step = metadata.training_phase, metadata.global_step
        if phase not in self.rows or type(step) is not int or step < 0:
            raise rl_episode.InvalidEpisode("skyrl_batch_metadata_invalid")
        ids = batch["trajectory_ids"]
        columns = [batch[k] for k in ("prompts", "env_classes", "env_extras")]
        if not ids or any(len(values) != len(ids) for values in columns):
            raise rl_episode.InvalidEpisode("skyrl_batch_column_mismatch")
        pairs, groups, configs = [], {}, []
        for index, identity in enumerate(ids):
            uid, repetition = identity.instance_id, identity.repetition_id
            if (
                type(uid) is not str
                or not uid.isdecimal()
                or str(int(uid)) != uid
                or int(uid) >= len(self.rows[phase])
                or type(repetition) is not int
                or not 0 <= repetition < self.repetitions[phase]
            ):
                raise rl_episode.InvalidEpisode("skyrl_trajectory_identity_invalid")
            pairs.append((uid, repetition))
            groups.setdefault(uid, []).append(repetition)
            row = self.rows[phase][int(uid)]
            extras = {k: v for k, v in row.items() if k not in {"prompt", "env_class"}}
            if (columns[0][index], columns[1][index], columns[2][index]) != (
                row["prompt"],
                row["env_class"],
                extras,
            ):
                raise rl_episode.InvalidEpisode("skyrl_native_input_drift")
            configs.append(json.loads(row["cyber_config_json"]))
        if len(set(pairs)) != len(pairs) or any(
            sorted(values) != list(range(self.repetitions[phase])) for values in groups.values()
        ):
            raise rl_episode.InvalidEpisode("skyrl_incomplete_or_duplicate_groups")
        binding = {"phase": phase, "global_step": step, "trajectory_ids": sorted(pairs)}
        identifier = fleet.sha256(fleet.canonical_json(binding)).removeprefix("sha256:")[:24]
        return binding, identifier, configs

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
            }
            fleet.write_json_once(directory / "STARTED.json", intent)
            key = os.environ.get("FLEET_API_KEY")
            if not key:
                raise rl_episode.InvalidEpisode("missing_fleet_auth")
            timeout = max(config["rl"]["episode_seconds"] for config in configs)
            semaphore = asyncio.Semaphore(self.concurrency)
            async with (
                httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=120,
                    transport=httpx.AsyncHTTPTransport(retries=0),
                    follow_redirects=False,
                ) as client,
                skyrl_episode.single_attempt_engine(self.engine, self.tokenizer, timeout) as engine,
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
                        recorder = skyrl_episode.Recorder(
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
                        if len(samples) != 1 or (
                            type(samples[0].reward) not in (float, int)
                            or not math.isfinite(samples[0].reward)
                            or not 0 <= samples[0].reward <= 1
                        ):
                            raise rl_episode.InvalidEpisode("skyrl_episode_result_invalid")
                        return samples[0], time.monotonic() - started

                async with asyncio.TaskGroup() as group:
                    tasks = [group.create_task(episode(i, c)) for i, c in enumerate(configs)]
            values, durations = zip(*(task.result() for task in tasks), strict=True)
            fleet.write_json_once(
                directory / "COLLECTED.json",
                {
                    **intent,
                    "sha256": fleet.sha256(fleet.canonical_json(intent)),
                },
            )
            return {
                "prompt_token_ids": [s.tokens[: -s.response_length] for s in values],
                "response_ids": [s.tokens[-s.response_length :] for s in values],
                "rewards": [s.reward for s in values],
                "loss_masks": [s.loss_mask for s in values],
                "rollout_logprobs": [s.rollout_log_probs for s in values],
                "stop_reasons": ["stop"] * len(values),
                "trajectory_ids": input_batch["trajectory_ids"],
                "trajectory_generation_times": list(durations),
                "is_last_step": [True] * len(values),
                "rollout_metrics": {"cyber/episodes": len(values)},
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
            if owned:
                fleet.write_json_once(directory / "FAILED.json", {"error_type": type(exc).__name__})
            raise rl_episode.InvalidEpisode("skyrl_batch_failed_no_replacement") from None
        finally:
            self.busy = False
