"""Strict batches for native Miles: sample once, clean up all siblings, or stop.

Miles still owns the dataset cursor, token recorder, weight updates and optimizer.
Its default collector replaces groups after exceptions and omits eval attempt IDs;
that policy is unsuitable for exact Fleet episodes. No retries or score filtering
belong here. Private episode artifacts remain separate from scalar telemetry.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
from pathlib import Path

from evals.fleet import opencode_self_hosted as fleet

from .rl_episode import InvalidEpisode, validate_samples


class Rollout:
    """Class-based synchronous Miles rollout function, shared by train and eval."""

    def __init__(self, input):
        from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
        from miles.utils.data import Dataset

        self.args, self.source = input.args, input.data_source
        args = self.args
        forbidden = (
            "partial_rollout",
            "fully_async",
            "group_rm",
            "apply_chat_template",
            "use_fault_tolerance",
            "eval_uses_snapshots",
            "dynamic_sampling_filter_path",
            "rollout_sample_filter_path",
            "rollout_all_samples_process_path",
            "custom_reward_post_process_path",
            "custom_convert_samples_to_train_data_path",
            "load_debug_rollout_data",
            "use_dynamic_global_batch_size",
        )
        if any(getattr(args, key, False) for key in forbidden) or (
            args.custom_generate_function_path != "training.rl_episode.generate"
            or args.rollout_function_path != args.eval_function_path
            or not args.rollout_global_dataset
        ):
            raise InvalidEpisode("unsupported_native_batch_policy")
        self.state = GenerateState(args)
        # Prompts already contain the frozen text template; no image processor
        # may reinterpret them, including the independently loaded dev dataset.
        self.state.processor = None
        manifest_path = Path(args.cyber_data_manifest)
        manifest = json.loads(manifest_path.read_bytes())
        if (
            manifest.get("schema") != "cyber_miles_data_v1"
            or manifest.get("sha256") != fleet.digest_without(manifest, "sha256")
            or manifest.get("name") != args.cyber_run_id
            or manifest.get("template_sha256")
            != fleet.sha256(self.state.tokenizer.chat_template.encode())
        ):
            raise InvalidEpisode("batch_manifest_mismatch")
        if len(args.eval_datasets) != 1:
            raise InvalidEpisode("one_frozen_dev_set_required")
        self.dev_config = cfg = args.eval_datasets[0]
        if cfg.metadata_overrides or cfg.custom_generate_function_path or cfg.rm_type:
            raise InvalidEpisode("dev_contract_override")
        self.dev = Dataset(
            cfg.path,
            self.state.tokenizer,
            self.state.processor,
            args.eval_max_prompt_len,
            prompt_key=cfg.input_key,
            metadata_key=cfg.metadata_key,
            apply_chat_template=False,
        )
        for split, path, dataset in (
            ("train", args.prompt_data, self.source.dataset),
            ("dev", cfg.path, self.dev),
        ):
            item = manifest["files"][split]
            if Path(path) != manifest_path.parent / item["path"]:
                raise InvalidEpisode("batch_data_path_mismatch")
            raw = Path(path).read_bytes()
            rows = [json.loads(line) for line in raw.splitlines()]
            if (
                fleet.sha256(raw) != item["sha256"]
                or len(rows) != item["rows"]
                or len(dataset) != len(rows)
                or not rows
                or any(
                    s.prompt != row["input"]
                    or s.metadata != row["metadata"]
                    or s.metadata.get("split") != split
                    or getattr(s, "multimodal_inputs", None) is not None
                    for s, row in zip(dataset.origin_samples, rows, strict=True)
                )
            ):
                raise InvalidEpisode("native_dataset_changed_or_filtered")
        if not 0 < args.rollout_batch_size <= len(self.source.dataset) or (
            args.over_sampling_batch_size != args.rollout_batch_size
            or args.n_samples_per_prompt < 2
            or cfg.n_samples_per_eval_prompt < 1
            or args.global_batch_size != args.rollout_batch_size * args.n_samples_per_prompt
        ):
            raise InvalidEpisode("batch_size_mismatch")
        self.manifest_sha256 = manifest["sha256"]
        self.root = Path(args.cyber_output_root) / "batches"
        if not self.root.is_absolute():
            raise InvalidEpisode("output_root_must_be_absolute")
        self.last_train = self.baseline = None
        self.busy = self.failed = False

    async def __call__(self, input):
        from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
        from miles.rollout.inference_rollout.inference_rollout_common import (
            compute_sampling_params,
            generate_and_rm_group,
        )

        rid, evaluation = input.rollout_id, input.evaluation
        if self.busy or self.failed or type(rid) is not int or not 0 <= rid < self.args.num_rollout:
            raise InvalidEpisode("batch_reentry_or_invalid_sequence")
        if evaluation:
            if input.generate_state is not None or input.hf_dir is not None:
                raise InvalidEpisode("snapshot_eval_not_qualified")
            if self.last_train is None:
                if self.baseline is not None:
                    raise InvalidEpisode("baseline_already_collected")
                kind = "dev-baseline"
            elif self.last_train == rid:
                kind = "dev-after"
            else:
                raise InvalidEpisode("eval_not_bound_to_latest_batch")
        else:
            if self.baseline is None or rid != (
                self.baseline if self.last_train is None else self.last_train + 1
            ):
                raise InvalidEpisode("training_requires_baseline_and_ordered_batches")
            kind = "train"
        batch_id = f"{kind}-r{rid}"
        directory = self.root / batch_id
        self.busy = True
        owned = False
        try:
            directory.mkdir(parents=True, mode=0o700, exist_ok=False)
            owned = True
            if evaluation:
                cfg = self.dev_config
                groups = [
                    [copy.deepcopy(sample) for _ in range(cfg.n_samples_per_eval_prompt)]
                    for sample in self.dev.samples
                ]
                for group_index, group in enumerate(groups):
                    for offset, sample in enumerate(group):
                        sample.index = group_index * cfg.n_samples_per_eval_prompt + offset
                        sample.group_index = group_index
                sampling = compute_sampling_params(
                    self.args,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    max_new_tokens=cfg.max_response_len,
                )
            else:
                # Exactly one cursor advance; never refill a failed batch.
                groups = self.source.get_samples(self.args.rollout_batch_size)
                if len(groups) != self.args.rollout_batch_size or any(
                    len(group) != self.args.n_samples_per_prompt for group in groups
                ):
                    raise InvalidEpisode("native_group_count_mismatch")
                sampling = self.state.sampling_params.copy()
            samples = [s for group in groups for s in group]
            indices = [s.index for s in samples]
            if any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(
                indices
            ):
                raise InvalidEpisode("native_sample_identity_mismatch")
            for sample in samples:
                if (
                    sample.response
                    or sample.reward is not None
                    or sample.generate_function_path
                    or sample.status.name != "PENDING"
                ):
                    raise InvalidEpisode("sample_is_not_unstarted")
                # Native rollout_id groups segments of ONE EPISODE for GRPO.
                # It must NOT be the batch counter shared by competing samples.
                sample.rollout_id = sample.index
                sample.metadata["cyber_batch"] = {"kind": kind, "rollout_id": rid}
            intent = {
                "schema": "cyber_miles_batch_v1",
                "batch_id": batch_id,
                "data_sha256": self.manifest_sha256,
                "episode_indices": indices,
                "native_rollout_id": rid,
                "evaluation": evaluation,
                "optimizer_step_verified": False,
            }
            fleet.write_json_once(directory / "STARTED.json", intent)
            async with asyncio.TaskGroup() as task_group:
                tasks = [
                    task_group.create_task(
                        generate_and_rm_group(
                            self.state,
                            group,
                            sampling.copy(),
                            evaluation=evaluation,
                        )
                    )
                    for group in groups
                ]
            data = [task.result() for task in tasks]
            for source_group, group in zip(groups, data, strict=True):
                if len(group) != len(source_group):
                    raise InvalidEpisode("generated_group_count_mismatch")
                for original, result in zip(source_group, group, strict=True):
                    segments = result if isinstance(result, list) else [result]
                    validate_samples(segments)
                    if any(
                        s.index != original.index
                        or s.rollout_id != original.index
                        or s.group_index != original.group_index
                        or type(s.reward) not in (float, int)
                        or not math.isfinite(s.reward)
                        or not 0 <= s.reward <= 1
                        or s.reward != segments[0].reward
                        for s in segments
                    ):
                        raise InvalidEpisode("generated_episode_identity_or_reward_mismatch")
            fleet.write_json_once(
                directory / "COLLECTED.json",
                {
                    **intent,
                    "sha256": fleet.sha256(fleet.canonical_json(intent)),
                },
            )
            if evaluation:
                if kind == "dev-baseline":
                    self.baseline = rid
                episodes = [item for group in data for item in group]
                # One terminal reward per episode, never one vote per segment.
                # Native sample diagnostics see the first segment; complete
                # recordings remain in the private per-episode artifacts.
                flat = [item[0] if isinstance(item, list) else item for item in episodes]
                return RolloutFnEvalOutput(
                    data={
                        self.dev_config.name: {
                            "rewards": [s.reward for s in flat],
                            "truncated": [False] * len(flat),
                            "samples": flat,
                        }
                    },
                    metrics={
                        "cyber/dev_is_baseline": int(kind == "dev-baseline"),
                        "cyber/dev_segments": sum(
                            len(item) if isinstance(item, list) else 1 for item in episodes
                        ),
                    },
                )
            self.last_train = rid
            return RolloutFnTrainOutput(samples=data, metrics={"cyber/episodes": len(samples)})
        except asyncio.CancelledError:
            self.failed = True
            # Cancellation is delivered at our await, after the claim is owned.
            fleet.write_json_once(directory / "FAILED.json", {"reason": "cancelled"})
            raise
        except Exception as exc:
            self.failed = True
            if owned:
                fleet.write_json_once(directory / "FAILED.json", {"error_type": type(exc).__name__})
            # TaskGroup has awaited every cancelled sibling's environment cleanup.
            # Never propagate its private ExceptionGroup into native/Ray logs.
            raise InvalidEpisode("batch_failed_no_replacement") from None
        finally:
            self.busy = False
