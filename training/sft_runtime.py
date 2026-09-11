"""Pinned SkyRL SFT with task-held-out loss, W&B, and recoverable saves.

The trainer/worker methods used here were inspected at SkyRL f5bc3b78. This
module deliberately keeps the native training loop and optimizer. It adds
metadata-preserving tokenization, one-pass task-macro validation, scalar
telemetry, and checkpoint retention. HF export is a separate zero-step job:
the old step-318 inline export could time out before the final checkpoint.

No task text, token array, private traceback, or sample table is sent to W&B.
Run as a staged module with --plan and its externally bound --plan-sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path

SKYRL_REVISION = "f5bc3b78dfddfb352870d5d7430cd226e5785838"
DENSE_SCHEMA = "cyber_sft_runtime_dense_v1"
DENSE_FORMAT = "pretokenized_assistant_segments_v1"
DENSE_EXCLUSION_REASONS = {"overlength_assistant_target", "overlength_required_previous_round"}
WATCHDOG_POLL_SECONDS = 60
WATCHDOG_STARTUP_SECONDS = 30 * 60
WATCHDOG_IDLE_SECONDS = 20 * 60
WATCHDOG_HARD_SECONDS = 8 * 60 * 60
WATCHDOG_DRAIN_SECONDS = 5 * 60
SOURCE_SHA256 = {
    "skyrl/train/sft_trainer.py": (
        "a5ef8a2e22de785b6760abffdd9353f1246a5898983b4d27b4aadd8089a3579a"
    ),
    "skyrl/train/config/sft_config.py": (
        "930812a153f8575b989d1492679f476d7a6495832bc08e623372d76efc407cd0"
    ),
    "skyrl/train/config/config.py": (
        "a3b36099c5308fc9bc658394f10cfff0da3f0cadcc067aa2b44e95870a609c90"
    ),
    "skyrl/train/dataset/collators.py": (
        "891fe6fdbeef5ede64514a265e8b573f9ca803115f64d705302ed235f4359e32"
    ),
    "skyrl/train/utils/callbacks.py": (
        "0b86c51d71db4a21f3fbd58b81e28a42ed00e1fe0aec5890a9f50a034c1be903"
    ),
    "skyrl/train/utils/tracking.py": (
        "51f6515201f654fd1c6a182bd62fa9840ee8c5f537ff3ae74e485b852faf7524"
    ),
    "skyrl/backends/skyrl_train/workers/worker.py": (
        "a3db942bcfb8f4a2f9cb6eae5717a2adde6699592ed469e164d60173e2bb2eab"
    ),
    # The ba288751 image adds the already-approved opt-in Torch GDN constructor
    # patch to f5bc3b78. Its forward/loss implementation remains unchanged.
    "skyrl/backends/skyrl_train/workers/model_wrapper.py": (
        "015b532717df5a3492ebf42fa03982bdeaa820c658c9207689bc64f26c07f8fa"
    ),
    "skyrl/backends/skyrl_train/patches/fleet_qwen_torch_gdn.py": (
        "36414572e7dd70f2cc098f062e4f5c0544f72b58c3c3c71927ea67edb9d5f07e"
    ),
    "skyrl/backends/skyrl_train/utils/torch_utils.py": (
        "c145636be365b9513197cd0947f24ef24f0c007919fc04a512e14f6b35d2763e"
    ),
    "skyrl/backends/skyrl_train/workers/worker_dispatch.py": (
        "e2da7e5d351699b2e38f2d326b905af2b7bedf0c04e6799c3eed260db6ca45aa"
    ),
    "skyrl/backends/skyrl_train/distributed/dispatch.py": (
        "165db3891b96b3fb92a98727f9290989f9f67b0a0dc8398a622b249762c9fe2e"
    ),
    "skyrl/backends/skyrl_train/distributed/fsdp_strategy.py": (
        "a987a4bd509a43af1d27a92b1a933455f1f812e0da8a5ad733a68bcf2d67cd63"
    ),
    "skyrl/backends/skyrl_train/distributed/fsdp_utils.py": (
        "68fa165b08a6341430468ffa7961bab76f2c87cbb250df800a68a5e1505cb726"
    ),
    "skyrl/backends/skyrl_train/training_batch.py": (
        "7668f7007dc983549a661a166c0948d12c8db74fbd7f8f4d37889f8cbf6fa00f"
    ),
    "skyrl/backends/skyrl_train/workers/worker_utils.py": (
        "effc71cb05327d74bf9163ca3de14370c0bf8fa51302bfd1ff4331a8bdd94fd0"
    ),
    "skyrl/backends/skyrl_train/utils/ppo_utils.py": (
        "0d7ae59af59b16cc3d0b6593251bd3ab5dbfd421d9e3ae1bfde00783ba6323d7"
    ),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unsigned_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_receipt(path: Path, value: dict, *, replace: bool = False) -> None:
    payload = {**value, "receipt_sha256": _unsigned_digest(value)}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp") if replace else path
    with temporary.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        temporary.replace(path)


def _checked_file(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or digest(path) != expected.removeprefix("sha256:"):
        raise ValueError("immutable file missing or digest mismatch")


def validate_plan(plan: dict, *, check_files: bool = True) -> None:
    if plan.get("schema") not in ("cyber_sft_runtime_v2", DENSE_SCHEMA):
        raise ValueError("unsupported SFT runtime plan")
    recipe = plan["recipe"]
    for key in (
        "epochs",
        "batch_size",
        "microbatch_per_gpu",
        "nodes",
        "gpus_per_node",
        "max_length",
        "eval_interval",
        "checkpoint_interval",
        "keep_checkpoints",
        "max_steps",
    ):
        if type(recipe[key]) is not int or recipe[key] <= 0:
            raise ValueError("recipe counts must be positive integers")
    if type(recipe["seed"]) is not int or not 0 <= recipe["seed"] < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if (
        type(recipe["lr"]) not in {int, float}
        or not math.isfinite(recipe["lr"])
        or not 0 < recipe["lr"] <= 1e-4
    ):
        raise ValueError("learning rate outside reviewed SFT range")
    if recipe["batch_size"] % (
        recipe["nodes"] * recipe["gpus_per_node"] * recipe["microbatch_per_gpu"]
    ):
        raise ValueError("global batch must divide evenly across GPU microbatches")
    if recipe["checkpoint_interval"] != recipe["eval_interval"]:
        raise ValueError("every periodic validation needs a corresponding saved checkpoint")
    if plan.get("resume_from"):
        raise ValueError(
            "v2 arms start from their exact base; resume needs a separately bound plan"
        )
    train, dev = (plan["datasets"][key] for key in ("train", "dev"))
    for dataset in (train, dev):
        if not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", dataset["sha256"]):
            raise ValueError("dataset digest missing or invalid")
        if any(not isinstance(k, str) or not k.strip() for k in dataset["task_keys"]):
            raise ValueError("task identities must be nonempty strings")
    if plan["schema"] == DENSE_SCHEMA:
        if train.get("format") != DENSE_FORMAT:
            raise ValueError("dense training requires explicitly masked segments")
        for key in (
            "supervised_tokens",
            "assistant_responses",
            "source_sessions",
            "source_total_assistant_responses",
        ):
            if type(train.get(key)) is not int or train[key] <= 0:
                raise ValueError("dense teacher source and target totals must be bound")
        excluded = train.get("excluded_assistant_responses")
        if (
            type(excluded) is not int
            or excluded < 0
            or train["assistant_responses"] + excluded != train["source_total_assistant_responses"]
        ):
            raise ValueError("dense eligible and excluded response totals must exhaust the source")
    elif train.get("format") == DENSE_FORMAT:
        raise ValueError("dense teacher data requires its separately versioned plan")
    # Pinned worker per-row diagnostics are suffix-only. Keep the bound dev set
    # on the qualified contiguous last-assistant path, never sparse-mask eval.
    if dev.get("format") == DENSE_FORMAT:
        raise ValueError("held-out evaluation must remain contiguous last-assistant windows")
    train_keys, dev_keys = set(train["task_keys"]), set(dev["task_keys"])
    if not dev_keys or len(dev_keys) != len(dev["task_keys"]):
        raise ValueError("validation must bind distinct nonempty held-out tasks")
    if not train_keys or train_keys & dev_keys:
        raise ValueError("train/dev task overlap or empty training task set")
    if train["path"] == dev["path"] or any(
        type(x["rows"]) is not int or x["rows"] <= 0 for x in (train, dev)
    ):
        raise ValueError("distinct nonempty train/dev artifacts required")
    if recipe["max_steps"] != math.ceil(train["rows"] / recipe["batch_size"]) * recipe["epochs"]:
        raise ValueError("max_steps must equal complete epochs with the native kept tail batch")
    model = plan["model"]
    if not re.fullmatch(r"[a-f0-9]{40}", model["revision"]) or not model["files"]:
        raise ValueError("exact model revision and file bindings required")
    if not {"config.json", "tokenizer_config.json"}.issubset({x["path"] for x in model["files"]}):
        raise ValueError("model config/tokenizer config bindings required")
    for item in model["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("model inventory path escapes model root")
    for key in ("project", "entity", "group", "run_id", "name"):
        if not isinstance(plan["wandb"][key], str) or not plan["wandb"][key].strip():
            raise ValueError("complete W&B run identity required")
    if check_files:
        for spec in (train, dev):
            _checked_file(Path(spec["path"]), spec["sha256"])
        for item in model["files"]:
            _checked_file(Path(model["root"]) / item["path"], item["sha256"])
        # Optional full model staging proof, beyond runtime sidecar bindings.
        if plan.get("staged_receipt"):
            _checked_file(Path(plan["staged_receipt"]["path"]), plan["staged_receipt"]["sha256"])


def validate_runtime_sources(root: Path | None = None) -> None:
    if root is None:
        spec = importlib.util.find_spec("skyrl")
        if spec is None or not spec.submodule_search_locations:
            raise ValueError("pinned SkyRL package is unavailable")
        root = Path(next(iter(spec.submodule_search_locations))).parent
    for path, expected in SOURCE_SHA256.items():
        _checked_file(root / path, expected)


def sft_overrides(plan: dict) -> dict:
    r, w = plan["recipe"], plan["wandb"]
    output = Path(plan["output_root"])
    options = {
        "strategy": "fsdp",
        "model.path": plan["model"]["root"],
        "max_length": r["max_length"],
        "train_on_what": (
            "all_assistant_messages" if plan["schema"] == DENSE_SCHEMA else "last_assistant_message"
        ),
        "batch_size": r["batch_size"],
        "micro_train_batch_size_per_gpu": r["microbatch_per_gpu"],
        "optimizer_config.lr": r["lr"],
        "optimizer_config.scheduler": "constant_with_warmup",
        "optimizer_config.num_warmup_steps": 0,
        "placement.num_nodes": r["nodes"],
        "placement.num_gpus_per_node": r["gpus_per_node"],
        "sequence_parallel_size": 1,
        "model_config_kwargs.fleet_force_qwen35_torch_gdn": True,
        "logger": "wandb",
        "project_name": w["project"],
        "run_name": w["name"],
        "tags": w.get("tags", []),
        "seed": r["seed"],
        "dataset_name": plan["datasets"]["train"]["path"],
        "dataset_split": "train",
        "eval_dataset_name": plan["datasets"]["dev"]["path"],
        "eval_dataset_split": "validation",
        "eval_before_train": True,
        "eval_interval": r["eval_interval"],
        "ckpt_path": str(output / "checkpoints"),
        "ckpt_interval": r["checkpoint_interval"],
        # This module retains latest N plus best; native deletion must be disabled.
        "max_ckpts_to_keep": -1,
        "hf_save_interval": 0,
        "cache_dir": str(output / "tokenized_cache"),
        "disable_cache": True,
        "num_epochs": r["epochs"],
        "max_training_steps": r["max_steps"],
        "num_workers": 0,
        "dataloader_num_workers": 0,
    }
    if plan["schema"] == DENSE_SCHEMA:
        options.update(
            {"fsdp_config.cpu_offload": False, "optimizer_config.offload_after_step": False}
        )
    return options


def build_runtime_configs(plan: dict):
    """Use native SFT configuration, with explicit single-policy dense residency."""
    from skyrl.train.config.sft_config import SFTConfig, build_skyrl_config_for_sft

    cfg = SFTConfig.from_cli_overrides(sft_overrides(plan))
    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    if plan["schema"] == DENSE_SCHEMA:
        # The native bridge disables colocate_all but inherits the RL default
        # colocate_policy_ref=True. This arm has no reference/inference actor.
        skyrl_cfg.trainer.placement.colocate_policy_ref = False
        assert not skyrl_cfg.trainer.placement.colocate_all
        assert not skyrl_cfg.trainer.policy.fsdp_config.cpu_offload
        assert not skyrl_cfg.trainer.policy.optimizer_config.offload_after_step
    return cfg, skyrl_cfg


def tokenize_rows(
    rows: list[dict], spec: dict, tokenizer, tokenize, *, max_length: int
) -> list[dict]:
    """Apply the pinned native target mask and reject any lost/truncated row."""
    if len(rows) != spec["rows"] or {r["task_key"] for r in rows} != set(spec["task_keys"]):
        raise ValueError("dataset row count or task identity mismatch")
    if len({r["window_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate training window")
    result = []
    for row in rows:
        if (
            not isinstance(row["messages"], list)
            or not row["messages"]
            or row["messages"][-1]["role"] != "assistant"
        ):
            raise ValueError("window must contain native messages and end at assistant target")
        if any(
            message.get(key)
            for message in row["messages"]
            for key in ("reasoning_content", "reasoning", "thinking")
        ):
            raise ValueError("visible-action objective must not contain private reasoning fields")
        item = tokenize(row, tokenizer, max_length=None)
        if item is None or not 0 < item["num_actions"] < len(item["input_ids"]) <= max_length:
            raise ValueError("empty target or overlength window; truncation is prohibited")
        if item["loss_mask"] != [1] * item["num_actions"]:
            raise ValueError("expected contiguous last-assistant target mask")
        if "token_count" in row and row["token_count"] != len(item["input_ids"]):
            raise ValueError("materialized and executed tokenization disagree")
        result.append({**item, "task_key": row["task_key"], "window_id": row["window_id"]})
    return result


def dense_rows(rows: list[dict], spec: dict, *, max_length: int, vocab_size: int) -> list[dict]:
    """Validate once-only eligible targets and adapt full masks to native suffix masks.

    ``num_actions`` is the WIDTH of the suffix, including masked interior tool
    tokens. It is not the number of supervised tokens. The native collator
    supplies token-normalization from the positive mask count.
    """
    if spec.get("format") != DENSE_FORMAT:
        raise ValueError("dense row format not explicitly selected")
    if len(rows) != spec["rows"] or {r["task_key"] for r in rows} != set(spec["task_keys"]):
        raise ValueError("dense dataset row count or task identity mismatch")
    if len({r["window_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate dense segment")
    sessions = {}
    supervised = responses = 0
    result = []
    for row in rows:
        ids, mask = row["input_ids"], row["loss_mask"]
        if (
            not isinstance(ids, list)
            or not 1 < len(ids) <= max_length
            or any(type(x) is not int or not 0 <= x < vocab_size for x in ids)
        ):
            raise ValueError("invalid or overlength pretokenized segment")
        if (
            not isinstance(mask, list)
            or len(mask) != len(ids)
            or mask[0] != 0
            or any(type(x) is not int or x not in (0, 1) for x in mask)
            or not sum(mask)
        ):
            raise ValueError("invalid full-token assistant mask or missing causal context")
        if row.get("token_count") != len(ids) or row.get("target_token_count") != sum(mask):
            raise ValueError("dense token inventory disagrees with payload")
        sid, count = row["source_session_id"], row["source_assistant_count"]
        if not isinstance(sid, str) or not sid or type(count) is not int or count <= 0:
            raise ValueError("dense source identity or response count missing")
        eligible, exclusions = row["eligible_assistant_indices"], row["excluded_assistant_targets"]
        if (
            not isinstance(eligible, list)
            or not eligible
            or any(type(x) is not int or not 0 <= x < count for x in eligible)
            or eligible != sorted(set(eligible))
            or not isinstance(exclusions, list)
        ):
            raise ValueError("invalid eligible-response inventory")
        excluded = {}
        for entry in exclusions:
            index, message, reason = (
                entry["assistant_index"],
                entry["source_message_index"],
                entry["reason"],
            )
            if (
                type(index) is not int
                or not 0 <= index < count
                or index in excluded
                or type(message) is not int
                or message < 0
                or reason not in DENSE_EXCLUSION_REASONS
            ):
                raise ValueError("invalid excluded-response provenance")
            excluded[index] = (message, reason)
        if (
            list(excluded) != sorted(excluded)
            or set(eligible) & set(excluded)
            or set(eligible) | set(excluded) != set(range(count))
        ):
            raise ValueError("eligible and excluded inventories are not disjoint and exhaustive")
        source = sessions.setdefault(
            sid,
            {
                "count": count,
                "task": row["task_key"],
                "targets": {},
                "eligible": tuple(eligible),
                "excluded": excluded,
            },
        )
        if (
            source["count"] != count
            or source["task"] != row["task_key"]
            or source["eligible"] != tuple(eligible)
            or source["excluded"] != excluded
        ):
            raise ValueError("dense source metadata changes across segments")
        copied = row["copied_context_assistant_indices"]
        if (
            not isinstance(copied, list)
            or len(set(copied)) != len(copied)
            or any(type(x) is not int or not 0 <= x < count for x in copied)
        ):
            raise ValueError("invalid copied-context assistant inventory")
        spans = row["target_spans"]
        if not isinstance(spans, list) or not spans:
            raise ValueError("dense segment lacks assistant target spans")
        expected_mask = [0] * len(ids)
        previous_end = 0
        previous_assistant = -1
        previous_message = -1
        for span in spans:
            index, message = span["assistant_index"], span["source_message_index"]
            start, end = span["token_start"], span["token_end"]
            if (
                any(type(x) is not int for x in (index, message, start, end))
                or not 0 <= index < count
                or message < 0
                or not 1 <= start < end <= len(ids)
                or start < previous_end
                or index <= previous_assistant
                or message <= previous_message
            ):
                raise ValueError("invalid, unordered or overlapping assistant target spans")
            if index not in source["eligible"] or index in source["targets"] or index in copied:
                raise ValueError("excluded, repeated, or copied-context assistant targeted")
            if span["source_target_sha256"].removeprefix("sha256:") != _unsigned_digest(
                ids[start:end]
            ):
                raise ValueError("assistant target tokens differ from source span digest")
            source["targets"][index] = message
            expected_mask[start:end] = [1] * (end - start)
            previous_end, previous_assistant, previous_message = end, index, message
        if mask != expected_mask:
            raise ValueError("loss mask trains outside declared assistant targets")
        first = mask.index(1)
        result.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "num_actions": len(ids) - first,
                "loss_mask": mask[first:],
                "task_key": row["task_key"],
                "window_id": row["window_id"],
            }
        )
        supervised += sum(mask)
        responses += len(spans)
    for source in sessions.values():
        if set(source["targets"]) != set(source["eligible"]):
            raise ValueError("eligible source assistant response coverage is incomplete")
        provenance = {**source["targets"], **{i: v[0] for i, v in source["excluded"].items()}}
        messages = [provenance[i] for i in range(source["count"])]
        if messages != sorted(set(messages)):
            raise ValueError("source assistant order or uniqueness differs across segments")
    if (supervised, responses, len(sessions)) != (
        spec["supervised_tokens"],
        spec["assistant_responses"],
        spec["source_sessions"],
    ):
        raise ValueError("dense supervised-token, response or source total mismatch")
    if (
        sum(s["count"] for s in sessions.values()),
        sum(len(s["excluded"]) for s in sessions.values()),
    ) != (spec["source_total_assistant_responses"], spec["excluded_assistant_responses"]):
        raise ValueError("dense original-source or excluded-response total mismatch")
    return result


def prepare_rows(rows, spec, tokenizer, tokenize, *, max_length):
    if spec.get("format") == DENSE_FORMAT:
        return dense_rows(rows, spec, max_length=max_length, vocab_size=len(tokenizer))
    return tokenize_rows(rows, spec, tokenizer, tokenize, max_length=max_length)


class EvalAccumulator:
    """Recover unnormalized NLL from native per-row, globally scaled losses."""

    def __init__(self):
        self.tasks = defaultdict(lambda: [0.0, 0, 0])

    def add_batch(
        self, metadata: list[dict], counts: list[int], outputs: list[dict], batch_loss: float
    ) -> None:
        if (
            len(metadata) != len(counts)
            or len(outputs) < len(metadata)
            or any(n <= 0 for n in counts)
        ):
            raise ValueError("eval row/target accounting mismatch")
        count = sum(counts)
        summed_scaled = 0.0
        for row, n, output in zip(metadata, counts, outputs, strict=False):
            values = output["elementwise_loss"]
            if len(values) != n or not all(
                math.isfinite(float(x)) and float(x) >= 0 for x in values
            ):
                raise ValueError("invalid native per-target loss output")
            scaled = math.fsum(float(x) for x in values)
            summed_scaled += scaled
            totals = self.tasks[row["task_key"]]
            totals[0] += scaled * count  # native mask scale is 1 / real batch targets
            totals[1] += n
            totals[2] += 1
        if any(item["elementwise_loss"] for item in outputs[len(metadata) :]):
            raise ValueError("padding contributed to validation loss")
        if not math.isfinite(batch_loss) or not math.isclose(
            summed_scaled, batch_loss, rel_tol=2e-4, abs_tol=2e-5
        ):
            raise ValueError("per-task loss does not reconcile with native global loss")

    def metrics(self, expected_tasks: set[str]) -> dict:
        if not expected_tasks or set(self.tasks) != expected_tasks:
            raise ValueError("validation task coverage differs from the frozen task set")
        nll = math.fsum(value[0] for value in self.tasks.values())
        tokens = sum(value[1] for value in self.tasks.values())
        macro = math.fsum(value[0] / value[1] for value in self.tasks.values()) / len(self.tasks)
        return {
            "eval_loss": nll / tokens,
            "task_macro_loss": macro,
            "supervised_tokens": tokens,
            "windows": sum(v[2] for v in self.tasks.values()),
            "tasks": len(self.tasks),
        }


def retention_steps(saved: set[int], best_step: int, keep_latest: int) -> set[int]:
    return set(sorted(saved)[-keep_latest:]) | ({best_step} if best_step in saved else set())


class ProgressWatchdog:
    """Bounded fail-closed resource watchdog; unavailable telemetry is not idle."""

    def __init__(self, started_at: float):
        self.started_at = started_at
        self.last_progress = started_at
        self.previous_marker = None
        self.previous_io = None
        self.idle_since = None

    def observe(
        self,
        now: float,
        marker,
        gpu_mean: float | None,
        process_io: int | None,
        *,
        checkpoint_advancing: bool = False,
    ) -> str | None:
        changed = marker != self.previous_marker
        if changed:
            self.last_progress = now
        self.previous_marker = marker
        io_advanced = (
            process_io is not None
            and self.previous_io is not None
            and process_io - self.previous_io >= 1024 * 1024
        )
        self.previous_io = process_io
        elapsed = now - self.started_at
        if elapsed >= WATCHDOG_HARD_SECONDS and (
            not checkpoint_advancing or elapsed >= WATCHDOG_HARD_SECONDS + WATCHDOG_DRAIN_SECONDS
        ):
            return "hard_runtime_bound"
        if changed or io_advanced or gpu_mean is None or process_io is None or gpu_mean >= 1:
            self.idle_since = None
        elif self.idle_since is None:
            self.idle_since = now
        if (
            elapsed >= WATCHDOG_STARTUP_SECONDS
            and self.idle_since is not None
            and now - self.idle_since >= WATCHDOG_IDLE_SECONDS
            and now - self.last_progress >= WATCHDOG_IDLE_SECONDS
        ):
            return "confirmed_no_progress_idle"
        return None


def _output_progress(output: Path) -> tuple[tuple, tuple]:
    def fingerprint(paths):
        found = []
        for path in paths:
            try:
                stat = path.stat()
                if path.is_file():
                    found.append((str(path.relative_to(output)), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:  # an atomic write/prune can race the observation
                continue
        return tuple(sorted(found))

    receipts = fingerprint(output / name for name in ("PROGRESS.json", "ACTIVITY.json"))
    checkpoints = fingerprint((output / "checkpoints").rglob("*"))
    return receipts, checkpoints


def _utilization_snapshot() -> tuple[float | None, int | None]:
    # No command lines, environment, logs or process contents are read or printed.
    gpu_mean = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if values and all(math.isfinite(value) and 0 <= value <= 100 for value in values):
            gpu_mean = sum(values) / len(values)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    io_bytes = 0
    observed = False
    # This container belongs entirely to this job; /proc PIDs are namespace local.
    # Physical I/O ignores normal tiny Ray heartbeat/log writes (<1MiB/sample).
    for path in Path("/proc").glob("[0-9]*/io"):
        try:
            counters = dict(line.split(":", 1) for line in path.read_text().splitlines())
            io_bytes += int(counters["read_bytes"]) + int(counters["write_bytes"])
            observed = True
        except (OSError, ValueError, KeyError):
            continue
    return gpu_mean, io_bytes if observed else None


def _wait_for_training(ray, task, output: Path) -> dict:
    watchdog = ProgressWatchdog(time.monotonic())
    previous_checkpoint = None
    while True:
        ready, _ = ray.wait([task], timeout=WATCHDOG_POLL_SECONDS)
        if ready:
            return ray.get(task)
        marker, checkpoint = _output_progress(output)
        gpu_mean, io_bytes = _utilization_snapshot()
        reason = watchdog.observe(
            time.monotonic(),
            (marker, checkpoint),
            gpu_mean,
            io_bytes,
            checkpoint_advancing=checkpoint != previous_checkpoint,
        )
        previous_checkpoint = checkpoint
        write_receipt(
            output / "WATCHDOG.json",
            {
                "observed_at_unix": time.time(),
                "gpu_mean_utilization_pct": gpu_mean,
                "process_io_available": io_bytes is not None,
                "state": reason or "monitoring",
            },
            replace=True,
        )
        if reason:
            ray.cancel(task, force=True, recursive=True)
            raise RuntimeError(reason)


def _configure_wandb(plan: dict) -> None:
    for key, value in {
        "WANDB_ENTITY": plan["wandb"]["entity"],
        "WANDB_RUN_GROUP": plan["wandb"]["group"],
        "WANDB_RUN_ID": plan["wandb"]["run_id"],
        "WANDB_RESUME": "never",
        "WANDB_MODE": "online",
        "WANDB_CONSOLE": "off",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_SAVE_CODE": "false",
        "WANDB_DISABLE_GIT": "true",
        "WANDB_DIR": str(Path(plan["output_root"]) / "wandb"),
    }.items():
        os.environ[key] = value
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("W&B secret injection missing")
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True, mode=0o700)


def explicit_tracking_class(base):
    """Native Tracking.__del__ assumes success; only our outer outcome may finish."""

    class ExplicitTracking(base):
        def __del__(self):
            pass

    return ExplicitTracking


def finalize_failed_run(trainer, output: Path, error: BaseException) -> list[str]:
    """Best-effort independent cleanup: storage/SDK defects cannot skip exit1."""
    failures = []
    try:
        with (output / "private-runtime-failure.log").open("a") as stream:
            traceback.print_exc(file=stream)
    except BaseException:
        failures.append("private_log")
    try:
        if trainer._ray_gpu_monitor is not None:
            trainer._ray_gpu_monitor.stop()
    except BaseException:
        failures.append("gpu_monitor")
    try:
        import wandb
    except ImportError:
        return failures + ["tracking_import"]
    try:
        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    "status": "failed",
                    "error_class": type(error).__name__,
                    "optimizer_step": trainer.global_step,
                }
            )
    except BaseException:
        failures.append("tracking_summary")
    try:
        # Also handles a run created before native tracker construction failed.
        if wandb.run is not None:
            wandb.finish(exit_code=1)
    except BaseException:
        failures.append("tracking_finish")
    return failures


def _make_trainer_class():
    from skyrl.backends.skyrl_train.training_batch import pad_training_input_batch
    from skyrl.train.sft_trainer import SFTTrainer, tokenize_chat_example
    from skyrl.train.utils.callbacks import TrainingCallback
    from skyrl.train.utils.tracking import Tracking
    from skyrl.train.utils.utils import Timer

    class EvidenceCallback(TrainingCallback):
        def on_log(self, trainer, event, control):
            event.logs["train/global_step"] = event.global_step
            event.logs["train/epoch_fraction"] = event.global_step / max(event.steps_per_epoch, 1)
            event.logs.update(trainer.extra_train_metrics if "train/loss" in event.logs else {})
            # Durable local scalar stream also covers W&B network interruptions.
            with (trainer.output / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {"optimizer_step": event.global_step, "time": time.time(), **event.logs}
                    )
                    + "\n"
                )
            write_receipt(
                trainer.output / "PROGRESS.json",
                {
                    "optimizer_step": event.global_step,
                    "phase": "validation"
                    if any(k.startswith("eval/") for k in event.logs)
                    else "training",
                    "observed_at_unix": time.time(),
                },
                replace=True,
            )

        def on_step_end(self, trainer, event, control):
            # Final checkpoint exists before final validation, including a tail batch.
            if event.global_step == event.total_steps:
                control.should_save = True

        def on_eval_end(self, trainer, event, control):
            metrics = event.metrics
            selected = (
                trainer.best is None or metrics["task_macro_loss"] < trainer.best["task_macro_loss"]
            )
            if selected:
                trainer.best = {
                    "optimizer_step": event.global_step,
                    "task_macro_loss": metrics["task_macro_loss"],
                    "token_weighted_loss": metrics["eval_loss"],
                    "checkpoint_path": plan_checkpoint(trainer.plan, event.global_step),
                }
                write_receipt(trainer.output / "BEST_CHECKPOINT.json", trainer.best, replace=True)
            write_receipt(
                trainer.output / "validation" / f"step-{event.global_step:06d}.json",
                {
                    "optimizer_step": event.global_step,
                    **metrics,
                    "plan_sha256": trainer.plan["plan_sha256"],
                },
            )
            trainer.prune_checkpoints()

    class AuditedSFTTrainer(SFTTrainer):
        def __init__(self, cfg, skyrl_cfg, plan):
            super().__init__(cfg, skyrl_cfg=skyrl_cfg, callbacks=[EvidenceCallback()])
            self.plan = plan
            self.output = Path(plan["output_root"])
            self.saved_steps = set()
            self.best = None
            self.extra_train_metrics = {}
            self.target_tokens_seen = 0

        def _init_workers(self):
            super()._init_workers()
            if self.plan["schema"] == DENSE_SCHEMA:
                # Pinned FSDP2 initialization broadcasts non-persistent buffers
                # (including RoPE inv_freq) back to CPU. Turning off colocation
                # skips the dispatcher's usual initial backload as well as its
                # repeated offloads. Explicitly finish initialization ONCE via
                # the native all-rank API; never recreate buffers or cast values.
                actor = self.dispatch._actor_groups["policy"]
                replies = actor.backload_to_gpu(backload_optimizer=False, backload_model=True)
                expected = self.plan["recipe"]["nodes"] * self.plan["recipe"]["gpus_per_node"]
                if (
                    not isinstance(replies, list)
                    or len(replies) != expected
                    or any(x is not None for x in replies)
                ):
                    raise ValueError("initial all-rank model backload was not acknowledged")
                write_receipt(
                    self.output / "DEVICE_INITIALIZED.json",
                    {
                        "plan_sha256": self.plan["plan_sha256"],
                        "ranks_acknowledged": expected,
                        "method": "native_backload_to_gpu_once_after_init",
                        "optimizer_steps": 0,
                        "observed_at_unix": time.time(),
                    },
                )

        def _init_tracker(self):
            self.tracker = explicit_tracking_class(Tracking)(
                project_name=self.cfg.trainer.project_name,
                experiment_name=self.cfg.trainer.run_name,
                backend=self.cfg.trainer.logger,
                config=self.sft_cfg,
                tags=self.cfg.trainer.tags,
            )
            import wandb

            run = wandb.run
            expected = self.plan["wandb"]
            if (
                run is None
                or run.id != expected["run_id"]
                or run.entity != expected["entity"]
                or run.project != expected["project"]
            ):
                raise ValueError("W&B run identity mismatch")
            run.define_metric("train/global_step")
            run.define_metric("train/*", step_metric="train/global_step")
            run.define_metric("eval/*", step_metric="train/global_step")
            run.define_metric("eval/task_macro_loss", summary="min")
            run.config.update(
                {
                    "experiment_plan_sha256": self.plan["plan_sha256"],
                    "model_repo": self.plan["model"]["repo"],
                    "model_revision": self.plan["model"]["revision"],
                    "split_manifest_sha256": self.plan.get("split_manifest_sha256"),
                    "train_rows": self.plan["datasets"]["train"]["rows"],
                    "dev_rows": self.plan["datasets"]["dev"]["rows"],
                    "dev_tasks": len(self.plan["datasets"]["dev"]["task_keys"]),
                    "target_policy": (
                        "visible_all_assistant_once"
                        if self.plan["schema"] == DENSE_SCHEMA
                        else "visible_last_assistant_message"
                    ),
                    "dev_target_policy": "visible_last_assistant_message",
                    "train_expected_supervised_tokens": self.plan["datasets"]["train"].get(
                        "supervised_tokens"
                    ),
                    "train_source_sessions": self.plan["datasets"]["train"].get("source_sessions"),
                    "train_assistant_responses": self.plan["datasets"]["train"].get(
                        "assistant_responses"
                    ),
                    "train_original_assistant_responses": self.plan["datasets"]["train"].get(
                        "source_total_assistant_responses"
                    ),
                    "train_excluded_assistant_responses": self.plan["datasets"]["train"].get(
                        "excluded_assistant_responses"
                    ),
                    "corpus_manifest_sha256": self.plan.get("corpus_manifest_sha256"),
                    "execution_resources": self.plan.get("execution", {}).get("resources"),
                    "selection_metric": "eval/task_macro_loss",
                    "inline_hf_export": False,
                },
                allow_val_change=False,
            )
            write_receipt(
                self.output / "WANDB.json",
                {"url": run.url, "run_id": run.id, "project": run.project, "entity": run.entity},
            )

        def _load_split(self, split):
            import pyarrow.parquet as pq

            spec = self.plan["datasets"][split]
            _checked_file(Path(spec["path"]), spec["sha256"])
            rows = pq.read_table(spec["path"]).to_pylist()
            return prepare_rows(
                rows,
                spec,
                self.tokenizer,
                tokenize_chat_example,
                max_length=self.sft_cfg.max_length,
            )

        def load_dataset(self):
            return self._load_split("train")

        def load_eval_dataset(self):
            self.dev_rows = self._load_split("dev")
            return self.dev_rows

        def run_eval(self):
            accumulator = EvalAccumulator()
            cursor = batches = 0
            for batch in self.eval_dataloader:
                n = batch["sequences"].shape[0]
                counts = (batch["loss_mask"] > 0).sum(dim=1).tolist()
                metadata = self.dev_rows[cursor : cursor + n]
                if n < self.eval_dataloader.batch_size:
                    batch = pad_training_input_batch(batch, self.eval_dataloader.batch_size - n)
                output = self.dispatch.forward(
                    "policy", batch, loss_fn="cross_entropy", loss_fn_config=None
                )
                accumulator.add_batch(
                    metadata, counts, output.loss_fn_outputs, float(output.metrics["loss"])
                )
                cursor += n
                batches += 1
                write_receipt(
                    self.output / "ACTIVITY.json",
                    {
                        "phase": "validation",
                        "optimizer_step": self.global_step,
                        "completed_batches": batches,
                        "observed_at_unix": time.time(),
                    },
                    replace=True,
                )
            if cursor != len(self.dev_rows):
                raise ValueError("validation did not consume the complete held-out dataset")
            return accumulator.metrics(set(self.plan["datasets"]["dev"]["task_keys"])), batches

        def train_step(self, batch, step):
            # Exact upstream calls, preserving the LR scalar it otherwise drops.
            timings = {}
            with Timer("forward_backward", timings):
                output = self.dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")
            with Timer("optim_step", timings):
                grad_norm = self.dispatch.optim_step("policy")
            if self._torch_profiler_enabled:
                self.dispatch.profile_step("policy")
            loss = float(output.metrics.get("final_loss", output.metrics.get("loss", float("nan"))))
            lr = float(output.metrics["lr"])
            if not all(math.isfinite(x) for x in (loss, lr, float(grad_norm))):
                raise ValueError("nonfinite training metric")
            targets = int((batch["loss_mask"] > 0).sum().item())
            self.target_tokens_seen += targets
            self.extra_train_metrics = {
                "train/lr": lr,
                "train/supervised_tokens": targets,
                "train/total_supervised_tokens": self.target_tokens_seen,
            }
            return {"loss": loss, "grad_norm": grad_norm, "timings": timings}

        def save_checkpoint(self):
            path = Path(super().save_checkpoint())
            # Native save catches dataloader-save errors: require it explicitly.
            if not (path / "data.pt").is_file() or not (path / "trainer_state.pt").is_file():
                raise ValueError("checkpoint lacks sampler/trainer resume state")
            self.saved_steps.add(self.global_step)
            write_receipt(
                self.output / "checkpoint_receipts" / f"step-{self.global_step:06d}.json",
                {
                    "optimizer_step": self.global_step,
                    "checkpoint_path": str(path),
                    "plan_sha256": self.plan["plan_sha256"],
                    "saved_at_unix": time.time(),
                },
            )
            self.prune_checkpoints()
            return str(path)

        def prune_checkpoints(self):
            keep = retention_steps(
                self.saved_steps,
                self.best["optimizer_step"] if self.best else 0,
                self.plan["recipe"]["keep_checkpoints"],
            )
            for step in sorted(self.saved_steps - keep):
                path = self.output / "checkpoints" / f"global_step_{step}"
                if (
                    path.is_symlink()
                    or path.resolve().parent != (self.output / "checkpoints").resolve()
                ):
                    raise ValueError("checkpoint retention target escaped owned root")
                shutil.rmtree(path)
                self.saved_steps.remove(step)

        def save_hf_model(self):
            raise RuntimeError(
                "inline HF export is prohibited; "
                "use the completed checkpoint in a separate export job"
            )

    return AuditedSFTTrainer


def plan_checkpoint(plan: dict, step: int) -> str:
    return (
        plan["model"]["root"]
        if step == 0
        else str(Path(plan["output_root"]) / "checkpoints" / f"global_step_{step}")
    )


def _run_training(plan: dict) -> dict:
    _configure_wandb(plan)
    cfg, skyrl_cfg = build_runtime_configs(plan)
    skyrl_cfg.trainer.log_path = str(Path(plan["output_root"]) / "private_logs")
    trainer = _make_trainer_class()(cfg, skyrl_cfg, plan)
    try:
        trainer.setup()
        trainer.train()
        expected = plan["recipe"]["max_steps"]
        pointer = Path(cfg.ckpt_path) / "latest_ckpt_global_step.txt"
        if int(pointer.read_text()) != expected or trainer.global_step != expected:
            raise ValueError("final checkpoint optimizer step mismatch")
        if (
            plan["schema"] == DENSE_SCHEMA
            and trainer.target_tokens_seen
            != plan["datasets"]["train"]["supervised_tokens"] * plan["recipe"]["epochs"]
        ):
            raise ValueError("completed epochs did not train every assistant target once per epoch")
        result = {
            "optimizer_step": expected,
            "checkpoint_path": plan_checkpoint(plan, expected),
            "best": trainer.best,
            "export_status": "pending_separate_zero_step_export",
            "supervised_tokens": trainer.target_tokens_seen,
            "plan_sha256": plan["plan_sha256"],
            "status": "training_complete",
        }
        import wandb

        wandb.run.summary.update(
            {
                "completed_optimizer_steps": expected,
                "best_checkpoint": trainer.best,
                "export_status": result["export_status"],
            }
        )
        trainer.shutdown()
        return result
    except BaseException as exc:
        # Do not use native log_exception: it uploads raw traceback and finishes exit0.
        incomplete = finalize_failed_run(trainer, Path(plan["output_root"]), exc)
        if incomplete:
            print(
                json.dumps({"status": "failure_cleanup_incomplete", "components": incomplete}),
                flush=True,
            )
        raise RuntimeError(f"SFT runtime failed: {type(exc).__name__}") from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preflight-tokenize", action="store_true")
    args = parser.parse_args()
    _checked_file(args.plan, args.plan_sha256)
    plan = json.loads(args.plan.read_text())
    plan["plan_sha256"] = args.plan_sha256.removeprefix("sha256:")
    validate_plan(plan)
    validate_runtime_sources()
    if args.preflight_tokenize:
        import pyarrow.parquet as pq
        from skyrl.train.sft_trainer import tokenize_chat_example
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            plan["model"]["root"], local_files_only=True, trust_remote_code=True
        )
        results = {}
        for split, spec in plan["datasets"].items():
            rows = prepare_rows(
                pq.read_table(spec["path"]).to_pylist(),
                spec,
                tokenizer,
                tokenize_chat_example,
                max_length=plan["recipe"]["max_length"],
            )
            results[split] = {
                "rows": len(rows),
                "tasks": len(spec["task_keys"]),
                "supervised_tokens": sum(sum(x["loss_mask"]) for x in rows),
            }
        print(json.dumps({"status": "tokenization_preflight_passed", **results}))
        return
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "dev_tasks": len(plan["datasets"]["dev"]["task_keys"]),
                    "max_steps": plan["recipe"]["max_steps"],
                }
            )
        )
        return
    output = Path(plan["output_root"])
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_receipt(
        output / "STARTED.json",
        {"plan_sha256": plan["plan_sha256"], "started_at_unix": time.time()},
    )
    ray = None
    try:
        import ray
        from skyrl.train.utils.utils import initialize_ray

        _, cfg = build_runtime_configs(plan)
        cfg.trainer.log_path = str(output / "private_logs")
        initialize_ray(cfg)
        task = ray.remote(num_cpus=1)(_run_training).remote(plan)
        result = _wait_for_training(ray, task, output)
        write_receipt(output / "TRAINING_COMPLETE.json", result)
        print(
            json.dumps({"status": "training_complete", "optimizer_step": result["optimizer_step"]})
        )
    except BaseException as exc:
        write_receipt(
            output / "FAILED.json",
            {
                "error_class": type(exc).__name__,
                "plan_sha256": plan["plan_sha256"],
                "failed_at_unix": time.time(),
                "status": "failed",
            },
        )
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    finally:
        if ray is not None and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
