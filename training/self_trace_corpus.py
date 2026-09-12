"""Compile exact direct-interface Qwen recordings into private dense SFT data.

This is a local/CPU data boundary, not a collector or a launch command.  It
accepts only reviewed, digest-bound ``training.rl_episode`` outputs produced by
the exact fresh Qwen base through the direct ``bash``, ``submit_report`` tool
surface.  Historical OpenCode transcripts are intentionally incompatible.

The recorded token IDs are the training source of truth.  They are never
re-tokenized from transcript text.  Every fitting assistant span is supervised
once, copied context and tool observations remain masked, and whole episodes
are capped per exact task version before publication.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

from . import rl_episode
from .corpus import local_tokenizer
from .dense import Excluded as DenseExcluded
from .dense import segment_record
from .io import atomic_write_json, digest_json, file_sha256
from .rl_data import AUTHORITY as RL_AUTHORITY
from .sft import read_mapping
from .sft_runtime import DENSE_FORMAT, dense_rows

INDEX_SCHEMA = "cyber_qwen_direct_self_trace_index_v1"
CONFIG_SCHEMA = "cyber_qwen_direct_self_trace_corpus_config_v1"
CORPUS_SCHEMA = "cyber_dense_sft_corpus_v1"
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")


class SelfTraceError(ValueError):
    """Payload-free rejection reason."""


def _sealed(value: dict, schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise SelfTraceError("sealed metadata digest mismatch")


def _sha(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


def _canonical_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.int != 0 and str(parsed) == value


def _canonical_instance_id(value: object) -> bool:
    return isinstance(value, str) and bool(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value)
    )


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str):
        raise SelfTraceError("episode directory must be relative")
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise SelfTraceError("episode directory must be relative")
    return path


def _real_child(root: Path, relative: Path) -> Path:
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise SelfTraceError("reviewed episode path contains a symlink")
    if not path.is_dir():
        raise SelfTraceError("reviewed episode directory is missing")
    return path


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except Exception:
        raise SelfTraceError("private source JSON is unreadable") from None
    if not isinstance(value, dict):
        raise SelfTraceError("private source JSON is not an object")
    return value


def _raw_receipt_digest(value: dict) -> str:
    return fleet.sha256(fleet.canonical_json({k: v for k, v in value.items() if k != "sha256"}))


def _runs(mask: list[int]) -> list[tuple[int, int]]:
    result = []
    start = None
    for index, value in enumerate([*mask, 0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    return result


def _conversation_shape(conversation: dict, spans: list[tuple[int, int]]) -> list[dict]:
    """Return only structural assistant/tool metadata; never source content."""
    messages = conversation.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SelfTraceError("direct conversation anchor is missing")
    if any(not isinstance(message, dict) for message in messages):
        raise SelfTraceError("direct conversation structure is invalid")
    if messages[0].get("role") != "user":
        raise SelfTraceError("direct conversation anchor is missing")
    if any(message.get("role") not in {"user", "assistant", "tool"} for message in messages):
        raise SelfTraceError("direct conversation contains an unsupported role")
    if any(message.get("role") == "user" for message in messages[1:]):
        raise SelfTraceError("direct conversation contains a later user transition")
    assistants = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if len(assistants) != len(spans) or not assistants:
        raise SelfTraceError("assistant structure differs from native token masks")
    result = []
    for ordinal, index in enumerate(assistants):
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
            raise SelfTraceError("assistant target lacks its direct tool observation")
        name = messages[index + 1].get("name")
        if name not in {"bash", "submit_report"}:
            raise SelfTraceError("assistant target used an unsupported direct tool")
        result.append({"ordinal": ordinal, "message_index": index, "tool": name})
    if messages[-1].get("role") != "tool" or result[-1]["tool"] != "submit_report":
        raise SelfTraceError("verified source lacks terminal direct report submission")
    return result


def _segment_native(
    *,
    episode_id: str,
    task_key: str,
    tokens: list[int],
    response_length: int,
    response_mask: list[int],
    shape: list[dict],
    max_length: int,
    context_tokens: int,
) -> tuple[list[dict], list[dict]]:
    """Adapt exact native tokens to the existing qualified dense segmenter."""
    prompt_length = len(tokens) - response_length
    spans = _runs([0] * prompt_length + response_mask)
    if len(spans) != len(shape):
        raise SelfTraceError("assistant structure differs from native token masks")
    chunks = []
    for ordinal, (start, end) in enumerate(spans):
        chunks.append(
            {
                "ids": tokens[start:end],
                "mask": [1] * (end - start),
                "message_index": shape[ordinal]["message_index"],
                "assistant_index": ordinal,
                "target": (0, end - start),
            }
        )
        next_start = spans[ordinal + 1][0] if ordinal + 1 < len(spans) else len(tokens)
        if end < next_start:
            chunks.append(
                {
                    "ids": tokens[end:next_start],
                    "mask": [0] * (next_start - end),
                    "message_index": shape[ordinal]["message_index"] + 1,
                    "assistant_index": None,
                    "target": None,
                }
            )
    record = {
        "record_id": episode_id,
        "lineage": {"task_key": task_key},
        "source": {"model": "Qwen/Qwen3.8-27B"},
    }
    try:
        rows = segment_record(
            record,
            tokens[:prompt_length],
            chunks,
            max_tokens=max_length,
            context_budget=context_tokens,
        )
    except DenseExcluded as exc:
        raise SelfTraceError("native direct trace has no qualified dense target") from exc
    return rows, rows[0]["excluded_assistant_targets"]


def _episode(
    root: Path,
    entry: dict,
    index: dict,
    train: set[tuple[str, str]],
    *,
    max_length: int,
    context_tokens: int,
) -> dict:
    required = {"episode_id", "directory", "task_key", "task_version_id", "accepted_sha256"}
    if (
        set(entry) != required
        or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", str(entry.get("episode_id", "")))
        or not isinstance(entry.get("task_key"), str)
        or not entry["task_key"]
        or not _canonical_uuid(entry.get("task_version_id"))
        or not _sha(entry.get("accepted_sha256"))
    ):
        raise SelfTraceError("reviewed episode index entry is incomplete")
    if (entry["task_key"], entry["task_version_id"]) not in train:
        raise SelfTraceError("reviewed self trace is outside the training split")
    directory = _real_child(root, _safe_relative(entry["directory"]))
    if directory.name != entry["episode_id"]:
        raise SelfTraceError("reviewed episode path or identity differs")
    accepted_path = directory / "ACCEPTED.json"
    if file_sha256(accepted_path) != entry["accepted_sha256"]:
        raise SelfTraceError("reviewed episode acceptance file differs")
    accepted = _read_json(accepted_path)
    if accepted.get("sha256") != _raw_receipt_digest(accepted):
        raise SelfTraceError("episode acceptance receipt digest differs")
    expected_files = {
        "binding.json",
        "instance.json",
        "conversation.json",
        "reward.json",
        "cleanup.json",
        "recording.json",
    }
    if (
        set(accepted)
        != {
            "task_version_id",
            "instance_id",
            "verifier_execution_id",
            "done_reason",
            "config_sha256",
            "sample_count",
            "files",
            "sha256",
        }
        or set(accepted.get("files") or {}) != expected_files
        or accepted.get("sample_count") != 1
    ):
        raise SelfTraceError("episode acceptance file inventory differs")
    paths = {name: directory / name for name in expected_files}
    for name, path in paths.items():
        if path.is_symlink() or fleet.sha256(path.read_bytes()) != accepted["files"][name]:
            raise SelfTraceError("episode payload digest differs from acceptance")
    binding = _read_json(paths["binding.json"])
    instance = _read_json(paths["instance.json"])
    reward = _read_json(paths["reward.json"])
    cleanup = _read_json(paths["cleanup.json"])
    recording = _read_json(paths["recording.json"])
    conversation = _read_json(paths["conversation.json"])
    if binding.get("config_sha256") != fleet.digest_without(binding, "config_sha256"):
        raise SelfTraceError("episode binding digest differs")
    interface, model = index["interface"], index["model"]
    try:
        rl_episode._validate(binding)
    except Exception:
        raise SelfTraceError("episode is not the exact direct RL binding") from None
    task = binding.get("task")
    bound_model = binding.get("model")
    environment = binding.get("environment")
    verifier = binding.get("verifier")
    execution = binding.get("execution")
    authority = binding.get("authority")
    limits = binding.get("rl")
    if (
        set(binding)
        != {
            "run_id",
            "model",
            "authority",
            "execution",
            "environment",
            "rl",
            "task",
            "verifier",
            "initial_prompt_sha256",
            "initial_prompt_tokens_sha256",
            "config_sha256",
        }
        or binding.get("run_id") != entry["episode_id"]
        or not isinstance(task, dict)
        or set(task)
        != {
            "key",
            "version_id",
            "prompt_sha256",
            "env_variables_sha256",
            "output_json_schema_sha256",
            "cyber_contract",
        }
        or task.get("key") != entry["task_key"]
        or task.get("version_id") != entry["task_version_id"]
        or any(
            not _sha(task.get(name))
            for name in ("prompt_sha256", "env_variables_sha256", "output_json_schema_sha256")
        )
        or task.get("cyber_contract") != RL_AUTHORITY["required_cyber_contract"]
        or not isinstance(bound_model, dict)
        or set(bound_model) != {"repo", "revision", "root", "runtime_chat_template_sha256"}
        or bound_model.get("repo") != model["repo"]
        or bound_model.get("revision") != model["revision"]
        or bound_model.get("root") != model["root"]
        or bound_model.get("runtime_chat_template_sha256")
        != interface["runtime_chat_template_sha256"]
        or authority != RL_AUTHORITY
        or not isinstance(execution, dict)
        or set(execution) != {"required_task_tools", "required_task_tool_catalog_sha256"}
        or execution.get("required_task_tools") != ["bash", "submit_report"]
        or execution.get("required_task_tool_catalog_sha256")
        != interface["required_task_tool_catalog_sha256"]
        or not _sha(execution.get("required_task_tool_catalog_sha256"))
        or not isinstance(limits, dict)
        or set(limits)
        != {
            "max_turns",
            "episode_seconds",
            "tool_seconds",
            "tool_result_chars",
            "max_tokens_per_turn",
            "context_tokens",
        }
        or not isinstance(environment, dict)
        or set(environment)
        != {
            "id",
            "version",
            "version_id",
            "data_id",
            "data_version",
            "runtime_seed_content_sha256",
            "ttl_seconds",
        }
        or any(
            not isinstance(environment.get(name), str) or not environment[name]
            for name in ("id", "version", "data_id", "data_version")
        )
        or not _canonical_uuid(environment.get("version_id"))
        or not _sha(environment.get("runtime_seed_content_sha256"))
        or not isinstance(verifier, dict)
        or set(verifier) != {"id", "version_id", "version", "sha256", "function_name"}
        or not _canonical_uuid(verifier.get("id"))
        or not _canonical_uuid(verifier.get("version_id"))
        or not isinstance(verifier.get("version"), str)
        or not verifier["version"]
        or not _sha(verifier.get("sha256"))
        or verifier.get("function_name") != "verify"
        or not _sha(binding.get("initial_prompt_sha256"))
        or not _sha(binding.get("initial_prompt_tokens_sha256"))
    ):
        raise SelfTraceError("episode is not the exact direct fresh-base interface")
    attestation = reward.get("direct_authority_attestation")
    context = attestation.get("context") if isinstance(attestation, dict) else None
    activity = attestation.get("activity") if isinstance(attestation, dict) else None
    shadow = attestation.get("shadow") if isinstance(attestation, dict) else None
    direct = shadow.get("direct_verifier") if isinstance(shadow, dict) else None
    minimization = attestation.get("data_minimization") if isinstance(attestation, dict) else None
    instance_id = accepted.get("instance_id")
    evidence_run_id = instance.get("evidence_run_id")
    verifier_execution_id = accepted.get("verifier_execution_id")
    if (
        set(instance) != {"instance_id", "evidence_run_id"}
        or not _canonical_instance_id(instance_id)
        or not _canonical_uuid(evidence_run_id)
        or not _canonical_uuid(verifier_execution_id)
        or accepted.get("config_sha256") != binding["config_sha256"]
        or accepted.get("task_version_id") != entry["task_version_id"]
        or instance_id != instance.get("instance_id")
        or set(cleanup)
        != {
            "create_attempted",
            "instance_created",
            "instance_id",
            "instance_closed",
            "possible_instance_leak",
        }
        or cleanup.get("create_attempted") is not True
        or cleanup.get("instance_created") is not True
        or cleanup.get("instance_id") != instance_id
        or cleanup.get("instance_closed") is not True
        or cleanup.get("possible_instance_leak") is not False
        or set(reward)
        != {
            "task_key",
            "task_version_id",
            "instance_id",
            "reward",
            "verifier_execution_id",
            "direct_authority_attestation",
        }
        or reward.get("task_key") != entry["task_key"]
        or reward.get("task_version_id") != entry["task_version_id"]
        or reward.get("instance_id") != instance_id
        or reward.get("verifier_execution_id") != verifier_execution_id
        or type(reward.get("reward")) not in {int, float}
        or isinstance(reward.get("reward"), bool)
        or not 0 < float(reward["reward"]) <= 1
        or accepted.get("done_reason") != "report_submitted"
        or not isinstance(attestation, dict)
        or attestation.get("schema_version") != fleet.DIRECT_AUTHORITY_ATTESTATION_SCHEMA
        or not isinstance(context, dict)
        or context.get("task_key") != entry["task_key"]
        or context.get("task_version_id") != entry["task_version_id"]
        or context.get("instance_id") != instance_id
        or context.get("evidence_run_id") != evidence_run_id
        or context.get("verifier_version_id") != verifier["version_id"]
        or context.get("scoring_payload_mode") != fleet.RUNTIME_EVIDENCE_ONLY_V3
        or not isinstance(activity, dict)
        or activity.get("result_schema_version") != "cyber_verification_result_v3"
        or activity.get("reward") != float(reward["reward"])
        or activity.get("task_version_id") != entry["task_version_id"]
        or activity.get("verifier_execution_id") != verifier_execution_id
        or not isinstance(shadow, dict)
        or shadow.get("mode") != "authoritative"
        or shadow.get("status") != "authoritative"
        or shadow.get("match") is not True
        or shadow.get("production_execution_id") != verifier_execution_id
        or not isinstance(direct, dict)
        or direct.get("status") != "authoritative"
        or direct.get("match") is not True
        or direct.get("execution_id") != verifier_execution_id
        or direct.get("verifier_contract_version")
        != RL_AUTHORITY["required_cyber_contract"]["verifier_contract"]
        or direct.get("context_schema_version") != "cyber_verification_context_v1"
        or minimization
        != {
            "components_included": False,
            "diagnostics_included": False,
            "evidence_payloads_included": False,
            "prompts_included": False,
            "traces_included": False,
            "flags_included": False,
        }
    ):
        raise SelfTraceError("episode is not a verified positive released outcome")
    samples = recording.get("samples")
    if not isinstance(samples, list) or len(samples) != 1 or not isinstance(samples[0], dict):
        raise SelfTraceError("native recording must contain exactly one sample")
    sample = samples[0]
    if set(sample) != {"tokens", "response_length", "loss_mask", "rollout_log_probs"}:
        raise SelfTraceError("native recording fields differ")
    tokens, response, mask, probabilities = (
        sample["tokens"],
        sample["response_length"],
        sample["loss_mask"],
        sample["rollout_log_probs"],
    )
    if (
        not isinstance(tokens, list)
        or type(response) is not int
        or not 0 < response < len(tokens)
        or not isinstance(mask, list)
        or not isinstance(probabilities, list)
        or len(mask) != response
        or len(probabilities) != response
        or any(type(token) is not int or token < 0 for token in tokens)
        or any(type(value) is not int or value not in {0, 1} for value in mask)
        or any(
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value > 1e-6
            for value in probabilities
        )
        or any(
            value != 0 for value, selected in zip(probabilities, mask, strict=True) if not selected
        )
        or not any(mask)
    ):
        raise SelfTraceError("native recording token inventory differs")
    prompt_length = len(tokens) - response
    if binding["initial_prompt_tokens_sha256"] != fleet.sha256(
        fleet.canonical_json(tokens[:prompt_length])
    ):
        raise SelfTraceError("recorded prompt token digest differs")
    messages = conversation.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or not isinstance(messages[0], dict)
        or not isinstance(messages[0].get("content"), str)
        or fleet.sha256(messages[0]["content"].encode()) != task["prompt_sha256"]
    ):
        raise SelfTraceError("recorded task prompt differs from the exact task binding")
    spans = _runs([0] * (len(tokens) - response) + mask)
    shape = _conversation_shape(conversation, spans)
    rows, excluded = _segment_native(
        episode_id=entry["episode_id"],
        task_key=entry["task_key"],
        tokens=tokens,
        response_length=response,
        response_mask=mask,
        shape=shape,
        max_length=max_length,
        context_tokens=context_tokens,
    )
    eligible = {span["assistant_index"] for row in rows for span in row["target_spans"]}
    categories = collections.Counter()
    submit_tokens = 0
    for ordinal in eligible:
        start, end = spans[ordinal]
        categories[shape[ordinal]["tool"]] += 1
        if shape[ordinal]["tool"] == "submit_report":
            submit_tokens += end - start
    if categories["bash"] < 1:
        raise SelfTraceError("self trace lacks a retained non-submission tool target")
    return {
        "entry": entry,
        "rows": rows,
        "excluded": excluded,
        "supervised_tokens": sum(row["target_token_count"] for row in rows),
        "assistant_responses": sum(len(row["target_spans"]) for row in rows),
        "submit_report_responses": categories["submit_report"],
        "submit_report_tokens": submit_tokens,
        "stable": {path: file_sha256(path) for path in [accepted_path, *paths.values()]},
    }


def build(config: dict, *, relative_to: Path) -> dict:
    """Create a private train-only Parquet corpus and sanitized manifest."""
    required = {
        "schema",
        "source_root",
        "source_index",
        "split",
        "model_lock",
        "model_weights",
        "tokenizer_root",
        "max_length",
        "context_tokens",
        "max_episodes_per_task_version",
        "max_supervised_tokens_per_task_version",
        "max_submit_token_share",
        "max_submit_response_share",
        "source_seed",
        "fleet_dev_protocol_sha256",
        "output",
    }
    if set(config) != required or config.get("schema") != CONFIG_SCHEMA:
        raise SelfTraceError("self-trace corpus configuration fields differ")
    for name in ("source_index", "split", "model_lock", "model_weights"):
        if set(config[name]) != {"path", "sha256"} or not _sha(config[name]["sha256"]):
            raise SelfTraceError("bound input requires an exact path and digest")
    for name in (
        "max_length",
        "context_tokens",
        "max_episodes_per_task_version",
        "max_supervised_tokens_per_task_version",
    ):
        if type(config[name]) is not int or config[name] < (0 if name == "context_tokens" else 1):
            raise SelfTraceError("self-trace corpus numeric bound is invalid")
    for name in ("max_submit_token_share", "max_submit_response_share"):
        if type(config[name]) not in {int, float} or not 0 < config[name] <= 0.5:
            raise SelfTraceError("submission-dominance bound must be in (0, 0.5]")
    if not isinstance(config["source_seed"], str) or not config["source_seed"]:
        raise SelfTraceError("deterministic source seed is missing")
    if not _sha(config["fleet_dev_protocol_sha256"]):
        raise SelfTraceError("Fleet development protocol digest is missing")
    source_root = Path(config["source_root"])
    if not source_root.is_absolute() or source_root.is_symlink() or not source_root.is_dir():
        raise SelfTraceError("source root must be an exact absolute directory")
    output = Path(config["output"])
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise SelfTraceError("output must be a new exact absolute directory")
    bound = {
        name: relative_to / config[name]["path"]
        for name in ("source_index", "split", "model_lock", "model_weights")
    }
    for name, path in bound.items():
        if path.is_symlink() or file_sha256(path) != config[name]["sha256"]:
            raise SelfTraceError("bound input file digest differs")
    index, split, lock, weights = (
        _read_json(bound[name]) for name in ("source_index", "split", "model_lock", "model_weights")
    )
    _sealed(index, INDEX_SCHEMA)
    _sealed(split, "cyber_task_split_v2")
    if (
        lock.get("repo") != "Qwen/Qwen3.8-27B"
        or weights.get("repo") != lock.get("repo")
        or weights.get("revision") != lock.get("revision")
        or index.get("model")
        != {
            "repo": lock["repo"],
            "revision": lock["revision"],
            "root": config["tokenizer_root"],
            "model_lock_file_sha256": config["model_lock"]["sha256"],
            "model_weights_file_sha256": config["model_weights"]["sha256"],
            "initialization": "exact_fresh_base",
        }
    ):
        raise SelfTraceError("self-trace model is not the exact fresh Qwen base")
    interface = index.get("interface")
    chat = next(
        (
            row["sha256"]
            for row in lock["tokenizer"]["files"]
            if row["path"] == "chat_template.jinja"
        ),
        None,
    )
    if (
        not isinstance(interface, dict)
        or interface.get("backend") != "skyrl_direct"
        or interface.get("required_task_tools") != ["bash", "submit_report"]
        or not _sha(interface.get("required_task_tool_catalog_sha256"))
        or interface.get("compaction") != "disabled"
        or interface.get("runtime_chat_template_sha256")
        != "sha256:" + str(chat).removeprefix("sha256:")
        or interface.get("recorder_source_sha256")
        != file_sha256(Path(__file__).with_name("skyrl_episode.py"))
        or not _sha(index.get("review_receipt_sha256"))
        or not _sha(index.get("producer_plan_sha256"))
        or not _sha(index.get("route_certificate_sha256"))
        or not IMAGE.fullmatch(str(index.get("runtime_image", "")))
    ):
        raise SelfTraceError("direct collector interface evidence is incomplete")
    tasks = split.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(
            set(row) != {"task_key", "task_version_id", "split"} or row["split"] != "train"
            for row in tasks
        )
    ):
        raise SelfTraceError("self-trace split must be an exact train-only projection")
    train = {(row["task_key"], row["task_version_id"]) for row in tasks}
    if len(train) != len(tasks):
        raise SelfTraceError("self-trace split contains duplicate tasks")
    entries = index.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise SelfTraceError("no reviewed direct self traces are frozen")
    if len({entry.get("episode_id") for entry in entries}) != len(entries):
        raise SelfTraceError("reviewed self trace identities are duplicated")
    tokenizer, identity = local_tokenizer(lock, Path(config["tokenizer_root"]))
    if fleet.sha256(tokenizer.chat_template.encode()) != interface["runtime_chat_template_sha256"]:
        raise SelfTraceError("collector and SFT chat templates differ")
    candidates = [
        _episode(
            source_root,
            entry,
            index,
            train,
            max_length=config["max_length"],
            context_tokens=config["context_tokens"],
        )
        for entry in entries
    ]
    by_task = collections.defaultdict(list)
    for candidate in candidates:
        entry = candidate["entry"]
        by_task[(entry["task_key"], entry["task_version_id"])].append(candidate)
    selected, excluded = [], collections.Counter()
    for task, values in sorted(by_task.items()):
        values.sort(
            key=lambda item: hashlib.sha256(
                (config["source_seed"] + "\0" + item["entry"]["episode_id"]).encode()
            ).digest()
        )
        used_tokens = 0
        for candidate in values:
            if (
                len(
                    [
                        row
                        for row in selected
                        if (row["entry"]["task_key"], row["entry"]["task_version_id"]) == task
                    ]
                )
                >= config["max_episodes_per_task_version"]
            ):
                excluded["task_episode_cap"] += 1
            elif (
                used_tokens + candidate["supervised_tokens"]
                > config["max_supervised_tokens_per_task_version"]
            ):
                excluded["task_supervised_token_cap"] += 1
            else:
                selected.append(candidate)
                used_tokens += candidate["supervised_tokens"]
    if not selected:
        raise SelfTraceError("no direct self trace survives the frozen source caps")
    rows = [row for candidate in selected for row in candidate["rows"]]
    totals = {
        "source_sessions": len(selected),
        "assistant_responses": sum(item["assistant_responses"] for item in selected),
        "supervised_tokens": sum(item["supervised_tokens"] for item in selected),
        "submit_report_responses": sum(item["submit_report_responses"] for item in selected),
        "submit_report_tokens": sum(item["submit_report_tokens"] for item in selected),
        "source_total_assistant_responses": sum(
            row["source_assistant_count"] for item in selected for row in item["rows"][:1]
        ),
        "excluded_assistant_responses": sum(len(item["excluded"]) for item in selected),
    }
    if (
        totals["submit_report_tokens"] / totals["supervised_tokens"]
        > config["max_submit_token_share"]
        or totals["submit_report_responses"] / totals["assistant_responses"]
        > config["max_submit_response_share"]
    ):
        raise SelfTraceError("terminal report targets dominate the selected self corpus")
    spec = {
        "path": "train.parquet",
        "rows": len(rows),
        "task_keys": sorted({row["task_key"] for row in rows}),
        "format": DENSE_FORMAT,
        **{
            key: totals[key]
            for key in (
                "source_sessions",
                "assistant_responses",
                "supervised_tokens",
                "source_total_assistant_responses",
                "excluded_assistant_responses",
            )
        },
    }
    dense_rows(rows, spec, max_length=config["max_length"], vocab_size=len(tokenizer))
    stable = {path: file_sha256(path) for path in bound.values()}
    for candidate in candidates:
        stable.update(candidate["stable"])
    if any(file_sha256(path) != expected for path, expected in stable.items()):
        raise SelfTraceError("self-trace source changed during compilation")
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(parents=True, mode=0o700)
    train_path = output / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), train_path, compression="zstd")
    os.chmod(train_path, 0o600)
    if pq.read_table(train_path).to_pylist() != rows:
        raise SelfTraceError("private self-trace Parquet readback differs")
    spec["sha256"] = file_sha256(train_path)
    manifest = {
        "schema": CORPUS_SCHEMA,
        "source_sha256": index["sha256"],
        "split_sha256": split["sha256"],
        "tokenizer": identity,
        "files": {"train": spec},
        "train_models": ["Qwen/Qwen3.8-27B"],
        "max_length": config["max_length"],
        "context_tokens": config["context_tokens"],
        "dev_windows": 0,
        "validation_mode": "task_outcomes_only",
        "fleet_dev_protocol_sha256": config["fleet_dev_protocol_sha256"],
        "source_selection": {
            "source_index_sha256": index["sha256"],
            "review_receipt_sha256": index["review_receipt_sha256"],
            "producer_plan_sha256": index["producer_plan_sha256"],
            "route_certificate_sha256": index["route_certificate_sha256"],
            "runtime_image": index["runtime_image"],
            "required_task_tool_catalog_sha256": interface["required_task_tool_catalog_sha256"],
            "selected_episode_count": len(selected),
            "excluded_episode_count": len(candidates) - len(selected),
            "policy": {
                key: config[key]
                for key in (
                    "max_episodes_per_task_version",
                    "max_supervised_tokens_per_task_version",
                    "max_submit_token_share",
                    "max_submit_response_share",
                    "source_seed",
                )
            },
            "exclusions": dict(sorted(excluded.items())),
            "terminal_submission_token_share": totals["submit_report_tokens"]
            / totals["supervised_tokens"],
            "terminal_submission_response_share": totals["submit_report_responses"]
            / totals["assistant_responses"],
        },
        "whole_source_exclusions": {},
        "builder_sha256": {
            "self_trace_corpus.py": file_sha256(Path(__file__)),
            "rl_data.py": file_sha256(Path(__file__).with_name("rl_data.py")),
            "rl_episode.py": file_sha256(Path(__file__).with_name("rl_episode.py")),
            "sft_runtime.py": file_sha256(Path(__file__).with_name("sft_runtime.py")),
            "skyrl_episode.py": file_sha256(Path(__file__).with_name("skyrl_episode.py")),
        },
        "limitations": [
            "Only exact visible sampled assistant tokens are supervised; "
            "copied context and observations are masked.",
            "Structural tool-round coverage does not establish semantic exploit quality.",
            "Task-family-held-out evaluation may share applications with training.",
        ],
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    atomic_write_json(output / "manifest.json", manifest, private=True)
    atomic_write_json(output / "split.json", split, private=True)
    atomic_write_json(output / "source-index.json", index, private=True)
    return {
        "manifest_sha256": manifest["sha256"],
        "source_sessions": totals["source_sessions"],
        "rows": len(rows),
        "assistant_responses": totals["assistant_responses"],
        "supervised_tokens": totals["supervised_tokens"],
        "task_versions": len(
            {(item["entry"]["task_key"], item["entry"]["task_version_id"]) for item in selected}
        ),
        "submitted_jobs": 0,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_mapping(args.config)
        result = build(config, relative_to=args.config.parent)
    except Exception:
        print(json.dumps({"status": "rejected", "reason": "self_trace_corpus_not_completed"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
