"""Normalize Fleet transcript exports into trainer-neutral trajectories."""

from __future__ import annotations

import collections
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .io import atomic_write_json, atomic_write_jsonl, digest_json, file_sha256, iter_jsonl
from .lineage import extract_lineage, leakage_group
from .rewards import compute_reward
from .secrets import redact_exact
from .splits import apply_splits

TERMINAL = {"completed", "succeeded", "success"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    messages: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, Mapping) or message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            continue
        clean = {
            key: item
            for key, item in message.items()
            if key in {"role", "content", "thinking", "tool_calls", "tool_call_id", "name"}
            and item is not None
        }
        messages.append(clean)
    return messages


def _initial_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") in {"system", "user"}:
            out.append(message)
        else:
            break
    return out


def normalize_export_row(row: Mapping[str, Any], *, secrets: Iterable[str] = ()) -> dict[str, Any]:
    if row.get("export_schema") != "fleet_session_export_v1":
        raise ValueError("unsupported or missing export_schema")
    source = _mapping(row.get("source"))
    session = _mapping(row.get("session"))
    envelope = _mapping(row.get("transcript_envelope"))
    task = _mapping(envelope.get("task"))
    # The legacy transcript payload omits several immutable task bindings that
    # are present in the authoritative job roster. Prefer those exact roster
    # values while retaining the prompt and other transcript-local fields.
    task.update(_mapping(source.get("roster_task_binding")))
    instance = _mapping(envelope.get("instance"))
    verifier = _mapping(envelope.get("verifier_execution"))
    messages = redact_exact(_messages(envelope.get("transcript")), secrets)
    lineage = extract_lineage(task, instance)

    status = str(session.get("status") or "").lower()
    score = verifier.get("score", session.get("score"))
    # Fleet's transcript route uses verifier_execution.success to mean that the
    # verifier process itself ran successfully. It is not the task outcome and
    # is true even for score 0. Only the deterministic numeric task score may
    # admit a trajectory to SFT.
    success = isinstance(score, (int, float)) and float(score) >= 1.0
    infra_valid = status in TERMINAL and bool(messages) and bool(task.get("key"))
    reward = compute_reward(verifier_score=score, infra_valid=infra_valid)
    has_assistant = any(message.get("role") == "assistant" for message in messages)
    has_tool = any(message.get("role") == "tool" for message in messages)

    record = {
        "schema": "fleet_cyber_trajectory_v1",
        "record_id": str(source.get("session_id") or session.get("session_id") or ""),
        "source": {
            "job_id": source.get("job_id"),
            "session_id": source.get("session_id"),
            "model": session.get("model"),
            "export_route": "/v1/sessions/{session_id}/transcript",
        },
        "lineage": lineage,
        "leakage_group": leakage_group(lineage),
        "prompt_messages": _initial_prompt(messages),
        "messages": messages,
        "environment": {
            "env_key": instance.get("env_key") or task.get("env_id"),
            "version": instance.get("version"),
            "data_key": instance.get("data_key") or task.get("data_id"),
            "data_version": instance.get("data_version") or task.get("data_version"),
        },
        "outcome": {
            "status": status,
            "score": score if isinstance(score, (int, float)) else None,
            "success": success,
            "infra_valid": infra_valid,
            "reward": reward,
        },
        "eligibility": {
            "sft": infra_valid and success and has_assistant and has_tool,
            "preference": infra_valid and has_assistant,
            "online_rl_prompt": infra_valid and bool(_initial_prompt(messages)),
            "reasons": [] if infra_valid else ["nonterminal_or_malformed_session"],
        },
    }
    digest_input = {key: value for key, value in record.items() if key != "content_digest"}
    record["content_digest"] = digest_json(digest_input)
    return record


def build_datasets(
    raw_path: Path, output_dir: Path, *, secrets: Iterable[str] = ()
) -> dict[str, Any]:
    normalized = [normalize_export_row(row, secrets=secrets) for row in iter_jsonl(raw_path)]
    normalized.sort(key=lambda row: (str(row["lineage"]["lineage_key"]), str(row["record_id"])))
    apply_splits(normalized)
    if len({row["record_id"] for row in normalized}) != len(normalized):
        raise ValueError("duplicate session record_id in export")

    sft_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    for row in normalized:
        if not row["eligibility"]["sft"]:
            continue
        sft_by_split[row["split"]].append(
            {
                "id": row["record_id"],
                "messages": row["messages"],
                "weight": row["outcome"]["reward"]["total"],
                "metadata": {
                    "lineage": row["lineage"],
                    "source": row["source"],
                    "split_unit": row["split_unit"],
                },
            }
        )
    rl_prompts_by_group: dict[tuple[str, str], dict[str, Any]] = {}
    for row in normalized:
        if not row["eligibility"]["online_rl_prompt"]:
            continue
        key = (row["leakage_group"], row["lineage"]["prompt_sha256"])
        rl_prompts_by_group.setdefault(
            key,
            {
                "id": row["lineage"]["lineage_key"],
                "messages": row["prompt_messages"],
                "task": {
                    "task_key": row["lineage"]["task_key"],
                    "task_version": row["lineage"]["task_version"],
                    "environment": row["environment"],
                },
                "metadata": {"leakage_group": row["leakage_group"], "lineage": row["lineage"]},
            },
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in normalized:
        if row["eligibility"]["preference"]:
            grouped[(row["leakage_group"], row["lineage"]["prompt_sha256"])].append(row)
    preferences: list[dict[str, Any]] = []
    for rows in grouped.values():
        chosen = max(rows, key=lambda row: float(row["outcome"]["reward"]["total"]))
        rejected = min(rows, key=lambda row: float(row["outcome"]["reward"]["total"]))
        if chosen["outcome"]["reward"]["total"] <= rejected["outcome"]["reward"]["total"]:
            continue
        preferences.append(
            {
                "id": f"{chosen['record_id']}::{rejected['record_id']}",
                "prompt": chosen["prompt_messages"],
                "chosen": chosen["messages"][len(chosen["prompt_messages"]) :],
                "rejected": rejected["messages"][len(rejected["prompt_messages"]) :],
                "chosen_reward": chosen["outcome"]["reward"],
                "rejected_reward": rejected["outcome"]["reward"],
                "metadata": {
                    "lineage": chosen["lineage"],
                    "leakage_group": chosen["leakage_group"],
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "trajectories": (output_dir / "trajectories.jsonl", normalized),
        "sft_train": (output_dir / "sft_train.jsonl", sft_by_split["train"]),
        "sft_dev": (output_dir / "sft_dev.jsonl", sft_by_split["dev"]),
        "sft_test": (output_dir / "sft_test.jsonl", sft_by_split["test"]),
        "preferences": (output_dir / "preferences.jsonl", preferences),
        "rl_prompts": (output_dir / "rl_prompts.jsonl", list(rl_prompts_by_group.values())),
    }
    files: dict[str, Any] = {}
    for name, (path, rows) in outputs.items():
        count = atomic_write_jsonl(path, rows, private=True)
        files[name] = {"path": path.name, "rows": count, "sha256": file_sha256(path)}
    manifest = {
        "schema": "fleet_cyber_dataset_manifest_v1",
        "source": {"path": str(raw_path), "sha256": file_sha256(raw_path)},
        "policy": {
            "fleet_scope": "all_eligible_exported_sessions",
            "sft": "execution-verified successes, lineage-safe train/dev/test",
            "split_unit": "application + vulnerability family + task lineage",
            "split_seed": "fleet-cyber-split-v1",
            "preferences": "best-vs-worst within exact prompt and leakage group",
            "online_rl": "one prompt per exact prompt and leakage group",
            "grader_output_included": False,
        },
        "counts": {
            "sessions": len(normalized),
            "successful": sum(bool(row["outcome"]["success"]) for row in normalized),
            "failed": sum(not bool(row["outcome"]["success"]) for row in normalized),
            "applications": dict(
                sorted(
                    collections.Counter(row["lineage"]["application"] for row in normalized).items()
                )
            ),
            "lineage_quality": dict(
                sorted(
                    collections.Counter(
                        row["lineage"]["lineage_quality"] for row in normalized
                    ).items()
                )
            ),
            "splits": dict(sorted(collections.Counter(row["split"] for row in normalized).items())),
            "sessions_missing_exact_bindings": sum(
                bool(row["lineage"]["missing_exact_bindings"]) for row in normalized
            ),
        },
        "files": files,
    }
    manifest["manifest_digest"] = digest_json(manifest)
    atomic_write_json(output_dir / "manifest.json", manifest, private=True)
    return manifest
