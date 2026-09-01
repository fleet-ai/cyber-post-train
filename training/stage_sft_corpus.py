"""Build a private Fleet SFT corpus shard from a sealed normalized export.

This is an operational bridge for clusters whose incremental ``trace-ingest``
cache has not yet materialized the source sessions.  It emits the exact raw
Parquet contract consumed by ``rl_rollout.sft_dataset``; it does not bypass the
trainer's task split, reward filter, or chat-template tokenization.

The generated files can contain challenge transcripts and must stay in ignored
private storage.  Only the converter, tests, and non-sensitive manifests belong
in git.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io import atomic_write_json, file_sha256, iter_jsonl

CORPUS_SCHEMA = "fleet_sft_corpus_stage_v1"
VALID_SPLITS = frozenset({"train", "dev", "test"})


class ChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: list[dict[str, Any]], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class SFTWindow:
    messages: list[dict[str, Any]]
    target_position: int
    token_count: int


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _task_key(row: Mapping[str, Any]) -> str:
    lineage = row.get("lineage")
    return str(lineage.get("task_key") or "") if isinstance(lineage, Mapping) else ""


def _eligible(row: Mapping[str, Any]) -> bool:
    eligibility = row.get("eligibility")
    outcome = row.get("outcome")
    return bool(
        isinstance(eligibility, Mapping)
        and eligibility.get("sft") is True
        and isinstance(outcome, Mapping)
        and outcome.get("infra_valid") is True
        and outcome.get("success") is True
        and float(outcome.get("score", 0)) >= 1.0
    )


def _normalized_for_template(message: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the trainer's Qwen-facing normalization for exact length checks."""
    normalized: dict[str, Any] = {
        "role": str(message.get("role") or ""),
        "content": _text(message.get("content")) or "",
    }
    calls = message.get("tool_calls")
    if calls:
        normalized_calls = []
        for raw_call in calls:
            call = dict(raw_call)
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                with contextlib.suppress(ValueError):
                    arguments = json.loads(arguments)
            function["arguments"] = arguments
            call["function"] = function
            normalized_calls.append(call)
        normalized["tool_calls"] = normalized_calls
    return normalized


def _chat_token_count(tokenizer: ChatTokenizer, messages: list[dict[str, Any]]) -> int:
    encoded = tokenizer.apply_chat_template(
        [_normalized_for_template(message) for message in messages],
        tokenize=True,
        add_generation_prompt=False,
    )
    # transformers 5 returns a BatchEncoding here; older releases return a list.
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return len(encoded)


def verify_tokenizer_lock(
    *,
    tokenizer: ChatTokenizer,
    tokenizer_repo: str,
    tokenizer_revision: str,
    model_lock_path: Path,
) -> dict[str, Any]:
    """Verify exact tokenizer files at an immutable revision and return safe identity."""
    from huggingface_hub import hf_hub_download

    lock = json.loads(model_lock_path.read_text())
    if lock.get("schema") != "huggingface_model_lock_v1":
        raise ValueError("tokenizer model lock has unsupported schema")
    if lock.get("repo") != tokenizer_repo or lock.get("revision") != tokenizer_revision:
        raise ValueError("tokenizer model lock does not match requested repo and revision")
    token_lock = lock.get("tokenizer")
    files = token_lock.get("files") if isinstance(token_lock, Mapping) else None
    if not isinstance(files, list) or not files:
        raise ValueError("tokenizer model lock has no tokenizer files")

    verified_files: list[dict[str, Any]] = []
    for index, row in enumerate(files):
        if not isinstance(row, Mapping):
            raise ValueError(f"tokenizer model lock file {index} is not an object")
        relative_path = row.get("path")
        expected = row.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise ValueError(f"tokenizer model lock file {index} is incomplete")
        local_path = Path(
            hf_hub_download(
                repo_id=tokenizer_repo,
                filename=relative_path,
                revision=tokenizer_revision,
            )
        )
        actual = file_sha256(local_path)
        if actual != f"sha256:{expected}":
            raise ValueError(f"tokenizer file digest mismatch: {relative_path}")
        verified_files.append({"path": relative_path, "sha256": actual})

    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer has no chat template")
    chat_template_sha256 = "sha256:" + hashlib.sha256(chat_template.encode()).hexdigest()
    locked_chat = next(
        (row["sha256"] for row in verified_files if row["path"] == "chat_template.jinja"),
        None,
    )
    if locked_chat != chat_template_sha256:
        raise ValueError("loaded tokenizer chat template does not match model lock")

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        raise ValueError("tokenizer has no serializable backend identity")
    backend_sha256 = "sha256:" + hashlib.sha256(backend.to_str().encode()).hexdigest()
    return {
        "repo": tokenizer_repo,
        "revision": tokenizer_revision,
        "model_lock_sha256": file_sha256(model_lock_path),
        "tokenizer_manifest_sha256": token_lock.get("manifest_sha256"),
        "verified_files": verified_files,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "chat_template_sha256": chat_template_sha256,
        "backend_tokenizer_sha256": backend_sha256,
    }


def _evenly_spaced_assistant_positions(
    messages: list[dict[str, Any]], targets_per_trajectory: int
) -> list[int]:
    assistants = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if not assistants:
        return []
    if targets_per_trajectory < 1:
        raise ValueError("targets_per_trajectory must be positive")
    if len(assistants) <= targets_per_trajectory:
        return assistants
    if targets_per_trajectory == 1:
        return [assistants[-1]]

    def is_submission(message: Mapping[str, Any]) -> bool:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or call.get("name") or "")
            if name in {"submit_report", "submit_patch"}:
                return True
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                with contextlib.suppress(ValueError):
                    arguments = json.loads(arguments)
            if name == "use_tool" and isinstance(arguments, Mapping):
                wrapped = str(arguments.get("tool_name") or "")
                if wrapped.endswith(("__submit_report", "__submit_patch")):
                    return True
        return False

    selected = {assistants[0], assistants[-1]}
    submission_positions = [index for index in assistants if is_submission(messages[index])]
    if submission_positions and len(selected) < targets_per_trajectory:
        selected.add(submission_positions[-1])
    tool_positions = [index for index in assistants if messages[index].get("tool_calls")]
    if tool_positions and len(selected) < targets_per_trajectory:
        selected.add(tool_positions[-1])

    ordinal = {position: index for index, position in enumerate(assistants)}
    while len(selected) < targets_per_trajectory:
        remaining = [position for position in assistants if position not in selected]
        # Farthest-point sampling maximizes trajectory coverage around the semantic anchors.
        chosen = max(
            remaining,
            key=lambda position: (
                min(abs(ordinal[position] - ordinal[current]) for current in selected),
                -ordinal[position],
            ),
        )
        selected.add(chosen)
    return sorted(selected)


def build_sft_windows(
    messages: list[dict[str, Any]],
    *,
    tokenizer: ChatTokenizer,
    max_tokens: int,
    targets_per_trajectory: int,
) -> tuple[list[SFTWindow], int]:
    """Build exact-token, assistant-ending windows; return windows and oversized targets.

    Every window retains the original system/task instruction and maximizes recent contiguous
    context. Only its final assistant turn is intended as a loss target. Evenly spaced targets
    cover trajectory progress, and the successful final assistant is always selected.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    canonical = [dict(message) for message in messages]
    first_user = next(
        (index for index, message in enumerate(canonical) if message.get("role") == "user"), None
    )
    if first_user is None:
        raise ValueError("windowed SFT requires a user task instruction")
    anchor_end = first_user + 1
    anchor = canonical[:anchor_end]
    windows: list[SFTWindow] = []
    oversized = 0
    for target in _evenly_spaced_assistant_positions(canonical, targets_per_trajectory):
        if target < anchor_end:
            continue
        # Start only at a user/assistant boundary; never orphan a tool observation. Search for the
        # earliest fitting boundary to maximize contiguous recent context.
        starts = [
            index
            for index in range(anchor_end, target + 1)
            if canonical[index].get("role") in {"user", "assistant"}
        ]
        if not starts:
            starts = [target]

        # Length decreases monotonically as old turns are removed. Binary search avoids repeatedly
        # tokenizing long prefixes for every possible message boundary.
        low, high = 0, len(starts) - 1
        best: tuple[list[dict[str, Any]], int] | None = None
        while low <= high:
            middle = (low + high) // 2
            current = anchor + canonical[starts[middle] : target + 1]
            token_count = _chat_token_count(tokenizer, current)
            if token_count <= max_tokens:
                best = (current, token_count)
                high = middle - 1
            else:
                low = middle + 1
        if best is None:
            oversized += 1
            continue
        windows.append(SFTWindow(best[0], target, best[1]))
    return windows, oversized


def build_stage(
    trajectories: Iterable[dict[str, Any]],
    output_root: Path,
    *,
    team_id: str,
    source_sha256: str,
    tokenizer: ChatTokenizer | None = None,
    window_max_tokens: int = 0,
    targets_per_trajectory: int = 5,
    artifact_stem: str = "chris-cyber-fleet-a62dd51f",
    corpus_job_id: str | None = None,
    include_splits: frozenset[str] | None = None,
    tokenizer_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write corpus and score shards, returning a non-secret integrity manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus_schema = pa.schema(
        [
            ("session_id", pa.string()),
            ("message_id", pa.string()),
            ("position", pa.int32()),
            ("task_key", pa.string()),
            ("model", pa.string()),
            ("job_id", pa.string()),
            ("attempt", pa.int32()),
            ("status", pa.string()),
            ("team_id", pa.string()),
            ("role", pa.string()),
            ("content", pa.string()),
            ("tool_calls", pa.string()),
            ("tool_call_id", pa.string()),
            ("tokens", pa.int64()),
            ("generated_tokens", pa.int64()),
            ("session_created_at", pa.string()),
            ("message_created_at", pa.string()),
        ]
    )
    score_schema = pa.schema([("session_id", pa.string()), ("score", pa.float64())])

    rows_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scores: list[dict[str, Any]] = []
    source_sessions: set[str] = set()
    emitted_sessions: set[str] = set()
    tasks_by_split: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = defaultdict(int)
    window_counts: dict[str, int] = defaultdict(int)
    window_token_counts: list[int] = []
    oversized_targets = 0
    source_job_ids: set[str] = set()
    excluded_rows_by_split: dict[str, int] = defaultdict(int)

    if (tokenizer is None) != (window_max_tokens == 0):
        raise ValueError("tokenizer and window_max_tokens must be provided together")
    if tokenizer is not None and tokenizer_identity is None:
        raise ValueError("windowed SFT requires a verified tokenizer_identity")
    if tokenizer is None and tokenizer_identity is not None:
        raise ValueError("tokenizer_identity requires a tokenizer")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", artifact_stem):
        raise ValueError("artifact_stem must be a safe filename stem")
    if include_splits is not None:
        if not include_splits:
            raise ValueError("include_splits must not be empty")
        unknown_splits = include_splits - VALID_SPLITS
        if unknown_splits:
            raise ValueError(f"unsupported include_splits: {sorted(unknown_splits)}")

    for row in trajectories:
        if row.get("schema") != "fleet_cyber_trajectory_v1":
            continue
        split = str(row.get("split") or "")
        if include_splits is not None and split not in include_splits:
            excluded_rows_by_split[split or "<missing>"] += 1
            continue
        if not _eligible(row):
            continue
        session_id = str(row.get("record_id") or "")
        task_key = _task_key(row)
        source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
        source_job_id = str(source.get("job_id") or "")
        if source_job_id:
            source_job_ids.add(source_job_id)
        environment = (
            row.get("environment") if isinstance(row.get("environment"), Mapping) else {}
        )
        messages = row.get("messages")
        env_key = str(environment.get("env_key") or "")
        if not session_id or not task_key or not env_key or not isinstance(messages, list):
            raise ValueError(f"malformed eligible trajectory {session_id or '<missing-id>'}")
        if session_id in source_sessions:
            raise ValueError(f"duplicate session id {session_id}")
        source_sessions.add(session_id)
        tasks_by_split[split].add(task_key)
        split_counts[split] += 1
        clean_messages: list[dict[str, Any]] = []
        for position, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"{session_id}: message {position} is not an object")
            role = str(message.get("role") or "")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"{session_id}: unsupported role {role!r}")
            clean_messages.append(dict(message))

        if tokenizer is None:
            staged_sessions = [(session_id, clean_messages)]
        else:
            windows, oversized = build_sft_windows(
                clean_messages,
                tokenizer=tokenizer,
                max_tokens=window_max_tokens,
                targets_per_trajectory=targets_per_trajectory,
            )
            oversized_targets += oversized
            window_counts[split] += len(windows)
            window_token_counts.extend(window.token_count for window in windows)
            staged_sessions = [
                (f"{session_id}::sftw::{window.target_position}", window.messages)
                for window in windows
            ]

        for staged_session_id, staged_messages in staged_sessions:
            if staged_session_id in emitted_sessions:
                raise ValueError(f"duplicate staged session id {staged_session_id}")
            emitted_sessions.add(staged_session_id)
            scores.append({"session_id": staged_session_id, "score": 1.0})
            for position, message in enumerate(staged_messages):
                role = str(message.get("role") or "")
                tool_calls = message.get("tool_calls")
                rows_by_env[env_key].append({
                    "session_id": staged_session_id,
                    "message_id": f"{session_id}:{position}",
                    "position": position,
                    "task_key": task_key,
                    "model": str(source.get("model") or ""),
                    "job_id": corpus_job_id or source_job_id,
                    "attempt": 0,
                    "status": str(row.get("outcome", {}).get("status") or "completed"),
                    "team_id": team_id,
                    "role": role,
                    "content": _text(message.get("content")),
                    "tool_calls": (
                        json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
                        if tool_calls
                        else None
                    ),
                    "tool_call_id": _text(message.get("tool_call_id")),
                    "tokens": None,
                    "generated_tokens": None,
                    "session_created_at": "",
                    "message_created_at": "",
                })

    if not source_sessions:
        raise ValueError("no SFT-eligible successful trajectories")
    if tokenizer is not None and not emitted_sessions:
        raise ValueError("windowing produced no usable SFT sessions")
    split_names = sorted(tasks_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = tasks_by_split[left] & tasks_by_split[right]
            if overlap:
                raise ValueError(f"task leakage across {left}/{right}: {sorted(overlap)[:3]}")

    corpus_outputs: list[dict[str, Any]] = []
    for env_key, rows in sorted(rows_by_env.items()):
        path = output_root / "session-corpus" / f"env_key={env_key}" / "date=2026-08-28"
        path.mkdir(parents=True, exist_ok=True)
        output = path / f"{artifact_stem}.parquet"
        temporary = output.with_suffix(".parquet.partial")
        pq.write_table(pa.Table.from_pylist(rows, schema=corpus_schema), temporary)
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
        corpus_outputs.append(
            {
                "relative_path": str(output.relative_to(output_root)),
                "rows": len(rows),
                "sha256": file_sha256(output),
            }
        )

    scores_dir = output_root / "session-scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores_output = scores_dir / f"{artifact_stem}.parquet"
    scores_temporary = scores_output.with_suffix(".parquet.partial")
    pq.write_table(pa.Table.from_pylist(scores, schema=score_schema), scores_temporary)
    os.chmod(scores_temporary, 0o600)
    os.replace(scores_temporary, scores_output)

    manifest = {
        "schema": CORPUS_SCHEMA,
        "source_trajectories_sha256": source_sha256,
        "source_job_ids": sorted(source_job_ids),
        "corpus_job_id": corpus_job_id,
        "team_id": team_id,
        "split_filter": {
            "included": sorted(include_splits) if include_splits is not None else None,
            "excluded_source_rows": dict(sorted(excluded_rows_by_split.items())),
        },
        "tokenizer_identity": dict(tokenizer_identity) if tokenizer_identity is not None else None,
        "eligible_sessions": len(source_sessions),
        "emitted_sessions": len(emitted_sessions),
        "task_counts": {key: len(value) for key, value in sorted(tasks_by_split.items())},
        "session_counts": dict(sorted(split_counts.items())),
        "corpus": corpus_outputs,
        "scores": {
            "relative_path": str(scores_output.relative_to(output_root)),
            "rows": len(scores),
            "sha256": file_sha256(scores_output),
        },
    }
    if tokenizer is not None:
        sorted_tokens = sorted(window_token_counts)
        manifest["windowing"] = {
            "max_tokens": window_max_tokens,
            "targets_per_trajectory": targets_per_trajectory,
            "train_on_what": "last_assistant_message",
            "window_counts": dict(sorted(window_counts.items())),
            "oversized_targets_excluded": oversized_targets,
            "token_count_min": sorted_tokens[0],
            "token_count_median": sorted_tokens[len(sorted_tokens) // 2],
            "token_count_max": sorted_tokens[-1],
        }
    atomic_write_json(output_root / "stage-manifest.json", manifest, private=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--tokenizer", help="HF model id or local tokenizer path")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument(
        "--tokenizer-model-lock",
        type=Path,
        help="model lock whose exact tokenizer files are verified before windowing",
    )
    parser.add_argument("--window-max-tokens", type=int, default=0)
    parser.add_argument("--targets-per-trajectory", type=int, default=5)
    parser.add_argument("--artifact-stem", default="chris-cyber-fleet-a62dd51f")
    parser.add_argument("--corpus-job-id")
    parser.add_argument(
        "--include-split",
        action="append",
        choices=sorted(VALID_SPLITS),
        dest="include_splits",
        help="repeat to admit only named splits before eligibility or tokenization",
    )
    args = parser.parse_args()
    tokenizer = None
    tokenizer_identity = None
    if args.tokenizer:
        if not args.tokenizer_revision or args.tokenizer_model_lock is None:
            parser.error(
                "--tokenizer requires --tokenizer-revision and --tokenizer-model-lock"
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            revision=args.tokenizer_revision,
            trust_remote_code=True,
        )
        tokenizer_identity = verify_tokenizer_lock(
            tokenizer=tokenizer,
            tokenizer_repo=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            model_lock_path=args.tokenizer_model_lock,
        )
    elif args.tokenizer_revision or args.tokenizer_model_lock is not None:
        parser.error("--tokenizer-revision/--tokenizer-model-lock require --tokenizer")
    manifest = build_stage(
        iter_jsonl(args.trajectories),
        args.output_root,
        team_id=args.team_id,
        source_sha256=file_sha256(args.trajectories),
        tokenizer=tokenizer,
        window_max_tokens=args.window_max_tokens,
        targets_per_trajectory=args.targets_per_trajectory,
        artifact_stem=args.artifact_stem,
        corpus_job_id=args.corpus_job_id,
        include_splits=(frozenset(args.include_splits) if args.include_splits else None),
        tokenizer_identity=tokenizer_identity,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
