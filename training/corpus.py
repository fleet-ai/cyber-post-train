"""Build complete-message SFT windows from evidence-bound successful sessions.

This pure planning boundary never slices token arrays or writes private data.
``count_tokens(messages, tools, target_index)`` must be the pinned trainer's
tokenizer/mask adapter and return (rendered tokens, supervised target tokens).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

TOOL_NAMES = {"fleet_bash", "fleet_submit_report"}
ALGORITHM = "complete_rounds_with_repeated_anchor_v1"


def sha256(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _rounds(messages: Any, aliases: Mapping[str, str], report_id: str) -> list[tuple[list[dict], int | None]]:
    if (not isinstance(messages, list) or len(messages) < 4
            or any(not isinstance(m, dict) for m in messages)
            or [m.get("role") for m in messages[:2]] != ["system", "user"]):
        raise ValueError("source lacks original system/task anchor")
    blocks: list[tuple[list[dict], int | None]] = []
    used_calls: set[str] = set()
    found_report = False
    index = 2
    while index < len(messages):
        message = copy.deepcopy(messages[index])
        role = message.get("role")
        if role == "user":
            blocks.append(([message], None))
            index += 1
            continue
        if role != "assistant":
            raise ValueError("orphan tool result or unsupported message role")
        assistant_index = index
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        call_ids = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise ValueError("malformed tool call")
            call_id = _text(call.get("id"), "tool call id")
            name = _text(call["function"].get("name"), "tool name")
            if call_id in used_calls or aliases.get(name) not in TOOL_NAMES:
                raise ValueError("duplicate call id or unknown tool")
            used_calls.add(call_id)
            call_ids.append(call_id)
            call["function"]["name"] = aliases[name]
            if call_id == report_id:
                if aliases[name] != "fleet_submit_report" or found_report:
                    raise ValueError("success evidence binds a non-report or duplicate call")
                found_report = True
        block = [message]
        pending = set(call_ids)
        for _ in call_ids:
            index += 1
            if index >= len(messages) or messages[index].get("role") != "tool":
                raise ValueError("tool call lacks an immediate result")
            result = copy.deepcopy(messages[index])
            call_id = result.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("tool result is duplicate or belongs to another round")
            pending.remove(call_id)
            block.append(result)
        blocks.append((block, assistant_index))
        index += 1
        if found_report:
            break  # No post-success text enters the training objective.
    if not found_report:
        raise ValueError("successful report and result absent from retained source")
    return blocks


def build_corpus(
    records: Iterable[dict], *, evidence: Mapping[str, dict], roster: Mapping[str, dict],
    tools: list[dict], aliases: Mapping[str, str], expected_sha256: Mapping[str, str],
    count_tokens: Callable[[list[dict], list[dict], int], tuple[int, int]],
    tokenizer_sha256: str, max_length: int, context_tokens: int,
    max_family_tokens: int, max_session_tokens: int, split: str = "train",
) -> tuple[list[dict], dict]:
    """Return private structured rows and a public-safe, self-digesting receipt.

    All input digests must be independently pinned by the caller. Whole sessions
    are admitted or excluded before packing, so a cap cannot truncate a success.
    """
    records = list(records)
    if split not in {"train", "dev"}:
        raise ValueError("only train and development teacher rows may be built")
    inputs = {"records": sorted(records, key=lambda r: r["session_id"]),
              "evidence": evidence, "roster": roster, "tools": tools, "aliases": aliases}
    if set(expected_sha256) != set(inputs):
        raise ValueError("all five input digests must be pinned")
    digests = {name: sha256(value) for name, value in inputs.items()}
    if digests != expected_sha256:
        raise ValueError("input digest differs from frozen selection")
    _text(tokenizer_sha256, "tokenizer digest")
    if not tokenizer_sha256.startswith("sha256:") or len(tokenizer_sha256) != 71:
        raise ValueError("exact tokenizer digest required")
    if (any(type(n) is not int or n <= 0 for n in (max_length, max_family_tokens,
                                                    max_session_tokens))
            or type(context_tokens) is not int or context_tokens < 0
            or context_tokens >= max_length):
        raise ValueError("invalid sequence or token cap")
    names = [t.get("function", {}).get("name") for t in tools]
    if len(names) != 2 or set(names) != TOOL_NAMES:
        raise ValueError("exact two model-facing Fleet tool definitions required")
    if any(target not in TOOL_NAMES for target in aliases.values()):
        raise ValueError("tool alias resolves outside the pinned catalog")
    family_splits: dict[str, str] = {}
    for identity, role in roster.items():
        _text(identity, "task version")
        family = _text(role.get("family_id"), "reviewed family")
        role_split = role.get("split")
        if (role_split not in {"train", "dev", "test"}
                or family_splits.setdefault(family, role_split) != role_split):
            raise ValueError("conflicting or missing reviewed family split")

    rows: list[dict] = []
    selected: list[str] = []
    family_tokens: dict[str, int] = {}
    excluded = {"other_split": 0, "session_cap": 0, "family_cap": 0}
    seen_sessions: set[str] = set()
    for record in inputs["records"]:
        session = _text(record.get("session_id"), "session id")
        version = _text(record.get("task_version_id"), "task version id")
        if session in seen_sessions or version not in roster:
            raise ValueError("duplicate session or unreviewed task version")
        seen_sessions.add(session)
        proof = evidence.get(session)
        if (not isinstance(proof, dict) or proof.get("verified_success") is not True
                or proof.get("source_sha256") != sha256(record)):
            raise ValueError("source lacks exact authoritative success evidence")
        report_id = _text(proof.get("report_call_id"), "successful report call id")
        family = roster[version]["family_id"]
        if roster[version]["split"] != split:
            excluded["other_split"] += 1
            continue
        source = record.get("messages")
        blocks = _rounds(source, aliases, report_id)
        anchor = copy.deepcopy(source[:2])
        prepared: list[dict] = []
        for block_index, (block, source_index) in enumerate(blocks):
            if source_index is None:
                continue
            target = block[0]
            base = anchor + [target]
            base_total, target_tokens = count_tokens(base, tools, 2)
            if (type(base_total) is not int or type(target_tokens) is not int
                    or not 0 < target_tokens <= base_total or base_total > max_length):
                raise ValueError("target cannot fit with its complete anchor")
            prior: list[dict] = []
            total = base_total
            for earlier, _ in reversed(blocks[:block_index]):
                trial = anchor + earlier + prior + [target]
                trial_total, trial_target = count_tokens(trial, tools, len(trial) - 1)
                if trial_target != target_tokens:
                    raise ValueError("target tokenization changes with copied history")
                if trial_total > max_length or trial_total - base_total > context_tokens:
                    break
                prior = earlier + prior
                total = trial_total
            messages = anchor + prior + [target]
            prepared.append({"messages": messages, "tools": copy.deepcopy(tools),
                             "target_message_index": len(messages) - 1,
                             "target_id": sha256([session, source_index]),
                             "task_version_id": version, "family_id": family,
                             "source_sha256": sha256(record),
                             "total_tokens": total, "supervised_tokens": target_tokens})
        amount = sum(row["supervised_tokens"] for row in prepared)
        if amount > max_session_tokens:
            excluded["session_cap"] += 1
        elif family_tokens.get(family, 0) + amount > max_family_tokens:
            excluded["family_cap"] += 1
        else:
            rows.extend(prepared)
            selected.append(session)
            family_tokens[family] = family_tokens.get(family, 0) + amount
    target_ids = [row["target_id"] for row in rows]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("assistant target selected more than once")
    receipt = {"algorithm": ALGORITHM, "split": split, "inputs_sha256": digests,
               "tokenizer_sha256": tokenizer_sha256, "max_length": max_length,
               "context_tokens": context_tokens, "max_family_tokens": max_family_tokens,
               "max_session_tokens": max_session_tokens, "selected_sessions": len(selected),
               "selected_session_ids_sha256": sha256(selected), "rows": len(rows),
               "supervised_tokens": sum(family_tokens.values()),
               "families": len(family_tokens), "excluded_sessions": excluded,
               "target_ids_sha256": sha256(target_ids), "rows_sha256": sha256(rows)}
    receipt["sha256"] = sha256(receipt)
    return rows, receipt


def materialize_pair(records: Iterable[dict], *, destination: Path, **kwargs: Any) -> dict:
    """Create private structured train/dev JSONL once; write receipt last.

    No trainer-ready token arrays are produced. A failed partial directory has
    no receipt and must be investigated, never overwritten or reused.
    """
    records = list(records)
    destination = Path(destination)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    partitions = {}
    for split in ("train", "dev"):
        rows, receipt = build_corpus(records, split=split, **kwargs)
        if not rows:
            raise ValueError("both partitions require verified successes; partial output has no receipt")
        name = f"{split}.jsonl"
        content_hash = hashlib.sha256()
        size = 0
        with os.fdopen(os.open(destination / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                              0o600), "wb") as file:
            for row in rows:
                line = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                file.write(line)
                content_hash.update(line)
                size += len(line)
            file.flush()
            os.fsync(file.fileno())
        partitions[split] = {"path": name, "bytes": size,
                             "sha256": "sha256:" + content_hash.hexdigest(),
                             "receipt": receipt}
    public = {"format": "structured_message_windows_v1", "partitions": partitions,
              "trainer_ready": False}
    public["sha256"] = sha256(public)
    with os.fdopen(os.open(destination / "RECEIPT.json",
                          os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as file:
        json.dump(public, file, sort_keys=True, separators=(",", ":"))
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    return public
