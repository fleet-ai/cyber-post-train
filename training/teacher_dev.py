"""Build a small, fixed teacher-CE DEV panel from accepted target-visible sessions.

Private messages stay on the preparation host. This never promotes provisional
source evidence; the independent target audit must already accept it.
"""

from __future__ import annotations

import copy
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import source, target_dense
from .dense_bridge import _digest, _file_sha, _legacy_digest
from .runtime import (BACKEND_SHA, MODEL, TOKENIZER_FILES, TOKENIZER_SHA,
                      load_pinned_tokenizer, pinned_tokenize)

SCHEMA = "qwen38_target_teacher_dev_v1"
FORMAT = "chat_messages_last_assistant_v2"
FROZEN_ROLES_FILE_SHA = "sha256:f961283dc8aeda872553f0702c556ccb39b83ad9a0baeb14123e93805f2ca5fc"


def _read_bound(root: Path, receipt: dict, key: str, name: str):
    path = root / name
    if path.is_symlink() or not path.is_file() or _file_sha(path) != receipt["files"][key]:
        raise ValueError("accepted source file differs")
    return path


def _target_indices(messages: list[dict]) -> list[int]:
    return [i for i, message in enumerate(messages[2:], 2)
            if message.get("role") == "assistant" and not any(
                call.get("function", {}).get("name") == "fleet_submit_report"
                for call in message.get("tool_calls") or [])]


def _window(messages: list[dict], index: int, tools: list, tokenizer, limit: int):
    # Keep the exact two anchors and the longest complete suffix of turns.
    starts = [2] + [i for i in range(3, index + 1) if messages[i]["role"] == "assistant"]
    for start in starts:
        selected = copy.deepcopy(messages[:2] + messages[start:index + 1])
        original = {"messages": selected, "tools": tools}
        ids, actions = pinned_tokenize(original, tokenizer)
        if len(ids) > limit:
            continue
        stored = copy.deepcopy(original)
        stored["tools"] = json.dumps(tools, sort_keys=True, separators=(",", ":"))
        for message in stored["messages"]:
            if message.get("tool_calls") is not None:
                message["tool_calls"] = json.dumps(message["tool_calls"], sort_keys=True,
                                                   separators=(",", ":"))
        if pinned_tokenize(stored, tokenizer) != (ids, actions) or not 0 < actions < len(ids):
            raise ValueError("serialized DEV row changes the native target mask")
        return stored, len(ids), actions, start
    raise ValueError("DEV target cannot fit without truncating its anchors or response")


def build(source_dir: Path, tokenizer_root: Path, output: Path, *, max_length: int,
          max_input_tokens: int) -> dict:
    """Create DEV Parquet and sealed aggregate evidence; no train/final rows."""
    source_dir, output = Path(source_dir), Path(output)
    if (source_dir.is_symlink() or not source_dir.is_dir() or not output.is_absolute()
            or output.exists() or output.is_symlink() or type(max_length) is not int
            or type(max_input_tokens) is not int or not 2 <= max_length <= 262144
            or max_input_tokens < 1):
        raise ValueError("unsafe DEV input, output, or budget")
    reviewed = target_dense.audit(source_dir)
    receipt_path = source_dir / "RECEIPT.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("accepted source receipt is absent")
    receipt = json.loads(receipt_path.read_text())
    if (reviewed.get("trainer_ready") is not True or receipt.get("training_ready") is not True
            or reviewed.get("source_receipt_sha256") != receipt.get("sha256")
            or reviewed.get("training_blocker") is not None
            or receipt.get("training_blocker") is not None
            or receipt.get("input_sha256", {}).get("roles") != FROZEN_ROLES_FILE_SHA):
        raise ValueError("live target/tool parity or frozen family roles are not accepted")
    roster = json.loads(_read_bound(source_dir, receipt, "roster", "family-roster.json").read_text())
    tools = json.loads(_read_bound(source_dir, receipt, "tools", "model-facing-tools.json").read_text())
    records = [json.loads(line) for line in _read_bound(
        source_dir, receipt, "dense_target_normalized", "dense-target-anchored.jsonl").read_text().splitlines()]
    if (not isinstance(roster, dict) or not isinstance(tools, list)
            or source.digest(tools, ascii=True) != receipt.get("model_facing_tools_sha256")):
        raise ValueError("accepted tool or role contract differs")
    roles, grouped, dev_count = {}, defaultdict(list), 0
    for role in roster.values():
        family, split = role["family_id"], role["split"]
        if split not in {"train", "dev", "test"} or roles.setdefault(family, split) != split:
            raise ValueError("family crosses train, DEV, or final")
    for record in records:
        version = record["lineage"]["eval_task_version_id"]
        role = roster.get(version)
        if (not role or record["lineage"]["task_key"] != role["task_key"]
                or role["split"] == "test"):
            raise ValueError("target record escapes the frozen split")
        if role["split"] == "dev":
            grouped[role["family_id"]].append(record)
            dev_count += 1
    if dev_count != receipt.get("retained_by_split", {}).get("dev") or not grouped:
        raise ValueError("accepted DEV source inventory differs")
    tokenizer = load_pinned_tokenizer(tokenizer_root)
    rows, selection, input_tokens = [], [], 0
    for family in sorted(grouped):
        ranked = sorted(grouped[family], key=lambda row: _digest(row["record_id"]))
        chosen = next((row for row in ranked if _target_indices(row["messages"])), None)
        if chosen is None:
            raise ValueError("DEV family has no nonterminal assistant target")
        candidates = _target_indices(chosen["messages"])
        for index in sorted({candidates[0], candidates[len(candidates) // 2], candidates[-1]}):
            value, length, actions, start = _window(chosen["messages"], index, tools,
                                                     tokenizer, max_length)
            input_tokens += length
            if input_tokens > max_input_tokens:
                raise ValueError("fixed DEV panel exceeds its input-token budget")
            identity = {"source": chosen["content_digest"], "target_index": index, "start": start}
            version = chosen["lineage"]["eval_task_version_id"]
            row = {**value, "task_key": chosen["lineage"]["task_key"],
                   "task_version_id": version, "family_id": family,
                   "window_id": _digest(identity), "source_session_sha256": _digest(chosen["record_id"]),
                   "source_target_sha256": _digest(chosen["messages"][index]),
                   "source_message_index": index, "token_count": length,
                   "supervised_tokens": actions}
            rows.append(row)
            selection.append(identity)
    output.mkdir(mode=0o700)
    parquet = output / "dev.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet, compression="zstd")
    os.chmod(parquet, 0o600)
    readback = pq.read_table(parquet).to_pylist()
    if len(readback) != len(rows) or any(
        pinned_tokenize(before, tokenizer) != pinned_tokenize(after, tokenizer)
        or before["window_id"] != after["window_id"]
        for before, after in zip(rows, readback, strict=True)
    ):
        raise ValueError("Parquet readback changes model-visible DEV bytes")
    entry = {"path": "dev.parquet", "sha256": _file_sha(parquet), "format": FORMAT,
             "rows": len(rows), "task_keys": sorted({r["task_key"] for r in rows}),
             "supervised_tokens": sum(r["supervised_tokens"] for r in rows)}
    manifest = {"schema": SCHEMA, "trainer_ready": True,
                "validation_mode": "teacher_cross_entropy", "files": {"dev": entry},
                "source_receipt_file_sha256": _file_sha(receipt_path),
                "source_audit_sha256": reviewed["sha256"],
                "family_roles_file_sha256": FROZEN_ROLES_FILE_SHA,
                "model_facing_tools_sha256": receipt["model_facing_tools_sha256"],
                "selection_sha256": _digest(selection), "families": len(grouped),
                "reserved_dev_families": sum(split == "dev" for split in roles.values()),
                "source_sessions": len({r["source_session_sha256"] for r in rows}),
                "input_tokens": input_tokens, "max_length": max_length,
                "tokenizer": {"repo": MODEL[0], "revision": MODEL[1], "sha256": TOKENIZER_SHA,
                              "files": [{"path": p, "sha256": d} for p, d in TOKENIZER_FILES.items()],
                              "backend_sha256": BACKEND_SHA}}
    manifest["sha256"] = _legacy_digest(manifest)
    source._write(output / "manifest.json", manifest)  # Last: absence means incomplete output.
    return {"manifest_sha256": manifest["sha256"], "rows": entry["rows"],
            "families": len(grouped), "supervised_tokens": entry["supervised_tokens"]}
