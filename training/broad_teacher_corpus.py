"""Build a train-only corpus from all verified teacher successes except holdouts.

This is the opportunistic counterpart to ``fleet_teacher_corpus``.  It keeps
the reviewed Fleet dev/final-test task families out across every historical
version, while permitting exact-version successes from the rest of the catalog.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
from pathlib import Path

from cyber_post_train.jobs import digest

from .corpus import local_tokenizer
from .dense import Excluded, _normalized_for_template, encode_record, native_helper, segment_record
from .fleet_teacher_corpus import _selection_row, _strip_finalize_suffix, _success
from .io import atomic_write_json, atomic_write_jsonl, digest_json, file_sha256, iter_jsonl
from .sft import _known, read_mapping
from .sft_runtime import DENSE_FORMAT, dense_rows

SCHEMA = "cyber_broad_teacher_corpus_request_v1"
EVIDENCE_SCHEMA = "fleet_broad_success_evidence_v1"
TOOL_ALIASES = {
    "fleet_environment__bash": "bash",
    "fleet_environment__fleet_environment__bash": "bash",
    "mcp__fleet_environment__bash": "bash",
    "fleet_environment__submit_report": "submit_report",
    "mcp__fleet_environment__submit_report": "submit_report",
    "mcp__fleet_environment__mcp__fleet_environment__submit_report": "submit_report",
}
FINAL_TOOL_ALIASES = {
    "fleet_environment__final_answer",
    "fleet_environment__complete",
    "fleet_environment__done",
    "fleet_environment__finalize",
    "fleet_environment__finish",
    "fleet_environment__submit_answer",
    "fleet_environment__submit_completion",
    "fleet_environment__submit_final",
    "fleet_environment__submit_final_answer",
}


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sealed(value: dict) -> bool:
    return value.get("sha256") == "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )


def _arrow_schema(pa):
    return pa.schema(
        [
            ("input_ids", pa.list_(pa.int64())),
            ("loss_mask", pa.list_(pa.int64())),
            ("token_count", pa.int64()),
            ("target_token_count", pa.int64()),
            ("task_key", pa.string()),
            ("window_id", pa.string()),
            ("segment_id", pa.string()),
            ("source_session_id", pa.string()),
            ("source_model", pa.string()),
            ("source_assistant_count", pa.int64()),
            ("eligible_assistant_indices", pa.list_(pa.int64())),
            (
                "excluded_assistant_targets",
                pa.list_(
                    pa.struct(
                        [
                            ("assistant_index", pa.int64()),
                            ("source_message_index", pa.int64()),
                            ("reason", pa.string()),
                        ]
                    )
                ),
            ),
            (
                "target_spans",
                pa.list_(
                    pa.struct(
                        [
                            ("assistant_index", pa.int64()),
                            ("source_message_index", pa.int64()),
                            ("token_start", pa.int64()),
                            ("token_end", pa.int64()),
                            ("source_target_sha256", pa.string()),
                        ]
                    )
                ),
            ),
            ("copied_context_assistant_indices", pa.list_(pa.int64())),
            ("context_start_message_index", pa.int64()),
            ("split", pa.string()),
            ("source_harness_mode", pa.string()),
            ("source_harness_sha256", pa.string()),
            ("source_acceptance_sha256", pa.string()),
            ("source_trace_sha256", pa.string()),
            ("source_normalized_record_sha256", pa.string()),
            ("source_task_version_id", pa.string()),
            ("source_transform_sha256", pa.string()),
        ]
    )


def _load_evidence(path: Path) -> dict[str, dict]:
    result = {}
    for item in iter_jsonl(path):
        sid = item.get("session_id")
        if (
            item.get("schema") != EVIDENCE_SCHEMA
            or not _sealed(item)
            or not isinstance(sid, str)
            or not sid
            or sid in result
        ):
            raise ValueError("invalid or duplicate broad success evidence")
        result[sid] = item
    return result


def _canonical_tool(name: str, arguments: dict) -> tuple[str, dict, bool]:
    wrapped = False
    if name == "use_tool" and set(arguments) >= {"tool_name", "tool_input"}:
        inner, payload = arguments["tool_name"], arguments["tool_input"]
        if not isinstance(inner, str) or not inner or not isinstance(payload, dict):
            raise Excluded("invalid_use_tool_wrapper")
        name, arguments, wrapped = inner, payload, True
    name = TOOL_ALIASES.get(name, name)
    if name in FINAL_TOOL_ALIASES:
        name = "submit_final_answer"
    if name == "shell" and set(arguments) == {"command"}:
        name, arguments = "bash", {"script": arguments["command"]}
    return name, arguments, wrapped


def _broad_messages(record: dict) -> tuple[dict, list[dict], str | None]:
    """Preserve valid visible turns and normalize only exact tool aliases."""
    transformed = copy.deepcopy(record)
    raw = transformed.get("messages")
    if not isinstance(raw, list) or len(raw) < 3 or [m.get("role") for m in raw[:2]] != [
        "system",
        "user",
    ]:
        raise Excluded("missing_original_system_and_task_anchor")
    pending, seen, operations = set(), set(), []
    for message in raw:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise Excluded("unsupported_message_role")
        if role != "tool" and pending:
            raise Excluded("missing_tool_result_before_next_message")
        calls = message.get("tool_calls") or []
        if role != "assistant" and calls:
            raise Excluded("tool_call_outside_assistant")
        for call in calls:
            call_id = call.get("id")
            function = call.get("function")
            if (
                not isinstance(call_id, str)
                or not call_id
                or call_id in seen
                or not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
            ):
                raise Excluded("invalid_or_duplicate_tool_call")
            arguments = function.get("arguments")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (TypeError, ValueError):
                raise Excluded("invalid_tool_arguments") from None
            if not isinstance(arguments, dict):
                raise Excluded("invalid_tool_arguments")
            original = function["name"]
            name, arguments, wrapped = _canonical_tool(original, arguments)
            if wrapped or name != original:
                operations.append({"call_id": call_id, "from": original, "to": name})
            function["name"], function["arguments"] = name, arguments
            seen.add(call_id)
            pending.add(call_id)
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise Excluded("orphan_or_duplicate_tool_result")
            pending.remove(call_id)
    if pending:
        raise Excluded("missing_terminal_tool_result")
    if not any(message.get("role") == "assistant" for message in raw):
        raise Excluded("no_assistant_targets")
    visible = [_normalized_for_template(message) for message in raw]
    receipt = None
    if operations:
        receipt = digest_json(
            {
                "schema": "cyber_exact_tool_alias_normalization_v1",
                "input_record_sha256": digest_json(record),
                "operations": operations,
                "output_messages_sha256": digest_json(visible),
            }
        )
    transformed["messages"] = visible
    return transformed, visible, receipt


def build(config: dict, *, relative_to: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _known(
        config,
        {
            "schema",
            "normalized",
            "evidence",
            "study_split",
            "model_lock",
            "tokenizer_root",
            "native_helper",
            "teacher_models",
            "max_length",
            "context_tokens",
            "output",
        },
        "broad Fleet teacher corpus",
    )
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported broad Fleet teacher corpus request")
    for name in ("normalized", "evidence"):
        _known(config[name], {"path", "sha256"}, name)
    teachers = config.get("teacher_models")
    if (
        not isinstance(teachers, list)
        or not teachers
        or len(teachers) != len(set(teachers))
        or any(not isinstance(model, str) or not model for model in teachers)
    ):
        raise ValueError("explicit unique stronger-teacher model list required")
    maximum, context = config.get("max_length"), config.get("context_tokens")
    if (
        type(maximum) is not int
        or type(context) is not int
        or maximum != 262144
        or not 32768 <= context <= maximum
    ):
        raise ValueError("broad teacher corpus must preserve a 262k maximum and useful context")
    output = _path(relative_to, config["output"])
    if output.exists():
        raise FileExistsError("create-once corpus destination exists")
    sources = {
        name: _path(relative_to, config[name]["path"]) for name in ("normalized", "evidence")
    }
    for name, path in sources.items():
        if file_sha256(path) != "sha256:" + config[name]["sha256"].removeprefix("sha256:"):
            raise ValueError(f"{name} digest mismatch")
    split_path = _path(relative_to, config["study_split"])
    split = read_mapping(split_path)
    held_out = {
        row["task_key"]
        for row in split.get("tasks", [])
        if row.get("split") in {"dev", "final_test"}
    }
    if len(held_out) != 25:
        raise ValueError("reviewed Fleet holdout no longer contains 25 task families")
    evidence = _load_evidence(sources["evidence"])
    tokenizer, tokenizer_identity = local_tokenizer(
        read_mapping(_path(relative_to, config["model_lock"])),
        _path(relative_to, config["tokenizer_root"]),
    )
    helper = native_helper(_path(relative_to, config["native_helper"]))

    output.mkdir(parents=True, mode=0o700)
    data_path = output / "train.parquet"
    selection_path = output / "source-selection.private.jsonl"
    writer = pq.ParquetWriter(data_path, _arrow_schema(pa), compression="zstd")
    selections, trajectories, window_payloads = [], set(), set()
    exclusions, models, harnesses = (
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
    )
    task_keys, task_versions, seen_records = set(), set(), set()
    row_count = supervised = responses = source_total = excluded_responses = 0
    min_tokens, max_tokens, token_lengths = None, 0, []
    suffixes_removed = 0
    try:
        for original in iter_jsonl(sources["normalized"]):
            sid = original.get("record_id")
            if not isinstance(sid, str) or not sid or sid in seen_records:
                raise ValueError("missing or duplicate normalized session identity")
            seen_records.add(sid)
            proof = evidence.get(sid)
            if proof is None:
                raise ValueError("normalized record lacks exact evidence")
            if original.get("content_digest") != digest_json(
                {key: item for key, item in original.items() if key != "content_digest"}
            ) or proof["normalized_record_sha256"] != digest_json(original):
                raise ValueError("normalized record digest mismatch")
            lineage, source = original.get("lineage", {}), original.get("source", {})
            identity = (lineage.get("task_key"), lineage.get("eval_task_version_id"))
            if any(
                (
                    proof.get("task_key") != identity[0],
                    proof.get("task_version_id") != identity[1],
                    proof.get("session_id") != source.get("session_id"),
                    proof.get("model_id") != source.get("model"),
                    proof.get("routes", {}).get("transcript")
                    != "GET /v1/sessions/{session_id}/transcript",
                )
            ):
                raise ValueError("source/evidence provenance mismatch")
            if identity[0] in held_out:
                raise ValueError("held-out task family entered the broad export")
            if not _success(original, proof, set(teachers)):
                exclusions["not_selected_verified_teacher_success"] += 1
                continue
            try:
                transformed, messages, interface_sha = _broad_messages(original)
                transformed, suffix_sha, removed = _strip_finalize_suffix(transformed)
                transform_sha = (
                    digest_json([interface_sha, suffix_sha]) if interface_sha else suffix_sha
                )
                suffixes_removed += int(removed)
                messages = transformed["messages"]
                trajectory = digest_json(messages)
                if trajectory in trajectories:
                    exclusions["exact_trajectory_duplicate"] += 1
                    continue
                anchor, chunks = encode_record(messages, tokenizer, helper)
                packed = segment_record(
                    transformed, anchor, chunks, max_tokens=maximum, context_budget=context
                )
                selection = _selection_row(
                    transformed,
                    proof,
                    packed,
                    "task-key:" + str(identity[0]),
                    transform_sha,
                )
            except Excluded as exc:
                exclusions[exc.reason] += 1
                continue
            for row in packed:
                fingerprint = digest_json([row["input_ids"], row["loss_mask"]])
                if fingerprint in window_payloads:
                    raise ValueError("partial source removal would break target coverage")
                window_payloads.add(fingerprint)
                row.update(
                    {
                        "source_harness_mode": source["harness_mode"],
                        "source_harness_sha256": source["harness_sha256"],
                        "source_acceptance_sha256": proof["sha256"],
                        "source_trace_sha256": proof["transcript_sha256"],
                        "source_normalized_record_sha256": proof["normalized_record_sha256"],
                        "source_task_version_id": identity[1],
                        "source_transform_sha256": transform_sha,
                    }
                )
            local_counts = {
                "rows": len(packed),
                "task_keys": [identity[0]],
                "format": DENSE_FORMAT,
                "source_sessions": 1,
                "supervised_tokens": sum(row["target_token_count"] for row in packed),
                "assistant_responses": sum(len(row["target_spans"]) for row in packed),
                "source_total_assistant_responses": packed[0]["source_assistant_count"],
                "excluded_assistant_responses": len(packed[0]["excluded_assistant_targets"]),
            }
            dense_rows(packed, local_counts, max_length=maximum, vocab_size=len(tokenizer))
            writer.write_table(pa.Table.from_pylist(packed, schema=_arrow_schema(pa)))
            trajectories.add(trajectory)
            selections.append(selection)
            row_count += len(packed)
            supervised += local_counts["supervised_tokens"]
            responses += local_counts["assistant_responses"]
            source_total += local_counts["source_total_assistant_responses"]
            excluded_responses += local_counts["excluded_assistant_responses"]
            lengths = [row["token_count"] for row in packed]
            token_lengths.extend(lengths)
            min_tokens = min(lengths) if min_tokens is None else min(min_tokens, *lengths)
            max_tokens = max(max_tokens, *lengths)
            task_keys.add(identity[0])
            task_versions.add(identity)
            models[source["model"]] += 1
            harnesses[source["harness_mode"]] += 1
    finally:
        writer.close()
    if set(evidence) != seen_records or not selections or row_count == 0:
        raise ValueError("normalized records/evidence are not one-to-one or corpus is empty")
    os.chmod(data_path, 0o600)
    atomic_write_jsonl(selection_path, selections, private=True)
    token_lengths.sort()

    def percentile(p: float) -> int:
        return token_lengths[min(len(token_lengths) - 1, math.ceil(p * len(token_lengths)) - 1)]

    counts = {
        "rows": row_count,
        "task_keys": sorted(task_keys),
        "format": DENSE_FORMAT,
        "source_sessions": len(selections),
        "supervised_tokens": supervised,
        "assistant_responses": responses,
        "source_total_assistant_responses": source_total,
        "excluded_assistant_responses": excluded_responses,
    }
    manifest = {
        "schema": "cyber_dense_sft_corpus_v1",
        "source_sha256": file_sha256(selection_path),
        "split_sha256": split["sha256"],
        "tokenizer": tokenizer_identity,
        "files": {"train": {"path": data_path.name, "sha256": file_sha256(data_path), **counts}},
        "train_models": sorted(models),
        "max_length": maximum,
        "context_tokens": context,
        "dev_windows": 0,
        "validation_mode": "task_outcomes_only",
        "whole_source_exclusions": dict(sorted(exclusions.items())),
        "catalog_provenance": {
            "study_split_file_sha256": file_sha256(split_path),
            "held_out_task_families_excluded_across_all_versions": len(held_out),
            "normalized_records_file_sha256": file_sha256(sources["normalized"]),
            "success_evidence_file_sha256": file_sha256(sources["evidence"]),
            "source_selection_file_sha256": file_sha256(selection_path),
            "selected_task_keys": len(task_keys),
            "selected_task_versions": len(task_versions),
            "selected_source_sessions": len(selections),
            "teacher_model_sessions": dict(sorted(models.items())),
            "harness_mode_sessions": dict(sorted(harnesses.items())),
            "terminal_finalize_suffixes_removed": suffixes_removed,
            "transcript_route": "GET /v1/sessions/{session_id}/transcript",
        },
        "token_length_distribution": {
            "min": min_tokens,
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p99": percentile(0.99),
            "max": max_tokens,
        },
        "limitations": [
            "Visible assistant actions only; private reasoning is omitted.",
            "All sources are exact transcript-verified full successes, but task-level QA "
            "certification is not required.",
            "The reviewed Fleet dev and final-test task families are excluded across every "
            "version.",
        ],
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    atomic_write_json(output / "manifest.json", manifest, private=True)
    receipt = {
        "schema": "cyber_broad_teacher_corpus_receipt_v1",
        "manifest_file_sha256": file_sha256(output / "manifest.json"),
        "manifest_sha256": manifest["sha256"],
        "train_parquet_sha256": file_sha256(data_path),
        "source_selection_sha256": file_sha256(selection_path),
        "rows": row_count,
        "source_sessions": len(selections),
        "task_versions": len(task_versions),
        "supervised_tokens": supervised,
    }
    receipt["sha256"] = "sha256:" + digest(receipt)
    atomic_write_json(output / "RECEIPT.json", receipt, private=True)
    return {
        "output": str(output),
        "manifest_sha256": manifest["sha256"],
        "receipt_sha256": receipt["sha256"],
        "train": {
            "rows": row_count,
            "source_sessions": len(selections),
            "task_versions": len(task_versions),
            "supervised_tokens": supervised,
        },
        "token_length_distribution": manifest["token_length_distribution"],
        "whole_source_exclusions": dict(sorted(exclusions.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_mapping(args.config)
    result = build(config, relative_to=args.config.parent)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
