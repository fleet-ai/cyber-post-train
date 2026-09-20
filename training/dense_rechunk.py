"""Re-window a sealed dense SFT corpus without changing its supervised targets.

The source corpus is already tokenized and independently qualified.  This
transform only changes how much contiguous history surrounds each assistant
target.  Every source target must appear exactly once in the successor loss
mask; repeated history is always masked.  The transform is streaming so a
long-context corpus does not have to fit in host memory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from cyber_post_train.jobs import digest

from .io import atomic_write_json, file_sha256
from .post_sft_staging import _rename_noreplace
from .sft import _known, read_mapping
from .sft_runtime import DENSE_FORMAT

REQUEST_SCHEMA = "cyber_dense_sft_rechunk_request_v1"
RECEIPT_SCHEMA = "cyber_dense_sft_rechunk_receipt_v1"
ALGORITHM = "contiguous_target_groups_with_masked_history_v1"
MINIMUM_SUPERVISED_TOKENS = 20_000_000


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sealed(value: dict, schema: str) -> None:
    expected = "sha256:" + digest({key: item for key, item in value.items() if key != "sha256"})
    if value.get("schema") != schema or value.get("sha256") != expected:
        raise ValueError(f"invalid {schema} object")


def _target_digest(ids: list[int]) -> str:
    return hashlib.sha256(__import__("json").dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _source_row(row: dict, *, maximum: int) -> None:
    ids, mask, spans = row.get("input_ids"), row.get("loss_mask"), row.get("target_spans")
    if (
        not isinstance(ids, list)
        or not ids
        or not isinstance(mask, list)
        or len(mask) != len(ids)
        or row.get("token_count") != len(ids)
        or row.get("target_token_count") != sum(mask)
        or len(ids) > maximum
        or not isinstance(spans, list)
        or not spans
    ):
        raise ValueError("source dense row shape differs")
    expected = [0] * len(ids)
    previous = (0, -1, -1)
    for span in spans:
        start, end = span.get("token_start"), span.get("token_end")
        assistant = span.get("assistant_index")
        message = span.get("source_message_index")
        if (
            any(type(value) is not int for value in (start, end, assistant, message))
            or not 1 <= start < end <= len(ids)
            or start < previous[0]
            or assistant <= previous[1]
            or message <= previous[2]
            or span.get("source_target_sha256", "").removeprefix("sha256:")
            != _target_digest(ids[start:end])
        ):
            raise ValueError("source target span differs")
        expected[start:end] = [1] * (end - start)
        previous = end, assistant, message
    if expected != mask:
        raise ValueError("source loss mask differs from declared targets")


def rechunk_row(row: dict, *, max_length: int, context_tokens: int) -> list[dict]:
    """Return contiguous windows that cover every source target once."""
    ids = row["input_ids"]
    spans = row["target_spans"]
    results: list[dict] = []
    first = 0
    while first < len(spans):
        start = max(0, spans[first]["token_start"] - context_tokens)
        last = first
        while last + 1 < len(spans) and spans[last + 1]["token_end"] - start <= max_length:
            last += 1
        if spans[last]["token_end"] - start > max_length:
            # One assistant response is larger than the complete requested
            # sequence.  Splitting its label would weaken provenance, so stop.
            raise ValueError("one assistant target exceeds the requested context ceiling")
        end = spans[last]["token_end"]
        selected = spans[first : last + 1]
        adjusted = []
        loss_mask = [0] * (end - start)
        for span in selected:
            item = dict(span)
            item["token_start"] -= start
            item["token_end"] -= start
            adjusted.append(item)
            loss_mask[item["token_start"] : item["token_end"]] = [1] * (
                item["token_end"] - item["token_start"]
            )
        targets = {span["assistant_index"] for span in selected}
        copied = {span["assistant_index"] for span in spans[:first] if span["token_end"] > start}
        copied.update(row.get("copied_context_assistant_indices") or [])
        copied.difference_update(targets)
        identity = {
            "parent": row["window_id"],
            "start": start,
            "end": end,
            "targets": sorted(targets),
            "algorithm": ALGORITHM,
        }
        window_id = f"{row['window_id']}::rechunk::{digest(identity)[:24]}"
        item = dict(row)
        item.update(
            {
                "input_ids": ids[start:end],
                "loss_mask": loss_mask,
                "token_count": end - start,
                "target_token_count": sum(loss_mask),
                "window_id": window_id,
                "segment_id": window_id,
                "target_spans": adjusted,
                "copied_context_assistant_indices": sorted(copied),
                "context_start_message_index": (
                    row["context_start_message_index"]
                    if start == 0 or first == 0
                    else spans[first - 1]["source_message_index"]
                ),
            }
        )
        results.append(item)
        first = last + 1
    return results


def build(config: dict, *, relative_to: Path) -> dict:
    """Stream a sealed parent Parquet into a create-once smaller-context corpus."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _known(
        config,
        {
            "schema",
            "source",
            "destination",
            "max_length",
            "context_tokens",
            "minimum_supervised_tokens",
            "sha256",
        },
        "dense rechunk request",
    )
    _sealed(config, REQUEST_SCHEMA)
    source_config = config.get("source")
    destination = config.get("destination")
    _known(
        source_config,
        {
            "manifest_path",
            "manifest_file_sha256",
            "manifest_sha256",
            "data_root",
            "train_parquet_sha256",
        },
        "rechunk source",
    )
    _known(destination, {"output_root", "create_once"}, "rechunk destination")
    if destination.get("create_once") is not True:
        raise ValueError("rechunk destination must be create-once")
    max_length = config.get("max_length")
    context_tokens = config.get("context_tokens")
    minimum = config.get("minimum_supervised_tokens")
    if (
        type(max_length) is not int
        or type(context_tokens) is not int
        or type(minimum) is not int
        or max_length < 2
        or not 1 <= context_tokens < max_length
        or minimum < MINIMUM_SUPERVISED_TOKENS
    ):
        raise ValueError("invalid dense rechunk bounds")

    manifest_path = _path(relative_to, source_config["manifest_path"])
    if file_sha256(manifest_path) != source_config["manifest_file_sha256"]:
        raise ValueError("source manifest file digest mismatch")
    manifest = read_mapping(manifest_path)
    _sealed(manifest, "cyber_dense_sft_corpus_v1")
    if manifest["sha256"] != source_config["manifest_sha256"]:
        raise ValueError("source manifest logical digest mismatch")
    if (
        set(manifest.get("files", {})) != {"train"}
        or manifest.get("validation_mode") != "task_outcomes_only"
    ):
        raise ValueError("source must be one train-only task-outcome corpus")
    train = manifest["files"]["train"]
    if (
        train.get("format") != DENSE_FORMAT
        or train.get("sha256") != source_config["train_parquet_sha256"]
    ):
        raise ValueError("source train binding differs")
    if manifest.get("max_length", 0) <= max_length:
        raise ValueError("successor context ceiling must be smaller than the source")
    source_path = Path(source_config["data_root"]) / train["path"]
    before = file_sha256(source_path)
    if before != train["sha256"]:
        raise ValueError("source Parquet digest mismatch")

    output = _path(relative_to, destination["output_root"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("create-once rechunk destination exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    os.chmod(partial, 0o700)
    parquet_path = partial / "train.parquet"
    writer = None
    source_rows = output_rows = supervised = responses = 0
    sessions: dict[str, dict] = {}
    window_ids: set[str] = set()
    task_keys: set[str] = set()
    task_versions: set[str] = set()
    try:
        parquet = pq.ParquetFile(source_path)
        for batch in parquet.iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            _source_row(row, maximum=manifest["max_length"])
            source_rows += 1
            sid = row["source_session_id"]
            metadata = {
                "task_key": row["task_key"],
                "source_model": row["source_model"],
                "source_assistant_count": row["source_assistant_count"],
                "eligible_assistant_indices": tuple(row["eligible_assistant_indices"]),
                "excluded_assistant_targets": tuple(
                    (item["assistant_index"], item["source_message_index"], item["reason"])
                    for item in row["excluded_assistant_targets"]
                ),
            }
            source_state = sessions.setdefault(sid, {"metadata": metadata, "targets": set()})
            if source_state["metadata"] != metadata:
                raise ValueError("source metadata changes across parent rows")
            for item in rechunk_row(row, max_length=max_length, context_tokens=context_tokens):
                if item["window_id"] in window_ids:
                    raise ValueError("rechunk window identity collision")
                window_ids.add(item["window_id"])
                targeted = {span["assistant_index"] for span in item["target_spans"]}
                if source_state["targets"] & targeted:
                    raise ValueError("one source assistant target was repeated")
                source_state["targets"].update(targeted)
                supervised += item["target_token_count"]
                responses += len(item["target_spans"])
                output_rows += 1
                task_keys.add(item["task_key"])
                version = item.get("source_task_version_id")
                if not isinstance(version, str) or not version:
                    raise ValueError("source row lacks exact task-version identity")
                task_versions.add(version)
                table = pa.Table.from_pylist([item], schema=parquet.schema_arrow)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
                writer.write_table(table)
        if writer is None:
            raise ValueError("source corpus is empty")
    except Exception:
        if writer is not None:
            writer.close()
        raise
    else:
        writer.close()

    for session in sessions.values():
        if session["targets"] != set(session["metadata"]["eligible_assistant_indices"]):
            raise ValueError("rechunked source target coverage is incomplete")
    if (
        source_rows != train["rows"]
        or supervised != train["supervised_tokens"]
        or responses != train["assistant_responses"]
        or len(sessions) != train["source_sessions"]
        or task_keys != set(train["task_keys"])
        or supervised < minimum
    ):
        raise ValueError("rechunked corpus totals differ from the sealed source")
    after = file_sha256(source_path)
    if before != after:
        raise ValueError("source Parquet changed while rechunking")
    os.chmod(parquet_path, 0o600)

    counts = {
        "path": "train.parquet",
        "sha256": file_sha256(parquet_path),
        "rows": output_rows,
        "task_keys": sorted(task_keys),
        "format": DENSE_FORMAT,
        "source_sessions": len(sessions),
        "supervised_tokens": supervised,
        "assistant_responses": responses,
        "source_total_assistant_responses": train["source_total_assistant_responses"],
        "excluded_assistant_responses": train["excluded_assistant_responses"],
    }
    successor = {
        "schema": "cyber_dense_sft_corpus_v1",
        "source_sha256": "sha256:"
        + digest(
            {
                "source_manifest_sha256": manifest["sha256"],
                "source_train_sha256": before,
                "algorithm": ALGORITHM,
                "max_length": max_length,
                "context_tokens": context_tokens,
            }
        ),
        "split_sha256": manifest["split_sha256"],
        "tokenizer": manifest["tokenizer"],
        "files": {"train": counts},
        "train_models": manifest["train_models"],
        "max_length": max_length,
        "context_tokens": context_tokens,
        "dev_windows": 0,
        "validation_mode": "task_outcomes_only",
        "split_exclusions": manifest.get("split_exclusions", {}),
        "whole_source_exclusions": manifest.get("whole_source_exclusions", {}),
        "rechunk_provenance": {
            "algorithm": ALGORITHM,
            "source_manifest_file_sha256": source_config["manifest_file_sha256"],
            "source_manifest_sha256": manifest["sha256"],
            "source_train_sha256": before,
            "source_max_length": manifest["max_length"],
            "source_context_tokens": manifest["context_tokens"],
            "supervised_target_occurrences_preserved": supervised,
            "task_versions_preserved": len(task_versions),
            "held_out_task_families_excluded_across_all_versions": manifest.get(
                "catalog_provenance", {}
            ).get("held_out_task_families_excluded_across_all_versions"),
        },
        "catalog_provenance": manifest.get("catalog_provenance", {}),
        "builder_sha256": {"dense_rechunk.py": file_sha256(Path(__file__))},
        "limitations": [
            "Visible assistant actions only; private reasoning is omitted.",
            "Windows after the first may begin inside earlier conversation history; "
            "all copied history is loss-masked.",
            "Task outcomes, not held-out teacher-token loss, select checkpoints.",
        ],
    }
    successor["sha256"] = "sha256:" + digest(successor)
    atomic_write_json(partial / "manifest.json", successor, private=True)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "request_sha256": config["sha256"],
        "manifest_file_sha256": file_sha256(partial / "manifest.json"),
        "manifest_sha256": successor["sha256"],
        "train_parquet_sha256": counts["sha256"],
        "source_train_parquet_sha256": before,
        "source_rows": source_rows,
        "rows": output_rows,
        "source_sessions": len(sessions),
        "task_versions": len(task_versions),
        "supervised_tokens": supervised,
    }
    receipt["sha256"] = "sha256:" + digest(receipt)
    atomic_write_json(partial / "RECEIPT.json", receipt, private=True)
    try:
        _rename_noreplace(partial, output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return {"output": str(output), **receipt, "submitted": False}
