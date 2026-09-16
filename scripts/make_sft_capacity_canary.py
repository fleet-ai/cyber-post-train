#!/usr/bin/env python3
"""Create a complete-source SFT canary from the longest Parquet row groups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cyber_post_train.jobs import digest
from training.io import atomic_write_json, atomic_write_jsonl, file_sha256, iter_jsonl
from training.sft_runtime import DENSE_FORMAT, dense_rows

QWEN38_VOCAB_SIZE = 248077


def build(source: Path, output: Path, groups: int) -> dict:
    if output.exists():
        raise FileExistsError("create-once canary destination exists")
    manifest = json.loads((source / "manifest.json").read_text())
    data = source / manifest["files"]["train"]["path"]
    if file_sha256(data) != manifest["files"]["train"]["sha256"]:
        raise ValueError("source corpus digest mismatch")
    parquet = pq.ParquetFile(data)
    ranked = []
    for index in range(parquet.num_row_groups):
        lengths = parquet.read_row_group(index, columns=["token_count"])["token_count"].to_pylist()
        ranked.append((max(lengths), index))
    selected = [index for _, index in sorted(ranked, reverse=True)[:groups]]
    tables = [parquet.read_row_group(index) for index in selected]
    table = pa.concat_tables(tables)
    rows = table.to_pylist()
    rows.sort(key=lambda row: (-row["token_count"], row["window_id"]))
    counts = {
        "rows": len(rows),
        "task_keys": sorted({row["task_key"] for row in rows}),
        "format": DENSE_FORMAT,
        "source_sessions": len({row["source_session_id"] for row in rows}),
        "supervised_tokens": sum(row["target_token_count"] for row in rows),
        "assistant_responses": sum(len(row["target_spans"]) for row in rows),
        "source_total_assistant_responses": sum(
            row["source_assistant_count"]
            for row in {item["source_session_id"]: item for item in rows}.values()
        ),
        "excluded_assistant_responses": sum(
            len(row["excluded_assistant_targets"])
            for row in {item["source_session_id"]: item for item in rows}.values()
        ),
    }
    dense_rows(
        rows,
        counts,
        max_length=manifest["max_length"],
        vocab_size=QWEN38_VOCAB_SIZE,
    )
    output.mkdir(parents=True, mode=0o700)
    data_out = output / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), data_out, compression="zstd")
    os.chmod(data_out, 0o600)
    session_ids = {row["source_session_id"] for row in rows}
    selections = [
        row
        for row in iter_jsonl(source / "source-selection.private.jsonl")
        if row["session_id"] in session_ids
    ]
    selection_out = output / "source-selection.private.jsonl"
    atomic_write_jsonl(selection_out, selections, private=True)
    lengths = sorted(row["token_count"] for row in rows)
    result = {
        **manifest,
        "source_sha256": file_sha256(selection_out),
        "files": {"train": {"path": data_out.name, "sha256": file_sha256(data_out), **counts}},
        "capacity_canary": {
            "policy": "all rows from the longest complete source row groups",
            "source_manifest_sha256": manifest["sha256"],
            "selected_source_groups": groups,
            "min_token_count": min(row["token_count"] for row in rows),
            "max_token_count": max(row["token_count"] for row in rows),
        },
        "token_length_distribution": {
            "min": lengths[0],
            "p50": lengths[(len(lengths) - 1) // 2],
            "p90": lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))],
            "p99": lengths[min(len(lengths) - 1, int(len(lengths) * 0.99))],
            "max": lengths[-1],
        },
    }
    result.pop("sha256", None)
    result["sha256"] = "sha256:" + digest(result)
    atomic_write_json(output / "manifest.json", result, private=True)
    receipt = {
        "schema": "cyber_sft_capacity_canary_corpus_receipt_v1",
        "manifest_file_sha256": file_sha256(output / "manifest.json"),
        "manifest_sha256": result["sha256"],
        "train_parquet_sha256": file_sha256(data_out),
        "source_selection_sha256": file_sha256(selection_out),
        "rows": len(rows),
        "source_sessions": counts["source_sessions"],
        "supervised_tokens": counts["supervised_tokens"],
    }
    receipt["sha256"] = "sha256:" + digest(receipt)
    atomic_write_json(output / "RECEIPT.json", receipt, private=True)
    return {
        "rows": len(rows),
        "source_sessions": counts["source_sessions"],
        "min_token_count": result["capacity_canary"]["min_token_count"],
        "max_token_count": result["capacity_canary"]["max_token_count"],
        "manifest_sha256": result["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.groups), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
