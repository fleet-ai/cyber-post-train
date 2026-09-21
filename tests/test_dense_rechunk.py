from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cyber_post_train.jobs import digest
from training import dense_rechunk
from training.io import file_sha256
from training.sft_runtime import DENSE_FORMAT, dense_rows


def _row() -> dict:
    ids = list(range(20))
    spans = [
        {
            "assistant_index": 0,
            "source_message_index": 2,
            "token_start": 5,
            "token_end": 8,
            "source_target_sha256": dense_rechunk._target_digest(ids[5:8]),
        },
        {
            "assistant_index": 1,
            "source_message_index": 4,
            "token_start": 15,
            "token_end": 19,
            "source_target_sha256": dense_rechunk._target_digest(ids[15:19]),
        },
    ]
    mask = [0] * 20
    mask[5:8] = [1] * 3
    mask[15:19] = [1] * 4
    return {
        "input_ids": ids,
        "loss_mask": mask,
        "token_count": 20,
        "target_token_count": 7,
        "task_key": "fleet-blackbox-train",
        "window_id": "source::dense::0-1",
        "segment_id": "source::dense::0-1",
        "source_session_id": "private-source",
        "source_model": "teacher",
        "source_assistant_count": 2,
        "eligible_assistant_indices": [0, 1],
        "excluded_assistant_targets": [],
        "target_spans": spans,
        "copied_context_assistant_indices": [],
        "context_start_message_index": 0,
        "split": "train",
        "source_task_version_id": "00000000-0000-0000-0000-000000000001",
    }


def _sealed(value: dict) -> dict:
    value["sha256"] = "sha256:" + digest(value)
    return value


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dense_rechunk, "MINIMUM_SUPERVISED_TOKENS", 7)
    data = tmp_path / "source"
    data.mkdir()
    parquet = data / "train.parquet"
    pq.write_table(pa.Table.from_pylist([_row()]), parquet, compression="zstd")
    manifest = _sealed(
        {
            "schema": "cyber_dense_sft_corpus_v1",
            "source_sha256": "sha256:" + "1" * 64,
            "split_sha256": "sha256:" + "2" * 64,
            "tokenizer": {"repo": "Qwen/Qwen3.8-27B", "revision": "exact"},
            "files": {
                "train": {
                    "path": "train.parquet",
                    "sha256": file_sha256(parquet),
                    "rows": 1,
                    "task_keys": ["fleet-blackbox-train"],
                    "format": DENSE_FORMAT,
                    "source_sessions": 1,
                    "supervised_tokens": 7,
                    "assistant_responses": 2,
                    "source_total_assistant_responses": 2,
                    "excluded_assistant_responses": 0,
                }
            },
            "train_models": ["teacher"],
            "max_length": 20,
            "context_tokens": 8,
            "dev_windows": 0,
            "validation_mode": "task_outcomes_only",
            "split_exclusions": {},
            "whole_source_exclusions": {},
            "catalog_provenance": {"held_out_task_families_excluded_across_all_versions": 25},
        }
    )
    manifest_path = data / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = _sealed(
        {
            "schema": dense_rechunk.REQUEST_SCHEMA,
            "source": {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": file_sha256(manifest_path),
                "manifest_sha256": manifest["sha256"],
                "data_root": str(data),
                "train_parquet_sha256": file_sha256(parquet),
            },
            "destination": {
                "output_root": str(tmp_path / "output"),
                "create_once": True,
            },
            "max_length": 10,
            "context_tokens": 4,
            "minimum_supervised_tokens": 7,
        }
    )
    path = tmp_path / "request.json"
    path.write_text(json.dumps(config))
    return path


def test_rechunk_preserves_each_target_once() -> None:
    rows = dense_rechunk.rechunk_row(_row(), max_length=10, context_tokens=4)
    assert [row["token_count"] for row in rows] == [7, 8]
    assert [row["target_token_count"] for row in rows] == [3, 4]
    assert [span["assistant_index"] for row in rows for span in row["target_spans"]] == [0, 1]
    assert sum(sum(row["loss_mask"]) for row in rows) == 7


def test_rechunk_rejects_one_target_larger_than_context() -> None:
    row = _row()
    row["target_spans"] = [
        {
            "assistant_index": 0,
            "source_message_index": 2,
            "token_start": 1,
            "token_end": 19,
            "source_target_sha256": dense_rechunk._target_digest(row["input_ids"][1:19]),
        }
    ]
    with pytest.raises(ValueError, match="assistant target exceeds"):
        dense_rechunk.rechunk_row(row, max_length=10, context_tokens=2)


def test_build_streams_create_once_and_preserves_dense_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _fixture(tmp_path, monkeypatch)
    result = dense_rechunk.build(
        json.loads(request_path.read_text()), relative_to=request_path.parent
    )
    assert result["supervised_tokens"] == 7
    assert result["rows"] == 2
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    spec = manifest["files"]["train"]
    rows = pq.read_table(tmp_path / "output/train.parquet").to_pylist()
    dense_rows(rows, spec, max_length=10, vocab_size=100)
    assert manifest["rechunk_provenance"] == {
        "algorithm": dense_rechunk.ALGORITHM,
        "source_manifest_file_sha256": json.loads(request_path.read_text())["source"][
            "manifest_file_sha256"
        ],
        "source_manifest_sha256": json.loads(request_path.read_text())["source"]["manifest_sha256"],
        "source_train_sha256": json.loads(request_path.read_text())["source"][
            "train_parquet_sha256"
        ],
        "source_max_length": 20,
        "source_context_tokens": 8,
        "supervised_target_occurrences_preserved": 7,
        "task_versions_preserved": 1,
        "held_out_task_families_excluded_across_all_versions": 25,
    }
    with pytest.raises(FileExistsError):
        dense_rechunk.build(json.loads(request_path.read_text()), relative_to=request_path.parent)


def test_build_fails_closed_on_source_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = _fixture(tmp_path, monkeypatch)
    value = json.loads(request_path.read_text())
    value["source"]["train_parquet_sha256"] = "sha256:" + "0" * 64
    value.pop("sha256")
    value = _sealed(value)
    with pytest.raises(ValueError, match="source train binding differs"):
        dense_rechunk.build(value, relative_to=request_path.parent)
