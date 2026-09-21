import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cyber_post_train.jobs import digest
from training import holdout_filter
from training.io import file_sha256
from training.sft_runtime import DENSE_FORMAT


def seal(value: dict) -> dict:
    value["sha256"] = "sha256:" + digest(value)
    return value


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def task(name: str, version: str, application: str, family: str) -> dict:
    return {
        "task_key": name,
        "task_version_id": version,
        "lineage_id": name,
        "taxonomy": {
            "application": {"status": "verified", "value": application},
            "task_family": {"status": "verified", "value": family},
        },
    }


def dense(session: str, key: str, version: str, *, segment: int = 0) -> dict:
    ids = [1, 2, 3]
    return {
        "input_ids": ids,
        "loss_mask": [0, 1, 1],
        "token_count": 3,
        "target_token_count": 2,
        "task_key": key,
        "window_id": f"window-{session}-{segment}",
        "segment_id": segment,
        "source_session_id": session,
        "source_model": "gpt-5.6-sol",
        "source_assistant_count": 1,
        "eligible_assistant_indices": [0],
        "excluded_assistant_targets": [],
        "target_spans": [
            {
                "assistant_index": 0,
                "source_message_index": 2,
                "token_start": 1,
                "token_end": 3,
                "source_target_sha256": "sha256:" + digest(ids[1:]),
            }
        ],
        "copied_context_assistant_indices": [],
        "context_start_message_index": 0,
        "split": "train",
        "source_task_version_id": version,
    }


def fixture(tmp_path: Path, *, mixed_session: bool = False) -> tuple[dict, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    rows = [
        dense("protected-session", "protected", "v-protected"),
        dense("retained-session", "retained", "v-retained"),
    ]
    if mixed_session:
        rows[1]["source_session_id"] = "protected-session"
    source_parquet = source_root / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source_parquet, compression="zstd")
    counts = {
        "path": "train.parquet",
        "sha256": file_sha256(source_parquet),
        "rows": 2,
        "task_keys": ["protected", "retained"],
        "format": DENSE_FORMAT,
        "source_sessions": 1 if mixed_session else 2,
        "supervised_tokens": 4,
        "assistant_responses": 2,
        "source_total_assistant_responses": 1 if mixed_session else 2,
        "excluded_assistant_responses": 0,
    }
    manifest = seal(
        {
            "schema": "cyber_dense_sft_corpus_v1",
            "source_sha256": "sha256:" + "1" * 64,
            "split_sha256": "sha256:" + "2" * 64,
            "tokenizer": {"repo": "Qwen/Qwen3.8-27B"},
            "files": {"train": counts},
            "train_models": ["gpt-5.6-sol"],
            "max_length": 16,
            "context_tokens": 4,
            "dev_windows": 0,
            "validation_mode": "task_outcomes_only",
            "split_exclusions": {},
            "whole_source_exclusions": {},
            "catalog_provenance": {},
            "builder_sha256": {},
            "limitations": [],
        }
    )
    manifest_path = tmp_path / "source-manifest.json"
    write(manifest_path, manifest)

    inventory = seal(
        {
            "schema": "cyber_study_inventory_v1",
            "tasks": [
                task("protected", "v-protected", "app-a", "family-a"),
                task("retained", "v-retained", "app-b", "family-b"),
            ],
        }
    )
    inventory_path = tmp_path / "inventory.json"
    write(inventory_path, inventory)
    final = seal(
        {
            "schema": "cyber_final_test_lock_v1",
            "inventory_sha256": inventory["sha256"],
            "tasks": [
                {
                    "task_key": "protected",
                    "task_version_id": "v-protected",
                    "group_id": "group-a",
                }
            ],
            "group_ids": ["group-a"],
        }
    )
    final_path = tmp_path / "final.json"
    write(final_path, final)
    selection = seal(
        {
            "schema": holdout_filter.SELECTION_SCHEMA,
            "source_split_sha256": manifest["split_sha256"],
            "protected_final_test_sha256": final["sha256"],
            "mapping_inventory_sha256": inventory["sha256"],
            "source_task_versions": 2,
            "protected_task_families": 1,
            "excluded_source_task_versions": [
                {
                    "application": "app-a",
                    "task_family": "family-a",
                    "task_key": "protected",
                    "task_version_id": "v-protected",
                }
            ],
            "excluded_source_task_version_count": 1,
            "retained_source_task_versions": 1,
        }
    )
    output = tmp_path / "output"
    request = seal(
        {
            "schema": holdout_filter.REQUEST_SCHEMA,
            "name": "test-filter",
            "source": {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": file_sha256(manifest_path),
                "manifest_sha256": manifest["sha256"],
                "data_root": str(source_root),
                "train_parquet_path": "train.parquet",
                "train_parquet_sha256": file_sha256(source_parquet),
                "source_selection_sha256": manifest["source_sha256"],
                "source_split_sha256": manifest["split_sha256"],
                "rows": 2,
                "source_sessions": counts["source_sessions"],
                "task_versions": 2,
                "supervised_tokens": 4,
            },
            "protection": {
                "final_test_lock_path": str(final_path),
                "final_test_lock_file_sha256": file_sha256(final_path),
                "final_test_lock_sha256": final["sha256"],
                "inventory_path": str(inventory_path),
                "inventory_file_sha256": file_sha256(inventory_path),
                "inventory_sha256": inventory["sha256"],
                "split_unit": ["application", "task_family"],
                "join_identity": ["task_key", "task_version_id"],
                "unknown_or_ambiguous_identity": "reject",
            },
            "selection": selection,
            "destination": {
                "data_root": str(output),
                "train_parquet_path": "train.parquet",
                "private_selection_path": "source-selection.private.jsonl",
                "manifest_path": "manifest.json",
                "receipt_path": "RECEIPT.json",
                "public_manifest_path": "unused.json",
                "create_once": True,
                "publication": "atomic_no_replace",
            },
            "materialization": {
                "row_identity": ["task_key", "source_task_version_id"],
                "algorithm": "test",
                "partial_source_session_removal": "reject",
                "unknown_or_ambiguous_source_row": "reject",
                "preserve_source_evidence_fields": True,
                "source_digest_before_after_must_match": True,
                "derive_counts_from_filtered_payload": True,
                "predicted_rows_sessions_tokens_or_file_sha256": False,
                "required_successor_bindings": [],
                "acceptance": {
                    "final_test_exact_identity_overlap": 0,
                    "final_test_task_family_overlap": 0,
                    "retained_task_versions": 1,
                    "all_output_task_identities_known_to_inventory": True,
                },
            },
        }
    )
    return request, source_parquet


def test_filter_removes_whole_protected_session_and_seals_outputs(tmp_path):
    request, source = fixture(tmp_path)
    before = file_sha256(source)
    result = holdout_filter.build(request, relative_to=tmp_path)

    assert result["train"] == {
        "rows": 1,
        "source_sessions": 1,
        "task_versions": 1,
        "supervised_tokens": 2,
    }
    assert file_sha256(source) == before
    output = Path(result["output"])
    manifest = json.loads((output / "manifest.json").read_text())
    receipt = json.loads((output / "RECEIPT.json").read_text())
    assert manifest["files"]["train"]["task_keys"] == ["retained"]
    assert manifest["files"]["train"]["rows"] == 1
    assert receipt["final_test_exact_identity_overlap"] == 0
    assert receipt["final_test_task_family_overlap"] == 0
    assert receipt["source_stable"] is True
    assert manifest["sha256"] == "sha256:" + digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    assert receipt["sha256"] == "sha256:" + digest(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )


def test_filter_rejects_partial_source_session_removal(tmp_path):
    request, _ = fixture(tmp_path, mixed_session=True)
    with pytest.raises(ValueError, match="partial source-session removal"):
        holdout_filter.build(request, relative_to=tmp_path)


def test_filter_rejects_source_drift(tmp_path):
    request, source = fixture(tmp_path)
    source.write_bytes(source.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="source train Parquet digest mismatch"):
        holdout_filter.build(request, relative_to=tmp_path)


def test_filter_is_create_once(tmp_path):
    request, _ = fixture(tmp_path)
    holdout_filter.build(request, relative_to=tmp_path)
    with pytest.raises(FileExistsError, match="create-once corpus destination exists"):
        holdout_filter.build(request, relative_to=tmp_path)
