import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.test_holdout_filter import dense, seal, task, write
from training import holdout_filter, holdout_filter_verify
from training.io import file_sha256
from training.sft_runtime import DENSE_FORMAT


def corpus_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    retained = [(f"retained-{index:02d}", f"v-{index:02d}") for index in range(35)]
    rows = [
        dense("protected-session", "protected", "v-protected"),
        dense("protected-alias-session", "alias-protected", "v-protected-alias"),
    ]
    rows.extend(
        dense(f"retained-session-{index:02d}", key, version)
        for index, (key, version) in enumerate(retained)
    )
    source_parquet = source_root / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source_parquet, compression="zstd")
    source_spec = {
        "path": "train.parquet",
        "sha256": file_sha256(source_parquet),
        "rows": 37,
        "task_keys": sorted(["protected", "alias-protected", *[key for key, _ in retained]]),
        "format": DENSE_FORMAT,
        "source_sessions": 37,
        "supervised_tokens": 74,
        "assistant_responses": 37,
        "source_total_assistant_responses": 37,
        "excluded_assistant_responses": 0,
    }
    manifest = seal(
        {
            "schema": "cyber_dense_sft_corpus_v1",
            "source_sha256": "sha256:" + "1" * 64,
            "split_sha256": "sha256:" + "2" * 64,
            "tokenizer": {"repo": "Qwen/Qwen3.8-27B"},
            "files": {"train": source_spec},
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
                task(
                    "protected",
                    "v-protected",
                    "protected-app",
                    "protected-family",
                ),
                task(
                    "alias-protected",
                    "v-protected-alias",
                    "protected-app",
                    "protected-family",
                ),
                *[
                    task(key, version, f"app-{index:02d}", f"family-{index:02d}")
                    for index, (key, version) in enumerate(retained)
                ],
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
                    "group_id": "protected-group",
                }
            ],
            "group_ids": ["protected-group"],
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
            "source_task_versions": 37,
            "protected_task_families": 1,
            "excluded_source_task_versions": [
                {
                    "application": "protected-app",
                    "task_family": "protected-family",
                    "task_key": "alias-protected",
                    "task_version_id": "v-protected-alias",
                },
                {
                    "application": "protected-app",
                    "task_family": "protected-family",
                    "task_key": "protected",
                    "task_version_id": "v-protected",
                },
            ],
            "excluded_source_task_version_count": 2,
            "retained_source_task_versions": 35,
        }
    )
    output = tmp_path / "output"
    request = seal(
        {
            "schema": holdout_filter.REQUEST_SCHEMA,
            "name": "verify-test-filter",
            "source": {
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": file_sha256(manifest_path),
                "manifest_sha256": manifest["sha256"],
                "data_root": str(source_root),
                "train_parquet_path": "train.parquet",
                "train_parquet_sha256": file_sha256(source_parquet),
                "source_selection_sha256": manifest["source_sha256"],
                "source_split_sha256": manifest["split_sha256"],
                "rows": 37,
                "source_sessions": 37,
                "task_versions": 37,
                "supervised_tokens": 74,
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
                "public_manifest_path": "public-manifest.json",
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
                    "retained_task_versions": 35,
                    "all_output_task_identities_known_to_inventory": True,
                },
            },
        }
    )
    holdout_filter.build(request, relative_to=tmp_path)
    return request, source_parquet, output


def test_verifier_reconstructs_successor_and_emits_no_private_rows(tmp_path):
    request, _, output = corpus_fixture(tmp_path)
    result = holdout_filter_verify.verify(request, relative_to=tmp_path)

    assert result["status"] == "verified"
    assert result["train"]["task_versions"] == 35
    assert result["final_test_exact_identity_overlap"] == 0
    assert result["final_test_task_family_overlap"] == 0
    assert result["public_manifest"] == json.loads((output / "manifest.json").read_text())
    serialized = json.dumps(result, sort_keys=True)
    assert "protected-session" not in serialized
    assert "retained-session" not in serialized


def test_cli_prints_only_sanitized_summary_and_public_manifest(tmp_path, monkeypatch, capsys):
    request, _, _ = corpus_fixture(tmp_path)
    request_path = tmp_path / "request.json"
    write(request_path, request)
    monkeypatch.setattr(sys, "argv", ["holdout-filter-verify", "--request", str(request_path)])

    holdout_filter_verify.main()

    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["status"] == "verified"
    assert parsed["public_manifest"]["files"]["train"]["rows"] == 35
    assert "protected-session" not in output
    assert "retained-session" not in output


def test_verifier_rejects_source_drift(tmp_path):
    request, source, _ = corpus_fixture(tmp_path)
    source.write_bytes(source.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="source train Parquet digest mismatch"):
        holdout_filter_verify.verify(request, relative_to=tmp_path)


def test_verifier_rejects_resealed_false_receipt(tmp_path):
    request, _, output = corpus_fixture(tmp_path)
    receipt_path = output / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.pop("sha256")
    receipt["source_stable"] = False
    seal(receipt)
    write(receipt_path, receipt)

    with pytest.raises(ValueError, match="successor receipt differs"):
        holdout_filter_verify.verify(request, relative_to=tmp_path)


def test_verifier_rejects_output_parquet_drift(tmp_path):
    request, _, output = corpus_fixture(tmp_path)
    data = output / "train.parquet"
    data.write_bytes(data.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="successor manifest/receipt file binding"):
        holdout_filter_verify.verify(request, relative_to=tmp_path)
