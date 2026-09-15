import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cyber_post_train.jobs import digest
from training import corpus_subset
from training.io import file_sha256
from training.sft_runtime import DENSE_FORMAT


def seal(value):
    value["sha256"] = "sha256:" + digest(value)
    return value


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def episode(name, task, version, policy, *, terminal_only=False):
    coverage = {
        "status": "certified",
        "target_policy_sha256": policy,
        "assistant_responses": 1,
        "submit_report_responses": int(terminal_only),
        "supervised_tokens": 2,
        "submit_report_tokens": 2 if terminal_only else 0,
        "non_submit_tool_responses": 1,
        "completed_non_submit_tool_rounds": 1,
    }
    return seal(
        {
            "schema": "cyber_sft_episode_metadata_v1",
            "episode_id": name,
            "task_key": task,
            "task_version_id": version,
            "source_kind": "teacher",
            "model_id": "gpt-5.6-sol",
            "validity": "valid",
            "verified_success": True,
            "trace_sha256": "sha256:" + "a" * 64,
            "acceptance_sha256": "sha256:" + "b" * 64,
            "normalized_record_sha256": "sha256:" + "c" * 64,
            "coverage": coverage,
        }
    )


def dense(name, task):
    ids = [1, 2, 3]
    return {
        "input_ids": ids,
        "loss_mask": [0, 1, 1],
        "token_count": 3,
        "target_token_count": 2,
        "task_key": task,
        "window_id": "window-" + name,
        "segment_id": 0,
        "source_session_id": name,
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
                "source_target_sha256": digest(ids[1:]),
            }
        ],
        "copied_context_assistant_indices": [],
        "context_start_message_index": 0,
        "split": "train",
    }


def fixture(tmp_path, monkeypatch, *, terminal_only=False):
    inventory = tmp_path / "inventory.json"
    write(inventory, {"sha256": "sha256:" + "1" * 64})
    split = {
        "schema": corpus_subset.SPLIT_SCHEMA,
        "sha256": "sha256:" + "2" * 64,
        "inventory": {"path": str(inventory), "logical_sha256": "sha256:" + "1" * 64},
        "training_split": {
            "tasks": [{"task_key": "train", "task_version_id": "v-train", "group_id": "g1"}]
        },
        "evaluation": {
            "dev": {"tasks": [{"task_key": "dev", "task_version_id": "v-dev"}]},
            "final_test": {"tasks": []},
        },
    }
    split_path = tmp_path / "split.json"
    write(split_path, split)
    monkeypatch.setattr(corpus_subset, "validate_split", lambda value, inventory_path: value)
    identity = {
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "d" * 40,
        "files": [],
        "chat_template_sha256": "e" * 64,
        "backend_sha256": "f" * 64,
    }
    monkeypatch.setattr(corpus_subset, "local_tokenizer", lambda lock, root: (range(32), identity))
    write(tmp_path / "lock.json", {})

    parent = tmp_path / "parent"
    policy = "sha256:" + "3" * 64
    episodes = [
        episode("train-source", "train", "v-train", policy, terminal_only=terminal_only),
        episode("dev-source", "dev", "v-dev", policy),
    ]
    parent.mkdir()
    (parent / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    coverage = seal(
        {
            "schema": "cyber_sft_coverage_extraction_v1",
            "target_policy_sha256": policy,
            "files": {"episodes.jsonl": file_sha256(parent / "episodes.jsonl")},
        }
    )
    write(parent / "coverage.json", coverage)
    evidence = [
        {
            "episode_id": row["episode_id"],
            "metadata_sha256": row["sha256"],
            "trace_sha256": row["trace_sha256"],
            "acceptance_sha256": row["acceptance_sha256"],
            "normalized_record_sha256": row["normalized_record_sha256"],
        }
        for row in episodes
    ]
    selection = seal(
        {
            "schema": "cyber_sft_source_selection_v1",
            "study_split_sha256": "sha256:" + "4" * 64,
            "target_policy_sha256": policy,
            "status": "ready",
            "models": ["gpt-5.6-sol"],
            "selected_episode_ids": [row["episode_id"] for row in episodes],
            "selected_evidence": evidence,
        }
    )
    write(parent / "selection.json", selection)
    rows = [dense("train-source", "train"), dense("dev-source", "dev")]
    pq.write_table(pa.Table.from_pylist(rows), parent / "train.parquet")
    train = {
        "path": "train.parquet",
        "sha256": file_sha256(parent / "train.parquet"),
        "rows": 2,
        "task_keys": ["train", "dev"],
        "format": DENSE_FORMAT,
    }
    manifest = seal(
        {
            "schema": "cyber_dense_sft_corpus_v1",
            "source_sha256": "sha256:" + "5" * 64,
            "split_sha256": "sha256:" + "6" * 64,
            "tokenizer": identity,
            "files": {"train": train},
            "max_length": 16,
            "context_tokens": 4,
            "validation_mode": "task_outcomes_only",
            "source_selection": {
                "study_split_sha256": selection["study_split_sha256"],
                "source_selection_sha256": selection["sha256"],
                "target_policy_sha256": policy,
                "selected_episode_count": 2,
            },
        }
    )
    write(parent / "manifest.json", manifest)
    return {
        "schema": "cyber_dense_sft_subset_request_v1",
        "inventory": str(inventory),
        "study_split": str(split_path),
        "model_lock": str(tmp_path / "lock.json"),
        "tokenizer_root": str(tmp_path),
        "source_kind": "teacher",
        "source_model": "gpt-5.6-sol",
        "parents": [
            {
                "corpus_manifest": str(parent / "manifest.json"),
                "source_selection": str(parent / "selection.json"),
                "coverage_manifest": str(parent / "coverage.json"),
            }
        ]
        * 2,
        "output": str(tmp_path / "output"),
    }


def test_subset_is_exact_train_only_and_deduplicates_parents(tmp_path, monkeypatch):
    result = corpus_subset.build(fixture(tmp_path, monkeypatch), relative_to=tmp_path)
    assert result["train"] == {
        "rows": 1,
        "source_sessions": 1,
        "task_versions": 1,
        "supervised_tokens": 2,
    }
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["validation_mode"] == "task_outcomes_only"
    assert manifest["files"]["train"]["task_keys"] == ["train"]
    assert set(manifest["files"]) == {"train"}


def test_terminal_submit_only_source_is_not_admitted(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="coverage is incomplete"):
        corpus_subset.build(
            fixture(tmp_path, monkeypatch, terminal_only=True), relative_to=tmp_path
        )
