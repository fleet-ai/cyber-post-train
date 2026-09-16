import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cyber_post_train.jobs import digest
from training import broad_teacher_corpus as corpus
from training.io import digest_json, file_sha256


class Tokenizer:
    def __len__(self):
        return 256

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2]


def _record(session: str, task: str, version: str) -> dict:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": session + "-bash",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"script": "true"}},
                }
            ],
        },
        {"role": "tool", "content": "ok", "tool_call_id": session + "-bash"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": session + "-report",
                    "type": "function",
                    "function": {
                        "name": "submit_report",
                        "arguments": {"flag": "flag", "explanation": "done"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "accepted", "tool_call_id": session + "-report"},
    ]
    row = {
        "schema": "fleet_cyber_trajectory_v1",
        "record_id": session,
        "source": {
            "session_id": session,
            "model": "gpt-5.6-sol",
            "harness_mode": "tool-use",
            "harness_sha256": "sha256:" + "a" * 64,
        },
        "lineage": {"task_key": task, "eval_task_version_id": version},
        "eligibility": {"sft": True},
        "outcome": {"infra_valid": True, "success": True, "score": 1},
        "messages": messages,
    }
    row["content_digest"] = digest_json(row)
    return row


def _evidence(row: dict) -> dict:
    value = {
        "schema": corpus.EVIDENCE_SCHEMA,
        "session_id": row["record_id"],
        "model_id": row["source"]["model"],
        "task_key": row["lineage"]["task_key"],
        "task_version_id": row["lineage"]["eval_task_version_id"],
        "normalized_record_sha256": digest_json(row),
        "transcript_sha256": "sha256:" + "b" * 64,
        "routes": {"transcript": "GET /v1/sessions/{session_id}/transcript"},
        "outcome": {
            "score_at_least_one": True,
            "status": "completed",
            "verifier_process_success": True,
        },
    }
    value["sha256"] = "sha256:" + digest(value)
    return value


def _fixture(tmp_path: Path, monkeypatch) -> dict:
    row = _record("session", "train-task", "train-version")
    normalized = tmp_path / "normalized.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    normalized.write_text(json.dumps(row) + "\n")
    evidence.write_text(json.dumps(_evidence(row)) + "\n")
    split = {
        "sha256": "sha256:" + "c" * 64,
        "tasks": [
            {
                "task_key": f"holdout-{index}",
                "task_version_id": f"version-{index}",
                "split": "dev" if index < 17 else "final_test",
            }
            for index in range(25)
        ],
    }
    (tmp_path / "split.json").write_text(json.dumps(split))
    (tmp_path / "lock.json").write_text("{}")
    (tmp_path / "helper.py").write_text("helper")
    monkeypatch.setattr(
        corpus,
        "local_tokenizer",
        lambda lock, root: (Tokenizer(), {"repo": "Qwen/Qwen3.8-27B", "revision": "d" * 40}),
    )

    def helper(messages, tokenizer, **kwargs):
        assistant = messages[0]["role"] == "assistant"
        return [1, 2, 3], [0, int(assistant), int(assistant)], None

    monkeypatch.setattr(corpus, "native_helper", lambda path: helper)
    return {
        "schema": corpus.SCHEMA,
        "normalized": {"path": str(normalized), "sha256": file_sha256(normalized)},
        "evidence": {"path": str(evidence), "sha256": file_sha256(evidence)},
        "study_split": str(tmp_path / "split.json"),
        "model_lock": str(tmp_path / "lock.json"),
        "tokenizer_root": str(tmp_path),
        "native_helper": str(tmp_path / "helper.py"),
        "teacher_models": ["gpt-5.6-sol"],
        "max_length": 262144,
        "context_tokens": 262144,
        "output": str(tmp_path / "output"),
    }


def test_streaming_builder_keeps_all_targets_and_262k_envelope(tmp_path, monkeypatch):
    config = _fixture(tmp_path, monkeypatch)
    result = corpus.build(config, relative_to=tmp_path)
    assert result["train"] == {
        "rows": 1,
        "source_sessions": 1,
        "task_versions": 1,
        "supervised_tokens": 4,
    }
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["max_length"] == manifest["context_tokens"] == 262144
    rows = pq.read_table(tmp_path / "output/train.parquet").to_pylist()
    assert rows[0]["task_key"] == "train-task"
    assert len(rows[0]["target_spans"]) == 2


def test_builder_rejects_any_holdout_family_version(tmp_path, monkeypatch):
    config = _fixture(tmp_path, monkeypatch)
    row = _record("session", "holdout-0", "historical-other-version")
    normalized = Path(config["normalized"]["path"])
    evidence = Path(config["evidence"]["path"])
    normalized.write_text(json.dumps(row) + "\n")
    evidence.write_text(json.dumps(_evidence(row)) + "\n")
    config["normalized"]["sha256"] = file_sha256(normalized)
    config["evidence"]["sha256"] = file_sha256(evidence)
    with pytest.raises(ValueError, match="held-out task family"):
        corpus.build(copy.deepcopy(config), relative_to=tmp_path)


def test_builder_salvages_only_complete_prefix_before_orphan_result(tmp_path, monkeypatch):
    config = _fixture(tmp_path, monkeypatch)
    row = _record("session", "train-task", "train-version")
    row["messages"].insert(
        4,
        {"role": "tool", "content": "legacy orphan", "tool_call_id": "missing-call"},
    )
    row["content_digest"] = digest_json({k: v for k, v in row.items() if k != "content_digest"})
    normalized = Path(config["normalized"]["path"])
    evidence = Path(config["evidence"]["path"])
    normalized.write_text(json.dumps(row) + "\n")
    evidence.write_text(json.dumps(_evidence(row)) + "\n")
    config["normalized"]["sha256"] = file_sha256(normalized)
    config["evidence"]["sha256"] = file_sha256(evidence)

    result = corpus.build(config, relative_to=tmp_path)

    assert result["train"]["source_sessions"] == 1
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["catalog_provenance"]["lifecycle_prefix_salvages"] == 1
    rows = pq.read_table(tmp_path / "output/train.parquet").to_pylist()
    assert len(rows[0]["target_spans"]) == 1
    selection = json.loads((tmp_path / "output/source-selection.private.jsonl").read_text())
    assert selection["lifecycle_prefix_salvage"] == {
        "reason": "orphan_or_duplicate_tool_result",
        "kept_messages": 4,
        "dropped_messages": 3,
        "original_assistant_responses": 2,
        "retained_assistant_responses": 1,
    }


def test_builder_discards_a_duplicate_payload_as_one_whole_source(tmp_path, monkeypatch):
    config = _fixture(tmp_path, monkeypatch)
    rows = [
        _record("session-a", "train-task-a", "train-version-a"),
        _record("session-b", "train-task-b", "train-version-b"),
    ]
    normalized = Path(config["normalized"]["path"])
    evidence = Path(config["evidence"]["path"])
    normalized.write_text("".join(json.dumps(row) + "\n" for row in rows))
    evidence.write_text("".join(json.dumps(_evidence(row)) + "\n" for row in rows))
    config["normalized"]["sha256"] = file_sha256(normalized)
    config["evidence"]["sha256"] = file_sha256(evidence)

    result = corpus.build(config, relative_to=tmp_path)

    assert result["train"]["source_sessions"] == 1
    assert result["whole_source_exclusions"] == {"exact_window_payload_duplicate": 1}
