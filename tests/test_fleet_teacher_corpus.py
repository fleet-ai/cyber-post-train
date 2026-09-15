import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from cyber_post_train.jobs import digest
from training import fleet_teacher_corpus as corpus
from training.io import digest_json, file_sha256


def seal(value):
    value.pop("sha256", None)
    value["sha256"] = "sha256:" + digest(value)
    return value


def message_record(name, task, version, *, final_suffix=False):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": task},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": name + "-bash",
                    "type": "function",
                    "function": {"name": "bash", "arguments": {"script": "true"}},
                }
            ],
        },
        {"role": "tool", "content": "ok", "tool_call_id": name + "-bash"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": name + "-report",
                    "type": "function",
                    "function": {
                        "name": "submit_report",
                        "arguments": {"flag": "test", "explanation": "test"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "accepted", "tool_call_id": name + "-report"},
    ]
    if final_suffix:
        messages += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": name + "-final",
                        "type": "function",
                        "function": {
                            "name": "submit_final_answer",
                            "arguments": {"answer": "done"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "done", "tool_call_id": name + "-final"},
        ]
    record = {
        "schema": corpus.RECORD_SCHEMA,
        "record_id": name,
        "source": {
            "session_id": name,
            "model": "gpt-5.6-sol",
            "export_route": "/v1/sessions/{session_id}/transcript",
            "harness_mode": "tool-use",
            "harness_sha256": "sha256:" + "a" * 64,
            "harness_tools": ["bash", "submit_report"],
            "job_id": None,
        },
        "lineage": {"task_key": task, "eval_task_version_id": version},
        "eligibility": {"sft": True},
        "outcome": {"infra_valid": True, "success": True, "score": 1},
        "messages": messages,
    }
    record["content_digest"] = digest_json(record)
    return record


def success_evidence(record, certification):
    return seal(
        {
            "schema": corpus.EVIDENCE_SCHEMA,
            "session_id": record["record_id"],
            "model_id": record["source"]["model"],
            "task_key": record["lineage"]["task_key"],
            "task_version_id": record["lineage"]["eval_task_version_id"],
            "task_certification_receipt_sha256": certification,
            "normalized_record_sha256": digest_json(record),
            "transcript_sha256": "sha256:" + "b" * 64,
            "verifier_execution_id": "verifier",
            "routes": {
                "summary": "/v1/sessions?task_key=x",
                "transcript": "GET /v1/sessions/{session_id}/transcript",
            },
            "outcome": {
                "score_at_least_one": True,
                "status": "completed",
                "verifier_process_success": True,
            },
        }
    )


class Tokenizer:
    def __len__(self):
        return 256

    def apply_chat_template(self, messages, **kwargs):
        return [1, sum(ord(char) for char in messages[-1]["content"]) % 251]


def fixture(tmp_path: Path, monkeypatch):
    certification = "sha256:" + "c" * 64
    inventory_rows = [
        {
            "task_key": f"task-{index}",
            "task_version_id": f"version-{index}",
            "provenance": {"certification": {"receipt_sha256": certification}},
        }
        for index in range(50)
    ]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"task_versions": inventory_rows, "sha256": "inventory"}))
    split = {
        "schema": corpus.SPLIT_SCHEMA,
        "inventory": {"path": str(inventory)},
        "training_split": {
            "tasks": [
                {
                    "task_key": row["task_key"],
                    "task_version_id": row["task_version_id"],
                    "group_id": f"group-{index}",
                }
                for index, row in enumerate(inventory_rows)
            ]
        },
        "sha256": "split",
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    monkeypatch.setattr(corpus, "validate_split", lambda value, inventory_path: value)
    records = [
        message_record("source-a", "task-0", "version-0"),
        message_record("source-b", "task-1", "version-1", final_suffix=True),
    ]
    normalized = tmp_path / "normalized.jsonl"
    normalized.write_text("".join(json.dumps(row) + "\n" for row in records))
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        "".join(json.dumps(success_evidence(row, certification)) + "\n" for row in records)
    )
    (tmp_path / "lock.json").write_text("{}")
    (tmp_path / "helper.py").write_text("helper")
    monkeypatch.setattr(
        corpus,
        "local_tokenizer",
        lambda lock, root: (
            Tokenizer(),
            {"repo": "Qwen/Qwen3.8-27B", "revision": "d" * 40},
        ),
    )

    def helper(messages, tokenizer, **kwargs):
        assistant = messages[0]["role"] == "assistant"
        return [1, 2, 3], [0, int(assistant), int(assistant)], None

    monkeypatch.setattr(corpus, "native_helper", lambda path: helper)
    config = {
        "schema": corpus.SCHEMA,
        "normalized": {"path": str(normalized), "sha256": file_sha256(normalized)},
        "evidence": {"path": str(evidence), "sha256": file_sha256(evidence)},
        "inventory": str(inventory),
        "study_split": str(split_path),
        "model_lock": str(tmp_path / "lock.json"),
        "tokenizer_root": str(tmp_path),
        "native_helper": str(tmp_path / "helper.py"),
        "teacher_models": ["gpt-5.6-sol"],
        "max_length": 64,
        "context_tokens": 16,
        "catalog_census": {
            "stronger_teacher_successes": 2,
            "exact_current_version_successes": 2,
            "different_version_successes": 0,
        },
        "output": str(tmp_path / "output-a"),
    }
    return config


def test_builder_is_train_only_task_rich_and_deterministic(tmp_path, monkeypatch):
    config = fixture(tmp_path, monkeypatch)
    first = corpus.build(config, relative_to=tmp_path)
    assert first["train"] == {
        "rows": 2,
        "source_sessions": 2,
        "task_versions": 2,
        "supervised_tokens": 8,
    }
    manifest = json.loads((tmp_path / "output-a/manifest.json").read_text())
    assert manifest["validation_mode"] == "task_outcomes_only"
    assert manifest["catalog_provenance"]["terminal_finalize_suffixes_removed"] == 1
    assert manifest["files"]["train"]["assistant_responses"] == 4
    rows = pq.read_table(tmp_path / "output-a/train.parquet").to_pylist()
    assert all(row["source_transform_sha256"] is None for row in rows[:1])
    assert any(row["source_transform_sha256"] is not None for row in rows)

    second_config = copy.deepcopy(config)
    second_config["output"] = str(tmp_path / "output-b")
    second = corpus.build(second_config, relative_to=tmp_path)
    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert file_sha256(tmp_path / "output-b/train.parquet") == file_sha256(
        tmp_path / "output-a/train.parquet"
    )


def test_builder_rejects_evidence_drift_before_writing(tmp_path, monkeypatch):
    config = fixture(tmp_path, monkeypatch)
    evidence = Path(config["evidence"]["path"])
    items = [json.loads(line) for line in evidence.read_text().splitlines()]
    items[0]["task_certification_receipt_sha256"] = "sha256:" + "f" * 64
    evidence.write_text("".join(json.dumps(seal(item)) + "\n" for item in items))
    config["evidence"]["sha256"] = file_sha256(evidence)
    with pytest.raises(ValueError):
        corpus.build(config, relative_to=tmp_path)
    assert not Path(config["output"]).exists()


def test_finalize_suffix_without_completed_report_is_not_normalized():
    record = message_record("source", "task", "version", final_suffix=True)
    record["messages"] = record["messages"][:4] + record["messages"][-2:]
    with pytest.raises(corpus.Excluded, match="completed_report"):
        corpus._strip_finalize_suffix(record)
