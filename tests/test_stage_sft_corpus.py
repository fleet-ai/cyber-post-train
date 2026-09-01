from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from training.stage_sft_corpus import build_sft_windows, build_stage, verify_tokenizer_lock


class _WordTokenizer:
    def apply_chat_template(self, conversation, **_kwargs):
        # Include per-message framing so tests exercise boundary selection too.
        return [0] * sum(
            2 + len(str(message.get("content") or "").split()) for message in conversation
        )


class _Backend:
    def to_str(self):
        return '{"backend":"exact"}'


class _LockedTokenizer(_WordTokenizer):
    chat_template = "exact template"
    backend_tokenizer = _Backend()
    model_max_length = 262_144

    def __len__(self):
        return 248_077


def _row(*, session_id: str = "s1", split: str = "train", task: str = "task-1") -> dict:
    return {
        "schema": "fleet_cyber_trajectory_v1",
        "record_id": session_id,
        "source": {"job_id": "job-1", "model": "teacher"},
        "lineage": {"task_key": task},
        "environment": {"env_key": "cyber-env"},
        "outcome": {"infra_valid": True, "success": True, "score": 1, "status": "completed"},
        "eligibility": {"sft": True},
        "split": split,
        "messages": [
            {"role": "user", "content": "find it"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}],
            },
            {"role": "tool", "content": {"ok": True}, "tool_call_id": "c1"},
        ],
    }


def test_build_stage_emits_trace_ingest_compatible_private_shards(tmp_path):
    manifest = build_stage([_row()], tmp_path, team_id="fleet-team", source_sha256="sha256:x")

    corpus_path = tmp_path / manifest["corpus"][0]["relative_path"]
    score_path = tmp_path / manifest["scores"]["relative_path"]
    rows = pq.read_table(corpus_path).to_pylist()
    scores = pq.read_table(score_path).to_pylist()

    assert manifest["eligible_sessions"] == 1
    assert manifest["emitted_sessions"] == 1
    assert manifest["task_counts"] == {"train": 1}
    assert [row["position"] for row in rows] == [0, 1, 2]
    assert [row["role"] for row in rows] == ["user", "assistant", "tool"]
    assert rows[1]["tool_calls"] and json.loads(rows[1]["tool_calls"])[0]["id"] == "c1"
    assert json.loads(rows[2]["content"]) == {"ok": True}
    assert scores == [{"session_id": "s1", "score": 1.0}]
    assert corpus_path.stat().st_mode & 0o777 == 0o600
    assert score_path.stat().st_mode & 0o777 == 0o600


def test_build_stage_rejects_task_leakage(tmp_path):
    with pytest.raises(ValueError, match="task leakage"):
        build_stage(
            [_row(), _row(session_id="s2", split="test")],
            tmp_path,
            team_id="fleet-team",
            source_sha256="sha256:x",
        )


def test_build_stage_filters_splits_before_eligibility_and_tokenization(tmp_path):
    manifest = build_stage(
        [
            _row(session_id="train-success", split="train", task="train-task"),
            _row(session_id="dev-success", split="dev", task="dev-task"),
            _row(session_id="test-success", split="test", task="test-task"),
        ],
        tmp_path,
        team_id="fleet-team",
        source_sha256="sha256:canonical-unfiltered-source",
        tokenizer=_WordTokenizer(),
        tokenizer_identity={"revision": "exact"},
        window_max_tokens=30,
        targets_per_trajectory=1,
        include_splits=frozenset({"train"}),
    )

    assert manifest["source_trajectories_sha256"] == "sha256:canonical-unfiltered-source"
    assert manifest["split_filter"] == {
        "included": ["train"],
        "excluded_source_rows": {"dev": 1, "test": 1},
    }
    assert manifest["task_counts"] == {"train": 1}
    assert manifest["session_counts"] == {"train": 1}
    assert manifest["windowing"]["window_counts"] == {"train": 1}
    assert manifest["eligible_sessions"] == 1
    assert manifest["emitted_sessions"] == 1


@pytest.mark.parametrize("include_splits", [frozenset(), frozenset({"holdout"})])
def test_build_stage_rejects_invalid_split_filters(tmp_path, include_splits):
    with pytest.raises(ValueError, match="include_splits"):
        build_stage(
            [_row()],
            tmp_path,
            team_id="fleet-team",
            source_sha256="sha256:x",
            include_splits=include_splits,
        )


def test_windowing_keeps_final_success_and_maximizes_recent_context():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "find flag"},
        {"role": "assistant", "content": "first"},
        {"role": "tool", "content": "one two three four five"},
        {"role": "assistant", "content": "middle"},
        {"role": "tool", "content": "six seven eight nine ten"},
        {"role": "assistant", "content": "final success"},
    ]
    windows, oversized = build_sft_windows(
        messages,
        tokenizer=_WordTokenizer(),
        max_tokens=14,
        targets_per_trajectory=2,
    )

    assert oversized == 0
    assert [window.target_position for window in windows] == [2, 6]
    assert windows[-1].messages[-1]["content"] == "final success"
    assert windows[-1].token_count <= 14
    assert windows[-1].messages[0]["role"] == "system"
    assert windows[-1].messages[1]["role"] == "user"


@pytest.mark.parametrize(
    "submission_call",
    [
        {"function": {"name": "submit_report", "arguments": "{}"}},
        {
            "function": {
                "name": "use_tool",
                "arguments": '{"tool_name":"fleet_environment__submit_report"}',
            }
        },
    ],
)
def test_windowing_prioritizes_submission_turn_before_final_text(submission_call):
    messages = [
        {"role": "user", "content": "find flag"},
        {"role": "assistant", "content": "start"},
        {"role": "tool", "content": "observation"},
        {"role": "assistant", "content": "probe one"},
        {"role": "tool", "content": "observation"},
        {"role": "assistant", "content": "probe two"},
        {"role": "tool", "content": "flag"},
        {"role": "assistant", "content": "submit", "tool_calls": [submission_call]},
        {"role": "tool", "content": "accepted"},
        {"role": "assistant", "content": "done"},
    ]
    windows, oversized = build_sft_windows(
        messages,
        tokenizer=_WordTokenizer(),
        max_tokens=50,
        targets_per_trajectory=3,
    )

    assert oversized == 0
    assert [window.target_position for window in windows] == [1, 7, 9]


def test_build_stage_emits_scored_synthetic_windows(tmp_path):
    manifest = build_stage(
        [_row()],
        tmp_path,
        team_id="fleet-team",
        source_sha256="sha256:x",
        tokenizer=_WordTokenizer(),
        tokenizer_identity={"revision": "exact"},
        window_max_tokens=30,
        targets_per_trajectory=1,
        artifact_stem="qwen-windowed-v1",
        corpus_job_id="qwen-windowed-corpus-v1",
    )

    assert manifest["eligible_sessions"] == 1
    assert manifest["emitted_sessions"] == 1
    assert manifest["windowing"]["train_on_what"] == "last_assistant_message"
    score_path = tmp_path / manifest["scores"]["relative_path"]
    scores = pq.read_table(score_path).to_pylist()
    assert scores[0]["session_id"].startswith("s1::sftw::")
    corpus_path = tmp_path / manifest["corpus"][0]["relative_path"]
    rows = pq.read_table(corpus_path).to_pylist()
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert {row["job_id"] for row in rows} == {"qwen-windowed-corpus-v1"}
    assert manifest["source_job_ids"] == ["job-1"]
    assert manifest["corpus_job_id"] == "qwen-windowed-corpus-v1"


def test_build_stage_rejects_unsafe_artifact_stem(tmp_path):
    with pytest.raises(ValueError, match="safe filename stem"):
        build_stage(
            [_row()],
            tmp_path,
            team_id="fleet-team",
            source_sha256="sha256:x",
            artifact_stem="../overwrite",
        )


def test_build_stage_rejects_unbound_window_tokenizer(tmp_path):
    with pytest.raises(ValueError, match="verified tokenizer_identity"):
        build_stage(
            [_row()],
            tmp_path,
            team_id="fleet-team",
            source_sha256="sha256:x",
            tokenizer=_WordTokenizer(),
            window_max_tokens=30,
        )


def test_verify_tokenizer_lock_hashes_exact_revision_files(tmp_path, monkeypatch):
    files = {
        "chat_template.jinja": b"exact template",
        "tokenizer.json": b'{"tokenizer":"exact"}',
    }
    for name, contents in files.items():
        (tmp_path / name).write_bytes(contents)
    lock = {
        "schema": "huggingface_model_lock_v1",
        "repo": "Qwen/exact",
        "revision": "a" * 40,
        "tokenizer": {
            "manifest_sha256": "sha256:manifest",
            "files": [
                {
                    "path": name,
                    "sha256": hashlib.sha256(contents).hexdigest(),
                }
                for name, contents in files.items()
            ],
        },
    }
    lock_path = tmp_path / "model-lock.json"
    lock_path.write_text(json.dumps(lock))

    def fake_download(*, repo_id, filename, revision):
        assert (repo_id, revision) == ("Qwen/exact", "a" * 40)
        return str(tmp_path / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )
    identity = verify_tokenizer_lock(
        tokenizer=_LockedTokenizer(),
        tokenizer_repo="Qwen/exact",
        tokenizer_revision="a" * 40,
        model_lock_path=lock_path,
    )

    assert identity["revision"] == "a" * 40
    assert identity["tokenizer_manifest_sha256"] == "sha256:manifest"
    assert identity["chat_template_sha256"] == (
        "sha256:" + hashlib.sha256(b"exact template").hexdigest()
    )
    assert identity["vocab_size"] == 248_077
    assert len(identity["verified_files"]) == 2
