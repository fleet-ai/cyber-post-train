"""Synthetic-only tests: no Fleet task text, transcripts, or credentials."""

import json

import pytest

from training import teacher_dev as dev
from training.dense_bridge import _digest, _file_sha


def _fixture(tmp_path, monkeypatch, *, accepted=True, overlap=False):
    source = tmp_path / "source"
    source.mkdir()
    tools = [{"type": "function", "function": {"name": "fleet_bash"}},
             {"type": "function", "function": {"name": "fleet_submit_report"}}]
    messages = [{"role": "system", "content": "Synthetic system"},
                {"role": "user", "content": "Synthetic task"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "fleet_bash"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "Synthetic result"},
                {"role": "assistant", "content": "Synthetic analysis"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "fleet_submit_report"}}]},
                {"role": "tool", "tool_call_id": "c2", "content": "Synthetic done"}]
    rows = [{"record_id": f"synthetic-{split}", "content_digest": _digest(split),
             "lineage": {"task_key": f"task-{split}", "eval_task_version_id": split},
             "messages": messages} for split in ("train", "dev")]
    roster = {split: {"task_key": f"task-{split}",
                      "family_id": "train" if overlap else split, "split": split}
              for split in ("train", "dev")}
    for name, value in (("family-roster.json", roster), ("model-facing-tools.json", tools)):
        (source / name).write_text(json.dumps(value))
    (source / "dense-target-anchored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = {"schema": "fleet_teacher_source_receipt_v1", "training_ready": accepted,
               "training_blocker": None if accepted else "live parity pending",
               "input_sha256": {"roles": dev.FROZEN_ROLES_FILE_SHA},
               "model_facing_tools_sha256": _digest(tools),
               "retained_by_split": {"train": 1, "dev": 1},
               "files": {key: _file_sha(source / name) for key, name in (
                   ("roster", "family-roster.json"), ("tools", "model-facing-tools.json"),
                   ("dense_target_normalized", "dense-target-anchored.jsonl"))}}
    receipt["sha256"] = _digest(receipt)
    (source / "RECEIPT.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(dev.target_dense, "audit", lambda _path: {
        "trainer_ready": accepted, "training_blocker": receipt["training_blocker"],
        "source_receipt_sha256": receipt["sha256"], "sha256": _digest("synthetic-audit")})
    monkeypatch.setattr(dev, "load_pinned_tokenizer", lambda _path: object())

    def tokenize(row, _tokenizer):
        messages = [{k: (json.loads(v) if k == "tool_calls" and isinstance(v, str) else v)
                     for k, v in message.items() if v is not None} for message in row["messages"]]
        tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        body = json.dumps([messages, tools], sort_keys=True, separators=(",", ":")).encode()
        target = json.dumps(messages[-1], sort_keys=True, separators=(",", ":")).encode()
        return [0, *body], len(target)

    monkeypatch.setattr(dev, "pinned_tokenize", tokenize)
    return source


def test_accepted_dev_is_family_disjoint_and_has_no_terminal_report(tmp_path, monkeypatch):
    source = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "dev"
    result = dev.build(source, tmp_path, output, max_length=4096, max_input_tokens=8192)
    assert result["rows"] == 2 and result["families"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["files"]["dev"]["task_keys"] == ["task-dev"]
    assert manifest["trainer_ready"] is True and manifest["sha256"] == result["manifest_sha256"]
    import pyarrow.parquet as pq
    rows = pq.read_table(output / "dev.parquet").to_pylist()
    assert {r["source_message_index"] for r in rows} == {2, 4}
    assert all(r["messages"][0]["content"] == "Synthetic system" for r in rows)
    with pytest.raises(ValueError, match="unsafe DEV"):
        dev.build(source, tmp_path, output, max_length=4096, max_input_tokens=8192)


@pytest.mark.parametrize("accepted,overlap,reason", [
    (False, False, "not accepted"), (True, True, "family crosses")])
def test_rejects_provisional_or_leaking_source(tmp_path, monkeypatch, accepted, overlap, reason):
    source = _fixture(tmp_path, monkeypatch, accepted=accepted, overlap=overlap)
    with pytest.raises(ValueError, match=reason):
        dev.build(source, tmp_path, tmp_path / "dev", max_length=4096, max_input_tokens=8192)
    assert not (tmp_path / "dev").exists()


def test_rejects_dev_panel_over_budget_before_writing(tmp_path, monkeypatch):
    source = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="input-token budget"):
        dev.build(source, tmp_path, tmp_path / "dev", max_length=4096, max_input_tokens=1)
    assert not (tmp_path / "dev").exists()
