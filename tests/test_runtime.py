"""Synthetic-only tests: no Fleet transcripts, tokens, or credentials."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from training.corpus import sha256
from training.runtime import FORMAT, materialize_parquet


def _render(row):
    tools = row["tools"]
    if isinstance(tools, str):
        tools = json.loads(tools)
    messages = copy.deepcopy(row["messages"])
    for message in messages:
        if isinstance(message.get("tool_calls"), str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        for key in list(message):
            if message[key] is None:
                del message[key]
    body = json.dumps([messages, tools], sort_keys=True, separators=(",", ":"))
    target = json.dumps(messages[-1], sort_keys=True, separators=(",", ":"))
    return [0, *body.encode()], len(target.encode())


def _source(root: Path, *, overlap=False):
    root.mkdir()
    tools = [{"type": "function", "function": {"name": "fleet_bash", "parameters": {}}},
             {"type": "function", "function": {"name": "fleet_submit_report", "parameters": {}}}]
    inputs = {"tools": sha256(tools), "roster": sha256({"family split": 1})}
    token_sha = "sha256:" + "1" * 64
    partitions = {}
    for split in ("train", "dev"):
        messages = [{"role": "system", "content": "Synthetic only"},
                    {"role": "user", "content": "Synthetic task"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "call-1", "type": "function", "function": {
                            "name": "fleet_bash", "arguments": {"command": "true"}}}]}]
        row = {"messages": messages, "tools": tools, "target_message_index": 2,
               "target_id": f"{split}-target", "task_version_id": f"{split}-version",
               "family_id": "shared" if overlap else split, "source_sha256": sha256(split)}
        ids, actions = _render(row)
        row.update(total_tokens=len(ids), supervised_tokens=actions)
        path = root / f"{split}.jsonl"
        raw = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        path.write_bytes(raw)
        proof = {"split": split, "rows": 1, "rows_sha256": sha256([row]),
                 "tokenizer_sha256": token_sha, "inputs_sha256": inputs}
        proof["sha256"] = sha256(proof)
        partitions[split] = {"path": path.name, "bytes": len(raw),
                             "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                             "receipt": proof}
    receipt = {"format": "structured_message_windows_v1", "trainer_ready": False,
               "partitions": partitions}
    receipt["sha256"] = sha256(receipt)
    (root / "RECEIPT.json").write_text(json.dumps(receipt))
    return token_sha


def test_materialize_tool_aware_parquet_and_refuse_replacement(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "ready"
    token_sha = _source(source)
    manifest = materialize_parquet(source, destination, tokenizer_root=tmp_path,
                                   _test_tokenize=_render, _test_tokenizer_sha256=token_sha)
    assert manifest["trainer_ready"] is False  # Test doubles cannot authorize a launch.
    assert manifest["validation_mode"] == "teacher_cross_entropy"
    assert manifest["files"]["train"]["format"] == FORMAT
    assert manifest["files"]["dev"]["rows"] == 1
    with pytest.raises(FileExistsError):
        materialize_parquet(source, destination, tokenizer_root=tmp_path,
                            _test_tokenize=_render, _test_tokenizer_sha256=token_sha)


def test_materialize_rejects_family_leak_and_source_drift(tmp_path):
    source = tmp_path / "source"
    token_sha = _source(source, overlap=True)
    with pytest.raises(ValueError, match="family overlap"):
        materialize_parquet(source, tmp_path / "ready", tokenizer_root=tmp_path,
                            _test_tokenize=_render, _test_tokenizer_sha256=token_sha)
    assert not (tmp_path / "ready").exists()
    (source / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="partition bytes"):
        materialize_parquet(source, tmp_path / "ready", tokenizer_root=tmp_path,
                            _test_tokenize=_render, _test_tokenizer_sha256=token_sha)
