import json
import pytest
from training import projection
from training.dense_bridge import _digest, _file_sha

def test_whole_session_projection_is_replayed(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    roles, identities, rows, proofs = {}, [], [], []
    for i in range(970):
        sid, version = f"synthetic-session-{i}", f"synthetic-version-{i}"
        split = "train" if i < 943 else "dev"
        family = f"synthetic-family-{i % 116}" if split == "train" else "synthetic-dev"
        roles[version] = {"task_key": "synthetic-task", "split": split, "family_id": family}
        identities.append({"task_key": "synthetic-task", "task_version_id": version,
                           "split": split, "group_id": family})
        row = json.dumps({"record_id": sid, "lineage": {"task_key": "synthetic-task",
                          "eval_task_version_id": version}}, sort_keys=True) + "\n"
        proof = json.dumps({"session_id": sid}, sort_keys=True) + "\n"
        rows.append(row)
        proofs.append(proof)
    names = {"dense_target_normalized": "dense-target-anchored.jsonl",
             "dense_success_evidence": "dense-success-evidence.jsonl", "roster": "family-roster.json"}
    for key, content in (("dense_target_normalized", "".join(rows)),
                         ("dense_success_evidence", "".join(reversed(proofs))),
                         ("roster", json.dumps(roles))):
        (source / names[key]).write_text(content)
    (output / "train-only-target.jsonl").write_text("".join(rows[:943]))
    (output / "train-only-evidence.jsonl").write_text("".join(proofs[:943]))
    legacy = tmp_path / "legacy.json"
    authority = {"root_role_anchor_id": projection.ROOT_ID,
                 "family_role_anchor_sha256": projection.ANCHOR_SHA, "identities": identities}
    authority["sha256"] = _digest(authority)
    legacy.write_text(json.dumps(authority))
    receipt = {"training_ready": False, "retained_by_split": {"train": 943, "dev": 27},
               "files": {key: _file_sha(source / name) for key, name in names.items()}}
    receipt["sha256"] = _digest(receipt)
    (source / "RECEIPT.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(projection, "SOURCE_RECEIPT_FILE_SHA", _file_sha(source / "RECEIPT.json"))
    monkeypatch.setattr(projection, "LEGACY_ROSTER_FILE_SHA", _file_sha(legacy))
    sidecar = output / "PROJECTION.json"
    assert projection.seal_projection(source, output, legacy, sidecar)["selected_sessions"] == 943
    assert projection.verify_projection(source, output, legacy, sidecar)["selected_families"] == 116
    selection = output / "selected-whole-sessions.json"
    selection.write_text(json.dumps({"schema": "qwen38_diagnostic_near96k_selection_v1",
                                     "rule": "lexicographic_first20_near90k_minrows8",
                                     "session_ids": [f"synthetic-session-{i}" for i in range(8)]}))
    monkeypatch.setattr(projection, "SELECTION_SHA", _file_sha(selection))
    child = output / "dense-child-v1"
    child.mkdir()
    (child / "train-target.jsonl").write_text("".join(rows[:8]))
    (child / "train-evidence.jsonl").write_text("".join(proofs[:8]))
    request = {"normalized": {"path": str(child / "train-target.jsonl"),
                               "sha256": _file_sha(child / "train-target.jsonl")},
               "evidence": {"path": str(child / "train-evidence.jsonl"),
                            "sha256": _file_sha(child / "train-evidence.jsonl")},
               "family_roster": {"sha256": _file_sha(legacy)}, "output": str(child / "native-output")}
    request["sha256"] = _digest(request)
    (child / "REQUEST.json").write_text(json.dumps(request))
    subset = projection.verify_subset(source, output, legacy, child, selection)
    (child / "SUBSET.json").write_text(json.dumps(subset))
    assert projection.verify_subset(source, output, legacy, child, selection, child / "SUBSET.json") == subset
    (child / "train-evidence.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="bytes differ"):
        projection.verify_subset(source, output, legacy, child, selection, child / "SUBSET.json")
    (output / "train-only-evidence.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="bytes differ"):
        projection.verify_projection(source, output, legacy, sidecar)
