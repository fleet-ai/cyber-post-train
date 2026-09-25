"""Replay the provisional, whole-session TRAIN projection; never certify training data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .dense_bridge import _digest, _file_sha

SCHEMA = "qwen38_diagnostic_train_projection_v1"
SOURCE_RECEIPT_FILE_SHA = "sha256:ffb3a687c4f1a325d2000dec30855f90cf6166acf7e03c7c6e13acbc5cf9a54a"
LEGACY_ROSTER_FILE_SHA = "sha256:2fa2e62adb9785355e86b44225040b67dbc79301e0b64cda8545a77c6e628c31"
ROOT_ID = "fleet-q38-teacher3k-transitive-roles-20260925-v1"
ANCHOR_SHA = "sha256:5b4d3d959d599a14235f0ecfdd8d8e699584424af4dd0b267440f388c0c2d224"
BUILDER_SHA = "sha256:365657758efcd55220be80d556dd83257d7d80a6ff77f9a308b1da9096511ebd"
SELECTION_SHA = "sha256:2be71521f4af4b0b8af78d276610a66647e40cde31e46ac0a6ea63f532dac675"
SELECTOR_CODE_SHA = "sha256:1eecda755665becf5475538fd3671b479ce27f7dc5210381d697021670c2260e"
SUBSET_BUILDER_SHA = "sha256:b43fd0ef20b3a528983c9ee2758a43d4f92fb4e8f4c795d865900028d137174d"

def _bound(path: Path, sha: str) -> None:
    if path.is_symlink() or not path.is_file() or _file_sha(path) != sha:
        raise ValueError("projection input or output bytes differ")


def verify_projection(source: Path, projected: Path, legacy: Path,
                      sidecar: Path | None = None) -> dict:
    """Reconstruct selected bytes from a pinned sealed source, then check outputs."""
    source, projected, legacy = Path(source), Path(projected), Path(legacy)
    _bound(source / "RECEIPT.json", SOURCE_RECEIPT_FILE_SHA)
    _bound(legacy, LEGACY_ROSTER_FILE_SHA)
    receipt, identities = json.loads((source / "RECEIPT.json").read_text()), json.loads(legacy.read_text())
    if (receipt.get("sha256") != _digest({k: v for k, v in receipt.items() if k != "sha256"})
            or receipt.get("training_ready") is not False or receipt.get("retained_by_split") != {"train": 943, "dev": 27}
            or identities.get("sha256") != _digest({k: v for k, v in identities.items() if k != "sha256"})
            or identities.get("root_role_anchor_id") != ROOT_ID
            or identities.get("family_role_anchor_sha256") != ANCHOR_SHA):
        raise ValueError("sealed source or family authority differs")
    names = {"dense_target_normalized": "dense-target-anchored.jsonl",
             "dense_success_evidence": "dense-success-evidence.jsonl", "roster": "family-roster.json"}
    for key, name in names.items():
        _bound(source / name, receipt["files"][key])
    roles = json.loads((source / names["roster"]).read_text())
    legacy_roles = {item["task_version_id"]: item for item in identities["identities"]}
    if len(legacy_roles) != len(identities["identities"]):
        raise ValueError("duplicate task versions in family authority")
    proofs = {}
    with (source / names["dense_success_evidence"]).open("rb") as stream:
        for line in stream:
            sid = json.loads(line)["session_id"]
            if sid in proofs:
                raise ValueError("duplicate source evidence session")
            proofs[sid] = line
    normal_hash, proof_hash = hashlib.sha256(), hashlib.sha256()
    all_ids, selected, families, role_rows = set(), [], set(), {"train": 0, "dev": 0}
    with (source / names["dense_target_normalized"]).open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            sid, lineage = row["record_id"], row["lineage"]
            version = lineage["eval_task_version_id"]
            role, old = roles[version], legacy_roles[version]
            if (sid in all_ids or sid not in proofs or role["task_key"] != lineage["task_key"]
                    or old["task_key"] != lineage["task_key"] or old["group_id"] != role["family_id"]
                    or old["split"] != role["split"] or role["split"] not in role_rows):
                raise ValueError("source session, proof, or family role differs")
            all_ids.add(sid)
            role_rows[role["split"]] += 1
            if role["split"] == "train":
                normal_hash.update(line)
                proof_hash.update(proofs[sid])
                selected.append((sid, version, role["family_id"]))
                families.add(role["family_id"])
    if (all_ids != set(proofs) or role_rows != receipt["retained_by_split"]
            or len(selected) != 943 or len(families) != 116):
        raise ValueError("projection omits a source session or includes DEV")
    expected = {"normalized": "sha256:" + normal_hash.hexdigest(),
                "evidence": "sha256:" + proof_hash.hexdigest()}
    _bound(projected / "train-only-target.jsonl", expected["normalized"])
    _bound(projected / "train-only-evidence.jsonl", expected["evidence"])
    proof = {"schema": SCHEMA, "source_receipt_file_sha256": SOURCE_RECEIPT_FILE_SHA,
             "source_receipt_sha256": receipt["sha256"],
             "source_files_sha256": {key: receipt["files"][key] for key in names},
             "legacy_roster_file_sha256": LEGACY_ROSTER_FILE_SHA,
             "family_role_anchor_sha256": ANCHOR_SHA, "projection_builder_sha256": BUILDER_SHA,
             "projected_files_sha256": expected, "selected_sessions": len(selected),
             "selected_families": len(families), "selected_identity_set_sha256": _digest(sorted(selected)),
             "whole_sessions": True, "diagnostic_only": True, "training_ready": False}
    proof["sha256"] = _digest(proof)
    if sidecar is not None:
        sidecar = Path(sidecar)
        if sidecar.is_symlink() or not sidecar.is_file() or json.loads(sidecar.read_text()) != proof:
            raise ValueError("projection sidecar does not match independent replay")
    return proof


def seal_projection(source: Path, projected: Path, legacy: Path, sidecar: Path) -> dict:
    """Create a private immutable sidecar only after independent replay."""
    proof = verify_projection(source, projected, legacy)
    fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(proof, stream, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        stream.write("\n")
    verify_projection(source, projected, legacy, sidecar)
    return proof


def verify_subset(source: Path, full: Path, legacy: Path, child: Path,
                  selection: Path, sidecar: Path | None = None) -> dict:
    """Replay a fixed eight-session child from the verified whole TRAIN projection."""
    full, child, selection = Path(full), Path(child), Path(selection)
    parent_file = full / "PROJECTION.json"
    parent = verify_projection(source, full, legacy, parent_file)
    _bound(selection, SELECTION_SHA)
    chosen = json.loads(selection.read_text())
    ids = chosen.get("session_ids")
    if (chosen.get("schema") != "qwen38_diagnostic_near96k_selection_v1"
            or chosen.get("rule") != "lexicographic_first20_near90k_minrows8"
            or not isinstance(ids, list) or len(ids) != len(set(ids)) or len(ids) != 8):
        raise ValueError("diagnostic selection differs")
    proofs = {}
    with (full / "train-only-evidence.jsonl").open("rb") as stream:
        for line in stream:
            sid = json.loads(line)["session_id"]
            if sid in proofs:
                raise ValueError("duplicate full-projection proof")
            proofs[sid] = line
    normal_hash, proof_hash, found = hashlib.sha256(), hashlib.sha256(), set()
    with (full / "train-only-target.jsonl").open("rb") as stream:
        for line in stream:
            sid = json.loads(line)["record_id"]
            if sid in ids:
                if sid in found or sid not in proofs:
                    raise ValueError("selected session is incomplete")
                found.add(sid)
                normal_hash.update(line)
                proof_hash.update(proofs[sid])
    if found != set(ids):
        raise ValueError("selected session is missing")
    expected = {"normalized": "sha256:" + normal_hash.hexdigest(),
                "evidence": "sha256:" + proof_hash.hexdigest()}
    _bound(child / "train-target.jsonl", expected["normalized"])
    _bound(child / "train-evidence.jsonl", expected["evidence"])
    request_file = child / "REQUEST.json"
    if request_file.is_symlink() or not request_file.is_file():
        raise ValueError("child request is missing or linked")
    request = json.loads(request_file.read_text())
    if (request.get("sha256") != _digest({k: v for k, v in request.items() if k != "sha256"})
            or any(request.get(key) != {"path": str(child / name), "sha256": expected[key]}
                   for key, name in (("normalized", "train-target.jsonl"), ("evidence", "train-evidence.jsonl")))
            or request.get("family_roster", {}).get("sha256") != _file_sha(Path(legacy))
            or request.get("output") != str(child / "native-output")):
        raise ValueError("child native request differs")
    proof = {"schema": "qwen38_diagnostic_whole_train_subset_v1",
             "source_receipt_file_sha256": parent["source_receipt_file_sha256"],
             "full_projection_file_sha256": _file_sha(parent_file), "full_projection_sha256": parent["sha256"],
             "selection_file_sha256": SELECTION_SHA, "selector_code_sha256": SELECTOR_CODE_SHA,
             "subset_builder_code_sha256": SUBSET_BUILDER_SHA,
             "selected_identity_set_sha256": _digest(sorted(ids)), "selected_sessions": len(ids),
             "child_files_sha256": expected, "child_request_file_sha256": _file_sha(request_file),
             "child_request_sha256": request["sha256"], "whole_sessions": True,
             "diagnostic_only": True, "training_ready": False}
    proof["sha256"] = _digest(proof)
    if sidecar is not None and (Path(sidecar).is_symlink() or not Path(sidecar).is_file()
                                or json.loads(Path(sidecar).read_text()) != proof):
        raise ValueError("child sidecar differs from independent replay")
    return proof
