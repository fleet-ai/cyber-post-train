"""Select bounded, direct-tool Qwen SFT windows; print aggregate evidence only."""
import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from training import source
from training.dense_bridge import _file_sha, _legacy_digest
def build(raw_root: Path, dense_root: Path, wire_path: Path, output: Path,
          family_cap: int) -> dict:
    if output.exists() or output.is_symlink() or family_cap < 1:
        raise ValueError("output exists or family cap invalid")
    receipt = json.loads((raw_root / "RECEIPT.json").read_text())
    roster = json.loads((raw_root / "family-roster.json").read_text())
    manifest = json.loads((dense_root / "manifest.json").read_text())
    dense_receipt = json.loads((dense_root / "RECEIPT.json").read_text())
    wire = json.loads(wire_path.read_text())
    if (receipt.get("sha256") != source.digest({k: v for k, v in receipt.items() if k != "sha256"})
        or receipt.get("files", {}).get("raw_sources") != _file_sha(raw_root / "raw-sources.private.jsonl")
        or receipt.get("files", {}).get("dense_target_normalized") != _file_sha(raw_root / "dense-target-anchored.jsonl")
        or receipt.get("files", {}).get("roster") != _file_sha(raw_root / "family-roster.json")
        or manifest.get("sha256") != _legacy_digest({k: v for k, v in manifest.items() if k != "sha256"})
        or dense_receipt.get("manifest_sha256") != manifest["sha256"]
        or dense_receipt.get("train_parquet_sha256") != _file_sha(dense_root / "train.parquet")
        or wire.get("sha256") != source.digest({k: v for k, v in wire.items() if k != "sha256"})
        or wire.get("status") != "live_wire_only_not_corpus_acceptance"
        or wire.get("harness", {}).get("name") != "opencode"
        or wire.get("harness", {}).get("version") != "1.18.27"
        or len(wire.get("requests", [])) < 2
        or any(r.get("model_facing_tools_sha256") != source.TOOL_DIGEST or
               r.get("system_sha256") != source.TARGET_SYSTEM_DIGEST for r in wire["requests"])
        or wire.get("release", {}).get("active_pods") != 0):
        raise ValueError("source, dense data, or actual target wire is not bound")
    parquet = pq.ParquetFile(dense_root / "train.parquet")
    if parquet.metadata.num_row_groups != parquet.metadata.num_rows:
        raise ValueError("expected one immutable window per Parquet row group")
    meta = parquet.read(columns=["source_session_id", "source_group_id", "target_token_count",
                                 "window_id", "source_normalized_record_sha256", "source_assistant_count",
                                 "split", "task_key"])
    rows = meta.to_pylist()
    if len(rows) != parquet.metadata.num_rows or any(r["split"] != "train" for r in rows):
        raise ValueError("parent rows or family roles differ")
    identities = {r["source_session_id"] for r in rows}
    normalized = {}
    with (raw_root / "dense-target-anchored.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            if item["record_id"] in identities:
                normalized[item["record_id"]] = source.digest(item)
    if set(normalized) != identities or any(
        r["source_normalized_record_sha256"] != normalized[r["source_session_id"]] for r in rows
    ):
        raise ValueError("parent row lacks exact verified trajectory")
    safe = set()
    selected_models = {}
    session_rows = {}
    for row in rows:
        session_rows.setdefault(row["source_session_id"], []).append(row)
    with (raw_root / "raw-sources.private.jsonl").open() as stream:
        for line in stream:
            item = json.loads(line)
            sid = item["selection"]["session_id"]
            if sid not in identities:
                continue
            role = roster.get(item["selection"]["task_version_id"])
            if (not isinstance(role, dict) or role.get("split") != "train"
                or any(row["source_group_id"] != role.get("family_id")
                       for row in session_rows[sid])):
                raise ValueError("source session crosses the reviewed train role")
            source._validate_envelope(item["selection"], item["summary"], item["transcript_envelope"])
            envelope = item["transcript_envelope"]
            if envelope["harness"]["job_launch_params"].get("harness") is not None:
                continue
            messages = source._training_messages(envelope["transcript"])
            report = source._report_call(messages) if messages else None
            end = next((i for i, m in enumerate(messages or []) if m.get("role") == "tool"
                        and m.get("tool_call_id") == report), None)
            if end is None or source._tool_operations(messages, end) is None:
                continue
            tool_text = [m.get("content") for m in messages[:end + 1] if m.get("role") == "tool"]
            if tool_text and all(isinstance(text, str) and len(text.encode()) <= 50 * 1024
                                 and text.count("\n") + 1 <= 2000 for text in tool_text):
                safe.add(sid)
                selected_models[sid] = item["selection"]["model_id"]
    session_tokens = Counter()
    session_family = {}
    for row in rows:
        sid = row["source_session_id"]
        if sid not in safe:
            continue
        family = row["source_group_id"]
        if sid in session_family and session_family[sid] != family:
            raise ValueError("source session crosses task families")
        session_family[sid] = family
        session_tokens[sid] += row["target_token_count"]
    family_used = Counter()
    selected_sessions = set()
    for sid in sorted(safe, key=lambda s: hashlib.sha256(s.encode()).digest()):
        family = session_family.get(sid)
        if family is not None and family_used[family] + session_tokens[sid] <= family_cap:
            family_used[family] += session_tokens[sid]
            selected_sessions.add(sid)
    selected = {i for i, row in enumerate(rows) if row["source_session_id"] in selected_sessions}
    if not selected or len(family_used) < 50:
        raise ValueError("safe, balanced corpus too small")
    output.mkdir(mode=0o700)
    train_path = output / "train.parquet"
    writer = pq.ParquetWriter(train_path, parquet.schema_arrow, compression="zstd")
    try:
        for index in sorted(selected):
            writer.write_table(parquet.read_row_group(index), row_group_size=1)
    finally:
        writer.close()
    os.chmod(train_path, 0o600)
    chosen = [rows[i] for i in sorted(selected)]
    sessions = {r["source_session_id"] for r in chosen}
    total_assistants = {r["source_session_id"]: r["source_assistant_count"] for r in rows if r["source_session_id"] in sessions}
    full_rows = pq.ParquetFile(train_path).read(columns=["target_spans", "loss_mask", "target_token_count"]).to_pylist()
    assistant_count = sum(len(r["target_spans"]) for r in full_rows)
    if (len(full_rows) != len(chosen) or any(sum(r["loss_mask"]) != r["target_token_count"] for r in full_rows)
        or assistant_count > sum(total_assistants.values())):
        raise ValueError("native masks or assistant counts differ")
    out_manifest = copy.deepcopy(manifest)
    out_manifest["parent_catalog_provenance_sha256"] = source.digest(out_manifest.pop("catalog_provenance"))
    out_manifest["parent_whole_source_exclusions_sha256"] = source.digest(out_manifest.pop("whole_source_exclusions"))
    out_manifest["train_models"] = sorted({selected_models[sid] for sid in sessions})
    out_manifest["files"] = {"train": {"path": "train.parquet", "format": "pretokenized_assistant_segments_v1",
        "storage_layout": "dense_single_row_group_v1", "sha256": _file_sha(train_path), "rows": len(chosen),
        "source_sessions": len(sessions), "supervised_tokens": sum(r["target_token_count"] for r in chosen),
        "assistant_responses": assistant_count, "source_total_assistant_responses": sum(total_assistants.values()),
        "excluded_assistant_responses": sum(total_assistants.values()) - assistant_count,
        "task_keys": sorted({r["task_key"] for r in chosen})}}
    out_manifest["source_sha256"] = source.digest(sorted(sessions))
    out_manifest["subset_provenance"] = {"parent_manifest_sha256": manifest["sha256"],
        "parent_receipt_sha256": dense_receipt["sha256"], "source_receipt_sha256": receipt["sha256"],
        "live_wire_file_sha256": _file_sha(wire_path), "family_cap_target_tokens": family_cap,
        "selection_sha256": source.digest(sorted(rows[i]["window_id"] for i in selected)),
        "result_policy": "direct_native_tool_text_below_opencode_limits_v1"}
    out_manifest["sha256"] = _legacy_digest({k: v for k, v in out_manifest.items() if k != "sha256"})
    (output / "manifest.json").write_text(json.dumps(out_manifest, sort_keys=True) + "\n")
    proof = {"schema": "qwen38_safe96_subset_v1", "manifest_file_sha256": _file_sha(output / "manifest.json"),
             "train_file_sha256": _file_sha(train_path), "parent_manifest_sha256": manifest["sha256"],
             "source_receipt_sha256": receipt["sha256"], "live_wire_file_sha256": _file_sha(wire_path),
             "rows": len(chosen), "sessions": len(sessions), "families": len(family_used),
             "supervised_tokens": out_manifest["files"]["train"]["supervised_tokens"],
             "target_wire": "opencode_1_18_27", "teacher_loss": "not_measured",
             "capability_outcome": "pending_matched_fleet_eval"}
    proof["sha256"] = source.digest(proof)
    (output / "SAFE_SUBSET.json").write_text(json.dumps(proof, sort_keys=True) + "\n")
    return {k: proof[k] for k in ("rows", "sessions", "families", "supervised_tokens", "sha256")}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("raw_root", "dense_root", "wire_path", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--family-cap", type=int, default=200_000)
    args = parser.parse_args()
    print(json.dumps(build(args.raw_root, args.dense_root, args.wire_path, args.output,
                           args.family_cap), sort_keys=True))
