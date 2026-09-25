"""Audit new target-anchor teacher data; historical code packs only token windows."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import source
from .dense_bridge import (_digest, _file_sha, _legacy_digest, build_dense, validate_request,
                           ALGORITHM as MECHANICS, MANIFEST_SCHEMA, RECEIPT_SCHEMA, TARGET_BUILDER_SHA,
                           TARGET_METHOD, LAYOUT, LAYOUT_BUILDER_SHA)

METHOD = TARGET_METHOD
LAZY_METHOD = "opencode_1_18_27_target_anchor_visible_only_multi_target_rowgroups_v1"
SCHEMA = "qwen38_target_anchor_visible_only_method_v1"
FILES = {"raw_sources": "raw-sources.private.jsonl", "dense_target_normalized": "dense-target-anchored.jsonl",
         "dense_success_evidence": "dense-success-evidence.jsonl", "tools": "model-facing-tools.json",
         "capture": "model-request-capture.json"}


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(source_dir: Path) -> dict:
    """Independently replay each raw→visible→target transformation."""
    source_dir = Path(source_dir)
    receipt_file = source_dir / "RECEIPT.json"
    if receipt_file.is_symlink() or not receipt_file.is_file():
        raise ValueError("private source receipt is missing or linked")
    receipt = json.loads(receipt_file.read_text())
    if (receipt.get("schema") != "fleet_teacher_source_receipt_v1"
            or receipt.get("sha256") != _digest({k: v for k, v in receipt.items() if k != "sha256"})
            or receipt.get("anchor_method") != source.ANCHOR_METHOD
            or receipt.get("visibility_method") != "visible_only_assistant_content_v1"
            or receipt.get("model_facing_tools_sha256") != source.TOOL_DIGEST
            or receipt.get("download_complete") is not True):
        raise ValueError("private source receipt is not the target-visible method")
    paths = {key: source_dir / name for key, name in FILES.items()}
    if any(p.is_symlink() or not p.is_file() or _file_sha(p) != receipt.get("files", {}).get(key)
           for key, p in paths.items()):
        raise ValueError("bound private source file differs")
    request_path = source_dir / "REQUEST.json"
    if (request_path.is_symlink() or not request_path.is_file()
            or _file_sha(request_path) != receipt.get("files", {}).get("request")):
        raise ValueError("bound source request differs")
    request = json.loads(request_path.read_text())
    live_binding = request.get("live_model_request_attestation")
    if (request.get("sha256") != source.digest({k: v for k, v in request.items() if k != "sha256"})
            or request["sha256"] != receipt.get("request_sha256")
            or receipt.get("input_sha256", {}).get("live_model_request_attestation")
            != receipt.get("files", {}).get("live_model_request_attestation")
            or not isinstance(live_binding, dict)
            or live_binding.get("sha256") != receipt.get("files", {}).get("live_model_request_attestation")):
        raise ValueError("live served-request binding differs")
    live = source._bound_file(live_binding)
    source._sealed(live, "cyber_qwen_live_model_request_attestation_v1")
    calls, bindings = live.get("requests"), request.get("target_anchors")
    model, proof = live.get("model"), live.get("serving_proof")
    roster_path = source_dir / "family-roster.json"
    if (roster_path.is_symlink() or not roster_path.is_file()
            or _file_sha(roster_path) != receipt.get("files", {}).get("roster")):
        raise ValueError("bound family roster differs")
    roster = json.loads(roster_path.read_text())
    capture = json.loads(paths["capture"].read_text())
    if (not isinstance(calls, list) or not isinstance(bindings, list) or len(calls) != len(bindings)
            or len(calls) < source.MIN_ANCHOR_PROBES or live.get("harness") != source.HARNESS
            or not isinstance(model, dict) or model.get("repo") != "Qwen/Qwen3.8-27B"
            or model.get("revision") != request.get("model_revision")
            or not source._sha(model.get("image_digest"))
            or any(not model.get(key) for key in ("served_alias", "inference_model_uid", "pod_uid"))
            or live.get("fleet", {}).get("team_id") != source.TEAM
            or live.get("tool_capture_file_sha256") != receipt["input_sha256"]["tool_capture"]
            or live.get("fleet", {}).get("model_facing_tools_sha256") != source.TOOL_DIGEST
            or not isinstance(proof, dict) or proof.get("kind") != "post_fixed_proxy_live_served_request"
            or proof.get("runner_cwd") != "/workspace"
            or proof.get("served_alias") != model["served_alias"]
            or not source._sha(proof.get("fixed_proxy_image_sha256"))
            or len(proof.get("request_ids", [])) != len(calls)
            or len(set(proof["request_ids"])) != len(calls)
            or receipt.get("live_attested_train_versions") != [call.get("task_version_id") for call in calls]
            or receipt.get("live_wire_review") != "pending_independent_readback"
            or receipt.get("tool_result_equivalence") != "unverified"):
        raise ValueError("live served-request evidence differs")
    for call, binding in zip(calls, bindings):
        anchor = source._bound_file(binding)
        source._sealed(anchor, "cyber_opencode_private_target_anchor_v1")
        if (call.get("target_anchor_file_sha256") != binding["sha256"]
                or call.get("task_version_id") != anchor.get("task_version_id")
                or roster.get(call["task_version_id"], {}).get("split") != "train"
                or call.get("request_envelope_sha256") != anchor.get("request_envelope_sha256")
                or call.get("system_sha256") != source.text_digest(anchor["messages"][0]["content"])
                or call.get("user_sha256") != source.text_digest(anchor["messages"][1]["content"])
                or call.get("model_facing_tools_sha256") != source.TOOL_DIGEST
                or not source._sha(call.get("response_sha256"))):
            raise ValueError("live TRAIN-version anchor differs")
    if calls[0]["request_envelope_sha256"] != capture.get("request_envelope_sha256"):
        raise ValueError("live first-request capture differs")
    raw_rows, target, proof_rows = (_rows(paths[key]) for key in
                                    ("raw_sources", "dense_target_normalized", "dense_success_evidence"))
    raw = {row["selection"]["session_id"]: row for row in raw_rows}
    proofs = {row["session_id"]: row for row in proof_rows}
    exact_tools = json.loads(paths["tools"].read_text())
    if (not target or len(raw) != len(raw_rows) or len(raw) < len(target)
            or len(target) != len(proofs) or len(proofs) != len(proof_rows)
            or {row.get("record_id") for row in target} != set(proofs)
            or len(target) != receipt.get("retained_sessions")
            or source.digest(exact_tools, ascii=True) != source.TOOL_DIGEST):
        raise ValueError("source, target, and proof identities differ")
    observed, hidden_total = [], 0
    for row in target:
        sid = row["record_id"]
        if sid not in raw or "discovery_transform" not in row:
            raise ValueError("target row lacks raw source or discovery state")
        entry, proof = raw[sid], proofs[sid]
        selection, envelope = entry["selection"], entry["transcript_envelope"]
        source._validate_envelope(selection, entry["summary"], envelope)
        original = envelope["transcript"]
        visible = source._training_messages(original)
        prompt = envelope["task"]["prompt"]
        if (visible is None or len(visible) < 4 or not isinstance(prompt, str)
                or source._text_content(original[1].get("content")) != prompt):
            raise ValueError("source has no exact visible task anchor")
        report = source._report_call(visible)
        end = next((i for i, msg in enumerate(visible)
                    if msg.get("role") == "tool" and msg.get("tool_call_id") == report), None)
        if end is None:
            raise ValueError("successful report is not a complete tool round")
        if row["discovery_transform"] is not None:
            raise ValueError("unreviewed tool discovery remains")
        checked = source._tool_operations(visible, end)
        if checked is None:
            raise ValueError("source calls cannot map to exact target calls")
        operations, expected = checked
        actual = row.get("messages")
        if not isinstance(actual, list) or len(actual) < 4:
            raise ValueError("target messages are absent")
        expected[0] = {**expected[0], "content": actual[0].get("content")}
        expected[1] = {**expected[1], "content": source._target_user(prompt)}
        if (actual != expected or source.text_digest(actual[0]["content"]) != source.TARGET_SYSTEM_DIGEST
                or any(call["function"]["name"] not in {"fleet_bash", "fleet_submit_report"}
                       for msg in actual for call in msg.get("tool_calls") or [])):
            raise ValueError("model-visible target rounds or anchors differ")
        system, user = (source._text_content(msg["content"]) for msg in original[:2])
        anchor = {"schema": "fleet_opencode_anchor_substitution_v1", "method": source.ANCHOR_METHOD,
                  "source_trace_sha256": source.digest(envelope),
                  "source_system_sha256": source.text_digest(system),
                  "source_user_sha256": source.text_digest(user),
                  "source_task_prompt_sha256": source.text_digest(prompt),
                  "target_system_sha256": source.text_digest(actual[0]["content"]),
                  "target_user_sha256": source.text_digest(actual[1]["content"]),
                  "probe_file_sha256": receipt["target_anchor_probe_file_sha256"]}
        anchor["sha256"] = source.digest(anchor)
        hidden = sum(key in msg for msg in original for key in ("thinking", "reasoning", "reasoning_content", "analysis"))
        visibility = {"schema": "fleet_visible_only_assistant_transform_v1",
                      "original_messages_sha256": source.digest(original),
                      "visible_messages_sha256": source.digest(visible),
                      "hidden_reasoning_fields_removed": hidden}
        visibility["sha256"] = source.digest(visibility)
        tools = {"schema": "fleet_source_tool_contract_check_v1",
                 "source_messages_sha256": source.digest(visible),
                 "accepted_prefix_result_index": end,
                 "operations": operations, "target_tool_schema_sha256": source.TOOL_DIGEST,
                 "target_messages_sha256": source.digest(actual)}
        tools["sha256"] = source.digest(tools)
        if (row.get("schema") != "fleet_cyber_trajectory_v1"
                or row.get("content_digest") != source.digest({k: v for k, v in row.items() if k != "content_digest"})
                or row.get("lineage") != {"task_key": selection["task_key"],
                                           "eval_task_version_id": selection["task_version_id"]}
                or row.get("anchor_transform") != anchor or row.get("visibility_transform") != visibility
                or row.get("tool_transform") != tools
                or proof.get("sha256") != source.digest({k: v for k, v in proof.items() if k != "sha256"})
                or proof.get("normalized_record_sha256") != source.digest(row)
                or proof.get("transcript_sha256") != selection["trace_sha256"]
                or proof.get("successful_report_call_id") != report
                or proof.get("verifier_execution_id") != envelope["verifier_execution"]["id"]
                or proof.get("outcome") != {"status": "completed", "verifier_process_success": True,
                                             "score_at_least_one": True}):
            raise ValueError("target transformation or verifier evidence changed")
        observed.append((sid, anchor["sha256"], visibility["sha256"], tools["sha256"]))
        hidden_total += hidden
    if receipt.get("exact_discovery_elided_sessions") != 0:
        raise ValueError("source discovery-elision total differs")
    result = {"schema": SCHEMA, "method": METHOD, "source_receipt_sha256": receipt["sha256"],
              "source_files_sha256": {key: receipt["files"][key] for key in FILES},
              "session_transform_set_sha256": _digest(sorted(observed)), "sessions": len(target),
              "hidden_reasoning_fields_removed": hidden_total,
              "exact_discovery_elided_sessions": 0,
              "trainer_ready": False, "training_blocker": receipt.get("training_blocker")}
    result["sha256"] = _digest(result)
    return result


def build(request_path: Path, source_dir: Path, output: Path, *, single_row_groups: bool = False) -> dict:
    """Run exact CPU-native mechanics and seal a distinct semantic method."""
    request_path, source_dir, output = Path(request_path).absolute(), Path(source_dir), Path(output)
    reviewed, request = audit(source_dir), validate_request(request_path)
    requested = Path(request["output"])
    requested = requested if requested.is_absolute() else request_path.parent / requested
    if not output.is_absolute() or requested.resolve() != output.resolve():
        raise ValueError("request output differs from target method destination")
    source_receipt = json.loads((source_dir / "RECEIPT.json").read_text())
    for key, name in (("normalized", "dense_target_normalized"), ("evidence", "dense_success_evidence"),
                      ("model_request_capture", "capture")):
        bound = Path(request[key]["path"])
        bound = bound if bound.is_absolute() else request_path.parent / bound
        if (bound.resolve() != (source_dir / FILES[name]).resolve()
                or request[key]["sha256"] != source_receipt["files"][name]):
            raise ValueError("CPU mechanics request binds another source")
    packed = build_dense(request_path, target_names=True, **({"single_row_groups": True}
                                                          if single_row_groups else {}))
    method = {"schema": SCHEMA, "method": LAZY_METHOD if single_row_groups else METHOD,
              "source_audit_sha256": reviewed["sha256"],
              "source_receipt_sha256": reviewed["source_receipt_sha256"],
              "request_sha256": request["sha256"], "mechanics_builder_sha256": (
                  LAYOUT_BUILDER_SHA if single_row_groups else TARGET_BUILDER_SHA),
              **({"storage_layout": LAYOUT} if single_row_groups else {}),
              "mechanics_manifest_sha256": packed["manifest_sha256"],
              "mechanics_receipt_sha256": packed["receipt_sha256"],
              "train_parquet_sha256": packed["train_sha256"],
              "trainer_ready": reviewed["trainer_ready"], "training_blocker": reviewed["training_blocker"]}
    method["sha256"] = _legacy_digest(method)
    with os.fdopen(os.open(output / "TARGET-METHOD.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(method, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return {"method_sha256": method["sha256"], "rows": packed["rows"],
            "supervised_tokens": packed["supervised_tokens"], "trainer_ready": False}


def verify_method(dense_dir: Path, source_dir: Path) -> dict:
    """Check semantic sidecar, immutable Parquet, and unique masked targets."""
    import pyarrow.parquet as pq
    dense_dir, reviewed = Path(dense_dir), audit(source_dir)
    def load(name: str, schema: str) -> dict:
        path = dense_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("target publication is incomplete or linked")
        value = json.loads(path.read_text())
        if (value.get("schema") != schema or value.get("sha256") !=
                _legacy_digest({k: v for k, v in value.items() if k != "sha256"})):
            raise ValueError("target publication seal differs")
        return value
    method = load("TARGET-METHOD.json", SCHEMA)
    manifest = load("manifest.json", MANIFEST_SCHEMA)
    receipt = load("RECEIPT.json", RECEIPT_SCHEMA)
    train, parquet = manifest.get("files", {}).get("train", {}), dense_dir / "train.parquet"
    material = manifest.get("materialization", {})
    lazy = method.get("method") == LAZY_METHOD
    builder = LAYOUT_BUILDER_SHA if lazy else TARGET_BUILDER_SHA
    if (method.get("method") not in {METHOD, LAZY_METHOD}
            or method.get("storage_layout") != (LAYOUT if lazy else None)
            or train.get("storage_layout") != (LAYOUT if lazy else None)
            or method.get("trainer_ready") is not reviewed["trainer_ready"]
            or method.get("source_audit_sha256") != reviewed["sha256"]
            or method.get("source_receipt_sha256") != reviewed["source_receipt_sha256"]
            or method.get("training_blocker") != reviewed["training_blocker"]
            or method.get("mechanics_builder_sha256") != builder
            or method.get("mechanics_manifest_sha256") != manifest["sha256"]
            or method.get("mechanics_receipt_sha256") != receipt["sha256"]
            or method.get("train_parquet_sha256") != train.get("sha256")
            or manifest.get("algorithm") != MECHANICS
            or manifest.get("builder_sha256", {}).get("message_aligned_teacher_corpus.py") != builder
            or material.get("request_sha256") != method.get("request_sha256")
            or material.get("normalized_sha256") != reviewed["source_files_sha256"]["dense_target_normalized"]
            or material.get("success_evidence_sha256") != reviewed["source_files_sha256"]["dense_success_evidence"]
            or material.get("model_request_capture_sha256") != reviewed["source_files_sha256"]["capture"]
            or receipt.get("manifest_sha256") != manifest["sha256"]
            or receipt.get("manifest_file_sha256") != _file_sha(dense_dir / "manifest.json")
            or receipt.get("train_parquet_sha256") != train.get("sha256")
            or receipt.get("rows") != train.get("rows")
            or receipt.get("supervised_tokens") != train.get("supervised_tokens")
            or train.get("format") != "pretokenized_assistant_segments_v1"
            or not isinstance(train.get("rows"), int) or train["rows"] < 1
            or parquet.is_symlink() or not parquet.is_file() or _file_sha(parquet) != train["sha256"]):
        raise ValueError("new target method differs from audited CPU mechanics")
    columns = ["source_session_id", "target_spans", "target_token_count", "loss_mask", "window_algorithm"]
    if lazy:
        reader = pq.ParquetFile(parquet)
        if (reader.num_row_groups != train["rows"] or any(
                reader.metadata.row_group(i).num_rows != 1 for i in range(reader.num_row_groups))):
            raise ValueError("target Parquet row-group layout differs")
        rows = (row for batch in reader.iter_batches(batch_size=1, columns=columns)
                for row in batch.to_pylist())
    else:
        rows = pq.read_table(parquet, columns=columns).to_pylist()
    seen, count = set(), 0
    for row in rows:
        spans = row["target_spans"]
        unique = {(row["source_session_id"], span["assistant_index"]) for span in spans}
        tokens = sum(span["token_end"] - span["token_start"] for span in spans)
        if (len(unique) != len(spans) or seen & unique or row["window_algorithm"] != MECHANICS
                or tokens != row["target_token_count"] or tokens != sum(row["loss_mask"])):
            raise ValueError("target token mask duplicates or differs")
        seen.update(unique)
        count += tokens
    if (reader.metadata.num_rows if lazy else len(rows)) != train["rows"] or count != train["supervised_tokens"]:
        raise ValueError("target token count differs from receipt")
    return {"method_sha256": method["sha256"], "manifest_sha256": manifest["sha256"],
            "trainer_ready": reviewed["trainer_ready"], "supervised_tokens": count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "build", "verify"))
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("request_or_dense_dir", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--single-row-groups", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = (audit(args.source_dir) if args.command == "audit" else
                  build(args.request_or_dense_dir, args.source_dir, args.output,
                        single_row_groups=args.single_row_groups) if args.command == "build" else
                  verify_method(args.request_or_dense_dir, args.source_dir))
    except Exception as error:
        print(f"target method {args.command} rejected: {type(error).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
