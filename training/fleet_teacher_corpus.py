"""Build a train-only dense corpus from exact Fleet success evidence.

The network census and transcript export happen separately.  This builder is
deliberately offline: it accepts digest-bound normalized records and evidence,
then admits only exact versions in the frozen training split.
"""

from __future__ import annotations

import collections
import copy
import math
import os
from pathlib import Path

from cyber_post_train.jobs import digest

from .corpus import local_tokenizer
from .dense import Excluded, compatible_messages, encode_record, native_helper, segment_record
from .io import atomic_write_json, atomic_write_jsonl, digest_json, file_sha256, iter_jsonl
from .sft import _known, read_mapping
from .sft_runtime import DENSE_FORMAT, dense_rows
from .study_split_v2 import SCHEMA as SPLIT_SCHEMA
from .study_split_v2 import validate as validate_split

SCHEMA = "cyber_fleet_teacher_corpus_request_v1"
TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
RECORD_SCHEMA = "fleet_cyber_trajectory_v1"
EVIDENCE_SCHEMA = "fleet_catalog_success_evidence_v1"
FINALIZE_POLICY = (
    "strip exact two-message submit_final_answer suffix only after completed "
    "submit_report; original trace digest retained"
)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sealed(value: dict, field: str = "sha256") -> bool:
    expected = "sha256:" + digest({key: item for key, item in value.items() if key != field})
    return value.get(field) == expected


def _strip_finalize_suffix(record: dict) -> tuple[dict, str | None, bool]:
    """Remove only Fleet's redundant final wrapper, never an exploit/report turn."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        return record, None, False
    assistant, result = messages[-2:]
    calls = assistant.get("tool_calls") if isinstance(assistant, dict) else None
    exact = bool(
        assistant.get("role") == "assistant"
        and isinstance(calls, list)
        and len(calls) == 1
        and (calls[0].get("function") or {}).get("name") == "submit_final_answer"
        and isinstance(calls[0].get("id"), str)
        and result.get("role") == "tool"
        and result.get("tool_call_id") == calls[0]["id"]
    )
    if not exact:
        return record, None, False
    retained = messages[:-2]
    report_ids = {
        call.get("id")
        for message in retained
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
        if (call.get("function") or {}).get("name") == "submit_report"
    }
    if not report_ids or not any(
        message.get("role") == "tool" and message.get("tool_call_id") in report_ids
        for message in retained
    ):
        raise Excluded("finalize_suffix_without_completed_report")
    transformed = copy.deepcopy(record)
    transformed["messages"] = retained
    receipt = {
        "schema": "cyber_terminal_finalize_suffix_strip_v1",
        "policy": FINALIZE_POLICY,
        "input_record_sha256": digest_json(record),
        "removed_suffix_sha256": digest_json(messages[-2:]),
        "retained_messages_sha256": digest_json(retained),
    }
    return transformed, digest_json(receipt), True


def _success(record: dict, evidence: dict, teachers: set[str]) -> bool:
    eligibility, outcome, proof = (
        record.get("eligibility"),
        record.get("outcome"),
        evidence.get("outcome"),
    )
    score = outcome.get("score") if isinstance(outcome, dict) else None
    return bool(
        record.get("schema") == RECORD_SCHEMA
        and record.get("source", {}).get("model") in teachers
        and isinstance(eligibility, dict)
        and eligibility.get("sft") is True
        and isinstance(outcome, dict)
        and outcome.get("infra_valid") is True
        and outcome.get("success") is True
        and type(score) in (int, float)
        and not isinstance(score, bool)
        and math.isfinite(score)
        and score >= 1
        and proof
        == {
            "score_at_least_one": True,
            "status": "completed",
            "verifier_process_success": True,
        }
    )


def _selection_row(record: dict, evidence: dict, rows: list[dict], group_id: str, transform):
    assistants = [message for message in record["messages"] if message["role"] == "assistant"]
    eligible = {span["assistant_index"] for row in rows for span in row["target_spans"]}
    if len(eligible) != sum(len(row["target_spans"]) for row in rows):
        raise ValueError("one source assistant target appeared more than once")
    counts = collections.Counter()
    submit_tokens = 0
    for index in sorted(eligible):
        calls = assistants[index].get("tool_calls") or []
        names = [(call.get("function") or {}).get("name") for call in calls]
        if not names and assistants[index].get("content"):
            counts["decision_responses"] += 1
        elif names and set(names) == {"bash"}:
            counts["non_submit_tool_responses"] += 1
            counts["completed_non_submit_tool_rounds"] += 1
        elif names == ["submit_report"]:
            counts["submit_report_responses"] += 1
        else:
            counts["other_responses"] += 1
    submit_indices = {
        index
        for index, message in enumerate(assistants)
        if [(call.get("function") or {}).get("name") for call in message.get("tool_calls") or []]
        == ["submit_report"]
    }
    for row in rows:
        for span in row["target_spans"]:
            if span["assistant_index"] in submit_indices:
                submit_tokens += span["token_end"] - span["token_start"]
    supervised = sum(row["target_token_count"] for row in rows)
    assistant_count = len(eligible)
    if not (
        counts["non_submit_tool_responses"] >= 1
        and counts["completed_non_submit_tool_rounds"] >= 1
        and assistant_count > counts["submit_report_responses"] >= 0
        and counts["submit_report_responses"] / assistant_count <= 0.5
        and supervised > 0
        and submit_tokens / supervised <= 0.5
    ):
        raise Excluded("insufficient_task_rich_targets")
    return {
        "session_id": record["record_id"],
        "task_key": record["lineage"]["task_key"],
        "task_version_id": record["lineage"]["eval_task_version_id"],
        "group_id": group_id,
        "model_id": record["source"]["model"],
        "harness_mode": record["source"]["harness_mode"],
        "harness_sha256": record["source"]["harness_sha256"],
        "trace_sha256": evidence["transcript_sha256"],
        "acceptance_sha256": evidence["sha256"],
        "normalized_record_sha256": evidence["normalized_record_sha256"],
        "transform_sha256": transform,
        "windows": len(rows),
        "supervised_tokens": supervised,
        "assistant_responses": assistant_count,
        "decision_responses": counts["decision_responses"],
        "non_submit_tool_responses": counts["non_submit_tool_responses"],
        "completed_non_submit_tool_rounds": counts["completed_non_submit_tool_rounds"],
        "submit_report_responses": counts["submit_report_responses"],
        "submit_report_tokens": submit_tokens,
        "other_responses": counts["other_responses"],
    }


def build(config: dict, *, relative_to: Path) -> dict:
    """Compile private Parquet and evidence receipts; return aggregates only."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _known(
        config,
        {
            "schema",
            "normalized",
            "evidence",
            "inventory",
            "study_split",
            "model_lock",
            "tokenizer_root",
            "native_helper",
            "teacher_models",
            "max_length",
            "context_tokens",
            "catalog_census",
            "output",
        },
        "Fleet teacher corpus",
    )
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported Fleet teacher corpus request")
    for name in ("normalized", "evidence"):
        _known(config[name], {"path", "sha256"}, name)
    teachers = config.get("teacher_models")
    if (
        not isinstance(teachers, list)
        or not teachers
        or any(not isinstance(model, str) or not model for model in teachers)
    ):
        raise ValueError("explicit stronger-teacher model list required")
    if len(teachers) != len(set(teachers)):
        raise ValueError("teacher model list contains duplicates")
    output = _path(relative_to, config["output"])
    if output.exists():
        raise FileExistsError("create-once corpus destination exists")
    sources = {
        name: _path(relative_to, config[name]["path"]) for name in ("normalized", "evidence")
    }
    for name, path in sources.items():
        if file_sha256(path) != "sha256:" + config[name]["sha256"].removeprefix("sha256:"):
            raise ValueError(f"{name} file digest mismatch")

    inventory_path = _path(relative_to, config["inventory"])
    split_path = _path(relative_to, config["study_split"])
    inventory, split = map(read_mapping, (inventory_path, split_path))
    if split.get("schema") != SPLIT_SCHEMA:
        raise ValueError("expected the reviewed representative study split")
    recorded_inventory = Path(split["inventory"]["path"])
    if recorded_inventory.resolve() != inventory_path.resolve():
        raise ValueError("configured inventory differs from the split binding")
    validate_split(split, inventory_path=recorded_inventory)
    train = {
        (row["task_key"], row["task_version_id"]): row["group_id"]
        for row in split["training_split"]["tasks"]
    }
    certification = {
        (row["task_key"], row["task_version_id"]): row["provenance"]["certification"][
            "receipt_sha256"
        ]
        for row in inventory["task_versions"]
    }
    if len(train) != 50 or not set(train) <= set(certification):
        raise ValueError("training split/inventory identity mismatch")

    evidence = {}
    for item in iter_jsonl(sources["evidence"]):
        if item.get("schema") != EVIDENCE_SCHEMA or not _sealed(item):
            raise ValueError("invalid success evidence receipt")
        session_id = item.get("session_id")
        if not isinstance(session_id, str) or not session_id or session_id in evidence:
            raise ValueError("missing or duplicate evidence session identity")
        evidence[session_id] = item
    records = sorted(iter_jsonl(sources["normalized"]), key=lambda row: row.get("record_id", ""))
    identities = {row.get("record_id") for row in records}
    if len(records) != len(evidence) or len(identities) != len(records):
        raise ValueError("normalized records and evidence are not one-to-one")

    tokenizer, tokenizer_identity = local_tokenizer(
        read_mapping(_path(relative_to, config["model_lock"])),
        _path(relative_to, config["tokenizer_root"]),
    )
    helper = native_helper(_path(relative_to, config["native_helper"]))
    maximum, context = config.get("max_length", 16384), config.get("context_tokens", 4096)
    if type(maximum) is not int or type(context) is not int or maximum < 2 or context < 0:
        raise ValueError("invalid data window bounds")

    rows, selections = [], []
    exclusions = collections.Counter()
    models = collections.Counter()
    harnesses = collections.Counter()
    trajectories, window_payloads = set(), set()
    suffixes_removed = 0
    for original in records:
        session_id = original.get("record_id")
        proof = evidence.get(session_id)
        if proof is None:
            raise ValueError("normalized record lacks evidence")
        if original.get("content_digest") != digest_json(
            {key: item for key, item in original.items() if key != "content_digest"}
        ) or proof["normalized_record_sha256"] != digest_json(original):
            raise ValueError("normalized record digest mismatch")
        lineage, source = original.get("lineage", {}), original.get("source", {})
        identity = (lineage.get("task_key"), lineage.get("eval_task_version_id"))
        if any(
            (
                proof.get("task_key") != identity[0],
                proof.get("task_version_id") != identity[1],
                proof.get("session_id") != source.get("session_id"),
                proof.get("model_id") != source.get("model"),
                proof.get("task_certification_receipt_sha256") != certification.get(identity),
                proof.get("routes", {}).get("transcript")
                != "GET /v1/sessions/{session_id}/transcript",
            )
        ):
            raise ValueError("source/evidence/task provenance mismatch")
        if identity not in train:
            exclusions["held_out_or_outside_exact_train"] += 1
            continue
        if not _success(original, proof, set(teachers)):
            exclusions["not_selected_verified_teacher_success"] += 1
            continue
        try:
            transformed, transform_sha, removed = _strip_finalize_suffix(original)
            suffixes_removed += int(removed)
            messages, _ = compatible_messages(transformed)
            trajectory = digest_json(messages)
            if trajectory in trajectories:
                exclusions["exact_trajectory_duplicate"] += 1
                continue
            anchor, chunks = encode_record(messages, tokenizer, helper)
            packed = segment_record(
                transformed, anchor, chunks, max_tokens=maximum, context_budget=context
            )
            selection = _selection_row(transformed, proof, packed, train[identity], transform_sha)
        except Excluded as exc:
            exclusions[exc.reason] += 1
            continue
        kept = []
        for row in packed:
            fingerprint = digest_json([row["input_ids"], row["loss_mask"]])
            if fingerprint in window_payloads:
                exclusions["exact_window_duplicate"] += 1
                continue
            window_payloads.add(fingerprint)
            row.update(
                {
                    "source_harness_mode": source["harness_mode"],
                    "source_harness_sha256": source["harness_sha256"],
                    "source_acceptance_sha256": proof["sha256"],
                    "source_trace_sha256": proof["transcript_sha256"],
                    "source_normalized_record_sha256": proof["normalized_record_sha256"],
                    "source_task_version_id": identity[1],
                    "source_transform_sha256": transform_sha,
                }
            )
            kept.append(row)
        if len(kept) != len(packed):
            raise ValueError("partial source removal would break assistant-target coverage")
        trajectories.add(trajectory)
        rows.extend(kept)
        selections.append(selection)
        models[source["model"]] += 1
        harnesses[source["harness_mode"]] += 1

    rows.sort(key=lambda row: (row["task_key"], row["source_session_id"], row["segment_id"]))
    counts = {
        "rows": len(rows),
        "task_keys": sorted({row["task_key"] for row in rows}),
        "format": DENSE_FORMAT,
        "source_sessions": len(selections),
        "supervised_tokens": sum(row["target_token_count"] for row in rows),
        "assistant_responses": sum(len(row["target_spans"]) for row in rows),
    }
    first = {}
    for row in rows:
        first.setdefault(row["source_session_id"], row)
    counts["source_total_assistant_responses"] = sum(
        row["source_assistant_count"] for row in first.values()
    )
    counts["excluded_assistant_responses"] = sum(
        len(row["excluded_assistant_targets"]) for row in first.values()
    )
    if not rows or len(selections) != len(first):
        raise ValueError("empty or incomplete teacher corpus")
    dense_rows(rows, counts, max_length=maximum, vocab_size=len(tokenizer))
    if any((row["task_key"], row["source_task_version_id"]) not in train for row in rows):
        raise ValueError("held-out task entered the training corpus")
    if any(
        file_sha256(sources[name]) != "sha256:" + config[name]["sha256"].removeprefix("sha256:")
        for name in sources
    ):
        raise ValueError("source files changed during preparation")

    output.mkdir(parents=True, mode=0o700)
    data_path = output / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), data_path, compression="zstd")
    os.chmod(data_path, 0o600)
    if pq.read_table(data_path).to_pylist() != rows:
        raise ValueError("Parquet readback differs")
    selection_path = output / "source-selection.private.jsonl"
    atomic_write_jsonl(selection_path, selections, private=True)
    census = config.get("catalog_census", {})
    if census and census.get("stronger_teacher_successes") != (
        census.get("exact_current_version_successes", 0)
        + census.get("different_version_successes", 0)
    ):
        raise ValueError("catalog census arithmetic mismatch")
    manifest = {
        "schema": "cyber_dense_sft_corpus_v1",
        "source_sha256": file_sha256(selection_path),
        "split_sha256": split["sha256"],
        "tokenizer": tokenizer_identity,
        "files": {"train": {"path": data_path.name, "sha256": file_sha256(data_path), **counts}},
        "train_models": sorted(models),
        "max_length": maximum,
        "context_tokens": context,
        "dev_windows": 0,
        "validation_mode": "task_outcomes_only",
        "split_exclusions": census,
        "whole_source_exclusions": dict(sorted(exclusions.items())),
        "catalog_provenance": {
            "fleet_team_id": TEAM_ID,
            "study_split_file_sha256": file_sha256(split_path),
            "study_split_sha256": split["sha256"],
            "high_quality_inventory_file_sha256": file_sha256(inventory_path),
            "normalized_records_file_sha256": file_sha256(sources["normalized"]),
            "success_evidence_file_sha256": file_sha256(sources["evidence"]),
            "source_selection_file_sha256": file_sha256(selection_path),
            "selected_task_versions": len({row["source_task_version_id"] for row in rows}),
            "selected_source_sessions": len(selections),
            "teacher_model_sessions": dict(sorted(models.items())),
            "harness_mode_sessions": dict(sorted(harnesses.items())),
            "terminal_finalize_suffixes_removed": suffixes_removed,
            "terminal_finalize_policy": FINALIZE_POLICY,
            "exact_trajectory_duplicates_removed": exclusions["exact_trajectory_duplicate"],
            "exact_window_duplicates_removed": exclusions["exact_window_duplicate"],
            "summary_route": "GET /v1/sessions?task_key=<exact-train-key>",
            "transcript_route": "GET /v1/sessions/{session_id}/transcript",
        },
        "builder_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("fleet_teacher_corpus.py", "dense.py", "corpus.py")
        }
        | {"native_helper": file_sha256(_path(relative_to, config["native_helper"]))},
        "limitations": [
            "Visible assistant actions only; private reasoning is omitted.",
            "Teacher sources are successful exact-version sessions on the frozen "
            "50-task train split only.",
            "Fresh Fleet development outcomes, not held-out teacher-token loss, "
            "select checkpoints.",
            "Opaque context compaction and unproven use_tool/search_tool interfaces are excluded.",
        ],
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    atomic_write_json(output / "manifest.json", manifest, private=True)
    receipt = {
        "schema": "cyber_fleet_teacher_corpus_receipt_v1",
        "manifest_file_sha256": file_sha256(output / "manifest.json"),
        "manifest_sha256": manifest["sha256"],
        "train_parquet_sha256": file_sha256(data_path),
        "source_selection_sha256": file_sha256(selection_path),
        "rows": counts["rows"],
        "source_sessions": counts["source_sessions"],
        "task_versions": manifest["catalog_provenance"]["selected_task_versions"],
        "supervised_tokens": counts["supervised_tokens"],
    }
    receipt["sha256"] = "sha256:" + digest(receipt)
    atomic_write_json(output / "RECEIPT.json", receipt, private=True)
    return {
        "output": str(output),
        "manifest_sha256": manifest["sha256"],
        "receipt_sha256": receipt["sha256"],
        "train": {
            "rows": counts["rows"],
            "source_sessions": counts["source_sessions"],
            "task_versions": manifest["catalog_provenance"]["selected_task_versions"],
            "supervised_tokens": counts["supervised_tokens"],
        },
        "teacher_models": dict(sorted(models.items())),
        "whole_source_exclusions": dict(sorted(exclusions.items())),
    }
