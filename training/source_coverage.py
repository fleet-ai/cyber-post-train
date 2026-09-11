"""Local, train-only SFT coverage extraction; private text/tokens never leave memory.

Reuse the corpus tokenizer and dense compiler. A reviewed evidence mapping, not
the content of a trace or a model answer, supplies authoritative success claims.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.metadata
import json
import os
from collections.abc import Iterable
from pathlib import Path

from . import dense
from .io import atomic_write_json, atomic_write_jsonl, digest_json, file_sha256, iter_jsonl
from .study_data import EPISODE_SCHEMA, _coverage, _sha, check_seal, seal

EVIDENCE_SCHEMA = "cyber_sft_reviewed_source_evidence_v1"
POLICY_SCHEMA = "cyber_sft_target_policy_v1"
COVERAGE_SCHEMA = "cyber_sft_source_target_coverage_v1"
COUNTS = (
    "assistant_responses",
    "supervised_tokens",
    "submit_report_responses",
    "submit_report_tokens",
    "non_submit_tool_responses",
    "decision_responses",
    "other_responses",
    "completed_non_submit_tool_rounds",
)
EVIDENCE_FIELDS = {
    "episode_id",
    "task_key",
    "task_version_id",
    "model_id",
    "source_kind",
    "validity",
    "verified_success",
    "acceptance_sha256",
    "trace_sha256",
    "normalized_record_sha256",
}


class CoverageError(ValueError):
    """Only static, non-payload error messages may escape the public boundary."""


def target_policy(
    tokenizer_identity: dict, helper_path: Path, *, max_length: int, context_tokens: int
) -> dict:
    """Shared policy fingerprint for this extractor and corpus qualification."""
    if (
        type(max_length) is not int
        or max_length < 2
        or type(context_tokens) is not int
        or context_tokens < 0
        or file_sha256(helper_path) != dense.NATIVE_HELPER_SHA
    ):
        raise CoverageError("invalid bounds or native helper identity")
    return seal(
        {
            "schema": POLICY_SCHEMA,
            "format": dense.FORMAT,
            "tokenizer": tokenizer_identity,
            "native_helper_sha256": file_sha256(helper_path),
            "max_length": max_length,
            "context_tokens": context_tokens,
            "libraries": {
                name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")
            },
            "code_sha256": {
                name: file_sha256(Path(__file__).with_name(name))
                for name in ("dense.py", "corpus.py", "source_coverage.py", "sft_runtime.py")
            },
            "classification": "submit_report first; other tools; visible decision; other",
            "completed_tool_round": (
                "non-submit assistant with all original call results structurally present"
            ),
            "scope": "eligible_dense_targets; copied context and tool observations excluded",
        }
    )


def reviewed_sources(evidence: dict, split: dict) -> dict:
    check_seal(evidence, EVIDENCE_SCHEMA)
    check_seal(split, "cyber_task_split_v2")
    if (
        set(evidence) != {"schema", "sha256", "origin", "review_receipt_sha256", "episodes"}
        or evidence["origin"] != "fleet_cyber"
        or not _sha(evidence["review_receipt_sha256"])
    ):
        raise CoverageError("explicit reviewed Fleet source evidence required")
    tasks = split["tasks"]
    if not tasks or any(
        set(r) != {"task_key", "task_version_id", "split"} or r["split"] != "train" for r in tasks
    ):
        raise CoverageError("coverage input split must be train-only and reference-free")
    train = {(r["task_key"], r["task_version_id"]) for r in tasks}
    if len(train) != len(tasks):
        raise CoverageError("duplicate exact training identity")
    result = {}
    for row in evidence["episodes"]:
        if (row["task_key"], row["task_version_id"]) not in train:
            raise CoverageError("review mapping contains a non-training identity")
        if (
            set(row) != EVIDENCE_FIELDS
            or not isinstance(row["episode_id"], str)
            or not row["episode_id"]
            or row["episode_id"] in result
            or row["source_kind"] not in {"teacher", "self"}
            or row["validity"] != "valid"
            or row["verified_success"] is not True
            or not isinstance(row["model_id"], str)
            or not row["model_id"]
            or any(
                not _sha(row[k])
                for k in ("acceptance_sha256", "trace_sha256", "normalized_record_sha256")
            )
        ):
            raise CoverageError("review mapping is duplicated, incomplete or not verified-success")
        result[row["episode_id"]] = row
    if not result:
        raise CoverageError("empty reviewed source mapping")
    return result


def _derive_record(
    record: dict, evidence: dict, policy: dict, tokenizer, helper
) -> tuple[dict, dict]:
    """Produce only target counts and hashes; no token IDs or source text returned."""
    from .corpus import _eligible
    from .sft_runtime import dense_rows

    check_seal(policy, POLICY_SCHEMA)
    lineage = record["lineage"]
    if (
        record["record_id"] != evidence["episode_id"]
        or lineage["task_key"] != evidence["task_key"]
        or lineage["eval_task_version_id"] != evidence["task_version_id"]
        or record["source"]["model"] != evidence["model_id"]
        or digest_json(record) != evidence["normalized_record_sha256"]
        or not _eligible(record)
    ):
        raise CoverageError("reviewed identity, record digest or corpus eligibility differs")
    # Call the very same normalization, renderer and eligibility implementation
    # as corpus.build. Never copy or reimplement tokenization/segmentation rules.
    messages, _ = dense.compatible_messages(record)
    anchor, chunks = dense.encode_record(messages, tokenizer, helper)
    rows = dense.segment_record(
        record,
        anchor,
        chunks,
        max_tokens=policy["max_length"],
        context_budget=policy["context_tokens"],
    )
    spans = [s for row in rows for s in row["target_spans"]]
    original, excluded = rows[0]["source_assistant_count"], rows[0]["excluded_assistant_targets"]
    counts = dict.fromkeys(COUNTS, 0)
    counts["assistant_responses"] = len(spans)
    counts["supervised_tokens"] = sum(row["target_token_count"] for row in rows)
    dense_rows(
        rows,
        {
            "format": dense.FORMAT,
            "rows": len(rows),
            "task_keys": [lineage["task_key"]],
            "source_sessions": 1,
            "supervised_tokens": counts["supervised_tokens"],
            "assistant_responses": len(spans),
            "source_total_assistant_responses": original,
            "excluded_assistant_responses": len(excluded),
        },
        max_length=policy["max_length"],
        vocab_size=len(tokenizer),
    )
    for span in spans:
        message = messages[span["source_message_index"]]
        if message["role"] != "assistant":
            raise CoverageError("native target does not identify an assistant response")
        names = {call["function"]["name"] for call in message.get("tool_calls", [])}
        if "submit_report" in names:
            counts["submit_report_responses"] += 1
            counts["submit_report_tokens"] += span["token_end"] - span["token_start"]
        elif names:
            counts["non_submit_tool_responses"] += 1
            # compatible_messages proved all original calls have unique results,
            # in causal order, before the next assistant. This is structural,
            # not a claim that a command succeeded or made exploit progress.
            counts["completed_non_submit_tool_rounds"] += 1
        elif message["content"].strip():
            counts["decision_responses"] += 1
        else:
            counts["other_responses"] += 1
    coverage = {
        "status": "certified",
        "scope": "eligible_dense_targets",
        "target_policy_sha256": policy["sha256"],
        **counts,
    }
    receipt = seal(
        {
            "schema": COVERAGE_SCHEMA,
            "episode_id": evidence["episode_id"],
            "normalized_record_sha256": evidence["normalized_record_sha256"],
            "acceptance_sha256": evidence["acceptance_sha256"],
            "trace_sha256": evidence["trace_sha256"],
            "target_policy_sha256": policy["sha256"],
            "coverage": coverage,
            "segments": len(rows),
            "source_assistant_responses": original,
            "eligible_assistant_indices_sha256": digest_json(rows[0]["eligible_assistant_indices"]),
            "excluded_assistant_responses": len(excluded),
            "excluded_reasons": dict(collections.Counter(r["reason"] for r in excluded)),
            "excluded_inventory_sha256": digest_json(excluded),
            "eligible_target_inventory_sha256": digest_json(
                [
                    {
                        k: s[k]
                        for k in ("assistant_index", "source_message_index", "source_target_sha256")
                    }
                    for s in spans
                ]
            ),
            "dense_rows_sha256": digest_json(rows),
        }
    )
    episode = seal(
        {
            "schema": EPISODE_SCHEMA,
            **evidence,
            "coverage": {**coverage, "receipt_sha256": receipt["sha256"]},
        }
    )
    _coverage(episode["coverage"], policy["sha256"])
    return episode, receipt


def extract(
    records: Iterable[dict], evidence: dict, split: dict, policy: dict, tokenizer, helper
) -> tuple[list, list, dict]:
    """Stream private records, restricting all substantive access to reviewed train IDs."""
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            return _extract(records, evidence, split, policy, tokenizer, helper)
    except CoverageError:
        raise
    except Exception:
        raise CoverageError("coverage extraction failed; private diagnostics withheld") from None


def _extract(records, evidence, split, policy, tokenizer, helper):
    reviewed = reviewed_sources(evidence, split)
    train = {(r["task_key"], r["task_version_id"]) for r in split["tasks"]}
    seen, episodes, receipts, skipped = set(), [], [], collections.Counter()
    excluded_sources = []
    for record in records:
        key = (record["lineage"]["task_key"], record["lineage"]["eval_task_version_id"])
        if key not in train:
            skipped["outside_training_split"] += 1
            continue
        sid = record["record_id"]
        if sid not in reviewed:
            skipped["not_in_reviewed_mapping"] += 1
            continue
        if sid in seen:
            raise CoverageError("duplicate reviewed private record")
        seen.add(sid)
        try:
            episode, receipt = _derive_record(record, reviewed[sid], policy, tokenizer, helper)
        except dense.Excluded as exc:
            skipped[exc.reason] += 1
            excluded_sources.append(
                {
                    "episode_id": sid,
                    "reason": exc.reason,
                    "normalized_record_sha256": reviewed[sid]["normalized_record_sha256"],
                }
            )
            continue
        episodes.append(episode)
        receipts.append(receipt)
    if seen != set(reviewed):
        raise CoverageError("reviewed private records are missing")
    episodes.sort(key=lambda r: r["episode_id"])
    receipts.sort(key=lambda r: r["episode_id"])
    return (
        episodes,
        receipts,
        {
            "skip_counts": dict(sorted(skipped.items())),
            "excluded_sources": sorted(excluded_sources, key=lambda r: r["episode_id"]),
        },
    )


def build(config: dict, *, relative_to: Path) -> dict:
    """Create metadata-only output once. Suppress private library logs and errors."""
    # A null sink avoids retaining or forwarding any tokenizer warning containing
    # private data. The command's final output is only the safe manifest summary.
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            return _build(config, relative_to=relative_to)
    except FileExistsError:
        raise CoverageError("create-once coverage destination already exists") from None
    except CoverageError:
        raise
    except Exception:
        raise CoverageError("coverage extraction failed; private diagnostics withheld") from None


def _build(config: dict, *, relative_to: Path) -> dict:
    from .corpus import local_tokenizer
    from .sft import read_mapping

    if set(config) != {
        "source",
        "split",
        "reviewed_evidence",
        "model_lock",
        "tokenizer_root",
        "native_helper",
        "max_length",
        "context_tokens",
        "output",
    }:
        raise CoverageError("unexpected or missing coverage configuration field")
    paths = {
        name: relative_to / config[name]
        for name in ("split", "model_lock", "tokenizer_root", "native_helper", "output")
    }
    for name in ("source", "reviewed_evidence"):
        if set(config[name]) != {"path", "sha256"} or not _sha(config[name]["sha256"]):
            raise CoverageError("source and review mapping require exact file digests")
        paths[name] = relative_to / config[name]["path"]
        if file_sha256(paths[name]) != config[name]["sha256"]:
            raise CoverageError("source or review mapping file digest mismatch")
    if paths["output"].exists():
        raise FileExistsError
    stable = {
        paths[k]: file_sha256(paths[k])
        for k in ("source", "reviewed_evidence", "split", "model_lock", "native_helper")
    }
    if any(stable[paths[k]] != config[k]["sha256"] for k in ("source", "reviewed_evidence")):
        raise CoverageError("source or review mapping changed before extraction")
    evidence, split = read_mapping(paths["reviewed_evidence"]), read_mapping(paths["split"])
    reviewed_sources(evidence, split)
    lock = read_mapping(paths["model_lock"])
    tokenizer, identity = local_tokenizer(lock, paths["tokenizer_root"])
    helper = dense.native_helper(paths["native_helper"])
    policy = target_policy(
        identity,
        paths["native_helper"],
        max_length=config["max_length"],
        context_tokens=config["context_tokens"],
    )
    stable.update(
        {
            paths["tokenizer_root"] / r["path"]: "sha256:" + r["sha256"].removeprefix("sha256:")
            for r in lock["tokenizer"]["files"]
        }
    )
    episodes, receipts, audit = extract(
        iter_jsonl(paths["source"]), evidence, split, policy, tokenizer, helper
    )
    if any(file_sha256(p) != sha for p, sha in stable.items()) or policy != target_policy(
        identity,
        paths["native_helper"],
        max_length=config["max_length"],
        context_tokens=config["context_tokens"],
    ):
        raise CoverageError("coverage inputs or compiler changed during extraction")
    output = paths["output"]
    output.mkdir(mode=0o700, parents=True)
    atomic_write_jsonl(output / "episodes.jsonl", episodes, private=True)
    atomic_write_jsonl(output / "coverage-receipts.jsonl", receipts, private=True)
    atomic_write_json(output / "target-policy.json", policy, private=True)
    manifest = seal(
        {
            "schema": "cyber_sft_coverage_extraction_v1",
            "source_sha256": stable[paths["source"]],
            "reviewed_evidence_sha256": evidence["sha256"],
            "review_receipt_sha256": evidence["review_receipt_sha256"],
            "split_sha256": split["sha256"],
            "target_policy_sha256": policy["sha256"],
            "reviewed_sources": len(evidence["episodes"]),
            "emitted_sources": len(episodes),
            "counts": {key: sum(row["coverage"][key] for row in episodes) for key in COUNTS},
            **audit,
            "files": {
                name: file_sha256(output / name)
                for name in ("episodes.jsonl", "coverage-receipts.jsonl", "target-policy.json")
            },
            "limitations": [
                "Acceptance claims are supplied by an explicit reviewed evidence mapping.",
                "Tool/decision counts do not measure semantic exploit progress.",
                "No held-out teacher references or GPU/model forward pass were used.",
            ],
        }
    )
    atomic_write_json(output / "manifest.json", manifest, private=True)
    return {
        "manifest_sha256": manifest["sha256"],
        "target_policy_sha256": policy["sha256"],
        "emitted_sources": len(episodes),
        "counts": manifest["counts"],
        "skip_counts": audit["skip_counts"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Create train-only, metadata-only SFT coverage receipts"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text())
        result = build(config, relative_to=args.config.parent)
    except Exception:
        print(json.dumps({"status": "rejected", "reason": "coverage_extraction_not_completed"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
