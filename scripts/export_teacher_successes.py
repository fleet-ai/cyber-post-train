#!/usr/bin/env python3
"""Export an explicit census of successful Fleet teacher sessions, read-only.

Raw transcripts and normalized trajectories are private.  The command prints
only aggregate counts and digests; it never prints response bodies.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import os
import tempfile
from contextlib import ExitStack, suppress
from pathlib import Path

from cyber_post_train.jobs import digest
from training.fleet import FleetClient
from training.io import atomic_write_json, canonical_json, digest_json, file_sha256
from training.normalize import normalize_export_row
from training.secrets import secret_values

SCHEMA = "fleet_broad_success_evidence_v1"
TERMINAL = {"completed", "succeeded", "success"}


def _candidates(path: Path, study_split: Path) -> tuple[list[dict], set[str], int]:
    rows = json.loads(path.read_text())
    split = json.loads(study_split.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty teacher success census")
    held_out = {
        row["task_key"]
        for row in split.get("tasks", [])
        if row.get("split") in {"dev", "final_test"}
    }
    if len(held_out) != 25:
        raise ValueError("reviewed Fleet holdout no longer contains 25 task families")
    selected, seen = [], set()
    for row in rows:
        sid = row.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("missing or duplicate census session id")
        seen.add(sid)
        if row.get("task_key") in held_out:
            continue
        selected.append(row)
    return selected, held_out, len(rows) - len(selected)


def _validated(candidate: dict, envelope: dict) -> tuple[dict, dict, dict]:
    task = envelope.get("task") if isinstance(envelope.get("task"), dict) else {}
    harness = envelope.get("harness") if isinstance(envelope.get("harness"), dict) else {}
    verifier = (
        envelope.get("verifier_execution")
        if isinstance(envelope.get("verifier_execution"), dict)
        else {}
    )
    messages = envelope.get("transcript")
    score = verifier.get("score")
    status = str(candidate.get("status") or "").lower()
    if (
        status not in TERMINAL
        or candidate.get("full") is not True
        or candidate.get("healthy") is not True
        or task.get("key") != candidate.get("task_key")
        or not isinstance(task.get("eval_task_version_id"), str)
        or not task["eval_task_version_id"]
        or verifier.get("success") is not True
        or type(score) not in (int, float)
        or isinstance(score, bool)
        or not math.isfinite(score)
        or score < 1
        or not isinstance(messages, list)
        or not messages
        or not isinstance(harness.get("mode"), str)
        or not isinstance(harness.get("tool_names"), list)
    ):
        raise ValueError("candidate no longer has an exact verified success transcript")
    source_binding_fields = {
        "key",
        "env_id",
        "created_at",
        "version",
        "eval_task_version_id",
        "data_id",
        "data_version",
        "multi_app_seed_versions",
        "verifier_id",
        "verifier_sha",
        "output_json_schema",
    }
    export = {
        "export_schema": "fleet_session_export_v1",
        "source": {
            "job_id": None,
            "job_name": None,
            "session_id": candidate["id"],
            "task_key_from_roster": candidate["task_key"],
            "roster_task_binding": {
                key: task[key]
                for key in sorted(source_binding_fields)
                if key in task and task[key] is not None
            },
        },
        "session": {
            "session_id": candidate["id"],
            "status": status,
            "model": candidate["model"],
            "score": score,
            "created_at": candidate.get("created_at"),
        },
        "transcript_envelope": envelope,
    }
    normalized = normalize_export_row(export, secrets=secret_values())
    normalized["source"].update(
        {
            "harness_mode": harness["mode"],
            "harness_tools": harness["tool_names"],
            "harness_sha256": digest_json(harness),
        }
    )
    normalized["content_digest"] = digest_json(
        {key: value for key, value in normalized.items() if key != "content_digest"}
    )
    evidence = {
        "schema": SCHEMA,
        "session_id": candidate["id"],
        "model_id": candidate["model"],
        "task_key": task["key"],
        "task_version_id": task["eval_task_version_id"],
        "normalized_record_sha256": digest_json(normalized),
        "transcript_sha256": digest_json(envelope),
        "verifier_execution_id": verifier.get("id"),
        "routes": {
            "summary": "GET /v1/sessions?eval_task_id=<project-task-id>",
            "transcript": "GET /v1/sessions/{session_id}/transcript",
        },
        "outcome": {
            "score_at_least_one": True,
            "status": "completed",
            "verifier_process_success": True,
        },
    }
    evidence["sha256"] = "sha256:" + digest(evidence)
    return export, normalized, evidence


def export(args: argparse.Namespace) -> dict:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise ValueError("FLEET_API_KEY must be injected through the environment")
    selected, held_out, held_out_sessions = _candidates(args.census, args.study_split)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("create-once export destination is not empty")
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    targets = {
        "exports": output / "exports.private.jsonl",
        "normalized": output / "normalized.private.jsonl",
        "evidence": output / "evidence.private.jsonl",
    }
    client = FleetClient(key, base_url=args.base_url, timeout=args.timeout)

    def fetch(row: dict):
        return row, client.transcript(row["id"])

    temporaries: dict[str, str] = {}
    handles = {}
    counts = collections.Counter()
    models = collections.Counter()
    task_versions = set()
    try:
        with ExitStack() as stack:
            for name, target in targets.items():
                fd, path = tempfile.mkstemp(prefix=f".{target.name}.", dir=output, text=True)
                os.fchmod(fd, 0o600)
                temporaries[name] = path
                handles[name] = stack.enter_context(os.fdopen(fd, "w", encoding="utf-8"))
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                for candidate, envelope in executor.map(fetch, selected):
                    try:
                        raw, normalized, evidence = _validated(candidate, envelope)
                    except ValueError:
                        counts["invalid_at_export"] += 1
                        continue
                    for name, value in (
                        ("exports", raw),
                        ("normalized", normalized),
                        ("evidence", evidence),
                    ):
                        handles[name].write(canonical_json(value) + "\n")
                    counts["exported"] += 1
                    models[candidate["model"]] += 1
                    task_versions.add((evidence["task_key"], evidence["task_version_id"]))
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        for name, target in targets.items():
            os.replace(temporaries[name], target)
        manifest = {
            "schema": "fleet_broad_teacher_export_manifest_v1",
            "census_sha256": file_sha256(args.census),
            "study_split_sha256": file_sha256(args.study_split),
            "input_candidate_sessions": len(selected) + held_out_sessions,
            "held_out_task_families": len(held_out),
            "held_out_sessions_excluded": held_out_sessions,
            "selected_candidate_sessions": len(selected),
            "exported_sessions": counts["exported"],
            "invalid_at_export": counts["invalid_at_export"],
            "task_versions": len(task_versions),
            "models": dict(sorted(models.items())),
            "files": {
                name: {"path": target.name, "sha256": file_sha256(target)}
                for name, target in targets.items()
            },
            "routes": {"transcript": "GET /v1/sessions/{session_id}/transcript"},
        }
        manifest["sha256"] = "sha256:" + digest(manifest)
        atomic_write_json(output / "manifest.json", manifest, private=True)
        return manifest
    except BaseException:
        for path in temporaries.values():
            with suppress(FileNotFoundError):
                os.unlink(path)
        raise


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--study-split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="https://orchestrator.fleetai.com")
    p.add_argument("--workers", type=int, default=32, choices=range(1, 65))
    p.add_argument("--timeout", type=float, default=120)
    return p


if __name__ == "__main__":
    result = export(parser().parse_args())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "input_candidate_sessions",
                    "held_out_sessions_excluded",
                    "exported_sessions",
                    "invalid_at_export",
                    "task_versions",
                    "sha256",
                )
            },
            indent=2,
        )
    )
