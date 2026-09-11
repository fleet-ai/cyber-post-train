"""Private-trace-safe evidence for Fleet Agent Runtime/native rollout parity.

The source export contains task prompts and model transcripts.  This module deliberately emits
only counts, hashes, durations, model names, task identifiers, and tool names; it never copies a
prompt, assistant message, tool arguments, or tool output into its result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _duration_seconds(session: dict[str, Any]) -> float | None:
    start = session.get("started_at")
    end = session.get("ended_at")
    if not start or not end:
        return None
    return (
        datetime.fromisoformat(str(end).replace("Z", "+00:00"))
        - datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    ).total_seconds()


def _text_content(content: Any) -> str | None:
    if isinstance(content, dict):
        return _text_content(content.get("content"))
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and isinstance(part.get("text"), str) for part in content
    ):
        return "".join(part["text"] for part in content)
    return None


def analyze_export(
    export_path: Path,
    *,
    treatment_task_keys: set[str] | None = None,
    export_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    models: Counter[str] = Counter()
    model_passes: Counter[str] = Counter()
    tool_surfaces: Counter[tuple[str, ...]] = Counter()
    launch_shapes: Counter[str] = Counter()
    system_prompts: Counter[tuple[str, int]] = Counter()
    tool_calls: Counter[str] = Counter()
    compaction_pre: list[float] = []
    compaction_post: list[float] = []
    compaction_sessions = 0
    assistant_turns: list[float] = []
    session_steps: list[float] = []
    durations: list[float] = []
    task_outcomes: dict[str, list[int]] = defaultdict(list)
    unique_versions: set[str] = set()
    source_job_ids: set[str] = set()
    records = 0
    passes = 0
    prompt_hydration_matches = 0
    bash_result_bytes: list[float] = []
    unparsed_bash_results = 0

    with export_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records += 1
            source = row["source"]
            session = row["session"]
            envelope = row["transcript_envelope"]
            harness = envelope["harness"]
            transcript = envelope["transcript"]
            task = envelope["task"]

            source_job_ids.add(str(source["job_id"]))
            model = str(session["model"])
            score = float((session.get("verifier_execution") or {}).get("score") or 0)
            passed = int(score > 0)
            passes += passed
            models[model] += 1
            model_passes[model] += passed
            task_key = str(task["key"])
            task_outcomes[task_key].append(passed)
            unique_versions.add(str(task["eval_task_version_id"]))
            first_user = next(
                (message for message in transcript if message.get("role") == "user"), None
            )
            prompt_hydration_matches += int(
                first_user is not None
                and _text_content(first_user.get("content")) == task.get("prompt")
            )

            tool_surfaces[tuple(harness.get("tool_names") or [])] += 1
            params = harness.get("job_launch_params") or {}
            launch_shapes[json.dumps(params, sort_keys=True, separators=(",", ":"))] += 1
            session_steps.append(float(session.get("step_count") or 0))
            duration = _duration_seconds(session)
            if duration is not None:
                durations.append(duration)
            assistant_turns.append(
                sum(1 for message in transcript if message.get("role") == "assistant")
            )

            compacted = False
            calls_by_id: dict[str, str] = {}
            for message in transcript:
                if message.get("role") == "system" and isinstance(message.get("content"), str):
                    content = message["content"]
                    system_prompts[(_sha256_text(content), len(content))] += 1
                for call in message.get("tool_calls") or []:
                    name = (call.get("function") or {}).get("name")
                    if not name:
                        continue
                    tool_calls[str(name)] += 1
                    if call.get("id"):
                        calls_by_id[str(call["id"])] = str(name)
                    if name != "context_compaction":
                        continue
                    compacted = True
                    try:
                        args = json.loads((call.get("function") or {}).get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if isinstance(args.get("pre_tokens"), (int, float)):
                        compaction_pre.append(float(args["pre_tokens"]))
                    if isinstance(args.get("post_tokens"), (int, float)):
                        compaction_post.append(float(args["post_tokens"]))
                if (
                    message.get("role") == "tool"
                    and calls_by_id.get(str(message.get("tool_call_id"))) == "bash"
                ):
                    content = _text_content(message.get("content"))
                    if content is None:
                        unparsed_bash_results += 1
                    else:
                        bash_result_bytes.append(float(len(content.encode())))
            compaction_sessions += int(compacted)

    if len(source_job_ids) != 1:
        raise ValueError(f"expected one source job, found {sorted(source_job_ids)!r}")

    treatment = None
    if treatment_task_keys is not None:
        outcomes = [
            outcome for key in treatment_task_keys for outcome in task_outcomes.get(key, [])
        ]
        treatment = {
            "requested_task_keys": len(treatment_task_keys),
            "observed_task_keys": sum(key in task_outcomes for key in treatment_task_keys),
            "sessions": len(outcomes),
            "passes": sum(outcomes),
            "pass_rate": sum(outcomes) / len(outcomes) if outcomes else None,
            "always_solved": sum(
                bool(task_outcomes.get(key)) and all(task_outcomes[key])
                for key in treatment_task_keys
            ),
            "never_solved": sum(
                bool(task_outcomes.get(key)) and not any(task_outcomes[key])
                for key in treatment_task_keys
            ),
            "mixed": sum(
                bool(task_outcomes.get(key))
                and any(task_outcomes[key])
                and not all(task_outcomes[key])
                for key in treatment_task_keys
            ),
        }

    evidence = {
        "schema": "fleet_harness_export_evidence_v1",
        "source_job_id": next(iter(source_job_ids)),
        "records": records,
        "passes": passes,
        "pass_rate": passes / records if records else None,
        "unique_task_keys": len(task_outcomes),
        "unique_task_version_ids": len(unique_versions),
        "task_prompt_hydration": {
            "byte_equal_after_text_envelope_normalization": prompt_hydration_matches,
            "mismatches": records - prompt_hydration_matches,
        },
        "models": {
            model: {
                "sessions": count,
                "passes": model_passes[model],
                "pass_rate": model_passes[model] / count,
            }
            for model, count in sorted(models.items())
        },
        "task_tool_surfaces": [
            {"tools": list(surface), "sessions": count}
            for surface, count in sorted(tool_surfaces.items())
        ],
        "launch_parameters": [
            {"parameters": json.loads(shape), "sessions": count}
            for shape, count in sorted(launch_shapes.items())
        ],
        "system_prompts": [
            {"sha256": digest, "character_length": length, "sessions": count}
            for (digest, length), count in sorted(system_prompts.items())
        ],
        "assistant_turns": {
            "min": min(assistant_turns),
            "median": statistics.median(assistant_turns),
            "p90_nearest_rank": _nearest_rank(assistant_turns, 0.9),
            "max": max(assistant_turns),
        },
        "session_steps": {
            "min": min(session_steps),
            "median": statistics.median(session_steps),
            "p90_nearest_rank": _nearest_rank(session_steps, 0.9),
            "max": max(session_steps),
        },
        "duration_seconds": {
            "min": min(durations),
            "median": statistics.median(durations),
            "p90_nearest_rank": _nearest_rank(durations, 0.9),
            "max": max(durations),
        },
        "transcript_tool_calls": dict(sorted(tool_calls.items())),
        "context_compaction": {
            "events": tool_calls["context_compaction"],
            "sessions": compaction_sessions,
            "pre_tokens_median": statistics.median(compaction_pre) if compaction_pre else None,
            "post_tokens_median": statistics.median(compaction_post) if compaction_post else None,
            "algorithm_visible": False,
        },
        "bash_tool_result_bytes": {
            "parsed": len(bash_result_bytes),
            "unparsed": unparsed_bash_results,
            "median": statistics.median(bash_result_bytes) if bash_result_bytes else None,
            "p90_nearest_rank": _nearest_rank(bash_result_bytes, 0.9),
            "p99_nearest_rank": _nearest_rank(bash_result_bytes, 0.99),
            "max": max(bash_result_bytes) if bash_result_bytes else None,
            "over_4000": sum(value > 4000 for value in bash_result_bytes),
            "over_16000": sum(value > 16000 for value in bash_result_bytes),
            "over_65536": sum(value > 65536 for value in bash_result_bytes),
        },
        "as_treated_train": treatment,
        "privacy": {
            "prompt_bodies_emitted": False,
            "assistant_or_tool_content_emitted": False,
            "tool_arguments_emitted": False,
        },
    }
    if export_manifest is not None:
        if export_manifest.get("sessions") != records:
            raise ValueError("export manifest session count does not match JSONL")
        manifest_jobs = {str(job) for job in export_manifest.get("jobs") or []}
        if manifest_jobs != source_job_ids:
            raise ValueError("export manifest jobs do not match JSONL")
        evidence["source_export"] = {
            "sessions": int(export_manifest["sessions"]),
            "sha256": str(export_manifest["sha256"]),
        }
    return evidence


def treatment_keys(split_path: Path, exclusions_path: Path) -> set[str]:
    split = json.loads(split_path.read_text())
    exclusions = json.loads(exclusions_path.read_text())
    excluded_ids = {row["task_version_id"] for row in exclusions["exclusions"]}
    return {
        row["task_key"]
        for row in split["tasks"]
        if row["split"] == "train" and row["task_version_id"] not in excluded_ids
    }


def tool_schema_receipt(tools: list[dict[str, Any]], *, required: list[str]) -> dict[str, Any]:
    """Canonical hashes for runtime-discovered schemas, without trusting discovery order."""
    by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = str((tool.get("function") or {}).get("name") or "")
        if not name or name in by_name:
            raise ValueError("runtime tool schemas must have unique non-empty function names")
        by_name[name] = tool
    if set(by_name) != set(required):
        raise ValueError(f"runtime tool set must be exactly {required!r}, got {sorted(by_name)!r}")
    canonical = [by_name[name] for name in required]
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "runtime_tool_schema_receipt_v1",
        "required_order": required,
        "canonical_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "per_tool_sha256": {
            name: "sha256:"
            + hashlib.sha256(
                json.dumps(by_name[name], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for name in required
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    keys = None
    if args.split or args.exclusions:
        if not args.split or not args.exclusions:
            parser.error("--split and --exclusions must be supplied together")
        keys = treatment_keys(args.split, args.exclusions)
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    print(
        json.dumps(
            analyze_export(args.export, treatment_task_keys=keys, export_manifest=manifest),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
