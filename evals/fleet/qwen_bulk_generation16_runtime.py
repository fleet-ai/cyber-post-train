"""Runtime adapter and immutable G15 gate for Qwen Generation-16 bulk."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_bulk_generation16 as authority
from evals.fleet import self_hosted


def validate_g15_gate(value: dict[str, Any]) -> None:
    expected_keys = {
        "api_session",
        "cell_id",
        "cleanup_completed",
        "config_sha256",
        "credentials_included",
        "execution_id",
        "inner_acceptance_receipt_sha256",
        "job",
        "model",
        "pod",
        "prompts_traces_flags_or_scores_included",
        "receipt_sha256",
        "schema_version",
        "status",
        "task_key",
        "task_version_id",
        "top_acceptance_receipt_sha256",
        "workload_uid",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != "fleet-qwen38-generation15-accepted-gate-v1"
        or value.get("status") != "ACCEPTED"
        or value.get("model") != "qwen3.8-27b"
        or value.get("cell_id")
        != "sha256:631c9d7cc5328849ce137393943927192b1b50dc60458cdb3425fbce893ecf5a"
        or value.get("cleanup_completed") is not True
        or value.get("job", {}).get("exclusive_complete") is not True
        or value.get("api_session", {}).get("exact_session_present") is not True
        or value.get("api_session", {}).get("model_projection") != "omitted"
        or value.get("api_session", {}).get("transcript_content_read") is not False
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or value.get("receipt_sha256")
        != self_hosted.digest_without(value, "receipt_sha256")
    ):
        raise RuntimeError("Generation-15 accepted Qwen gate drifted")
    for field in (
        "config_sha256",
        "execution_id",
        "inner_acceptance_receipt_sha256",
        "top_acceptance_receipt_sha256",
    ):
        if authority.SHA256_RE.fullmatch(str(value.get(field))) is None:
            raise RuntimeError("Generation-15 accepted digest binding drifted")
    for value_id in (
        value.get("workload_uid"),
        value.get("job", {}).get("uid"),
        value.get("pod", {}).get("uid"),
        value.get("api_session", {}).get("session_id"),
        value.get("api_session", {}).get("verifier_execution_id"),
        value.get("task_version_id"),
    ):
        if authority.UUID_RE.fullmatch(str(value_id)) is None:
            raise RuntimeError("Generation-15 accepted UUID binding drifted")


def runtime_gate(plan: dict[str, Any]) -> None:
    expected_digest = os.environ.get("QWEN_G15_ACCEPTED_GATE_SHA256")
    package_commit = os.environ.get("BULK_PACKAGE_COMMIT")
    if (
        expected_digest is None
        or authority.SHA256_RE.fullmatch(expected_digest) is None
        or package_commit is None
        or authority.COMMIT_RE.fullmatch(package_commit) is None
    ):
        raise RuntimeError("Generation-16 runtime environment gate is incomplete")
    gate = authority.load(Path(plan["repo_root"]) / authority.G15_GATE_PATH)
    validate_g15_gate(gate)
    if gate["receipt_sha256"] != expected_digest:
        raise RuntimeError("Generation-15 accepted gate digest drifted")
    plans = authority.validate_all(Path(plan["repo_root"]))
    attempts = [row for built in plans.values() for row in built["attempts"]]
    if (
        len(attempts) != 395
        or any(row["selection_rank"] == 2 for row in attempts)
        or any(row["selection_rank"] == 4 and row["attempt"] == 1 for row in attempts)
        or plan["execution"]["endpoint_lease"]["maximum_streams"] != 2
        or plan["execution"]["attempts_per_task_sequential"] is not True
    ):
        raise RuntimeError("Generation-16 runtime partition gate drifted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "validate-gate"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--proxy", type=Path)
    args = parser.parse_args()
    if args.command == "validate-gate":
        validate_g15_gate(authority.load(Path(authority.G15_GATE_PATH)))
        return 0
    if args.plan is None or args.out is None or args.proxy is None:
        parser.error("run requires --plan, --out, and --proxy")
    plan = authority.load(args.plan)
    previous_bulk = engine.bulk
    try:
        engine.bulk = authority
        engine.run_controller(
            plan,
            out=args.out,
            proxy=args.proxy,
            runtime_gate_check=runtime_gate,
        )
    finally:
        engine.bulk = previous_bulk
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
