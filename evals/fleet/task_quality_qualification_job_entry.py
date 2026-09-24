"""Run one sealed task-quality plan inside its bounded CPU Job."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
from typing import Any

from evals.fleet import task_quality_qualification as qualification

ENTRY_RECEIPT_SCHEMA = "cyber_task_quality_cpu_job_receipt_v1"


class JobEntryError(RuntimeError):
    """The one-cell Job did not complete its exact qualification contract."""


def _mkdir_once(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.mkdir(mode=0o700)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _all_one(value: object) -> bool:
    return isinstance(value, dict) and bool(value) and all(item == 1 for item in value.values())


def run(
    *, plan_path: Path, attestation_path: Path, private_root: Path, api_key: str
) -> dict[str, Any]:
    previous = os.environ.get(qualification.PACKAGED_SOURCE_ENV)
    os.environ[qualification.PACKAGED_SOURCE_ENV] = str(attestation_path)
    try:
        return _run(plan_path=plan_path, private_root=private_root, api_key=api_key)
    finally:
        if previous is None:
            os.environ.pop(qualification.PACKAGED_SOURCE_ENV, None)
        else:
            os.environ[qualification.PACKAGED_SOURCE_ENV] = previous


def _run(*, plan_path: Path, private_root: Path, api_key: str) -> dict[str, Any]:
    plan = qualification._read(plan_path, "qualification plan")  # noqa: SLF001
    qualification._sealed(plan, qualification.PLAN_SCHEMA, "qualification plan")  # noqa: SLF001
    qualification._source_matches_plan(  # noqa: SLF001
        plan.get("source"),
        plan_sha256=plan.get("sha256"),
        exact_task_identity=plan.get("selection", {}).get("exact_task_identity"),
    )
    if (
        len(plan.get("tasks", [])) != 1
        or plan.get("execution", {}).get("concurrency") != 1
        or plan.get("execution", {}).get("model_calls") != 0
    ):
        raise JobEntryError("CPU Job accepts only one model-free qualification cell")

    _mkdir_once(private_root)
    qualification._write_once(private_root / "PLAN.json", plan)  # noqa: SLF001
    aggregate: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        aggregate = qualification.execute_plan(plan, private_root, api_key=api_key)
    except BaseException as error:  # noqa: BLE001 - cleanup must run for every failure
        failure = error
    finally:
        if (private_root / "cells").is_dir():
            cleanup = qualification.cleanup_plan(plan, private_root, api_key=api_key)

    expected_counts = {
        "qualified": 1,
        "infrastructure_invalid": 0,
        "quarantined_ambiguous": 0,
    }
    if (
        failure is not None
        or aggregate is None
        or aggregate.get("planned_task_versions") != 1
        or aggregate.get("counts") != expected_counts
        or not _all_one(aggregate.get("gate_pass_counts"))
        or cleanup is None
        or cleanup.get("resolved_task_versions") != 1
        or cleanup.get("unresolved_task_versions") != 0
    ):
        raise JobEntryError("one-cell qualification or exact environment cleanup did not pass")
    body = {
        "schema": ENTRY_RECEIPT_SCHEMA,
        "plan_sha256": plan["sha256"],
        "aggregate_receipt_sha256": aggregate["sha256"],
        "cleanup_receipt_sha256": cleanup["sha256"],
        "qualified_task_versions": 1,
        "model_calls": 0,
        "training_data_eligible": False,
        "prompts_traces_answers_flags_scores_or_credentials_included": False,
    }
    receipt = qualification.sealed(body)
    qualification._write_once(private_root / "JOB_RECEIPT.json", receipt)  # noqa: SLF001
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-attestation", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def terminate(_signum: int, _frame: object) -> None:
        raise JobEntryError("CPU Job received a termination signal")

    signal.signal(signal.SIGTERM, terminate)
    try:
        result = run(
            plan_path=args.plan,
            attestation_path=args.source_attestation,
            private_root=args.private_root,
            api_key=os.environ.get("FLEET_API_KEY", ""),
        )
    except BaseException as error:  # noqa: BLE001 - content-free terminal marker
        print(
            json.dumps(
                {
                    "schema": "cyber_task_quality_cpu_job_failure_v1",
                    "status": "failed_no_automatic_retry",
                    "failure_code": type(error).__name__.lower(),
                    "private_content_included": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
