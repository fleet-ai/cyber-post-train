"""Runtime wrapper for the held rank-15/rank-16 hosted successors."""

from __future__ import annotations

import argparse
import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_whole_task_successor_v1 as successor
from evals.fleet import self_hosted

CANARY_PHASES = (
    "runtime-entry",
    "build-plans-started",
    "build-plans-done",
    "package-source-load-done",
    "package-source-digest-done",
    "plan-match-done",
    "held-load-started",
    "held-load-done",
    "held-digest-done",
    "package-source-validate-done",
    "held-validate-done",
)


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def runtime_gate_check(
    plans: dict[str, dict[str, Any]],
    package_source: dict[str, Any],
    *,
    canary: bool = False,
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the exact packaged authority gate used at the engine boundary."""
    if canary:
        if phase_callback is not None:
            phase_callback("held-load-started")
        raw_path = os.environ.get("QWEN_HOSTED_WHOLE_TASK_RELEASE_PATH")
        expected = os.environ.get("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256")
        if not raw_path or not Path(raw_path).is_absolute() or not expected:
            raise RuntimeError("hosted whole-task canary held binding drifted")
        held = successor.load(Path(raw_path))
        if phase_callback is not None:
            phase_callback("held-load-done")
        if held.get("receipt_sha256") != expected:
            raise RuntimeError("hosted whole-task canary held digest drifted")
        if phase_callback is not None:
            phase_callback("held-digest-done")
        controller = str(package_source.get("controller"))
        projected = {row["controller"]: row for row in held.get("controllers") or []}
        sources = {
            name: package_source
            if name == controller
            else {"receipt_sha256": projected.get(name, {}).get("package_source_receipt_sha256")}
            for name in plans
        }
        successor.validate_package_source_receipt(package_source, controller, plans[controller])
        if phase_callback is not None:
            phase_callback("package-source-validate-done")
        successor.validate_held(held, plans, sources)
        if phase_callback is not None:
            phase_callback("held-validate-done")
        return held
    return successor.load_runtime_release(plans, package_source)


def _load_bound_inputs(
    plan: dict[str, Any],
    *,
    package_source_path: Path = Path("/bootstrap/package-source.json"),
    phase_callback: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if phase_callback is not None:
        phase_callback("build-plans-started")
    plans = successor.build_plans(Path(plan["repo_root"]))
    if phase_callback is not None:
        phase_callback("build-plans-done")
    if not package_source_path.is_absolute():
        raise RuntimeError("hosted whole-task package source path must be absolute")
    package_source = successor.load(package_source_path)
    if phase_callback is not None:
        phase_callback("package-source-load-done")
    if package_source.get("receipt_sha256") != os.environ.get(
        "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256"
    ):
        raise RuntimeError("hosted whole-task package source environment binding drifted")
    if phase_callback is not None:
        phase_callback("package-source-digest-done")
    if plan != plans.get(plan.get("controller")):
        raise RuntimeError("hosted whole-task runtime plan drifted")
    if phase_callback is not None:
        phase_callback("plan-match-done")
    return plans, package_source


def run_gate_canary(
    plan: dict[str, Any],
    *,
    receipt_path: Path,
    package_source_path: Path = Path("/bootstrap/package-source.json"),
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Revalidate the packaged runtime gate and exit before any scored boundary."""
    plans, package_source = _load_bound_inputs(
        plan,
        package_source_path=package_source_path,
        phase_callback=phase_callback,
    )
    authority = runtime_gate_check(
        plans,
        package_source,
        canary=True,
        phase_callback=phase_callback,
    )
    body = {
        "schema_version": successor.RUNTIME_GATE_CANARY_SCHEMA,
        "status": "PASS",
        "controller": plan["controller"],
        "plan_sha256": plan["plan_sha256"],
        "job_uid": os.environ.get("JOB_UID"),
        "pod_uid": os.environ.get("POD_UID"),
        "authority_schema_version": authority["schema_version"],
        "authority_receipt_sha256": authority["receipt_sha256"],
        "package_source_receipt_sha256": package_source["receipt_sha256"],
        "bootstrap_stage_reached": "06-runtime-exec",
        "output_roots_created": 0,
        "endpoint_leases_acquired": 0,
        "canonical_claims_created": 0,
        "model_calls": 0,
        "task_calls": 0,
        "session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "api_mutations": 0,
        "scores_included": False,
        "prompts_or_traces_included": False,
        "credentials_included": False,
    }
    receipt = _seal(body)
    successor.validate_runtime_gate_canary(receipt)
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    self_hosted.write_json_once(receipt_path, receipt)
    return receipt


def write_gate_canary_failure(
    receipt_path: Path, exc: Exception, *, last_completed_phase: str
) -> dict[str, Any]:
    """Persist only a sanitized exception class and digest after stage 06."""
    if last_completed_phase not in CANARY_PHASES:
        raise RuntimeError("hosted whole-task canary failure phase drifted")
    body = {
        "schema_version": "fleet-qwen38-hosted-whole-task-runtime-gate-failure-v2",
        "status": "FAILED",
        "last_completed_stage": "06-runtime-exec",
        "last_completed_phase": last_completed_phase,
        "error_type": type(exc).__name__,
        "error_sha256": self_hosted.sha256(str(exc).encode()),
        "job_uid": os.environ.get("JOB_UID"),
        "pod_uid": os.environ.get("POD_UID"),
        "authority_receipt_sha256": os.environ.get("QWEN_HOSTED_WHOLE_TASK_RELEASE_SHA256"),
        "package_source_receipt_sha256": os.environ.get(
            "QWEN_HOSTED_WHOLE_TASK_PACKAGE_SOURCE_SHA256"
        ),
        "scores_included": False,
        "prompts_or_traces_included": False,
        "credentials_included": False,
    }
    receipt = _seal(body)
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with contextlib.suppress(FileExistsError):
        self_hosted.write_json_once(receipt_path, receipt)
    return receipt


def run(plan: dict[str, Any], *, out: Path, proxy: Path, diagnostic_root: Path) -> dict[str, Any]:
    if out.is_symlink():
        raise RuntimeError("hosted whole-task output root is unsafe")
    plans, package_source = _load_bound_inputs(plan)
    runtime_gate_check(plans, package_source)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    provider = successor.AtomicWholeTaskClaims(plan, out, key=key)
    state: dict[str, Any] = {"stage": None, "item": None}

    def observe(stage: str, item: dict[str, Any] | None) -> None:
        state.update(stage=stage, item=item)
        if stage == "07-claim-written":
            if item is None:
                raise RuntimeError("hosted whole-task model boundary item is missing")
            provider.mark_model_boundary(item)

    prior = engine.bulk
    try:
        engine.bulk = successor
        return engine.run_controller(
            plan,
            out=out,
            proxy=proxy,
            runtime_gate_check=lambda _plan: runtime_gate_check(plans, package_source),
            claim_provider=provider,
            stage_observer=observe,
        )
    except Exception as exc:
        item = state["item"]
        body: dict[str, Any] = {
            "schema_version": "fleet-qwen38-hosted-whole-task-runtime-failure-v1",
            "status": "FAILED",
            "controller": plan["controller"],
            "plan_sha256": plan["plan_sha256"],
            "last_completed_stage": state["stage"],
            "reservation_committed": out.joinpath("RESERVATION.json").is_file(),
            "error_type": type(exc).__name__,
            "error_sha256": self_hosted.sha256(str(exc).encode()),
            "job_uid": os.environ.get("JOB_UID"),
            "pod_uid": os.environ.get("POD_UID"),
            "scores_included": False,
            "prompts_or_traces_included": False,
            "credentials_included": False,
        }
        if item is not None:
            body.update(
                cell_id=item["cell_id"],
                execution_id=item["execution_id"],
                selection_rank=item["selection_rank"],
                attempt=item["attempt"],
            )
        diagnostic_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):
            self_hosted.write_json_once(diagnostic_root / "FAILED.json", _seal(body))
        raise
    finally:
        engine.bulk = prior


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--runtime-gate-canary-receipt", type=Path)
    parser.add_argument(
        "--package-source",
        type=Path,
        default=Path("/bootstrap/package-source.json"),
    )
    args = parser.parse_args()
    plan = successor.load(args.plan)
    if args.runtime_gate_canary_receipt is not None:
        phase = {"last_completed_phase": "runtime-entry"}

        def phase_callback(value: str) -> None:
            phase["last_completed_phase"] = value

        try:
            run_gate_canary(
                plan,
                receipt_path=args.runtime_gate_canary_receipt,
                package_source_path=args.package_source,
                phase_callback=phase_callback,
            )
        except Exception as exc:
            write_gate_canary_failure(
                args.runtime_gate_canary_receipt.with_name("RUNTIME-GATE-CANARY-FAILED.json"),
                exc,
                last_completed_phase=phase["last_completed_phase"],
            )
            raise
    else:
        run(plan, out=args.out, proxy=args.proxy, diagnostic_root=args.diagnostic_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
