"""Fresh score-free rank-29 release diagnostic with runtime-plan binding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_v1 as prior
from evals.fleet import hosted_glm_rank29_a3a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank29-release-phase-diagnostic-v2"
JOB_NAME = "chris-glm53-exact100-hosted-r029-a3a4-release-diagnostic-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "DIAGNOSTIC.json"
STATIC_PLAN_SHA256 = prior.PLAN_SHA256
RUNTIME_PLAN_SHA256 = (
    "sha256:2c5d68d34f9370188e4fb3446be05a02a9e716e2cc6d9b0b674ecd9b13afa60d"
)
INVENTORY_RECEIPT_SHA256 = (
    "sha256:bf7fae379833aca5c914e1a2761ab608a805ee6f2424c714c3ef060e318fb9d3"
)


def _build_runtime_plan(root: Path, state: dict[str, Any]) -> None:
    static_plan = successor.validate_all(root)[successor.CONTROLLER]
    if static_plan.get("plan_sha256") != STATIC_PLAN_SHA256:
        raise RuntimeError("rank-29 static plan binding drifted")
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    if inventory.get("receipt_sha256") != INVENTORY_RECEIPT_SHA256:
        raise RuntimeError("rank-29 inventory receipt binding drifted")
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    if plan.get("plan_sha256") != RUNTIME_PLAN_SHA256:
        raise RuntimeError("rank-29 runtime plan binding drifted")
    state["inventory_receipt_sha256"] = INVENTORY_RECEIPT_SHA256
    state["static_plan_sha256"] = STATIC_PLAN_SHA256
    state["runtime_plan_sha256"] = RUNTIME_PLAN_SHA256
    state["plan"] = plan


PHASES: tuple[prior.Phase, ...] = (
    *prior.PHASES[:6],
    ("07-runtime-plan", _build_runtime_plan),
    *prior.PHASES[7:],
)


def _receipt(
    *,
    completed: list[str],
    failed_phase: str | None,
    error: Exception | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    body = prior._receipt(  # noqa: SLF001
        completed=completed,
        failed_phase=failed_phase,
        error=error,
        state=state,
    )
    body.pop("receipt_sha256")
    body.update(
        schema_version=SCHEMA,
        diagnostic_job=JOB_NAME,
        diagnostic_configmap=CONFIGMAP_NAME,
        phase_order=[name for name, _function in PHASES],
        prior_diagnostic_job=prior.JOB_NAME,
        prior_diagnostic_receipt_sha256=(
            "sha256:751d2a4b1b9b40762c21a7416992111777384b4c463eb6c0ed62b25fbd95948d"
        ),
        inventory_receipt_sha256=state.get("inventory_receipt_sha256"),
        static_plan_sha256=state.get("static_plan_sha256"),
        runtime_plan_sha256=state.get("runtime_plan_sha256"),
    )
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body


def run(
    root: Path,
    *,
    output_path: Path = OUTPUT_PATH,
    phases: Sequence[prior.Phase] = PHASES,
) -> int:
    if [name for name, _function in phases] != [name for name, _function in PHASES]:
        raise RuntimeError("diagnostic phase order drifted")
    completed: list[str] = []
    failed_phase: str | None = None
    error: Exception | None = None
    state: dict[str, Any] = {}
    for name, function in phases:
        try:
            function(root, state)
        except Exception as exc:  # terminal evidence must survive a phase failure
            failed_phase = name
            error = exc
            break
        completed.append(name)
    receipt = _receipt(
        completed=completed,
        failed_phase=failed_phase,
        error=error,
        state=state,
    )
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(output_path, receipt)  # noqa: SLF001
    return int(error is not None)


def main() -> int:
    return run(Path.cwd())


if __name__ == "__main__":
    raise SystemExit(main())
