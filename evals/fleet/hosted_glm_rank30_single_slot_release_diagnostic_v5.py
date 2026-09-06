"""Fresh score-free diagnostic for the corrected rank-30 runtime-plan boundary."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v4 as prior
from evals.fleet import hosted_glm_rank30_single_slot_v4 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank30-release-phase-diagnostic-v5"
JOB_NAME = "chris-glm53-r030-release-diagnostic-v5"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "DIAGNOSTIC.json"
PRIOR_RECEIPT_SHA256 = (
    "sha256:bc309d8bb9e834f1eb8b717dcd884ec989e7b75506316bd74fd3cce32d07cd05"
)


def _build_plan(root: Path, state: dict[str, Any]) -> None:
    plan = successor.build_runtime_plan(
        successor.CONTROLLER, state["inventory"], root
    )
    state["plan"] = plan
    state["plan_sha256"] = plan["plan_sha256"]


PHASES: tuple[prior.Phase, ...] = (
    *prior.PHASES[:4],
    ("05-runtime-plan", _build_plan),
    *prior.PHASES[5:],
)


def _receipt(
    *, completed: list[str], failed_phase: str | None, error: Exception | None,
    state: dict[str, Any]
) -> dict[str, Any]:
    body = prior._receipt(  # noqa: SLF001
        completed=completed, failed_phase=failed_phase, error=error, state=state
    )
    body.pop("receipt_sha256")
    body.update(
        schema_version=SCHEMA,
        diagnostic_job=JOB_NAME,
        diagnostic_configmap=CONFIGMAP_NAME,
        prior_diagnostic_job=prior.JOB_NAME,
        prior_diagnostic_receipt_sha256=PRIOR_RECEIPT_SHA256,
        phase_order=[name for name, _function in PHASES],
    )
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body


def run(
    root: Path, *, output_path: Path = OUTPUT_PATH,
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
        except Exception as exc:  # terminal receipt is the purpose of this job
            failed_phase = name
            error = exc
            break
        completed.append(name)
    receipt = _receipt(
        completed=completed, failed_phase=failed_phase, error=error, state=state
    )
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(output_path, receipt)  # noqa: SLF001
    return int(error is not None)


def main() -> int:
    return run(Path.cwd())


if __name__ == "__main__":
    raise SystemExit(main())
