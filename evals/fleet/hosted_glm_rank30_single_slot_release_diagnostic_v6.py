"""Fresh score-free diagnostic for the rank-30 release-projection boundary."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_rank30_single_slot_release_diagnostic_v5 as prior
from evals.fleet import hosted_glm_rank30_single_slot_v4 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank30-release-phase-diagnostic-v6"
JOB_NAME = "chris-glm53-r030-release-diagnostic-v6"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "DIAGNOSTIC.json"
PRIOR_RECEIPT_SHA256 = (
    "sha256:9e8899367fd436ef06aea859447f3063afc0774c89577612f66b10950e3d1d02"
)


def _validate_projection(_root: Path, state: dict[str, Any]) -> None:
    projection = successor.release_projection(state["plan"])
    expected = [
        {
            "attempt": row["attempt"],
            "cell_id": row["cell_id"],
            "execution_id": row["execution_id"],
            "run_id": row["run_id"],
        }
        for row in state["plan"]["attempts"]
    ]
    cells = projection.get("cells")
    if any(
        (
            projection.get("selection_rank") != 30,
            not isinstance(cells, list),
            cells != expected,
            [row.get("attempt") for row in cells or []] != [1, 2, 3, 4],
        )
    ):
        raise RuntimeError("rank-30 release projection cells drifted")
    state["projection_validated"] = True
    state["projected_attempts"] = [1, 2, 3, 4]


PHASES: tuple[prior.prior.Phase, ...] = (
    *prior.PHASES[:8],
    ("09-release-projection", _validate_projection),
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
        projected_attempts=state.get("projected_attempts", []),
    )
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body


def run(
    root: Path, *, output_path: Path = OUTPUT_PATH,
    phases: Sequence[prior.prior.Phase] = PHASES,
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
