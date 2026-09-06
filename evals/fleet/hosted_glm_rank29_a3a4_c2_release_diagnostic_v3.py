"""Fresh score-free rank-29 diagnostic with bounded global-scan phases."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_diagnostic_v2 as prior
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_v1 as release
from evals.fleet import hosted_glm_rank29_a3a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank29-release-phase-diagnostic-v3"
JOB_NAME = "chris-glm53-exact100-hosted-r029-a3a4-release-diagnostic-v3"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "DIAGNOSTIC.json"
PRIOR_RECEIPT_SHA256 = (
    "sha256:11e06f2607b357835f5d116a026f3a9c0e5d00644008df754de9fc8510084891"
)
ERROR_TYPE_CLASSES = {
    AttributeError: "attribute-error",
    KeyError: "key-error",
    TypeError: "type-error",
    UnicodeDecodeError: "unicode-decode-error",
    RuntimeError: "runtime-error",
    ValueError: "value-error",
    OSError: "filesystem-error",
}


def _safe_error_type_class(error: Exception | None) -> str | None:
    if error is None:
        return None
    for error_type, classification in ERROR_TYPE_CLASSES.items():
        if isinstance(error, error_type):
            return classification
    return "unclassified-exception"


def _scan_global_evidence(
    plan: dict[str, Any],
    state: dict[str, Any],
    *,
    jobs_root: Path = release.JOBS_ROOT,
    claim_root: Path | str = successor.CLAIM_ROOT,
) -> dict[str, int]:
    scan = {
        "subphase": "root-validation",
        "path_category": "root",
        "claim_files_examined_before_terminal": 0,
        "accepted_files_examined_before_terminal": 0,
        "output_run_ids_examined_before_terminal": 0,
    }
    state["global_scan"] = scan
    jobs_root = Path(jobs_root)
    claim_root = Path(claim_root)
    if (
        jobs_root.is_symlink()
        or not jobs_root.is_dir()
        or claim_root.is_symlink()
        or not claim_root.is_dir()
    ):
        raise RuntimeError("rank-29 global evidence roots are unsafe")
    current = plan["attempts"]
    retired = plan["partition"]["source_generation_1_retired_unclaimed"]
    cells = {row["cell_id"] for row in current}
    fresh_executions = {row["execution_id"] for row in current}
    retired_executions = {row["execution_id"] for row in retired}
    run_ids = {row["run_id"] for row in current} | {row["run_id"] for row in retired}
    target_identities = cells | fresh_executions | retired_executions | run_ids
    result = {
        "global_claim_files_examined": 0,
        "global_accepted_files_examined": 0,
        "fresh_generation_claim_collisions": 0,
        "retired_generation_claim_collisions": 0,
        "global_cell_claim_collisions": 0,
        "global_accepted_evidence_collisions": 0,
        "global_output_evidence_collisions": 0,
    }
    scan.update(subphase="claim-scan", path_category="claim")
    for path in sorted(claim_root.glob("*.json")):
        result["global_claim_files_examined"] += 1
        scan["claim_files_examined_before_terminal"] += 1
        identities = release._identity_values(release._safe_receipt(path))  # noqa: SLF001
        path_execution = "sha256:" + path.stem
        result["fresh_generation_claim_collisions"] += int(
            path_execution in fresh_executions
            or bool(identities.intersection(fresh_executions))
        )
        result["retired_generation_claim_collisions"] += int(
            path_execution in retired_executions
            or bool(identities.intersection(retired_executions))
        )
        result["global_cell_claim_collisions"] += int(
            bool(identities.intersection(cells))
        )
    scan.update(subphase="accepted-scan", path_category="accepted")
    for path in sorted(jobs_root.glob("*/accepted/*.json")):
        result["global_accepted_files_examined"] += 1
        scan["accepted_files_examined_before_terminal"] += 1
        identities = release._identity_values(release._safe_receipt(path))  # noqa: SLF001
        result["global_accepted_evidence_collisions"] += int(
            path.stem in run_ids or bool(identities.intersection(target_identities))
        )
    scan.update(subphase="output-scan", path_category="output")
    for run_id in run_ids:
        scan["output_run_ids_examined_before_terminal"] += 1
        result["global_output_evidence_collisions"] += sum(
            int(path.exists() or path.is_symlink())
            for path in jobs_root.glob(f"*/attempts/{run_id}")
        )
    scan.update(subphase="complete", path_category=None)
    return result


def _validate_global_evidence(
    _root: Path, state: dict[str, Any]
) -> None:
    evidence = _scan_global_evidence(state["plan"], state)
    if any(value for key, value in evidence.items() if key.endswith("collisions")):
        raise RuntimeError("rank-29 global collision evidence is not clear")
    state["global_evidence"] = evidence


PHASES: tuple[prior.prior.Phase, ...] = (
    *prior.PHASES[:7],
    ("08-global-evidence", _validate_global_evidence),
    prior.PHASES[8],
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
    scan = state.get("global_scan", {})
    body.update(
        schema_version=SCHEMA,
        diagnostic_job=JOB_NAME,
        diagnostic_configmap=CONFIGMAP_NAME,
        phase_order=[name for name, _function in PHASES],
        prior_diagnostic_job=prior.JOB_NAME,
        prior_diagnostic_receipt_sha256=PRIOR_RECEIPT_SHA256,
        error_type_class=_safe_error_type_class(error),
        global_scan_subphase=scan.get("subphase"),
        global_scan_path_category=scan.get("path_category"),
        claim_files_examined_before_terminal=scan.get(
            "claim_files_examined_before_terminal", 0
        ),
        accepted_files_examined_before_terminal=scan.get(
            "accepted_files_examined_before_terminal", 0
        ),
        output_run_ids_examined_before_terminal=scan.get(
            "output_run_ids_examined_before_terminal", 0
        ),
    )
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body


def run(
    root: Path,
    *,
    output_path: Path = OUTPUT_PATH,
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
