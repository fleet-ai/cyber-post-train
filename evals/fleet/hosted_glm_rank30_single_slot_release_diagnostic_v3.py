"""Score-free phased diagnostic for the pre-receipt rank-30 observer failure."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_single_slot_release_v1 as release
from evals.fleet import hosted_glm_rank30_single_slot_v1 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank30-release-phase-diagnostic-v3"
JOB_NAME = "chris-glm53-exact100-hosted-r030-single-slot-release-diagnostic-v3"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "DIAGNOSTIC.json"
V2_JOB_UID = "6f626b8b-1daf-46c9-8be7-c3e3ccd96b3a"
V2_POD_UID = "ba5cc5f7-e6ba-47e6-9751-d0dd15407128"
ErrorPhase = tuple[str, Callable[[Path, dict[str, Any]], None]]
ERROR_CLASSES = {
    AttributeError: "attribute-error",
    KeyError: "key-error",
    TypeError: "type-error",
    UnicodeDecodeError: "unicode-decode-error",
    RuntimeError: "runtime-error",
    ValueError: "value-error",
    OSError: "filesystem-error",
}


def _bindings(_root: Path, state: dict[str, Any]) -> None:
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    key = os.environ.get("FLEET_API_KEY", "")
    source_sha = os.environ.get("GLM_HOSTED_R30_SOURCE_SHA256", "")
    if (
        not key
        or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid))
        or successor.SHA256_RE.fullmatch(source_sha) is None
    ):
        raise RuntimeError("rank-30 diagnostic bindings are invalid")
    state.update(job_uid=job_uid, pod_uid=pod_uid, key=key, source_sha=source_sha)


def _inventory(_root: Path, state: dict[str, Any]) -> None:
    state["inventory"] = successor.load(source_runtime.INVENTORY_PATH)


def _runtime_plan(root: Path, state: dict[str, Any]) -> None:
    state["plan"] = successor.build_runtime_plan(
        successor.CONTROLLER, state["inventory"], root
    )


def _live_peer(_root: Path, state: dict[str, Any]) -> None:
    state["peer"] = release._validate_live_peer()  # noqa: SLF001


def _lease_slots(_root: Path, state: dict[str, Any]) -> None:
    state["lease"] = release._probe_slots()  # noqa: SLF001


def _collisions(_root: Path, state: dict[str, Any]) -> None:
    state["collision"] = release._collisions(  # noqa: SLF001
        state["plan"], state["key"]
    )


def _release_validation(_root: Path, state: dict[str, Any]) -> None:
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR",
        "checked_at_utc": release._now(),  # noqa: SLF001
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_cap": 1,
        "controller": successor.release_projection(state["plan"]),
        "source_package_sha256": state["source_sha"],
        "ledger_authority": release.whole.LEDGER_AUTHORITY,
        "selection_authority": release.whole.SELECTION_AUTHORITY,
        "live_rank29_peer": state["peer"],
        "endpoint_lease_observer": state["lease"],
        "fresh_collision_reconciliation": state["collision"],
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    candidate = {
        **body,
        "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256"),
    }
    successor.validate_release(candidate, state["plan"], state["source_sha"])
    state["candidate_valid"] = True


PHASES: tuple[ErrorPhase, ...] = (
    ("01-bindings", _bindings),
    ("02-inventory", _inventory),
    ("03-runtime-plan", _runtime_plan),
    ("04-live-peer", _live_peer),
    ("05-lease-slots", _lease_slots),
    ("06-collisions", _collisions),
    ("07-release-validation", _release_validation),
)


def _error_class(error: Exception | None) -> str | None:
    if error is None:
        return None
    for error_type, classification in ERROR_CLASSES.items():
        if isinstance(error, error_type):
            return classification
    return "unclassified-exception"


def _receipt(
    completed: list[str],
    failed_phase: str | None,
    error: Exception | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    collision = state.get("collision", {})
    body = {
        "schema_version": SCHEMA,
        "status": "PASS" if error is None else "FAILED",
        "diagnostic_job": JOB_NAME,
        "diagnostic_configmap": CONFIGMAP_NAME,
        "prior_failed_job_uid": V2_JOB_UID,
        "prior_failed_pod_uid": V2_POD_UID,
        "phase_order": [name for name, _function in PHASES],
        "completed_phases": completed,
        "failed_phase": failed_phase,
        "error_type_class": _error_class(error),
        "error_message_sha256": (
            None if error is None else self_hosted.sha256(str(error).encode())
        ),
        "safe_counts": {
            key: collision.get(key, 0)
            for key in (
                "claim_files_examined",
                "accepted_files_examined",
                "session_rows_examined",
                "canonical_claim_collisions",
                "authoritative_session_collisions",
                "accepted_evidence_collisions",
                "output_root_collisions",
                "new_job_collisions",
                "new_configmap_collisions",
                "api_mutations",
            )
        },
        "score_blind_side_effects": {
            "model_calls": 0,
            "task_mutations": 0,
            "session_mutations": 0,
            "verifier_mutations": 0,
            "scoring_calls": 0,
            "claims_created": 0,
            "scored_output_roots_created": 0,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_read": False,
            "scores_read": False,
        },
    }
    return {
        **body,
        "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256"),
    }


def run(
    root: Path,
    *,
    output_path: Path = OUTPUT_PATH,
    phases: Sequence[ErrorPhase] = PHASES,
) -> int:
    if [name for name, _function in phases] != [name for name, _function in PHASES]:
        raise RuntimeError("rank-30 diagnostic phase order drifted")
    completed: list[str] = []
    failed_phase: str | None = None
    error: Exception | None = None
    state: dict[str, Any] = {}
    for name, function in phases:
        try:
            function(root, state)
        except Exception as exc:  # terminal receipt is the diagnostic purpose
            failed_phase = name
            error = exc
            break
        completed.append(name)
    receipt = _receipt(completed, failed_phase, error, state)
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    release.engine._write_once(output_path, receipt)  # noqa: SLF001
    return int(error is not None)


def main() -> int:
    return run(Path.cwd())


if __name__ == "__main__":
    raise SystemExit(main())
