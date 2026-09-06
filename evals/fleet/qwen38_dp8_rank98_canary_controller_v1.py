"""Executable rank-98/a1 DP8 canary behind an explicit root release.

The controller reserves all four canonical G19 claims atomically, executes only
attempt 1, and leaves attempts 2-4 held after formal acceptance.  This module is
packaged for future review but must not run without a digest-valid root release.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import qwen38_dedicated_scored_canary_v1 as legacy
from evals.fleet import qwen38_dp8_rank98_canary_partition_v1 as partition
from evals.fleet import self_hosted

ROOT_RELEASE_SCHEMA = "fleet-qwen38-dp8-r98-a1-root-release-v1"
PLAN_SCHEMA = "fleet-qwen38-dp8-r98-canary-plan-v1"
CLAIM_SCHEMA = "fleet-exact-pass4-bulk-cell-execution-claim-v3"
RESERVATION_SCHEMA = "fleet-qwen38-dp8-r98-four-claim-reservation-v1"
CONTROLLER = "qwen-dedicated-dp8-r98-a1-canary-v1"
SOURCE = Path("evals/fleet/configs/qwen-hosted-generation19-qwen-b-v4.json")
CLAIM_ROOT = Path("/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1")
ROLLBACK_ROOT = Path("/mnt/sfs/claim-reservation-rollbacks/opencode11827-autocontinue-v1")


class ControllerError(RuntimeError):
    """The executable canary boundary is not authorized or exact."""


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _fleet_failure_evidence(exc: BaseException) -> dict[str, Any]:
    if not isinstance(exc, self_hosted.FleetRequestError):
        return {}
    return {
        "error_code": "fleet_http_error",
        "method": exc.method,
        "route": exc.route,
        "http_status": exc.status_code,
    }


def validate_root_release(
    value: Mapping[str, Any],
    binding: Mapping[str, Any],
    qualification_result: Mapping[str, Any],
    held_partition: Mapping[str, Any],
    held_package_sha256: str,
) -> None:
    if (
        binding.get("receipt_sha256")
        != self_hosted.digest_without(dict(binding), "receipt_sha256")
        or qualification_result.get("receipt_sha256")
        != self_hosted.digest_without(dict(qualification_result), "receipt_sha256")
        or held_partition.get("receipt_sha256")
        != self_hosted.digest_without(dict(held_partition), "receipt_sha256")
        or not isinstance(value.get("root_review_receipt_sha256"), str)
        or not value["root_review_receipt_sha256"].startswith("sha256:")
        or len(value["root_review_receipt_sha256"]) != 71
        or value.get("root_review_authority") != "root"
        or value.get("receipt_sha256")
        != self_hosted.digest_without(dict(value), "receipt_sha256")
        or value.get("schema_version") != ROOT_RELEASE_SCHEMA
        or value.get("status") != "RELEASED_ONE_A1_CANARY"
        or value.get("root_reviewed") is not True
        or value.get("server_binding_receipt_sha256") != binding.get("receipt_sha256")
        or value.get("qualification_result_sha256")
        != qualification_result.get("receipt_sha256")
        or value.get("held_partition_receipt_sha256")
        != held_partition.get("receipt_sha256")
        or value.get("held_package_sha256") != held_package_sha256
        or value.get("fresh_live_scan_receipt_sha256")
        != held_partition.get("fresh_live_scan_receipt_sha256")
        or value.get("controller_job_name")
        != held_partition["held_controller_identity"]["job_name"]
        or value.get("controller_configmap_name")
        != held_partition["held_controller_identity"]["configmap_name"]
        or value.get("active_attempt") != 1
        or value.get("held_attempts") != [2, 3, 4]
        or value.get("all_four_claims_absent") is not True
        or value.get("all_four_outputs_absent") is not True
        or value.get("exact_session_and_verifier_matches") != 0
        or value.get("accepted_receipt_matches") != 0
        or value.get("launch_authorized") is not True
        or value.get("scoring_authorized") is not True
        or value.get("controller_create_permitted") is not True
        or value.get("prompts_traces_flags_scores_or_model_outputs_included") is not False
    ):
        raise ControllerError("explicit root release is absent or stale")


def _assert_formal_acceptance(value: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    item = plan["item"]
    if (
        value.get("accepted") is not True
        or value.get("credited") is not True
        or value.get("retry_allowed") is not False
        or value.get("cell_id") != item["cell_id"]
        or value.get("execution_id") != item["execution_id"]
        or value.get("run_id") != item["run_id"]
        or value.get("attempt") != 1
        or value.get("session_ingest_completed") is not True
        or value.get("cleanup_completed") is not True
        or not value.get("session_id")
        or not value.get("verifier_execution_id")
    ):
        raise ControllerError("attempt 1 lacks formal authoritative acceptance")


@contextmanager
def _binding(binding: Mapping[str, Any]) -> Iterator[None]:
    replacements = {
        "CONTROLLER": CONTROLLER,
        "SOURCE": SOURCE,
        "SERVICE_ORIGIN": binding["service_origin"],
        "SERVICE_UID": binding["service_uid"],
        "RAYJOB_UID": binding["rayjob_uid"],
        "WORKLOAD_UID": binding["workload_uid"],
        "TRAFFIC": Path(binding["traffic_path"]),
        "CLAIM_ROOT": CLAIM_ROOT,
    }
    prior = {name: getattr(legacy, name) for name in replacements}
    try:
        for name, item in replacements.items():
            setattr(legacy, name, item)
        yield
    finally:
        for name, item in prior.items():
            setattr(legacy, name, item)


def build_plans(root: Path, binding: Mapping[str, Any], release: Mapping[str, Any]) -> list[dict]:
    plans = []
    with _binding(binding):
        for attempt, cell, execution, run in zip(
            range(1, 5),
            partition.CELL_IDS,
            partition.EXECUTION_IDS,
            partition.RUN_IDS,
            strict=True,
        ):
            plan = legacy.build_plan(
                root,
                attempt,
                selection_rank=partition.SELECTION_RANK,
                execution_generation=19,
                run_id_override=run,
                expected_cell_id=cell,
                expected_execution_id=execution,
                expected_task_version_id=partition.TASK_VERSION_ID,
            )
            plan["schema_version"] = PLAN_SCHEMA
            plan["controller"] = CONTROLLER
            plan["config"]["serving"] = {
                "kind": "dedicated_qwen_dp8_canary_v1",
                "serving_block": binding["serving_block"],
                "service_origin": binding["service_origin"],
                "service_uid": binding["service_uid"],
                "rayjob_uid": binding["rayjob_uid"],
                "workload_uid": binding["workload_uid"],
                "parity_receipt_sha256": binding["parity_receipt_sha256"],
                "data_parallel_size": 8,
                "hosted_and_dedicated_results_must_remain_explicit_blocks": True,
            }
            plan["config"]["config_sha256"] = self_hosted.digest_without(
                plan["config"], "config_sha256"
            )
            plan["root_release_receipt_sha256"] = release["receipt_sha256"]
            plan["reservation_state"] = "active" if attempt == 1 else "reservation_only"
            plan["launch_authorized"] = attempt == 1
            plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
            plans.append(plan)
    return plans


def canonical_claim_path(execution_id: str, claim_root: Path = CLAIM_ROOT) -> Path:
    if not execution_id.startswith("sha256:") or len(execution_id) != 71:
        raise ValueError("invalid execution identity")
    return claim_root / f"{execution_id.removeprefix('sha256:')}.json"


def preclaim_checks(
    root: Path,
    binding: Mapping[str, Any],
    release: Mapping[str, Any],
    key: str,
    *,
    claim_root: Path = CLAIM_ROOT,
) -> list[dict]:
    plans = build_plans(root, binding, release)
    if claim_root.is_symlink() or not claim_root.is_dir():
        raise ControllerError("canonical claim mount is absent")
    for plan in plans:
        if Path(plan["output_root"]).exists() or canonical_claim_path(
            plan["item"]["execution_id"], claim_root
        ).exists():
            raise ControllerError("fresh claim or output boundary changed")
        with _binding(binding):
            legacy._live_checks(plan, key)  # noqa: SLF001 - identical production callback
    return plans


def _assert_authoritative_sessions_clear(plans: list[dict], key: str) -> None:
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}"}, timeout=1800
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise ControllerError("Fleet team identity drifted")
        for plan in plans:
            item = plan["item"]
            sessions = self_hosted._task_sessions(client, plan["config"]["task"]["key"])
            if any(
                (row.get("metadata") or {}).get("cell_id") == item["cell_id"]
                or (row.get("metadata") or {}).get("execution_id") == item["execution_id"]
                or (row.get("metadata") or {}).get("run_id") == item["run_id"]
                for row in sessions
            ):
                raise ControllerError("authoritative session collision appeared")


def _claim_payload(
    plan: Mapping[str, Any], job_uid: str, pod_uid: str, release: Mapping[str, Any]
) -> dict[str, Any]:
    uuid.UUID(job_uid)
    uuid.UUID(pod_uid)
    return _seal(
        {
            "schema_version": CLAIM_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "controller": CONTROLLER,
            "cell_id": plan["item"]["cell_id"],
            "execution_id": plan["item"]["execution_id"],
            "execution_generation": 19,
            "run_id": plan["item"]["run_id"],
            "selection_rank": partition.SELECTION_RANK,
            "attempt": plan["item"]["attempt"],
            "reservation_state": plan["reservation_state"],
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "root_release_receipt_sha256": release["receipt_sha256"],
            "immutable": True,
            "automatic_retry": False,
            "model_call_started_when_claim_written": False,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    )


def _rollback_pre_model(
    created: list[tuple[dict, dict, Path]],
    *,
    job_uid: str,
    pod_uid: str,
    reservation_run_id: str,
    rollback_root: Path = ROLLBACK_ROOT,
) -> None:
    if any(Path(plan["output_root"]).exists() for plan, _claim, _path in created):
        raise ControllerError("rollback crossed output or model boundary")
    for _plan, claim, path in created:
        if (
            path.is_symlink()
            or json.loads(path.read_text()) != claim
            or claim.get("job_uid") != job_uid
            or claim.get("pod_uid") != pod_uid
            or claim.get("model_call_started_when_claim_written") is not False
        ):
            raise ControllerError("rollback ownership or claim bytes drifted")
    rollback_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    proof = _seal(
        {
            "schema_version": "fleet-qwen38-dp8-r98-pre-model-claim-rollback-v1",
            "status": "ROLLBACK_AUTHORIZED_SAME_UID_OWNER_ZERO_MODEL_REQUESTS",
            "reservation_run_id": reservation_run_id,
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "claim_receipt_sha256s": [claim["receipt_sha256"] for _p, claim, _x in created],
            "model_requests": 0,
            "prompts_traces_flags_scores_or_model_outputs_included": False,
        }
    )
    self_hosted.write_json_once(rollback_root / f"{reservation_run_id}.json", proof)
    for _plan, _claim, path in created:
        path.unlink()


def reserve_all(
    plans: list[dict],
    release: Mapping[str, Any],
    job_uid: str,
    pod_uid: str,
    *,
    claim_root: Path = CLAIM_ROOT,
    rollback_root: Path = ROLLBACK_ROOT,
) -> dict[int, dict]:
    created: list[tuple[dict, dict, Path]] = []
    try:
        for plan in reversed(plans):
            claim = _claim_payload(plan, job_uid, pod_uid, release)
            path = canonical_claim_path(plan["item"]["execution_id"], claim_root)
            self_hosted.write_json_once(path, claim)
            created.append((plan, claim, path))
    except Exception:
        if created:
            _rollback_pre_model(
                created,
                job_uid=job_uid,
                pod_uid=pod_uid,
                reservation_run_id=release["reservation_run_id"],
                rollback_root=rollback_root,
            )
        raise
    return {plan["item"]["attempt"]: claim for plan, claim, _path in created}


def prepare_before_model_call(
    root: Path,
    binding: Mapping[str, Any],
    release: Mapping[str, Any],
    key: str,
    job_uid: str,
    pod_uid: str,
) -> tuple[list[dict], dict[int, dict], Path]:
    plans = preclaim_checks(root, binding, release, key)
    claims = reserve_all(plans, release, job_uid, pod_uid)
    try:
        _assert_authoritative_sessions_clear(plans, key)
    except Exception:
        created = [
            (
                plan,
                claims[plan["item"]["attempt"]],
                canonical_claim_path(plan["item"]["execution_id"]),
            )
            for plan in plans
        ]
        _rollback_pre_model(
            created,
            job_uid=job_uid,
            pod_uid=pod_uid,
            reservation_run_id=release["reservation_run_id"],
        )
        raise
    reservation_root = Path("/mnt/sfs/jobs") / release["reservation_run_id"]
    reservation = _seal(
        {
            "schema_version": RESERVATION_SCHEMA,
            "status": "RESERVED_A1_ACTIVE_A2_A4_HELD",
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "root_release_receipt_sha256": release["receipt_sha256"],
            "plan_sha256s": [plan["plan_sha256"] for plan in plans],
            "claim_receipt_sha256s": [claims[index]["receipt_sha256"] for index in range(1, 5)],
            "active_attempt": 1,
            "held_attempts": [2, 3, 4],
            "all_four_claims_before_model_call": True,
            "g19_skips_all_four": True,
            "prompts_traces_flags_scores_or_model_outputs_included": False,
        }
    )
    try:
        reservation_root.mkdir(mode=0o700, parents=False)
        self_hosted.write_json_once(reservation_root / "RESERVATION.json", reservation)
    except Exception:
        created = [
            (
                plan,
                claims[plan["item"]["attempt"]],
                canonical_claim_path(plan["item"]["execution_id"]),
            )
            for plan in plans
        ]
        _rollback_pre_model(
            created,
            job_uid=job_uid,
            pod_uid=pod_uid,
            reservation_run_id=release["reservation_run_id"],
        )
        raise
    return plans, claims, reservation_root


def run_a1(
    root: Path,
    proxy: Path,
    binding: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key or not job_uid or not pod_uid:
        raise ControllerError("controller credentials or downward UIDs are absent")
    plans, claims, reservation_root = prepare_before_model_call(
        root, binding, release, key, job_uid, pod_uid
    )
    plan = plans[0]
    output = Path(plan["output_root"])
    output.mkdir(mode=0o700, parents=False)
    self_hosted.write_json_once(output / "PLAN.json", plan)
    stop = threading.Event()
    with _binding(binding):
        watcher = threading.Thread(target=legacy._traffic_loop, args=(stop,), daemon=True)
        watcher.start()
        try:
            attempt_root = output / "attempt"
            self_hosted.run(plan["config"], attempt_root, proxy)
            accepted = legacy._classify(attempt_root, plan, claims[1], key)
            _assert_formal_acceptance(accepted, plan)
            accepted["serving_block"] = binding["serving_block"]
            accepted = _seal(
                {key: item for key, item in accepted.items() if key != "receipt_sha256"}
            )
            self_hosted.write_json_once(output / "ACCEPTED.json", accepted)
            hold = _seal(
                {
                    "schema_version": "fleet-qwen38-dp8-r98-a2-a4-hold-v1",
                    "status": "HELD_AFTER_FORMAL_A1_ACCEPTANCE",
                    "accepted_a1_receipt_sha256": accepted["receipt_sha256"],
                    "held_attempts": [2, 3, 4],
                    "same_job_uid": job_uid,
                    "same_pod_uid": pod_uid,
                    "same_server_binding_receipt_sha256": binding["receipt_sha256"],
                    "continuation_authorized": False,
                    "prompts_traces_flags_scores_or_model_outputs_included": False,
                }
            )
            self_hosted.write_json_once(reservation_root / "ATTEMPTS-2-4-HELD.json", hold)
            return hold
        except BaseException as exc:
            abort = _seal(
                {
                    "schema_version": "fleet-qwen38-dp8-r98-a1-controller-abort-v1",
                    "status": "CLAIMS_RETAINED_REVIEW_REQUIRED",
                    "error_type": type(exc).__name__,
                    **_fleet_failure_evidence(exc),
                    "a1_retry_authorized": False,
                    "attempts_2_to_4_transfer_authorized": False,
                    "prompts_traces_flags_scores_or_model_outputs_included": False,
                }
            )
            self_hosted.write_json_once(reservation_root / "CANARY-ABORT.json", abort)
            raise
        finally:
            stop.set()
            watcher.join(timeout=5)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ControllerError(f"controller input is not an object: {path}")
    return value


def main() -> int:
    root = Path(os.environ.get("CYBER_ROOT", "/workspace/cyber-post-train"))
    binding = _read(root / "server-binding.json")
    qualification = _read(root / "qualification-result.json")
    held = _read(root / "release.json")
    release = _read(root / "root-release.json")
    validate_root_release(
        release,
        binding,
        qualification,
        held,
        os.environ.get("QWEN_DP8_HELD_PACKAGE_SHA256", ""),
    )
    run_a1(root, root / "evals/fleet/fixed_proxy.py", binding, release)
    return 0


if __name__ == "__main__":
    sys.exit(main())
