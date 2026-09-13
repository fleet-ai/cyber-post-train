"""Accept a terminal Miles reload only after independent GPU release evidence.

The reload process deliberately cannot certify its own Kubernetes teardown.  This
offline gate binds its immutable config and plan to separate controller-terminal
and post-terminal release observations, then reopens the sealed checkpoint.  It
does not call an API, Kubernetes, Miles, Ray, a verifier, or W&B.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, digest

from . import miles_reload
from .miles_conversion import _write
from .rl_runtime import sealed

CONTROLLER_SCHEMA = "cyber_miles_rl_reload_controller_terminal_v1"
RELEASE_SCHEMA = "cyber_miles_rl_reload_external_release_v1"
ACCEPTED_SCHEMA = "cyber_miles_rl_reload_accepted_v1"
NAMESPACE = "fleet-train-jobs"

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_SHA256 = re.compile(r"[a-f0-9]{64}")


def _fields(value: str) -> frozenset[str]:
    return frozenset(value.split())  # noqa: SIM905 - more reviewable schema declarations


_PROCESS_FIELDS = _fields(
    "schema status plan_sha256 source_manifest_sha256 source_terminal_acceptance_sha256 "
    "source_policy_delta_observation_sha256 world_size ranks restored_rollout_index "
    "next_rollout_id probe_set_sha256 rank_state_commitment_method "
    "rank_state_commitments_sha256 all_rank_model_loaded "
    "all_rank_optimizer_loaded all_rank_scheduler_loaded all_rank_rng_loaded "
    "all_rank_state_commitments_match state_stable_across_zero_updates optimizer_updates "
    "rollouts verifier_calls forwards backwards checkpoint_writes wandb_events "
    "source_checkpoint_unchanged external_gpu_release_verified "
    "scientific_rl_acceptance completed_at sha256"
)
_CONTROLLER_FIELDS = _fields(
    "schema status cluster api_base_url kube_context namespace namespace_uid run_name "
    "plan_sha256 request_sha256 api_run_id api_run_name rayjob_name rayjob_uid "
    "workload_name workload_uid workload_owner_rayjob_uid raycluster_name raycluster_uid "
    "raycluster_owner_rayjob_uid pods api_status controller_status effective_priority "
    "automatic_requeue workers gpus_per_worker total_gpus observed_at sha256"
)
_POD_FIELDS = _fields(
    "name uid owner_raycluster_uid phase exit_code termination_reason terminated_at "
    "runtime_image_id container_restarts gpus"
)
_RELEASE_FIELDS = _fields(
    "schema status cluster api_base_url kube_context namespace namespace_uid run_name "
    "plan_sha256 request_sha256 controller_terminal_path controller_terminal_file_sha256 "
    "controller_terminal_sha256 api_run_id api_run_name rayjob_name rayjob_uid "
    "workload_name workload_uid raycluster_name raycluster_uid pod_uids api_status "
    "controller_status rayjob_present workload_present quota_reservation_present "
    "raycluster_present gpu_pods_present active_gpu_pod_uids active_gpus observed_at sha256"
)
_WORK_FIELDS = _fields(
    "rollouts verifier_calls forwards backwards optimizer_updates checkpoint_writes wandb_events"
)
_ACCEPTED_FIELDS = _fields(
    "schema status reload_config_path reload_config_file_sha256 reload_config "
    "reload_config_sha256 reload_plan_path reload_plan_file_sha256 reload_plan "
    "reload_plan_sha256 request_sha256 reload_result_path reload_result_file_sha256 "
    "reload_result controller_terminal_path controller_terminal_file_sha256 "
    "controller_terminal external_release_path external_release_file_sha256 "
    "external_release source_manifest_sha256 source_checkpoint_unchanged_after_release "
    "source_terminal_acceptance_sha256 source_policy_delta_observation_sha256 "
    "rank_state_commitments_sha256 exact_rank_state_commitments_verified work_executed "
    "external_gpu_release_verified production_promotion_requires_this_receipt sha256"
)


def _snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Read one regular file once and reject replacement while it is read."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("acceptance input is missing or indirect")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in attributes):
        raise ValueError("acceptance input changed while being read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("acceptance input is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("acceptance input must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _time(value: object) -> float:
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("terminal evidence timestamp is invalid") from exc
        if parsed.tzinfo is not None:
            return parsed.timestamp()
    raise ValueError("terminal evidence timestamp is invalid")


def _integer(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _image_digest(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("runtime image ID is absent")
    match = re.fullmatch(
        r"(?:containerd|docker-pullable)://(?:[^@\s]+@)?sha256:([a-f0-9]{64})", value
    )
    if match is None:
        raise ValueError("runtime image ID is not an immutable observed digest")
    return match.group(1)


def _process(value: dict, plan: dict) -> float:
    sealed(value, miles_reload.RESULT_SCHEMA)
    manifest = plan["source_manifest"]
    world_size = manifest["world_size"]
    if (
        set(value) != _PROCESS_FIELDS
        or value.get("status") != "reload_validated"
        or value.get("plan_sha256") != digest(plan)
        or value.get("source_manifest_sha256") != manifest["sha256"].removeprefix("sha256:")
        or value.get("source_terminal_acceptance_sha256")
        != plan["source_terminal_acceptance"]["receipt_sha256"]
        or value.get("source_policy_delta_observation_sha256")
        != plan["source_policy_delta_observation"]["receipt_sha256"]
        or value.get("rank_state_commitment_method")
        != miles_reload.RANK_STATE_COMMITMENT_METHOD
        or not _integer(value.get("world_size"), world_size)
        or value.get("ranks") != list(range(world_size))
        or not _integer(value.get("restored_rollout_index"), manifest["rollout_index"])
        or not _integer(value.get("next_rollout_id"), manifest["next_rollout_id"])
        or not isinstance(value.get("probe_set_sha256"), str)
        or _SHA256.fullmatch(value["probe_set_sha256"].removeprefix("sha256:")) is None
        or value.get("rank_state_commitments_sha256")
        != digest(plan["expected_rank_state_commitments"])
        or any(
            value.get(key) is not True
            for key in (
                "all_rank_model_loaded",
                "all_rank_optimizer_loaded",
                "all_rank_scheduler_loaded",
                "all_rank_rng_loaded",
                "all_rank_state_commitments_match",
                "state_stable_across_zero_updates",
                "source_checkpoint_unchanged",
            )
        )
        or any(
            not _integer(value.get(key), 0)
            for key in (
                "optimizer_updates",
                "rollouts",
                "verifier_calls",
                "forwards",
                "backwards",
                "checkpoint_writes",
                "wandb_events",
            )
        )
        or value.get("external_gpu_release_verified") is not False
        or value.get("scientific_rl_acceptance") is not False
    ):
        raise ValueError("Miles reload process receipt is incomplete or conflicting")
    return _time(value.get("completed_at"))


def _controller(value: dict, plan: dict, request: dict, completed_at: float) -> float:
    sealed(value, CONTROLLER_SCHEMA)
    manifest = plan["source_manifest"]
    workers = manifest["topology"]["nodes"]
    gpus_per_worker = manifest["topology"]["gpus_per_node"]
    run_id = value.get("api_run_id")
    expected_name = f"{plan['run_name']}-{str(run_id)[:8]}"
    pods = value.get("pods")
    observed_at = _time(value.get("observed_at"))
    expected_image = plan["execution"]["image"].rsplit("@sha256:", 1)[-1]
    if (
        set(value) != _CONTROLLER_FIELDS
        or value.get("status") != "succeeded"
        or value.get("cluster") != "dev"
        or value.get("api_base_url") != API_URLS["dev"]
        or not isinstance(value.get("kube_context"), str)
        or not value["kube_context"]
        or value.get("namespace") != NAMESPACE
        or _UUID.fullmatch(str(value.get("namespace_uid"))) is None
        or value.get("run_name") != plan["run_name"]
        or value.get("plan_sha256") != digest(plan)
        or value.get("request_sha256") != digest(request)
        or _UUID.fullmatch(str(run_id)) is None
        or value.get("api_run_name") != expected_name
        or value.get("rayjob_name") != expected_name
        or any(
            _UUID.fullmatch(str(value.get(key))) is None
            for key in ("rayjob_uid", "workload_uid", "raycluster_uid")
        )
        or any(
            not isinstance(value.get(key), str) or not value[key]
            for key in ("workload_name", "raycluster_name")
        )
        or value.get("workload_owner_rayjob_uid") != value.get("rayjob_uid")
        or value.get("raycluster_owner_rayjob_uid") != value.get("rayjob_uid")
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or not _integer(value.get("effective_priority"), 10000)
        or value.get("automatic_requeue") is not False
        or not _integer(value.get("workers"), workers)
        or not _integer(value.get("gpus_per_worker"), gpus_per_worker)
        or not _integer(value.get("total_gpus"), workers * gpus_per_worker)
        or observed_at < completed_at
        or not isinstance(pods, list)
        or len(pods) != workers
    ):
        raise ValueError("Miles terminal controller evidence is incomplete or mismatched")
    pod_uids: list[str] = []
    for pod in pods:
        if (
            not isinstance(pod, dict)
            or set(pod) != _POD_FIELDS
            or not isinstance(pod.get("name"), str)
            or not pod["name"]
            or _UUID.fullmatch(str(pod.get("uid"))) is None
            or pod.get("owner_raycluster_uid") != value["raycluster_uid"]
            or pod.get("phase") != "Succeeded"
            or not _integer(pod.get("exit_code"), 0)
            or pod.get("termination_reason") != "Completed"
            or _time(pod.get("terminated_at")) < completed_at
            or _time(pod.get("terminated_at")) > observed_at
            or _image_digest(pod.get("runtime_image_id")) != expected_image
            or not _integer(pod.get("container_restarts"), 0)
            or not _integer(pod.get("gpus"), gpus_per_worker)
        ):
            raise ValueError("Miles terminal Pod evidence is incomplete or mismatched")
        pod_uids.append(pod["uid"])
    identities = [
        run_id,
        value["rayjob_uid"],
        value["workload_uid"],
        value["raycluster_uid"],
        *pod_uids,
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Miles terminal controller identities are not distinct")
    return observed_at


def _release(
    value: dict,
    plan: dict,
    request: dict,
    controller: dict,
    controller_path: Path,
    controller_file_sha256: str,
    not_before: float,
) -> None:
    sealed(value, RELEASE_SCHEMA)
    observed_at = _time(value.get("observed_at"))
    identity_keys = (
        "api_run_id",
        "api_run_name",
        "rayjob_name",
        "rayjob_uid",
        "workload_name",
        "workload_uid",
        "raycluster_name",
        "raycluster_uid",
    )
    if (
        set(value) != _RELEASE_FIELDS
        or value.get("status") != "released"
        or value.get("cluster") != "dev"
        or value.get("api_base_url") != API_URLS["dev"]
        or value.get("kube_context") != controller["kube_context"]
        or value.get("namespace") != NAMESPACE
        or value.get("namespace_uid") != controller["namespace_uid"]
        or value.get("run_name") != plan["run_name"]
        or value.get("plan_sha256") != digest(plan)
        or value.get("request_sha256") != digest(request)
        or value.get("controller_terminal_path") != str(controller_path)
        or value.get("controller_terminal_file_sha256") != controller_file_sha256
        or value.get("controller_terminal_sha256") != controller["sha256"]
        or any(value.get(key) != controller[key] for key in identity_keys)
        or value.get("pod_uids") != [pod["uid"] for pod in controller["pods"]]
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or value.get("rayjob_present") is not False
        or value.get("workload_present") is not False
        or value.get("quota_reservation_present") is not False
        or value.get("raycluster_present") is not False
        or value.get("gpu_pods_present") is not False
        or value.get("active_gpu_pod_uids") != []
        or not _integer(value.get("active_gpus"), 0)
        or observed_at < not_before
    ):
        raise ValueError("Miles external GPU release is incomplete or mismatched")


def _request(plan: dict) -> dict:
    request = miles_reload.job_request(plan)
    env = request.get("env", {})
    if (
        request.get("workers") != plan["source_manifest"]["topology"]["nodes"]
        or request.get("gpus_per_worker") != plan["source_manifest"]["topology"]["gpus_per_node"]
        or request.get("priority_class") != "c1"
        or request.get("requeueIfPreempted") is not False
        or request.get("secrets") != []
        or env.get("WANDB_MODE") != "disabled"
        or env.get("WANDB_DISABLED") != "true"
        or any("WANDB" in key and key not in {"WANDB_MODE", "WANDB_DISABLED"} for key in env)
    ):
        raise ValueError("Miles reload request does not prove a zero-W&B dev execution")
    return request


def _validate_values(
    *,
    config: dict,
    plan: dict,
    result: dict,
    controller: dict,
    release: dict,
    config_path: Path,
    controller_path: Path,
    controller_file_sha256: str,
    recompile: bool,
) -> dict:
    if recompile:
        rebuilt = miles_reload.compile_reload(config, relative_to=config_path.parent)
        if rebuilt != plan:
            raise ValueError("Miles reload config does not reconstruct the exact plan")
    request = _request(plan)
    completed_at = _process(result, plan)
    terminal_at = _controller(controller, plan, request, completed_at)
    _release(
        release,
        plan,
        request,
        controller,
        controller_path,
        controller_file_sha256,
        terminal_at,
    )
    if any(
        (Path(plan["output_root"]) / name).exists()
        for name in ("FAILED.json", "RELOAD_REJECTED.json", "REJECTED.json")
    ):
        raise ValueError("Miles reload output contains conflicting terminal evidence")
    miles_reload._verify_checkpoint(plan["source_manifest"], hashes=True)
    return request


def accept_reload(
    *,
    config_path: Path,
    plan_path: Path,
    result_path: Path,
    controller_path: Path,
    release_path: Path,
    output: Path,
) -> dict:
    """Create one promotion receipt from immutable, independently sourced evidence."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Miles reload acceptance destination already exists")
    config, config_file_sha256 = _snapshot(config_path)
    plan, plan_file_sha256 = _snapshot(plan_path)
    result, result_file_sha256 = _snapshot(result_path)
    controller, controller_file_sha256 = _snapshot(controller_path)
    release, release_file_sha256 = _snapshot(release_path)
    root = Path(plan.get("output_root", ""))
    expected = {
        result_path: root / "RELOAD_VALIDATED.json",
        controller_path: root / "RELOAD_CONTROLLER_TERMINAL.json",
        release_path: root / "RELOAD_RELEASE.json",
        output: root / "RELOAD_ACCEPTED.json",
    }
    if any(actual != wanted for actual, wanted in expected.items()):
        raise ValueError("Miles reload acceptance inputs are outside the bound output root")
    request = _validate_values(
        config=config,
        plan=plan,
        result=result,
        controller=controller,
        release=release,
        config_path=config_path,
        controller_path=controller_path,
        controller_file_sha256=controller_file_sha256,
        recompile=True,
    )
    # The exact reload runtime executes no rollout/verifier path, and its exact
    # request disables W&B.  The process receipt independently supplies the
    # remaining zero-work counters.  Seal all seven conclusions explicitly.
    zero_work = {key: result[key] for key in _WORK_FIELDS}
    accepted = {
        "schema": ACCEPTED_SCHEMA,
        "status": "accepted",
        "reload_config_path": str(config_path),
        "reload_config_file_sha256": config_file_sha256,
        "reload_config": config,
        "reload_config_sha256": digest(config),
        "reload_plan_path": str(plan_path),
        "reload_plan_file_sha256": plan_file_sha256,
        "reload_plan": plan,
        "reload_plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "reload_result_path": str(result_path),
        "reload_result_file_sha256": result_file_sha256,
        "reload_result": result,
        "controller_terminal_path": str(controller_path),
        "controller_terminal_file_sha256": controller_file_sha256,
        "controller_terminal": controller,
        "external_release_path": str(release_path),
        "external_release_file_sha256": release_file_sha256,
        "external_release": release,
        "source_manifest_sha256": plan["source_manifest"]["sha256"],
        "source_checkpoint_unchanged_after_release": True,
        "source_terminal_acceptance_sha256": plan["source_terminal_acceptance"][
            "receipt_sha256"
        ],
        "source_policy_delta_observation_sha256": plan[
            "source_policy_delta_observation"
        ]["receipt_sha256"],
        "rank_state_commitments_sha256": result["rank_state_commitments_sha256"],
        "exact_rank_state_commitments_verified": True,
        "work_executed": zero_work,
        "external_gpu_release_verified": True,
        "production_promotion_requires_this_receipt": True,
    }
    return _write(output, accepted)


def validate_accepted(value: dict, *, check_files: bool = True) -> dict:
    """Recompute an accepted receipt before a production-promotion consumer uses it."""
    from . import miles_policy_observer

    if value.get("schema") == miles_policy_observer.RELOAD_ACCEPTED_SCHEMA:
        return miles_policy_observer.validate_observer_reload_accepted(
            value,
            check_files=check_files,
        )
    if not check_files:
        raise ValueError("Miles reload promotion requires reopening every referenced file")
    sealed(value, ACCEPTED_SCHEMA)
    if set(value) != _ACCEPTED_FIELDS:
        raise ValueError("Miles accepted reload receipt fields changed")
    config = value.get("reload_config")
    plan = value.get("reload_plan")
    result = value.get("reload_result")
    controller = value.get("controller_terminal")
    release = value.get("external_release")
    work = value.get("work_executed")
    if not all(isinstance(item, dict) for item in (config, plan, result, controller, release)):
        raise ValueError("Miles accepted reload receipt has malformed embedded evidence")
    if (
        value.get("status") != "accepted"
        or value.get("reload_config_sha256") != digest(config)
        or value.get("reload_plan_sha256") != digest(plan)
        or value.get("source_manifest_sha256") != plan["source_manifest"]["sha256"]
        or value.get("source_terminal_acceptance_sha256")
        != plan["source_terminal_acceptance"]["receipt_sha256"]
        or value.get("source_policy_delta_observation_sha256")
        != plan["source_policy_delta_observation"]["receipt_sha256"]
        or value.get("rank_state_commitments_sha256")
        != digest(plan["expected_rank_state_commitments"])
        or value.get("exact_rank_state_commitments_verified") is not True
        or value.get("source_checkpoint_unchanged_after_release") is not True
        or not isinstance(work, dict)
        or set(work) != _WORK_FIELDS
        or any(not _integer(work.get(key), 0) for key in _WORK_FIELDS)
        or value.get("external_gpu_release_verified") is not True
        or value.get("production_promotion_requires_this_receipt") is not True
    ):
        raise ValueError("Miles accepted reload promotion binding is incomplete")
    config_path = Path(value["reload_config_path"])
    plan_path = Path(value["reload_plan_path"])
    result_path = Path(value["reload_result_path"])
    controller_path = Path(value["controller_terminal_path"])
    release_path = Path(value["external_release_path"])
    for path, expected, file_key in (
        (config_path, config, "reload_config_file_sha256"),
        (plan_path, plan, "reload_plan_file_sha256"),
        (result_path, result, "reload_result_file_sha256"),
        (controller_path, controller, "controller_terminal_file_sha256"),
        (release_path, release, "external_release_file_sha256"),
    ):
        observed, file_sha256 = _snapshot(path)
        if observed != expected or file_sha256 != value[file_key]:
            raise ValueError("Miles accepted reload input changed after acceptance")
    request = _validate_values(
        config=config,
        plan=plan,
        result=result,
        controller=controller,
        release=release,
        config_path=config_path,
        controller_path=controller_path,
        controller_file_sha256=value["controller_terminal_file_sha256"],
        recompile=True,
    )
    if value.get("request_sha256") != digest(request):
        raise ValueError("Miles accepted reload request binding changed")
    return {
        "reload_plan_sha256": value["reload_plan_sha256"],
        "request_sha256": value["request_sha256"],
        "reload_result_sha256": result["sha256"],
        "controller_terminal_sha256": controller["sha256"],
        "external_release_sha256": release["sha256"],
        "source_manifest_sha256": value["source_manifest_sha256"],
        "source_terminal_acceptance_sha256": value["source_terminal_acceptance_sha256"],
        "rank_state_commitments_sha256": value["rank_state_commitments_sha256"],
    }
