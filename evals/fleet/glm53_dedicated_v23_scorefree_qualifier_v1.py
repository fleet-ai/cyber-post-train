"""Fail-closed, score-free concurrency qualifier for a future GLM v23 server."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import endpoint_lease
from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v22_concurrency_qualification_v1 as engine
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as watchdog_runtime
from evals.fleet import opencode_actual_harness_parity_v1 as actual_harness

SCHEMA = "fleet-glm53-dedicated-v23-scorefree-qualification-held-v1"
AUTH_SCHEMA = "fleet-glm53-dedicated-v23-scorefree-authorization-v1"
RAW_SCHEMA = "fleet-glm53-dedicated-v23-scorefree-raw-v1"
VERDICT_SCHEMA = "fleet-glm53-dedicated-v23-scorefree-qualified-v1"
JOB_NAME = "chris-glm53-dedicated-v23-scorefree-qualification-v1"
RESULT_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
SERVER_TITLE = "chris-cyber-evalserve-glm53-tp8-a-v23"
SERVER_RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v23"
LEASE_ROOT = Path("/mnt/sfs/endpoint-leases/opencode11827-dedicated-v23-v1")
MODEL_REVISION = "30333038ada1f1dacb294a93270305a890b50c14"
SERVED_ID = "glm-5.3"
CONTEXT_LENGTH = 262144
IDLE_RELEASE_SECONDS = 600
CONCURRENCY = (1, 2, 4)
GPU_OBSERVER_SCHEMA = "fleet-glm53-dedicated-v23-scorefree-gpu-wave-v1"
GPU_OBSERVER_WAIT_SECONDS = 300
GPU_OBSERVER_MODULE = "evals.fleet.glm53_dedicated_v23_scorefree_gpu_observer_v1"


class QualificationError(RuntimeError):
    """The score-free v23 qualification gate failed closed."""


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise QualificationError("evidence_shape_invalid")
    return value


def build_held() -> dict[str, Any]:
    """Return the immutable no-launch contract before a v23 server exists."""

    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "READY_HELD",
        "server": {
            "title": SERVER_TITLE,
            "run_dir": SERVER_RUN_DIR,
            "nodes": 1,
            "gpus": 8,
            "served_id": SERVED_ID,
            "model_revision": MODEL_REVISION,
            "context_length": CONTEXT_LENGTH,
            "priority_class": "fleet-infra-quiet",
            "preemption_policy": "Never",
        },
        "score_free_boundary": {
            "concurrency_waves": list(CONCURRENCY),
            "fleet_task_instance_calls": 0,
            "fleet_session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "persist_response_or_tool_argument_content": False,
            "persist_prompts_traces_flags_scores": False,
        },
        "required_live_evidence": {
            "fresh_uid_bound_server_binding": True,
            "actual_opencode_parity": "PASSED_NON_SCORED",
            "request_counter_source": "sglang_num_requests_total",
            "request_counter_changes_only_on_model_requests": True,
            "idle_release_seconds": IDLE_RELEASE_SECONDS,
            "server_ready_and_zero_restarts": True,
            "workload_admitted_and_never_preempted": True,
            "no_active_scored_controller": True,
            "fresh_result_root_and_exclusive_endpoint_lease": True,
        },
        "removed_false_prerequisite": {
            "rank51_attempt2_accepted_required": False,
            "reason": "rank51_attempt2_is_sealed_nonrepeatable_uncredited",
        },
        "server_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def render(_root: Path) -> dict[str, Any]:
    """Compatibility entrypoint for the UID-bound GPU observer."""

    return build_held()


def _validate_binding(binding: dict[str, Any]) -> None:
    if (
        binding.get("server_title") != SERVER_TITLE
        or binding.get("server_run_dir") != SERVER_RUN_DIR
        or binding.get("served_id") != SERVED_ID
        or binding.get("model_revision") != MODEL_REVISION
        or binding.get("context_length") != CONTEXT_LENGTH
        or not all(
            isinstance(binding.get(key), str) and binding[key]
            for key in (
                "api_run_id",
                "rayjob_uid",
                "workload_uid",
                "head_pod_uid",
                "service_uid",
                "service_origin",
            )
        )
    ):
        raise QualificationError("v23_server_binding_invalid")


def canonical_model_binding(binding: dict[str, Any]) -> dict[str, Any]:
    """Project expanded v23 identity to the parity runner's exact seven keys."""

    _validate_binding(binding)
    return {
        key: binding[key]
        for key in (
            "api_run_id",
            "rayjob_uid",
            "head_pod_uid",
            "service_uid",
            "served_id",
            "model_revision",
            "context_length",
        )
    }


def authorize(
    binding: dict[str, Any],
    parity: dict[str, Any],
    watchdog: dict[str, Any],
    live: dict[str, Any],
) -> dict[str, Any]:
    """Authorize only the score-free qualifier from fresh UID-bound evidence."""

    _validate_binding(binding)
    treatment = actual_harness.exact.EXPECTED_TREATMENT
    expected_model = actual_harness.exact.EXPECTED_MODELS["glm-5.3"]
    harness = parity.get("harness") or {}
    tools = parity.get("tool_contract") or {}
    privacy = parity.get("privacy") or {}
    if (
        parity.get("receipt_sha256") != crypto.digest_without(parity, "receipt_sha256")
        or parity.get("schema_version") != actual_harness.SCHEMA
        or parity.get("status") != "PASSED_NON_SCORED"
        or parity.get("classification") != "ACTUAL_HARNESS_PARITY"
        or parity.get("endpoint", {}).get("kind") != "dedicated_uid_bound_inference"
        or parity.get("endpoint", {}).get("server_binding") != canonical_model_binding(binding)
        or parity.get("execution", {}).get("final_marker_observed") is not True
        or parity.get("execution", {}).get("harness_exit_code") != 0
        or not isinstance(parity.get("execution", {}).get("model_requests"), int)
        or parity.get("execution", {}).get("model_requests", 0) <= 0
        or parity.get("execution", {}).get("scored_launch_authorized") is not False
        or parity.get("execution", {}).get("task_instance_session_verifier_scoring_calls") != 0
        or harness.get("name") != treatment["harness"]
        or harness.get("version") != treatment["harness_version"]
        or harness.get("image") != actual_harness.IMAGE
        or harness.get("image_id") != actual_harness.IMAGE_ID
        or harness.get("release_asset_sha256") != treatment["release_asset_sha256"]
        or harness.get("provider_adapter") != treatment["provider_adapter"]
        or harness.get("context_management") != treatment["context_management"]
        or harness.get("context_window_size") != treatment["context_window_size"]
        or harness.get("compaction_headroom_tokens") != treatment["compaction_headroom_tokens"]
        or harness.get("max_output_tokens") != treatment["max_output_tokens"]
        or harness.get("max_model_requests") != treatment["max_model_requests"]
        or harness.get("timeout_seconds") != treatment["timeout_seconds"]
        or parity.get("model")
        != {
            "repository": expected_model["repository"],
            "revision": expected_model["revision"],
            "served_id": expected_model["served_id"],
            "session_model": expected_model["session_model"],
        }
        or tools.get("names") != treatment["tools"]
        or tools.get("calls_observed_in_order") != actual_harness.EXPECTED_CALL_ORDER
        or tools.get("mcp_catalog_sha256") != treatment["tool_catalog_sha256"]
        or tools.get("arguments_structurally_valid") is not True
        or tools.get("model_request_catalog_exact") is not True
        or tools.get("model_request_tool_names_exact") is not True
        or tools.get("model_request_tool_descriptions_exact") is not True
        or tools.get("model_request_tool_parameters_exact") is not True
        or not isinstance(tools.get("model_requests_with_tools"), int)
        or tools.get("model_requests_with_tools", 0) <= 0
        or any(
            privacy.get(field) is not False
            for field in (
                "benchmark_content_included",
                "credentials_included",
                "prompt_included",
                "responses_or_model_outputs_included",
                "stderr_or_stdout_included",
                "tool_arguments_included",
            )
        )
    ):
        raise QualificationError("v23_parity_invalid")
    if (
        watchdog.get("receipt_sha256") != crypto.digest_without(watchdog, "receipt_sha256")
        or watchdog.get("schema_version") != watchdog_runtime.SCHEMA
        or watchdog.get("status") != "ACTIVE_UID_BOUND"
        or watchdog.get("server_binding") != binding
        or watchdog.get("metric") != watchdog_runtime.METRIC
        or watchdog.get("idle_release_seconds") != IDLE_RELEASE_SECONDS
        or watchdog.get("health_or_process_liveness_refreshes") is not False
        or watchdog.get("model_request_counter_growth_refreshes") is not True
        or watchdog.get("release_via_jobs_api") is not True
        or watchdog.get("release_route") != watchdog_runtime.RELEASE_ROUTE
        or watchdog.get("implementation_id") != watchdog_runtime.IMPLEMENTATION_ID
        or watchdog.get("implementation_module")
        != "evals.fleet.glm53_dedicated_v23_request_counter_watchdog_v1"
        or watchdog.get("implementation_sha256") != watchdog_runtime.source_sha256()
        or not _valid_uuid(watchdog.get("watcher_job_uid"))
        or not _valid_uuid(watchdog.get("watcher_pod_uid"))
        or not isinstance(watchdog.get("initial_request_counter"), int)
        or watchdog.get("initial_request_counter", -1) < 0
        or not isinstance(watchdog.get("ready_at_epoch"), (int, float))
        or watchdog.get("ready_at_epoch", 0) <= 0
    ):
        raise QualificationError("v23_request_counter_watchdog_invalid")
    if (
        live.get("schema_version") != "fleet-glm53-dedicated-v23-scorefree-live-state-v1"
        or live.get("receipt_sha256") != crypto.digest_without(live, "receipt_sha256")
        or live.get("server_binding") != binding
        or live.get("rayjob_running") is not True
        or live.get("workload_admitted") is not True
        or live.get("workload_preemption_events") != 0
        or live.get("head_pod_ready") is not True
        or live.get("head_pod_restarts") != 0
        or live.get("active_scored_controller_count") != 0
        or live.get("qualification_result_root_absent") is not True
        or live.get("endpoint_lease_available") is not True
        or live.get("watcher_job_uid") != watchdog.get("watcher_job_uid")
        or live.get("watcher_pod_uid") != watchdog.get("watcher_pod_uid")
        or live.get("watcher_job_active") != 1
        or live.get("watcher_pod_ready") is not True
        or live.get("watcher_pod_restarts") != 0
        or live.get("seconds_since_last_model_request") not in range(IDLE_RELEASE_SECONDS)
    ):
        raise QualificationError("v23_live_scorefree_boundary_invalid")
    body: dict[str, Any] = {
        "schema_version": AUTH_SCHEMA,
        "status": "AUTHORIZED_SCORE_FREE_ONLY",
        "server_binding": binding,
        "parity_receipt_sha256": parity["receipt_sha256"],
        "watchdog_receipt_sha256": watchdog["receipt_sha256"],
        "live_receipt_sha256": live["receipt_sha256"],
        "idle_release_seconds": IDLE_RELEASE_SECONDS,
        "no_active_scored_controller": True,
        "qualification_result_root_absent": True,
        "endpoint_lease_exclusive": True,
        "qualification_launch_authorized": True,
        "scored_launch_authorized": False,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


Observer = Callable[[Path, int, dict[str, Any]], dict[str, Any]]


def validate_gpu_wave(
    observed: dict[str, Any],
    concurrency: int,
    server: dict[str, Any],
    binding: dict[str, Any],
    qualifier_identity: dict[str, str] | None = None,
) -> None:
    utilization = observed.get("max_utilization_percent_by_device")
    identity = observed.get("identity")
    if (
        observed.get("schema_version") != GPU_OBSERVER_SCHEMA
        or observed.get("status") != "OBSERVED_SCORE_FREE_WAVE"
        or observed.get("concurrency") != concurrency
        or observed.get("server") != server
        or observed.get("devices_seen") != 8
        or not isinstance(observed.get("samples_per_device"), int)
        or observed.get("samples_per_device", 0) < 1
        or not isinstance(utilization, list)
        or len(utilization) != 8
        or any(type(value) not in {int, float} or not 0 < value <= 100 for value in utilization)
        or not isinstance(identity, dict)
        or set(identity)
        != {
            "server_rayjob_uid",
            "server_head_pod_uid",
            "qualifier_job_uid",
            "qualifier_pod_uid",
        }
        or any(not _valid_uuid(value) for value in identity.values())
        or identity.get("server_rayjob_uid") != binding.get("rayjob_uid")
        or identity.get("server_head_pod_uid") != binding.get("head_pod_uid")
        or (
            qualifier_identity is not None
            and (
                identity.get("qualifier_job_uid") != qualifier_identity.get("job_uid")
                or identity.get("qualifier_pod_uid") != qualifier_identity.get("pod_uid")
            )
        )
        or observed.get("server_identity_unchanged") is not True
        or observed.get("qualifier_identity_unchanged") is not True
        or observed.get("receipt_sha256") != crypto.digest_without(observed, "receipt_sha256")
    ):
        raise QualificationError("v23_gpu_wave_invalid")


def _valid_uuid(value: object) -> bool:
    try:
        return uuid.UUID(str(value)).int != 0
    except ValueError:
        return False


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")


def validate_raw(raw: dict[str, Any]) -> None:
    authorization = raw.get("authorization")
    binding = raw.get("server_binding")
    if not isinstance(authorization, dict) or not isinstance(binding, dict):
        raise QualificationError("v23_raw_authority_invalid")
    _validate_binding(binding)
    if (
        raw.get("schema_version") != RAW_SCHEMA
        or raw.get("status") != "COMPLETED_SCORE_FREE_WAVES"
        or raw.get("receipt_sha256") != crypto.digest_without(raw, "receipt_sha256")
        or authorization.get("schema_version") != AUTH_SCHEMA
        or authorization.get("receipt_sha256")
        != crypto.digest_without(authorization, "receipt_sha256")
        or raw.get("authorization_receipt_sha256") != authorization.get("receipt_sha256")
        or authorization.get("server_binding") != binding
        or authorization.get("qualification_launch_authorized") is not True
        or authorization.get("scored_launch_authorized") is not False
        or authorization.get("status") != "AUTHORIZED_SCORE_FREE_ONLY"
        or authorization.get("idle_release_seconds") != IDLE_RELEASE_SECONDS
        or authorization.get("no_active_scored_controller") is not True
        or authorization.get("qualification_result_root_absent") is not True
        or authorization.get("endpoint_lease_exclusive") is not True
        or any(
            authorization.get(field) != 0
            for field in (
                "fleet_task_instance_calls",
                "fleet_session_calls",
                "verifier_calls",
                "scoring_calls",
            )
        )
        or not all(
            _valid_sha(authorization.get(field))
            for field in (
                "parity_receipt_sha256",
                "watchdog_receipt_sha256",
                "live_receipt_sha256",
            )
        )
        or any(
            raw.get(field) != 0
            for field in (
                "fleet_task_instance_calls",
                "fleet_session_calls",
                "verifier_calls",
                "scoring_calls",
            )
        )
        or raw.get("scored_launch_authorized") is not False
        or not isinstance(raw.get("waves"), list)
        or not isinstance(raw.get("gpu_waves"), list)
    ):
        raise QualificationError("v23_raw_authority_invalid")


def evaluate(
    waves: list[dict[str, Any]],
    gpu_waves: list[dict[str, Any]],
    binding: dict[str, Any],
) -> dict[str, Any]:
    if [row.get("concurrency") for row in waves] != list(CONCURRENCY) or len(gpu_waves) != 3:
        raise QualificationError("v23_wave_order_invalid")
    failures: list[str] = []
    for wave, observed in zip(waves, gpu_waves, strict=True):
        concurrency = int(wave["concurrency"])
        try:
            engine.validate_runtime_ramp(wave, baseline=waves[0] if concurrency != 1 else None)
            validate_gpu_wave(observed, concurrency, build_held()["server"], binding)
        except (engine.QualificationError, QualificationError):
            failures.append(f"c{concurrency}_protocol_latency_gpu_or_identity")
    baseline = float(waves[0]["throughput_streams_per_second"])
    if baseline <= 0 or float(waves[2]["throughput_streams_per_second"]) / baseline < 1.5:
        failures.append("c4_throughput_ratio")
    latencies = [float(row["stream_latency_seconds"]["p95"]) for row in waves]
    if statistics.median(latencies) <= 0:
        failures.append("latency_invalid")
    identities = [row["identity"] for row in gpu_waves]
    if any(row != identities[0] for row in identities[1:]):
        failures.append("uid_identity_changed_between_waves")
    return {
        "status": "PASSED_SCORE_FREE" if not failures else "FAILED",
        "failures": failures,
        "qualified_concurrency_ceiling": 4 if not failures else 0,
        "server_binding": binding,
        "qualifier_identity": identities[0],
        "scored_concurrency_change_authorized": False,
    }


def execute(
    authorization: dict[str, Any],
    out: Path,
    *,
    origin: str,
    runner: engine.HarnessRunner,
    counter: Callable[[str], int],
    observer: Observer,
    qualifier_identity: dict[str, str],
) -> dict[str, Any]:
    """Run c1/c2/c4 actual-OpenCode waves without Fleet API calls."""

    binding = authorization.get("server_binding")
    if (
        authorization.get("schema_version") != AUTH_SCHEMA
        or authorization.get("qualification_launch_authorized") is not True
        or authorization.get("scored_launch_authorized") is not False
        or authorization.get("receipt_sha256")
        != crypto.digest_without(authorization, "receipt_sha256")
        or not isinstance(binding, dict)
    ):
        raise QualificationError("v23_scorefree_authorization_invalid")
    _validate_binding(binding)
    if set(qualifier_identity) != {"job_uid", "pod_uid"} or any(
        not _valid_uuid(value) for value in qualifier_identity.values()
    ):
        raise QualificationError("v23_runtime_qualifier_identity_invalid")
    if out.exists() or out.is_symlink():
        raise QualificationError("v23_qualification_result_collision")
    waves: list[dict[str, Any]] = []
    gpu_waves: list[dict[str, Any]] = []
    with endpoint_lease.acquire_endpoint_lease(
        lease_root=LEASE_ROOT,
        endpoint_key=binding["api_run_id"],
        maximum_streams=1,
    ):
        for concurrency in CONCURRENCY:
            phase = {
                "schema_version": "fleet-glm53-dedicated-v23-scorefree-phase-v1",
                "status": "WAVE_STARTED_SCORE_FREE",
                "concurrency": concurrency,
                "server": build_held()["server"],
                "fleet_task_instance_calls": 0,
                "fleet_session_calls": 0,
                "verifier_calls": 0,
                "scoring_calls": 0,
            }
            engine.write_once(out.parent / f"PHASE-c{concurrency}.json", phase)
            wave = engine.run_wave(
                concurrency,
                origin=origin,
                binding=canonical_model_binding(binding),
                runner=runner,
                counter=counter,
            )
            engine.validate_runtime_ramp(wave, baseline=waves[0] if waves else None)
            observed = observer(out.parent, concurrency, build_held()["server"])
            validate_gpu_wave(
                observed,
                concurrency,
                build_held()["server"],
                binding,
                qualifier_identity,
            )
            waves.append(wave)
            gpu_waves.append(observed)
    body: dict[str, Any] = {
        "schema_version": RAW_SCHEMA,
        "status": "COMPLETED_SCORE_FREE_WAVES",
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "authorization": authorization,
        "server_binding": binding,
        "waves": waves,
        "gpu_waves": gpu_waves,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "scored_launch_authorized": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    out.parent.mkdir(parents=True, exist_ok=True)
    engine.write_once(out, body)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "validate"))
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--origin")
    parser.add_argument("--gpu-observer-root", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(build_held(), sort_keys=True))
    elif args.command == "run":
        if not all((args.authorization, args.origin, args.gpu_observer_root, args.out)):
            parser.error("run requires authorization, origin, gpu-observer-root, and out")
        execute(
            load(args.authorization),
            args.out,
            origin=args.origin,
            runner=engine.actual_harness.run,
            counter=engine.read_request_counter,
            observer=lambda root, concurrency, server: engine.await_gpu_observer(
                args.gpu_observer_root, concurrency, server
            ),
            qualifier_identity={
                "job_uid": os.environ.get("JOB_UID", ""),
                "pod_uid": os.environ.get("POD_UID", ""),
            },
        )
    else:
        if args.raw is None or args.out is None:
            parser.error("validate requires raw and out")
        raw = load(args.raw)
        validate_raw(raw)
        verdict = {
            "schema_version": VERDICT_SCHEMA,
            **evaluate(raw["waves"], raw["gpu_waves"], raw["server_binding"]),
            "raw_receipt_sha256": raw["receipt_sha256"],
            "fleet_task_instance_calls": 0,
            "fleet_session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "scored_launch_authorized": False,
        }
        engine.write_once(args.out, verdict)
        if verdict["status"] != "PASSED_SCORE_FREE":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
