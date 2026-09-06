"""Fail-closed, score-free concurrency qualifier for a future GLM v23 server."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import endpoint_lease
from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v22_concurrency_qualification_v1 as engine

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
            for key in ("api_run_id", "rayjob_uid", "workload_uid", "head_pod_uid", "service_uid")
        )
    ):
        raise QualificationError("v23_server_binding_invalid")


def authorize(
    binding: dict[str, Any],
    parity: dict[str, Any],
    watchdog: dict[str, Any],
    live: dict[str, Any],
) -> dict[str, Any]:
    """Authorize only the score-free qualifier from fresh UID-bound evidence."""

    _validate_binding(binding)
    if (
        parity.get("receipt_sha256") != crypto.digest_without(parity, "receipt_sha256")
        or parity.get("status") != "PASSED_NON_SCORED"
        or parity.get("endpoint", {}).get("server_binding") != binding
        or parity.get("execution", {}).get("task_instance_session_verifier_scoring_calls") != 0
    ):
        raise QualificationError("v23_parity_invalid")
    if (
        watchdog.get("receipt_sha256") != crypto.digest_without(watchdog, "receipt_sha256")
        or watchdog.get("status") != "ACTIVE_UID_BOUND"
        or watchdog.get("server_binding") != binding
        or watchdog.get("metric") != "sglang_num_requests_total"
        or watchdog.get("idle_release_seconds") != IDLE_RELEASE_SECONDS
        or watchdog.get("health_or_process_liveness_refreshes") is not False
        or watchdog.get("model_request_counter_growth_refreshes") is not True
        or watchdog.get("release_via_jobs_api") is not True
    ):
        raise QualificationError("v23_request_counter_watchdog_invalid")
    if (
        live.get("server_binding") != binding
        or live.get("rayjob_running") is not True
        or live.get("workload_admitted") is not True
        or live.get("workload_preemption_events") != 0
        or live.get("head_pod_ready") is not True
        or live.get("head_pod_restarts") != 0
        or live.get("active_scored_controller_count") != 0
        or live.get("qualification_result_root_absent") is not True
        or live.get("endpoint_lease_available") is not True
        or live.get("seconds_since_last_model_request") not in range(IDLE_RELEASE_SECONDS)
    ):
        raise QualificationError("v23_live_scorefree_boundary_invalid")
    body: dict[str, Any] = {
        "schema_version": AUTH_SCHEMA,
        "status": "AUTHORIZED_SCORE_FREE_ONLY",
        "server_binding": binding,
        "parity_receipt_sha256": parity["receipt_sha256"],
        "watchdog_receipt_sha256": watchdog["receipt_sha256"],
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


def execute(
    authorization: dict[str, Any],
    out: Path,
    *,
    origin: str,
    runner: engine.HarnessRunner,
    counter: Callable[[str], int],
    observer: Observer,
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
            wave = engine.run_wave(
                concurrency,
                origin=origin,
                binding=binding,
                runner=runner,
                counter=counter,
            )
            engine.validate_runtime_ramp(wave, baseline=waves[0] if waves else None)
            observed = observer(out.parent, concurrency, build_held()["server"])
            engine.validate_gpu_wave(observed, concurrency, build_held()["server"])
            waves.append(wave)
            gpu_waves.append(observed)
    body: dict[str, Any] = {
        "schema_version": RAW_SCHEMA,
        "status": "COMPLETED_SCORE_FREE_WAVES",
        "authorization_receipt_sha256": authorization["receipt_sha256"],
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
        )
    else:
        if args.raw is None or args.out is None:
            parser.error("validate requires raw and out")
        raw = load(args.raw)
        verdict = {
            "schema_version": VERDICT_SCHEMA,
            **engine.evaluate(
                raw["waves"], {"server": build_held()["server"], "waves": raw["gpu_waves"]}
            ),
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
