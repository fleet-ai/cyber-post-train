"""Create-once scored canary for the exact dedicated Qwen DP8 server."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dedicated_scored_canary_v1 as legacy
from evals.fleet import self_hosted

CONTROLLER = "qwen-dedicated-dp8-a-rank4-a2-g22-v1"
SERVING_BLOCK = "dedicated-qwen-dp8-a-v1"
PARITY = Path(
    "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-dp8-v1-actual-opencode-parity.json"
)
PARITY_SHA256 = "sha256:22b33d598baab0a35f9d8340110e0f2783aba824894c3efda97298226d729966"
SERVICE_ORIGIN = (
    "http://ft-run-f8094254-ssnbz-head-svc.fleet-train-jobs.svc.cluster.local:8000"
)
SERVICE_UID = "b88f67d1-8369-4fa3-b8a3-d2a53c44399c"
RAYJOB_UID = "a3f06d5d-b6ff-4b06-ba7b-9f8f05972106"
WORKLOAD_UID = "9c8ad74d-7ecd-4102-98b1-b48d14386f0b"
HEAD_POD_UID = "0fe269e3-8796-4164-9fa6-310a36e88aae"
TRAFFIC = Path("/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-a-v1/lifecycle/traffic")
EXECUTION_GENERATION = 22
SELECTION_RANK = 4
ATTEMPT = 2
CELL_ID = "sha256:f5dced06656b0bf069930628978ce31b6ff682f437c86d55c6b241b2f83f4f48"
EXECUTION_ID = "sha256:18f44011c472d154c8be8eea7f4e18f00fcad794709401c6875047f27bcba0b1"
TASK_VERSION_ID = "02dd4e3f-d85d-4bf8-9976-eae2f102384d"
RUN_ID = "chris-cyber-q38-opencode11827-ded-dp8-r004-a2-g22-v1"
RELEASE_SCHEMA = "fleet-qwen38-dedicated-dp8-canary-release-v1"


@contextmanager
def _binding():
    replacements = {
        "CONTROLLER": CONTROLLER,
        "SERVICE_ORIGIN": SERVICE_ORIGIN,
        "SERVICE_UID": SERVICE_UID,
        "RAYJOB_UID": RAYJOB_UID,
        "WORKLOAD_UID": WORKLOAD_UID,
        "TRAFFIC": TRAFFIC,
    }
    previous = {name: getattr(legacy, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(legacy, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(legacy, name, value)


def build_plan(root: Path) -> dict[str, Any]:
    with _binding():
        plan = legacy.build_plan(
            root,
            ATTEMPT,
            selection_rank=SELECTION_RANK,
            execution_generation=EXECUTION_GENERATION,
            run_id_override=RUN_ID,
            expected_cell_id=CELL_ID,
            expected_execution_id=EXECUTION_ID,
            expected_task_version_id=TASK_VERSION_ID,
        )
    plan["controller"] = CONTROLLER
    config = plan["config"]
    config["serving"] = {
        "kind": "dedicated_qwen_dp8_a_v1",
        "serving_block": SERVING_BLOCK,
        "service_origin": SERVICE_ORIGIN,
        "service_uid": SERVICE_UID,
        "rayjob_uid": RAYJOB_UID,
        "workload_uid": WORKLOAD_UID,
        "head_pod_uid": HEAD_POD_UID,
        "parity_receipt_sha256": PARITY_SHA256,
        "hosted_and_dedicated_results_must_remain_explicit_blocks": True,
    }
    config["config_sha256"] = self_hosted.digest_without(config, "config_sha256")
    plan["launch_authorized"] = False
    plan["required_release"] = RELEASE_SCHEMA
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def _release(root: Path) -> dict[str, Any]:
    raw = os.environ.get("QWEN_DP8_RELEASE_PATH")
    if not raw:
        raise RuntimeError("dedicated Qwen DP8 release receipt is required")
    value = json.loads(Path(raw).read_text())
    if (
        value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256")
        or value.get("schema_version") != RELEASE_SCHEMA
        or value.get("status") != "RELEASED_ONE_CANARY"
        or value.get("cell_id") != CELL_ID
        or value.get("execution_id") != EXECUTION_ID
        or value.get("selection_rank") != SELECTION_RANK
        or value.get("attempt") != ATTEMPT
        or value.get("serving_block") != SERVING_BLOCK
        or value.get("parity_receipt_sha256") != PARITY_SHA256
        or value.get("rayjob_uid") != RAYJOB_UID
        or value.get("head_pod_uid") != HEAD_POD_UID
        or value.get("service_uid") != SERVICE_UID
        or value.get("fresh_global_ledger_clear") is not True
        or value.get("fresh_authoritative_sessions_clear") is not True
        or value.get("claim_clear") is not True
        or value.get("output_root_clear") is not True
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise RuntimeError("dedicated Qwen DP8 release receipt drifted")
    return value


def _validate_parity(root: Path) -> dict[str, Any]:
    value = json.loads((root / PARITY).read_text())
    binding = (value.get("endpoint") or {}).get("server_binding") or {}
    execution = value.get("execution") or {}
    tools = value.get("tool_contract") or {}
    if (
        value.get("receipt_sha256") != PARITY_SHA256
        or self_hosted.digest_without(value, "receipt_sha256") != PARITY_SHA256
        or value.get("status") != "PASSED_NON_SCORED"
        or binding.get("rayjob_uid") != RAYJOB_UID
        or binding.get("head_pod_uid") != HEAD_POD_UID
        or binding.get("service_uid") != SERVICE_UID
        or binding.get("model_revision") != legacy.MODEL_REVISION
        or binding.get("context_length") != 262144
        or execution.get("harness_exit_code") != 0
        or execution.get("task_instance_session_verifier_scoring_calls") != 0
        or tools.get("calls_observed_in_order") != ["bash", "submit_report"]
        or tools.get("model_request_catalog_exact") is not True
        or tools.get("arguments_structurally_valid") is not True
    ):
        raise RuntimeError("dedicated Qwen DP8 actual-harness parity drifted")
    return value


def released_plan(root: Path) -> dict[str, Any]:
    plan = build_plan(root)
    release = _release(root)
    plan["launch_authorized"] = True
    plan["release_receipt_sha256"] = release["receipt_sha256"]
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def run(root: Path, proxy: Path) -> dict[str, Any]:
    with _binding():
        plan = released_plan(root)
        output = Path(plan["output_root"])
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        _validate_parity(root)
        legacy._live_checks(plan, key)
        output.mkdir(mode=0o700, parents=False)
        self_hosted.write_json_once(output / "PLAN.json", plan)
        claim = legacy._claim(plan, os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", ""))
        stop = threading.Event()
        watcher = threading.Thread(target=legacy._traffic_loop, args=(stop,), daemon=True)
        watcher.start()
        try:
            attempt_root = output / "attempt"
            self_hosted.run(plan["config"], attempt_root, proxy)
            accepted = legacy._classify(attempt_root, plan, claim, key)
            accepted.pop("receipt_sha256", None)
            accepted["serving_block"] = SERVING_BLOCK
            accepted = legacy._seal(accepted)
            self_hosted.write_json_once(output / "ACCEPTED.json", accepted)
            terminal = legacy._seal(
                {
                    "schema_version": legacy.TERMINAL_SCHEMA,
                    "status": "ACCEPTED",
                    "accepted": True,
                    "serving_block": SERVING_BLOCK,
                    "cell_id": CELL_ID,
                    "execution_id": EXECUTION_ID,
                    "claim_sha256": claim["receipt_sha256"],
                    "accepted_receipt_sha256": accepted["receipt_sha256"],
                    "job_uid": os.environ.get("JOB_UID", ""),
                    "pod_uid": os.environ.get("POD_UID", ""),
                    "scores_included": False,
                    "prompts_or_traces_included": False,
                }
            )
            self_hosted.write_json_once(output / "TERMINAL.json", terminal)
            return terminal
        finally:
            stop.set()
            watcher.join(timeout=5)


def main() -> int:
    root = Path(os.environ.get("CYBER_ROOT", "/workspace/cyber-post-train"))
    run(root, root / "evals/fleet/fixed_proxy.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
