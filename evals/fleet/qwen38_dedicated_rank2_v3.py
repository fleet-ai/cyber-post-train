"""Score-blind rank-2 continuation on the exact dedicated Qwen v3 server."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dedicated_scored_canary_v1 as legacy
from evals.fleet import self_hosted

CONTROLLER = "qwen-dedicated-tp1-rank2-v3"
SERVING_BLOCK = "dedicated-qwen-tp1-v3"
PARITY = Path(
    "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-v3-actual-opencode-parity.json"
)
PARITY_SHA256 = "sha256:2ecd7996bc71c504cfbaf8babf0cadcca83b9495729b6b1598e4fb142dae8cd7"
SERVICE_ORIGIN = "http://ft-run-59fbbb3e-cb8wx-head-svc.fleet-train-jobs.svc.cluster.local:8000"
SERVICE_UID = "bc8e0add-f8ec-4dbe-820d-8d78c4a8bc8d"
RAYJOB_UID = "ce6ee4d7-fd6d-4ea9-94f3-7c8fa18f58fe"
WORKLOAD_UID = "1532e82a-91f9-40f3-81a7-8247e79d4126"
TRAFFIC = Path("/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-a-v3/lifecycle/traffic")
RUN_ID = "chris-cyber-q38-opencode11827-ded-tp1-r002-a2-v3"
OUTPUT_ROOT = Path(f"/mnt/sfs/jobs/{RUN_ID}")


@contextmanager
def _binding() -> Iterator[None]:
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
        or binding.get("service_uid") != SERVICE_UID
        or binding.get("model_revision") != legacy.MODEL_REVISION
        or binding.get("context_length") != 262144
        or execution.get("harness_exit_code") != 0
        or execution.get("task_instance_session_verifier_scoring_calls") != 0
        or tools.get("calls_observed_in_order") != ["bash", "submit_report"]
        or tools.get("model_request_catalog_exact") is not True
        or tools.get("arguments_structurally_valid") is not True
    ):
        raise RuntimeError("dedicated Qwen v3 actual-harness parity drifted")
    return value


def build_plan(root: Path) -> dict[str, Any]:
    with _binding():
        plan = legacy.build_plan(root, 2)
    plan["controller"] = CONTROLLER
    plan["item"]["run_id"] = RUN_ID
    plan["output_root"] = str(OUTPUT_ROOT)
    config = plan["config"]
    config["run_id"] = RUN_ID
    config["execution"]["network"] = RUN_ID
    config["serving"] = {
        "kind": "dedicated_qwen_tp1_v3",
        "serving_block": SERVING_BLOCK,
        "service_origin": SERVICE_ORIGIN,
        "service_uid": SERVICE_UID,
        "rayjob_uid": RAYJOB_UID,
        "workload_uid": WORKLOAD_UID,
        "parity_receipt_sha256": PARITY_SHA256,
        "hosted_and_dedicated_results_must_remain_explicit_blocks": True,
    }
    config["config_sha256"] = self_hosted.digest_without(config, "config_sha256")
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def run(root: Path, proxy: Path) -> dict[str, Any]:
    with _binding():
        plan = build_plan(root)
        key = os.environ.get("FLEET_API_KEY")
        job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
        if not key:
            raise RuntimeError("FLEET_API_KEY is required")
        _validate_parity(root)
        legacy._live_checks(plan, key)
        OUTPUT_ROOT.mkdir(mode=0o700, parents=False)
        self_hosted.write_json_once(OUTPUT_ROOT / "PLAN.json", plan)
        claim = legacy._claim(plan, job_uid, pod_uid)
        stop = threading.Event()
        watcher = threading.Thread(target=legacy._traffic_loop, args=(stop,), daemon=True)
        watcher.start()
        attempt_root = OUTPUT_ROOT / "attempt"
        try:
            self_hosted.run(plan["config"], attempt_root, proxy)
            accepted = legacy._classify(attempt_root, plan, claim, key)
            accepted.pop("receipt_sha256", None)
            accepted["serving_block"] = SERVING_BLOCK
            accepted = legacy._seal(accepted)
            self_hosted.write_json_once(OUTPUT_ROOT / "ACCEPTED.json", accepted)
            terminal = legacy._seal(
                {
                    "schema_version": legacy.TERMINAL_SCHEMA,
                    "status": "ACCEPTED",
                    "accepted": True,
                    "serving_block": SERVING_BLOCK,
                    "cell_id": plan["item"]["cell_id"],
                    "execution_id": plan["item"]["execution_id"],
                    "claim_sha256": claim["receipt_sha256"],
                    "accepted_receipt_sha256": accepted["receipt_sha256"],
                    "job_uid": job_uid,
                    "pod_uid": pod_uid,
                    "scores_included": False,
                    "prompts_or_traces_included": False,
                }
            )
            self_hosted.write_json_once(OUTPUT_ROOT / "TERMINAL.json", terminal)
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
