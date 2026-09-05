"""Fresh retry-safe rank3/a2 successor after a pre-plan packaging failure."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dedicated_rank3_v3 as prior
from evals.fleet import qwen38_dedicated_scored_canary_v1 as legacy
from evals.fleet import self_hosted

CONTROLLER = "qwen-dedicated-tp1-rank3-a2-g22-v1"
RUN_ID = "chris-cyber-q38-opencode11827-ded-tp1-r003-a2-g22-v1"
CELL_ID = "sha256:6edac950ae44ff62c07074a775afdb65cc9b394047883e0b57fcad84e74d2fa7"
EXECUTION_ID = "sha256:6a64d050625fdb15dde3a1a734ec88afd43ef1d07d7c6be8f1951addac62112e"


def build_plan(root: Path) -> dict[str, Any]:
    with prior._binding():
        plan = legacy.build_plan(
            root,
            2,
            selection_rank=3,
            execution_generation=22,
            run_id_override=RUN_ID,
            expected_cell_id=CELL_ID,
            expected_execution_id=EXECUTION_ID,
            expected_task_version_id=prior.TASK_VERSION_ID,
        )
    plan["controller"] = CONTROLLER
    plan["config"]["serving"] = {
        "kind": "dedicated_qwen_tp1_b_v2",
        "serving_block": prior.SERVING_BLOCK,
        "service_origin": prior.SERVICE_ORIGIN,
        "service_uid": prior.SERVICE_UID,
        "rayjob_uid": prior.RAYJOB_UID,
        "workload_uid": prior.WORKLOAD_UID,
        "parity_receipt_sha256": prior.PARITY_SHA256,
        "hosted_and_dedicated_results_must_remain_explicit_blocks": True,
    }
    plan["config"]["config_sha256"] = self_hosted.digest_without(plan["config"], "config_sha256")
    plan["launch_authorized"] = False
    plan["required_release"] = "fleet-qwen38-dedicated-rank3-a2-g22-release-v1"
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def _released_plan(root: Path, *, release_path: Path | None = None) -> dict[str, Any]:
    plan = build_plan(root)
    if release_path is None:
        raw_path = os.environ.get("QWEN_RANK3_G22_RELEASE_PATH")
        if not raw_path:
            raise RuntimeError("rank3/a2 g22 release receipt is required")
        release_path = Path(raw_path)
    value = json.loads(release_path.read_text())
    if (
        value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256")
        or value.get("schema_version") != "fleet-qwen38-dedicated-rank3-a2-g22-release-v1"
        or value.get("status") != "RELEASED_TO_DEDICATED"
        or value.get("selection_rank") != 3
        or value.get("attempt") != 2
        or value.get("cell_id") != CELL_ID
        or value.get("execution_id") != EXECUTION_ID
        or value.get("serving_block") != prior.SERVING_BLOCK
        or value.get("prior_attempt1_accepted_validated") is not True
        or value.get("fresh_global_ledger_clear") is not True
        or value.get("fresh_authoritative_sessions_clear") is not True
        or value.get("claim_clear") is not True
        or value.get("output_root_clear") is not True
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise RuntimeError("rank3/a2 g22 release receipt drifted")
    plan["launch_authorized"] = True
    plan["release_receipt_sha256"] = value["receipt_sha256"]
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def run(root: Path, proxy: Path) -> dict[str, Any]:
    prior._require_previous_validated(2)
    prior._validate_parity(root)
    plan = _released_plan(root)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    with prior._binding():
        legacy.CONTROLLER = CONTROLLER
        legacy._live_checks(plan, key)
        output = Path(plan["output_root"])
        output.mkdir(mode=0o700, parents=False)
        self_hosted.write_json_once(output / "PLAN.json", plan)
        claim = legacy._claim(plan, os.environ["JOB_UID"], os.environ["POD_UID"])
        stop = threading.Event()
        watcher = threading.Thread(target=legacy._traffic_loop, args=(stop,), daemon=True)
        watcher.start()
        try:
            attempt_root = output / "attempt"
            self_hosted.run(plan["config"], attempt_root, proxy)
            accepted = legacy._classify(attempt_root, plan, claim, key)
            accepted.pop("receipt_sha256", None)
            accepted["serving_block"] = prior.SERVING_BLOCK
            accepted = legacy._seal(accepted)
            self_hosted.write_json_once(output / "ACCEPTED.json", accepted)
            terminal = legacy._seal(
                {
                    "schema_version": legacy.TERMINAL_SCHEMA,
                    "status": "ACCEPTED",
                    "accepted": True,
                    "serving_block": prior.SERVING_BLOCK,
                    "cell_id": CELL_ID,
                    "execution_id": EXECUTION_ID,
                    "claim_sha256": claim["receipt_sha256"],
                    "accepted_receipt_sha256": accepted["receipt_sha256"],
                    "job_uid": os.environ["JOB_UID"],
                    "pod_uid": os.environ["POD_UID"],
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
