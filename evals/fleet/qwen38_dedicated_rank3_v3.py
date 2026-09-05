"""Held sequential rank-3 block for the exact dedicated Qwen TP1-B server."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dedicated_scored_canary_v1 as legacy
from evals.fleet import self_hosted

CONTROLLER = "qwen-dedicated-tp1-rank3-g21-v2"
SERVING_BLOCK = "dedicated-qwen-tp1-b-v2"
PARITY = Path(
    "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-tp1-b-v2-actual-opencode-parity.json"
)
PARITY_SHA256 = "sha256:06920a9cec20cff7b86312ffad6d2ce4a2a4ec3f9779f3a57c45b4f1b6fb70e7"
SERVICE_ORIGIN = "http://ft-run-76c7f016-pmzv6-head-svc.fleet-train-jobs.svc.cluster.local:8000"
SERVICE_UID = "c0dbe1ca-e10f-48b0-8221-f5a36b555b1d"
RAYJOB_UID = "5e153629-8166-4b79-a707-0e3fde8ab321"
WORKLOAD_UID = "87aef71c-b63f-4352-a1c0-9869232a4050"
TRAFFIC = Path("/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2/lifecycle/traffic")
EXECUTION_GENERATION = 21
TASK_VERSION_ID = "33d37078-0669-478e-af39-43cd245f0da8"
EXPECTED_IDENTITIES = {
    1: (
        "sha256:f728f5712478f233b05c0f2acfd1aba5ae9f63862e1b759ea40a508e188557d8",
        "sha256:acf0be00dcb0d87203e6805b506ba061434bcd1b2c56181b220fbfd6c31bfd2c",
    ),
    2: (
        "sha256:6edac950ae44ff62c07074a775afdb65cc9b394047883e0b57fcad84e74d2fa7",
        "sha256:49b30d65bac2afca7c21d6a4b763b8bea136d9f8e2314d0cd563aa50e7d82fab",
    ),
    3: (
        "sha256:ffb5b0fb6a3eb702b78c75aca66446eb39668d0949bc92b4628ad77c2102bcae",
        "sha256:677af5f417f6df4f3fd58ba34df0951e832822e4dc0587fdefdcec9af6c09215",
    ),
    4: (
        "sha256:47e1093f8b03d4179d334a4feac13aba526c5c36f8eb98bc29a681d27aa68e36",
        "sha256:0507ac14037c875483a4f7ac87bcf48cc1daa4d50d326afde8c8072496fbf289",
    ),
}
RUN_IDS = {
    attempt: f"chris-cyber-q38-opencode11827-ded-tp1-r003-a{attempt}-g21-v2"
    for attempt in EXPECTED_IDENTITIES
}


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


def build_plan(root: Path, attempt: int) -> dict[str, Any]:
    if attempt not in EXPECTED_IDENTITIES:
        raise ValueError("dedicated Qwen rank3 attempt must be 1 through 4")
    cell_id, execution_id = EXPECTED_IDENTITIES[attempt]
    with _binding():
        plan = legacy.build_plan(
            root,
            attempt,
            selection_rank=3,
            execution_generation=EXECUTION_GENERATION,
            run_id_override=RUN_IDS[attempt],
            expected_cell_id=cell_id,
            expected_execution_id=execution_id,
            expected_task_version_id=TASK_VERSION_ID,
        )
    plan["controller"] = CONTROLLER
    config = plan["config"]
    config["serving"] = {
        "kind": "dedicated_qwen_tp1_b_v2",
        "serving_block": SERVING_BLOCK,
        "service_origin": SERVICE_ORIGIN,
        "service_uid": SERVICE_UID,
        "rayjob_uid": RAYJOB_UID,
        "workload_uid": WORKLOAD_UID,
        "parity_receipt_sha256": PARITY_SHA256,
        "hosted_and_dedicated_results_must_remain_explicit_blocks": True,
    }
    config["config_sha256"] = self_hosted.digest_without(config, "config_sha256")
    plan["launch_authorized"] = False
    plan["required_release"] = "fleet-qwen38-dedicated-rank3-release-v1"
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def _released_plan(root: Path, attempt: int) -> dict[str, Any]:
    plan = build_plan(root, attempt)
    raw_path = os.environ.get("QWEN_RANK3_RELEASE_PATH")
    if not raw_path:
        raise RuntimeError("dedicated Qwen rank3 release receipt is required")
    release = json.loads(Path(raw_path).read_text())
    if (
        release.get("receipt_sha256") != self_hosted.digest_without(release, "receipt_sha256")
        or release.get("schema_version") != "fleet-qwen38-dedicated-rank3-release-v1"
        or release.get("status") != "RELEASED_TO_DEDICATED"
        or release.get("selection_rank") != 3
        or release.get("cell_ids") != [EXPECTED_IDENTITIES[index][0] for index in (1, 2, 3, 4)]
        or release.get("execution_ids") != [EXPECTED_IDENTITIES[index][1] for index in (1, 2, 3, 4)]
        or release.get("serving_block") != SERVING_BLOCK
        or release.get("laptop_hold_transferred") is not True
        or release.get("fresh_global_ledger_clear") is not True
        or release.get("fresh_authoritative_sessions_clear") is not True
        or release.get("claims_clear") is not True
        or release.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise RuntimeError("dedicated Qwen rank3 release receipt drifted")
    plan["launch_authorized"] = True
    plan["release_receipt_sha256"] = release["receipt_sha256"]
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def _require_previous_validated(attempt: int, jobs_root: Path = Path("/mnt/sfs/jobs")) -> None:
    if attempt == 1:
        return
    previous = json.loads(
        (jobs_root / RUN_IDS[attempt - 1] / "ACCEPTED_VALIDATED.json").read_text()
    )
    cell_id, execution_id = EXPECTED_IDENTITIES[attempt - 1]
    if (
        previous.get("receipt_sha256") != self_hosted.digest_without(previous, "receipt_sha256")
        or previous.get("status") != "ACCEPTED_VALIDATED"
        or previous.get("accepted") is not True
        or previous.get("credited") is not True
        or previous.get("retry_allowed") is not False
        or previous.get("all_artifact_byte_digests_matched") is not True
        or previous.get("fresh_authoritative_session_reconciled") is not True
        or previous.get("selection_rank") != 3
        or previous.get("attempt") != attempt - 1
        or previous.get("cell_id") != cell_id
        or previous.get("execution_id") != execution_id
        or previous.get("serving_block") != SERVING_BLOCK
    ):
        raise RuntimeError("previous dedicated Qwen rank3 cell is not fully validated")


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
        raise RuntimeError("dedicated Qwen TP1-B actual-harness parity drifted")
    return value


def run(root: Path, proxy: Path, attempt: int) -> dict[str, Any]:
    with _binding():
        _require_previous_validated(attempt)
        plan = _released_plan(root, attempt)
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
                    "cell_id": plan["item"]["cell_id"],
                    "execution_id": plan["item"]["execution_id"],
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
    attempt = int(os.environ["QWEN_DEDICATED_ATTEMPT"])
    run(root, root / "evals/fleet/fixed_proxy.py", attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
