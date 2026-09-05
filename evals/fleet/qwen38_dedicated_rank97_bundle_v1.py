"""Reserve and run one complete rank-97 Qwen task on dedicated TP1-c.

All four generation-19 execution claims are created by one UID-bound Job before
the first model call.  This makes the immutable hosted G19 owner skip the whole
task without changing that live controller.  A partial pre-model reservation is
rolled back only after byte-validating every claim created by this process.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dedicated_scored_canary_v1 as legacy
from evals.fleet import self_hosted

CONTROLLER = "qwen-dedicated-tp1-rank97-whole-task-g19-v1"
SERVING_BLOCK = "dedicated-qwen-tp1-d-v1"
SELECTION_RANK = 97
EXECUTION_GENERATION = 19
TASK_VERSION_ID = "d4f5e664-d4fd-4492-a103-429260ad9e99"
SERVICE_ORIGIN = "http://ft-run-ef3b2fcb-hrt2w-head-svc.fleet-train-jobs.svc.cluster.local:8000"
SERVICE_UID = "cb53a63e-9af3-4652-9c67-d74bb14563af"
RAYJOB_UID = "0956e41f-c4ce-4809-b930-8b892f7d13d0"
WORKLOAD_UID = "e56ff660-27b8-42b9-9db7-19ff7815912f"
TRAFFIC = Path("/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-d-v1/lifecycle/traffic")
PARITY = Path(
    "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-tp1-d-v1-actual-opencode-parity.json"
)
PARITY_SHA256 = "sha256:2fdd5e2c1e86d4ad144a324a95426ad6358ab56852a1f48ee5bbe2a3bc2a6f1e"
BUNDLE_RUN_ID = "chris-cyber-q38-opencode11827-ded-tp1-r097-g19-c-bundle-v1"
BUNDLE_ROOT = Path(f"/mnt/sfs/jobs/{BUNDLE_RUN_ID}")
EXPECTED_IDENTITIES = {
    1: (
        "sha256:59df171f31662465c8a9a98844b254a3fcf6bb242e53347501248a8e499bb068",
        "sha256:c7ec5b2d485989c9e2e437714f4c6eeb94a2c2bb7f771b29a76da978e5a42ad1",
    ),
    2: (
        "sha256:160c68de576263202b614038ded2140e6bcf72435a2086fc7af6da6a3a19ba78",
        "sha256:dc6722e3954e719722491a2734a5b138f672b4a59dff4a6b6b38051b0a418c36",
    ),
    3: (
        "sha256:209f2350068810f475a61771b588a2ec4ed8aaa99b82197e037eaca57f929413",
        "sha256:d1e51b4814866658a73a791333798e8fddd05e0fa9b8fe9d20075e07aba4d86c",
    ),
    4: (
        "sha256:0652b1bc6305c8d1ff4cc4184390d69f67ee2a365551ba59408377dddf333869",
        "sha256:ed052dc0f396e72602a4f4d55e7c12036f57cd801c41b4d7079f046e7c62c4bf",
    ),
}
RUN_IDS = {
    attempt: f"chris-cyber-q38-opencode11827-ded-tp1-r097-a{attempt}-g19-c-v1"
    for attempt in EXPECTED_IDENTITIES
}


@contextmanager
def _binding():
    replacements = {
        "CONTROLLER": CONTROLLER,
        "SOURCE": Path("evals/fleet/configs/qwen-hosted-generation19-qwen-a-v4.json"),
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
        raise ValueError("rank97 attempt must be 1 through 4")
    cell_id, execution_id = EXPECTED_IDENTITIES[attempt]
    with _binding():
        plan = legacy.build_plan(
            root,
            attempt,
            selection_rank=SELECTION_RANK,
            execution_generation=EXECUTION_GENERATION,
            run_id_override=RUN_IDS[attempt],
            expected_cell_id=cell_id,
            expected_execution_id=execution_id,
            expected_task_version_id=TASK_VERSION_ID,
        )
    plan["controller"] = CONTROLLER
    plan["config"]["serving"] = {
        "kind": "dedicated_qwen_tp1_d_v1",
        "serving_block": SERVING_BLOCK,
        "service_origin": SERVICE_ORIGIN,
        "service_uid": SERVICE_UID,
        "rayjob_uid": RAYJOB_UID,
        "workload_uid": WORKLOAD_UID,
        "parity_receipt_sha256": PARITY_SHA256,
        "hosted_and_dedicated_results_must_remain_explicit_blocks": True,
    }
    plan["config"]["config_sha256"] = self_hosted.digest_without(plan["config"], "config_sha256")
    plan["launch_authorized"] = False
    plan["required_release"] = "fleet-qwen38-dedicated-rank97-whole-task-release-v1"
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


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
        raise RuntimeError("dedicated Qwen TP1-d actual-harness parity drifted")
    return value


def _released_plans(root: Path) -> list[dict[str, Any]]:
    plans = [build_plan(root, attempt) for attempt in (1, 2, 3, 4)]
    raw_path = os.environ.get("QWEN_RANK97_RELEASE_PATH")
    if not raw_path:
        raise RuntimeError("rank97 whole-task release receipt is required")
    release = json.loads(Path(raw_path).read_text())
    if (
        release.get("receipt_sha256") != self_hosted.digest_without(release, "receipt_sha256")
        or release.get("schema_version") != "fleet-qwen38-dedicated-rank97-whole-task-release-v1"
        or release.get("status") != "RELEASED_TO_DEDICATED"
        or release.get("selection_rank") != SELECTION_RANK
        or release.get("cell_ids") != [EXPECTED_IDENTITIES[index][0] for index in (1, 2, 3, 4)]
        or release.get("execution_ids") != [EXPECTED_IDENTITIES[index][1] for index in (1, 2, 3, 4)]
        or release.get("serving_block") != SERVING_BLOCK
        or release.get("prior_owner") != "qwen-a-generation19-v4"
        or release.get("single_job_four_claim_reservation_required") is not True
        or release.get("fresh_global_ledger_clear") is not True
        or release.get("fresh_authoritative_sessions_clear") is not True
        or release.get("claims_clear") is not True
        or release.get("output_roots_clear") is not True
        or release.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise RuntimeError("rank97 whole-task release receipt drifted")
    for plan in plans:
        plan["launch_authorized"] = True
        plan["release_receipt_sha256"] = release["receipt_sha256"]
        plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plans


def _claim_path(plan: dict[str, Any]) -> Path:
    execution_id = plan["item"]["execution_id"].removeprefix("sha256:")
    return legacy.CLAIM_ROOT / f"{execution_id}.json"


def _rollback_created_claims(
    created: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    # No output root or model call may exist at this compensation boundary.
    if any(Path(plan["output_root"]).exists() for plan, _claim in created):
        raise RuntimeError("partial reservation rollback crossed model/output boundary")
    for plan, claim in reversed(created):
        path = _claim_path(plan)
        if path.is_symlink() or json.loads(path.read_text()) != claim:
            raise RuntimeError("partial reservation claim bytes drifted")
        if claim.get("model_call_started_when_claim_written") is not False:
            raise RuntimeError("partial reservation claim crossed model boundary")
        path.unlink()


def reserve_all(
    plans: list[dict[str, Any]], job_uid: str, pod_uid: str
) -> dict[int, dict[str, Any]]:
    created: list[tuple[dict[str, Any], dict[str, Any]]] = []
    try:
        # Reverse attempt order minimizes the race with G19's forward traversal.
        for plan in reversed(plans):
            with _binding():
                claim = legacy._claim(plan, job_uid, pod_uid)
            created.append((plan, claim))
    except Exception:
        _rollback_created_claims(created)
        raise
    return {plan["item"]["attempt"]: claim for plan, claim in created}


def _run_one(
    root: Path, proxy: Path, plan: dict[str, Any], claim: dict[str, Any]
) -> dict[str, Any]:
    output = Path(plan["output_root"])
    output.mkdir(mode=0o700, parents=False)
    self_hosted.write_json_once(output / "PLAN.json", plan)
    stop = threading.Event()
    with _binding():
        watcher = threading.Thread(target=legacy._traffic_loop, args=(stop,), daemon=True)
        watcher.start()
        try:
            attempt_root = output / "attempt"
            self_hosted.run(plan["config"], attempt_root, proxy)
            accepted = legacy._classify(attempt_root, plan, claim, os.environ["FLEET_API_KEY"])
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
                    "job_uid": os.environ["JOB_UID"],
                    "pod_uid": os.environ["POD_UID"],
                    "scores_included": False,
                    "prompts_or_traces_included": False,
                }
            )
            self_hosted.write_json_once(output / "TERMINAL.json", terminal)
            return accepted
        finally:
            stop.set()
            watcher.join(timeout=5)


def run(root: Path, proxy: Path) -> None:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    _validate_parity(root)
    plans = _released_plans(root)
    if BUNDLE_ROOT.exists():
        raise RuntimeError("rank97 bundle output root already exists")
    for plan in plans:
        with _binding():
            legacy._live_checks(plan, key)
    claims = reserve_all(plans, os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", ""))
    BUNDLE_ROOT.mkdir(mode=0o700, parents=False)
    self_hosted.write_json_once(
        BUNDLE_ROOT / "RESERVATION.json",
        legacy._seal(
            {
                "schema_version": "fleet-qwen38-dedicated-rank97-four-claim-reservation-v1",
                "status": "RESERVED",
                "selection_rank": SELECTION_RANK,
                "job_uid": os.environ["JOB_UID"],
                "pod_uid": os.environ["POD_UID"],
                "plan_sha256s": [plan["plan_sha256"] for plan in plans],
                "claim_sha256s": [claims[index]["receipt_sha256"] for index in (1, 2, 3, 4)],
                "all_claims_before_model_call": True,
                "attempt_order": [1, 2, 3, 4],
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
        ),
    )
    for plan in plans:
        _run_one(root, proxy, plan, claims[plan["item"]["attempt"]])


def main() -> int:
    root = Path(os.environ.get("CYBER_ROOT", "/workspace/cyber-post-train"))
    run(root, root / "evals/fleet/fixed_proxy.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
