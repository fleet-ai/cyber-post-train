"""Create-once hosted GLM rank-1/attempt-1 exact-pass@4 canary.

The runner consumes the already accepted score-blind exact-100 inventory from
SFS, re-builds the frozen bulk authority, and executes only the first cell of a
whole-task reservation.  It deliberately writes the canonical bulk claim and
acceptance schemas so the global 800-cell ledger can reconcile the result.
"""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import endpoint_lease, self_hosted
from evals.fleet import exact_pass4_bulk_runtime_v3 as runtime
from evals.fleet import exact_pass4_bulk_v3 as bulk

MODEL = "glm-5.3"
CONTROLLER = "glm-b"
SELECTION_RANK = 1
ATTEMPT = 1
JOB_NAME = "chris-glm53-exact100-hosted-r001-a1-canary-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
INVENTORY_PATH = Path("/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v2/TERMINAL.json")
CLAIM_ROOT = Path(bulk.CLAIM_ROOT)
PARITY_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-glm53-hosted-actual-opencode-parity-v1.json"
)
PARITY_SHA256 = "sha256:9e0f00521061468f3ed77aed49ec52a586f5dc73702d26b605557dc86856b992"
SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": self_hosted.digest_without(value, "receipt_sha256")}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    runtime._write_once(path, value)  # noqa: SLF001 - canonical reviewed receipt writer


def _item(plan: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in plan["attempts"]
        if row["selection_rank"] == SELECTION_RANK and row["attempt"] == ATTEMPT
    ]
    if len(rows) != 1:
        raise RuntimeError("hosted GLM canary cell is not unique")
    return rows[0]


def _validate_parity(root: Path) -> None:
    receipt = bulk.load(root / PARITY_PATH)
    tool = receipt.get("tool_contract") or {}
    harness = receipt.get("harness") or {}
    model = receipt.get("model") or {}
    execution = receipt.get("execution") or {}
    if any(
        (
            receipt.get("receipt_sha256") != PARITY_SHA256,
            receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"),
            receipt.get("status") != "PASSED_NON_SCORED",
            receipt.get("classification") != "ACTUAL_HARNESS_PARITY",
            model.get("served_id") != MODEL,
            model.get("revision") != bulk.exact.EXPECTED_MODELS[MODEL]["revision"],
            harness.get("name") != "opencode",
            harness.get("version") != "1.18.27",
            harness.get("context_window_size") != 262144,
            harness.get("max_output_tokens") != 32768,
            harness.get("max_model_requests") != 600,
            harness.get("compaction_headroom_tokens") != 20000,
            tool.get("names") != ["bash", "submit_report"],
            tool.get("mcp_catalog_sha256")
            != "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a",
            tool.get("calls_observed_in_order") != ["bash", "submit_report"],
            tool.get("model_request_catalog_exact") is not True,
            tool.get("arguments_structurally_valid") is not True,
            execution.get("harness_exit_code") != 0,
            execution.get("task_instance_session_verifier_scoring_calls") != 0,
        )
    ):
        raise RuntimeError("hosted GLM actual-harness parity receipt drifted")


def build(root: Path, inventory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the full executable plan and its exact static claim authority."""
    _validate_parity(root)
    bulk.validate_inventory_gate(inventory, root)
    executable = bulk.build_runtime_plan(CONTROLLER, inventory, root)
    authority = bulk.validate_all(root)[CONTROLLER]
    item = _item(executable)
    authority_item = _item(authority)
    if any(
        (
            executable.get("treatment") != bulk.exact.EXPECTED_TREATMENT,
            executable.get("model", {}).get("served_id") != MODEL,
            executable.get("harness", {}).get("version") != "1.18.27",
            executable.get("execution", {}).get("required_task_tools") != ["bash", "submit_report"],
            executable.get("execution", {}).get("required_task_tool_catalog_sha256")
            != "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a",
            item["cell_id"] != authority_item["cell_id"],
            item["execution_id"] != authority_item["execution_id"],
            item["run_id"] != authority_item["run_id"],
            item["task_version_id"] != authority_item["task_version_id"],
        )
    ):
        raise RuntimeError("hosted GLM canary immutable plan drifted")
    return executable, authority


def _assert_fresh(config: dict[str, Any], item: dict[str, Any], key: str) -> None:
    runtime._assert_run_absent(config, key)  # noqa: SLF001 - exact reviewed duplicate gate
    persisted_model = self_hosted.persisted_session_model_identity(config)
    with runtime._client(key) as client:  # noqa: SLF001 - content-free session metadata
        account = self_hosted._request(client, "GET", "/v1/account")
        rows = self_hosted._task_sessions(client, config["task"]["key"])
    if account.get("team_id") != TEAM_ID or account.get("team_name") != "fleet":
        raise RuntimeError("Fleet-team credential authority drifted")
    collisions = []
    for row in rows:
        metadata = row.get("metadata") or {}
        if row.get("model") == persisted_model or any(
            metadata.get(field) == value
            for field, value in {
                "run_id": config["run_id"],
                "cell_id": item["cell_id"],
                "execution_id": item["execution_id"],
            }.items()
        ):
            collisions.append(row.get("session_id"))
    if collisions:
        raise RuntimeError("hosted GLM canary statistical cell is not globally fresh")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    if (
        os.environ.get("JOB_NAME") != JOB_NAME
        or os.environ.get("SECRET_UID") != SECRET_UID
        or not os.environ.get("FLEET_API_KEY")
    ):
        raise RuntimeError("hosted GLM canary runtime authority drifted")
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    for value in (job_uid, pod_uid):
        parsed = uuid.UUID(value)
        if parsed.int == 0:
            raise RuntimeError("hosted GLM canary requires nonzero runtime UIDs")
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise RuntimeError("hosted GLM canary output root already exists")

    inventory = bulk.load(INVENTORY_PATH)
    plan, authority = build(root, inventory)
    item = _item(plan)
    authority_item = _item(authority)
    task = runtime._task_for_item(plan, item)  # noqa: SLF001
    config = runtime._attempt_config(plan, task, item)  # noqa: SLF001
    key = os.environ["FLEET_API_KEY"]
    _assert_fresh(config, item, key)
    runtime._fresh_route_check(plan, key)  # noqa: SLF001
    claim_path = CLAIM_ROOT / runtime.claim_filename(item["execution_id"])
    if claim_path.exists() or claim_path.is_symlink():
        raise RuntimeError("hosted GLM canary execution is already claimed")

    OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in ("attempts", "accepted"):
        (OUTPUT_ROOT / name).mkdir(mode=0o700)
    _write_once(
        OUTPUT_ROOT / "PLAN.json",
        _seal(
            {
                "schema_version": "fleet-hosted-glm-exact-canary-plan-v1",
                "controller": CONTROLLER,
                "selection_rank": SELECTION_RANK,
                "attempt": ATTEMPT,
                "cell_id": item["cell_id"],
                "execution_id": item["execution_id"],
                "run_id": item["run_id"],
                "bulk_plan_sha256": authority["plan_sha256"],
                "runtime_plan_sha256": plan["plan_sha256"],
                "inventory_receipt_sha256": inventory["receipt_sha256"],
                "actual_harness_parity_receipt_sha256": PARITY_SHA256,
                "whole_task_reservation": {"selection_rank": 1, "attempts": [1, 2, 3, 4]},
                "created_at_utc": _now(),
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
        ),
    )
    lease = plan["execution"]["endpoint_lease"]
    with endpoint_lease.acquire_endpoint_lease(
        lease_root=Path(lease["lease_root"]),
        endpoint_key=lease["endpoint_key"],
        maximum_streams=lease["maximum_streams"],
    ):
        claim = runtime.claim_cell(
            authority,
            authority_item,
            claim_root=CLAIM_ROOT,
            job_uid=job_uid,
            pod_uid=pod_uid,
        )
        if claim is None:
            raise RuntimeError("hosted GLM canary lost the atomic claim race")
        _write_once(OUTPUT_ROOT / "CLAIM.json", claim)
        attempt_out = OUTPUT_ROOT / "attempts" / item["run_id"]
        try:
            self_hosted.run(config, attempt_out, proxy)
            accepted = runtime._classify_result(  # noqa: SLF001
                attempt_out, config, item, claim, key
            )
            _write_once(OUTPUT_ROOT / "accepted" / f"{item['run_id']}.json", accepted)
        except Exception as exc:
            _write_once(
                OUTPUT_ROOT / "NONREPEATABLE.json",
                _seal(
                    {
                        "schema_version": "fleet-hosted-glm-exact-canary-nonrepeatable-v1",
                        "controller": CONTROLLER,
                        "cell_id": item["cell_id"],
                        "execution_id": item["execution_id"],
                        "run_id": item["run_id"],
                        "claim_sha256": claim["receipt_sha256"],
                        "error_type": type(exc).__name__,
                        "retry_allowed": False,
                        "requires_score_blind_reconciliation": True,
                        "terminal_at_utc": _now(),
                        "scores_included": False,
                        "prompts_or_traces_included": False,
                    }
                ),
            )
            raise
    terminal = _seal(
        {
            "schema_version": "fleet-hosted-glm-exact-canary-terminal-v1",
            "status": "ACCEPTED",
            "controller": CONTROLLER,
            "cell_id": item["cell_id"],
            "execution_id": item["execution_id"],
            "run_id": item["run_id"],
            "claim_sha256": claim["receipt_sha256"],
            "accepted_receipt_sha256": accepted["receipt_sha256"],
            "session_id": accepted["session_id"],
            "verifier_execution_id": accepted["verifier_execution_id"],
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "terminal_at_utc": _now(),
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    )
    _write_once(OUTPUT_ROOT / "ACCEPTED.json", accepted)
    _write_once(OUTPUT_ROOT / "TERMINAL.json", terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    if args.command == "validate":
        _validate_parity(root)
        bulk.validate_all(root)
        return 0
    if args.proxy is None:
        parser.error("run requires --proxy")
    run(root, args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
