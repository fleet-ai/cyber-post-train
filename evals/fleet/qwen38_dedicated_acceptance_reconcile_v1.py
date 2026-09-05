"""Read-only Fleet reconciliation for the completed dedicated Qwen canary."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from evals.fleet import qwen38_dedicated_scored_canary_v1 as canary
from evals.fleet import self_hosted

SOURCE_TERMINAL_SHA256 = "sha256:68fa8b63d14d26e8bb7f48f4bd103eafb5ac0165545c72e3d1765d7609aeb746"


def reconcile(root: Path, key: str, *, job_uid: str, pod_uid: str) -> dict:
    uuid.UUID(job_uid)
    uuid.UUID(pod_uid)
    plan = canary.build_plan(root)
    output = Path(plan["output_root"])
    accepted_path, terminal_path = output / "ACCEPTED.json", output / "TERMINAL.json"
    if accepted_path.exists() or terminal_path.exists():
        raise RuntimeError("dedicated Qwen acceptance already reconciled")
    stored_plan = json.loads((output / "PLAN.json").read_text())
    if stored_plan != plan:
        raise RuntimeError("stored dedicated Qwen plan drifted")
    claim_path = canary.CLAIM_ROOT / (
        plan["item"]["execution_id"].removeprefix("sha256:") + ".json"
    )
    claim = json.loads(claim_path.read_text())
    if (
        claim.get("receipt_sha256") != self_hosted.digest_without(claim, "receipt_sha256")
        or claim.get("cell_id") != plan["item"]["cell_id"]
        or claim.get("execution_id") != plan["item"]["execution_id"]
        or claim.get("run_id") != plan["item"]["run_id"]
    ):
        raise RuntimeError("dedicated Qwen execution claim drifted")
    accepted = canary._classify(output / "attempt", plan, claim, key)
    if accepted.get("authoritative_projection_rule") != (
        "legacy_list_fields_may_be_null_but_never_mismatched_v1"
    ):
        raise RuntimeError("legacy projection reconciliation rule is absent")
    accepted["reconciled_after_source_job_terminal"] = True
    accepted["source_terminal_receipt_sha256"] = SOURCE_TERMINAL_SHA256
    accepted["receipt_sha256"] = self_hosted.digest_without(accepted, "receipt_sha256")
    terminal = {
        "schema_version": "fleet-qwen38-dedicated-tp1-acceptance-reconcile-v1",
        "status": "ACCEPTED",
        "accepted": True,
        "cell_id": plan["item"]["cell_id"],
        "execution_id": plan["item"]["execution_id"],
        "session_id": accepted["session_id"],
        "verifier_execution_id": accepted["verifier_execution_id"],
        "accepted_receipt_sha256": accepted["receipt_sha256"],
        "source_terminal_receipt_sha256": SOURCE_TERMINAL_SHA256,
        "collector_job_uid": job_uid,
        "collector_pod_uid": pod_uid,
        "fleet_api_mutations": 0,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    terminal["receipt_sha256"] = self_hosted.digest_without(terminal, "receipt_sha256")
    self_hosted.write_json_once(accepted_path, accepted)
    self_hosted.write_json_once(terminal_path, terminal)
    return terminal


def main() -> int:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    terminal = reconcile(
        Path(os.environ.get("CYBER_ROOT", "/workspace/cyber-post-train")),
        key,
        job_uid=os.environ.get("JOB_UID", ""),
        pod_uid=os.environ.get("POD_UID", ""),
    )
    print(json.dumps({"status": terminal["status"], "receipt_sha256": terminal["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
