"""Exercise the dedicated controller plan rebuild and release gate without live work."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import glm53_dedicated_v14_scored_canary_runtime_v1 as runtime
from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as canary
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as inventory
from evals.fleet import self_hosted

OUTPUT = Path("/mnt/sfs/jobs/chris-glm53-dedicated-v17-r051-runtime-gate-v1/GATE.json")


def main() -> int:
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    uuid.UUID(job_uid)
    uuid.UUID(pod_uid)
    stages = []
    engine.validate_bulk_adapter(canary)
    stages.append("01-adapter-interface-valid")
    inventory_receipt = canary.load(inventory.INVENTORY_PATH)
    plan = canary.build_runtime_plan(canary.CONTROLLER, inventory_receipt, Path.cwd())
    stages.append("02-runtime-plan-built")
    rebuilt = canary.build_runtime_plan(
        canary.CONTROLLER, plan["inventory_receipt"], Path.cwd()
    )
    if plan != rebuilt:
        raise RuntimeError("runtime gate observer plan rebuild drifted")
    stages.append("03-runtime-plan-rebuilt-exactly")
    runtime._runtime_gate(plan)  # noqa: SLF001
    stages.append("04-release-gate-valid")
    body = {
        "schema_version": "fleet-glm53-dedicated-runtime-gate-observer-v1",
        "status": "PASSED_PRECLAIM",
        "checked_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "job_uid": job_uid,
        "pod_uid": pod_uid,
        "plan_sha256": plan["plan_sha256"],
        "stages": stages,
        "claim_calls": 0,
        "model_requests": 0,
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_read": False,
    }
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    OUTPUT.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    OUTPUT.write_bytes(self_hosted.canonical_json(body) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
