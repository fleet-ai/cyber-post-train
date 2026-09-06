"""Score-blind rank-30 release observer bound to the successful v6 diagnostic."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_single_slot_release_v3 as prior
from evals.fleet import hosted_glm_rank30_single_slot_v5 as successor
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-r030-single-slot-release-v4"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
engine = prior.engine
whole = prior.whole


def build(root: Path) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    source_sha = os.environ.get("GLM_HOSTED_R30_SOURCE_SHA256", "")
    if (
        not key
        or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid))
        or successor.SHA256_RE.fullmatch(source_sha) is None
    ):
        raise RuntimeError("rank-30 v4 observer bindings are invalid")
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    peer = prior._validate_live_peer()  # noqa: SLF001
    lease = prior.prior._probe_slots()  # noqa: SLF001
    collision = prior.prior._collisions(plan, key)  # noqa: SLF001
    zero_fields = (
        "canonical_claim_collisions",
        "authoritative_session_collisions",
        "accepted_evidence_collisions",
        "output_root_collisions",
        "new_job_collisions",
        "new_configmap_collisions",
        "api_mutations",
    )
    if any(collision[field] for field in zero_fields):
        raise RuntimeError("rank-30 v4 collision gate is not clear")
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR",
        "checked_at_utc": prior.prior._now(),  # noqa: SLF001
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_cap": 1,
        "controller": successor.release_projection(plan),
        "source_package_sha256": source_sha,
        "ledger_authority": prior.whole.LEDGER_AUTHORITY,
        "selection_authority": prior.whole.SELECTION_AUTHORITY,
        "live_rank29_peer": peer,
        "endpoint_lease_observer": lease,
        "fresh_collision_reconciliation": collision,
        "runtime_gate_diagnostic": successor.validate_runtime_gate_diagnostic(),
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    receipt = {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}
    successor.validate_release(receipt, plan, source_sha)
    return receipt


def main() -> int:
    receipt = build(Path.cwd())
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
