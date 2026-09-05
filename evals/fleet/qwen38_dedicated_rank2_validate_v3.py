"""Append-only full artifact-chain validation for dedicated Qwen rank2 cells."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dedicated_rank2_v3 as lane
from evals.fleet import qwen38_dedicated_scored_canary_v1 as legacy
from evals.fleet import self_hosted


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("artifact is not an object")
    return value


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _self_digest(value: dict[str, Any]) -> str:
    digest = value.get("receipt_sha256")
    if digest != self_hosted.digest_without(value, "receipt_sha256"):
        raise RuntimeError("artifact self-digest mismatch")
    return str(digest)


def validate(root: Path, key: str, attempt: int, job_uid: str, pod_uid: str) -> dict[str, Any]:
    uuid.UUID(job_uid)
    uuid.UUID(pod_uid)
    plan = lane.build_plan(root, attempt)
    output = Path(plan["output_root"])
    destination = output / "ACCEPTED_VALIDATED.json"
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("validated acceptance already exists")
    claim_path = legacy.CLAIM_ROOT / (
        plan["item"]["execution_id"].removeprefix("sha256:") + ".json"
    )
    paths = {
        "plan": output / "PLAN.json",
        "claim": claim_path,
        "result": output / "attempt/result.json",
        "reward": output / "attempt/reward-result.json",
        "session_ingest": output / "attempt/session-ingest.json",
        "cleanup": output / "attempt/cleanup.json",
        "accepted": output / "ACCEPTED.json",
        "terminal": output / "TERMINAL.json",
    }
    digests = {name: _file_sha(path) for name, path in paths.items()}
    stored_plan = _load(paths["plan"])
    claim, accepted, terminal = (_load(paths[name]) for name in ("claim", "accepted", "terminal"))
    if stored_plan != plan:
        raise RuntimeError("stored plan drifted")
    claim_digest = _self_digest(claim)
    accepted_digest = _self_digest(accepted)
    terminal_digest = _self_digest(terminal)
    item = plan["item"]
    if (
        any(claim.get(field) != item.get(field) for field in ("cell_id", "execution_id", "run_id"))
        or accepted.get("cell_id") != item["cell_id"]
        or accepted.get("execution_id") != item["execution_id"]
        or accepted.get("run_id") != item["run_id"]
        or accepted.get("selection_rank") != 2
        or accepted.get("attempt") != attempt
        or accepted.get("accepted") is not True
        or accepted.get("credited") is not True
        or accepted.get("retry_allowed") is not False
        or accepted.get("serving_block") != lane.SERVING_BLOCK
        or accepted.get("claim_sha256") != claim_digest
        or terminal.get("accepted") is not True
        or terminal.get("status") != "ACCEPTED"
        or terminal.get("accepted_receipt_sha256") != accepted_digest
        or terminal.get("claim_sha256") != claim_digest
    ):
        raise RuntimeError("acceptance identity chain drifted")
    with lane._binding():
        current = legacy._classify(output / "attempt", plan, claim, key)
    compared = (
        "cell_id",
        "execution_id",
        "run_id",
        "selection_rank",
        "attempt",
        "task_version_id",
        "session_id",
        "verifier_execution_id",
        "claim_sha256",
        "config_sha256",
        "authoritative_projection_omissions",
        "authoritative_projection_rule",
    )
    if any(current.get(field) != accepted.get(field) for field in compared):
        raise RuntimeError("fresh authoritative reconciliation drifted")
    validated = {
        "schema_version": "fleet-qwen38-dedicated-tp1-accepted-validated-v2",
        "status": "ACCEPTED_VALIDATED",
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        **{field: accepted[field] for field in compared},
        "serving_block": lane.SERVING_BLOCK,
        "serving_parity_receipt_sha256": lane.PARITY_SHA256,
        "plan_sha256": plan["plan_sha256"],
        "artifact_file_sha256": digests,
        "all_artifact_byte_digests_matched": True,
        "claim_receipt_sha256": claim_digest,
        "accepted_receipt_sha256": accepted_digest,
        "terminal_receipt_sha256": terminal_digest,
        "source_job_uid": terminal["job_uid"],
        "source_pod_uid": terminal["pod_uid"],
        "validator_job_uid": job_uid,
        "validator_pod_uid": pod_uid,
        "fresh_authoritative_session_reconciled": True,
        "fleet_api_mutations": 0,
        "scores_included": False,
        "prompts_or_traces_included": False,
        "credentials_included": False,
    }
    validated["receipt_sha256"] = self_hosted.digest_without(validated, "receipt_sha256")
    self_hosted.write_json_once(destination, validated)
    return validated


def main() -> int:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    attempt = int(os.environ.get("QWEN_DEDICATED_ATTEMPT", "2"))
    receipt = validate(
        Path(os.environ.get("CYBER_ROOT", "/workspace/cyber-post-train")),
        key,
        attempt,
        os.environ.get("JOB_UID", ""),
        os.environ.get("POD_UID", ""),
    )
    print(json.dumps({"status": receipt["status"], "receipt_sha256": receipt["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
