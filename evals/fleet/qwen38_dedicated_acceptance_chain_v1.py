"""Cryptographically chain the dedicated Qwen acceptance to source evidence."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dedicated_scored_canary_v1 as canary
from evals.fleet import self_hosted

SOURCE_TERMINAL_SHA256 = "sha256:68fa8b63d14d26e8bb7f48f4bd103eafb5ac0165545c72e3d1765d7609aeb746"
SOURCE_CORRECTION_SHA256 = "sha256:d756b797be1c493fcc7ed630e75787a0dad9b14c592f93e0ca6183ab5ae707f2"
OUTPUT_ROOT = Path("/mnt/sfs/jobs/chris-cyber-q38-opencode11827-ded-tp1-r002-a1-v2")
CLAIM_PATH = Path(
    "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1/"
    "d84a03547e45240e9702097b47d68ee122312987c5b18fcfa6ab7ee2ab738f0f.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path.name}")
    return value


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _require_self_digest(receipt: dict[str, Any], expected: str | None = None) -> str:
    actual = receipt.get("receipt_sha256")
    if actual != self_hosted.digest_without(receipt, "receipt_sha256"):
        raise RuntimeError("receipt self-digest mismatch")
    if expected is not None and actual != expected:
        raise RuntimeError("receipt does not match reviewed digest")
    return str(actual)


def validate_source_chain(
    source_receipt_path: Path,
    correction_receipt_path: Path,
    output: Path,
    claim_path: Path,
    *,
    expected_source_digest: str,
    expected_correction_digest: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Validate reviewed terminal evidence and every named immutable input."""
    source = _load(source_receipt_path)
    correction = _load(correction_receipt_path)
    _require_self_digest(correction, expected_correction_digest)
    if (
        _file_sha256(source_receipt_path) != correction.get("source_file_sha256")
        or source.get("receipt_sha256") != correction.get("source_claimed_receipt_sha256")
        or self_hosted.digest_without(source, "receipt_sha256")
        != correction.get("source_actual_canonical_receipt_sha256")
        or source.get("receipt_sha256") != expected_source_digest
        or correction.get("source_evidence_authoritative_after_raw_file_binding") is not True
        or correction.get("accepted_or_credited_by_this_receipt") is not False
    ):
        raise RuntimeError("source terminal digest correction does not bind reviewed evidence")
    evidence = source.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("source terminal evidence is absent")
    mount_root = Path(os.environ.get("SFS_MOUNT_ROOT", "/mnt/sfs"))
    expected_output = "/shared/" + str(output.relative_to(mount_root))
    expected_claim = "/shared/" + str(claim_path.relative_to(mount_root))
    if (
        evidence.get("output_root") != expected_output
        or evidence.get("claim_path") != expected_claim
    ):
        raise RuntimeError("source terminal paths do not bind mounted evidence")
    files = {
        "plan_file_sha256": output / "PLAN.json",
        "result_file_sha256": output / "attempt/result.json",
        "reward_file_sha256": output / "attempt/reward-result.json",
        "session_ingest_file_sha256": output / "attempt/session-ingest.json",
        "cleanup_file_sha256": output / "attempt/cleanup.json",
        "claim_file_sha256": claim_path,
    }
    observed: dict[str, str] = {}
    for field, path in files.items():
        observed[field] = _file_sha256(path)
        if observed[field] != evidence.get(field):
            raise RuntimeError(f"source evidence byte digest mismatch: {field}")
    plan, claim = _load(output / "PLAN.json"), _load(claim_path)
    if claim.get("receipt_sha256") != evidence.get("claim_receipt_sha256"):
        raise RuntimeError("claim receipt digest differs from source evidence")
    _require_self_digest(claim, str(evidence["claim_receipt_sha256"]))
    cell, local = source.get("cell"), source.get("local_completion")
    if not isinstance(cell, dict) or not isinstance(local, dict):
        raise RuntimeError("source identity observations are absent")
    item = plan.get("item")
    if not isinstance(item, dict) or any(
        item.get(field) != cell.get(field)
        for field in ("cell_id", "execution_id", "run_id", "selection_rank", "attempt")
    ):
        raise RuntimeError("source cell identity differs from stored plan")
    if any(claim.get(field) != item.get(field) for field in ("cell_id", "execution_id", "run_id")):
        raise RuntimeError("claim identity differs from stored plan")
    result = _load(output / "attempt/result.json")
    if any(
        result.get(field) != local.get(field)
        for field in (
            "session_id",
            "verifier_execution_id",
            "agent_exit_code",
            "agent_termination",
            "session_ingest_status",
        )
    ):
        raise RuntimeError("source local completion differs from stored result")
    return source, correction, plan, claim, observed


def validate_acceptance(
    source_receipt_path: Path,
    correction_receipt_path: Path,
    output: Path,
    claim_path: Path,
    key: str,
    *,
    validator_job_uid: str,
    validator_pod_uid: str,
) -> dict[str, Any]:
    uuid.UUID(validator_job_uid)
    uuid.UUID(validator_pod_uid)
    validated_path = output / "ACCEPTED_VALIDATED.json"
    if validated_path.exists():
        raise RuntimeError("dedicated Qwen acceptance chain already validated")
    source, correction, plan, claim, artifact_digests = validate_source_chain(
        source_receipt_path,
        correction_receipt_path,
        output,
        claim_path,
        expected_source_digest=SOURCE_TERMINAL_SHA256,
        expected_correction_digest=SOURCE_CORRECTION_SHA256,
    )
    accepted, terminal = _load(output / "ACCEPTED.json"), _load(output / "TERMINAL.json")
    accepted_digest = _require_self_digest(accepted)
    terminal_digest = _require_self_digest(terminal)
    if (
        accepted.get("source_terminal_receipt_sha256") != source["receipt_sha256"]
        or terminal.get("source_terminal_receipt_sha256") != source["receipt_sha256"]
        or terminal.get("accepted_receipt_sha256") != accepted_digest
    ):
        raise RuntimeError("provisional acceptance is not chained to source evidence")
    current = canary._classify(output / "attempt", plan, claim, key)
    compared = (
        "cell_id",
        "execution_id",
        "run_id",
        "selection_rank",
        "attempt",
        "task_version_id",
        "serving_block",
        "session_id",
        "verifier_execution_id",
        "claim_sha256",
        "config_sha256",
        "authoritative_projection_omissions",
        "authoritative_projection_rule",
    )
    for field in compared:
        if current.get(field) != accepted.get(field):
            raise RuntimeError(f"fresh authoritative reconciliation drifted: {field}")
    validated = {
        "schema_version": "fleet-qwen38-dedicated-tp1-accepted-validated-v1",
        "status": "ACCEPTED_VALIDATED",
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        **{field: accepted[field] for field in compared},
        "plan_file_sha256": artifact_digests["plan_file_sha256"],
        "claim_file_sha256": artifact_digests["claim_file_sha256"],
        "claim_receipt_sha256": claim["receipt_sha256"],
        "artifact_file_sha256": artifact_digests,
        "all_artifact_byte_digests_matched": True,
        "source_terminal_receipt_sha256": correction["receipt_sha256"],
        "source_terminal_stale_claim_sha256": source["receipt_sha256"],
        "source_terminal_actual_canonical_sha256": correction[
            "source_actual_canonical_receipt_sha256"
        ],
        "accepted_receipt_sha256": accepted_digest,
        "acceptance_terminal_receipt_sha256": terminal_digest,
        "collector_job_uid": terminal["collector_job_uid"],
        "collector_pod_uid": terminal["collector_pod_uid"],
        "validator_job_uid": validator_job_uid,
        "validator_pod_uid": validator_pod_uid,
        "fleet_api_mutations": 0,
        "fresh_authoritative_session_reconciled": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
        "credentials_included": False,
    }
    validated["receipt_sha256"] = self_hosted.digest_without(validated, "receipt_sha256")
    self_hosted.write_json_once(validated_path, validated)
    return validated


def main() -> int:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    receipt = validate_acceptance(
        Path(os.environ.get("SOURCE_TERMINAL_RECEIPT", "/bootstrap/source-terminal.json")),
        Path(os.environ.get("SOURCE_CORRECTION_RECEIPT", "/bootstrap/source-correction.json")),
        Path(os.environ.get("OUTPUT_ROOT", str(OUTPUT_ROOT))),
        Path(os.environ.get("CLAIM_PATH", str(CLAIM_PATH))),
        key,
        validator_job_uid=os.environ.get("JOB_UID", ""),
        validator_pod_uid=os.environ.get("POD_UID", ""),
    )
    print(json.dumps({"status": receipt["status"], "receipt_sha256": receipt["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
