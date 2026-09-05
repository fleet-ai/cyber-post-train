import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dedicated_acceptance_chain_v1 as chain
from evals.fleet import self_hosted


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return chain._file_sha256(path)


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path, str, str]:
    mount = tmp_path / "mnt/sfs"
    monkeypatch.setenv("SFS_MOUNT_ROOT", str(mount))
    output = mount / "jobs/run"
    claim_path = mount / "cell-execution-claims/x/claim.json"
    item = {
        "cell_id": "sha256:" + "1" * 64,
        "execution_id": "sha256:" + "2" * 64,
        "run_id": "run",
        "selection_rank": 2,
        "attempt": 1,
    }
    plan = {"item": item, "config": {"task": {"key": "task", "version_id": "v"}}}
    claim = {k: item[k] for k in ("cell_id", "execution_id", "run_id")}
    claim["receipt_sha256"] = self_hosted.digest_without(claim, "receipt_sha256")
    result = {
        "session_id": "11111111-1111-4111-8111-111111111111",
        "verifier_execution_id": "22222222-2222-4222-8222-222222222222",
        "agent_exit_code": 0,
        "agent_termination": "completed",
        "session_ingest_status": "completed",
    }
    files = {
        "plan_file_sha256": (output / "PLAN.json", plan),
        "result_file_sha256": (output / "attempt/result.json", result),
        "reward_file_sha256": (output / "attempt/reward-result.json", {"ok": True}),
        "session_ingest_file_sha256": (output / "attempt/session-ingest.json", {"ok": True}),
        "cleanup_file_sha256": (output / "attempt/cleanup.json", {"ok": True}),
        "claim_file_sha256": (claim_path, claim),
    }
    evidence = {name: _write(path, value) for name, (path, value) in files.items()}
    evidence.update(
        {
            "output_root": str(output).replace(str(mount), "/shared"),
            "claim_path": str(claim_path).replace(str(mount), "/shared"),
            "claim_receipt_sha256": claim["receipt_sha256"],
        }
    )
    source = {
        "cell": item,
        "local_completion": {
            "session_id": result["session_id"],
            "verifier_execution_id": result["verifier_execution_id"],
            "agent_exit_code": 0,
            "agent_termination": "completed",
            "session_ingest_status": "completed",
        },
        "evidence": evidence,
    }
    source["receipt_sha256"] = self_hosted.digest_without(source, "receipt_sha256")
    source_path = tmp_path / "source.json"
    source_file_sha = _write(source_path, source)
    correction = {
        "source_file_sha256": source_file_sha,
        "source_claimed_receipt_sha256": source["receipt_sha256"],
        "source_actual_canonical_receipt_sha256": self_hosted.digest_without(
            source, "receipt_sha256"
        ),
        "source_evidence_authoritative_after_raw_file_binding": True,
        "accepted_or_credited_by_this_receipt": False,
    }
    correction["receipt_sha256"] = self_hosted.digest_without(correction, "receipt_sha256")
    correction_path = tmp_path / "correction.json"
    _write(correction_path, correction)
    return (
        source_path,
        correction_path,
        output,
        claim_path,
        source["receipt_sha256"],
        correction["receipt_sha256"],
    )


def test_source_chain_behaviorally_validates_every_named_file(tmp_path: Path, monkeypatch):
    source, correction, output, claim, source_digest, correction_digest = _fixture(
        tmp_path, monkeypatch
    )
    observed, fixed, plan, observed_claim, file_digests = chain.validate_source_chain(
        source,
        correction,
        output,
        claim,
        expected_source_digest=source_digest,
        expected_correction_digest=correction_digest,
    )
    assert observed["receipt_sha256"] == source_digest
    assert fixed["receipt_sha256"] == correction_digest
    assert plan["item"]["run_id"] == "run"
    assert observed_claim["run_id"] == "run"
    assert set(file_digests) == {
        "plan_file_sha256",
        "result_file_sha256",
        "reward_file_sha256",
        "session_ingest_file_sha256",
        "cleanup_file_sha256",
        "claim_file_sha256",
    }


@pytest.mark.parametrize(
    "relative",
    [
        "PLAN.json",
        "attempt/result.json",
        "attempt/reward-result.json",
        "attempt/session-ingest.json",
        "attempt/cleanup.json",
    ],
)
def test_source_chain_rejects_artifact_tampering(tmp_path: Path, monkeypatch, relative: str):
    source, correction, output, claim, source_digest, correction_digest = _fixture(
        tmp_path, monkeypatch
    )
    (output / relative).write_text("{}\n")
    with pytest.raises(RuntimeError, match="byte digest mismatch"):
        chain.validate_source_chain(
            source,
            correction,
            output,
            claim,
            expected_source_digest=source_digest,
            expected_correction_digest=correction_digest,
        )


def test_source_chain_rejects_claim_tampering(tmp_path: Path, monkeypatch):
    source, correction, output, claim, source_digest, correction_digest = _fixture(
        tmp_path, monkeypatch
    )
    claim.write_text("{}\n")
    with pytest.raises(RuntimeError, match="byte digest mismatch"):
        chain.validate_source_chain(
            source,
            correction,
            output,
            claim,
            expected_source_digest=source_digest,
            expected_correction_digest=correction_digest,
        )


def test_source_chain_rejects_source_receipt_tampering(tmp_path: Path, monkeypatch):
    source, correction, output, claim, source_digest, correction_digest = _fixture(
        tmp_path, monkeypatch
    )
    body = json.loads(source.read_text())
    body["cell"]["attempt"] = 2
    source.write_text(json.dumps(body))
    with pytest.raises(RuntimeError, match="digest correction does not bind"):
        chain.validate_source_chain(
            source,
            correction,
            output,
            claim,
            expected_source_digest=source_digest,
            expected_correction_digest=correction_digest,
        )


def test_validate_acceptance_writes_sufficient_create_once_envelope(tmp_path: Path, monkeypatch):
    source_path, correction_path, output, claim_path, source_digest, correction_digest = _fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(chain, "SOURCE_TERMINAL_SHA256", source_digest)
    monkeypatch.setattr(chain, "SOURCE_CORRECTION_SHA256", correction_digest)
    plan = json.loads((output / "PLAN.json").read_text())
    claim = json.loads(claim_path.read_text())
    item = plan["item"]
    current = {
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
        "task_version_id": "v",
        "serving_block": "dedicated-qwen-tp1-v1",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "verifier_execution_id": "22222222-2222-4222-8222-222222222222",
        "claim_sha256": claim["receipt_sha256"],
        "config_sha256": "sha256:" + "3" * 64,
        "authoritative_projection_omissions": ["metadata", "model", "task_version_id"],
        "authoritative_projection_rule": ("legacy_list_fields_may_be_null_but_never_mismatched_v1"),
    }
    accepted = {
        **current,
        "source_terminal_receipt_sha256": source_digest,
    }
    accepted["receipt_sha256"] = self_hosted.digest_without(accepted, "receipt_sha256")
    terminal = {
        "source_terminal_receipt_sha256": source_digest,
        "accepted_receipt_sha256": accepted["receipt_sha256"],
        "collector_job_uid": "33333333-3333-4333-8333-333333333333",
        "collector_pod_uid": "44444444-4444-4444-8444-444444444444",
    }
    terminal["receipt_sha256"] = self_hosted.digest_without(terminal, "receipt_sha256")
    _write(output / "ACCEPTED.json", accepted)
    _write(output / "TERMINAL.json", terminal)
    monkeypatch.setattr(chain.canary, "_classify", lambda *_args: current)
    receipt = chain.validate_acceptance(
        source_path,
        correction_path,
        output,
        claim_path,
        "key",
        validator_job_uid="55555555-5555-4555-8555-555555555555",
        validator_pod_uid="66666666-6666-4666-8666-666666666666",
    )
    assert receipt["accepted"] is True
    assert receipt["credited"] is True
    assert receipt["retry_allowed"] is False
    assert receipt["all_artifact_byte_digests_matched"] is True
    assert receipt["collector_job_uid"] == terminal["collector_job_uid"]
    assert receipt["validator_job_uid"] == "55555555-5555-4555-8555-555555555555"
    assert receipt["authoritative_projection_omissions"] == [
        "metadata",
        "model",
        "task_version_id",
    ]
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")
    assert json.loads((output / "ACCEPTED_VALIDATED.json").read_text()) == receipt
