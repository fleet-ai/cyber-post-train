import json
import uuid
from pathlib import Path

import pytest

from evals.fleet import qwen38_dedicated_rank2_validate_v3 as validator
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-r002-a2-accepted-validated-v2.json"
)
A3_RECEIPT = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-r002-a3-accepted-validated-v2.json"
)


def test_rank2_a2_validated_acceptance_is_self_digesting_and_sealed_safe() -> None:
    value = json.loads(RECEIPT.read_text())
    assert value["schema_version"] == ("fleet-qwen38-dedicated-tp1-accepted-validated-v2")
    assert value["status"] == "ACCEPTED_VALIDATED"
    assert value["accepted"] is True
    assert value["credited"] is True
    assert value["retry_allowed"] is False
    assert value["all_artifact_byte_digests_matched"] is True
    assert value["fresh_authoritative_session_reconciled"] is True
    assert value["fleet_api_mutations"] == 0
    assert value["prompts_or_traces_included"] is False
    assert value["scores_included"] is False
    assert value["credentials_included"] is False
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert set(value["artifact_file_sha256"]) == {
        "accepted",
        "claim",
        "cleanup",
        "plan",
        "result",
        "reward",
        "session_ingest",
        "terminal",
    }


def test_rank2_a3_validated_acceptance_is_self_digesting_and_sealed_safe() -> None:
    value = json.loads(A3_RECEIPT.read_text())
    assert value["schema_version"] == "fleet-qwen38-dedicated-tp1-accepted-validated-v2"
    assert value["status"] == "ACCEPTED_VALIDATED"
    assert value["selection_rank"] == 2
    assert value["attempt"] == 3
    assert value["accepted"] is True
    assert value["credited"] is True
    assert value["retry_allowed"] is False
    assert value["all_artifact_byte_digests_matched"] is True
    assert value["fresh_authoritative_session_reconciled"] is True
    assert value["fleet_api_mutations"] == 0
    assert value["prompts_or_traces_included"] is False
    assert value["scores_included"] is False
    assert value["credentials_included"] is False
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert set(value["artifact_file_sha256"]) == {
        "accepted",
        "claim",
        "cleanup",
        "plan",
        "result",
        "reward",
        "session_ingest",
        "terminal",
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    output = tmp_path / "output"
    attempt_root = output / "attempt"
    attempt_root.mkdir(parents=True)
    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    item = {
        "cell_id": "sha256:" + "1" * 64,
        "execution_id": "sha256:" + "2" * 64,
        "run_id": "run",
        "selection_rank": 2,
        "attempt": 2,
    }
    plan = {
        "item": item,
        "config": {"task": {"version_id": "task-version"}},
        "output_root": str(output),
        "plan_sha256": "sha256:" + "3" * 64,
    }
    claim = validator.legacy._seal(
        {**item, "job_uid": str(uuid.uuid4()), "pod_uid": str(uuid.uuid4())}
    )
    accepted = validator.legacy._seal(
        {
            **item,
            "accepted": True,
            "credited": True,
            "retry_allowed": False,
            "serving_block": validator.lane.SERVING_BLOCK,
            "task_version_id": "task-version",
            "session_id": str(uuid.uuid4()),
            "verifier_execution_id": str(uuid.uuid4()),
            "claim_sha256": claim["receipt_sha256"],
            "config_sha256": "sha256:" + "4" * 64,
            "authoritative_projection_omissions": ["metadata"],
            "authoritative_projection_rule": (
                "legacy_list_fields_may_be_null_but_never_mismatched_v1"
            ),
        }
    )
    terminal = validator.legacy._seal(
        {
            "accepted": True,
            "status": "ACCEPTED",
            "accepted_receipt_sha256": accepted["receipt_sha256"],
            "claim_sha256": claim["receipt_sha256"],
            "job_uid": claim["job_uid"],
            "pod_uid": claim["pod_uid"],
        }
    )
    self_hosted.write_json_once(output / "PLAN.json", plan)
    self_hosted.write_json_once(
        claim_root / (item["execution_id"].removeprefix("sha256:") + ".json"), claim
    )
    for name, value in {
        "result.json": {},
        "reward-result.json": {},
        "session-ingest.json": {},
        "cleanup.json": {},
    }.items():
        self_hosted.write_json_once(attempt_root / name, value)
    self_hosted.write_json_once(output / "ACCEPTED.json", accepted)
    self_hosted.write_json_once(output / "TERMINAL.json", terminal)
    monkeypatch.setattr(validator.lane, "build_plan", lambda _root, _attempt: plan)
    monkeypatch.setattr(validator.legacy, "CLAIM_ROOT", claim_root)
    monkeypatch.setattr(validator.legacy, "_classify", lambda *_args: accepted)
    return plan, output


def test_validator_behaviorally_chains_all_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan, output = _fixture(tmp_path, monkeypatch)
    value = validator.validate(ROOT, "unused", 2, str(uuid.uuid4()), str(uuid.uuid4()))
    assert value["status"] == "ACCEPTED_VALIDATED"
    assert value["all_artifact_byte_digests_matched"] is True
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert json.loads((output / "ACCEPTED_VALIDATED.json").read_text()) == value


def test_validator_rejects_tampered_self_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _plan, output = _fixture(tmp_path, monkeypatch)
    accepted_path = output / "ACCEPTED.json"
    accepted = json.loads(accepted_path.read_text())
    accepted["credited"] = False
    accepted_path.write_text(json.dumps(accepted))
    with pytest.raises(RuntimeError, match="self-digest mismatch"):
        validator.validate(ROOT, "unused", 2, str(uuid.uuid4()), str(uuid.uuid4()))
