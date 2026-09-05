from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_bulk_runtime_v3 as runtime
from evals.fleet import exact_pass4_ledger as ledger
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import qwen38_dedicated_scored_canary_v1 as qwen_dedicated
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / ledger.DEFAULT_CAMPAIGN


@pytest.fixture(scope="module")
def authority() -> ledger.Authority:
    return ledger._build_authority(ROOT, CAMPAIGN)


def _first_bulk(authority: ledger.Authority, model: str = "qwen3.8-27b") -> tuple[dict, dict, dict]:
    for plan, item in authority.bulk_items.values():
        cell = authority.cells[item["cell_id"]]
        if cell["model"] == model:
            return plan, item, cell
    raise AssertionError("bulk fixture missing")


def _accepted(authority: ledger.Authority, model: str = "qwen3.8-27b") -> dict:
    plan, item, cell = _first_bulk(authority, model)
    value = {
        "schema_version": "fleet-exact-pass4-bulk-cell-accepted-v3",
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        "controller": plan["controller"],
        "cell_id": cell["cell_id"],
        "execution_id": item["execution_id"],
        "run_id": item["run_id"],
        "selection_rank": cell["selection_rank"],
        "attempt": cell["attempt"],
        "task_key": cell["task_key"],
        "task_version_id": cell["task_version_id"],
        "session_id": str(uuid.uuid4()),
        "verifier_execution_id": str(uuid.uuid4()),
        "agent_exit_code": 1,
        "agent_process_exit_success": False,
        "config_sha256": "sha256:" + "c" * 64,
        "claim_sha256": "sha256:" + "a" * 64,
        "cleanup_completed": True,
        "session_ingest_completed": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def _claim(
    tmp_path: Path,
    authority: ledger.Authority,
    model: str = "qwen3.8-27b",
) -> tuple[Path, dict]:
    plan, item, _cell = _first_bulk(authority, model)
    root = tmp_path / "claims"
    value = runtime.claim_cell(
        plan,
        item,
        claim_root=root,
        job_uid="11111111-1111-4111-8111-111111111111",
        pod_uid="22222222-2222-4222-8222-222222222222",
    )
    assert value is not None
    path = root / f"{item['execution_id'].removeprefix('sha256:')}.json"
    return path, value


def _tombstone(cell: dict, generation: int = 1) -> dict:
    execution = exact.execution_for(cell["cell_id"], generation)
    value = {
        "schema_version": exact.TOMBSTONE_SCHEMA,
        "cell_id": cell["cell_id"],
        "execution_id": execution["execution_id"],
        "execution_generation": generation,
        "classification": "pre_model_infrastructure_failure",
        "terminal": True,
        "model_called": False,
        "verifier_called": False,
        "session_created": False,
        "authoritative_outcome_created": False,
        "retry_allowed": True,
        "evidence_receipt_sha256": "sha256:" + "b" * 64,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _dedicated_qwen_claim(authority: ledger.Authority) -> dict:
    plan, item = next(iter(authority.dedicated_qwen_items.values()))
    value = {
        "schema_version": qwen_dedicated.CLAIM_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "controller": plan["controller"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "execution_generation": item["execution_generation"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
        "job_uid": str(uuid.uuid4()),
        "pod_uid": str(uuid.uuid4()),
        "claimed_at_utc": "2026-09-05T12:00:00Z",
        "immutable": True,
        "automatic_retry": False,
        "model_call_started_when_claim_written": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _dedicated_qwen_accepted(authority: ledger.Authority) -> dict:
    plan, item = next(iter(authority.dedicated_qwen_items.values()))
    value = {
        "schema_version": qwen_dedicated.ACCEPTED_SCHEMA,
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        "serving_block": plan["config"]["serving"]["serving_block"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "run_id": item["run_id"],
        "selection_rank": item["selection_rank"],
        "attempt": item["attempt"],
        "task_version_id": item["task_version_id"],
        "session_id": str(uuid.uuid4()),
        "verifier_execution_id": str(uuid.uuid4()),
        "agent_exit_code": 0,
        "agent_process_exit_success": True,
        "claim_sha256": "sha256:" + "a" * 64,
        "config_sha256": plan["config"]["config_sha256"],
        "cleanup_completed": True,
        "session_ingest_completed": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _generation7_terminal(authority: ledger.Authority, model: str) -> dict:
    binding = next(item for item in authority.generation7.values() if item["model"] == model)
    value = {
        "schema_version": ledger.GENERATION7_TERMINAL_SCHEMA,
        "generation7_spec_sha256": binding["spec_sha256"],
        "plan_sha256": binding["plan_sha256"],
        "cell_id": binding["cell_id"],
        "execution_id": binding["execution_id"],
        "execution_generation": 7,
        "generation_claim_receipt_sha256": "sha256:" + "1" * 64,
        "job_uid": str(uuid.uuid4()),
        "pod_uid": str(uuid.uuid4()),
        "terminal_at_utc": "2026-09-05T12:00:00Z",
        "scoring_release": {},
        "root_authorization": {},
        "result": {
            "accepted": True,
            "quarantined": False,
            "claim_sha256": "sha256:" + "2" * 64,
            "attempt_config_sha256": "sha256:" + "3" * 64,
            "acceptance_receipt_sha256": "sha256:" + "4" * 64,
            "session_id": str(uuid.uuid4()),
            "verifier_execution_id": str(uuid.uuid4()),
            "session_ingest_completed": True,
            "cleanup_completed": True,
        },
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _generation15_claim(authority: ledger.Authority, model: str) -> dict:
    binding = next(item for item in authority.generation15.values() if item["model"] == model)
    value = {
        "schema_version": ledger.GENERATION15_CLAIM_SCHEMA,
        "spec_sha256": binding["spec_sha256"],
        "cell_id": binding["cell_id"],
        "execution_id": binding["execution_id"],
        "execution_generation": 15,
        "run_id": binding["run_id"],
        "job_uid": str(uuid.uuid4()),
        "pod_uid": str(uuid.uuid4()),
        "claimed_at_utc": "2026-09-05T12:00:00Z",
        "generation14_tombstone_receipt_sha256": "sha256:" + "5" * 64,
        "automatic_retry": False,
        "immutable": True,
        "prompts_or_traces_included": False,
        "scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_empty_ledger_has_the_exact_800_cell_denominator(
    authority: ledger.Authority,
) -> None:
    result = ledger.reconcile(
        authority,
        accepted=[],
        active_claims=[],
        blocked_claims=[],
        tombstones=[],
    )
    assert result["universe_cell_count"] == 800
    assert result["models"] == {
        "qwen3.8-27b": {
            "accepted": 0,
            "active": 0,
            "retryable_infra_failed": 0,
            "blocked_nonrepeatable": 0,
            "unstarted": 400,
        },
        "glm-5.3": {
            "accepted": 0,
            "active": 0,
            "retryable_infra_failed": 0,
            "blocked_nonrepeatable": 0,
            "unstarted": 400,
        },
    }
    assert result["privacy"] == {
        "prompts_read": False,
        "traces_read": False,
        "flags_read": False,
        "scores_read": False,
    }
    table = ledger.render_table(result)
    assert "qwen3.8-27b  400" in table
    assert "glm-5.3      400" in table
    assert "TOTAL        800" in table


def test_digest_valid_exact_bulk_acceptance_is_counted_score_blind(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    path = _write(tmp_path / "ACCEPTED.json", _accepted(authority))
    accepted = ledger.accepted_evidence(path, authority)
    result = ledger.reconcile(
        authority,
        accepted=[accepted],
        active_claims=[],
        blocked_claims=[],
        tombstones=[],
    )
    assert result["models"]["qwen3.8-27b"]["accepted"] == 1
    assert result["models"]["qwen3.8-27b"]["unstarted"] == 399


def test_nullable_api_projection_is_accepted_only_as_sealed_omission_metadata(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    value = _accepted(authority)
    value[ledger.BULK_ACCEPTED_OPTIONAL_PROJECTION_FIELD] = {
        "model": "omitted",
        "task_version_id": "matched",
        "run_id": "omitted",
        "execution_id": "omitted",
        "cell_id": "omitted",
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED.json", value)
    assert ledger.accepted_evidence(path, authority).state == "accepted"

    value[ledger.BULK_ACCEPTED_OPTIONAL_PROJECTION_FIELD]["model"] = "unchecked"
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    _write(path, value)
    with pytest.raises(ledger.LedgerError, match="projection evidence drifted"):
        ledger.accepted_evidence(path, authority)


def test_producer_path_can_map_to_read_only_observer_mount(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    observer_root = tmp_path / "shared" / "jobs"
    observed = _write(observer_root / "run" / "ACCEPTED.json", _accepted(authority))
    mappings = ledger._mount_maps([f"/mnt/sfs/jobs={observer_root}"])
    paths = ledger._paths([Path("/mnt/sfs/jobs/run/ACCEPTED.json")], [], "ACCEPTED.json", mappings)
    assert paths == [observed.absolute()]
    assert ledger.accepted_evidence(paths[0], authority).state == "accepted"


def test_active_and_nonrepeatable_claims_are_distinct_states(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    q_path, _ = _claim(tmp_path / "q", authority, "qwen3.8-27b")
    g_path, _ = _claim(tmp_path / "g", authority, "glm-5.3")
    active = ledger.claim_evidence(q_path, authority, active=True)
    blocked = ledger.claim_evidence(g_path, authority, active=False)
    result = ledger.reconcile(
        authority,
        accepted=[],
        active_claims=[active],
        blocked_claims=[blocked],
        tombstones=[],
    )
    assert result["models"]["qwen3.8-27b"]["active"] == 1
    assert result["models"]["glm-5.3"]["blocked_nonrepeatable"] == 1


def test_fixed_historical_adapters_remain_score_blind(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    accepted_path = _write(
        tmp_path / "generation7" / "ACCEPTED.json",
        _generation7_terminal(authority, "qwen3.8-27b"),
    )
    claim_path = _write(
        tmp_path / "generation15" / "claim.json",
        _generation15_claim(authority, "glm-5.3"),
    )
    result = ledger.reconcile(
        authority,
        accepted=[ledger.accepted_evidence(accepted_path, authority)],
        active_claims=[ledger.claim_evidence(claim_path, authority, active=True)],
        blocked_claims=[],
        tombstones=[],
    )
    assert result["models"]["qwen3.8-27b"]["accepted"] == 1
    assert result["models"]["glm-5.3"]["active"] == 1


def test_generation16_hosted_claim_uses_its_exact_successor_plan(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    plan, item = next(iter(authority.qwen_generation16_items.values()))
    root = tmp_path / "claims"
    value = runtime.claim_cell(
        plan,
        item,
        claim_root=root,
        job_uid="11111111-1111-4111-8111-111111111111",
        pod_uid="22222222-2222-4222-8222-222222222222",
    )
    assert value is not None
    path = root / f"{item['execution_id'].removeprefix('sha256:')}.json"
    evidence = ledger.claim_evidence(path, authority, active=True)
    assert evidence.cell_id == item["cell_id"]
    assert evidence.execution_generation == 16


def test_dedicated_qwen_claim_and_acceptance_are_strict_score_blind_adapters(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    claim = _dedicated_qwen_claim(authority)
    claim_path = _write(
        tmp_path / "claims" / f"{claim['execution_id'].removeprefix('sha256:')}.json",
        claim,
    )
    active = ledger.claim_evidence(claim_path, authority, active=True)
    active_result = ledger.reconcile(
        authority,
        accepted=[],
        active_claims=[active],
        blocked_claims=[],
        tombstones=[],
    )
    assert active_result["models"]["qwen3.8-27b"]["active"] == 1

    accepted_path = _write(
        tmp_path / "accepted" / "ACCEPTED.json",
        _dedicated_qwen_accepted(authority),
    )
    accepted = ledger.accepted_evidence(accepted_path, authority)
    accepted_result = ledger.reconcile(
        authority,
        accepted=[accepted],
        active_claims=[],
        blocked_claims=[],
        tombstones=[],
    )
    assert accepted_result["models"]["qwen3.8-27b"]["accepted"] == 1


def test_dedicated_qwen_adapter_rejects_serving_block_drift(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    value = _dedicated_qwen_accepted(authority)
    value["serving_block"] = "hosted"
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED.json", value)
    with pytest.raises(ledger.LedgerError, match="treatment drifted"):
        ledger.accepted_evidence(path, authority)


def test_historical_adapter_rejects_fixed_spec_drift(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    value = _generation7_terminal(authority, "qwen3.8-27b")
    value["generation7_spec_sha256"] = "sha256:" + "f" * 64
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED.json", value)
    with pytest.raises(ledger.LedgerError, match="treatment drifted"):
        ledger.accepted_evidence(path, authority)


def test_retry_safe_tombstone_is_counted_without_consuming_the_cell(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    _plan, _item, cell = _first_bulk(authority)
    path = _write(tmp_path / "tombstone.json", _tombstone(cell))
    tombstone = ledger.tombstone_evidence(path, authority)
    result = ledger.reconcile(
        authority,
        accepted=[],
        active_claims=[],
        blocked_claims=[],
        tombstones=[tombstone],
    )
    assert result["models"]["qwen3.8-27b"]["retryable_infra_failed"] == 1
    assert result["models"]["qwen3.8-27b"]["accepted"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("receipt_sha256", "sha256:" + "0" * 64), "digest"),
        (lambda value: value.__setitem__("task_version_id", str(uuid.uuid4())), "binding"),
    ],
)
def test_acceptance_digest_identity_and_treatment_drift_fail_closed(
    tmp_path: Path,
    authority: ledger.Authority,
    mutation,
    message: str,
) -> None:
    value = _accepted(authority)
    mutation(value)
    if message != "digest":
        value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED.json", value)
    with pytest.raises(ledger.LedgerError, match=message):
        ledger.accepted_evidence(path, authority)


def test_bulk_treatment_drift_fails_before_any_receipt_is_ingested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ledger.bulk.validate_all(ROOT)
    changed = copy.deepcopy(original)
    changed["qwen-a"]["treatment"]["context_window_size"] = 1
    monkeypatch.setattr(ledger.bulk, "validate_all", lambda _root: changed)
    with pytest.raises(ledger.LedgerError, match="treatment drifted"):
        ledger._build_authority(ROOT, CAMPAIGN)


def test_sensitive_content_field_is_rejected_before_ingestion(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    value = _accepted(authority)
    value["score"] = 1
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED.json", value)
    with pytest.raises(ledger.LedgerError, match="prohibited content"):
        ledger.accepted_evidence(path, authority)


def test_unsafe_tombstone_is_never_reclassified_as_retryable(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    _plan, _item, cell = _first_bulk(authority)
    value = _tombstone(cell)
    value["model_called"] = True
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "tombstone.json", value)
    with pytest.raises(ledger.LedgerError, match="not retry-safe"):
        ledger.tombstone_evidence(path, authority)


def test_duplicate_and_contradictory_cell_evidence_fail_closed(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    first = _write(tmp_path / "one.json", _accepted(authority))
    second_value = copy.deepcopy(json.loads(first.read_text()))
    second_value["session_id"] = str(uuid.uuid4())
    second_value["receipt_sha256"] = self_hosted.digest_without(second_value, "receipt_sha256")
    second = _write(tmp_path / "two.json", second_value)
    with pytest.raises(ledger.LedgerError, match="duplicate accepted"):
        ledger.reconcile(
            authority,
            accepted=[
                ledger.accepted_evidence(first, authority),
                ledger.accepted_evidence(second, authority),
            ],
            active_claims=[],
            blocked_claims=[],
            tombstones=[],
        )

    claim_path, _ = _claim(tmp_path / "claim", authority)
    with pytest.raises(ledger.LedgerError, match="also has"):
        ledger.reconcile(
            authority,
            accepted=[ledger.accepted_evidence(first, authority)],
            active_claims=[ledger.claim_evidence(claim_path, authority, active=True)],
            blocked_claims=[],
            tombstones=[],
        )


def test_tombstone_generations_must_be_contiguous(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    _plan, _item, cell = _first_bulk(authority)
    path = _write(tmp_path / "g2.json", _tombstone(cell, generation=2))
    with pytest.raises(ledger.LedgerError, match="contiguous"):
        ledger.reconcile(
            authority,
            accepted=[],
            active_claims=[],
            blocked_claims=[],
            tombstones=[ledger.tombstone_evidence(path, authority)],
        )


def test_duplicate_json_keys_and_symlinks_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"receipt_sha256":"x","receipt_sha256":"y"}')
    with pytest.raises(ledger.LedgerError, match="duplicate JSON key"):
        ledger.load_receipt(duplicate)

    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ledger.LedgerError, match="non-symlink"):
        ledger.load_receipt(link)


def test_cli_is_read_only_and_emits_json_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ledger.main(["--repo-root", str(ROOT), "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["universe_cell_count"] == 800
    assert len(value["cells"]) == 800
