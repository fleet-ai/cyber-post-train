from __future__ import annotations

import copy
import json
import uuid
from dataclasses import replace
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


def _first_g19_v4(authority: ledger.Authority) -> tuple[dict, dict, dict]:
    plan, item = next(iter(authority.qwen_generation19_v4_items.values()))
    return plan, item, authority.cells[item["cell_id"]]


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


def _evidence_manifest(entries: list[dict], *, schema: str | None = None) -> dict:
    value = {
        "schema_version": schema or ledger.EVIDENCE_MANIFEST_SCHEMA,
        "campaign_id": exact.EXPECTED_CAMPAIGN_ID,
        "campaign_path": str(CAMPAIGN.relative_to(ROOT)),
        "entries": entries,
        "privacy": {
            "prompts_read": False,
            "traces_read": False,
            "flags_read": False,
            "scores_read": False,
        },
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


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


def _g19_v4_claim(tmp_path: Path, authority: ledger.Authority) -> tuple[Path, dict]:
    plan, item, _cell = _first_g19_v4(authority)
    root = tmp_path / "claims"
    value = runtime.claim_cell(
        plan,
        item,
        claim_root=root,
        job_uid="11111111-1111-4111-8111-111111111111",
        pod_uid="22222222-2222-4222-8222-222222222222",
    )
    assert value is not None
    return root / f"{item['execution_id'].removeprefix('sha256:')}.json", value


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


def _dedicated_qwen_validated(authority: ledger.Authority) -> dict:
    binding = ledger.DEDICATED_QWEN_ATTEMPT1_BINDING
    value = {
        "schema_version": ledger.DEDICATED_QWEN_VALIDATED_SCHEMA,
        "status": "ACCEPTED_VALIDATED",
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        "serving_block": binding["serving_block"],
        "cell_id": binding["cell_id"],
        "execution_id": binding["execution_id"],
        "run_id": binding["run_id"],
        "selection_rank": binding["selection_rank"],
        "attempt": binding["attempt"],
        "task_version_id": binding["task_version_id"],
        "session_id": binding["session_id"],
        "verifier_execution_id": binding["verifier_execution_id"],
        "claim_sha256": binding["claim_sha256"],
        "config_sha256": binding["config_sha256"],
        "authoritative_projection_omissions": ["metadata", "model", "task_version_id"],
        "authoritative_projection_rule": ("legacy_list_fields_may_be_null_but_never_mismatched_v1"),
        "plan_file_sha256": binding["artifact_file_sha256"]["plan_file_sha256"],
        "claim_file_sha256": binding["artifact_file_sha256"]["claim_file_sha256"],
        "claim_receipt_sha256": binding["claim_sha256"],
        "artifact_file_sha256": copy.deepcopy(binding["artifact_file_sha256"]),
        "all_artifact_byte_digests_matched": True,
        "source_terminal_receipt_sha256": binding["source_terminal_receipt_sha256"],
        "source_terminal_stale_claim_sha256": binding["source_terminal_stale_claim_sha256"],
        "source_terminal_actual_canonical_sha256": binding[
            "source_terminal_actual_canonical_sha256"
        ],
        "accepted_receipt_sha256": binding["accepted_receipt_sha256"],
        "acceptance_terminal_receipt_sha256": binding["acceptance_terminal_receipt_sha256"],
        "collector_job_uid": binding["collector_job_uid"],
        "collector_pod_uid": binding["collector_pod_uid"],
        "validator_job_uid": binding["validator_job_uid"],
        "validator_pod_uid": binding["validator_pod_uid"],
        "fleet_api_mutations": 0,
        "fresh_authoritative_session_reconciled": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
        "credentials_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    assert value["receipt_sha256"] == binding["validated_receipt_sha256"]
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


def _rank99_rollforward_evidence(
    authority: ledger.Authority,
    *,
    state: str = "active",
    tmp_path: Path | None = None,
) -> tuple[ledger.Evidence, ledger.Evidence]:
    cell_id, binding = next(iter(authority.rollforward_precedence.items()))
    prior_execution_id, successor_execution_id, prior_receipt, successor_claim = binding
    blocked = ledger.Evidence(
        state="blocked_nonrepeatable",
        cell_id=cell_id,
        execution_id=prior_execution_id,
        execution_generation=19,
        receipt_sha256=prior_receipt,
        path=Path("/score-blind/prior-claim.json"),
    )
    current_path = Path("/score-blind/current-evidence.json")
    current_receipt = successor_claim
    if state == "accepted":
        assert tmp_path is not None
        accepted_receipt = {"claim_sha256": successor_claim}
        accepted_receipt["receipt_sha256"] = self_hosted.digest_without(
            accepted_receipt, "receipt_sha256"
        )
        current_path = _write(tmp_path / "accepted.json", accepted_receipt)
        current_receipt = accepted_receipt["receipt_sha256"]
    current = ledger.Evidence(
        state=state,
        cell_id=cell_id,
        execution_id=successor_execution_id,
        execution_generation=23,
        receipt_sha256=current_receipt,
        path=current_path,
    )
    return blocked, current


def test_exact_append_only_rollforward_selects_newer_active_generation(
    authority: ledger.Authority,
) -> None:
    blocked, active = _rank99_rollforward_evidence(authority)
    result = ledger.reconcile(
        authority,
        accepted=[],
        active_claims=[active],
        blocked_claims=[blocked],
        tombstones=[],
    )
    row = next(item for item in result["cells"] if item["cell_id"] == active.cell_id)
    assert row["state"] == "active"
    assert row["latest_execution_generation"] == 23
    assert result["models"]["qwen3.8-27b"]["blocked_nonrepeatable"] == 0


def test_exact_append_only_rollforward_allows_validated_successor_acceptance(
    tmp_path: Path, authority: ledger.Authority,
) -> None:
    blocked, accepted = _rank99_rollforward_evidence(
        authority, state="accepted", tmp_path=tmp_path
    )
    result = ledger.reconcile(
        authority,
        accepted=[accepted],
        active_claims=[],
        blocked_claims=[blocked],
        tombstones=[],
    )
    row = next(item for item in result["cells"] if item["cell_id"] == accepted.cell_id)
    assert row["state"] == "accepted"
    assert row["latest_execution_generation"] == 23

    receipt = json.loads(accepted.path.read_text())
    receipt["claim_sha256"] = "sha256:" + "0" * 64
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    changed_path = _write(tmp_path / "accepted-drift.json", receipt)
    changed = replace(
        accepted,
        receipt_sha256=receipt["receipt_sha256"],
        path=changed_path,
    )
    with pytest.raises(ledger.LedgerError, match="no exact append-only roll-forward"):
        ledger.reconcile(
            authority,
            accepted=[changed],
            active_claims=[],
            blocked_claims=[blocked],
            tombstones=[],
        )


def test_unmapped_or_receipt_drifted_rollforward_fails_closed(
    authority: ledger.Authority,
) -> None:
    blocked, active = _rank99_rollforward_evidence(authority)
    without_mapping = replace(authority, rollforward_precedence={})
    with pytest.raises(ledger.LedgerError, match="no exact append-only roll-forward"):
        ledger.reconcile(
            without_mapping,
            accepted=[],
            active_claims=[active],
            blocked_claims=[blocked],
            tombstones=[],
        )

    drifted = replace(
        authority,
        rollforward_precedence={
            **authority.rollforward_precedence,
            active.cell_id: (
                blocked.execution_id,
                active.execution_id,
                blocked.receipt_sha256,
                "sha256:" + "0" * 64,
            ),
        },
    )
    with pytest.raises(ledger.LedgerError, match="no exact append-only roll-forward"):
        ledger.reconcile(
            drifted,
            accepted=[],
            active_claims=[active],
            blocked_claims=[blocked],
            tombstones=[],
        )


def test_supplemental_rollforward_rejects_tampered_clearance_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = ROOT / ledger.SUPPLEMENTAL_RUNTIME_AUTHORITY_PATH
    value = json.loads(source.read_text())
    value["execution_rollforwards"][0]["preserver_clearance_receipt_sha256"] = (
        "sha256:" + "0" * 64
    )
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    changed = _write(tmp_path / "supplemental.json", value)
    monkeypatch.setattr(ledger, "SUPPLEMENTAL_RUNTIME_AUTHORITY_PATH", changed)
    with pytest.raises(ledger.LedgerError, match="preserver_clearance binding drifted"):
        ledger._build_authority(ROOT, CAMPAIGN)


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


def test_reviewed_legacy_glm_generation7_acceptance_is_exact(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    source = ledger.load_receipt(ROOT / ledger.LEGACY_GLM_GENERATION7_SOURCE)
    value = next(
        row["receipt"]
        for row in source["accepted_cells"]
        if row.get("model_block") == "glm_hosted_v12"
        and row.get("source_rank") == 13
        and row.get("attempt") == 1
    )
    path = _write(tmp_path / "legacy-glm-accepted.json", value)
    evidence = ledger.accepted_evidence(path, authority)
    assert evidence.state == "accepted"
    assert evidence.execution_generation == 7
    assert evidence.cell_id == ledger.GENERATION7_BINDINGS["glm-5.3"]["cell_id"]

    changed = copy.deepcopy(value)
    changed["run_id"] += "-drift"
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    with pytest.raises(ledger.LedgerError, match="identity or outcome drifted"):
        ledger.accepted_evidence(_write(tmp_path / "drift.json", changed), authority)


def test_durable_generation15_gate_is_exact_accepted_evidence(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    source = (
        ROOT / "docs/evidence/qwen38-study/2026-09-05-qwen38-generation15-accepted-gate-v1.json"
    )
    accepted = ledger.accepted_evidence(source, authority)
    assert accepted.state == "accepted"
    assert accepted.cell_id == ledger.GENERATION15_BINDINGS["qwen3.8-27b"]["cell_id"]

    changed = json.loads(source.read_text())
    changed["api_session"]["model_projection"] = "matched"
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    changed_path = _write(tmp_path / "g15-gate.json", changed)
    with pytest.raises(ledger.LedgerError, match="drifted"):
        ledger.accepted_evidence(changed_path, authority)


def test_evidence_manifest_runs_exact_score_blind_ledger_without_reconstructing_paths(
    tmp_path: Path, authority: ledger.Authority, capsys: pytest.CaptureFixture[str]
) -> None:
    claim_path, claim = _g19_v4_claim(tmp_path, authority)
    producer_claim = Path("/mnt/sfs/claims") / claim_path.name
    manifest = _write(
        tmp_path / "manifest.json",
        _evidence_manifest(
            [
                {
                    "kind": "accepted",
                    "path": (
                        "docs/evidence/qwen38-study/"
                        "2026-09-05-qwen38-generation15-accepted-gate-v1.json"
                    ),
                    "expected_receipt_sha256": (
                        "sha256:e0aef9a97d146fe5fcc686efafd4399bd7c2ee65a325e64fc089613507ab6745"
                    ),
                },
                {
                    "kind": "active_claim",
                    "path": str(producer_claim),
                    "expected_receipt_sha256": claim["receipt_sha256"],
                },
            ]
        ),
    )

    assert (
        ledger.main(
            [
                "--repo-root",
                str(ROOT),
                "--evidence-manifest",
                str(manifest),
                "--mount-map",
                f"/mnt/sfs={tmp_path}",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "qwen3.8-27b  400     1         1" in output


def test_evidence_manifest_rejects_receipt_digest_drift(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    claim_path, _claim_value = _g19_v4_claim(tmp_path, authority)
    manifest = _write(
        tmp_path / "manifest.json",
        _evidence_manifest(
            [
                {
                    "kind": "active_claim",
                    "path": str(Path("/mnt/sfs/claims") / claim_path.name),
                    "expected_receipt_sha256": "sha256:" + "0" * 64,
                }
            ]
        ),
    )
    mappings = ledger._mount_maps([f"/mnt/sfs={tmp_path}"])
    with pytest.raises(ledger.LedgerError, match="receipt digest drifted"):
        ledger._manifest_paths(
            manifest,
            repo_root=ROOT,
            campaign=CAMPAIGN,
            mappings=mappings,
        )


def test_v2_evidence_manifest_binds_operational_incident_without_counting_a_cell(
    tmp_path: Path, authority: ledger.Authority, capsys: pytest.CaptureFixture[str]
) -> None:
    incident = {
        "schema_version": "test-score-blind-operational-incident-v1",
        "status": "TERMINAL_PRECLAIM_INFRASTRUCTURE_FAILURE",
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    incident["receipt_sha256"] = self_hosted.digest_without(incident, "receipt_sha256")
    incident_path = _write(tmp_path / "incident.json", incident)
    manifest = _write(
        tmp_path / "manifest-v2.json",
        _evidence_manifest(
            [
                {
                    "kind": "accepted",
                    "path": (
                        "docs/evidence/qwen38-study/"
                        "2026-09-05-qwen38-generation15-accepted-gate-v1.json"
                    ),
                    "expected_receipt_sha256": (
                        "sha256:e0aef9a97d146fe5fcc686efafd4399bd7c2ee65a325e64fc089613507ab6745"
                    ),
                },
                {
                    "kind": "operational_incident",
                    "path": str(Path("/mnt/sfs/incidents") / incident_path.name),
                    "expected_receipt_sha256": incident["receipt_sha256"],
                },
            ],
            schema=ledger.EVIDENCE_MANIFEST_V2_SCHEMA,
        ),
    )

    assert (
        ledger.main(
            [
                "--repo-root",
                str(ROOT),
                "--evidence-manifest",
                str(manifest),
                "--mount-map",
                f"/mnt/sfs/incidents={tmp_path}",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "qwen3.8-27b  400     1" in output


def test_supplemental_runtime_authority_resolves_ambiguous_bulk_identity(
    authority: ledger.Authority,
) -> None:
    key = (
        "sha256:83d88543af87e4cb8983953165582ff724ca4db7f8dfb5d40b7a549c9ec8902b",
        "sha256:fcad0a06b3321a28675af9a2e3e826574ceffbfaa2e4ff2b3428923fbf87e2e6",
    )

    plan, item = ledger._bulk_pair(authority, key, "glm-hosted-s2") or ({}, {})

    assert plan["plan_sha256"] == (
        "sha256:c0cc69202751fbbea53dfeea98a62efe41b2638b4e9ad60307b0c0bdbeefd83f"
    )
    assert item["run_id"] == "chris-glm53-ac-bulk-a-r026-a2-g1-493bc3d8"
    assert item["task_key"] == authority.cells[key[0]]["task_key"]


def test_supplemental_runtime_authority_resolves_live_glm_rank27_attempt3(
    authority: ledger.Authority,
) -> None:
    key = (
        "sha256:600b2af93e99baa4c0442f0b0b48e36e5183ae42135a9549918c61b49ccd9f71",
        "sha256:a8227c016231192540bdba4d3058f284727c8a97a5d7e50ea378f8294bc434cb",
    )

    plan, item = ledger._bulk_pair(authority, key, "glm-hosted-s2") or ({}, {})

    assert plan == {
        "controller": "glm-hosted-s2",
        "plan_sha256": (
            "sha256:c0cc69202751fbbea53dfeea98a62efe41b2638b4e9ad60307b0c0bdbeefd83f"
        ),
    }
    assert item["run_id"] == "chris-glm53-ac-bulk-b-r027-a3-g1-36dd3619"
    assert item["selection_rank"] == 27
    assert item["attempt"] == 3
    assert item["task_key"] == authority.cells[key[0]]["task_key"]


def test_supplemental_runtime_authority_resolves_live_glm_rank27_attempt4(
    authority: ledger.Authority,
) -> None:
    key = (
        "sha256:22b36c9e6b2f43ae2fcce7b6d729cb0791375f38dd8da770968bb295f8c14bbc",
        "sha256:ca83617d5bec99592b0e175afe44d44fa8a611958c97efa21fe9c942205dff9f",
    )

    plan, item = ledger._bulk_pair(authority, key, "glm-hosted-s2") or ({}, {})

    assert plan == {
        "controller": "glm-hosted-s2",
        "plan_sha256": (
            "sha256:c0cc69202751fbbea53dfeea98a62efe41b2638b4e9ad60307b0c0bdbeefd83f"
        ),
    }
    assert item["run_id"] == "chris-glm53-ac-bulk-b-r027-a4-g1-36dd3619"
    assert item["selection_rank"] == 27
    assert item["attempt"] == 4
    assert item["task_key"] == authority.cells[key[0]]["task_key"]


@pytest.mark.parametrize(
    ("attempt", "cell_id", "execution_id"),
    [
        (
            1,
            "sha256:b036345f7f10180cfc4f9226cf495943e528394b3ca096634c89498010f80f34",
            "sha256:bc7c2d7d0d4330d1cfe5634872ef498092457250496d892de2cb7feb6ddde741",
        ),
        (
            2,
            "sha256:161518898b56ae52cfe4dbb15d9fc82499c59a1e48c4e625389c46449473ec0b",
            "sha256:dc9833e1f42f57bdc9fef690e07c83bb0a3207986c0d51d935b13f1de3577b2a",
        ),
        (
            3,
            "sha256:a39b169770345597cd53c3d8042254e3298313e3e161a4600388f2f9e26a5410",
            "sha256:3ac02e2bb743218ccc72f6d3b5b86a864831457398665e6cc36b5897336540bf",
        ),
        (
            4,
            "sha256:31cdbce338d8afb2d61c59cdd64d3edf91e420e393889bc0d7828206b12d75f0",
            "sha256:51419b57963a75e7c387b9bae607a2760e55cf56c5267e8f26b3112b2a0b0c11",
        ),
    ],
)
def test_supplemental_runtime_authority_resolves_live_glm_rank28(
    authority: ledger.Authority,
    attempt: int,
    cell_id: str,
    execution_id: str,
) -> None:
    plan, item = ledger._bulk_pair(
        authority, (cell_id, execution_id), "glm-hosted-s2"
    ) or ({}, {})

    assert plan == {
        "controller": "glm-hosted-s2",
        "plan_sha256": (
            "sha256:c0cc69202751fbbea53dfeea98a62efe41b2638b4e9ad60307b0c0bdbeefd83f"
        ),
    }
    assert item["run_id"] == (
        f"chris-glm53-ac-bulk-a-r028-a{attempt}-g1-c3fbe2bd"
    )
    assert item["selection_rank"] == 28
    assert item["attempt"] == attempt
    assert item["task_key"] == authority.cells[cell_id]["task_key"]


def test_supplemental_runtime_authority_resolves_live_glm_rank29_attempt1(
    authority: ledger.Authority,
) -> None:
    key = (
        "sha256:77dde32eb233a7bfae8594b2873caff0279d7e67600d51f2ce384dd486cb1361",
        "sha256:0312a3507bbf4f6763a7ea84815053c6d89c540632458a143e3e5e9a61eda9a4",
    )

    plan, item = ledger._bulk_pair(authority, key, "glm-hosted-s2") or ({}, {})

    assert plan == {
        "controller": "glm-hosted-s2",
        "plan_sha256": (
            "sha256:c0cc69202751fbbea53dfeea98a62efe41b2638b4e9ad60307b0c0bdbeefd83f"
        ),
    }
    assert item["run_id"] == "chris-glm53-ac-bulk-b-r029-a1-g1-b51782f9"
    assert item["selection_rank"] == 29
    assert item["attempt"] == 1
    assert item["task_key"] == authority.cells[key[0]]["task_key"]


def test_supplemental_runtime_authority_resolves_live_glm_rank3_attempt1(
    authority: ledger.Authority,
) -> None:
    key = (
        "sha256:7568e59b6949088e67f3a98a566a22927640771abfd331ce5094364eb7cbac04",
        "sha256:4c64408156e2d084a6bc5bf593be06dc82203bdc7236f1bf511f642e730f1cb0",
    )

    plan, item = ledger._bulk_pair(authority, key, "glm-hosted-r3-a1-canary") or (
        {},
        {},
    )

    assert plan == {
        "controller": "glm-hosted-r3-a1-canary",
        "plan_sha256": (
            "sha256:2cb3111dd8af5e5b165f864fec71d7697cf51f3c44e84d3ead5d0e5d76b29fc5"
        ),
    }
    assert item["run_id"] == "chris-glm53-ac-bulk-b-r003-a1-g1-33d37078"
    assert item["selection_rank"] == 3
    assert item["attempt"] == 1
    assert item["task_key"] == authority.cells[key[0]]["task_key"]


@pytest.mark.parametrize(
    "name",
    [
        "2026-09-06-qwen38-hosted-a-rank13-a1-deadline-terminal-reconciliation-v1.json",
        "2026-09-06-qwen38-hosted-b-rank14-a1-deadline-terminal-reconciliation-v1.json",
    ],
)
def test_hosted_qwen_deadline_tombstones_are_score_blind_and_nonrepeatable(
    name: str,
) -> None:
    path = ROOT / "docs/evidence/qwen38-study" / name
    receipt = ledger.load_receipt(path)

    assert receipt["schema_version"] == (
        "fleet-qwen38-hosted-deadline-terminal-reconciliation-v1"
    )
    assert receipt["classification"] == "POST_MODEL_NONREPEATABLE_UNCREDITED"
    assert receipt["status"] == "NO_AUTHORITATIVE_SESSION_OR_VERIFIER"
    assert receipt["retry_allowed"] is False
    assert receipt["fresh_authoritative_reconciliation"]["matching_session_count"] == 0
    assert receipt["fresh_authoritative_reconciliation"][
        "verifier_execution_presence"
    ] is False
    assert receipt["request_counts"]["api_mutations"] == 0
    assert receipt["privacy"] == {
        "credentials_included": False,
        "prompts_traces_flags_read": False,
        "scores_read": False,
    }
    assert receipt["model_stream"] == {
        **receipt["model_stream"],
        "content_read": False,
        "present": True,
    }


def test_evidence_manifest_cannot_be_mixed_with_individual_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _write(
        tmp_path / "manifest.json",
        _evidence_manifest(
            [
                {
                    "kind": "accepted",
                    "path": (
                        "docs/evidence/qwen38-study/"
                        "2026-09-05-qwen38-generation15-accepted-gate-v1.json"
                    ),
                    "expected_receipt_sha256": (
                        "sha256:e0aef9a97d146fe5fcc686efafd4399bd7c2ee65a325e64fc089613507ab6745"
                    ),
                }
            ]
        ),
    )
    assert (
        ledger.main(
            [
                "--repo-root",
                str(ROOT),
                "--evidence-manifest",
                str(manifest),
                "--accepted",
                str(
                    ROOT / "docs/evidence/qwen38-study/"
                    "2026-09-05-qwen38-generation15-accepted-gate-v1.json"
                ),
            ]
        )
        == 2
    )
    assert "cannot be combined" in capsys.readouterr().err


def test_current_qwen_hosted_claim_uses_its_exact_successor_plan(
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
    assert evidence.execution_generation == item["execution_generation"]


def test_generation18_qwen_claim_uses_its_exact_successor_plan(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    pairs = list(authority.qwen_generation18_items.values())
    assert len(pairs) == 1
    plan, item = pairs[0]
    assert plan["plan_sha256"] == (
        "sha256:6770e4a0f423154101f9d7c14277593f38ce8ff5539b5155fc1f1f4e58b9003b"
    )
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
    assert evidence.execution_generation == 18


def test_hosted_glm_bulk_claim_uses_exact_partition_authority(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    assert len(authority.hosted_glm_bulk_items) == 394
    plan, item = next(iter(authority.hosted_glm_bulk_items.values()))
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


@pytest.mark.parametrize(
    "authority_field",
    ["dedicated_qwen_rank3_items", "dedicated_qwen_rank97_items"],
)
def test_released_dedicated_successor_claims_use_exact_plan_authority(
    tmp_path: Path, authority: ledger.Authority, authority_field: str
) -> None:
    pairs = getattr(authority, authority_field)
    plan, item = max(pairs.values(), key=lambda pair: pair[1]["execution_generation"])
    root = tmp_path / authority_field
    value = runtime.claim_cell(
        plan,
        item,
        claim_root=root,
        job_uid="11111111-1111-4111-8111-111111111111",
        pod_uid="22222222-2222-4222-8222-222222222222",
    )
    assert value is not None
    path = root / f"{item['execution_id'].removeprefix('sha256:')}.json"
    evidence = ledger.claim_evidence(path, authority, active=False)
    assert evidence.cell_id == item["cell_id"]
    assert evidence.state == "blocked_nonrepeatable"


def test_dedicated_qwen_v3_claim_uses_exact_rank2_attempt2_authority(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    plan, item = next(iter(authority.dedicated_qwen_v3_items.values()))
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
    assert evidence.execution_generation == 1


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


def test_dedicated_attempt1_keeps_its_immutable_historical_plan_authority(
    authority: ledger.Authority,
) -> None:
    key = (
        ledger.DEDICATED_QWEN_ATTEMPT1_BINDING["cell_id"],
        ledger.DEDICATED_QWEN_ATTEMPT1_BINDING["execution_id"],
    )
    plan, item = authority.dedicated_qwen_items[key]
    assert plan["plan_sha256"] == ledger.DEDICATED_QWEN_ATTEMPT1_BINDING["plan_sha256"]
    assert (
        plan["config"]["config_sha256"] == ledger.DEDICATED_QWEN_ATTEMPT1_BINDING["config_sha256"]
    )
    assert item == qwen_dedicated.build_plan(ROOT, 1)["item"]
    assert plan["plan_sha256"] != qwen_dedicated.build_plan(ROOT, 1)["plan_sha256"]


def test_dedicated_qwen_adapter_rejects_serving_block_drift(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    value = _dedicated_qwen_accepted(authority)
    value["serving_block"] = "hosted"
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED.json", value)
    with pytest.raises(ledger.LedgerError, match="treatment drifted"):
        ledger.accepted_evidence(path, authority)


def test_validated_dedicated_qwen_chain_is_the_only_post_terminal_credit(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    value = _dedicated_qwen_validated(authority)
    path = _write(tmp_path / "ACCEPTED_VALIDATED.json", value)
    evidence = ledger.accepted_evidence(path, authority)
    assert evidence.state == "accepted"
    assert evidence.cell_id == ledger.DEDICATED_QWEN_ATTEMPT1_BINDING["cell_id"]

    provisional = _dedicated_qwen_accepted(authority)
    provisional.update(
        {
            "authoritative_session_task_key_matched": True,
            "authoritative_projection_omissions": ["metadata", "model", "task_version_id"],
            "authoritative_projection_rule": (
                "legacy_list_fields_may_be_null_but_never_mismatched_v1"
            ),
            "reconciled_after_source_job_terminal": True,
            "source_terminal_receipt_sha256": ledger.DEDICATED_QWEN_ATTEMPT1_BINDING[
                "source_terminal_receipt_sha256"
            ],
        }
    )
    provisional["receipt_sha256"] = self_hosted.digest_without(provisional, "receipt_sha256")
    provisional_path = _write(tmp_path / "ACCEPTED.json", provisional)
    with pytest.raises(ledger.LedgerError, match="fields drifted"):
        ledger.accepted_evidence(provisional_path, authority)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["artifact_file_sha256"].__setitem__(
            "result_file_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value.__setitem__("accepted_receipt_sha256", "sha256:" + "1" * 64),
        lambda value: value.__setitem__("authoritative_projection_omissions", ["model"]),
        lambda value: value.__setitem__("fleet_api_mutations", 1),
    ],
)
def test_validated_dedicated_qwen_chain_drift_fails_closed(
    tmp_path: Path, authority: ledger.Authority, mutation
) -> None:
    value = _dedicated_qwen_validated(authority)
    mutation(value)
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED_VALIDATED.json", value)
    with pytest.raises(ledger.LedgerError, match="drifted|not authoritative"):
        ledger.accepted_evidence(path, authority)


def test_direct_dedicated_qwen_projection_omissions_are_explicit_and_bounded(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    value = _dedicated_qwen_accepted(authority)
    value.update(
        {
            "authoritative_session_task_key_matched": True,
            "authoritative_projection_omissions": ["metadata", "model"],
            "authoritative_projection_rule": (
                "legacy_list_fields_may_be_null_but_never_mismatched_v1"
            ),
        }
    )
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED.json", value)
    assert ledger.accepted_evidence(path, authority).state == "accepted"

    value["authoritative_projection_omissions"] = ["metadata", "unexpected"]
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    _write(path, value)
    with pytest.raises(ledger.LedgerError, match="projection evidence drifted"):
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


def test_generation19_v4_claim_is_bound_to_exact_frozen_plan(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    path, value = _g19_v4_claim(tmp_path, authority)
    evidence = ledger.claim_evidence(path, authority, active=True)
    assert evidence.state == "active"

    value["controller"] = "qwen-b" if value["controller"] == "qwen-a" else "qwen-a"
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    _write(path, value)
    with pytest.raises(ledger.LedgerError, match="absent from exact frozen plans"):
        ledger.claim_evidence(path, authority, active=True)


def test_dedicated_qwen_v2_validated_acceptance_is_digest_only_and_plan_bound(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    source = ROOT / (
        "docs/evidence/qwen38-study/"
        "2026-09-05-qwen38-dedicated-r002-a2-accepted-validated-v2.json"
    )
    evidence = ledger.accepted_evidence(source, authority)
    assert evidence.state == "accepted"

    value = json.loads(source.read_text())
    value["artifact_file_sha256"]["reward"] = "protected-content"
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED_VALIDATED.json", value)
    with pytest.raises(ledger.LedgerError, match="digests only"):
        ledger.accepted_evidence(path, authority)


def test_glm_c2_validated_acceptance_uses_reviewed_runtime_plan_mapping(
    tmp_path: Path, authority: ledger.Authority
) -> None:
    source = ROOT / (
        "docs/evidence/qwen38-study/"
        "2026-09-05-glm53-hosted-c2-accepted-validated-v1.json"
    )
    evidence = ledger.accepted_evidence(source, authority)
    assert evidence.state == "accepted"
    assert evidence.cell_id == (
        "sha256:ef0273d94a3e967f41ae53ee12c808907efee86377d6c85b55740018370b598c"
    )

    value = json.loads(source.read_text())
    value["plan_sha256"] = "sha256:" + "0" * 64
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED_VALIDATED.json", value)
    with pytest.raises(ledger.LedgerError, match="source receipt chain drifted"):
        ledger.accepted_evidence(path, authority)

    value = json.loads(source.read_text())
    value["release_receipt_sha256"] = "sha256:" + "1" * 64
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    _write(path, value)
    with pytest.raises(ledger.LedgerError, match="source receipt chain drifted"):
        ledger.accepted_evidence(path, authority)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["accepted"].__setitem__("cell_id", "sha256:" + "0" * 64),
        lambda value: value["session_inventory"].__setitem__("matching_count", 0),
        lambda value: value["verifier"].__setitem__("present", False),
        lambda value: value["cleanup"].__setitem__("instance_closed", False),
        lambda value: value["accepted"].__setitem__("file_sha256", "not-a-digest"),
    ],
)
def test_hosted_glm_validated_acceptance_is_exact_and_score_blind(
    tmp_path: Path, authority: ledger.Authority, mutation
) -> None:
    source = ROOT / (
        "docs/evidence/glm53-study/"
        "2026-09-06-glm53-hosted-s2-rank27-a1-accepted-validated.json"
    )
    evidence = ledger.accepted_evidence(source, authority)
    assert evidence.state == "accepted"
    assert evidence.cell_id == (
        "sha256:309baef0b19473c8e7b19340adcb881d12fb559118c2a3a3b7576ccb3bb19c00"
    )

    value = json.loads(source.read_text())
    mutation(value)
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = _write(tmp_path / "ACCEPTED_VALIDATED.json", value)
    with pytest.raises(ledger.LedgerError):
        ledger.accepted_evidence(path, authority)


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


def test_rank99_rollforward_acceptance_binds_plan_server_and_session_chain(
    authority: ledger.Authority,
) -> None:
    path = (
        ROOT
        / "docs/evidence/qwen38-study/"
        "2026-09-05-qwen38-dedicated-rank99-g23-a1-accepted-validated-v1.json"
    )
    evidence = ledger.accepted_evidence(path, authority)
    assert evidence.state == "accepted"
    assert evidence.cell_id == (
        "sha256:bf9f6d8aee8d775f6ce5dbd238d0b3a7abc544da0687c9cb943b3ba090408b2a"
    )
    assert evidence.execution_generation == 23


def test_cli_is_read_only_and_emits_json_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ledger.main(["--repo-root", str(ROOT), "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["universe_cell_count"] == 800
    assert len(value["cells"]) == 800
