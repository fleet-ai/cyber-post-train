from __future__ import annotations

import copy
import json
import shutil
import uuid
from base64 import b64decode
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.fleet import heldout_stored_session_reconciliation as packet
from evals.fleet import rollout_ledger
from evals.fleet import stored_session_reconciliation_v2 as reconciliation

ROOT = Path(__file__).parents[1]
JOB_NAME = "chris-q38-s47-base-existing-v1"
CONFIG_MAP_NAME = "chris-q38-s47-base-existing-code-v1"
SECRET_NAME = "chris-q38-s47-base-existing-intent-v1"
OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-s47-base-existing-v1"


def _source_package(tmp_path: Path) -> SimpleNamespace:
    launch = tmp_path / "LAUNCH_PACKET.json"
    launch.write_bytes(b'{"sealed":"source-packet"}\n')
    return SimpleNamespace(
        packet=SimpleNamespace(
            namespace=packet.NAMESPACE,
            job_name="chris-q38-dev17-s47-base-p1-v1",
            config_map_name="chris-q38-dev17-s47-base-code-v1",
            output_root="/mnt/sfs/jobs/chris-q38-fleet-dev17-s47-base-p1-v1",
            database="q38_dev17_s47_base_p1_v1",
            identity_sha256="sha256:" + "1" * 64,
            identity={
                "comparison_protocol_sha256": "sha256:" + "2" * 64,
                "protocol_id": "q38-dev17-s47-base-t3k32s1000-p1-v1",
                "arm_id": "base",
                "retry_limit": 0,
            },
            path=launch,
        ),
        evaluation_config={
            "pass_k": 1,
            "max_reviewed_infrastructure_retries": 0,
            "routes": {"base": {"task_versions": [str(uuid.uuid4()) for _ in range(17)]}},
        },
    )


def _terminal(source: SimpleNamespace, *, selected: int = 2, local_results: int = 17) -> dict:
    states = {state: 0 for state in rollout_ledger.STATES}
    states.update(accepted=17 - selected, retry_review=selected)
    value = {
        "schema": packet.heldout_launch.TERMINAL_SCHEMA,
        "observed_at": "2026-09-23T00:00:00+00:00",
        "evaluation_identity_sha256": source.packet.identity_sha256,
        "comparison_protocol_sha256": source.packet.identity["comparison_protocol_sha256"],
        "protocol_id": source.packet.identity["protocol_id"],
        "arm_id": source.packet.identity["arm_id"],
        "job": {
            "name": source.packet.job_name,
            "uid": "11111111-1111-4111-8111-111111111111",
            "terminal_condition": "Failed",
            "succeeded": 0,
            "failed": 1,
        },
        "config_map": {
            "name": source.packet.config_map_name,
            "uid": "22222222-2222-4222-8222-222222222222",
        },
        "workloads": [],
        "pods": [
            {
                "name": "source-pod",
                "uid": "33333333-3333-4333-8333-333333333333",
                "phase": "Failed",
            }
        ],
        "database": {
            "name": source.packet.database,
            "summary": {
                "total": 17,
                "local_results": local_results,
                "by_state": states,
                "by_serving_block": [
                    {"serving_block": "base", "state": state, "count": count}
                    for state, count in states.items()
                    if count
                ],
                "stale_active": 0,
                "plan_sha256": "a" * 64,
            },
        },
        "output_root": {"path": source.packet.output_root, "exists": True},
        "decision": {
            "capability_result_status": "not_interpreted",
            "score_blind_reconciliation_required": True,
            "unresolved_cells": selected,
            "rollout_retry_performed": False,
            "score_read_or_generated": False,
        },
        "privacy": {
            "prompts_responses_flags_rewards_or_trace_content_included": False,
            "score_values_included": False,
            "credentials_included": False,
        },
    }
    return {**value, "sha256": packet._canonical_digest(value)}  # noqa: SLF001


def _terminal_path(tmp_path: Path, terminal: dict) -> Path:
    path = tmp_path / "TERMINAL_OBSERVATION.json"
    path.write_text(json.dumps(terminal), encoding="utf-8")
    return path


def _create_evidence(
    tmp_path: Path, source: SimpleNamespace, terminal: dict, *, gpus: int | bool = 0
) -> Path:
    zero = {"jobs": 0, "config_maps": 0, "pods": 0}
    intent = {
        "schema": "cyber_fleet_heldout_create_intent_v1",
        "state": "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
        "packet_sha256": packet._file_digest_bytes(source.packet.path.read_bytes()),  # noqa: SLF001
        "evaluation_identity_sha256": source.packet.identity_sha256,
        "comparison_protocol_sha256": source.packet.identity["comparison_protocol_sha256"],
        "protocol_id": source.packet.identity["protocol_id"],
        "arm_id": source.packet.identity["arm_id"],
        "namespace": source.packet.namespace,
        "job_name": source.packet.job_name,
        "config_map_name": source.packet.config_map_name,
        "output_root": source.packet.output_root,
        "database": source.packet.database,
        "server_preview_sha256": "sha256:" + "3" * 64,
        "first_duplicate_census": zero,
        "final_duplicate_census": zero,
    }
    response = {
        "state": "KUBECTL_CREATE_RESPONSE",
        "submitted": True,
        "gpus": gpus,
        "job_name": source.packet.job_name,
        "job_uid": terminal["job"]["uid"],
        "config_map_name": source.packet.config_map_name,
        "config_map_uid": terminal["config_map"]["uid"],
        "evaluation_identity_sha256": source.packet.identity_sha256,
        "comparison_protocol_sha256": source.packet.identity["comparison_protocol_sha256"],
    }
    path = tmp_path / f"CREATE_INTENT-{uuid.uuid4()}.jsonl"
    path.write_text(json.dumps(intent) + "\n" + json.dumps(response) + "\n", encoding="utf-8")
    return path


def _build_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, repo_root: Path = ROOT
) -> tuple[dict, SimpleNamespace, dict, Path, Path]:
    source = _source_package(tmp_path)
    terminal = _terminal(source)
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    value = packet.build_private_intent_value(
        repo_root=repo_root,
        source_launch_packet=source.packet.path,
        source_terminal_receipt=terminal_path,
        source_create_evidence=evidence,
        selected_cell_ids=[str(uuid.uuid4()), str(uuid.uuid4())],
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        secret_name=SECRET_NAME,
        output_root=OUTPUT_ROOT,
    )
    return value, source, terminal, terminal_path, evidence


def _render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    value, source, terminal, terminal_path, evidence = _build_authorization(tmp_path, monkeypatch)
    intent = tmp_path / "reviewed-intent.json"
    assert packet.write_private_intent(intent, value) == value["sha256"]
    rendered = packet.render(
        repo_root=ROOT,
        source_launch_packet=source.packet.path,
        source_terminal_receipt=terminal_path,
        source_create_evidence=evidence,
        private_intent=intent,
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        secret_name=SECRET_NAME,
        output_root=OUTPUT_ROOT,
    )
    return rendered, value, source, terminal, terminal_path, evidence, intent


def _build_subset_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = _source_package(tmp_path)
    terminal = _terminal(source, selected=7)
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    selected = [str(uuid.uuid4()) for _ in range(4)]
    unselected = [str(uuid.uuid4()) for _ in range(13)]
    value = packet.build_private_intent_value(
        repo_root=ROOT,
        source_launch_packet=source.packet.path,
        source_terminal_receipt=terminal_path,
        source_create_evidence=evidence,
        selected_cell_ids=selected,
        unselected_cell_ids=unselected,
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        secret_name=SECRET_NAME,
        output_root=OUTPUT_ROOT,
    )
    return value, source, terminal, terminal_path, evidence


def _build_s47_local_gap_authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = _source_package(tmp_path)
    terminal = _terminal(source, selected=7, local_results=15)
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    selected = [str(uuid.uuid4()) for _ in range(4)]
    unselected = [str(uuid.uuid4()) for _ in range(13)]
    missing = unselected[:2]
    value = packet.build_private_intent_value(
        repo_root=ROOT,
        source_launch_packet=source.packet.path,
        source_terminal_receipt=terminal_path,
        source_create_evidence=evidence,
        selected_cell_ids=selected,
        unselected_cell_ids=unselected,
        missing_local_result_cell_ids=missing,
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        secret_name=SECRET_NAME,
        output_root=OUTPUT_ROOT,
    )
    return value, source, terminal, terminal_path, evidence, selected, unselected, missing


def test_build_authorization_binds_source_create_closure_and_exact_bundle(tmp_path, monkeypatch):
    value, source, terminal, _, evidence = _build_authorization(tmp_path, monkeypatch)
    runtime = value["runtime_intent"]
    closure = value["executable_closure"]
    assert value["schema"] == packet.PRIVATE_INTENT_SCHEMA
    assert value["source_launch_packet_sha256"] == packet._file_digest_bytes(  # noqa: SLF001
        source.packet.path.read_bytes()
    )
    assert value["source_terminal_receipt_sha256"] == terminal["sha256"]
    assert value["source_create_evidence_sha256"] == packet._file_digest_bytes(  # noqa: SLF001
        evidence.read_bytes()
    )
    assert runtime["schema_version"] == reconciliation.INTENT_SCHEMA
    assert runtime["selected_cell_ids"]
    assert closure["dependency_pins"] == packet.DEPENDENCY_PINS
    assert set(closure["files_sha256"]) == {*packet.CODE_FILES, "run.sh"}
    assert closure["runtime_intent_file_sha256"].startswith("sha256:")
    assert value["bundle_sha256"].startswith("sha256:")
    assert value["sha256"] == packet._canonical_digest(  # noqa: SLF001
        {key: item for key, item in value.items() if key != "sha256"}
    )


def test_subset_authorization_binds_full_mixed_arm_census(tmp_path, monkeypatch):
    value, source, terminal, terminal_path, evidence = _build_subset_authorization(
        tmp_path, monkeypatch
    )
    runtime = value["runtime_intent"]
    assert runtime["schema_version"] == reconciliation.SUBSET_INTENT_SCHEMA
    assert len(runtime["selected_cell_ids"]) == 4
    assert len(runtime["unselected_cell_ids"]) == 13
    assert set(runtime["selected_cell_ids"]).isdisjoint(runtime["unselected_cell_ids"])
    assert runtime["expected_arm_state_counts"] == terminal["database"]["summary"]["by_state"]
    intent_path = tmp_path / "subset-intent.json"
    packet.write_private_intent(intent_path, value)
    rendered = packet.render(
        repo_root=ROOT,
        source_launch_packet=source.packet.path,
        source_terminal_receipt=terminal_path,
        source_create_evidence=evidence,
        private_intent=intent_path,
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        secret_name=SECRET_NAME,
        output_root=OUTPUT_ROOT,
    )
    assert rendered.proof["subset_reconciliation"] is True
    assert rendered.proof["selected_cell_count"] == 4
    assert rendered.proof["source_retry_review_count"] == 7
    assert rendered.proof["source_total_cell_count"] == 17
    assert rendered.proof["nonselected_cell_count"] == 13
    assert rendered.proof["nonselected_cells_preserved_byte_for_byte_and_state_for_state"] is True
    assert rendered.proof["model_generation_allowed"] is False
    assert rendered.proof["scoring_call_allowed"] is False
    pod = rendered.job["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(pod["containers"][0]["resources"])
    assert rendered.job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    first = _server_preview(rendered, "11111111-1111-4111-8111-111111111111")
    second = _server_preview(rendered, "22222222-2222-4222-8222-222222222222")
    assert packet.validate_server_previews(rendered, first, second).startswith("sha256:")


def test_process_error_subset_authorization_is_exact_and_score_blind(tmp_path, monkeypatch):
    source = _source_package(tmp_path)
    terminal = _terminal(source, selected=1)
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    selected = [str(uuid.uuid4())]
    unselected = [str(uuid.uuid4()) for _ in range(16)]

    value = packet.build_private_intent_value(
        repo_root=ROOT,
        source_launch_packet=source.packet.path,
        source_terminal_receipt=terminal_path,
        source_create_evidence=evidence,
        selected_cell_ids=selected,
        unselected_cell_ids=unselected,
        expected_agent_exit_code=1,
        expected_agent_termination="process_error",
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        secret_name=SECRET_NAME,
        output_root=OUTPUT_ROOT,
    )
    runtime = value["runtime_intent"]
    assert runtime["expected_agent_exit_code"] == 1
    assert runtime["expected_agent_termination"] == "process_error"
    assert runtime["expected_arm_state_counts"] == terminal["database"]["summary"]["by_state"]
    intent_path = tmp_path / "process-error-intent.json"
    packet.write_private_intent(intent_path, value)

    rendered = packet.render(
        repo_root=ROOT,
        source_launch_packet=source.packet.path,
        source_terminal_receipt=terminal_path,
        source_create_evidence=evidence,
        private_intent=intent_path,
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        secret_name=SECRET_NAME,
        output_root=OUTPUT_ROOT,
    )
    assert rendered.proof["selected_cell_count"] == 1
    assert rendered.proof["nonselected_cell_count"] == 16
    assert rendered.proof["model_generation_allowed"] is False
    assert rendered.proof["scoring_call_allowed"] is False
    assert rendered.job["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert rendered.job["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert "nvidia.com/gpu" not in json.dumps(
        rendered.job["spec"]["template"]["spec"]["containers"][0]["resources"]
    )
    assert all(cell_id not in json.dumps(rendered.proof) for cell_id in selected + unselected)


@pytest.mark.parametrize(
    ("exit_code", "termination"),
    [
        (0, "process_error"),
        (1, "output_limit"),
        (2, "process_error"),
        (0.0, "output_limit"),
        (1.0, "process_error"),
    ],
)
def test_private_authorization_rejects_unsupported_agent_outcome(
    tmp_path, monkeypatch, exit_code, termination
):
    source = _source_package(tmp_path)
    terminal = _terminal(source, selected=1)
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)

    with pytest.raises(packet.ReconciliationPacketError, match="intent is invalid"):
        packet.build_private_intent_value(
            repo_root=ROOT,
            source_launch_packet=source.packet.path,
            source_terminal_receipt=terminal_path,
            source_create_evidence=evidence,
            selected_cell_ids=[str(uuid.uuid4())],
            unselected_cell_ids=[str(uuid.uuid4()) for _ in range(16)],
            expected_agent_exit_code=exit_code,
            expected_agent_termination=termination,
            job_name=JOB_NAME,
            config_map_name=CONFIG_MAP_NAME,
            secret_name=SECRET_NAME,
            output_root=OUTPUT_ROOT,
        )


def test_s47_authorization_binds_exact_two_unselected_local_result_gaps(tmp_path, monkeypatch):
    (
        value,
        source,
        terminal,
        terminal_path,
        evidence,
        selected,
        unselected,
        missing,
    ) = _build_s47_local_gap_authorization(tmp_path, monkeypatch)
    runtime = value["runtime_intent"]
    assert runtime["schema_version"] == reconciliation.SUBSET_LOCAL_GAPS_INTENT_SCHEMA
    assert runtime["expected_local_result_count"] == 15
    assert runtime["missing_local_result_cell_ids"] == missing
    assert set(missing).issubset(unselected)
    assert set(missing).isdisjoint(selected)
    intent_path = tmp_path / "s47-gap-intent.json"
    packet.write_private_intent(intent_path, value)

    rendered = packet.render(
        repo_root=ROOT,
        source_launch_packet=source.packet.path,
        source_terminal_receipt=terminal_path,
        source_create_evidence=evidence,
        private_intent=intent_path,
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        secret_name=SECRET_NAME,
        output_root=OUTPUT_ROOT,
    )

    assert rendered.proof["source_local_result_count"] == 15
    assert rendered.proof["missing_local_result_count"] == 2
    assert rendered.proof["missing_local_results_are_unselected"] is True
    assert rendered.proof["selected_cells_have_local_results"] is True
    assert rendered.proof["missing_local_result_cells_preserved"] is True
    assert rendered.proof["model_generation_allowed"] is False
    assert rendered.proof["scoring_call_allowed"] is False
    assert terminal["database"]["summary"]["total"] == 17
    serialized_proof = json.dumps(rendered.proof)
    assert all(cell_id not in serialized_proof for cell_id in selected + unselected)


def test_partial_local_results_require_exact_private_missing_roster(tmp_path, monkeypatch):
    source = _source_package(tmp_path)
    terminal = _terminal(source, selected=7, local_results=15)
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    selected = [str(uuid.uuid4()) for _ in range(4)]
    unselected = [str(uuid.uuid4()) for _ in range(13)]

    with pytest.raises(packet.ReconciliationPacketError, match="review-held"):
        packet.build_private_intent_value(
            repo_root=ROOT,
            source_launch_packet=source.packet.path,
            source_terminal_receipt=terminal_path,
            source_create_evidence=evidence,
            selected_cell_ids=selected,
            unselected_cell_ids=unselected,
            job_name=JOB_NAME,
            config_map_name=CONFIG_MAP_NAME,
            secret_name=SECRET_NAME,
            output_root=OUTPUT_ROOT,
        )


@pytest.mark.parametrize("fault", ["one", "three", "selected", "wrong-count"])
def test_partial_local_result_packet_rejects_any_broader_exception(tmp_path, monkeypatch, fault):
    source = _source_package(tmp_path)
    terminal = _terminal(
        source,
        selected=7,
        local_results=14 if fault == "wrong-count" else 15,
    )
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    selected = [str(uuid.uuid4()) for _ in range(4)]
    unselected = [str(uuid.uuid4()) for _ in range(13)]
    missing = unselected[:2]
    if fault == "one":
        missing = missing[:1]
    elif fault == "three":
        missing.append(unselected[2])
    elif fault == "selected":
        missing[0] = selected[0]

    with pytest.raises(packet.ReconciliationPacketError, match="private reconciliation intent"):
        packet.build_private_intent_value(
            repo_root=ROOT,
            source_launch_packet=source.packet.path,
            source_terminal_receipt=terminal_path,
            source_create_evidence=evidence,
            selected_cell_ids=selected,
            unselected_cell_ids=unselected,
            missing_local_result_cell_ids=missing,
            job_name=JOB_NAME,
            config_map_name=CONFIG_MAP_NAME,
            secret_name=SECRET_NAME,
            output_root=OUTPUT_ROOT,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "overlap",
        "missing-complement",
        "route-summary-drift",
        "selected-overflow",
        "active-state",
    ],
)
def test_subset_authorization_rejects_nonexhaustive_or_drifting_census(
    tmp_path, monkeypatch, fault
):
    source = _source_package(tmp_path)
    terminal = _terminal(source, selected=7)
    selected = [str(uuid.uuid4()) for _ in range(4)]
    unselected = [str(uuid.uuid4()) for _ in range(13)]
    if fault == "overlap":
        unselected[0] = selected[0]
    elif fault == "missing-complement":
        unselected.pop()
    elif fault == "route-summary-drift":
        terminal["database"]["summary"]["by_state"]["accepted"] -= 1
        terminal["database"]["summary"]["by_state"]["retry_review"] += 1
    elif fault == "selected-overflow":
        selected.extend(str(uuid.uuid4()) for _ in range(4))
        unselected = unselected[:-4]
    else:
        terminal["database"]["summary"]["by_state"]["accepted"] -= 1
        terminal["database"]["summary"]["by_state"]["running"] = 1
        terminal["database"]["summary"]["by_serving_block"] = [
            {"serving_block": "base", "state": state, "count": count}
            for state, count in terminal["database"]["summary"]["by_state"].items()
            if count
        ]
    terminal["sha256"] = packet._canonical_digest(  # noqa: SLF001
        {key: item for key, item in terminal.items() if key != "sha256"}
    )
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    with pytest.raises((packet.ReconciliationPacketError, rollout_ledger.LedgerError)):
        packet.build_private_intent_value(
            repo_root=ROOT,
            source_launch_packet=source.packet.path,
            source_terminal_receipt=terminal_path,
            source_create_evidence=evidence,
            selected_cell_ids=selected,
            unselected_cell_ids=unselected,
            job_name=JOB_NAME,
            config_map_name=CONFIG_MAP_NAME,
            secret_name=SECRET_NAME,
            output_root=OUTPUT_ROOT,
        )


def test_private_authorization_is_exclusive_private(tmp_path, monkeypatch):
    value, *_ = _build_authorization(tmp_path, monkeypatch)
    path = tmp_path / "intent.json"
    assert packet.write_private_intent(path, value) == value["sha256"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_bytes()) == value
    with pytest.raises(packet.ReconciliationPacketError, match="already exists"):
        packet.write_private_intent(path, value)


def test_render_is_cpu_only_score_blind_and_exact(tmp_path, monkeypatch):
    rendered, value, source, terminal, *_ = _render(tmp_path, monkeypatch)
    pod_metadata = rendered.job["spec"]["template"]["metadata"]
    pod = rendered.job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert rendered.job["metadata"]["annotations"] == {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
    }
    assert rendered.job["spec"]["backoffLimit"] == 0
    assert pod_metadata["labels"]["cyber-post-train.fleet.ai/postgres-client"] == "true"
    assert pod["automountServiceAccountToken"] is False
    assert pod["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(container["resources"])
    assert rendered.proof["source_job_uid"] == terminal["job"]["uid"]
    assert rendered.proof["source_config_map_uid"] == terminal["config_map"]["uid"]
    assert rendered.proof["operation"] == "accept_existing_scored_session"
    assert rendered.proof["schema"] == packet.PROOF_SCHEMA
    assert "subset_reconciliation" not in rendered.proof
    assert "nonselected_cell_count" not in rendered.proof
    assert rendered.proof["model_generation_allowed"] is False
    assert rendered.proof["scoring_call_allowed"] is False
    assert rendered.proof["reviewed_intent_sha256"] == value["sha256"]
    assert rendered.proof["bundle_sha256"] == packet._canonical_digest(rendered.bundle)  # noqa: SLF001
    assert set(rendered.secret["data"]) == {"intent.json", "closure.json"}
    assert (
        json.loads(b64decode(rendered.secret["data"]["intent.json"], validate=True))
        == value["runtime_intent"]
    )
    assert all(
        cell not in json.dumps(rendered.proof)
        for cell in value["runtime_intent"]["selected_cell_ids"]
    )
    assert container["env"][0]["value"] == source.packet.output_root


def test_render_rejects_support_file_drift_in_full_executable_closure(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    for relative in packet.CODE_FILES.values():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    value, source, _, terminal_path, evidence = _build_authorization(
        tmp_path, monkeypatch, repo_root=repo
    )
    intent = tmp_path / "intent.json"
    packet.write_private_intent(intent, value)
    support = repo / packet.CODE_FILES["rollout_worker.py"]
    support.write_text(support.read_text() + "\n# support drift\n", encoding="utf-8")
    with pytest.raises(packet.ReconciliationPacketError, match="executable closure differs"):
        packet.render(
            repo_root=repo,
            source_launch_packet=source.packet.path,
            source_terminal_receipt=terminal_path,
            source_create_evidence=evidence,
            private_intent=intent,
            job_name=JOB_NAME,
            config_map_name=CONFIG_MAP_NAME,
            secret_name=SECRET_NAME,
            output_root=OUTPUT_ROOT,
        )


@pytest.mark.parametrize(
    "candidate",
    [
        "/mnt/sfs/jobs/chris-q38-fleet-dev17-s47-base-p1-v1",
        "/mnt/sfs/jobs/chris-q38-fleet-dev17-s47-base-p1-v1/child",
        "/mnt/sfs/jobs",
    ],
)
def test_output_root_rejects_equal_child_or_parent(candidate):
    with pytest.raises(packet.ReconciliationPacketError, match="output root"):
        packet._output_root(  # noqa: SLF001
            candidate, source="/mnt/sfs/jobs/chris-q38-fleet-dev17-s47-base-p1-v1"
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(score=0.5),
        lambda value: value["decision"].update(raw_score=0.5),
        lambda value: value["database"]["summary"].update(score_values=[]),
        lambda value: value["pods"][0].update(raw={}),
    ],
)
def test_terminal_receipt_rejects_unknown_score_or_raw_fields(tmp_path, monkeypatch, mutate):
    source = _source_package(tmp_path)
    terminal = _terminal(source)
    mutate(terminal)
    terminal["sha256"] = packet._canonical_digest(  # noqa: SLF001
        {key: item for key, item in terminal.items() if key != "sha256"}
    )
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    with pytest.raises(packet.ReconciliationPacketError, match="shape differs"):
        _build_value(source, terminal_path, evidence)


def _build_value(source, terminal_path, evidence):
    return packet.build_private_intent_value(
        repo_root=ROOT,
        source_launch_packet=source.packet.path,
        source_terminal_receipt=terminal_path,
        source_create_evidence=evidence,
        selected_cell_ids=[str(uuid.uuid4()), str(uuid.uuid4())],
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        secret_name=SECRET_NAME,
        output_root=OUTPUT_ROOT,
    )


@pytest.mark.parametrize("fault", ["wrong-job-uid", "wrong-config-uid", "bool-gpu", "bool-census"])
def test_source_create_evidence_is_exact_and_uid_bound(tmp_path, monkeypatch, fault):
    source = _source_package(tmp_path)
    terminal = _terminal(source)
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    records = [json.loads(line) for line in evidence.read_text().splitlines()]
    if fault == "wrong-job-uid":
        records[1]["job_uid"] = "44444444-4444-4444-8444-444444444444"
    elif fault == "wrong-config-uid":
        records[1]["config_map_uid"] = "55555555-5555-4555-8555-555555555555"
    elif fault == "bool-gpu":
        records[1]["gpus"] = False
    else:
        records[0]["first_duplicate_census"]["jobs"] = False
    evidence.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    with pytest.raises(packet.ReconciliationPacketError, match="create evidence identity"):
        _build_value(source, terminal_path, evidence)


def test_source_create_evidence_rejects_symlink(tmp_path, monkeypatch):
    source = _source_package(tmp_path)
    terminal = _terminal(source)
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    link = tmp_path / "journal-link.jsonl"
    link.symlink_to(evidence)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    with pytest.raises(packet.ReconciliationPacketError, match="unreadable"):
        _build_value(source, terminal_path, link)


def _server_preview(rendered: packet.Package, uid: str) -> dict:
    response = copy.deepcopy(rendered.bundle)
    job = next(item for item in response["items"] if item["kind"] == "Job")
    job["metadata"].update(uid=uid, resourceVersion="1")
    job["spec"]["selector"] = {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}}
    job["spec"]["template"]["metadata"]["labels"].update(
        {
            "batch.kubernetes.io/controller-uid": uid,
            "controller-uid": uid,
        }
    )
    return response


def test_two_exact_server_previews_are_stable(tmp_path, monkeypatch):
    rendered, *_ = _render(tmp_path, monkeypatch)
    first = _server_preview(rendered, "11111111-1111-4111-8111-111111111111")
    second = _server_preview(rendered, "22222222-2222-4222-8222-222222222222")
    assert packet.validate_server_previews(rendered, first, second).startswith("sha256:")


@pytest.mark.parametrize("suspend", [False, True])
def test_enumerated_api_defaults_and_non_job_metadata_are_accepted(tmp_path, monkeypatch, suspend):
    rendered, *_ = _render(tmp_path, monkeypatch)
    previews = [
        _server_preview(rendered, "11111111-1111-4111-8111-111111111111"),
        _server_preview(rendered, "22222222-2222-4222-8222-222222222222"),
    ]
    for preview in previews:
        for item in preview["items"]:
            item["metadata"].update(creationTimestamp=None, generation=1)
        job = next(item for item in preview["items"] if item["kind"] == "Job")
        job["spec"].update(
            parallelism=1,
            completions=1,
            completionMode="NonIndexed",
            manualSelector=False,
            suspend=suspend,
            podReplacementPolicy="TerminatingOrFailed",
        )
        pod = job["spec"]["template"]["spec"]
        pod.update(
            dnsPolicy="ClusterFirst",
            enableServiceLinks=True,
            preemptionPolicy="PreemptLowerPriority",
            priority=10_000,
            schedulerName="default-scheduler",
            securityContext={},
            serviceAccount="default",
            serviceAccountName="default",
            terminationGracePeriodSeconds=30,
        )
        pod["containers"][0].update(
            imagePullPolicy="IfNotPresent",
            terminationMessagePath="/dev/termination-log",
            terminationMessagePolicy="File",
        )
    assert packet.validate_server_previews(rendered, *previews).startswith("sha256:")


@pytest.mark.parametrize(
    "fault",
    [
        "command",
        "create-once",
        "init-container",
        "postgres-network-label",
        "service-account",
        "environment",
        "volume",
        "secret-data",
    ],
)
def test_server_preview_rejects_behavioral_or_private_mutation(tmp_path, monkeypatch, fault):
    rendered, *_ = _render(tmp_path, monkeypatch)
    first = _server_preview(rendered, "11111111-1111-4111-8111-111111111111")
    second = _server_preview(rendered, "22222222-2222-4222-8222-222222222222")
    job = next(item for item in second["items"] if item["kind"] == "Job")
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if fault == "command":
        container["command"] = ["/bin/true"]
    elif fault == "create-once":
        job["metadata"]["annotations"][packet.heldout_launch.CREATE_ONCE_ANNOTATION] = "false"
    elif fault == "init-container":
        pod["initContainers"] = [{"name": "extra", "image": "busybox"}]
    elif fault == "postgres-network-label":
        job["spec"]["template"]["metadata"]["labels"].pop(
            "cyber-post-train.fleet.ai/postgres-client"
        )
    elif fault == "service-account":
        pod["serviceAccountName"] = "privileged"
    elif fault == "environment":
        container["env"].append({"name": "EXTRA", "value": "1"})
    elif fault == "volume":
        pod["volumes"].append({"name": "extra", "emptyDir": {}})
    else:
        secret = next(item for item in second["items"] if item["kind"] == "Secret")
        secret["data"]["extra"] = "unexpected"
    with pytest.raises(packet.ReconciliationPacketError, match="package differs"):
        packet.validate_server_previews(rendered, first, second)


def test_validate_rejects_post_render_bundle_mutation(tmp_path, monkeypatch):
    rendered, *_ = _render(tmp_path, monkeypatch)
    rendered.job["spec"]["template"]["spec"]["containers"][0]["command"] = ["/bin/true"]
    with pytest.raises(packet.ReconciliationPacketError, match="unsafe"):
        packet.validate(rendered)


@pytest.mark.parametrize("automount", [None, True])
def test_validate_requires_service_account_token_automount_disabled(
    tmp_path, monkeypatch, automount
):
    rendered, *_ = _render(tmp_path, monkeypatch)
    pod = rendered.job["spec"]["template"]["spec"]
    if automount is None:
        pod.pop("automountServiceAccountToken")
    else:
        pod["automountServiceAccountToken"] = automount
    with pytest.raises(packet.ReconciliationPacketError, match="unsafe"):
        packet.validate(rendered)


def test_validate_rejects_bundle_mutation_even_if_mutable_proof_is_rehashed(tmp_path, monkeypatch):
    rendered, *_ = _render(tmp_path, monkeypatch)
    rendered.job["spec"]["template"]["spec"]["containers"][0]["env"].append(
        {"name": "EXTRA", "value": "1"}
    )
    rendered.proof["bundle_sha256"] = packet._canonical_digest(rendered.bundle)  # noqa: SLF001
    rendered.proof["sha256"] = packet._canonical_digest(  # noqa: SLF001
        {key: value for key, value in rendered.proof.items() if key != "sha256"}
    )
    with pytest.raises(packet.ReconciliationPacketError, match="unsafe"):
        packet.validate(rendered)


@pytest.mark.parametrize("fault", ["selector", "batch-controller", "legacy-controller"])
def test_server_preview_rejects_unproven_job_uid_projection(tmp_path, monkeypatch, fault):
    rendered, *_ = _render(tmp_path, monkeypatch)
    first = _server_preview(rendered, "11111111-1111-4111-8111-111111111111")
    second = _server_preview(rendered, "22222222-2222-4222-8222-222222222222")
    job = next(item for item in second["items"] if item["kind"] == "Job")
    if fault == "selector":
        job["spec"]["selector"] = {"matchLabels": {"evil": "x"}}
    else:
        job["spec"]["template"]["metadata"]["labels"][
            "batch.kubernetes.io/controller-uid"
            if fault == "batch-controller"
            else "controller-uid"
        ] = "totally-wrong"
    with pytest.raises(packet.ReconciliationPacketError, match="UID projections"):
        packet.validate_server_previews(rendered, first, second)


def test_runtime_script_verifies_full_closure_before_uv(tmp_path, monkeypatch):
    rendered, value, *_ = _render(tmp_path, monkeypatch)
    run = rendered.config_map["data"]["run.sh"]
    assert 'Path("/intent/closure.json")' in run
    assert "reconciliation executable closure differs" in run
    assert run.index("reconciliation executable closure differs") < run.index("exec uv run")
    assert value["executable_closure"]["image"] == packet.EVALUATOR_IMAGE
    for version in packet.DEPENDENCY_PINS.values():
        assert version in run


def test_terminal_job_must_be_exact_failed_source(tmp_path, monkeypatch):
    source = _source_package(tmp_path)
    terminal = _terminal(source)
    terminal["job"].update(terminal_condition="Complete", succeeded=1, failed=0)
    terminal["sha256"] = packet._canonical_digest(  # noqa: SLF001
        {key: item for key, item in terminal.items() if key != "sha256"}
    )
    terminal_path = _terminal_path(tmp_path, terminal)
    evidence = _create_evidence(tmp_path, source, terminal)
    monkeypatch.setattr(packet.heldout_launch, "build_package", lambda _: source)
    with pytest.raises(packet.ReconciliationPacketError, match="terminal"):
        _build_value(source, terminal_path, evidence)
