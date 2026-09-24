from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from evals.fleet import task_quality_qualification as qualification

ROOT = Path(__file__).parents[1]
PACKET = ROOT / "configs/qualification/fleet-blackbox-qa33-zero-model-canary-20260924-v1.json"
PACKET_COMMIT = "0ae3e0923c1e74e65bea3a91f102c1ac7a12917d"


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _packet_source(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{PACKET_COMMIT}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout


def test_packet_is_self_digested_and_binds_exact_inputs():
    packet = _load(PACKET)
    assert packet["sha256"] == qualification.digest(
        {key: item for key, item in packet.items() if key != "sha256"}
    )
    for binding in packet["inputs"].values():
        path = ROOT / binding["path"]
        source = _packet_source(path)
        assert "sha256:" + hashlib.sha256(source).hexdigest() == binding["file_sha256"]
        if "logical_sha256" in binding:
            assert json.loads(source)["sha256"] == binding["logical_sha256"]


def test_packet_candidate_is_the_exact_deterministic_lineage_join():
    packet = _load(PACKET)
    candidate = packet["candidate"]
    live = _load(ROOT / packet["inputs"]["live_lineage"]["path"])
    census = _load(ROOT / packet["inputs"]["lineage_census"]["path"])
    qa = _load(ROOT / packet["inputs"]["qa_candidates"]["path"])
    identity = (candidate["task_key"], candidate["task_version_id"])

    qa_rows = [row for row in qa["tasks"] if (row["task_key"], row["task_version_id"]) == identity]
    live_rows = [
        row
        for row in live["task_versions"]
        if (row["task_key"], row["task_version_id"]) == identity
    ]
    census_rows = [
        row
        for row in census["candidate_task_versions"]
        if (row["task_key"], row["task_version_id"]) == identity
    ]
    assert len(qa_rows) == len(live_rows) == len(census_rows) == 1
    assert qa_rows[0]["task_id"] == candidate["task_id"]
    assert qa_rows[0]["qa_status"] == candidate["qa_status"] == "clean"
    for field in (
        "atom_artifact_keys",
        "atom_source_locators",
        "lineage",
        "lineage_binding_sha256",
        "task_graph_source_locator",
        "task_id",
    ):
        assert live_rows[0][field] == candidate[field]
    assert census_rows[0]["component_id"] == candidate["shared_atom_component_id"]
    assert census_rows[0]["runtime_receipt_qualified"] is False
    assert census_rows[0]["admission"] == "excluded_missing_complete_runtime_receipts"
    assert census_rows[0]["teacher3k_exact_version_exposed"] is False
    assert census_rows[0]["teacher3k_shared_atom_exposed"] is False

    clean = [row for row in live["task_versions"] if row["qa_status"] == "clean"]
    modal = [
        row
        for row in clean
        if row["lineage"]["difficulty"] == "medium"
        and len(row["atom_artifact_keys"]) == 1
        and isinstance(row["lineage"]["application"], str)
    ]
    exposure = {
        (row["task_key"], row["task_version_id"]): row for row in census["candidate_task_versions"]
    }
    unexposed = [
        row
        for row in modal
        if exposure[(row["task_key"], row["task_version_id"])]["teacher3k_exact_version_exposed"]
        is False
        and exposure[(row["task_key"], row["task_version_id"])]["teacher3k_shared_atom_exposed"]
        is False
    ]
    assert (len(live["task_versions"]), len(clean), len(modal), len(unexposed)) == (33, 8, 7, 1)
    assert (unexposed[0]["task_key"], unexposed[0]["task_version_id"]) == identity


def test_packet_is_inert_and_does_not_claim_unobserved_evidence():
    packet = _load(PACKET)
    assert packet["source"]["review_packet_merged_to_origin_main"] is False
    assert packet["no_launch_receipt"] == {
        "external_mutations": 0,
        "kubernetes_objects_created": 0,
        "launch_authorized": False,
        "launch_performed": False,
        "launchable": False,
        "server_previews_performed": 0,
        "status": "prepared_no_launch",
    }
    assert packet["missing_launch_gates"]
    rendered = json.dumps(packet, sort_keys=True)
    assert "job_uid" not in packet["no_launch_receipt"]
    assert "preview_sha256" not in rendered
    assert packet["privacy"]["credentials_included"] is False


def test_future_contract_is_zero_model_cpu_only_create_once_and_cleanup_bound():
    packet = _load(PACKET)
    qualification_contract = packet["qualification"]
    contract = packet["future_cpu_canary_contract"]
    assert qualification_contract["candidate_count"] == 1
    assert qualification_contract["concurrency"] == 1
    assert qualification_contract["model_calls"] == 0
    assert qualification_contract["training_data_eligible"] is False
    assert qualification_contract["fresh_plan_exact_task_identity"] == {
        "exact_task_key": packet["candidate"]["task_key"],
        "exact_task_version_id": packet["candidate"]["task_version_id"],
    }
    assert qualification_contract["required_live_tool_names"] == ["bash", "submit_report"]
    assert qualification_contract["required_checks"] == [
        "exact_task_binding",
        "durable_create_claim_routes_deployed",
        "environment_started",
        "bash_reachable",
        "submit_report_reachable",
        "verifier_completed",
        "finite_authoritative_outcome",
        "metadata_only_session_ingested",
        "environment_cleanup_completed",
    ]
    assert contract["root_kind"] == "Job"
    assert contract["root_metadata_annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert contract["priority_class_name"] == "c1"
    assert contract["gpu_requests"] == contract["gpu_limits"] == 0
    assert contract["zero_gpu_must_hold_for_all_init_regular_and_ephemeral_containers"] is True
    assert contract["zero_gpu_resource_claims_forbidden"] is True
    assert contract["active_deadline_seconds_maximum"] == 1800
    assert contract["backoff_limit"] == 0
    assert contract["restart_policy"] == "Never"
    assert contract["server_preview_count"] == 2
    assert contract["normalized_server_previews_must_match"] is True
    assert contract["server_preview_rejects_unreviewed_container_fields"] is True
    assert contract["authorization_bound_create_journal"] is True
    assert contract["create_journal_ignores_only_an_unterminated_final_tail"] is True
    assert contract["uncertain_create_stable_observation_seconds_minimum"] == 10
    assert contract["exact_name_absence_checks"] == [
        "before_server_previews",
        "after_server_previews_before_create_intent",
    ]
    assert contract["create_intent"] == {
        "exclusive_create": True,
        "fsync_file_and_parent": True,
        "must_precede_create": True,
    }
    assert contract["create_attempts_maximum"] == 1
    assert contract["automatic_create_retry"] is False
    assert contract["uncertain_create_response"] == "reconcile_only_never_retry"
    assert contract["uncertain_partial_create"] == ("explicit_exact_uid_cleanup_then_final_absence")
    assert contract["packaged_source"] == {
        "code_in_immutable_config_map": True,
        "exact_file_bytes_attested": True,
        "merged_source_required": True,
        "private_plan_in_immutable_secret": True,
    }
    assert contract["create_response_must_bind"][-3:] == [
        "secret_name",
        "secret_uid",
        "secret_resource_version",
    ]
    assert contract["outer_kubernetes_cleanup"]["delete_preconditions"] == [
        "exact_name",
        "exact_uid",
        "exact_resource_version",
    ]
    assert contract["outer_kubernetes_cleanup"]["foreground_propagation"] is True
    assert (
        contract["outer_kubernetes_cleanup"][
            "cleanup_resumes_from_fsynced_intent_after_local_crash"
        ]
        is True
    )
    assert (
        contract["outer_kubernetes_cleanup"]["root_config_map_and_secret_absence_required"] is True
    )
    assert (
        contract["outer_kubernetes_cleanup"]["fresh_predelete_readback_must_match_exact_uid"]
        is True
    )
    assert contract["inner_fleet_cleanup"]["durable_create_request_claim_required"] is True
