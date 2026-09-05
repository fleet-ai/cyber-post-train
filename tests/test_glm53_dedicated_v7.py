from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v7 as v7
from evals.fleet import self_hosted

ROOT = Path.cwd()


def _seal(body: dict) -> dict:
    value = copy.deepcopy(body)
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _parity(value: dict, replica: str = "A") -> dict:
    payload = v7.server_payload(value, ROOT, replica)
    selected = value["replicas"][replica]
    api_run_id = "ft-run-abcdef12"
    ray_cluster_name = "ft-run-abcdef12-cluster"
    service_name = f"{ray_cluster_name}-head-svc"
    service_dns = f"{service_name}.{v7.NAMESPACE}.svc.cluster.local"
    probe_pod_name = f"{selected['probe_job_name']}-abcde"
    return _seal(
        {
            "schema_version": v7.PARITY_SCHEMA,
            "status": "PASSED",
            "replica": replica,
            "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
            "run_dir": value["replicas"][replica]["run_dir"],
            "image": v7.IMAGE,
            "resolved_image_id": v7.IMAGE,
            "model_revision": v7.MODEL_REVISION,
            "model_path": v7.MODEL_PATH,
            "served_id": "glm-5.3",
            "context_length": 262144,
            "priority_class": "fleet-infra-quiet",
            "preemption_policy": "Never",
            "gpus": 8,
            "pod_restarts": 0,
            "health_http_200": True,
            "service_port": 8000,
            "service_health_http_200": True,
            "server_arguments_sha256": v7.CANONICAL_SERVER_ARGUMENTS_SHA256,
            "rendered_command_sha256": self_hosted.sha256(payload["command"].encode()),
            "structured_tools": ["bash", "submit_report"],
            "tool_catalog_sha256": v7.TOOL_CATALOG_SHA,
            "content_blind_probe": True,
            "non_scored_model_requests": 2,
            "scored_sessions": 0,
            "concurrency_qualification": {
                "mode": "parallel_barrier",
                "requested_concurrency": 2,
                "maximum_observed_inflight": 2,
                "completed_requests": 2,
                "failed_requests": 0,
                "scored_requests": 0,
                "endpoint_traversal": {
                    "source_kind": "Pod",
                    "source_namespace": v7.NAMESPACE,
                    "source_name": probe_pod_name,
                    "via_service_dns": service_dns,
                    "scheme": "http",
                    "port": 8000,
                    "path": "/v1/chat/completions",
                    "direct_pod_bypass": False,
                },
            },
            "probe_resource_policy": {
                "cpu_only": True,
                "gpus": 0,
                "priority_class": "fleet-serve-low",
                "preemption_policy": "Never",
            },
            "api_run_id": api_run_id,
            "api_run_uid": "11111111-1111-4111-8111-111111111111",
            "kubernetes_namespace": v7.NAMESPACE,
            "ray_job_kind": "RayJob",
            "ray_job_name": api_run_id,
            "ray_job_uid": "22222222-2222-4222-8222-222222222222",
            "workload_kind": "Workload",
            "workload_name": "rayjob-ft-run-abcdef12-12345",
            "workload_uid": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "workload_owner_ray_job_uid": "22222222-2222-4222-8222-222222222222",
            "ray_cluster_kind": "RayCluster",
            "ray_cluster_name": ray_cluster_name,
            "ray_cluster_uid": "33333333-3333-4333-8333-333333333333",
            "ray_cluster_owner_ray_job_uid": "22222222-2222-4222-8222-222222222222",
            "head_pod_kind": "Pod",
            "head_pod_name": "ft-run-abcdef12-head-abcde",
            "head_pod_uid": "44444444-4444-4444-8444-444444444444",
            "head_pod_owner_ray_cluster_uid": "33333333-3333-4333-8333-333333333333",
            "service_kind": "Service",
            "service_name": service_name,
            "service_uid": "55555555-5555-4555-8555-555555555555",
            "service_owner_ray_cluster_uid": "33333333-3333-4333-8333-333333333333",
            "service_dns": service_dns,
            "service_selector": {
                "ray.io/cluster": ray_cluster_name,
                "ray.io/node-type": "head",
            },
            "service_target": {"port": 8000, "target_port": 8000, "protocol": "TCP"},
            "probe_job_kind": "Job",
            "probe_job_namespace": v7.NAMESPACE,
            "probe_job_name": selected["probe_job_name"],
            "probe_job_uid": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "probe_pod_kind": "Pod",
            "probe_pod_name": probe_pod_name,
            "probe_pod_uid": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            "probe_pod_owner_job_uid": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "endpoint_traversal": {
                "source_kind": "Pod",
                "source_namespace": v7.NAMESPACE,
                "source_name": probe_pod_name,
                "via_service_dns": service_dns,
                "scheme": "http",
                "port": 8000,
                "path": "/v1/chat/completions",
                "direct_pod_bypass": False,
            },
            "prompts_traces_flags_or_scores_included": False,
        }
    )


def test_held_spec_and_receipt_are_exact_and_create_nothing() -> None:
    value = v7.spec(ROOT)
    v7.validate_held(v7.load(ROOT / v7.HELD_PATH), ROOT)
    preview = v7.held_preview(ROOT)
    assert value["status"] == "HELD"
    assert value["launch_authorized"] is False
    assert preview["objects_created"] is False
    assert preview["jobs_api_priority_preview_passed"] is True


def test_generation7_binding_and_all_hosted_bulk_conflict_are_explicit() -> None:
    value = v7.spec(ROOT)
    assert value["generation7"] == v7.held_receipt(ROOT)["generation7_release"]
    assert value["bulk_handoff"] == {
        "current_generation7_bulk_commit": v7.PACKAGE_COMMIT,
        "current_generation7_bulk_serving_block": "glm-hosted-autocontinue-v1",
        "current_generation7_bulk_is_dedicated_compatible": False,
        "current_generation7_bulk_must_remain_unreleased_for_dedicated_path": True,
        "dedicated_aware_append_only_successor_required": True,
        "fresh_prebulk_reconciliation_required": True,
        "whole_task_partition_required": True,
        "pooling_across_serving_blocks_allowed": False,
    }
    changed = copy.deepcopy(value)
    changed["generation7"]["release_receipt_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="generation-7 binding"):
        v7.validate_spec(changed, ROOT)


def test_partition_is_399_disjoint_cells_at_whole_task_boundaries() -> None:
    controllers = v7.controller_specs(v7.spec(ROOT), ROOT)
    assert {key: row["new_session_count"] for key, row in controllers.items()} == {
        "hosted-1": 99,
        "hosted-2": 100,
        "dedicated-a-1": 51,
        "dedicated-a-2": 48,
        "dedicated-b-1": 51,
        "dedicated-b-2": 48,
    }
    canaries = [v7.dedicated_canary_cell(v7.spec(ROOT), ROOT, replica) for replica in ("A", "B")]
    cells = [cell for controller in controllers.values() for cell in controller["cells"]] + canaries
    assert len(cells) == len({row["cell_id"] for row in cells}) == 399
    owners: dict[int, set[str]] = {}
    for controller in controllers.values():
        for cell in controller["cells"]:
            owners.setdefault(cell["selection_rank"], set()).add(controller["serving_block"])
    assert all(len(blocks) == 1 for blocks in owners.values())
    assert [
        row["attempt"] for row in controllers["hosted-1"]["cells"] if row["selection_rank"] == 13
    ] == [2, 3, 4]
    assert v7.dedicated_canary_cell(v7.spec(ROOT), ROOT, "A")["selection_rank"] == 51
    assert v7.dedicated_canary_cell(v7.spec(ROOT), ROOT, "B")["selection_rank"] == 76
    assert all(
        plan["runtime_gate_required_before_canary"] is False
        for plan in (
            v7.dedicated_canary_plan(v7.spec(ROOT), ROOT, "A"),
            v7.dedicated_canary_plan(v7.spec(ROOT), ROOT, "B"),
        )
    )


def test_server_argv_is_exact_and_rejects_duplicate_or_trailing_overrides() -> None:
    value = v7.spec(ROOT)
    assert value["server_arguments"] == list(v7.CANONICAL_SERVER_ARGUMENTS)
    assert value["server_arguments_sha256"] == v7.CANONICAL_SERVER_ARGUMENTS_SHA256
    for extra in (["--context-length", "1"], ["--disable-radix-cache"]):
        changed = copy.deepcopy(value)
        changed["server_arguments"].extend(extra)
        with pytest.raises(ValueError, match="canonical server argv"):
            v7.validate_spec(changed, ROOT)


def test_server_requests_hold_exact_runtime_and_requested_scheduler_policy() -> None:
    value = v7.spec(ROOT)
    payloads = {replica: v7.server_payload(value, ROOT, replica) for replica in ("A", "B")}
    assert all(set(payload) == v7.EXPECTED_API_FIELDS for payload in payloads.values())
    assert all(payload["image"] == v7.IMAGE for payload in payloads.values())
    assert all(payload["gpus_per_worker"] == 8 for payload in payloads.values())
    assert all(payload["priority_class"] == "fleet-infra-quiet" for payload in payloads.values())
    assert all("IDLE_SECONDS=600" in payload["command"] for payload in payloads.values())
    assert payloads["A"]["run_dir"] != payloads["B"]["run_dir"]


def test_preview_and_duplicate_gates_fail_closed() -> None:
    value = v7.spec(ROOT)
    payload = v7.server_payload(value, ROOT, "A")
    preview = _seal(
        {
            "schema_version": v7.PREVIEW_SCHEMA,
            "status": "PASSED",
            "replica": "A",
            "authenticated": True,
            "base_url": "https://api.ft.flt.build",
            "route": "/v1/runs/preview",
            "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
            "requested_image": v7.IMAGE,
            "rendered_image": v7.IMAGE,
            "rendered_image_pull_secrets": ["ghcr-pull"],
            "rendered_workers": 1,
            "rendered_gpus_per_worker": 8,
            "rendered_priority_class": "fleet-infra-quiet",
            "rendered_preemption_policy": "Never",
            "rendered_run_dir": value["replicas"]["A"]["run_dir"],
            "post_put_patch_delete": 0,
            "credentials_included": False,
        }
    )
    v7.validate_authenticated_preview(preview, value, ROOT, "A")
    invalid = copy.deepcopy(preview)
    invalid["rendered_priority_class"] = "fleet-train-high"
    invalid = _seal({k: item for k, item in invalid.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError, match="authenticated preview"):
        v7.validate_authenticated_preview(invalid, value, ROOT, "A")

    cells = [
        cell
        for name, controller in v7.controller_specs(value, ROOT).items()
        if name.startswith("dedicated-a-")
        for cell in controller["cells"]
    ]
    cells.append(v7.dedicated_canary_plan(value, ROOT, "A")["cell"])
    duplicate = _seal(
        {
            "schema_version": v7.DUPLICATE_SCHEMA,
            "status": "PASSED",
            "replica": "A",
            "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
            "title": value["replicas"]["A"]["title"],
            "run_dir": value["replicas"]["A"]["run_dir"],
            "jobs_api_title_matches": 0,
            "kubernetes_label_or_name_matches": 0,
            "sfs_run_dir_exists": False,
            "checked_sources": [
                "fleet_sessions",
                "kubernetes_jobs_and_pods",
                "sfs_attempt_outputs",
                "global_cell_claims",
                "model_started_receipts",
            ],
            "prebulk_reconciliation_receipt_sha256": "sha256:" + "e" * 64,
            "assigned_cell_count": 100,
            "assigned_cell_ids_sha256": self_hosted.sha256(
                self_hosted.canonical_json(sorted(cell["cell_id"] for cell in cells))
            ),
            "accepted_active_claimed_or_model_started_cell_collisions": 0,
            "checked_immediately_before_create": True,
            "post_put_patch_delete": 0,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
    )
    v7.validate_duplicate_gate(duplicate, value, ROOT, "A")
    invalid_duplicate = copy.deepcopy(duplicate)
    invalid_duplicate["jobs_api_title_matches"] = 1
    invalid_duplicate = _seal(
        {k: item for k, item in invalid_duplicate.items() if k != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="create-once"):
        v7.validate_duplicate_gate(invalid_duplicate, value, ROOT, "A")


def test_uid_parity_claim_session_verifier_ingest_and_cleanup_gate_stream_two() -> None:
    value = v7.spec(ROOT)
    parity = _parity(value)
    v7.validate_parity_gate(parity, value, ROOT, "A")
    cell = v7.dedicated_canary_cell(value, ROOT, "A")
    canary = _seal(
        {
            "schema_version": v7.CANARY_SCHEMA,
            "accepted": True,
            "credited": True,
            "retry_allowed": False,
            "replica": "A",
            "serving_block": value["replicas"]["A"]["serving_block"],
            **cell,
            "run_id": "chris-cyber-glm53-dedicated-a-v7-r051-a1-test",
            "session_id": "66666666-6666-4666-8666-666666666666",
            "verifier_execution_id": "77777777-7777-4777-8777-777777777777",
            "claim_execution_id": cell["execution_id"],
            "claim_path": (
                f"{v7.bulk_v3.CLAIM_ROOT}/"
                f"{cell['execution_id'].removeprefix('sha256:')}.json"
            ),
            "claim_job_kind": "Job",
            "claim_job_namespace": v7.NAMESPACE,
            "claim_job_name": "chris-cyber-exact100-glm-dedicated-a-canary-v4",
            "claim_job_uid": "88888888-8888-4888-8888-888888888888",
            "claim_pod_kind": "Pod",
            "claim_pod_name": "chris-cyber-glm53-dedicated-a-v7-r051-a1-abcde",
            "claim_pod_uid": "99999999-9999-4999-8999-999999999999",
            "claim_pod_owner_job_uid": "88888888-8888-4888-8888-888888888888",
            "claim_receipt_sha256": "sha256:" + "c" * 64,
            "source_binding_valid": True,
            "session_model": v7.exact.EXPECTED_MODELS["glm-5.3"]["session_model"],
            "model_revision": v7.MODEL_REVISION,
            "authoritative_metadata_run_id": "chris-cyber-glm53-dedicated-a-v7-r051-a1-test",
            "serving_parity_receipt_sha256": parity["receipt_sha256"],
            "serving_runtime": v7._parity_runtime_identity(parity, value, "A"),
            "authoritative_session_id": "66666666-6666-4666-8666-666666666666",
            "authoritative_verifier_execution_id": "77777777-7777-4777-8777-777777777777",
            "authoritative_session_matches": 1,
            "agent_exit_code": 0,
            "agent_process_exit_success": True,
            "session_ingest_completed": True,
            "cleanup_completed": True,
            "cleanup_scope": "exact_session_and_task_environment",
            "scores_included": False,
            "prompts_traces_flags_or_scores_included": False,
        }
    )
    v7.validate_canary_acceptance(canary, parity, value, ROOT, "A")
    runtime = _seal(
        {
            "schema_version": v7.RUNTIME_SCHEMA,
            "status": "PASSED",
            "replica": "A",
            "parity_receipt_sha256": parity["receipt_sha256"],
            "serving_runtime": v7._parity_runtime_identity(parity, value, "A"),
            "scored_canary_accepted": True,
            "scored_canary_cell_id": canary["cell_id"],
            "scored_canary_execution_id": canary["execution_id"],
            "scored_canary_claim_path": canary["claim_path"],
            "scored_canary_claim_receipt_sha256": canary["claim_receipt_sha256"],
            "scored_canary_run_id": canary["run_id"],
            "scored_canary_session_id": canary["session_id"],
            "scored_canary_verifier_execution_id": canary["verifier_execution_id"],
            "scored_canary_authoritative_session_matches": 1,
            "scored_canary_session_ingest_completed": True,
            "scored_canary_cleanup_completed": True,
            "scored_canary_receipt_sha256": canary["receipt_sha256"],
            "dependent_bulk_release_authorized": True,
            "qualified_controller_concurrency": 2,
            "released_streams": ["dedicated-a-1", "dedicated-a-2"],
            "maximum_streams_released": 2,
            "prompts_traces_flags_or_scores_included": False,
        }
    )
    v7.validate_runtime_gate(runtime, parity, canary, value, ROOT, "A")
    broken = copy.deepcopy(canary)
    broken["cleanup_completed"] = False
    broken = _seal({k: item for k, item in broken.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError, match="canary acceptance"):
        v7.validate_canary_acceptance(broken, parity, value, ROOT, "A")
    broken_parity = copy.deepcopy(parity)
    broken_parity["service_name"] = "other-service"
    broken_parity = _seal(
        {k: item for k, item in broken_parity.items() if k != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="Service identity"):
        v7.validate_parity_gate(broken_parity, value, ROOT, "A")
    underqualified = copy.deepcopy(parity)
    underqualified["concurrency_qualification"]["maximum_observed_inflight"] = 1
    underqualified = _seal(
        {k: item for k, item in underqualified.items() if k != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="parity gate"):
        v7.validate_parity_gate(underqualified, value, ROOT, "A")


def test_shell_is_syntax_valid_and_submit_requires_release_evidence() -> None:
    script = ROOT / v7.SUBMIT_PATH
    subprocess.run(["bash", "-n", str(script)], check=True)
    source = script.read_text()
    assert 'client.get("/v1/runs", params={"limit": 200, "offset": offset})' in source
    assert 'client.get(f"/v1/runs/' not in source
    assert 'value.get("uid")' not in source
    completed = subprocess.run(
        [str(script), "submit"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert completed.returncode != 0
