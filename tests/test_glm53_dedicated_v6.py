from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v6 as v6
from evals.fleet import self_hosted

ROOT = Path.cwd()


def _seal(body: dict) -> dict:
    receipt = copy.deepcopy(body)
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    return receipt


def test_held_spec_and_receipt_are_exact() -> None:
    value = v6.spec(ROOT)
    v6.validate_held(v6.load(ROOT / v6.HELD_PATH), ROOT)
    assert value["status"] == "HELD"
    assert value["launch_authorized"] is False


def test_server_payloads_use_current_jobs_api_shape_and_proven_runtime() -> None:
    value = v6.spec(ROOT)
    payloads = {replica: v6.server_payload(value, ROOT, replica) for replica in ("A", "B")}
    assert all(set(payload) == v6.EXPECTED_API_FIELDS for payload in payloads.values())
    assert all("image_pull_secrets" not in payload for payload in payloads.values())
    assert all(payload["image"] == v6.IMAGE for payload in payloads.values())
    assert all(payload["workers"] == 1 for payload in payloads.values())
    assert all(payload["gpus_per_worker"] == 8 for payload in payloads.values())
    assert all(payload["priority_class"] == "fleet-serve-low" for payload in payloads.values())
    assert all("IDLE_SECONDS=600" in payload["command"] for payload in payloads.values())
    assert payloads["A"]["run_dir"] != payloads["B"]["run_dir"]


def test_controller_partition_is_whole_task_complete_and_two_stream_bounded() -> None:
    value = v6.spec(ROOT)
    controllers = v6.controller_specs(value, ROOT)
    assert {key: row["new_session_count"] for key, row in controllers.items()} == {
        "hosted-1": 99,
        "hosted-2": 100,
        "dedicated-a-1": 52,
        "dedicated-a-2": 48,
        "dedicated-b-1": 52,
        "dedicated-b-2": 48,
    }
    assert sum(row["new_session_count"] for row in controllers.values()) == 399
    assert all(row["maximum_endpoint_streams"] == 2 for row in controllers.values())

    rank_blocks: dict[int, set[str]] = {}
    cell_ids: list[str] = []
    for row in controllers.values():
        for cell in row["cells"]:
            rank_blocks.setdefault(cell["selection_rank"], set()).add(row["serving_block"])
            cell_ids.append(cell["cell_id"])
    assert all(len(blocks) == 1 for blocks in rank_blocks.values())
    assert len(cell_ids) == len(set(cell_ids)) == 399
    assert [
        row["attempt"] for row in controllers["hosted-1"]["cells"] if row["selection_rank"] == 13
    ] == [2, 3, 4]


def test_treatment_blocks_cannot_be_silently_reassigned() -> None:
    value = v6.spec(ROOT)
    changed = copy.deepcopy(value)
    changed["partition"]["blocks"][2]["serving_block"] = "glm-hosted-autocontinue-v1"
    with pytest.raises(ValueError, match="block rank or treatment"):
        v6.validate_spec(changed, ROOT)


def test_authenticated_preview_requires_exact_private_pull_rendering() -> None:
    value = v6.spec(ROOT)
    payload = v6.server_payload(value, ROOT, "A")
    body = {
        "schema_version": v6.PREVIEW_SCHEMA,
        "status": "PASSED",
        "replica": "A",
        "authenticated": True,
        "route": "/v1/runs/preview",
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "requested_image": v6.IMAGE,
        "rendered_image": v6.IMAGE,
        "rendered_image_pull_secrets": ["ghcr-pull"],
        "rendered_workers": 1,
        "rendered_gpus_per_worker": 8,
        "rendered_priority_class": "fleet-serve-low",
        "rendered_preemption_policy": "Never",
        "rendered_run_dir": value["replicas"]["A"]["run_dir"],
        "post_put_patch_delete": 0,
        "credentials_included": False,
    }
    receipt = _seal(body)
    v6.validate_authenticated_preview(receipt, value, ROOT, "A")
    changed = copy.deepcopy(receipt)
    changed["rendered_image_pull_secrets"] = ["ecr-pull"]
    changed = _seal({key: item for key, item in changed.items() if key != "receipt_sha256"})
    with pytest.raises(ValueError, match="authenticated preview gate"):
        v6.validate_authenticated_preview(changed, value, ROOT, "A")


def test_create_once_gate_requires_zero_names_and_absent_sfs_root() -> None:
    value = v6.spec(ROOT)
    payload = v6.server_payload(value, ROOT, "A")
    body = {
        "schema_version": v6.DUPLICATE_SCHEMA,
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
            self_hosted.canonical_json(
                sorted(
                    cell["cell_id"]
                    for name, controller in v6.controller_specs(value, ROOT).items()
                    if name.startswith("dedicated-a-")
                    for cell in controller["cells"]
                )
            )
        ),
        "accepted_active_claimed_or_model_started_cell_collisions": 0,
        "checked_immediately_before_create": True,
        "post_put_patch_delete": 0,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    v6.validate_duplicate_gate(_seal(body), value, ROOT, "A")
    changed = copy.deepcopy(body)
    changed["jobs_api_title_matches"] = 1
    with pytest.raises(ValueError, match="create-once gate"):
        v6.validate_duplicate_gate(_seal(changed), value, ROOT, "A")


def test_uid_bound_parity_then_runtime_gate_requires_scored_canary() -> None:
    value = v6.spec(ROOT)
    payload = v6.server_payload(value, ROOT, "A")
    body = {
        "schema_version": v6.PARITY_SCHEMA,
        "status": "PASSED",
        "replica": "A",
        "request_sha256": self_hosted.sha256(self_hosted.canonical_json(payload)),
        "run_dir": value["replicas"]["A"]["run_dir"],
        "image": v6.IMAGE,
        "resolved_image_id": v6.IMAGE,
        "model_revision": v6.MODEL_REVISION,
        "model_path": v6.MODEL_PATH,
        "served_id": "glm-5.3",
        "context_length": 262144,
        "priority_class": "fleet-serve-low",
        "preemption_policy": "Never",
        "gpus": 8,
        "pod_restarts": 0,
        "health_http_200": True,
        "service_port": 8000,
        "service_health_http_200": True,
        "server_arguments_sha256": self_hosted.sha256(" ".join(value["server_arguments"]).encode()),
        "rendered_command_sha256": self_hosted.sha256(payload["command"].encode()),
        "structured_tools": ["bash", "submit_report"],
        "tool_catalog_sha256": v6.TOOL_CATALOG_SHA,
        "content_blind_probe": True,
        "non_scored_model_requests": 1,
        "scored_sessions": 0,
        "api_run_uid": "11111111-1111-4111-8111-111111111111",
        "ray_job_uid": "22222222-2222-4222-8222-222222222222",
        "ray_cluster_uid": "33333333-3333-4333-8333-333333333333",
        "head_pod_uid": "44444444-4444-4444-8444-444444444444",
        "service_uid": "55555555-5555-4555-8555-555555555555",
        "prompts_traces_flags_or_scores_included": False,
    }
    parity = _seal(body)
    v6.validate_parity_gate(parity, value, ROOT, "A")
    cell = v6.dedicated_canary_cell(value, ROOT, "A")
    canary_body = {
        "schema_version": v6.CANARY_SCHEMA,
        "accepted": True,
        "credited": True,
        "retry_allowed": False,
        "replica": "A",
        "serving_block": value["replicas"]["A"]["serving_block"],
        **cell,
        "run_id": "chris-cyber-glm53-dedicated-a-v6-r051-a1-test",
        "session_id": "66666666-6666-4666-8666-666666666666",
        "verifier_execution_id": "77777777-7777-4777-8777-777777777777",
        "claim_receipt_sha256": "sha256:" + "c" * 64,
        "session_model": v6.exact.EXPECTED_MODELS["glm-5.3"]["session_model"],
        "model_revision": v6.MODEL_REVISION,
        "authoritative_metadata_run_id": ("chris-cyber-glm53-dedicated-a-v6-r051-a1-test"),
        **{
            field: parity[field]
            for field in (
                "api_run_uid",
                "ray_job_uid",
                "ray_cluster_uid",
                "head_pod_uid",
                "service_uid",
            )
        },
        "authoritative_session_matches": 1,
        "agent_exit_code": 0,
        "agent_process_exit_success": True,
        "session_ingest_completed": True,
        "cleanup_completed": True,
        "scores_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    canary = _seal(canary_body)
    v6.validate_canary_acceptance(canary, parity, value, ROOT, "A")
    runtime_body = {
        "schema_version": v6.RUNTIME_SCHEMA,
        "status": "PASSED",
        "replica": "A",
        "parity_receipt_sha256": parity["receipt_sha256"],
        **{
            field: parity[field]
            for field in (
                "api_run_uid",
                "ray_job_uid",
                "ray_cluster_uid",
                "head_pod_uid",
                "service_uid",
            )
        },
        "scored_canary_accepted": True,
        "scored_canary_cell_id": canary["cell_id"],
        "scored_canary_session_id": canary["session_id"],
        "scored_canary_verifier_execution_id": canary["verifier_execution_id"],
        "scored_canary_receipt_sha256": canary["receipt_sha256"],
        "maximum_streams_released": 2,
        "prompts_traces_flags_or_scores_included": False,
    }
    runtime = _seal(runtime_body)
    v6.validate_runtime_gate(runtime, parity, canary, value, ROOT, "A")
    changed = copy.deepcopy(runtime_body)
    changed["scored_canary_accepted"] = False
    with pytest.raises(ValueError, match="runtime gate"):
        v6.validate_runtime_gate(_seal(changed), parity, canary, value, ROOT, "A")

    wrong_canary_body = copy.deepcopy(canary_body)
    wrong_canary_body["cell_id"] = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="canary acceptance"):
        v6.validate_canary_acceptance(_seal(wrong_canary_body), parity, value, ROOT, "A")


def test_lifecycle_and_submit_shells_are_syntax_valid_and_held() -> None:
    lifecycle = ROOT / v6.LIFECYCLE_PATH
    heartbeat = ROOT / v6.HEARTBEAT_PATH
    submit = ROOT / "evals/fleet/scripts/submit_glm53_dedicated_v6.sh"
    subprocess.run(["bash", "-n", str(lifecycle)], check=True)
    subprocess.run(["bash", "-n", str(heartbeat)], check=True)
    subprocess.run(["bash", "-n", str(submit)], check=True)
    lifecycle_text = lifecycle.read_text()
    assert "IDLE_SECONDS=600" in lifecycle_text
    assert "if (( modified > now ))" in lifecycle_text
    assert "modified=$now" in lifecycle_text
    completed = subprocess.run(
        [str(submit), "submit"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 78
    assert "HELD" in completed.stderr
