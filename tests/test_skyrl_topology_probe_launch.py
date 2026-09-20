"""Offline tests for evidence-bound SkyRL topology-probe launch authority."""

from __future__ import annotations

import copy

import pytest

from cyber_post_train.jobs import digest
from training import skyrl_topology_probe as probe
from training import skyrl_topology_probe_launch as launch


def _evidence():
    plan = probe.compile_probe(probe.CONFIG_PATH)
    plan_sha = digest(plan)
    cpu_manifest_sha = digest(probe.preflight_job_manifest(plan))
    fleet_manifest_sha = digest(probe.fleetjob_manifest(plan))
    cpu_receipt = probe._seal(
        {
            "schema": launch.CPU_RECEIPT_SCHEMA,
            "status": "passed",
            "gpus": 0,
            "runtime_user": {"uid": 1000, "gid": 100},
            "plan_sha256": plan_sha,
            "request_sha256": digest(
                probe.request(plan, fleetjob_transport=True, cpu_preflight=True)
            ),
            "fleetjob_manifest_sha256": fleet_manifest_sha,
            "preflight_job_manifest_sha256": cpu_manifest_sha,
            "task_rows_read": 0,
            "rollout_episodes": 0,
            "optimizer_steps": 0,
            "create_once_output_absent": True,
        }
    )
    cpu_result = launch._seal(
        {
            "schema": launch.OBSERVER_RESULT_SCHEMA,
            "status": "released",
            "context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "kind": "job",
            "name": probe.PREFLIGHT_NAME,
            "plan_sha256": "sha256:" + plan_sha,
            "manifest_sha256": "sha256:" + cpu_manifest_sha,
            "expected_gpus": 0,
            "uid": "00000000-0000-0000-0000-000000000001",
            "created_at": "2026-09-20T00:00:00Z",
            "terminal_status": "Succeeded",
            "pod_uids": ["00000000-0000-0000-0000-000000000002"],
            "image_ids": [probe.IMAGE],
            "exit_codes": [0],
            "restarts": 0,
            "peak_gpus": 0,
            "receipt": cpu_receipt,
            "deletion_requested_at": "2026-09-20T00:01:00Z",
            "target_present": False,
            "pods_present": False,
            "active_gpus": 0,
            "release_observed_at": "2026-09-20T00:02:00Z",
        }
    )
    cpu_preview = probe._seal(
        {
            "schema": probe.PREFLIGHT_PREVIEW_SCHEMA,
            "status": "passed",
            "plan_sha256": plan_sha,
            "manifest_sha256": cpu_manifest_sha,
            "server_render_sha256": "1" * 64,
            "kubernetes_context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "name": probe.PREFLIGHT_NAME,
            "gpu_nodes": 0,
            "gpus": 0,
            "runtime_user": {"uid": 1000, "gid": 100},
            "submitted": False,
        }
    )
    fleetjob_preview = probe._seal(
        {
            "schema": probe.FLEETJOB_PREVIEW_SCHEMA,
            "status": "passed",
            "plan_sha256": plan_sha,
            "manifest_sha256": fleet_manifest_sha,
            "server_render_sha256": "2" * 64,
            "kubernetes_context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "name": plan["run_name"],
            "gpu_nodes": 1,
            "gpus": 8,
            "runtime_user": {"uid": 1000, "gid": 100},
            "submitted": False,
        }
    )
    observer = launch._seal(
        {
            "schema": launch.OBSERVER_ARMED_SCHEMA,
            "status": "armed",
            "context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "kind": "fleetjob",
            "name": plan["run_name"],
            "maximum_seconds": 1800,
            "expected_gpus": 8,
            "plan_sha256": "sha256:" + plan_sha,
            "manifest_sha256": "sha256:" + fleet_manifest_sha,
            "armed_at": "2026-09-20T00:03:00Z",
            "observer_pid": 12345,
        }
    )
    return plan, cpu_result, cpu_preview, fleetjob_preview, observer


def test_launch_authorization_binds_passed_cpu_release_previews_and_observer() -> None:
    plan, cpu, cpu_preview, fleet_preview, observer = _evidence()
    value = launch.authorize(
        plan,
        cpu_result=cpu,
        cpu_preview=cpu_preview,
        fleetjob_preview=fleet_preview,
        fleetjob_observer=observer,
    )
    assert value["status"] == "authorized_for_one_create"
    assert value["plan_sha256"] == digest(plan)
    launch.validate(plan, value)


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("cpu_result", "status"), "released_without_accepted_execution"),
        (("cpu_result", "active_gpus"), 8),
        (("cpu_result", "receipt", "optimizer_steps"), 1),
        (("fleetjob_preview", "gpus"), 7),
        (("fleetjob_observer", "expected_gpus"), 0),
        (("fleetjob_observer", "name"), "other"),
    ],
)
def test_launch_authorization_rejects_substituted_evidence(path, replacement) -> None:
    plan, cpu, cpu_preview, fleet_preview, observer = _evidence()
    evidence = {
        "cpu_result": copy.deepcopy(cpu),
        "cpu_preview": copy.deepcopy(cpu_preview),
        "fleetjob_preview": copy.deepcopy(fleet_preview),
        "fleetjob_observer": copy.deepcopy(observer),
    }
    target = evidence
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    target.pop("sha256", None)
    if path[0] in {"cpu_result", "fleetjob_observer"}:
        evidence[path[0]] = launch._seal(evidence[path[0]])
    with pytest.raises(ValueError):
        launch.authorize(plan, **evidence)
