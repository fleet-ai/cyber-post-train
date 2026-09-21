"""Direct root-RayJob transport tests for the sealed SkyRL topology probe."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cyber_post_train.jobs import JobsError, digest
from training import dev_cleanup_observer as cleanup
from training import skyrl_topology_probe as probe
from training import skyrl_topology_rayjob as direct

CONFIG = Path("configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json")


@pytest.fixture
def plan() -> dict:
    return probe.compile_probe(CONFIG)


def _rendered(plan: dict) -> dict:
    value = copy.deepcopy(direct.manifest(plan))
    value["metadata"].update(
        {
            "creationTimestamp": "2026-09-20T00:00:00Z",
            "generation": 1,
            "uid": "00000000-0000-0000-0000-000000000001",
        }
    )
    value["spec"].update({"submissionMode": "K8sJobMode", "ttlSecondsAfterFinished": 0})
    value["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["numOfHosts"] = 1
    return value


def _preview(plan: dict) -> dict:
    return direct.validate_preview(plan, direct.manifest(plan), _rendered(plan))


def _cpu_result(plan: dict) -> dict:
    receipt = probe._seal(
        {
            "schema": "cyber_skyrl_topology_probe_cpu_preflight_v1",
            "status": "passed",
            "gpus": 0,
            "runtime_user": {"uid": 1000, "gid": 100},
            "plan_sha256": digest(plan),
            "request_sha256": digest(
                probe.request(plan, fleetjob_transport=True, cpu_preflight=True)
            ),
            "fleetjob_manifest_sha256": digest(probe.fleetjob_manifest(plan)),
            "preflight_job_manifest_sha256": digest(probe.preflight_job_manifest(plan)),
            "task_rows_read": 0,
            "rollout_episodes": 0,
            "optimizer_steps": 0,
            "create_once_output_absent": True,
            "vllm_sampler_environment": probe.VLLM_SAMPLER_ENV,
        }
    )
    return cleanup._seal(
        {
            "schema": cleanup.RESULT_SCHEMA,
            "status": "released",
            "context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "kind": "job",
            "name": probe.PREFLIGHT_NAME,
            "plan_sha256": "sha256:" + digest(plan),
            "manifest_sha256": "sha256:" + digest(probe.preflight_job_manifest(plan)),
            "expected_gpus": 0,
            "maximum_seconds": 1200,
            "armed_at": "2026-09-20T00:00:00Z",
            "uid": "00000000-0000-0000-0000-000000000002",
            "created_at": "2026-09-20T00:00:01Z",
            "terminal_status": "Succeeded",
            "job_id": "",
            "rayjob_name": "",
            "rayjob_uid": "",
            "workload_name": "",
            "workload_uid": "",
            "raycluster_name": "",
            "raycluster_uid": "",
            "pod_names": ["preflight-pod"],
            "pod_uids": ["00000000-0000-0000-0000-000000000003"],
            "image_ids": [probe.IMAGE],
            "exit_codes": [0],
            "termination_reasons": ["Completed"],
            "restarts": 0,
            "peak_gpus": 0,
            "receipt": receipt,
            "observer_error_class": "",
            "deletion_requested_at": "2026-09-20T00:01:00Z",
            "target_present": False,
            "pods_present": False,
            "rayjob_present": False,
            "workload_present": False,
            "raycluster_present": False,
            "active_gpus": 0,
            "release_observed_at": "2026-09-20T00:01:01Z",
        }
    )


def _cpu_preview(plan: dict) -> dict:
    return probe._seal(
        {
            "schema": probe.PREFLIGHT_PREVIEW_SCHEMA,
            "status": "passed",
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(probe.preflight_job_manifest(plan)),
            "server_render_sha256": "0" * 64,
            "kubernetes_context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "name": probe.PREFLIGHT_NAME,
            "gpu_nodes": 0,
            "gpus": 0,
            "runtime_user": {"uid": 1000, "gid": 100},
            "submitted": False,
        }
    )


def _receipt_preview(plan: dict) -> dict:
    return probe._seal(
        {
            "schema": probe.RECEIPT_VERIFY_PREVIEW_SCHEMA,
            "status": "passed",
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(probe.receipt_verify_job_manifest(plan)),
            "server_render_sha256": "1" * 64,
            "kubernetes_context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "name": probe.RECEIPT_VERIFY_NAME,
            "gpu_nodes": 0,
            "gpus": 0,
            "runtime_user": {"uid": 1000, "gid": 100},
            "submitted": False,
        }
    )


def _observer(plan: dict) -> dict:
    return cleanup._seal(
        {
            "schema": cleanup.ARMED_SCHEMA,
            "status": "armed",
            "context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "kind": "rayjob",
            "name": plan["run_name"],
            "maximum_seconds": 1800,
            "expected_gpus": 8,
            "plan_sha256": "sha256:" + digest(plan),
            "manifest_sha256": "sha256:" + digest(direct.manifest(plan)),
            "armed_at": "2026-09-20T00:00:00Z",
            "observer_pid": os.getpid(),
        }
    )


def _authorization(plan: dict) -> dict:
    return direct.authorize(
        plan,
        cpu_result=_cpu_result(plan),
        cpu_preview=_cpu_preview(plan),
        receipt_preview=_receipt_preview(plan),
        rayjob_preview=_preview(plan),
        observer=_observer(plan),
    )


def test_direct_manifest_is_exact_v17_projection_with_alert_safe_root(plan) -> None:
    source = probe.fleetjob_manifest(plan)["spec"]["job"]
    value = direct.manifest(plan)
    assert value["apiVersion"] == source["apiVersion"] == "ray.io/v1"
    assert value["kind"] == source["kind"] == "RayJob"
    assert value["spec"]["entrypoint"] == source["spec"]["entrypoint"]
    assert value["spec"]["activeDeadlineSeconds"] == source["spec"]["activeDeadlineSeconds"]
    assert value["spec"]["backoffLimit"] == source["spec"]["backoffLimit"] == 0
    assert value["spec"]["shutdownAfterJobFinishes"] is True
    assert value["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert value["metadata"]["annotations"]["kueue.x-k8s.io/elastic-job"] == "true"
    assert value["metadata"]["annotations"]["fleet.ai/run-dir"] == plan["output_root"]
    assert value["metadata"]["labels"] == {
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q1",
    }
    cluster = value["spec"]["rayClusterSpec"]
    assert cluster["headGroupSpec"]["template"]["spec"]["priorityClassName"] == "c1"
    assert (
        cluster["headGroupSpec"]["template"]["spec"]["containers"][0]["resources"]["requests"][
            "nvidia.com/gpu"
        ]
        == "8"
    )
    assert cluster["workerGroupSpecs"][0]["replicas"] == 0
    assert cluster["workerGroupSpecs"][0]["maxReplicas"] == 1
    assert value["spec"]["suspend"] is True
    assert cluster["headGroupSpec"]["template"]["spec"]["initContainers"][0]["name"] == (
        "create-once-output"
    )
    source_groups = [
        source["spec"]["rayClusterSpec"]["headGroupSpec"],
        source["spec"]["rayClusterSpec"]["workerGroupSpecs"][0],
    ]
    direct_groups = [cluster["headGroupSpec"], cluster["workerGroupSpecs"][0]]
    for source_group, direct_group in zip(source_groups, direct_groups, strict=True):
        source_container = source_group["template"]["spec"]["containers"][0]
        direct_container = direct_group["template"]["spec"]["containers"][0]
        assert direct_container["image"] == source_container["image"]
        assert direct_container["resources"] == source_container["resources"]
        assert direct_container["securityContext"] == source_container["securityContext"]
        direct_env = {row["name"]: row for row in direct_container["env"]}
        assert all(direct_env[row["name"]] == row for row in source_container["env"])


def test_direct_preview_rejects_alert_priority_and_topology_drift(plan) -> None:
    expected = direct.manifest(plan)
    proof = direct.validate_preview(plan, expected, _rendered(plan))
    assert proof["failure_alerts"] == "off"
    assert proof["gpus"] == 8 and proof["gpu_nodes"] == 1
    for mutate in (
        lambda value: value["metadata"]["annotations"].pop("fleet.ai/failure-alerts"),
        lambda value: value["metadata"]["labels"].update({"kueue.x-k8s.io/priority-class": "q0"}),
        lambda value: value["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"].update(
            {"priorityClassName": "c0"}
        ),
        lambda value: value["spec"].update({"ttlSecondsAfterFinished": 10}),
        lambda value: value["spec"]["rayClusterSpec"]["workerGroupSpecs"][0].update(
            {"numOfHosts": 2}
        ),
    ):
        changed = _rendered(plan)
        mutate(changed)
        with pytest.raises(JobsError, match="server dry-run"):
            direct.validate_preview(plan, expected, changed)


def test_direct_authorization_binds_cpu_preview_observer_and_manifest(plan) -> None:
    value = _authorization(plan)
    direct.validate_authorization(plan, value)
    assert value["status"] == "authorized_for_one_create"
    assert value["manifest_sha256"] == digest(direct.manifest(plan))
    changed = copy.deepcopy(value)
    changed["observer"]["kind"] = "fleetjob"
    changed["observer"] = cleanup._seal(changed["observer"])
    with pytest.raises(JobsError, match="observer"):
        direct.validate_authorization(plan, changed)


def test_direct_release_requires_success_exact_uids_and_zero_gpus(plan) -> None:
    receipt = probe._seal(
        {
            "schema": probe.RECEIPT_SCHEMA,
            "status": "setup_and_internal_cleanup_passed",
            "plan_sha256": digest(plan),
            **plan["scientific_work"],
            "engines_started": 2,
            "tensor_parallel_size": 4,
            "runtime_users": {
                "driver": {"uid": 1000, "gid": 100, "physical_node": "gpu-node-1"},
                "gpu_pods": [{"uid": 1000, "gid": 100, "physical_node": "gpu-node-1"}],
            },
            "ray_shutdown_called": True,
            "external_release_required": True,
        }
    )
    observation = cleanup._seal(
        {
            "schema": cleanup.RESULT_SCHEMA,
            "status": "released",
            "context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "kind": "rayjob",
            "name": plan["run_name"],
            "plan_sha256": "sha256:" + digest(plan),
            "manifest_sha256": "sha256:" + digest(direct.manifest(plan)),
            "expected_gpus": 8,
            "maximum_seconds": 1800,
            "armed_at": "2026-09-20T00:00:00Z",
            "uid": "00000000-0000-0000-0000-000000000030",
            "created_at": "2026-09-20T00:00:01Z",
            "terminal_status": "Succeeded",
            "job_id": "",
            "rayjob_name": plan["run_name"],
            "rayjob_uid": "00000000-0000-0000-0000-000000000030",
            "workload_name": "workload-probe",
            "workload_uid": "00000000-0000-0000-0000-000000000031",
            "raycluster_name": "cluster-probe",
            "raycluster_uid": "00000000-0000-0000-0000-000000000032",
            "pod_names": ["probe-pod"],
            "pod_uids": ["00000000-0000-0000-0000-000000000033"],
            "image_ids": [probe.IMAGE],
            "exit_codes": [0],
            "termination_reasons": ["Completed"],
            "restarts": 0,
            "peak_gpus": 8,
            "receipt": receipt,
            "observer_error_class": "",
            "deletion_requested_at": "2026-09-20T00:20:00Z",
            "target_present": False,
            "pods_present": False,
            "rayjob_present": False,
            "workload_present": False,
            "raycluster_present": False,
            "active_gpus": 0,
            "release_observed_at": "2026-09-20T00:20:01Z",
        }
    )
    release = direct.validate_release(plan, receipt, observation)
    assert release["status"] == "released" and release["active_gpus"] == 0
    changed = copy.deepcopy(observation)
    changed["active_gpus"] = 8
    changed = cleanup._seal(changed)
    with pytest.raises(JobsError, match="release was not proven"):
        direct.validate_release(plan, receipt, changed)


class EmptyJobs:
    def __init__(self, _token: str, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def all_runs(self) -> list[dict]:
        return []


def test_direct_create_journals_before_exactly_one_nonretried_create(
    plan, tmp_path, monkeypatch
) -> None:
    expected = direct.manifest(plan)
    path = tmp_path / "direct-rayjob.json"
    path.write_text(json.dumps(expected))
    authorization = _authorization(plan)
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        arguments = argv[5:]
        if arguments[:2] == ["get", "rayjob"] or arguments[:2] == ["get", "fleetjob"]:
            return NS(returncode=0, stdout=json.dumps({"items": []}), stderr="")
        if arguments[:2] == ["get", "job"]:
            return NS(returncode=0, stdout=json.dumps({"items": []}), stderr="")
        if arguments[:2] == ["create", "--dry-run=server"]:
            return NS(returncode=0, stdout=json.dumps(_rendered(plan)), stderr="")
        if arguments[:1] == ["create"]:
            assert (tmp_path / "DIRECT_RAYJOB_CREATE.jsonl").exists()
            created = _rendered(plan)
            return NS(returncode=0, stdout=json.dumps(created), stderr="")
        raise AssertionError(arguments)

    monkeypatch.setattr(direct.os, "kill", lambda *_: None)
    result = direct.create_once(
        tmp_path,
        plan,
        expected,
        _preview(plan),
        authorization,
        token="test-token",
        runner=runner,
        jobs_factory=EmptyJobs,
    )
    mutating = [
        call for call in calls if call[5:6] == ["create"] and "--dry-run=server" not in call
    ]
    assert len(mutating) == 1
    assert all("apply" not in call and "patch" not in call for call in calls)
    assert result["status"] == "created"
    journal = json.loads((tmp_path / "DIRECT_RAYJOB_CREATE.jsonl").read_text())
    assert journal["state"] == "CREATE_INTENT_DO_NOT_RETRY"
    with pytest.raises(JobsError, match="intent already exists"):
        direct.create_once(
            tmp_path,
            plan,
            expected,
            _preview(plan),
            authorization,
            token="test-token",
            runner=runner,
            jobs_factory=EmptyJobs,
        )


def test_direct_duplicate_check_fails_closed_on_any_existing_object(plan) -> None:
    def runner(argv, **_kwargs):
        arguments = argv[5:]
        existing = arguments[:2] == ["get", "fleetjob"] and direct.DEV_CONTEXT in argv
        items = [{"metadata": {"name": plan["run_name"]}}] if existing else []
        return NS(returncode=0, stdout=json.dumps({"items": items}), stderr="")

    with pytest.raises(JobsError, match="already owns"):
        direct.exhaustive_duplicate_checks(
            plan, token="test-token", runner=runner, jobs_factory=EmptyJobs
        )


def test_direct_ambiguous_create_is_journaled_and_never_retried(
    plan, tmp_path, monkeypatch
) -> None:
    expected = direct.manifest(plan)
    (tmp_path / "direct-rayjob.json").write_text(json.dumps(expected))
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        arguments = argv[5:]
        if arguments[:1] == ["get"]:
            return NS(returncode=0, stdout=json.dumps({"items": []}), stderr="")
        if arguments[:2] == ["create", "--dry-run=server"]:
            return NS(returncode=0, stdout=json.dumps(_rendered(plan)), stderr="")
        if arguments[:1] == ["create"]:
            assert (tmp_path / "DIRECT_RAYJOB_CREATE.jsonl").exists()
            return NS(returncode=1, stdout="", stderr="connection closed")
        raise AssertionError(arguments)

    monkeypatch.setattr(direct.os, "kill", lambda *_: None)
    arguments = (
        tmp_path,
        plan,
        expected,
        _preview(plan),
        _authorization(plan),
    )
    with pytest.raises(JobsError, match="ambiguous; reconcile journal, never retry"):
        direct.create_once(
            *arguments,
            token="test-token",
            runner=runner,
            jobs_factory=EmptyJobs,
        )
    with pytest.raises(JobsError, match="intent already exists"):
        direct.create_once(
            *arguments,
            token="test-token",
            runner=runner,
            jobs_factory=EmptyJobs,
        )
    mutating = [
        call for call in calls if call[5:6] == ["create"] and "--dry-run=server" not in call
    ]
    assert len(mutating) == 1
