"""Offline tests for the pre-armed, UID-bound development cleanup observer."""

from __future__ import annotations

import copy
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cyber_post_train.jobs import digest
from scripts import probe_qwen38_prod8_terminal as probe
from training import dev_cleanup_observer as cleanup


def _seal(value: dict) -> dict:
    return {**value, "sha256": "sha256:" + digest(value)}


def _terminal_probe_receipt(**overrides) -> dict:
    value = {
        "schema": probe.RECEIPT_SCHEMA,
        "status": "inspected",
        "target": probe.TARGET,
        "training_plan_sha256": probe.TRAINING_PLAN_SHA256,
        "checked_at": "2026-09-21T12:00:00Z",
        "gpus": 0,
        "sfs_mount_read_only": True,
        "private_payloads_read": False,
        "root_exists": False,
        "root_direct": False,
        "scan": {"attempts": 1, "stable": True},
        "terminal_classification": "root_missing",
        "root_markers": [
            {"name": name, "present": False, "accepted": False, "reason": "absent"}
            for name in probe.ROOT_MARKERS
        ],
        "batch_inventory": probe._empty_batch_inventory(accepted=True),
        "checkpoint_inventory": probe._empty_checkpoint_inventory(accepted=True),
        "receipt_size_limit_bytes": 3500,
        "receipt_size_bytes": 0,
    }
    value.update(overrides)
    while True:
        value.pop("sha256", None)
        value = _seal(value)
        size = len((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
        if value["receipt_size_bytes"] == size:
            return value
        value["receipt_size_bytes"] = size


def _metadata(name: str, uid: int) -> dict:
    return {
        "name": name,
        "uid": f"00000000-0000-0000-0000-{uid:012d}",
        "creationTimestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


class FakeJobCluster:
    def __init__(self) -> None:
        self.target_reads = 0
        self.deleted = False
        self.delete_calls = 0
        self.receipt = _seal(
            {
                "schema": "cyber_skyrl_topology_probe_cpu_preflight_v1",
                "status": "passed",
                "gpus": 0,
            }
        )

    def __call__(self, argv, **_kwargs):
        args = argv[5:]
        if args[:2] == ["get", "job"]:
            self.target_reads += 1
            value = None
            if self.target_reads > 1 and not self.deleted:
                value = {
                    "metadata": _metadata("preflight", 1),
                    "status": {"conditions": [{"type": "Complete", "status": "True"}]},
                }
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:2] == ["get", "pod"] and "--selector" in args:
            pod = {
                "metadata": _metadata("preflight-pod", 2),
                "spec": {"containers": [{"resources": {"requests": {}}}]},
                "status": {
                    "containerStatuses": [
                        {
                            "restartCount": 0,
                            "imageID": "registry/image@sha256:" + "a" * 64,
                            "state": {"terminated": {"message": json.dumps(self.receipt)}},
                        }
                    ]
                },
            }
            return NS(
                returncode=0,
                stdout=json.dumps({"items": [] if self.deleted else [pod]}),
                stderr="",
            )
        if args[:3] == ["get", "pod", "preflight-pod"]:
            return NS(returncode=0, stdout="", stderr="")
        if args[:2] == ["delete", "job"]:
            self.deleted = True
            self.delete_calls += 1
            return NS(returncode=0, stdout="job.batch/preflight\n", stderr="")
        raise AssertionError(args)


def _observer(tmp_path: Path, runner, **overrides) -> cleanup.Observer:
    values = {
        "context": cleanup.DEV_CONTEXT,
        "namespace": cleanup.NAMESPACE,
        "kind": "job",
        "name": "preflight",
        "maximum_seconds": 1200,
        "expected_gpus": 0,
        "plan_sha256": "sha256:" + "1" * 64,
        "manifest_sha256": "sha256:" + "2" * 64,
        "armed_path": tmp_path / "ARMED.json",
        "result_path": tmp_path / "RESULT.json",
        "poll_seconds": 0.001,
        "run": runner,
    }
    values.update(overrides)
    return cleanup.Observer(**values)


def test_job_observer_arms_before_creation_captures_receipt_and_releases(tmp_path) -> None:
    cluster = FakeJobCluster()
    observer = _observer(tmp_path, cluster)
    result = observer.run()
    assert result["status"] == "released"
    assert result["terminal_status"] == "Succeeded"
    assert result["receipt"]["status"] == "passed"
    assert result["peak_gpus"] == result["expected_gpus"] == 0
    assert result["target_present"] is result["pods_present"] is False
    assert result["active_gpus"] == 0
    assert cluster.delete_calls == 1
    assert json.loads((tmp_path / "ARMED.json").read_text())["status"] == "armed"
    assert json.loads((tmp_path / "RESULT.json").read_text()) == result


class Prod8TerminalProbeCluster:
    def __init__(self) -> None:
        self.target_reads = 0
        self.deleted = False
        self.delete_calls = 0
        self.receipt = _terminal_probe_receipt()
        self.mutate_job = lambda value: value
        self.mutate_pod = lambda value: value

    @property
    def job_uid(self) -> str:
        return _metadata(probe.NAME, 101)["uid"]

    def _job(self) -> dict:
        value = copy.deepcopy(probe.manifest())
        value["metadata"].update(_metadata(probe.NAME, 101))
        labels = {
            "batch.kubernetes.io/controller-uid": self.job_uid,
            "batch.kubernetes.io/job-name": probe.NAME,
            "controller-uid": self.job_uid,
            "job-name": probe.NAME,
        }
        value["metadata"]["labels"] = labels
        value["spec"].update(
            {
                "completionMode": "NonIndexed",
                "completions": 1,
                "manualSelector": False,
                "parallelism": 1,
                "podReplacementPolicy": "TerminatingOrFailed",
                "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": self.job_uid}},
                "suspend": False,
            }
        )
        value["spec"]["template"]["metadata"]["labels"] = labels
        value["status"] = {"conditions": [{"type": "Complete", "status": "True"}]}
        return self.mutate_job(value)

    def _pod(self) -> dict:
        owner = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": probe.NAME,
            "uid": self.job_uid,
            "controller": True,
            "blockOwnerDeletion": True,
        }
        return self.mutate_pod(
            {
                "metadata": {
                    **_metadata("prod8-terminal-probe-pod", 102),
                    "labels": {
                        "batch.kubernetes.io/controller-uid": self.job_uid,
                        "job-name": probe.NAME,
                    },
                    "ownerReferences": [owner],
                },
                "spec": {
                    "priorityClassName": "c1",
                    "containers": [
                        {
                            "name": "terminal-probe",
                            "image": probe.IMAGE,
                            "resources": {"requests": {}, "limits": {}},
                        }
                    ],
                },
                "status": {
                    "containerStatuses": [
                        {
                            "name": "terminal-probe",
                            "restartCount": 0,
                            "imageID": "registry/image@" + probe.IMAGE.rsplit("@", 1)[1],
                            "state": {
                                "terminated": {
                                    "exitCode": 0,
                                    "reason": "Completed",
                                    "message": json.dumps(self.receipt),
                                }
                            },
                        }
                    ]
                },
            }
        )

    def __call__(self, argv, **_kwargs):
        args = argv[5:]
        if args[:2] == ["get", "job"]:
            self.target_reads += 1
            value = self._job() if self.target_reads > 1 and not self.deleted else None
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:2] == ["get", "pod"] and "--selector" in args:
            return NS(
                returncode=0,
                stdout=json.dumps({"items": [] if self.deleted else [self._pod()]}),
                stderr="",
            )
        if args[:3] == ["get", "pod", "prod8-terminal-probe-pod"]:
            return NS(returncode=0, stdout="", stderr="")
        if args[:2] == ["delete", "job"]:
            self.deleted = True
            self.delete_calls += 1
            return NS(returncode=0, stdout="job.batch/prod8\n", stderr="")
        raise AssertionError(args)


def _prod8_observer(tmp_path: Path, runner: Prod8TerminalProbeCluster) -> cleanup.Observer:
    return _observer(
        tmp_path,
        runner,
        name=probe.NAME,
        maximum_seconds=1200,
        manifest_sha256=probe.manifest_digest(),
    )


def test_job_observer_binds_prod8_receipt_to_exact_job_pod_and_image(tmp_path) -> None:
    cluster = Prod8TerminalProbeCluster()
    result = _prod8_observer(tmp_path, cluster).run()
    assert result["status"] == "released"
    assert result["receipt"] == cluster.receipt
    binding = result["prod8_live_binding"]
    assert binding["job_uid"] == cluster.job_uid
    assert binding["receipt_container_bound"] is True
    assert binding["binding_error"] == ""
    assert binding["containers"] == [
        {
            "pod_name": "prod8-terminal-probe-pod",
            "pod_uid": _metadata("prod8-terminal-probe-pod", 102)["uid"],
            "container_name": "terminal-probe",
            "requested_image": probe.IMAGE,
            "requested_image_digest": probe.IMAGE.rsplit("@", 1)[1],
            "runtime_image_id": "registry/image@" + probe.IMAGE.rsplit("@", 1)[1],
            "runtime_image_digest": probe.IMAGE.rsplit("@", 1)[1],
            "exit_code": 0,
            "termination_reason": "Completed",
            "restart_count": 0,
        }
    ]
    assert result["active_gpus"] == 0
    assert cluster.delete_calls == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda job: job["metadata"]["annotations"].pop("fleet.ai/failure-alerts"),
        lambda job: job["spec"].update({"backoffLimit": False}),
        lambda job: job["spec"]["template"]["spec"].update({"priorityClassName": "c2"}),
        lambda job: job["spec"]["template"]["spec"]["containers"][0]["resources"][
            "requests"
        ].update({"nvidia.com/gpu": "1"}),
        lambda job: job["spec"]["template"]["spec"].update({"overhead": {"nvidia.com/gpu": "1"}}),
        lambda job: job["spec"]["template"]["spec"]["containers"][0].update(
            {"envFrom": [{"secretRef": {"name": "unreviewed"}}]}
        ),
        lambda job: job["spec"]["template"]["spec"].update(
            {
                "initContainers": [
                    {
                        "name": "injected",
                        "resources": {
                            "requests": {"nvidia.com/gpu": "1"},
                            "limits": {"nvidia.com/gpu": "1"},
                        },
                    }
                ]
            }
        ),
    ],
)
def test_job_observer_rejects_manifest_binding_drift(tmp_path, mutate) -> None:
    cluster = Prod8TerminalProbeCluster()

    def mutate_job(value):
        value = copy.deepcopy(value)
        mutate(value)
        return value

    cluster.mutate_job = mutate_job
    result = _prod8_observer(tmp_path, cluster).run()
    assert result["status"] == "released_without_accepted_execution"
    assert result["receipt"] is None
    assert result["prod8_live_binding"]["binding_error"] in {
        "manifest_malformed",
        "manifest_mismatch",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pod: pod["metadata"]["ownerReferences"][0].update(
            {"uid": _metadata("other", 103)["uid"]}
        ),
        lambda pod: pod["metadata"]["labels"].update(
            {"batch.kubernetes.io/controller-uid": _metadata("other", 103)["uid"]}
        ),
        lambda pod: pod["spec"]["containers"][0].update(
            {"image": "registry/example@sha256:" + "0" * 64}
        ),
        lambda pod: pod["status"]["containerStatuses"][0].update(
            {"imageID": "registry/example@sha256:" + "0" * 64}
        ),
        lambda pod: pod["status"]["containerStatuses"][0].update(
            {"imageID": "unbound-runtime-image"}
        ),
        lambda pod: pod["spec"]["containers"][0]["resources"]["limits"].update(
            {"nvidia.com/gpu": "1"}
        ),
        lambda pod: pod["spec"].update({"overhead": {"nvidia.com/gpu": "1"}}),
        lambda pod: pod["spec"].update(
            {"initContainers": [{"name": "sidecar", "resources": {"requests": {}, "limits": {}}}]}
        ),
        lambda pod: pod["status"]["containerStatuses"].append(
            {
                "name": "sidecar",
                "restartCount": 0,
                "imageID": "registry/sidecar@sha256:" + "0" * 64,
                "state": {
                    "terminated": {
                        "exitCode": 0,
                        "message": json.dumps(_terminal_probe_receipt()),
                    }
                },
            }
        ),
    ],
)
def test_job_observer_rejects_foreign_or_unsafe_prod8_pod(tmp_path, mutate) -> None:
    cluster = Prod8TerminalProbeCluster()

    def mutate_pod(value):
        value = copy.deepcopy(value)
        mutate(value)
        return value

    cluster.mutate_pod = mutate_pod
    result = _prod8_observer(tmp_path, cluster).run()
    assert result["status"] == "released_without_accepted_execution"
    assert result["receipt"] is None
    assert result["prod8_live_binding"]["binding_error"]
    assert result["active_gpus"] == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pod: pod["status"]["containerStatuses"][0].update({"restartCount": 1}),
        lambda pod: pod["status"]["containerStatuses"][0]["state"]["terminated"].update(
            {"exitCode": 1}
        ),
    ],
)
def test_job_observer_records_exact_nonaccepted_terminal_container_state(tmp_path, mutate) -> None:
    cluster = Prod8TerminalProbeCluster()

    def mutate_pod(value):
        value = copy.deepcopy(value)
        mutate(value)
        return value

    cluster.mutate_pod = mutate_pod
    result = _prod8_observer(tmp_path, cluster).run()
    assert result["status"] == "released_without_accepted_execution"
    assert result["receipt"] is None
    observed = result["prod8_live_binding"]["containers"]
    assert len(observed) == 1
    assert observed[0]["restart_count"] == 1 or observed[0]["exit_code"] == 1


def test_job_observer_rejects_and_does_not_persist_closed_schema_drift(tmp_path) -> None:
    cluster = Prod8TerminalProbeCluster()
    receipt = _terminal_probe_receipt()
    receipt["unexpected_private_payload"] = "must-not-persist"
    cluster.receipt = _terminal_probe_receipt(**receipt)
    result = _prod8_observer(tmp_path, cluster).run()
    assert result["status"] == "released_without_accepted_execution"
    assert result["receipt"] is None
    assert result["prod8_live_binding"]["receipt_container_bound"] is False


class FakeFleetCluster:
    def __init__(self) -> None:
        self.target_reads = 0
        self.deleted = False
        self.receipt = _seal(
            {
                "schema": "cyber_skyrl_topology_probe_receipt_v1",
                "status": "setup_and_internal_cleanup_passed",
            }
        )

    def _object(self, name: str, uid: int, **extra):
        return {"metadata": _metadata(name, uid), **extra}

    def __call__(self, argv, **_kwargs):
        args = argv[5:]
        if args[:2] == ["get", "fleetjob"]:
            self.target_reads += 1
            value = None
            if self.target_reads > 1 and not self.deleted:
                value = self._object(
                    "probe",
                    10,
                    status={
                        "jobId": "00000000-0000-0000-0000-000000000011",
                        "rayJobName": "ray-probe",
                    },
                )
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:3] == ["get", "rayjob", "ray-probe"]:
            value = None
            if not self.deleted:
                value = self._object(
                    "ray-probe",
                    12,
                    status={"rayClusterName": "cluster-probe", "jobStatus": "SUCCEEDED"},
                )
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:3] == ["get", "workload", "--output"]:
            items = []
            if not self.deleted:
                row = self._object("workload-probe", 13)
                row["metadata"]["ownerReferences"] = [{"name": "ray-probe"}]
                items.append(row)
            return NS(returncode=0, stdout=json.dumps({"items": items}), stderr="")
        if args[:3] == ["get", "workload", "workload-probe"]:
            return NS(returncode=0, stdout="", stderr="")
        if args[:3] == ["get", "raycluster", "cluster-probe"]:
            value = None if self.deleted else self._object("cluster-probe", 14)
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:2] == ["get", "pod"] and "--selector" in args:
            items = []
            if not self.deleted:
                items.append(
                    self._object(
                        "probe-pod-15",
                        15,
                        spec={"containers": [{"resources": {"requests": {"nvidia.com/gpu": 8}}}]},
                        status={
                            "containerStatuses": [
                                {
                                    "restartCount": 0,
                                    "imageID": "registry/image@sha256:" + "b" * 64,
                                    "state": {"terminated": {"message": json.dumps(self.receipt)}},
                                }
                            ]
                        },
                    )
                )
            return NS(returncode=0, stdout=json.dumps({"items": items}), stderr="")
        if args[:2] == ["get", "pod"] and len(args) > 2:
            return NS(returncode=0, stdout="", stderr="")
        if args[:2] == ["delete", "fleetjob"]:
            self.deleted = True
            return NS(returncode=0, stdout="fleetjob.fleet.ai/probe\n", stderr="")
        raise AssertionError(args)


class DelayedFleetReceiptCluster(FakeFleetCluster):
    """Ray reports success one observation before the Pod exposes its receipt."""

    def __init__(self) -> None:
        super().__init__()
        self.rayjob_reads = 0
        self.pod_list_reads = 0

    def __call__(self, argv, **kwargs):
        args = argv[5:]
        if args[:3] == ["get", "rayjob", "ray-probe"]:
            self.rayjob_reads += 1
            value = None
            if not self.deleted:
                value = self._object(
                    "ray-probe",
                    12,
                    status={
                        "rayClusterName": "cluster-probe",
                        "jobStatus": "RUNNING" if self.rayjob_reads == 1 else "SUCCEEDED",
                    },
                )
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:2] == ["get", "pod"] and "--selector" in args:
            self.pod_list_reads += 1
            items = []
            if not self.deleted:
                state = (
                    {"running": {}}
                    if self.pod_list_reads < 3
                    else {"terminated": {"message": json.dumps(self.receipt)}}
                )
                items.append(
                    self._object(
                        "probe-pod-15",
                        15,
                        spec={"containers": [{"resources": {"requests": {"nvidia.com/gpu": 8}}}]},
                        status={
                            "containerStatuses": [
                                {
                                    "restartCount": 0,
                                    "imageID": "registry/image@sha256:" + "b" * 64,
                                    "state": state,
                                }
                            ]
                        },
                    )
                )
            return NS(returncode=0, stdout=json.dumps({"items": items}), stderr="")
        return super().__call__(argv, **kwargs)


def test_fleetjob_observer_binds_all_uids_and_exact_eight_gpus(tmp_path) -> None:
    observer = _observer(
        tmp_path,
        FakeFleetCluster(),
        kind="fleetjob",
        name="probe",
        maximum_seconds=1800,
        expected_gpus=8,
    )
    result = observer.run()
    assert result["status"] == "released"
    assert result["peak_gpus"] == 8
    assert len(result["pod_uids"]) == 1
    assert result["rayjob_uid"].endswith("000000000012")
    assert result["workload_uid"].endswith("000000000013")
    assert result["raycluster_uid"].endswith("000000000014")
    assert result["receipt"]["status"] == "setup_and_internal_cleanup_passed"


def test_fleetjob_observer_waits_bounded_time_for_terminal_receipt(tmp_path) -> None:
    cluster = DelayedFleetReceiptCluster()
    observer = _observer(
        tmp_path,
        cluster,
        kind="fleetjob",
        name="probe",
        maximum_seconds=1800,
        expected_gpus=8,
    )
    result = observer.run()
    assert result["status"] == "released"
    assert result["terminal_status"] == "Succeeded"
    assert result["receipt"]["status"] == "setup_and_internal_cleanup_passed"
    assert cluster.rayjob_reads >= 3
    assert cluster.pod_list_reads >= 3


class FakeDirectRayJobCluster:
    def __init__(self) -> None:
        self.target_reads = 0
        self.deleted = False
        self.receipt = _seal(
            {
                "schema": "cyber_skyrl_topology_probe_receipt_v1",
                "status": "setup_and_internal_cleanup_passed",
            }
        )

    @staticmethod
    def _object(name: str, uid: int, **extra):
        return {"metadata": _metadata(name, uid), **extra}

    def __call__(self, argv, **_kwargs):
        args = argv[5:]
        if args[:2] == ["get", "rayjob"]:
            self.target_reads += 1
            value = None
            if self.target_reads > 1 and not self.deleted:
                value = self._object(
                    "probe",
                    20,
                    status={"rayClusterName": "cluster-probe", "jobStatus": "SUCCEEDED"},
                )
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:3] == ["get", "workload", "--output"]:
            items = []
            if not self.deleted:
                row = self._object("workload-probe", 21)
                row["metadata"]["ownerReferences"] = [{"name": "probe"}]
                items.append(row)
            return NS(returncode=0, stdout=json.dumps({"items": items}), stderr="")
        if args[:3] == ["get", "workload", "workload-probe"]:
            return NS(returncode=0, stdout="", stderr="")
        if args[:3] == ["get", "raycluster", "cluster-probe"]:
            value = None if self.deleted else self._object("cluster-probe", 22)
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:2] == ["get", "pod"] and "--selector" in args:
            items = []
            if not self.deleted:
                items.append(
                    self._object(
                        "probe-pod-23",
                        23,
                        spec={"containers": [{"resources": {"requests": {"nvidia.com/gpu": 8}}}]},
                        status={
                            "containerStatuses": [
                                {
                                    "restartCount": 0,
                                    "imageID": "registry/image@sha256:" + "c" * 64,
                                    "state": {"terminated": {"message": json.dumps(self.receipt)}},
                                }
                            ]
                        },
                    )
                )
            return NS(returncode=0, stdout=json.dumps({"items": items}), stderr="")
        if args[:2] == ["get", "pod"] and len(args) > 2:
            return NS(returncode=0, stdout="", stderr="")
        if args[:2] == ["delete", "rayjob"]:
            self.deleted = True
            return NS(returncode=0, stdout="rayjob.ray.io/probe\n", stderr="")
        raise AssertionError(args)


def test_direct_rayjob_observer_binds_children_receipt_and_releases(tmp_path) -> None:
    observer = _observer(
        tmp_path,
        FakeDirectRayJobCluster(),
        kind="rayjob",
        name="probe",
        maximum_seconds=1800,
        expected_gpus=8,
    )
    result = observer.run()
    assert result["status"] == "released"
    assert result["uid"] == result["rayjob_uid"]
    assert result["peak_gpus"] == 8
    assert result["workload_uid"].endswith("000000000021")
    assert result["raycluster_uid"].endswith("000000000022")
    assert result["receipt"]["status"] == "setup_and_internal_cleanup_passed"
    assert result["deletion_reason"] == "terminal_status"


def test_production_recovery_observer_adopts_only_the_exact_live_uid(tmp_path) -> None:
    cluster = FakeDirectRayJobCluster()
    cluster.target_reads = 1  # The exact RayJob already exists before arming.
    uid = "00000000-0000-0000-0000-000000000020"
    observer = _observer(
        tmp_path,
        cluster,
        context=cleanup.PROD_CONTEXT,
        kind="rayjob",
        name="probe",
        maximum_seconds=1800,
        expected_gpus=8,
        profile="production-recovery",
        expected_uid=uid,
    )
    result = observer.run()
    armed = json.loads((tmp_path / "ARMED.json").read_text())
    assert armed["schema"] == cleanup.RECOVERY_ARMED_SCHEMA
    assert armed["expected_uid"] == uid
    assert armed["recovered_existing_target"] is True
    assert result["schema"] == cleanup.RECOVERY_RESULT_SCHEMA
    assert result["recovered_existing_target_uid"] == uid
    assert result["status"] == "released"


def test_production_recovery_observer_rejects_another_live_uid(tmp_path) -> None:
    cluster = FakeDirectRayJobCluster()
    cluster.target_reads = 1
    observer = _observer(
        tmp_path,
        cluster,
        context=cleanup.PROD_CONTEXT,
        kind="rayjob",
        name="probe",
        maximum_seconds=1800,
        expected_gpus=8,
        profile="production-recovery",
        expected_uid="00000000-0000-0000-0000-000000000099",
    )
    with pytest.raises(cleanup.ObserverError, match="UID differs"):
        observer.arm()


def test_observer_rejects_prod_route_or_excess_deadline(tmp_path) -> None:
    with pytest.raises(cleanup.ObserverError, match="development cluster"):
        _observer(tmp_path, FakeJobCluster(), context="prod")
    with pytest.raises(cleanup.ObserverError, match="deadline"):
        _observer(tmp_path, FakeJobCluster(), maximum_seconds=1801)


def test_observer_accepts_only_digest_valid_sanitized_failure_receipt() -> None:
    receipt = _seal(
        {
            "schema": "cyber_skyrl_topology_probe_cpu_preflight_rejection_v1",
            "status": "rejected",
            "phase": "model_inventory",
            "error_class": "ValueError",
        }
    )
    assert cleanup._validated_receipt(json.dumps(receipt), kind="job") == receipt
    receipt["phase"] = "changed"
    assert cleanup._validated_receipt(json.dumps(receipt), kind="job") is None


def test_observer_accepts_digest_valid_model_stage_receipt() -> None:
    receipt = _seal(
        {
            "schema": "cyber_skyrl_model_artifact_stage_receipt_v1",
            "status": "published",
            "plan_sha256": "0" * 64,
            "files": 28,
        }
    )
    assert cleanup._validated_receipt(json.dumps(receipt), kind="job") == receipt


def test_observer_accepts_digest_valid_topology_receipt_verifier() -> None:
    receipt = _seal(
        {
            "schema": "cyber_skyrl_topology_probe_receipt_verification_v1",
            "status": "passed",
            "plan_sha256": "0" * 64,
            "receipt_sha256": "sha256:" + "1" * 64,
            "gpus": 0,
        }
    )
    assert cleanup._validated_receipt(json.dumps(receipt), kind="job") == receipt


def test_observer_retries_one_transient_kubectl_timeout(tmp_path) -> None:
    cluster = FakeJobCluster()
    calls = 0

    def flaky(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return cluster(argv, **kwargs)

    result = _observer(tmp_path, flaky).run()
    assert calls > 1
    assert result["status"] == "released"
    assert result["observer_error_class"] == ""


def test_observer_recovers_after_one_exhausted_kubectl_read(tmp_path) -> None:
    cluster = FakeJobCluster()
    timeouts = 0

    def flaky(argv, **kwargs):
        nonlocal timeouts
        args = argv[5:]
        if (
            args[:2] == ["get", "pod"]
            and "--selector" in args
            and not cluster.deleted
            and timeouts < 3
        ):
            timeouts += 1
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return cluster(argv, **kwargs)

    result = _observer(tmp_path, flaky).run()
    assert timeouts == 3
    assert cluster.deleted is True
    assert result["status"] == "released"
    assert result["observer_error_class"] == ""
    assert result["target_present"] is result["pods_present"] is False
    assert json.loads((tmp_path / "RESULT.json").read_text()) == result


class FlakyActiveDirectRayJobCluster(FakeDirectRayJobCluster):
    """An active 8-GPU run survives five fully exhausted Pod-list reads."""

    def __init__(self) -> None:
        super().__init__()
        self.pod_list_reads = 0
        self.timeouts = 0
        self.recovered_active_reads = 0
        self.terminal = False
        self.delete_calls = 0
        self.deleted_while_running = False

    def __call__(self, argv, **kwargs):
        args = argv[5:]
        if args[:2] == ["get", "rayjob"]:
            self.target_reads += 1
            value = None
            if self.target_reads > 1 and not self.deleted:
                self.terminal = (
                    self.timeouts
                    >= cleanup.KUBECTL_ATTEMPTS * cleanup.MAX_CONSECUTIVE_OBSERVATION_FAILURES
                    and self.recovered_active_reads >= 1
                )
                value = self._object(
                    "probe",
                    20,
                    status={
                        "rayClusterName": "cluster-probe",
                        "jobStatus": "SUCCEEDED" if self.terminal else "RUNNING",
                    },
                )
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:2] == ["get", "pod"] and "--selector" in args:
            self.pod_list_reads += 1
            if (
                self.pod_list_reads > 1
                and self.timeouts
                < cleanup.KUBECTL_ATTEMPTS * cleanup.MAX_CONSECUTIVE_OBSERVATION_FAILURES
            ):
                self.timeouts += 1
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            if self.timeouts:
                self.recovered_active_reads += int(not self.terminal)
            items = []
            if not self.deleted:
                state = (
                    {"terminated": {"message": json.dumps(self.receipt)}}
                    if self.terminal
                    else {"running": {}}
                )
                items.append(
                    self._object(
                        "probe-pod-23",
                        23,
                        spec={"containers": [{"resources": {"requests": {"nvidia.com/gpu": 8}}}]},
                        status={
                            "containerStatuses": [
                                {
                                    "restartCount": 0,
                                    "imageID": "registry/image@sha256:" + "c" * 64,
                                    "state": state,
                                }
                            ]
                        },
                    )
                )
            return NS(returncode=0, stdout=json.dumps({"items": items}), stderr="")
        if args[:2] == ["delete", "rayjob"]:
            self.delete_calls += 1
            self.deleted_while_running = not self.terminal
        return super().__call__(argv, **kwargs)


def test_observer_read_failures_cannot_delete_an_active_gpu_run(tmp_path) -> None:
    cluster = FlakyActiveDirectRayJobCluster()
    result = _observer(
        tmp_path,
        cluster,
        kind="rayjob",
        name="probe",
        maximum_seconds=1800,
        expected_gpus=8,
    ).run()
    assert cluster.timeouts == (
        cleanup.KUBECTL_ATTEMPTS * cleanup.MAX_CONSECUTIVE_OBSERVATION_FAILURES
    )
    assert cluster.recovered_active_reads >= 1
    assert cluster.deleted_while_running is False
    assert cluster.delete_calls == 1
    assert result["status"] == "released"
    assert result["terminal_status"] == "Succeeded"
    assert result["peak_gpus"] == 8
    assert result["observation_failures"] >= cleanup.MAX_CONSECUTIVE_OBSERVATION_FAILURES
    assert (
        result["max_consecutive_observation_failures"]
        >= cleanup.MAX_CONSECUTIVE_OBSERVATION_FAILURES
    )
    assert result["last_observation_error_code"] == "kubectl_timeout"
    assert result["deletion_reason"] == "terminal_status"
    assert result["observer_error_class"] == ""
