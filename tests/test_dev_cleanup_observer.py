"""Offline tests for the pre-armed, UID-bound development cleanup observer."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cyber_post_train.jobs import digest
from training import dev_cleanup_observer as cleanup


def _seal(value: dict) -> dict:
    return {**value, "sha256": "sha256:" + digest(value)}


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
