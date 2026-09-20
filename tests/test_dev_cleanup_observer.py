"""Offline tests for the pre-armed, UID-bound development cleanup observer."""

from __future__ import annotations

import json
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
                    "status": {
                        "conditions": [{"type": "Complete", "status": "True"}]
                    },
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
                            "state": {
                                "terminated": {"message": json.dumps(self.receipt)}
                            },
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
                for index, gpu in ((15, 0), (16, 8)):
                    state = {}
                    if not gpu:
                        state = {"terminated": {"message": json.dumps(self.receipt)}}
                    items.append(
                        self._object(
                            f"probe-pod-{index}",
                            index,
                            spec={
                                "containers": [
                                    {
                                        "resources": {
                                            "requests": {"nvidia.com/gpu": gpu}
                                        }
                                    }
                                ]
                            },
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
        if args[:2] == ["get", "pod"] and len(args) > 2:
            return NS(returncode=0, stdout="", stderr="")
        if args[:2] == ["delete", "fleetjob"]:
            self.deleted = True
            return NS(returncode=0, stdout="fleetjob.fleet.ai/probe\n", stderr="")
        raise AssertionError(args)


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
    assert len(result["pod_uids"]) == 2
    assert result["rayjob_uid"].endswith("000000000012")
    assert result["workload_uid"].endswith("000000000013")
    assert result["raycluster_uid"].endswith("000000000014")
    assert result["receipt"]["status"] == "setup_and_internal_cleanup_passed"


def test_observer_rejects_prod_route_or_excess_deadline(tmp_path) -> None:
    with pytest.raises(cleanup.ObserverError, match="development cluster"):
        _observer(tmp_path, FakeJobCluster(), context="prod")
    with pytest.raises(cleanup.ObserverError, match="deadline"):
        _observer(tmp_path, FakeJobCluster(), maximum_seconds=1801)
