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


def _canonical_terminal_probe_receipt(value: dict) -> str:
    return probe.canonical_receipt_bytes(value).decode("utf-8")


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
        if args[:2] == ["delete", "--raw"]:
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
        self.hide_pod_from_controller_selector = False

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
        pod_spec = copy.deepcopy(self._job()["spec"]["template"]["spec"])
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
                "spec": pod_spec,
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
                                    "message": _canonical_terminal_probe_receipt(self.receipt),
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
                stdout=json.dumps(
                    {
                        "items": (
                            []
                            if self.deleted or self.hide_pod_from_controller_selector
                            else [self._pod()]
                        )
                    }
                ),
                stderr="",
            )
        if args[:3] == ["get", "pod", "prod8-terminal-probe-pod"]:
            return NS(returncode=0, stdout="", stderr="")
        if args[:2] == ["delete", "--raw"]:
            assert args[2] == (f"/apis/batch/v1/namespaces/{cleanup.NAMESPACE}/jobs/{probe.NAME}")
            assert args[-2:] == ["-f", "-"]
            assert json.loads(_kwargs["input"]) == {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "propagationPolicy": "Foreground",
                "preconditions": {"uid": self.job_uid},
            }
            self.deleted = True
            self.delete_calls += 1
            return NS(returncode=0, stdout="job.batch/prod8\n", stderr="")
        raise AssertionError(args)


def _prod8_observer(
    tmp_path: Path, runner: Prod8TerminalProbeCluster, **overrides: object
) -> cleanup.Observer:
    values: dict[str, object] = {
        "context": cleanup.PROD_CONTEXT,
        "name": probe.NAME,
        "maximum_seconds": 1200,
        "plan_sha256": "sha256:" + probe.TRAINING_PLAN_SHA256,
        "manifest_sha256": probe.manifest_digest(),
        "profile": "production-cpu",
        "expected_uid": runner.job_uid,
    }
    values.update(overrides)
    return _observer(tmp_path, runner, **values)


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
    assert result["creation_bound_uid"] == cluster.job_uid
    armed = json.loads((tmp_path / "ARMED.json").read_text())
    assert armed["creation_bound_uid"] == cluster.job_uid


def test_prod8_post_arm_same_name_collision_never_binds_or_deletes(tmp_path) -> None:
    cluster = Prod8TerminalProbeCluster()
    unrelated_uid = _metadata("unrelated", 999)["uid"]
    with pytest.raises(cleanup.ObserverError, match="UID changed"):
        _prod8_observer(tmp_path, cluster, expected_uid=unrelated_uid).run()
    assert cluster.delete_calls == 0
    assert cluster.deleted is False


def test_prod8_hidden_controller_owned_pod_never_authorizes_delete(tmp_path) -> None:
    cluster = Prod8TerminalProbeCluster()
    # Model a completed Job whose actual Pod is still live but has had its
    # mutable controller label removed, so the selector cannot prove its full
    # owner/spec/status binding.
    cluster.hide_pod_from_controller_selector = True
    with pytest.raises(cleanup.ObserverError, match="no uniquely bound live Pod"):
        _prod8_observer(tmp_path, cluster).run()
    assert cluster.delete_calls == 0
    assert cluster.deleted is False


def test_prod8_delete_revalidates_live_pod_before_raw_delete(tmp_path) -> None:
    cluster = Prod8TerminalProbeCluster()
    observer = _prod8_observer(tmp_path, cluster)
    observer.arm()
    observer.observe(cluster._job())
    cluster.hide_pod_from_controller_selector = True
    with pytest.raises(cleanup.ObserverError, match="no uniquely bound live Pod"):
        observer.delete()
    assert cluster.delete_calls == 0
    assert cluster.deleted is False


def test_prod8_observer_rejects_mislabeled_training_plan_before_arming(tmp_path) -> None:
    cluster = Prod8TerminalProbeCluster()
    with pytest.raises(cleanup.ObserverError, match="training-plan binding"):
        _prod8_observer(tmp_path, cluster, plan_sha256="sha256:" + "0" * 64)
    assert not (tmp_path / "ARMED.json").exists()
    assert cluster.delete_calls == 0


class UidPreconditionRaceCluster(Prod8TerminalProbeCluster):
    """The API atomically rejects a recreated same-name Job at DELETE time."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_delete_bodies: list[dict] = []

    def __call__(self, argv, **kwargs):
        args = argv[5:]
        if args[:2] == ["delete", "--raw"]:
            self.raw_delete_bodies.append(json.loads(kwargs["input"]))
            return NS(returncode=1, stdout="", stderr="Conflict")
        return super().__call__(argv, **kwargs)


def test_prod8_uid_precondition_prevents_delete_after_name_reuse_race(tmp_path) -> None:
    cluster = UidPreconditionRaceCluster()
    observer = _prod8_observer(tmp_path, cluster)
    observer.arm()
    observer.observe(cluster._job())
    with pytest.raises(cleanup.ObserverError, match="observation failed"):
        observer.delete()
    assert cluster.deleted is False
    assert len(cluster.raw_delete_bodies) == cleanup.KUBECTL_ATTEMPTS
    assert all(
        body["preconditions"] == {"uid": cluster.job_uid}
        and body["propagationPolicy"] == "Foreground"
        for body in cluster.raw_delete_bodies
    )


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
    with pytest.raises(cleanup.ObserverError, match="manifest binding failed"):
        _prod8_observer(tmp_path, cluster).run()
    assert cluster.delete_calls == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pod: pod["metadata"].update({"uid": "not-a-uuid"}),
        lambda pod: pod["metadata"].update({"creationTimestamp": "not-a-timestamp"}),
        lambda pod: pod.update({"metadata": None}),
        lambda pod: pod["metadata"]["ownerReferences"][0].update(
            {"uid": _metadata("other", 103)["uid"]}
        ),
        lambda pod: pod["metadata"]["labels"].update(
            {"batch.kubernetes.io/controller-uid": _metadata("other", 103)["uid"]}
        ),
        lambda pod: pod["spec"]["containers"][0].update(
            {"image": "registry/example@sha256:" + "0" * 64}
        ),
        lambda pod: pod["spec"]["containers"][0].update(
            {"command": ["sh", "-c", "echo unreviewed"]}
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
        lambda pod: pod["spec"]["containers"][0]["resources"]["limits"].update(
            {"nvidia.com/mig-1g.5gb": "1"}
        ),
        # Exact resource-map comparison also rejects unrecognized extended
        # resource names; it is not limited to a hand-maintained GPU alias
        # list.
        lambda pod: pod["spec"]["containers"][0]["resources"]["limits"].update(
            {"vendor.example/custom-device": "1"}
        ),
        lambda pod: pod["spec"].update({"overhead": {"nvidia.com/gpu": "1"}}),
        lambda pod: pod["spec"].update(
            {"initContainers": [{"name": "sidecar", "resources": {"requests": {}, "limits": {}}}]}
        ),
        lambda pod: pod["spec"]["containers"][0].update(
            {"envFrom": [{"secretRef": {"name": "unreviewed"}}]}
        ),
        lambda pod: pod["spec"]["containers"][0].update(
            {"lifecycle": {"postStart": {"exec": {"command": ["sh", "-c", "id"]}}}}
        ),
        lambda pod: pod["spec"].update({"resourceClaims": [{"name": "unreviewed"}]}),
        lambda pod: pod["status"]["containerStatuses"].append(
            {
                "name": "sidecar",
                "restartCount": 0,
                "imageID": "registry/sidecar@sha256:" + "0" * 64,
                "state": {
                    "terminated": {
                        "exitCode": 0,
                        "message": _canonical_terminal_probe_receipt(_terminal_probe_receipt()),
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
    with pytest.raises(cleanup.ObserverError, match="Pod binding failed"):
        _prod8_observer(tmp_path, cluster).run()
    # A rejected live-Pod binding is not authorization to clean up the Job.
    # In particular, an admission-mutated command or accelerator request must
    # remain for an operator rather than be silently deleted by this probe.
    assert cluster.delete_calls == 0
    assert cluster.deleted is False


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
    with pytest.raises(cleanup.ObserverError, match="receipt was rejected"):
        _prod8_observer(tmp_path, cluster).run()
    assert cluster.delete_calls == 0
    assert cluster.deleted is False


def test_job_observer_rejects_padded_prod8_receipt_without_deleting(tmp_path) -> None:
    cluster = Prod8TerminalProbeCluster()

    def mutate_pod(value):
        value = copy.deepcopy(value)
        terminated = value["status"]["containerStatuses"][0]["state"]["terminated"]
        terminated["message"] += " "
        return value

    cluster.mutate_pod = mutate_pod
    with pytest.raises(cleanup.ObserverError, match="receipt was rejected"):
        _prod8_observer(tmp_path, cluster).run()
    assert cluster.delete_calls == 0
    assert cluster.deleted is False


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
        if args[:2] == ["delete", "--raw"]:
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
        if args[:2] == ["delete", "--raw"]:
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


def test_observer_rejects_oversized_termination_message_before_utf8_copy() -> None:
    class OversizedMessage(str):
        def encode(self, *args, **kwargs):  # pragma: no cover - must not execute
            raise AssertionError("oversized message was encoded")

    assert cleanup._validated_receipt(OversizedMessage("x" * 16385), kind="job") is None


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
        if args[:2] == ["delete", "--raw"]:
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


class FakeJobsApiPrefixGuardCluster:
    """Offline Kubernetes surface for the non-destructive Jobs API guard."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.exact: dict | None = None
        self.exact_after_reads = 0
        self.exact_reads = 0
        self.calls: list[list[str]] = []
        self.delete_calls = 0

    @staticmethod
    def rayjob(
        name: str = "collector-diag-1a2b3c4d",
        uid: int = 301,
        *,
        run_dir: str = "/mnt/sfs/jobs/collector-diag-v1",
        image: str = "registry/image@sha256:" + "d" * 64,
    ) -> dict:
        return {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "metadata": {
                **_metadata(name, uid),
                "namespace": cleanup.NAMESPACE,
                "annotations": {
                    "fleet.ai/run-dir": run_dir,
                    "fleet.ai/failure-alerts": "off",
                },
                "labels": {
                    "kueue.x-k8s.io/queue-name": "training-lq",
                    "kueue.x-k8s.io/priority-class": "q1",
                    "fleet.ai/requeue-if-preempted": "false",
                },
            },
            "spec": {
                "shutdownAfterJobFinishes": True,
                "backoffLimit": 0,
                "rayClusterSpec": {
                    "headGroupSpec": {
                        "template": {
                            "spec": {
                                "priorityClassName": "c1",
                                "containers": [
                                    {
                                        "image": image,
                                        "resources": {
                                            "requests": {"nvidia.com/gpu": 8},
                                            "limits": {"nvidia.com/gpu": 8},
                                        },
                                    }
                                ],
                            }
                        }
                    },
                    "workerGroupSpecs": [{"replicas": 0}],
                },
            },
        }

    def __call__(self, argv, **_kwargs):
        args = argv[5:]
        self.calls.append(args)
        if args == ["get", "rayjob", "--output", "json"]:
            return NS(returncode=0, stdout=json.dumps({"items": self.rows}), stderr="")
        if args[:2] == ["get", "rayjob"] and len(args) == 6:
            assert args[3:] == ["--ignore-not-found", "--output", "json"]
            self.exact_reads += 1
            value = (
                self.exact
                if self.exact
                and self.exact_reads > self.exact_after_reads
                and args[2] == self.exact["metadata"]["name"]
                else None
            )
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:2] == ["delete", "--raw"]:
            self.delete_calls += 1
        raise AssertionError(args)


def _jobs_api_prefix_guard(
    tmp_path: Path, runner: FakeJobsApiPrefixGuardCluster, **overrides: object
) -> cleanup.JobsApiPrefixGuard:
    values: dict[str, object] = {
        "context": cleanup.DEV_CONTEXT,
        "namespace": cleanup.NAMESPACE,
        "run_name_prefix": "collector-diag",
        "run_dir": "/mnt/sfs/jobs/collector-diag-v1",
        "image": "registry/image@sha256:" + "d" * 64,
        "plan_sha256": "sha256:" + "1" * 64,
        "manifest_sha256": "sha256:" + "2" * 64,
        "maximum_seconds": 1800,
        "expected_gpus": 8,
        "armed_path": tmp_path / "PREFIX_GUARD_ARMED.json",
        "binding_path": tmp_path / "EXACT_BINDING.json",
        "run": runner,
    }
    values.update(overrides)
    return cleanup.JobsApiPrefixGuard(**values)


def _creator_identity(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "jobs_api_run_name": "collector-diag-1a2b3c4d",
        "jobs_api_run_id": "00000000-0000-0000-0000-000000000302",
        "run_dir": "/mnt/sfs/jobs/collector-diag-v1",
    }
    value.update(overrides)
    return value


def test_jobs_api_prefix_guard_arms_without_deletion_then_binds_exact_creator_name(
    tmp_path,
) -> None:
    cluster = FakeJobsApiPrefixGuardCluster()
    guard = _jobs_api_prefix_guard(tmp_path, cluster)
    armed = guard.arm()
    assert armed["schema"] == cleanup.JOBS_API_PREFIX_GUARD_SCHEMA
    assert armed["status"] == "armed_non_destructive_prefix_guard"
    assert armed["prefix_collision_count_before_post"] == 0
    assert cluster.calls == [["get", "rayjob", "--output", "json"]]
    assert cluster.delete_calls == 0

    cluster.exact = cluster.rayjob()
    binding = guard.bind_exact(_creator_identity())
    assert binding["schema"] == cleanup.JOBS_API_EXACT_BINDING_SCHEMA
    assert binding["status"] == "bound_exact_uid_cleanup_not_started"
    assert binding["jobs_api_run_name"] == binding["rayjob_name"] == "collector-diag-1a2b3c4d"
    assert binding["jobs_api_run_id"] == "00000000-0000-0000-0000-000000000302"
    assert binding["rayjob_uid"] == _metadata("unused", 301)["uid"]
    assert binding["prefix_guard_sha256"] == armed["sha256"]
    # The only post-POST lookup uses the exact name returned by the creator;
    # there is no second list/prefix discovery and no delete.
    assert cluster.calls == [
        ["get", "rayjob", "--output", "json"],
        [
            "get",
            "rayjob",
            "collector-diag-1a2b3c4d",
            "--ignore-not-found",
            "--output",
            "json",
        ],
    ]
    assert cluster.delete_calls == 0


def test_jobs_api_guard_accepts_bounded_prod_one_gpu_observer(tmp_path) -> None:
    cluster = FakeJobsApiPrefixGuardCluster()
    guard = _jobs_api_prefix_guard(
        tmp_path,
        cluster,
        context=cleanup.PROD_CONTEXT,
        expected_gpus=1,
        maximum_seconds=7200,
    )
    assert guard.context == cleanup.PROD_CONTEXT
    assert guard.expected_gpus == 1
    assert guard.maximum_seconds == 7200


def test_jobs_api_exact_observer_accepts_server_suffix_after_maximum_prefix(tmp_path) -> None:
    prefix = "a" * 31
    exact_name = prefix + "-1a2b3c4d"
    binding = _jobs_api_exact_binding(
        tmp_path,
        jobs_api_run_name=exact_name,
        rayjob_name=exact_name,
    )
    observer = cleanup.JobsApiExactUidObserver(
        binding_path=binding,
        result_path=tmp_path / "result.json",
    )
    assert observer.root_name == exact_name


def test_jobs_api_exact_binding_can_resume_from_a_sealed_pre_post_guard(tmp_path) -> None:
    cluster = FakeJobsApiPrefixGuardCluster()
    _jobs_api_prefix_guard(tmp_path, cluster).arm()
    # The POST and its response handling may be a separate process.  It must
    # be able to reopen only the exact sealed guard, never rediscover a name by
    # prefix after creation.
    cluster.exact = cluster.rayjob()
    binding = _jobs_api_prefix_guard(tmp_path, cluster).bind_exact(_creator_identity())
    assert binding["rayjob_name"] == "collector-diag-1a2b3c4d"
    assert [call[:3] for call in cluster.calls] == [
        ["get", "rayjob", "--output"],
        ["get", "rayjob", "collector-diag-1a2b3c4d"],
    ]
    assert cluster.delete_calls == 0


def test_jobs_api_exact_binding_waits_only_for_the_creator_returned_name(tmp_path) -> None:
    cluster = FakeJobsApiPrefixGuardCluster()
    ticks = iter((0.0, 0.0, 0.1))
    _jobs_api_prefix_guard(
        tmp_path,
        cluster,
        bind_wait_seconds=1.0,
        sleep=lambda _seconds: None,
        monotonic=lambda: next(ticks),
    ).arm()
    cluster.exact = cluster.rayjob()
    cluster.exact_after_reads = 1
    binding = _jobs_api_prefix_guard(
        tmp_path,
        cluster,
        bind_wait_seconds=1.0,
        sleep=lambda _seconds: None,
        monotonic=lambda: 0.1,
    ).bind_exact(_creator_identity())
    assert binding["rayjob_name"] == "collector-diag-1a2b3c4d"
    assert cluster.exact_reads == 2
    assert cluster.delete_calls == 0


@pytest.mark.parametrize(
    "existing_name",
    ["collector-diag", "collector-diag-deadbeef", "collector-diag-x"],
)
def test_jobs_api_prefix_guard_rejects_any_preexisting_prefix_collision(
    tmp_path, existing_name
) -> None:
    cluster = FakeJobsApiPrefixGuardCluster()
    cluster.rows = [FakeJobsApiPrefixGuardCluster.rayjob(name=existing_name)]
    guard = _jobs_api_prefix_guard(tmp_path, cluster)
    with pytest.raises(cleanup.ObserverError, match="prefix is already in use"):
        guard.arm()
    assert not (tmp_path / "PREFIX_GUARD_ARMED.json").exists()
    assert cluster.delete_calls == 0


@pytest.mark.parametrize(
    "mutate_identity, mutate_rayjob, error",
    [
        (
            lambda identity: identity.update(jobs_api_run_name="collector-diag-not-hex"),
            lambda rayjob: None,
            "creator identity is invalid",
        ),
        (
            lambda identity: identity.update(jobs_api_run_id="not-a-uuid"),
            lambda rayjob: None,
            "creator identity is invalid",
        ),
        (
            lambda identity: None,
            lambda rayjob: rayjob["metadata"]["annotations"].pop("fleet.ai/failure-alerts"),
            "does not match",
        ),
        (
            lambda identity: None,
            lambda rayjob: rayjob["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
                "containers"
            ][0].update({"image": "registry/other@sha256:" + "e" * 64}),
            "resource binding changed",
        ),
        (
            lambda identity: None,
            lambda rayjob: rayjob["spec"]["rayClusterSpec"].update(
                {"workerGroupSpecs": [{"replicas": 1}]}
            ),
            "resource binding changed",
        ),
        (
            lambda identity: None,
            lambda rayjob: rayjob["spec"].update({"backoffLimit": 1}),
            "does not match",
        ),
    ],
)
def test_jobs_api_exact_binding_fails_closed_without_cleanup(
    tmp_path, mutate_identity, mutate_rayjob, error
) -> None:
    cluster = FakeJobsApiPrefixGuardCluster()
    guard = _jobs_api_prefix_guard(tmp_path, cluster)
    guard.arm()
    identity = _creator_identity()
    rayjob = cluster.rayjob()
    mutate_identity(identity)
    mutate_rayjob(rayjob)
    cluster.exact = rayjob
    with pytest.raises(cleanup.ObserverError, match=error):
        guard.bind_exact(identity)
    assert not (tmp_path / "EXACT_BINDING.json").exists()
    assert cluster.delete_calls == 0


def test_jobs_api_exact_binding_rejects_a_rayjob_that_predates_its_guard(tmp_path) -> None:
    cluster = FakeJobsApiPrefixGuardCluster()
    guard = _jobs_api_prefix_guard(tmp_path, cluster)
    guard.arm()
    cluster.exact = cluster.rayjob()
    cluster.exact["metadata"]["creationTimestamp"] = "2000-01-01T00:00:00Z"
    with pytest.raises(cleanup.ObserverError, match="predates"):
        guard.bind_exact(_creator_identity())
    assert not (tmp_path / "EXACT_BINDING.json").exists()
    assert cluster.delete_calls == 0


def test_jobs_api_exact_binding_requires_the_sealed_pre_post_guard(tmp_path) -> None:
    cluster = FakeJobsApiPrefixGuardCluster()
    guard = _jobs_api_prefix_guard(tmp_path, cluster)
    cluster.exact = cluster.rayjob()
    with pytest.raises(cleanup.ObserverError, match="has not been armed"):
        guard.bind_exact(_creator_identity())
    assert cluster.calls == []
    assert cluster.delete_calls == 0


def _jobs_api_exact_binding(tmp_path: Path, **overrides: object) -> Path:
    created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    value: dict[str, object] = {
        "schema": cleanup.JOBS_API_EXACT_BINDING_SCHEMA,
        "status": "bound_exact_uid_cleanup_not_started",
        "prefix_guard_sha256": "sha256:" + "a" * 64,
        "context": cleanup.DEV_CONTEXT,
        "namespace": cleanup.NAMESPACE,
        "jobs_api_run_name": "collector-diag-1a2b3c4d",
        "jobs_api_run_id": "00000000-0000-0000-0000-000000000302",
        "run_dir": "/mnt/sfs/jobs/collector-diag-v1",
        "image": "registry/image@sha256:" + "b" * 64,
        "rayjob_name": "collector-diag-1a2b3c4d",
        "rayjob_uid": _metadata("unused", 401)["uid"],
        "rayjob_created_at": created,
        "bound_at": created,
        "failure_alerts": "off",
        "maximum_seconds": 1800,
        "expected_gpus": 8,
        "cleanup_started": False,
    }
    value.update(overrides)
    sealed = _seal(value)
    path = tmp_path / "EXACT_BINDING.json"
    path.write_text(json.dumps(sealed))
    return path


def _jobs_api_release_contract(tmp_path: Path, binding: dict) -> Path:
    value = {
        "schema": cleanup.JOBS_API_RELEASE_CONTRACT_SCHEMA,
        "status": "creator_authorized_exact_uid_release",
        "binding_sha256": binding["sha256"],
        "context": binding["context"],
        "namespace": binding["namespace"],
        "jobs_api_run_name": binding["jobs_api_run_name"],
        "jobs_api_run_id": binding["jobs_api_run_id"],
        "rayjob_name": binding["rayjob_name"],
        "rayjob_uid": binding["rayjob_uid"],
        "authorized_at": binding["bound_at"],
        "release_route": "raw_rayjob_uid_precondition_v1",
    }
    path = tmp_path / "RELEASE_CONTRACT.json"
    path.write_text(json.dumps(_seal(value)))
    return path


class FakeJobsApiExactObserverCluster:
    """Fake exact-name Kubernetes reads; it has no prefix or peer route."""

    def __init__(self, binding: dict, *, auto_release: bool = True) -> None:
        self.binding = binding
        self.auto_release = auto_release
        self.root_reads = 0
        self.deleted = False
        self.calls: list[tuple[list[str], object]] = []
        self.root_uid = binding["rayjob_uid"]
        self.cluster_uid = _metadata("unused", 402)["uid"]
        self.workload_uid = _metadata("unused", 403)["uid"]
        self.pod_uid = _metadata("unused", 404)["uid"]
        self.replacement_root: dict | None = None
        self.bad_workload_owner = False

    def _owner(self, *, kind: str, name: str, uid: str) -> list[dict]:
        return [{"kind": kind, "name": name, "uid": uid, "controller": True}]

    def _root(self, *, uid: str | None = None) -> dict:
        return {
            "kind": "RayJob",
            "metadata": {
                **_metadata(self.binding["rayjob_name"], 401),
                "uid": uid or self.root_uid,
                "creationTimestamp": self.binding["rayjob_created_at"],
                "namespace": cleanup.NAMESPACE,
                "annotations": {
                    "fleet.ai/run-dir": self.binding["run_dir"],
                    "fleet.ai/failure-alerts": "off",
                },
            },
            "status": {"jobStatus": "SUCCEEDED", "rayClusterName": "collector-cluster"},
        }

    def _workload(self) -> dict:
        owner_uid = (
            self.root_uid if not self.bad_workload_owner else _metadata("unused", 499)["uid"]
        )
        return {
            "kind": "Workload",
            "metadata": {
                **_metadata("collector-workload", 403),
                "namespace": cleanup.NAMESPACE,
                "labels": {"kueue.x-k8s.io/job-uid": self.root_uid},
                "ownerReferences": self._owner(
                    kind="RayJob", name=self.binding["rayjob_name"], uid=owner_uid
                ),
            },
        }

    def _cluster(self) -> dict:
        return {
            "kind": "RayCluster",
            "metadata": {
                **_metadata("collector-cluster", 402),
                "namespace": cleanup.NAMESPACE,
                "ownerReferences": self._owner(
                    kind="RayJob", name=self.binding["rayjob_name"], uid=self.root_uid
                ),
            },
        }

    def _pod(self) -> dict:
        return {
            "kind": "Pod",
            "metadata": {
                **_metadata("collector-pod", 404),
                "namespace": cleanup.NAMESPACE,
                "ownerReferences": self._owner(
                    kind="RayCluster", name="collector-cluster", uid=self.cluster_uid
                ),
            },
            "spec": {
                "containers": [
                    {
                        "name": "trainer",
                        "image": self.binding["image"],
                        "resources": {
                            "requests": {"nvidia.com/gpu": self.binding["expected_gpus"]},
                            "limits": {"nvidia.com/gpu": self.binding["expected_gpus"]},
                        },
                    }
                ]
            },
            "status": {
                "containerStatuses": [
                    {
                        "name": "trainer",
                        "imageID": "registry/image@sha256:" + "b" * 64,
                    }
                ]
            },
        }

    def __call__(self, argv, **kwargs):
        args = argv[5:]
        self.calls.append((args, kwargs.get("input")))
        root_absent = self.deleted or (self.auto_release and self.root_reads > 1)
        if args[:3] == ["get", "rayjob", self.binding["rayjob_name"]]:
            self.root_reads += 1
            if self.replacement_root is not None and self.root_reads > 1:
                return NS(returncode=0, stdout=json.dumps(self.replacement_root), stderr="")
            value = None if root_absent else self._root()
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:3] == ["get", "workload", "--selector"]:
            assert args[3] == "kueue.x-k8s.io/job-uid=" + self.root_uid
            return NS(returncode=0, stdout=json.dumps({"items": [self._workload()]}), stderr="")
        if args[:3] == ["get", "raycluster", "collector-cluster"]:
            value = None if root_absent else self._cluster()
            return NS(returncode=0, stdout=json.dumps(value) if value else "", stderr="")
        if args[:3] == ["get", "pod", "--selector"]:
            assert args[3] == "ray.io/cluster=collector-cluster"
            return NS(returncode=0, stdout=json.dumps({"items": [self._pod()]}), stderr="")
        if args[:3] in (
            ["get", "workload", "collector-workload"],
            ["get", "pod", "collector-pod"],
        ):
            return NS(returncode=0, stdout="", stderr="")
        if args[:2] == ["delete", "--raw"]:
            assert args[2] == (
                f"/apis/ray.io/v1/namespaces/{cleanup.NAMESPACE}/rayjobs/"
                + self.binding["rayjob_name"]
            )
            assert json.loads(kwargs["input"]) == {
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "propagationPolicy": "Foreground",
                "preconditions": {"uid": self.root_uid},
            }
            self.deleted = True
            return NS(returncode=0, stdout="", stderr="")
        raise AssertionError(args)


def _jobs_api_exact_observer(
    tmp_path: Path,
    cluster: FakeJobsApiExactObserverCluster,
    *,
    release_contract_path: Path | None = None,
) -> cleanup.JobsApiExactUidObserver:
    return cleanup.JobsApiExactUidObserver(
        binding_path=tmp_path / "EXACT_BINDING.json",
        result_path=tmp_path / "EXACT_OBSERVER_RESULT.json",
        release_contract_path=release_contract_path,
        poll_seconds=0.001,
        run=cluster,
    )


def test_jobs_api_exact_observer_accepts_bounded_prod_one_gpu_binding(tmp_path) -> None:
    _jobs_api_exact_binding(
        tmp_path,
        context=cleanup.PROD_CONTEXT,
        expected_gpus=1,
        maximum_seconds=7200,
    )
    binding = json.loads((tmp_path / "EXACT_BINDING.json").read_text())
    observer = _jobs_api_exact_observer(tmp_path, FakeJobsApiExactObserverCluster(binding))
    assert observer.context == cleanup.PROD_CONTEXT
    assert observer.binding["expected_gpus"] == 1
    assert observer.maximum_seconds == 7200
    result = observer.run()
    assert result["release_confirmed"] is True
    assert result["runtime_image_identity_complete"] is True
    assert result["pods"][0]["gpus"] == 1


def test_jobs_api_exact_uid_observer_releases_only_proven_owned_children(tmp_path) -> None:
    binding_path = _jobs_api_exact_binding(tmp_path)
    binding = json.loads(binding_path.read_text())
    cluster = FakeJobsApiExactObserverCluster(binding)
    result = _jobs_api_exact_observer(tmp_path, cluster).run()
    assert result["status"] == "released_after_terminal"
    assert result["release_confirmed"] is True
    assert result["terminal_status"] == "Succeeded"
    assert result["peak_gpus"] == 8
    assert result["requested_image"] == binding["image"]
    assert result["runtime_image_identity_complete"] is True
    assert result["cleanup_status"] == "not_authorized"
    assert result["private_logs_read"] is False
    assert [entry["uid"] for entry in result["workloads"]] == [cluster.workload_uid]
    assert [entry["uid"] for entry in result["rayclusters"]] == [cluster.cluster_uid]
    assert [entry["uid"] for entry in result["pods"]] == [cluster.pod_uid]
    assert result["pods"] == [
        {
            "name": "collector-pod",
            "uid": cluster.pod_uid,
            "gpus": 8,
            "runtime_image_id": "registry/image@sha256:" + "b" * 64,
            "runtime_image_digest": "sha256:" + "b" * 64,
        }
    ]
    commands = [call[0] for call in cluster.calls]
    assert ["get", "rayjob", "--output", "json"] not in commands
    assert not any(command[:2] == ["delete", "--raw"] for command in commands)


def test_jobs_api_exact_uid_observer_reports_owner_mismatch_without_cleanup(tmp_path) -> None:
    binding_path = _jobs_api_exact_binding(tmp_path)
    binding = json.loads(binding_path.read_text())
    cluster = FakeJobsApiExactObserverCluster(binding)
    cluster.bad_workload_owner = True
    result = _jobs_api_exact_observer(tmp_path, cluster).run()
    assert result["status"] == "release_uncertain"
    assert result["release_confirmed"] is False
    assert result["cleanup_requested"] is False
    assert not any(call[0][:2] == ["delete", "--raw"] for call in cluster.calls)


def test_jobs_api_exact_uid_observer_requires_matching_runtime_image_id(tmp_path) -> None:
    binding_path = _jobs_api_exact_binding(tmp_path)
    binding = json.loads(binding_path.read_text())
    cluster = FakeJobsApiExactObserverCluster(binding)
    pod = cluster._pod()
    pod["status"]["containerStatuses"][0]["imageID"] = ""
    cluster._pod = lambda: pod
    result = _jobs_api_exact_observer(tmp_path, cluster).run()
    assert result["status"] == "release_uncertain"
    assert result["reason"] == "runtime_image_identity_not_observed"
    assert result["release_confirmed"] is False
    assert result["runtime_image_identity_complete"] is False

    other = tmp_path / "other"
    other.mkdir()
    binding_path = _jobs_api_exact_binding(other)
    binding = json.loads(binding_path.read_text())
    cluster = FakeJobsApiExactObserverCluster(binding)
    pod = cluster._pod()
    pod["status"]["containerStatuses"][0]["imageID"] = "registry/image@sha256:" + "c" * 64
    cluster._pod = lambda: pod
    result = _jobs_api_exact_observer(other, cluster).run()
    assert result["status"] == "release_uncertain"
    assert result["reason"] == "observer_error"
    assert result["release_confirmed"] is False


def test_jobs_api_exact_uid_observer_raw_deletes_only_with_creator_contract(tmp_path) -> None:
    binding_path = _jobs_api_exact_binding(tmp_path)
    binding = json.loads(binding_path.read_text())
    cluster = FakeJobsApiExactObserverCluster(binding, auto_release=False)
    contract = _jobs_api_release_contract(tmp_path, binding)
    result = _jobs_api_exact_observer(tmp_path, cluster, release_contract_path=contract).run()
    assert result["status"] == "released_after_terminal"
    assert result["cleanup_requested"] is True
    assert result["cleanup_status"] == "requested_exact_uid_precondition"
    assert sum(call[0][:2] == ["delete", "--raw"] for call in cluster.calls) == 1


def test_jobs_api_exact_uid_observer_rejects_same_name_uid_reuse(tmp_path) -> None:
    binding_path = _jobs_api_exact_binding(tmp_path)
    binding = json.loads(binding_path.read_text())
    cluster = FakeJobsApiExactObserverCluster(binding, auto_release=False)
    cluster.replacement_root = cluster._root(uid=_metadata("unused", 498)["uid"])
    result = _jobs_api_exact_observer(tmp_path, cluster).run()
    assert result["status"] == "release_uncertain"
    assert result["release_confirmed"] is False
    assert result["cleanup_requested"] is False
    assert not any(call[0][:2] == ["delete", "--raw"] for call in cluster.calls)


def test_jobs_api_exact_uid_observer_never_claims_release_without_a_bound_cluster(tmp_path) -> None:
    binding_path = _jobs_api_exact_binding(tmp_path)
    binding = json.loads(binding_path.read_text())
    cluster = FakeJobsApiExactObserverCluster(binding)
    root = cluster._root()
    root["status"] = {"jobStatus": "SUCCEEDED"}
    cluster._root = lambda **_kwargs: root
    # The root disappears after the first observation, but no RayCluster UID
    # was ever bound.  A successful release claim would be unsafe.
    cluster.auto_release = True
    result = _jobs_api_exact_observer(tmp_path, cluster).run()
    assert result["status"] == "release_uncertain"
    assert result["reason"] == "bound_root_absent_without_terminal_inventory"
    assert result["raycluster_identity_observed"] is False
    assert result["release_confirmed"] is False


def test_jobs_api_exact_uid_observer_queue_wait_has_no_active_deadline(tmp_path) -> None:
    binding_path = _jobs_api_exact_binding(
        tmp_path,
        rayjob_created_at="2000-01-01T00:00:00Z",
        bound_at="2000-01-01T00:00:00Z",
        maximum_seconds=1,
    )
    binding = json.loads(binding_path.read_text())
    cluster = FakeJobsApiExactObserverCluster(binding, auto_release=False)
    root = cluster._root()
    root["status"] = {"jobStatus": "PENDING"}
    observer = _jobs_api_exact_observer(tmp_path, cluster)

    cluster_name = observer._validate_root(root)
    observer._observe_owned_children(cluster_name)

    assert cluster_name == ""
    assert observer.allocated_at is None
    assert observer.deadline_at is None
    assert observer.cleanup_requested is False
    assert not any(call[0][:2] == ["delete", "--raw"] for call in cluster.calls)


def test_jobs_api_exact_uid_observer_deadline_is_bound_to_gpu_pod_creation(tmp_path) -> None:
    binding_path = _jobs_api_exact_binding(
        tmp_path,
        rayjob_created_at="2000-01-01T00:00:00Z",
        bound_at="2000-01-01T00:00:00Z",
        maximum_seconds=1,
    )
    binding = json.loads(binding_path.read_text())
    cluster = FakeJobsApiExactObserverCluster(binding, auto_release=False)
    root = cluster._root()
    root["status"] = {
        "jobStatus": "RUNNING",
        "rayClusterName": "collector-cluster",
    }
    cluster._root = lambda **_kwargs: root
    pod = cluster._pod()
    pod["metadata"]["creationTimestamp"] = "2000-01-01T00:00:00Z"
    cluster._pod = lambda: pod
    result = _jobs_api_exact_observer(tmp_path, cluster).run()
    assert result["status"] == "release_uncertain"
    assert result["reason"] == "allocation_bound_deadline_elapsed"
    assert result["allocated_at"] == "2000-01-01T00:00:00Z"
    assert result["deadline_at"] == "2000-01-01T00:00:01Z"
    assert result["cleanup_requested"] is False
    assert not any(call[0][:2] == ["delete", "--raw"] for call in cluster.calls)
