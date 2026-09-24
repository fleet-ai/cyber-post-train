"""Offline tests for the one-cell task-quality CPU Job."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from evals.fleet import rollout_worker
from evals.fleet import task_quality_qualification as qualification
from evals.fleet import task_quality_qualification_job as job
from evals.fleet import task_quality_qualification_job_entry as entry

ROOT = Path(__file__).resolve().parents[1]
CANARY = ROOT / "configs/qualification/fleet-blackbox-qa33-zero-model-canary-20260924-v1.json"
SOURCE_COMMIT = "a" * 40
SOURCE_TREE = "b" * 40
MAIN_COMMIT = "c" * 40
JOB_UID = "11111111-1111-4111-8111-111111111111"
CONFIG_UID = "22222222-2222-4222-8222-222222222222"
SECRET_UID = "33333333-3333-4333-8333-333333333333"
POD_UID = "44444444-4444-4444-8444-444444444444"
WORKLOAD_UID = "55555555-5555-4555-8555-555555555555"


def _source() -> dict[str, Any]:
    return {
        "git_commit": SOURCE_COMMIT,
        "git_tree": SOURCE_TREE,
        "origin_main_commit": MAIN_COMMIT,
        "merged_to_origin_main": True,
        "controller_path": "evals/fleet/task_quality_qualification.py",
        "controller_file_sha256": job.file_sha256(
            ROOT / "evals/fleet/task_quality_qualification.py"
        ),
    }


def _plan() -> dict[str, Any]:
    canary = json.loads(CANARY.read_text())
    candidate = canary["candidate"]
    return qualification.sealed(
        {
            "schema": qualification.PLAN_SCHEMA,
            "wave_id": "qa33-one-canary-v1",
            "source": _source(),
            "selection": {
                "exact_task_identity": {
                    "task_key": candidate["task_key"],
                    "task_version_id": candidate["task_version_id"],
                }
            },
            "execution": {
                "concurrency": 1,
                "model_calls": 0,
                "external_mutations_authorized": True,
            },
            "tasks": [
                {
                    "task_key": candidate["task_key"],
                    "task_version_id": candidate["task_version_id"],
                }
            ],
        }
    )


@pytest.fixture()
def packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    plan = tmp_path / "PLAN.json"
    plan.write_text(json.dumps(_plan()))
    canary = json.loads(CANARY.read_text())
    for label, relative in job.CANARY_SOURCE_INPUTS.items():
        canary["inputs"][label]["file_sha256"] = job.file_sha256(ROOT / relative)
    canary["sha256"] = job.canonical_digest(
        {key: value for key, value in canary.items() if key != "sha256"}
    )
    canary_path = tmp_path / "CANARY.json"
    canary_path.write_text(json.dumps(canary))

    def fake_git(_repo: Path, *args: str) -> str:
        values = {
            ("status", "--porcelain"): "",
            ("rev-parse", "HEAD"): SOURCE_COMMIT,
            ("rev-parse", "HEAD^{tree}"): SOURCE_TREE,
        }
        return values[args]

    monkeypatch.setattr(job, "git", fake_git)
    monkeypatch.setattr(
        job,
        "merge_witness",
        lambda _repo, _commit, refresh: {
            "canonical_remote": "https://github.com/fleet-ai/cyber-post-train.git",
            "source_commit": SOURCE_COMMIT,
            "observed_main_commit": MAIN_COMMIT,
            "is_ancestor": True,
        },
    )
    result = job.prepare_packet(
        repo_root=ROOT,
        plan_path=plan,
        canary_path=canary_path,
        output=tmp_path / "packet",
        expected_source_commit=SOURCE_COMMIT,
    )
    assert result["external_mutations"] == 0
    return Path(result["packet"])


class FakeCluster:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.previews = 0
        self.creates = 0
        self.preview_mutator = None
        self.create_mutator = None
        self.create_raises = False
        self.job: dict[str, Any] | None = None
        self.config_map: dict[str, Any] | None = None
        self.secret: dict[str, Any] | None = None
        self.pods: list[dict[str, Any]] = []
        self.workloads: list[dict[str, Any]] = []
        self.deletes: list[tuple[str, str, str, str]] = []

    def list(
        self,
        resource: str,
        namespace: str,
        *,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ) -> dict[str, Any]:
        assert namespace == job.NAMESPACE
        self.events.append("list:" + resource)
        values = {
            "jobs.batch": [] if self.job is None else [self.job],
            "configmaps": [] if self.config_map is None else [self.config_map],
            "secrets": [] if self.secret is None else [self.secret],
            "pods": self.pods,
            "workloads.kueue.x-k8s.io": self.workloads,
        }
        return {"apiVersion": "v1", "kind": "List", "items": copy.deepcopy(values[resource])}

    def get_optional(self, resource: str, namespace: str, name: str) -> dict[str, Any] | None:
        assert namespace == job.NAMESPACE
        value: dict[str, Any] | None
        if resource == "jobs.batch":
            value = self.job
        elif resource == "configmaps":
            value = self.config_map
        elif resource == "secrets":
            value = self.secret
        else:
            rows = self.pods if resource == "pods" else self.workloads
            value = next((row for row in rows if row["metadata"]["name"] == name), None)
        if value is None or value["metadata"]["name"] != name:
            return None
        return copy.deepcopy(value)

    def server_dry_run(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        assert namespace == job.NAMESPACE
        self.previews += 1
        self.events.append(f"preview:{self.previews}")
        value = copy.deepcopy(bundle)
        if self.preview_mutator is not None:
            self.preview_mutator(value, self.previews)
        return value

    def create_once(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        assert namespace == job.NAMESPACE
        self.creates += 1
        self.events.append("create")
        rows = copy.deepcopy(bundle["items"])
        self.config_map = next(row for row in rows if row["kind"] == "ConfigMap")
        self.secret = next(row for row in rows if row["kind"] == "Secret")
        self.job = next(row for row in rows if row["kind"] == "Job")
        for value, uid, rv in (
            (self.job, JOB_UID, "101"),
            (self.config_map, CONFIG_UID, "102"),
            (self.secret, SECRET_UID, "103"),
        ):
            value["metadata"].update(uid=uid, resourceVersion=rv)
        if self.create_mutator is not None:
            self.create_mutator(rows)
        if self.create_raises:
            raise RuntimeError("response lost")
        return {"apiVersion": "v1", "kind": "List", "items": rows}

    def delete_uid(
        self,
        resource: str,
        namespace: str,
        name: str,
        uid: str,
        resource_version: str,
        *,
        confirmed_live: bool,
    ) -> dict[str, Any]:
        assert namespace == job.NAMESPACE and confirmed_live is True
        self.deletes.append((resource, name, uid, resource_version))
        if resource == "jobs.batch":
            self.job = None
            self.pods = []
            self.workloads = []
        elif resource == "configmaps":
            self.config_map = None
        elif resource == "secrets":
            self.secret = None
        return {"apiVersion": "v1", "kind": "Status", "status": "Success"}

    def terminal(self, *, failed: bool = False) -> None:
        assert self.job is not None
        self.job["metadata"]["resourceVersion"] = "201"
        self.config_map["metadata"]["resourceVersion"] = "202"
        self.secret["metadata"]["resourceVersion"] = "203"
        self.job["status"] = {
            "active": 0,
            "failed" if failed else "succeeded": 1,
            "conditions": [{"type": "Failed" if failed else "Complete", "status": "True"}],
        }
        owner = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": self.job["metadata"]["name"],
            "uid": JOB_UID,
            "controller": True,
        }
        self.pods = [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "qa33-pod",
                    "namespace": job.NAMESPACE,
                    "uid": POD_UID,
                    "ownerReferences": [owner],
                },
                "status": {"phase": "Succeeded"},
            }
        ]
        self.workloads = [
            {
                "apiVersion": "kueue.x-k8s.io/v1beta1",
                "kind": "Workload",
                "metadata": {
                    "name": "qa33-workload",
                    "namespace": job.NAMESPACE,
                    "uid": WORKLOAD_UID,
                    "ownerReferences": [owner],
                },
            }
        ]


def _authorization(packet: Path, tmp_path: Path) -> Path:
    path = tmp_path / "authorization.json"
    job.write_launch_authorization(packet, path, root_authorization_id="root-authorization-test-v1")
    return path


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _reconcile(
    packet: Path, *, cluster: FakeCluster, journal: Path, **kwargs: Any
) -> dict[str, Any]:
    clock = FakeClock()
    return job.reconcile_uncertain(
        packet,
        cluster=cluster,
        journal=journal,
        timeout_seconds=60,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        **kwargs,
    )


def test_runtime_authority_does_not_drift() -> None:
    assert qualification.RUNTIME_AUTHORITY == rollout_worker.AUTHORITY


def test_package_is_cpu_only_alert_suppressed_and_keeps_plan_private(packet: Path) -> None:
    package = job.build_package(packet)
    root = package.job
    assert root["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert root["spec"]["activeDeadlineSeconds"] == 1800
    assert root["spec"]["backoffLimit"] == 0
    assert root["spec"]["ttlSecondsAfterFinished"] == 3600
    assert root["spec"]["template"]["spec"]["priorityClassName"] == "c1"
    job._assert_zero_accelerators(root["spec"]["template"]["spec"])  # noqa: SLF001
    assert "plan.json" not in package.config_map["data"]
    assert "plan.json" in package.secret["data"]


@pytest.mark.parametrize("ttl", [None, 3599, 3601])
def test_job_missing_or_wrong_terminal_ttl_is_rejected(packet: Path, ttl: int | None) -> None:
    package = job.build_package(packet)
    if ttl is None:
        package.job["spec"].pop("ttlSecondsAfterFinished")
    else:
        package.job["spec"]["ttlSecondsAfterFinished"] = ttl
    with pytest.raises(job.QualificationJobError, match="safety contract"):
        job._assert_exact_job(package.job, package.packet)  # noqa: SLF001


@pytest.mark.parametrize(
    "location", [("containers", 0, "requests"), ("initContainers", 0, "limits")]
)
def test_accelerator_in_any_container_is_rejected(
    packet: Path, location: tuple[str, int, str]
) -> None:
    package = job.build_package(packet)
    pod = package.job["spec"]["template"]["spec"]
    if location[0] == "initContainers":
        pod["initContainers"] = [copy.deepcopy(pod["containers"][0])]
    pod[location[0]][location[1]]["resources"][location[2]]["nvidia.com/gpu"] = "1"
    with pytest.raises(job.QualificationJobError, match="accelerator|unsafe"):
        job._assert_zero_accelerators(pod)  # noqa: SLF001


@pytest.mark.parametrize("resource", ["nvidia.com/mig-1g.10gb", "aws.amazon.com/neuron"])
def test_any_extended_container_resource_is_rejected(packet: Path, resource: str) -> None:
    pod = job.build_package(packet).job["spec"]["template"]["spec"]
    pod["containers"][0]["resources"]["requests"][resource] = "1"
    with pytest.raises(job.QualificationJobError, match="accelerator"):
        job._assert_zero_accelerators(pod)  # noqa: SLF001


def test_pod_overhead_accelerator_is_rejected(packet: Path) -> None:
    pod = job.build_package(packet).job["spec"]["template"]["spec"]
    pod["overhead"] = {"nvidia.com/gpu": "1"}
    with pytest.raises(job.QualificationJobError, match="overhead"):
        job._assert_zero_accelerators(pod)  # noqa: SLF001


@pytest.mark.parametrize("field", ["lifecycle", "envFrom"])
def test_server_preview_cannot_inject_private_plan_readers(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    cluster = FakeCluster()

    def mutate(value: dict[str, Any], _preview: int) -> None:
        root = next(row for row in value["items"] if row["kind"] == "Job")
        container = root["spec"]["template"]["spec"]["containers"][0]
        container[field] = (
            {"postStart": {"exec": {"command": ["cat", "/private/plan.json"]}}}
            if field == "lifecycle"
            else [{"secretRef": {"name": "other"}}]
        )

    cluster.preview_mutator = mutate
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    with pytest.raises(job.QualificationJobError, match="unreviewed fields"):
        job.launch_once(
            packet,
            authorization_path=_authorization(packet, tmp_path),
            repo_root=ROOT,
            cluster=cluster,
            journal=tmp_path / "create.jsonl",
        )
    assert cluster.creates == 0


def _reseal_packet(packet: Path, mutate: Any) -> None:
    value = json.loads(packet.read_text())
    mutate(value)
    value["sha256"] = job.canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    packet.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["runtime"].__setitem__(
            "image", "example.invalid/other@sha256:" + "0" * 64
        ),
        lambda value: value["execution"].__setitem__("launch_authorized", True),
        lambda value: value["runtime"].__setitem__(
            "fleet_secret", {"name": "other", "key": "FLEET_API_KEY"}
        ),
    ],
)
def test_resealed_runtime_or_execution_drift_is_rejected(packet: Path, mutate: Any) -> None:
    _reseal_packet(packet, mutate)
    with pytest.raises(job.QualificationJobError, match="runtime|execution"):
        job.load_packet(packet)


def test_launch_orders_two_previews_absence_intent_and_one_create(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    monkeypatch.setattr(
        job,
        "merge_witness",
        lambda *_args, **_kwargs: {
            "canonical_remote": "https://github.com/fleet-ai/cyber-post-train.git",
            "source_commit": SOURCE_COMMIT,
            "observed_main_commit": MAIN_COMMIT,
            "is_ancestor": True,
        },
    )
    journal = tmp_path / "create.jsonl"
    result = job.launch_once(
        packet,
        authorization_path=_authorization(packet, tmp_path),
        repo_root=ROOT,
        cluster=cluster,
        journal=journal,
    )
    assert cluster.previews == 2 and cluster.creates == 1
    assert cluster.events[-1] == "create"
    assert result["job_uid"] == JOB_UID and result["job_resource_version"] == "101"
    assert result["config_map_uid"] == CONFIG_UID
    assert result["secret_uid"] == SECRET_UID
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[0]["state"] == "KUBERNETES_CREATE_INTENT_DO_NOT_RETRY"
    assert rows[1]["state"] == "KUBERNETES_CREATE_RESPONSE"


def test_authorization_cannot_be_reused_with_another_journal(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    with pytest.raises(job.QualificationJobError, match="authorization-bound journal"):
        job.launch_once(
            packet,
            authorization_path=_authorization(packet, tmp_path),
            repo_root=ROOT,
            cluster=cluster,
            journal=tmp_path / "different.jsonl",
        )
    assert cluster.previews == cluster.creates == 0


def test_two_identically_unsafe_previews_cannot_pass(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()

    def mutate(value: dict[str, Any], _preview: int) -> None:
        root = next(row for row in value["items"] if row["kind"] == "Job")
        root["metadata"]["annotations"]["fleet.ai/failure-alerts"] = "on"

    cluster.preview_mutator = mutate
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    with pytest.raises(job.QualificationJobError, match="safety contract"):
        job.launch_once(
            packet,
            authorization_path=_authorization(packet, tmp_path),
            repo_root=ROOT,
            cluster=cluster,
            journal=tmp_path / "create.jsonl",
        )
    assert cluster.creates == 0


def test_live_create_mutation_after_previews_is_rejected(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    clock = FakeClock()

    def mutate(rows: list[dict[str, Any]]) -> None:
        root = next(row for row in rows if row["kind"] == "Job")
        root["spec"]["template"]["spec"]["containers"][0]["command"] = ["false"]

    cluster.create_mutator = mutate
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    journal = tmp_path / "create.jsonl"
    with pytest.raises(job.QualificationJobError, match="server preview.*released"):
        job.launch_once(
            packet,
            authorization_path=_authorization(packet, tmp_path),
            repo_root=ROOT,
            cluster=cluster,
            journal=journal,
            cleanup_timeout_seconds=60,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    assert cluster.creates == 1
    assert cluster.job is cluster.config_map is cluster.secret is None
    assert [row[0] for row in cluster.deletes] == ["jobs.batch", "configmaps", "secrets"]
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["state"] for row in rows] == [
        "KUBERNETES_CREATE_INTENT_DO_NOT_RETRY",
        "PARTIAL_CREATE_CLEANUP_INTENT",
        "KUBERNETES_CREATE_RECONCILED_PARTIAL_RELEASED",
    ]
    assert rows[1]["validation_mode"] == job.MUTATED_CREATE_RESPONSE_CLEANUP
    assert rows[2]["kubernetes_release_confirmed"] is True
    assert rows[2]["environment_cleanup_proven"] is False


def test_live_create_mutation_cleanup_resumes_without_revalidating_preview(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    clock = FakeClock()

    def mutate(rows: list[dict[str, Any]]) -> None:
        root = next(row for row in rows if row["kind"] == "Job")
        root["spec"]["template"]["spec"]["containers"][0]["command"] = ["false"]

    cluster.create_mutator = mutate
    original_delete = cluster.delete_uid
    attempts = 0

    def interrupt_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("local cleanup observer interrupted")
        return original_delete(*args, **kwargs)

    cluster.delete_uid = interrupt_once  # type: ignore[method-assign]
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    journal = tmp_path / "create.jsonl"
    with pytest.raises(job.QualificationJobError, match="uncertain"):
        job.launch_once(
            packet,
            authorization_path=_authorization(packet, tmp_path),
            repo_root=ROOT,
            cluster=cluster,
            journal=journal,
            cleanup_timeout_seconds=60,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    cluster.delete_uid = original_delete  # type: ignore[method-assign]
    result = _reconcile(packet, cluster=cluster, journal=journal)
    assert result["state"] == "KUBERNETES_CREATE_RECONCILED_PARTIAL_RELEASED"
    assert cluster.job is cluster.config_map is cluster.secret is None
    assert cluster.creates == 1


def _source_witness() -> dict[str, Any]:
    return {
        "canonical_remote": "https://github.com/fleet-ai/cyber-post-train.git",
        "source_commit": SOURCE_COMMIT,
        "observed_main_commit": MAIN_COMMIT,
        "is_ancestor": True,
    }


def test_uncertain_create_is_never_retried(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    cluster.create_raises = True
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    journal = tmp_path / "create.jsonl"
    authorization = _authorization(packet, tmp_path)
    with pytest.raises(job.QualificationJobError, match="uncertain"):
        job.launch_once(
            packet,
            authorization_path=authorization,
            repo_root=ROOT,
            cluster=cluster,
            journal=journal,
        )
    assert cluster.creates == 1
    with pytest.raises(job.QualificationJobError, match="journal"):
        job.launch_once(
            packet,
            authorization_path=authorization,
            repo_root=ROOT,
            cluster=cluster,
            journal=journal,
        )
    assert cluster.creates == 1


def test_lost_mutated_create_requires_confirmation_then_releases_exact_uids(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()

    def mutate(rows: list[dict[str, Any]]) -> None:
        root = next(row for row in rows if row["kind"] == "Job")
        root["spec"]["template"]["spec"]["containers"][0]["command"] = ["false"]

    cluster.create_mutator = mutate
    cluster.create_raises = True
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    journal = tmp_path / "create.jsonl"
    with pytest.raises(job.QualificationJobError, match="uncertain"):
        job.launch_once(
            packet,
            authorization_path=_authorization(packet, tmp_path),
            repo_root=ROOT,
            cluster=cluster,
            journal=journal,
        )
    with pytest.raises(job.QualificationJobError, match="explicit exact-UID cleanup"):
        _reconcile(packet, cluster=cluster, journal=journal)
    result = _reconcile(
        packet,
        cluster=cluster,
        journal=journal,
        confirm_partial_cleanup=True,
    )
    assert result["state"] == "KUBERNETES_CREATE_RECONCILED_PARTIAL_RELEASED"
    assert cluster.job is cluster.config_map is cluster.secret is None
    assert [row[0] for row in cluster.deletes] == ["jobs.batch", "configmaps", "secrets"]
    assert cluster.creates == 1


def _uncertain_create(
    packet: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cluster: FakeCluster,
) -> Path:
    cluster.create_raises = True
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    journal = tmp_path / "create.jsonl"
    with pytest.raises(job.QualificationJobError, match="uncertain"):
        job.launch_once(
            packet,
            authorization_path=_authorization(packet, tmp_path),
            repo_root=ROOT,
            cluster=cluster,
            journal=journal,
        )
    assert cluster.creates == 1
    return journal


def test_uncertain_create_reconciles_three_exact_objects_without_retry(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    result = _reconcile(packet, cluster=cluster, journal=journal)
    assert result["state"] == "KUBERNETES_CREATE_RECONCILED"
    assert result["job_uid"] == JOB_UID
    assert cluster.creates == 1 and cluster.deletes == []


def test_intent_only_crash_is_reconciled_without_retry(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    intent = journal.read_text().splitlines()[0]
    journal.write_text(intent + "\n")
    result = _reconcile(packet, cluster=cluster, journal=journal)
    assert result["state"] == "KUBERNETES_CREATE_RECONCILED"
    assert cluster.creates == 1


def test_torn_final_journal_row_does_not_hide_fsynced_create_intent(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    journal.write_bytes(journal.read_bytes() + b'{"state":"TORN')
    result = _reconcile(packet, cluster=cluster, journal=journal)
    assert result["state"] == "KUBERNETES_CREATE_RECONCILED"
    assert journal.read_bytes().endswith(b"\n")


def test_reconcile_rejects_live_mutation_from_reviewed_preview(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    assert cluster.job is not None
    cluster.job["spec"]["template"]["spec"]["containers"][0]["command"] = ["false"]
    with pytest.raises(job.QualificationJobError, match="server preview"):
        _reconcile(packet, cluster=cluster, journal=journal)
    assert cluster.creates == 1


def test_uncertain_create_reconciles_complete_absence_without_retry(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    cluster.job = cluster.config_map = cluster.secret = None
    result = _reconcile(packet, cluster=cluster, journal=journal)
    assert result == {
        "state": "KUBERNETES_CREATE_RECONCILED_ABSENT",
        "submitted": "unknown",
        "kubernetes_release_confirmed": False,
        "environment_cleanup_proven": False,
        "release_confirmed": False,
        "possible_external_environment_leak": True,
        "gpus": 0,
    }
    assert cluster.creates == 1


def test_uncertain_absence_waits_long_enough_to_see_a_late_list_member(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    late_config = copy.deepcopy(cluster.config_map)
    cluster.job = cluster.config_map = cluster.secret = None
    reads = 0
    original_get = cluster.get_optional

    def late_get(resource: str, namespace: str, name: str) -> dict[str, Any] | None:
        nonlocal reads
        if resource == "configmaps":
            reads += 1
            if reads == 2:
                cluster.config_map = copy.deepcopy(late_config)
        return original_get(resource, namespace, name)

    monkeypatch.setattr(cluster, "get_optional", late_get)
    with pytest.raises(job.QualificationJobError, match="partial List create"):
        _reconcile(packet, cluster=cluster, journal=journal)
    assert reads > 2


def test_partial_create_requires_confirmation_then_releases_exact_uid(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    cluster.job = cluster.secret = None
    with pytest.raises(job.QualificationJobError, match="explicit exact-UID cleanup"):
        _reconcile(packet, cluster=cluster, journal=journal)
    assert cluster.deletes == []
    result = _reconcile(
        packet,
        cluster=cluster,
        journal=journal,
        confirm_partial_cleanup=True,
    )
    assert result["state"] == "KUBERNETES_CREATE_RECONCILED_PARTIAL_RELEASED"
    assert result["release_confirmed"] is False
    assert result["possible_external_environment_leak"] is True
    assert cluster.deletes == [
        (
            "configmaps",
            job.build_package(packet).packet.value["config_map_name"],
            CONFIG_UID,
            "102",
        )
    ]
    assert cluster.creates == 1


def test_partial_cleanup_resumes_from_sealed_intent_after_interruption(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    cluster.job = cluster.secret = None
    original_delete = cluster.delete_uid
    attempts = 0

    def interrupt_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("local observer interrupted")
        return original_delete(*args, **kwargs)

    monkeypatch.setattr(cluster, "delete_uid", interrupt_once)
    with pytest.raises(RuntimeError, match="interrupted"):
        _reconcile(
            packet,
            cluster=cluster,
            journal=journal,
            confirm_partial_cleanup=True,
        )
    monkeypatch.setattr(cluster, "delete_uid", original_delete)
    result = _reconcile(packet, cluster=cluster, journal=journal)
    assert result["release_confirmed"] is False
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["state"] for row in rows].count("PARTIAL_CREATE_CLEANUP_INTENT") == 1
    assert cluster.creates == 1


def test_partial_create_releases_root_before_private_source(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    cluster.config_map = None
    result = _reconcile(
        packet,
        cluster=cluster,
        journal=journal,
        confirm_partial_cleanup=True,
    )
    assert result["release_confirmed"] is False
    assert [row[0] for row in cluster.deletes] == ["jobs.batch", "secrets"]
    assert cluster.creates == 1


def test_partial_create_rejects_unbound_source_bytes_before_delete(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    journal = _uncertain_create(packet, tmp_path, monkeypatch, cluster)
    cluster.job = cluster.secret = None
    assert cluster.config_map is not None
    cluster.config_map["data"]["evals-init.py"] += "# changed\n"
    with pytest.raises(job.QualificationJobError, match="differs from the exact packet"):
        _reconcile(
            packet,
            cluster=cluster,
            journal=journal,
            confirm_partial_cleanup=True,
        )
    assert cluster.deletes == [] and cluster.creates == 1


def test_resealed_substitute_canary_identity_is_rejected() -> None:
    canary = json.loads(CANARY.read_text())
    canary["candidate"]["task_version_id"] = "00000000-0000-4000-8000-000000000000"
    canary["sha256"] = job.canonical_digest(
        {key: value for key, value in canary.items() if key != "sha256"}
    )
    plan = _plan()
    plan["tasks"][0]["task_version_id"] = canary["candidate"]["task_version_id"]
    plan["selection"]["exact_task_identity"]["task_version_id"] = canary["candidate"][
        "task_version_id"
    ]
    plan["sha256"] = job.canonical_digest(
        {key: value for key, value in plan.items() if key != "sha256"}
    )
    with pytest.raises(job.QualificationJobError, match="exact one-cell"):
        job._validate_exact_plan(plan, canary)  # noqa: SLF001


def test_merge_witness_rejects_noncanonical_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(job, "git", lambda *_args: "https://example.invalid/fork.git")
    with pytest.raises(job.QualificationJobError, match="canonical"):
        job.merge_witness(tmp_path, SOURCE_COMMIT, refresh=False)


def test_cleanup_uses_fresh_resource_versions_and_proves_child_absence(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    journal = tmp_path / "create.jsonl"
    job.launch_once(
        packet,
        authorization_path=_authorization(packet, tmp_path),
        repo_root=ROOT,
        cluster=cluster,
        journal=journal,
    )
    cluster.terminal()
    receipt = job.cleanup_once(
        packet,
        cluster=cluster,
        journal=journal,
        receipt_path=tmp_path / "cleanup.json",
        sleep=lambda _seconds: None,
    )
    assert receipt["release_confirmed"] is True
    assert (
        "jobs.batch",
        job.build_package(packet).packet.value["job_name"],
        JOB_UID,
        "201",
    ) in cluster.deletes
    assert any(row[0] == "configmaps" and row[3] == "202" for row in cluster.deletes)
    assert any(row[0] == "secrets" and row[3] == "203" for row in cluster.deletes)


def test_cleanup_resumes_after_job_was_deleted_and_local_observer_crashed(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    journal = tmp_path / "create.jsonl"
    job.launch_once(
        packet,
        authorization_path=_authorization(packet, tmp_path),
        repo_root=ROOT,
        cluster=cluster,
        journal=journal,
    )
    cluster.terminal()
    original_delete = cluster.delete_uid

    def delete_then_crash(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_delete(*args, **kwargs)
        if args[0] == "jobs.batch":
            raise RuntimeError("observer crashed after Job deletion")
        return result

    monkeypatch.setattr(cluster, "delete_uid", delete_then_crash)
    receipt_path = tmp_path / "cleanup.json"
    with pytest.raises(RuntimeError, match="after Job deletion"):
        job.cleanup_once(
            packet,
            cluster=cluster,
            journal=journal,
            receipt_path=receipt_path,
            sleep=lambda _seconds: None,
        )
    assert cluster.job is None and receipt_path.with_suffix(".json.intent").is_file()
    monkeypatch.setattr(cluster, "delete_uid", original_delete)
    receipt = job.cleanup_once(
        packet,
        cluster=cluster,
        journal=journal,
        receipt_path=receipt_path,
        sleep=lambda _seconds: None,
    )
    assert receipt["release_confirmed"] is True


def test_cleanup_fresh_uid_census_catches_late_owned_child(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    journal = tmp_path / "create.jsonl"
    job.launch_once(
        packet,
        authorization_path=_authorization(packet, tmp_path),
        repo_root=ROOT,
        cluster=cluster,
        journal=journal,
    )
    cluster.terminal()
    assert cluster.pods
    late_pod = copy.deepcopy(cluster.pods[0])
    late_pod["metadata"].update(name="late-owned-pod", uid="66666666-6666-4666-8666-666666666666")
    original_delete = cluster.delete_uid

    def delete_with_late_child(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_delete(*args, **kwargs)
        if args[0] == "jobs.batch":
            cluster.pods = [copy.deepcopy(late_pod)]
        return result

    monkeypatch.setattr(cluster, "delete_uid", delete_with_late_child)
    clock = FakeClock()
    receipt_path = tmp_path / "cleanup.json"
    with pytest.raises(job.QualificationJobError, match="remains incomplete"):
        job.cleanup_once(
            packet,
            cluster=cluster,
            journal=journal,
            receipt_path=receipt_path,
            timeout_seconds=2,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    assert not receipt_path.exists()
    cluster.pods = []
    monkeypatch.setattr(cluster, "delete_uid", original_delete)
    receipt = job.cleanup_once(
        packet,
        cluster=cluster,
        journal=journal,
        receipt_path=receipt_path,
        sleep=lambda _seconds: None,
    )
    assert receipt["kubernetes_release_confirmed"] is True


def test_failed_job_is_released_but_reports_possible_external_leak(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cluster = FakeCluster()
    monkeypatch.setattr(job, "merge_witness", lambda *_args, **_kwargs: _source_witness())
    journal = tmp_path / "create.jsonl"
    job.launch_once(
        packet,
        authorization_path=_authorization(packet, tmp_path),
        repo_root=ROOT,
        cluster=cluster,
        journal=journal,
    )
    cluster.terminal(failed=True)
    receipt = job.cleanup_once(
        packet,
        cluster=cluster,
        journal=journal,
        receipt_path=tmp_path / "cleanup.json",
        sleep=lambda _seconds: None,
    )
    assert receipt["kubernetes_release_confirmed"] is True
    assert receipt["environment_cleanup_proven"] is False
    assert receipt["release_confirmed"] is False
    assert receipt["possible_external_environment_leak"] is True


def test_packaged_source_guard_works_without_git(
    packet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = job.build_package(packet)
    attestation = package.packet.files["source_attestation"]
    plan = json.loads(package.packet.files["plan"].read_text())
    monkeypatch.setenv(qualification.PACKAGED_SOURCE_ENV, str(attestation))
    monkeypatch.setattr(qualification, "_git", lambda *_args: pytest.fail("git must not run"))
    qualification._source_matches_plan(  # noqa: SLF001
        plan["source"],
        plan_sha256=plan["sha256"],
        exact_task_identity=plan["selection"]["exact_task_identity"],
    )


def test_packaged_source_has_an_isolated_import_closure(packet: Path, tmp_path: Path) -> None:
    package = job.build_package(packet)
    root = tmp_path / "isolated"
    (root / "evals/fleet").mkdir(parents=True)
    destinations = {
        "evals-init.py": root / "evals/__init__.py",
        "fleet-init.py": root / "evals/fleet/__init__.py",
        "opencode_self_hosted.py": root / "evals/fleet/opencode_self_hosted.py",
        "task_quality_qualification.py": root / "evals/fleet/task_quality_qualification.py",
        "task_quality_qualification_job_entry.py": (
            root / "evals/fleet/task_quality_qualification_job_entry.py"
        ),
    }
    for key, destination in destinations.items():
        destination.write_bytes(package.packet.files[key].read_bytes())
    script = (
        "import sys;"
        f"sys.path.insert(0,{str(root)!r});"
        "import evals.fleet.task_quality_qualification_job_entry;"
        "assert 'evals.fleet.rollout_worker' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_job_entry_requires_all_one_cell_gates_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(qualification.PACKAGED_SOURCE_ENV, raising=False)
    plan = _plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    attestation = tmp_path / "attestation.json"
    attestation.write_text("{}")
    gate_names = (
        "exact_task_binding",
        "durable_create_claim_routes_deployed",
        "environment_started",
        "bash_reachable",
        "submit_report_reachable",
        "verifier_completed",
        "finite_authoritative_outcome",
        "metadata_only_session_ingested",
        "environment_cleanup_completed",
    )
    monkeypatch.setattr(qualification, "_source_matches_plan", lambda *_args, **_kwargs: None)

    def execute(*_args: object, **_kwargs: object) -> dict[str, Any]:
        (tmp_path / "operation/cells").mkdir()
        return qualification.sealed(
            {
                "planned_task_versions": 1,
                "counts": {
                    "qualified": 1,
                    "infrastructure_invalid": 0,
                    "quarantined_ambiguous": 0,
                },
                "gate_pass_counts": dict.fromkeys(gate_names, 1),
            }
        )

    monkeypatch.setattr(qualification, "execute_plan", execute)
    monkeypatch.setattr(
        qualification,
        "cleanup_plan",
        lambda *_args, **_kwargs: qualification.sealed(
            {"resolved_task_versions": 1, "unresolved_task_versions": 0}
        ),
    )
    receipt = entry.run(
        plan_path=plan_path,
        attestation_path=attestation,
        private_root=tmp_path / "operation",
        api_key="redacted-test-key",
    )
    assert receipt["qualified_task_versions"] == 1
    assert receipt["model_calls"] == 0
    assert qualification.PACKAGED_SOURCE_ENV not in os.environ
