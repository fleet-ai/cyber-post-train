"""Offline tests for the exact-once visible-action collection CPU Job."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from evals.fleet import visible_action_collection_job as job
from evals.fleet import visible_action_collection_job_entry as entry

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v2"
DELETE_DRYRUN_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-study/2026-09-23-kubernetes-delete-dryrun-applied-v1.json"
)
SOURCE_COMMIT = "a" * 40
SOURCE_TREE = "b" * 40
JOB_UID = "11111111-1111-4111-8111-111111111111"
CONFIG_MAP_UID = "22222222-2222-4222-8222-222222222222"
POD_UID = "33333333-3333-4333-8333-333333333333"
WORKLOAD_UID = "44444444-4444-4444-8444-444444444444"
JOB_RESOURCE_VERSION = "101"
CONFIG_MAP_RESOURCE_VERSION = "102"


@pytest.fixture(scope="module")
def packet_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("collection-job-packet")

    def fake_git(_repo: Path, *arguments: str) -> str:
        if arguments == ("status", "--porcelain"):
            return ""
        if arguments == ("rev-parse", "HEAD"):
            return SOURCE_COMMIT
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return SOURCE_TREE
        raise AssertionError(arguments)

    original = job._git  # noqa: SLF001
    job._git = fake_git  # type: ignore[assignment] # noqa: SLF001
    try:
        result = job.prepare_packet(
            repo_root=ROOT,
            config_path=CAMPAIGN / "eval-config.json",
            task_selection_path=CAMPAIGN / "task-selection.json",
            authorization_path=CAMPAIGN / "operation-authorization.json",
            collection_packet_path=CAMPAIGN / "collection-packet.json",
            output=root / "packet",
            expected_source_commit=SOURCE_COMMIT,
        )
    finally:
        job._git = original  # type: ignore[assignment] # noqa: SLF001
    assert result["external_mutations"] == 0
    return Path(result["packet"])


class FakeCluster:
    def __init__(self) -> None:
        self.previews = 0
        self.creates = 0
        self.deletes: list[tuple[str, str, str, str]] = []
        self.job: dict[str, Any] | None = None
        self.config_map: dict[str, Any] | None = None
        self.pods: list[dict[str, Any]] = []
        self.workloads: list[dict[str, Any]] = []
        self.preview_drift = False
        self.create_raises = False

    def list(
        self,
        resource: str,
        namespace: str,
        *,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ) -> dict[str, Any]:
        assert namespace == job.NAMESPACE
        if resource == "jobs.batch":
            rows = [self.job] if self.job is not None else []
        elif resource == "configmaps":
            rows = [self.config_map] if self.config_map is not None else []
        elif resource == "pods":
            rows = self.pods
        elif resource == "workloads.kueue.x-k8s.io":
            rows = self.workloads
        else:
            raise AssertionError(resource)
        return {"apiVersion": "v1", "kind": "List", "items": copy.deepcopy(rows)}

    def get(self, resource: str, namespace: str, name: str) -> dict[str, Any]:
        value = self.get_optional(resource, namespace, name)
        if value is None:
            raise AssertionError((resource, name))
        return value

    def get_optional(self, resource: str, namespace: str, name: str) -> dict[str, Any] | None:
        assert namespace == job.NAMESPACE
        value = self.job if resource == "jobs.batch" else self.config_map
        if value is None or value["metadata"]["name"] != name:
            return None
        return copy.deepcopy(value)

    def server_dry_run(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        assert namespace == job.NAMESPACE
        self.previews += 1
        value = copy.deepcopy(bundle)
        rendered_job = next(row for row in value["items"] if row["kind"] == "Job")
        pod = rendered_job["spec"]["template"]["spec"]
        for container in pod["initContainers"] + pod["containers"]:
            for environment in container.get("env", []):
                if environment.get("value") == "":
                    environment.pop("value")
        if self.preview_drift and self.previews == 2:
            rendered_job["metadata"]["annotations"]["admission.example/drift"] = "true"
        return value

    def create_once(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        assert namespace == job.NAMESPACE
        self.creates += 1
        rows = copy.deepcopy(bundle["items"])
        self.config_map = next(row for row in rows if row["kind"] == "ConfigMap")
        self.job = next(row for row in rows if row["kind"] == "Job")
        self.config_map["metadata"].update(
            uid=CONFIG_MAP_UID, resourceVersion=CONFIG_MAP_RESOURCE_VERSION
        )
        self.job["metadata"].update(uid=JOB_UID, resourceVersion=JOB_RESOURCE_VERSION)
        response = {"apiVersion": "v1", "kind": "List", "items": rows}
        if self.create_raises:
            raise RuntimeError("response lost")
        return response

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
        assert namespace == job.NAMESPACE
        assert confirmed_live is True
        self.deletes.append((resource, name, uid, resource_version))
        if resource == "jobs.batch":
            assert self.job is not None and self.job["metadata"]["uid"] == uid
            assert self.job["metadata"]["resourceVersion"] == resource_version
            self.job = None
            self.pods = []
            self.workloads = []
        elif resource == "configmaps":
            assert self.config_map is not None and self.config_map["metadata"]["uid"] == uid
            assert self.config_map["metadata"]["resourceVersion"] == resource_version
            self.config_map = None
        else:
            raise AssertionError(resource)
        return {"apiVersion": "v1", "kind": "Status", "status": "Success"}

    def make_terminal(self, *, active_pod: bool = False) -> None:
        assert self.job is not None
        self.job["status"] = {
            "active": 0,
            "succeeded": 1,
            "conditions": [{"type": "Complete", "status": "True"}],
        }
        phase = "Running" if active_pod else "Succeeded"
        self.pods = [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {
                    "name": "collection-pod",
                    "namespace": job.NAMESPACE,
                    "uid": POD_UID,
                    "ownerReferences": [
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": self.job["metadata"]["name"],
                            "uid": JOB_UID,
                            "controller": True,
                        }
                    ],
                },
                "status": {"phase": phase},
            }
        ]
        self.workloads = [
            {
                "apiVersion": "kueue.x-k8s.io/v1beta1",
                "kind": "Workload",
                "metadata": {
                    "name": "collection-workload",
                    "namespace": job.NAMESPACE,
                    "uid": WORKLOAD_UID,
                    "ownerReferences": [
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": self.job["metadata"]["name"],
                            "uid": JOB_UID,
                            "controller": True,
                        }
                    ],
                },
            }
        ]


def test_package_binds_exact_cpu_job_and_source(packet_path: Path) -> None:
    package = job.build_package(packet_path)
    assert package.packet.value["source"]["git_commit"] == SOURCE_COMMIT
    assert package.packet.value["source"]["git_tree"] == SOURCE_TREE
    assert package.packet.value["operation"] == {
        "campaign_id": "q38-base-train50-actions-p4-v1",
        "plan_sha256": ("sha256:22ee8eef7949c1db748a4cc025a50e04b4ce3ebaebed11096d3b5d99a77c8277"),
        "planned_cell_universe_sha256": (
            "sha256:a0d70673b46efd09eec8283c87ce699aa56515a5e441013ad0243add5c84f10f"
        ),
        "operation_authorization_sha256": (
            "sha256:1a1a0e21c76aeb3fd00c5583e5a3eb5fc1ea86b48a19641b76fb35eb5733a67e"
        ),
        "collection_packet_sha256": (
            "sha256:f59989dc54d6b92503df5e37f19fa9f258a604c7644cdc2e887ceac5b174373b"
        ),
        "planned_cells": 200,
        "private_root": "/mnt/sfs/jobs",
        "operation_root": ("/mnt/sfs/jobs/q38-base-train50-actions-p4-v1-22ee8eef7949"),
        "database": "q38_base_train50_actions_p4_v2_22ee8eef",
        "route": "base",
        "worker_id": "base-v2",
    }
    rendered = package.job
    assert rendered["kind"] == "Job"
    assert rendered["metadata"]["annotations"][job.FAILURE_ALERT_ANNOTATION] == "off"
    assert rendered["spec"]["activeDeadlineSeconds"] == 768600
    assert rendered["spec"]["backoffLimit"] == 0
    pod = rendered["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "c1"
    assert pod["restartPolicy"] == "Never"
    assert pod["nodeSelector"]["kubernetes.io/arch"] == "amd64"
    assert next(
        item for item in pod["initContainers"][1]["env"] if item["name"] == "DOCKER_TLS_CERTDIR"
    ) == {"name": "DOCKER_TLS_CERTDIR"}
    assert next(
        item for item in pod["containers"][0]["env"] if item["name"] == "DOCKER_TLS_CERTDIR"
    ) == {"name": "DOCKER_TLS_CERTDIR"}
    for container in pod["initContainers"] + pod["containers"]:
        assert "nvidia.com/gpu" not in container["resources"]["requests"]
        assert "nvidia.com/gpu" not in container["resources"]["limits"]
    serialized = json.dumps(package.config_map, separators=(",", ":")).encode()
    assert len(serialized) < job.CONFIG_MAP_MAX_SERIALIZED_BYTES


def test_launch_proves_two_previews_and_creates_once(packet_path: Path, tmp_path: Path) -> None:
    cluster = FakeCluster()
    journal = tmp_path / "create.jsonl"
    result = job.launch_once(packet_path, cluster=cluster, journal=journal)
    assert result["submitted"] is True
    assert result["gpus"] == 0
    assert result["job_uid"] == JOB_UID
    assert result["config_map_uid"] == CONFIG_MAP_UID
    assert cluster.previews == 2
    assert cluster.creates == 1
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[0]["state"] == "KUBERNETES_CREATE_INTENT_DO_NOT_RETRY"
    assert rows[1]["state"] == "KUBERNETES_CREATE_RESPONSE"
    with pytest.raises(job.CollectionJobError, match="journal exists"):
        job.launch_once(packet_path, cluster=cluster, journal=journal)
    assert cluster.creates == 1


def test_preview_drift_stops_before_intent_or_create(packet_path: Path, tmp_path: Path) -> None:
    cluster = FakeCluster()
    cluster.preview_drift = True
    journal = tmp_path / "create.jsonl"
    with pytest.raises(job.CollectionJobError, match="changed across"):
        job.launch_once(packet_path, cluster=cluster, journal=journal)
    assert cluster.previews == 2
    assert cluster.creates == 0
    assert not journal.exists()


def test_uncertain_create_is_recorded_and_never_retried(packet_path: Path, tmp_path: Path) -> None:
    cluster = FakeCluster()
    cluster.create_raises = True
    journal = tmp_path / "create.jsonl"
    with pytest.raises(job.CollectionJobError, match="uncertain"):
        job.launch_once(packet_path, cluster=cluster, journal=journal)
    assert cluster.creates == 1
    assert "KUBERNETES_CREATE_RESPONSE_UNCERTAIN_DO_NOT_RETRY" in journal.read_text()
    with pytest.raises(job.CollectionJobError, match="journal exists"):
        job.launch_once(packet_path, cluster=cluster, journal=journal)
    assert cluster.creates == 1


def test_terminal_cleanup_uses_exact_uids_and_proves_release(
    packet_path: Path, tmp_path: Path
) -> None:
    cluster = FakeCluster()
    journal = tmp_path / "create.jsonl"
    job.launch_once(packet_path, cluster=cluster, journal=journal)
    cluster.make_terminal()
    receipt_path = tmp_path / "cleanup.json"
    receipt = job.cleanup_once(
        packet_path,
        cluster=cluster,
        journal=journal,
        receipt_path=receipt_path,
    )
    assert receipt["release_confirmed"] is True
    assert receipt["gpus"] == 0
    assert receipt["logs_traces_scores_or_credentials_read"] is False
    assert cluster.deletes == [
        (
            "jobs.batch",
            "chris-q38-base-train50-p4-v2-22ee8eef",
            JOB_UID,
            JOB_RESOURCE_VERSION,
        ),
        (
            "configmaps",
            "chris-q38-base-train50-p4-v2-22ee8eef-code",
            CONFIG_MAP_UID,
            CONFIG_MAP_RESOURCE_VERSION,
        ),
    ]
    assert cluster.job is None
    assert cluster.config_map is None
    assert not cluster.pods
    assert not cluster.workloads
    assert (
        job.cleanup_once(
            packet_path,
            cluster=cluster,
            journal=journal,
            receipt_path=receipt_path,
        )
        == receipt
    )
    assert len(cluster.deletes) == 2


def test_cleanup_refuses_active_owned_pod_without_delete(packet_path: Path, tmp_path: Path) -> None:
    cluster = FakeCluster()
    journal = tmp_path / "create.jsonl"
    job.launch_once(packet_path, cluster=cluster, journal=journal)
    cluster.make_terminal(active_pod=True)
    with pytest.raises(job.CollectionJobError, match="active or unknown Pod"):
        job.cleanup_once(
            packet_path,
            cluster=cluster,
            journal=journal,
            receipt_path=tmp_path / "cleanup.json",
        )
    assert cluster.deletes == []


def test_status_is_exact_uid_bound_and_content_free(packet_path: Path, tmp_path: Path) -> None:
    cluster = FakeCluster()
    journal = tmp_path / "create.jsonl"
    job.launch_once(packet_path, cluster=cluster, journal=journal)
    running = job.observe_status(packet_path, cluster=cluster, journal=journal)
    assert running["state"] == "running_or_queued"
    assert running["logs_traces_scores_or_credentials_read"] is False
    cluster.make_terminal()
    terminal = job.observe_status(packet_path, cluster=cluster, journal=journal)
    assert terminal["state"] == "terminal_ready_for_cleanup"
    assert terminal["job"]["uid"] == JOB_UID
    assert terminal["owned_pod_phases"] == {"Succeeded": 1}
    job.cleanup_once(
        packet_path,
        cluster=cluster,
        journal=journal,
        receipt_path=tmp_path / "cleanup.json",
    )
    released = job.observe_status(packet_path, cluster=cluster, journal=journal)
    assert released["state"] == "release_confirmed"


def test_cleanup_resumes_same_uid_after_lost_delete_response(
    packet_path: Path, tmp_path: Path
) -> None:
    class LostResponseCluster(FakeCluster):
        lose_once = True

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
            if resource == "jobs.batch" and self.lose_once:
                self.lose_once = False
                self.deletes.append((resource, name, uid, resource_version))
                raise RuntimeError("transport lost before server mutation")
            return super().delete_uid(
                resource,
                namespace,
                name,
                uid,
                resource_version,
                confirmed_live=confirmed_live,
            )

    cluster = LostResponseCluster()
    journal = tmp_path / "create.jsonl"
    job.launch_once(packet_path, cluster=cluster, journal=journal)
    cluster.make_terminal()
    receipt = tmp_path / "cleanup.json"
    with pytest.raises(job.CollectionJobError, match="resume the sealed cleanup intent"):
        job.cleanup_once(
            packet_path,
            cluster=cluster,
            journal=journal,
            receipt_path=receipt,
        )
    assert receipt.with_name(receipt.name + ".intent").is_file()
    assert not receipt.exists()
    result = job.cleanup_once(
        packet_path,
        cluster=cluster,
        journal=journal,
        receipt_path=receipt,
    )
    assert result["release_confirmed"] is True
    assert cluster.deletes == [
        (
            "jobs.batch",
            "chris-q38-base-train50-p4-v2-22ee8eef",
            JOB_UID,
            JOB_RESOURCE_VERSION,
        ),
        (
            "jobs.batch",
            "chris-q38-base-train50-p4-v2-22ee8eef",
            JOB_UID,
            JOB_RESOURCE_VERSION,
        ),
        (
            "configmaps",
            "chris-q38-base-train50-p4-v2-22ee8eef-code",
            CONFIG_MAP_UID,
            CONFIG_MAP_RESOURCE_VERSION,
        ),
    ]


def test_packet_source_mutation_is_rejected(packet_path: Path) -> None:
    packet = json.loads(packet_path.read_text())
    source = packet_path.parent / packet["source"]["files"]["evaluate.py"]["path"]
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n")
        with pytest.raises(job.CollectionJobError, match="bytes differ"):
            job.build_package(packet_path)
    finally:
        source.write_bytes(original)


def test_job_entry_orders_preflight_before_create_and_emits_only_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = tmp_path / "harness.tar"
    harness.write_bytes(b"qualified-harness")
    harness_sha = "sha256:" + hashlib.sha256(harness.read_bytes()).hexdigest()
    operation_root = tmp_path / "operation"
    operation_root.mkdir()
    events: list[str] = []
    written: dict[str, Any] = {}

    monkeypatch.setenv("ROLLOUT_DATABASE_URL", "postgresql://db.example/admin")
    monkeypatch.setattr(
        entry.cluster_entry,
        "stage_images",
        lambda **_kwargs: events.append("stage_images"),
    )
    monkeypatch.setattr(
        entry.collection,
        "prepare",
        lambda *_args, **_kwargs: events.append("prepare") or {"prepared": str(operation_root)},
    )
    monkeypatch.setattr(
        entry.collection,
        "preflight",
        lambda *_args, **_kwargs: events.append("preflight") or {"receipt_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        entry.cluster_entry,
        "dedicated_dsn",
        lambda *_args, **_kwargs: (
            events.append("dedicated_dsn") or "postgresql://db.example/dedicated"
        ),
    )
    monkeypatch.setattr(
        entry.cluster_entry,
        "create_database_once",
        lambda *_args, **_kwargs: (
            events.append("create_database") or "postgresql://db.example/dedicated"
        ),
    )
    monkeypatch.setattr(
        entry.collection,
        "_initialize_authorized_ledger",
        lambda *_args, **_kwargs: (
            events.append("initialize_ledger") or {"cells": 200, "plan_sha256": "ledger-plan"}
        ),
    )

    def initialize_once(_directory, *, dsn, initializer):
        events.append("initialize_once")
        authorization = json.loads((CAMPAIGN / "operation-authorization.json").read_text())
        created = initializer(dsn, operation_root / "plan.csv", authorization)
        assert created == {"cells": 200, "plan_sha256": "ledger-plan"}
        return {
            "ledger_plan_sha256": "ledger-plan",
            "sha256": "sha256:" + "e" * 64,
        }

    monkeypatch.setattr(entry.collection, "initialize_once", initialize_once)
    monkeypatch.setattr(
        entry.collection,
        "run",
        lambda *_args, **_kwargs: events.append("run") or {"accepted": 197},
    )
    monkeypatch.setattr(
        entry.rollout_postgres,
        "summary",
        lambda *_args, **_kwargs: (
            events.append("summary")
            or {
                "total": 200,
                "local_results": 197,
                "by_state": {
                    "pending": 0,
                    "claimed": 0,
                    "running": 0,
                    "grading": 0,
                    "accepted": 197,
                    "retry_review": 3,
                    "terminal": 0,
                },
                "by_serving_block": [
                    {"serving_block": "base", "state": "accepted", "count": 197},
                    {"serving_block": "base", "state": "retry_review", "count": 3},
                ],
                "stale_active": 0,
                "plan_sha256": "ledger-plan",
            }
        ),
    )

    def write_receipt(path: Path, value: dict[str, Any]) -> None:
        events.append("write_receipt")
        written["path"] = path
        written["value"] = value

    monkeypatch.setattr(entry.rollout_worker, "_safe_write_once", write_receipt)
    args = Namespace(
        config=str(CAMPAIGN / "eval-config.json"),
        authorization=str(CAMPAIGN / "operation-authorization.json"),
        private_root=str(tmp_path),
        database="q38_base_train50_actions_p4_v2_22ee8eef",
        harness_tar=str(harness),
        harness_tar_sha256=harness_sha,
        harness_receipt=str(tmp_path / "BUILD.json"),
        harness_receipt_sha256="sha256:" + "d" * 64,
        route="base",
        worker_id="base-v2",
        limit=200,
    )
    receipt = entry.execute(args)
    assert events == [
        "stage_images",
        "prepare",
        "preflight",
        "dedicated_dsn",
        "initialize_once",
        "create_database",
        "initialize_ledger",
        "run",
        "summary",
        "write_receipt",
    ]
    assert receipt["accepted_cells"] == 197
    assert receipt["unresolved_cells"] == 0
    assert receipt["private_content_included"] is False
    assert receipt["score_read_or_generated"] is False
    assert "results" not in receipt
    assert written["path"] == operation_root / entry.TERMINAL_FILE


def test_kubectl_cleanup_uses_raw_uid_and_resource_version_preconditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def run(arguments, *, input, **_kwargs):
        calls.append((arguments, input))
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=json.dumps({"apiVersion": "v1", "kind": "Status", "status": "Success"}),
            stderr="",
        )

    monkeypatch.setattr(job.subprocess, "run", run)
    cluster = job.KubectlCluster(job.CONTEXT)
    cluster.delete_uid(
        "jobs.batch",
        job.NAMESPACE,
        "chris-q38-base-train50-p4-v2-22ee8eef",
        JOB_UID,
        JOB_RESOURCE_VERSION,
        confirmed_live=True,
    )
    arguments, body = calls[0]
    assert arguments[-5:] == [
        "delete",
        "--raw",
        ("/apis/batch/v1/namespaces/fleet-train-jobs/jobs/chris-q38-base-train50-p4-v2-22ee8eef"),
        "-f",
        "-",
    ]
    assert json.loads(body or "null") == {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "propagationPolicy": "Foreground",
        "preconditions": {"uid": JOB_UID, "resourceVersion": JOB_RESOURCE_VERSION},
    }


def test_kubectl_cleanup_rejects_delete_dry_run_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("live delete must not run")

    monkeypatch.setattr(job.subprocess, "run", run)
    cluster = job.KubectlCluster(job.CONTEXT)
    with pytest.raises(job.CollectionJobError, match="no safe dry-run preview"):
        cluster.delete_uid(
            "jobs.batch",
            job.NAMESPACE,
            "chris-q38-base-train50-p4-v2-22ee8eef",
            JOB_UID,
            JOB_RESOURCE_VERSION,
            confirmed_live=False,
        )
    assert called is False


def test_delete_dry_run_incident_evidence_is_self_digested_and_score_blind() -> None:
    evidence = json.loads(DELETE_DRYRUN_EVIDENCE.read_text())
    expected_digest = evidence.pop("sha256")
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    assert expected_digest == "sha256:" + hashlib.sha256(canonical).hexdigest()

    privacy = evidence["privacy"]
    assert privacy == {
        "credentials_included": False,
        "flags_included": False,
        "prompts_or_traces_included": False,
        "raw_logs_included": False,
        "score_blind": True,
        "scores_included": False,
        "session_ids_included": False,
    }
    before = evidence["pre_release_state"]
    request = evidence["delete_request_and_response"]
    assert request["query"] == {"dryRun": "All"}
    assert request["request"]["preconditions"] == {
        "uid": before["job"]["uid"],
        "resourceVersion": before["job"]["resourceVersion"],
    }
    assert request["response"]["uid"] == before["job"]["uid"]
    for resource in ("job", "pod", "workload"):
        assert all(before[resource][key] for key in ("kind", "name", "uid", "resourceVersion"))
    after = evidence["post_release_state"]
    assert all(
        after[key] == 0
        for key in (
            "job_count_by_exact_name",
            "pod_count_by_exact_job_name",
            "pod_count_by_exact_job_uid",
            "workload_count_by_exact_name",
            "workload_count_by_exact_job_uid",
        )
    )
