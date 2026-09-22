"""Offline tests for the exact-once visible-action collection CPU Job."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from evals.fleet import visible_action_collection_job as job
from evals.fleet import visible_action_collection_job_entry as entry

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v2"
SOURCE_COMMIT = "a" * 40
SOURCE_TREE = "b" * 40
JOB_UID = "11111111-1111-4111-8111-111111111111"
CONFIG_MAP_UID = "22222222-2222-4222-8222-222222222222"
POD_UID = "33333333-3333-4333-8333-333333333333"
WORKLOAD_UID = "44444444-4444-4444-8444-444444444444"


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
        self.deletes: list[tuple[str, str, str]] = []
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
        self.config_map["metadata"]["uid"] = CONFIG_MAP_UID
        self.job["metadata"]["uid"] = JOB_UID
        response = {"apiVersion": "v1", "kind": "List", "items": rows}
        if self.create_raises:
            raise RuntimeError("response lost")
        return response

    def delete_uid(self, resource: str, namespace: str, name: str, uid: str) -> dict[str, Any]:
        assert namespace == job.NAMESPACE
        self.deletes.append((resource, name, uid))
        if resource == "jobs.batch":
            assert self.job is not None and self.job["metadata"]["uid"] == uid
            self.job = None
            self.pods = []
            self.workloads = []
        elif resource == "configmaps":
            assert self.config_map is not None and self.config_map["metadata"]["uid"] == uid
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
        "plan_sha256": ("sha256:fb0d54ecca6bb5a8942016dbf339e7b51c902ec39458eb610b4d1eae2585154f"),
        "planned_cell_universe_sha256": (
            "sha256:a031182c01783b316dfa1190c8104fcc24fe738cbdf63d0a411942a0891c7f6c"
        ),
        "operation_authorization_sha256": (
            "sha256:1597d141b236a9f8fea16e92bf0b4a4f9b16045677931bf72bb5ca26618e4684"
        ),
        "collection_packet_sha256": (
            "sha256:d164781e9d704611c35637a23a61976bab0372906cfd65ee2ccd014a9d1caeba"
        ),
        "planned_cells": 200,
        "private_root": "/mnt/sfs/jobs",
        "operation_root": ("/mnt/sfs/jobs/q38-base-train50-actions-p4-v1-fb0d54ecca6b"),
        "database": "q38_base_train50_actions_p4_v2_fb0d54ec",
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


def test_v2_staged_entrypoint_does_not_require_v3_source(packet_path: Path, tmp_path: Path) -> None:
    packet = json.loads(packet_path.read_text())
    staged = tmp_path / "staged"
    (staged / "cyber_post_train").mkdir(parents=True)
    (staged / "evals/fleet").mkdir(parents=True)
    for package in ("cyber_post_train", "evals", "evals/fleet"):
        (staged / package / "__init__.py").touch()
    for name, reference in packet["source"]["files"].items():
        if name == "run.sh":
            continue
        destination = (
            staged / "cyber_post_train/jobs.py"
            if name == "jobs.py"
            else staged / "evals/fleet" / name
        )
        destination.write_bytes((packet_path.parent / reference["path"]).read_bytes())
    assert not (staged / "evals/fleet/visible_action_collection_v3.py").exists()
    command = (
        "from evals.fleet import visible_action_collection_job_entry as entry; "
        "assert entry._collection_runtime({"  # noqa: SLF001
        "'collection_runtime': {'schema': 'cyber_visible_action_collection_runtime_v2'}"
        "}) is entry.collection"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=staged,
        env={**os.environ, "PYTHONPATH": str(staged)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
        ("jobs.batch", "chris-q38-base-train50-p4-v2-fb0d54ec", JOB_UID),
        (
            "configmaps",
            "chris-q38-base-train50-p4-v2-fb0d54ec-code",
            CONFIG_MAP_UID,
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

        def delete_uid(self, resource: str, namespace: str, name: str, uid: str) -> dict[str, Any]:
            if resource == "jobs.batch" and self.lose_once:
                self.lose_once = False
                self.deletes.append((resource, name, uid))
                raise RuntimeError("transport lost before server mutation")
            return super().delete_uid(resource, namespace, name, uid)

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
        ("jobs.batch", "chris-q38-base-train50-p4-v2-fb0d54ec", JOB_UID),
        ("jobs.batch", "chris-q38-base-train50-p4-v2-fb0d54ec", JOB_UID),
        (
            "configmaps",
            "chris-q38-base-train50-p4-v2-fb0d54ec-code",
            CONFIG_MAP_UID,
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
        database="q38_base_train50_actions_p4_v2_fb0d54ec",
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


def test_kubectl_cleanup_uses_raw_uid_precondition(
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
        "chris-q38-base-train50-p4-v2-fb0d54ec",
        JOB_UID,
    )
    arguments, body = calls[0]
    assert arguments[-5:] == [
        "delete",
        "--raw",
        ("/apis/batch/v1/namespaces/fleet-train-jobs/jobs/chris-q38-base-train50-p4-v2-fb0d54ec"),
        "-f",
        "-",
    ]
    assert json.loads(body or "null") == {
        "apiVersion": "v1",
        "kind": "DeleteOptions",
        "propagationPolicy": "Foreground",
        "preconditions": {"uid": JOB_UID},
    }
