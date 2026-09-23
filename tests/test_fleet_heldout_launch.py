"""Focused no-network regressions for the one-shot Fleet held-out launcher."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from evals.fleet import heldout_launch as launch

NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-heldout-base-a1"
CONFIG_MAP_NAME = "chris-heldout-base-code-a1"
OUTPUT_ROOT = "/mnt/sfs/jobs/chris-heldout-base-a1"
DATABASE = "chris_heldout_base_a1"
JOB_UID = "11111111-2222-4333-8444-555555555555"
CONFIG_MAP_UID = "66666666-7777-4888-8999-aaaaaaaaaaaa"
POD_UID = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
WORKLOAD_UID = "12345678-1234-4234-8234-123456789abc"
ROOT = Path(__file__).resolve().parents[1]


class FakeDatabase:
    def __init__(self) -> None:
        self.present = False
        self.exists_calls = 0
        self.summary_calls = 0
        self.summary_value = {
            "total": 2,
            "local_results": 2,
            "by_state": {"accepted": 2, "claimed": 0, "pending": 0, "retry_review": 0},
            "by_serving_block": [{"serving_block": "base", "state": "accepted", "count": 2}],
            "stale_active": 0,
            "plan_sha256": "sha256:" + "a" * 64,
        }

    def exists(self, database: str) -> bool:
        assert database == DATABASE
        self.exists_calls += 1
        return self.present

    def summary(self, database: str) -> dict[str, Any]:
        assert database == DATABASE
        self.summary_calls += 1
        return copy.deepcopy(self.summary_value)


class FakeCluster:
    def __init__(self) -> None:
        self.inventories: dict[str, dict[str, Any]] = {
            "jobs.batch": {"kind": "JobList", "items": []},
            "configmaps": {"kind": "ConfigMapList", "items": []},
            "workloads.kueue.x-k8s.io": {"kind": "WorkloadList", "items": []},
            "pods": {"kind": "PodList", "items": []},
        }
        self.preview_calls = 0
        self.create_calls = 0
        self.list_calls: list[tuple[str, str | None, str | None]] = []
        self.missing_root_alert = False
        self.raise_on_create = False
        self.add_job_after_second_preview = False
        self.created: dict[str, Any] | None = None

    def list(
        self,
        resource: str,
        namespace: str,
        *,
        field_selector: str | None = None,
        label_selector: str | None = None,
    ) -> dict[str, Any]:
        assert namespace == NAMESPACE
        assert (field_selector is None) != (label_selector is None)
        self.list_calls.append((resource, field_selector, label_selector))
        inventory = copy.deepcopy(self.inventories[resource])
        if field_selector is not None:
            assert field_selector.startswith("metadata.name=")
            name = field_selector.removeprefix("metadata.name=")
            inventory["items"] = [
                item for item in inventory["items"] if item.get("metadata", {}).get("name") == name
            ]
        else:
            assert label_selector is not None
            key, value = label_selector.split("=", 1)
            inventory["items"] = [
                item
                for item in inventory["items"]
                if item.get("metadata", {}).get("labels", {}).get(key) == value
            ]
        return inventory

    def get(self, resource: str, namespace: str, name: str) -> dict[str, Any]:
        assert namespace == NAMESPACE
        if self.created is None:
            raise AssertionError("terminal collection before create")
        items = self.created["items"]
        kind = {"jobs.batch": "Job", "configmaps": "ConfigMap"}[resource]
        for item in items:
            if item["kind"] == kind and item["metadata"]["name"] == name:
                return copy.deepcopy(item)
        raise AssertionError(f"missing created {kind}")

    def _server_response(self, bundle: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(bundle)
        job = next(item for item in result["items"] if item["kind"] == "Job")
        job["metadata"]["uid"] = JOB_UID
        job["spec"]["selector"] = {"matchLabels": {"batch.kubernetes.io/controller-uid": JOB_UID}}
        template_metadata = job["spec"]["template"].setdefault("metadata", {})
        template_metadata.setdefault("labels", {})["batch.kubernetes.io/controller-uid"] = JOB_UID
        if self.missing_root_alert:
            job["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
            template_metadata.setdefault("annotations", {})["fleet.ai/failure-alerts"] = "off"
        config_map = next(item for item in result["items"] if item["kind"] == "ConfigMap")
        config_map["metadata"]["uid"] = CONFIG_MAP_UID
        return result

    def server_dry_run(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        assert namespace == NAMESPACE
        self.preview_calls += 1
        if self.add_job_after_second_preview and self.preview_calls == 2:
            self.inventories["jobs.batch"]["items"].append({"metadata": {"name": JOB_NAME}})
        return self._server_response(bundle)

    def create_once(self, namespace: str, bundle: dict[str, Any]) -> dict[str, Any]:
        assert namespace == NAMESPACE
        self.create_calls += 1
        if self.raise_on_create:
            raise OSError("transport interruption")
        self.created = self._server_response(bundle)
        return copy.deepcopy(self.created)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _packet(tmp_path: Path) -> Path:
    split_manifest = {
        "schema": "cyber_parameterized_task_family_split_v1",
        "tasks": [
            {
                "task_key": "task-a",
                "task_version_id": "version-a",
                "group_id": "sha256:" + "1" * 64,
                "split": "dev",
            }
        ],
    }
    split_manifest["sha256"] = launch._canonical_digest(split_manifest)  # noqa: SLF001
    _write_json(tmp_path / "split-manifest.json", split_manifest)
    task_set = {
        "schema": "cyber_eval_task_selection_v2",
        "selection_role": "dev",
        "split_manifest": {
            "path": "split-manifest.json",
            "file_sha256": _sha(tmp_path / "split-manifest.json"),
            "object_sha256": split_manifest["sha256"],
        },
        "task_count": 1,
        "tasks": [{"task_key": "task-a", "task_version_id": "version-a"}],
    }
    config = {
        "name": "heldout-base-a1",
        "task_set": "task-set.json",
        "models": {"base": {"revision": "base-revision"}},
        "routes": {"base": {"model": "base", "task_versions": ["version-a"]}},
        "harness": {
            "harness": "opencode",
            "harness_version": "1.18.27",
            "context_management": "native-compaction",
        },
        "sampling": {"seed": 44, "temperature": 0.6},
        "images": {
            "agent": "registry.example/agent@sha256:" + "a" * 64,
            "proxy": "registry.example/proxy@sha256:" + "b" * 64,
        },
        "pass_k": 1,
        "max_reviewed_infrastructure_retries": 1,
        "training_data_eligible": False,
    }
    _write_json(tmp_path / "task-set.json", task_set)
    _write_json(tmp_path / "config.json", config)
    (tmp_path / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "serving.json").write_text("{}\n", encoding="utf-8")
    _write_json(tmp_path / "ledger.json", {"evaluations": []})
    comparison_protocol = {
        "schema": launch.COMPARISON_PROTOCOL_SCHEMA,
        "protocol_id": "heldout-protocol-a1",
        "comparison_arms": ["base", "candidate"],
        "model_revisions": {"base": "base-revision", "candidate": "candidate-revision"},
        "task_selection_sha256": _sha(tmp_path / "task-set.json"),
        "split_manifest_file_sha256": _sha(tmp_path / "split-manifest.json"),
        "split_manifest_sha256": split_manifest["sha256"],
        "harness": config["harness"],
        "sampling": config["sampling"],
        "images": config["images"],
        "pass_k": config["pass_k"],
        "retry_limit": config["max_reviewed_infrastructure_retries"],
    }
    comparison_protocol["sha256"] = launch._canonical_digest(comparison_protocol)  # noqa: SLF001
    _write_json(tmp_path / "comparison-protocol.json", comparison_protocol)
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "immutable": True,
        "data": {"config.json": config_text, "heldout.json": config_text, "run.sh": "true\n"},
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
            "annotations": {
                "fleet.ai/failure-alerts": "off",
                "cyber-post-train.fleet.ai/create-once": "true",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 3600,
            "template": {
                "spec": {
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "evaluator",
                            "env": [
                                {"name": "EVAL_CONFIG_NAME", "value": "heldout.json"},
                                {"name": "EVAL_OUTPUT", "value": OUTPUT_ROOT},
                                {"name": "EVAL_DATABASE", "value": DATABASE},
                            ],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "1Gi"},
                                "limits": {"cpu": "2", "memory": "2Gi"},
                            },
                        }
                    ],
                    "volumes": [{"name": "bootstrap", "configMap": {"name": CONFIG_MAP_NAME}}],
                }
            },
        },
    }
    (tmp_path / "config-map.yaml").write_text(yaml.safe_dump(config_map), encoding="utf-8")
    (tmp_path / "job.yaml").write_text(yaml.safe_dump(job), encoding="utf-8")
    files = {
        "evaluation_config": {"path": "config.json", "sha256": _sha(tmp_path / "config.json")},
        "task_set": {"path": "task-set.json", "sha256": _sha(tmp_path / "task-set.json")},
        "split_manifest": {
            "path": "split-manifest.json",
            "sha256": _sha(tmp_path / "split-manifest.json"),
        },
        "comparison_protocol": {
            "path": "comparison-protocol.json",
            "sha256": _sha(tmp_path / "comparison-protocol.json"),
        },
        "config_map": {"path": "config-map.yaml", "sha256": _sha(tmp_path / "config-map.yaml")},
        "job": {"path": "job.yaml", "sha256": _sha(tmp_path / "job.yaml")},
        "checkpoint_provenance": {
            "path": "checkpoint.json",
            "sha256": _sha(tmp_path / "checkpoint.json"),
        },
        "serving_route_proof": {"path": "serving.json", "sha256": _sha(tmp_path / "serving.json")},
        "evaluation_ledger": {"path": "ledger.json"},
    }
    identity = {
        "protocol_id": "heldout-protocol-a1",
        "comparison_arms": ["base", "candidate"],
        "arm_id": "base",
        "evaluation_config_name": config["name"],
        "evaluation_config_sha256": files["evaluation_config"]["sha256"],
        "task_selection_sha256": files["task_set"]["sha256"],
        "split_manifest_file_sha256": files["split_manifest"]["sha256"],
        "split_manifest_sha256": split_manifest["sha256"],
        "comparison_protocol_file_sha256": files["comparison_protocol"]["sha256"],
        "comparison_protocol_sha256": comparison_protocol["sha256"],
        "checkpoint_provenance_sha256": files["checkpoint_provenance"]["sha256"],
        "serving_route_proof_sha256": files["serving_route_proof"]["sha256"],
        "model_revision": "base-revision",
        "harness": "opencode",
        "harness_version": "1.18.27",
        "context_management": "native-compaction",
        "sampling_seed": 44,
        "pass_k": 1,
        "retry_limit": 1,
        "output_root": OUTPUT_ROOT,
        "database": DATABASE,
    }
    packet = {
        "schema": launch.PACKET_SCHEMA,
        "namespace": NAMESPACE,
        "job_name": JOB_NAME,
        "config_map_name": CONFIG_MAP_NAME,
        "output_root": OUTPUT_ROOT,
        "database": DATABASE,
        "files": files,
        "evaluation_identity": identity,
        "evaluation_identity_sha256": launch._canonical_digest(identity),  # noqa: SLF001
    }
    path = tmp_path / "packet.json"
    _write_json(path, packet)
    return path


def _launch(
    packet: Path, cluster: FakeCluster, database: FakeDatabase, journal: Path
) -> dict[str, Any]:
    return launch.launch_once(
        packet,
        cluster=cluster,
        database=database,
        journal=journal,
        output_exists=lambda _: False,
    )


def _reseal_packet(packet: Path, raw: dict[str, Any]) -> None:
    identity = raw["evaluation_identity"]
    identity["evaluation_config_sha256"] = raw["files"]["evaluation_config"]["sha256"]
    identity["task_selection_sha256"] = raw["files"]["task_set"]["sha256"]
    identity["split_manifest_file_sha256"] = raw["files"]["split_manifest"]["sha256"]
    raw["evaluation_identity_sha256"] = launch._canonical_digest(identity)  # noqa: SLF001
    _write_json(packet, raw)


def test_existing_job_stops_before_server_preview_or_create(tmp_path):
    packet = _packet(tmp_path)
    cluster, database = FakeCluster(), FakeDatabase()
    cluster.inventories["jobs.batch"]["items"] = [{"metadata": {"name": JOB_NAME}}]
    with pytest.raises(launch.HeldoutLaunchError, match="Job already exists"):
        _launch(packet, cluster, database, tmp_path / "intent.jsonl")
    assert cluster.preview_calls == 0
    assert cluster.create_calls == 0
    assert cluster.list_calls == [("jobs.batch", f"metadata.name={JOB_NAME}", None)]


def test_duplicate_census_uses_only_bounded_exact_identity_queries(tmp_path):
    packet = _packet(tmp_path)
    cluster, database = FakeCluster(), FakeDatabase()
    package = launch.build_package(packet)

    census = launch.duplicate_census(
        package, cluster=cluster, database=database, output_exists=lambda _: False
    )

    assert census == {"jobs": 0, "config_maps": 0, "pods": 0}
    assert cluster.list_calls == [
        ("jobs.batch", f"metadata.name={JOB_NAME}", None),
        ("configmaps", f"metadata.name={CONFIG_MAP_NAME}", None),
        ("pods", None, f"job-name={JOB_NAME}"),
        ("pods", None, f"batch.kubernetes.io/job-name={JOB_NAME}"),
    ]


def test_matching_complete_identity_in_ledger_stops_before_server_preview_or_create(tmp_path):
    packet = _packet(tmp_path)
    identity = json.loads(packet.read_text())["evaluation_identity_sha256"]
    _write_json(
        tmp_path / "ledger.json", {"evaluations": [{"evaluation_identity_sha256": identity}]}
    )
    cluster, database = FakeCluster(), FakeDatabase()
    with pytest.raises(launch.HeldoutLaunchError, match="ledger already contains"):
        _launch(packet, cluster, database, tmp_path / "intent.jsonl")
    assert cluster.preview_calls == 0
    assert cluster.create_calls == 0


def test_server_preview_requires_root_annotation_not_just_pod_template(tmp_path):
    packet = _packet(tmp_path)
    cluster, database = FakeCluster(), FakeDatabase()
    cluster.missing_root_alert = True
    with pytest.raises(launch.HeldoutLaunchError, match="root Job is missing failure-alerts off"):
        _launch(packet, cluster, database, tmp_path / "intent.jsonl")
    assert cluster.preview_calls == 1
    assert cluster.create_calls == 0


def test_task_selection_must_bind_the_exact_sealed_split_manifest(tmp_path):
    packet = _packet(tmp_path)
    raw = json.loads(packet.read_text())
    task_set_path = tmp_path / raw["files"]["task_set"]["path"]
    task_set = json.loads(task_set_path.read_text())
    task_set["split_manifest"]["object_sha256"] = "sha256:" + "2" * 64
    _write_json(task_set_path, task_set)
    raw["files"]["task_set"]["sha256"] = _sha(task_set_path)
    _reseal_packet(packet, raw)

    with pytest.raises(launch.HeldoutLaunchError, match="split binding differs"):
        launch.build_package(packet)


def test_heldout_evaluator_rejects_a_training_selection_role(tmp_path):
    packet = _packet(tmp_path)
    raw = json.loads(packet.read_text())
    task_set_path = tmp_path / raw["files"]["task_set"]["path"]
    task_set = json.loads(task_set_path.read_text())
    task_set["selection_role"] = "train"
    _write_json(task_set_path, task_set)
    raw["files"]["task_set"]["sha256"] = _sha(task_set_path)
    _reseal_packet(packet, raw)

    with pytest.raises(launch.HeldoutLaunchError, match="cannot use a training role"):
        launch.build_package(packet)


def test_route_must_evaluate_exactly_the_sealed_selection(tmp_path):
    packet = _packet(tmp_path)
    raw = json.loads(packet.read_text())
    config_path = tmp_path / raw["files"]["evaluation_config"]["path"]
    config = json.loads(config_path.read_text())
    config["routes"]["base"]["task_versions"] = ["version-b"]
    _write_json(config_path, config)
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_map_path = tmp_path / raw["files"]["config_map"]["path"]
    config_map = yaml.safe_load(config_map_path.read_text())
    config_map["data"]["config.json"] = config_text
    config_map["data"]["heldout.json"] = config_text
    config_map_path.write_text(yaml.safe_dump(config_map), encoding="utf-8")
    raw["files"]["evaluation_config"]["sha256"] = _sha(config_path)
    raw["files"]["config_map"]["sha256"] = _sha(config_map_path)
    _reseal_packet(packet, raw)

    with pytest.raises(launch.HeldoutLaunchError, match="route task versions differ"):
        launch.build_package(packet)


def test_comparison_protocol_rejects_sampling_treatment_drift(tmp_path):
    packet = _packet(tmp_path)
    raw = json.loads(packet.read_text())
    config_path = tmp_path / raw["files"]["evaluation_config"]["path"]
    config = json.loads(config_path.read_text())
    config["sampling"]["temperature"] = 0.7
    _write_json(config_path, config)
    config_text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_map_path = tmp_path / raw["files"]["config_map"]["path"]
    config_map = yaml.safe_load(config_map_path.read_text())
    config_map["data"]["config.json"] = config_text
    config_map["data"]["heldout.json"] = config_text
    config_map_path.write_text(yaml.safe_dump(config_map), encoding="utf-8")
    raw["files"]["evaluation_config"]["sha256"] = _sha(config_path)
    raw["files"]["config_map"]["sha256"] = _sha(config_map_path)
    _reseal_packet(packet, raw)

    with pytest.raises(launch.HeldoutLaunchError, match="comparison protocol differs"):
        launch.build_package(packet)


def test_kubectl_adapter_rejects_an_unsafe_context_before_any_operation():
    with pytest.raises(launch.HeldoutLaunchError, match="explicit Kubernetes context"):
        launch.KubectlCluster("--other-context")
    with pytest.raises(launch.HeldoutLaunchError, match="explicit Kubernetes context"):
        launch.KubectlCluster("fleet;unexpected")


def test_kubectl_adapter_rejects_unbounded_lists_and_sends_only_server_side_selectors(
    monkeypatch,
):
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        return launch.subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"kind": "List", "items": []}), stderr=""
        )

    monkeypatch.setattr(launch.subprocess, "run", run)
    cluster = launch.KubectlCluster("fleet-context")
    with pytest.raises(launch.HeldoutLaunchError, match="exactly one scoped selector"):
        cluster.list("pods", NAMESPACE)
    cluster.list("jobs.batch", NAMESPACE, field_selector=f"metadata.name={JOB_NAME}")
    cluster.list("pods", NAMESPACE, label_selector=f"job-name={JOB_NAME}")
    cluster.list(
        "workloads.kueue.x-k8s.io",
        NAMESPACE,
        label_selector=f"kueue.x-k8s.io/job-uid={JOB_UID}",
    )
    assert len(calls) == 3
    assert f"--field-selector=metadata.name={JOB_NAME}" in calls[0]
    assert f"--selector=job-name={JOB_NAME}" in calls[1]
    assert f"--selector=kueue.x-k8s.io/job-uid={JOB_UID}" in calls[2]
    assert all("--output=json" in call for call in calls)


def test_kubectl_adapter_decodes_concatenated_multi_object_json(monkeypatch):
    objects = [
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": CONFIG_MAP_NAME}},
        {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": JOB_NAME}},
    ]

    def run(command, **kwargs):
        return launch.subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(json.dumps(value) for value in objects) + "\n",
            stderr="",
        )

    monkeypatch.setattr(launch.subprocess, "run", run)
    cluster = launch.KubectlCluster("fleet-context")
    assert cluster.server_dry_run(NAMESPACE, {"apiVersion": "v1", "kind": "List"}) == {
        "apiVersion": "v1",
        "kind": "List",
        "items": objects,
    }


def test_kubectl_adapter_rejects_trailing_non_json(monkeypatch):
    def run(command, **kwargs):
        return launch.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"kind": "List", "items": []}) + "\nprivate noise",
            stderr="",
        )

    monkeypatch.setattr(launch.subprocess, "run", run)
    cluster = launch.KubectlCluster("fleet-context")
    with pytest.raises(launch.HeldoutLaunchError, match="invalid JSON"):
        cluster.list("jobs.batch", NAMESPACE, field_selector=f"metadata.name={JOB_NAME}")


def test_preview_containment_accepts_only_server_defaults_inside_fixed_lists():
    expected = [{"name": "evaluator", "env": [{"name": "EMPTY", "value": ""}]}]
    actual = [
        {
            "name": "evaluator",
            "imagePullPolicy": "IfNotPresent",
            "env": [{"name": "EMPTY"}],
        }
    ]
    assert launch._contains(actual, expected)  # noqa: SLF001
    assert not launch._contains(actual + [{"name": "unexpected"}], expected)  # noqa: SLF001
    assert not launch._contains([{"name": "other", "env": [{"name": "EMPTY"}]}], expected)  # noqa: SLF001


def test_workload_binding_requires_the_created_job_uid_not_only_a_matching_name():
    workload = {
        "metadata": {
            "labels": {"kueue.x-k8s.io/job-uid": JOB_UID},
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": JOB_NAME,
                    "uid": JOB_UID,
                    "controller": True,
                }
            ],
        }
    }
    assert launch._workload_binds_created_job(workload, JOB_NAME, JOB_UID)  # noqa: SLF001
    workload["metadata"]["ownerReferences"][0]["uid"] = CONFIG_MAP_UID
    assert not launch._workload_binds_created_job(workload, JOB_NAME, JOB_UID)  # noqa: SLF001


def test_current_fresh75_selection_contract_remains_compatible():
    selection = json.loads(
        (ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-task-set-v1.json").read_text()
    )
    split_path = ROOT / selection["split_manifest"]["path"]
    split_manifest = json.loads(split_path.read_text())
    packet = launch.LaunchPacket(
        path=ROOT / "synthetic-packet.json",
        namespace=NAMESPACE,
        job_name=JOB_NAME,
        config_map_name=CONFIG_MAP_NAME,
        output_root=OUTPUT_ROOT,
        database=DATABASE,
        files={},
        file_sha256={"split_manifest": selection["split_manifest"]["file_sha256"]},
        identity={"split_manifest_sha256": selection["split_manifest"]["object_sha256"]},
        identity_sha256="sha256:" + "0" * 64,
    )
    split_index = launch._split_index(  # noqa: SLF001
        split_manifest,
        selection["split_manifest"]["object_sha256"],
    )
    role, selected = launch._selection_index(  # noqa: SLF001
        selection,
        split_index=split_index,
        packet=packet,
    )
    assert role == "dev"
    assert len(selected) == len(selection["tasks"])
    assert set(selected) == {row["task_version_id"] for row in selection["tasks"]}


def test_fresh_final_census_blocks_a_duplicate_that_appears_during_preview(tmp_path):
    packet = _packet(tmp_path)
    cluster, database = FakeCluster(), FakeDatabase()
    cluster.add_job_after_second_preview = True
    with pytest.raises(launch.HeldoutLaunchError, match="Job already exists"):
        _launch(packet, cluster, database, tmp_path / "intent.jsonl")
    assert cluster.preview_calls == 2
    assert cluster.create_calls == 0


def test_successful_two_preview_gate_writes_intent_and_creates_once_at_most_once(tmp_path):
    packet = _packet(tmp_path)
    cluster, database = FakeCluster(), FakeDatabase()
    journal = tmp_path / "intent.jsonl"
    result = _launch(packet, cluster, database, journal)
    assert result == {
        "submitted": True,
        "gpus": 0,
        "job_name": JOB_NAME,
        "job_uid": JOB_UID,
        "config_map_name": CONFIG_MAP_NAME,
        "config_map_uid": CONFIG_MAP_UID,
        "evaluation_identity_sha256": json.loads(packet.read_text())["evaluation_identity_sha256"],
        "comparison_protocol_sha256": json.loads(packet.read_text())["evaluation_identity"][
            "comparison_protocol_sha256"
        ],
    }
    assert cluster.preview_calls == 2
    assert cluster.create_calls == 1
    lines = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [line["state"] for line in lines] == [
        "KUBECTL_CREATE_INTENT_DO_NOT_RETRY",
        "KUBECTL_CREATE_RESPONSE",
    ]
    with pytest.raises(launch.HeldoutLaunchError, match="journal already exists"):
        _launch(packet, cluster, database, journal)
    assert cluster.create_calls == 1


def test_uncertain_create_is_observed_but_never_retried(tmp_path):
    packet = _packet(tmp_path)
    cluster, database = FakeCluster(), FakeDatabase()
    cluster.raise_on_create = True
    journal = tmp_path / "intent.jsonl"
    with pytest.raises(launch.HeldoutLaunchError, match="create response is uncertain"):
        _launch(packet, cluster, database, journal)
    assert cluster.create_calls == 1
    lines = [json.loads(line) for line in journal.read_text().splitlines()]
    assert lines[-1]["state"] == "KUBECTL_CREATE_RESPONSE_UNCERTAIN_DO_NOT_RETRY"


@pytest.mark.parametrize("database_present", [True, False])
def test_terminal_collection_is_score_blind_and_never_retries_or_scores(tmp_path, database_present):
    packet = _packet(tmp_path)
    cluster, database = FakeCluster(), FakeDatabase()
    _launch(packet, cluster, database, tmp_path / "intent.jsonl")
    preterminal_exists_calls = database.exists_calls
    database.present = database_present
    assert cluster.created is not None
    job = next(item for item in cluster.created["items"] if item["kind"] == "Job")
    job["status"] = {
        "conditions": [{"type": "Complete", "status": "True"}],
        "succeeded": 1,
        "failed": 0,
    }
    cluster.inventories["pods"]["items"] = [
        {
            "metadata": {
                "name": JOB_NAME + "-abc",
                "uid": POD_UID,
                "labels": {"job-name": JOB_NAME},
            },
            "status": {"phase": "Succeeded"},
        }
    ]
    cluster.inventories["workloads.kueue.x-k8s.io"]["items"] = [
        {
            "metadata": {
                "name": "job-" + JOB_NAME,
                "uid": WORKLOAD_UID,
                "labels": {"kueue.x-k8s.io/job-uid": JOB_UID},
                "ownerReferences": [
                    {
                        "apiVersion": "batch/v1",
                        "kind": "Job",
                        "name": JOB_NAME,
                        "uid": JOB_UID,
                        "controller": True,
                    }
                ],
            },
            "spec": {"podSets": []},
            "status": {"phase": "Finished"},
        }
    ]
    receipt = launch.collect_terminal(
        packet,
        cluster=cluster,
        database=database,
        receipt_path=tmp_path / "terminal.json",
        output_exists=lambda _: True,
    )
    assert receipt["job"]["terminal_condition"] == "Complete"
    assert receipt["protocol_id"] == "heldout-protocol-a1"
    assert receipt["arm_id"] == "base"
    if database_present:
        assert receipt["database"]["summary"]["by_state"] == {
            "accepted": 2,
            "claimed": 0,
            "pending": 0,
            "retry_review": 0,
        }
        assert database.exists_calls - preterminal_exists_calls == 1
        assert database.summary_calls == 1
    else:
        assert receipt["database"]["summary"] == {
            "total": 0,
            "local_results": 0,
            "by_state": {
                "accepted": 0,
                "claimed": 0,
                "grading": 0,
                "pending": 0,
                "retry_review": 0,
                "running": 0,
                "terminal": 0,
            },
            "by_serving_block": [],
            "stale_active": 0,
            "plan_sha256": None,
        }
        assert database.exists_calls - preterminal_exists_calls == 2
        assert database.summary_calls == 0
    assert receipt["decision"] == {
        "capability_result_status": "not_interpreted",
        "score_blind_reconciliation_required": False,
        "unresolved_cells": 0,
        "rollout_retry_performed": False,
        "score_read_or_generated": False,
    }
    assert receipt["privacy"]["score_values_included"] is False
    assert (
        "workloads.kueue.x-k8s.io",
        None,
        f"kueue.x-k8s.io/job-uid={JOB_UID}",
    ) in cluster.list_calls


def test_postgres_summary_preserves_uri_scheme_when_selecting_database(monkeypatch):
    from evals.fleet import rollout_postgres

    monkeypatch.setenv(
        "TEST_ROLLOUT_DATABASE_URL",
        "postgresql://user:password@postgres.example:5432/rollout?sslmode=disable",
    )
    observed: dict[str, str] = {}

    def fake_summary(dsn: str) -> dict[str, Any]:
        observed["dsn"] = dsn
        return {"total": 17}

    monkeypatch.setattr(rollout_postgres, "summary", fake_summary)
    result = launch.PostgresDatabase("TEST_ROLLOUT_DATABASE_URL").summary(DATABASE)
    assert result == {"total": 17}
    assert observed["dsn"] == (
        "postgresql://user:password@postgres.example:5432/" + DATABASE + "?sslmode=disable"
    )


@pytest.mark.parametrize(
    "dsn",
    [
        "host=postgres.example dbname=rollout",
        "https://postgres.example/rollout",
        "postgresql:///rollout",
        "postgresql://postgres.example/rollout#fragment",
    ],
)
def test_postgres_summary_rejects_non_uri_or_ambiguous_database_targets(monkeypatch, dsn):
    monkeypatch.setenv("TEST_ROLLOUT_DATABASE_URL", dsn)
    with pytest.raises(launch.HeldoutLaunchError, match="supported PostgreSQL URI"):
        launch.PostgresDatabase("TEST_ROLLOUT_DATABASE_URL").summary(DATABASE)


@pytest.mark.parametrize(
    "query",
    [
        "dbname=wrong_database",
        "%64bname=wrong_database",
        "sslmode=disable&dbname=wrong_database",
        "sslmode=disable&DATABASE=",
        "dbname=wrong_database&dbname=another_database",
    ],
)
def test_postgres_summary_rejects_query_database_overrides(monkeypatch, query):
    monkeypatch.setenv(
        "TEST_ROLLOUT_DATABASE_URL",
        f"postgresql://user:password@postgres.example:5432/rollout?{query}",
    )
    with pytest.raises(
        launch.HeldoutLaunchError,
        match="select its database only by URI path",
    ):
        launch.PostgresDatabase("TEST_ROLLOUT_DATABASE_URL").summary(DATABASE)


def test_terminal_receipt_path_uses_existing_evaluation_output(tmp_path):
    output = tmp_path / "evaluation"
    output.mkdir()
    assert (
        launch.terminal_receipt_path(str(output), JOB_NAME, fallback_root=tmp_path)
        == output / "TERMINAL_OBSERVATION.json"
    )


def test_terminal_receipt_path_falls_back_when_evaluation_never_started(tmp_path):
    missing = tmp_path / "evaluation-never-created"
    assert (
        launch.terminal_receipt_path(str(missing), JOB_NAME, fallback_root=tmp_path)
        == tmp_path / f"{JOB_NAME}-TERMINAL_OBSERVATION.json"
    )


def test_terminal_receipt_path_rejects_ambiguous_roots(tmp_path):
    regular = tmp_path / "regular"
    regular.write_text("not a directory", encoding="utf-8")
    with pytest.raises(launch.HeldoutLaunchError, match="not a directory"):
        launch.terminal_receipt_path(str(regular), JOB_NAME, fallback_root=tmp_path)

    symlink = tmp_path / "symlink"
    symlink.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(launch.HeldoutLaunchError, match="must not be a symlink"):
        launch.terminal_receipt_path(str(symlink), JOB_NAME, fallback_root=tmp_path)
