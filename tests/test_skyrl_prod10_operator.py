from __future__ import annotations

import copy
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from cyber_post_train import skyrl_prod10_operator_launch as operator_launch
from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, JobsError, digest
from training import dev_cleanup_observer as cleanup
from training import incluster_kubernetes
from training import skyrl_prod9_direct as direct
from training import skyrl_prod9_training as training
from training import skyrl_prod10_operator as operator
from training import skyrl_reward_rayjob as historical

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod10-identity-v1.json"
PREDECESSOR = ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json"


def _cpu_render(value: dict) -> dict:
    rendered = copy.deepcopy(value)
    uid = "00000000-0000-4000-8000-000000000002"
    name = value["metadata"]["name"]
    generated = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": name,
        "controller-uid": uid,
        "job-name": name,
    }
    rendered["metadata"].update(
        {
            "creationTimestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generation": 1,
            "uid": uid,
            "labels": {**value["metadata"].get("labels", {}), **generated},
        }
    )
    rendered["status"] = {}
    rendered["spec"].update(
        {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
        }
    )
    if "suspend" not in value["spec"]:
        rendered["spec"]["suspend"] = False
    rendered["spec"]["template"]["metadata"]["labels"] = generated
    rendered["spec"]["template"]["spec"].update(
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
        }
    )
    rendered["spec"]["template"]["spec"]["containers"][0]["imagePullPolicy"] = "IfNotPresent"
    return rendered


def _stage_inputs() -> tuple[historical.RailIdentity, dict, dict, dict]:
    identity = historical.load_identity(IDENTITY)
    predecessor = json.loads(PREDECESSOR.read_bytes())
    stage = training.stage_spec(identity, predecessor)
    manifest = direct.stage_job_manifest(stage, identity=identity)
    dev_preview = direct.validate_cpu_preview(
        manifest,
        _cpu_render(manifest),
        context=direct.DEV_CONTEXT,
        purpose="stage",
    )
    duplicate = direct._seal(
        {
            "schema": direct.CPU_DUPLICATE_PROOF_SCHEMA,
            "status": "identity_absent",
            "context": direct.DEV_CONTEXT,
            "namespace": direct.NAMESPACE,
            "name": identity.stage_name,
            "kubernetes_inventories_checked": 5,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    return identity, stage, dev_preview, duplicate


def test_prod10_operator_package_is_exact_alert_off_c1_q1_zero_gpu() -> None:
    identity, stage, preview, duplicate = _stage_inputs()
    packet = operator_job.stage_packet(
        identity=identity,
        stage=stage,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)

    assert set(packet) == {
        "schema",
        "phase",
        "operator_name",
        "identity",
        "stage",
        "precreate_recovery",
        "sha256",
    }
    assert packet["precreate_recovery"] == operator.stage_recovery_binding()
    assert proof["failure_alerts"] == "off"
    assert proof["priority"] == "c1"
    assert proof["queue_priority"] == "q1"
    assert proof["gpus"] == 0
    assert proof["source_bytes"] < 1024 * 1024
    assert package.job["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] == "off"
    pod = package.job["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "default"
    assert pod["automountServiceAccountToken"] is True
    [container] = pod["containers"]
    environment = {value["name"]: value for value in container["env"]}
    assert environment["OPERATOR_JOB_NAME"] == {
        "name": "OPERATOR_JOB_NAME",
        "value": operator.OPERATOR_NAMES["stage"],
    }
    assert environment["OPERATOR_POD_NAME"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.name"
    assert environment["OPERATOR_POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
    assert "OPERATOR_JOB_UID" not in environment
    assert container["securityContext"]["runAsUser"] == 1000
    assert container["securityContext"]["runAsGroup"] == 100
    assert container["resources"] == {
        "requests": {"cpu": "2", "memory": "8Gi"},
        "limits": {"cpu": "4", "memory": "16Gi"},
    }
    mounts = {value["name"]: value for value in container["volumeMounts"]}
    assert mounts["sfs"] == {"name": "sfs", "mountPath": "/mnt/sfs"}
    assert mounts["controls-rw"] == {
        "name": "controls-rw",
        "mountPath": operator_job.CONTROLS_PATH,
        "subPath": operator_job.CONTROLS_SUBPATH,
    }
    volumes = {value["name"]: value for value in pod["volumes"]}
    assert volumes["sfs"] == {
        "name": "sfs",
        "persistentVolumeClaim": {"claimName": operator_job.PVC},
    }
    assert "nvidia.com/gpu" not in json.dumps(package.job, sort_keys=True)


def test_prod10_manifest_operator_is_distinct_read_only_and_recovery_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, stage, _preview, _duplicate = _stage_inputs()
    launch = {"schema": "synthetic-stage-launch"}
    monkeypatch.setattr(direct, "_direct_stage_launch", lambda value, *_args, **_kwargs: value)
    packet = operator_job.manifest_packet(
        identity=identity,
        stage=stage,
        stage_launch_result=launch,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    container = package.job["spec"]["template"]["spec"]["containers"][0]
    mounts = {item["name"]: item for item in container["volumeMounts"]}

    assert packet["preflight_v1_failure"] == operator.preflight_v1_failure_binding()
    assert proof == {
        **proof,
        "phase": "manifest",
        "name": "chris-q38-prod10-manifest-operator-v1",
        "failure_alerts": "off",
        "priority": "c1",
        "queue_priority": "q1",
        "gpus": 0,
    }
    assert mounts["sfs"]["readOnly"] is True
    assert "nvidia.com/gpu" not in json.dumps(package.job, sort_keys=True)

    changed = copy.deepcopy(packet)
    changed["preflight_v1_failure"]["operator_job_uid"] = (
        "00000000-0000-4000-8000-000000000001"
    )
    changed["preflight_v1_failure"] = operator._seal(changed["preflight_v1_failure"])
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="recovery"):
        operator_job.build_operator_package(changed)


def test_prod10_manifest_result_exports_only_sanitized_public_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, stage, _preview, _duplicate = _stage_inputs()
    successor = copy.deepcopy(stage["predecessor_manifest"])
    successor["name"] = identity.run_name
    successor["sha256"] = "sha256:" + digest(
        {key: value for key, value in successor.items() if key != "sha256"}
    )
    result_path = operator.hardening.stage_operation_root(stage) / "STAGE_OPERATOR_RESULT.json"
    stage_result = operator._seal(
        {"schema": operator.DIRECT_STAGE_RESULT_SCHEMA, "status": "stage_ready"}
    )
    launch = operator._seal(
        {
            "schema": direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA,
            "observer": {
                "receipt": {
                    "result_path": str(result_path),
                    "result_sha256": stage_result["sha256"],
                },
                "workload_name": "job-stage-abcde",
                "pod_names": [],
            },
            "created": {
                "source_config_map": {"name": operator.OPERATOR_NAMES["stage"] + "-source"},
                "packet_config_map": {"name": operator.OPERATOR_NAMES["stage"] + "-packet"},
            },
        }
    )
    monkeypatch.setattr(direct, "_direct_stage_launch", lambda *_args, **_kwargs: launch)
    monkeypatch.setattr(operator, "_read_recovery_file", lambda *_args: stage_result)
    monkeypatch.setattr(
        direct,
        "_direct_stage_rebound_evidence",
        lambda *_args, **_kwargs: (stage_result, launch, successor),
    )
    absent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        operator,
        "_resource_absent",
        lambda _runner, resource, name, **_kwargs: absent.append((resource, name)),
    )

    result = operator.run_manifest(
        {
            "identity": identity.sealed_mapping(),
            "stage": stage,
            "stage_launch_result": launch,
        },
        runner=object(),
    )

    assert result["schema"] == operator.MANIFEST_RESULT_SCHEMA
    assert result["successor_manifest"] == successor
    assert result["private_rows_exported"] is False
    assert result["nested_jobs_created"] == result["gpus"] == 0
    assert len(json.dumps(result, sort_keys=True, separators=(",", ":"))) < 3900
    assert ("job", "chris-q38-prod10-preflight-operator-v1") in absent
    assert ("job", identity.preflight_name) in absent


def test_prod10_runtime_derives_root_job_uid_from_exact_pod_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, stage, preview, duplicate = _stage_inputs()
    packet = operator_job.stage_packet(
        identity=identity,
        stage=stage,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    name = operator.OPERATOR_NAMES["stage"]
    pod_name = name + "-abcde"
    pod_uid = "00000000-0000-4000-8000-000000000010"
    job_uid = "00000000-0000-4000-8000-000000000011"

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        resource, target = command[6:8]
        if resource == "pod":
            value = {
                "metadata": {
                    "name": pod_name,
                    "uid": pod_uid,
                    "ownerReferences": [
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "name": name,
                            "uid": job_uid,
                            "controller": True,
                        }
                    ],
                }
            }
        else:
            value = copy.deepcopy(package.job)
            value["metadata"]["uid"] = job_uid
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    monkeypatch.setattr(operator.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(operator.os, "getegid", lambda: 100)
    monkeypatch.setenv("OPERATOR_JOB_NAME", name)
    monkeypatch.setenv("OPERATOR_POD_NAME", pod_name)
    monkeypatch.setenv("OPERATOR_POD_UID", pod_uid)
    monkeypatch.setenv("OPERATOR_PACKET_SHA256", packet["sha256"])
    monkeypatch.setenv("OPERATOR_SOURCE_SHA256", proof["source_sha256"])
    assert operator._validate_runtime(packet, runner) == job_uid

    monkeypatch.setenv("OPERATOR_JOB_NAME", pod_name)
    with pytest.raises(operator.OperatorFailure, match="runtime_job_name_rejected"):
        operator._validate_runtime(packet, runner)


def test_stage_v7_preserves_released_v6_and_rebinds_without_nested_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, stage, preview, duplicate = _stage_inputs()
    packet = operator_job.stage_packet(identity=identity, stage=stage)
    root = tmp_path / "prod9-create-once-v1"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(operator.hardening, "CREATE_ONCE_ROOT", root)
    root_identity = root.stat()
    monkeypatch.setattr(operator, "RUNTIME_UID", root_identity.st_uid)
    monkeypatch.setattr(operator, "RUNTIME_GID", root_identity.st_gid)
    monkeypatch.setattr(direct, "live_create_is_available", lambda: True)
    expected = direct.stage_job_manifest(stage, identity=identity)
    manifest_sha256 = "sha256:" + direct.digest(expected)
    operation_root = operator.hardening.stage_operation_root(stage)
    operation_root.mkdir(mode=0o700)
    v4_recovery = operator._seal(operator._STAGE_V4_RECOVERY)
    v4_intent = operator._seal(
        {
            "schema": "cyber_skyrl_prod10_operator_intent_v1",
            "phase": "stage",
            "packet_sha256": v4_recovery["previous_packet_sha256"],
            "operator_job_uid": v4_recovery["previous_operator_job_uid"],
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    v4_armed = direct._seal(
        {
            "schema": "cyber_direct_cleanup_observer_armed_v1",
            "status": "armed",
            "context": direct.PROD_CONTEXT,
            "namespace": direct.NAMESPACE,
            "kind": "job",
            "name": identity.stage_name,
            "maximum_seconds": direct.CPU_MAXIMUM_SECONDS,
            "expected_gpus": 0,
            "plan_sha256": stage["sha256"],
            "manifest_sha256": "sha256:" + direct.digest(expected),
            "armed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observer_pid": 6,
            "creator_binding_path": str(
                operator.hardening.creator_binding_path(operation_root, "stage")
            ),
        }
    )
    operator._write_once(
        operation_root / "STAGE_OPERATOR_INTENT.v4.failed.json", v4_intent
    )
    operator._write_once(
        operation_root / "STAGE_OBSERVER_ARMED.v4.failed.json", v4_armed
    )
    operator._write_once(
        operation_root / "STAGE_OPERATOR_RECOVERY_V5.json",
        operator._seal(
            {
                "schema": "cyber_skyrl_prod10_stage_precreate_recovery_receipt_v1",
                "status": "v4_precreate_evidence_preserved",
                "binding_sha256": v4_recovery["sha256"],
                "previous_intent_sha256": v4_intent["sha256"],
                "previous_observer_sha256": v4_armed["sha256"],
                "archived_files": [
                    "STAGE_OBSERVER_ARMED.v4.failed.json",
                    "STAGE_OPERATOR_INTENT.v4.failed.json",
                ],
                "gpus": 0,
            }
        ),
    )
    v5_recovery = operator._seal(operator._STAGE_V5_RECOVERY)
    v5_intent = operator._seal(
        {
            "schema": "cyber_skyrl_prod10_operator_intent_v1",
            "phase": "stage",
            "packet_sha256": v5_recovery["previous_packet_sha256"],
            "operator_job_uid": v5_recovery["previous_operator_job_uid"],
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    v5_armed = direct._seal(
        {
            **{key: value for key, value in v4_armed.items() if key != "sha256"},
            "observer_pid": 7,
        }
    )
    operator._write_once(
        operation_root / "STAGE_OPERATOR_INTENT.v5.failed.json", v5_intent
    )
    operator._write_once(
        operation_root / "STAGE_OBSERVER_ARMED.v5.failed.json", v5_armed
    )
    operator._write_once(
        operation_root / "STAGE_OPERATOR_RECOVERY_V6.json",
        operator._seal(
            {
                "schema": "cyber_skyrl_prod10_stage_precreate_recovery_receipt_v1",
                "status": "v5_precreate_evidence_preserved",
                "binding_sha256": v5_recovery["sha256"],
                "previous_intent_sha256": v5_intent["sha256"],
                "previous_observer_sha256": v5_armed["sha256"],
                "archived_files": [
                    "STAGE_OBSERVER_ARMED.v5.failed.json",
                    "STAGE_OPERATOR_INTENT.v5.failed.json",
                ],
                "gpus": 0,
            }
        ),
    )
    recovery = operator.stage_recovery_binding()
    v6_intent = operator._seal(
        {
            "schema": "cyber_skyrl_prod10_operator_intent_v1",
            "phase": "stage",
            "packet_sha256": recovery["previous_packet_sha256"],
            "operator_job_uid": recovery["previous_operator_job_uid"],
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    operator._write_once(operation_root / "STAGE_OPERATOR_INTENT.json", v6_intent)
    v6_armed = direct._seal(
        {
            **{key: value for key, value in v4_armed.items() if key != "sha256"},
            "observer_pid": 8,
        }
    )
    operator._write_once(
        operation_root / "STAGE_OBSERVER_ARMED.json", v6_armed
    )
    creator = direct._seal(
        {
            "schema": cleanup.CREATOR_BINDING_SCHEMA,
            "status": "created_once",
            "context": direct.PROD_CONTEXT,
            "namespace": direct.NAMESPACE,
            "kind": "job",
            "name": identity.stage_name,
            "plan_sha256": stage["sha256"],
            "manifest_sha256": manifest_sha256,
            "uid": recovery["previous_target_job_uid"],
        }
    )
    operator._write_once(
        operation_root / "STAGE_OBSERVER_ARMED.json.created.json", creator
    )
    authorization_sha256 = "sha256:" + "1" * 64
    created = direct._seal(
        {
            "schema": direct.CPU_CREATED_SCHEMA,
            "status": "created_once",
            "purpose": "stage",
            "name": identity.stage_name,
            "plan_sha256": stage["sha256"],
            "manifest_sha256": manifest_sha256,
            "authorization_sha256": authorization_sha256,
            "job_uid": recovery["previous_target_job_uid"],
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    journal = operation_root / "PROD9_STAGE_CREATE.jsonl"
    journal.write_text(
        "\n".join(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in (
                {
                    "state": "CREATE_INTENT_DO_NOT_RETRY",
                    "purpose": "stage",
                    "plan_sha256": stage["sha256"],
                    "manifest_sha256": manifest_sha256,
                    "authorization_sha256": authorization_sha256,
                    "duplicate_checks": {
                        "kubernetes_inventories_checked": 10,
                        "development_duplicate_proof_sha256": "sha256:" + "2" * 64,
                    },
                },
                created,
            )
        )
        + "\n"
    )
    journal.chmod(0o600)
    released = direct._seal(
        {
            "schema": cleanup.DIRECT_RESULT_SCHEMA,
            "status": "released_without_accepted_execution",
            "context": direct.PROD_CONTEXT,
            "namespace": direct.NAMESPACE,
            "kind": "job",
            "name": identity.stage_name,
            "plan_sha256": stage["sha256"],
            "manifest_sha256": manifest_sha256,
            "expected_gpus": 0,
            "maximum_seconds": direct.CPU_MAXIMUM_SECONDS,
            "armed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "uid": recovery["previous_target_job_uid"],
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "terminal_status": "Deleted",
            "job_id": "",
            "rayjob_name": "",
            "rayjob_uid": "",
            "workload_name": recovery["previous_target_workload_name"],
            "workload_uid": recovery["previous_target_workload_uid"],
            "raycluster_name": "",
            "raycluster_uid": "",
            "pod_names": [],
            "pod_uids": [],
            "image_ids": [],
            "exit_codes": [],
            "termination_reasons": [],
            "restarts": 0,
            "peak_gpus": 0,
            "receipt": None,
            "observer_error_class": "",
            "observation_failures": 0,
            "max_consecutive_observation_failures": 0,
            "last_observation_error_code": "",
            "deletion_reason": "target_absent",
            "deletion_requested_at": "",
            "target_present": False,
            "pods_present": False,
            "rayjob_present": False,
            "workload_present": False,
            "raycluster_present": False,
            "active_gpus": 0,
            "release_observed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    operator._write_once(operation_root / "STAGE_OBSERVER_RESULT.json", released)

    calls: list[tuple[str, str]] = []

    def request(method: str, path: str, *_: object) -> tuple[int, bytes]:
        calls.append((method, path))
        return 404, b'{"kind":"Status"}'

    runner = incluster_kubernetes.InClusterKubernetesRunner(request=request)
    raw_receipt = {
        "schema": training.STAGE_RECEIPT_SCHEMA,
        "status": "published",
        "gpus": 0,
    }
    monkeypatch.setattr(training, "stage_rebind", lambda *_args, **_kwargs: raw_receipt)
    validated: list[dict] = []
    monkeypatch.setattr(
        direct,
        "_stage_receipt",
        lambda _stage, receipt, **_kwargs: validated.append(receipt),
    )
    monkeypatch.setenv(
        "OPERATOR_JOB_UID", "00000000-0000-4000-8000-000000000099"
    )
    monkeypatch.setenv("OPERATOR_SOURCE_SHA256", "sha256:" + "9" * 64)

    result = operator.run_stage(packet, runner=runner)

    assert result["schema"] == operator.DIRECT_STAGE_RESULT_SCHEMA
    assert result["status"] == "stage_ready"
    assert result["execution"] == {
        "kind": "job",
        "name": operator.OPERATOR_NAMES["stage"],
        "uid": "00000000-0000-4000-8000-000000000099",
        "image": stage["image"],
        "source_sha256": "sha256:" + "9" * 64,
        "sfs_output": stage["destination"],
        "receipt_sha256": digest(raw_receipt),
        "nested_jobs_created": 0,
    }
    assert not {"authorization", "created", "release"}.intersection(result)
    assert result["execution"]["nested_jobs_created"] == 0
    assert validated == [{**raw_receipt, "receipt_sha256": digest(raw_receipt)}]
    assert calls and all(method == "GET" for method, _path in calls)
    assert not (operation_root / "STAGE_OPERATOR_INTENT.v6.failed.json").is_symlink()
    assert (operation_root / "STAGE_OPERATOR_INTENT.v6.failed.json").is_file()
    assert (operation_root / "STAGE_OBSERVER_RESULT.v6.failed.json").is_file()
    assert (operation_root / "PROD9_STAGE_CREATE.v6.failed.jsonl").is_file()
    assert (operation_root / "STAGE_OPERATOR_INTENT.json").is_file()
    receipt = json.loads(
        (operation_root / "STAGE_OPERATOR_RECOVERY_V7.json").read_bytes()
    )
    assert receipt["status"] == "v6_released_child_evidence_preserved"
    assert receipt == operator._seal(receipt)


def test_prod10_operator_package_rejects_root_alert_or_packet_drift() -> None:
    identity, stage, preview, duplicate = _stage_inputs()
    packet = operator_job.stage_packet(
        identity=identity,
        stage=stage,
    )
    package = operator_job.build_operator_package(packet)
    package.job["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] = "on"
    with pytest.raises(ValueError, match="tracked exact bytes"):
        operator_job.validate_operator_package(package)

    changed = copy.deepcopy(packet)
    changed["manifest_sha256"] = "sha256:" + "0" * 64
    changed = operator._seal(changed)
    with pytest.raises((JobsError, ValueError), match="packet|manifest"):
        operator_job.build_operator_package(changed)

    changed = copy.deepcopy(packet)
    changed["precreate_recovery"]["previous_release_sha256"] = "sha256:" + "0" * 64
    changed["precreate_recovery"] = operator._seal(changed["precreate_recovery"])
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="recovery"):
        operator_job.build_operator_package(changed)


def test_prod10_operator_launcher_requires_two_alert_off_c1_q1_previews_and_absence() -> None:
    identity, stage, preview, duplicate = _stage_inputs()
    package = operator_job.build_operator_package(
        operator_job.stage_packet(
            identity=identity,
            stage=stage,
        )
    )

    class FakeKubectl:
        def __init__(self, context: str) -> None:
            self.context = context

        def dry_run(self, manifest: dict) -> dict:
            value = copy.deepcopy(manifest)
            if value.get("kind") == "Job":
                value["metadata"]["uid"] = (
                    "00000000-0000-4000-8000-000000000021"
                    if self.context == direct.DEV_CONTEXT
                    else "00000000-0000-4000-8000-000000000022"
                )
            return value

        def list_operator_resources(self, resource: str) -> dict:
            raise AssertionError(f"unbounded inventory used for {resource}")

        def get_operator_object(self, resource: str, name: str) -> None:
            assert resource in {
                "configmap",
                "job",
                "workload",
                "rayjob",
                "raycluster",
            }
            return None

        def list_operator_pods(self, job_name: str) -> dict:
            assert job_name == operator.OPERATOR_NAMES["stage"]
            return {"kind": "PartialObjectMetadataList", "items": []}

    previews = operator_launch.server_previews(package, factory=FakeKubectl)
    assert {value["context"] for value in previews} == {
        direct.DEV_CONTEXT,
        direct.PROD_CONTEXT,
    }
    assert all(
        value["failure_alerts"] == "off"
        and value["priority"] == "c1"
        and value["queue_priority"] == "q1"
        and value["gpus"] == 0
        for value in previews
    )
    absence = operator_launch.duplicate_proof(
        package, previews=previews, factory=FakeKubectl
    )
    assert absence["kubernetes_inventories_checked"] == 12
    assert absence["derived_workload_names"] == {
        direct.DEV_CONTEXT: "job-chris-q38-prod10-stage-operator-v7-12a46",
        direct.PROD_CONTEXT: "job-chris-q38-prod10-stage-operator-v7-3286c",
    }
    assert operator_launch._validate_duplicate(package, absence, previews) == absence

    changed = copy.deepcopy(previews)
    changed[0]["failure_alerts"] = "on"
    changed[0] = operator_launch._seal(changed[0])
    with pytest.raises(JobsError, match="preview"):
        operator_launch._preview_set(package, changed)


def test_prod10_operator_workload_duplicate_gate_is_exact_and_never_lists_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert operator_launch._derived_workload_name(
        "chris-q38-dev17-s51-base-p1-v1",
        "75d47351-61c7-4bd1-995a-93cbfd112d0d",
    ) == "job-chris-q38-dev17-s51-base-p1-v1-c7f37"
    client = operator_launch.Kubectl(direct.PROD_CONTEXT)
    calls: list[list[str]] = []

    def run_text(args: list[str], *, manifest: dict | None = None) -> str:
        assert manifest is None
        calls.append(args)
        return ""

    monkeypatch.setattr(client, "_run_text", run_text)
    name = "job-chris-q38-prod10-stage-operator-v7-3286c"
    assert client.get_operator_object("workload", name) is None
    assert client.list_operator_pods(operator.OPERATOR_NAMES["stage"])["items"] == []
    assert calls[0][:3] == ["get", "workload", name]
    assert "--selector=batch.kubernetes.io/job-name=" + operator.OPERATOR_NAMES["stage"] in calls[1]
    with pytest.raises(JobsError, match="unsupported"):
        client.list_operator_resources("workloads.kueue.x-k8s.io")
    assert all(
        not (call[:2] == ["get", "workloads.kueue.x-k8s.io"] and len(call) < 3)
        for call in calls
    )


def test_cpu_remote_duplicate_proof_still_rechecks_production() -> None:
    identity, _, _, _ = _stage_inputs()
    commands: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    proof = direct.cpu_duplicate_proof(
        identity.stage_name, context=direct.DEV_CONTEXT, runner=runner
    )
    assert len(commands) == 5
    assert all(direct.DEV_CONTEXT in command for command in commands)
    commands.clear()

    result = direct._cpu_duplicate_checks(
        identity.stage_name, runner=runner, dev_proof=proof
    )
    assert result["kubernetes_inventories_checked"] == 10
    assert result["development_duplicate_proof_sha256"] == proof["sha256"]
    assert len(commands) == 5
    assert all(direct.PROD_CONTEXT in command for command in commands)


def test_incluster_runner_allows_only_prod_get_create_and_uid_cas_delete() -> None:
    calls: list[tuple[str, str, bytes | None]] = []

    def request(
        method: str, path: str, body: bytes | None, _: dict[str, str]
    ) -> tuple[int, bytes]:
        calls.append((method, path, body))
        if method == "GET":
            return 200, b'{"kind":"List","items":[]}'
        if method == "POST":
            value = json.loads(body or b"{}")
            value["metadata"]["managedFields"] = [
                {"manager": "cyber-post-train-prod10", "operation": "Update"}
            ]
            return 201, json.dumps(value).encode()
        return 200, b'{"kind":"Status","status":"Success"}'

    runner = incluster_kubernetes.InClusterKubernetesRunner(request=request)
    prefix = [
        "kubectl",
        "--context",
        direct.PROD_CONTEXT,
        "--namespace",
        direct.NAMESPACE,
    ]
    result = runner(prefix + ["get", "pod", "--selector", "job-name=exact", "--output", "json"])
    assert result.returncode == 0
    assert calls[-1][:2] == (
        "GET",
        "/api/v1/namespaces/fleet-train-jobs/pods?labelSelector=job-name%3Dexact",
    )

    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "exact", "namespace": direct.NAMESPACE},
        "spec": {},
    }
    result = runner(
        prefix + ["create", "--dry-run=server", "-f", "-", "-o", "json"],
        input=json.dumps(manifest),
    )
    assert result.returncode == 0
    assert calls[-1][0] == "POST"
    assert "dryRun=All" in calls[-1][1]
    assert "managedFields" not in json.loads(result.stdout)["metadata"]

    delete = json.dumps(
        {
            "apiVersion": "v1",
            "kind": "DeleteOptions",
            "propagationPolicy": "Foreground",
            "preconditions": {"uid": "00000000-0000-4000-8000-000000000001"},
        }
    )
    path = "/apis/batch/v1/namespaces/fleet-train-jobs/jobs/exact"
    result = runner(prefix + ["delete", "--raw", path, "-f", "-"], input=delete)
    assert result.returncode == 0
    assert calls[-1][0] == "DELETE"

    with pytest.raises(incluster_kubernetes.InClusterKubernetesError, match="production"):
        runner(
            [
                "kubectl",
                "--context",
                direct.DEV_CONTEXT,
                "--namespace",
                direct.NAMESPACE,
                "get",
                "pod",
                "--output=json",
            ]
        )
    with pytest.raises(incluster_kubernetes.InClusterKubernetesError, match="UID-CAS"):
        runner(prefix + ["delete", "job", "exact"])
    with pytest.raises(incluster_kubernetes.InClusterKubernetesError, match="kind"):
        runner(
            prefix + ["create", "-f", "-", "-o", "json"],
            input=json.dumps({"apiVersion": "v1", "kind": "Pod", "metadata": {"namespace": direct.NAMESPACE}}),
        )


def test_operator_termination_receipt_is_accepted_by_exact_observer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = operator._seal({"schema": operator.RESULT_SCHEMA, "status": "stage_ready"})
    target = tmp_path / "termination-log"
    monkeypatch.setattr(operator, "_TERMINATION_PATH", target)
    operator._write_termination(phase="stage", result_path=Path("/fixed/result"), result=result)
    value = json.loads(target.read_text())
    assert value["schema"] == operator.TERMINATION_SCHEMA
    assert value["status"] == "passed"
    assert value["sha256"] == "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )


def test_operator_failure_termination_receipt_is_sanitized_and_not_accepted_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "termination-log"
    monkeypatch.setattr(operator, "_TERMINATION_PATH", target)
    operator._write_failure_termination(
        phase="stage", error=operator.OperatorFailure("sfs_control_parent_rejected")
    )
    value = json.loads(target.read_text())
    assert value == operator._seal(
        {
            "schema": operator.FAILURE_TERMINATION_SCHEMA,
            "status": "failed",
            "phase": "stage",
            "error_class": "OperatorFailure",
            "error_code": "sfs_control_parent_rejected",
            "gpus": 0,
        }
    )
    assert cleanup._validated_receipt(json.dumps(value), kind="job") == value
    assert cleanup._receipt_execution_accepted(value) is False
    assert "bootstrap_failure" in operator_job._BOOTSTRAP
    assert "cyber_skyrl_prod10_bootstrap_failure_v1" in operator_job._BOOTSTRAP
