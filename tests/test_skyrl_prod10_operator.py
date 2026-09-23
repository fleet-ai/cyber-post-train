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
from training import skyrl_prod10_operator as operator
from training import skyrl_prod9_direct as direct
from training import skyrl_prod9_training as training
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
        dev_preview=preview,
        dev_duplicate_proof=duplicate,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)

    assert packet["precreate_recovery"] == operator.stage_recovery_binding()
    assert proof["failure_alerts"] == "off"
    assert proof["priority"] == "c1"
    assert proof["queue_priority"] == "q1"
    assert proof["gpus"] == 0
    assert proof["source_bytes"] < 1024 * 1024
    assert preview["queue_priority"] == "q1"
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
    mounts = {value["name"]: value for value in container["volumeMounts"]}
    assert mounts["sfs-ro"] == {"name": "sfs-ro", "mountPath": "/mnt/sfs", "readOnly": True}
    assert mounts["controls-rw"] == {
        "name": "controls-rw",
        "mountPath": operator_job.CONTROLS_PATH,
        "subPath": operator_job.CONTROLS_SUBPATH,
    }
    assert "nvidia.com/gpu" not in json.dumps(package.job, sort_keys=True)


def test_prod10_runtime_derives_root_job_uid_from_exact_pod_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, stage, preview, duplicate = _stage_inputs()
    packet = operator_job.stage_packet(
        identity=identity,
        stage=stage,
        dev_preview=preview,
        dev_duplicate_proof=duplicate,
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


def test_stage_v5_preserves_and_reconciles_exact_v4_precreate_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, stage, preview, duplicate = _stage_inputs()
    packet = operator_job.stage_packet(
        identity=identity,
        stage=stage,
        dev_preview=preview,
        dev_duplicate_proof=duplicate,
    )
    expected = direct.stage_job_manifest(stage, identity=identity)
    root = tmp_path / "prod9-create-once-v1"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(operator.hardening, "CREATE_ONCE_ROOT", root)
    monkeypatch.setattr(direct, "RUNTIME_UID", os.geteuid())
    monkeypatch.setattr(direct, "RUNTIME_GID", os.getegid())
    operation_root = operator.hardening.stage_operation_root(stage)
    operation_root.mkdir(mode=0o700)
    recovery = operator.stage_recovery_binding()
    operator._write_once(
        operation_root / "STAGE_OPERATOR_INTENT.json",
        operator._seal(
            {
                "schema": "cyber_skyrl_prod10_operator_intent_v1",
                "phase": "stage",
                "packet_sha256": recovery["previous_packet_sha256"],
                "operator_job_uid": recovery["previous_operator_job_uid"],
                "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ),
    )
    operator._write_once(
        operation_root / "STAGE_OBSERVER_ARMED.json",
        direct._seal(
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
                "observer_pid": 7,
                "creator_binding_path": str(
                    operator.hardening.creator_binding_path(operation_root, "stage")
                ),
            }
        ),
    )

    runner = incluster_kubernetes.InClusterKubernetesRunner(
        request=lambda *_args: (404, b'{"kind":"Status"}')
    )
    assert operator._reconcile_stage_v4_precreate(
        packet,
        stage=stage,
        expected=expected,
        identity=identity,
        runner=runner,
    ) == operation_root
    assert not (operation_root / "STAGE_OPERATOR_INTENT.json").exists()
    assert not (operation_root / "STAGE_OBSERVER_ARMED.json").exists()
    assert (operation_root / "STAGE_OPERATOR_INTENT.v4.failed.json").is_file()
    assert (operation_root / "STAGE_OBSERVER_ARMED.v4.failed.json").is_file()
    receipt = json.loads(
        (operation_root / "STAGE_OPERATOR_RECOVERY_V5.json").read_bytes()
    )
    assert receipt["status"] == "v4_precreate_evidence_preserved"
    assert receipt == operator._seal(receipt)


def test_prod10_operator_package_rejects_root_alert_or_packet_drift() -> None:
    identity, stage, preview, duplicate = _stage_inputs()
    packet = operator_job.stage_packet(
        identity=identity,
        stage=stage,
        dev_preview=preview,
        dev_duplicate_proof=duplicate,
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
            dev_preview=preview,
            dev_duplicate_proof=duplicate,
        )
    )

    class FakeKubectl:
        def __init__(self, context: str) -> None:
            self.context = context

        def dry_run(self, manifest: dict) -> dict:
            return copy.deepcopy(manifest)

        def list_operator_resources(self, resource: str) -> dict:
            assert resource in operator_launch._RESOURCES
            return {"kind": "List", "items": []}

        def get_operator_object(self, resource: str, name: str) -> None:
            assert resource == "configmap" and name.endswith(("-source", "-packet"))
            return None

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
    absence = operator_launch.duplicate_proof(package, factory=FakeKubectl)
    assert absence["kubernetes_inventories_checked"] == 12
    refreshed_preview = copy.deepcopy(preview)
    refreshed_preview["server_render_sha256"] = "sha256:" + "f" * 64
    refreshed_preview = direct._seal(refreshed_preview)
    refreshed_package = operator_job.build_operator_package(
        operator_job.stage_packet(
            identity=identity,
            stage=stage,
            dev_preview=refreshed_preview,
            dev_duplicate_proof=duplicate,
        )
    )
    assert refreshed_package.packet["sha256"] != package.packet["sha256"]
    assert operator_launch._validate_duplicate(refreshed_package, absence) == absence

    changed = copy.deepcopy(previews)
    changed[0]["failure_alerts"] = "on"
    changed[0] = operator_launch._seal(changed[0])
    with pytest.raises(JobsError, match="preview"):
        operator_launch._preview_set(package, changed)


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
        return 201 if method == "POST" else 200, b'{"kind":"Status","status":"Success"}'

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
