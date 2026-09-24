from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train import skyrl_prod10_operator_job as operator_job
from cyber_post_train import skyrl_prod10_operator_launch as operator_launch
from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, JobsError, digest
from training import dev_cleanup_observer as cleanup
from training import incluster_kubernetes
from training import skyrl_prod9_direct as direct
from training import skyrl_prod9_hardening as hardening
from training import skyrl_prod9_training as training
from training import skyrl_prod10_direct as launch_direct
from training import skyrl_prod10_operator as operator
from training import skyrl_reward_rayjob as historical

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod10-identity-v1.json"
PREDECESSOR = ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json"


def _dev_preview_provenance(preview: dict) -> dict:
    return direct._seal(
        {
            "schema": launch_direct.SEALED_DEV_PREVIEW_PROVENANCE_SCHEMA,
            "status": "fresh_sealed_external_dev_server_preview_validated",
            "context": direct.DEV_CONTEXT,
            "sealed_dev_server_preview_sha256": preview["sha256"],
            "checked_at": preview["checked_at"],
        }
    )


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
    assert package.job["spec"]["activeDeadlineSeconds"] == 2400
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


def test_prod10_launch_package_is_alert_off_c1_q1_and_capacity_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = historical.load_identity(IDENTITY)
    plan = {"schema": training.SCHEMA}
    request = {"workers": 1, "gpus_per_worker": 8}
    monkeypatch.setattr(direct, "_identity", lambda _plan, bound: bound)
    monkeypatch.setattr(training, "job_request", lambda _plan: request)
    monkeypatch.setattr(launch_direct, "_preflight_launch", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(direct, "_source", lambda value: value)
    monkeypatch.setattr(launch_direct, "_duplicate", lambda value, _identity, **_kwargs: value)
    preflight = direct._seal({"schema": direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA, "gpus": 0})
    preview = direct._seal(
        {
            "schema": direct.PREVIEW_SCHEMA,
            "context": direct.DEV_CONTEXT,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    duplicate = direct._seal(
        {"schema": launch_direct.DUPLICATE_SCHEMA, "status": "identities_absent"}
    )
    capacity = {
        "schema": "cyber_project_gpu_capacity_census_v1",
        "limits": {"nodes": 10, "gpus": 80},
        "planned": {"nodes": 1, "gpus": 8},
    }
    capacity["sha256"] = digest(capacity)
    packet = operator_job.launch_packet(
        identity=identity,
        plan=plan,
        request=request,
        preflight_launch_result=preflight,
        source_preview={"manifest_yaml": "{}"},
        manifest_sha256="sha256:" + "1" * 64,
        dev_preview=preview,
        dev_preview_provenance=_dev_preview_provenance(preview),
        duplicate_proof=duplicate,
        capacity_census=capacity,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    container = package.job["spec"]["template"]["spec"]["containers"][0]

    assert proof["phase"] == "launch"
    assert proof["name"] == "chris-q38-prod10-launch-operator-v8"
    assert proof["failure_alerts"] == "off"
    assert proof["priority"] == "c1"
    assert proof["queue_priority"] == "q1"
    assert proof["gpus"] == 0
    assert (
        package.job["spec"]["activeDeadlineSeconds"]
        == operator_job.LAUNCH_ACTIVE_DEADLINE_SECONDS
        > direct.MAXIMUM_SECONDS
    )
    assert (
        operator_launch.LAUNCH_OPERATOR_JOIN_SECONDS
        > operator_launch.LAUNCH_OPERATOR_OBSERVER_SECONDS
        > operator_job.LAUNCH_ACTIVE_DEADLINE_SECONDS
        > direct.MAXIMUM_SECONDS
    )
    outer = operator_launch._operator_observer(proof, tmp_path)
    assert outer.profile == "prod10-launch-operator"
    assert outer.maximum_seconds == operator_launch.LAUNCH_OPERATOR_OBSERVER_SECONDS
    with pytest.raises(cleanup.ObserverError, match="deadline"):
        cleanup.Observer(
            context=direct.PROD_CONTEXT,
            namespace=direct.NAMESPACE,
            kind="job",
            name=proof["name"],
            maximum_seconds=operator_launch.LAUNCH_OPERATOR_OBSERVER_SECONDS,
            expected_gpus=0,
            plan_sha256=proof["packet_sha256"],
            manifest_sha256=proof["job_manifest_sha256"],
            armed_path=tmp_path / "old-profile-armed.json",
            result_path=tmp_path / "old-profile-result.json",
            profile="production-operator",
        )
    assert container["envFrom"] == [
        {"secretRef": {"name": "fleet-api"}},
        {"secretRef": {"name": "wandb-api"}},
    ]
    environment = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    assert environment["HF_DATASETS_CACHE"] == "/work/hf-datasets"
    assert "WANDB_API_KEY" not in environment
    assert "private_rows" not in json.dumps(packet, sort_keys=True)

    changed = copy.deepcopy(packet)
    changed["capacity_census"]["limits"] = {"nodes": 8, "gpus": 64}
    changed["capacity_census"]["sha256"] = digest(
        {key: value for key, value in changed["capacity_census"].items() if key != "sha256"}
    )
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="capacity"):
        operator_job.build_operator_package(changed)

    changed = copy.deepcopy(packet)
    changed["launch_v1_failure"]["operator_job_uid"] = "00000000-0000-4000-8000-000000000001"
    changed["launch_v1_failure"] = operator._seal(changed["launch_v1_failure"])
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="failure predecessor"):
        operator_job.build_operator_package(changed)

    changed = copy.deepcopy(packet)
    changed["probe_v6_success"]["operator_job_uid"] = "00000000-0000-4000-8000-000000000001"
    changed["probe_v6_success"] = operator._seal(changed["probe_v6_success"])
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="repair proof"):
        operator_job.build_operator_package(changed)

    for key, match in (
        ("launch_v2_failure", "launch-v2 failure predecessor"),
        ("probe_v7_failure", "probe-v7 predecessor"),
        ("probe_v8_failure", "probe-v8 predecessor"),
        ("probe_v9_success", "probe-v9 predecessor"),
        ("launch_v3_recovery", "launch-v3 recovery predecessor"),
        ("inspect_v4_success", "inspector-v4 predecessor"),
        ("launch_v4_failure", "launch-v4 failure predecessor"),
        ("inspect_v5_success", "inspector-v5 predecessor"),
        ("launch_v5_failure", "launch-v5 failure predecessor"),
        ("inspect_v6_success", "inspector-v6 predecessor"),
        ("launch_v6_failure", "launch-v6 failure predecessor"),
        ("launch_v7_failure", "launch-v7 failure predecessor"),
    ):
        changed = copy.deepcopy(packet)
        changed[key]["operator_job_uid"] = "00000000-0000-4000-8000-000000000001"
        changed[key] = operator._seal(changed[key])
        changed = operator._seal(changed)
        with pytest.raises(ValueError, match=match):
            operator_job.build_operator_package(changed)

    freshness: list[bool] = []

    def duplicate(value: dict, _identity: object, *, fresh: bool = True) -> dict:
        freshness.append(fresh)
        if fresh:
            raise JobsError("stale host proof")
        return value

    monkeypatch.setattr(launch_direct, "_duplicate", duplicate)
    with pytest.raises(JobsError, match="stale host proof"):
        operator_job.launch_packet(
            identity=identity,
            plan=plan,
            request=request,
            preflight_launch_result=preflight,
            source_preview={"manifest_yaml": "{}"},
            manifest_sha256="sha256:" + "1" * 64,
            dev_preview=preview,
            dev_preview_provenance=_dev_preview_provenance(preview),
            duplicate_proof=packet["duplicate_proof"],
            capacity_census=capacity,
        )
    assert operator_job._validate_packet_semantics(packet) == packet
    assert freshness == [True, False]


def test_prod10_launch_recovery_observer_binds_existing_job_at_construction(
    tmp_path: Path,
) -> None:
    uid = "00000000-0000-4000-8000-000000000021"
    proof = {
        "phase": "launch",
        "name": "chris-q38-prod10-launch-operator-v3",
        "failure_alerts": "off",
        "priority": "c1",
        "queue_priority": "q1",
        "gpus": 0,
        "packet_sha256": "sha256:" + "1" * 64,
        "job_manifest_sha256": "sha256:" + "2" * 64,
    }

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert command[5:8] == ["get", "job", proof["name"]]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "metadata": {
                        "uid": uid,
                        "creationTimestamp": "2026-09-23T10:04:17Z",
                    }
                }
            ),
            stderr="",
        )

    observer = operator_launch._Prod10LaunchRecoveryObserver(
        proof,
        tmp_path,
        expected_uid=uid,
        run=runner,
    )

    assert observer.kind == "job"
    assert observer.profile == "production-recovery"
    assert observer.expected_uid == uid
    assert observer.requires_creator_binding is False
    assert observer.armed_schema == cleanup.RECOVERY_ARMED_SCHEMA
    assert observer.result_schema == cleanup.RECOVERY_RESULT_SCHEMA
    assert observer.armed_path == tmp_path / "OPERATOR_RECOVERY_OBSERVER_ARMED.json"
    assert observer.result_path == tmp_path / "OPERATOR_RECOVERY_OBSERVER_RESULT.json"
    assert observer.snapshot.uid == ""
    armed = observer.arm()
    assert armed["schema"] == cleanup.RECOVERY_ARMED_SCHEMA
    assert armed["recovered_existing_target"] is True
    assert armed["expected_uid"] == uid
    assert armed["target_created_at"] == "2026-09-23T10:04:17Z"
    with pytest.raises(cleanup.ObserverError, match="only a recovery observer"):
        cleanup.Observer(
            context=direct.PROD_CONTEXT,
            namespace=direct.NAMESPACE,
            kind="job",
            name=proof["name"],
            maximum_seconds=2500,
            expected_gpus=0,
            plan_sha256=proof["packet_sha256"],
            manifest_sha256=proof["job_manifest_sha256"],
            armed_path=tmp_path / "generic-armed.json",
            result_path=tmp_path / "generic-result.json",
            profile="production-operator",
            expected_uid=uid,
        )


def test_prod10_launch_recovery_observer_rejects_drift_before_arming(
    tmp_path: Path,
) -> None:
    proof = {
        "phase": "launch",
        "name": "chris-q38-prod10-launch-operator-v3",
        "failure_alerts": "off",
        "priority": "c1",
        "queue_priority": "q1",
        "gpus": 0,
        "packet_sha256": "sha256:" + "1" * 64,
        "job_manifest_sha256": "sha256:" + "2" * 64,
    }
    (tmp_path / "OPERATOR_RECOVERY_OBSERVER_RESULT.json").write_text("existing")

    with pytest.raises(cleanup.ObserverError, match="binding"):
        operator_launch._Prod10LaunchRecoveryObserver(
            proof,
            tmp_path,
            expected_uid="00000000-0000-4000-8000-000000000021",
        )
    with pytest.raises(cleanup.ObserverError, match="exact UID"):
        operator_launch._Prod10LaunchRecoveryObserver(
            proof,
            tmp_path / "unused",
            expected_uid="not-a-uid",
        )


def test_prod10_inspector_is_read_only_alert_off_c1_q1_zero_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = historical.load_identity(IDENTITY)
    plan = {"schema": training.SCHEMA}
    launch = direct._seal({"schema": direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA})
    monkeypatch.setattr(direct, "_identity", lambda _plan, bound: bound)
    monkeypatch.setattr(launch_direct, "_preflight_launch", lambda value, *_args, **_kw: value)

    packet = operator_job.inspect_packet(
        identity=identity,
        plan=plan,
        preflight_launch_result=launch,
    )
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    pod = package.job["spec"]["template"]["spec"]
    [container] = pod["containers"]
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    volumes = {item["name"]: item for item in pod["volumes"]}

    assert proof == {
        **proof,
        "phase": "inspect",
        "name": "chris-q38-prod10-launch-inspect-v6",
        "failure_alerts": "off",
        "priority": "c1",
        "queue_priority": "q1",
        "gpus": 0,
    }
    assert package.job["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] == "off"
    assert package.job["spec"]["activeDeadlineSeconds"] == 2400
    assert package.job["spec"]["backoffLimit"] == 0
    assert mounts["sfs"]["readOnly"] is True
    assert volumes["sfs"]["persistentVolumeClaim"]["readOnly"] is True
    assert "controls-rw" not in mounts
    assert "controls-rw" not in volumes
    assert "envFrom" not in container
    assert "secretRef" not in json.dumps(package.job, sort_keys=True)
    assert "nvidia.com/gpu" not in json.dumps(package.job, sort_keys=True)
    assert packet["launch_v5_failure"] == operator.launch_v5_failure_binding()
    assert packet["launch_v5_failure"]["config_maps_absent"] is True
    assert packet["launch_v5_failure"]["config_map_cleanup_result_sha256"] == (
        "sha256:0becb6f7dd07640ce5f3a8e24ed2bcb3e967dc1539683b65693e716bcb96e5e2"
    )

    changed = copy.deepcopy(packet)
    changed["launch_v5_failure"]["operator_job_uid"] = "00000000-0000-4000-8000-000000000001"
    changed["launch_v5_failure"] = operator._seal(changed["launch_v5_failure"])
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="predecessor"):
        operator_job.build_operator_package(changed)


def test_prod10_inspector_reads_only_allowlisted_path_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_root = tmp_path / "operation"
    operation_root.mkdir()
    archive = operation_root / operator._GUARD_ARCHIVE_NAME
    archive.touch()
    archive.chmod(0o600)
    archive_receipt = operation_root / operator._GUARD_ARCHIVE_RECEIPT_NAME
    archive_receipt.touch()
    archive_receipt.chmod(0o600)
    current_guard = operation_root / "TRAINING_JOBS_API_PREFIX_GUARD.json"
    current_guard.touch()
    current_guard.chmod(0o600)
    plan = {
        "schema": training.SCHEMA,
    }
    fake_identity = SimpleNamespace(output_root=str(tmp_path / "unused-output"))
    checked_launch = operator._seal(
        {
            "schema": direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA,
            "observer": {
                "receipt": {
                    "result_path": str(tmp_path / "unused-preflight.json"),
                    "result_sha256": "sha256:" + "1" * 64,
                }
            },
        }
    )
    monkeypatch.setattr(operator, "_identity", lambda _value: fake_identity)
    monkeypatch.setattr(direct, "_identity", lambda _plan, bound: bound)
    monkeypatch.setattr(operator.hardening, "training_operation_root", lambda _plan: operation_root)
    monkeypatch.setattr(
        launch_direct,
        "_preflight_launch",
        lambda *_args, **_kwargs: checked_launch,
    )
    monkeypatch.setattr(
        operator,
        "Jobs",
        lambda *_args, **_kwargs: pytest.fail("inspector contacted the Jobs API"),
    )
    monkeypatch.setattr(
        direct,
        "server_dry_run",
        lambda *_args, **_kwargs: pytest.fail("inspector contacted Kubernetes"),
    )
    monkeypatch.setattr(
        launch_direct,
        "jit_duplicate_proof",
        lambda *_args, **_kwargs: pytest.fail("inspector ran a duplicate scan"),
    )
    monkeypatch.setattr(
        operator,
        "_fresh_capacity_census",
        lambda *_args, **_kwargs: pytest.fail("inspector ran a capacity census"),
    )

    result = operator.run_inspect(
        {
            "identity": {},
            "plan": plan,
            "preflight_launch_result": checked_launch,
            "launch_v5_failure": operator.launch_v5_failure_binding(),
        }
    )

    assert result["schema"] == operator.MANIFEST_RESULT_SCHEMA
    assert result["status"] == "passed"
    assert result["phase"] == "inspect"
    assert result["contents_exported"] is False
    assert result["nested_jobs_created"] == result["gpus"] == 0
    assert result["paths"]["v3_guard_archive"]["state"] == "present"
    assert result["paths"]["v3_guard_archive_receipt"]["state"] == "present"
    assert result["paths"]["current_guard"]["state"] == "present"
    assert result["paths"]["create_journal"] == {"state": "absent"}
    assert result["paths"]["creator_binding"] == {"state": "absent"}
    assert set(result["paths"]) == {
        "v3_guard_archive",
        "v3_guard_archive_receipt",
        "current_guard",
        "create_journal",
        "creator_binding",
    }
    assert result["launch_v5_failure_sha256"] == operator.launch_v5_failure_binding()["sha256"]
    assert result["launch_packet_sha256"] == (operator.launch_v5_failure_binding()["packet_sha256"])
    assert result["preflight_launch_sha256"] == checked_launch["sha256"]
    assert result["launch_boundary"] == "after_new_guard_before_intent"
    assert result["inspection_external_calls"] == 0
    assert result["runtime_binding_kubernetes_reads"] == 2
    assert result["archive_attempted"] is False
    assert result["guard_constructed"] is False
    assert result["create_intent_written"] is False
    assert result["provider_create_called"] is False
    for metadata in result["paths"].values():
        assert set(metadata) <= {"state", "kind", "mode", "mtime_ns", "sha256"}
        assert not {"uid", "gid", "readable", "traversable", "path", "content"} & set(metadata)
        if metadata["state"] == "present":
            assert metadata["kind"] == "regular"
            assert metadata["sha256"] == (
                "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            )
            assert type(metadata["mtime_ns"]) is int
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    assert len(encoded.encode()) < 3900
    assert str(tmp_path) not in encoded

    termination = tmp_path / "termination.log"
    monkeypatch.setattr(operator, "_TERMINATION_PATH", termination)
    operator._write_manifest_termination(result)
    message = termination.read_text()
    assert cleanup._validated_receipt(message, kind="job") == result
    assert cleanup._receipt_execution_accepted(result) is True


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (
            ("absent", "absent", "present", "absent", "absent"),
            "before_v3_guard_archive_or_archive_write",
        ),
        (
            ("present", "absent", "absent", "absent", "absent"),
            "v3_guard_archived_receipt_missing",
        ),
        (
            ("present", "present", "absent", "absent", "absent"),
            "after_v3_guard_archive_before_new_guard",
        ),
        (
            ("present", "present", "present", "absent", "absent"),
            "after_new_guard_before_intent",
        ),
        (
            ("present", "present", "present", "present", "absent"),
            "intent_crossed_never_retry",
        ),
        (
            ("present", "present", "present", "present", "present"),
            "binding_present",
        ),
        (
            ("absent", "present", "absent", "absent", "absent"),
            "indeterminate_or_inconsistent",
        ),
        (
            ("lstat_error", "absent", "present", "absent", "absent"),
            "indeterminate_or_inconsistent",
        ),
    ],
)
def test_prod10_inspection_boundary_is_fixed_and_fail_closed(
    states: tuple[str, str, str, str, str], expected: str
) -> None:
    present = {
        "state": "present",
        "kind": "regular",
        "mode": "0600",
        "mtime_ns": 1,
        "sha256": "sha256:" + "1" * 64,
    }
    probes = {
        name: dict(present) if state == "present" else {"state": state}
        for name, state in zip(
            (
                "v3_guard_archive",
                "v3_guard_archive_receipt",
                "current_guard",
                "create_journal",
                "creator_binding",
            ),
            states,
            strict=True,
        )
    }
    assert operator._inspection_boundary(probes) == expected


@pytest.mark.parametrize(
    "malformed",
    [
        {
            "state": "present",
            "kind": "symlink",
            "mode": "0777",
            "mtime_ns": 1,
            "sha256": None,
        },
        {
            "state": "present",
            "kind": "directory",
            "mode": "0700",
            "mtime_ns": 1,
            "sha256": None,
        },
        {
            "state": "present",
            "kind": "regular",
            "mode": "0600",
            "mtime_ns": 1,
            "sha256": None,
        },
    ],
)
def test_prod10_inspection_boundary_rejects_noncanonical_present_marker(
    malformed: dict[str, object],
) -> None:
    present = {
        "state": "present",
        "kind": "regular",
        "mode": "0600",
        "mtime_ns": 1,
        "sha256": "sha256:" + "1" * 64,
    }
    probes = {
        "v3_guard_archive": dict(present),
        "v3_guard_archive_receipt": dict(present),
        "current_guard": malformed,
        "create_journal": {"state": "absent"},
        "creator_binding": {"state": "absent"},
    }

    assert operator._inspection_boundary(probes) == "indeterminate_or_inconsistent"


def test_prod10_inspection_digest_never_follows_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "private-target"
    target.write_text("private contents must not be exported")
    link = tmp_path / "marker"
    link.symlink_to(target)

    observed = operator._inspect_path(link)

    assert observed == {
        "state": "present",
        "kind": "symlink",
        "mode": f"{stat.S_IMODE(link.lstat().st_mode):04o}",
        "mtime_ns": link.lstat().st_mtime_ns,
        "sha256": None,
    }
    assert "private" not in json.dumps(observed, sort_keys=True)


_PRE_GUARD_STAGES = (
    "plan_identity",
    "request_identity",
    "preflight_launch_binding",
    "preflight_result_read",
    "preflight_result_digest",
    "sealed_preflight_validate",
    "fresh_preflight_revalidate",
    "image_identity",
    "source_preview_shape",
    "manifest_rebuild",
    "manifest_digest",
    "operation_root_validate",
    "output_absence",
    "server_dry_run",
    "server_preview_validate",
)


@pytest.mark.parametrize("stage", _PRE_GUARD_STAGES)
@pytest.mark.parametrize(
    ("failure", "expected_class", "expected_code"),
    [
        (OSError("private path and errno must not escape"), "OSError", "oserror"),
        (AssertionError("private assertion must not escape"), "AssertionError", "assertionerror"),
        (JobsError("private Jobs response must not escape"), "JobsError", "jobserror"),
        (
            incluster_kubernetes.InClusterKubernetesError(
                "private Kubernetes response must not escape"
            ),
            "InClusterKubernetesError",
            "inclusterkuberneteserror",
        ),
    ],
)
def test_prod10_phase_probe_is_sanitized_read_only_and_zero_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    failure: Exception,
    expected_class: str,
    expected_code: str,
) -> None:
    identity = historical.load_identity(IDENTITY)
    plan = {"schema": training.SCHEMA}
    request = {"workers": 1, "gpus_per_worker": 8}
    launch = direct._seal({"schema": direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA, "gpus": 0})
    preview = direct._seal(
        {
            "schema": direct.PREVIEW_SCHEMA,
            "context": direct.DEV_CONTEXT,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    duplicate = direct._seal(
        {"schema": launch_direct.DUPLICATE_SCHEMA, "status": "identities_absent"}
    )
    capacity = {
        "schema": "cyber_project_gpu_capacity_census_v1",
        "limits": {"nodes": 10, "gpus": 80},
        "planned": {"nodes": 1, "gpus": 8},
    }
    capacity["sha256"] = digest(capacity)
    monkeypatch.setattr(direct, "_identity", lambda _plan, bound: bound)
    monkeypatch.setattr(training, "job_request", lambda _plan: request)
    monkeypatch.setattr(launch_direct, "_preflight_launch", lambda value, *_args, **_kw: value)
    monkeypatch.setattr(direct, "_source", lambda value: value)
    monkeypatch.setattr(launch_direct, "_duplicate", lambda value, _identity, **_kwargs: value)
    source = operator_job.launch_packet(
        identity=identity,
        plan=plan,
        request=request,
        preflight_launch_result=launch,
        source_preview={"manifest_yaml": "{}"},
        manifest_sha256="sha256:" + "1" * 64,
        dev_preview=preview,
        dev_preview_provenance=_dev_preview_provenance(preview),
        duplicate_proof=duplicate,
        capacity_census=capacity,
    )
    source = copy.deepcopy(source)
    source["operator_name"] = operator._LAUNCH_V2_FAILURE["operator_name"]
    for key in (
        "launch_v2_failure",
        "probe_v7_failure",
        "probe_v8_failure",
        "probe_v9_success",
    ):
        source.pop(key)
    source = operator._seal(source)
    monkeypatch.setitem(operator._LAUNCH_V2_FAILURE, "launch_packet_sha256", source["sha256"])
    packet = operator_job.probe_packet(launch_packet=source)
    package = operator_job.build_operator_package(packet)
    proof = operator_job.validate_operator_package(package)
    pod = package.job["spec"]["template"]["spec"]
    [container] = pod["containers"]
    mounts = {item["name"]: item for item in container["volumeMounts"]}
    volumes = {item["name"]: item for item in pod["volumes"]}

    assert proof["name"] == "chris-q38-prod10-launch-probe-v9"
    assert proof["phase"] == "probe"
    assert proof["failure_alerts"] == "off"
    assert proof["priority"] == "c1" and proof["queue_priority"] == "q1"
    assert proof["gpus"] == 0
    assert mounts["sfs"]["readOnly"] is True
    assert volumes["sfs"]["persistentVolumeClaim"]["readOnly"] is True
    assert mounts["controls-rw"] == {
        "name": "controls-rw",
        "mountPath": operator_job.CONTROLS_PATH,
        "subPath": operator_job.CONTROLS_SUBPATH,
    }
    assert volumes["controls-rw"] == {
        "name": "controls-rw",
        "persistentVolumeClaim": {"claimName": operator_job.PVC},
    }
    assert "envFrom" not in container
    assert "secretRef" not in json.dumps(package.job, sort_keys=True)
    assert "nvidia.com/gpu" not in json.dumps(package.job, sort_keys=True)
    environment = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    assert environment["WANDB_MODE"] == "disabled"
    assert environment["WANDB_API_KEY"] == "diagnostic-not-a-credential"
    assert environment["HF_DATASETS_CACHE"] == "/work/hf-datasets"
    assert {
        "HOME",
        "TMPDIR",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    }.isdisjoint(environment)
    assert len(json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()) < 1024 * 1024

    operation_root = tmp_path / "operation"
    operation_root.mkdir()
    monkeypatch.setattr(hardening, "training_operation_root", lambda _plan: operation_root)

    def fail_pre_guard(_packet: dict, _runner: object) -> dict:
        operator._LAUNCH_STAGE = stage
        training._PREFLIGHT_STAGE = "native_config_validate"
        raise failure

    monkeypatch.setattr(operator, "_pre_guard_launch", fail_pre_guard)
    monkeypatch.setattr(
        cleanup.JobsApiPrefixGuard,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("guard must not be constructed"),
    )
    monkeypatch.setattr(
        operator,
        "_fresh_capacity_census",
        lambda *_args, **_kwargs: pytest.fail("capacity must not be read"),
    )
    monkeypatch.setattr(
        launch_direct,
        "create_once",
        lambda *_args, **_kwargs: pytest.fail("Jobs API must not be called"),
    )
    result = operator.run_probe(packet, runner=object())
    assert result["status"] == "passed"
    assert result["diagnosis"] == "exception_localized"
    assert result["error_class"] == expected_class
    assert result["error_code"] == f"launch_{stage}_{expected_code}"
    assert result["launch_stage"] == stage
    assert result["preflight_stage"] == "native_config_validate"
    assert result["error_path_exported"] is False
    assert result["error_errno_exported"] is False
    assert result["error_message_exported"] is False
    assert result["expected_diagnosis"] == "before_guard_passed"
    assert result["probe_v7_failure_sha256"] == operator.probe_v7_failure_binding()["sha256"]
    assert result["probe_v8_failure_sha256"] == operator.probe_v8_failure_binding()["sha256"]
    assert result["nested_jobs_created"] == result["gpus"] == 0
    encoded = json.dumps(result, sort_keys=True)
    assert "private" not in encoded
    assert len(encoded.encode()) < 3900

    changed = copy.deepcopy(packet)
    changed["inspect_v3_success"]["operator_job_uid"] = "00000000-0000-4000-8000-000000000001"
    changed["inspect_v3_success"] = operator._seal(changed["inspect_v3_success"])
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="inspection predecessor"):
        operator_job.build_operator_package(changed)

    changed = copy.deepcopy(packet)
    changed["probe_v7_failure"]["operator_job_uid"] = "00000000-0000-4000-8000-000000000001"
    changed["probe_v7_failure"] = operator._seal(changed["probe_v7_failure"])
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="v7 predecessor"):
        operator_job.build_operator_package(changed)

    changed = copy.deepcopy(packet)
    changed["probe_v8_failure"]["operator_job_uid"] = "00000000-0000-4000-8000-000000000001"
    changed["probe_v8_failure"] = operator._seal(changed["probe_v8_failure"])
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="v8 predecessor"):
        operator_job.build_operator_package(changed)

    changed = copy.deepcopy(packet)
    changed["expected_diagnosis"] = "exception_localized"
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="expected diagnosis"):
        operator_job.build_operator_package(changed)

    changed = copy.deepcopy(packet)
    changed["writable_controls_probe"] = False
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="writable-controls"):
        operator_job.build_operator_package(changed)

    changed = copy.deepcopy(packet)
    changed["launch_packet"]["unexpected"] = True
    changed["launch_packet"] = operator._seal(changed["launch_packet"])
    changed = operator._seal(changed)
    with pytest.raises(ValueError, match="packet predecessor"):
        operator_job.build_operator_package(changed)


def test_prod10_launch_uses_shared_pre_guard_path_before_any_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("shared pre-guard sentinel")

    def fail_shared(_packet: dict, _runner: object) -> dict:
        raise failure

    monkeypatch.setattr(operator, "_pre_guard_launch", fail_shared)
    monkeypatch.setattr(
        cleanup.JobsApiPrefixGuard,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("guard constructed before shared path"),
    )
    with pytest.raises(RuntimeError, match="shared pre-guard sentinel"):
        operator.run_launch({}, runner=object())


def test_prod10_jit_duplicate_uses_fresh_host_dev_proof_and_only_live_prod_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = historical.load_identity(IDENTITY)
    host = direct._seal(
        {
            "schema": launch_direct.DUPLICATE_SCHEMA,
            "status": "identity_and_output_absent",
            "identity_sha256": identity.sealed_mapping()["sha256"],
            "run_name": identity.run_name,
            "output_root": identity.output_root,
            "kubernetes_inventories_checked": 10,
            "jobs_api_rows_checked": 0,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    items: list[object] = []
    contexts: list[str] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        context = command[command.index("--context") + 1]
        contexts.append(context)
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"items": items}), stderr=""
        )

    bases: list[str] = []
    rows: list[dict] = []

    class FakeJobs:
        def __init__(self, _token: str, *, base_url: str):
            bases.append(base_url)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def all_runs(self):
            return rows

    first = launch_direct.jit_duplicate_proof(
        identity, host, token="token", runner=runner, jobs_factory=FakeJobs
    )
    second = launch_direct.jit_duplicate_proof(
        identity,
        host,
        token="token",
        prior_duplicate=first,
        runner=runner,
        jobs_factory=FakeJobs,
    )
    assert contexts == [direct.PROD_CONTEXT] * 10
    assert bases == [launch_direct.API_URLS["prod"]] * 2
    assert second["prior_jit_duplicate_sha256"] == first["sha256"]
    assert second["runtime_prod_kubernetes_inventories_checked"] == 5
    assert second["runtime_jobs_api_targets_checked"] == ["prod"]
    assert second["fresh_host_all_context_duplicate_sha256"] == host["sha256"]
    assert "jobs_api_targets_checked" not in second
    assert "host_duplicate_sha256" not in second

    items.append({"metadata": {"name": identity.run_name}, "spec": {}})
    with pytest.raises(JobsError, match="Kubernetes identity/output"):
        launch_direct.jit_duplicate_proof(
            identity, host, token="token", runner=runner, jobs_factory=FakeJobs
        )
    items[:] = [None]
    with pytest.raises(JobsError, match="inventory is invalid"):
        launch_direct.jit_duplicate_proof(
            identity, host, token="token", runner=runner, jobs_factory=FakeJobs
        )
    items.clear()
    rows.append({"name": identity.run_name, "run_dir": "other"})
    with pytest.raises(JobsError, match="Jobs API history"):
        launch_direct.jit_duplicate_proof(
            identity, host, token="token", runner=runner, jobs_factory=FakeJobs
        )
    rows.clear()
    monkeypatch.setattr(
        launch_direct,
        "Path",
        lambda _value: SimpleNamespace(exists=lambda: True, is_symlink=lambda: False),
    )
    with pytest.raises(JobsError, match="output root already exists"):
        launch_direct.jit_duplicate_proof(
            identity, host, token="token", runner=runner, jobs_factory=FakeJobs
        )

    stale = direct._seal(
        {
            **{key: value for key, value in host.items() if key != "sha256"},
            "checked_at": (datetime.now(UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    with pytest.raises(JobsError, match="stale or future-dated"):
        launch_direct.jit_duplicate_proof(
            identity, stale, token="token", runner=runner, jobs_factory=FakeJobs
        )


def test_prod10_dead_guard_is_atomically_archived_and_crash_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = historical.load_identity(IDENTITY)
    operation_root = tmp_path / "operation"
    operation_root.mkdir()
    plan = {"schema": training.SCHEMA}
    request = {
        "name": identity.run_name,
        "run_dir": identity.output_root,
        "image": "image@sha256:" + "1" * 64,
        "workers": 1,
        "gpus_per_worker": 8,
    }
    expected = {"kind": "RayJob"}
    packet = {"duplicate_proof": {"sha256": "sha256:" + "2" * 64}}
    first = {"sha256": "sha256:" + "3" * 64}
    final = {"sha256": "sha256:" + "4" * 64}
    guard_path = direct.jobs_api_guard_path(operation_root, "training")
    archive_path = operation_root / operator._GUARD_ARCHIVE_NAME
    receipt_path = operation_root / operator._GUARD_ARCHIVE_RECEIPT_NAME
    stored = operator._seal({"schema": cleanup.JOBS_API_PREFIX_GUARD_SCHEMA})
    operator._write_once(guard_path, stored)
    source_inode = guard_path.stat().st_ino
    guard_file_sha256 = "sha256:" + hashlib.sha256(guard_path.read_bytes()).hexdigest()
    monkeypatch.setattr(operator, "RUNTIME_UID", os.getuid())
    monkeypatch.setattr(operator, "RUNTIME_GID", os.getgid())
    monkeypatch.setattr(
        launch_direct,
        "_jit_duplicate",
        lambda value, *_args, **_kwargs: value,
    )
    monkeypatch.setattr(
        cleanup.JobsApiPrefixGuard,
        "_validate_armed",
        lambda _self, value: value,
    )
    monkeypatch.setattr(
        operator,
        "inspect_v6_success_binding",
        lambda: {
            "sha256": "sha256:" + "5" * 64,
            "current_guard_sha256": "sha256:" + "0" * 64,
        },
    )
    with pytest.raises(operator.OperatorFailure, match="archive_file"):
        operator._archive_launch_v3_guard(
            packet,
            identity=identity,
            plan=plan,
            request=request,
            expected=expected,
            operation_root=operation_root,
            jit_before_guard=first,
            jit_before_intent=final,
            runner=object(),
        )
    inspection_v5 = {
        "sha256": "sha256:" + "5" * 64,
        "current_guard_sha256": guard_file_sha256,
    }
    monkeypatch.setattr(operator, "inspect_v5_success_binding", lambda: inspection_v5)
    inspection_v6 = {
        "sha256": "sha256:" + "6" * 64,
        "current_guard_sha256": guard_file_sha256,
    }
    monkeypatch.setattr(operator, "inspect_v6_success_binding", lambda: inspection_v6)

    journal = operation_root / "PROD10_DIRECT_V3_CREATE.jsonl"
    journal.write_text("intent")
    with pytest.raises(operator.OperatorFailure, match="archive_state"):
        operator._archive_launch_v3_guard(
            packet,
            identity=identity,
            plan=plan,
            request=request,
            expected=expected,
            operation_root=operation_root,
            jit_before_guard=first,
            jit_before_intent=final,
            runner=object(),
        )
    journal.unlink()

    receipt = operator._archive_launch_v3_guard(
        packet,
        identity=identity,
        plan=plan,
        request=request,
        expected=expected,
        operation_root=operation_root,
        jit_before_guard=first,
        jit_before_intent=final,
        runner=object(),
    )
    assert not guard_path.exists()
    assert archive_path.stat().st_ino == source_inode
    assert json.loads(archive_path.read_bytes()) == stored
    assert receipt["archived_via_atomic_rename"] is True
    assert receipt["recovered_after_atomic_rename"] is False
    assert receipt["inspect_v5_success_sha256"] == inspection_v5["sha256"]
    assert receipt["inspect_v6_success_sha256"] == inspection_v6["sha256"]
    assert receipt["guard_file_sha256"] == guard_file_sha256
    assert receipt["launch_v4_failure_sha256"] == operator.launch_v4_failure_binding()["sha256"]
    assert receipt["launch_v5_failure_sha256"] == operator.launch_v5_failure_binding()["sha256"]
    assert receipt["launch_v6_failure_sha256"] == operator.launch_v6_failure_binding()["sha256"]
    assert receipt["launch_v7_failure_sha256"] == operator.launch_v7_failure_binding()["sha256"]

    receipt_path.unlink()
    recovered = operator._archive_launch_v3_guard(
        packet,
        identity=identity,
        plan=plan,
        request=request,
        expected=expected,
        operation_root=operation_root,
        jit_before_guard=first,
        jit_before_intent=final,
        runner=object(),
    )
    assert recovered["recovered_after_atomic_rename"] is True
    assert archive_path.stat().st_ino == source_inode
    with pytest.raises(operator.OperatorFailure, match="archive_state"):
        operator._archive_launch_v3_guard(
            packet,
            identity=identity,
            plan=plan,
            request=request,
            expected=expected,
            operation_root=operation_root,
            jit_before_guard=first,
            jit_before_intent=final,
            runner=object(),
        )


def test_prod10_launch_orders_all_reads_before_new_guard_and_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = historical.load_identity(IDENTITY)
    plan = {
        "schema": training.SCHEMA,
        "arguments": {
            "wandb_entity": "thefleet",
            "wandb_project": "cyber-post-train",
            "wandb_run_id": identity.wandb_run_id,
        },
    }
    request = {
        "name": identity.run_name,
        "run_dir": identity.output_root,
        "image": "image@sha256:" + "1" * 64,
        "workers": 1,
        "gpus_per_worker": 8,
    }
    source, expected = {"name": identity.run_name + "-00000000"}, {"kind": "RayJob"}
    preview = {"checked_at": "2026-09-23T10:00:00Z", "sha256": "sha256:" + "2" * 64}
    events: list[str] = []
    monkeypatch.setenv("FLEET_API_KEY", "token")
    monkeypatch.setenv("WANDB_API_KEY", "token")
    monkeypatch.setattr(
        operator,
        "_pre_guard_launch",
        lambda *_args: {
            "identity": identity,
            "plan": plan,
            "request": request,
            "preflight": {"sha256": "sha256:" + "a" * 64},
            "revalidation": {"sha256": "sha256:" + "b" * 64},
            "image_identity": {},
            "source_preview": source,
            "expected": expected,
            "operation_root": Path("/operation"),
        },
    )
    monkeypatch.setattr(direct, "server_dry_run", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(direct, "validate_preview", lambda *_args, **_kwargs: preview)
    monkeypatch.setattr(
        launch_direct,
        "_sealed_dev_preview_provenance",
        lambda *_args, **_kwargs: events.append("dev_provenance") or preview,
    )
    jit = iter(
        [
            {"sha256": "sha256:" + "3" * 64},
            {"sha256": "sha256:" + "4" * 64},
        ]
    )
    monkeypatch.setattr(
        launch_direct,
        "jit_duplicate_proof",
        lambda *_args, **_kwargs: events.append("jit") or next(jit),
    )
    monkeypatch.setattr(
        direct,
        "_wandb_exists_default",
        lambda *_args: pytest.fail("launch performed the non-authoritative W&B lookup"),
    )

    class FakeJobs:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def preview(self, _request):
            events.append("live_preview")
            return source

    monkeypatch.setattr(operator, "Jobs", FakeJobs)
    monkeypatch.setattr(direct, "manifest", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(
        operator,
        "_fresh_capacity_census",
        lambda *_args: events.append("capacity") or {},
    )
    monkeypatch.setattr(launch_direct, "capacity_gate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(direct, "_fresh_at", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        operator,
        "_archive_launch_v3_guard",
        lambda *_args, **_kwargs: events.append("archive") or {"sha256": "sha256:" + "5" * 64},
    )

    class FakeGuard:
        def __init__(self, **_kwargs):
            events.append("guard_construct")

        def arm(self):
            events.append("guard_arm")
            return {}

    monkeypatch.setattr(cleanup, "JobsApiPrefixGuard", FakeGuard)
    monkeypatch.setattr(
        launch_direct,
        "authorize",
        lambda *_args, **_kwargs: events.append("authorize") or {"sha256": "sha256:" + "6" * 64},
    )

    def create_once(*_args, **kwargs):
        assert "wandb_absent" not in kwargs
        events.append("create_once")
        return {"capacity_gate_sha256": "sha256:" + "7" * 64, "gpus": 8}

    monkeypatch.setattr(launch_direct, "create_once", create_once)
    monkeypatch.setattr(
        operator,
        "_observe_created_run",
        lambda *_args, **_kwargs: events.append("observe") or {},
    )
    packet = {
        "sha256": "sha256:" + "8" * 64,
        "dev_preview": {},
        "sealed_dev_preview_provenance": {},
        "duplicate_proof": {},
        "capacity_census": {"sha256": "sha256:" + "9" * 64},
    }
    result = operator.run_launch(packet, runner=object())
    assert result["status"] == "gpu_run_succeeded_and_released"
    assert events == [
        "dev_provenance",
        "jit",
        "live_preview",
        "capacity",
        "jit",
        "archive",
        "guard_construct",
        "guard_arm",
        "authorize",
        "create_once",
        "observe",
    ]


def test_prod10_probe_rechecks_markers_and_stops_before_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operation_root = tmp_path / "operation"
    operation_root.mkdir()
    launch_packet = {
        "plan": {"schema": training.SCHEMA},
        "sha256": "sha256:" + "1" * 64,
    }
    packet = {"launch_packet": launch_packet}
    checks: list[Path] = []
    monkeypatch.setattr(hardening, "training_operation_root", lambda _plan: operation_root)
    monkeypatch.setattr(
        operator,
        "_assert_probe_markers_absent",
        lambda root: checks.append(root),
    )

    def pass_shared(_packet: dict, _runner: object) -> dict:
        operator._LAUNCH_STAGE = "before_guard_passed"
        return {"operation_root": operation_root}

    monkeypatch.setattr(operator, "_pre_guard_launch", pass_shared)
    monkeypatch.setattr(
        cleanup.JobsApiPrefixGuard,
        "__init__",
        lambda *_args, **_kwargs: pytest.fail("probe constructed a guard"),
    )
    result = operator.run_probe(packet, runner=object())
    assert checks == [operation_root, operation_root]
    assert result["diagnosis"] == result["launch_stage"] == "before_guard_passed"
    assert result["expected_diagnosis"] == "before_guard_passed"
    assert result["probe_v7_failure_sha256"] == operator.probe_v7_failure_binding()["sha256"]
    assert result["probe_v8_failure_sha256"] == operator.probe_v8_failure_binding()["sha256"]
    assert result["nested_jobs_created"] == result["gpus"] == 0


def test_prod10_direct_v3_rejects_unsealed_preflight_and_posts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = historical.load_identity(IDENTITY)
    monkeypatch.setattr(direct, "_identity", lambda _plan, bound: bound)
    with pytest.raises(JobsError, match="digest"):
        launch_direct.preflight_result(
            {"schema": training.SCHEMA},
            {},
            {"schema": launch_direct.PREFLIGHT_RESULT_SCHEMA},
            identity=identity,
        )

    root = tmp_path / "operation"
    root.mkdir()
    plan = {
        "schema": training.SCHEMA,
        "arguments": {
            "wandb_entity": "thefleet",
            "wandb_project": "cyber-post-train",
            "wandb_run_id": identity.wandb_run_id,
        },
    }
    request = {
        "name": identity.run_name,
        "run_dir": identity.output_root,
        "workers": 1,
        "gpus_per_worker": 8,
    }
    source, expected = {"source": "preview"}, {"kind": "RayJob"}
    auth = direct._seal(
        {
            "schema": launch_direct.AUTHORIZATION_SCHEMA,
            "operation_root": str(root),
            "preflight_result": {
                "receipt": {
                    "wandb_create_once": {
                        "entity": "thefleet",
                        "project": "cyber-post-train",
                        "run_id": identity.wandb_run_id,
                        "resume": "never",
                    }
                }
            },
            "preflight_revalidation": {},
            "dev_preview": {},
            "sealed_dev_preview_provenance": {
                "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            },
            "prod_preview": {},
            "observer": {},
            "image_identity_receipt": {},
        }
    )
    monkeypatch.setattr(operator.hardening, "training_operation_root", lambda _plan: root)
    monkeypatch.setattr(launch_direct, "authorize", lambda *_args, **_kwargs: auth)
    monkeypatch.setattr(launch_direct, "_duplicate", lambda value, _identity, **_kwargs: value)
    monkeypatch.setattr(
        launch_direct,
        "_jit_duplicate",
        lambda value, _identity, _host, **_kwargs: value,
    )
    jit_before_guard = {
        "sha256": "sha256:" + "5" * 64,
        "fresh_host_all_context_duplicate_sha256": "sha256:" + "6" * 64,
    }
    jit_before_intent = {
        "sha256": "sha256:" + "7" * 64,
        "fresh_host_all_context_duplicate_sha256": "sha256:" + "6" * 64,
    }
    monkeypatch.setattr(
        launch_direct,
        "jit_duplicate_proof",
        lambda *_args, **_kwargs: jit_before_intent,
    )
    monkeypatch.setattr(direct, "manifest", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(direct, "server_dry_run", lambda *_args, **_kwargs: expected)
    live_preview = {
        "sha256": "sha256:" + "2" * 64,
        "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    monkeypatch.setattr(direct, "validate_preview", lambda *_args, **_kwargs: live_preview)
    monkeypatch.setattr(launch_direct, "_preview_binding", lambda *_args, **_kwargs: live_preview)
    monkeypatch.setattr(
        launch_direct, "capacity_gate", lambda *_args, **_kwargs: {"sha256": "sha256:" + "3" * 64}
    )
    monkeypatch.setattr(direct, "_jobs_api_prefix_guard", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(direct, "_fresh_at", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        direct,
        "_bind_jobs_api_created",
        lambda **_kwargs: {
            "rayjob_name": identity.run_name + "-1a2b3c4d",
            "rayjob_uid": "00000000-0000-4000-8000-000000000007",
            "sha256": "sha256:" + "4" * 64,
            "rayjob_created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "failure_alerts": "off",
        },
    )
    posts: list[tuple[str, str]] = []

    class FakeJobs:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def preview(self, _request):
            return source

        def request(self, method, path, **_kwargs):
            posts.append((method, path))
            return {
                "name": identity.run_name + "-1a2b3c4d",
                "job_id": "00000000-0000-4000-8000-000000000006",
                "run_dir": identity.output_root,
            }

    def create():
        return launch_direct.create_once(
            root,
            plan,
            request,
            source,
            expected,
            auth,
            token="token",
            identity=identity,
            duplicate=jit_before_guard,
            final_duplicate=jit_before_intent,
            host_duplicate={},
            census={},
            live_preview=live_preview,
            jobs_factory=FakeJobs,
        )

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(JobsError, match="runtime create-once binding"):
        create()
    assert not (root / "PROD10_DIRECT_V3_CREATE.jsonl").exists()
    monkeypatch.setenv("WANDB_API_KEY", "credential-present")
    created = create()
    assert posts == [("POST", "/v1/runs")]
    assert created["failure_alerts"] == "off"
    journal_lines = (root / "PROD10_DIRECT_V3_CREATE.jsonl").read_text().splitlines()
    intent = json.loads(journal_lines[0])
    assert "wandb_run_id_absent" not in intent
    assert intent["wandb_runtime_create_once"] == {
        "credential_present": True,
        "enforced_by": "training.skyrl_training.ScalarTracking.wandb.init",
        "remote_lookup_performed": False,
        "wandb_create_once": {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": identity.wandb_run_id,
            "resume": "never",
        },
    }
    with pytest.raises(JobsError, match="create intent"):
        create()
    assert len(posts) == 1


def test_prod10_launch_validates_sealed_dev_preview_without_network_or_redating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = historical.load_identity(IDENTITY)
    plan = {"schema": training.SCHEMA}
    request = {"name": identity.run_name, "workers": 1, "gpus_per_worker": 8}
    source = {"name": identity.run_name + "-00000000"}
    expected = {"kind": "RayJob"}
    fresh_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def preview(context: str, checked_at: str) -> dict:
        return direct._seal(
            {
                "schema": direct.PREVIEW_SCHEMA,
                "status": "passed",
                "context": context,
                "plan_sha256": digest(plan),
                "request_sha256": digest(request),
                "manifest_sha256": digest(expected),
                "server_render_sha256": "0" * 64,
                "name": source["name"],
                "run_name_prefix": identity.run_name,
                "nodes": 1,
                "gpus": 8,
                "priority": "c1",
                "queue_priority": "q1",
                "failure_alerts": "off",
                "submitted": False,
                "checked_at": checked_at,
            }
        )

    dev = preview(direct.DEV_CONTEXT, fresh_at)
    prod = preview(direct.PROD_CONTEXT, fresh_at)

    class ForbiddenJobs:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("production operator contacted the development Jobs API")

    monkeypatch.setattr(direct, "_identity", lambda _plan, bound: bound)
    monkeypatch.setattr(direct, "manifest", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(launch_direct, "Jobs", ForbiddenJobs)
    image = {"status": "passed"}
    provenance = launch_direct.sealed_dev_preview_provenance(
        plan,
        request,
        source,
        expected,
        dev,
        image_identity_receipt=image,
        identity=identity,
    )
    assert provenance["status"] == "fresh_sealed_external_dev_server_preview_validated"
    assert provenance["sealed_dev_server_preview_sha256"] == dev["sha256"]
    assert provenance["checked_at"] == dev["checked_at"]
    assert "dev_jobs_preview_refresh" not in json.dumps(provenance, sort_keys=True)
    assert "fresh_dev_jobs_preview_passed" not in json.dumps(provenance, sort_keys=True)

    preflight = {"sha256": "sha256:" + "1" * 64}
    revalidation = {"sha256": "sha256:" + "2" * 64}
    monkeypatch.setattr(launch_direct, "preflight_result", lambda *_args, **_kwargs: preflight)
    monkeypatch.setattr(launch_direct, "_revalidation", lambda *_args, **_kwargs: revalidation)
    monkeypatch.setattr(launch_direct, "image_identity", lambda *_args, **_kwargs: image)
    monkeypatch.setattr(direct, "_jobs_api_prefix_guard", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(direct, "_canonical_operation_root", lambda _root: None)
    authorization = launch_direct.authorize(
        plan,
        request,
        source,
        expected,
        preflight,
        revalidation,
        dev_preview=dev,
        dev_provenance=provenance,
        prod_preview=prod,
        observer={},
        identity=identity,
    )
    assert authorization["dev_preview"] == dev
    assert authorization["sealed_dev_preview_provenance"] == provenance
    assert "dev_preview_refresh" not in authorization
    assert launch_direct.AUTHORIZATION_SCHEMA.endswith("_v4")
    assert launch_direct.JIT_DUPLICATE_SCHEMA.endswith("_v2")
    assert (
        launch_direct.SEALED_DEV_PREVIEW_PROVENANCE_SCHEMA
        == "cyber_skyrl_prod10_sealed_external_dev_server_preview_provenance_v1"
    )
    assert "dev_preview_refresh" not in operator._POST_PRE_GUARD_LAUNCH_STAGES
    assert "sealed_dev_preview_validate" in operator._POST_PRE_GUARD_LAUNCH_STAGES
    assert "sealed_dev_preview_freshness" not in operator._POST_PRE_GUARD_LAUNCH_STAGES

    stale = preview(
        direct.DEV_CONTEXT,
        (datetime.now(UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    with pytest.raises(JobsError, match="stale or future-dated"):
        launch_direct.sealed_dev_preview_provenance(
            plan,
            request,
            source,
            expected,
            stale,
            image_identity_receipt=image,
            identity=identity,
        )

    stale_provenance = direct._seal(
        {
            **{
                key: value
                for key, value in provenance.items()
                if key not in {"sha256", "sealed_dev_server_preview_sha256", "checked_at"}
            },
            "sealed_dev_server_preview_sha256": stale["sha256"],
            "checked_at": stale["checked_at"],
        }
    )
    monkeypatch.setattr(
        direct,
        "_fresh_at",
        lambda *_args, **_kwargs: pytest.fail(
            "runtime re-enforced sealed development preview wall-clock freshness"
        ),
    )
    assert (
        launch_direct._sealed_dev_preview_provenance(
            stale_provenance,
            plan,
            request,
            source,
            expected,
            stale,
            identity=identity,
        )
        == stale_provenance
    )

    drifted = {**dev, "manifest_sha256": "different"}
    drifted = direct._seal({key: value for key, value in drifted.items() if key != "sha256"})
    with pytest.raises(JobsError, match="server preview changed"):
        launch_direct.sealed_dev_preview_provenance(
            plan,
            request,
            source,
            expected,
            drifted,
            image_identity_receipt=image,
            identity=identity,
        )


def test_prod10_launch_reruns_preflight_and_freshly_dates_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = historical.load_identity(IDENTITY)
    raw_receipt = {"status": "passed", "gpus": 0}
    receipt = {**raw_receipt, "receipt_sha256": digest(raw_receipt)}
    checked = {"receipt": receipt, "sha256": "sha256:" + "1" * 64}
    calls: list[dict] = []
    monkeypatch.setattr(
        launch_direct,
        "preflight_result",
        lambda *_args, **_kwargs: checked,
    )
    monkeypatch.setattr(
        training,
        "preflight",
        lambda plan: calls.append(plan) or raw_receipt,
    )
    monkeypatch.setattr(
        direct,
        "_preflight_receipt",
        lambda _plan, _request, value, **_kwargs: value,
    )
    fresh: list[str] = []
    monkeypatch.setattr(direct, "_fresh_at", lambda value, **_kwargs: fresh.append(value))

    plan, request = {"schema": training.SCHEMA}, {}
    result = launch_direct.revalidate_preflight(plan, request, checked, identity=identity)
    assert calls == [plan]
    assert result["status"] == "fresh_exact_image_preflight_passed"
    assert launch_direct._revalidation(result, plan, request, checked, identity=identity) == result
    assert fresh == [result["revalidated_at"]]

    monkeypatch.setattr(training, "preflight", lambda _plan: {"status": "changed"})
    with pytest.raises(JobsError, match="fresh preflight differs"):
        launch_direct.revalidate_preflight(plan, request, checked, identity=identity)


def test_prod10_fresh_preflight_receipt_uses_cpu_writer_digest_and_rejects_tamper() -> None:
    identity = historical.load_identity(IDENTITY)
    schema = "cyber_skyrl_prod9_training_cpu_preflight_v1"
    plan = {
        "schema": training.SCHEMA,
        "run_name": identity.run_name,
        "output_root": identity.output_root,
        "arguments": {
            "steps": 1,
            "wandb_run_id": identity.wandb_run_id,
            "data_manifest": identity.data_root + "/manifest.json",
            "train_data": identity.data_root + "/train.jsonl",
            "dev_data": identity.data_root + "/dev.jsonl",
        },
        "data": {"name": identity.run_name},
    }
    request: dict = {}
    raw = {
        "schema": schema,
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": 1000, "gid": 100},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "prod9_runtime": training._binding(),
        "native_parser_checked": True,
        "ordered_multi_tool_parser_checked": True,
        "chunk_continuation_checked": True,
        "compaction_checked": True,
        "stepwise_prompt_checked": True,
        "ordered_multi_tool_execution_checked": True,
        "output_limit_gradeable_checked": True,
        "output_limit_partial_tool_blocked_checked": True,
        "fresh_recorder_checked": True,
        "tool_result_token_safe": True,
        "recorder_implementation": "training.skyrl_prod9_hardening.Recorder",
        "counts": {"train": 1, "dev": 1},
        "planned_steps": 1,
        "output_absent": True,
        "wandb_create_once": {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": identity.wandb_run_id,
            "resume": "never",
        },
    }
    with pytest.raises(JobsError, match="digest/schema"):
        direct._receipt(raw, schema)

    sealed = launch_direct._seal_fresh_preflight_receipt(raw)
    assert sealed == {**raw, "receipt_sha256": digest(raw)}
    assert direct._preflight_receipt(plan, request, sealed, identity=identity) == sealed

    changed_body = {**raw, "counts": {"train": 2, "dev": 1}}
    changed = {**changed_body, "receipt_sha256": digest(changed_body)}
    with pytest.raises(JobsError, match="receipt is incomplete"):
        direct._preflight_receipt(plan, request, changed, identity=identity)
    with pytest.raises(JobsError, match="body changed"):
        launch_direct._seal_fresh_preflight_receipt(sealed)


def test_prod10_launch_observes_only_created_uid_to_valid_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operation_root = tmp_path / "operation"
    operation_root.mkdir()
    rayjob_uid = "00000000-0000-4000-8000-000000000041"
    bound_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    binding = direct._seal(
        {
            "schema": cleanup.JOBS_API_EXACT_BINDING_SCHEMA,
            "status": "bound_exact_uid_cleanup_not_started",
            "prefix_guard_sha256": "sha256:" + "2" * 64,
            "context": direct.PROD_CONTEXT,
            "namespace": direct.NAMESPACE,
            "jobs_api_run_name": "chris-q38-prod10-12345678",
            "jobs_api_run_id": "00000000-0000-4000-8000-000000000042",
            "run_dir": "/mnt/sfs/jobs/chris-q38-prod10",
            "image": "registry/image@sha256:" + "3" * 64,
            "rayjob_name": "chris-q38-prod10-12345678",
            "rayjob_uid": rayjob_uid,
            "rayjob_created_at": bound_at,
            "bound_at": bound_at,
            "failure_alerts": "off",
            "maximum_seconds": direct.MAXIMUM_SECONDS,
            "expected_gpus": 8,
            "cleanup_started": False,
        }
    )
    binding_path = operator.hardening.creator_binding_path(operation_root, "training")
    binding_path.write_text(json.dumps(binding))
    plan = {"arguments": {"steps": 1, "eval_interval": 1}}
    receipt_body = {
        "status": "native_loop_returned",
        "plan_sha256": digest(plan),
        "checkpoint_global_step": 1,
        "completed_batches": 3,
        "completed_at": 1.0,
        "optimizer_update_independently_verified": False,
        "checkpoint_reload_verified": False,
    }
    receipt = {**receipt_body, "sha256": digest(receipt_body)}
    result = {
        "status": "released_after_terminal",
        "release_confirmed": True,
        "binding_sha256": binding["sha256"],
        "rayjob_uid": rayjob_uid,
        "peak_gpus": 8,
        "runtime_image_identity_complete": True,
        "restarts": 0,
        "exit_codes": [0, 0],
        "terminal_status": "Succeeded",
        "receipt": receipt,
    }
    captured: dict = {}

    class FakeObserver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return result

    monkeypatch.setattr(operator, "_Prod10ExactUidObserver", FakeObserver)
    created = {"rayjob_uid": rayjob_uid, "gpus": 8}
    assert (
        operator._observe_created_run(
            operation_root, plan, created, runner=lambda *_args, **_kwargs: None
        )
        == result
    )
    assert captured["binding_path"] == binding_path
    assert captured["result_path"] == operation_root / "PROD10_EXACT_OBSERVER_RESULT.json"
    contract = json.loads((operation_root / "PROD10_EXACT_RELEASE_CONTRACT.json").read_bytes())
    assert contract["rayjob_uid"] == rayjob_uid
    assert contract["binding_sha256"] == binding["sha256"]

    accepted = copy.deepcopy(result)
    rejected = []
    for change in ("exit", "empty_exit", "missing_receipt", "non_native", "wrong_step"):
        candidate = copy.deepcopy(accepted)
        if change == "exit":
            candidate["exit_codes"] = [0, 1]
        elif change == "empty_exit":
            candidate["exit_codes"] = []
        elif change == "missing_receipt":
            candidate["receipt"] = None
        else:
            candidate["receipt"][
                "status" if change == "non_native" else "checkpoint_global_step"
            ] = "changed" if change == "non_native" else 2
            body = {key: value for key, value in candidate["receipt"].items() if key != "sha256"}
            candidate["receipt"]["sha256"] = digest(body)
        result.clear()
        result.update(candidate)
        (operation_root / "PROD10_EXACT_RELEASE_CONTRACT.json").unlink()
        with pytest.raises(operator.OperatorFailure, match="exact_observer_rejected"):
            operator._observe_created_run(
                operation_root, plan, created, runner=lambda *_args, **_kwargs: None
            )
        rejected.append(change)
    assert rejected == [
        "exit",
        "empty_exit",
        "missing_receipt",
        "non_native",
        "wrong_step",
    ]


def test_prod10_exact_observer_waits_for_terminal_receipt_before_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = object.__new__(operator._Prod10ExactUidObserver)
    observer.terminal_status = "Succeeded"
    observer.receipt = None
    observer.deadline_at = datetime.now(UTC) + timedelta(seconds=1)
    observer.poll_seconds = 0.001
    observer.root_name = "chris-q38-prod10-12345678"
    observer._get = lambda _kind, _name: {"kind": "RayJob"}
    observer._validate_root = lambda _root: "cluster"
    observations: list[str] = []

    def observe(_cluster: str) -> None:
        observations.append("receipt")
        observer.receipt = {"status": "native_loop_returned"}

    observer._observe_owned_children = observe
    releases: list[str] = []
    monkeypatch.setattr(
        cleanup.JobsApiExactUidObserver,
        "_request_exact_uid_cleanup",
        lambda _self: releases.append("released"),
    )

    observer._request_exact_uid_cleanup()

    assert observations == ["receipt"]
    assert releases == ["released"]


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
    changed["preflight_v1_failure"]["operator_job_uid"] = "00000000-0000-4000-8000-000000000001"
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
    operator._write_once(operation_root / "STAGE_OPERATOR_INTENT.v4.failed.json", v4_intent)
    operator._write_once(operation_root / "STAGE_OBSERVER_ARMED.v4.failed.json", v4_armed)
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
    operator._write_once(operation_root / "STAGE_OPERATOR_INTENT.v5.failed.json", v5_intent)
    operator._write_once(operation_root / "STAGE_OBSERVER_ARMED.v5.failed.json", v5_armed)
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
    operator._write_once(operation_root / "STAGE_OBSERVER_ARMED.json", v6_armed)
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
    operator._write_once(operation_root / "STAGE_OBSERVER_ARMED.json.created.json", creator)
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
    monkeypatch.setenv("OPERATOR_JOB_UID", "00000000-0000-4000-8000-000000000099")
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
    receipt = json.loads((operation_root / "STAGE_OPERATOR_RECOVERY_V7.json").read_bytes())
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
    absence = operator_launch.duplicate_proof(package, previews=previews, factory=FakeKubectl)
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
    assert (
        operator_launch._derived_workload_name(
            "chris-q38-dev17-s51-base-p1-v1",
            "75d47351-61c7-4bd1-995a-93cbfd112d0d",
        )
        == "job-chris-q38-dev17-s51-base-p1-v1-c7f37"
    )
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
        not (call[:2] == ["get", "workloads.kueue.x-k8s.io"] and len(call) < 3) for call in calls
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

    result = direct._cpu_duplicate_checks(identity.stage_name, runner=runner, dev_proof=proof)
    assert result["kubernetes_inventories_checked"] == 10
    assert result["development_duplicate_proof_sha256"] == proof["sha256"]
    assert len(commands) == 5
    assert all(direct.PROD_CONTEXT in command for command in commands)


def test_incluster_runner_allows_only_prod_get_create_and_uid_cas_delete() -> None:
    calls: list[tuple[str, str, bytes | None]] = []

    def request(method: str, path: str, body: bytes | None, _: dict[str, str]) -> tuple[int, bytes]:
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

    assert set(runner.capacity_inventory()) == {
        "pods",
        "inference_models",
        "rayjobs",
        "workloads",
    }
    assert [path for method, path, _body in calls[-4:] if method == "GET"] == [
        "/api/v1/pods",
        "/apis/inference.fleet.ai/v1alpha1/inferencemodels",
        "/apis/ray.io/v1/rayjobs",
        "/apis/kueue.x-k8s.io/v1beta2/workloads",
    ]
    census = operator._fresh_capacity_census({"workers": 1, "gpus_per_worker": 8}, runner)
    assert census["limits"] == {"nodes": 10, "gpus": 80}
    assert census["planned"] == {"nodes": 1, "gpus": 8}
    assert census["projected"] == {"nodes": 1, "gpus": 8}
    assert census["qualified"] is True

    pods = {
        "items": [
            {
                "metadata": {
                    "name": f"chris-q38-live-{index}",
                    "namespace": direct.NAMESPACE,
                    "uid": f"00000000-0000-4000-8000-{index:012d}",
                    "resourceVersion": str(index),
                    "labels": {"fleet.ai/run-name": f"chris-q38-live-{index}"},
                },
                "spec": {
                    "nodeName": f"node-{index}",
                    "containers": [
                        {
                            "resources": {
                                "requests": {"nvidia.com/gpu": "8"},
                                "limits": {"nvidia.com/gpu": "8"},
                            }
                        }
                    ],
                },
                "status": {"phase": "Running", "containerStatuses": []},
            }
            for index in range(10)
        ]
    }

    class FullCapacity:
        @staticmethod
        def capacity_inventory():
            return {
                "pods": pods,
                "inference_models": {"items": []},
                "rayjobs": {"items": []},
                "workloads": {"items": []},
            }

    with pytest.raises(operator.OperatorFailure, match="capacity"):
        operator._fresh_capacity_census({"workers": 1, "gpus_per_worker": 8}, FullCapacity())

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
            input=json.dumps(
                {"apiVersion": "v1", "kind": "Pod", "metadata": {"namespace": direct.NAMESPACE}}
            ),
        )

    for status, payload, message in (
        (403, b'{"kind":"Status"}', "read failed"),
        (200, b'{"kind":"List","items":[1]}', "not a list"),
    ):
        denied = incluster_kubernetes.InClusterKubernetesRunner(
            request=lambda *_args, status=status, payload=payload: (status, payload)
        )
        with pytest.raises(incluster_kubernetes.InClusterKubernetesError, match=message):
            denied.capacity_inventory()


def test_creator_binding_is_invisible_until_its_complete_bytes_are_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_sha256 = "sha256:" + "1" * 64
    manifest_sha256 = "sha256:" + "2" * 64
    uid = "00000000-0000-4000-8000-000000000123"
    observer = cleanup.Observer(
        context=direct.PROD_CONTEXT,
        namespace=direct.NAMESPACE,
        kind="job",
        name="exact-preflight",
        maximum_seconds=direct.CPU_MAXIMUM_SECONDS,
        expected_gpus=0,
        plan_sha256=plan_sha256,
        manifest_sha256=manifest_sha256,
        armed_path=tmp_path / "armed.json",
        result_path=tmp_path / "result.json",
        profile="production-cpu",
    )
    target = observer.creator_binding_path
    partial_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with pytest.raises(cleanup.ObserverError, match="creator binding is invalid"):
            observer._creator_uid()
    finally:
        os.close(partial_fd)
        target.unlink()

    link_started = threading.Event()
    release_link = threading.Event()
    real_link = os.link

    def paused_link(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        link_started.set()
        assert not target.exists()
        assert release_link.wait(5)
        real_link(source, destination)

    monkeypatch.setattr(direct.os, "link", paused_link)
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            direct._publish_creator_binding(
                {"creator_binding_path": str(target)},
                kind="job",
                name=observer.name,
                plan_sha256=plan_sha256,
                manifest_sha256=manifest_sha256,
                uid=uid,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=publish)
    thread.start()
    assert link_started.wait(5)
    assert observer._creator_uid() is None
    release_link.set()
    thread.join(5)

    assert not thread.is_alive()
    assert errors == []
    assert observer._creator_uid() == uid
    assert [path.name for path in tmp_path.iterdir()] == [target.name]


def test_incluster_runner_allows_rayjob_only_for_server_dry_run() -> None:
    calls: list[tuple[str, str, bytes | None]] = []

    def request(
        method: str, path: str, body: bytes | None, _headers: dict[str, str]
    ) -> tuple[int, bytes]:
        calls.append((method, path, body))
        return 201, body or b"{}"

    runner = incluster_kubernetes.InClusterKubernetesRunner(request=request)
    prefix = [
        "kubectl",
        "--context",
        direct.PROD_CONTEXT,
        "--namespace",
        direct.NAMESPACE,
    ]
    manifest = {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {"name": "exact", "namespace": direct.NAMESPACE},
        "spec": {},
    }
    result = runner(
        prefix + ["create", "--dry-run=server", "-f", "-", "-o", "json"],
        input=json.dumps(manifest),
    )
    assert result.returncode == 0
    assert calls == [
        (
            "POST",
            (
                "/apis/ray.io/v1/namespaces/fleet-train-jobs/rayjobs"
                "?dryRun=All&fieldManager=cyber-post-train-prod10"
            ),
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
        )
    ]

    with pytest.raises(
        incluster_kubernetes.InClusterKubernetesError,
        match="reviewed only for server dry-run",
    ):
        runner(
            prefix + ["create", "-f", "-", "-o", "json"],
            input=json.dumps(manifest),
        )
    assert len(calls) == 1


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


@pytest.mark.parametrize("stage", operator._POST_PRE_GUARD_LAUNCH_STAGES)
def test_launch_jobs_error_receipt_exports_only_allowlisted_stage(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "termination-log"
    monkeypatch.setattr(operator, "_TERMINATION_PATH", target)
    monkeypatch.setattr(operator, "_LAUNCH_STAGE", stage)

    operator._write_failure_termination(
        phase="launch", error=JobsError("private path and provider response")
    )

    value = json.loads(target.read_text())
    assert value == operator._seal(
        {
            "schema": operator.FAILURE_TERMINATION_SCHEMA,
            "status": "failed",
            "phase": "launch",
            "error_class": "JobsError",
            "error_code": "operator_unclassified",
            "launch_stage": stage,
            "gpus": 0,
        }
    )
    assert "private" not in json.dumps(value)
    assert "provider" not in json.dumps(value)


def test_launch_failure_receipt_rejects_unlisted_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "termination-log"
    monkeypatch.setattr(operator, "_TERMINATION_PATH", target)
    monkeypatch.setattr(operator, "_LAUNCH_STAGE", "attacker-controlled-private-path")

    operator._write_failure_termination(phase="launch", error=JobsError("secret"))

    value = json.loads(target.read_text())
    assert "launch_stage" not in value
    assert "attacker" not in json.dumps(value)
    assert "secret" not in json.dumps(value)
