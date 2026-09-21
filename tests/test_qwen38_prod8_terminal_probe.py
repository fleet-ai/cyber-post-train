from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cyber_post_train.jobs import JobsError, digest
from scripts import probe_qwen38_prod8_terminal as probe
from training import dev_cleanup_observer as cleanup
from training import skyrl_reward_rayjob as direct


def _rendered() -> dict:
    value = copy.deepcopy(probe.manifest())
    uid = "12345678-1234-4123-8123-123456789abc"
    labels = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": probe.NAME,
        "controller-uid": uid,
        "job-name": probe.NAME,
    }
    value["metadata"].update(
        {
            "creationTimestamp": "2026-09-21T12:00:00Z",
            "generation": 1,
            "labels": labels,
            "uid": uid,
        }
    )
    value["spec"].update(
        {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
            "suspend": False,
        }
    )
    value["spec"]["template"]["metadata"]["labels"] = labels
    pod = value["spec"]["template"]["spec"]
    pod.update(
        {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
        }
    )
    pod["containers"][0]["imagePullPolicy"] = "IfNotPresent"
    value["status"] = {}
    return value


def _seal(value: dict) -> dict:
    value = copy.deepcopy(value)
    value["sha256"] = digest(value)
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_seal(value), sort_keys=True, separators=(",", ":")))


def _write_unsealed(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _receipt(root: Path, receipt_path: Path) -> dict:
    env = {
        **os.environ,
        "PROBE_TARGET": str(root),
        "PROBE_RECEIPT_SCHEMA": probe.RECEIPT_SCHEMA,
        "PROBE_TRAINING_PLAN_SHA256": probe.TRAINING_PLAN_SHA256,
        "PROBE_RECEIPT_PATH": str(receipt_path),
    }
    result = subprocess.run(
        [sys.executable, "-c", probe.RUNTIME],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout)["status"] == "inspected"
    return json.loads(receipt_path.read_bytes())


def test_manifest_is_bounded_read_only_alert_safe_and_zero_gpu() -> None:
    value = probe.manifest()
    pod = value["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert value["metadata"]["annotations"]["fleet.ai/failure-alerts"] == "off"
    assert value["spec"]["activeDeadlineSeconds"] == 300
    assert value["spec"]["backoffLimit"] == 0
    assert pod["priorityClassName"] == "c1"
    assert pod["automountServiceAccountToken"] is False
    assert container["volumeMounts"] == [{"name": "sfs", "mountPath": "/mnt/sfs", "readOnly": True}]
    assert pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert "nvidia.com/gpu" not in json.dumps(value, sort_keys=True)


def test_preview_is_exactly_validated() -> None:
    proof = probe.preview_evidence(_rendered(), direct.PROD_CONTEXT)
    assert proof["sfs_mount_read_only"] is True
    assert proof["gpus"] == 0
    assert proof["failure_alerts"] == "off"
    changed = _rendered()
    changed["metadata"]["annotations"].pop("fleet.ai/failure-alerts")
    with pytest.raises(JobsError):
        probe.preview_evidence(changed, direct.PROD_CONTEXT)


def test_runtime_reads_only_sanitized_markers_and_inventory(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _write(
        root / "STARTED.json",
        {"plan_sha256": probe.TRAINING_PLAN_SHA256, "started_at": 1.0},
    )
    _write(
        root / "NATIVE_FAILURE.json",
        {
            "plan_sha256": probe.TRAINING_PLAN_SHA256,
            "causes": [
                {
                    "error_class": "ActorDiedError",
                    "actor_init_failed": False,
                    "remote_error_classes": ["RuntimeError"],
                    "remote_frames": [
                        {"file": "skyrl_episode.py", "line": 42, "function": "generate"}
                    ],
                    "local_frames": [{"file": "rl_runtime.py", "line": 159, "function": "run"}],
                }
            ],
        },
    )
    _write(
        root / "FAILED.json",
        {
            "status": "failed",
            "plan_sha256": probe.TRAINING_PLAN_SHA256,
            "error_class": "RuntimeError",
            "watchdog_reason": None,
        },
    )
    _write_unsealed(
        root / "episodes/batches/batch-secret/STARTED.json",
        {
            "schema": "cyber_skyrl_batch_v1",
            "phase": "eval",
            "global_step": 0,
            "optimizer_step_verified": False,
            "trajectory_ids": [["private-trajectory-id", 0]],
            "prompt": "PRIVATE PROMPT MUST NEVER LEAK",
        },
    )
    private = root / "private-skyrl.log"
    private.write_text("PRIVATE LOG MUST NEVER LEAK")
    (root / "episodes/batches/batch-secret/FAILED.json").symlink_to(private)
    checkpoint = root / "checkpoints/global_step_1/policy/model.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not-inspected-tensor-bytes")
    (root / "checkpoints/global_step_1/data.pt").write_bytes(b"sampler")
    (root / "checkpoints/latest_ckpt_global_step.txt").write_text("1\n")

    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt, target=str(root)) == receipt
    assert receipt["terminal_classification"] == "failed"
    serialized = json.dumps(receipt, sort_keys=True)
    assert "PRIVATE" not in serialized
    assert "private-trajectory-id" not in serialized
    assert "batch-secret" not in serialized
    failed = next(item for item in receipt["root_markers"] if item["name"] == "FAILED.json")
    native = next(item for item in receipt["root_markers"] if item["name"] == "NATIVE_FAILURE.json")
    assert failed["selected"]["error_class"] == "RuntimeError"
    assert native["selected"]["causes"][0]["error_class"] == "ActorDiedError"
    assert receipt["batch_inventory"]["batch_directories"] == 1
    batch = receipt["batch_inventory"]["markers"]["STARTED.json"]
    assert batch["count"] == batch["accepted_count"] == 1
    assert batch["latest"]["selected"] == {
        "schema": "cyber_skyrl_batch_v1",
        "phase": "eval",
        "global_step": 0,
        "optimizer_step_verified": False,
    }
    failed_batch = receipt["batch_inventory"]["markers"]["FAILED.json"]
    assert failed_batch == {
        "count": 1,
        "accepted_count": 0,
        "unaccepted_count": 1,
        "latest": None,
    }
    assert receipt["checkpoint_inventory"] == {
        "directory_present": True,
        "accepted": True,
        "global_steps": [1],
        "file_count": 2,
        "total_bytes": len(b"not-inspected-tensor-bytes") + len(b"sampler"),
        "latest_pointer_step": 1,
    }


def test_runtime_does_not_copy_unexpected_failure_fields(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _write(
        root / "FAILED.json",
        {
            "status": "failed",
            "plan_sha256": probe.TRAINING_PLAN_SHA256,
            "error_class": "RuntimeError",
            "watchdog_reason": None,
            "message": "PRIVATE ERROR TEXT MUST NEVER LEAK",
        },
    )
    receipt = _receipt(root, tmp_path / "receipt.json")
    marker = next(item for item in receipt["root_markers"] if item["name"] == "FAILED.json")
    assert marker["accepted"] is False
    assert marker["reason"] == "schema_or_binding_mismatch"
    assert "PRIVATE" not in json.dumps(receipt, sort_keys=True)
    assert receipt["terminal_classification"] == "unaccepted_failed_marker"


def test_runtime_receipt_stays_below_kubernetes_limit_at_maximum_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    causes = []
    for cause in range(8):
        frames = [
            {"file": f"source_{index}.py", "line": index + 1, "function": f"call_{index}"}
            for index in range(20)
        ]
        causes.append(
            {
                "error_class": f"ErrorClass{cause}",
                "actor_init_failed": cause == 0,
                "remote_error_classes": [f"RemoteError{index}" for index in range(32)],
                "remote_frames": frames,
                "local_frames": frames,
            }
        )
    _write(
        root / "NATIVE_FAILURE.json",
        {"plan_sha256": probe.TRAINING_PLAN_SHA256, "causes": causes},
    )
    _write(
        root / "FAILED.json",
        {
            "status": "failed",
            "plan_sha256": probe.TRAINING_PLAN_SHA256,
            "error_class": "RuntimeError",
            "watchdog_reason": None,
        },
    )
    for index in range(100):
        batch = root / f"episodes/batches/batch-{index:03d}"
        common = {
            "schema": "cyber_skyrl_batch_v1",
            "phase": "train" if index else "eval",
            "global_step": index,
            "optimizer_step_verified": False,
            "trajectory_ids": [[str(index), 0]],
        }
        _write_unsealed(batch / "STARTED.json", common)
        _write(batch / "COLLECTED.json", common)
        _write_unsealed(batch / "REJECTED.json", {"reason": "episode_seconds"})
        _write_unsealed(batch / "FAILED.json", {"error_type": "InvalidEpisode"})

    receipt_path = tmp_path / "receipt.json"
    receipt = _receipt(root, receipt_path)
    assert receipt["receipt_size_limit_bytes"] == 3500
    assert receipt["receipt_size_bytes"] == len(receipt_path.read_bytes())
    assert receipt["receipt_size_bytes"] <= 3500
    assert probe.validate_receipt(receipt, target=str(root)) == receipt
    native = next(item for item in receipt["root_markers"] if item["name"] == "NATIVE_FAILURE.json")
    assert native["selected"]["causes_total"] == 8
    assert len(native["selected"]["causes"]) == 2
    assert len(native["selected"]["causes"][0]["remote_error_classes"]) == 4
    assert len(native["selected"]["causes"][0]["remote_frames"]) == 2
    for name in ("STARTED.json", "COLLECTED.json", "REJECTED.json", "FAILED.json"):
        summary = receipt["batch_inventory"]["markers"][name]
        assert summary["count"] == summary["accepted_count"] == 100
        assert summary["unaccepted_count"] == 0
        assert summary["latest"] is not None


def test_receipt_and_observer_reject_drift() -> None:
    receipt = {
        "schema": probe.RECEIPT_SCHEMA,
        "status": "inspected",
        "target": probe.TARGET,
        "training_plan_sha256": probe.TRAINING_PLAN_SHA256,
        "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpus": 0,
        "sfs_mount_read_only": True,
        "private_payloads_read": False,
        "root_exists": True,
        "root_direct": True,
        "terminal_classification": "failed",
        "root_markers": [],
        "batch_inventory": {},
        "checkpoint_inventory": {},
        "receipt_size_limit_bytes": 3500,
        "receipt_size_bytes": 0,
    }
    while True:
        receipt.pop("sha256", None)
        receipt["sha256"] = "sha256:" + digest(receipt)
        size = len((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode())
        if receipt["receipt_size_bytes"] == size:
            break
        receipt["receipt_size_bytes"] = size
    assert probe.validate_receipt(receipt) == receipt
    assert cleanup._validated_receipt(json.dumps(receipt), kind="job") == receipt
    for key, replacement in (
        ("gpus", 1),
        ("private_payloads_read", True),
        ("target", "/mnt/sfs/jobs/other"),
    ):
        changed = copy.deepcopy(receipt)
        changed[key] = replacement
        body = {name: item for name, item in changed.items() if name != "sha256"}
        changed["sha256"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        with pytest.raises(JobsError):
            probe.validate_receipt(changed)
