from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
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


def _receipt(root: Path, receipt_path: Path, *, nofile_limit: int | None = None) -> dict:
    env = {
        **os.environ,
        "PROBE_TARGET": str(root),
        "PROBE_RECEIPT_SCHEMA": probe.RECEIPT_SCHEMA,
        "PROBE_TRAINING_PLAN_SHA256": probe.TRAINING_PLAN_SHA256,
        "PROBE_RECEIPT_PATH": str(receipt_path),
    }
    preexec_fn = None
    if nofile_limit is not None:

        def set_nofile_limit() -> None:
            _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (nofile_limit, hard))

        preexec_fn = set_nofile_limit
    result = subprocess.run(
        [sys.executable, "-c", probe.RUNTIME],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
        preexec_fn=preexec_fn,
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
    checkpoint_runtime = probe.RUNTIME.split("def checkpoint_inventory(root_fd):", 1)[1].split(
        "\ntry:\n    root_fd=os.open", 1
    )[0]
    assert ".iterdir(" not in checkpoint_runtime
    assert "os.walk(" not in checkpoint_runtime
    assert "os.listdir(parent_fd)" in checkpoint_runtime
    assert "read_small_direct_at(parent_fd" in checkpoint_runtime
    assert "batch_inventory(root_fd)" in probe.RUNTIME
    assert "checkpoint_inventory(root_fd)" in probe.RUNTIME
    assert "file_info_at(root_fd,name)" in probe.RUNTIME
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
    (root / "episodes/batches/batch-secret/STARTED.json").chmod(0)
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
    assert batch["count"] == batch["direct_regular_count"] == 1
    assert batch["indirect_count"] == 0
    assert batch["latest"]["size_bytes"] > 0
    assert type(batch["latest"]["mtime_ns"]) is int
    failed_batch = receipt["batch_inventory"]["markers"]["FAILED.json"]
    assert failed_batch == {
        "count": 1,
        "direct_regular_count": 0,
        "indirect_count": 1,
        "latest": None,
    }
    assert receipt["checkpoint_inventory"] == {
        "directory_present": True,
        "accepted": True,
        "global_step_count": 1,
        "minimum_step": 1,
        "maximum_step": 1,
        "boundary_steps": [1],
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
    long_name = "E" + "x" * 63
    long_function = "f" + "x" * 63
    long_file = "x" * 60 + ".py"
    causes = []
    for cause in range(8):
        frames = [
            {"file": long_file, "line": 2_147_483_647, "function": long_function} for _ in range(20)
        ]
        causes.append(
            {
                "error_class": long_name,
                "actor_init_failed": cause == 0,
                "remote_error_classes": [long_name for _ in range(32)],
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
    for index in range(1000):
        (root / f"checkpoints/global_step_{index}").mkdir(parents=True)
    private_pointer = root / "private-pointer.txt"
    private_pointer.write_text("999\n")
    (root / "checkpoints/latest_ckpt_global_step.txt").symlink_to(private_pointer)

    receipt_path = tmp_path / "receipt.json"
    receipt = _receipt(root, receipt_path)
    assert receipt["receipt_size_limit_bytes"] == 3500
    assert receipt["receipt_size_bytes"] == len(receipt_path.read_bytes())
    assert receipt["receipt_size_bytes"] <= 3500
    assert probe.validate_receipt(receipt, target=str(root)) == receipt
    assert receipt["receipt_compaction_level"] in {0, 1, 2, 3}
    assert receipt["cause_summary"]["native_causes_total"] == 8
    assert receipt["cause_summary"]["native_error_class"] == long_name
    assert receipt["cause_summary"]["native_first_remote_frame"]["line"] == 2_147_483_647
    for name in ("STARTED.json", "COLLECTED.json", "REJECTED.json", "FAILED.json"):
        summary = receipt["batch_inventory"]["markers"][name]
        assert summary["count"] == summary["direct_regular_count"] == 100
        assert summary["indirect_count"] == 0
        assert summary["latest"] is not None
    checkpoint = receipt["checkpoint_inventory"]
    assert checkpoint["global_step_count"] == 1000
    assert checkpoint["boundary_steps"] == [0, 1, 998, 999]
    assert checkpoint["latest_pointer_step"] is None


def test_runtime_compacts_invalid_oversized_cause_without_losing_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _write(
        root / "NATIVE_FAILURE.json",
        {
            "plan_sha256": probe.TRAINING_PLAN_SHA256,
            "causes": [
                {
                    "error_class": "ActorDiedError",
                    "actor_init_failed": False,
                    "remote_error_classes": [],
                    "remote_frames": [
                        {"file": "x" * 4000 + ".py", "line": 1, "function": "generate"}
                    ],
                    "local_frames": [],
                }
            ],
        },
    )
    receipt_path = tmp_path / "receipt.json"
    receipt = _receipt(root, receipt_path)
    assert receipt["receipt_size_bytes"] <= 3500
    assert probe.validate_receipt(receipt, target=str(root)) == receipt
    native = next(item for item in receipt["root_markers"] if item["name"] == "NATIVE_FAILURE.json")
    assert native["accepted"] is False
    assert native["reason"] == "schema_or_binding_mismatch"


def test_runtime_rejects_unbounded_frame_line_without_losing_receipt(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _write(
        root / "NATIVE_FAILURE.json",
        {
            "plan_sha256": probe.TRAINING_PLAN_SHA256,
            "causes": [
                {
                    "error_class": "ActorDiedError",
                    "actor_init_failed": False,
                    "remote_error_classes": [],
                    "remote_frames": [
                        {"file": "episode.py", "line": 10**3400, "function": "generate"}
                    ],
                    "local_frames": [],
                }
            ],
        },
    )
    receipt_path = tmp_path / "receipt.json"
    receipt = _receipt(root, receipt_path)
    assert receipt["receipt_size_bytes"] <= 3500
    assert probe.validate_receipt(receipt, target=str(root)) == receipt
    native = next(item for item in receipt["root_markers"] if item["name"] == "NATIVE_FAILURE.json")
    assert native["accepted"] is False
    assert native["reason"] == "schema_or_binding_mismatch"


def test_runtime_rejects_invalid_training_complete_without_losing_receipt(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    _write(
        root / "NATIVE_TRAINING_COMPLETE.json",
        {
            "status": "native_loop_returned",
            "plan_sha256": probe.TRAINING_PLAN_SHA256,
            "checkpoint_global_step": -1,
            "completed_batches": -99,
            "completed_at": {},
            "optimizer_update_independently_verified": True,
            "checkpoint_reload_verified": False,
        },
    )
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt, target=str(root)) == receipt
    marker = next(
        item for item in receipt["root_markers"] if item["name"] == "NATIVE_TRAINING_COMPLETE.json"
    )
    assert marker["accepted"] is False
    assert marker["reason"] == "schema_or_binding_mismatch"
    assert receipt["terminal_classification"] == "unaccepted_complete_marker"


def test_runtime_rejects_recursive_json_without_losing_receipt(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    marker = root / "NATIVE_FAILURE.json"
    marker.write_text("[" * 20_000 + "0" + "]" * 20_000)
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt, target=str(root)) == receipt
    native = next(item for item in receipt["root_markers"] if item["name"] == "NATIVE_FAILURE.json")
    assert native["accepted"] is False
    assert native["reason"] == "invalid_json"


def test_runtime_does_not_block_on_fifo_markers_or_pointer(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "checkpoints").mkdir(parents=True)
    os.mkfifo(root / "FAILED.json")
    os.mkfifo(root / "checkpoints/latest_ckpt_global_step.txt")
    receipt = _receipt(root, tmp_path / "receipt.json")
    assert probe.validate_receipt(receipt, target=str(root)) == receipt
    failed = next(item for item in receipt["root_markers"] if item["name"] == "FAILED.json")
    assert failed["accepted"] is False
    assert failed["reason"] == "not_small_regular_file"
    assert receipt["checkpoint_inventory"]["latest_pointer_step"] is None


def test_checkpoint_inventory_is_complete_under_low_fd_limit(tmp_path: Path) -> None:
    root = tmp_path / "run"
    step = root / "checkpoints/global_step_1"
    for index in range(100):
        directory = step / f"shard-{index:03d}"
        directory.mkdir(parents=True)
        (directory / "state.bin").write_bytes(b"x")
    receipt = _receipt(root, tmp_path / "receipt.json", nofile_limit=32)
    assert probe.validate_receipt(receipt, target=str(root)) == receipt
    checkpoint = receipt["checkpoint_inventory"]
    assert checkpoint["accepted"] is True
    assert checkpoint["global_step_count"] == 1
    assert checkpoint["file_count"] == checkpoint["total_bytes"] == 100


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
        "receipt_compaction_level": 0,
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
        ("gpus", False),
        ("private_payloads_read", True),
        ("target", "/mnt/sfs/jobs/other"),
        ("receipt_size_limit_bytes", 3500.0),
        ("receipt_compaction_level", True),
    ):
        changed = copy.deepcopy(receipt)
        changed[key] = replacement
        while True:
            body = {name: item for name, item in changed.items() if name != "sha256"}
            changed["sha256"] = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            )
            size = len((json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n").encode())
            if changed["receipt_size_bytes"] == size:
                break
            changed["receipt_size_bytes"] = size
        with pytest.raises(JobsError):
            probe.validate_receipt(changed)
