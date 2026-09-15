from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from training import model_stage_bridge as bridge

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs/qualification/qwen38-27b-stage-bridge-v1.json"
LOCK = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
JOB = ROOT / "cluster/jobs/chris-cyber-qwen38-stage-bridge-1d4bf0f2-v1.yaml"


def _sign(value: dict, field: str) -> dict:
    value[field] = bridge.digest_json(value)
    return value


def test_committed_plan_is_exact_and_dry_run_only() -> None:
    plan, lock = bridge.load_and_validate(PLAN, LOCK)
    assert plan["status"] == "dry_run_only_unapproved"
    assert plan["source"]["revision"] == lock["revision"]
    assert (
        plan["source"]["weights_manifest_sha256"]
        == "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
    )
    assert plan["storage_topology"]["claims_are_distinct"] is True
    assert plan["storage_topology"]["direct_cross_namespace_mount_supported"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda plan: plan["source"].__setitem__("revision", "main"), "immutable model lock"),
        (lambda plan: plan["execution"]["resources"].__setitem__("gpus", 1), "zero-GPU"),
        (lambda plan: plan["execution"].__setitem__("suspend", False), "suspended"),
        (
            lambda plan: plan["storage_topology"].__setitem__(
                "direct_cross_namespace_mount_supported", True
            ),
            "storage topology",
        ),
        (
            lambda plan: plan["storage_topology"].__setitem__("claims_are_distinct", False),
            "storage topology",
        ),
        (
            lambda plan: plan["storage_topology"]["training_claim"].__setitem__(
                "volume_name", "pvc-stale"
            ),
            "training_claim identity",
        ),
        (
            lambda plan: plan["storage_topology"]["inference_claim"].__setitem__(
                "pv_uid", "stale-pv-uid"
            ),
            "inference_claim identity",
        ),
        (
            lambda plan: plan["storage_topology"]["training_claim"]["pv_claim_ref"].__setitem__(
                "uid", "wrong-pvc-uid"
            ),
            "claimRef",
        ),
        (
            lambda plan: plan["storage_topology"]["training_claim"].__setitem__(
                "pv_snapshot_sha256", "invalid"
            ),
            "pv_snapshot_sha256",
        ),
        (lambda plan: plan["destination"].__setitem__("atomic_transaction", "mv"), "no-replace"),
        (lambda plan: plan.__setitem__("status", "approved"), "approval status"),
    ],
)
def test_plan_fails_closed_on_drift(tmp_path: Path, mutate, message: str) -> None:
    plan = json.loads(PLAN.read_text())
    plan.pop("plan_sha256")
    mutate(plan)
    _sign(plan, "plan_sha256")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=message):
        bridge.load_and_validate(path, LOCK)


def test_jointly_tampered_plan_and_lock_still_fail(tmp_path: Path) -> None:
    plan = json.loads(PLAN.read_text())
    lock = json.loads(LOCK.read_text())
    lock["revision"] = "attacker-revision"
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    plan["source"]["revision"] = "attacker-revision"
    plan["model_lock_file_sha256"] = bridge.file_sha256(lock_path)
    plan.pop("plan_sha256")
    _sign(plan, "plan_sha256")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="destination root"):
        bridge.load_and_validate(plan_path, lock_path)


def _synthetic_plan(tmp_path: Path, small_bytes: bytes, shard_bytes: bytes) -> dict:
    shard = {
        "path": "model-00001-of-00001.safetensors",
        "size": len(shard_bytes),
        "sha256": hashlib.sha256(shard_bytes).hexdigest(),
    }
    plan = {
        "status": "approved_for_execution",
        "source": {
            "repository": "example/model",
            "revision": "a" * 40,
            "weight_shards": 1,
            "weight_bytes": len(shard_bytes),
            "weights_manifest_sha256": bridge.digest_json([shard]),
        },
        "files": {"config.json": "sha256:" + hashlib.sha256(small_bytes).hexdigest()},
        "destination": {"immutable_root": str(tmp_path / "final")},
        "execution": {"job_name": "chris-test"},
        "plan_sha256": "sha256:" + "1" * 64,
    }
    return plan


def test_execute_is_atomic_exact_and_idempotent(tmp_path: Path, monkeypatch) -> None:
    small = b"config"
    shard_bytes = b"weights"
    plan = _synthetic_plan(tmp_path, small, shard_bytes)
    shard = {
        "path": "model-00001-of-00001.safetensors",
        "size": len(shard_bytes),
        "sha256": hashlib.sha256(shard_bytes).hexdigest(),
    }
    lock_path = tmp_path / "model-lock.json"
    lock_path.write_text("{}")
    monkeypatch.setattr(bridge, "load_and_validate", lambda *_args: (copy.deepcopy(plan), {}))
    monkeypatch.setattr(bridge, "fetch_shard_manifest", lambda _plan: [shard])

    calls = 0

    def fake_download(argv, check):
        nonlocal calls
        calls += 1
        root = Path(argv[argv.index("--local-dir") + 1])
        (root / "config.json").write_bytes(small)
        (root / shard["path"]).write_bytes(shard_bytes)
        (root / ".cache").mkdir()
        (root / ".cache" / "untrusted-metadata").write_text("removed")

    def fake_rename(source: Path, destination: Path) -> None:
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(source, destination)

    monkeypatch.setattr(bridge.subprocess, "run", fake_download)
    monkeypatch.setattr(bridge, "_rename_noreplace", fake_rename)
    first = bridge.execute(Path("ignored"), lock_path, tmp_path / "transactions")
    second = bridge.execute(Path("ignored"), lock_path, tmp_path / "transactions")
    assert calls == 1
    assert first == second
    assert first["destination"]["no_replace"] is True
    assert first["payload"]["file_count"] == 4
    assert not (tmp_path / "final" / ".cache").exists()


def test_committed_dry_run_plan_cannot_execute(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="not been approved"):
        bridge.execute(PLAN, LOCK, tmp_path)


def test_existing_tree_with_extra_file_is_rejected(tmp_path: Path, monkeypatch) -> None:
    test_execute_is_atomic_exact_and_idempotent(tmp_path, monkeypatch)
    (tmp_path / "final" / "extra.bin").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="type inventory"):
        bridge.execute(Path("ignored"), tmp_path / "model-lock.json", tmp_path / "transactions-2")


def test_out_of_tree_provenance_symlink_is_rejected(tmp_path: Path, monkeypatch) -> None:
    test_execute_is_atomic_exact_and_idempotent(tmp_path, monkeypatch)
    provenance = tmp_path / "final" / "source-tree.json"
    outside = tmp_path / "outside-source-tree.json"
    outside.write_bytes(provenance.read_bytes())
    provenance.unlink()
    provenance.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        bridge.execute(Path("ignored"), tmp_path / "model-lock.json", tmp_path / "transactions-2")


@pytest.mark.parametrize("directory", ["empty", ".cache"])
def test_unexpected_empty_or_cache_directory_is_rejected(
    tmp_path: Path, monkeypatch, directory: str
) -> None:
    test_execute_is_atomic_exact_and_idempotent(tmp_path, monkeypatch)
    (tmp_path / "final" / directory).mkdir()
    with pytest.raises(ValueError, match="unexpected directory"):
        bridge.execute(Path("ignored"), tmp_path / "model-lock.json", tmp_path / "transactions-2")


def test_cache_removal_failure_is_fatal(tmp_path: Path, monkeypatch) -> None:
    small = b"config"
    shard_bytes = b"weights"
    plan = _synthetic_plan(tmp_path, small, shard_bytes)
    shard = {
        "path": "model-00001-of-00001.safetensors",
        "size": len(shard_bytes),
        "sha256": hashlib.sha256(shard_bytes).hexdigest(),
    }
    lock_path = tmp_path / "model-lock.json"
    lock_path.write_text("{}")
    monkeypatch.setattr(bridge, "load_and_validate", lambda *_args: (copy.deepcopy(plan), {}))
    monkeypatch.setattr(bridge, "fetch_shard_manifest", lambda _plan: [shard])

    def fake_download(argv, check):
        root = Path(argv[argv.index("--local-dir") + 1])
        (root / "config.json").write_bytes(small)
        (root / shard["path"]).write_bytes(shard_bytes)
        (root / ".cache").mkdir()

    monkeypatch.setattr(bridge.subprocess, "run", fake_download)
    monkeypatch.setattr(
        bridge.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("simulated cache removal failure")),
    )
    with pytest.raises(OSError, match="cache removal failure"):
        bridge.execute(Path("ignored"), lock_path, tmp_path / "transactions")
    assert not (tmp_path / "final").exists()


def test_job_is_suspended_queue_owned_zero_gpu_and_mounts_only_training_sfs() -> None:
    job = yaml.safe_load(JOB.read_text())
    plan = json.loads(PLAN.read_text())
    assert job["metadata"]["name"] == "chris-cyber-qwen38-stage-bridge-1d4bf0f2-v1"
    assert job["metadata"]["namespace"] == "fleet-train-jobs"
    assert job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "training-lq"
    assert job["spec"]["suspend"] is True
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert "nvidia.com/gpu" not in container["resources"]["requests"]
    assert "nvidia.com/gpu" not in container["resources"]["limits"]
    assert "@sha256:" in container["image"]
    assert container["image"] == plan["execution"]["image"]
    assert container["command"] == plan["execution"]["command"]
    claims = [
        volume.get("persistentVolumeClaim", {}).get("claimName")
        for volume in job["spec"]["template"]["spec"]["volumes"]
    ]
    assert claims == [None, "sfs-shared"]


def test_preview_script_has_no_create_path() -> None:
    script = (ROOT / "scripts/preview_qwen38_stage_bridge.sh").read_text()
    assert "dry-run only" in script
    assert script.count("--dry-run=server") == 2
    assert "kubectl apply" not in script
    assert "kubectl create -f" not in script
