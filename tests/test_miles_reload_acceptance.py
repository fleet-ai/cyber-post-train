"""Miles reload acceptance is a pure, fail-closed evidence join."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from cyber_post_train.jobs import API_URLS, digest
from training import miles_reload
from training import miles_reload_acceptance as acceptance
from training.miles_conversion import _write

IMAGE = "registry.invalid/miles@sha256:" + "a" * 64
API_RUN_ID = "11111111-1111-4111-8111-111111111111"
NAMESPACE_UID = "22222222-2222-4222-8222-222222222222"
RAYJOB_UID = "33333333-3333-4333-8333-333333333333"
WORKLOAD_UID = "44444444-4444-4444-8444-444444444444"
RAYCLUSTER_UID = "55555555-5555-4555-8555-555555555555"
POD_UID = "66666666-6666-4666-8666-666666666666"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite(path: Path, value: dict) -> dict:
    path.unlink()
    return _write(path, {key: item for key, item in value.items() if key != "sha256"})


def _manifest(tmp_path: Path) -> dict:
    root = tmp_path / "source" / "checkpoints"
    generation = root / "iter_0000000"
    generation.mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text("0\n")
    (generation / ".metadata").write_bytes(b"metadata")
    (generation / "common.pt").write_bytes(b"common")
    for rank in range(8):
        (generation / f"__{rank}_0.distcp").write_bytes(f"rank-{rank}".encode())
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    body = {
        "schema": miles_reload.CHECKPOINT_SCHEMA,
        "image": IMAGE,
        "root": str(root),
        "rollout_index": 0,
        "next_rollout_id": 1,
        "world_size": 8,
        "topology": {"nodes": 1, "gpus_per_node": 8},
        "model": {"repo": "Qwen/Qwen3.8-27B", "revision": "exact-revision"},
        "source": {
            "run_name": "source-run",
            "output_root": str(tmp_path / "source"),
            "plan_sha256": "b" * 64,
            "completion_sha256": "c" * 64,
            "arguments": {},
            "execution": {"image": IMAGE},
            "native_driver_sha256": miles_reload.NATIVE_DRIVER_SHA256,
        },
        "files": files,
        "source_optimizer_update_claimed": False,
        "gpu_reload_verified": False,
        "optimizer_update_during_reload": False,
    }
    return {**body, "sha256": digest(body)}


def _case(tmp_path: Path, monkeypatch) -> dict:
    root = tmp_path / "reload"
    root.mkdir()
    manifest = _manifest(tmp_path)
    config = {"schema": miles_reload.CONFIG_SCHEMA, "identity": "exact-config"}
    plan = {
        "schema": miles_reload.RELOAD_SCHEMA,
        "run_name": "miles-reload",
        "output_root": str(root),
        "source_manifest": manifest,
        "source_manifest_file_sha256": "d" * 64,
        "source_terminal_acceptance": {
            "path": str(tmp_path / "source/MILES_TERMINAL_ACCEPTED.json"),
            "file_sha256": "e" * 64,
            "receipt_sha256": "f" * 64,
        },
        "source_policy_delta_observation": {
            "path": str(tmp_path / "source/POLICY_DELTA.json"),
            "file_sha256": "1" * 64,
            "receipt_sha256": "2" * 64,
        },
        "rank_state_commitment_method": miles_reload.RANK_STATE_COMMITMENT_METHOD,
        "expected_rank_state_commitments": [
            {
                "rank": rank,
                "model_tensor_count": 10,
                "model_local_numel": 100,
                "model_structure_sha256": f"{rank + 10:064x}",
                "model_value_sha256": f"{rank + 20:064x}",
                "optimizer_value_sha256": f"{rank + 30:064x}",
                "scheduler_value_sha256": f"{rank + 40:064x}",
                "rng_value_sha256": f"{rank + 50:064x}",
            }
            for rank in range(8)
        ],
        "runtime_sha256": "e" * 64,
        "native_driver_sha256": miles_reload.NATIVE_DRIVER_SHA256,
        "optimizer_updates": 0,
        "rollouts": 0,
        "deadline_seconds": miles_reload.DEADLINE_SECONDS,
        "execution": {
            "cluster_target": "dev",
            "image": IMAGE,
            "priority": "c1",
            "resources": {},
        },
    }
    request = {
        "name": plan["run_name"],
        "run_dir": plan["output_root"],
        "image": IMAGE,
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "secrets": [],
        "env": {"WANDB_MODE": "disabled", "WANDB_DISABLED": "true"},
    }

    def compile_reload(value, *, relative_to):
        assert relative_to == tmp_path
        if value != config:
            raise ValueError("config drift")
        return plan

    def job_request(value):
        if value != plan:
            raise ValueError("plan drift")
        return request

    monkeypatch.setattr(acceptance.miles_reload, "compile_reload", compile_reload)
    monkeypatch.setattr(acceptance.miles_reload, "job_request", job_request)

    config_path = tmp_path / "reload-config.json"
    plan_path = tmp_path / "reload-plan.json"
    config_path.write_text(json.dumps(config, indent=2))
    plan_path.write_text(json.dumps(plan, indent=2))
    result_path = root / "RELOAD_VALIDATED.json"
    result = _write(
        result_path,
        {
            "schema": miles_reload.RESULT_SCHEMA,
            "status": "reload_validated",
            "plan_sha256": digest(plan),
            "source_manifest_sha256": manifest["sha256"],
            "source_terminal_acceptance_sha256": plan["source_terminal_acceptance"][
                "receipt_sha256"
            ],
            "source_policy_delta_observation_sha256": plan["source_policy_delta_observation"][
                "receipt_sha256"
            ],
            "rank_state_commitment_method": miles_reload.RANK_STATE_COMMITMENT_METHOD,
            "world_size": 8,
            "ranks": list(range(8)),
            "restored_rollout_index": 0,
            "next_rollout_id": 1,
            "probe_set_sha256": "f" * 64,
            "rank_state_commitments_sha256": digest(plan["expected_rank_state_commitments"]),
            "all_rank_model_loaded": True,
            "all_rank_optimizer_loaded": True,
            "all_rank_scheduler_loaded": True,
            "all_rank_rng_loaded": True,
            "all_rank_state_commitments_match": True,
            "state_stable_across_zero_updates": True,
            "optimizer_updates": 0,
            "rollouts": 0,
            "verifier_calls": 0,
            "forwards": 0,
            "backwards": 0,
            "checkpoint_writes": 0,
            "wandb_events": 0,
            "source_checkpoint_unchanged": True,
            "external_gpu_release_verified": False,
            "scientific_rl_acceptance": False,
            "completed_at": 100.0,
        },
    )
    api_run_name = f"{plan['run_name']}-{API_RUN_ID[:8]}"
    controller_path = root / "RELOAD_CONTROLLER_TERMINAL.json"
    controller = _write(
        controller_path,
        {
            "schema": acceptance.CONTROLLER_SCHEMA,
            "status": "succeeded",
            "cluster": "dev",
            "api_base_url": API_URLS["dev"],
            "kube_context": "exact-dev-context",
            "namespace": acceptance.NAMESPACE,
            "namespace_uid": NAMESPACE_UID,
            "run_name": plan["run_name"],
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
            "api_run_id": API_RUN_ID,
            "api_run_name": api_run_name,
            "rayjob_name": api_run_name,
            "rayjob_uid": RAYJOB_UID,
            "workload_name": "workload-exact",
            "workload_uid": WORKLOAD_UID,
            "workload_owner_rayjob_uid": RAYJOB_UID,
            "raycluster_name": "raycluster-exact",
            "raycluster_uid": RAYCLUSTER_UID,
            "raycluster_owner_rayjob_uid": RAYJOB_UID,
            "pods": [
                {
                    "name": "worker-exact",
                    "uid": POD_UID,
                    "owner_raycluster_uid": RAYCLUSTER_UID,
                    "phase": "Succeeded",
                    "exit_code": 0,
                    "termination_reason": "Completed",
                    "terminated_at": 101.0,
                    "runtime_image_id": "containerd://sha256:" + "a" * 64,
                    "container_restarts": 0,
                    "gpus": 8,
                }
            ],
            "api_status": "SUCCEEDED",
            "controller_status": "SUCCEEDED",
            "effective_priority": 10000,
            "automatic_requeue": False,
            "workers": 1,
            "gpus_per_worker": 8,
            "total_gpus": 8,
            "observed_at": 102.0,
        },
    )
    release_path = root / "RELOAD_RELEASE.json"
    identity = {
        key: controller[key]
        for key in (
            "api_run_id",
            "api_run_name",
            "rayjob_name",
            "rayjob_uid",
            "workload_name",
            "workload_uid",
            "raycluster_name",
            "raycluster_uid",
        )
    }
    release = _write(
        release_path,
        {
            "schema": acceptance.RELEASE_SCHEMA,
            "status": "released",
            "cluster": "dev",
            "api_base_url": API_URLS["dev"],
            "kube_context": controller["kube_context"],
            "namespace": acceptance.NAMESPACE,
            "namespace_uid": NAMESPACE_UID,
            "run_name": plan["run_name"],
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
            "controller_terminal_path": str(controller_path),
            "controller_terminal_file_sha256": _file_sha256(controller_path),
            "controller_terminal_sha256": controller["sha256"],
            **identity,
            "pod_uids": [POD_UID],
            "api_status": "SUCCEEDED",
            "controller_status": "SUCCEEDED",
            "rayjob_present": False,
            "workload_present": False,
            "quota_reservation_present": False,
            "raycluster_present": False,
            "gpu_pods_present": False,
            "active_gpu_pod_uids": [],
            "active_gpus": 0,
            "observed_at": 103.0,
        },
    )
    return {
        "root": root,
        "config": config,
        "config_path": config_path,
        "plan": plan,
        "plan_path": plan_path,
        "request": request,
        "result": result,
        "result_path": result_path,
        "controller": controller,
        "controller_path": controller_path,
        "release": release,
        "release_path": release_path,
        "output": root / "RELOAD_ACCEPTED.json",
    }


def _accept(case: dict) -> dict:
    return acceptance.accept_reload(
        config_path=case["config_path"],
        plan_path=case["plan_path"],
        result_path=case["result_path"],
        controller_path=case["controller_path"],
        release_path=case["release_path"],
        output=case["output"],
    )


def test_acceptance_binds_config_plan_terminal_zero_work_release_and_checkpoint(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch)
    result = _accept(case)

    assert result["schema"] == acceptance.ACCEPTED_SCHEMA
    assert result["reload_config"] == case["config"]
    assert result["reload_plan"] == case["plan"]
    assert set(result["work_executed"].values()) == {0}
    assert result["source_checkpoint_unchanged_after_release"] is True
    assert result["source_terminal_acceptance_sha256"] == "f" * 64
    assert result["exact_rank_state_commitments_verified"] is True
    assert result["external_gpu_release_verified"] is True
    assert result["production_promotion_requires_this_receipt"] is True
    assert result["sha256"] == digest(
        {key: value for key, value in result.items() if key != "sha256"}
    )
    validated = acceptance.validate_accepted(result)
    assert validated["reload_plan_sha256"] == digest(case["plan"])
    assert validated["external_release_sha256"] == case["release"]["sha256"]


@pytest.mark.parametrize(
    "field",
    [
        "optimizer_updates",
        "rollouts",
        "verifier_calls",
        "forwards",
        "backwards",
        "checkpoint_writes",
        "wandb_events",
    ],
)
def test_acceptance_rejects_any_process_work(tmp_path, monkeypatch, field):
    case = _case(tmp_path, monkeypatch)
    changed = {**case["result"], field: 1}
    _rewrite(case["result_path"], changed)

    with pytest.raises(ValueError, match="process receipt"):
        _accept(case)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rayjob_present", True),
        ("workload_present", True),
        ("quota_reservation_present", True),
        ("raycluster_present", True),
        ("gpu_pods_present", True),
        ("active_gpu_pod_uids", [POD_UID]),
        ("active_gpus", 8),
    ],
)
def test_acceptance_rejects_incomplete_external_release(tmp_path, monkeypatch, field, value):
    case = _case(tmp_path, monkeypatch)
    _rewrite(case["release_path"], {**case["release"], field: value})

    with pytest.raises(ValueError, match="external GPU release"):
        _accept(case)


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("controller", "workload_owner_rayjob_uid", POD_UID, "terminal controller"),
        ("controller", "effective_priority", 9999, "terminal controller"),
        ("controller", "automatic_requeue", True, "terminal controller"),
        ("controller", "controller_status", "FAILED", "terminal controller"),
        ("controller_pod", "container_restarts", 1, "terminal Pod"),
        ("controller_pod", "runtime_image_id", "containerd://sha256:" + "b" * 64, "terminal Pod"),
    ],
)
def test_acceptance_rejects_controller_identity_or_runtime_drift(
    tmp_path, monkeypatch, target, field, value, message
):
    case = _case(tmp_path, monkeypatch)
    controller = deepcopy(case["controller"])
    if target == "controller_pod":
        controller["pods"][0][field] = value
    else:
        controller[field] = value
    _rewrite(case["controller_path"], controller)

    with pytest.raises(ValueError, match=message):
        _accept(case)


def test_acceptance_rejects_checkpoint_change_after_reload(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    checkpoint = Path(case["plan"]["source_manifest"]["root"])
    (checkpoint / "iter_0000000/__7_0.distcp").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="inventory changed|payload changed"):
        _accept(case)


def test_acceptance_rejects_wandb_capability_in_exact_request(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    case["request"]["env"]["WANDB_PROJECT"] = "unexpected"

    with pytest.raises(ValueError, match="zero-W&B"):
        _accept(case)


@pytest.mark.parametrize("fault", ["config", "plan", "conflict"])
def test_acceptance_rejects_source_binding_drift_or_conflicting_terminal(
    tmp_path, monkeypatch, fault
):
    case = _case(tmp_path, monkeypatch)
    if fault == "config":
        case["config_path"].write_text("{}")
    elif fault == "plan":
        changed = {**case["plan"], "deadline_seconds": 1}
        case["plan_path"].write_text(json.dumps(changed))
    else:
        (case["root"] / "FAILED.json").write_text("{}")

    with pytest.raises(ValueError):
        _accept(case)


def test_accepted_receipt_rejects_input_tampering_and_cannot_be_replaced(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    result = _accept(case)
    case["config_path"].write_text("{}")

    with pytest.raises(ValueError, match="changed after acceptance"):
        acceptance.validate_accepted(result)
    with pytest.raises(FileExistsError, match="destination already exists"):
        _accept(case)


def test_accepted_receipt_forbids_receipt_only_promotion(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    result = _accept(case)

    with pytest.raises(ValueError, match="reopening every referenced file"):
        acceptance.validate_accepted(result, check_files=False)


def test_acceptance_accepts_kubernetes_rfc3339_timestamps(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    result = _rewrite(
        case["result_path"],
        {**case["result"], "completed_at": "2026-09-12T10:00:00Z"},
    )
    controller = deepcopy(case["controller"])
    controller["pods"][0]["terminated_at"] = "2026-09-12T10:00:01Z"
    controller["observed_at"] = "2026-09-12T10:00:02Z"
    controller = _rewrite(case["controller_path"], controller)
    release = deepcopy(case["release"])
    release.update(
        {
            "controller_terminal_file_sha256": _file_sha256(case["controller_path"]),
            "controller_terminal_sha256": controller["sha256"],
            "observed_at": "2026-09-12T10:00:03Z",
        }
    )
    _rewrite(case["release_path"], release)
    case["result"] = result
    case["controller"] = controller

    assert _accept(case)["status"] == "accepted"
