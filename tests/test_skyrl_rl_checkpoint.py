"""CPU-only fail-closed tests for native SkyRL RL checkpoint reload."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from typer.testing import CliRunner

from cyber_post_train import cli
from cyber_post_train.jobs import digest
from training import skyrl_rl_checkpoint as checkpoint

RUNNER = CliRunner()
ZERO = "0" * 64
ONE = "1" * 64
UUIDS = {
    "runtime_run_id": "00000000-0000-4000-8000-000000000001",
    "rayjob_uid": "00000000-0000-4000-8000-000000000002",
    "workload_uid": "00000000-0000-4000-8000-000000000003",
    "raycluster_uid": "00000000-0000-4000-8000-000000000004",
    "pod_uids": ["00000000-0000-4000-8000-000000000005"],
}


def sealed(value: dict) -> dict:
    return {**value, "sha256": digest(value)}


def implementation(start: str, end: str) -> str:
    source = Path(checkpoint.__file__).read_text()
    return source[source.index("def " + start) : source.index("def " + end)]


def source_plan() -> dict:
    args = checkpoint.skyrl.SkyRLConfig(
        name=checkpoint.SOURCE_RUN_NAME,
        output_root=checkpoint.SOURCE_OUTPUT_ROOT,
        model="Qwen/Qwen3.8-27B",
        model_root=checkpoint.SOURCE_MODEL_ROOT,
        train_data=checkpoint.SOURCE_DATA_ROOT + "/train.jsonl",
        dev_data=checkpoint.SOURCE_DATA_ROOT + "/dev.jsonl",
        data_manifest=checkpoint.SOURCE_DATA_ROOT + "/manifest.json",
        train_rows=1,
        dev_rows=1,
        wandb_entity="thefleet",
        wandb_project="cyber-post-train",
        wandb_run_id=checkpoint.SOURCE_RUN_NAME,
        nodes=1,
        steps=1,
        groups=1,
        samples_per_prompt=8,
        lr=1e-6,
        eval_interval=1,
        checkpoint_interval=1,
        keep_checkpoints=2,
        seed=42,
        context_tokens=98304,
        response_tokens=81920,
        tokens_per_turn=4096,
        max_turns=600,
        engine_start_timeout_seconds=1800,
        engine_cleanup_timeout_seconds=300,
    )
    return {
        "schema": "cyber_skyrl_training_v1",
        "run_name": checkpoint.SOURCE_RUN_NAME,
        "output_root": checkpoint.SOURCE_OUTPUT_ROOT,
        "model": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": checkpoint.SOURCE_MODEL_REVISION,
            "root": checkpoint.SOURCE_MODEL_ROOT,
            "weight_manifest_sha256": (
                "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
            ),
            "files": [{"path": "config.json", "sha256": ZERO}],
        },
        "data": {
            "schema": "cyber_skyrl_data_v1",
            "name": checkpoint.SOURCE_RUN_NAME,
            "selection_sha256": (
                "sha256:608b8c47790fd6e9fef6e86115a7b624273589b8b11e920224db54b7b848e965"
            ),
            "split_sha256": (
                "sha256:fe77cff7256c8c7554eceaf228b855ccc506c2035882d459499fe9b5505ea7db"
            ),
            "tool_catalog_sha256": (
                "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
            ),
            "limits": {
                "context_tokens": 98304,
                "response_tokens": 81920,
                "max_tokens_per_turn": 4096,
                "max_turns": 600,
                "episode_seconds": 2400,
                "tool_seconds": 330,
                "tool_result_chars": 50000,
            },
            "tokenizer": {
                "repo": "Qwen/Qwen3.8-27B",
                "revision": checkpoint.SOURCE_MODEL_REVISION,
            },
            "files": {
                "train": {"path": "train.jsonl", "rows": 1},
                "dev": {"path": "dev.jsonl", "rows": 1},
            },
        },
        "native_sources": {"source": ZERO},
        "runtime_sha256": ZERO,
        "native_overrides": checkpoint.skyrl.overrides(args),
        "arguments": vars(args),
        "execution": {
            "image": "registry/skyrl@sha256:" + ZERO,
            "cluster_target": "dev",
            "priority": "c1",
            "resources": checkpoint._RESOURCES,
            "runtime_user": {"uid": 1000, "gid": 100, "run_as_non_root": True},
            "engine_diagnostic_prerequisite": {"exact": "accepted-dev8"},
            "reward_canary_source": {"exact": "registry-source"},
        },
    }


def release(
    plan: dict,
    observed_at: str = "2026-09-12T01:01:00Z",
    *,
    request_sha256: str = ZERO,
) -> dict:
    reload = plan.get("schema") == checkpoint.RELOAD_SCHEMA
    api_run_name = plan["run_name"] + "-1234abcd"
    api_job_id = api_run_name + "-abcde"
    rayjob_name = api_run_name
    audit_name = "RELOAD_CONTROLLER_AUDIT.json" if reload else "SOURCE_CONTROLLER_AUDIT.json"
    value = {
        "schema": checkpoint.RELEASE_SCHEMA,
        "status": "released",
        "cluster": "dev",
        "kube_context": checkpoint.DEV_KUBE_CONTEXT,
        "namespace": checkpoint.NAMESPACE,
        "namespace_uid": checkpoint.NAMESPACE_UID,
        "run_name": plan["run_name"],
        "plan_sha256": digest(plan),
        "request_sha256": request_sha256,
        "api_base_url": "https://api.ft.dev.flt.build",
        "submission_journal_path": str(
            Path(plan["output_root"])
            / ("RELOAD_SUBMISSION.jsonl" if reload else "SOURCE_SUBMISSION.jsonl")
        ),
        "submission_journal_file_sha256": ZERO,
        "controller_audit_path": str(Path(plan["output_root"]) / audit_name),
        "controller_audit_file_sha256": ZERO,
        "controller_audit_sha256": ZERO,
        "api_run_name": api_run_name,
        "post_response_job_id": None,
        "api_job_id": api_job_id,
        **UUIDS,
        "rayjob_name": rayjob_name,
        "workload_name": plan["run_name"] + "-workload",
        "raycluster_name": plan["run_name"] + "-raycluster",
        "pod_terminal_observations": [
            {
                "name": plan["run_name"] + "-pod",
                "uid": UUIDS["pod_uids"][0],
                "owner_raycluster_uid": UUIDS["raycluster_uid"],
                "phase": "Succeeded",
                "exit_code": 0,
                "termination_reason": "Completed",
                "terminated_at": observed_at,
                "runtime_image_id": "containerd://sha256:" + ZERO,
                "runtime_uid": 1000,
                "runtime_gid": 100,
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
        "runtime_image_id": "containerd://sha256:" + ZERO,
        "container_restarts": 0,
        "raycluster_present": False,
        "gpu_pods_present": False,
        "active_gpus": 0,
        "gpu_release_proven": True,
        "observed_at": observed_at,
    }
    if not reload:
        value.update(
            source_pod_uid=UUIDS["pod_uids"][0],
            source_runtime_image_id="containerd://sha256:" + ZERO,
            source_runtime_uid=1000,
            source_runtime_gid=100,
        )
    value["controller_audit_sha256"] = controller_audit(plan, value)["sha256"]
    return sealed(value)


def controller_audit(plan: dict, value: dict) -> dict:
    return sealed(
        {
            "schema": checkpoint.CONTROLLER_AUDIT_SCHEMA,
            "cluster": "dev",
            "kube_context": checkpoint.DEV_KUBE_CONTEXT,
            "namespace": checkpoint.NAMESPACE,
            "namespace_uid": checkpoint.NAMESPACE_UID,
            "api_base_url": "https://api.ft.dev.flt.build",
            "run_name": plan["run_name"],
            "request_sha256": value["request_sha256"],
            "api_run_name": value["api_run_name"],
            "post_response_job_id": value["post_response_job_id"],
            "api_job_id": value["api_job_id"],
            "runtime_run_id": value["runtime_run_id"],
            "rayjob_name": value["rayjob_name"],
            "rayjob_uid": value["rayjob_uid"],
            "workload_name": value["workload_name"],
            "workload_uid": value["workload_uid"],
            "workload_owner_rayjob_uid": value["rayjob_uid"],
            "raycluster_name": value["raycluster_name"],
            "raycluster_uid": value["raycluster_uid"],
            "raycluster_owner_rayjob_uid": value["rayjob_uid"],
            "pods": [
                {
                    "name": plan["run_name"] + "-pod",
                    "uid": value["pod_uids"][0],
                    "owner_raycluster_uid": value["raycluster_uid"],
                    "runtime_image_id": value["runtime_image_id"],
                    "runtime_uid": 1000,
                    "runtime_gid": 100,
                    "container_restarts": 0,
                    "gpus": 8,
                }
            ],
            "effective_priority": 10000,
            "automatic_requeue": False,
            "workers": 1,
            "gpus_per_worker": 8,
            "total_gpus": 8,
            "observed_at": "2026-09-12T01:00:30Z",
        }
    )


def inventory() -> dict:
    names = checkpoint._required_checkpoint_files() | {
        "policy/huggingface/config.json",
        "policy/huggingface/tokenizer_config.json",
    }
    return {
        name: {
            "bytes": 1,
            "sha256": ZERO,
        }
        for name in sorted(names)
    }


def checkpoint_candidate() -> dict:
    source = source_plan()
    files = inventory()
    completion = sealed(
        {
            "status": "native_loop_returned",
            "plan_sha256": digest(source),
            "checkpoint_global_step": 1,
            "completed_batches": 3,
            "completed_at": 1_757_642_400.0,
            "optimizer_update_independently_verified": False,
            "checkpoint_reload_verified": False,
        }
    )
    value = {
        "schema": checkpoint.CHECKPOINT_CANDIDATE_SCHEMA,
        "source_plan_sha256": digest(source),
        "source_plan": source,
        "runtime": {
            "expected_image": checkpoint.IMAGE,
            "runtime_sha256": digest(checkpoint._runtime()),
            "native_sources_sha256": digest(source["native_sources"]),
            "module_sha256": checkpoint._hash(Path(checkpoint.__file__)),
            "python": "3.11.0",
            "packages": {"ray": "2.56.0", "torch": "2.9.0", "transformers": "5.0.0"},
            "uid": 1000,
            "gid": 100,
            "cuda_available": False,
        },
        "completion": {
            "path": checkpoint.SOURCE_OUTPUT_ROOT + "/NATIVE_TRAINING_COMPLETE.json",
            "file_sha256": ZERO,
            "receipt": completion,
        },
        "release": {
            "path": checkpoint.SOURCE_OUTPUT_ROOT + "/SOURCE_RELEASE.json",
            "file_sha256": ZERO,
            "receipt": release(source),
        },
        "source_stat_sha256": ZERO,
        "checkpoint": {
            "path": checkpoint.SOURCE_OUTPUT_ROOT + "/checkpoints/global_step_1",
            "step": 1,
            "world_size": 8,
            "files": files,
            "total_bytes": sum(row["bytes"] for row in files.values()),
            "latest_pointer": {
                "path": checkpoint.SOURCE_OUTPUT_ROOT + "/checkpoints/latest_ckpt_global_step.txt",
                "file_sha256": ZERO,
            },
            "native_config_sha256": ZERO,
            "sampler_batches_in_epoch": 1,
            "ranks": [
                {
                    "rank": rank,
                    "policy_state_sha256": ONE if rank == 0 else ZERO,
                    "optimizer_state_sha256": ONE,
                    "scheduler_state_sha256": ONE,
                    "rng_state_sha256": ONE,
                    "optimizer_states": 3,
                    "optimizer_step_states": 3,
                    "optimizer_step": 1,
                    "scheduler_last_epoch": 1,
                    "rng_present": True,
                }
                for rank in range(8)
            ],
        },
        "created_at": 1_757_642_500.0,
    }
    return sealed(value)


def sealer_contract(source: dict) -> dict:
    return {
        "schema": "cyber_skyrl_rl_checkpoint_sealer_job_v1",
        "run_name": checkpoint.SEALER_RUN_NAME,
        "output_root": checkpoint.SEALER_OUTPUT_ROOT,
        "source_plan_sha256": digest(source),
        "source_request_sha256": digest({"source": "request"}),
        "checkpoint_global_step": 1,
        "candidate_path": checkpoint.SEALER_OUTPUT_ROOT + "/RL_CHECKPOINT_STEP_1_CANDIDATE.json",
        "image": checkpoint.IMAGE,
        "kube_context": checkpoint.DEV_KUBE_CONTEXT,
        "namespace": checkpoint.NAMESPACE,
        "namespace_uid": checkpoint.NAMESPACE_UID,
        "security_context": {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True},
        "restart_policy": "Never",
        "backoff_limit": 0,
        "gpus": 0,
    }


def sealer_audit(candidate: dict) -> dict:
    path = checkpoint.SEALER_OUTPUT_ROOT + "/RL_CHECKPOINT_STEP_1_CANDIDATE.json"
    value = {
        "schema": checkpoint.CHECKPOINT_SEALER_AUDIT_SCHEMA,
        "cluster": "dev",
        "kube_context": checkpoint.DEV_KUBE_CONTEXT,
        "namespace": checkpoint.NAMESPACE,
        "namespace_uid": checkpoint.NAMESPACE_UID,
        "run_name": checkpoint.SEALER_RUN_NAME,
        "source_plan_sha256": candidate["source_plan_sha256"],
        "source_request_sha256": digest({"source": "request"}),
        "checkpoint_global_step": 1,
        "candidate_path": path,
        "candidate_file_sha256": ZERO,
        "candidate_self_sha256": candidate["sha256"],
        "request_sha256": digest(sealer_contract(candidate["source_plan"])),
        "job_name": checkpoint.SEALER_RUN_NAME,
        "job_uid": "00000000-0000-4000-8000-000000000011",
        "pod_name": checkpoint.SEALER_RUN_NAME + "-abcde",
        "pod_uid": "00000000-0000-4000-8000-000000000012",
        "pod_owner_job_uid": "00000000-0000-4000-8000-000000000011",
        "runtime_image_id": "containerd://sha256:" + checkpoint.IMAGE.rsplit("@sha256:", 1)[-1],
        "effective_security_context": {
            "runAsUser": 1000,
            "runAsGroup": 100,
            "runAsNonRoot": True,
        },
        "container_restarts": 0,
        "gpus": 0,
        "observed_at": "2026-09-12T01:02:00Z",
    }
    value["kubernetes_observation"] = {
        "authority": "kubernetes_api",
        "verb": "GET",
        "kube_context": value["kube_context"],
        "namespace": value["namespace"],
        "namespace_uid": value["namespace_uid"],
        "job": {
            "api_version": "batch/v1",
            "kind": "Job",
            "name": value["job_name"],
            "uid": value["job_uid"],
            "resource_version": "101",
            "complete": False,
            "failed": False,
        },
        "pod": {
            "api_version": "v1",
            "kind": "Pod",
            "name": value["pod_name"],
            "uid": value["pod_uid"],
            "resource_version": "102",
            "owner_job_uid": value["job_uid"],
            "phase": "Running",
            "container_name": "training",
            "runtime_image_id": value["runtime_image_id"],
            "security_context": value["effective_security_context"],
            "restart_count": 0,
            "gpu_limit": 0,
            "exit_code": None,
        },
        "observed_at": value["observed_at"],
    }
    return sealed(value)


def sealer_release(candidate: dict, audit: dict) -> dict:
    value = {
        "schema": checkpoint.CHECKPOINT_SEALER_RELEASE_SCHEMA,
        "status": "released",
        "cluster": "dev",
        "kube_context": checkpoint.DEV_KUBE_CONTEXT,
        "namespace": checkpoint.NAMESPACE,
        "namespace_uid": checkpoint.NAMESPACE_UID,
        "run_name": checkpoint.SEALER_RUN_NAME,
        "source_plan_sha256": candidate["source_plan_sha256"],
        "checkpoint_global_step": 1,
        "candidate_path": checkpoint.SEALER_OUTPUT_ROOT + "/RL_CHECKPOINT_STEP_1_CANDIDATE.json",
        "candidate_file_sha256": ZERO,
        "candidate_self_sha256": candidate["sha256"],
        "controller_audit_path": checkpoint.SEALER_OUTPUT_ROOT
        + "/CHECKPOINT_SEALER_CONTROLLER_AUDIT.json",
        "controller_audit_file_sha256": ZERO,
        "controller_audit_self_sha256": audit["sha256"],
        "job_name": audit["job_name"],
        "job_uid": audit["job_uid"],
        "pod_name": audit["pod_name"],
        "pod_uid": audit["pod_uid"],
        "job_complete": True,
        "job_failed": False,
        "pod_succeeded": True,
        "container_exit_code": 0,
        "terminated_at": "2026-09-12T01:03:00Z",
        "job_present": False,
        "pod_present": False,
        "active_gpus": 0,
        "observed_at": "2026-09-12T01:04:00Z",
    }
    value["terminal_observation"] = {
        **audit["kubernetes_observation"],
        "job": {
            **audit["kubernetes_observation"]["job"],
            "resource_version": "103",
            "complete": True,
        },
        "pod": {
            **audit["kubernetes_observation"]["pod"],
            "resource_version": "104",
            "phase": "Succeeded",
            "exit_code": 0,
        },
        "observed_at": value["terminated_at"],
    }
    value["absence_observation"] = {
        "authority": "kubernetes_api",
        "verb": "GET",
        "kube_context": checkpoint.DEV_KUBE_CONTEXT,
        "namespace": checkpoint.NAMESPACE,
        "namespace_uid": checkpoint.NAMESPACE_UID,
        "job_name": value["job_name"],
        "job_http_status": 404,
        "pod_name": value["pod_name"],
        "pod_http_status": 404,
        "active_gpu_pod_uids": [],
        "observed_at": value["observed_at"],
    }
    return sealed(value)


def manifest() -> dict:
    candidate = checkpoint_candidate()
    audit = sealer_audit(candidate)
    external_release = sealer_release(candidate, audit)
    runtime = candidate["runtime"]
    return sealed(
        {
            "schema": checkpoint.CHECKPOINT_SCHEMA,
            **{
                key: candidate[key]
                for key in (
                    "source_plan_sha256",
                    "source_plan",
                    "completion",
                    "release",
                    "source_stat_sha256",
                    "checkpoint",
                )
            },
            "sealer": {
                "image": checkpoint.IMAGE,
                **{
                    key: runtime[key]
                    for key in (
                        "runtime_sha256",
                        "native_sources_sha256",
                        "module_sha256",
                        "python",
                        "packages",
                        "uid",
                        "gid",
                    )
                },
                "candidate": {
                    "path": checkpoint.SEALER_OUTPUT_ROOT + "/RL_CHECKPOINT_STEP_1_CANDIDATE.json",
                    "file_sha256": ZERO,
                    "receipt": candidate,
                },
                "controller_audit": {
                    "path": checkpoint.SEALER_OUTPUT_ROOT
                    + "/CHECKPOINT_SEALER_CONTROLLER_AUDIT.json",
                    "file_sha256": ZERO,
                    "receipt": audit,
                },
                "release": {
                    "path": checkpoint.SEALER_OUTPUT_ROOT + "/CHECKPOINT_SEALER_RELEASE.json",
                    "file_sha256": ZERO,
                    "receipt": external_release,
                },
            },
        }
    )


def rank_replies() -> list[dict]:
    return [
        {
            "rank": rank,
            "world_size": 8,
            "base_policy_sha256": ZERO,
            "loaded_policy_sha256": ONE if rank == 0 else ZERO,
            "post_forward_policy_sha256": ONE if rank == 0 else ZERO,
            "optimizer_states": 3,
            "optimizer_step_states": 3,
            "optimizer_step": 1,
            "loaded_optimizer_sha256": ONE,
            "post_forward_optimizer_sha256": ONE,
            "loaded_scheduler_sha256": ONE,
            "post_forward_scheduler_sha256": ONE,
            "rng_state_sha256": ONE,
            "post_forward_rng_sha256": ONE,
            "forward_finite": True,
            "forward_batch": 1,
            "forward_tokens": 4,
            "gradients_created": 0,
            "optimizer_updates": 0,
            "runtime_uid": 1000,
            "runtime_gid": 100,
            "native_sources_sha256": digest(source_plan()["native_sources"]),
        }
        for rank in range(8)
    ]


def test_release_requires_terminal_uid_bound_post_completion_gpu_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = source_plan()
    monkeypatch.setattr(checkpoint, "_submission", lambda *_: ({"request_sha256": ZERO}, ZERO))
    monkeypatch.setattr(
        checkpoint,
        "_controller_audit",
        lambda _plan, value: (controller_audit(_plan, value), ZERO),
    )
    terminal = checkpoint._observed_epoch("2026-09-12T01:00:00Z")
    checkpoint._release(release(plan), plan, not_before=terminal)
    for field, value in (
        ("active_gpus", 1),
        ("active_gpus", False),
        ("raycluster_present", True),
        ("container_restarts", 1),
        ("controller_status", "FAILED"),
        ("effective_priority", 9999),
        ("automatic_requeue", True),
        ("runtime_image_id", "containerd://sha256:" + ONE),
        ("runtime_run_id", "not-a-uuid"),
        ("api_run_name", "wrong-name"),
        ("source_runtime_uid", 0),
        ("source_runtime_gid", 0),
    ):
        changed = {**release(plan), field: value}
        changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
        with pytest.raises(ValueError, match="release"):
            checkpoint._release(changed, plan, not_before=terminal)
    with pytest.raises(ValueError, match="predates"):
        checkpoint._release(
            release(plan, "2026-09-12T00:00:00Z"),
            plan,
            not_before=terminal,
        )


def test_release_keeps_api_runtime_and_kubernetes_identities_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = source_plan()
    value = release(plan)
    monkeypatch.setattr(checkpoint, "_submission", lambda *_: ({"request_sha256": ZERO}, ZERO))
    monkeypatch.setattr(
        checkpoint,
        "_controller_audit",
        lambda _plan, current: (controller_audit(_plan, current), ZERO),
    )
    checkpoint._release(value, plan)

    # A real create-once journal may contain a null POST job_id; the later API
    # and Kubernetes observations separately bind the generated RayJob name.
    assert value["post_response_job_id"] is None
    assert value["api_run_name"] == plan["run_name"] + "-1234abcd"
    assert value["rayjob_name"] == value["api_run_name"]
    assert value["api_job_id"] == value["api_run_name"] + "-abcde"

    for mutation in (
        {"rayjob_name": value["api_job_id"]},
        {"api_job_id": None},
        {"api_job_id": value["api_run_name"]},
        {"post_response_job_id": UUIDS["rayjob_uid"]},
        {"runtime_run_id": UUIDS["rayjob_uid"]},
        {"rayjob_uid": UUIDS["runtime_run_id"]},
    ):
        changed = {**value, **mutation}
        changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
        with pytest.raises(ValueError, match="release"):
            checkpoint._release(changed, plan)


def test_release_is_bound_to_exact_jobs_journal_and_controller_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = source_plan()
    plan["output_root"] = str(tmp_path)
    plan["arguments"]["output_root"] = str(tmp_path)
    request = {"exact": "request"}
    request_sha256 = digest(request)
    monkeypatch.setattr(checkpoint, "_request", lambda _: request)
    path = tmp_path / "SOURCE_SUBMISSION.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "state": "POST_INTENT_DO_NOT_RETRY",
                    "api_base_url": "https://api.ft.dev.flt.build",
                    "request_sha256": request_sha256,
                },
                {
                    "state": "POST_RESPONSE",
                    "name": plan["run_name"] + "-1234abcd",
                    "job_id": None,
                    "run_dir": str(tmp_path),
                },
            )
        )
        + "\n"
    )
    value = release(plan, request_sha256=request_sha256)
    value["submission_journal_file_sha256"] = checkpoint._hash(path)
    audit = controller_audit(plan, value)
    audit_path = tmp_path / "SOURCE_CONTROLLER_AUDIT.json"
    audit_path.write_text(json.dumps(audit))

    wrong_user = copy.deepcopy(audit)
    wrong_user["pods"][0]["runtime_uid"] = 0
    wrong_user["sha256"] = digest(
        {key: item for key, item in wrong_user.items() if key != "sha256"}
    )
    audit_path.write_text(json.dumps(wrong_user))
    changed = {
        **value,
        "controller_audit_file_sha256": checkpoint._hash(audit_path),
        "controller_audit_sha256": wrong_user["sha256"],
    }
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="controller audit"):
        checkpoint._release(changed, plan)
    audit_path.write_text(json.dumps(audit))
    value["controller_audit_file_sha256"] = checkpoint._hash(audit_path)
    value["controller_audit_sha256"] = audit["sha256"]
    value["sha256"] = digest({key: item for key, item in value.items() if key != "sha256"})
    checkpoint._release(value, plan)

    substituted = {**value, "workload_uid": "00000000-0000-4000-8000-000000000099"}
    substituted["sha256"] = digest(
        {key: item for key, item in substituted.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="controller audit"):
        checkpoint._release(substituted, plan)

    wrong_owner = {
        **audit,
        "workload_owner_rayjob_uid": "00000000-0000-4000-8000-000000000099",
    }
    wrong_owner["sha256"] = digest(
        {key: item for key, item in wrong_owner.items() if key != "sha256"}
    )
    audit_path.write_text(json.dumps(wrong_owner))
    changed = {
        **value,
        "controller_audit_file_sha256": checkpoint._hash(audit_path),
        "controller_audit_sha256": wrong_owner["sha256"],
    }
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="controller audit"):
        checkpoint._release(changed, plan)
    audit_path.write_text(json.dumps(audit))

    for field, invalid in (
        ("kube_context", "wrong-context"),
        ("namespace_uid", "00000000-0000-4000-8000-000000000099"),
    ):
        invalid_audit = {**audit, field: invalid}
        invalid_audit["sha256"] = digest(
            {key: item for key, item in invalid_audit.items() if key != "sha256"}
        )
        audit_path.write_text(json.dumps(invalid_audit))
        invalid_release = {
            **value,
            "controller_audit_file_sha256": checkpoint._hash(audit_path),
            "controller_audit_sha256": invalid_audit["sha256"],
        }
        invalid_release["sha256"] = digest(
            {key: item for key, item in invalid_release.items() if key != "sha256"}
        )
        with pytest.raises(ValueError, match="controller audit"):
            checkpoint._release(invalid_release, plan)

    impossible = copy.deepcopy(audit)
    impossible["pods"][0]["uid"] = impossible["raycluster_uid"]
    impossible["sha256"] = digest(
        {key: item for key, item in impossible.items() if key != "sha256"}
    )
    audit_path.write_text(json.dumps(impossible))
    impossible_release = {
        **value,
        "pod_uids": [impossible["raycluster_uid"]],
        "controller_audit_file_sha256": checkpoint._hash(audit_path),
        "controller_audit_sha256": impossible["sha256"],
    }
    impossible_release["sha256"] = digest(
        {key: item for key, item in impossible_release.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="controller audit|release"):
        checkpoint._release(impossible_release, plan)
    audit_path.write_text(json.dumps(audit))

    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="journal"):
        checkpoint._release(value, plan)


def test_source_plan_binds_exact_reward_canary_data_recipe_and_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checkpoint, "training_request", lambda _: {})
    monkeypatch.setattr(checkpoint, "is_reward_canary", lambda _: True)
    plan = source_plan()
    checkpoint._source_plan(plan)
    for mutate in (
        lambda item: item["arguments"].update(model_root="/mnt/sfs/models/other"),
        lambda item: item["arguments"].update(train_data="/mnt/sfs/data/other/train.jsonl"),
        lambda item: item["data"]["files"]["dev"].update(rows=2),
        lambda item: item["data"].update(selection_sha256="sha256:" + ONE),
        lambda item: item["data"].update(split_sha256="sha256:" + ONE),
        lambda item: item["data"].update(tool_catalog_sha256="sha256:" + ONE),
        lambda item: item["data"]["limits"].update(max_turns=80),
        lambda item: item["arguments"].update(groups=2),
        lambda item: item["arguments"].update(samples_per_prompt=4),
        lambda item: item["arguments"].update(lr=2e-6),
        lambda item: item["arguments"].update(seed=43),
        lambda item: item["arguments"].update(engine_start_timeout_seconds=1799),
        lambda item: item["execution"].pop("reward_canary_source"),
        lambda item: item["execution"].pop("engine_diagnostic_prerequisite"),
        lambda item: item["execution"].update(
            resources={**checkpoint._RESOURCES, "cpu_limit": "63"}
        ),
        lambda item: item.update(run_name="different-source"),
    ):
        changed = copy.deepcopy(plan)
        mutate(changed)
        with pytest.raises(ValueError, match="reward canary|execution shape|data differs"):
            checkpoint._source_plan(changed)


def test_structure_only_manifest_verification_never_reads_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = manifest()
    monkeypatch.setattr(checkpoint, "_source_plan", lambda _: {"source": "request"})
    monkeypatch.setattr(
        checkpoint,
        "_checkpoint_files",
        lambda *_: pytest.fail("structure-only verification read checkpoint storage"),
    )
    checkpoint.verify_manifest(value, check_files=False)
    changed = copy.deepcopy(value)
    changed["sealer"]["uid"] = 0
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="audited candidate"):
        checkpoint.verify_manifest(changed, check_files=False)
    monkeypatch.setattr(checkpoint, "_validate_final_sealer", lambda *_args, **_kwargs: None)

    missing = copy.deepcopy(value)
    del missing["checkpoint"]["files"]["policy/optim_world_size_8_rank_7.pt"]
    missing["checkpoint"]["total_bytes"] -= 1
    missing["sha256"] = digest({key: item for key, item in missing.items() if key != "sha256"})
    with pytest.raises(ValueError, match="topology"):
        checkpoint.verify_manifest(missing, check_files=False)

    changed = copy.deepcopy(value)
    changed["checkpoint"]["ranks"][3]["optimizer_step"] = 0
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="state proof"):
        checkpoint.verify_manifest(changed, check_files=False)

    changed = copy.deepcopy(value)
    changed["checkpoint"]["files"]["data.pt"]["sha256"] = int("1" * 64)
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="inventory"):
        checkpoint.verify_manifest(changed, check_files=False)


def test_checkpoint_manifest_needs_observed_exact_image_and_clean_sealer_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = checkpoint_candidate()
    audit = sealer_audit(candidate)
    monkeypatch.setattr(
        checkpoint,
        "checkpoint_sealer_contract",
        lambda *_: sealer_contract(candidate["source_plan"]),
    )
    candidate_path = Path(checkpoint.SEALER_OUTPUT_ROOT) / "RL_CHECKPOINT_STEP_1_CANDIDATE.json"
    audit_path = Path(checkpoint.SEALER_OUTPUT_ROOT) / "CHECKPOINT_SEALER_CONTROLLER_AUDIT.json"
    checkpoint._validate_sealer_audit(audit, candidate, candidate_path, ZERO)

    spoofed = {**audit, "runtime_image_id": "containerd://sha256:" + ZERO}
    spoofed["sha256"] = digest({key: item for key, item in spoofed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="controller audit"):
        checkpoint._validate_sealer_audit(spoofed, candidate, candidate_path, ZERO)

    external_release = sealer_release(candidate, audit)
    checkpoint._validate_sealer_release(
        external_release,
        candidate,
        candidate_path,
        ZERO,
        audit,
        audit_path,
        ZERO,
    )
    failed = {**external_release, "container_exit_code": 1, "job_complete": False}
    failed["sha256"] = digest({key: item for key, item in failed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="release"):
        checkpoint._validate_sealer_release(
            failed,
            candidate,
            candidate_path,
            ZERO,
            audit,
            audit_path,
            ZERO,
        )


def test_native_checkpoint_layout_and_rank_state_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkpoint"
    for name in checkpoint._required_checkpoint_files() | {
        "policy/huggingface/config.json",
        "policy/huggingface/tokenizer_config.json",
    }:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    (root / "policy/fsdp_config.json").write_text(
        json.dumps({"fsdp_strategy": "fsdp", "world_size": 8})
    )
    files = checkpoint._checkpoint_files(root)
    assert set(files) == checkpoint._required_checkpoint_files() | {
        "policy/huggingface/config.json",
        "policy/huggingface/tokenizer_config.json",
    }

    values = {
        "trainer_state.pt": {"global_step": 1, "config": {"native": True}},
        "data.pt": {"_num_yielded": 1},
    }
    for rank in range(8):
        values[f"optim_world_size_8_rank_{rank}.pt"] = {
            "state": {0: {"step": torch.tensor(1)}},
            "param_groups": [{"params": [0]}],
        }
        values[f"model_world_size_8_rank_{rank}.pt"] = {
            "weight": torch.tensor([rank], dtype=torch.float32)
        }
        values[f"extra_state_world_size_8_rank_{rank}.pt"] = {
            "rank": rank,
            "world_size": 8,
            "fsdp_strategy": "fsdp",
            "lr_scheduler": {"last_epoch": 1},
            "rng": {"torch": torch.tensor([1], dtype=torch.uint8)},
        }
    monkeypatch.setattr(checkpoint, "_native_config_dict", lambda _: {"native": True})
    monkeypatch.setattr(checkpoint, "_torch_load", lambda path: values[path.name])
    state = checkpoint._checkpoint_state(source_plan(), files, 1)
    assert [row["rank"] for row in state["ranks"]] == list(range(8))
    assert all(row["optimizer_step"] == 1 for row in state["ranks"])

    values["extra_state_world_size_8_rank_7.pt"]["lr_scheduler"]["last_epoch"] = 0
    with pytest.raises(ValueError, match="scheduler"):
        checkpoint._checkpoint_state(source_plan(), files, 1)

    (root / "policy/model_world_size_8_rank_7.pt").unlink()
    with pytest.raises(ValueError, match="layout"):
        checkpoint._checkpoint_files(root)


def test_reload_plan_and_request_are_strictly_dev_c1_zero_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / checkpoint.SOURCE_RUN_NAME
    value = manifest()
    monkeypatch.setattr(checkpoint, "SOURCE_OUTPUT_ROOT", str(source_root))
    value["source_plan"]["output_root"] = str(source_root)
    value["source_plan"]["arguments"]["output_root"] = str(source_root)
    value["source_plan_sha256"] = digest(value["source_plan"])
    value["completion"]["path"] = str(source_root / "NATIVE_TRAINING_COMPLETE.json")
    value["completion"]["receipt"]["plan_sha256"] = value["source_plan_sha256"]
    value["completion"]["receipt"] = sealed(
        {key: item for key, item in value["completion"]["receipt"].items() if key != "sha256"}
    )
    value["release"]["receipt"] = release(value["source_plan"])
    value["release"]["path"] = str(source_root / "SOURCE_RELEASE.json")
    value["checkpoint"]["path"] = str(source_root / "checkpoints/global_step_1")
    value["checkpoint"]["latest_pointer"]["path"] = str(
        source_root / "checkpoints/latest_ckpt_global_step.txt"
    )
    value = sealed({key: item for key, item in value.items() if key != "sha256"})
    path = source_root / "RL_CHECKPOINT_STEP_1_MANIFEST.json"
    path.parent.mkdir()
    path.write_text(json.dumps(value))
    source_request = {"source": "request"}
    monkeypatch.setattr(checkpoint, "training_request", lambda _: source_request)
    monkeypatch.setattr(checkpoint, "_source_plan", lambda _: source_request)
    monkeypatch.setattr(checkpoint, "_validate_final_sealer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(checkpoint, "_validate_reward_terminal", lambda *_args: None)
    monkeypatch.setattr(checkpoint, "_canonical_sfs", lambda value, **_: value)
    reward_path = source_root / "REWARD_CANARY_TERMINAL.json"
    reward = sealed(
        {
            "schema": checkpoint.REWARD_TERMINAL_SCHEMA,
            "status": "accepted",
            "source_run_name": checkpoint.SOURCE_RUN_NAME,
            "source_plan_sha256": digest(value["source_plan"]),
            "source_request_sha256": digest(source_request),
            "checkpoint_global_step": 1,
            "checkpoint_manifest_path": str(path),
            "checkpoint_manifest_file_sha256": checkpoint._hash(path),
            "checkpoint_manifest_self_sha256": value["sha256"],
            "source_external_release_path": value["release"]["path"],
            "source_external_release_file_sha256": value["release"]["file_sha256"],
            "source_external_release_self_sha256": value["release"]["receipt"]["sha256"],
            "reward_acquisition_verified": True,
            "within_group_reward_variance_verified": True,
            "optimizer_update_independently_verified": True,
            "wandb_identity_and_finite_scalars_verified": True,
            "source_gpu_release_verified": True,
        }
    )
    reward_path.write_text(json.dumps(reward))
    source = {
        "run_name": checkpoint.SOURCE_RUN_NAME,
        "plan_sha256": digest(value["source_plan"]),
        "request_sha256": digest(source_request),
        "checkpoint_global_step": 1,
        "checkpoint_manifest_path": str(path),
        "checkpoint_manifest_file_sha256": checkpoint._hash(path),
        "checkpoint_manifest_self_sha256": value["sha256"],
        "reward_terminal_receipt_path": str(reward_path),
        "reward_terminal_receipt_file_sha256": checkpoint._hash(reward_path),
        "reward_terminal_receipt_self_sha256": reward["sha256"],
        "source_external_release_path": value["release"]["path"],
        "source_external_release_file_sha256": value["release"]["file_sha256"],
        "source_external_release_self_sha256": value["release"]["receipt"]["sha256"],
    }
    config = {
        "schema": checkpoint.RELOAD_CONFIG_SCHEMA,
        "name": "q38-rl-reload-dev1",
        "output_root": "/mnt/sfs/jobs/q38-rl-reload-dev1",
        "source": source,
        "cluster": {
            "target": "dev",
            "priority": "c1",
            "resources": copy.deepcopy(checkpoint._RESOURCES),
        },
        "lifecycle": copy.deepcopy(checkpoint._LIFECYCLE),
    }
    plan = checkpoint.compile_reload(config, relative_to=tmp_path)
    request = checkpoint.reload_request(plan)
    assert plan["execution"] == {
        "image": checkpoint.IMAGE,
        "cluster_target": "dev",
        "priority": "c1",
        "resources": checkpoint._RESOURCES,
        "workers": 1,
        "gpus_per_worker": 8,
        "requeue_if_preempted": False,
    }
    assert request["workers"] == 1 and request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == []
    assert request["env"]["WANDB_MODE"] == "disabled"
    assert plan["source"] == source
    assert plan["source"]["run_name"] == checkpoint.SOURCE_RUN_NAME
    assert plan["source"]["checkpoint_global_step"] == 1
    assert plan["source"]["reward_terminal_receipt_self_sha256"] == reward["sha256"]

    for mutate in (
        lambda item: item.update(output_root=str(source_root)),
        lambda item: item.update(source_manifest_path=str(tmp_path / "copy.json")),
        lambda item: item.update(source_manifest_file_sha256=ONE),
        lambda item: item["source"].update(plan_sha256=ONE),
    ):
        changed = copy.deepcopy(plan)
        mutate(changed)
        with pytest.raises(ValueError):
            checkpoint.reload_request(changed)

    for field, invalid in (("target", "prod"), ("priority", "c2")):
        changed = copy.deepcopy(config)
        changed["cluster"][field] = invalid
        with pytest.raises(ValueError, match="dev c1"):
            checkpoint.compile_reload(changed, relative_to=tmp_path)

    missing_reward = copy.deepcopy(config)
    missing_reward["source"]["reward_terminal_receipt_path"] = str(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="terminal receipt path"):
        checkpoint.compile_reload(missing_reward, relative_to=tmp_path)

    wrong_step = copy.deepcopy(value)
    wrong_step["checkpoint"]["step"] = 2
    wrong_step["sha256"] = digest(
        {key: item for key, item in wrong_step.items() if key != "sha256"}
    )
    path.write_text(json.dumps(wrong_step))
    with pytest.raises(ValueError, match="step-1"):
        checkpoint._source_binding(source, wrong_step, path, checkpoint._hash(path))


def test_reward_terminal_rejects_legacy_self_attested_booleans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checkpoint, "training_request", lambda _: {"source": "request"})
    legacy = sealed(
        {
            "schema": checkpoint.REWARD_TERMINAL_SCHEMA,
            "status": "accepted",
            "source_run_name": checkpoint.SOURCE_RUN_NAME,
            "source_plan_sha256": digest(source_plan()),
            "source_request_sha256": ZERO,
            "checkpoint_global_step": 1,
            "checkpoint_manifest_path": checkpoint.SOURCE_OUTPUT_ROOT + "/manifest.json",
            "checkpoint_manifest_file_sha256": ZERO,
            "checkpoint_manifest_self_sha256": ZERO,
            "source_external_release_path": checkpoint.SOURCE_OUTPUT_ROOT + "/release.json",
            "source_external_release_file_sha256": ZERO,
            "source_external_release_self_sha256": ZERO,
            "reward_acquisition_verified": True,
            "within_group_reward_variance_verified": True,
            "optimizer_update_independently_verified": True,
            "wandb_identity_and_finite_scalars_verified": True,
            "source_gpu_release_verified": True,
        }
    )
    with pytest.raises(ValueError, match="terminal identity"):
        checkpoint._validate_reward_terminal(
            legacy,
            source_plan(),
            manifest(),
            Path(checkpoint.SOURCE_OUTPUT_ROOT) / "manifest.json",
            ZERO,
        )


def test_reward_reference_reopens_file_and_self_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "receipt.json"
    value = sealed({"status": "observed"})
    path.write_text(json.dumps(value))
    monkeypatch.setattr(checkpoint, "_canonical_sfs", lambda item, **_: item)
    reference = {
        "path": str(path),
        "file_sha256": checkpoint._hash(path),
        "receipt_self_sha256": value["sha256"],
    }
    assert checkpoint._reward_reference(reference, path, None)[0] == value

    changed = {**reference, "receipt_self_sha256": ONE}
    with pytest.raises(ValueError, match="digest"):
        checkpoint._reward_reference(changed, path, None)
    path.write_text(json.dumps(sealed({"status": "substituted"})))
    with pytest.raises(ValueError, match="digest"):
        checkpoint._reward_reference(reference, path, None)


def test_reward_wandb_recomputes_local_remote_scalar_binding(tmp_path: Path) -> None:
    from training.skyrl_training import (
        REWARD_CANARY_WANDB_SCHEMA,
        _finite_scalar_rows,
        _scalar_step_view,
    )

    source = source_plan()
    source["output_root"] = str(tmp_path)
    source["arguments"]["output_root"] = str(tmp_path)
    episode = sealed(
        {
            "schema": "cyber_skyrl_reward_canary_episode_audit_v1",
            "train_reward_mean": 0.25,
            "train_nonzero_count": 2,
            "train_reward_population_variance": 0.125,
        }
    )
    update = sealed(
        {
            "schema": "cyber_skyrl_reward_canary_optimizer_update_v1",
            "final_optimizer_step": 1,
        }
    )
    step_one = {
        "trainer/global_step": 1.0,
        "reward/avg_raw_reward": 0.25,
        "cyber/train/episodes": 8.0,
        "cyber/train/reward_mean": 0.25,
        "cyber/train/reward_nonzero_count": 2.0,
        "cyber/train/reward_population_variance": 0.125,
        "cyber/optimizer_step": 1.0,
        "policy/policy_loss": 0.5,
        "policy/policy_lr": 1e-6,
        "policy/grad_norm": 1.25,
    }
    remote = [
        {"_step": 0.0, "trainer/global_step": 0.0},
        {"_step": 1.0, **step_one},
    ]
    local_raw = [
        {"optimizer_step": 0, "time": 1.0, "trainer/global_step": 0.0},
        {"optimizer_step": 1, "time": 2.0, **step_one},
    ]
    local_path = tmp_path / "metrics.jsonl"
    local_path.write_text("\n".join(json.dumps(row) for row in local_raw) + "\n")
    local = _finite_scalar_rows(local_raw, label="test local")
    value = sealed(
        {
            "schema": REWARD_CANARY_WANDB_SCHEMA,
            "source_plan_sha256": digest(source),
            "identity": {
                "entity": source["arguments"]["wandb_entity"],
                "project": source["arguments"]["wandb_project"],
                "run_id": source["arguments"]["wandb_run_id"],
                "name": source["arguments"]["name"],
            },
            "state": "finished",
            "remote_scalar_history": remote,
            "remote_scalar_history_sha256": "sha256:" + digest(remote),
            "remote_history_rows": 2,
            "local_metrics_path": str(local_path),
            "local_metrics_file_sha256": checkpoint._hash(local_path),
            "local_scalar_history_sha256": "sha256:" + digest(local),
            "local_history_rows": 2,
            "local_remote_step_history_sha256": "sha256:"
            + digest(_scalar_step_view(local, remote=False)),
            "required_step_1_scalars": step_one,
            "episode_audit_sha256": episode["sha256"],
            "optimizer_update_proof_sha256": update["sha256"],
            "logged_artifact_count": 0,
            "rich_payload_count": 0,
        }
    )
    checkpoint._validate_reward_wandb(value, source, episode, update)

    changed = copy.deepcopy(value)
    changed["required_step_1_scalars"]["cyber/train/reward_mean"] = 0.5
    changed = sealed({key: item for key, item in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="W&B evidence"):
        checkpoint._validate_reward_wandb(changed, source, episode, update)


def test_reward_update_requires_eight_reopened_base_ranks_and_policy_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.skyrl_training import (
        REWARD_CANARY_BASE_POLICY_RANK_SCHEMA,
        REWARD_CANARY_UPDATE_PROOF_SCHEMA,
    )

    source = source_plan()
    source["output_root"] = str(tmp_path)
    source["arguments"]["output_root"] = str(tmp_path)
    value = manifest()
    value["source_plan"] = source
    value["source_plan_sha256"] = digest(source)
    value = sealed({key: item for key, item in value.items() if key != "sha256"})
    manifest_path = tmp_path / "RL_CHECKPOINT_STEP_1_MANIFEST.json"
    manifest_path.write_text(json.dumps(value))
    monkeypatch.setattr(checkpoint, "_canonical_sfs", lambda item, **_: item)
    base_refs, ranks = [], []
    for rank, checkpoint_rank in enumerate(value["checkpoint"]["ranks"]):
        base_digest = ZERO if rank == 0 else checkpoint_rank["policy_state_sha256"]
        base = sealed(
            {
                "schema": REWARD_CANARY_BASE_POLICY_RANK_SCHEMA,
                "source_plan_sha256": digest(source),
                "model_root": source["model"]["root"],
                "model_revision": source["model"]["revision"],
                "weight_manifest_sha256": source["model"]["weight_manifest_sha256"],
                "rank": rank,
                "world_size": 8,
                "initial_optimizer_states": 0,
                "policy_state_sha256": base_digest,
            }
        )
        path = tmp_path / "reward_canary_base_policy" / f"rank-{rank}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(base))
        base_refs.append(
            {
                "rank": rank,
                "path": str(path),
                "file_sha256": checkpoint._hash(path),
                "receipt_self_sha256": base["sha256"],
                "policy_state_sha256": base_digest,
            }
        )
        ranks.append(
            {
                **{
                    key: checkpoint_rank[key]
                    for key in (
                        "rank",
                        "policy_state_sha256",
                        "optimizer_state_sha256",
                        "optimizer_states",
                        "optimizer_step_states",
                        "optimizer_step",
                        "scheduler_last_epoch",
                    )
                },
                "base_policy_state_sha256": base_digest,
                "checkpoint_policy_state_sha256": checkpoint_rank["policy_state_sha256"],
                "policy_changed": rank == 0,
            }
        )
    proof = sealed(
        {
            "schema": REWARD_CANARY_UPDATE_PROOF_SCHEMA,
            "source_plan_sha256": digest(source),
            "checkpoint_manifest": {
                "path": str(manifest_path),
                "file_sha256": checkpoint._hash(manifest_path),
                "receipt_self_sha256": value["sha256"],
            },
            "base_policy_ranks": base_refs,
            "initial_optimizer_step": 0,
            "final_optimizer_step": 1,
            "optimizer_updates": 1,
            "changed_optimizer_ranks": 8,
            "delta_basis": "decoded_step1_optimizer_state_after_exact_no_resume_initialization",
            "changed_policy_ranks": 1,
            "policy_delta_basis": (
                "native_rank_state_digest_before_first_optim_step_vs_sealed_step1_checkpoint"
            ),
            "ranks": ranks,
        }
    )
    checkpoint._validate_reward_update(proof, source, value, manifest_path)

    unchanged = copy.deepcopy(proof)
    unchanged["base_policy_ranks"][0]["policy_state_sha256"] = value["checkpoint"]["ranks"][0][
        "policy_state_sha256"
    ]
    unchanged["ranks"][0]["base_policy_state_sha256"] = value["checkpoint"]["ranks"][0][
        "policy_state_sha256"
    ]
    unchanged["ranks"][0]["policy_changed"] = False
    unchanged["changed_policy_ranks"] = 0
    unchanged = sealed({key: item for key, item in unchanged.items() if key != "sha256"})
    with pytest.raises(ValueError, match="rank update|policy delta"):
        checkpoint._validate_reward_update(unchanged, source, value, manifest_path)


def test_rank_receipts_require_all_ranks_and_unchanged_policy_optimizer() -> None:
    replies = rank_replies()
    expected = manifest()["checkpoint"]
    native_sha256 = digest(source_plan()["native_sources"])
    assert checkpoint._validate_rank_replies(replies, expected, native_sha256) == replies
    for transform in (
        lambda rows: rows.pop(),
        lambda rows: rows[2].update(rank=1),
        lambda rows: rows[4].update(optimizer_updates=1),
        lambda rows: rows[0].update(loaded_policy_sha256=ZERO),
        lambda rows: rows[3].update(
            loaded_optimizer_sha256=ZERO, post_forward_optimizer_sha256=ZERO
        ),
        lambda rows: rows[6].update(post_forward_optimizer_sha256=ZERO),
        lambda rows: rows[5].update(rng_state_sha256=int("1" * 64)),
        lambda rows: rows[1].update(runtime_uid=0),
        lambda rows: rows[7].update(native_sources_sha256=ZERO),
    ):
        changed = copy.deepcopy(replies)
        transform(changed)
        with pytest.raises(ValueError):
            checkpoint._validate_rank_replies(changed, expected, native_sha256)

    rejected = copy.deepcopy(replies)
    rejected[7] = {
        "rank": 7,
        "world_size": 8,
        "rejected": "native_state_restore_contract",
        "runtime_uid": 1000,
        "runtime_gid": 100,
        "native_sources_sha256": native_sha256,
    }
    with pytest.raises(checkpoint._ReloadContractRejected):
        checkpoint._validate_rank_replies(rejected, expected, native_sha256)

    rejected[2].pop("forward_tokens")
    with pytest.raises(ValueError) as malformed:
        checkpoint._validate_rank_replies(rejected, expected, native_sha256)
    assert type(malformed.value) is ValueError


def test_initialization_rejection_cannot_mask_a_malformed_peer() -> None:
    native_sha256 = digest(source_plan()["native_sources"])
    initialized = [
        {
            "rank": rank,
            "world_size": 8,
            "base_policy_sha256": ZERO,
            "initial_optimizer_states": 0,
            "runtime_uid": 1000,
            "runtime_gid": 100,
            "native_sources_sha256": native_sha256,
        }
        for rank in range(8)
    ]
    checkpoint._validate_initialization_replies(initialized, native_sha256)
    initialized[7] = {
        "rank": 7,
        "world_size": 8,
        "rejected": "base_initialization_contract",
        "runtime_uid": 1000,
        "runtime_gid": 100,
        "native_sources_sha256": native_sha256,
    }
    with pytest.raises(checkpoint._ReloadContractRejected):
        checkpoint._validate_initialization_replies(initialized, native_sha256)
    initialized[2].pop("base_policy_sha256")
    with pytest.raises(ValueError) as malformed:
        checkpoint._validate_initialization_replies(initialized, native_sha256)
    assert type(malformed.value) is ValueError


def test_nested_state_digest_is_order_stable_and_change_sensitive() -> None:
    left = {2: {"step": torch.tensor(1), "value": torch.tensor([1.0, 2.0])}, 1: None}
    right = {1: None, 2: {"value": torch.tensor([1.0, 2.0]), "step": torch.tensor(1)}}
    assert checkpoint._state_digest(left) == checkpoint._state_digest(right)
    right[2]["value"][1] = 3
    assert checkpoint._state_digest(left) != checkpoint._state_digest(right)
    assert checkpoint._state_digest([[1], 2]) != checkpoint._state_digest([[1, 2]])


def test_source_stat_snapshot_detects_change_without_rehashing_payloads(tmp_path: Path) -> None:
    value = manifest()
    checkpoint_root = tmp_path / "checkpoint"
    for name in value["checkpoint"]["files"]:
        path = checkpoint_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_bytes(b"x")
    for name in ("latest", "completion", "release", "submission", "controller-audit"):
        (tmp_path / name).write_bytes(b"x")
    value["source_plan"]["model"]["root"] = str(model_root)
    value["checkpoint"]["path"] = str(checkpoint_root)
    value["checkpoint"]["latest_pointer"]["path"] = str(tmp_path / "latest")
    value["completion"]["path"] = str(tmp_path / "completion")
    value["release"]["path"] = str(tmp_path / "release")
    value["release"]["receipt"]["submission_journal_path"] = str(tmp_path / "submission")
    value["release"]["receipt"]["controller_audit_path"] = str(tmp_path / "controller-audit")
    before, before_sha = checkpoint._source_stat_snapshot(value)
    target = checkpoint_root / "data.pt"
    target.write_bytes(b"y")
    after, after_sha = checkpoint._source_stat_snapshot(value)
    assert before != after and before_sha != after_sha


def test_recursive_output_evidence_rejects_nested_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "run_name": "q38-rl-reload-dev1",
        "source_manifest": {"sha256": ZERO, "source_plan": source_plan()},
        "source": {"exact": "source"},
    }
    root = tmp_path
    runtime = root / ".runtime"
    runtime.mkdir()
    (runtime / "plan.json").write_text("exact")
    logs = root / "private-native-logs"
    logs.mkdir(mode=0o700)
    infra = logs / "infra.log"
    infra.write_text("")
    infra.chmod(0o600)
    identity = (UUIDS["runtime_run_id"], plan["run_name"] + "-1234abcd")
    request = {"exact": "request"}
    started = sealed(
        {
            "schema": checkpoint.RELOAD_STARTED_SCHEMA,
            "status": "started",
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
            "runtime_run_id": identity[0],
            "rayjob_name": identity[1],
            "checkpoint_manifest_sha256": ZERO,
            "source": plan["source"],
            "driver_runtime_identity": {
                "uid": 1000,
                "gid": 100,
                "native_sources_sha256": digest(source_plan()["native_sources"]),
            },
            "optimizer_updates": 0,
            "rollouts": 0,
        }
    )
    (root / "RELOAD_STARTED.json").write_text(json.dumps(started))
    monkeypatch.setattr(checkpoint, "_reload_runtime_files", lambda _: {"plan.json": "exact"})
    monkeypatch.setattr(checkpoint, "reload_request", lambda _: request)
    checkpoint._output_evidence(plan, root, identity=identity)

    (runtime / "hidden.txt").write_text("forbidden")
    with pytest.raises(ValueError, match="runtime bundle"):
        checkpoint._output_evidence(plan, root, identity=identity)
    (runtime / "hidden.txt").unlink()
    nested = logs / "nested"
    nested.mkdir()
    (nested / "hidden.txt").write_text("forbidden")
    with pytest.raises(ValueError, match="private-log"):
        checkpoint._output_evidence(plan, root, identity=identity)


def test_acceptance_is_create_once_and_requires_later_external_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest = manifest()
    root = tmp_path / "reload-output"
    root.mkdir()
    plan = {
        "schema": checkpoint.RELOAD_SCHEMA,
        "run_name": "q38-rl-reload-dev1",
        "output_root": str(root),
        "source_manifest_file_sha256": ZERO,
        "source_manifest": source_manifest,
        "source": {
            "exact": "source-provenance",
            "reward_terminal_receipt_self_sha256": ZERO,
        },
        "execution": {"image": "registry/skyrl@sha256:" + ZERO},
    }
    runtime_identity = {
        "uid": 1000,
        "gid": 100,
        "native_sources_sha256": digest(source_plan()["native_sources"]),
    }
    result = sealed(
        {
            "schema": checkpoint.RELOAD_RESULT_SCHEMA,
            "status": "reload_validated",
            "plan_sha256": digest(plan),
            "request_sha256": digest({}),
            "runtime_run_id": UUIDS["runtime_run_id"],
            "rayjob_name": plan["run_name"] + "-1234abcd",
            "source_manifest_sha256": source_manifest["sha256"],
            "source_manifest_file_sha256": ZERO,
            "checkpoint_global_step": 1,
            "world_size": 8,
            "ranks": rank_replies(),
            "sampler": {"sampler_restored": True, "sampler_batches_in_epoch": 1},
            "policy_changed_from_base": True,
            "base_model_and_tokenizer_verified": True,
            "source_checkpoint_unchanged": True,
            "optimizer_updates_executed": 0,
            "rollouts_executed": 0,
            "verifier_calls_executed": 0,
            "new_checkpoints_created": 0,
            "wandb_initialized": False,
            "internal_ray_resources_released": True,
            "internal_cleanup": {
                "active_owned_actors": 0,
                "active_owned_placement_groups": 0,
                "tracked_ray_actors": 9,
                "tracked_engine_actors": 0,
                "tracked_placement_groups": 1,
                "tracked_routers": 0,
                "graceful_actor_shutdown": True,
                "cleanup_proven": True,
            },
            "external_job_gpu_release_verified": False,
            "external_release_evidence_required_after_process_exit": True,
            "source": plan["source"],
            "driver_runtime_identity": runtime_identity,
            "completed_at": checkpoint._observed_epoch("2026-09-12T01:00:00Z"),
        }
    )
    result_path = root / "RELOAD_VALIDATED.json"
    result_path.write_text(json.dumps(result))
    release_path = root / "RELOAD_RELEASE.json"
    release_path.write_text(
        json.dumps(
            release(
                plan,
                "2026-09-12T01:01:00Z",
                request_sha256=digest({}),
            )
        )
    )
    monkeypatch.setattr(checkpoint, "reload_request", lambda _: {})
    monkeypatch.setattr(
        checkpoint, "_runtime_identity_proof", lambda _: (runtime_identity, {"native": object()})
    )
    monkeypatch.setattr(
        checkpoint, "_submission", lambda *_: ({"request_sha256": digest({})}, ZERO)
    )
    monkeypatch.setattr(
        checkpoint,
        "_controller_audit",
        lambda _plan, value: (controller_audit(_plan, value), ZERO),
    )
    monkeypatch.setattr(checkpoint, "_output_evidence", lambda *_, **__: None)
    monkeypatch.setattr(checkpoint, "_manifest_file", lambda _: None)
    monkeypatch.setattr(checkpoint, "verify_manifest", lambda *_, **__: None)
    monkeypatch.setattr(checkpoint, "_canonical_sfs", lambda value, **_: value)
    output = root / "RELOAD_ACCEPTED.json"
    accepted = checkpoint.accept_reload(plan, release_path, output)
    assert accepted["sha256"] == digest(
        {key: item for key, item in accepted.items() if key != "sha256"}
    )
    assert accepted["external_job_gpu_release_verified"] is True
    assert accepted["optimizer_updates_executed"] == 0
    assert accepted["verifier_calls_executed"] == 0
    assert accepted["source"] == plan["source"]
    assert (
        checkpoint.validate_reload_accepted(accepted, check_files=False)["source_manifest_sha256"]
        == source_manifest["sha256"]
    )
    with pytest.raises(FileExistsError):
        checkpoint.accept_reload(plan, release_path, output)

    output.unlink()
    for field, invalid in (
        ("source_manifest_file_sha256", ONE),
        ("checkpoint_global_step", 2),
        ("world_size", 7),
        ("source", {"wrong": "source"}),
        ("driver_runtime_identity", {**runtime_identity, "uid": 0}),
    ):
        changed = {**result, field: invalid}
        changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
        result_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="incomplete"):
            checkpoint.accept_reload(plan, release_path, output)
    changed = copy.deepcopy(result)
    changed["internal_cleanup"]["tracked_ray_actors"] = 7
    changed["sha256"] = digest({key: item for key, item in changed.items() if key != "sha256"})
    result_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cleanup"):
        checkpoint.accept_reload(plan, release_path, output)
    result_path.write_text(json.dumps(result))

    release_path.write_text(
        json.dumps(
            release(
                plan,
                "2026-09-12T00:59:00Z",
                request_sha256=digest({}),
            )
        )
    )
    with pytest.raises(ValueError, match="predates"):
        checkpoint.accept_reload(plan, release_path, output)


def test_worker_source_blocks_optimizer_and_uses_native_all_rank_restore() -> None:
    source = implementation("_reload_worker", "_validate_rank_replies")
    assert "load_optimizer_states=True" in source
    assert "load_lr_scheduler_states=True" in source
    assert "torch.inference_mode()" in source
    assert "def optim_step" in source
    assert "zero-update reload must never call the optimizer" in source
    assert "post_forward_optimizer_sha256" in source
    assert "post_forward_policy_sha256" in source
    assert "_runtime_identity_proof" in source
    assert "runtime_uid" in source and "runtime_gid" in source


def test_source_drift_is_an_unexpected_failure_not_clean_rejection() -> None:
    source = implementation("_run_reload", "accept_reload")
    assert 'RuntimeError("source changed before reload")' in source
    assert 'RuntimeError("source changed during reload")' in source
    assert '_ReloadContractRejected("source_changed_during_reload")' not in source
    with pytest.raises(ValueError, match="unknown"):
        checkpoint._ReloadContractRejected("source_changed_during_reload")


def test_recognized_reload_rejection_exits_zero_without_failed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = {"run_name": "q38-rl-reload-dev1", "source": {"exact": "source"}}
    root = tmp_path
    monkeypatch.setattr(
        sys, "argv", ["reload", "--plan", "plan.json", "--sha256", digest(plan), "--run"]
    )
    monkeypatch.setattr(checkpoint, "_json", lambda _: plan)
    monkeypatch.setattr(checkpoint, "_owned_output_root", lambda _: root)
    monkeypatch.setattr(
        checkpoint,
        "_run_reload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            checkpoint._ReloadContractRejected("native_state_restore_contract")
        ),
    )
    monkeypatch.setattr(checkpoint, "_output_evidence", lambda *_, **__: None)
    monkeypatch.setattr(checkpoint, "reload_request", lambda _: {"exact": "request"})
    checkpoint.main()
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"
    rejected = json.loads((root / "RELOAD_REJECTED.json").read_text())
    assert rejected["source"] == plan["source"]
    assert not (root / "FAILED.json").exists()


def test_checkpoint_candidate_rejects_wrong_runtime_user_before_trusted_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint.os, "geteuid", lambda: 501)
    monkeypatch.setattr(checkpoint.os, "getegid", lambda: 20)
    with pytest.raises(ValueError, match="1000:100"):
        checkpoint.checkpoint_candidate(
            {}, 1, tmp_path / "candidate.json", tmp_path / "release.json"
        )
    source = implementation("checkpoint_candidate", "_bound_receipt")
    assert source.index("native_source()") < source.index("_checkpoint_state(")
    assert source.index("check_inputs(plan)") < source.index("_checkpoint_state(")


def test_acceptance_rejects_wrong_runtime_user_before_native_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint.os, "geteuid", lambda: 501)
    monkeypatch.setattr(checkpoint.os, "getegid", lambda: 20)
    with pytest.raises(ValueError, match="1000:100"):
        checkpoint.accept_reload(
            {"source_manifest": {"source_plan": source_plan()}},
            tmp_path / "release.json",
            tmp_path / "accepted.json",
        )
    source = implementation("accept_reload", "_record_failure")
    assert source.index("_runtime_identity_proof(") < source.index("reload_request(plan)")
    assert source.index("reload_request(plan)") < source.index("verify_manifest(")


@pytest.mark.parametrize("command", ["preview", "submit"])
def test_cli_never_routes_rl_reload_to_production(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"schema": checkpoint.RELOAD_SCHEMA, "execution": {"cluster_target": "dev"}}
    request = {"request": "not reached"}
    monkeypatch.setattr(cli, "_prepared", lambda _: (plan, request))
    monkeypatch.setattr(cli, "_client", lambda _: pytest.fail("reached cluster API"))
    monkeypatch.setattr(cli, "_read", lambda _: pytest.fail("read proof before cluster gate"))
    result = RUNNER.invoke(cli.app, [command, str(tmp_path), "--cluster", "prod"])
    assert result.exit_code == 2
    assert "dev-cluster-only" in result.stderr


def test_reload_preview_requires_exact_nonroot_runtime_user() -> None:
    request = {
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "requeueIfPreempted": False,
    }
    manifest = {
        "metadata": {"namespace": checkpoint.NAMESPACE},
        "spec": {
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "spec": {
                            "securityContext": {
                                "runAsUser": 1000,
                                "runAsGroup": 100,
                                "runAsNonRoot": True,
                            },
                            "containers": [{}],
                        }
                    }
                },
                "workerGroupSpecs": [],
            }
        },
    }
    preview = {"manifest_yaml": yaml.safe_dump(manifest)}
    assert checkpoint.validate_reload_preview(request, preview)["runtime_user"] == {
        "uid": 1000,
        "gid": 100,
    }
    manifest["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["securityContext"][
        "runAsUser"
    ] = 0
    with pytest.raises(checkpoint.JobsError, match="1000:100"):
        checkpoint.validate_reload_preview(request, {"manifest_yaml": yaml.safe_dump(manifest)})


def test_reload_submit_repreviews_security_context_before_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "schema": checkpoint.RELOAD_SCHEMA,
        "execution": {"cluster_target": "dev"},
    }
    request = {"exact": "request"}
    proof = sealed(
        {
            "schema": checkpoint.PREFLIGHT_SCHEMA,
            "status": "passed",
            "gpus": 0,
            "plan_sha256": digest(plan),
            "request_sha256": digest(request),
        }
    )
    calls: list[str] = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def preview(self, value):
            assert value == request
            calls.append("preview")
            return {"manifest_yaml": "exact"}

        def submit_once(self, value, journal):
            assert value == request and journal == tmp_path / "SUBMISSION.jsonl"
            calls.append("submit")
            return {"submitted": True}

    monkeypatch.setattr(cli, "_prepared", lambda _: (plan, request))
    monkeypatch.setattr(cli, "_read", lambda _: proof)
    monkeypatch.setattr(cli, "_client", lambda _: Client())
    monkeypatch.setattr(cli, "validate_preview", lambda *_: {})
    monkeypatch.setattr(checkpoint, "validate_reload_preview", lambda *_: calls.append("security"))
    result = RUNNER.invoke(cli.app, ["submit", str(tmp_path), "--cluster", "dev"])
    assert result.exit_code == 0
    assert calls == ["preview", "security", "submit"]
