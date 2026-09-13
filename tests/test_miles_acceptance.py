"""Synthetic terminal evidence only; no Fleet, W&B, Kubernetes, or GPU calls."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import shlex
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import API_URLS, digest
from evals.fleet import opencode_self_hosted as fleet
from training import miles
from training import miles_acceptance as acceptance
from training.miles_conversion import _hash


def _seal(value: dict) -> dict:
    value = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = "sha256:" + digest(value)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fleet.canonical_json(value))


def _source_config(run_name: str, split: str, prompt: str, catalog_sha: str) -> dict:
    task_version = str(uuid.uuid5(uuid.NAMESPACE_URL, f"task-{split}"))
    verifier_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"verifier-{split}"))
    verifier_version = str(uuid.uuid5(uuid.NAMESPACE_URL, f"verifier-version-{split}"))
    prefix = "/v1/rollout-rewards/{task_key}/versions/{task_version_id}"
    value = {
        "run_id": run_name,
        "model": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "synthetic-revision",
            "root": "/mnt/sfs/models/synthetic-qwen",
            "tito_family": "qwen35",
            "runtime_chat_template_sha256": "sha256:" + "1" * 64,
        },
        "authority": {
            "provisioning_route_template": prefix + "/instances",
            "scoring_route_template": prefix,
            "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
            "scoring_mode": "full",
            "multi_app_aggregation_mode": "binary",
            "required_cyber_contract": {"verifier_contract": "v3"},
        },
        "execution": {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": catalog_sha,
        },
        "environment": {
            "id": "synthetic-env",
            "version": "v1",
            "version_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"environment-{split}")),
            "data_id": "synthetic-data",
            "data_version": "v1",
            "runtime_seed_content_sha256": "sha256:" + "2" * 64,
            "ttl_seconds": 32400,
        },
        "task": {
            "key": f"synthetic-{split}",
            "version_id": task_version,
            "prompt_sha256": fleet.sha256(prompt.encode()),
            "env_variables_sha256": fleet.sha256(fleet.canonical_json({})),
            "output_json_schema_sha256": fleet.sha256(fleet.canonical_json(None)),
            "cyber_contract": {"verifier_contract": "v3"},
        },
        "verifier": {
            "id": verifier_id,
            "version_id": verifier_version,
            "version": "v1",
            "sha256": "sha256:" + ("3" if split == "train" else "4") * 64,
            "function_name": "verify",
        },
        "rl": {
            "max_turns": 4,
            "episode_seconds": 60,
            "tool_seconds": 5,
            "tool_result_chars": 1000,
            "max_tokens_per_turn": 4096,
            "context_tokens": 32768,
        },
        "initial_prompt_sha256": fleet.sha256(prompt.encode()),
    }
    value["config_sha256"] = fleet.digest_without(value, "config_sha256")
    return value


def _episode_config(source: dict, kind: str, sample: int) -> dict:
    value = copy.deepcopy(source)
    value["run_id"] += f"-{kind}-r0-s{sample}"
    value["native_batch"] = {"kind": kind, "rollout_id": 0}
    value["sampling"] = {"temperature": 1.0, "top_p": 1.0}
    value["config_sha256"] = fleet.digest_without(value, "config_sha256")
    return value


def _attestation(config: dict, instance_id: str, evidence_id: str, execution_id: str, score: float):
    return {
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
        "reward": score,
        "verifier_execution_id": execution_id,
        "direct_authority_attestation": {
            "schema_version": fleet.DIRECT_AUTHORITY_ATTESTATION_SCHEMA,
            "context": {
                "task_key": config["task"]["key"],
                "task_version_id": config["task"]["version_id"],
                "instance_id": instance_id,
                "evidence_run_id": evidence_id,
                "verifier_version_id": config["verifier"]["version_id"],
                "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
            },
            "activity": {
                "result_schema_version": "cyber_verification_result_v3",
                "reward": score,
                "task_version_id": config["task"]["version_id"],
                "verifier_execution_id": execution_id,
            },
            "shadow": {
                "mode": "authoritative",
                "status": "authoritative",
                "match": True,
                "production_execution_id": execution_id,
                "direct_verifier": {
                    "status": "authoritative",
                    "match": True,
                    "execution_id": execution_id,
                    "verifier_contract_version": "v3",
                    "context_schema_version": "cyber_verification_context_v1",
                },
            },
            "data_minimization": {
                "components_included": False,
                "diagnostics_included": False,
                "evidence_payloads_included": False,
                "prompts_included": False,
                "traces_included": False,
                "flags_included": False,
            },
        },
    }


def _write_episode(root: Path, source: dict, kind: str, sample: int, score: float) -> Path:
    config = _episode_config(source, kind, sample)
    path = root / "episodes" / config["run_id"]
    path.mkdir(parents=True)
    instance_id = f"instance-{kind}-{sample}"
    evidence_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"evidence-{kind}-{sample}"))
    execution_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"execution-{kind}-{sample}"))
    values = {
        "binding.json": config,
        "create-intent.json": {"run_id": config["run_id"]},
        "instance.json": {"instance_id": instance_id, "evidence_run_id": evidence_id},
        "conversation.json": {
            "messages": [
                {"role": "user", "content": "synthetic"},
                {"role": "assistant", "content": "synthetic"},
                {"role": "tool", "name": "submit_report", "content": "synthetic"},
            ]
        },
        "score-intent.json": {
            "instance_id": instance_id,
            "scoring_mode": "full",
            "multi_app_aggregation_mode": "binary",
        },
        "reward.json": _attestation(config, instance_id, evidence_id, execution_id, score),
        "cleanup.json": {
            "create_attempted": True,
            "instance_created": True,
            "instance_id": instance_id,
            "instance_closed": True,
            "possible_instance_leak": False,
        },
        "recording.json": {
            "samples": [
                {
                    "tokens": [1, 2, 3],
                    "response_length": 2,
                    "loss_mask": [1, 0],
                    "rollout_log_probs": [-0.2, 0.0],
                }
            ]
        },
    }
    for name, value in values.items():
        _write_json(path / name, value)
    accepted = {
        "task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
        "verifier_execution_id": execution_id,
        "done_reason": "report_submitted",
        "config_sha256": config["config_sha256"],
        "sample_count": 1,
        "files": {name: fleet.sha256((path / name).read_bytes()) for name in values},
    }
    _write_json(path / "ACCEPTED.json", _seal(accepted))
    return path


def _replace_episode_file(path: Path, name: str, value: dict) -> None:
    _write_json(path / name, value)
    accepted = json.loads((path / "ACCEPTED.json").read_bytes())
    accepted["files"][name] = fleet.sha256((path / name).read_bytes())
    _write_json(path / "ACCEPTED.json", _seal(accepted))


@pytest.fixture
def case(tmp_path: Path) -> dict:
    run_name = "synthetic-miles-reward-canary"
    root = tmp_path / "run"
    root.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    catalog_sha = "sha256:" + "5" * 64
    prompts = {"train": "synthetic training task", "dev": "synthetic dev task"}
    configs = {
        split: _source_config(run_name, split, prompt, catalog_sha)
        for split, prompt in prompts.items()
    }
    files = {}
    for split in ("train", "dev"):
        row = {
            "input": prompts[split],
            "metadata": {"split": split, "cyber_config": configs[split]},
        }
        payload = fleet.canonical_json(row) + b"\n"
        path = data_root / f"{split}.jsonl"
        path.write_bytes(payload)
        files[split] = {
            "path": path.name,
            "rows": 1,
            "sha256": fleet.sha256(payload),
        }
    limits = {**configs["train"]["rl"], "response_tokens": 8192}
    data = _seal(
        {
            "schema": "cyber_miles_data_v1",
            "name": run_name,
            "selection_sha256": "sha256:" + "6" * 64,
            "split_sha256": "sha256:" + "7" * 64,
            "tokenizer": {"repo": "Qwen/Qwen3.8-27B", "revision": "synthetic-revision"},
            "template_sha256": "sha256:" + "1" * 64,
            "tool_catalog_sha256": catalog_sha,
            "limits": limits,
            "files": files,
            "gpus": 0,
            "environment_creates": 0,
        }
    )
    manifest_path = data_root / "manifest.json"
    _write_json(manifest_path, data)
    arguments = {
        "name": run_name,
        "output_root": str(root),
        "model_root": "/mnt/sfs/models/synthetic-qwen",
        "torch_dist_root": "/mnt/sfs/jobs/synthetic-base/checkpoint",
        "train_data": str(data_root / "train.jsonl"),
        "dev_data": str(data_root / "dev.jsonl"),
        "data_manifest": str(manifest_path),
        "wandb_entity": "synthetic",
        "wandb_project": "synthetic",
        "wandb_run_id": run_name,
        "model": "Qwen/Qwen3.8-27B",
        "nodes": 1,
        "gpus_per_node": 8,
        "steps": 1,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 1e-6,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "seed": 42,
        "context_tokens": 32768,
        "response_tokens": 8192,
        "tokens_per_turn": 4096,
    }
    model = {
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "synthetic-revision",
        "root": arguments["model_root"],
    }
    base_checkpoint = _seal(
        {
            "schema": "cyber_miles_checkpoint_v1",
            "image": miles.IMAGE,
            "root": "/mnt/sfs/jobs/synthetic-base/checkpoint",
            "model": model,
            "files": [{"path": "base", "size": 1, "sha256": "a" * 64}],
            "optimizer_steps": 0,
        }
    )
    plan = {
        "schema": "cyber_miles_training_v1",
        "run_name": run_name,
        "output_root": str(root),
        "model": model,
        "data": data,
        "checkpoint": base_checkpoint,
        "arguments": arguments,
        "runtime_sha256": "8" * 64,
        "native_driver_sha256": "9" * 64,
        "execution": {
            "image": miles.IMAGE,
            "priority": "c1",
            "resources": {
                "cpu_request": "64",
                "cpu_limit": "128",
                "memory_request": "1536Gi",
                "memory_limit": "2048Gi",
            },
        },
    }
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_files = {
        path: subprocess.check_output(["git", "show", f"{source_commit}:{path}"], text=True)
        for path in acceptance._SUBMITTED_RUNTIME_FILES
    }
    plan["runtime_sha256"] = digest(source_files)
    _write_json(root / "plan.json", plan)
    bundle_files = {
        **source_files,
        "training/__init__.py": "",
        "evals/__init__.py": "",
        "evals/fleet/__init__.py": "",
        "cyber_post_train/__init__.py": "",
        "plan.json": json.dumps(plan, sort_keys=True, separators=(",", ":")),
    }
    payload = json.dumps(
        {
            "files": bundle_files,
            "module": "training.miles_training",
            "argv": ["--plan", "plan.json", "--sha256", digest(plan)],
        },
        sort_keys=True,
    ).encode()
    blob = gzip.compress(payload, mtime=0)
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    request = {
        "name": run_name,
        "title": run_name + " native Miles RL",
        "run_dir": str(root),
        "image": miles.IMAGE,
        "workers": 1,
        "gpus_per_worker": 8,
        "resources": plan["execution"]["resources"],
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "secrets": ["fleet-api", "wandb-api"],
        "env": {
            **acceptance._SUBMITTED_ENV,
            "WANDB_RUN_ID": run_name,
            "CYBER_RUNTIME_BUNDLE": base64.b64encode(blob).decode(),
        },
        "command": "python -c " + shlex.quote("assert " + repr(blob_sha256)),
    }
    request_path = tmp_path / "prepared/request.json"
    _write_json(request_path, request)
    submission_path = tmp_path / "evidence/SUBMITTED_EXECUTION.json"
    submission_path.parent.mkdir()
    submission = acceptance.compile_submission_binding(
        plan_path=root / "plan.json",
        request_path=request_path,
        source_commit=source_commit,
        api_run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        submitted_at="2026-09-13T03:06:15Z",
        output=submission_path,
    )
    completion = _seal(
        {
            "status": "native_loop_returned",
            "plan_sha256": digest(plan),
            "checkpoint_rollout_index": 0,
            "completed_batches": 3,
            "completed_at": 1000.0,
            "optimizer_update_independently_verified": False,
            "checkpoint_reload_verified": False,
        }
    )
    _write_json(root / "NATIVE_TRAINING_COMPLETE.json", completion)
    batches = {
        "dev-baseline-r0": ("dev-baseline", [0]),
        "train-r0": ("train", list(range(8))),
        "dev-after-r0": ("dev-after", [0]),
    }
    for name, (kind, indices) in batches.items():
        intent = {
            "schema": "cyber_miles_batch_v1",
            "batch_id": name,
            "data_sha256": data["sha256"],
            "episode_indices": indices,
            "native_rollout_id": 0,
            "evaluation": kind != "train",
            "optimizer_step_verified": False,
        }
        directory = root / "episodes/batches" / name
        _write_json(directory / "STARTED.json", intent)
        _write_json(directory / "COLLECTED.json", _seal(intent))
    episodes = []
    episodes.append(_write_episode(root, configs["dev"], "dev-baseline", 0, 0.0))
    for index in range(8):
        episodes.append(_write_episode(root, configs["train"], "train", index, float(index % 2)))
    episodes.append(_write_episode(root, configs["dev"], "dev-after", 0, 1.0))

    completion_sha = completion["sha256"].removeprefix("sha256:")
    checkpoint_root = root / "checkpoints"
    generation = checkpoint_root / "iter_0000000"
    generation.mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("0\n")
    (generation / ".metadata").write_bytes(b"synthetic metadata")
    (generation / "common.pt").write_bytes(b"synthetic common state")
    for rank in range(8):
        (generation / f"__{rank}_0.distcp").write_bytes(f"rank-{rank}-trained".encode())
    checkpoint_files = [
        {
            "path": str(path.relative_to(checkpoint_root)),
            "size": path.stat().st_size,
            "sha256": _hash(path),
        }
        for path in sorted(checkpoint_root.rglob("*"))
        if path.is_file()
    ]
    checkpoint = _seal(
        {
            "schema": "cyber_miles_training_checkpoint_v1",
            "image": miles.IMAGE,
            "root": str(checkpoint_root),
            "rollout_index": 0,
            "next_rollout_id": 1,
            "world_size": 8,
            "topology": {"nodes": 1, "gpus_per_node": 8},
            "model": model,
            "source": {
                "run_name": run_name,
                "output_root": str(root),
                "plan_sha256": digest(plan),
                "completion_sha256": completion_sha,
                "arguments": arguments,
                "execution": plan["execution"],
                "native_driver_sha256": plan["native_driver_sha256"],
            },
            "files": checkpoint_files,
            "source_optimizer_update_claimed": False,
            "gpu_reload_verified": False,
            "optimizer_update_during_reload": False,
        }
    )
    checkpoint_path = tmp_path / "evidence/MILES_TRAINING_CHECKPOINT.json"
    _write_json(checkpoint_path, checkpoint)

    wandb = _seal(
        {
            "schema": acceptance.WANDB_SCHEMA,
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_request_sha256": "sha256:" + digest(request),
            "identity": {
                "entity": "synthetic",
                "project": "synthetic",
                "run_id": run_name,
                "name": run_name,
            },
            "state": "finished",
            "history_rows": 2,
            "remote_non_sensitive_metric_schema": [
                ["train/grad_norm", "train/loss", "train/lr-pg_0", "train/step"],
                [],
            ],
            "remote_metric_schema_sha256": "sha256:"
            + digest(
                [
                    ["train/grad_norm", "train/loss", "train/lr-pg_0", "train/step"],
                    [],
                ]
            ),
            "observed_train_steps": [0],
            "optimizer_update_count": 1,
            "update_zero_metric_names": [
                "train/grad_norm",
                "train/loss",
                "train/lr-pg_0",
                "train/step",
            ],
            "positive_finite_gradient_observed": True,
            "finite_train_loss_observed": True,
            "learning_rate_matches_plan": True,
            "sensitive_metric_values_redacted": True,
            "logged_artifact_count": 0,
            "rich_payload_count": 0,
            "reward_values_included": False,
        }
    )
    wandb_path = tmp_path / "evidence/WANDB.json"
    _write_json(wandb_path, wandb)

    policy_delta = _seal(
        {
            "schema": acceptance.POLICY_DELTA_SCHEMA,
            "source_plan_sha256": "sha256:" + digest(plan),
            "base_checkpoint_receipt_sha256": base_checkpoint["sha256"],
            "trained_checkpoint_receipt_sha256": checkpoint["sha256"],
            "world_size": 8,
            "comparison_method": "all_rank_named_policy_tensor_value_sha256_v1",
            "ranks": [
                {
                    "rank": rank,
                    "policy_tensor_count": 10,
                    "local_policy_numel": 100,
                    "base_policy_structure_sha256": "sha256:" + f"{rank + 1:064x}",
                    "trained_policy_structure_sha256": "sha256:" + f"{rank + 1:064x}",
                    "base_policy_value_sha256": "sha256:" + f"{rank + 20:064x}",
                    "trained_policy_value_sha256": "sha256:"
                    + f"{rank + (40 if rank == 0 else 20):064x}",
                    "trained_optimizer_value_sha256": "sha256:" + f"{rank + 60:064x}",
                    "trained_scheduler_value_sha256": "sha256:" + f"{rank + 80:064x}",
                    "trained_rng_value_sha256": "sha256:" + f"{rank + 100:064x}",
                    "policy_changed": rank == 0,
                }
                for rank in range(8)
            ],
            "changed_policy_ranks": [0],
            "policy_structure_matches": True,
            "optimizer_state_used_for_delta": False,
            "scheduler_state_used_for_delta": False,
            "rng_state_used_for_delta": False,
            "metadata_used_for_delta": False,
            "checkpoint_state_commitment_method": acceptance.RANK_STATE_COMMITMENT_METHOD,
            "reward_values_included": False,
        }
    )
    policy_delta_path = tmp_path / "evidence/POLICY_DELTA.json"
    _write_json(policy_delta_path, policy_delta)

    run_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    api_name = run_name + "-aaaaaaaa"
    rayjob_uid = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    workload_uid = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    raycluster_uid = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    pod_uid = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
    controller = _seal(
        {
            "schema": acceptance.CONTROLLER_SCHEMA,
            "status": "succeeded",
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_request_sha256": "sha256:" + digest(request),
            "api": {
                "base_url": API_URLS["dev"],
                "run_id": run_id,
                "run_name": api_name,
                "status": "SUCCEEDED",
            },
            "kubernetes": {
                "cluster": "dev",
                "namespace": acceptance.NAMESPACE,
                "namespace_uid": acceptance.NAMESPACE_UID,
                "rayjob": {"name": api_name, "uid": rayjob_uid, "status": "SUCCEEDED"},
                "workload": {
                    "name": "workload-synthetic",
                    "uid": workload_uid,
                    "owner_rayjob_uid": rayjob_uid,
                },
                "raycluster": {
                    "name": "raycluster-synthetic",
                    "uid": raycluster_uid,
                    "owner_rayjob_uid": rayjob_uid,
                },
                "pods": [
                    {
                        "name": "worker-synthetic",
                        "uid": pod_uid,
                        "owner_raycluster_uid": raycluster_uid,
                        "phase": "Succeeded",
                        "exit_code": 0,
                        "termination_reason": "Completed",
                        "runtime_image_id": "containerd://image@sha256:"
                        + miles.IMAGE.rsplit("@sha256:", 1)[1],
                        "container_restarts": 0,
                        "gpus": 8,
                    }
                ],
            },
            "execution": {
                "requested_image": miles.IMAGE,
                "priority_class": "c1",
                "effective_priority": 10000,
                "automatic_requeue": False,
                "workers": 1,
                "gpus_per_worker": 8,
                "total_gpus": 8,
            },
            "observed_at": 1010.0,
        }
    )
    controller_path = tmp_path / "evidence/CONTROLLER.json"
    _write_json(controller_path, controller)
    release = _seal(
        {
            "schema": acceptance.RELEASE_SCHEMA,
            "status": "released",
            "source_plan_sha256": "sha256:" + digest(plan),
            "source_request_sha256": "sha256:" + digest(request),
            "controller_observation_sha256": controller["sha256"],
            "controller_observation_file_sha256": "sha256:" + _hash(controller_path),
            "api_status": "SUCCEEDED",
            "controller_status": "SUCCEEDED",
            "identities": {
                "api_run_id": run_id,
                "api_run_name": api_name,
                "rayjob_uid": rayjob_uid,
                "workload_uid": workload_uid,
                "raycluster_uid": raycluster_uid,
                "pod_uids": [pod_uid],
            },
            "raycluster_present": False,
            "rayjob_present": False,
            "workload_present": False,
            "quota_reservation_present": False,
            "gpu_pods_present": False,
            "active_gpus": 0,
            "gpu_release_proven": True,
            "observed_at": 1020.0,
        }
    )
    release_path = tmp_path / "evidence/RELEASE.json"
    _write_json(release_path, release)
    return {
        "plan": plan,
        "request": request,
        "submission": submission,
        "submission_path": submission_path,
        "root": root,
        "episodes": episodes,
        "checkpoint": checkpoint_path,
        "wandb": wandb_path,
        "policy_delta": policy_delta_path,
        "controller": controller_path,
        "release": release_path,
    }


def _accept(case: dict) -> dict:
    return acceptance.accept_terminal(
        case["plan"],
        submission_binding_path=case["submission_path"],
        checkpoint_manifest_path=case["checkpoint"],
        policy_delta_observation_path=case["policy_delta"],
        controller_observation_path=case["controller"],
        release_observation_path=case["release"],
        wandb_observation_path=case["wandb"],
        output=case["root"] / "MILES_TERMINAL_ACCEPTED.json",
    )


def test_terminal_acceptance_reopens_every_gate_without_reward_values(case: dict) -> None:
    result = _accept(case)

    assert result["status"] == "accepted"
    assert result["optimizer_update_proof"]["optimizer_updates"] == 1
    assert result["episode_audit"]["train_within_group_reward_variance_present"] is True
    assert result["episode_audit"]["unique_challenge_instance_count"] == 10
    assert result["episode_audit"]["unique_evidence_run_count"] == 10
    assert result["reward_values_included"] is False
    assert result["external_gpu_release_verified"] is True
    assert "synthetic training task" not in json.dumps(result)
    assert (
        acceptance.validate_terminal(result, check_files=True)["terminal_receipt_sha256"]
        == result["sha256"]
    )
    with pytest.raises(FileExistsError):
        _accept(case)


@pytest.mark.parametrize(
    "fault",
    [
        "plan",
        "submission",
        "completion",
        "conflict",
        "interaction",
        "reward_variance",
        "wandb",
        "checkpoint",
        "policy_delta",
        "controller",
        "release",
    ],
)
def test_terminal_acceptance_fails_closed_on_each_scientific_or_release_gate(
    case: dict, fault: str
) -> None:
    if fault == "plan":
        case["plan"]["arguments"]["steps"] = 2
    elif fault == "submission":
        value = json.loads(Path(case["submission"]["source_request_path"]).read_bytes())
        value["requeueIfPreempted"] = True
        _write_json(Path(case["submission"]["source_request_path"]), value)
    elif fault == "completion":
        value = json.loads((case["root"] / "NATIVE_TRAINING_COMPLETE.json").read_bytes())
        value["completed_batches"] = 2
        _write_json(case["root"] / "NATIVE_TRAINING_COMPLETE.json", _seal(value))
    elif fault == "conflict":
        _write_json(case["root"] / "FAILED.json", {})
    elif fault == "interaction":
        path = case["episodes"][1]
        value = json.loads((path / "conversation.json").read_bytes())
        value["messages"] = [item for item in value["messages"] if item["role"] != "tool"]
        _replace_episode_file(path, "conversation.json", value)
    elif fault == "reward_variance":
        for path in case["episodes"][1:9]:
            value = json.loads((path / "reward.json").read_bytes())
            value["reward"] = 0.0
            value["direct_authority_attestation"]["activity"]["reward"] = 0.0
            _replace_episode_file(path, "reward.json", value)
    elif fault == "wandb":
        value = json.loads(case["wandb"].read_bytes())
        value["positive_finite_gradient_observed"] = False
        _write_json(case["wandb"], _seal(value))
    elif fault == "checkpoint":
        (case["root"] / "checkpoints/iter_0000000/__7_0.distcp").write_bytes(b"tampered")
    elif fault == "policy_delta":
        value = json.loads(case["policy_delta"].read_bytes())
        for row in value["ranks"]:
            row["trained_policy_value_sha256"] = row["base_policy_value_sha256"]
            row["policy_changed"] = False
        value["changed_policy_ranks"] = []
        _write_json(case["policy_delta"], _seal(value))
    elif fault == "controller":
        value = json.loads(case["controller"].read_bytes())
        value["kubernetes"]["pods"][0]["container_restarts"] = 1
        _write_json(case["controller"], _seal(value))
    else:
        value = json.loads(case["release"].read_bytes())
        value["rayjob_present"] = True
        _write_json(case["release"], _seal(value))
    with pytest.raises((ValueError, KeyError)):
        _accept(case)


def test_wandb_observer_validates_remote_identity_and_omits_reward_values(
    case: dict,
) -> None:
    plan = case["plan"]
    history = [
        {
            "_step": 0,
            "train/step": 0.0,
            "train/grad_norm": 1.25,
            "train/loss": 0.5,
            "train/lr-pg_0": 1e-6,
            "rollout/episode_raw_reward": 0.731,
            "eval/heldout-cyber": 0.42,
        },
    ]
    run = SimpleNamespace(
        entity="synthetic",
        project="synthetic",
        id=plan["run_name"],
        name=plan["run_name"],
        state="finished",
        summary={"train/step": 0.0, "rollout/episode_raw_reward": 0.731},
        logged_artifacts=lambda: [],
        files=lambda: [],
        scan_history=lambda: copy.deepcopy(history),
    )
    api = SimpleNamespace(run=lambda path: run)

    result = acceptance.observe_wandb(plan, case["submission"], api=api)

    assert result["observed_train_steps"] == [0]
    assert result["optimizer_update_count"] == 1
    assert result["update_zero_metric_names"] == sorted(acceptance._MILES_UPDATE_METRICS)
    assert "0.731" not in json.dumps(result)
    assert "0.42" not in json.dumps(result)
    assert "episode_raw_reward" not in json.dumps(result)
    first = copy.deepcopy(result)
    history[0]["rollout/episode_raw_reward"] = 0.123
    history[0]["eval/heldout-cyber"] = 0.99
    run.summary = {"train/step": 0.0, "rollout/episode_raw_reward": 0.123}
    assert acceptance.observe_wandb(plan, case["submission"], api=api) == first
    assert result["reward_values_included"] is False
    history.append(
        {
            "_step": 1,
            "train/step": 1.0,
            "train/grad_norm": 1.0,
            "train/loss": 0.4,
            "train/lr-pg_0": 1e-6,
        }
    )
    with pytest.raises(ValueError, match="exactly one optimizer"):
        acceptance.observe_wandb(plan, case["submission"], api=api)


def test_historical_acceptance_never_reconstructs_request_with_current_code(
    case: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        acceptance,
        "job_request",
        lambda plan: (_ for _ in ()).throw(AssertionError("current compiler must not run")),
    )

    result = _accept(case)

    assert result["source_request_sha256"] == case["submission"]["source_request_sha256"]
    assert result["submission_binding"]["receipt_sha256"].removeprefix("sha256:") == case[
        "submission"
    ]["sha256"].removeprefix("sha256:")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.pop("train/loss"), "telemetry"),
        (lambda row: row.__setitem__("train/grad_norm", 0.0), "telemetry"),
        (lambda row: row.__setitem__("train/lr-pg_0", 2e-6), "telemetry"),
        (lambda row: row.__setitem__("train/step", 1.0), "exactly one optimizer"),
        (
            lambda row: (row.pop("train/step"), row.__setitem__("trainer/global_step", 1.0)),
            "exactly one optimizer",
        ),
    ],
)
def test_wandb_observer_rejects_missing_or_synthetic_update_telemetry(
    case: dict, mutation, message: str
) -> None:
    row = {
        "_step": 0,
        "train/step": 0.0,
        "train/grad_norm": 1.25,
        "train/loss": 0.5,
        "train/lr-pg_0": 1e-6,
    }
    mutation(row)
    run = SimpleNamespace(
        entity="synthetic",
        project="synthetic",
        id=case["plan"]["run_name"],
        name=case["plan"]["run_name"],
        state="finished",
        summary={"train/step": 0.0},
        logged_artifacts=lambda: [],
        files=lambda: [],
        scan_history=lambda: [row],
    )
    with pytest.raises(ValueError, match=message):
        acceptance.observe_wandb(
            case["plan"], case["submission"], api=SimpleNamespace(run=lambda path: run)
        )


def test_optimizer_metadata_delta_cannot_masquerade_as_policy_change(case: dict) -> None:
    """The checkpoint already has different optimizer/metadata file hashes."""

    value = json.loads(case["policy_delta"].read_bytes())
    for row in value["ranks"]:
        row["trained_policy_value_sha256"] = row["base_policy_value_sha256"]
        row["policy_changed"] = False
    value["changed_policy_ranks"] = []
    _write_json(case["policy_delta"], _seal(value))

    with pytest.raises(ValueError, match="no independently observed policy tensor delta"):
        _accept(case)


def test_episode_rejects_non_digest_authoritative_verifier(case: dict) -> None:
    path = case["episodes"][1]
    binding = json.loads((path / "binding.json").read_bytes())
    binding["verifier"]["sha256"] = "not-a-digest"
    binding["config_sha256"] = fleet.digest_without(binding, "config_sha256")
    _replace_episode_file(path, "binding.json", binding)
    accepted = json.loads((path / "ACCEPTED.json").read_bytes())
    accepted["config_sha256"] = binding["config_sha256"]
    _write_json(path / "ACCEPTED.json", _seal(accepted))

    with pytest.raises(ValueError, match="verifier identity"):
        acceptance._episode(path, acceptance._dynamic_config(binding), "train")


def test_episode_audit_rejects_reused_challenge_evidence_identity(case: dict) -> None:
    source = case["episodes"][1]
    target = case["episodes"][2]
    reused = json.loads((source / "instance.json").read_bytes())["evidence_run_id"]
    instance = json.loads((target / "instance.json").read_bytes())
    instance["evidence_run_id"] = reused
    _replace_episode_file(target, "instance.json", instance)
    reward = json.loads((target / "reward.json").read_bytes())
    reward["direct_authority_attestation"]["context"]["evidence_run_id"] = reused
    _replace_episode_file(target, "reward.json", reward)

    with pytest.raises(ValueError, match="unique authoritative reward variance"):
        acceptance.episode_audit(case["plan"])


def test_shallow_terminal_validation_still_rejects_extra_or_false_claims(case: dict) -> None:
    result = _accept(case)
    with pytest.raises(ValueError, match="requires reopening"):
        acceptance.validate_terminal(result, check_files=False)
    changed = copy.deepcopy(result)
    changed["private_reward"] = 1.0
    changed = _seal(changed)
    with pytest.raises(ValueError):
        acceptance.validate_terminal(changed, check_files=True)


def test_terminal_consumer_detects_later_evidence_substitution(case: dict) -> None:
    result = _accept(case)
    value = json.loads(case["controller"].read_bytes())
    value["observed_at"] = 9999.0
    _write_json(case["controller"], _seal(value))

    with pytest.raises(ValueError):
        acceptance.validate_terminal(result, check_files=True)
