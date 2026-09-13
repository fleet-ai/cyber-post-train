"""Synthetic terminal evidence only; no Fleet, W&B, Kubernetes, or GPU calls."""

from __future__ import annotations

import copy
import json
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
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
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
    plan = {
        "schema": "cyber_miles_training_v1",
        "run_name": run_name,
        "output_root": str(root),
        "model": model,
        "data": data,
        "checkpoint": {
            "schema": "cyber_miles_checkpoint_v1",
            "model": model,
            "files": [{"path": "base", "size": 1, "sha256": "a" * 64}],
        },
        "arguments": arguments,
        "runtime_sha256": "8" * 64,
        "native_driver_sha256": "9" * 64,
        "execution": {
            "image": miles.IMAGE,
            "priority": "c1",
            "resources": {},
        },
    }
    request = {
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "requeueIfPreempted": False,
        "secrets": ["fleet-api", "wandb-api"],
    }
    monkeypatch.setattr(acceptance, "job_request", lambda value: request)
    _write_json(root / "plan.json", plan)
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
            "remote_scalar_history_sha256": "sha256:" + "b" * 64,
            "observed_optimizer_steps": [0, 1],
            "step_one_metric_names": [
                "policy/grad_norm",
                "policy/policy_loss",
                "policy/policy_lr",
                "trainer/global_step",
            ],
            "step_one_metrics": {
                "policy/grad_norm": 1.25,
                "policy/policy_loss": 0.5,
                "policy/policy_lr": 1e-6,
                "trainer/global_step": 1.0,
            },
            "positive_finite_gradient_observed": True,
            "finite_policy_loss_observed": True,
            "learning_rate_matches_plan": True,
            "logged_artifact_count": 0,
            "rich_payload_count": 0,
            "reward_values_included": False,
        }
    )
    wandb_path = tmp_path / "evidence/WANDB.json"
    _write_json(wandb_path, wandb)

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
                        "runtime_uid": 1000,
                        "runtime_gid": 100,
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
        "root": root,
        "episodes": episodes,
        "checkpoint": checkpoint_path,
        "wandb": wandb_path,
        "controller": controller_path,
        "release": release_path,
    }


def _accept(case: dict) -> dict:
    return acceptance.accept_terminal(
        case["plan"],
        checkpoint_manifest_path=case["checkpoint"],
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
        "completion",
        "conflict",
        "interaction",
        "reward_variance",
        "wandb",
        "checkpoint",
        "controller",
        "release",
    ],
)
def test_terminal_acceptance_fails_closed_on_each_scientific_or_release_gate(
    case: dict, fault: str
) -> None:
    if fault == "plan":
        case["plan"]["arguments"]["steps"] = 2
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
        value["step_one_metrics"]["policy/grad_norm"] = 0.0
        _write_json(case["wandb"], _seal(value))
    elif fault == "checkpoint":
        (case["root"] / "checkpoints/iter_0000000/__7_0.distcp").write_bytes(b"tampered")
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
        {"_step": 0, "trainer/global_step": 0.0},
        {
            "_step": 1,
            "trainer/global_step": 1.0,
            "policy/grad_norm": 1.25,
            "policy/policy_loss": 0.5,
            "policy/policy_lr": 1e-6,
            "reward/avg_raw_reward": 0.731,
        },
    ]
    run = SimpleNamespace(
        entity="synthetic",
        project="synthetic",
        id=plan["run_name"],
        name=plan["run_name"],
        state="finished",
        summary={"trainer/global_step": 1.0},
        logged_artifacts=lambda: [],
        files=lambda: [],
        scan_history=lambda: copy.deepcopy(history),
    )
    api = SimpleNamespace(run=lambda path: run)

    result = acceptance.observe_wandb(plan, api=api)

    assert result["observed_optimizer_steps"] == [0, 1]
    assert result["step_one_metrics"]["policy/grad_norm"] == 1.25
    assert "0.731" not in json.dumps(result)
    assert result["reward_values_included"] is False
    history.append({"_step": 2, "trainer/global_step": 2.0})
    with pytest.raises(ValueError, match="exactly one optimizer"):
        acceptance.observe_wandb(plan, api=api)


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


def test_shallow_terminal_validation_still_rejects_extra_or_false_claims(case: dict) -> None:
    result = _accept(case)
    changed = copy.deepcopy(result)
    changed["private_reward"] = 1.0
    changed = _seal(changed)
    with pytest.raises(ValueError, match="not promotion-safe"):
        acceptance.validate_terminal(changed, check_files=False)


def test_terminal_consumer_detects_later_evidence_substitution(case: dict) -> None:
    result = _accept(case)
    value = json.loads(case["controller"].read_bytes())
    value["observed_at"] = 9999.0
    _write_json(case["controller"], _seal(value))

    with pytest.raises(ValueError):
        acceptance.validate_terminal(result, check_files=True)
