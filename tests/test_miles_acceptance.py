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
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from cyber_post_train.jobs import API_URLS, digest
from evals.fleet import opencode_self_hosted as fleet
from training import miles, miles_reload_acceptance
from training import miles_acceptance as acceptance
from training import miles_event_evidence as event_evidence
from training import miles_policy_observer as observer
from training.miles_conversion import _hash


def _seal(value: dict) -> dict:
    value = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = "sha256:" + digest(value)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fleet.canonical_json(value))


def _bundle_without_path_validation(
    request: dict, files: dict[str, str], module: str, argv: list[str]
) -> dict:
    payload = json.dumps(
        {"files": files, "module": module, "argv": argv}, sort_keys=True
    ).encode()
    blob = gzip.compress(payload, mtime=0)
    encoded = base64.b64encode(blob).decode()
    return {
        **request,
        "command": "python -c " + shlex.quote("assert " + repr(hashlib.sha256(blob).hexdigest())),
        "env": {**request.get("env", {}), "CYBER_RUNTIME_BUNDLE": encoded},
    }


def _owner(kind: str, name: str, uid: str) -> list[dict]:
    return [{"kind": kind, "name": name, "uid": uid, "controller": True}]


def _kube_event(
    kind: str,
    name: str,
    uid: str,
    *,
    version: int,
    owner: list[dict] | None = None,
    spec: dict | None = None,
    status: dict | None = None,
    event_type: str = "MODIFIED",
) -> dict:
    return {
        "type": event_type,
        "object": {
            "kind": kind,
            "metadata": {
                "namespace": acceptance.NAMESPACE,
                "name": name,
                "uid": uid,
                "resourceVersion": str(version),
                "creationTimestamp": "2026-09-13T03:07:01Z",
                "ownerReferences": owner or [],
            },
            "spec": spec or {},
            "status": status or {},
        },
    }


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


def _policy_observer_evidence(
    *,
    plan: dict,
    plan_path: Path,
    source_submission: dict,
    source_submission_path: Path,
    checkpoint: dict,
    checkpoint_path: Path,
    tmp_path: Path,
) -> tuple[Path, dict]:
    output_root = tmp_path / "observer-output"
    runtime = observer._runtime()
    observer_plan = {
        "schema": observer.PLAN_SCHEMA,
        "run_name": "synthetic-policy-observer",
        "output_root": str(output_root),
        "source_plan": plan,
        "source_plan_sha256": "sha256:" + digest(plan),
        "source_plan_file_sha256": "sha256:" + _hash(plan_path),
        "source_plan_path": str(plan_path),
        "source_submission": {
            "path": str(source_submission_path),
            "file_sha256": "sha256:" + _hash(source_submission_path),
            "receipt_sha256": source_submission["sha256"],
        },
        "trained_checkpoint": checkpoint,
        "trained_checkpoint_reference": {
            "path": str(checkpoint_path),
            "file_sha256": "sha256:" + _hash(checkpoint_path),
            "receipt_sha256": checkpoint["sha256"],
        },
        "base_checkpoint_receipt_sha256": plan["checkpoint"]["sha256"],
        "world_size": 8,
        "comparison_method": observer.COMPARISON_METHOD,
        "rank_state_commitment_method": observer.RANK_STATE_COMMITMENT_METHOD,
        "work_authorized": dict(observer._OBSERVER_WORK),
        "runtime_sha256": digest(runtime),
        "native_driver_sha256": observer.NATIVE_DRIVER_SHA256,
        "deadline_seconds": observer.DEADLINE_SECONDS,
        "restore_method": observer.RESTORE_METHOD,
        "sentinel_markers": {
            "base": observer.BASE_SENTINEL,
            "trained_reference": observer.REFERENCE_SENTINEL,
            "trained_reload": observer.RELOAD_SENTINEL,
        },
        "execution": {
            "cluster_target": "dev",
            "image": miles.IMAGE,
            "priority": "c1",
            "resources": plan["execution"]["resources"],
        },
    }
    observer_plan_path = tmp_path / "observer/plan.json"
    _write_json(observer_plan_path, observer_plan)
    request = observer.job_request(observer_plan)
    request_path = tmp_path / "observer/request.json"
    _write_json(request_path, request)
    watch = tmp_path / "observer/watch"
    observer.start_capture_intent(
        observer_plan,
        request,
        namespace_uid=acceptance.NAMESPACE_UID,
        started_at="2026-09-13T03:06:59Z",
        directory=watch,
        kube_context=event_evidence.DEV_KUBE_CONTEXT,
    )
    run_id = "11111111-1111-4111-8111-111111111111"
    journal_path = tmp_path / "observer/SUBMISSION.jsonl"
    journal_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "state": "POST_INTENT_DO_NOT_RETRY",
                    "api_base_url": API_URLS["dev"],
                    "request_sha256": digest(request),
                    "manifest_sha256": "1" * 64,
                    "nodes": 1,
                    "gpus": 8,
                    "image": miles.IMAGE,
                },
                {
                    "state": "POST_RESPONSE",
                    "name": observer_plan["run_name"] + "-" + run_id[:8],
                    "job_id": run_id,
                    "run_dir": observer_plan["output_root"],
                    "status": "QUEUED",
                    "created_at": "2026-09-13T03:07:00Z",
                    "finished_at": None,
                },
            )
        )
        + "\n"
    )
    submission_path = tmp_path / "observer/SUBMITTED.json"
    submission = observer.compile_submission_binding(
        plan_path=observer_plan_path,
        request_path=request_path,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        submission_journal_path=journal_path,
        output=submission_path,
    )

    def state(seed: int, rank: int, *, progress: int = 1) -> dict:
        return {
            "model": {
                "tensors": 10,
                "local_numel": 100,
                "structure_sha256": "sha256:" + f"{rank + 1:064x}",
                "value_sha256": "sha256:" + f"{seed + rank:064x}",
            },
            "optimizer": {
                "objects": 1,
                "state_entries": 2,
                "parameter_groups": 1,
                "structure_sha256": "sha256:" + f"{rank + 10:064x}",
                "value_sha256": "sha256:" + f"{seed + rank + 100:064x}",
            },
            "scheduler": {
                "positive_progress_counters": progress,
                "structure_sha256": "sha256:" + f"{rank + 11:064x}",
                "value_sha256": "sha256:" + f"{seed + rank + 200:064x}",
            },
            "rng_sha256": "sha256:" + f"{seed + rank + 300:064x}",
        }

    def load(marker: int, preload_seed: int, restored_seed: int, rank: int) -> dict:
        restored = state(restored_seed, rank)
        return {
            "marker": marker,
            "load_calls": 1,
            "preload": state(preload_seed, rank),
            "restored": restored,
            "live": copy.deepcopy(restored),
        }

    def base_load(rank: int) -> dict:
        value = load(observer.BASE_SENTINEL, 1000, 2000, rank)
        for key in ("optimizer", "scheduler"):
            value["restored"][key] = copy.deepcopy(value["preload"][key])
        value["restored"]["rng_sha256"] = value["preload"]["rng_sha256"]
        value["live"] = copy.deepcopy(value["restored"])
        return value

    rows = [
        {
            "rank": rank,
            "policy_tensor_count": 10,
            "local_policy_numel": 100,
            "base": base_load(rank),
            "trained_reference": load(observer.REFERENCE_SENTINEL, 3000, 4000, rank),
            "trained_reload": load(observer.RELOAD_SENTINEL, 5000, 4000, rank),
            "policy_changed": True,
        }
        for rank in range(8)
    ]
    result = _seal(
        {
            "schema": observer.RESULT_SCHEMA,
            "status": "observed",
            "observer_plan_sha256": "sha256:" + digest(observer_plan),
            "source_plan_sha256": observer_plan["source_plan_sha256"],
            "base_checkpoint_receipt_sha256": plan["checkpoint"]["sha256"],
            "trained_checkpoint_receipt_sha256": checkpoint["sha256"],
            "world_size": 8,
            "comparison_method": observer.COMPARISON_METHOD,
            "ranks": rows,
            "changed_policy_ranks": list(range(8)),
            "restored_next_rollout_id": checkpoint["next_rollout_id"],
            "checkpoint_state_commitment_method": observer.RANK_STATE_COMMITMENT_METHOD,
            "restore_method": observer.RESTORE_METHOD,
            "trained_restore_count": 2,
            "prediction_probe": {
                "schema": observer.PREDICTION_SCHEMA,
                "probe_id": observer.PREDICTION_PROBE_ID,
                "input_ids_sha256": "sha256:"
                + digest(list(observer.PREDICTION_INPUT_IDS)),
                "sequence_length": len(observer.PREDICTION_INPUT_IDS),
                "top_k": observer.PREDICTION_TOP_K,
                "selection_margin_threshold": observer.PREDICTION_MARGIN,
                "selection_margin_satisfied": True,
                "prediction_sha256": "sha256:" + "d" * 64,
                "logits_included": False,
                "task_content_included": False,
                "benchmark_content_included": False,
            },
            "state_stable_across_zero_updates": True,
            "work_executed": dict(observer._OBSERVER_WORK),
            "base_checkpoint_unchanged": True,
            "trained_checkpoint_unchanged": True,
            "completed_at": "2026-09-13T03:07:10Z",
            "reward_values_included": False,
            "task_content_included": False,
            "tensor_values_included": False,
        }
    )
    result_path = output_root / "POLICY_OBSERVER_RESULT.json"
    _write_json(result_path, result)

    observer.start_capture(
        observer_plan,
        submission,
        intent_path=watch / "CAPTURE_INTENT.json",
    )
    api_name = submission["api"]["run_name"]
    run_id = submission["api"]["run_id"]
    rayjob_uid = "22222222-2222-4222-8222-222222222222"
    workload_uid = "33333333-3333-4333-8333-333333333333"
    raycluster_uid = "44444444-4444-4444-8444-444444444444"
    pod_uid = "55555555-5555-4555-8555-555555555555"
    cluster_name = "synthetic-observer-cluster"
    rayjob = _kube_event(
        "RayJob",
        api_name,
        rayjob_uid,
        version=1,
        event_type="ADDED",
        spec={
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 0,
            "rayClusterSpec": {
                "headGroupSpec": {"template": {"spec": {"priorityClassName": "c1"}}}
            },
        },
    )
    rayjob["object"]["metadata"]["labels"] = {"fleet.ai/run-id": run_id}
    workload = _kube_event(
        "Workload",
        "synthetic-observer-workload",
        workload_uid,
        version=2,
        owner=_owner("RayJob", api_name, rayjob_uid),
        status={
            "conditions": [
                {
                    "type": "Admitted",
                    "status": "True",
                    "lastTransitionTime": "2026-09-13T03:07:02Z",
                }
            ]
        },
    )
    cluster = _kube_event(
        "RayCluster",
        cluster_name,
        raycluster_uid,
        version=3,
        owner=_owner("RayJob", api_name, rayjob_uid),
        event_type="ADDED",
    )
    pod = _kube_event(
        "Pod",
        "synthetic-observer-pod",
        pod_uid,
        version=4,
        owner=_owner("RayCluster", cluster_name, raycluster_uid),
        spec={
            "containers": [
                {"name": "ray-head", "resources": {"limits": {"nvidia.com/gpu": "8"}}}
            ]
        },
        status={
            "phase": "Succeeded",
            "containerStatuses": [
                {
                    "name": "ray-head",
                    "imageID": "containerd://registry/image@" + miles.IMAGE.rsplit("@", 1)[1],
                    "restartCount": 0,
                    "state": {
                        "terminated": {
                            "exitCode": 0,
                            "reason": "Completed",
                            "finishedAt": "2026-09-13T03:07:11Z",
                        }
                    },
                }
            ],
        },
    )
    terminal = copy.deepcopy(rayjob)
    terminal["type"] = "DELETED"
    terminal["object"]["metadata"]["resourceVersion"] = "5"
    terminal["object"]["status"] = {
        "jobStatus": "SUCCEEDED",
        "rayClusterName": cluster_name,
    }
    for sequence, event in enumerate((rayjob, workload, cluster, pod, terminal)):
        event_evidence.record_event(
            watch,
            event,
            observed_at=f"2026-09-13T03:07:{sequence + 1:02d}Z",
            sequence=sequence,
        )
    controller_path = tmp_path / "observer/CONTROLLER.json"
    observer.compile_controller(
        observer_plan,
        submission,
        directory=watch,
        api_status="SUCCEEDED",
        observed_at="2026-09-13T03:07:12Z",
        output=controller_path,
    )
    release_path = tmp_path / "observer/RELEASE.json"
    query_path = tmp_path / "observer/RELEASE_QUERY.json"

    class Jobs:
        def status(self, name: str) -> dict:
            return {
                "name": name,
                "job_id": run_id,
                "run_dir": observer_plan["output_root"],
                "status": "SUCCEEDED",
                "created_at": "2026-09-13T03:07:00Z",
                "finished_at": "2026-09-13T03:07:12Z",
            }

    def kubectl(command: list[str], **_: object) -> SimpleNamespace:
        if command[4] == "namespace":
            payload = {"metadata": {"uid": acceptance.NAMESPACE_UID}}
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload).encode())
        return SimpleNamespace(returncode=0, stdout=b"")

    observer.collect_release_query(
        observer_plan,
        submission,
        controller_path=controller_path,
        jobs=Jobs(),
        output=query_path,
        run_command=kubectl,
        clock=lambda: datetime.fromisoformat("2026-09-13T03:07:13+00:00").timestamp(),
    )
    observer.compile_release(
        observer_plan,
        submission,
        controller_path=controller_path,
        release_query_path=query_path,
        output=release_path,
    )
    policy_path = output_root / "POLICY_DELTA.json"
    policy = observer.accept_policy_delta(
        observer_plan,
        submission_path=submission_path,
        result_path=result_path,
        controller_path=controller_path,
        release_path=release_path,
        output=policy_path,
    )
    return policy_path, policy


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(observer, "bundled_request", _bundle_without_path_validation)
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
    base_root = tmp_path / "base-checkpoint"
    (base_root / "release").mkdir(parents=True)
    (base_root / "latest_checkpointed_iteration.txt").write_text("release\n")
    (base_root / "release/.metadata").write_bytes(b"synthetic base metadata")
    (base_root / "release/__0_0.distcp").write_bytes(b"synthetic base weights")
    arguments = {
        "name": run_name,
        "output_root": str(root),
        "model_root": "/mnt/sfs/models/synthetic-qwen",
        "torch_dist_root": str(base_root),
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
            "root": str(base_root),
            "model": model,
            "files": [
                {
                    "path": str(path.relative_to(base_root)),
                    "size": path.stat().st_size,
                    "sha256": _hash(path),
                }
                for path in sorted(base_root.rglob("*"))
                if path.is_file()
            ],
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

    policy_delta_path, policy_delta = _policy_observer_evidence(
        plan=plan,
        plan_path=root / "plan.json",
        source_submission=submission,
        source_submission_path=submission_path,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        tmp_path=tmp_path,
    )

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


def _observer_files(case: dict) -> tuple[dict, Path, dict]:
    policy = json.loads(case["policy_delta"].read_bytes())
    submission = json.loads(Path(policy["observer_submission"]["path"]).read_bytes())
    plan = json.loads(Path(submission["observer_plan_path"]).read_bytes())
    result_path = Path(policy["observer_result"]["path"])
    return plan, result_path, json.loads(result_path.read_bytes())


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


def test_single_observer_run_also_satisfies_native_reload(case: dict) -> None:
    terminal = _accept(case)
    terminal_path = case["root"] / "MILES_TERMINAL_ACCEPTED.json"
    policy = json.loads(case["policy_delta"].read_bytes())
    output = Path(policy["observer_result"]["path"]).parent / "RELOAD_ACCEPTED.json"

    result = observer.accept_observer_reload(
        terminal_path=terminal_path,
        output=output,
    )

    assert result["source_terminal_acceptance_sha256"] == terminal["sha256"]
    assert result["source_manifest_sha256"] == terminal["checkpoint_manifest"][
        "receipt_sha256"
    ]
    assert result["observer_gpu_jobs"] == 1
    assert result["additional_reload_gpu_jobs"] == 0
    assert result["state_stable_across_zero_updates"] is True
    assert result["external_gpu_release_verified"] is True
    validated = miles_reload_acceptance.validate_accepted(result, check_files=True)
    assert validated["source_terminal_acceptance_sha256"] == terminal["sha256"]
    assert validated["source_manifest_sha256"] == terminal["checkpoint_manifest"][
        "receipt_sha256"
    ]
    with pytest.raises(ValueError, match="reopening every referenced file"):
        miles_reload_acceptance.validate_accepted(result, check_files=False)
    with pytest.raises(FileExistsError):
        observer.accept_observer_reload(terminal_path=terminal_path, output=output)


def test_observer_reload_rejects_commitment_or_result_replacement(case: dict) -> None:
    _accept(case)
    terminal_path = case["root"] / "MILES_TERMINAL_ACCEPTED.json"
    policy = json.loads(case["policy_delta"].read_bytes())
    output = Path(policy["observer_result"]["path"]).parent / "RELOAD_ACCEPTED.json"
    accepted = observer.accept_observer_reload(
        terminal_path=terminal_path,
        output=output,
    )

    forged = copy.deepcopy(accepted)
    forged["rank_state_commitments_sha256"] = "sha256:" + "f" * 64
    forged = _seal(forged)
    with pytest.raises(ValueError, match="differs from rederived evidence"):
        miles_reload_acceptance.validate_accepted(forged, check_files=True)

    result_path = Path(policy["observer_result"]["path"])
    result = json.loads(result_path.read_bytes())
    result["ranks"][0]["trained_reload"]["restored"]["model"]["value_sha256"] = (
        "sha256:" + "e" * 64
    )
    _write_json(result_path, _seal(result))
    with pytest.raises(
        ValueError,
        match="reference changed|differs from rederived observer evidence|restore commitments",
    ):
        miles_reload_acceptance.validate_accepted(accepted, check_files=True)


def test_policy_observer_accepts_historical_bundle_after_current_code_changes(
    case: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = json.loads(case["policy_delta"].read_bytes())
    checkpoint = json.loads(case["checkpoint"].read_bytes())
    before = observer.validate_policy_evidence(policy, case["plan"], checkpoint)
    monkeypatch.setattr(observer, "_runtime", lambda: {"future.py": "changed"})

    after = observer.validate_policy_evidence(policy, case["plan"], checkpoint)

    assert before == after == 8


def test_policy_observer_result_can_truthfully_report_no_delta_but_not_accept(
    case: dict,
) -> None:
    policy = json.loads(case["policy_delta"].read_bytes())
    submission_path = Path(policy["observer_submission"]["path"])
    submission = json.loads(submission_path.read_bytes())
    plan = json.loads(Path(submission["observer_plan_path"]).read_bytes())
    result_path = Path(policy["observer_result"]["path"])
    result = json.loads(result_path.read_bytes())
    for row in result["ranks"]:
        value = row["base"]["restored"]["model"]["value_sha256"]
        row["trained_reference"]["restored"]["model"]["value_sha256"] = value
        row["trained_reload"]["restored"]["model"]["value_sha256"] = value
        row["policy_changed"] = False
    result["changed_policy_ranks"] = []
    _write_json(result_path, _seal(result))
    with pytest.raises(ValueError, match="restore commitments|changed-rank summary"):
        observer.validate_result(plan, json.loads(result_path.read_bytes()))
    case["policy_delta"].unlink()

    with pytest.raises(ValueError, match="restore commitments|changed-rank summary"):
        observer.accept_policy_delta(
            plan,
            submission_path=submission_path,
            result_path=result_path,
            controller_path=Path(policy["observer_controller"]["path"]),
            release_path=Path(policy["observer_release"]["path"]),
            output=case["policy_delta"],
        )
    assert not case["policy_delta"].exists()


@pytest.mark.parametrize(
    "fault",
    [
        "base_nonpolicy_changed",
        "trained_preloads_collide",
        "trained_restore_not_overwritten",
        "trained_restores_disagree",
        "live_state_changed",
    ],
)
def test_policy_observer_rejects_circular_or_unstable_restore_evidence(
    case: dict,
    fault: str,
) -> None:
    plan, result_path, result = _observer_files(case)
    row = result["ranks"][0]
    if fault == "base_nonpolicy_changed":
        row["base"]["restored"]["optimizer"]["value_sha256"] = "sha256:" + "a" * 64
        row["base"]["live"] = copy.deepcopy(row["base"]["restored"])
    elif fault == "trained_preloads_collide":
        row["trained_reload"]["preload"]["model"]["value_sha256"] = row[
            "trained_reference"
        ]["preload"]["model"]["value_sha256"]
    elif fault == "trained_restore_not_overwritten":
        row["trained_reference"]["restored"]["optimizer"]["value_sha256"] = row[
            "trained_reference"
        ]["preload"]["optimizer"]["value_sha256"]
        row["trained_reference"]["live"] = copy.deepcopy(
            row["trained_reference"]["restored"]
        )
    elif fault == "trained_restores_disagree":
        row["trained_reload"]["restored"]["rng_sha256"] = "sha256:" + "b" * 64
        row["trained_reload"]["live"] = copy.deepcopy(row["trained_reload"]["restored"])
    else:
        row["trained_reload"]["live"]["model"]["value_sha256"] = "sha256:" + "c" * 64
    _write_json(result_path, _seal(result))

    with pytest.raises(ValueError, match="restore|load state|no-load|preload|zero-update"):
        observer.validate_result(plan, json.loads(result_path.read_bytes()))


def test_policy_observer_does_not_claim_an_unproven_rng_delta(case: dict) -> None:
    plan, result_path, result = _observer_files(case)
    for row in result["ranks"]:
        base_rng = row["base"]["restored"]["rng_sha256"]
        for label in ("trained_reference", "trained_reload"):
            row[label]["restored"]["rng_sha256"] = base_rng
            row[label]["live"]["rng_sha256"] = base_rng
    _write_json(result_path, _seal(result))

    assert observer.validate_result(plan, json.loads(result_path.read_bytes())) == list(
        range(observer.WORLD_SIZE)
    )


def test_observer_submission_rejects_an_appended_post_journal_row(case: dict) -> None:
    policy = json.loads(case["policy_delta"].read_bytes())
    submission_path = Path(policy["observer_submission"]["path"])
    submission = json.loads(submission_path.read_bytes())
    plan = json.loads(Path(submission["observer_plan_path"]).read_bytes())
    with Path(submission["submission_journal_path"]).open("a") as stream:
        stream.write(json.dumps({"state": "POST_RESPONSE"}) + "\n")

    with pytest.raises(ValueError, match="one intent and one response"):
        observer.validate_submission_binding(submission, plan, check_files=True)


def test_observer_release_rejects_a_still_present_exact_uid(case: dict) -> None:
    policy = json.loads(case["policy_delta"].read_bytes())
    release = json.loads(Path(policy["observer_release"]["path"]).read_bytes())
    query_path = Path(release["release_query"]["path"])
    query = json.loads(query_path.read_bytes())
    query["kubernetes"]["objects"]["pod"]["present"] = True
    query["kubernetes"]["objects"]["pod"]["observed_uid"] = query["kubernetes"][
        "objects"
    ]["pod"]["expected_uid"]
    _write_json(query_path, _seal(query))
    checkpoint = json.loads(case["checkpoint"].read_bytes())

    with pytest.raises(ValueError, match="release query|exact absence|reference changed"):
        observer.validate_policy_evidence(policy, case["plan"], checkpoint)


def test_policy_observer_rejects_ttl_zero_event_replacement(case: dict) -> None:
    policy = json.loads(case["policy_delta"].read_bytes())
    controller = json.loads(Path(policy["observer_controller"]["path"]).read_bytes())
    event_path = Path(controller["event_journal"]["events"][-1]["path"])
    event = json.loads(event_path.read_bytes())
    event["controller_status"] = "FAILED"
    _write_json(event_path, _seal(event))
    checkpoint = json.loads(case["checkpoint"].read_bytes())

    with pytest.raises(ValueError, match="event journal|controller|journal missed terminal"):
        observer.validate_policy_evidence(policy, case["plan"], checkpoint)


def test_policy_observer_public_receipts_exclude_task_and_metric_values(case: dict) -> None:
    terminal = _accept(case)
    policy = json.loads(case["policy_delta"].read_bytes())
    output = Path(policy["observer_result"]["path"]).parent / "RELOAD_ACCEPTED.json"
    reload_accepted = observer.accept_observer_reload(
        terminal_path=case["root"] / "MILES_TERMINAL_ACCEPTED.json",
        output=output,
    )

    public = json.dumps([policy, terminal, reload_accepted], sort_keys=True)
    assert "synthetic training task" not in public
    assert "rollout/episode_raw_reward" not in public
    assert "eval/heldout-cyber" not in public
    assert policy["reward_values_included"] is False
    assert reload_accepted["reward_values_included"] is False


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
            base = row["base"]["restored"]["model"]["value_sha256"]
            row["trained_reference"]["restored"]["model"]["value_sha256"] = base
            row["trained_reload"]["restored"]["model"]["value_sha256"] = base
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
        base = row["base"]["restored"]["model"]["value_sha256"]
        row["trained_reference"]["restored"]["model"]["value_sha256"] = base
        row["trained_reload"]["restored"]["model"]["value_sha256"] = base
        row["policy_changed"] = False
    value["changed_policy_ranks"] = []
    _write_json(case["policy_delta"], _seal(value))

    with pytest.raises(
        ValueError,
        match="no independently observed policy tensor delta|differs from rederived observer",
    ):
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
