#!/usr/bin/env python3
"""Build the offline launch-readiness audit for the Qwen3.8 SkyRL queue.

This command compiles only local, sanitized plans.  It never creates a Jobs
client, contacts Kubernetes or W&B, reads private task payloads, or mutates an
external destination.  Missing live evidence is recorded as a blocker rather
than inferred from an older observation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from unittest import mock

from cyber_post_train.jobs import digest
from scripts import prepare_qwen38_skyrl_production_queue as queue
from training import rl_reward_canary as canary
from training import sft, skyrl, skyrl_topology_probe, skyrl_training
from training.sft_runtime import (
    WATCHDOG_DRAIN_SECONDS,
    WATCHDOG_HARD_SECONDS,
    WATCHDOG_IDLE_SECONDS,
    WATCHDOG_POLL_SECONDS,
    WATCHDOG_STARTUP_SECONDS,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/evidence/qwen38-study/2026-09-20-skyrl-launch-readiness-audit-v1.json"
NEXT_GATES = ROOT / "docs/evidence/qwen38-study/2026-09-20-skyrl-next-gates-queue-v1.json"
TOPOLOGY_CONFIG = ROOT / "configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json"
CANARY_DATA = ROOT / "configs/qualification/qwen38-rl-reward-canary-data-prod-v4.json"
CANARY_RUN = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v4.json"
CANARY_QUALIFICATION = ROOT / canary.QUALIFICATION_PATH
MODEL_LOCK = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
MODEL_WEIGHTS = ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"


def load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> dict:
    result = {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": file_sha256(path),
    }
    if path.suffix == ".json":
        value = json.loads(path.read_bytes())
        if isinstance(value, dict) and (self_sha256 := value.get("sha256")) is not None:
            result["self_sha256"] = self_sha256
    return result


def seal(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = "sha256:" + digest(result)
    return result


def raw(value: dict) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def prod4_metadata(run: dict, next_gates: dict) -> dict:
    """Reconstruct the committed sanitized manifest surface, never task text."""
    local = next_gates["local_data_preparation"]
    value = {
        "schema": "cyber_skyrl_data_v1",
        "name": run["name"],
        "selection_sha256": canary.TASK_SET_SELF_SHA256,
        "split_sha256": canary.SPLIT_SELF_SHA256,
        "tokenizer": {
            "backend_sha256": ("ffb7a28b27dabcc333662fd3e0b0005d9e79a1c22e31453ab5a3017fbd5f25c0"),
            "chat_template_sha256": (
                "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
            ),
            "files": [
                {
                    "path": "tokenizer.json",
                    "sha256": ("0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
                },
                {
                    "path": "tokenizer_config.json",
                    "sha256": ("b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
                },
                {
                    "path": "chat_template.jinja",
                    "sha256": ("c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
                },
                {
                    "path": "merges.txt",
                    "sha256": ("a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
                },
                {
                    "path": "vocab.json",
                    "sha256": ("ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
                },
            ],
            "repo": canary.MODEL["repo"],
            "revision": canary.MODEL["revision"],
        },
        "template_sha256": (
            "sha256:c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
        ),
        "tool_catalog_sha256": canary.TOOL_CATALOG_SHA256,
        "limits": copy.deepcopy(canary.LIMITS),
        "files": {
            split: {
                "path": split + ".jsonl",
                "sha256": local[split]["sha256"],
                "rows": local[split]["rows"],
                "max_prompt_tokens": local[split]["max_prompt_tokens"],
            }
            for split in ("train", "dev")
        },
        "gpus": 0,
        "environment_creates": 0,
    }
    value["sha256"] = "sha256:" + digest(value)
    if value["sha256"] != local["manifest_self_sha256"]:
        raise ValueError("sanitized prod4 manifest no longer matches the prepared candidate")
    return value


def compile_prod4(run: dict, metadata: dict) -> tuple[dict, dict]:
    """Exercise the real compiler against digests only, without touching SFS."""
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(run["data"]["manifest"]):
            return copy.deepcopy(metadata)
        return original(path)

    with mock.patch.object(sft, "read_mapping", side_effect=read):
        plan = skyrl_training.compile_rl(run, relative_to=CANARY_RUN.parent)
    return plan, skyrl_training.job_request(plan)


def episode_ceiling(run: dict, *, dev_rows: int, episode_seconds: int) -> dict:
    recipe = run["recipe"]
    concurrency = recipe["groups"] * recipe["samples_per_prompt"]
    eval_steps = {0, recipe["steps"]} | set(
        range(recipe["eval_interval"], recipe["steps"] + 1, recipe["eval_interval"])
    )
    dev_waves = math.ceil(dev_rows / concurrency)
    ceiling = (recipe["steps"] + len(eval_steps) * dev_waves) * episode_seconds
    return {
        "episode_seconds": episode_seconds,
        "train_steps": recipe["steps"],
        "evaluation_points": len(eval_steps),
        "development_rows": dev_rows,
        "episode_concurrency": concurrency,
        "development_waves_per_evaluation": dev_waves,
        "legal_episode_ceiling_seconds_excluding_startup_and_optimization": ceiling,
        "watchdog_hard_seconds": WATCHDOG_HARD_SECONDS,
        "fits_watchdog_hard_bound": ceiling <= WATCHDOG_HARD_SECONDS,
    }


def topology_audit(qualification: dict) -> dict:
    config = load(TOPOLOGY_CONFIG)
    plan = skyrl_topology_probe.compile_probe(TOPOLOGY_CONFIG)
    request = skyrl_topology_probe.request(plan)
    fleetjob = skyrl_topology_probe.fleetjob_manifest(plan)
    preflight = skyrl_topology_probe.preflight_job_manifest(plan)
    verifier = skyrl_topology_probe.receipt_verify_job_manifest(plan)
    expected = qualification["topology_successor"]
    computed = {
        "plan_sha256": "sha256:" + digest(plan),
        "request_sha256": "sha256:" + digest(request),
        "fleetjob_manifest_sha256": "sha256:" + digest(fleetjob),
        "preflight_manifest_sha256": "sha256:" + digest(preflight),
        "receipt_verifier_manifest_sha256": "sha256:" + digest(verifier),
    }
    for key, value in computed.items():
        if value != expected[key]:
            raise ValueError(f"V17 {key} differs from the frozen qualification")
    job = fleetjob["spec"]["job"]["spec"]
    cluster = job["rayClusterSpec"]
    head = cluster["headGroupSpec"]["template"]["spec"]["containers"][0]
    workers = cluster["workerGroupSpecs"]
    if (
        request["workers"] != 1
        or request["gpus_per_worker"] != 8
        or request["priority_class"] != "c1"
        or head["resources"]["limits"]["nvidia.com/gpu"] != "8"
        or sum(group["replicas"] for group in workers) != 0
        or job["activeDeadlineSeconds"] != 1800
        or job["shutdownAfterJobFinishes"] is not True
    ):
        raise ValueError("V17 resource or release shape changed")
    return {
        "state": "prepared_not_submitted",
        "config": source(TOPOLOGY_CONFIG),
        "compiled_digests": computed,
        "identity": {
            "name": plan["run_name"],
            "cluster_target": plan["execution"]["cluster_target"],
            "image": plan["execution"]["image"],
            "output_root": plan["output_root"],
        },
        "resource_shape": {
            "priority": "c1",
            "physical_nodes": 1,
            "gpus": 8,
            "gpu_head_pods": 1,
            "gpu_worker_replicas": 0,
            "engines": 2,
            "tensor_parallel_size": 4,
        },
        "science": plan["scientific_work"],
        "bounds": {
            "setup_seconds": plan["deadlines"]["setup_seconds"],
            "cleanup_seconds": plan["deadlines"]["cleanup_seconds"],
            "process_total_seconds": plan["deadlines"]["total_seconds"],
            "fleetjob_active_deadline_seconds": job["activeDeadlineSeconds"],
            "shutdown_after_job_finishes": job["shutdownAfterJobFinishes"],
            "receipt_grace_seconds": expected["terminal_receipt_grace_seconds"],
            "external_UID_bound_release_required": True,
        },
        "wandb": {
            "enabled": False,
            "secret_in_request": "wandb-api" in request.get("secrets", []),
        },
        "encoded_absence_checks": {
            "create_once_output_checked_by_CPU_preflight": True,
            "Kubernetes_name_checked_in_dev_and_prod_before_create": True,
            "fresh_output_check_at_create_time": False,
        },
        "submission_gate": config["submission_gate"],
        "ready": False,
    }


def prod4_audit(next_gates: dict) -> dict:
    run = load(CANARY_RUN)
    metadata = prod4_metadata(run, next_gates)
    plan, request = compile_prod4(run, metadata)
    frozen = next_gates["scientific_canary"]
    if (
        "sha256:" + digest(plan) != frozen["plan_sha256"]
        or "sha256:" + digest(request) != frozen["request_sha256"]
        or request["workers"] != 1
        or request["gpus_per_worker"] != 8
        or request["priority_class"] != "c1"
        or request["secrets"] != ["fleet-api", "wandb-api"]
        or request["env"]["WANDB_RUN_ID"] != run["name"]
        or request["env"]["WANDB_MODE"] != "online"
        or plan["model"]["revision"] != canary.MODEL["revision"]
        or plan["data"]["sha256"] != next_gates["local_data_preparation"]["manifest_self_sha256"]
    ):
        raise ValueError("prod4 plan, request, resource, model, data, or W&B binding changed")
    return {
        "state": "compiled_from_sanitized_manifest_not_staged_or_submitted",
        "inputs": {
            "data_config": source(CANARY_DATA),
            "run_config": source(CANARY_RUN),
            "qualification": source(CANARY_QUALIFICATION),
            "model_lock": source(MODEL_LOCK),
            "model_weights": source(MODEL_WEIGHTS),
        },
        "compiled_digests": {
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "sanitized_manifest_self_sha256": metadata["sha256"],
            "staged_manifest_observed": False,
        },
        "identity": {
            "name": run["name"],
            "cluster_target": plan["execution"]["cluster_target"],
            "image": request["image"],
            "output_root": run["output_root"],
        },
        "resource_shape": {
            "priority": request["priority_class"],
            "nodes": request["workers"],
            "gpus_per_node": request["gpus_per_worker"],
            "gpus": request["workers"] * request["gpus_per_worker"],
            "requeue_if_preempted": request["requeueIfPreempted"],
        },
        "bindings": {
            "model": {
                key: plan["model"][key]
                for key in ("repo", "revision", "root", "weight_manifest_sha256")
            },
            "dataset": {
                "selection_sha256": metadata["selection_sha256"],
                "split_sha256": metadata["split_sha256"],
                "tool_catalog_sha256": metadata["tool_catalog_sha256"],
                "rows": {key: value["rows"] for key, value in metadata["files"].items()},
                "payload_file_sha256": {
                    key: value["sha256"] for key, value in metadata["files"].items()
                },
                "task_environment_verifier_bindings": [
                    {
                        key: task[key]
                        for key in (
                            "task_key",
                            "task_version_id",
                            "environment_version_id",
                            "verifier_version_id",
                            "split",
                        )
                    }
                    for task in canary.TASKS
                ],
            },
            "reward": {
                "mode": "authoritative_Fleet_fractional_partial_cyber_score",
                "ordered_tools": ["bash", "submit_report"],
                "one_verifier_execution_ID_per_scored_episode_required": True,
            },
        },
        "wandb": {
            **run["wandb"],
            "secret_reference": "wandb-api",
            "resume": "never",
            "scalar_only": True,
            "code_console_and_trajectory_uploads_disabled": True,
            "fresh_run_ID_absence_checked_before_submit": False,
        },
        "watchdog": {
            "poll_seconds": WATCHDOG_POLL_SECONDS,
            "startup_seconds": WATCHDOG_STARTUP_SECONDS,
            "confirmed_idle_seconds": WATCHDOG_IDLE_SECONDS,
            "hard_seconds": WATCHDOG_HARD_SECONDS,
            "drain_seconds": WATCHDOG_DRAIN_SECONDS,
            "child_SIGTERM_then_SIGKILL_seconds": 60,
            "episode_budget": episode_ceiling(
                run,
                dev_rows=metadata["files"]["dev"]["rows"],
                episode_seconds=metadata["limits"]["episode_seconds"],
            ),
            "Jobs_API_preview_must_render_shutdown_after_job_finishes": True,
            "independent_UID_bound_release_observer_recorded": False,
        },
        "encoded_absence_checks": {
            "submission_journal_create_once": True,
            "Jobs_API_name_prefix_and_output_history": True,
            "CPU_preflight_output_absence": True,
            "runtime_new_checkpoint_and_episode_trees": True,
            "fresh_Kubernetes_name_absence": False,
            "fresh_SFS_output_absence_immediately_before_create": False,
            "fresh_WandB_run_ID_absence_immediately_before_create": False,
        },
        "submission_gate": plan["qualification"]["submission_gate"],
        "ready": False,
    }


def production_arms_audit() -> dict:
    expected = queue.expected()
    stale = [
        str(path.relative_to(ROOT))
        for path, value in expected.items()
        if not path.exists() or path.read_bytes() != queue.raw(value)
    ]
    if stale:
        raise ValueError("production queue artifacts changed: " + ", ".join(stale))
    qualification = load(queue.QUALIFICATION)
    staging = load(queue.STAGING_PACKET)
    controls = qualification["fixed_controls"]
    arms = []
    for arm in queue.ARMS:
        run_path = queue.path_for("run", arm)
        data_path = queue.path_for("data", arm)
        run, data = load(run_path), load(data_path)
        args = skyrl.SkyRLConfig(
            name=run["name"],
            output_root=run["output_root"],
            model=canary.MODEL["repo"],
            model_root=run["model"]["root"],
            train_data=run["data"]["root"] + "/train.jsonl",
            dev_data=run["data"]["root"] + "/dev.jsonl",
            data_manifest=run["data"]["manifest"],
            train_rows=59,
            dev_rows=20,
            wandb_entity=run["wandb"]["entity"],
            wandb_project=run["wandb"]["project"],
            wandb_run_id=run["wandb"]["run_id"],
            context_tokens=data["limits"]["context_tokens"],
            response_tokens=data["limits"]["response_tokens"],
            tokens_per_turn=data["limits"]["max_tokens_per_turn"],
            max_turns=data["limits"]["max_turns"],
            **run["recipe"],
        )
        overrides = skyrl.overrides(args)
        if (
            run["cluster"]["priority"] != "c1"
            or overrides["trainer.placement.policy_num_nodes"] != 1
            or overrides["trainer.placement.policy_num_gpus_per_node"] != 8
            or run["wandb"]["run_id"] != run["name"]
            or run["model"] != load(CANARY_RUN)["model"]
            or data["limits"]
            != {
                "context_tokens": controls["context_tokens"],
                "response_tokens": controls["response_tokens"],
                "max_tokens_per_turn": controls["max_tokens_per_turn"],
                "max_turns": controls["max_turns"],
                "episode_seconds": controls["episode_seconds"],
                "tool_seconds": 330,
                "tool_result_chars": 50000,
            }
        ):
            raise ValueError(f"production arm {arm['id']} changed a fixed launch control")
        stage = next(item for item in staging["arms"] if item["name"] == arm["name"])
        arms.append(
            {
                "id": arm["id"],
                "dispatch_order": arm["priority"],
                "name": run["name"],
                "treatment": arm["treatment"],
                "run_config": source(run_path),
                "data_config": source(data_path),
                "plan_sha256": None,
                "request_sha256": None,
                "plan_state": "not_compilable_until_exact_staged_manifest_and_validator_exist",
                "resource_shape": {
                    "priority": "c1",
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "gpus": 8,
                },
                "recipe": run["recipe"],
                "model": run["model"],
                "dataset": {
                    "root": run["data"]["root"],
                    "manifest": run["data"]["manifest"],
                    "manifest_file_sha256": stage["data_manifest_file_sha256"],
                    "manifest_self_sha256": stage["data_manifest_sha256"],
                    "task_set_sha256": controls["task_set_sha256"],
                    "split_sha256": controls["split_sha256"],
                    "counts": controls["counts"],
                    "ordered_tools": controls["ordered_tools"],
                    "staged": False,
                },
                "reward": controls["reward"],
                "wandb": {
                    **run["wandb"],
                    "resume": "never",
                    "fresh_run_ID_absence_checked_before_submit": False,
                },
                "watchdog": episode_ceiling(
                    run,
                    dev_rows=20,
                    episode_seconds=data["limits"]["episode_seconds"],
                ),
                "ready": False,
            }
        )
    if (
        len({arm["name"] for arm in arms}) != 5
        or len({arm["wandb"]["run_id"] for arm in arms}) != 5
    ):
        raise ValueError("production create-once identities are not unique")
    return {
        "state": "offline_configs_sealed_plans_not_prepared",
        "qualification": source(queue.QUALIFICATION),
        "data_staging": source(queue.STAGING_PACKET),
        "task_set": source(queue.TASK_SET),
        "split": source(queue.SPLIT),
        "tool_catalog": source(queue.TOOL_CATALOG),
        "model_lock": source(MODEL_LOCK),
        "model_weights": source(MODEL_WEIGHTS),
        "common_reward": controls["reward"],
        "arms": arms,
        "all_c1_one_node_eight_GPU": all(
            arm["resource_shape"] == {"priority": "c1", "nodes": 1, "gpus_per_node": 8, "gpus": 8}
            for arm in arms
        ),
        "all_legal_episode_ceilings_fit_shared_watchdog": all(
            arm["watchdog"]["fits_watchdog_hard_bound"] for arm in arms
        ),
        "submission_gate": qualification["submission_gate"],
        "ready": False,
    }


def build() -> dict:
    next_gates = load(NEXT_GATES)
    qualification = load(CANARY_QUALIFICATION)
    if next_gates["failure_budget"] != {
        "used": 10,
        "limit": 10,
        "reset_recorded": False,
        "external_cluster_post_stop": True,
    }:
        raise ValueError("cluster failure budget no longer matches the frozen stop")
    topology = topology_audit(qualification)
    prod4 = prod4_audit(next_gates)
    production = production_arms_audit()
    blockers = [
        "cluster_failure_budget_is_10_of_10_and_has_not_been_reset",
        "v17_CPU_preflight_preview_observer_execution_receipt_and_release_are_not_accepted",
        "prod4_private_data_is_not_create_once_staged_and_digest_verified",
        "prod4_exact_image_CPU_preflight_and_Jobs_API_preview_are_not_recorded",
        "fresh_Jobs_Kubernetes_SFS_and_WandB_absence_checks_are_not_complete_or_encoded_at_submit",
        "prod4_authoritative_reward_optimizer_checkpoint_and_UID_release_receipt_is_absent",
        "full_arm_exact_staged_manifests_training_image_validator_plan_request_preflight_and_preview_are_absent",
        "full_arm_legal_episode_ceilings_exceed_the_shared_eight_hour_watchdog_bound",
        "full_arm_independent_UID_bound_release_observers_are_not_recorded",
    ]
    return seal(
        {
            "schema": "cyber_qwen38_skyrl_launch_readiness_audit_v1",
            "status": "not_launch_ready_failure_budget_closed_and_gates_absent",
            "scope": (
                "offline_no_submit_no_launch_no_cancel_no_private_logs_no_external_reads_or_writes"
            ),
            "source_queue": source(NEXT_GATES),
            "external_activity": {
                "Jobs_API_requests": 0,
                "Kubernetes_requests": 0,
                "WandB_requests": 0,
                "cluster_mutations": 0,
                "private_logs_read": False,
            },
            "failure_budget": next_gates["failure_budget"],
            "v17_development_topology_gate": topology,
            "prod4_one_step_reward_gate": prod4,
            "full_c1_arms": production,
            "tooling_bindings": {
                "Jobs_client": source(ROOT / "cyber_post_train/jobs.py"),
                "CLI": source(ROOT / "cyber_post_train/cli.py"),
                "RL_runtime": source(ROOT / "training/rl_runtime.py"),
                "SkyRL_training": source(ROOT / "training/skyrl_training.py"),
                "watchdog": source(ROOT / "training/sft_runtime.py"),
            },
            "dry_run_commands": [
                (
                    "uv run --locked cyber-post-train rl-topology-probe "
                    "configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json "
                    "--output <new-temporary-directory>"
                ),
                "uv run --locked python scripts/prepare_qwen38_skyrl_production_queue.py --check",
                "uv run --locked python scripts/prepare_qwen38_skyrl_posttrain_packets.py --check",
                ("uv run --locked python scripts/audit_qwen38_skyrl_launch_readiness.py --check"),
            ],
            "blockers": blockers,
            "launch_authorized": False,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    value = build()
    if args.write:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        AUDIT.write_bytes(raw(value))
    elif not AUDIT.exists() or AUDIT.read_bytes() != raw(value):
        raise SystemExit("SkyRL launch-readiness audit is stale")
    print(
        json.dumps(
            {
                "audit": str(AUDIT.relative_to(ROOT)),
                "sha256": value["sha256"],
                "launch_authorized": False,
                "external_activity": value["external_activity"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
