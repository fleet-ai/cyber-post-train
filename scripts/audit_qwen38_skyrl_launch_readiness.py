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
from training import (
    sft,
    skyrl,
    skyrl_production,
    skyrl_production_training,
    skyrl_reward_rayjob,
    skyrl_topology_probe,
    skyrl_training,
)
from training.sft_runtime import (
    WATCHDOG_DRAIN_SECONDS,
    WATCHDOG_HARD_SECONDS,
    WATCHDOG_IDLE_SECONDS,
    WATCHDOG_POLL_SECONDS,
    WATCHDOG_STARTUP_SECONDS,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/evidence/qwen38-study/2026-09-23-skyrl-launch-readiness-audit-v2.json"
NEXT_GATES = ROOT / "docs/evidence/qwen38-study/2026-09-20-skyrl-next-gates-queue-v1.json"
TOPOLOGY_CONFIG = ROOT / "configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json"
CANARY_DATA = ROOT / "configs/qualification/qwen38-rl-reward-canary-data-prod-v8.json"
CANARY_RUN = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod-v8.json"
CANARY_MANIFEST = ROOT / "configs/qualification/qwen38-rl-reward-canary-manifest-prod-v8.json"
CANARY_QUALIFICATION = ROOT / canary.FAST_UPDATE_3_QUALIFICATION_PATH
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


def prod8_metadata(run: dict, next_gates: dict) -> dict:
    """Load the current committed sanitized manifest surface, never task text."""
    del next_gates
    value = load(CANARY_MANIFEST)
    if value.get("name") != run.get("name"):
        raise ValueError("sanitized reward-canary manifest identity changed")
    return value


def compile_prod8(run: dict, metadata: dict) -> tuple[dict, dict]:
    """Exercise the real compiler against digests only, without touching SFS."""
    original = sft.read_mapping

    def read(path: Path) -> dict:
        if Path(path) == Path(run["data"]["manifest"]):
            return copy.deepcopy(metadata)
        return original(path)

    with mock.patch.object(sft, "read_mapping", side_effect=read):
        plan = skyrl_training.compile_rl(run, relative_to=CANARY_RUN.parent)
    return plan, skyrl_training.job_request(plan)


def episode_ceiling(
    run: dict,
    *,
    dev_rows: int,
    episode_seconds: int,
    watchdog: dict | None = None,
) -> dict:
    recipe = run["recipe"]
    concurrency = recipe["groups"] * recipe["samples_per_prompt"]
    eval_steps = {0, recipe["steps"]} | set(
        range(recipe["eval_interval"], recipe["steps"] + 1, recipe["eval_interval"])
    )
    dev_waves = math.ceil(dev_rows / concurrency)
    ceiling = (recipe["steps"] + len(eval_steps) * dev_waves) * episode_seconds
    watchdog = watchdog or {
        "poll_seconds": WATCHDOG_POLL_SECONDS,
        "startup_seconds": WATCHDOG_STARTUP_SECONDS,
        "idle_seconds": WATCHDOG_IDLE_SECONDS,
        "hard_seconds": WATCHDOG_HARD_SECONDS,
        "drain_seconds": WATCHDOG_DRAIN_SECONDS,
        "episode_ceiling_seconds": ceiling,
    }
    if watchdog["episode_ceiling_seconds"] != ceiling:
        raise ValueError("plan watchdog episode ceiling differs from the legal schedule")
    minimum = ceiling + watchdog["startup_seconds"] + watchdog["drain_seconds"]
    return {
        "episode_seconds": episode_seconds,
        "train_steps": recipe["steps"],
        "evaluation_points": len(eval_steps),
        "development_rows": dev_rows,
        "episode_concurrency": concurrency,
        "development_waves_per_evaluation": dev_waves,
        "legal_episode_ceiling_seconds_excluding_startup_and_optimization": ceiling,
        "watchdog": watchdog,
        "minimum_hard_seconds_without_shortening_episode": minimum,
        "fits_watchdog_hard_bound": watchdog["hard_seconds"] >= minimum,
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


def prod8_audit(next_gates: dict) -> dict:
    run = load(CANARY_RUN)
    metadata = prod8_metadata(run, next_gates)
    plan, request = compile_prod8(run, metadata)
    shared_budget = episode_ceiling(
        run,
        dev_rows=metadata["files"]["dev"]["rows"],
        episode_seconds=metadata["limits"]["episode_seconds"],
    )
    direct_budget = episode_ceiling(
        run,
        dev_rows=metadata["files"]["dev"]["rows"],
        episode_seconds=metadata["limits"]["episode_seconds"],
        watchdog={
            **shared_budget["watchdog"],
            "hard_seconds": skyrl_reward_rayjob.MAXIMUM_SECONDS,
        },
    )
    if (
        request["workers"] != 1
        or request["gpus_per_worker"] != 8
        or request["priority_class"] != "c1"
        or request["secrets"] != ["fleet-api", "wandb-api"]
        or request["env"]["WANDB_RUN_ID"] != run["name"]
        or request["env"]["WANDB_MODE"] != "online"
        or plan["model"]["revision"] != canary.MODEL["revision"]
        or plan["data"]["sha256"] != metadata["sha256"]
    ):
        raise ValueError("prod8 plan, request, resource, model, data, or W&B binding changed")
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
            "hard_seconds": skyrl_reward_rayjob.MAXIMUM_SECONDS,
            "drain_seconds": WATCHDOG_DRAIN_SECONDS,
            "child_SIGTERM_then_SIGKILL_seconds": 60,
            "episode_budget": direct_budget,
            "Jobs_API_preview_must_render_shutdown_after_job_finishes": True,
            "independent_UID_bound_release_observer_recorded": False,
        },
        "encoded_absence_checks": {
            "submission_journal_create_once": True,
            "Jobs_API_name_prefix_and_output_history": True,
            "CPU_preflight_output_absence": True,
            "runtime_new_checkpoint_and_episode_trees": True,
            "fresh_Kubernetes_name_absence_encoded": True,
            "fresh_SFS_manifest_payload_and_output_absence_encoded": True,
            "fresh_WandB_run_ID_absence_encoded": True,
            "caller_guard_runs_after_preview_before_create": True,
            "maximum_guard_seconds": 120,
            "fresh_guard_executed": False,
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
        manifest_path = queue.path_for("manifest", arm)
        plan_path = queue.path_for("plan", arm)
        preview_path = queue.path_for("preview", arm)
        observer_path = queue.path_for("observer", arm)
        run, data = load(run_path), load(data_path)
        plan = expected[plan_path]
        compiled, request = queue.compile_arm(expected, arm)
        preview = expected[preview_path]
        observer = expected[observer_path]
        if (
            plan != compiled
            or preview != skyrl_production.offline_preview(plan, request)
            or observer != skyrl_production.release_observer_contract(plan, request)
            or plan["schema"] != skyrl_production_training.SCHEMA
            or request["workers"] != 1
            or request["gpus_per_worker"] != 8
            or request["priority_class"] != "c1"
            or request["image"] != skyrl_production.IMAGE
            or plan["data"] != expected[manifest_path]
            or plan["watchdog"] != skyrl_production.WATCHDOGS[arm["steps"]]
        ):
            raise ValueError(f"production arm {arm['id']} plan/request closure changed")
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
                "sanitized_manifest": source(manifest_path),
                "plan": source(plan_path),
                "plan_sha256": "sha256:" + digest(plan),
                "request_sha256": "sha256:" + digest(request),
                "offline_preview": source(preview_path),
                "server_preview_recorded": False,
                "plan_state": "exact_offline_plan_compiled_not_live_preflighted",
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
                    "sanitized_manifest_bound": True,
                    "SFS_staged": False,
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
                    watchdog=plan["watchdog"],
                ),
                "fresh_absence_guard": {
                    "encoded": True,
                    "executed": False,
                    "checks": [
                        "Jobs_history_name_and_output",
                        "Kubernetes_name_and_output",
                        "SFS_staged_tree_and_output",
                        "WandB_run_ID",
                    ],
                    "maximum_seconds": 120,
                },
                "release_observer": {
                    "contract": source(observer_path),
                    "arm_before_POST": observer["arm_before_jobs_post"],
                    "UID_fields": observer["bind_after_jobs_response"],
                    "armed": False,
                    "release_receipt_recorded": False,
                },
                "ready": False,
            }
        )
    if (
        len({arm["name"] for arm in arms}) != 5
        or len({arm["wandb"]["run_id"] for arm in arms}) != 5
    ):
        raise ValueError("production create-once identities are not unique")
    return {
        "state": "offline_plans_validators_previews_and_observer_contracts_complete",
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
        "all_legal_episode_ceilings_fit_plan_watchdogs": all(
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
        raise ValueError("historical next-gates receipt no longer matches its frozen record")
    topology = topology_audit(qualification)
    prod8 = prod8_audit(next_gates)
    production = production_arms_audit()
    blockers = [
        "prod8_private_data_is_not_create_once_staged_and_digest_verified",
        "prod8_exact_image_output_limit_CPU_preflight_and_server_previews_are_not_recorded",
        "fresh_prod8_Jobs_Kubernetes_SFS_and_WandB_absence_guard_is_not_executed",
        "prod8_authoritative_reward_optimizer_checkpoint_and_UID_release_receipt_is_absent",
        "prod8_step1_to_step2_native_resume_is_not_qualified",
        "full_arm_prod8_prerequisite_is_not_accepted",
        "full_arm_horizon_is_not_ported_to_the_prod8_long_context_contract",
        "full_arm_private_data_is_not_create_once_staged_and_verified_on_SFS",
        "full_arm_exact_image_CPU_preflights_and_server_previews_are_not_recorded",
        "full_arm_fresh_absence_guards_and_UID_bound_release_observers_are_not_executed",
    ]
    return seal(
        {
            "schema": "cyber_qwen38_skyrl_launch_readiness_audit_v2",
            "status": "offline_repairs_complete_external_execution_gates_pending",
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
            "failure_policy": {
                "numeric_failure_budget": None,
                "historical_source_receipt_preserved": True,
                "historical_source_receipt_is_not_a_current_submission_gate": True,
                "failures_require_evidence_repair_and_resource_release": True,
            },
            "legacy_v17_development_topology_evidence": {
                **topology,
                "gating": False,
                "reason": (
                    "The prod8 direct authorization path does not consume a V17 receipt; live "
                    "prod6/prod7 execution already exercised the one-node topology."
                ),
            },
            "prod8_one_step_reward_gate": prod8,
            "full_c1_arms": production,
            "tooling_bindings": {
                "Jobs_client": source(ROOT / "cyber_post_train/jobs.py"),
                "CLI": source(ROOT / "cyber_post_train/cli.py"),
                "RL_runtime": source(ROOT / "training/rl_runtime.py"),
                "SkyRL_training": source(ROOT / "training/skyrl_training.py"),
                "SkyRL_production_training": source(ROOT / "training/skyrl_production_training.py"),
                "SkyRL_production_validator": source(ROOT / "training/skyrl_production.py"),
                "SkyRL_caller_guard": source(ROOT / "training/skyrl_launch_guard.py"),
                "watchdog": source(ROOT / "training/sft_runtime.py"),
            },
            "dry_run_commands": [
                (
                    "uv run --locked cyber-post-train rl-topology-probe "
                    "configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json "
                    "--output <new-temporary-directory>"
                ),
                "uv run --locked python scripts/prepare_qwen38_skyrl_production_queue.py --check",
                (
                    "uv run --locked cyber-post-train rl "
                    "configs/runs/qwen38-skyrl-production-a1-v1.json "
                    "--output <new-temporary-directory>"
                ),
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
