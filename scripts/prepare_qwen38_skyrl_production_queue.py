#!/usr/bin/env python3
"""Build the sealed, no-submit Qwen3.8 SkyRL production experiment queue.

This command is deliberately local and deterministic.  It reads only committed
study inputs, writes no private task text, and never contacts Fleet or the Jobs
API.  ``--check`` is the CI/review mode; ``--write`` materializes the reviewed
JSON artifacts after an intentional source change.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest import mock

from cyber_post_train.jobs import digest as request_digest
from training import sft, skyrl_production, skyrl_production_training

ROOT = Path(__file__).resolve().parents[1]
ELIGIBLE = ROOT / "configs/data/qwen-blackbox-eligible-v1.json"
INVENTORY = ROOT / "configs/data/qwen-blackbox-study-inventory-v1.json"
SOURCE_SPLIT = ROOT / "configs/data/qwen-blackbox-study-split-b-v1.json"
TOOL_CATALOG = ROOT / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"
DECISION_SPACE = ROOT / "docs/TRAINING_DECISION_SPACE.md"

TASK_SET = ROOT / "configs/data/qwen38-skyrl-production-task-set-v1.json"
SPLIT = ROOT / "configs/data/qwen38-skyrl-production-split-v1.json"
QUALIFICATION = ROOT / "configs/qualification/qwen38-skyrl-production-queue-v1.json"
STAGING_PACKET = ROOT / "configs/qualification/qwen38-skyrl-production-data-staging-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-20-skyrl-production-experiment-queue-v1.json"

ARMS = (
    {
        "id": "a1",
        "priority": 1,
        "name": "chris-q38-skyrl10-a1",
        "steps": 10,
        "lr": 1e-6,
        "seed": 42,
        "treatment": "anchor",
        "rationale": "Ten-update anchor at the learning rate already used by the one-step canary.",
    },
    {
        "id": "lr3e7",
        "priority": 2,
        "name": "chris-q38-skyrl10-lr3e7",
        "steps": 10,
        "lr": 3e-7,
        "seed": 42,
        "treatment": "learning_rate",
        "rationale": (
            "Lower geometric learning-rate boundary; every other scientific control matches a1."
        ),
    },
    {
        "id": "lr3e6",
        "priority": 3,
        "name": "chris-q38-skyrl10-lr3e6",
        "steps": 10,
        "lr": 3e-6,
        "seed": 42,
        "treatment": "learning_rate",
        "rationale": (
            "Higher geometric learning-rate boundary; every other scientific control matches a1."
        ),
    },
    {
        "id": "seed43",
        "priority": 4,
        "name": "chris-q38-skyrl10-seed43",
        "steps": 10,
        "lr": 1e-6,
        "seed": 43,
        "treatment": "random_seed",
        "rationale": (
            "Independent anchor repeat to distinguish a stable effect from one stochastic run."
        ),
    },
    {
        "id": "dose50",
        "priority": 5,
        "name": "chris-q38-skyrl50-a1",
        "steps": 50,
        "lr": 1e-6,
        "seed": 42,
        "treatment": "optimizer_update_dose",
        "rationale": (
            "Medium-dose continuation of a1; launch only after the ten-update anchor remains "
            "healthy."
        ),
    },
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sealed(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = "sha256:" + digest(result)
    return result


def load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def source_file(path: Path) -> dict:
    result = {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if path.suffix == ".json":
        result["self_sha256"] = load(path).get("sha256")
    return result


def build_task_inputs() -> tuple[dict, dict]:
    eligible, inventory, source_split = map(load, (ELIGIBLE, INVENTORY, SOURCE_SPLIT))
    eligible_rows = {
        (row["task_key"], row["task_version_id"]): row for row in eligible["task_versions"]
    }
    inventory_rows = {(row["task_key"], row["task_version_id"]): row for row in inventory["tasks"]}
    split_rows = {(row["task_key"], row["task_version_id"]): row for row in source_split["tasks"]}
    if not eligible_rows.keys() == inventory_rows.keys() == split_rows.keys():
        raise ValueError("eligible inventory and frozen split identities differ")

    tasks, assignments = [], []
    split_map = {"train": "train", "dev": "dev", "final_test": "test"}
    for key in sorted(eligible_rows):
        row, taxonomy = eligible_rows[key], inventory_rows[key]["taxonomy"]
        if any(taxonomy[field]["status"] != "verified" for field in ("application", "task_family")):
            raise ValueError("production RL lineage must be verified")
        environment = row["environment"]
        tasks.append(
            {
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "env_key": environment["id"],
                "env_version": environment["version"],
                "environment_version_id": environment["version_id"],
                "data_key": environment["data_id"],
                "data_version": environment["data_version"],
                "lineage": {
                    "application": taxonomy["application"]["value"],
                    "task_family": taxonomy["task_family"]["value"],
                },
            }
        )
        assignments.append(
            {
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "split": split_map[split_rows[key]["split"]],
                "reference_session_id": None,
            }
        )

    task_set = sealed(
        {
            "schema": "cyber_rl_task_set_v1",
            "purpose": (
                "Broad Qwen3.8 SkyRL production search over exact Fleet blackbox task versions; "
                "prompts remain private and are fetched only by the GET-only data builder"
            ),
            "source_manifest_sha256": "sha256:" + eligible["sha256"],
            "tool_catalog_sha256": (
                "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
            ),
            "training_data_eligible": True,
            "reward_signal_provenance": {
                "kind": "execution_proven_exact_current_bindings",
                "eligible_manifest": source_file(ELIGIBLE),
                "inventory": source_file(INVENTORY),
                "source_split": source_file(SOURCE_SPLIT),
                "rule": (
                    "Each task version passed the committed current-binding and execution-evidence "
                    "audit. Historical outcomes are eligibility evidence only and are not rewards."
                ),
            },
            "tasks": tasks,
        }
    )
    split = sealed(
        {
            "schema": "cyber_task_split_v1",
            "purpose": (
                "SkyRL search split preserving Split B exactly: 59 train, 20 development, "
                "and 10 untouched final-test task families"
            ),
            "parent_split_sha256": source_split["sha256"],
            "tasks": assignments,
        }
    )
    counts = {
        name: sum(row["split"] == name for row in assignments) for name in ("train", "dev", "test")
    }
    if counts != {"train": 59, "dev": 20, "test": 10}:
        raise ValueError("broad production split counts changed")
    return task_set, split


def data_config(arm: dict) -> dict:
    root = f"/mnt/sfs/jobs/chris-q38-study-corpora-v1/{arm['name']}-inputs-v1/data"
    return {
        "name": arm["name"],
        "backend": "skyrl",
        "task_set": "../data/qwen38-skyrl-production-task-set-v1.json",
        "split": "../data/qwen38-skyrl-production-split-v1.json",
        "tool_catalog": "../data/qwen38-rl-filtered-canary-tool-catalog-v1.json",
        "model_lock": "../models/qwen38-27b-1d4bf0f2.lock.json",
        "model_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "output": root,
        "limits": {
            "context_tokens": 98304,
            "response_tokens": 81920,
            "max_tokens_per_turn": 4096,
            "max_turns": 600,
            "episode_seconds": 2400,
            "tool_seconds": 330,
            "tool_result_chars": 50000,
        },
    }


def run_config(arm: dict) -> dict:
    data_root = f"/mnt/sfs/jobs/chris-q38-study-corpora-v1/{arm['name']}-inputs-v1/data"
    return {
        "backend": "skyrl",
        "name": arm["name"],
        "output_root": f"/mnt/sfs/jobs/{arm['name']}",
        "model": {
            "lock": "../models/qwen38-27b-1d4bf0f2.lock.json",
            "weights": "../models/qwen38-27b-1d4bf0f2.weights.json",
            "root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        },
        "data": {"manifest": data_root + "/manifest.json", "root": data_root},
        "recipe": {
            "nodes": 1,
            "steps": arm["steps"],
            "groups": 1,
            "samples_per_prompt": 8,
            "lr": arm["lr"],
            "eval_interval": 10,
            "checkpoint_interval": 10,
            "keep_checkpoints": 2,
            "seed": arm["seed"],
        },
        "wandb": {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": arm["name"],
        },
        "cluster": {
            "priority": "c1",
            "target": "prod",
            "resources": {
                "cpu_request": "64",
                "cpu_limit": "64",
                "memory_request": "512Gi",
                "memory_limit": "768Gi",
            },
        },
        # The current compiler recognizes only the broad-queue qualification.
        # Carrying this path makes an accidental early compile fail closed.  A
        # post-prod8 release change must add and test the production validator;
        # it may not silently drop this binding.
        "qualification": "../qualification/qwen38-skyrl-production-queue-v1.json",
    }


def path_for(kind: str, arm: dict) -> Path:
    if kind == "data":
        return ROOT / f"configs/qualification/qwen38-skyrl-production-data-{arm['id']}-v1.json"
    if kind in {"manifest", "plan", "preview", "observer"}:
        return ROOT / (f"configs/qualification/qwen38-skyrl-production-{kind}-{arm['id']}-v1.json")
    return ROOT / f"configs/runs/qwen38-skyrl-production-{arm['id']}-v1.json"


def build() -> dict[Path, dict]:
    task_set, split = build_task_inputs()
    staging = load(STAGING_PACKET)
    if (
        staging.get("schema") != "cyber_qwen38_skyrl_production_data_staging_v1"
        or staging.get("sha256")
        != "sha256:" + digest({key: value for key, value in staging.items() if key != "sha256"})
        or staging.get("state") != "locally_built_not_staged"
        or staging.get("external_mutations") != 0
        or [arm.get("name") for arm in staging.get("arms", [])] != [arm["name"] for arm in ARMS]
        or task_set["sha256"] != skyrl_production.TASK_SET_SHA256
        or split["sha256"] != skyrl_production.SPLIT_SHA256
    ):
        raise ValueError("private data staging packet differs from the reviewed queue")
    staged_by_name = {arm["name"]: arm for arm in staging["arms"]}
    if any(
        not isinstance(arm.get("sanitized_manifest"), dict)
        or arm["sanitized_manifest"].get("sha256") != arm["data_manifest_sha256"]
        for arm in staging["arms"]
    ):
        raise ValueError("private staging packet lacks its sanitized manifests")
    qualification = sealed(
        {
            "schema": "cyber_qwen38_skyrl_production_queue_v1",
            "profile": skyrl_production.PROFILE,
            "purpose": (
                "Exact offline-compiled queue behind the prod8 reward-acquisition canary; "
                "all external preview and submission gates remain closed."
            ),
            "execution": {
                "cluster_target": "prod",
                "jobs_api_base_url": "https://api.ft.flt.build",
                "kubernetes_context": skyrl_production.PROD_CONTEXT,
                "namespace": skyrl_production.NAMESPACE,
                "image": skyrl_production.IMAGE,
                "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
            },
            "canary_prerequisite": {
                "run_name": "chris-q38-rlreward-prod8",
                "required": [
                    "eight_real_nontruncated_rollouts",
                    "one_authoritative_verifier_execution_id_per_rollout",
                    "reward_max_strictly_greater_than_reward_min",
                    "one_finite_nonzero_optimizer_update",
                    "reloadable_step_1_checkpoint",
                    "all_owned_gpu_resources_released",
                ],
            },
            "fixed_controls": copy.deepcopy(skyrl_production.FIXED_CONTROLS),
            "submission_gate": {
                "preview_authorized": False,
                "submission_authorized": False,
                "blockers": [
                    "prod8_terminal_acceptance_receipt_absent",
                    "prod8_step1_to_step2_native_resume_not_yet_qualified",
                    "broad_horizon_not_yet_ported_to_prod8_long_context_contract",
                    "broad_get_only_data_manifests_not_yet_create_once_staged_on_SFS",
                    "exact_image_cpu_preflights_not_recorded",
                    "jobs_api_server_previews_and_fresh_guard_receipts_not_recorded",
                    "uid_bound_release_observers_not_yet_armed",
                ],
            },
            "release_observer": {
                "contract_schema": skyrl_production.RELEASE_CONTRACT_SCHEMA,
                "armed_receipt_schema": "cyber_skyrl_release_observer_armed_v1",
                "arm_before_jobs_post": True,
                "bind_server_run_and_kubernetes_uids_after_post": True,
                "release_requires_bound_descendants_absent_and_active_gpus_zero": True,
                "peer_workload_mutation_authorized": False,
            },
            "production_arms": [
                {
                    "id": arm["id"],
                    "dispatch_order": arm["priority"],
                    "name": arm["name"],
                    "recipe": {
                        "nodes": 1,
                        "steps": arm["steps"],
                        "groups": 1,
                        "samples_per_prompt": 8,
                        "lr": arm["lr"],
                        "eval_interval": 10,
                        "checkpoint_interval": 10,
                        "keep_checkpoints": 2,
                        "seed": arm["seed"],
                    },
                    "manifest": copy.deepcopy(staged_by_name[arm["name"]]["sanitized_manifest"]),
                    "watchdog": copy.deepcopy(skyrl_production.WATCHDOGS[arm["steps"]]),
                    "staged_data": {
                        "root": staged_by_name[arm["name"]]["create_once_target"],
                        "candidate_manifest_path": str(path_for("manifest", arm).relative_to(ROOT)),
                        "manifest_file_sha256": staged_by_name[arm["name"]][
                            "data_manifest_file_sha256"
                        ],
                        "manifest_self_sha256": staged_by_name[arm["name"]]["data_manifest_sha256"],
                        "files": copy.deepcopy(staged_by_name[arm["name"]]["files"]),
                        "state": "local_private_candidate_not_SFS_staged",
                    },
                }
                for arm in ARMS
            ],
            "private_data": {
                "staging_packet": source_file(STAGING_PACKET),
                "state": staging["state"],
                "external_mutations": staging["external_mutations"],
                "fleet_reads": staging["fleet_reads"],
                "runtime_ownership": staging["runtime_ownership"],
                "arms": [
                    {
                        "name": arm["name"],
                        "data_manifest_sha256": arm["data_manifest_sha256"],
                        "data_manifest_file_sha256": arm["data_manifest_file_sha256"],
                        "rows": arm["rows"],
                    }
                    for arm in staging["arms"]
                ],
            },
            "research_basis": [
                {
                    "source": "https://arxiv.org/abs/2402.03300",
                    "observed": (
                        "DeepSeekMath reports GRPO policy learning rate 1e-6, 64 samples per "
                        "question, KL coefficient 0.04, and one update per exploration stage."
                    ),
                    "inference": (
                        "Use 1e-6 as the anchor and bracket it geometrically at 3e-7 and 3e-6; "
                        "retain one native update per fresh batch."
                    ),
                    "transfer_warning": (
                        "The paper studies a 7B math model with 1,024-token answers, not a 27B "
                        "tool-using cyber agent with long episodes."
                    ),
                },
                {
                    "source": "https://www.jmlr.org/papers/v18/16-558.html",
                    "observed": (
                        "Hyperband allocates a small resource budget broadly and increases the "
                        "budget only for candidates retained by successive halving."
                    ),
                    "inference": (
                        "Screen the anchor and learning-rate boundaries for ten updates before "
                        "spending fifty updates on the retained anchor treatment."
                    ),
                    "transfer_warning": (
                        "This queue uses a predeclared staged design, not the complete Hyperband "
                        "algorithm, and Fleet task success rather than training loss selects arms."
                    ),
                },
            ],
            "local_design_basis": source_file(DECISION_SPACE),
            "successive_halving": {
                "short_search_steps": 10,
                "medium_dose_steps": 50,
                "dose50_extra_gate": (
                    "a1 must have finite metrics, valid rewards, a reloadable step-10 checkpoint, "
                    "and no development regression large enough to stop the arm"
                ),
                "selection_inputs": [
                    "Fleet development task outcomes",
                    "reward variation and invalid-attempt rate",
                    "finite loss, KL, entropy, and gradient/update telemetry",
                    "checkpoint reload evidence",
                ],
                "external_benchmark_selection_forbidden": True,
            },
        }
    )
    artifacts: dict[Path, dict] = {TASK_SET: task_set, SPLIT: split, QUALIFICATION: qualification}
    for arm in ARMS:
        artifacts[path_for("data", arm)] = data_config(arm)
        artifacts[path_for("run", arm)] = run_config(arm)
        artifacts[path_for("manifest", arm)] = copy.deepcopy(
            staged_by_name[arm["name"]]["sanitized_manifest"]
        )
    skyrl_production.validate_qualification(qualification)
    for arm in ARMS:
        plan, request = compile_arm(artifacts, arm)
        artifacts[path_for("plan", arm)] = plan
        artifacts[path_for("preview", arm)] = skyrl_production.offline_preview(plan, request)
        artifacts[path_for("observer", arm)] = skyrl_production.release_observer_contract(
            plan, request
        )
    return artifacts


def raw(value: dict) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def compile_arm(artifacts: dict[Path, dict], arm: dict) -> tuple[dict, dict]:
    """Compile one exact no-submit plan from its sanitized staged manifest."""
    run_path = path_for("run", arm)
    run = artifacts[run_path]
    manifest_path = Path(run["data"]["manifest"])
    original = sft.read_mapping

    def read(path: Path) -> dict:
        candidate = Path(path)
        if candidate == manifest_path:
            return copy.deepcopy(artifacts[path_for("manifest", arm)])
        if candidate.resolve() == QUALIFICATION.resolve():
            return copy.deepcopy(artifacts[QUALIFICATION])
        return original(candidate)

    with mock.patch.object(sft, "read_mapping", side_effect=read):
        plan = skyrl_production_training.compile_rl(run, relative_to=run_path.parent)
    request = skyrl_production_training.job_request(plan)
    return plan, request


def queue_evidence(artifacts: dict[Path, dict]) -> dict:
    qualification = artifacts[QUALIFICATION]
    staging = load(STAGING_PACKET)
    staged_by_name = {arm["name"]: arm for arm in staging["arms"]}
    arms = []
    for arm in ARMS:
        data_path, run_path = path_for("data", arm), path_for("run", arm)
        manifest_path = path_for("manifest", arm)
        plan_path = path_for("plan", arm)
        preview_path = path_for("preview", arm)
        observer_path = path_for("observer", arm)
        plan, request = compile_arm(artifacts, arm)
        if plan != artifacts[plan_path]:
            raise ValueError("offline SkyRL production plan is not reproducible")
        arms.append(
            {
                **arm,
                "data_config": str(data_path.relative_to(ROOT)),
                "data_config_file_sha256": "sha256:"
                + hashlib.sha256(raw(artifacts[data_path])).hexdigest(),
                "run_config": str(run_path.relative_to(ROOT)),
                "run_config_file_sha256": "sha256:"
                + hashlib.sha256(raw(artifacts[run_path])).hexdigest(),
                "local_data_manifest_sha256": staged_by_name[arm["name"]]["data_manifest_sha256"],
                "local_data_manifest_file_sha256": staged_by_name[arm["name"]][
                    "data_manifest_file_sha256"
                ],
                "sanitized_manifest": {
                    "path": str(manifest_path.relative_to(ROOT)),
                    "self_sha256": artifacts[manifest_path]["sha256"],
                    "file_sha256": "sha256:"
                    + hashlib.sha256(raw(artifacts[manifest_path])).hexdigest(),
                    "cluster_staged": False,
                },
                "plan": {
                    "path": str(plan_path.relative_to(ROOT)),
                    "sha256": "sha256:" + request_digest(plan),
                    "file_sha256": "sha256:"
                    + hashlib.sha256(raw(artifacts[plan_path])).hexdigest(),
                },
                "request_sha256": "sha256:" + request_digest(request),
                "offline_preview": {
                    "path": str(preview_path.relative_to(ROOT)),
                    "self_sha256": artifacts[preview_path]["sha256"],
                    "server_preview_requested": False,
                },
                "release_observer_contract": {
                    "path": str(observer_path.relative_to(ROOT)),
                    "self_sha256": artifacts[observer_path]["sha256"],
                    "armed": False,
                },
                "plan_request_state": "exact_offline_plan_and_request_digest_compiled",
                "submitted": False,
            }
        )
    return sealed(
        {
            "schema": "cyber_qwen38_skyrl_production_experiment_queue_evidence_v1",
            "regenerated_at": "2026-09-21T09:00:00Z",
            "source_base_commit": "40d55fae71510e04ed0f5a1c8c820ce34b7f612c",
            "historical_origin": {
                "observed_at": "2026-09-20T17:19:29Z",
                "git_branch": "codex/q38-skyrl-full-queue-v1",
                "current_plan_or_request_binding": False,
            },
            "scope": "offline_no_submit_no_stage_no_serve_no_cancel",
            "qualification": {
                "path": str(QUALIFICATION.relative_to(ROOT)),
                "self_sha256": qualification["sha256"],
                "file_sha256": "sha256:" + hashlib.sha256(raw(qualification)).hexdigest(),
            },
            "data": {
                "task_set": {
                    "path": str(TASK_SET.relative_to(ROOT)),
                    "self_sha256": artifacts[TASK_SET]["sha256"],
                    "file_sha256": "sha256:" + hashlib.sha256(raw(artifacts[TASK_SET])).hexdigest(),
                },
                "split": {
                    "path": str(SPLIT.relative_to(ROOT)),
                    "self_sha256": artifacts[SPLIT]["sha256"],
                    "file_sha256": "sha256:" + hashlib.sha256(raw(artifacts[SPLIT])).hexdigest(),
                },
                "counts": {"train": 59, "dev": 20, "test_untouched": 10},
                "private_task_text_committed": False,
                "local_private_build": {
                    "packet": source_file(STAGING_PACKET),
                    "state": staging["state"],
                    "fleet_reads": staging["fleet_reads"],
                    "runtime_ownership": staging["runtime_ownership"],
                    "repository_payload_bytes": staging["privacy"]["repository_payload_bytes"],
                },
            },
            "arms": arms,
            "common_release_gate": qualification["canary_prerequisite"],
            "blockers": qualification["submission_gate"]["blockers"],
            "duplicate_output_checks": {
                "state": "caller_guard_encoded_but_must_execute_fresh_before_submit",
                "caller_guard": {
                    "jobs_history": True,
                    "kubernetes_name_and_output": True,
                    "sfs_staged_tree_and_output": True,
                    "wandb_run_id": True,
                    "maximum_guard_seconds": 120,
                    "runs_after_server_preview_immediately_before_create": True,
                },
                "observed_at": "2026-09-20T17:10:39Z",
                "jobs_api": {
                    "method": "GET_only",
                    "history_rows": 934,
                    "matching_name_or_output_rows": 0,
                },
                "kubernetes": {
                    "context": "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6",
                    "namespace": "fleet-train-jobs",
                    "kinds": ["RayJob", "RayCluster", "Job", "Pod", "Workload"],
                    "matching_objects": 0,
                },
                "sfs": "not_observable_from_this_host",
                "wandb": "not_observable_without_a_local_read_credential",
                "reason": (
                    "The encoded caller guard must recheck all four destinations; this "
                    "historical observation is not reusable."
                ),
            },
            "next_actions_after_gate": [
                "Accept prod8 only after every common release-gate fact is independently verified.",
                (
                    "Qualify native step-1 to step-2 continuation and port the broad queue to the "
                    "prod8 262,144-token compaction and output-limit contract before launch."
                ),
                (
                    "Create-once stage and digest-verify each arm's manifest, train JSONL, dev "
                    "JSONL, split, and task set against the committed sanitized manifest."
                ),
                "Recompile the immutable plan/request and run exact-image CPU preflight.",
                (
                    "Recheck Jobs, Kubernetes, SFS, and W&B duplicates/absence and validate the "
                    "server preview."
                ),
                "Submit each eligible c1 arm at most once after every remaining gate passes.",
            ],
            "external_mutations": 0,
        }
    )


def expected() -> dict[Path, dict]:
    artifacts = build()
    artifacts[EVIDENCE] = queue_evidence(artifacts)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    args = parser.parse_args()
    artifacts = expected()
    if args.write:
        for path, value in artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw(value))
        print(json.dumps({"written": len(artifacts), "external_mutations": 0}, sort_keys=True))
        return
    changed = [
        str(path.relative_to(ROOT))
        for path, value in artifacts.items()
        if not path.is_file() or path.read_bytes() != raw(value)
    ]
    if changed:
        raise SystemExit("queue artifacts differ: " + ", ".join(changed))
    print(json.dumps({"checked": len(artifacts), "external_mutations": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
