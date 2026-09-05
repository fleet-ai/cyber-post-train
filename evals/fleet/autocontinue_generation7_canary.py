"""Generation-7 successors for terminal generation-6 CLI-volume evictions."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation6_canary as generation6
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-opencode-autocontinue-generation7-canary-spec-v1"
FAILURE_SCHEMA = "fleet-opencode-autocontinue-generation6-cli-volume-failure-v1"
FAILURE_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-autocontinue-generation6-cli-volume-failure-v1.json"
)
DOCKER_IMAGE = (
    "docker.io/library/docker@sha256:"
    "f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
MODULE_PATH = "evals/fleet/autocontinue_generation7_canary.py"
INTENT_NAME = "chris-ac-g7-canary-submit-v1"
G7_SPEC_PATHS = {
    "qwen3.8-27b": "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation7-v1.json",
    "glm-5.3": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation7-v1.json",
}
EXPECTED = {
    "qwen3.8-27b": {
        "job_name": "chris-q38-ac-r004-a1-g7-v1",
        "configmap_name": "chris-q38-ac-r004-a1-g7-run-v1",
        "run_id": "chris-q38-ac-g7-r004-a1-77deaade",
        "network": "q38-ac-g7-r004-a1-77deaade",
        "generation6_job_uid": "674ac8ec-0d20-49e0-a260-b77d96e9cd81",
        "generation6_pod_uid": "2a2e94ed-c86d-4159-80d5-45ef94f89bf8",
    },
    "glm-5.3": {
        "job_name": "chris-glm53-ac-r013-a1-g7-v1",
        "configmap_name": "chris-glm53-ac-r013-a1-g7-run-v1",
        "run_id": "chris-glm53-ac-g7-r013-a1-b6338d53",
        "network": "glm53-ac-g7-r013-a1-b6338d53",
        "generation6_job_uid": "859378a9-b2ce-4edc-9b25-18d701201e5f",
        "generation6_pod_uid": "ea288ab8-aea8-45b4-84b8-7d011a2a5e8f",
    },
}


def load(path: Path) -> dict[str, Any]:
    return generation6.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return generation6.digest(value, field)


def file_sha256(path: Path) -> str:
    return generation6.file_sha256(path)


def validate_failure(receipt: dict[str, Any], root: Path) -> None:
    expected_jobs = {
        model: {
            "model": model,
            "job_name": generation6.EXPECTED[model]["job_name"],
            "job_uid": EXPECTED[model]["generation6_job_uid"],
            "pod_uid": EXPECTED[model]["generation6_pod_uid"],
            "job_terminal_condition": "Failed",
            "job_terminal_reason": "BackoffLimitExceeded",
            "pod_phase": "Failed",
            "pod_reason": "Evicted",
            "docker_cli_init_exit_code": 0,
            "evaluator_exit_code": 137,
            "dind_exit_code": 137,
            "aggregate_restart_count": 2,
            "generation6_output_root_absent": True,
            "generation6_execution_claim_absent": True,
            "exact_treatment_sessions": 0,
        }
        for model in EXPECTED
    }
    expected = {
        "schema_version": FAILURE_SCHEMA,
        "append_only": True,
        "status": "SEALED_PRE_MODEL_INFRASTRUCTURE_TOMBSTONES",
        "observed_at_utc": "2026-09-05T11:23:21Z",
        "classification": "docker_cli_emptydir_capacity_eviction",
        "root_cause": {
            "runner_boundary": "docker_cli_shared_emptydir",
            "docker_cli_emptydir_size_limit": "100Mi",
            "docker_cli_emptydir_limit_bytes": 104857600,
            "exact_copied_binary_bytes": 105594160,
            "minimum_deficit_bytes": 736560,
            "successor_size_limit": "256Mi",
            "successor_size_limit_bytes": 268435456,
            "successor_headroom_bytes": 162841296,
            "kubernetes_eviction_message": "EmptyDir docker-cli exceeds 100Mi",
            "logs_used": False,
        },
        "jobs": [expected_jobs[model] for model in EXPECTED],
        "side_effects": {
            "model_calls": 0,
            "sessions": 0,
            "verifier_executions": 0,
            "generation6_claims": 0,
            "output_roots": 0,
        },
        "replacement": {
            "permitted_execution_generation": 7,
            "same_statistical_cells": True,
            "generation6_nonrepeatable": True,
            "fresh_execution_ids_required": True,
            "docker_cli_emptydir_minimum_bytes": 268435456,
        },
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        not (root / FAILURE_PATH).is_file()
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-6 CLI failure tombstone drifted")


def _render_plan(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-7 model is unsupported")
    g6_spec = load(root / generation6.G6_SPEC_PATHS[model])
    old_plan = generation6.validate_spec(g6_spec, root)
    expected = EXPECTED[model]
    plan = copy.deepcopy(old_plan)
    plan["campaign_id"] = expected["job_name"]
    plan["source_job_id"] = expected["job_name"]
    plan["preflight_job_name"] = expected["job_name"] + "-preflight"
    plan["scored_job_name"] = expected["job_name"]
    plan["sfs_root"] = expected["job_name"]
    plan["attempts"][0].update(
        {
            "run_id": expected["run_id"],
            "network": expected["network"],
            "execution_generation": 7,
        }
    )
    plan["execution"].update(
        {
            "retry_policy": "generation7_only_after_exact_generation6_pre_model_cli_tombstone",
            "required_priority_class": "fleet-serve-low",
            "execution_generation": 7,
            "old_claims_preserved": True,
            "launch_authorized": False,
            "docker_cli_source_image": DOCKER_IMAGE,
            "docker_cli_version": "27.5.1",
            "docker_cli_buildx_version": "0.20.1",
            "docker_cli_copied_binary_bytes": 105594160,
            "docker_cli_emptydir_size_limit_bytes": 268435456,
        }
    )
    plan["source"] = {
        "exact_pass4_campaign_path": old_plan["source"]["exact_pass4_campaign_path"],
        "statistical_cell_id": g6_spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "generation6_spec_sha256": g6_spec["generation6_spec_sha256"],
        "generation6_plan_sha256": old_plan["plan_sha256"],
        "generation6_failure_receipt_sha256": load(root / FAILURE_PATH)[
            "receipt_sha256"
        ],
    }
    plan["plan_sha256"] = digest(plan, "plan_sha256")
    return plan


def validate_spec(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-7 model is unsupported")
    validate_failure(load(root / FAILURE_PATH), root)
    g6_spec = load(root / generation6.G6_SPEC_PATHS[model])
    g6_plan = generation6.validate_spec(g6_spec, root)
    execution = exact.execution_for(g6_spec["statistical_cell"]["cell_id"], 7)
    expected = EXPECTED[model]
    expected_spec = {
        "schema_version": SPEC_SCHEMA,
        "model": model,
        "statistical_cell": g6_spec["statistical_cell"],
        "execution": execution,
        "predecessor_generation6": {
            "spec_path": generation6.G6_SPEC_PATHS[model],
            "spec_sha256": g6_spec["generation6_spec_sha256"],
            "plan_sha256": g6_plan["plan_sha256"],
            "job_uid": expected["generation6_job_uid"],
            "pod_uid": expected["generation6_pod_uid"],
        },
        "supersession": {
            "failure_path": FAILURE_PATH,
            "failure_receipt_sha256": load(root / FAILURE_PATH)["receipt_sha256"],
            "prior_generation": 6,
            "new_generation": 7,
            "same_statistical_cell": True,
            "generation6_nonrepeatable": True,
        },
        "identities": {
            "job_name": expected["job_name"],
            "configmap_name": expected["configmap_name"],
            "intent_configmap_name": INTENT_NAME,
            "run_id": expected["run_id"],
            "network": expected["network"],
            "sfs_root": f"/mnt/sfs/jobs/{expected['job_name']}",
            "generation_claim_root": generation6.generation5.CLAIM_ROOT,
        },
        "rendered_plan_sha256": spec.get("rendered_plan_sha256"),
        "launch_authorized": False,
    }
    if (
        spec.get("generation7_spec_sha256") != digest(spec, "generation7_spec_sha256")
        or {key: value for key, value in spec.items() if key != "generation7_spec_sha256"}
        != expected_spec
    ):
        raise ValueError("generation-7 specification drifted")
    plan = _render_plan(spec, root)
    if plan["plan_sha256"] != spec["rendered_plan_sha256"]:
        raise ValueError("generation-7 rendered plan digest drifted")
    settings = self_hosted.opencode_settings(plan)
    canonical = self_hosted.canonical_json(settings)
    if (
        plan["harness"]["settings_canonical_sha256"] != self_hosted.sha256(canonical)
        or plan["harness"]["settings_file_sha256"]
        != self_hosted.sha256(canonical + b"\n")
        or settings.get("compaction") != {"auto": True, "reserved": 20000}
        or "plugin" in settings
        or plan["execution"]["required_task_tools"] != ["bash", "submit_report"]
    ):
        raise ValueError("generation-7 treatment renderer drifted")
    return plan


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-failure", "validate-spec"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "validate-failure":
        validate_failure(load(args.repo / FAILURE_PATH), args.repo)
        return 0
    if not args.spec:
        parser.error("validate-spec requires --spec")
    validate_spec(load(args.spec), args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
