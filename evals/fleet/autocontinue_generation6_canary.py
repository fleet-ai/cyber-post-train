"""Generation-6 successors for the terminal generation-5 CLI bootstrap failures."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation5_canary as generation5
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-opencode-autocontinue-generation6-canary-spec-v1"
FAILURE_SCHEMA = "fleet-opencode-autocontinue-generation5-cli-failure-v1"
FAILURE_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-autocontinue-generation5-cli-failure-v1.json"
)
EVALUATOR_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
    "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)
DOCKER_IMAGE = (
    "docker.io/library/docker@sha256:"
    "f649ef046008ca7f926a2571c32b0ac22e5c59eb61b959617f9acc2a4c638cf5"
)
MODULE_PATH = "evals/fleet/autocontinue_generation6_canary.py"
INTENT_NAME = "chris-ac-g6-canary-submit-v1"
G6_SPEC_PATHS = {
    "qwen3.8-27b": "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation6-v1.json",
    "glm-5.3": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation6-v1.json",
}
EXPECTED = {
    "qwen3.8-27b": {
        "job_name": "chris-q38-ac-r004-a1-g6-v1",
        "configmap_name": "chris-q38-ac-r004-a1-g6-run-v1",
        "run_id": "chris-q38-ac-g6-r004-a1-fa0c9b85",
        "network": "q38-ac-g6-r004-a1-fa0c9b85",
        "generation5_job_uid": "d9ea67ab-1e9e-4ef2-b878-20c9b1e965c7",
        "generation5_pod_uid": "7428a445-f47d-44d0-bf9e-0d6ea5c85913",
    },
    "glm-5.3": {
        "job_name": "chris-glm53-ac-r013-a1-g6-v1",
        "configmap_name": "chris-glm53-ac-r013-a1-g6-run-v1",
        "run_id": "chris-glm53-ac-g6-r013-a1-a63dbb28",
        "network": "glm53-ac-g6-r013-a1-a63dbb28",
        "generation5_job_uid": "3bdecde1-b95c-4773-be2e-3240216a8167",
        "generation5_pod_uid": "47e55a4e-7d3e-4f33-a4c8-6127665c6616",
    },
}


def load(path: Path) -> dict[str, Any]:
    return generation5.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return generation5.digest(value, field)


def file_sha256(path: Path) -> str:
    return generation5.file_sha256(path)


def validate_failure(receipt: dict[str, Any], root: Path) -> None:
    expected_jobs = {
        model: {
            "model": model,
            "job_name": generation5.EXPECTED[model]["job_name"],
            "job_uid": EXPECTED[model]["generation5_job_uid"],
            "pod_uid": EXPECTED[model]["generation5_pod_uid"],
            "job_terminal_condition": "Failed",
            "job_terminal_reason": "BackoffLimitExceeded",
            "pod_phase": "Failed",
            "evaluator_exit_code": 127,
            "evaluator_restarts": 0,
            "generation5_output_root_absent": True,
            "generation5_execution_claim_absent": True,
            "exact_treatment_sessions": 0,
        }
        for model in EXPECTED
    }
    expected = {
        "schema_version": FAILURE_SCHEMA,
        "append_only": True,
        "status": "SEALED_PRE_MODEL_INFRASTRUCTURE_TOMBSTONES",
        "observed_at_utc": "2026-09-05T09:45:00Z",
        "classification": "evaluator_image_missing_docker_cli",
        "root_cause": {
            "runner_boundary": "docker_info_readiness_loop",
            "hidden_command_not_found_attempts": 120,
            "terminal_shell_exit_code": 127,
            "evaluator_image": EVALUATOR_IMAGE,
            "evaluator_image_command_v_docker_exit_code": 127,
            "dind_image": DOCKER_IMAGE,
            "dind_image_docker_cli_path": "/usr/local/bin/docker",
            "dind_image_docker_cli_version": "27.5.1",
            "dind_image_buildx_version": "0.20.1",
            "logs_used": False,
        },
        "jobs": [expected_jobs[model] for model in EXPECTED],
        "side_effects": {
            "model_calls": 0,
            "sessions": 0,
            "verifier_executions": 0,
            "generation5_claims": 0,
            "output_roots": 0,
        },
        "replacement": {
            "permitted_execution_generation": 6,
            "same_statistical_cells": True,
            "generation5_nonrepeatable": True,
            "fresh_execution_ids_required": True,
            "pinned_cli_copy_required": True,
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
        raise ValueError("generation-5 CLI failure tombstone drifted")


def _render_plan(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-6 model is unsupported")
    g5_spec = load(root / generation5.G5_SPEC_PATHS[model])
    old_plan = generation5.validate_spec(g5_spec, root)
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
            "execution_generation": 6,
        }
    )
    plan["execution"].update(
        {
            "retry_policy": "generation6_only_after_exact_generation5_pre_model_cli_tombstone",
            "required_priority_class": "fleet-serve-low",
            "execution_generation": 6,
            "old_claims_preserved": True,
            "launch_authorized": False,
            "docker_cli_source_image": DOCKER_IMAGE,
            "docker_cli_version": "27.5.1",
        }
    )
    plan["source"] = {
        "exact_pass4_campaign_path": old_plan["source"]["exact_pass4_campaign_path"],
        "statistical_cell_id": g5_spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "generation5_spec_sha256": g5_spec["generation5_spec_sha256"],
        "generation5_plan_sha256": old_plan["plan_sha256"],
        "generation5_failure_receipt_sha256": load(root / FAILURE_PATH)[
            "receipt_sha256"
        ],
    }
    plan["plan_sha256"] = digest(plan, "plan_sha256")
    return plan


def validate_spec(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-6 model is unsupported")
    validate_failure(load(root / FAILURE_PATH), root)
    g5_spec = load(root / generation5.G5_SPEC_PATHS[model])
    g5_plan = generation5.validate_spec(g5_spec, root)
    execution = exact.execution_for(g5_spec["statistical_cell"]["cell_id"], 6)
    expected = EXPECTED[model]
    expected_spec = {
        "schema_version": SPEC_SCHEMA,
        "model": model,
        "statistical_cell": g5_spec["statistical_cell"],
        "execution": execution,
        "predecessor_generation5": {
            "spec_path": generation5.G5_SPEC_PATHS[model],
            "spec_sha256": g5_spec["generation5_spec_sha256"],
            "plan_sha256": g5_plan["plan_sha256"],
            "job_uid": expected["generation5_job_uid"],
            "pod_uid": expected["generation5_pod_uid"],
        },
        "supersession": {
            "failure_path": FAILURE_PATH,
            "failure_receipt_sha256": load(root / FAILURE_PATH)["receipt_sha256"],
            "prior_generation": 5,
            "new_generation": 6,
            "same_statistical_cell": True,
            "generation5_nonrepeatable": True,
        },
        "identities": {
            "job_name": expected["job_name"],
            "configmap_name": expected["configmap_name"],
            "intent_configmap_name": INTENT_NAME,
            "run_id": expected["run_id"],
            "network": expected["network"],
            "sfs_root": f"/mnt/sfs/jobs/{expected['job_name']}",
            "generation_claim_root": generation5.CLAIM_ROOT,
        },
        "rendered_plan_sha256": spec.get("rendered_plan_sha256"),
        "launch_authorized": False,
    }
    if (
        spec.get("generation6_spec_sha256") != digest(spec, "generation6_spec_sha256")
        or {key: value for key, value in spec.items() if key != "generation6_spec_sha256"}
        != expected_spec
    ):
        raise ValueError("generation-6 specification drifted")
    plan = _render_plan(spec, root)
    if plan["plan_sha256"] != spec["rendered_plan_sha256"]:
        raise ValueError("generation-6 rendered plan digest drifted")
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
        raise ValueError("generation-6 treatment renderer drifted")
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
