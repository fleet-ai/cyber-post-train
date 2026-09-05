"""Held generation-4 successor for generation-3 pre-model bootstrap failures."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_generation2_canary_v3 as generation2
from evals.fleet import autocontinue_generation3_authority_v1 as generation3_authority
from evals.fleet import autocontinue_generation3_canary as generation3
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-opencode-autocontinue-generation4-canary-spec-v1"
HELD_SCHEMA = "fleet-opencode-autocontinue-generation4-canary-held-v1"
CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v5"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation4-terminal-v1"
INCIDENT_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation3-pre-model-failure-v1.json"
)
INCIDENT_FILE_SHA = "sha256:8ca8de29fb7d4a46a09cec448a695d73084075bc0eb83bc05916aeb0c58c4647"
INCIDENT_SHA = "sha256:89d4e6a673a9084fdf68a37bb3b52647f489f12db88381b821985c5ca4a4ac98"
TOMBSTONE_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation3-pre-model-tombstones-v1.json"
)
TOMBSTONE_FILE_SHA = "sha256:423b735af28ff7c081b0a1e56e79eccc2ee0a66abbc0ce54bee4206f95cf863a"
TOMBSTONE_SHA = "sha256:47036e86edcec407212ae1b6118c8973f85d3f023dfd59e11336e6998e09635b"
HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation4-canaries-held-v1.json"
)
MODULE_PATH = "evals/fleet/autocontinue_generation4_canary.py"
PACKAGE_PATH = "evals/fleet/autocontinue_generation4_executable_package_v1.py"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation4-held-v1.yaml"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation4_held_v1.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation4_held_v1.sh"
CLAIM_ROOT = generation3.CLAIM_ROOT
INTENT_NAME = "chris-ac-g4-canary-submit-v1"
G4_SPEC_PATHS = {
    "qwen3.8-27b": "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation4-v1.json",
    "glm-5.3": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation4-v1.json",
}
EXPECTED = {
    "qwen3.8-27b": {
        "job_name": "chris-q38-ac-r004-a1-g4-v1",
        "configmap_name": "chris-q38-ac-r004-a1-g4-run-v1",
        "run_id": "chris-q38-ac-g4-r004-a1-02dd4e3f",
        "network": "q38-ac-g4-r004-a1-02dd4e3f",
        "generation3_job_uid": "395148cc-94b2-43d4-aa15-1c69d19b25e9",
        "generation3_pod_uid": "35ed25b8-c2b0-4ced-807c-1fa7df7dcc6e",
    },
    "glm-5.3": {
        "job_name": "chris-glm53-ac-r013-a1-g4-v1",
        "configmap_name": "chris-glm53-ac-r013-a1-g4-run-v1",
        "run_id": "chris-glm53-ac-g4-r013-a1-9375a9b9",
        "network": "glm53-ac-g4-r013-a1-9375a9b9",
        "generation3_job_uid": "f20e3a1f-f1d6-4c97-859c-559bfaaf1fc7",
        "generation3_pod_uid": "ab1f6c75-cd7d-4b1f-87f6-ecd60a496eff",
    },
}


def load(path: Path) -> dict[str, Any]:
    return generation3.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return generation3.digest(value, field)


def file_sha256(path: Path) -> str:
    return generation3.file_sha256(path)


def validate_incident(receipt: dict[str, Any], root: Path) -> None:
    if (
        file_sha256(root / INCIDENT_PATH) != INCIDENT_FILE_SHA
        or receipt.get("receipt_sha256") != INCIDENT_SHA
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or receipt.get("status") != "INFRASTRUCTURE_INVALID_PRE_MODEL"
        or receipt.get("classification") != "bootstrap_cwd_module_resolution_failure"
        or receipt.get("package_commit") != "36e86d2bee67dd5b3ed59cea5cba079d05699186"
        or receipt.get("side_effects")
        != {
            "model_calls": 0,
            "sessions": 0,
            "verifier_executions": 0,
            "generation3_claims": 0,
            "output_roots": 0,
        }
        or receipt.get("source_ordering_proof", {}).get(
            "working_directory_changed_to_repo_before_module_validation"
        )
        is not False
        or receipt.get("source_ordering_proof", {}).get(
            "first_module_validation_precedes_route_claim_and_model"
        )
        is not True
        or receipt.get("package_dependency_proof")
        != {
            "authority_call_chain": (
                "validate_release->semantic_held_binding->semantic.validate_held"
            ),
            "core_b_omitted_required_paths": [
                generation3_authority.SEMANTIC_HELD_PATH,
                "evals/fleet/cluster/opencode-autocontinue-generation3-executable-held-v3.yaml",
                "evals/fleet/scripts/run_opencode_autocontinue_generation3_executable_v3.sh",
                "evals/fleet/scripts/submit_opencode_autocontinue_generation3_executable_v3.sh",
            ],
            "all_four_paths_required_before_route_claim_or_model": True,
            "generation4_package_includes_all_four": True,
        }
        or len(receipt.get("jobs", [])) != 2
        or any(
            row.get("uid") != EXPECTED[row.get("model", "")]["generation3_job_uid"]
            or row.get("pod", {}).get("uid")
            != EXPECTED[row.get("model", "")]["generation3_pod_uid"]
            or row.get("pod", {}).get("evaluator_exit_code") != 1
            or row.get("pod", {}).get("evaluator_restarts") != 0
            for row in receipt.get("jobs", [])
        )
    ):
        raise ValueError("generation-3 pre-model incident drifted")


def validate_tombstones(receipt: dict[str, Any], root: Path) -> None:
    validate_incident(load(root / INCIDENT_PATH), root)
    if (
        file_sha256(root / TOMBSTONE_PATH) != TOMBSTONE_FILE_SHA
        or receipt.get("receipt_sha256") != TOMBSTONE_SHA
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or receipt.get("status") != "SEALED_PRE_MODEL_INFRASTRUCTURE_TOMBSTONES"
        or receipt.get("incident")
        != {
            "path": INCIDENT_PATH,
            "file_sha256": INCIDENT_FILE_SHA,
            "receipt_sha256": INCIDENT_SHA,
        }
        or len(receipt.get("models", [])) != 2
        or any(
            row.get("generation3_job", {}).get("uid")
            != EXPECTED[row.get("model", "")]["generation3_job_uid"]
            or row.get("generation3_pod", {}).get("uid")
            != EXPECTED[row.get("model", "")]["generation3_pod_uid"]
            or row.get("generation3_output_root", {}).get("absent") is not True
            or row.get("generation3_execution_claim", {}).get("absent") is not True
            or row.get("authoritative_api")
            != {"exact_treatment_sessions": 0, "planned_run_id_collisions": 0}
            or row.get("replacement")
            != {
                "permitted_execution_generation": 4,
                "same_cell": True,
                "generation3_nonrepeatable": True,
            }
            for row in receipt.get("models", [])
        )
    ):
        raise ValueError("generation-3 pre-model tombstones drifted")


def _render_plan(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-4 model is unsupported")
    old_spec = load(root / generation3.G3_SPEC_PATHS[model])
    old_plan = generation3.validate_spec(old_spec, root)
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
            "execution_generation": 4,
        }
    )
    plan["execution"].update(
        {
            "retry_policy": "generation4_only_after_exact_generation3_pre_model_tombstone",
            "required_priority_class": "fleet-serve-low",
            "execution_generation": 4,
            "generation_claim_root": CLAIM_ROOT,
            "old_claims_preserved": True,
            "launch_authorized": False,
        }
    )
    plan["source"] = {
        "exact_pass4_campaign_path": old_plan["source"]["exact_pass4_campaign_path"],
        "statistical_cell_id": old_spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "generation3_spec_sha256": old_spec["generation3_spec_sha256"],
        "generation3_plan_sha256": old_plan["plan_sha256"],
        "generation3_incident_receipt_sha256": INCIDENT_SHA,
        "generation3_tombstone_receipt_sha256": TOMBSTONE_SHA,
    }
    plan["plan_sha256"] = digest(plan, "plan_sha256")
    return plan


def validate_spec(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-4 model is unsupported")
    validate_incident(load(root / INCIDENT_PATH), root)
    validate_tombstones(load(root / TOMBSTONE_PATH), root)
    old_spec = load(root / generation3.G3_SPEC_PATHS[model])
    old_plan = generation3.validate_spec(old_spec, root)
    execution = exact.execution_for(old_spec["statistical_cell"]["cell_id"], 4)
    expected = EXPECTED[model]
    expected_spec = {
        "schema_version": SPEC_SCHEMA,
        "model": model,
        "statistical_cell": old_spec["statistical_cell"],
        "execution": execution,
        "predecessor_generation3": {
            "spec_path": generation3.G3_SPEC_PATHS[model],
            "spec_sha256": old_spec["generation3_spec_sha256"],
            "plan_sha256": old_plan["plan_sha256"],
            "job_uid": expected["generation3_job_uid"],
            "pod_uid": expected["generation3_pod_uid"],
        },
        "supersession": {
            "incident_path": INCIDENT_PATH,
            "incident_file_sha256": INCIDENT_FILE_SHA,
            "incident_receipt_sha256": INCIDENT_SHA,
            "tombstone_path": TOMBSTONE_PATH,
            "tombstone_file_sha256": TOMBSTONE_FILE_SHA,
            "tombstone_receipt_sha256": TOMBSTONE_SHA,
            "prior_generation": 3,
            "new_generation": 4,
            "same_statistical_cell": True,
            "generation3_nonrepeatable": True,
        },
        "identities": {
            "job_name": expected["job_name"],
            "configmap_name": expected["configmap_name"],
            "intent_configmap_name": INTENT_NAME,
            "run_id": expected["run_id"],
            "network": expected["network"],
            "sfs_root": f"/mnt/sfs/jobs/{expected['job_name']}",
            "generation_claim_root": CLAIM_ROOT,
        },
        "rendered_plan_sha256": spec.get("rendered_plan_sha256"),
        "launch_authorized": False,
    }
    if (
        spec.get("generation4_spec_sha256") != digest(spec, "generation4_spec_sha256")
        or {key: value for key, value in spec.items() if key != "generation4_spec_sha256"}
        != expected_spec
    ):
        raise ValueError("generation-4 specification drifted")
    plan = _render_plan(spec, root)
    if plan["plan_sha256"] != spec["rendered_plan_sha256"]:
        raise ValueError("generation-4 rendered plan digest drifted")
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
        raise ValueError("generation-4 executable treatment renderer drifted")
    return plan


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    specs = [load(root / G4_SPEC_PATHS[model]) for model in EXPECTED]
    plans = [validate_spec(spec, root) for spec in specs]
    expected = {
        "schema_version": HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "observed_at_utc": receipt.get("observed_at_utc"),
        "incident": {
            "path": INCIDENT_PATH,
            "file_sha256": INCIDENT_FILE_SHA,
            "receipt_sha256": INCIDENT_SHA,
        },
        "tombstones": {
            "path": TOMBSTONE_PATH,
            "file_sha256": TOMBSTONE_FILE_SHA,
            "receipt_sha256": TOMBSTONE_SHA,
        },
        "generation4_spec_sha256s": [spec["generation4_spec_sha256"] for spec in specs],
        "rendered_plan_sha256s": [plan["plan_sha256"] for plan in plans],
        "jobs": [EXPECTED[spec["model"]]["job_name"] for spec in specs],
        "execution_generation": 4,
        "statistical_cells": 2,
        "priority": {"class": "fleet-serve-low", "value": 100, "preemption_policy": "Never"},
        "launch_authorized": False,
        "bulk_release_authorized": False,
        "dedicated_serving_authorized": False,
        "remaining_gates": [
            "independent_generation4_package_audit",
            "fresh_hosted_route_and_duplicate_inventory",
            "append_only_root_authorization_and_model_releases",
        ],
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        not generation3._canonical_utc(receipt.get("observed_at_utc"))
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-4 held receipt drifted")


def validate_claim(
    claim: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    *,
    root: Path,
) -> None:
    if plan != validate_spec(spec, root):
        raise ValueError("generation-4 claim spec-plan chain drifted")
    expected = {
        "schema_version": CLAIM_SCHEMA,
        "generation4_spec_sha256": spec["generation4_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 4,
        "run_id": plan["attempts"][0]["run_id"],
        "job_uid": claim.get("job_uid"),
        "pod_uid": claim.get("pod_uid"),
        "claimed_at_utc": claim.get("claimed_at_utc"),
        "generation3_tombstone_receipt_sha256": TOMBSTONE_SHA,
        "prior_claims_preserved": True,
        "immutable": True,
        "automatic_retry": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if (
        not legacy._is_uuid(claim.get("job_uid"))
        or not legacy._is_uuid(claim.get("pod_uid"))
        or not generation3._canonical_utc(claim.get("claimed_at_utc"))
        or claim.get("receipt_sha256") != digest(claim, "receipt_sha256")
        or {key: value for key, value in claim.items() if key != "receipt_sha256"}
        != expected
    ):
        raise ValueError("generation-4 claim is not authoritative")


def terminal_receipt(
    spec: dict[str, Any],
    plan: dict[str, Any],
    claim: dict[str, Any],
    result: dict[str, Any],
    *,
    root: Path,
    terminal_at_utc: str | None = None,
) -> dict[str, Any]:
    validate_claim(claim, spec, plan, root=root)
    receipt = {
        "schema_version": TERMINAL_SCHEMA,
        "generation4_spec_sha256": spec["generation4_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 4,
        "generation_claim_receipt_sha256": claim["receipt_sha256"],
        "job_uid": claim["job_uid"],
        "pod_uid": claim["pod_uid"],
        "terminal_at_utc": terminal_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "result": generation2._validated_result(result, plan),
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
    return receipt


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("validate-evidence", "validate-spec", "validate-held", "preview"),
    )
    parser.add_argument("--spec", action="append", type=Path, default=[])
    parser.add_argument("--held", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "validate-evidence":
        validate_incident(load(args.repo / INCIDENT_PATH), args.repo)
        validate_tombstones(load(args.repo / TOMBSTONE_PATH), args.repo)
        return 0
    if args.command == "validate-spec":
        if len(args.spec) != 1:
            parser.error("validate-spec requires one spec")
        validate_spec(load(args.spec[0]), args.repo)
        return 0
    if not args.held:
        parser.error(f"{args.command} requires held receipt")
    validate_held(load(args.held), args.repo)
    if args.command == "preview":
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "HELD",
                    "execution_generation": 4,
                    "statistical_cells": 2,
                    "priority_class": "fleet-serve-low",
                    "preemption_policy": "Never",
                    "launch_authorized": False,
                    "objects_created": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
