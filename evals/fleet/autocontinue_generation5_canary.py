"""Held generation-5 successor for generation-4 pre-model bootstrap failures."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_generation2_canary_v3 as generation2
from evals.fleet import autocontinue_generation3_canary as generation3
from evals.fleet import autocontinue_generation4_canary as generation4
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-opencode-autocontinue-generation5-canary-spec-v1"
HELD_SCHEMA = "fleet-opencode-autocontinue-generation5-canary-held-v1"
CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v7"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation5-terminal-v3"
INCIDENT_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-autocontinue-generation4-pre-model-failure-v1.json"
)
INCIDENT_FILE_SHA = "sha256:21431a369cd427d57ef4b9cc764b0dce06d50437c98e86c7fd7cf78152774718"
INCIDENT_SHA = "sha256:f7567ef3545d451db2eb34bc967433f3ea5011a6f37a464e3a5a336d91415d0a"
TOMBSTONE_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-autocontinue-generation4-pre-model-tombstones-v1.json"
)
TOMBSTONE_FILE_SHA = "sha256:efeb730097bba431d338848583b297caeae4b4df2269a4b133428664e55dd650"
TOMBSTONE_SHA = "sha256:15de0c2d1d973555cd5c115034dc4707958663c800651c4f163355b3470eb6b8"
HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-05-opencode-autocontinue-generation5-canaries-held-v1.json"
)
MODULE_PATH = "evals/fleet/autocontinue_generation5_canary.py"
CLAIM_ROOT = generation3.CLAIM_ROOT
INTENT_NAME = "chris-ac-g5-canary-submit-v1"
G5_SPEC_PATHS = {
    "qwen3.8-27b": "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation5-v1.json",
    "glm-5.3": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation5-v1.json",
}
EXPECTED = {
    "qwen3.8-27b": {
        "job_name": "chris-q38-ac-r004-a1-g5-v1",
        "configmap_name": "chris-q38-ac-r004-a1-g5-run-v1",
        "run_id": "chris-q38-ac-g5-r004-a1-02dd4e3f",
        "network": "q38-ac-g5-r004-a1-02dd4e3f",
        "generation4_job_uid": "28cb20fd-71b0-406e-bc0c-ee6d57e93200",
        "generation4_pod_uid": "633815f2-45e3-4481-ac77-b12ebeeee87d",
    },
    "glm-5.3": {
        "job_name": "chris-glm53-ac-r013-a1-g5-v1",
        "configmap_name": "chris-glm53-ac-r013-a1-g5-run-v1",
        "run_id": "chris-glm53-ac-g5-r013-a1-9375a9b9",
        "network": "glm53-ac-g5-r013-a1-9375a9b9",
        "generation4_job_uid": "08cef047-4901-4d75-be26-981f71818af6",
        "generation4_pod_uid": "f0c739e3-e70a-43f1-a72d-a04bd8ed5d42",
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
        or receipt.get("classification")
        != "bootstrap_release_transport_and_dependency_closure_failure"
        or receipt.get("package_commit") != "b46d9ddccb616734566b1d50f66fcc4354bfde39"
        or receipt.get("side_effects")
        != {
            "model_calls": 0,
            "sessions": 0,
            "verifier_executions": 0,
            "generation4_claims": 0,
            "output_roots": 0,
        }
        or receipt.get("bootstrap_failure_proof")
        != {
            "failures": [
                "release_file_sha256_bound_pretty_repository_bytes_but_mounted_canonical_bytes",
                "model_projection_omitted_opposite_generation4_spec_and_transitive_predecessor_specs",
            ],
            "runner_reached_route_claim_or_model": False,
            "claim_precedes_first_model_call": True,
        }
        or len(receipt.get("jobs", [])) != 2
        or any(
            row.get("uid") != EXPECTED[row.get("model", "")]["generation4_job_uid"]
            or row.get("pod", {}).get("uid")
            != EXPECTED[row.get("model", "")]["generation4_pod_uid"]
            or row.get("pod", {}).get("evaluator_exit_code") != 1
            or row.get("pod", {}).get("evaluator_restarts") != 0
            for row in receipt.get("jobs", [])
        )
    ):
        raise ValueError("generation-4 pre-model incident drifted")


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
            row.get("generation4_job", {}).get("uid")
            != EXPECTED[row.get("model", "")]["generation4_job_uid"]
            or row.get("generation4_pod", {}).get("uid")
            != EXPECTED[row.get("model", "")]["generation4_pod_uid"]
            or row.get("generation4_output_root", {}).get("absent") is not True
            or row.get("generation4_execution_claim", {}).get("absent") is not True
            or row.get("authoritative_api")
            != {"exact_treatment_sessions": 0, "planned_run_id_collisions": 0}
            or row.get("replacement")
            != {
                "permitted_execution_generation": 5,
                "same_cell": True,
                "generation4_nonrepeatable": True,
            }
            for row in receipt.get("models", [])
        )
    ):
        raise ValueError("generation-4 pre-model tombstones drifted")


def _render_plan(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-5 model is unsupported")
    old_spec = load(root / generation4.G4_SPEC_PATHS[model])
    old_plan = generation4.validate_spec(old_spec, root)
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
            "execution_generation": 5,
        }
    )
    plan["execution"].update(
        {
            "retry_policy": "generation5_only_after_exact_generation4_pre_model_tombstone",
            "required_priority_class": "fleet-serve-low",
            "execution_generation": 5,
            "generation_claim_root": CLAIM_ROOT,
            "old_claims_preserved": True,
            "launch_authorized": False,
        }
    )
    plan["source"] = {
        "exact_pass4_campaign_path": old_plan["source"]["exact_pass4_campaign_path"],
        "statistical_cell_id": old_spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "generation4_spec_sha256": old_spec["generation4_spec_sha256"],
        "generation4_plan_sha256": old_plan["plan_sha256"],
        "generation4_incident_receipt_sha256": INCIDENT_SHA,
        "generation4_tombstone_receipt_sha256": TOMBSTONE_SHA,
    }
    plan["plan_sha256"] = digest(plan, "plan_sha256")
    return plan


def validate_spec(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-5 model is unsupported")
    validate_incident(load(root / INCIDENT_PATH), root)
    validate_tombstones(load(root / TOMBSTONE_PATH), root)
    old_spec = load(root / generation4.G4_SPEC_PATHS[model])
    old_plan = generation4.validate_spec(old_spec, root)
    execution = exact.execution_for(old_spec["statistical_cell"]["cell_id"], 5)
    expected = EXPECTED[model]
    expected_spec = {
        "schema_version": SPEC_SCHEMA,
        "model": model,
        "statistical_cell": old_spec["statistical_cell"],
        "execution": execution,
        "predecessor_generation4": {
            "spec_path": generation4.G4_SPEC_PATHS[model],
            "spec_sha256": old_spec["generation4_spec_sha256"],
            "plan_sha256": old_plan["plan_sha256"],
            "job_uid": expected["generation4_job_uid"],
            "pod_uid": expected["generation4_pod_uid"],
        },
        "supersession": {
            "incident_path": INCIDENT_PATH,
            "incident_file_sha256": INCIDENT_FILE_SHA,
            "incident_receipt_sha256": INCIDENT_SHA,
            "tombstone_path": TOMBSTONE_PATH,
            "tombstone_file_sha256": TOMBSTONE_FILE_SHA,
            "tombstone_receipt_sha256": TOMBSTONE_SHA,
            "prior_generation": 4,
            "new_generation": 5,
            "same_statistical_cell": True,
            "generation4_nonrepeatable": True,
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
        spec.get("generation5_spec_sha256") != digest(spec, "generation5_spec_sha256")
        or {key: value for key, value in spec.items() if key != "generation5_spec_sha256"}
        != expected_spec
    ):
        raise ValueError("generation-5 specification drifted")
    plan = _render_plan(spec, root)
    if plan["plan_sha256"] != spec["rendered_plan_sha256"]:
        raise ValueError("generation-5 rendered plan digest drifted")
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
        raise ValueError("generation-5 executable treatment renderer drifted")
    return plan


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    specs = [load(root / G5_SPEC_PATHS[model]) for model in EXPECTED]
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
        "generation5_spec_sha256s": [spec["generation5_spec_sha256"] for spec in specs],
        "rendered_plan_sha256s": [plan["plan_sha256"] for plan in plans],
        "jobs": [EXPECTED[spec["model"]]["job_name"] for spec in specs],
        "execution_generation": 5,
        "statistical_cells": 2,
        "priority": {"class": "fleet-serve-low", "value": 100, "preemption_policy": "Never"},
        "launch_authorized": False,
        "bulk_release_authorized": False,
        "dedicated_serving_authorized": False,
        "remaining_gates": [
            "independent_generation5_package_audit",
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
        raise ValueError("generation-5 held receipt drifted")


def validate_claim(
    claim: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    *,
    root: Path,
) -> None:
    if plan != validate_spec(spec, root):
        raise ValueError("generation-5 claim spec-plan chain drifted")
    expected = {
        "schema_version": CLAIM_SCHEMA,
        "generation5_spec_sha256": spec["generation5_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 5,
        "run_id": plan["attempts"][0]["run_id"],
        "job_uid": claim.get("job_uid"),
        "pod_uid": claim.get("pod_uid"),
        "claimed_at_utc": claim.get("claimed_at_utc"),
        "generation4_tombstone_receipt_sha256": TOMBSTONE_SHA,
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
        raise ValueError("generation-5 claim is not authoritative")


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
        "generation5_spec_sha256": spec["generation5_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 5,
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
                    "execution_generation": 5,
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
