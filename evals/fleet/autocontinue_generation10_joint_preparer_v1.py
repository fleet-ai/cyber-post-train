"""Prepare, but never launch, fresh G10 canaries after sealed G7 failures.

This module has no Kubernetes or Fleet mutation path.  It derives a fresh
execution identity from each exact frozen G8 treatment only after a
self-digesting G7 post-claim/pre-instance tombstone proves that the predecessor
created no instance, model call, session, verifier execution, or score.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_generation8_optimized_v1 as generation8
from evals.fleet import autocontinue_generation8_package_v1 as generation8_package
from evals.fleet import self_hosted

SCHEMA = "fleet-opencode-autocontinue-generation10-joint-preparer-v1"
TOMBSTONE_SCHEMA = "fleet-opencode-autocontinue-generation7-postclaim-preinstance-tombstone-v1"
TOMBSTONE_STATUS = "G7_FAILED_AFTER_CLAIM_BEFORE_INSTANCE_MODEL_SESSION_VERIFIER"
SHA_RE = re.compile(r"sha256:[0-9a-f]{64}")
CLAIM_ROOT = "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1"
DIAGNOSIS_PATH = (
    "docs/evidence/qwen38-study/2026-09-05-qwen38-autocontinue-authority-diagnosis-v2.json"
)
HELD_PATH = (
    "docs/evidence/qwen38-study/2026-09-05-opencode-autocontinue-generation10-joint-held-v1.json"
)
OPTIMIZED_RUNTIME = {
    "validation_mode": "generation8_single_pass_immutable_validation",
    "runtime_module": generation8.MODULE_PATH,
    "package_module": generation8.PACKAGE_PATH,
    "run_script": generation8.RUN_PATH,
    "minimum_validation_speedup": generation8.REQUIRED_VALIDATION_SPEEDUP,
    "measured_g7_one_model_validate_seconds": (generation8.MEASURED_G7_ONE_MODEL_VALIDATE_SECONDS),
}

MODELS: dict[str, dict[str, Any]] = {
    "qwen3.8-27b": {
        "short": "q38",
        "rank": 4,
        "g7_job": "chris-q38-ac-r004-a1-g7-v1",
        "g7_job_uid": "123c0026-0de9-43ab-8703-c92b34063fe1",
        "g7_pod_uid": "30229380-8ac8-4f95-b769-dead83397752",
        "g7_execution_id": (
            "sha256:77deaade2d420d18602ffdfde93d1a2a9957a32a00d1f644d42053ee72c31535"
        ),
        "cell_id": ("sha256:631c9d7cc5328849ce137393943927192b1b50dc60458cdb3425fbce893ecf5a"),
        "g8_execution_id": (
            "sha256:88b1f006fdd8e8cf9ba8212e75caf260ff7d6e963e94ad431f79483dcf4b008f"
        ),
        "g10_execution_id": (
            "sha256:883071ccd7ce04147e5213b43c43954ead7ad2e9fea69c2803e5b4b4c83e08b9"
        ),
        "g8_spec": "evals/fleet/configs/q38-opencode-autocontinue-canary-generation8-v1.json",
        "g8_plan": (
            "evals/fleet/configs/q38-opencode-autocontinue-canary-generation8-plan-v1.json"
        ),
        "tombstone": (
            "docs/evidence/qwen38-study/"
            "2026-09-05-qwen38-autocontinue-generation7-postclaim-preinstance-tombstone-v1.json"
        ),
    },
    "glm-5.3": {
        "short": "glm53",
        "rank": 13,
        "g7_job": "chris-glm53-ac-r013-a1-g7-v1",
        "g7_job_uid": "2f2dbf17-3d78-49be-8969-5ab0b8280f8e",
        "g7_pod_uid": "ec18d919-9e2e-47d8-8c0a-b5e727db4921",
        "g7_execution_id": (
            "sha256:b6338d535870aee0afa28519c3078dd6749db8d6f961b3a51e84dfc65cc13524"
        ),
        "cell_id": ("sha256:905051f141077d3aa5086c1f5dc6ad015d5ee6173a1ea7515025805cf9a24b41"),
        "g8_execution_id": (
            "sha256:a8b2f9b65f55c3c1f6cccc8189a5fece1023dd6cb238a02391766c28b3ba3bb8"
        ),
        "g10_execution_id": (
            "sha256:de0a1826ebd0f6d08634dd06c263e5f9640da06353ec1b08dcc72df2ad98ffb1"
        ),
        "g8_spec": ("evals/fleet/configs/glm53-opencode-autocontinue-canary-generation8-v1.json"),
        "g8_plan": (
            "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation8-plan-v1.json"
        ),
        "tombstone": (
            "docs/evidence/qwen38-study/"
            "2026-09-05-glm53-autocontinue-generation7-postclaim-preinstance-tombstone-v1.json"
        ),
    },
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def digest(value: Mapping[str, Any], field: str = "receipt_sha256") -> str:
    return sha256(canonical({key: item for key, item in value.items() if key != field}))


def read_canonical(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent immutable file: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise ValueError(f"non-canonical immutable file: {path}")
    return value


def _names(model: str) -> dict[str, str]:
    row = MODELS[model]
    suffix = row["g10_execution_id"].removeprefix("sha256:")[:8]
    stem = f"chris-{row['short']}-ac-r{row['rank']:03d}-a1-g10-v1"
    run_id = f"chris-{row['short']}-ac-g10-r{row['rank']:03d}-a1-{suffix}"
    return {
        "job": stem,
        "configmap": f"{stem}-run",
        "root": f"/mnt/sfs/jobs/{stem}",
        "run_id": run_id,
        "network": f"{row['short']}-ac-g10-r{row['rank']:03d}-a1-{suffix}",
    }


def validate_preinstance_tombstone(model: str, value: Mapping[str, Any]) -> None:
    """Require evidence equivalent to the sealed Qwen G7 pre-instance failure."""
    row = MODELS[model]
    job = value.get("generation7_job", {})
    pod = value.get("generation7_pod", {})
    attempt = value.get("attempt_root", {})
    cleanup = value.get("cleanup", {})
    fleet = value.get("fleet_evidence", {})
    claim_hashes = (
        value.get("generation7_global_claim", {}).get("receipt_sha256"),
        value.get("generation7_local_claim_receipt_sha256"),
        value.get("generation7_task_claim_receipt_sha256"),
    )
    if (
        value.get("schema_version") != TOMBSTONE_SCHEMA
        or value.get("status") != TOMBSTONE_STATUS
        or value.get("model") != model
        or value.get("cell_id") != row["cell_id"]
        or value.get("generation7_execution_id") != row["g7_execution_id"]
        or value.get("generation8_execution_id") != row["g8_execution_id"]
        or value.get("generation8_status") != "SKIPPED_UNCLAIMED_ABSENT"
        or job.get("name") != row["g7_job"]
        or job.get("uid") != row["g7_job_uid"]
        or job.get("terminal") != "Failed"
        or job.get("failed") != 1
        or job.get("restarts") != 0
        or pod.get("uid") != row["g7_pod_uid"]
        or pod.get("owner_job_uid") != row["g7_job_uid"]
        or pod.get("phase") != "Failed"
        or pod.get("evaluator_exit_code") != 1
        or pod.get("evaluator_restarts") != 0
        or pod.get("dind_restarts") != 0
        or attempt.get("attempt_directory_count") != 1
        or attempt.get("binding_or_runtime_files") != 0
        or attempt.get("accepted_or_terminal_files") != 0
        or sorted(attempt.get("files", [])) != ["cleanup.json", "failure.json"]
        or cleanup.get("instance_created") is not False
        or cleanup.get("instance_closed") is not False
        or fleet.get("matching_sessions") != 0
        or fleet.get("matching_ingestion_records") != 0
        or fleet.get("matching_verifier_records") != 0
        or value.get("model_called") is not False
        or value.get("scored_outcome_created") is not False
        or value.get("retry_generation7_identity_permitted") is not False
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or not all(isinstance(item, str) and SHA_RE.fullmatch(item) for item in claim_hashes)
        or value.get("receipt_sha256") != digest(value)
    ):
        raise ValueError(f"{model} G7 postclaim/preinstance tombstone drifted")


def validate_authority_diagnosis(value: Mapping[str, Any]) -> None:
    probe = value.get("behavioral_probe", {})
    openapi = value.get("openapi", {})
    source = value.get("source_inspection", {})
    supersedes = value.get("supersedes", {})
    startup = value.get("startup_cost_assessment", {})
    if (
        value.get("schema_version")
        != "fleet-opencode-autocontinue-preinstance-authority-diagnosis-v2"
        or value.get("status") != "ROUTES_DEPLOYED_OPENAPI_HIDDEN_NONMUTATING_PROBE_PASSED"
        or probe
        != {
            "authentication": "none",
            "method": "GET",
            "method_not_allowed_proves_concrete_route_match": True,
            "provisioning_status": 405,
            "response_bodies_read": False,
            "scoring_status": 405,
        }
        or openapi
        != {
            "intentional_hidden_registration": True,
            "route_templates_present": False,
            "router_include_in_schema": False,
        }
        or source
        != {
            "provisioning_post_handler_registered": True,
            "scoring_post_handler_registered": True,
        }
        or supersedes.get("receipt_sha256")
        != "sha256:e34da473d008420b1f2f065e4ca5b65097628a9340b926def2f79076f9eb1171"
        or value.get("g9_authority_diagnosis_status") != "SUPERSEDED_HELD_DO_NOT_USE_FOR_RELEASE"
        or startup.get("g8_validation_path") != "single_pass_immutable_validation"
        or startup.get("likely_dominant_local_startup_after_g8_optimization") is not True
        or startup.get("timing_receipt_available") is not False
        or value.get("model_instance_session_verifier_or_scoring_calls") != 0
        or value.get("mutations_performed") is not False
        or value.get("prompts_traces_flags_answers_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or value.get("receipt_sha256") != digest(value)
    ):
        raise ValueError("corrected non-mutating authority diagnosis drifted")


def load_static(root: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for model, row in MODELS.items():
        spec = read_canonical(root / row["g8_spec"])
        plan = read_canonical(root / row["g8_plan"])
        if (
            spec.get("model") != model
            or spec.get("execution", {}).get("execution_id") != row["g8_execution_id"]
            or spec.get("rendered_plan_sha256") != plan.get("plan_sha256")
            or plan.get("plan_sha256") != digest(plan, "plan_sha256")
            or plan.get("harness", {}).get("name") != "opencode"
            or plan.get("harness", {}).get("version") != "1.18.27"
            or plan.get("harness", {}).get("context_window_size") != 262144
            or plan.get("execution", {}).get("required_task_tools") != ["bash", "submit_report"]
            or plan.get("execution", {}).get("required_priority_class") != "fleet-serve-low"
            or plan.get("task_count") != 1
            or plan.get("new_session_count") != 1
        ):
            raise ValueError(f"{model} frozen G8 treatment drifted")
        loaded[model] = {"spec": spec, "plan": plan}
    return loaded


def prepare(root: Path) -> dict[str, Any]:
    """Return a sealed-safe held package; never create files or external objects."""
    static = load_static(root)
    diagnosis = read_canonical(root / DIAGNOSIS_PATH)
    validate_authority_diagnosis(diagnosis)
    models: dict[str, Any] = {}
    for model, row in MODELS.items():
        path = root / row["tombstone"]
        if not path.is_file() or path.is_symlink():
            models[model] = {
                "status": "HELD_PREINSTANCE_TOMBSTONE_REQUIRED",
                "tombstone_path": row["tombstone"],
            }
            continue
        tombstone = read_canonical(path)
        validate_preinstance_tombstone(model, tombstone)
        g8_spec = static[model]["spec"]
        plan = copy.deepcopy(static[model]["plan"])
        names = _names(model)
        plan["attempts"][0].update(
            execution_generation=10,
            network=names["network"],
            run_id=names["run_id"],
        )
        plan.update(
            campaign_id=names["job"],
            source_job_id=names["job"],
            scored_job_name=names["job"],
            sfs_root=names["job"],
            preflight_job_name=f"{names['job']}-preflight",
        )
        plan["execution"].update(
            execution_generation=10,
            launch_authorized=False,
            preemption_policy="Never",
            retry_policy=(
                "generation10_only_after_exact_generation7_postclaim_preinstance_"
                "tombstone_and_nonmutating_authority_route_probe"
            ),
        )
        plan["source"].update(
            execution_id=row["g10_execution_id"],
            generation7_execution_id=row["g7_execution_id"],
            generation7_tombstone_path=row["tombstone"],
            generation7_tombstone_receipt_sha256=tombstone["receipt_sha256"],
            generation8_execution_id=row["g8_execution_id"],
            generation8_status="SKIPPED_UNCLAIMED_ABSENT",
            required_authority_gate_mode="behavioral_method_not_allowed",
            required_authority_probe_method="GET",
            required_authority_probe_status=405,
            corrected_authority_diagnosis_path=DIAGNOSIS_PATH,
            corrected_authority_diagnosis_receipt_sha256=diagnosis["receipt_sha256"],
            required_optimized_runtime=copy.deepcopy(OPTIMIZED_RUNTIME),
        )
        plan["plan_sha256"] = digest(plan, "plan_sha256")
        spec: dict[str, Any] = {
            "schema_version": "fleet-opencode-autocontinue-generation10-canary-spec-v1",
            "launch_authorized": False,
            "model": model,
            "statistical_cell": copy.deepcopy(g8_spec["statistical_cell"]),
            "execution": {
                "schema_version": "fleet-statistical-cell-execution-v1",
                "cell_id": g8_spec["statistical_cell"]["cell_id"],
                "execution_generation": 10,
                "execution_id": row["g10_execution_id"],
            },
            "identities": {
                "job_name": names["job"],
                "configmap_name": names["configmap"],
                "sfs_root": names["root"],
                "generation_claim_root": CLAIM_ROOT,
                "run_id": names["run_id"],
                "network": names["network"],
            },
            "predecessor_generation7": {
                "job_name": row["g7_job"],
                "job_uid": row["g7_job_uid"],
                "pod_uid": row["g7_pod_uid"],
                "execution_id": row["g7_execution_id"],
                "tombstone_path": row["tombstone"],
                "tombstone_receipt_sha256": tombstone["receipt_sha256"],
            },
            "skipped_generation8": {
                "execution_id": row["g8_execution_id"],
                "claim_absent": True,
                "output_absent": True,
            },
            "required_authority_gate": {
                "mode": "behavioral_method_not_allowed",
                "method": "GET",
                "statuses": {"provisioning": 405, "scoring": 405},
                "response_body_read": False,
                "diagnosis_path": DIAGNOSIS_PATH,
                "diagnosis_receipt_sha256": diagnosis["receipt_sha256"],
            },
            "required_optimized_runtime": copy.deepcopy(OPTIMIZED_RUNTIME),
            "rendered_plan_sha256": plan["plan_sha256"],
        }
        spec["generation10_spec_sha256"] = digest(spec, "generation10_spec_sha256")
        models[model] = {
            "status": "HELD_RELEASE_AND_DUPLICATE_PREFLIGHT_REQUIRED",
            "spec": spec,
            "plan": plan,
            "tombstone_receipt_sha256": tombstone["receipt_sha256"],
        }
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "HELD",
        "launch_authorized": False,
        "objects_created": False,
        "authority_gate_source_sha256": sha256((root / "evals/fleet/self_hosted.py").read_bytes()),
        "corrected_authority_diagnosis_receipt_sha256": diagnosis["receipt_sha256"],
        "models": models,
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    body["package_sha256"] = digest(body, "package_sha256")
    return body


def assert_route_gate_implementation() -> None:
    """Pin the repair semantics without making a network request."""
    source = Path(self_hosted.__file__).read_text()
    required = (
        'with client.stream("GET", f"{ORCHESTRATOR}{path}") as response:',
        "if status_code != 405:",
        '"mode": "behavioral_method_not_allowed"',
    )
    forbidden = (
        "qwen-route-probe",
        "supported_shape_guard",
        'client.post(f"{ORCHESTRATOR}{path}"',
    )
    if not all(marker in source for marker in required) or any(
        marker in source for marker in forbidden
    ):
        raise ValueError("non-mutating authoritative route gate is not installed")


def assert_optimized_runtime_implementation(root: Path) -> None:
    """Require G8's single-pass runtime rather than the recursive G7 path."""
    if (
        generation8.MODULE_PATH != "evals/fleet/autocontinue_generation8_optimized_v1.py"
        or generation8.PACKAGE_PATH != "evals/fleet/autocontinue_generation8_package_v1.py"
        or generation8.REQUIRED_VALIDATION_SPEEDUP < 10
        or generation8_package.G8_PATHS.count(generation8.MODULE_PATH) != 1
    ):
        raise ValueError("Generation-8 optimized validation binding drifted")
    run_script = (root / generation8.RUN_PATH).read_text()
    if (
        "autocontinue_generation8_optimized_v1 run" not in run_script
        or "autocontinue_generation7_authority_v1 run" in run_script
    ):
        raise ValueError("Generation-8 optimized runtime script drifted")


def validate_held_receipt(root: Path, package: Mapping[str, Any]) -> None:
    value = read_canonical(root / HELD_PATH)
    qwen = package["models"]["qwen3.8-27b"]
    glm = package["models"]["glm-5.3"]
    if (
        value.get("schema_version") != "fleet-opencode-autocontinue-generation10-joint-held-v1"
        or value.get("status") != "HELD"
        or value.get("launch_authorized") is not False
        or value.get("objects_created") is not False
        or value.get("package_sha256") != package.get("package_sha256")
        or value.get("authority_gate_source_sha256") != package.get("authority_gate_source_sha256")
        or value.get("corrected_authority_diagnosis_receipt_sha256")
        != package.get("corrected_authority_diagnosis_receipt_sha256")
        or value.get("qwen_release_independent_of_glm") is not True
        or value.get("models", {}).get("qwen3.8-27b")
        != {
            "status": qwen["status"],
            "execution_id": qwen["spec"]["execution"]["execution_id"],
            "spec_sha256": qwen["spec"]["generation10_spec_sha256"],
            "plan_sha256": qwen["plan"]["plan_sha256"],
        }
        or value.get("models", {}).get("glm-5.3")
        != {
            "status": glm["status"],
            "tombstone_path": MODELS["glm-5.3"]["tombstone"],
            "execution_id": MODELS["glm-5.3"]["g10_execution_id"],
        }
        or value.get("required_optimized_runtime")
        != {
            "validation_mode": OPTIMIZED_RUNTIME["validation_mode"],
            "runtime_module": OPTIMIZED_RUNTIME["runtime_module"],
            "package_module": OPTIMIZED_RUNTIME["package_module"],
            "run_script": OPTIMIZED_RUNTIME["run_script"],
        }
        or value.get("prompts_traces_flags_or_scores_included") is not False
        or value.get("credentials_included") is not False
        or value.get("receipt_sha256") != digest(value)
    ):
        raise ValueError("Generation-10 held receipt drifted")
