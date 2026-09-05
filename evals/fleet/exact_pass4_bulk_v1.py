"""Build and validate the held four-controller exact pass@4 bulk successor.

The module deliberately separates statistical cells from execution machinery.
It expands the frozen easiest-100 universe, removes the two generation-5
canary cells, and partitions the remaining 798 cells into four one-stream
hosted controllers.  Nothing in this module submits work or authorizes launch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-exact-pass4-bulk-controller-spec-v1"
PLAN_SCHEMA = "fleet-exact-pass4-bulk-controller-plan-v1"
HELD_SCHEMA = "fleet-exact-pass4-bulk-held-v1"
RELEASE_SCHEMA = "fleet-exact-pass4-bulk-release-v1"
INVENTORY_GATE_SCHEMA = "fleet-exact-easiest100-task-inventory-terminal-v1"
CANARY_GATE_SCHEMA = "fleet-exact-pass4-generation5-observer-v1"
RECONCILIATION_GATE_SCHEMA = "fleet-exact-pass4-bulk-reconciliation-v1"
RECONCILIATION_OBSERVATION_SCHEMA = "fleet-exact-pass4-prebulk-observation-v1"
RECONCILIATION_ROOT = Path("/mnt/sfs/jobs/chris-cyber-exact100-prebulk-reconcile-v1")
RECONCILIATION_ACCEPT_JOB = "chris-cyber-exact100-prebulk-reconcile-accept-v1"

CAMPAIGN_PATH = "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
SELECTION_PATH = exact.EXPECTED_SELECTION_PATH
HELD_PATH = (
    "docs/evidence/qwen38-study/2026-09-05-opencode-exact-pass4-bulk-four-controller-held-v1.json"
)
MODULE_PATH = "evals/fleet/exact_pass4_bulk_v1.py"
PACKAGE_PATH = "evals/fleet/exact_pass4_bulk_package_v1.py"
RUNTIME_PATH = "evals/fleet/exact_pass4_bulk_runtime_v1.py"
RUN_PATH = "evals/fleet/scripts/run_exact_pass4_bulk_controller_v1.sh"
RENDER_PATH = "evals/fleet/exact_pass4_bulk_release_renderer_v1.py"
SUBMIT_PATH = "evals/fleet/scripts/submit_exact_pass4_bulk_v1.sh"
MANIFEST_PATH = "evals/fleet/cluster/opencode-exact-pass4-bulk-held-v1.yaml"
SNAPSHOT_PATH = "evals/fleet/immutable_submission_snapshot.py"
CLAIM_ROOT = "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1"
LEASE_ROOT = "/mnt/sfs/endpoint-leases/opencode11827-autocontinue-primary-v1"

CANARIES = {
    "qwen3.8-27b": {"selection_rank": 4, "attempt": 1, "execution_generation": 5},
    "glm-5.3": {"selection_rank": 13, "attempt": 1, "execution_generation": 5},
}
CANARY_SPEC_PATHS = {
    "qwen3.8-27b": ("evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation5-v1.json"),
    "glm-5.3": ("evals/fleet/configs/glm53-opencode-autocontinue-canary-generation5-v1.json"),
}
RUNTIME_TEMPLATE_PATHS = {
    "qwen3.8-27b": "evals/fleet/configs/qwen38-opencode-autocontinue-canary1-v1.json",
    "glm-5.3": "evals/fleet/configs/glm53-opencode-autocontinue-canary1-v1.json",
}

CONTROLLERS = {
    "qwen-a": {
        "model": "qwen3.8-27b",
        "job_name": "chris-q38-ac-exact100-bulk-a199-v1",
        "configmap_name": "chris-q38-ac-exact100-bulk-a199-run-v1",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "full_ranks": list(range(1, 98, 2)),
        "partial_attempts": {4: [2, 3, 4]},
        "cell_count": 199,
    },
    "qwen-b": {
        "model": "qwen3.8-27b",
        "job_name": "chris-q38-ac-exact100-bulk-b200-v1",
        "configmap_name": "chris-q38-ac-exact100-bulk-b200-run-v1",
        "serving_block": "qwen-hosted-autocontinue-v1",
        "full_ranks": [2, *range(6, 99, 2), 99, 100],
        "partial_attempts": {},
        "cell_count": 200,
    },
    "glm-a": {
        "model": "glm-5.3",
        "job_name": "chris-glm53-ac-exact100-bulk-a199-v1",
        "configmap_name": "chris-glm53-ac-exact100-bulk-a199-run-v1",
        "serving_block": "glm-hosted-autocontinue-v1",
        "full_ranks": list(range(2, 100, 2)),
        "partial_attempts": {13: [2, 3, 4]},
        "cell_count": 199,
    },
    "glm-b": {
        "model": "glm-5.3",
        "job_name": "chris-glm53-ac-exact100-bulk-b200-v1",
        "configmap_name": "chris-glm53-ac-exact100-bulk-b200-run-v1",
        "serving_block": "glm-hosted-autocontinue-v1",
        "full_ranks": [*range(1, 13, 2), *range(15, 100, 2), 100],
        "partial_attempts": {},
        "cell_count": 200,
    },
}

SPEC_PATHS = {key: f"evals/fleet/configs/exact-pass4-bulk-{key}-v1.json" for key in CONTROLLERS}
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
RELEASE_STATEMENT = (
    "Authorize exactly four create-once one-stream hosted controllers for the 798 "
    "untouched corrected-treatment cells remaining after the two accepted generation-5 "
    "canaries, only after all bound inventory, canary, and duplicate-reconciliation "
    "gates pass; never repeat any accepted, active, claimed, or model-started cell."
)
NONZERO_EXIT_POLICY = "credit_only_if_reward_ingest_cleanup_and_authoritative_session_match"
FIXED_GATE_PATHS = {
    "exact100_inventory": ("/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v2/TERMINAL.json"),
    "exact100_inventory_package": str(RECONCILIATION_ROOT / "exact100-inventory-package.json"),
    "qwen_generation5_canary": str(RECONCILIATION_ROOT / "qwen3.8-27b-generation5-gate.json"),
    "glm_generation5_canary": str(RECONCILIATION_ROOT / "glm-5.3-generation5-gate.json"),
    "duplicate_reconciliation": str(RECONCILIATION_ROOT / "TERMINAL.json"),
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or absent JSON file: {path}")
    value = json.loads(path.read_text(), object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def digest(value: dict[str, Any], field: str) -> str:
    return self_hosted.digest_without(value, field)


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _universe(root: Path) -> dict[str, Any]:
    campaign = exact.read_object(root / CAMPAIGN_PATH)
    return exact.build_universe(campaign, root)


def _cells_by_key(root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    universe = _universe(root)
    return {(row["model"], row["selection_rank"], row["attempt"]): row for row in universe["cells"]}


def _selected_keys(controller: str) -> list[tuple[str, int, int]]:
    expected = CONTROLLERS[controller]
    model = expected["model"]
    selected: list[tuple[str, int, int]] = []
    partial = {int(rank): attempts for rank, attempts in expected["partial_attempts"].items()}
    for rank in sorted(set(expected["full_ranks"]) | set(partial)):
        attempts = partial.get(rank, [1, 2, 3, 4])
        for attempt in attempts:
            selected.append((model, rank, attempt))
    if len(selected) != expected["cell_count"] or len(set(selected)) != len(selected):
        raise ValueError(f"{controller} partition count or uniqueness drifted")
    return selected


def build_spec(controller: str, root: Path) -> dict[str, Any]:
    if controller not in CONTROLLERS:
        raise ValueError("unsupported bulk controller")
    expected = CONTROLLERS[controller]
    cells = _cells_by_key(root)
    selected = [cells[key] for key in _selected_keys(controller)]
    compact_cells = [
        {
            "cell_id": row["cell_id"],
            "selection_rank": row["selection_rank"],
            "attempt": row["attempt"],
            "task_version_id": row["task_version_id"],
            "execution_generation": 1,
            "execution_id": row["initial_execution"]["execution_id"],
        }
        for row in selected
    ]
    body = {
        "schema_version": SPEC_SCHEMA,
        "controller": controller,
        "model": expected["model"],
        "job_name": expected["job_name"],
        "configmap_name": expected["configmap_name"],
        "serving_block": expected["serving_block"],
        "partition": {
            "full_ranks": expected["full_ranks"],
            "partial_attempts": {
                str(rank): attempts for rank, attempts in expected["partial_attempts"].items()
            },
            "ordering": "selection_rank_ascending_then_attempt_ascending",
            "complete_task_boundaries_except_explicit_canary_remainder": True,
            "cell_count": expected["cell_count"],
        },
        "cells_sha256": self_hosted.sha256(self_hosted.canonical_json(compact_cells)),
        "execution": {
            "workers": 1,
            "same_task_max_inflight": 1,
            "attempts_per_task_sequential": True,
            "global_execution_claim_before_model_call": True,
            "claim_root": CLAIM_ROOT,
            "accepted_active_or_claimed_cells_are_nonrepeatable": True,
            "automatic_retry": False,
            "future_nonzero_exit_policy": NONZERO_EXIT_POLICY,
            "infrastructure_failure_policy": "quarantine_exact_cell_and_continue_tail",
            "tail_survives_single_cell_failure": True,
            "endpoint_lease": {
                "lease_root": LEASE_ROOT,
                "endpoint_key": expected["serving_block"],
                "maximum_streams": 2,
            },
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        },
        "treatment": exact.EXPECTED_TREATMENT,
        "launch_authorized": False,
    }
    return {**body, "spec_sha256": self_hosted.sha256(self_hosted.canonical_json(body))}


def validate_spec(spec: dict[str, Any], controller: str, root: Path) -> dict[str, Any]:
    expected = build_spec(controller, root)
    if spec != expected or spec.get("spec_sha256") != digest(spec, "spec_sha256"):
        raise ValueError(f"{controller} bulk specification drifted")
    return build_plan(spec, root)


def build_plan(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    controller = spec.get("controller")
    if controller not in CONTROLLERS:
        raise ValueError("unsupported bulk controller")
    if spec != build_spec(controller, root):
        raise ValueError("bulk plan source specification drifted")
    selection = exact.validate_selection(exact.read_object(root / CAMPAIGN_PATH), root)
    tasks = {row["rank"]: row for row in selection}
    model = spec["model"]
    model_binding = exact.EXPECTED_MODELS[model]
    attempts: list[dict[str, Any]] = []
    cells_by_key = _cells_by_key(root)
    selected_cells = [cells_by_key[key] for key in _selected_keys(controller)]
    compact_cells = [
        {
            "cell_id": row["cell_id"],
            "selection_rank": row["selection_rank"],
            "attempt": row["attempt"],
            "task_version_id": row["task_version_id"],
            "execution_generation": 1,
            "execution_id": row["initial_execution"]["execution_id"],
        }
        for row in selected_cells
    ]
    if spec["cells_sha256"] != self_hosted.sha256(self_hosted.canonical_json(compact_cells)):
        raise ValueError("bulk specification cell digest drifted")
    for ordinal, cell in enumerate(compact_cells, 1):
        task = tasks[cell["selection_rank"]]
        model_short = "q38" if model == "qwen3.8-27b" else "glm53"
        run_id = (
            f"chris-{model_short}-ac-bulk-{controller[-1]}-r"
            f"{cell['selection_rank']:03d}-a{cell['attempt']}-g1-"
            f"{task['task_version_id'].split('-')[0]}"
        )
        attempts.append(
            {
                "ordinal": ordinal,
                "cell_id": cell["cell_id"],
                "execution_id": cell["execution_id"],
                "execution_generation": 1,
                "selection_rank": cell["selection_rank"],
                "attempt": cell["attempt"],
                "task_key": task["task_key"],
                "task_version_id": task["task_version_id"],
                "environment_version_id": task["environment_version_id"],
                "run_id": run_id,
                "network": run_id.removeprefix("chris-")[:63],
            }
        )
    body = {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": exact.EXPECTED_CAMPAIGN_ID,
        "controller": controller,
        "job_name": spec["job_name"],
        "configmap_name": spec["configmap_name"],
        "sfs_root": f"/mnt/sfs/jobs/{spec['job_name']}",
        "model": model_binding,
        "serving_block": spec["serving_block"],
        "task_count": len({row["selection_rank"] for row in attempts}),
        "new_session_count": len(attempts),
        "attempts": attempts,
        "execution": spec["execution"],
        "treatment": spec["treatment"],
        "release_required": True,
        "launch_authorized": False,
        "privacy": {
            "credentials_included": False,
            "prompts_included": False,
            "scores_included": False,
            "transcripts_included": False,
        },
    }
    return {**body, "plan_sha256": self_hosted.sha256(self_hosted.canonical_json(body))}


def all_specs(root: Path) -> dict[str, dict[str, Any]]:
    return {key: load(root / path) for key, path in SPEC_PATHS.items()}


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    specs = all_specs(root)
    plans = {key: validate_spec(value, key, root) for key, value in specs.items()}
    universe = _universe(root)
    expected = {
        (row["model"], row["selection_rank"], row["attempt"]): row["cell_id"]
        for row in universe["cells"]
    }
    selected: dict[tuple[str, int, int], str] = {}
    for key in specs:
        for identity in _selected_keys(key):
            if identity in selected:
                raise ValueError(f"bulk controller overlap at {identity}")
            selected[identity] = expected[identity]
    canary_keys = {
        (model, value["selection_rank"], value["attempt"]) for model, value in CANARIES.items()
    }
    if set(selected) != set(expected) - canary_keys:
        raise ValueError("bulk partitions do not cover exactly universe minus canaries")
    if any(selected[key] != expected[key] for key in selected):
        raise ValueError("bulk partition statistical cell identity drifted")
    if {key: len(plan["attempts"]) for key, plan in plans.items()} != {
        "qwen-a": 199,
        "qwen-b": 200,
        "glm-a": 199,
        "glm-b": 200,
    }:
        raise ValueError("bulk partition counts drifted")
    return plans


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def validate_inventory_gate(receipt: dict[str, Any], root: Path) -> None:
    from evals.fleet import exact_pass4_task_inventory as inventory

    prepared = inventory.prepare_expected(root)
    universe = _universe(root)
    tasks = receipt.get("tasks")
    campaign = exact.read_object(root / CAMPAIGN_PATH)
    selected = exact.validate_selection(campaign, root)
    expected_identities = [
        (
            row["rank"],
            row["task_key"],
            row["task_version"],
            row["task_version_id"],
            row["env_key"],
            row["env_version"],
            row["environment_version_id"],
            row["data_key"],
            row["data_version"],
        )
        for row in selected
    ]
    actual_identities = [
        (
            row.get("selection_rank"),
            (row.get("task") or {}).get("key"),
            (row.get("task") or {}).get("version"),
            (row.get("task") or {}).get("version_id"),
            (row.get("environment") or {}).get("id"),
            (row.get("environment") or {}).get("version"),
            (row.get("environment") or {}).get("version_id"),
            (row.get("environment") or {}).get("data_id"),
            (row.get("environment") or {}).get("data_version"),
        )
        for row in tasks or []
        if isinstance(row, dict)
    ]
    task_shapes_valid = all(
        isinstance(row, dict)
        and set(row)
        == {
            "selection_rank",
            "task",
            "environment",
            "verifier",
            "exact_version_request",
            "response_version_id_observable",
        }
        and set(row.get("task") or {})
        == {
            "key",
            "version",
            "version_id",
            "prompt_sha256",
            "env_variables_sha256",
            "output_json_schema_sha256",
            "cyber_contract",
        }
        and set(row.get("environment") or {})
        == {
            "id",
            "version",
            "version_id",
            "version_id_authority",
            "data_id",
            "data_version",
            "runtime_seed_content_sha256",
            "runtime_seed_file_count",
        }
        and set(row.get("verifier") or {})
        == {"id", "version_id", "version", "sha256", "function_name"}
        and row.get("exact_version_request") is True
        and (row.get("environment") or {}).get("version_id_authority") == "immutable_selection"
        for row in tasks or []
    )
    if (
        receipt.get("schema_version") != INVENTORY_GATE_SCHEMA
        or receipt.get("status") != "PASSED"
        or receipt.get("campaign_id") != exact.EXPECTED_CAMPAIGN_ID
        or receipt.get("campaign_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(campaign))
        or receipt.get("universe_sha256") != universe["universe_sha256"]
        or receipt.get("selection_sha256") != exact.EXPECTED_SELECTION_SHA256
        or receipt.get("models") != exact.EXPECTED_MODELS
        or receipt.get("expected_inventory_sha256") != prepared["expected_sha256"]
        or receipt.get("task_count") != 100
        or receipt.get("cell_counts") != {"qwen3.8-27b": 400, "glm-5.3": 400}
        or receipt.get("total_cell_count") != 800
        or receipt.get("attempts") != [1, 2, 3, 4]
        or receipt.get("binding_mismatch_count") != 0
        or not isinstance(tasks, list)
        or not task_shapes_valid
        or len(tasks) != 100
        or [row.get("selection_rank") for row in tasks if isinstance(row, dict)]
        != list(range(1, 101))
        or actual_identities != expected_identities
        or receipt.get("task_bindings_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(tasks))
        or receipt.get("fleet_account")
        != {
            "team_name": "fleet",
            "team_id": "a1025f0b-ad67-49fc-a023-51800ab43e84",
        }
        or receipt.get("request_counts")
        != {
            "account_get": 1,
            "exact_task_version_get": 100,
            "redirects_followed": 0,
            "post_put_patch_delete": 0,
            "model_or_scoring_calls": 0,
            "session_calls": 0,
        }
        or receipt.get("privacy")
        != {
            "task_payloads_persisted": False,
            "task_content_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
    ):
        raise ValueError("exact-100 inventory gate is not accepted")
    _require_digest(receipt.get("expected_inventory_sha256"), "expected inventory digest")
    _require_digest(receipt.get("task_bindings_sha256"), "task bindings digest")
    inventory._validate_expected(prepared)
    runtime = receipt.get("runtime")
    if (
        not isinstance(runtime, dict)
        or UUID_RE.fullmatch(str(runtime.get("job_uid"))) is None
        or UUID_RE.fullmatch(str(runtime.get("pod_uid"))) is None
    ):
        raise ValueError("exact-100 inventory runtime UIDs are invalid")


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Hydrate an executable controller from the accepted GET-only inventory."""
    validate_inventory_gate(inventory_receipt, root)
    compact = validate_all(root)[controller]
    template = load(root / RUNTIME_TEMPLATE_PATHS[compact["model"]["served_id"]])
    tasks_by_rank = {row["selection_rank"]: row for row in inventory_receipt["tasks"]}
    tasks: list[dict[str, Any]] = []
    for rank in sorted({row["selection_rank"] for row in compact["attempts"]}):
        source = tasks_by_rank[rank]
        environment = copy.deepcopy(source["environment"])
        environment.pop("version_id_authority", None)
        environment.pop("runtime_seed_file_count", None)
        environment["ttl_seconds"] = 32400
        task = copy.deepcopy(source["task"])
        task.pop("version", None)
        tasks.append(
            {
                "rank": rank,
                "source_rank": rank,
                "environment": environment,
                "task": task,
                "verifier": copy.deepcopy(source["verifier"]),
            }
        )
    attempts = [
        {
            **copy.deepcopy(row),
            "rank": row["selection_rank"],
            "source_rank": row["selection_rank"],
        }
        for row in compact["attempts"]
    ]
    execution = copy.deepcopy(compact["execution"])
    execution.update(
        {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": template["execution"][
                "required_task_tool_catalog_sha256"
            ],
            "training_data_eligible": False,
        }
    )
    runtime_harness = copy.deepcopy(template["harness"])
    runtime_harness["compaction_headroom_tokens"] = compact["treatment"][
        "compaction_headroom_tokens"
    ]
    body = {
        "schema_version": "fleet-exact-pass4-bulk-executable-plan-v1",
        "controller": controller,
        "campaign_id": compact["job_name"],
        "source_job_id": compact["job_name"],
        "sfs_root": compact["sfs_root"],
        "repo_root": ".",
        "inventory_receipt": inventory_receipt,
        "inventory_receipt_sha256": inventory_receipt["receipt_sha256"],
        "model": copy.deepcopy(template["model"]),
        "harness": runtime_harness,
        "authority": copy.deepcopy(template["authority"]),
        "serving_block": compact["serving_block"],
        "tasks": tasks,
        "attempts": attempts,
        "execution": execution,
        "treatment": compact["treatment"],
        "release_required": True,
        "launch_authorized": True,
        "privacy": compact["privacy"],
    }
    plan = {**body, "plan_sha256": self_hosted.digest_without(body, "plan_sha256")}
    settings = self_hosted.opencode_settings(plan)
    canonical = self_hosted.canonical_json(settings)
    if (
        plan["harness"]["settings_canonical_sha256"] != self_hosted.sha256(canonical)
        or plan["harness"]["settings_file_sha256"] != self_hosted.sha256(canonical + b"\n")
        or settings.get("compaction") != {"auto": True, "reserved": 20000}
        or "plugin" in settings
        or plan["execution"]["required_task_tools"] != ["bash", "submit_report"]
    ):
        raise ValueError("bulk executable treatment renderer drifted")
    return plan


def validate_canary_gate(
    receipt: dict[str, Any],
    model: str,
    root: Path,
    *,
    evidence_root: Path | None = None,
) -> None:
    """Validate a fresh observer over the actual generation-5 terminal chain."""
    from evals.fleet import autocontinue_generation5_authority_v1 as authority_module
    from evals.fleet import autocontinue_generation5_canary as generation5

    if model not in CANARIES:
        raise ValueError("unsupported canary gate model")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "claim_path",
        "terminal_path",
        "release_path",
        "authority_path",
        "package_commit",
    }:
        raise ValueError("generation-5 observer evidence paths are incomplete")
    spec = load(root / CANARY_SPEC_PATHS[model])
    plan = generation5.validate_spec(spec, root)
    claim = load(
        _evidence_path(root, evidence["claim_path"], "canary claim", evidence_root=evidence_root)
    )
    terminal_path = _evidence_path(
        root, evidence["terminal_path"], "canary terminal", evidence_root=evidence_root
    )
    terminal = load(terminal_path)
    release_path = _evidence_path(
        root,
        evidence["release_path"],
        "canary release",
        evidence_root=evidence_root,
    )
    scoring_release = load(release_path)
    root_authority = load(_evidence_path(root, evidence["authority_path"], "canary authority"))
    package_commit = evidence["package_commit"]
    authority_module.validate_terminal(
        terminal,
        spec,
        plan,
        claim,
        scoring_release,
        file_sha256(release_path),
        root_authority,
        root,
        package_commit,
    )
    result = terminal.get("result") or {}
    if (
        receipt.get("schema_version") != CANARY_GATE_SCHEMA
        or receipt.get("status") != "ACCEPTED"
        or receipt.get("model") != model
        or receipt.get("generation5_spec_sha256") != spec["generation5_spec_sha256"]
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("execution_id") != spec["execution"]["execution_id"]
        or receipt.get("job_name") != spec["identities"]["job_name"]
        or receipt.get("run_id") != plan["attempts"][0]["run_id"]
        or receipt.get("terminal_file_sha256") != file_sha256(terminal_path)
        or receipt.get("terminal_receipt_sha256") != terminal.get("receipt_sha256")
        or receipt.get("session_id") != result.get("session_id")
        or receipt.get("verifier_execution_id") != result.get("verifier_execution_id")
        or result.get("accepted") is not True
        or result.get("quarantined") is not False
        or result.get("session_ingest_completed") is not True
        or result.get("cleanup_completed") is not True
        or receipt.get("job_succeeded") is not True
        or receipt.get("pod_restarts") != 0
        or receipt.get("lease_probe_acquired_and_released") is not True
        or receipt.get("claim_job_uid") != claim.get("job_uid")
        or receipt.get("claim_pod_uid") != claim.get("pod_uid")
        or receipt.get("prompts_traces_flags_or_scores_included") is not False
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
    ):
        raise ValueError(f"{model} generation-5 canary gate is not accepted")
    if (
        UUID_RE.fullmatch(str(claim.get("job_uid"))) is None
        or UUID_RE.fullmatch(str(claim.get("pod_uid"))) is None
        or UUID_RE.fullmatch(str((receipt.get("observer_runtime") or {}).get("job_uid"))) is None
        or UUID_RE.fullmatch(str((receipt.get("observer_runtime") or {}).get("pod_uid"))) is None
        or not isinstance(receipt.get("observed_at_utc"), str)
        or UTC_RE.fullmatch(receipt["observed_at_utc"]) is None
    ):
        raise ValueError("canary gate terminal UIDs are invalid")


def validate_reconciliation_gate(
    receipt: dict[str, Any], root: Path, *, evidence_root: Path | None = None
) -> None:
    plans = validate_all(root)
    remaining_cells = sorted(
        (
            {
                "model": plan["model"]["served_id"],
                "controller": controller,
                "selection_rank": row["selection_rank"],
                "attempt": row["attempt"],
                "task_version_id": row["task_version_id"],
                "cell_id": row["cell_id"],
                "execution_id": row["execution_id"],
                "run_id": row["run_id"],
            }
            for controller, plan in plans.items()
            for row in plan["attempts"]
        ),
        key=lambda row: (row["model"], row["selection_rank"], row["attempt"]),
    )
    execution_ids = sorted(
        row["execution_id"] for plan in plans.values() for row in plan["attempts"]
    )
    expected_digest = self_hosted.sha256(self_hosted.canonical_json(execution_ids))
    remaining_digest = self_hosted.sha256(self_hosted.canonical_json(remaining_cells))
    if (
        receipt.get("schema_version") != RECONCILIATION_GATE_SCHEMA
        or receipt.get("status") != "CLEAR"
        or receipt.get("planned_execution_count") != 798
        or receipt.get("planned_execution_ids_sha256") != expected_digest
        or receipt.get("remaining_cells") != remaining_cells
        or receipt.get("remaining_cells_sha256") != remaining_digest
        or receipt.get("fleet_api_collisions") != 0
        or receipt.get("kubernetes_job_or_pod_collisions") != 0
        or receipt.get("sfs_output_collisions") != 0
        or receipt.get("global_claim_collisions") != 0
        or receipt.get("active_or_accepted_cell_collisions") != 0
        or receipt.get("checked_immediately_before_release") is not True
        or receipt.get("observer_job_succeeded") is not True
        or receipt.get("observer_pod_restarts") != 0
        or receipt.get("methods") != ["GET"]
        or receipt.get("mutation_calls") != 0
        or receipt.get("checked_job_names") != sorted(plan["job_name"] for plan in plans.values())
        or receipt.get("checked_output_roots")
        != sorted(plan["sfs_root"] for plan in plans.values())
        or receipt.get("prompts_traces_flags_or_scores_included") is not False
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
    ):
        raise ValueError("bulk duplicate reconciliation gate is not clear")
    observer = receipt.get("runtime")
    collector = receipt.get("collector_runtime")
    if (
        not isinstance(observer, dict)
        or UUID_RE.fullmatch(str(observer.get("job_uid"))) is None
        or UUID_RE.fullmatch(str(observer.get("pod_uid"))) is None
        or not isinstance(receipt.get("observed_at_utc"), str)
        or UTC_RE.fullmatch(receipt["observed_at_utc"]) is None
        or COMMIT_RE.fullmatch(str(receipt.get("observer_package_commit"))) is None
        or not isinstance(collector, dict)
        or UUID_RE.fullmatch(str(collector.get("job_uid"))) is None
        or UUID_RE.fullmatch(str(collector.get("pod_uid"))) is None
    ):
        raise ValueError("bulk reconciliation observer provenance is invalid")

    source_ref = receipt.get("source_observation")
    if not isinstance(source_ref, dict) or set(source_ref) != {
        "path",
        "file_sha256",
        "receipt_sha256",
    }:
        raise ValueError("bulk reconciliation source observation binding is invalid")
    expected_source_path = str(RECONCILIATION_ROOT / "OBSERVATION.json")
    source_path = _evidence_path(
        root,
        source_ref["path"],
        "source observation",
        evidence_root=evidence_root,
    )
    source = load(source_path)
    expected_job_names = sorted(plan["job_name"] for plan in plans.values())
    expected_output_roots = sorted(plan["sfs_root"] for plan in plans.values())
    source_requests = source.get("request_counts") or {}
    source_generation5 = source.get("generation5")
    if (
        source_ref["path"] != expected_source_path
        or source_ref["file_sha256"] != file_sha256(source_path)
        or source_ref["receipt_sha256"] != source.get("receipt_sha256")
        or source.get("receipt_sha256") != digest(source, "receipt_sha256")
        or source.get("schema_version") != RECONCILIATION_OBSERVATION_SCHEMA
        or source.get("status") != "CLEAR_PENDING_SOURCE_JOB_ACCEPTANCE"
        or source.get("campaign_id") != exact.EXPECTED_CAMPAIGN_ID
        or source.get("universe_sha256") != _universe(root)["universe_sha256"]
        or source.get("remaining_cells") != remaining_cells
        or source.get("remaining_cells_sha256") != remaining_digest
        or source.get("planned_execution_count") != 798
        or source.get("planned_execution_ids_sha256") != expected_digest
        or source.get("collisions")
        != {
            "fleet_api": 0,
            "kubernetes_job_or_pod": 0,
            "sfs_output": 0,
            "global_claim": 0,
            "active_or_accepted_cell": 0,
        }
        or source.get("runtime") != observer
        or source.get("checked_job_names") != expected_job_names
        or source.get("checked_output_roots") != expected_output_roots
        or type(source_requests.get("task_session_queries")) is not int
        or not 1 <= source_requests["task_session_queries"] <= 100
        or type(source_requests.get("session_rows_examined")) is not int
        or source_requests["session_rows_examined"] < 2
        or source_requests.get("transcript_queries") != 0
        or source_requests.get("mutation_calls") != 0
        or UTC_RE.fullmatch(str(source.get("observed_at_utc"))) is None
        or not isinstance(source_generation5, dict)
        or set(source_generation5) != set(CANARIES)
        or source.get("privacy")
        != {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
    ):
        raise ValueError("bulk reconciliation source observation drifted")

    inventory_ref = receipt.get("exact100_inventory")
    if not isinstance(inventory_ref, dict) or set(inventory_ref) != {
        "path",
        "file_sha256",
        "receipt_sha256",
    }:
        raise ValueError("bulk reconciliation inventory binding is invalid")
    inventory_path = _evidence_path(
        root,
        inventory_ref["path"],
        "reconciliation inventory",
        evidence_root=evidence_root,
    )
    inventory = load(inventory_path)
    if (
        inventory_ref["file_sha256"] != file_sha256(inventory_path)
        or inventory_ref["receipt_sha256"] != inventory.get("receipt_sha256")
        or source.get("exact100_inventory") != inventory_ref
        or source_requests["task_session_queries"]
        != len({row["task"]["key"] for row in inventory.get("tasks", [])})
    ):
        raise ValueError("bulk reconciliation inventory binding drifted")
    validate_inventory_gate(inventory, root)

    inventory_package_ref = receipt.get("exact100_inventory_package")
    inventory_package_fields = {
        "path",
        "file_sha256",
        "package_sha256",
        "package_commit",
        "bootstrap_configmap",
        "bootstrap_configmap_uid",
        "intent_configmap",
        "intent_configmap_uid",
        "producer_runtime",
    }
    if (
        not isinstance(inventory_package_ref, dict)
        or set(inventory_package_ref) != inventory_package_fields
        or inventory_package_ref.get("path")
        != str(RECONCILIATION_ROOT / "exact100-inventory-package.json")
        or SHA256_RE.fullmatch(str(inventory_package_ref.get("file_sha256"))) is None
        or SHA256_RE.fullmatch(str(inventory_package_ref.get("package_sha256"))) is None
        or COMMIT_RE.fullmatch(str(inventory_package_ref.get("package_commit"))) is None
        or inventory_package_ref.get("bootstrap_configmap")
        != "chris-cyber-exact100-pass4-inventory-bootstrap-v2"
        or inventory_package_ref.get("intent_configmap")
        != "chris-cyber-exact100-pass4-inventory-intent-v2"
        or UUID_RE.fullmatch(str(inventory_package_ref.get("bootstrap_configmap_uid"))) is None
        or UUID_RE.fullmatch(str(inventory_package_ref.get("intent_configmap_uid"))) is None
    ):
        raise ValueError("bulk reconciliation inventory package binding is invalid")
    producer_runtime = inventory_package_ref.get("producer_runtime")
    if (
        not isinstance(producer_runtime, dict)
        or producer_runtime.get("job_uid") != (inventory.get("runtime") or {}).get("job_uid")
        or producer_runtime.get("pod_uid") != (inventory.get("runtime") or {}).get("pod_uid")
        or producer_runtime.get("pod_restarts") != 0
    ):
        raise ValueError("bulk reconciliation inventory producer runtime drifted")
    inventory_package_path = _evidence_path(
        root,
        inventory_package_ref["path"],
        "reconciliation inventory package",
        evidence_root=evidence_root,
    )
    inventory_package = load(inventory_package_path)
    from evals.fleet import exact_pass4_task_inventory_package as inventory_packager

    inventory_packager.validate_package_manifest(inventory_package)
    if (
        inventory_package_ref["file_sha256"] != file_sha256(inventory_package_path)
        or inventory_package_ref["package_sha256"] != inventory_package.get("package_sha256")
        or inventory_package_ref["package_commit"] != inventory_package.get("package_commit")
        or source.get("exact100_inventory_package") != inventory_package_ref
    ):
        raise ValueError("bulk reconciliation inventory package bytes drifted")

    gate_refs = receipt.get("generation5_gate_receipts")
    if not isinstance(gate_refs, dict) or set(gate_refs) != set(CANARIES):
        raise ValueError("bulk reconciliation generation-5 gate bindings are invalid")
    for model, ref in gate_refs.items():
        if not isinstance(ref, dict) or set(ref) != {"path", "receipt_sha256"}:
            raise ValueError("bulk reconciliation generation-5 gate binding is invalid")
        gate_path = _evidence_path(
            root,
            ref["path"],
            f"{model} reconciliation canary",
            evidence_root=evidence_root,
        )
        expected_path = str(RECONCILIATION_ROOT / f"{model}-generation5-gate.json")
        gate = load(gate_path)
        if ref["path"] != expected_path or ref["receipt_sha256"] != gate.get("receipt_sha256"):
            raise ValueError("bulk reconciliation generation-5 gate binding drifted")
        source_gate = source_generation5[model]
        scoring_release_path = _evidence_path(
            root,
            gate["evidence"]["release_path"],
            f"{model} copied scoring release",
            evidence_root=evidence_root,
        )
        scoring_release = load(scoring_release_path)
        if (
            not isinstance(source_gate, dict)
            or source_gate.get("cell_id")
            != load(root / CANARY_SPEC_PATHS[model])["statistical_cell"]["cell_id"]
            or source_gate.get("execution_id") != gate.get("execution_id")
            or source_gate.get("session_id") != gate.get("session_id")
            or source_gate.get("verifier_execution_id") != gate.get("verifier_execution_id")
            or source_gate.get("claim_receipt_sha256")
            != load(
                _evidence_path(
                    root,
                    gate["evidence"]["claim_path"],
                    f"{model} canary claim",
                    evidence_root=evidence_root,
                )
            ).get("receipt_sha256")
            or source_gate.get("terminal_receipt_sha256") != gate.get("terminal_receipt_sha256")
            or source_gate.get("scoring_release_file_sha256") != file_sha256(scoring_release_path)
            or source_gate.get("scoring_release_receipt_sha256")
            != scoring_release.get("receipt_sha256")
        ):
            raise ValueError("bulk source observation generation-5 binding drifted")
        validate_canary_gate(gate, model, root, evidence_root=evidence_root)


def validate_reconciliation_collector_completion(
    receipt: dict[str, Any], job: dict[str, Any], pods: dict[str, Any]
) -> None:
    """Bind the emitted terminal to the accepting Job's later terminal state."""
    collector = receipt.get("collector_runtime") or {}
    metadata = job.get("metadata") or {}
    conditions = (job.get("status") or {}).get("conditions") or []
    pod_rows = pods.get("items") if isinstance(pods, dict) else None
    if (
        metadata.get("name") != RECONCILIATION_ACCEPT_JOB
        or metadata.get("namespace") != "fleet-train-jobs"
        or metadata.get("uid") != collector.get("job_uid")
        or not any(
            isinstance(row, dict) and row.get("type") == "Complete" and row.get("status") == "True"
            for row in conditions
        )
        or any(
            isinstance(row, dict) and row.get("type") == "Failed" and row.get("status") == "True"
            for row in conditions
        )
        or not isinstance(pod_rows, list)
        or len(pod_rows) != 1
    ):
        raise ValueError("bulk reconciliation accept Job is not exclusively Complete")
    pod = pod_rows[0]
    status = pod.get("status") or {}
    all_statuses = [
        *status.get("initContainerStatuses", []),
        *status.get("containerStatuses", []),
    ]
    terminated = [
        (row.get("state") or {}).get("terminated") for row in all_statuses if isinstance(row, dict)
    ]
    if (
        (pod.get("metadata") or {}).get("uid") != collector.get("pod_uid")
        or status.get("phase") != "Succeeded"
        or not all_statuses
        or any(row.get("restartCount") != 0 for row in all_statuses)
        or any(not isinstance(row, dict) or row.get("exitCode") != 0 for row in terminated)
        or len(terminated) != len(all_statuses)
    ):
        raise ValueError("bulk reconciliation accept Pod is not cleanly Succeeded")


def _evidence_path(
    root: Path,
    value: Any,
    label: str,
    *,
    evidence_root: Path | None = None,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} gate path is absent")
    relative = Path(value)
    if ".." in relative.parts:
        raise ValueError(f"{label} gate path is unsafe")
    if relative.is_absolute():
        if not relative.is_relative_to(Path("/mnt/sfs/jobs")):
            raise ValueError(f"{label} absolute evidence must be under the jobs SFS")
        if evidence_root is not None:
            resolved_root = evidence_root.resolve(strict=True)
            mapped = (resolved_root / relative.relative_to(Path("/mnt/sfs/jobs"))).resolve(
                strict=True
            )
            if not mapped.is_relative_to(resolved_root):
                raise ValueError(f"{label} mapped evidence escapes the evidence root")
            return mapped
        resolved = relative.resolve(strict=True)
        if not resolved.is_relative_to(Path("/mnt/sfs/jobs")):
            raise ValueError(f"{label} SFS evidence escapes the jobs root")
        return resolved
    resolved_root = root.resolve(strict=True)
    resolved = (resolved_root / relative).resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} gate path escapes the repository")
    return resolved


def _gate_path(root: Path, value: Any, label: str) -> Path:
    return _evidence_path(root, value, label)


def validate_package_commit(root: Path, package_commit: str) -> None:
    """Bind launch bytes to one clean exact Git commit, not mutable working files."""
    from evals.fleet import exact_pass4_bulk_package_v1 as package

    if COMMIT_RE.fullmatch(package_commit) is None:
        raise ValueError("bulk release package commit is invalid")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("bulk package Git authority unavailable") from exc
    if head != package_commit:
        raise ValueError("bulk package commit is not checked out exactly")
    controller_paths = {path for rows in package.CONTROLLER_PATHS.values() for path in rows}
    paths = sorted(set(package.CORE_PATHS) | controller_paths)
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("bulk package paths are not clean at the exact commit")
    for relative in paths:
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"{package_commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        local = root / relative
        if local.is_symlink() or not local.is_file() or local.read_bytes() != committed:
            raise ValueError(f"bulk package byte binding drifted: {relative}")


def _release_body(
    root: Path,
    *,
    package_commit: str,
    released_at: str,
    gate_paths: dict[str, str],
    plans: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from evals.fleet import exact_pass4_bulk_package_v1 as package

    plans = validate_all(root) if plans is None else plans
    built_package = package.build_package(root)
    return {
        "schema_version": RELEASE_SCHEMA,
        "status": "RELEASED",
        "launch_authorized": True,
        "campaign_id": exact.EXPECTED_CAMPAIGN_ID,
        "universe_sha256": _universe(root)["universe_sha256"],
        "package_commit": package_commit,
        "released_at_utc": released_at,
        "package": {
            "schema_version": package.SCHEMA,
            "aggregate_sha256": built_package["aggregate_sha256"],
            "objects": sorted(built_package["configmaps"]),
            "object_json_bytes": built_package["object_json_bytes"],
            "release_included": False,
        },
        "implementation": {
            "module_path": MODULE_PATH,
            "module_sha256": file_sha256(root / MODULE_PATH),
            "package_path": PACKAGE_PATH,
            "package_sha256": file_sha256(root / PACKAGE_PATH),
            "runtime_path": RUNTIME_PATH,
            "runtime_sha256": file_sha256(root / RUNTIME_PATH),
            "run_path": RUN_PATH,
            "run_sha256": file_sha256(root / RUN_PATH),
            "renderer_path": RENDER_PATH,
            "renderer_sha256": file_sha256(root / RENDER_PATH),
            "submit_path": SUBMIT_PATH,
            "submit_sha256": file_sha256(root / SUBMIT_PATH),
            "manifest_path": MANIFEST_PATH,
            "manifest_sha256": file_sha256(root / MANIFEST_PATH),
            "snapshot_path": SNAPSHOT_PATH,
            "snapshot_sha256": file_sha256(root / SNAPSHOT_PATH),
            "held_path": HELD_PATH,
            "held_receipt_sha256": load(root / HELD_PATH)["receipt_sha256"],
        },
        "controllers": [
            {
                "controller": key,
                "spec_path": SPEC_PATHS[key],
                "spec_sha256": all_specs(root)[key]["spec_sha256"],
                "plan_sha256": plans[key]["plan_sha256"],
                "job_name": plans[key]["job_name"],
                "cell_count": plans[key]["new_session_count"],
            }
            for key in CONTROLLERS
        ],
        "gate_receipts": gate_paths,
        "authorization": {
            "author": "/root",
            "statement": RELEASE_STATEMENT,
            "create_once": True,
        },
        "policy": {
            "four_one_stream_jobs": True,
            "maximum_streams_per_hosted_endpoint": 2,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "create_once": True,
            "global_claim_before_model_call": True,
            "accepted_active_or_claimed_cells_nonrepeatable": True,
            "infrastructure_failure_quarantines_only_exact_cell": True,
        },
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }


def build_release(
    root: Path,
    *,
    package_commit: str,
    released_at: str,
    evidence_root: Path,
    collector_job: dict[str, Any],
    collector_pods: dict[str, Any],
) -> dict[str, Any]:
    """Build and fully validate one append-only release from fixed gate paths."""
    body = _release_body(
        root,
        package_commit=package_commit,
        released_at=released_at,
        gate_paths=copy.deepcopy(FIXED_GATE_PATHS),
    )
    release = {**body, "receipt_sha256": self_hosted.sha256(self_hosted.canonical_json(body))}
    validate_release(
        release,
        root,
        evidence_root=evidence_root,
        collector_job=collector_job,
        collector_pods=collector_pods,
        reconciliation_max_age_seconds=900,
    )
    return release


def validate_release(
    release: dict[str, Any],
    root: Path,
    *,
    package_commit_authority: Path | None = None,
    evidence_root: Path | None = None,
    collector_job: dict[str, Any] | None = None,
    collector_pods: dict[str, Any] | None = None,
    reconciliation_max_age_seconds: int | None = None,
) -> None:
    from evals.fleet import exact_pass4_bulk_package_v1 as package

    plans = validate_all(root)
    gate_paths = release.get("gate_receipts")
    if not isinstance(gate_paths, dict):
        raise ValueError("bulk release gate receipts are absent")
    if set(gate_paths) != {
        "exact100_inventory",
        "exact100_inventory_package",
        "qwen_generation5_canary",
        "glm_generation5_canary",
        "duplicate_reconciliation",
    }:
        raise ValueError("bulk release gate receipt set drifted")
    inventory = load(
        _evidence_path(
            root,
            gate_paths.get("exact100_inventory"),
            "inventory",
            evidence_root=evidence_root,
        )
    )
    inventory_package = load(
        _evidence_path(
            root,
            gate_paths.get("exact100_inventory_package"),
            "inventory package",
            evidence_root=evidence_root,
        )
    )
    qwen = load(
        _evidence_path(
            root,
            gate_paths.get("qwen_generation5_canary"),
            "qwen canary",
            evidence_root=evidence_root,
        )
    )
    glm = load(
        _evidence_path(
            root,
            gate_paths.get("glm_generation5_canary"),
            "glm canary",
            evidence_root=evidence_root,
        )
    )
    reconciliation = load(
        _evidence_path(
            root,
            gate_paths.get("duplicate_reconciliation"),
            "reconciliation",
            evidence_root=evidence_root,
        )
    )
    validate_inventory_gate(inventory, root)
    from evals.fleet import exact_pass4_task_inventory_package as inventory_packager

    inventory_packager.validate_package_manifest(inventory_package)
    if inventory_package["runtime_contract"]["sfs_terminal_path"] != gate_paths.get(
        "exact100_inventory"
    ):
        raise ValueError("inventory package does not bind the selected producer terminal")
    validate_canary_gate(qwen, "qwen3.8-27b", root, evidence_root=evidence_root)
    validate_canary_gate(glm, "glm-5.3", root, evidence_root=evidence_root)
    validate_reconciliation_gate(reconciliation, root, evidence_root=evidence_root)
    if reconciliation_max_age_seconds is not None:
        if type(reconciliation_max_age_seconds) is not int or not (
            1 <= reconciliation_max_age_seconds <= 3600
        ):
            raise ValueError("bulk reconciliation freshness bound is invalid")
        observed = datetime.strptime(
            reconciliation["observed_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=UTC)
        age = (datetime.now(UTC) - observed).total_seconds()
        if age < -60 or age > reconciliation_max_age_seconds:
            raise ValueError("bulk reconciliation is not fresh at release")
    gate_refs = reconciliation["generation5_gate_receipts"]
    if (
        gate_paths["duplicate_reconciliation"] != str(RECONCILIATION_ROOT / "TERMINAL.json")
        or gate_paths["exact100_inventory"] != reconciliation["exact100_inventory"]["path"]
        or gate_paths["exact100_inventory_package"]
        != reconciliation["exact100_inventory_package"]["path"]
        or gate_paths["qwen_generation5_canary"] != gate_refs["qwen3.8-27b"]["path"]
        or gate_paths["glm_generation5_canary"] != gate_refs["glm-5.3"]["path"]
        or gate_refs["qwen3.8-27b"]["receipt_sha256"] != qwen["receipt_sha256"]
        or gate_refs["glm-5.3"]["receipt_sha256"] != glm["receipt_sha256"]
    ):
        raise ValueError("bulk release canaries do not match reconciliation")
    if collector_job is None or collector_pods is None:
        raise ValueError("bulk release requires live reconciliation collector evidence")
    validate_reconciliation_collector_completion(reconciliation, collector_job, collector_pods)
    package_commit = release.get("package_commit")
    released_at = release.get("released_at_utc")
    if not isinstance(package_commit, str) or COMMIT_RE.fullmatch(package_commit) is None:
        raise ValueError("bulk release package commit is invalid")
    if reconciliation.get("observer_package_commit") != package_commit:
        raise ValueError("bulk reconciliation observer is not from the released package commit")
    if package_commit_authority is None:
        validate_package_commit(root, package_commit)
    else:
        from evals.fleet import immutable_submission_snapshot as snapshot

        controller_paths = {path for rows in package.CONTROLLER_PATHS.values() for path in rows}
        snapshot.verify_paths(
            package_commit_authority.resolve(),
            package_commit,
            root.resolve(),
            set(package.CORE_PATHS) | controller_paths,
        )
    if not isinstance(released_at, str) or UTC_RE.fullmatch(released_at) is None:
        raise ValueError("bulk release timestamp is invalid")
    release_time = datetime.strptime(released_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    observed_time = datetime.strptime(
        reconciliation["observed_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=UTC)
    if release_time < observed_time or (release_time - datetime.now(UTC)).total_seconds() > 60:
        raise ValueError("bulk release timestamp does not follow reconciliation")
    expected = _release_body(
        root,
        package_commit=package_commit,
        released_at=released_at,
        gate_paths=gate_paths,
        plans=plans,
    )
    if {
        key: value for key, value in release.items() if key != "receipt_sha256"
    } != expected or release.get("receipt_sha256") != digest(release, "receipt_sha256"):
        raise ValueError("bulk release drifted")


def build_held(root: Path) -> dict[str, Any]:
    plans = validate_all(root)
    body = {
        "schema_version": HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "campaign_id": exact.EXPECTED_CAMPAIGN_ID,
        "universe_sha256": _universe(root)["universe_sha256"],
        "remaining_cells": {"qwen3.8-27b": 399, "glm-5.3": 399, "total": 798},
        "controllers": [
            {
                "controller": key,
                "spec_path": SPEC_PATHS[key],
                "spec_sha256": all_specs(root)[key]["spec_sha256"],
                "plan_sha256": plans[key]["plan_sha256"],
                "job_name": plans[key]["job_name"],
                "cell_count": plans[key]["new_session_count"],
            }
            for key in CONTROLLERS
        ],
        "required_gates": [
            "digest_valid_exact100_get_only_inventory_receipt",
            "accepted_qwen_generation5_canary_with_session_verifier_ingestion_cleanup_zero_restarts_and_released_lease",
            "accepted_glm_generation5_canary_with_session_verifier_ingestion_cleanup_zero_restarts_and_released_lease",
            "fresh_exhaustive_api_kubernetes_sfs_and_global_claim_reconciliation",
            "append_only_bulk_release_receipt",
        ],
        "execution_policy": {
            "four_one_stream_jobs": True,
            "maximum_streams_per_hosted_endpoint": 2,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "create_once": True,
            "global_claim_before_model_call": True,
            "accepted_active_or_claimed_cells_nonrepeatable": True,
            "infrastructure_failure_quarantines_only_exact_cell": True,
            "launch_authorized": False,
        },
        "objects_created": False,
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    return {**body, "receipt_sha256": self_hosted.sha256(self_hosted.canonical_json(body))}


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    expected = build_held(root)
    if receipt != expected or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256"):
        raise ValueError("bulk held receipt drifted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preview", "validate-held", "build-release", "validate-release")
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--collector-job", type=Path)
    parser.add_argument("--collector-pods", type=Path)
    parser.add_argument("--package-commit")
    parser.add_argument("--released-at-utc")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    if args.command == "preview":
        plans = validate_all(root)
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "HELD",
                    "controllers": {key: plan["new_session_count"] for key, plan in plans.items()},
                    "launch_authorized": False,
                    "objects_created": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-release":
        if any(
            value is None
            for value in (
                args.evidence_root,
                args.collector_job,
                args.collector_pods,
                args.package_commit,
                args.released_at_utc,
                args.output,
            )
        ):
            parser.error(
                "build-release requires --evidence-root, --collector-job, "
                "--collector-pods, --package-commit, --released-at-utc, and --output"
            )
        output = args.output.resolve()
        if output.exists() or output.is_relative_to(root):
            parser.error(
                "build-release output must be an absent append-only path outside the repository"
            )
        release = build_release(
            root,
            package_commit=args.package_commit,
            released_at=args.released_at_utc,
            evidence_root=args.evidence_root.resolve(),
            collector_job=load(args.collector_job),
            collector_pods=load(args.collector_pods),
        )
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self_hosted.write_json_once(output, release)
        return 0
    if args.receipt is None:
        parser.error(f"{args.command} requires --receipt")
    receipt = load(args.receipt)
    if args.command == "validate-held":
        validate_held(receipt, root)
    else:
        if any(
            value is None for value in (args.evidence_root, args.collector_job, args.collector_pods)
        ):
            parser.error(
                "validate-release requires --evidence-root, --collector-job, and --collector-pods"
            )
        validate_release(
            receipt,
            root,
            evidence_root=args.evidence_root.resolve(),
            collector_job=load(args.collector_job),
            collector_pods=load(args.collector_pods),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
