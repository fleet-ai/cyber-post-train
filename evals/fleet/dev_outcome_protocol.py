"""Freeze score-blind Fleet development-task outcome protocols.

The input binding audit is private because the upstream task response contains
private fields.  This compiler accepts only its already-sanitized projection and
emits exact task/runtime/verifier metadata; it never reads a prompt, trace,
answer, flag, or historical score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from training.io import atomic_write_json, canonical_json, digest_json

TASK_SET_SCHEMA = "cyber_fleet_eval_task_set_v1"
PROTOCOL_SCHEMA = "cyber_fleet_dev_outcome_protocol_v1"
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DEV_TASKS = 20
PASS_K = 4
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")

HARNESS = {
    "harness": "opencode",
    "harness_version": "1.18.27",
    "release_asset_sha256": (
        "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
    ),
    "provider_adapter": "@ai-sdk/openai-compatible",
    "context_management": "opencode_1.18.27_native_compaction_autocontinue_v2",
    "context_window_size": 262144,
    "compaction_headroom_tokens": 20000,
    "max_output_tokens": 32768,
    "max_model_requests": 600,
    "timeout_seconds": 28800,
    "tools": ["bash", "submit_report"],
    "tool_catalog_sha256": (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    ),
}
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "base_seed": 42,
    "attempt_seeds": [42, 43, 44, 45],
    "seed_rule": "attempt N uses base_seed + N - 1 modulo 2^31",
}
UNBOUND_CHECKPOINT_FIELDS = (
    "checkpoint_seal_sha256",
    "export_receipt_sha256",
    "weights_manifest_sha256",
    "tokenizer_manifest_sha256",
    "chat_template_sha256",
    "staging_receipt_sha256",
    "serving_registration_receipt_sha256",
    "live_parity_receipt_sha256",
    "served_model_id",
    "serving_route_profile_sha256",
    "agent_image_digest",
    "proxy_image_digest",
)


def _sealed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "sha256": digest_json(value)}


def _check_seal(value: dict[str, Any], schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid {schema} seal")


def _uuid(value: Any, label: str) -> str:
    try:
        canonical = str(uuid.UUID(value)) if isinstance(value, str) else None
    except ValueError:
        canonical = None
    if canonical != value:
        raise ValueError(f"{label} must be a canonical UUID")
    return value


def _sha(value: Any, label: str, *, prefixed: bool = True) -> str:
    candidate = value if prefixed else "sha256:" + str(value)
    if not isinstance(candidate, str) or SHA256.fullmatch(candidate) is None:
        raise ValueError(f"{label} must be an exact SHA-256")
    return candidate


def _pretty_file_sha256(value: dict[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_task_set(split: dict[str, Any], audit: dict[str, Any], variant: str) -> dict[str, Any]:
    """Join one frozen dev split to exact sanitized live task bindings."""
    _check_seal(split, "cyber_study_split_v1")
    _check_seal(audit, "cyber_task_current_binding_audit_v1")
    if variant not in {"a", "b"}:
        raise ValueError("split variant must be a or b")
    dev = split.get("evaluation", {}).get("dev")
    if not isinstance(dev, dict):
        raise ValueError("study split lacks its sealed development selection")
    _check_seal(dev, "cyber_eval_task_selection_v1")
    selected = dev.get("tasks")
    if not isinstance(selected, list) or len(selected) != DEV_TASKS:
        raise ValueError("Fleet development selection must contain exactly 20 tasks")
    if any(row.get("split") != "dev" for row in selected):
        raise ValueError("development selection contains a non-development task")
    selected_ids = [row.get("task_version_id") for row in selected]
    selected_groups = [row.get("group_id") for row in selected]
    if len(set(selected_ids)) != DEV_TASKS or len(set(selected_groups)) != DEV_TASKS:
        raise ValueError("development task versions and families must be unique")

    audited = audit.get("task_versions")
    if not isinstance(audited, list):
        raise ValueError("binding audit lacks task rows")
    by_id = {row.get("task_version_id"): row for row in audited}
    if len(by_id) != len(audited):
        raise ValueError("binding audit contains duplicate task versions")

    rows = []
    for selected_row in selected:
        version = _uuid(selected_row.get("task_version_id"), "task version")
        binding = by_id.get(version)
        if not isinstance(binding, dict):
            raise ValueError("development task is absent from the binding audit")
        _check_seal(binding, "cyber_task_current_binding_v1")
        if (
            binding.get("status") != "valid"
            or binding.get("task_key") != selected_row.get("task_key")
            or binding.get("private_fields_persisted") is not False
        ):
            raise ValueError("development task has no clean exact binding")
        task = binding.get("task")
        environment = binding.get("environment")
        verifier = binding.get("verifier")
        if not all(isinstance(item, dict) for item in (task, environment, verifier)):
            raise ValueError("development task binding is incomplete")
        if task.get("key") != binding["task_key"] or task.get("version_id") != version:
            raise ValueError("task identity differs inside its binding")
        if environment.get("id") is None or environment.get("data_id") is None:
            raise ValueError("environment or starting-data identity is absent")
        _uuid(environment.get("version_id"), "environment version")
        _uuid(verifier.get("id"), "verifier")
        _uuid(verifier.get("version_id"), "verifier version")
        for key in ("prompt_sha256", "env_variables_sha256", "output_json_schema_sha256"):
            _sha(task.get(key), f"task {key}")
        _sha(environment.get("runtime_seed_content_sha256"), "runtime seed", prefixed=False)
        _sha(verifier.get("sha256"), "verifier source", prefixed=False)
        exact = {"task": task, "environment": environment, "verifier": verifier}
        rows.append(
            {
                "task_key": binding["task_key"],
                "task_version_id": version,
                "env_key": environment["id"],
                "env_version": environment["version"],
                "environment_version_id": environment["version_id"],
                "data_key": environment["data_id"],
                "data_version": environment["data_version"],
                "group_id": selected_row["group_id"],
                "exact_binding": exact,
                "exact_binding_sha256": digest_json([task, environment, verifier]),
            }
        )
    rows.sort(key=lambda row: row["task_version_id"])
    return _sealed(
        {
            "schema": TASK_SET_SCHEMA,
            "selection_role": "fleet_dev_hpo",
            "split_variant": variant,
            "study_split_sha256": split["sha256"],
            "study_dev_selection_sha256": dev["sha256"],
            "final_test_lock_sha256": split["final_test_lock_sha256"],
            "binding_audit_sha256": audit["sha256"],
            "binding_observed_at": audit["observed_at"],
            "source_job_id": None,
            "tasks": rows,
            "task_count": len(rows),
            "training_data_eligible": False,
            "private_fields_persisted": False,
        }
    )


def build_protocol(task_set: dict[str, Any], *, variant: str, task_set_path: str) -> dict[str, Any]:
    """Build the frozen parent protocol used by outcome-only SFT manifests."""
    validate_task_set(task_set)
    if task_set.get("split_variant") != variant or task_set.get("task_count") != DEV_TASKS:
        raise ValueError("task set does not match the requested development split")
    if not isinstance(task_set_path, str) or not task_set_path.startswith("configs/"):
        raise ValueError("task set needs a repository-relative configs/ path")
    unbound = {field: None for field in UNBOUND_CHECKPOINT_FIELDS}
    return _sealed(
        {
            "schema": PROTOCOL_SCHEMA,
            "purpose": "Qwen3.8-27B SFT hyperparameter selection on fresh Fleet task outcomes",
            "split_variant": variant,
            "task_set": {
                "path": task_set_path,
                "sha256": task_set["sha256"],
                "file_sha256": _pretty_file_sha256(task_set),
                "task_count": DEV_TASKS,
                "selection_role": "fleet_dev_hpo",
                "training_data_eligible": False,
                "exact_task_binding_required_at_preflight_and_execution": True,
            },
            "model": {
                "repository": MODEL_REPOSITORY,
                "base_revision": MODEL_REVISION,
                "tokenizer_revision": MODEL_REVISION,
                "comparison": "matched base versus one post-SFT checkpoint",
                "only_intended_difference": "post-SFT weight manifest",
            },
            "checkpoint_binding_template": {
                "state": "unbound",
                "launchable": False,
                "fields": unbound,
                "binding_rule": (
                    "An arm-specific immutable child may fill every null exactly once, must "
                    "reference this parent protocol SHA-256, and may not change any other field."
                ),
                "acceptance_rule": (
                    "Require accepted checkpoint seal, BF16 export and exact reload, create-once "
                    "staging, serving registration, and live base/post parity before prepare."
                ),
            },
            "harness": HARNESS,
            "sampling": SAMPLING,
            "pass_k": PASS_K,
            "concurrency_per_worker": 1,
            "outcome_authority": {
                "required_cyber_contract": {
                    "evidence_schema": "1.0.0",
                    "verifier_contract": "3.0.0",
                    "submission_protocol": "2.0.0",
                },
                "provisioning_route_template": (
                    "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
                ),
                "scoring_route_template": (
                    "/v1/rollout-rewards/{task_key}/versions/{task_version_id}"
                ),
                "valid_outcome_requires": [
                    "exact frozen task, environment, starting-data and verifier bindings",
                    "normal terminal OpenCode event stream without an unfinished final step",
                    "authoritative version-bound grading",
                    "complete private local result evidence",
                    "challenge environment and all harness containers cleaned",
                ],
            },
            "metrics": {
                "primary": {
                    "name": "fleet_dev_pass_at_1",
                    "unit": "task_family",
                    "attempt": 1,
                    "seed": 42,
                    "definition": (
                        "Mean full-task solve indicator across the 20 planned task families, "
                        "using only attempt 1 authoritative valid outcomes."
                    ),
                    "hpo_direction": "maximize",
                    "eligibility": "all 20 attempt-1 outcomes must be valid",
                },
                "descriptive_only": [
                    "fleet_dev_pass_at_4",
                    "attempt_1_mean_authoritative_fractional_reward",
                    "best_of_4_mean_authoritative_fractional_reward",
                    "valid_outcome_rate",
                    "infrastructure_invalid_count",
                    "report_submission_rate",
                    "model_requests_tokens_and_elapsed_time",
                ],
                "uncertainty": {
                    "method": "paired task-family bootstrap",
                    "resamples": 10000,
                    "seed": 20260911,
                    "interval": 0.95,
                },
                "invalid_outcome_policy": (
                    "Never score infrastructure-invalid, interrupted, output-limited, or unknown "
                    "attempts as model failures; the affected metric remains incomplete."
                ),
            },
            "selection_rule": {
                "signal": "fleet_dev_pass_at_1 only",
                "ties": "retain tied arms for a fresh predeclared confirmation",
                "forbidden_tiebreakers": [
                    "training loss",
                    "teacher-reference cross-entropy",
                    "Fleet pass@4 descriptive metrics",
                    "Fleet final-test outcomes",
                    "WebExploitBench outcomes",
                ],
            },
            "heldout_policy": {
                "teacher_reference_cross_entropy": "forbidden",
                "development_tasks_in_training_preferences_or_rl": False,
                "training_loss_role": "fitting diagnostic only",
                "external_benchmarks": (
                    "run after each frozen checkpoint but keep sealed from HPO decisions"
                ),
            },
            "retry_policy": {
                "automatic_retry": False,
                "valid_outcome_retry": "forbidden",
                "ambiguous_or_invalid_attempt": "hold for evidence review",
                "replacement_authority": (
                    "none in this parent protocol; a separately reviewed symmetric amendment is "
                    "required before any replacement execution"
                ),
            },
            "launch_gate": {
                "status": "not_launchable_until_checkpoint_and_serving_bindings_are_complete",
                "dev_first": True,
                "prepare_must_reject_unbound_checkpoint_fields": True,
                "paid_or_scored_work_authorized_by_this_file": False,
            },
        }
    )


def validate_task_set(task_set: dict[str, Any]) -> None:
    _check_seal(task_set, TASK_SET_SCHEMA)
    if (
        task_set.get("selection_role") != "fleet_dev_hpo"
        or task_set.get("training_data_eligible") is not False
        or task_set.get("private_fields_persisted") is not False
    ):
        raise ValueError("task set is not a held-out, sanitized Fleet development set")
    rows = task_set.get("tasks")
    if (
        not isinstance(rows, list)
        or task_set.get("task_count") != DEV_TASKS
        or len(rows) != DEV_TASKS
    ):
        raise ValueError("task set must contain exactly 20 development tasks")
    versions = set()
    groups = set()
    for row in rows:
        version = _uuid(row.get("task_version_id"), "task version")
        environment_version = _uuid(row.get("environment_version_id"), "environment version")
        exact = row.get("exact_binding")
        if not isinstance(exact, dict) or set(exact) != {"task", "environment", "verifier"}:
            raise ValueError("task set lacks its exact binding projection")
        if row.get("exact_binding_sha256") != digest_json(
            [exact["task"], exact["environment"], exact["verifier"]]
        ):
            raise ValueError("task binding projection changed")
        if (
            exact["task"].get("key") != row.get("task_key")
            or exact["task"].get("version_id") != version
            or exact["environment"].get("id") != row.get("env_key")
            or exact["environment"].get("version") != row.get("env_version")
            or exact["environment"].get("version_id") != environment_version
            or exact["environment"].get("data_id") != row.get("data_key")
            or exact["environment"].get("data_version") != row.get("data_version")
        ):
            raise ValueError("flat evaluator tuple differs from its exact task binding")
        if version in versions or row.get("group_id") in groups:
            raise ValueError("task versions and families must remain unique")
        versions.add(version)
        groups.add(row.get("group_id"))


def validate_protocol(protocol: dict[str, Any], task_set: dict[str, Any]) -> None:
    _check_seal(protocol, PROTOCOL_SCHEMA)
    validate_task_set(task_set)
    if protocol.get("task_set", {}).get("sha256") != task_set["sha256"]:
        raise ValueError("protocol and task set differ")
    if protocol["task_set"].get("file_sha256") != _pretty_file_sha256(task_set):
        raise ValueError("protocol and task-set file bytes differ")
    if protocol.get("harness") != HARNESS or protocol.get("sampling") != SAMPLING:
        raise ValueError("Fleet harness or sampling treatment drifted")
    if protocol.get("pass_k") != PASS_K:
        raise ValueError("Fleet development pass@k drifted")
    template = protocol.get("checkpoint_binding_template", {})
    if template.get("launchable") is not False or template.get("state") != "unbound":
        raise ValueError("parent protocol must remain an unbound non-launchable template")
    fields = template.get("fields")
    if (
        not isinstance(fields, dict)
        or set(fields) != set(UNBOUND_CHECKPOINT_FIELDS)
        or any(value is not None for value in fields.values())
    ):
        raise ValueError("checkpoint placeholder fields changed")
    heldout = protocol.get("heldout_policy", {})
    if heldout.get("teacher_reference_cross_entropy") != "forbidden":
        raise ValueError("teacher-reference CE must not select blackbox capability")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--variant", choices=("a", "b"), required=True)
    parser.add_argument("--task-set", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.task_set.exists() or args.protocol.exists():
        raise FileExistsError("refusing to replace a frozen task set or protocol")
    split = json.loads(args.split.read_text())
    audit = json.loads(args.bindings.read_text())
    task_set = build_task_set(split, audit, args.variant)
    try:
        task_set_path = args.task_set.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError as error:
        raise ValueError("task set output must be inside the current repository") from error
    protocol = build_protocol(task_set, variant=args.variant, task_set_path=task_set_path)
    validate_protocol(protocol, task_set)
    atomic_write_json(args.task_set, task_set)
    atomic_write_json(args.protocol, protocol)
    print(canonical_json({"protocol_sha256": protocol["sha256"], "submitted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
