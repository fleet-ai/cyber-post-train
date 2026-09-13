"""Freeze and bind the exact Fleet final-test comparison for one post-training arm.

The checked-in parent is score-blind and non-launchable.  Child materialization
is offline, create-once, and remains non-launchable: a later reviewed evaluator
step must still reopen the referenced receipts, perform live task/route
preflight, and create its own execution claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from evals.fleet import dev_outcome_protocol as final_set
from training.io import canonical_json, digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[2]
TASK_SET_REPO_PATH = "configs/evaluation/qwen38-blackbox-fleet-final-task-set-v1.json"
TASK_SET_PATH = ROOT / TASK_SET_REPO_PATH
PARENT_SCHEMA = "cyber_fleet_final_post_training_parent_v1"
BINDINGS_SCHEMA = "cyber_fleet_final_post_training_bindings_v1"
CHILD_SCHEMA = "cyber_fleet_final_post_training_child_v1"
CANDIDATE_ARM = "rl"
SUPPORTED_CANDIDATE_ARMS = ("teacher_sft", "self_sft", "rl", "sft_then_rl")

EXACT_TASK_SET_SHA256 = "sha256:718936798883348fe16570561679e49d7725e174a1cee4fa0f7821225675548e"
EXACT_TASK_SET_FILE_SHA256 = (
    "sha256:018e9a0fa3de5055b1f19a4ac4db94b5419359625fee5e638c9c306dfce510c7"
)
EXACT_FINAL_LOCK_SHA256 = "sha256:dc625115be8c775de8ecf476bc0d8e06e71b6c7f4160ed50b25eb0308b30d8d7"

ARM_FIELDS = (
    "checkpoint_identity",
    "checkpoint_acceptance_receipt_sha256",
    "export_receipt_sha256",
    "weights_manifest_sha256",
    "tokenizer_manifest_sha256",
    "chat_template_sha256",
    "staging_receipt_sha256",
    "serving_registration_receipt_sha256",
    "served_model_id",
    "serving_route_profile_sha256",
    "serving_execution_contract_sha256",
    "live_pair_parity_receipt_sha256",
    "serving_image",
    "agent_image",
    "proxy_image",
)
MATCHED_ARM_FIELDS = (
    "tokenizer_manifest_sha256",
    "chat_template_sha256",
    "serving_execution_contract_sha256",
    "live_pair_parity_receipt_sha256",
    "serving_image",
    "agent_image",
    "proxy_image",
)
TRAINING_BOUNDARY_FIELDS = (
    "source_class",
    "training_arm_id",
    "training_plan_sha256",
    "training_terminal_acceptance_receipt_sha256",
    "checkpoint_freeze_receipt_sha256",
    "training_input_manifest_sha256",
    "external_benchmark_inputs_used",
    "external_benchmark_outputs_used",
    "external_benchmark_outcomes_used",
    "external_benchmark_derived_inputs_used",
)
RESULT_ISOLATION_FIELDS = (
    "campaign_id",
    "prepared_output_root",
    "result_root",
    "roots_absent",
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_IMMUTABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:-]{2,255}")
_IMAGE = re.compile(r"[A-Za-z0-9._/-]+@sha256:[0-9a-f]{64}")


def _sealed(value: dict[str, Any], field: str = "sha256") -> dict[str, Any]:
    result = copy.deepcopy(value)
    result[field] = digest_json(result)
    return result


def _check_seal(value: dict[str, Any], schema: str, field: str = "sha256") -> None:
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get("schema") != schema or value.get(field) != digest_json(unsigned):
        raise ValueError(f"invalid {schema} seal")


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("protocol input must be a regular nonsymlink file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable):
        raise ValueError("protocol input changed while reading")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("protocol input is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("protocol input must be a JSON object")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    raw = (canonical_json(value) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact SHA-256")
    return value


def _immutable_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or _IMMUTABLE_ID.fullmatch(value) is None
        or any(marker in value.lower() for marker in ("latest", "current", "mutable"))
    ):
        raise ValueError(f"{label} must be a unique immutable identity")
    return value


def _image(value: object, label: str) -> str:
    if not isinstance(value, str) or _IMAGE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an immutable OCI digest reference")
    return value


def _exact_keys(value: object, fields: tuple[str, ...] | set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label} fields differ")
    return value


def _fresh_path(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute fresh path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
        raise ValueError(f"{label} must be an absolute fresh path")
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label} already exists")
    return path


def load_exact_task_set() -> dict[str, Any]:
    task_set = _read(TASK_SET_PATH)
    final_set.validate_final_task_set(task_set)
    if (
        task_set.get("sha256") != EXACT_TASK_SET_SHA256
        or file_sha256(TASK_SET_PATH) != EXACT_TASK_SET_FILE_SHA256
        or task_set.get("final_test_lock_sha256") != EXACT_FINAL_LOCK_SHA256
    ):
        raise ValueError("Fleet final task-set identity changed")
    return task_set


def _candidate_label(candidate_arm: str) -> str:
    labels = {
        "teacher_sft": "teacher-SFT",
        "self_sft": "self-SFT",
        "rl": "RL",
        "sft_then_rl": "SFT-then-RL",
    }
    try:
        return labels[candidate_arm]
    except KeyError as error:
        raise ValueError("unsupported post-training candidate arm") from error


def _parent_body(task_set: dict[str, Any], *, candidate_arm: str = CANDIDATE_ARM) -> dict[str, Any]:
    label = _candidate_label(candidate_arm)
    empty_arm = {field: None for field in ARM_FIELDS}
    empty_boundary = {field: None for field in TRAINING_BOUNDARY_FIELDS}
    return {
        "schema": PARENT_SCHEMA,
        "purpose": f"Sealed Qwen3.8-27B base-versus-{label} Fleet final capability comparison",
        "candidate_arm": candidate_arm,
        "launchable": False,
        "paid_or_scored_work_authorized": False,
        "task_set": {
            "path": TASK_SET_REPO_PATH,
            "sha256": task_set["sha256"],
            "file_sha256": EXACT_TASK_SET_FILE_SHA256,
            "task_count": final_set.FINAL_TASKS,
            "selection_role": "fleet_final_test",
            "final_test_lock_sha256": task_set["final_test_lock_sha256"],
            "training_data_eligible": False,
            "hyperparameter_or_checkpoint_selection_eligible": False,
            "exact_task_binding_required_at_preflight_and_execution": True,
        },
        "model": {
            "repository": final_set.MODEL_REPOSITORY,
            "base_revision": final_set.MODEL_REVISION,
            "tokenizer_revision": final_set.MODEL_REVISION,
            "comparison": f"matched base versus one frozen {label} checkpoint",
            "only_intended_difference": "candidate weight manifest",
        },
        "binding_template": {
            "state": "unbound",
            "candidate_arm": candidate_arm,
            "training_boundary": empty_boundary,
            "arms": {"base": dict(empty_arm), "candidate": dict(empty_arm)},
            "result_isolation": {field: None for field in RESULT_ISOLATION_FIELDS},
            "binding_rule": (
                "One immutable child must fill every null exactly once from accepted receipts, "
                "reference this exact parent, and remain non-launchable."
            ),
        },
        "harness": copy.deepcopy(final_set.HARNESS),
        "sampling": copy.deepcopy(final_set.SAMPLING),
        "pass_k": final_set.PASS_K,
        "concurrency_per_worker": 1,
        "paired_execution": {
            "pairing_unit": "exact task_version_id and attempt seed",
            "same_tasks_and_attempt_seeds": True,
            "same_retry_and_validity_policy": True,
            "matched_arm_fields": list(MATCHED_ARM_FIELDS),
            "scientific_difference_allowed": ["weights_manifest_sha256"],
            "nonsemantic_identity_differences_allowed": [
                "checkpoint_identity",
                "checkpoint_acceptance_receipt_sha256",
                "export_receipt_sha256",
                "staging_receipt_sha256",
                "serving_registration_receipt_sha256",
                "served_model_id",
                "serving_route_profile_sha256",
            ],
            "hosted_and_dedicated_routes_may_not_be_pooled": True,
        },
        "outcome_authority": {
            "required_cyber_contract": {
                "evidence_schema": "1.0.0",
                "verifier_contract": "3.0.0",
                "submission_protocol": "2.0.0",
            },
            "provisioning_route_template": (
                "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
            ),
            "scoring_route_template": ("/v1/rollout-rewards/{task_key}/versions/{task_version_id}"),
            "valid_outcome_requires": [
                "exact frozen task, environment, starting-data and verifier bindings",
                "normal terminal OpenCode event stream without an unfinished final step",
                "authoritative version-bound grading",
                "complete private local result evidence",
                "challenge environment and all harness containers cleaned",
            ],
        },
        "metrics": {
            "primary_after_unsealing": {
                "name": "fleet_final_paired_pass_at_1_delta",
                "unit": "task_family",
                "attempt": 1,
                "seed": 42,
                "definition": (
                    "Candidate minus base full-task solve rate across the exact paired task "
                    "families using attempt 1 authoritative valid outcomes."
                ),
                "eligibility": "all paired attempt-1 outcomes must be valid",
            },
            "descriptive_after_unsealing": [
                "base_and_candidate_fleet_final_pass_at_4",
                "paired_task_level_outcomes",
                "authoritative_fractional_reward_delta",
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
                "Never score infrastructure-invalid, interrupted, output-limited or unknown "
                "attempts as model failures; the affected comparison is incomplete."
            ),
        },
        "sealed_use_policy": {
            "state": "sealed_during_training_and_checkpoint_selection",
            "training_data_eligible": False,
            "preference_or_reward_data_eligible": False,
            "prompt_retrieval_or_skill_eligible": False,
            "hyperparameter_selection_eligible": False,
            "checkpoint_selection_eligible": False,
            "failure_analysis_before_freeze": "forbidden",
            "unseal_only_after": (
                "The selected recipe and checkpoint identities are frozen in the exact "
                "checkpoint-freeze receipt bound by the child."
            ),
        },
        "external_benchmark_isolation": {
            "webexploitbench_state": "sealed_external_evaluation_only",
            "training_input_eligible": False,
            "reward_input_eligible": False,
            "checkpoint_or_hyperparameter_selection_eligible": False,
            "retry_or_stopping_input_eligible": False,
            "failure_analysis_before_checkpoint_freeze": "forbidden",
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
            "status": "nonlaunchable_binding_artifact_only",
            "dev_first": True,
            "prepare_must_reopen_all_bound_receipts": True,
            "fresh_exact_task_and_route_preflight_required": True,
            "paid_or_scored_work_authorized_by_this_file": False,
        },
    }


def build_parent(task_set: dict[str, Any], *, candidate_arm: str = CANDIDATE_ARM) -> dict[str, Any]:
    final_set.validate_final_task_set(task_set)
    if (
        task_set.get("sha256") != EXACT_TASK_SET_SHA256
        or task_set.get("final_test_lock_sha256") != EXACT_FINAL_LOCK_SHA256
    ):
        raise ValueError("Fleet final task-set identity changed")
    return _sealed(_parent_body(task_set, candidate_arm=candidate_arm))


def validate_parent(value: dict[str, Any], task_set: dict[str, Any]) -> None:
    _check_seal(value, PARENT_SCHEMA)
    candidate_arm = value.get("candidate_arm")
    if candidate_arm not in SUPPORTED_CANDIDATE_ARMS:
        raise ValueError("unsupported post-training candidate arm")
    expected = build_parent(task_set, candidate_arm=candidate_arm)
    if value != expected:
        raise ValueError("post-training final parent differs from its exact frozen contract")


def _validate_arm(value: object, label: str) -> dict[str, Any]:
    arm = _exact_keys(value, ARM_FIELDS, label)
    _immutable_id(arm.get("checkpoint_identity"), f"{label} checkpoint")
    _immutable_id(arm.get("served_model_id"), f"{label} served model")
    for field in ARM_FIELDS:
        if field.endswith("_sha256"):
            _digest(arm.get(field), f"{label} {field}")
    for field in ("serving_image", "agent_image", "proxy_image"):
        _image(arm.get(field), f"{label} {field}")
    return arm


def validate_bindings(
    value: dict[str, Any], parent: dict[str, Any], task_set: dict[str, Any]
) -> None:
    validate_parent(parent, task_set)
    _exact_keys(
        value,
        {"schema", "candidate_arm", "training_boundary", "arms", "result_isolation"},
        "binding",
    )
    candidate_arm = parent["candidate_arm"]
    if value.get("schema") != BINDINGS_SCHEMA or value.get("candidate_arm") != candidate_arm:
        raise ValueError("binding must select the exact post-training candidate arm")

    boundary = _exact_keys(
        value.get("training_boundary"), TRAINING_BOUNDARY_FIELDS, "training boundary"
    )
    if boundary.get("source_class") != "fleet_blackbox_tasks_only":
        raise ValueError("post-training source must remain Fleet blackbox only")
    _immutable_id(boundary.get("training_arm_id"), "training arm")
    for field in TRAINING_BOUNDARY_FIELDS:
        if field.endswith("_sha256"):
            _digest(boundary.get(field), field)
    if any(boundary.get(field) is not False for field in TRAINING_BOUNDARY_FIELDS[6:]):
        raise ValueError("external benchmark-derived training inputs are forbidden")

    arms = _exact_keys(value.get("arms"), {"base", "candidate"}, "arms")
    base = _validate_arm(arms.get("base"), "base arm")
    candidate = _validate_arm(arms.get("candidate"), "candidate arm")
    for field in MATCHED_ARM_FIELDS:
        if base[field] != candidate[field]:
            raise ValueError(f"matched arm field differs: {field}")
    if base["weights_manifest_sha256"] == candidate["weights_manifest_sha256"]:
        raise ValueError("post-training candidate must bind a distinct weight manifest")
    if base["checkpoint_identity"] == candidate["checkpoint_identity"]:
        raise ValueError("post-training candidate must bind a distinct checkpoint identity")
    if base["served_model_id"] == candidate["served_model_id"]:
        raise ValueError(
            "base and post-training candidate require distinct served model identities"
        )

    result = _exact_keys(value.get("result_isolation"), RESULT_ISOLATION_FIELDS, "result isolation")
    _immutable_id(result.get("campaign_id"), "campaign")
    prepared = _fresh_path(result.get("prepared_output_root"), "prepared output root")
    result_root = _fresh_path(result.get("result_root"), "result root")
    if result_root == prepared or not result_root.is_relative_to(prepared):
        raise ValueError("result root must be a fresh child of the prepared output root")
    if result.get("roots_absent") is not True:
        raise ValueError("Fleet result roots must be declared absent")


def _parent_reference(parent: dict[str, Any], parent_path: Path) -> dict[str, str]:
    contract = {
        key: parent[key]
        for key in (
            "task_set",
            "model",
            "harness",
            "sampling",
            "paired_execution",
            "outcome_authority",
            "metrics",
            "sealed_use_policy",
            "external_benchmark_isolation",
            "retry_policy",
        )
    }
    return {
        "schema": parent["schema"],
        "path": str(parent_path),
        "file_sha256": file_sha256(parent_path),
        "protocol_sha256": parent["sha256"],
        "evaluation_contract_sha256": digest_json(contract),
    }


def build_child(
    parent: dict[str, Any],
    parent_path: Path,
    bindings: dict[str, Any],
    task_set: dict[str, Any],
) -> dict[str, Any]:
    validate_bindings(bindings, parent, task_set)
    child = _build_child_unchecked(parent, parent_path, bindings)
    validate_child(child, parent, parent_path, task_set)
    return child


def validate_child(
    value: dict[str, Any],
    parent: dict[str, Any],
    parent_path: Path,
    task_set: dict[str, Any],
) -> None:
    _check_seal(value, CHILD_SCHEMA, "child_sha256")
    if (
        value.get("state") != "bound_nonlaunchable"
        or value.get("launchable") is not False
        or value.get("paid_or_scored_work_authorized") is not False
    ):
        raise ValueError("Fleet final child must remain non-launchable")
    bindings = {
        "schema": BINDINGS_SCHEMA,
        "candidate_arm": value.get("candidate_arm"),
        "training_boundary": value.get("training_boundary"),
        "arms": value.get("arms"),
        "result_isolation": value.get("result_isolation"),
    }
    validate_bindings(bindings, parent, task_set)
    rebuilt = _build_child_unchecked(parent, parent_path, bindings)
    if value != rebuilt:
        raise ValueError("Fleet final child differs from its exact parent and bindings")


def _build_child_unchecked(
    parent: dict[str, Any], parent_path: Path, bindings: dict[str, Any]
) -> dict[str, Any]:
    """Construct the canonical child after callers have validated all inputs."""
    value = {
        "schema": CHILD_SCHEMA,
        "state": "bound_nonlaunchable",
        "candidate_arm": parent["candidate_arm"],
        "launchable": False,
        "paid_or_scored_work_authorized": False,
        "parent": _parent_reference(parent, parent_path),
        "task_set": copy.deepcopy(parent["task_set"]),
        "training_boundary": copy.deepcopy(bindings["training_boundary"]),
        "arms": copy.deepcopy(bindings["arms"]),
        "frozen_treatment": {
            "harness_sha256": digest_json(parent["harness"]),
            "sampling_sha256": digest_json(parent["sampling"]),
            "pass_k": parent["pass_k"],
            "concurrency_per_worker": parent["concurrency_per_worker"],
            "paired_execution_sha256": digest_json(parent["paired_execution"]),
            "outcome_authority_sha256": digest_json(parent["outcome_authority"]),
            "retry_policy_sha256": digest_json(parent["retry_policy"]),
        },
        "result_isolation": copy.deepcopy(bindings["result_isolation"]),
        "sealed_use_policy": copy.deepcopy(parent["sealed_use_policy"]),
        "external_benchmark_isolation": copy.deepcopy(parent["external_benchmark_isolation"]),
        "launch_gate": {
            "status": "bound_but_nonlaunchable",
            "all_receipt_digests_present": True,
            "live_receipts_must_be_reopened_before_prepare": True,
            "fresh_exact_task_and_route_preflight_required": True,
            "separate_reviewed_execution_claim_required": True,
            "paid_or_scored_work_authorized": False,
        },
    }
    return _sealed(value, "child_sha256")


def materialize_child(parent_path: Path, bindings_path: Path, output: Path) -> dict[str, Any]:
    parent_path = parent_path.absolute()
    parent = _read(parent_path)
    bindings = _read(bindings_path.absolute())
    task_set = load_exact_task_set()
    child = build_child(parent, parent_path, bindings, task_set)
    output = output.absolute()
    reserved = [
        Path(bindings["result_isolation"][field])
        for field in ("prepared_output_root", "result_root")
    ]
    if any(
        output == path or output.is_relative_to(path) or path.is_relative_to(output)
        for path in reserved
    ):
        raise ValueError("child output must remain disjoint from result roots")
    _write_once(output, child)
    return child


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parent_parser = subparsers.add_parser("validate-parent")
    validate_parent_parser.add_argument("--parent", type=Path, required=True)
    materialize = subparsers.add_parser("materialize-child")
    materialize.add_argument("--parent", type=Path, required=True)
    materialize.add_argument("--bindings", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    validate_child_parser = subparsers.add_parser("validate-child")
    validate_child_parser.add_argument("--parent", type=Path, required=True)
    validate_child_parser.add_argument("--child", type=Path, required=True)
    args = parser.parse_args(argv)

    task_set = load_exact_task_set()
    parent_path = args.parent.absolute()
    parent = _read(parent_path)
    if args.command == "validate-parent":
        validate_parent(parent, task_set)
        print(canonical_json({"launchable": False, "sha256": parent["sha256"]}))
        return 0
    if args.command == "materialize-child":
        child = materialize_child(parent_path, args.bindings, args.output)
        print(canonical_json({"child_sha256": child["child_sha256"], "launchable": False}))
        return 0
    child = _read(args.child.absolute())
    validate_child(child, parent, parent_path, task_set)
    print(canonical_json({"child_sha256": child["child_sha256"], "launchable": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
