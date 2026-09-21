"""Render the first large Qwen self visible-reasoning collection plan.

This module is deliberately source-only.  It validates existing public task,
split, model, harness, and runtime metadata and writes an independently
reviewable plan.  It never contacts Fleet, reads a trace, creates a session, or
authorizes collection.  The later collection authority must still issue the
immutable source authorization, template round-trip proof, collection packet,
and verified-success evidence required by ``fleet_visible_reasoning_corpus``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training import collection_campaign, task_family_split
from training import fleet_visible_reasoning_corpus as corpus

SPEC_SCHEMA = "cyber_qwen_opencode_visible_reasoning_campaign_spec_v1"
PLAN_SCHEMA = "cyber_qwen_opencode_visible_reasoning_campaign_plan_v1"
WAVE_SCHEMA = "cyber_qwen_opencode_visible_reasoning_wave_plan_v1"
RECEIPT_SCHEMA = "cyber_qwen_opencode_visible_reasoning_campaign_review_v1"

STATUS = "source_only_review_required"
TRAIN_TASKS = 50
DEV_TASKS = 17
FINAL_TASKS = 8
ATTEMPTS_PER_TASK = 400
ATTEMPTS_PER_TASK_PER_WAVE = 10
WAVES = 40
CELLS_PER_WAVE = 500
PLANNED_CELLS = 20_000
BASE_SEED = 43
MINIMUM_SUCCESSFUL_FAMILIES = 20


def _digest(value: object) -> str:
    return collection_campaign.canonical_digest(value)


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sealed(value: dict[str, Any], schema: str, label: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != _digest(
        {name: item for name, item in value.items() if name != "sha256"}
    ):
        raise ValueError(f"invalid sealed {label}")


def _reference(root: Path, value: object, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"path", "file_sha256", "logical_sha256"}:
        raise ValueError(f"{label} needs an exact file and logical reference")
    path = root / value["path"]
    if path.is_symlink() or not path.is_file() or _file_sha256(path) != value["file_sha256"]:
        raise ValueError(f"{label} file digest mismatch")
    loaded = _load(path)
    if loaded.get("sha256", _digest(loaded)) != value["logical_sha256"]:
        raise ValueError(f"{label} logical digest mismatch")
    return path, loaded


def _validate_spec(spec: dict[str, Any]) -> None:
    _sealed(spec, SPEC_SCHEMA, "visible-reasoning campaign spec")
    if set(spec) != {
        "schema",
        "campaign_name",
        "status",
        "purpose",
        "inputs",
        "source",
        "selection",
        "collection",
        "admission",
        "execution",
        "review",
        "sha256",
    }:
        raise ValueError("visible-reasoning campaign spec shape drift")
    if spec["status"] != STATUS:
        raise ValueError("visible-reasoning campaign must remain source-only pending review")
    if set(spec["inputs"]) != {
        "model_lock",
        "source_profile",
        "inventory",
        "family_split",
        "role_anchor",
        "protected_family_lock",
        "runtime_bindings",
        "task_selection",
        "historical_token_basis",
    }:
        raise ValueError("visible-reasoning campaign input set drift")
    if spec["selection"] != {
        "allowed_role": "train",
        "train_task_versions": TRAIN_TASKS,
        "dev_task_versions": DEV_TASKS,
        "final_test_task_versions": FINAL_TASKS,
        "maximum_task_versions_per_family": 1,
        "collect_every_train_family_each_wave": True,
        "heldout_collection_cells": 0,
        "external_benchmark_collection_cells": 0,
    }:
        raise ValueError("visible-reasoning family selection drift")
    if spec["collection"] != {
        "attempts_per_task_version": ATTEMPTS_PER_TASK,
        "attempts_per_task_version_per_wave": ATTEMPTS_PER_TASK_PER_WAVE,
        "waves": WAVES,
        "cells_per_full_wave": CELLS_PER_WAVE,
        "maximum_planned_cells": PLANNED_CELLS,
        "maximum_rollouts_per_day": CELLS_PER_WAVE,
        "finish_active_wave_before_target_stop": True,
        "minimum_unique_supervised_tokens": corpus.MINIMUM_SUPERVISED_TOKENS,
        "minimum_successful_families": MINIMUM_SUCCESSFUL_FAMILIES,
        "maximum_family_target_token_fraction": corpus.MAXIMUM_FAMILY_TOKEN_FRACTION,
    }:
        raise ValueError("visible-reasoning collection budget drift")
    if spec["admission"] != {
        "eligible_terminal_outcome": "verified_success",
        "reasoning_visibility": "student_visible",
        "paired_target_kinds": ["student_visible_reasoning", "visible_action"],
        "deduplication_order": [
            "source_session_identity",
            "normalized_trajectory_digest",
            "packed_window_payload_digest",
        ],
        "reject": [
            "private_or_unknown_reasoning",
            "teacher_or_provider_private_reasoning",
            "inferred_or_reconstructed_reasoning",
            "opaque_compaction",
            "unknown_serialization",
            "heldout_family",
            "missing_or_mismatched_verifier_execution",
        ],
    }:
        raise ValueError("visible-reasoning admission policy drift")
    if spec["execution"] != {
        "source_only": True,
        "external_submission": False,
        "fleet_api_calls": 0,
        "model_calls": 0,
        "trace_or_score_reads": 0,
        "create_once_cell_intents": True,
        "dedicated_ledger_required": True,
        "ambiguous_cell_replay_allowed": False,
        "same_cell_retry_allowed": False,
        "required_job_root_annotation": {"fleet.ai/failure-alerts": "off"},
    }:
        raise ValueError("visible-reasoning execution boundary drift")
    missing = spec["review"].get("required_before_collection")
    if (
        spec["review"].get("independent_review_required") is not True
        or spec["review"].get("review_completed") is not False
        or not isinstance(missing, list)
        or set(missing)
        != {
            "immutable_source_authorization_registry_artifact",
            "exact_qwen_template_roundtrip_receipt",
            "reasoning_preservation_request_and_response_receipt",
            "exact_compaction_continuation_recording_receipt",
            "collection_packet_and_operation_authorization",
            "independent_source_review",
        }
    ):
        raise ValueError("visible-reasoning review gate drift")


def _model_and_harness(
    spec: dict[str, Any], model_lock: dict[str, Any], profile: dict[str, Any]
) -> None:
    source = spec["source"]
    if source.get("kind") != "qwen_self" or source.get("model") != {
        "repository": corpus.QWEN_REPOSITORY,
        "revision": model_lock.get("revision"),
        "session_model": "qwen/qwen3.8-27b",
    }:
        raise ValueError("visible-reasoning source is not the exact Qwen self model")
    if model_lock.get("repo") != corpus.QWEN_REPOSITORY:
        raise ValueError("model lock is not Qwen3.8-27B")
    tokenizer = model_lock.get("tokenizer", {})
    if source.get("tokenizer") != {
        "manifest_sha256": tokenizer.get("manifest_sha256"),
        "backend_sha256": (
            "sha256:ffb7a28b27dabcc333662fd3e0b0005d9e79a1c22e31453ab5a3017fbd5f25c0"
        ),
        "chat_template_sha256": (
            "sha256:c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
        ),
    }:
        raise ValueError("visible-reasoning tokenizer identity drift")
    model = profile.get("models", {}).get("qwen3.8-27b-base")
    route = profile.get("routes", {}).get("base")
    harness = profile.get("harness")
    if (
        model != source["model"]
        or not isinstance(route, dict)
        or route.get("model") != ("qwen3.8-27b-base")
    ):
        raise ValueError("source profile changes the exact Qwen route")
    expected_harness = {
        "harness": corpus.OPENCODE_HARNESS,
        "harness_version": corpus.OPENCODE_VERSION,
        "release_asset_sha256": harness.get("release_asset_sha256"),
        "tool_catalog_sha256": harness.get("tool_catalog_sha256"),
        "tools": ["bash", "submit_report"],
        "context_management": corpus.ONLINE_COMPACTION,
        "context_window_tokens": 262_144,
        "context_headroom_tokens": 20_000,
        "max_output_tokens": 32_768,
        "max_model_requests": 600,
        "timeout_seconds": 28_800,
    }
    if source.get("opencode") != expected_harness:
        raise ValueError("visible-reasoning OpenCode treatment drift")
    if source.get("thinking") != {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_visibility": "student_visible",
    }:
        raise ValueError("visible reasoning must be explicitly generated and preserved")
    if source.get("sampling") != {
        "temperature": 0.6,
        "top_p": 0.95,
        "base_seed": BASE_SEED,
        "seed_rule": "base_seed_plus_attempt_minus_one",
    }:
        raise ValueError("visible-reasoning sampling treatment drift")


def _task_boundary(inputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = inputs["inventory"]
    split = inputs["family_split"]
    anchor = inputs["role_anchor"]
    lock = inputs["protected_family_lock"]
    runtime = inputs["runtime_bindings"]
    selection = inputs["task_selection"]
    inventory_rows = collection_campaign._inventory_rows(inventory)  # noqa: SLF001
    if len(inventory_rows) != TRAIN_TASKS + DEV_TASKS + FINAL_TASKS:
        raise ValueError("visible-reasoning inventory count drift")
    task_family_split.require_trusted_fleet_collection_root_anchor(anchor)
    task_family_split.validate(split, inventory_rows, role_anchor=anchor)
    counts = {name: split["counts"][name]["task_versions"] for name in split["counts"]}
    if counts != {"train": TRAIN_TASKS, "dev": DEV_TASKS, "final_test": FINAL_TASKS}:
        raise ValueError("visible-reasoning split count drift")
    heldout = {row["group_id"] for row in split["tasks"] if row["split"] != "train"}
    if (
        lock.get("source_split_sha256") != split.get("sha256")
        or set(lock.get("heldout_group_ids", [])) != heldout
        or len(heldout) != DEV_TASKS + FINAL_TASKS
    ):
        raise ValueError("visible-reasoning protected-family lock drift")
    bindings = collection_campaign._runtime_bindings(  # noqa: SLF001
        runtime,
        inventory,
        inventory_rows,
    )
    role_by_identity = {(row["task_key"], row["task_version_id"]): row for row in split["tasks"]}
    rows = selection.get("tasks")
    if not isinstance(rows, list) or len(rows) != TRAIN_TASKS:
        raise ValueError("visible-reasoning task selection must cover every train family")
    groups: set[str] = set()
    identities: set[tuple[str, str]] = set()
    selected: list[dict[str, Any]] = []
    for row in rows:
        identity = (row.get("task_key"), row.get("task_version_id"))
        role = role_by_identity.get(identity)
        if (
            identity in identities
            or identity not in bindings
            or role is None
            or role.get("split") != "train"
            or role.get("group_id") in heldout
            or role.get("group_id") in groups
        ):
            raise ValueError("visible-reasoning selection leaks or duplicates a task family")
        identities.add(identity)
        groups.add(role["group_id"])
        selected.append({**row, "group_id": role["group_id"]})
    return sorted(selected, key=lambda row: (row["task_key"], row["task_version_id"]))


def _cell_intent(campaign_name: str, task: dict[str, Any], attempt: int) -> dict[str, Any]:
    payload = {
        "campaign_name": campaign_name,
        "task_key": task["task_key"],
        "task_version_id": task["task_version_id"],
        "group_id": task["group_id"],
        "attempt": attempt,
        "seed": BASE_SEED + attempt - 1,
    }
    digest = _digest(payload)
    return {**payload, "cell_id": "qvrc-" + digest.removeprefix("sha256:")[:24]}


def _waves(campaign_name: str, tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    waves: list[dict[str, Any]] = []
    universe: list[dict[str, Any]] = []
    for wave_index in range(WAVES):
        first = wave_index * ATTEMPTS_PER_TASK_PER_WAVE + 1
        last = first + ATTEMPTS_PER_TASK_PER_WAVE - 1
        cells = [
            _cell_intent(campaign_name, task, attempt)
            for task in tasks
            for attempt in range(first, last + 1)
        ]
        universe.extend(cells)
        waves.append(
            {
                "wave": wave_index + 1,
                "attempt_first": first,
                "attempt_last": last,
                "seed_first": BASE_SEED + first - 1,
                "seed_last": BASE_SEED + last - 1,
                "task_versions": len(tasks),
                "planned_cells": len(cells),
                "cell_intents_sha256": _digest(cells),
            }
        )
    if len(universe) != PLANNED_CELLS or len({cell["cell_id"] for cell in universe}) != (
        PLANNED_CELLS
    ):
        raise ValueError("visible-reasoning cell identity universe is not exactly unique")
    return waves, _digest(universe)


def render(spec: dict[str, Any], *, root: Path) -> dict[str, dict[str, Any]]:
    """Render a deterministic review packet without external calls or trace access."""
    _validate_spec(spec)
    loaded: dict[str, dict[str, Any]] = {}
    for name, reference in spec["inputs"].items():
        _path, loaded[name] = _reference(root, reference, name.replace("_", " "))
    _model_and_harness(spec, loaded["model_lock"], loaded["source_profile"])
    tasks = _task_boundary(loaded)
    historical = loaded["historical_token_basis"]
    if (
        historical.get("files", {}).get("train", {}).get("source_sessions") != 2_886
        or historical.get("files", {}).get("train", {}).get("supervised_tokens") != 57_384_881
    ):
        raise ValueError("historical aggregate token-yield basis drift")
    mean_tokens = (
        historical["files"]["train"]["supervised_tokens"]
        / historical["files"]["train"]["source_sessions"]
    )
    successes_needed = int(corpus.MINIMUM_SUPERVISED_TOKENS // mean_tokens) + 1
    waves, identity_universe_sha256 = _waves(spec["campaign_name"], tasks)
    plan = collection_campaign.sealed(
        {
            "schema": PLAN_SCHEMA,
            "campaign_name": spec["campaign_name"],
            "status": STATUS,
            "source_spec_sha256": spec["sha256"],
            "task_boundary": {
                "inventory_sha256": loaded["inventory"]["sha256"],
                "family_split_sha256": loaded["family_split"]["sha256"],
                "root_role_anchor_id": task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID,
                "family_role_anchor_sha256": loaded["role_anchor"]["sha256"],
                "protected_family_lock_sha256": loaded["protected_family_lock"]["sha256"],
                "runtime_bindings_sha256": loaded["runtime_bindings"]["sha256"],
                "task_selection_sha256": loaded["task_selection"]["sha256"],
                "train_task_versions": TRAIN_TASKS,
                "heldout_task_versions": DEV_TASKS + FINAL_TASKS,
                "heldout_collection_cells": 0,
            },
            "source_treatment": spec["source"],
            "collection": spec["collection"],
            "admission": spec["admission"],
            "planning_basis": {
                "historical_aggregate_manifest_sha256": historical["sha256"],
                "historical_verified_success_sessions": 2_886,
                "historical_unique_supervised_tokens": 57_384_881,
                "historical_mean_supervised_tokens_per_success": round(mean_tokens, 3),
                "reference_successes_needed_for_20m": successes_needed,
                "minimum_reference_success_rate_over_planned_cells": round(
                    successes_needed / PLANNED_CELLS, 6
                ),
                "not_a_success_or_token_yield_guarantee": True,
            },
            "identity": {
                "algorithm": "sha256(canonical_json(campaign,task,version,family,attempt,seed))",
                "cell_id_prefix": "qvrc-",
                "cell_identity_universe_sha256": identity_universe_sha256,
                "planned_unique_cell_ids": PLANNED_CELLS,
                "ambiguous_cell_replay_allowed": False,
            },
            "future_artifact_schemas": {
                "source_profile": corpus.SOURCE_PROFILE_SCHEMA,
                "source_authorization": corpus.SOURCE_AUTHORIZATION_SCHEMA,
                "collection_packet": corpus.PACKET_SCHEMA,
                "success_evidence": corpus.SUCCESS_EVIDENCE_SCHEMA,
                "private_selection": corpus.SELECTION_SCHEMA,
                "record": corpus.RECORD_SCHEMA,
                "source_census": "cyber_qwen_opencode_student_visible_reasoning_census_v1",
                "corpus": corpus.CORPUS_SCHEMA,
                "training_selection": (
                    "cyber_qwen_opencode_visible_reasoning_training_selection_v1"
                ),
            },
            "compaction": {
                "online": corpus.ONLINE_COMPACTION,
                "accepted_offline_kind": corpus.EXACT_COMPACTION,
                "actual_post_summary_prompt_required": True,
                "compaction_summary_loss_mask": 0,
                "opaque_or_unreconstructable_continuation": "reject_target_and_later_continuation",
            },
            "matched_ablation": {
                "derive_action_only_from_same_selected_turns": True,
                "visible_action_spans_identical": True,
                "reasoning_spans_masked_only_in_action_arm": True,
                "mix_corpora": False,
            },
            "review": spec["review"],
            "submitted": False,
            "fleet_api_calls": 0,
            "model_calls": 0,
            "trace_or_score_reads": 0,
        }
    )
    wave_plan = collection_campaign.sealed(
        {
            "schema": WAVE_SCHEMA,
            "campaign_plan_sha256": plan["sha256"],
            "waves": waves,
            "scheduling": {
                "maximum_rollouts_per_day": CELLS_PER_WAVE,
                "one_full_wave_per_day": True,
                "finish_active_wave_before_target_stop": True,
                "start_next_wave_only_if_aggregate_target_not_reached": True,
            },
            "submitted": False,
        }
    )
    outputs = {"campaign-plan.json": plan, "wave-plan.json": wave_plan}
    receipt = collection_campaign.sealed(
        {
            "schema": RECEIPT_SCHEMA,
            "status": STATUS,
            "source_spec_sha256": spec["sha256"],
            "input_file_sha256": {
                name: spec["inputs"][name]["file_sha256"] for name in sorted(spec["inputs"])
            },
            "output_sha256": {name: value["sha256"] for name, value in sorted(outputs.items())},
            "train_task_versions": TRAIN_TASKS,
            "heldout_task_versions": DEV_TASKS + FINAL_TASKS,
            "planned_cells": PLANNED_CELLS,
            "minimum_unique_supervised_tokens": corpus.MINIMUM_SUPERVISED_TOKENS,
            "private_or_hidden_reasoning_allowed": False,
            "external_submission": False,
            "review_completed": False,
            "required_before_collection": spec["review"]["required_before_collection"],
        }
    )
    outputs["review-receipt.json"] = receipt
    return outputs


def write_once(destination: Path, rendered: dict[str, dict[str, Any]]) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("visible-reasoning campaign destination already exists")
    destination.mkdir(parents=True, mode=0o700)
    for name, value in sorted(rendered.items()):
        path = destination / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")


def check(destination: Path, rendered: dict[str, dict[str, Any]]) -> None:
    if set(path.name for path in destination.iterdir()) != set(rendered):
        raise ValueError("visible-reasoning campaign file set drift")
    for name, expected in rendered.items():
        if _load(destination / name) != expected:
            raise ValueError(f"visible-reasoning campaign drift: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        rendered = render(_load(args.spec), root=args.root)
        if args.write:
            write_once(args.output, rendered)
        else:
            check(args.output, rendered)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {type(error).__name__}: {error}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": STATUS,
                "planned_cells": PLANNED_CELLS,
                "submitted": False,
                "files": sorted(rendered),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
