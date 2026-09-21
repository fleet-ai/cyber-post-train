"""Admit sealed Fleet collection metadata to the action-SFT input boundary.

This module is intentionally a *metadata-only* bridge.  It never opens a
transcript, prompt, tool result, reasoning field, or packed training row.  A
later private corpus step uses the returned selection to bind the matching
normalized records and then delegates message validation/windowing to the
existing dense-SFT implementation.

The boundary is deliberately stricter than ``training_data_eligible`` alone:
an admitted session must be an exact planned campaign cell, a verified success,
fully ingested, action-only, uncompacted, assigned to the immutable train
family split, unique, and within an explicit per-task-version cap.
"""

from __future__ import annotations

import collections
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest

from .io import atomic_write_json, file_sha256, iter_jsonl
from .sft import _known
from .task_family_split import is_supported_schema, requires_role_anchor
from .task_family_split import validate as validate_split

REQUEST_SCHEMA = "cyber_fleet_collection_admission_request_v1"
ATTEMPT_SCHEMA = "cyber_fleet_collection_attempt_metadata_v1"
SELECTION_SCHEMA = "cyber_fleet_collection_selection_v1"
RECEIPT_SCHEMA = "cyber_fleet_collection_admission_receipt_v1"
PROTECTED_FAMILY_LOCK_SCHEMA = "cyber_protected_task_family_lock_v1"
CAMPAIGN_SCHEMA = "cyber_fleet_eval_v1"
HANDOFF_KIND = "metadata_evidence_handoff_only"

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_CAMPAIGN_SHA256 = re.compile(r"[0-9a-f]{64}")
_REJECTION_REASONS = (
    "ambiguous_campaign_cell",
    "duplicate_session",
    "exact_trajectory_duplicate",
    "heldout_or_nontrain",
    "incomplete_ingestion",
    "invalid_authoritative_success",
    "missing_planned_attempt",
    "not_visible_actions_only",
    "opaque_compaction",
    "per_task_session_cap",
    "private_or_unknown_reasoning",
    "unapproved_compaction",
    "unplanned_campaign_cell",
    "wrong_campaign_binding",
    "wrong_model_binding",
    "wrong_template_binding",
    "wrong_tool_binding",
)
_PUBLIC_FORBIDDEN_FIELD_NAMES = {
    "session_id",
    "task_key",
    "task_version_id",
    "trace",
    "transcript",
    "prompt",
    "message",
    "reasoning",
    "thinking",
    "content",
    "flag",
    "credential",
}


def _sealed(value: Mapping[str, Any], *, field: str = "sha256") -> bool:
    return value.get(field) == "sha256:" + digest(
        {key: item for key, item in value.items() if key != field}
    )


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"exact {label} SHA-256 is required")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"nonempty {label} is required")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _input_path(root: Path, reference: object, label: str) -> tuple[Path, str]:
    value = _mapping(reference, label)
    _known(value, {"path", "sha256"}, label)
    path = _path(root, _string(value.get("path"), f"{label} path"))
    expected = _sha(value.get("sha256"), f"{label} file")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    if file_sha256(path) != expected:
        raise ValueError(f"{label} file digest mismatch")
    return path, expected


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    return _mapping(value, label)


def _campaign_sha(plan: Mapping[str, Any]) -> str:
    value = plan.get("sha256")
    if not isinstance(value, str) or _CAMPAIGN_SHA256.fullmatch(value) is None:
        raise ValueError("campaign plan is missing its exact digest")
    if value != digest({key: item for key, item in plan.items() if key != "sha256"}):
        raise ValueError("campaign plan digest mismatch")
    return "sha256:" + value


def _cell_sha(
    *,
    campaign_sha256: str,
    task_key: str,
    task_version_id: str,
    model_alias: str,
    model_revision: str,
    attempt: int,
) -> str:
    return "sha256:" + digest(
        {
            "campaign_plan_sha256": campaign_sha256,
            "task_key": task_key,
            "task_version_id": task_version_id,
            "model_alias": model_alias,
            "model_revision": model_revision,
            "attempt": attempt,
        }
    )


def _campaign_cells(
    plan: dict[str, Any], model_alias: str
) -> tuple[
    dict[tuple[str, str, str, int], dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Return exact cells for one model without trusting a mutable job roster."""
    if plan.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported Fleet campaign plan schema")
    campaign_sha = _campaign_sha(plan)
    if plan.get("training_data_eligible") is not True:
        raise ValueError("campaign is not explicitly eligible for training-data collection")
    models = _mapping(plan.get("models"), "campaign models")
    model = _mapping(models.get(model_alias), "selected campaign model")
    _known(model, {"repository", "revision", "session_model"}, "selected campaign model")
    for field in ("repository", "revision", "session_model"):
        _string(model.get(field), f"selected campaign model {field}")
    treatment = _mapping(plan.get("treatment"), "campaign treatment")
    tools = treatment.get("tools")
    if tools != ["bash", "submit_report"]:
        raise ValueError("campaign must bind exactly the reviewed action tool surface")
    _sha(treatment.get("tool_catalog_sha256"), "campaign tool catalog")
    task_rows = plan.get("tasks")
    if not isinstance(task_rows, list) or not task_rows:
        raise ValueError("campaign plan has no task bindings")
    tasks: dict[str, dict[str, str]] = {}
    for raw in task_rows:
        row = _mapping(raw, "campaign task")
        key = _string(row.get("task_key"), "campaign task key")
        version = _string(row.get("task_version_id"), "campaign task version")
        if version in tasks or set(row) - {
            "task_key",
            "task_version_id",
            "env_key",
            "env_version",
            "environment_version_id",
            "data_key",
            "data_version",
        }:
            raise ValueError("campaign task bindings are malformed or ambiguous")
        tasks[version] = {"task_key": key, "task_version_id": version}
    pass_k = plan.get("pass_k")
    if type(pass_k) is not int or pass_k < 1:
        raise ValueError("campaign pass_k is invalid")
    routes = _mapping(plan.get("routes"), "campaign routes")
    selected_versions: set[str] = set()
    for raw in routes.values():
        route = _mapping(raw, "campaign route")
        if route.get("model") != model_alias:
            continue
        versions = route.get("task_versions")
        if not isinstance(versions, list) or not versions:
            raise ValueError("selected campaign route lacks task versions")
        for version in versions:
            if not isinstance(version, str) or version not in tasks or version in selected_versions:
                raise ValueError("selected campaign route task binding is ambiguous")
            selected_versions.add(version)
    if not selected_versions:
        raise ValueError("selected model has no campaign task assignments")
    expected: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for version in sorted(selected_versions):
        for attempt in range(1, pass_k + 1):
            task = tasks[version]
            key = (task["task_key"], version, model_alias, attempt)
            expected[key] = {
                **task,
                "model_alias": model_alias,
                "attempt": attempt,
                "campaign_plan_sha256": campaign_sha,
                "cell_sha256": _cell_sha(
                    campaign_sha256=campaign_sha,
                    task_key=task["task_key"],
                    task_version_id=version,
                    model_alias=model_alias,
                    model_revision=model["revision"],
                    attempt=attempt,
                ),
            }
    return expected, model, treatment


def _split_assignments(
    split: dict[str, Any], inventory: dict[str, Any], *, role_anchor: dict[str, Any] | None
) -> dict[tuple[str, str], dict[str, str]]:
    if not is_supported_schema(split):
        raise ValueError("collection admission requires a supported family split")
    rows = inventory.get("task_versions")
    if not isinstance(rows, list):
        raise ValueError("sanitized split inventory is missing task_versions")
    if not _sealed(inventory):
        raise ValueError("sanitized split inventory digest mismatch")
    if split.get("inventory_sha256") != inventory["sha256"]:
        raise ValueError("family split is not bound to the current sanitized task catalog")
    validate_split(split, rows, role_anchor=role_anchor)
    if set(split.get("counts", {})) != {"train", "dev", "final_test"}:
        raise ValueError("collection admission requires immutable train/dev/final_test roles")
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in split["tasks"]:
        identity = (row["task_key"], row["task_version_id"])
        if identity in result or row["split"] not in {"train", "dev", "final_test"}:
            raise ValueError("family split task assignments are ambiguous")
        result[identity] = {"group_id": row["group_id"], "split": row["split"]}
    return result


def _protected_groups(value: dict[str, Any]) -> set[str]:
    if value.get("schema") != PROTECTED_FAMILY_LOCK_SCHEMA or not _sealed(value):
        raise ValueError("invalid protected-family lock")
    _known(
        value,
        {"schema", "source_split_sha256", "heldout_group_ids", "sha256"},
        "protected-family lock",
    )
    _sha(value.get("source_split_sha256"), "protected-family source split")
    groups = value.get("heldout_group_ids")
    if not isinstance(groups, list) or not groups:
        raise ValueError("protected-family lock must name at least one held-out family")
    if any(not isinstance(group, str) or _SHA256.fullmatch(group) is None for group in groups):
        raise ValueError("protected-family lock has an invalid group digest")
    if groups != sorted(set(groups)):
        raise ValueError("protected-family lock groups must be sorted and unique")
    return set(groups)


def _attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one metadata record and reject any hidden trace payload field."""
    row = dict(value)
    _known(
        row,
        {
            "schema",
            "session_id",
            "campaign_plan_sha256",
            "cell_sha256",
            "task_key",
            "task_version_id",
            "model_alias",
            "attempt",
            "model",
            "harness",
            "template_sha256",
            "outcome",
            "ingestion",
            "content_policy",
            "sha256",
        },
        "collection attempt metadata",
    )
    if row.get("schema") != ATTEMPT_SCHEMA or not _sealed(row):
        raise ValueError("collection attempt metadata digest/schema mismatch")
    for field in ("session_id", "task_key", "task_version_id", "model_alias"):
        _string(row.get(field), f"attempt {field}")
    _sha(row.get("campaign_plan_sha256"), "attempt campaign plan")
    _sha(row.get("cell_sha256"), "attempt cell")
    if type(row.get("attempt")) is not int or row["attempt"] < 1:
        raise ValueError("attempt number must be a positive integer")
    model = _mapping(row.get("model"), "attempt model")
    _known(model, {"repository", "revision", "session_model"}, "attempt model")
    for field in ("repository", "revision", "session_model"):
        _string(model.get(field), f"attempt model {field}")
    harness = _mapping(row.get("harness"), "attempt harness")
    _known(harness, {"treatment_sha256", "tool_catalog_sha256"}, "attempt harness")
    _sha(harness.get("treatment_sha256"), "attempt harness treatment")
    _sha(harness.get("tool_catalog_sha256"), "attempt tool catalog")
    _sha(row.get("template_sha256"), "attempt template")
    outcome = _mapping(row.get("outcome"), "attempt outcome")
    _known(
        outcome,
        {"status", "verifier_process_success", "score_at_least_one"},
        "attempt outcome",
    )
    if "score_at_least_one" not in outcome:
        raise ValueError("attempt outcome is missing authoritative grading")
    if not isinstance(outcome.get("status"), str) or any(
        type(outcome.get(field)) is not bool
        for field in ("verifier_process_success", "score_at_least_one")
    ):
        raise ValueError("attempt outcome is malformed")
    ingestion = _mapping(row.get("ingestion"), "attempt ingestion")
    _known(
        ingestion,
        {
            "status",
            "normalized_record_sha256",
            "normalized_trajectory_sha256",
            "transcript_sha256",
        },
        "attempt ingestion",
    )
    if ingestion.get("status") not in {"complete", "incomplete"}:
        raise ValueError("attempt ingestion status is malformed")
    for field in (
        "normalized_record_sha256",
        "normalized_trajectory_sha256",
        "transcript_sha256",
    ):
        _sha(ingestion.get(field), f"attempt ingestion {field}")
    policy = _mapping(row.get("content_policy"), "attempt content policy")
    _known(
        policy,
        {"target_mode", "reasoning_visibility", "compaction"},
        "attempt content policy",
    )
    for field in ("target_mode", "reasoning_visibility", "compaction"):
        _string(policy.get(field), f"attempt content policy {field}")
    return row


def _reason_for_candidate(
    row: dict[str, Any],
    *,
    model: dict[str, Any],
    treatment: dict[str, Any],
    campaign_sha256: str,
    template_sha256: str,
) -> str | None:
    if row["campaign_plan_sha256"] != campaign_sha256:
        return "wrong_campaign_binding"
    expected_cell = _cell_sha(
        campaign_sha256=campaign_sha256,
        task_key=row["task_key"],
        task_version_id=row["task_version_id"],
        model_alias=row["model_alias"],
        model_revision=model["revision"],
        attempt=row["attempt"],
    )
    if row["cell_sha256"] != expected_cell:
        return "wrong_campaign_binding"
    if row["model"] != model:
        return "wrong_model_binding"
    if row["harness"]["treatment_sha256"] != "sha256:" + digest(treatment):
        return "wrong_campaign_binding"
    if row["harness"]["tool_catalog_sha256"] != treatment["tool_catalog_sha256"]:
        return "wrong_tool_binding"
    if row["template_sha256"] != template_sha256:
        return "wrong_template_binding"
    outcome = row["outcome"]
    if outcome != {
        "status": "completed",
        "verifier_process_success": True,
        "score_at_least_one": True,
    }:
        return "invalid_authoritative_success"
    if row["ingestion"]["status"] != "complete":
        return "incomplete_ingestion"
    policy = row["content_policy"]
    if policy["target_mode"] != "visible_actions_only":
        return "not_visible_actions_only"
    if policy["reasoning_visibility"] != "absent":
        return "private_or_unknown_reasoning"
    if policy["compaction"] == "opaque":
        return "opaque_compaction"
    if policy["compaction"] != "none":
        return "unapproved_compaction"
    return None


def _selection_rank(row: Mapping[str, Any]) -> str:
    """Stable private ordering for dedupe/caps; never use outcome quality."""
    return digest(
        {
            "cell_sha256": row["cell_sha256"],
            "session_id": row["session_id"],
            "normalized_trajectory_sha256": row["ingestion"]["normalized_trajectory_sha256"],
        }
    )


def _assert_public(value: object) -> None:
    """Keep the receipt aggregate-only even when selection data is private."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _PUBLIC_FORBIDDEN_FIELD_NAMES:
                raise ValueError("public admission receipt would expose private source metadata")
            _assert_public(item)
    elif isinstance(value, list):
        for item in value:
            _assert_public(item)


def _write_output(output: Path, selection: dict[str, Any], receipt: dict[str, Any]) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("collection admission destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(output, 0o700)
    except FileExistsError as error:
        raise FileExistsError("collection admission destination already exists") from error
    atomic_write_json(output / "selection.private.json", selection, private=True)
    atomic_write_json(output / "ADMISSION.json", receipt, private=True)


def build(config: dict[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Make one create-once, aggregate-safe selection for a generic campaign.

    The command is CPU-only and offline.  It consumes only sealed metadata and
    does not create a rollout, train, score, or read raw session content.
    """
    _known(
        config,
        {
            "schema",
            "campaign",
            "inventory",
            "family_split",
            "role_anchor",
            "protected_family_lock",
            "attempts",
            "source",
            "max_sessions_per_task_version",
            "output",
        },
        "collection admission request",
    )
    if config.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported collection admission request")
    source = _mapping(config.get("source"), "collection source")
    _known(source, {"kind", "model_alias", "template_sha256"}, "collection source")
    if source.get("kind") not in {"teacher", "self"}:
        raise ValueError("collection source kind must be teacher or self")
    model_alias = _string(source.get("model_alias"), "collection source model alias")
    template_sha256 = _sha(source.get("template_sha256"), "collection source template")
    cap = config.get("max_sessions_per_task_version")
    if type(cap) is not int or not 1 <= cap <= 10_000:
        raise ValueError("max_sessions_per_task_version must be an explicit positive bound")
    output = _path(relative_to, _string(config.get("output"), "collection admission output"))
    if output.exists() or output.is_symlink():
        raise FileExistsError("collection admission destination already exists")

    campaign_path, campaign_file_sha = _input_path(relative_to, config.get("campaign"), "campaign")
    inventory_path, inventory_file_sha = _input_path(
        relative_to, config.get("inventory"), "inventory"
    )
    split_path, split_file_sha = _input_path(
        relative_to, config.get("family_split"), "family split"
    )
    role_anchor_path = None
    role_anchor_file_sha = None
    if "role_anchor" in config:
        role_anchor_path, role_anchor_file_sha = _input_path(
            relative_to, config.get("role_anchor"), "family role anchor"
        )
    lock_path, lock_file_sha = _input_path(
        relative_to, config.get("protected_family_lock"), "protected-family lock"
    )
    attempts_path, attempts_file_sha = _input_path(relative_to, config.get("attempts"), "attempts")
    plan = _read_json(campaign_path, "campaign")
    inventory = _read_json(inventory_path, "inventory")
    split = _read_json(split_path, "family split")
    role_anchor = (
        None if role_anchor_path is None else _read_json(role_anchor_path, "family role anchor")
    )
    if requires_role_anchor(split) and role_anchor is None:
        raise ValueError("anchored family split requires its role_anchor input")
    if not requires_role_anchor(split) and role_anchor is not None:
        raise ValueError("parameterized family split must not carry a role_anchor input")
    lock = _read_json(lock_path, "protected-family lock")
    expected, model, treatment = _campaign_cells(plan, model_alias)
    campaign_sha256 = _campaign_sha(plan)
    assignments = _split_assignments(split, inventory, role_anchor=role_anchor)
    protected_groups = _protected_groups(lock)
    if lock["source_split_sha256"] != split["sha256"]:
        raise ValueError("protected-family lock is not bound to the current family split")
    nontrain_groups = {
        assignment["group_id"]
        for assignment in assignments.values()
        if assignment["split"] != "train"
    }
    if protected_groups != nontrain_groups:
        raise ValueError(
            "protected-family lock must name exactly every immutable nontraining family"
        )
    for assignment in assignments.values():
        if assignment["split"] == "train" and assignment["group_id"] in protected_groups:
            raise ValueError("family split assigns an immutable held-out family to training")

    attempts = [_attempt(row) for row in iter_jsonl(attempts_path)]
    rejections = collections.Counter({reason: 0 for reason in _REJECTION_REASONS})
    by_cell: dict[tuple[str, str, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in attempts:
        key = (row["task_key"], row["task_version_id"], row["model_alias"], row["attempt"])
        if key not in expected:
            rejections["unplanned_campaign_cell"] += 1
            continue
        by_cell[key].append(row)
    for key in expected:
        if key not in by_cell:
            rejections["missing_planned_attempt"] += 1

    singletons: list[dict[str, Any]] = []
    for _key, rows in by_cell.items():
        if len(rows) != 1:
            rejections["ambiguous_campaign_cell"] += len(rows)
            continue
        singletons.append(rows[0])
    by_session: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in singletons:
        by_session[row["session_id"]].append(row)
    uniquely_sourced: list[dict[str, Any]] = []
    for rows in by_session.values():
        if len(rows) != 1:
            rejections["duplicate_session"] += len(rows)
            continue
        uniquely_sourced.append(rows[0])

    candidates: list[dict[str, Any]] = []
    for row in uniquely_sourced:
        reason = _reason_for_candidate(
            row,
            model=model,
            treatment=treatment,
            campaign_sha256=campaign_sha256,
            template_sha256=template_sha256,
        )
        if reason is not None:
            rejections[reason] += 1
            continue
        assignment = assignments.get((row["task_key"], row["task_version_id"]))
        if assignment is None or assignment["split"] != "train":
            rejections["heldout_or_nontrain"] += 1
            continue
        if assignment["group_id"] in protected_groups:
            raise ValueError("protected held-out family reached a train candidate")
        candidates.append({**row, "group_id": assignment["group_id"]})

    by_trajectory: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates:
        by_trajectory[row["ingestion"]["normalized_trajectory_sha256"]].append(row)
    unique_trajectories: list[dict[str, Any]] = []
    for rows in by_trajectory.values():
        ordered = sorted(rows, key=_selection_rank)
        unique_trajectories.append(ordered[0])
        rejections["exact_trajectory_duplicate"] += len(ordered) - 1

    selected: list[dict[str, Any]] = []
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in unique_trajectories:
        by_task[(row["task_key"], row["task_version_id"])].append(row)
    for rows in by_task.values():
        ordered = sorted(rows, key=_selection_rank)
        selected.extend(ordered[:cap])
        rejections["per_task_session_cap"] += max(0, len(ordered) - cap)
    selected.sort(
        key=lambda row: (
            row["task_key"],
            row["task_version_id"],
            _selection_rank(row),
        )
    )

    selected_rows = [
        {
            "session_id": row["session_id"],
            "campaign_plan_sha256": row["campaign_plan_sha256"],
            "cell_sha256": row["cell_sha256"],
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "group_id": row["group_id"],
            "attempt": row["attempt"],
            "source_kind": source["kind"],
            "model": row["model"],
            "harness": row["harness"],
            "template_sha256": row["template_sha256"],
            "normalized_record_sha256": row["ingestion"]["normalized_record_sha256"],
            "normalized_trajectory_sha256": row["ingestion"]["normalized_trajectory_sha256"],
            "transcript_sha256": row["ingestion"]["transcript_sha256"],
        }
        for row in selected
    ]
    selection = {
        "schema": SELECTION_SCHEMA,
        "artifact_kind": HANDOFF_KIND,
        "trainable_corpus_created": False,
        "parquet_created": False,
        "source_text_read": False,
        "next_required_gate": "bind_private_normalized_records_then_run_existing_dense_sft_builder",
        "campaign_plan_sha256": campaign_sha256,
        "catalog_inventory_sha256": inventory["sha256"],
        "family_split_sha256": split["sha256"],
        "family_role_anchor_sha256": None if role_anchor is None else role_anchor["sha256"],
        "protected_family_lock_sha256": lock["sha256"],
        "source_kind": source["kind"],
        "source_model_alias": model_alias,
        "source_model": model,
        "harness_treatment_sha256": "sha256:" + digest(treatment),
        "tool_catalog_sha256": treatment["tool_catalog_sha256"],
        "template_sha256": template_sha256,
        "max_sessions_per_task_version": cap,
        "selected": selected_rows,
    }
    selection["sha256"] = "sha256:" + digest(selection)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "artifact_kind": HANDOFF_KIND,
        "trainable_corpus_created": False,
        "parquet_created": False,
        "source_text_read": False,
        "next_required_gate": "bind_private_normalized_records_then_run_existing_dense_sft_builder",
        "campaign_plan_sha256": campaign_sha256,
        "input_file_sha256": {
            "campaign": campaign_file_sha,
            "inventory": inventory_file_sha,
            "family_split": split_file_sha,
            "role_anchor": role_anchor_file_sha,
            "protected_family_lock": lock_file_sha,
            "attempts": attempts_file_sha,
        },
        "catalog_inventory_sha256": inventory["sha256"],
        "family_split_sha256": split["sha256"],
        "family_role_anchor_sha256": None if role_anchor is None else role_anchor["sha256"],
        "protected_family_lock_sha256": lock["sha256"],
        "protected_family_count": len(protected_groups),
        "source_kind": source["kind"],
        "source_model_alias": model_alias,
        "source_model_identity_sha256": "sha256:" + digest(model),
        "harness_treatment_sha256": "sha256:" + digest(treatment),
        "template_sha256": template_sha256,
        "tool_catalog_sha256": treatment["tool_catalog_sha256"],
        "max_sessions_per_task_version": cap,
        "counts": {
            "attempt_metadata_records": len(attempts),
            "planned_cells_for_source_model": len(expected),
            "train_cells_for_source_model": sum(
                assignments.get((cell["task_key"], cell["task_version_id"]), {}).get("split")
                == "train"
                for cell in expected.values()
            ),
            "admitted_sessions": len(selected_rows),
            "admitted_task_versions": len(
                {(row["task_key"], row["task_version_id"]) for row in selected_rows}
            ),
            "rejections": {reason: rejections[reason] for reason in _REJECTION_REASONS},
        },
        "policy": {
            "training_data_eligible_required": True,
            "success_requires_completed_authoritative_verifier": True,
            "visible_actions_only": True,
            "private_or_unknown_reasoning_rejected": True,
            "opaque_or_unapproved_compaction_rejected": True,
            "exact_session_and_trajectory_deduplication": True,
            "per_task_version_cap_is_deterministic": True,
            "heldout_families_admitted": 0,
            "raw_source_payload_read": False,
        },
        "selection_sha256": selection["sha256"],
    }
    _assert_public(receipt)
    receipt["sha256"] = "sha256:" + digest(receipt)

    # Inputs are rehashed immediately before publication so a changing private
    # collection file cannot be confused with the reviewed admission result.
    for path, expected_sha, label in (
        (campaign_path, campaign_file_sha, "campaign"),
        (inventory_path, inventory_file_sha, "inventory"),
        (split_path, split_file_sha, "family split"),
        (role_anchor_path, role_anchor_file_sha, "family role anchor"),
        (lock_path, lock_file_sha, "protected-family lock"),
        (attempts_path, attempts_file_sha, "attempts"),
    ):
        if path is not None and file_sha256(path) != expected_sha:
            raise ValueError(f"{label} changed during collection admission")
    _write_output(output, selection, receipt)
    return {
        "submitted": False,
        "artifact_kind": HANDOFF_KIND,
        "trainable_corpus_created": False,
        "parquet_created": False,
        "source_text_read": False,
        "next_required_gate": "bind_private_normalized_records_then_run_existing_dense_sft_builder",
        "receipt_sha256": receipt["sha256"],
        "selection_sha256": selection["sha256"],
        "catalog_inventory_sha256": inventory["sha256"],
        "admitted_sessions": len(selected_rows),
        "admitted_task_versions": receipt["counts"]["admitted_task_versions"],
        "rejections": receipt["counts"]["rejections"],
    }
