"""Render a source-only, family-safe Fleet trajectory-collection campaign.

This module deliberately has no Fleet client, credential handling, workload
submission, transcript reader, or scorer.  It takes only sealed metadata about
exact task bindings and family roles, renders the existing generic OpenCode
evaluation inputs, and records the data-admission policy that a later private
adapter must enforce.

The rendered files are inputs to ``cyber-post-train eval prepare``; rendering
them is not a launch and does not contact Fleet.

This v1 packet collects only visible actions from verifier-confirmed successes.
It rejects teacher hidden-thinking fields.  A corpus with student-visible
reasoning, if separately authorized and safety-reviewed, must use a new
immutable campaign packet; this policy cannot be relaxed in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import evaluate
from training import task_family_split

INVENTORY_SCHEMA = "cyber_collection_metadata_inventory_v1"
RUNTIME_BINDINGS_SCHEMA = "cyber_collection_runtime_bindings_v1"
REQUEST_SCHEMA = "cyber_trajectory_collection_request_v1"
SELECTION_SCHEMA = "cyber_collection_task_selection_v1"
PACKET_SCHEMA = "cyber_trajectory_collection_packet_v1"
VISIBLE_ACTIONS_ONLY = "visible_actions_only_v1"
OPAQUE_COMPACTION_REJECT = "reject_opaque_context_compaction_v1"
ONLINE_COMPACTION = "opencode_1.18.27_native_compaction_autocontinue_v2"
MINIMUM_VISIBLE_TARGET_TOKENS = 20_000_000
EXACT_TASK_FIELDS = (
    "task_key",
    "task_version_id",
    "env_key",
    "env_version",
    "environment_version_id",
    "data_key",
    "data_version",
)
METADATA_TASK_FIELDS = ("task_key", "task_version_id", "lineage")
LINEAGE_FIELDS = {
    "application",
    "environment",
    "difficulty",
    "vulnerability_family",
    "task_family",
}
FORBIDDEN_CONTENT_KEYS = {
    "prompt",
    "prompts",
    "trace",
    "traces",
    "answer",
    "answers",
    "flag",
    "flags",
    "credential",
    "credentials",
    "session_id",
    "score",
    "reasoning_content",
    "thinking",
    "reasoning",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sealed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["sha256"] = canonical_digest(result)
    return result


def raw(value: dict[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError(f"exact {label} SHA-256 is required")
    return value


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", value):
        raise ValueError(f"safe lowercase {label} is required")
    return value


def _sealed(value: dict[str, Any], schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid sealed {schema}")


def _no_private_content_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if key.lower() in FORBIDDEN_CONTENT_KEYS:
                raise ValueError("collection metadata must not contain private content fields")
            _no_private_content_keys(item)
    elif isinstance(value, list):
        for item in value:
            _no_private_content_keys(item)


def _exact_uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"canonical {label} UUID is required")
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError
    except ValueError as error:
        raise ValueError(f"canonical {label} UUID is required") from error
    return value


def _inventory_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    _sealed(inventory, INVENTORY_SCHEMA)
    if set(inventory) != {
        "schema",
        "task_validity_receipt_sha256",
        "task_versions",
        "sha256",
    }:
        raise ValueError("collection inventory has unknown or missing fields")
    _sha256(inventory["task_validity_receipt_sha256"], "task-validity receipt")
    _no_private_content_keys(inventory)
    rows = inventory["task_versions"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("collection inventory needs task_versions")
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(METADATA_TASK_FIELDS):
            raise ValueError("collection inventory row has unknown or missing fields")
        identity = (row["task_key"], _exact_uuid(row["task_version_id"], "task version"))
        if not isinstance(identity[0], str) or not identity[0] or identity in seen:
            raise ValueError("collection inventory identities must be unique")
        seen.add(identity)
        lineage = row["lineage"]
        if not isinstance(lineage, dict) or set(lineage) != LINEAGE_FIELDS:
            raise ValueError("collection inventory lineage is incomplete")
        for key in LINEAGE_FIELDS - {"vulnerability_family"}:
            if not isinstance(lineage[key], str) or not lineage[key] or lineage[key] == "unknown":
                raise ValueError("collection inventory needs reviewed family metadata")
        families = lineage["vulnerability_family"]
        if (
            not isinstance(families, list)
            or not families
            or any(not isinstance(item, str) or not item or item == "unknown" for item in families)
        ):
            raise ValueError("collection inventory needs reviewed vulnerability-family metadata")
    return rows


def _runtime_bindings(
    value: dict[str, Any], inventory: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    _sealed(value, RUNTIME_BINDINGS_SCHEMA)
    if set(value) != {
        "schema",
        "metadata_inventory_sha256",
        "task_validity_receipt_sha256",
        "task_versions",
        "sha256",
    }:
        raise ValueError("runtime bindings have unknown or missing fields")
    if value["metadata_inventory_sha256"] != inventory["sha256"]:
        raise ValueError("runtime bindings are for a different metadata inventory")
    if value["task_validity_receipt_sha256"] != inventory["task_validity_receipt_sha256"]:
        raise ValueError("runtime bindings use a different task-validity receipt")
    _no_private_content_keys(value)
    bindings = value["task_versions"]
    if not isinstance(bindings, list):
        raise ValueError("runtime bindings need task_versions")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in bindings:
        if not isinstance(row, dict) or set(row) != set(EXACT_TASK_FIELDS):
            raise ValueError("runtime binding row has unknown or missing fields")
        identity = (row["task_key"], _exact_uuid(row["task_version_id"], "task version"))
        if not isinstance(identity[0], str) or not identity[0] or identity in result:
            raise ValueError("runtime binding identities must be unique")
        _exact_uuid(row["environment_version_id"], "environment version")
        if any(not isinstance(row[field], str) or not row[field] for field in EXACT_TASK_FIELDS):
            raise ValueError("runtime bindings need exact task/environment/data values")
        result[identity] = row
    expected = {(row["task_key"], row["task_version_id"]) for row in rows}
    if set(result) != expected:
        raise ValueError("runtime bindings must cover exactly the metadata inventory")
    return result


def _request(value: dict[str, Any]) -> None:
    allowed = {
        "schema",
        "campaign_name",
        "source_kind",
        "source_model",
        "template_sha256",
        "source_authorization_receipt_sha256",
        "teacher_strength_receipt_sha256",
        "route",
        "harness",
        "images",
        "sampling",
        "concurrency",
        "attempts_per_task",
        "target_unique_visible_action_tokens",
        "reasoning_policy",
        "offline_compaction_policy",
    }
    required = allowed - {"teacher_strength_receipt_sha256"}
    if not required <= set(value) or set(value) - allowed or value.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unknown or missing collection-request fields")
    _name(value.get("campaign_name"), "campaign name")
    if value.get("source_kind") not in {"self", "teacher"}:
        raise ValueError("source_kind must be self or teacher")
    model = value.get("source_model")
    if not isinstance(model, dict) or set(model) != {"repository", "revision", "session_model"}:
        raise ValueError("source_model needs repository, revision and session_model")
    if not all(isinstance(item, str) and item for item in model.values()):
        raise ValueError("source_model identity must be nonempty")
    if re.fullmatch(r"(?:[a-f0-9]{40}|sha256:[a-f0-9]{64})", model["revision"]) is None:
        raise ValueError("source_model requires an immutable revision")
    _sha256(value.get("template_sha256"), "source template")
    _sha256(value.get("source_authorization_receipt_sha256"), "source-authorization receipt")
    if value["source_kind"] == "self":
        if set(value) != required:
            raise ValueError("self collection has an invalid request shape")
        if model["repository"] != "Qwen/Qwen3.8-27B":
            raise ValueError("self collection is reserved for the exact Qwen3.8 student")
    else:
        if "teacher_strength_receipt_sha256" not in value:
            raise ValueError("teacher-strength receipt is required for teacher collection")
        if set(value) != required | {"teacher_strength_receipt_sha256"}:
            raise ValueError("teacher collection has an invalid request shape")
        _sha256(value.get("teacher_strength_receipt_sha256"), "teacher-strength receipt")
    if value.get("reasoning_policy") != VISIBLE_ACTIONS_ONLY:
        raise ValueError("collection currently permits visible actions only")
    if value.get("offline_compaction_policy") != OPAQUE_COMPACTION_REJECT:
        raise ValueError("opaque compaction must be rejected for offline SFT")
    if value.get("attempts_per_task") != 4:
        raise ValueError("high-throughput collection uses exactly four predeclared attempts")
    if type(value.get("concurrency")) is not int or not 1 <= value["concurrency"] <= 32:
        raise ValueError("concurrency must be an integer from 1 through 32")
    if (
        type(value.get("target_unique_visible_action_tokens")) is not int
        or value["target_unique_visible_action_tokens"] < MINIMUM_VISIBLE_TARGET_TOKENS
    ):
        raise ValueError("collection target must be at least 20M unique visible target tokens")
    if not isinstance(value.get("route"), dict) or set(value["route"]) != {
        "name",
        "served_id",
        "catalog",
        "model_info",
        "server_info",
        "endpoint_origin",
    }:
        raise ValueError("collection route must bind one complete serving profile")
    _name(value["route"]["name"], "route name")


def _split_roles(
    split: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    task_family_split.validate(split, rows)
    roles = {row["split"] for row in split["tasks"]}
    if roles != {"train", "dev", "final_test"}:
        raise ValueError("scale-up collection requires train, dev and final_test roles")
    result = {(row["task_key"], row["task_version_id"]): row for row in split["tasks"]}
    if len(result) != len(rows):
        raise ValueError("family split does not assign every inventory task exactly once")
    if not any(row["split"] == "train" for row in result.values()):
        raise ValueError("family split has no training task")
    if not any(row["split"] in {"dev", "final_test"} for row in result.values()):
        raise ValueError("family split has no held-out task")
    return result


def _selection(
    rows: list[dict[str, Any]],
    bindings: dict[tuple[str, str], dict[str, Any]],
    roles: dict[tuple[str, str], dict[str, Any]],
    inventory: dict[str, Any],
    split: dict[str, Any],
    runtime_bindings: dict[str, Any],
) -> dict[str, Any]:
    train = [
        {
            field: bindings[(row["task_key"], row["task_version_id"])][field]
            for field in EXACT_TASK_FIELDS
        }
        for row in rows
        if roles[(row["task_key"], row["task_version_id"])]["split"] == "train"
    ]
    train.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    held_out = [row for row in roles.values() if row["split"] in {"dev", "final_test"}]
    return sealed(
        {
            "schema": SELECTION_SCHEMA,
            "inventory_sha256": inventory["sha256"],
            "family_split_sha256": split["sha256"],
            "runtime_bindings_sha256": runtime_bindings["sha256"],
            "task_validity_receipt_sha256": inventory["task_validity_receipt_sha256"],
            "tasks": train,
            "held_out_task_version_count": len(held_out),
            "family_leakage_check": {
                "exact_identity_overlap": 0,
                "reviewed_family_overlap": 0,
                "held_out_roles": ["dev", "final_test"],
            },
        }
    )


def _config(request: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    route = dict(request["route"])
    route["model"] = "source"
    route["task_versions"] = [row["task_version_id"] for row in selection["tasks"]]
    harness = request["harness"]
    if not isinstance(harness, dict):
        raise ValueError("complete OpenCode harness is required")
    if (
        harness.get("context_management") != ONLINE_COMPACTION
        or harness.get("context_window_size") != 262144
        or harness.get("tools") != ["bash", "submit_report"]
    ):
        raise ValueError("collection requires qualified 262K OpenCode actions-only treatment")
    config = {
        "name": request["campaign_name"],
        "task_set": "task-selection.json",
        "models": {"source": request["source_model"]},
        "routes": {route.pop("name"): route},
        "harness": harness,
        "images": request["images"],
        "pass_k": request["attempts_per_task"],
        "concurrency": request["concurrency"],
        "max_reviewed_infrastructure_retries": 0,
        "training_data_eligible": True,
        "sampling": request["sampling"],
    }
    return config


def _packet(
    request: dict[str, Any],
    selection: dict[str, Any],
    config: dict[str, Any],
    runtime_bindings: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    source = {
        "kind": request["source_kind"],
        # ``source`` is the only model alias rendered by this source-only
        # collection packet.  Seal it so the later admission/materialization
        # path cannot silently bind a session from a second model route.
        "model_alias": "source",
        "model": request["source_model"],
        "template_sha256": request["template_sha256"],
        "source_authorization_receipt_sha256": request["source_authorization_receipt_sha256"],
    }
    if request["source_kind"] == "teacher":
        source["teacher_strength_receipt_sha256"] = request["teacher_strength_receipt_sha256"]
    return sealed(
        {
            "schema": PACKET_SCHEMA,
            "source": source,
            "task_selection_sha256": selection["sha256"],
            "runtime_bindings_sha256": runtime_bindings["sha256"],
            "eval_config_sha256": canonical_digest(config),
            "eval_plan_sha256": "sha256:" + plan["sha256"],
            # Generic eval plans retain only selection.source_job_id.  Generic
            # source-only collection has no historical Job to name, so this is
            # deliberately null rather than a fabricated provenance claim.
            "generic_plan_source_job_id": None,
            "training_data_eligible": True,
            "corpus_scope": {
                "current": "verified_success_visible_actions_only_v1",
                "visible_reasoning_included": False,
                "future_visible_reasoning_requires": [
                    "separate_immutable_campaign_packet",
                    "student_visible_source_evidence",
                    "explicit_authorization_and_safety_evidence",
                ],
            },
            "admission_policy": {
                # Collection may preserve failed attempts as operational
                # evidence, but only a verifier-confirmed success may enter
                # either the self-SFT or teacher-SFT corpus.
                "eligible_terminal_outcome": "verified_success_v1",
                "reasoning_policy": request["reasoning_policy"],
                "offline_compaction_policy": request["offline_compaction_policy"],
                "minimum_unique_visible_action_target_tokens": request[
                    "target_unique_visible_action_tokens"
                ],
                # A final submit alone is not useful supervision for the
                # general black-box exploitation skill.  The later admission
                # adapter must certify the preceding visible tool work too.
                "minimum_non_submit_tool_responses": 1,
                "minimum_completed_non_submit_tool_rounds": 1,
                "maximum_submit_report_response_fraction": 0.5,
                "maximum_submit_report_target_token_fraction": 0.5,
                # Per-task caps alone cannot stop one unusually long family
                # from consuming a broad campaign.  Keep a fixed family
                # concentration ceiling in the immutable packet.
                "maximum_family_target_token_fraction": 0.25,
                "deduplication_order": [
                    "source_session_identity",
                    "normalized_trajectory_digest",
                    "packed_window_payload_digest",
                ],
                "held_out_roles_excluded": ["dev", "final_test"],
                "adapter_must_bind": ["collection_packet_sha256", "eval_plan_sha256"],
            },
            "metrics_required": [
                "valid_success_count",
                "opaque_compaction_rejection_count",
                "private_or_unknown_reasoning_rejection_count",
                "distinct_visible_action_target_tokens",
                "family_token_concentration",
            ],
        }
    )


def render(
    request: dict[str, Any],
    inventory: dict[str, Any],
    split: dict[str, Any],
    runtime_bindings: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Render only local campaign inputs; this function performs no I/O or network calls."""
    _request(request)
    rows = _inventory_rows(inventory)
    bindings = _runtime_bindings(runtime_bindings, inventory, rows)
    roles = _split_roles(split, rows)
    selection = _selection(rows, bindings, roles, inventory, split, runtime_bindings)
    config = _config(request, selection)
    plan = _compile_local(selection, config)
    packet = _packet(request, selection, config, runtime_bindings, plan)
    rendered = {
        "task-selection.json": selection,
        "eval-config.json": config,
        "collection-packet.json": packet,
    }
    return rendered


def _compile_local(selection: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Exercise the existing generic evaluator locally, without preflight or a request."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "task-selection.json").write_bytes(raw(selection))
        plan = evaluate.compile_eval(config, relative_to=root)
    if plan["training_data_eligible"] is not True or plan["pass_k"] != 4:
        raise ValueError("rendered collection plan lost its data-eligibility contract")
    if plan["selection"] != {"source_job_id": None}:
        raise ValueError("source-only collection must not invent a historical source job")
    return plan


def write_once(output: Path, rendered: dict[str, dict[str, Any]]) -> None:
    """Create one private packet directory; existing collection evidence is immutable."""
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FileExistsError("collection packet destination already exists") from error
    for name, value in rendered.items():
        path = output / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw(value))
            stream.flush()
            os.fsync(stream.fileno())


def check(output: Path, rendered: dict[str, dict[str, Any]]) -> None:
    for name, value in rendered.items():
        path = output / name
        if not path.is_file() or path.read_bytes() != raw(value):
            raise ValueError(f"collection packet drift: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--family-split", type=Path, required=True)
    parser.add_argument("--runtime-bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render(
        _load(args.request),
        _load(args.inventory),
        _load(args.family_split),
        _load(args.runtime_bindings),
    )
    if args.write:
        write_once(args.output, rendered)
        print(json.dumps({"written": sorted(rendered), "submitted": False}, sort_keys=True))
    else:
        check(args.output, rendered)
        print(json.dumps({"checked": sorted(rendered), "submitted": False}, sort_keys=True))


if __name__ == "__main__":
    main()
