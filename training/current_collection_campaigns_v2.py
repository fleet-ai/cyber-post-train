"""Materialize the immutable exactly-once v2 successor campaign.

The reviewed inventory, split, model, OpenCode harness, sampling and 200-cell
scientific design are identical to v1.  Only execution safety is versioned:
v2 binds the deterministic cell identities and a create-once operation
authorization consumed by the qualified amd64 CPU Job launcher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training import collection_campaign as campaign_v1
from training import collection_campaign_v2 as campaign_v2
from training import current_collection_campaigns as current
from training import fleet_collection_admission as admission
from training import task_family_split

SPEC_SCHEMA = "cyber_collection_campaign_source_spec_v2"
RECEIPT_SCHEMA = "cyber_collection_campaign_materialization_receipt_v2"


def _validate_spec(spec: dict[str, Any]) -> None:
    current._sealed(spec, SPEC_SCHEMA)  # noqa: SLF001
    if set(spec) != {
        "schema",
        "campaign_name",
        "purpose",
        "inputs",
        "source",
        "selection",
        "collection",
        "authorization",
        "safety",
        "sha256",
    }:
        raise ValueError("collection v2 source spec has unknown or missing fields")
    expected_safety = {
        "source_only_materialization": True,
        "fleet_api_calls": 0,
        "model_calls": 0,
        "trace_or_score_reads": 0,
        "canonical_private_operation_root_required": True,
        "exclusive_pre_mutation_intent_required": True,
        "dedicated_empty_ledger_required": True,
        "same_path_retry_allowed": False,
        "alternate_path_retry_allowed": False,
        "ambiguous_external_mutation_replay_allowed": False,
        "cluster_wrapper_supported": True,
        "cluster_wrapper_requires_two_stable_server_previews": True,
        "required_root_annotation": {campaign_v1.FAILURE_ALERT_ANNOTATION: "off"},
    }
    if spec.get("safety") != expected_safety:
        raise ValueError("collection v2 safety policy drift")
    # Reuse the reviewed v1 source-policy validator without pretending that a
    # v1 artifact authorizes execution.  The derived object exists in memory
    # only and is never written or used as evidence.
    compatible = {
        **spec,
        "schema": current.SPEC_SCHEMA,
        "safety": {
            "source_only_materialization": True,
            "fleet_api_calls": 0,
            "model_calls": 0,
            "trace_or_score_reads": 0,
            "exact_prelaunch_duplicate_census_required": True,
            "ambiguous_cell_replay_allowed": False,
            "cluster_wrapper_supported": False,
            "cluster_wrapper_requires_two_stable_server_previews": True,
            "required_root_annotation": {campaign_v1.FAILURE_ALERT_ANNOTATION: "off"},
        },
    }
    compatible["sha256"] = campaign_v1.canonical_digest(
        {key: value for key, value in compatible.items() if key != "sha256"}
    )
    current._validate_spec(compatible)  # noqa: SLF001


def render(spec: dict[str, Any], *, root: Path) -> dict[str, dict[str, Any]]:
    """Render exact v2 bytes without network, credentials, traces or scores."""
    _validate_spec(spec)
    inventory_ref = spec["inputs"]["inventory"]
    _inventory_path, source_inventory = current._reference(  # noqa: SLF001
        root, inventory_ref, "inventory"
    )
    _split_path, source_split = current._reference(  # noqa: SLF001
        root, spec["inputs"]["family_split"], "family split"
    )
    _runtime_path, source_runtime = current._reference(  # noqa: SLF001
        root, spec["inputs"]["runtime_catalog"], "runtime catalog"
    )
    _profile_path, source_profile = current._reference(  # noqa: SLF001
        root, spec["inputs"]["source_profile"], "source profile"
    )
    expected = spec["selection"]["expected_counts"]
    inventory = current._source_inventory(  # noqa: SLF001
        source_inventory, sum(expected.values())
    )
    split, role_anchor = current._source_split(  # noqa: SLF001
        source_split, source_inventory, inventory_ref, inventory, expected
    )
    protected_family_lock = campaign_v1.sealed(
        {
            "schema": admission.PROTECTED_FAMILY_LOCK_SCHEMA,
            "source_split_sha256": split["sha256"],
            "heldout_group_ids": sorted(
                {row["group_id"] for row in split["tasks"] if row["split"] in {"dev", "final_test"}}
            ),
        }
    )
    runtime_bindings, excluded_runtime_rows = current._runtime_bindings(  # noqa: SLF001
        source_runtime,
        inventory,
        spec["inputs"]["runtime_catalog_expected_task_versions"],
    )
    request = current._request(spec, source_profile)  # noqa: SLF001
    rendered = campaign_v2.render(
        request, inventory, split, runtime_bindings, role_anchor=role_anchor
    )
    train_tasks = len(rendered["task-selection.json"]["tasks"])
    planned_cells = train_tasks * request["attempts_per_task"]
    if train_tasks != expected["train"] or planned_cells != 200:
        raise ValueError("materialized v2 campaign count or cell budget drift")
    outputs = {
        "metadata-inventory.json": inventory,
        "family-split.json": split,
        "role-anchor.json": role_anchor,
        "protected-family-lock.json": protected_family_lock,
        "runtime-bindings.json": runtime_bindings,
        "collection-request.json": request,
        **rendered,
    }
    operation = rendered["operation-authorization.json"]
    receipt = campaign_v1.sealed(
        {
            "schema": RECEIPT_SCHEMA,
            "source_spec_sha256": spec["sha256"],
            "source_input_logical_sha256": {
                name: reference["logical_sha256"]
                for name, reference in spec["inputs"].items()
                if isinstance(reference, dict) and "logical_sha256" in reference
            },
            "output_logical_sha256": {
                name: value.get("sha256", campaign_v1.canonical_digest(value))
                for name, value in outputs.items()
            },
            "inventory_task_versions": sum(expected.values()),
            "train_task_versions": train_tasks,
            "held_out_task_versions": expected["dev"] + expected["final_test"],
            "root_role_anchor_id": task_family_split.TRUSTED_FLEET_COLLECTION_ROOT_ID,
            "family_role_anchor_sha256": role_anchor["sha256"],
            "protected_family_lock_sha256": protected_family_lock["sha256"],
            "excluded_unqualified_runtime_catalog_rows": excluded_runtime_rows,
            "attempts_per_task": request["attempts_per_task"],
            "planned_cells": planned_cells,
            "operation_authorization_sha256": operation["sha256"],
            "operation_root_name": operation["operation_root_name"],
            "dedicated_ledger_id": operation["dedicated_ledger_id"],
            "identity_map_sha256": operation["identity_map_sha256"],
            "training_data_eligible": True,
            "reasoning_generation": campaign_v1.THINKING_DISABLED,
            "reasoning_targets_included": False,
            "context_window_size": request["harness"]["context_window_size"],
            "context_management": request["harness"]["context_management"],
            "execution_mode": rendered["eval-config.json"]["collection_runtime"]["execution_mode"],
            "submitted": False,
            "fleet_api_calls": 0,
            "model_calls": 0,
            "trace_or_score_reads": 0,
        }
    )
    outputs["materialization-receipt.json"] = receipt
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render(json.loads(args.spec.read_text()), root=args.root)
    if args.write:
        current.write_once(args.output, rendered)
    else:
        current.check(args.output, rendered)
    print(
        json.dumps(
            {
                "files": sorted(rendered),
                "planned_cells": rendered["materialization-receipt.json"]["planned_cells"],
                "submitted": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
