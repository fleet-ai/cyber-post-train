"""Materialize the immutable completion-budget-safe v3 action campaign."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evals.fleet import visible_action_collection_v3 as runtime
from training import collection_campaign as campaign_v1
from training import collection_campaign_v3 as campaign_v3
from training import current_collection_campaigns as current
from training import current_collection_campaigns_v2 as current_v2

SPEC_SCHEMA = "cyber_collection_campaign_source_spec_v3"
RECEIPT_SCHEMA = "cyber_collection_campaign_materialization_receipt_v3"


def _v2_compatible_spec(spec: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(spec)
    value["schema"] = current_v2.SPEC_SCHEMA
    value["safety"].pop("completion_budget")
    value["sha256"] = campaign_v1.canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


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
        raise ValueError("collection v3 source spec has unknown or missing fields")
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
        "completion_budget": runtime.COMPLETION_BUDGET_POLICY,
    }
    if spec.get("safety") != expected_safety:
        raise ValueError("collection v3 safety policy drift")
    current_v2._validate_spec(_v2_compatible_spec(spec))  # noqa: SLF001


def render(spec: dict[str, Any], *, root: Path) -> dict[str, dict[str, Any]]:
    """Render exact v3 bytes without network, credentials, traces or scores."""
    _validate_spec(spec)
    v2_outputs = current_v2.render(_v2_compatible_spec(spec), root=root)
    request = copy.deepcopy(v2_outputs["collection-request.json"])
    request["source_authorization_receipt_sha256"] = spec["sha256"]
    rendered = campaign_v3.render(
        request,
        v2_outputs["metadata-inventory.json"],
        v2_outputs["family-split.json"],
        v2_outputs["runtime-bindings.json"],
        role_anchor=v2_outputs["role-anchor.json"],
    )
    outputs = {
        **{
            name: value
            for name, value in v2_outputs.items()
            if name
            not in {
                "collection-request.json",
                "eval-config.json",
                "task-selection.json",
                "operation-authorization.json",
                "collection-packet.json",
                "materialization-receipt.json",
            }
        },
        "collection-request.json": request,
        **rendered,
    }
    operation = rendered["operation-authorization.json"]
    old_receipt = v2_outputs["materialization-receipt.json"]
    receipt = campaign_v1.sealed(
        {
            **{
                key: item
                for key, item in old_receipt.items()
                if key
                not in {
                    "schema",
                    "sha256",
                    "source_spec_sha256",
                    "output_logical_sha256",
                    "operation_authorization_sha256",
                    "operation_root_name",
                    "dedicated_ledger_id",
                    "identity_map_sha256",
                }
            },
            "schema": RECEIPT_SCHEMA,
            "source_spec_sha256": spec["sha256"],
            "output_logical_sha256": {
                name: value.get("sha256", campaign_v1.canonical_digest(value))
                for name, value in outputs.items()
            },
            "operation_authorization_sha256": operation["sha256"],
            "operation_root_name": operation["operation_root_name"],
            "dedicated_ledger_id": operation["dedicated_ledger_id"],
            "identity_map_sha256": operation["identity_map_sha256"],
            "completion_budget_runtime_sha256": rendered["collection-packet.json"][
                "completion_budget_runtime_sha256"
            ],
            "completion_budget": runtime.COMPLETION_BUDGET_POLICY,
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
