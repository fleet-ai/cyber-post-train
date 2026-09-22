"""Freeze one train family x pass@4 completion-budget collection canary."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from training import collection_campaign as v1
from training import collection_campaign_v3 as v3
from training import current_collection_campaigns as current

SPEC_SCHEMA = "cyber_visible_action_collection_budget_canary_source_v1"
RECEIPT_SCHEMA = "cyber_visible_action_collection_budget_canary_receipt_v1"


def _load(root: Path, relative: object, label: str) -> dict[str, Any]:
    if not isinstance(relative, str):
        raise ValueError(f"{label} path is required")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must stay inside the repository") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_spec(spec: dict[str, Any]) -> None:
    current._sealed(spec, SPEC_SCHEMA)  # noqa: SLF001
    if set(spec) != {
        "schema",
        "campaign_name",
        "parent",
        "selected_task_version_id",
        "selected_family_group_id",
        "attempts_per_task",
        "maximum_planned_cells",
        "authorization",
        "sha256",
    }:
        raise ValueError("collection canary spec has unknown or missing fields")
    if (
        spec.get("attempts_per_task") != 4
        or spec.get("maximum_planned_cells") != 4
        or spec.get("authorization")
        != {
            "external_launch_by_this_artifact": False,
            "reasoning_generation": "disabled",
            "bounded_canary_before_scale": True,
        }
    ):
        raise ValueError("collection canary safety policy drift")


def render(spec: dict[str, Any], *, root: Path) -> dict[str, dict[str, Any]]:
    _validate_spec(spec)
    parent = spec["parent"]
    if not isinstance(parent, dict) or set(parent) != {
        "eval_config",
        "task_selection",
        "collection_packet",
        "family_split",
    }:
        raise ValueError("collection canary parent bindings are incomplete")
    parent_config = _load(root, parent["eval_config"], "parent eval config")
    parent_selection = _load(root, parent["task_selection"], "parent task selection")
    parent_packet = _load(root, parent["collection_packet"], "parent collection packet")
    parent_split = _load(root, parent["family_split"], "parent family split")
    selected = [
        row
        for row in parent_selection.get("tasks", [])
        if row.get("task_version_id") == spec["selected_task_version_id"]
    ]
    roles = [
        row
        for row in parent_split.get("tasks", [])
        if row.get("task_version_id") == spec["selected_task_version_id"]
    ]
    if (
        len(selected) != 1
        or len(roles) != 1
        or roles[0].get("split") != "train"
        or roles[0].get("group_id") != spec["selected_family_group_id"]
    ):
        raise ValueError("canary task is not one exact admitted train family")

    selection = copy.deepcopy(parent_selection)
    selection.pop("sha256")
    selection["tasks"] = selected
    selection = v1.sealed(selection)
    config = copy.deepcopy(parent_config)
    config["name"] = spec["campaign_name"]
    config["routes"]["base"]["task_versions"] = [spec["selected_task_version_id"]]
    config["collection_runtime"]["maximum_planned_cells"] = 4
    plan = v3.compile_local(selection, config)
    authorization = v3.runtime.build_operation_authorization(plan)
    packet_input = copy.deepcopy(parent_packet)
    packet_input["task_selection_sha256"] = selection["sha256"]
    packet = v3._packet(packet_input, config, plan, authorization)  # noqa: SLF001
    outputs = {
        "eval-config.json": config,
        "task-selection.json": selection,
        "operation-authorization.json": authorization,
        "collection-packet.json": packet,
    }
    receipt = v1.sealed(
        {
            "schema": RECEIPT_SCHEMA,
            "source_spec_sha256": spec["sha256"],
            "parent_collection_packet_sha256": parent_packet["sha256"],
            "selected_task_version_id": spec["selected_task_version_id"],
            "selected_family_group_id": spec["selected_family_group_id"],
            "heldout_roles_excluded": ["dev", "final_test"],
            "planned_cells": 4,
            "operation_authorization_sha256": authorization["sha256"],
            "completion_budget_runtime_sha256": packet["completion_budget_runtime_sha256"],
            "output_logical_sha256": {
                name: value.get("sha256", v1.canonical_digest(value))
                for name, value in outputs.items()
            },
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
                "planned_cells": 4,
                "submitted": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
