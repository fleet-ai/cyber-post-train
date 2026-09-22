"""Render the completion-budget-safe v3 visible-action campaign packet.

The v1/v2 renderers and committed artifacts remain unchanged.  V3 preserves
their task, source, harness, sampling, compaction, and admission science while
binding the successor runtime and its exact chat-completion budget policy.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

from evals.fleet import visible_action_collection_v3 as runtime
from training import collection_campaign as v1
from training import collection_campaign_v2 as v2

PACKET_SCHEMA = "cyber_trajectory_collection_packet_v3"
JOB_EXECUTION_REQUIREMENTS = v2.JOB_EXECUTION_REQUIREMENTS


def _runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    old = value["collection_runtime"]
    value["collection_runtime"] = {
        "schema": runtime.RUNTIME_SCHEMA,
        "source_template_sha256": old["source_template_sha256"],
        "reasoning_generation": runtime.THINKING_DISABLED,
        "reasoning_request_override": {"chat_template_kwargs": {"enable_thinking": False}},
        "opencode_model_reasoning": False,
        "opencode_cli_thinking_flag": False,
        "maximum_planned_cells": old["maximum_planned_cells"],
        "operation_authorization_required": True,
        "canonical_private_operation_root_required": True,
        "exclusive_pre_mutation_intent_required": True,
        "dedicated_empty_ledger_required": True,
        "automatic_replay_of_ambiguous_cells": False,
        "external_submission": False,
        "execution_mode": runtime.EXECUTION_MODE,
        "cluster_wrapper_supported": True,
        "completion_budget": copy.deepcopy(runtime.COMPLETION_BUDGET_POLICY),
    }
    return value


def _packet(
    packet: dict[str, Any],
    config: dict[str, Any],
    plan: dict[str, Any],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(packet)
    value.pop("sha256")
    runtime_sha = v1.canonical_digest(runtime.runtime_identity())
    value["schema"] = PACKET_SCHEMA
    value["eval_config_sha256"] = v1.canonical_digest(config)
    value["eval_plan_sha256"] = "sha256:" + plan["sha256"]
    value["operation_authorization_sha256"] = authorization["sha256"]
    value["completion_budget_runtime_sha256"] = runtime_sha
    value["admission_policy"]["adapter_must_bind"] = [
        "collection_packet_sha256",
        "eval_plan_sha256",
        "operation_authorization_sha256",
        "completion_budget_runtime_sha256",
    ]
    value["execution_safety"] = {
        "execution_mode": runtime.EXECUTION_MODE,
        "planned_cells": plan["planned_cells"],
        "maximum_planned_cells": config["collection_runtime"]["maximum_planned_cells"],
        "operation_authorization_required": True,
        "operation_authorization_sha256": authorization["sha256"],
        "canonical_private_operation_root_required": True,
        "operation_root_name": authorization["operation_root_name"],
        "exclusive_pre_mutation_intent_required": True,
        "dedicated_empty_ledger_required": True,
        "dedicated_ledger_id": authorization["dedicated_ledger_id"],
        "identity_map_sha256": authorization["identity_map_sha256"],
        "automatic_replay_of_ambiguous_cells": False,
        "same_path_retry_allowed": False,
        "alternate_path_retry_allowed": False,
        "external_submission": False,
        "cluster_wrapper_supported": True,
        "cluster_job_execution_requirements": copy.deepcopy(JOB_EXECUTION_REQUIREMENTS),
        "completion_budget_runtime_sha256": runtime_sha,
        "completion_budget": copy.deepcopy(runtime.COMPLETION_BUDGET_POLICY),
    }
    return v1.sealed(value)


def compile_local(selection: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "task-selection.json").write_bytes(v1.raw(selection))
        return runtime.compile_eval(config, relative_to=root)


def render(
    request: dict[str, Any],
    inventory: dict[str, Any],
    split: dict[str, Any],
    runtime_bindings: dict[str, Any],
    *,
    role_anchor: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    rendered = v1.render(
        request,
        inventory,
        split,
        runtime_bindings,
        role_anchor=role_anchor,
    )
    config = _runtime_config(rendered["eval-config.json"])
    plan = compile_local(rendered["task-selection.json"], config)
    authorization = runtime.build_operation_authorization(plan)
    packet = _packet(rendered["collection-packet.json"], config, plan, authorization)
    return {
        **rendered,
        "eval-config.json": config,
        "operation-authorization.json": authorization,
        "collection-packet.json": packet,
    }
