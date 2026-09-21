"""Render the exactly-once v2 successor to a visible-action campaign packet.

The v1 renderer remains unchanged and continues to describe the historical,
non-launchable duplicate-census design.  This module reuses its reviewed task,
model, harness, sampling and admission policy, then replaces only the execution
contract with the v2 operation authorization and runtime.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

from evals.fleet import visible_action_collection_v2 as runtime
from training import collection_campaign as v1

PACKET_SCHEMA = "cyber_trajectory_collection_packet_v2"
JOB_ACTIVE_DEADLINE_SECONDS = 768_600
JOB_EXECUTION_REQUIREMENTS = {
    "root_kind": "Job",
    "architecture": "amd64",
    "priority_class_name": "c1",
    "completions": 1,
    "parallelism": 1,
    "active_deadline_seconds": JOB_ACTIVE_DEADLINE_SECONDS,
    "backoff_limit": 0,
    "restart_policy": "Never",
    "zero_gpu_requests_and_limits_across_regular_and_init_containers": True,
    "required_top_level_annotation": {v1.FAILURE_ALERT_ANNOTATION: "off"},
    "server_preview_count": 2,
    "identical_normalized_server_preview_digests_required": True,
    "exclusive_local_cluster_create_intent_required": True,
    "cluster_create_attempts": 1,
    "ambiguous_cluster_create_retry_allowed": False,
    "terminal_exact_name_and_uid_observation_required": True,
    "exact_uid_foreground_cleanup_required": True,
    "owned_child_absence_and_no_idle_proof_required": True,
    "exact_launcher_bindings_required": [
        "operation_authorization_sha256",
        "operation_root_name",
        "dedicated_ledger_id",
        "source_git_commit",
        "source_git_tree",
        "agent_image_digest",
        "proxy_image_digest",
        "controller_image_digest",
        "database_identity",
        "sfs_pvc_identity",
        "kube_context",
        "namespace",
        "queue_name",
    ],
}


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
    value["schema"] = PACKET_SCHEMA
    value["eval_config_sha256"] = v1.canonical_digest(config)
    value["eval_plan_sha256"] = "sha256:" + plan["sha256"]
    value["operation_authorization_sha256"] = authorization["sha256"]
    value["admission_policy"]["adapter_must_bind"] = [
        "collection_packet_sha256",
        "eval_plan_sha256",
        "operation_authorization_sha256",
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
    }
    return v1.sealed(value)


def render(
    request: dict[str, Any],
    inventory: dict[str, Any],
    split: dict[str, Any],
    runtime_bindings: dict[str, Any],
    *,
    role_anchor: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Render v2 locally without contacting Fleet or opening private content."""
    rendered = v1.render(
        request,
        inventory,
        split,
        runtime_bindings,
        role_anchor=role_anchor,
    )
    config = _runtime_config(rendered["eval-config.json"])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "task-selection.json").write_bytes(v1.raw(rendered["task-selection.json"]))
        plan = runtime.compile_eval(config, relative_to=root)
    authorization = runtime.build_operation_authorization(plan)
    packet = _packet(rendered["collection-packet.json"], config, plan, authorization)
    return {
        **rendered,
        "eval-config.json": config,
        "operation-authorization.json": authorization,
        "collection-packet.json": packet,
    }
