"""Fail-closed parity assessment for Fleet managed evaluation jobs.

The public ``/v1/jobs`` schema can describe a managed evaluation without
proving every immutable identity required by the exact pass@4 campaign.  This
module keeps those two statements separate: a field being accepted by the API
is not evidence that the resulting session is poolable with the frozen
treatment.

This module performs no network requests and creates no jobs.  Callers provide
the public OpenAPI document and a sanitized runtime observation assembled from
reviewed deployment source or an immutable execution receipt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "fleet-managed-jobs-route-parity-v1"
JOB_CREATE_SCHEMA = "JobCreateRequestPayload"

REQUIRED_TREATMENT = {
    "harness": "opencode",
    "harness_version": "1.18.27",
    "context_window_size": 262_144,
    "max_output_tokens": 32_768,
    "compaction_headroom_tokens": 20_000,
    "context_management": "opencode_native_compaction_autocontinue",
    "tools": ["bash", "submit_report"],
}

REQUIRED_REQUEST_FIELDS = {
    "models",
    "harness",
    "agent_runtime",
    "ci",
    "task_version_id",
    "task_group_id",
    "tools",
    "metadata",
}

# A managed service may enforce these internally, but the public create request
# cannot pin them.  They therefore require immutable downstream attestation
# before a session may enter the primary campaign ledger.
MISSING_IMMUTABLE_REQUEST_CONTROLS = (
    "model_repository_revision",
    "provider_route_revision",
    "harness_image_digest",
    "harness_release_asset_digest",
    "harness_version",
    "max_output_tokens",
    "native_compaction_headroom_tokens",
    "global_statistical_cell_claim",
)


class ManagedRouteParityError(ValueError):
    """The supplied schema or sanitized observation is incomplete."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _job_create_schema(openapi: Mapping[str, Any]) -> Mapping[str, Any]:
    components = openapi.get("components")
    schemas = components.get("schemas") if isinstance(components, Mapping) else None
    schema = schemas.get(JOB_CREATE_SCHEMA) if isinstance(schemas, Mapping) else None
    if not isinstance(schema, Mapping):
        raise ManagedRouteParityError("managed_jobs_create_schema_missing")
    return schema


def inspect_public_contract(openapi: Mapping[str, Any]) -> dict[str, Any]:
    """Return only content-free capability facts from the public schema."""
    schema = _job_create_schema(openapi)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ManagedRouteParityError("managed_jobs_create_properties_missing")
    present = sorted(REQUIRED_REQUEST_FIELDS & set(properties))
    missing = sorted(REQUIRED_REQUEST_FIELDS - set(properties))

    tools = properties.get("tools")
    tool_description = str(tools.get("description") or "") if isinstance(tools, Mapping) else ""
    task_group = properties.get("task_group_id")
    group_description = (
        str(task_group.get("description") or "") if isinstance(task_group, Mapping) else ""
    )
    compaction = properties.get("compaction_threshold_tokens")
    compaction_description = (
        str(compaction.get("description") or "") if isinstance(compaction, Mapping) else ""
    )

    return {
        "required_fields_present": present,
        "required_fields_missing": missing,
        "supports_agent_runtime_selection": "agent_runtime" in properties,
        "supports_harness_alias_selection": "harness" in properties,
        "supports_exact_task_version_selector": "task_version_id" in properties,
        "task_version_selector_requires_ci_contract": (
            "task_version_id" in properties and "ci" in properties
        ),
        "supports_pinned_task_group_selector": (
            "task_group_id" in properties and "pinned eval_task_version_id" in group_description
        ),
        "metadata_is_storage_not_uniqueness": "metadata" in properties,
        "native_tools_are_independent_of_fleet_mcp_tools": (
            "Fleet MCP servers are independent" in tool_description
        ),
        "generic_compaction_is_not_opencode_native_compaction": (
            "computer-use sessions" in compaction_description
        ),
        "immutable_request_controls_absent": list(MISSING_IMMUTABLE_REQUEST_CONTROLS),
    }


def _require_model_observation(models: Mapping[str, Any], model_key: str) -> Mapping[str, Any]:
    observation = models.get(model_key)
    if not isinstance(observation, Mapping):
        raise ManagedRouteParityError(f"runtime_observation_missing:{model_key}")
    required = {
        "managed_model_id",
        "live_managed_catalog_attested",
        "context_window_size",
        "max_output_tokens",
        "exact_model_revision_attested",
        "provider_route_revision_attested",
        "required_tool_behavior_attested",
    }
    missing = sorted(required - set(observation))
    if missing:
        raise ManagedRouteParityError(
            f"runtime_observation_incomplete:{model_key}:{','.join(missing)}"
        )
    return observation


def assess(openapi: Mapping[str, Any], runtime_observation: Mapping[str, Any]) -> dict[str, Any]:
    """Assess whether managed jobs are admissible to the primary treatment.

    ``runtime_observation`` must contain sanitized, non-content facts only.  A
    truthy catalog alias is deliberately insufficient: exact revision, route,
    tool behavior, and execution treatment must be attested independently.
    """
    contract = inspect_public_contract(openapi)
    models = runtime_observation.get("models")
    if not isinstance(models, Mapping):
        raise ManagedRouteParityError("runtime_models_missing")

    results: dict[str, Any] = {}
    for model_key in ("qwen3.8-27b", "glm-5.3"):
        observed = _require_model_observation(models, model_key)
        checks = {
            "managed_route_alias_supported_by_source": bool(observed["managed_model_id"]),
            "live_managed_catalog_attested": (observed["live_managed_catalog_attested"] is True),
            "context_window_exact": (
                observed["context_window_size"] == REQUIRED_TREATMENT["context_window_size"]
            ),
            "max_output_tokens_exact": (
                observed["max_output_tokens"] == REQUIRED_TREATMENT["max_output_tokens"]
            ),
            "exact_model_revision_attested": (observed["exact_model_revision_attested"] is True),
            "provider_route_revision_attested": (
                observed["provider_route_revision_attested"] is True
            ),
            "required_tool_behavior_attested": (
                observed["required_tool_behavior_attested"] is True
            ),
            "opencode_native_compaction_headroom_attested": (
                runtime_observation.get("native_compaction_autocontinue") is True
                and runtime_observation.get("compaction_headroom_tokens")
                == REQUIRED_TREATMENT["compaction_headroom_tokens"]
            ),
            "immutable_harness_execution_attested": (
                runtime_observation.get("immutable_harness_execution_attested") is True
            ),
            "global_cell_claim_enforced": (
                runtime_observation.get("global_cell_claim_enforced") is True
            ),
        }
        results[model_key] = {
            "managed_model_id": observed["managed_model_id"],
            "observed_context_window_size": observed["context_window_size"],
            "observed_max_output_tokens": observed["max_output_tokens"],
            "required_context_window_size": REQUIRED_TREATMENT["context_window_size"],
            "required_max_output_tokens": REQUIRED_TREATMENT["max_output_tokens"],
            "checks": checks,
            "primary_treatment_poolable": all(checks.values()),
        }

    exact_task_binding_supported = (
        contract["supports_exact_task_version_selector"]
        or contract["supports_pinned_task_group_selector"]
    )
    contract_checks = {
        "public_schema_complete": not contract["required_fields_missing"],
        "agent_runtime_and_opencode_selectable": (
            contract["supports_agent_runtime_selection"]
            and contract["supports_harness_alias_selection"]
        ),
        "exact_registered_task_version_selectable": exact_task_binding_supported,
        "direct_task_version_selection_requires_ci_true": contract[
            "task_version_selector_requires_ci_contract"
        ],
        "fleet_mcp_tools_not_confused_with_native_tools": (
            contract["native_tools_are_independent_of_fleet_mcp_tools"]
        ),
        "generic_compaction_not_misread_as_native_opencode_compaction": (
            contract["generic_compaction_is_not_opencode_native_compaction"]
        ),
    }
    primary = all(contract_checks.values()) and all(
        result["primary_treatment_poolable"] for result in results.values()
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            "PRIMARY_TREATMENT_ELIGIBLE" if primary else "PRIMARY_TREATMENT_BLOCKED"
        ),
        "scored_launch_authorized": False,
        "authority": {
            key: runtime_observation.get(key)
            for key in (
                "public_openapi_sha256",
                "public_openapi_observed_at",
                "deployment_source_commit",
                "runtime_image_tag",
                "runtime_source_commit",
                "opencode_version",
                "observation_method",
            )
        },
        "public_contract": contract,
        "contract_checks": contract_checks,
        "models": results,
        "treatment_decision": {
            "primary_campaign_poolable": primary,
            "separate_managed_service_block_only": not primary,
            "reason": (
                "all_exact_identities_attested"
                if primary
                else "managed_create_fields_do_not_prove_the_frozen_execution_identity"
            ),
        },
        "side_effects": {
            "jobs_created": 0,
            "sessions_created": 0,
            "scored_model_calls": 0,
            "mutations": 0,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_scores_or_model_outputs_included": False,
        },
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt
