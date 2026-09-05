from __future__ import annotations

import copy

import pytest

from evals.fleet import managed_jobs_route_parity_v1 as managed


def _openapi() -> dict:
    return {
        "components": {
            "schemas": {
                "JobCreateRequestPayload": {
                    "properties": {
                        "models": {"type": "array"},
                        "harness": {"type": "string"},
                        "agent_runtime": {"type": "boolean"},
                        "ci": {"type": "boolean"},
                        "task_version_id": {"type": "string"},
                        "task_group_id": {
                            "type": "string",
                            "description": "Runs pinned eval_task_version_id members.",
                        },
                        "tools": {
                            "type": "array",
                            "description": (
                                "Fleet MCP servers are independent and remain available."
                            ),
                        },
                        "context_management": {"type": "object"},
                        "compaction_threshold_tokens": {
                            "type": "integer",
                            "description": "In-loop compaction for computer-use sessions.",
                        },
                        "metadata": {"type": "object"},
                    }
                }
            }
        }
    }


def _runtime() -> dict:
    return {
        "native_compaction_autocontinue": True,
        "compaction_headroom_tokens": 20_000,
        "immutable_harness_execution_attested": False,
        "global_cell_claim_enforced": False,
        "models": {
            "qwen3.8-27b": {
                "managed_model_id": "fleet-qwen/qwen3.8-27b",
                "live_managed_catalog_attested": False,
                "context_window_size": 262_144,
                "max_output_tokens": 16_384,
                "exact_model_revision_attested": False,
                "provider_route_revision_attested": False,
                "required_tool_behavior_attested": False,
            },
            "glm-5.3": {
                "managed_model_id": "fleet-glm/glm-5.3-fleet",
                "live_managed_catalog_attested": False,
                "context_window_size": 262_144,
                "max_output_tokens": 128_000,
                "exact_model_revision_attested": False,
                "provider_route_revision_attested": False,
                "required_tool_behavior_attested": False,
            },
        },
    }


def test_managed_route_fails_closed_on_exact_treatment_gaps() -> None:
    receipt = managed.assess(_openapi(), _runtime())

    assert receipt["classification"] == "PRIMARY_TREATMENT_BLOCKED"
    assert receipt["scored_launch_authorized"] is False
    assert receipt["treatment_decision"] == {
        "primary_campaign_poolable": False,
        "separate_managed_service_block_only": True,
        "reason": "managed_create_fields_do_not_prove_the_frozen_execution_identity",
    }
    assert receipt["models"]["qwen3.8-27b"]["checks"]["max_output_tokens_exact"] is False
    assert receipt["models"]["glm-5.3"]["checks"]["max_output_tokens_exact"] is False
    assert receipt["models"]["qwen3.8-27b"]["observed_max_output_tokens"] == 16_384
    assert receipt["models"]["glm-5.3"]["observed_max_output_tokens"] == 128_000
    assert receipt["side_effects"]["jobs_created"] == 0
    assert receipt["receipt_sha256"].startswith("sha256:")


def test_task_version_is_supported_but_metadata_is_not_a_cell_claim() -> None:
    contract = managed.inspect_public_contract(_openapi())

    assert contract["supports_exact_task_version_selector"] is True
    assert contract["task_version_selector_requires_ci_contract"] is True
    assert contract["supports_pinned_task_group_selector"] is True
    assert contract["metadata_is_storage_not_uniqueness"] is True
    assert "global_statistical_cell_claim" in contract["immutable_request_controls_absent"]


def test_native_and_fleet_mcp_tool_controls_are_not_conflated() -> None:
    contract = managed.inspect_public_contract(_openapi())

    assert contract["native_tools_are_independent_of_fleet_mcp_tools"] is True
    assert contract["generic_compaction_is_not_opencode_native_compaction"] is True


def test_every_exact_attestation_is_required_for_primary_pooling() -> None:
    runtime = _runtime()
    for row in runtime["models"].values():
        row["max_output_tokens"] = 32_768
        row["exact_model_revision_attested"] = True
        row["provider_route_revision_attested"] = True
        row["required_tool_behavior_attested"] = True
        row["live_managed_catalog_attested"] = True
    runtime["immutable_harness_execution_attested"] = True
    runtime["global_cell_claim_enforced"] = True

    receipt = managed.assess(_openapi(), runtime)

    assert receipt["classification"] == "PRIMARY_TREATMENT_ELIGIBLE"
    assert receipt["treatment_decision"]["primary_campaign_poolable"] is True
    # Eligibility is evidence, not blanket launch authority.
    assert receipt["scored_launch_authorized"] is False


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda api, runtime: api["components"]["schemas"].clear(),
            "managed_jobs_create_schema_missing",
        ),
        (
            lambda api, runtime: runtime["models"].pop("glm-5.3"),
            "runtime_observation_missing:glm-5.3",
        ),
        (
            lambda api, runtime: runtime["models"]["qwen3.8-27b"].pop("managed_model_id"),
            "runtime_observation_incomplete:qwen3.8-27b:managed_model_id",
        ),
    ],
)
def test_incomplete_authority_is_rejected(mutation, error: str) -> None:
    api = copy.deepcopy(_openapi())
    runtime = copy.deepcopy(_runtime())
    mutation(api, runtime)

    with pytest.raises(managed.ManagedRouteParityError, match=error):
        managed.assess(api, runtime)
