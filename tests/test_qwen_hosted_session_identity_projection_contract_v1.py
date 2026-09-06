from __future__ import annotations

import copy

import pytest

from evals.fleet import qwen_hosted_session_identity_projection_contract_v1 as contract


def test_contract_is_exact_self_digested_and_held() -> None:
    value = contract.contract()
    contract.validate(value)
    assert value["status"] == "HELD_BLOCKED_UNDEPLOYED_METADATA_CONTRACT"
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["deployment_gate"]["merge_or_deploy_authorized"] is False
    assert value["authoritative_tally"] == {
        "accepted": 51,
        "active": 0,
        "blocked_nonrepeatable": 9,
        "unstarted": 340,
    }


def test_no_current_route_has_both_required_identity_legs_score_blind() -> None:
    routes = contract.contract()["reviewed_theseus"]["source_route_matrix"]
    for route in routes.values():
        assert not (
            route.get("exact_task_version") is True
            and route.get("exact_model_identity") is True
            and not any(key.startswith("contains_protected") for key in route)
        )


def test_projection_is_metadata_only_and_fail_closed() -> None:
    projection = contract.contract()["required_projection"]
    assert projection["team_scoped"] is True
    assert projection["requires_exactly_one_task_selector"] is True
    assert projection["model_resolution"] == {
        "source": "sessions.model_identity write-once server provenance",
        "provider_and_model_required": True,
        "missing_or_invalid_is_ambiguous": True,
        "mutable_provider_catalog_join_forbidden": True,
        "caller_metadata_identity_forbidden": True,
        "historical_dropped_trace_ingest_model_remains_ambiguous": True,
    }
    assert projection["pagination"] == {
        "limit_max": 500,
        "includes_archived_sessions": True,
        "stable_order": ["created_at_desc", "session_id_desc"],
        "cursor": "immutable_created_at_session_id_keyset",
        "snapshot_head_returned": True,
        "has_more_probe": "limit_plus_one",
    }
    assert set(projection["fields"]).isdisjoint(projection["forbidden_fields"])
    assert {
        "prompt",
        "transcript",
        "reference_traces",
        "verifier_execution",
        "score",
        "flag",
        "metadata",
        "workflow_input_json",
        "cell_id",
        "execution_id",
        "run_id",
    } <= set(projection["forbidden_fields"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("launch_authorized", True),
        lambda value: value["authoritative_tally"].__setitem__("active", 1),
        lambda value: value["required_projection"]["fields"].append("score"),
        lambda value: value["required_projection"]["model_resolution"].__setitem__(
            "mutable_provider_catalog_join_forbidden", False
        ),
        lambda value: value["required_projection"]["pagination"].__setitem__(
            "includes_archived_sessions", False
        ),
        lambda value: value["deployment_gate"].__setitem__(
            "merge_or_deploy_authorized", True
        ),
        lambda value: value.__setitem__("unexpected", True),
    ],
)
def test_rehashed_mutations_remain_rejected(mutate) -> None:
    value = copy.deepcopy(contract.contract())
    mutate(value)
    value["receipt_sha256"] = contract.self_hosted.digest_without(
        value, "receipt_sha256"
    )
    with pytest.raises(ValueError, match="contract drifted"):
        contract.validate(value)
