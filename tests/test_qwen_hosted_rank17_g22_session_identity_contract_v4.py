from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_rank17_g22_session_identity_contract_v4 as contract

ROOT = Path(__file__).parents[1]


def test_contract_is_exact_self_digested_and_held() -> None:
    actual = json.loads((ROOT / contract.HELD_PATH).read_text())
    contract.validate(actual)
    assert actual["status"] == "HELD_BLOCKED_METADATA_ONLY_SESSION_IDENTITY_API"
    assert actual["launch_authorized"] is False
    assert actual["scoring_authorized"] is False
    assert actual["rank17_ledger_reclassified"] is False
    assert actual["reserved_uncreated_v4_identity"]["created"] is False
    assert actual["v3_privacy_caveat"] == {
        "session_list_route_exposed_verifier_execution": True,
        "v3_json_decoder_materialized_full_session_rows": True,
        "protected_value_persisted_or_emitted": False,
        "protected_value_inspected_by_human": False,
        "v3_score_blind_release_claim_retracted": True,
        "v3_result_not_usable_as_release_authority": True,
    }


def test_public_source_cannot_resolve_exact_session_version_score_blind() -> None:
    actual = contract.held_contract()
    source = actual["reviewed_public_source"]
    assert source["session_task_version_id_exposed"] is False
    assert source["generic_session_detail_route_exposed"] is False
    assert source["verifier_execution_summary_exposed"] is True
    assert source["task_response_exact_eval_task_id_exposed"] is False
    assert source["task_response_exact_task_version_id_exposed"] is False
    assert source["task_response_contains_protected_task_or_verifier_fields"] is True


def test_unblock_contract_forbids_all_protected_surfaces() -> None:
    unblock = contract.held_contract()["required_unblock_contract"]
    assert unblock["metadata_only"] is True
    assert unblock["list_detail_identity_agreement_required"] is True
    assert unblock["missing_conflicting_or_extra_detail_fails_closed"] is True
    assert {
        "prompt",
        "transcript",
        "reference_traces",
        "tool_use_workflow",
        "verifier",
        "verifier_execution",
        "score",
        "flag",
    } == set(unblock["forbidden_fields"])


def test_rehashed_relaxation_is_rejected() -> None:
    for mutate in (
        lambda value: value.__setitem__("launch_authorized", True),
        lambda value: value.__setitem__("rank17_ledger_reclassified", True),
        lambda value: value["required_unblock_contract"].__setitem__(
            "metadata_only", False
        ),
        lambda value: value["privacy"].__setitem__("scores_read", True),
    ):
        changed = copy.deepcopy(contract.held_contract())
        mutate(changed)
        changed["receipt_sha256"] = contract.self_hosted.digest_without(
            changed, "receipt_sha256"
        )
        with pytest.raises(ValueError, match="contract drifted"):
            contract.validate(changed)
