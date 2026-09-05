from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-04-opencode-compaction-autocontinue-audit-v1.json"
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _digest_without(value: dict[str, object], field: str) -> str:
    body = {key: item for key, item in value.items() if key != field}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def test_compaction_audit_receipt_is_self_consistent_and_score_blind() -> None:
    receipt = json.loads(RECEIPT.read_text(), object_pairs_hook=_strict_object)
    assert receipt["receipt_sha256"] == _digest_without(receipt, "receipt_sha256")

    repository = receipt["repository_provenance"]
    assert repository["origin_main_contains_merge_commit"] is False
    assert repository["merge_commit"] == "0bc8dbc0cf7c50554277d4bcc27dc3bc1683a93a"

    fixed = receipt["fixed_treatment"]
    assert fixed["policy_name"] == "opencode_1.18.27_native_compaction_autocontinue_v1"
    assert fixed["effective_compaction_threshold_tokens"] == 209376
    assert fixed["autocontinue_disable_plugin_present"] is False
    assert fixed["caps"]["native_opencode_finite_agent_step_cap"] is None
    assert fixed["caps"]["proxy_hard_request_cap"] == 630

    smoke = receipt["score_free_smoke"]
    assert smoke["merge_ci"]["conclusion"] == "success"
    assert smoke["supplemental_operator_attestation"]["tests_passed"] == 27
    assert smoke["design"]["fleet_api_contacted"] is False
    assert smoke["design"]["scoring_executed"] is False

    historical = receipt["historical_claim_audit"]
    raw = historical["raw_authoritative_accepted_markers"]
    reconciled = historical["append_only_reconciled_completed_exit1"]
    unique = historical["unique_accepted_or_reconciled"]
    assert raw["qwen3.8-27b"] + reconciled["qwen3.8-27b"] == unique["qwen3.8-27b"] == 28
    assert raw["glm-5.3"] + reconciled["glm-5.3"] == unique["glm-5.3"] == 44
    assert unique["total"] == 72
    assert historical["reported_74_explanation"]["reported"]["total"] == 74
    assert historical["compaction"]["unique_accepted_or_reconciled"]["total"] == 23
    assert historical["compaction"]["reported"]["total"] == 24
    assert historical["hard_limit"]["attempts_reaching_context_window"] == 0

    privacy = receipt["privacy"]
    assert all(
        privacy[field] is False
        for field in (
            "prompts_read_or_included",
            "message_or_part_content_read_or_included",
            "tool_arguments_read_or_included",
            "scores_read_or_included",
            "flags_read_or_included",
            "traces_read_or_included",
        )
    )
    assert receipt["scientific_disposition"]["new_scored_launch_authorized"] is False
