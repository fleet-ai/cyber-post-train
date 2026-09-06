from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_metric_observer_v2 as observer
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
INCIDENT = ROOT / (
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-b-v1-baseline-deadlock-terminal.json"
)


def _anchors(*, ranks: range = range(6)) -> str:
    return "\n".join(
        f'{observer.STARTUP_ANCHOR_FAMILY}{{dp_rank="{rank}",model_name="qwen3.8-27b"}} 1024'
        for rank in ranks
    )


def test_fresh_server_uses_startup_anchor_for_zero_request_baseline() -> None:
    assert observer.request_counters(_anchors()) == [0] * 6


def test_request_counter_partitions_are_summed_per_rank() -> None:
    metrics = "\n".join(
        [
            _anchors(),
            f'{observer.REQUEST_FAMILY}{{dp_rank="0",is_streaming="true"}} 2',
            f'{observer.REQUEST_FAMILY}{{dp_rank="0",is_streaming="false"}} 3',
            f'{observer.REQUEST_FAMILY}{{dp_rank="5",is_streaming="true"}} 7',
        ]
    )
    assert observer.request_counters(metrics) == [5, 0, 0, 0, 0, 7]


def test_request_parser_fails_closed_without_complete_startup_anchor() -> None:
    with pytest.raises(ValueError, match="startup anchor"):
        observer.request_counters("")
    with pytest.raises(ValueError, match="startup anchor"):
        observer.request_counters(_anchors(ranks=range(5)))
    with pytest.raises(ValueError, match="duplicated"):
        observer.request_counters(
            "\n".join(
                [
                    _anchors(),
                    f'{observer.REQUEST_FAMILY}{{dp_rank="0",is_streaming="true"}} 1',
                    f'{observer.REQUEST_FAMILY}{{dp_rank="0",is_streaming="true"}} 1',
                ]
            )
        )


def test_observer_error_status_is_classified_digest_valid_and_content_free() -> None:
    receipt = observer.observer_status(
        "FAILED", "metric_schema", observed_at_epoch=123, error_type="ValueError"
    )
    claimed = receipt.pop("receipt_sha256")
    import hashlib

    assert (
        claimed
        == "sha256:"
        + hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert receipt["phase"] == "metric_schema"
    assert receipt["error_type"] == "ValueError"
    assert receipt["request_or_response_bodies_included"] is False
    assert receipt["prompts_traces_flags_or_scores_included"] is False
    assert "message" not in receipt


def test_terminal_incident_is_digest_valid_and_keeps_successor_closed() -> None:
    receipt = json.loads(INCIDENT.read_text())
    assert receipt["receipt_sha256"] == self_hosted.digest_without(receipt, "receipt_sha256")
    assert receipt["side_effects"] == {
        "model_requests": 0,
        "task_calls": 0,
        "instance_calls": 0,
        "session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
    }
    assert receipt["successor_gate"]["fresh_server_and_qualifier_identities_required"]
    assert receipt["successor_gate"]["launch_authorized"] is False
    assert receipt["successor_gate"]["scoring_authorized"] is False
