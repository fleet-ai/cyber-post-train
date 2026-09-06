from __future__ import annotations

import json
from pathlib import Path

from evals.fleet import qwen38_dp6_metric_observer_v3 as observer
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
INCIDENT = ROOT / (
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-c-v1-metric-schema-terminal.json"
)


def test_schema_observation_contains_shape_but_never_values() -> None:
    metrics = "\n".join(
        [
            '# HELP sglang:max_total_num_tokens Maximum total tokens',
            'sglang:max_total_num_tokens{dp_rank="0",model_name="qwen3.8-27b"} 1024',
            'sglang:max_total_num_tokens{dp_rank="1",model_name="qwen3.8-27b"} 2048',
            'sglang:num_requests_total{dp_rank="0",is_streaming="false"} 7',
        ]
    )
    value = observer.schema_observation(metrics, observed_at_epoch=123)
    assert value["receipt_sha256"] == observer._digest(value)  # noqa: SLF001
    assert value["metric_values_included"] is False
    assert value["warmup_attempt"] == 1
    assert value["request_or_response_bodies_included"] is False
    assert "1024" not in str(value)
    assert "2048" not in str(value)
    assert " 7" not in str(value)
    anchor = value["target_families"][observer.prior.STARTUP_ANCHOR_FAMILY]
    assert anchor == {
        "raw_family_line_count": 2,
        "parsed_sample_count": 2,
        "label_key_sets": [["dp_rank", "model_name"]],
    }
    requests = value["target_families"][observer.prior.REQUEST_FAMILY]
    assert requests == {
        "raw_family_line_count": 1,
        "parsed_sample_count": 1,
        "label_key_sets": [["dp_rank", "is_streaming"]],
    }


def test_schema_observation_classifies_unranked_live_shape() -> None:
    metrics = "\n".join(
        [
            "sglang:max_total_num_tokens{engine_type=\"unified\","
            "model_name=\"qwen3.8-27b\",moe_ep_rank=\"0\",pp_rank=\"0\","
            "tp_rank=\"0\"} 1",
            "sglang:num_requests_total{engine_type=\"unified\","
            "is_streaming=\"false\",model_name=\"qwen3.8-27b\"} 0",
            "sglang:num_requests_total{engine_type=\"unified\","
            "is_streaming=\"true\",model_name=\"qwen3.8-27b\"} 0",
        ]
    )
    value = observer.schema_observation(metrics, observed_at_epoch=456)
    assert value["target_families"][observer.prior.STARTUP_ANCHOR_FAMILY] == {
        "raw_family_line_count": 1,
        "parsed_sample_count": 1,
        "label_key_sets": [
            ["engine_type", "model_name", "moe_ep_rank", "pp_rank", "tp_rank"]
        ],
    }
    assert value["target_families"][observer.prior.REQUEST_FAMILY] == {
        "raw_family_line_count": 2,
        "parsed_sample_count": 2,
        "label_key_sets": [["engine_type", "is_streaming", "model_name"]],
    }


def test_bounded_warmup_allows_ranks_to_appear_incrementally(tmp_path) -> None:
    snapshots = []
    for count in (1, 3, 6):
        snapshots.append(
            "\n".join(
                f'{observer.prior.STARTUP_ANCHOR_FAMILY}{{dp_rank="{rank}"}} 1024'
                for rank in range(count)
            )
        )
    calls = 0

    def fetch() -> str:
        nonlocal calls
        value = snapshots[calls]
        calls += 1
        return value

    clock = iter((0.0, 1.0, 2.0))
    path = tmp_path / "METRIC-SCHEMA.json"
    assert observer.wait_for_request_counters(
        fetch, path, monotonic=lambda: next(clock), sleep=lambda _: None
    ) == [0] * 6
    assert calls == 3
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "CONVERGED"
    assert receipt["warmup_attempt"] == 3
    assert receipt["metric_values_included"] is False


def test_schema_warmup_never_refreshes_real_traffic_marker(tmp_path) -> None:
    schema_path = tmp_path / "METRIC-SCHEMA.json"
    traffic_path = tmp_path / "traffic"
    clock = iter((0.0, 31.0))
    try:
        observer.wait_for_request_counters(
            lambda: "",
            schema_path,
            monotonic=lambda: next(clock),
            sleep=lambda _: None,
        )
    except observer.MetricSchemaWarmupTimeout:
        pass
    else:
        raise AssertionError("incomplete schema unexpectedly converged")
    assert not traffic_path.exists()
    receipt = json.loads(schema_path.read_text())
    assert receipt["status"] == "INCOMPLETE_DURING_BOUNDED_WARMUP"
    assert receipt["metric_values_included"] is False


def test_c_v1_incident_is_zero_request_and_closes_launch() -> None:
    value = json.loads(INCIDENT.read_text())
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )
    assert value["status"] == "INFRASTRUCTURE_INVALID_ZERO_REQUEST_METRIC_SCHEMA"
    assert value["side_effects"] == {
        "instance_calls": 0,
        "model_requests": 0,
        "scoring_calls": 0,
        "session_calls": 0,
        "statistical_cells_selected": 0,
        "task_calls": 0,
        "verifier_calls": 0,
    }
    assert value["successor_gate"]["launch_authorized"] is False
    assert value["successor_gate"]["scoring_authorized"] is False
    assert value["privacy"]["prompts_traces_flags_or_scores_included"] is False
