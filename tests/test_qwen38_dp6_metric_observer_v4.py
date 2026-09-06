from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_metric_observer_v4 as observer
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
INCIDENT = ROOT / (
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-d-v1-v6-metric-schema-terminal.json"
)
HELD = ROOT / (
    "docs/evidence/qwen38-study/2026-09-06-qwen38-dp6-observer-v4-held-successor-v1.json"
)
IDLE_RELEASE = ROOT / ("docs/evidence/qwen38-study/2026-09-06-qwen38-tp1-j-v1-idle-release.json")


def _metrics(*, false: int = 0, true: int | None = None) -> str:
    rows = [
        "# HELP sglang:max_total_num_tokens Maximum total tokens",
        'unrelated_family{arbitrary="label"} 99',
    ]
    rows.extend(
        "sglang:max_total_num_tokens{"
        f'dp_rank="{rank}",engine_type="unified",model_name="qwen3.8-27b",'
        'moe_ep_rank="0",pp_rank="0",tp_rank="0"} 262144'
        for rank in range(6)
    )
    for family in (observer.RUNNING_FAMILY, observer.QUEUE_FAMILY):
        rows.extend(
            f'{family}{{dp_rank="{rank}",engine_type="unified",'
            'model_name="qwen3.8-27b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 0'
            for rank in range(6)
        )
    rows.append(
        'sglang:num_requests_total{engine_type="unified",'
        f'is_streaming="false",model_name="qwen3.8-27b"}} {false}'
    )
    if true is not None:
        rows.append(
            'sglang:num_requests_total{engine_type="unified",'
            f'is_streaming="true",model_name="qwen3.8-27b"}} {true}'
        )
    return "\n".join(rows)


def _identity() -> dict[str, object]:
    return {
        "server_run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-e-v1",
        "pod_name": "head-pod",
        "pod_uid": "11111111-1111-4111-8111-111111111111",
        "api_run_id": "ft-run-example",
        "service_uid": "22222222-2222-4222-8222-222222222222",
        "server_binding_receipt_sha256": "sha256:" + "a" * 64,
        "observed_at_epoch": 123,
    }


def test_live_global_counter_partitions_are_summed() -> None:
    assert observer._global_request_total(_metrics(false=7, true=3)) == 10  # noqa: SLF001


def test_live_one_series_zero_schema_is_valid() -> None:
    assert observer._global_request_total(_metrics()) == 0  # noqa: SLF001


def test_schema_receipt_covers_all_activity_families_without_values() -> None:
    value = observer.schema_observation(
        _metrics(false=123, true=45),
        observed_at_epoch=1,
        warmup_attempt=1,
        validation_status="CONVERGED_ACTIVITY_SCHEMA",
    )
    assert set(value["target_families"]) == observer.TARGET_FAMILIES
    assert value["metric_values_included"] is False
    assert all(
        set(shape) == {"raw_family_line_count", "parsed_sample_count", "label_key_sets"}
        for shape in value["target_families"].values()
    )


@pytest.mark.parametrize(
    "metrics",
    [
        _metrics().replace('dp_rank="5",', ""),
        _metrics().replace('is_streaming="false",', 'dp_rank="0",is_streaming="false",'),
        _metrics() + "\n" + _metrics().splitlines()[-1],
        _metrics().replace('is_streaming="false"', 'is_streaming="maybe"'),
        _metrics().replace('engine_type="unified"', 'engine_type="other"', 1),
        _metrics().replace('model_name="qwen3.8-27b"', 'model_name="other"', 1),
        _metrics().replace('moe_ep_rank="0"', 'moe_ep_rank="1"', 1),
        _metrics() + '\nsglang:num_running_reqs{dp_rank="0" BROKEN',
    ],
)
def test_schema_drift_fails_closed(metrics: str) -> None:
    with pytest.raises(ValueError):
        observer._global_request_total(metrics)  # noqa: SLF001


def test_bounded_schema_warmup_never_refreshes_traffic(tmp_path: Path) -> None:
    snapshots = [_metrics().replace('dp_rank="5",', ""), _metrics(false=4)]
    calls = 0

    def fetch() -> str:
        nonlocal calls
        value = snapshots[calls]
        calls += 1
        return value

    clock = iter((0.0, 1.0))
    schema_path = tmp_path / "METRIC-SCHEMA.json"
    traffic_path = tmp_path / "traffic"
    assert (
        observer.wait_for_global_request_total(
            fetch, schema_path, monotonic=lambda: next(clock), sleep=lambda _: None
        )
        == 4
    )
    assert calls == 2
    assert not traffic_path.exists()
    receipt = json.loads(schema_path.read_text())
    assert receipt["status"] == "CONVERGED_ACTIVITY_SCHEMA"
    assert receipt["metric_values_included"] is False


def test_global_counter_monotonicity_and_six_device_evidence() -> None:
    value = observer.traffic_observation(
        10,
        13,
        memory_mib=[40000] * 6,
        utilization_percent=[80, 70, 60, 50, 40, 30],
        **_identity(),
    )
    assert value is not None
    assert value["global_request_delta"] == 3
    assert value["per_rank_request_attribution_claimed"] is False
    assert value["activity_reasons"] == [
        "completed_request_counter_increased",
        "gpu_utilization_positive",
    ]
    assert value["gpu_memory_used_mib_by_device"] == [40000] * 6
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    with pytest.raises(ValueError, match="decreased"):
        observer.traffic_observation(
            13,
            12,
            memory_mib=[40000] * 6,
            utilization_percent=[0] * 6,
            **_identity(),
        )


def test_stable_sample_is_not_traffic() -> None:
    assert (
        observer.traffic_observation(
            10,
            10,
            memory_mib=[40000] * 6,
            utilization_percent=[0] * 6,
            **_identity(),
        )
        is None
    )


def test_long_inflight_request_blocks_idle_release_after_600_seconds() -> None:
    assert (
        observer.idle_release_eligible(
            10,
            10,
            running_requests=1,
            queued_requests=0,
            utilization_percent=[75] * 6,
            activity_age_seconds=900,
        )
        is False
    )
    value = observer.traffic_observation(
        10,
        10,
        memory_mib=[40000] * 6,
        utilization_percent=[75] * 6,
        running_requests=1,
        queued_requests=0,
        **_identity(),
    )
    assert value is not None
    assert "running_requests_positive" in value["activity_reasons"]
    assert (
        observer.idle_release_eligible(
            10,
            10,
            running_requests=0,
            queued_requests=0,
            utilization_percent=[0] * 6,
            activity_age_seconds=600,
        )
        is True
    )


def test_incident_closes_consumed_identities_and_has_no_calls() -> None:
    value = json.loads(INCIDENT.read_text())
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["identity_disposition"] == {
        "fresh_identity_required_after_reviewed_fix": True,
        "qualifier_v6_consumed": True,
        "repeat_authorized": False,
        "scored_successor_authorized": False,
        "server_d_v1_consumed": True,
    }
    assert value["effects"] == {
        "model_requests": 0,
        "scoring_calls": 0,
        "session_calls": 0,
        "statistical_cells_selected": 0,
        "task_instance_calls": 0,
        "valid_model_outcomes": 0,
        "verifier_calls": 0,
    }


def test_held_successor_is_fresh_score_free_and_not_launchable() -> None:
    value = json.loads(HELD.read_text())
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["status"] == "HELD_PENDING_REVIEW"
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["fresh_identities"] == {
        "qualifier": "chris-cyber-q38-dp6-e-qualifier-v7",
        "server_run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-e-v1",
        "server_title": "chris-cyber-evalserve-q38-dp6-e-v1",
    }
    assert value["side_effect_budget"] == {
        "model_requests": 0,
        "scoring_calls": 0,
        "session_calls": 0,
        "statistical_cells": 0,
        "task_instance_calls": 0,
        "verifier_calls": 0,
    }


def test_idle_tp1_was_released_before_successor_work() -> None:
    value = json.loads(IDLE_RELEASE.read_text())
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["status"] == "RELEASED_IDLE_GPU_AFTER_600_SECOND_TRAFFIC_TIMEOUT"
    assert (
        value["idle_gate"]["observed_traffic_age_seconds"]
        > value["idle_gate"]["maximum_idle_seconds"]
    )
    assert value["idle_gate"]["active_chris_qwen_scored_controller_count"] == 0
    assert value["release"] == {
        "gpu_quota_released": True,
        "head_pod_absent_after": True,
        "http_status": 204,
        "jobs_api_mutations": 1,
        "rayjob_absent_after": True,
        "route": "DELETE /v1/runs/ft-run-e87e2bd4",
        "service_absent_after": True,
        "workload_absent_after": True,
    }
    assert all(value["effects"][key] == 0 for key in value["effects"])
    assert value["privacy"]["prompts_traces_flags_scores_logs_or_credentials_included"] is False
