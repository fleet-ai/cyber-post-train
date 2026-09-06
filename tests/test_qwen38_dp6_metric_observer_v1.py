from __future__ import annotations

import copy
import shlex
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_early_qualification_v1 as early
from evals.fleet import qwen38_dp6_metric_observer_v1 as observer

ROOT = Path(__file__).resolve().parents[1]
BINDING = {
    "api_run_id": "ft-run-example",
    "head_pod_name": "example-head",
    "head_pod_uid": "11111111-1111-4111-8111-111111111111",
    "service_uid": "22222222-2222-4222-8222-222222222222",
    "receipt_sha256": "sha256:" + "a" * 64,
}


def _metrics(*, offset: int = 0, family: str = "sglang_requests_total") -> str:
    return "\n".join(
        f'{family}{{dp_rank="{rank}"}} {rank + offset}' for rank in range(6)
    )


def test_request_parser_requires_one_complete_six_rank_family() -> None:
    assert observer.request_counters(_metrics(offset=2)) == list(range(2, 8))
    with pytest.raises(ValueError):
        observer.request_counters("health 1")
    with pytest.raises(ValueError):
        observer.request_counters(_metrics() + "\n" + _metrics(family="other_requests_total"))


def test_real_counter_increase_is_the_only_traffic_observation() -> None:
    before = [0] * 6
    assert (
        observer.observation(
            before,
            before,
            memory_mib=[250_000] * 6,
            utilization_percent=[0] * 6,
            server_run_dir=early.RUN_DIR,
            pod_name="example-head",
            pod_uid=BINDING["head_pod_uid"],
            api_run_id=BINDING["api_run_id"],
            service_uid=BINDING["service_uid"],
            server_binding_receipt_sha256=BINDING["receipt_sha256"],
            observed_at_epoch=1,
        )
        is None
    )
    receipt = observer.observation(
        before,
        [1] + [0] * 5,
        memory_mib=[250_000] * 6,
        utilization_percent=[100] + [0] * 5,
        server_run_dir=early.RUN_DIR,
        pod_name="example-head",
        pod_uid=BINDING["head_pod_uid"],
        api_run_id=BINDING["api_run_id"],
        service_uid=BINDING["service_uid"],
        server_binding_receipt_sha256=BINDING["receipt_sha256"],
        observed_at_epoch=2,
    )
    assert receipt is not None
    assert receipt["status"] == "REAL_REQUEST_COUNTER_INCREASED"
    assert receipt["request_deltas_by_rank"] == [1] + [0] * 5
    assert receipt["prompts_traces_flags_or_scores_included"] is False


def test_counter_decrease_and_invalid_gpu_samples_fail_closed() -> None:
    with pytest.raises(ValueError):
        observer.observation(
            [1] + [0] * 5,
            [0] * 6,
            memory_mib=[250_000] * 6,
            utilization_percent=[0] * 6,
            server_run_dir=early.RUN_DIR,
            pod_name="example-head",
            pod_uid=BINDING["head_pod_uid"],
            api_run_id=BINDING["api_run_id"],
            service_uid=BINDING["service_uid"],
            server_binding_receipt_sha256=BINDING["receipt_sha256"],
            observed_at_epoch=3,
        )
    rows = "\n".join(f"{rank}, 250000, 0" for rank in range(6))
    memory, utilization = observer.gpu_sample(rows)
    assert memory == [250_000] * 6
    assert utilization == [0] * 6
    with pytest.raises(ValueError):
        observer.gpu_sample("\n".join(rows.splitlines()[:-1]))


def test_receipt_digest_changes_if_safe_evidence_is_tampered() -> None:
    receipt = observer.observation(
        [0] * 6,
        [1] * 6,
        memory_mib=[250_000] * 6,
        utilization_percent=[100] * 6,
        server_run_dir=early.RUN_DIR,
        pod_name="example-head",
        pod_uid=BINDING["head_pod_uid"],
        api_run_id=BINDING["api_run_id"],
        service_uid=BINDING["service_uid"],
        server_binding_receipt_sha256=BINDING["receipt_sha256"],
        observed_at_epoch=4,
    )
    assert receipt is not None
    changed = copy.deepcopy(receipt)
    changed["request_deltas_by_rank"][0] = 2
    assert changed != receipt


def test_stable_bound_baseline_is_score_free_and_not_traffic() -> None:
    receipt = observer.baseline_observation(
        list(range(6)),
        server_run_dir=early.RUN_DIR,
        pod_name="example-head",
        pod_uid=BINDING["head_pod_uid"],
        api_run_id=BINDING["api_run_id"],
        service_uid=BINDING["service_uid"],
        server_binding_receipt_sha256=BINDING["receipt_sha256"],
        observed_at_epoch=5,
    )
    assert receipt["status"] == "STABLE_BOUND_COUNTER_BASELINE"
    assert receipt["request_counters_by_rank"] == list(range(6))
    assert receipt["request_delta_since_prior_sample"] == 0
    assert receipt["traffic_refresh_performed"] is False


def test_lifecycle_refreshes_idle_only_via_real_counter_observer() -> None:
    lifecycle = (ROOT / early.LIFECYCLE_V2_PATH).read_text()
    assert 'python3 "$QWEN38_DP6_OBSERVER_SCRIPT"' in lifecycle
    assert '--traffic-path "$TRAFFIC_FILE"' in lifecycle
    assert '--binding-path "$SERVER_BINDING"' in lifecycle
    assert '--baseline-path "$COUNTER_BASELINE"' in lifecycle
    assert '--event-dir "$TRAFFIC_EVENT_DIR"' in lifecycle
    assert 'urlopen("http://127.0.0.1:8000/health"' in lifecycle
    assert 'touch "$TRAFFIC_FILE"' not in lifecycle
    source = (ROOT / early.OBSERVER_V2_PATH).read_text()
    assert 'STARTUP_ANCHOR_FAMILY = "sglang:max_total_num_tokens"' in source
    assert "absent request series are genuine zeros" in source
    assert 'parser.add_argument("--status-path"' in source
    payload = early.jobs_payload(ROOT)
    assert str(early.RUNTIME_OBSERVER_PATH) in payload["command"]
    assert observer.__file__ is not None
    bootstrap = shlex.split(payload["command"])[2]
    assert repr((ROOT / early.OBSERVER_V2_PATH).read_text()) in bootstrap
