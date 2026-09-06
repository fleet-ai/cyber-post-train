from __future__ import annotations

import copy
import shlex
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp8_early_qualification_v1 as early
from evals.fleet import qwen38_dp8_metric_observer_v2 as observer

ROOT = Path(__file__).resolve().parents[1]


def _metrics(*, offset: int = 0, family: str = "sglang_requests_total") -> str:
    return "\n".join(
        f'{family}{{dp_rank="{rank}"}} {rank + offset}' for rank in range(8)
    )


def test_request_parser_requires_one_complete_eight_rank_family() -> None:
    assert observer.request_counters(_metrics(offset=2)) == list(range(2, 10))
    with pytest.raises(ValueError):
        observer.request_counters("health 1")
    with pytest.raises(ValueError):
        observer.request_counters(_metrics() + "\n" + _metrics(family="other_requests_total"))


def test_real_counter_increase_is_the_only_traffic_observation() -> None:
    before = [0] * 8
    assert (
        observer.observation(
            before,
            before,
            memory_mib=[250_000] * 8,
            utilization_percent=[0] * 8,
            server_run_dir="/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-c-v1",
            pod_name="example-head",
            observed_at_epoch=1,
        )
        is None
    )
    receipt = observer.observation(
        before,
        [1] + [0] * 7,
        memory_mib=[250_000] * 8,
        utilization_percent=[100] + [0] * 7,
        server_run_dir="/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-c-v1",
        pod_name="example-head",
        observed_at_epoch=2,
    )
    assert receipt is not None
    assert receipt["status"] == "REAL_REQUEST_COUNTER_INCREASED"
    assert receipt["request_deltas_by_rank"] == [1] + [0] * 7
    assert receipt["prompts_traces_flags_or_scores_included"] is False


def test_counter_decrease_and_invalid_gpu_samples_fail_closed() -> None:
    with pytest.raises(ValueError):
        observer.observation(
            [1] + [0] * 7,
            [0] * 8,
            memory_mib=[250_000] * 8,
            utilization_percent=[0] * 8,
            server_run_dir="/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-c-v1",
            pod_name="example-head",
            observed_at_epoch=3,
        )
    rows = "\n".join(f"{rank}, 250000, 0" for rank in range(8))
    memory, utilization = observer.gpu_sample(rows)
    assert memory == [250_000] * 8
    assert utilization == [0] * 8
    with pytest.raises(ValueError):
        observer.gpu_sample("\n".join(rows.splitlines()[:-1]))


def test_receipt_digest_changes_if_safe_evidence_is_tampered() -> None:
    receipt = observer.observation(
        [0] * 8,
        [1] * 8,
        memory_mib=[250_000] * 8,
        utilization_percent=[100] * 8,
        server_run_dir="/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp8-c-v1",
        pod_name="example-head",
        observed_at_epoch=4,
    )
    assert receipt is not None
    changed = copy.deepcopy(receipt)
    changed["request_deltas_by_rank"][0] = 2
    assert changed != receipt


def test_lifecycle_refreshes_idle_only_via_real_counter_observer() -> None:
    lifecycle = (ROOT / early.LIFECYCLE_V2_PATH).read_text()
    assert 'python3 "$QWEN38_OBSERVER_SCRIPT"' in lifecycle
    assert '--traffic-path "$TRAFFIC_FILE"' in lifecycle
    assert 'urlopen("http://127.0.0.1:8000/health"' in lifecycle
    assert 'touch "$TRAFFIC_FILE"' not in lifecycle
    payload = early.jobs_payload(ROOT)
    assert str(early.RUNTIME_OBSERVER_PATH) in payload["command"]
    assert observer.__file__ is not None
    bootstrap = shlex.split(payload["command"])[2]
    assert repr((ROOT / early.OBSERVER_V2_PATH).read_text()) in bootstrap
