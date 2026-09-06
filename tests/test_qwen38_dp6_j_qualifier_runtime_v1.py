from __future__ import annotations

from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_j_qualifier_runtime_v1 as runtime


def test_resource_sample_gate_replays_h_startup_race_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    expected = {
        "observed_at_epoch": 1,
        "oom_kill": 0,
        "nr_throttled": 0,
        "throttled_usec": 0,
    }

    def sample() -> dict[str, int]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("dind resource observer unavailable")
        return expected

    monkeypatch.setattr(runtime, "_dind_resource_sample", sample)
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)
    assert runtime._wait_dind_resource_sample() == expected  # noqa: SLF001
    assert attempts == 3


def test_resource_sample_gate_times_out_before_any_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter((0.0, 0.0, 31.0))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        runtime,
        "_dind_resource_sample",
        lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)
    with pytest.raises(runtime.PreRequestResourceSampleError):
        runtime._wait_dind_resource_sample()  # noqa: SLF001


def test_pre_request_resource_failure_is_classified_without_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"receipt_sha256": "sha256:" + "a" * 64}
    monkeypatch.setattr(runtime, "validate_plan", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "observe_wave",
        lambda *_args: (_ for _ in ()).throw(
            runtime.PreRequestResourceSampleError("classified")
        ),
    )
    value = runtime.run(plan, {}, Path("."))
    row = value["levels"][0]
    assert row["status"] == "FAILED"
    assert row["failure_stage"] == "pre_request_resource_sample"
    assert row["error_type"] == "PreRequestResourceSampleError"
    assert row["completed_stream_count"] == row["observed_model_request_count"] == 0
    assert value["scored_calls"] == 0
    assert value["prompts_traces_flags_or_scores_included"] is False


def test_failure_metadata_rejects_unstructured_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"receipt_sha256": "sha256:" + "a" * 64}
    monkeypatch.setattr(runtime, "validate_plan", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "observe_wave",
        lambda *_args: (_ for _ in ()).throw(
            runtime.PreRequestResourceSampleError("classified")
        ),
    )
    value = runtime.run(plan, {}, Path("."))
    value["levels"][0]["error_type"] = "secret-like unstructured text"
    value["receipt_sha256"] = runtime._digest(value)  # noqa: SLF001
    with pytest.raises(ValueError, match="not sanitized"):
        runtime.validate_result(value, plan, Path("."))
