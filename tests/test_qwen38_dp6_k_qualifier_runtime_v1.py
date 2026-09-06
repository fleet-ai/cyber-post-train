from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_k_qualifier_package_v1 as package
from evals.fleet import qwen38_dp6_k_qualifier_runtime_v1 as runtime


def test_cgroup_leaf_producer_to_shared_consumer_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cgroup_root = tmp_path / "sys-fs-cgroup"
    cgroup_leaf = cgroup_root / "kubepods.slice" / "pod.slice" / "container.scope"
    cgroup_leaf.mkdir(parents=True)
    (cgroup_leaf / "memory.events").write_text("low 0\nhigh 0\noom 0\noom_kill 0\n")
    (cgroup_leaf / "cpu.stat").write_text(
        "usage_usec 100\nnr_periods 2\nnr_throttled 3\nthrottled_usec 4\n"
    )
    membership = tmp_path / "proc-self-cgroup"
    membership.write_text("0::/kubepods.slice/pod.slice/container.scope\n")
    sample = tmp_path / "workspace" / "dind-resource-samples.tsv"
    ready = tmp_path / "workspace" / "dind-resource-producer.ready"
    sample.parent.mkdir()
    script = package.resource_sampler_script(
        sample_path=str(sample),
        ready_path=str(ready),
        cgroup_membership_path=str(membership),
        cgroup_root=str(cgroup_root),
        one_shot=True,
    )
    subprocess.run(["sh", "-ceu", "--", script], check=True)
    monkeypatch.setattr(runtime, "DIND_RESOURCE_SAMPLES_PATH", sample)
    monkeypatch.setattr(runtime, "DIND_RESOURCE_READY_PATH", ready)
    value = runtime._dind_resource_sample()  # noqa: SLF001
    assert value["oom_kill"] == 0
    assert value["nr_throttled"] == 3
    assert value["throttled_usec"] == 4
    assert "/sys/fs/cgroup/memory.events" not in script
    assert str(cgroup_leaf / "memory.events") not in script


def test_resource_consumer_rejects_missing_ready_and_stale_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = tmp_path / "samples.tsv"
    ready = tmp_path / "ready"
    sample.write_text("1 0 0 0\n")
    monkeypatch.setattr(runtime, "DIND_RESOURCE_SAMPLES_PATH", sample)
    monkeypatch.setattr(runtime, "DIND_RESOURCE_READY_PATH", ready)
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime._dind_resource_sample()  # noqa: SLF001
    ready.write_bytes(runtime.DIND_RESOURCE_READY_BYTES)
    with pytest.raises(RuntimeError, match="stale"):
        runtime._dind_resource_sample()  # noqa: SLF001


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


def test_resource_control_waits_for_a_strictly_newer_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter(
        (
            {"observed_at_epoch": 10, "oom_kill": 0, "nr_throttled": 0, "throttled_usec": 0},
            {"observed_at_epoch": 11, "oom_kill": 0, "nr_throttled": 0, "throttled_usec": 0},
        )
    )
    times = iter((0.0, 0.0, 0.1))
    monkeypatch.setattr(runtime, "_dind_resource_sample", lambda: next(values))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(runtime.time, "sleep", lambda _seconds: None)
    value = runtime._wait_newer_dind_resource_sample(10)  # noqa: SLF001
    assert value["observed_at_epoch"] == 11


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
