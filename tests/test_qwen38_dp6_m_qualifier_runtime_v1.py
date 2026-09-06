from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
import tarfile
import urllib.error
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp6_m_qualifier_package_v1 as package
from evals.fleet import qwen38_dp6_m_qualifier_runtime_v1 as runtime
from evals.fleet import qwen38_dp6_metric_observer_v5 as observer


def test_runtime_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"safe","status":"SECRET_PROTECTED_VALUE"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runtime._load(path)  # noqa: SLF001


def test_qualifier_archive_imports_in_isolated_materialization(tmp_path: Path) -> None:
    archive = package.archive_bytes(Path(__file__).resolve().parents[1])
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        bundle.extractall(tmp_path, filter="data")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "import evals.fleet.qwen38_dp6_m_qualifier_runtime_v1 as runtime; "
                "assert runtime.RESULT_SCHEMA == "
                "'fleet-qwen38-dp6-m-qualification-result-v1'"
            ),
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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
    assert row["failure_code"] == "pre_request_resource_sample_unavailable"
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
    value["levels"][0]["failure_code"] = "secret-like unstructured text"
    value["receipt_sha256"] = runtime._digest(value)  # noqa: SLF001
    with pytest.raises(ValueError, match="not sanitized"):
        runtime.validate_result(value, plan, Path("."))


def test_wave_uses_exact_before_after_metrics_across_first_request_observer_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "receipt_sha256": "sha256:" + "a" * 64,
        "service_origin": "http://server:8000",
        "server_binding": {"receipt_sha256": "sha256:" + "b" * 64},
    }
    rows = [{"receipt_sha256": f"sha256:{index:064x}"} for index in range(1, 5)]
    totals = iter((0, 4))
    event = {
        "observed_at_epoch": 10,
        "receipt_sha256": "sha256:" + "c" * 64,
        # The asynchronous sampler missed request 0 -> 1 and only saw 1 -> 4.
        "global_request_total_before": 1,
        "global_request_total_after": 4,
        "global_request_delta": 3,
        "gpu_utilization_percent_by_device": [0] * 6,
        "gpu_memory_used_mib_by_device": [100] * 6,
    }
    samples = iter(
        (
            {"observed_at_epoch": 1, "oom_kill": 0, "nr_throttled": 0, "throttled_usec": 0},
            {"observed_at_epoch": 2, "oom_kill": 0, "nr_throttled": 0, "throttled_usec": 0},
        )
    )
    monkeypatch.setattr(runtime, "COUNTER_SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(runtime, "EVENT_DIR", tmp_path / "events")
    monkeypatch.setattr(runtime, "_stable_counter_baseline", lambda _binding: {"frozen": 0})
    monkeypatch.setattr(runtime, "_fetch_global_request_total", lambda _origin: next(totals))
    monkeypatch.setattr(runtime, "_wait_dind_resource_sample", lambda: next(samples))
    monkeypatch.setattr(runtime, "_wait_newer_dind_resource_sample", lambda _epoch: next(samples))
    monkeypatch.setattr(runtime, "_new_events", lambda *_args: [event])
    observed_rows, distribution, resource, baseline = runtime.observe_wave(
        4, lambda: rows, plan, {}
    )
    assert observed_rows == [runtime._failed_stream("probe_evidence_invalid")] * 4  # noqa: SLF001
    assert baseline == {"frozen": 0}
    assert resource["status"] == "PASSED_NO_CONTROLLER_RESOURCE_ERROR"
    assert distribution["global_request_total_before"] == 0
    assert distribution["global_request_total_after"] == 4
    assert distribution["global_request_delta"] == 4
    assert distribution["gpu_peak_utilization_percent_by_device"] == [0] * 6
    runtime.validate_distribution_receipt(distribution, plan, 4, observed_rows)
    before_snapshot = json.loads((tmp_path / "snapshots/c4-before.json").read_text())
    after_snapshot = json.loads((tmp_path / "snapshots/c4-after.json").read_text())
    assert before_snapshot["global_request_total"] == 0
    assert after_snapshot["global_request_total"] == 4


def test_validation_exception_preserves_sanitized_wave_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"receipt_sha256": "sha256:" + "a" * 64}
    rows = [
        {
            "execution": {"model_requests": 4},
            "privacy": {"responses_or_model_outputs_included": False},
        }
    ]
    distribution = {"global_request_delta": 4, "evidence": "sanitized"}
    resource = {
        "status": "PASSED_NO_CONTROLLER_RESOURCE_ERROR",
        "receipt_sha256": "sha256:" + "b" * 64,
    }
    baseline = {"global_request_total": 0, "receipt_sha256": "sha256:" + "c" * 64}
    monkeypatch.setattr(runtime, "validate_plan", lambda *_args: None)
    monkeypatch.setattr(
        runtime, "observe_wave", lambda *_args: (rows, distribution, resource, baseline)
    )
    monkeypatch.setattr(
        runtime,
        "validate_distribution_receipt",
        lambda *_args: (_ for _ in ()).throw(ValueError("sensitive_validation_detail")),
    )
    value = runtime.run(plan, {}, Path("."))
    failed = value["levels"][0]
    assert failed["failure_code"] == "wave_distribution_invalid"
    assert failed["stream_receipts"] == [
        runtime._failed_stream("probe_evidence_invalid")  # noqa: SLF001
    ]
    assert failed["distribution_receipt"] == {}
    assert failed["controller_resource_receipt"] == {}
    assert failed["stable_counter_baseline_receipt"] == {}
    assert "sensitive_validation_detail" not in json.dumps(value)


def test_observation_exception_preserves_evidence_captured_before_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"receipt_sha256": "sha256:" + "a" * 64}
    rows = [
        {
            "execution": {"model_requests": 1},
            "privacy": {"responses_or_model_outputs_included": False},
        }
    ]
    baseline = {"global_request_total": 0, "receipt_sha256": "sha256:" + "b" * 64}

    def fail_after_capture(
        _level: int,
        _execute: object,
        _plan: object,
        _binding: object,
        evidence_sink: dict[str, object],
    ) -> object:
        evidence_sink["stable_counter_baseline_receipt"] = baseline
        evidence_sink["stream_receipts"] = rows
        raise RuntimeError("sensitive_post_request_observer_detail")

    monkeypatch.setattr(runtime, "validate_plan", lambda *_args: None)
    monkeypatch.setattr(runtime, "observe_wave", fail_after_capture)
    value = runtime.run(plan, {}, Path("."))
    failed = value["levels"][0]
    assert failed["failure_code"] == "wave_observation_unavailable"
    assert failed["stream_receipts"] == [
        runtime._failed_stream("probe_evidence_invalid")  # noqa: SLF001
    ]
    assert failed["stable_counter_baseline_receipt"] == {}
    assert "sensitive_post_request_observer_detail" not in json.dumps(value)


def test_unexpected_exception_class_text_and_hash_are_not_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {"receipt_sha256": "sha256:" + "a" * 64}
    secret = "sensitive_exception_detail"
    secret_error = type(secret, (RuntimeError,), {})
    monkeypatch.setattr(runtime, "validate_plan", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "observe_wave",
        lambda *_args: (_ for _ in ()).throw(secret_error(secret)),
    )
    value = runtime.run(plan, {}, Path("."))
    raw = json.dumps(value)
    assert secret not in raw
    assert runtime._digest({"secret": secret}) not in raw  # noqa: SLF001
    assert value["levels"][0]["failure_code"] == "wave_observation_unavailable"


def test_invalid_stream_is_projected_to_fixed_content_free_schema() -> None:
    secret = "SECRET_PROTECTED_STREAM_FIELD"
    raw = {
        "schema_version": "rehashed-invalid",
        "unexpected": {"score": 0.75, "payload": secret},
        "receipt_sha256": "sha256:" + "a" * 64,
    }
    projected = runtime._project_stream(raw, {})  # noqa: SLF001
    assert projected == runtime._failed_stream("probe_evidence_invalid")  # noqa: SLF001
    assert secret not in json.dumps(projected)
    assert set(projected) == runtime.FAILED_STREAM_KEYS


@pytest.mark.parametrize(
    "failure",
    [
        OSError("SECRET_OBSERVER_DETAIL"),
        urllib.error.URLError("SECRET_OBSERVER_DETAIL"),
        TimeoutError("SECRET_OBSERVER_DETAIL"),
    ],
)
def test_result_and_level_reject_rehashed_extra_fields(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    plan = {"receipt_sha256": "sha256:" + "a" * 64}
    monkeypatch.setattr(runtime, "validate_plan", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "observe_wave",
        lambda *_args: (_ for _ in ()).throw(failure),
    )
    valid = runtime.run(plan, {}, Path("."))
    assert "SECRET_OBSERVER_DETAIL" not in json.dumps(valid)
    runtime.validate_result(valid, plan, Path("."))

    top_extra = copy.deepcopy(valid)
    top_extra["protected"] = "SECRET"
    top_extra["receipt_sha256"] = runtime._digest(top_extra)  # noqa: SLF001
    with pytest.raises(ValueError, match="result drifted"):
        runtime.validate_result(top_extra, plan, Path("."))

    level_extra = copy.deepcopy(valid)
    level_extra["levels"][0]["protected"] = "SECRET"
    level_extra["receipt_sha256"] = runtime._digest(level_extra)  # noqa: SLF001
    with pytest.raises(ValueError, match="ladder order drifted"):
        runtime.validate_result(level_extra, plan, Path("."))

    scalar_content = copy.deepcopy(valid)
    scalar_content["levels"][0]["elapsed_milliseconds"] = "SECRET"
    scalar_content["receipt_sha256"] = runtime._digest(scalar_content)  # noqa: SLF001
    with pytest.raises(ValueError, match="ladder order drifted"):
        runtime.validate_result(scalar_content, plan, Path("."))

    stream_extra = copy.deepcopy(valid)
    stream_extra["levels"][0]["stream_receipts"] = [
        runtime._failed_stream("probe_evidence_invalid")  # noqa: SLF001
    ]
    stream_extra["levels"][0]["stream_receipts"][0]["protected"] = "SECRET"
    stream_extra["levels"][0]["stream_receipts"][0]["receipt_sha256"] = runtime._digest(  # noqa: SLF001
        stream_extra["levels"][0]["stream_receipts"][0]
    )
    stream_extra["receipt_sha256"] = runtime._digest(stream_extra)  # noqa: SLF001
    with pytest.raises(ValueError, match="failed stream receipt"):
        runtime.validate_result(stream_extra, plan, Path("."))


def test_nested_evidence_keysets_reject_rehashed_extras() -> None:
    plan = {
        "receipt_sha256": "sha256:" + "a" * 64,
        "service_origin": "http://server:8000",
        "server_binding": {"receipt_sha256": "sha256:" + "b" * 64},
    }
    counter = runtime._counter_snapshot("BEFORE_WAVE", 0, 1, plan)  # noqa: SLF001
    counter["protected"] = "SECRET"
    counter["receipt_sha256"] = runtime._digest(counter)  # noqa: SLF001
    with pytest.raises(ValueError, match="counter snapshot contains"):
        runtime._validate_counter_snapshot(counter, "BEFORE_WAVE", 1, plan)  # noqa: SLF001

    identity = {
        "server_run_dir": runtime.early.RUN_DIR,
        "pod_name": "pod",
        "pod_uid": "11111111-1111-4111-8111-111111111111",
        "api_run_id": "ft-run-one",
        "service_uid": "22222222-2222-4222-8222-222222222222",
        "server_binding_receipt_sha256": "sha256:" + "c" * 64,
    }
    baseline = observer.baseline_observation(0, observed_at_epoch=1, **identity)
    baseline["protected"] = "SECRET"
    baseline["receipt_sha256"] = runtime._digest(baseline)  # noqa: SLF001
    binding = {
        "api_run_id": identity["api_run_id"],
        "head_pod_name": identity["pod_name"],
        "head_pod_uid": identity["pod_uid"],
        "service_uid": identity["service_uid"],
        "server_binding_receipt_sha256": identity["server_binding_receipt_sha256"],
    }
    with pytest.raises(ValueError, match="baseline receipt contains"):
        runtime._validate_baseline_evidence(baseline, binding)  # noqa: SLF001
    with pytest.raises(ValueError, match="baseline receipt contains"):
        runtime._validate_optional_level_evidence(  # noqa: SLF001
            {
                "stream_receipts": [],
                "stable_counter_baseline_receipt": baseline,
                "distribution_receipt": {},
                "controller_resource_receipt": {},
            },
            {"server_binding": binding},
        )

    before = {"observed_at_epoch": 1, "oom_kill": 0, "nr_throttled": 0, "throttled_usec": 0}
    after = {"observed_at_epoch": 2, "oom_kill": 0, "nr_throttled": 0, "throttled_usec": 0}
    resource = runtime._dind_resource_control(before, after)  # noqa: SLF001
    resource["protected"] = "SECRET"
    resource["receipt_sha256"] = runtime._digest(resource)  # noqa: SLF001
    with pytest.raises(ValueError, match="resource receipt contains"):
        runtime._validate_resource_control(resource)  # noqa: SLF001


def test_event_rejects_rehashed_extra_or_identity_drift(tmp_path: Path) -> None:
    receipt_sha = "sha256:" + "a" * 64
    identity = {
        "server_run_dir": runtime.early.RUN_DIR,
        "pod_name": "pod",
        "pod_uid": "11111111-1111-4111-8111-111111111111",
        "api_run_id": "ft-run-one",
        "service_uid": "22222222-2222-4222-8222-222222222222",
        "server_binding_receipt_sha256": receipt_sha,
    }
    event = observer.traffic_observation(
        0,
        1,
        memory_mib=[1] * runtime.RANKS,
        utilization_percent=[1] * runtime.RANKS,
        observed_at_epoch=1,
        **identity,
    )
    assert event is not None
    event["protected"] = "SECRET"
    event["receipt_sha256"] = runtime._digest(event)  # noqa: SLF001
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event))
    binding = {
        "api_run_id": identity["api_run_id"],
        "head_pod_name": identity["pod_name"],
        "head_pod_uid": identity["pod_uid"],
        "service_uid": identity["service_uid"],
        "receipt_sha256": receipt_sha,
    }
    with pytest.raises(ValueError, match="traffic event drifted"):
        runtime._event(path, binding)  # noqa: SLF001


def test_initial_sampler_baseline_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "REQUEST-COUNTER-BASELINE.json"
    identity = {
        "server_run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-dp6-m-v1",
        "pod_name": "pod",
        "pod_uid": "11111111-1111-4111-8111-111111111111",
        "api_run_id": "ft-run-one",
        "service_uid": "22222222-2222-4222-8222-222222222222",
        "server_binding_receipt_sha256": "sha256:" + "a" * 64,
    }
    first = observer.baseline_observation(0, observed_at_epoch=1, **identity)
    later = observer.baseline_observation(4, observed_at_epoch=2, **identity)
    assert observer.preserve_initial_baseline(path, first) == first
    assert observer.preserve_initial_baseline(path, later) == first
    assert json.loads(path.read_text()) == first


def test_observer_failure_status_uses_only_fixed_code() -> None:
    value = observer.observer_status(
        "FAILED",
        "metric_schema_warmup",
        observed_at_epoch=1,
        failure_code="observer_failed",
    )
    raw = json.dumps(value)
    assert value["failure_code"] == "observer_failed"
    assert "error_type" not in value
    assert "exception" not in raw.lower()


def test_c6_still_requires_all_six_instantaneous_gpu_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {
        "receipt_sha256": "sha256:" + "a" * 64,
        "service_origin": "http://server:8000",
        "server_binding": {"receipt_sha256": "sha256:" + "b" * 64},
    }
    rows = [{"receipt_sha256": f"sha256:{index:064x}"} for index in range(1, 7)]
    before = runtime._counter_snapshot("BEFORE_WAVE", 0, 6, plan)  # noqa: SLF001
    after = runtime._counter_snapshot("AFTER_WAVE", 6, 6, plan)  # noqa: SLF001
    distribution = {
        "schema_version": runtime.DISTRIBUTION_SCHEMA,
        "status": "PASSED_NON_SCORED_DISTRIBUTION",
        "plan_receipt_sha256": plan["receipt_sha256"],
        "server_binding": plan["server_binding"],
        "service_origin": plan["service_origin"],
        "concurrency": 6,
        "stream_receipt_sha256s": [row["receipt_sha256"] for row in rows],
        "counter_before_snapshot": before,
        "counter_after_snapshot": after,
        "global_request_total_before": 0,
        "global_request_total_after": 6,
        "global_request_delta": 6,
        "per_rank_request_attribution_claimed": False,
        "gpu_peak_utilization_percent_by_device": [1, 1, 1, 1, 1, 0],
        "gpu_peak_memory_used_mib_by_device": [100] * 6,
        "gpu_device_count": 6,
        "gpu_memory_loaded_count": 6,
        "sampling_seconds": 5,
        "observer_errors": [],
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    distribution["receipt_sha256"] = runtime._digest(distribution)  # noqa: SLF001
    with_extra = copy.deepcopy(distribution)
    with_extra["protected"] = "SECRET"
    with_extra["receipt_sha256"] = runtime._digest(with_extra)  # noqa: SLF001
    with pytest.raises(ValueError, match="distribution receipt contains"):
        runtime.validate_distribution_receipt(with_extra, plan, 6, rows)
    with pytest.raises(ValueError, match="six-rank"):
        runtime.validate_distribution_receipt(distribution, plan, 6, rows)
