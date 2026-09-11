"""Exercise the actual observation/cancellation boundary without cluster access."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training import sft_runtime as runtime


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("0\n 10 \n\n20\n", 10.0),
        ("", None),
        ("N/A", None),
        ("nan", None),
        ("inf", None),
        ("-1", None),
        ("101", None),
        ("0\nN/A", None),
        (FileNotFoundError(), None),
        (subprocess.TimeoutExpired("synthetic", 10), None),
    ],
)
def test_gpu_observation_is_bounded_and_missing_is_not_zero(monkeypatch, reply, expected):
    command = Mock()
    if isinstance(reply, Exception):
        command.side_effect = reply
    else:
        command.return_value = SimpleNamespace(stdout=reply)
    monkeypatch.setattr(runtime.subprocess, "run", command)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [])
    assert runtime._utilization_snapshot() == (expected, None)
    command.assert_called_once_with(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )


def test_process_observation_sums_only_readable_physical_io(tmp_path, monkeypatch):
    payloads = [
        "read_bytes: 10\nwrite_bytes: 20",
        "read_bytes: 5\nwrite_bytes: 7",
        "read_bytes: n/a\nwrite_bytes: 0",
        "read_bytes: 12",
        "malformed",
    ]
    paths = [tmp_path / str(i) for i in range(len(payloads))]
    for path, text in zip(paths, payloads, strict=True):
        path.write_text(text)
    paths.append(tmp_path / "gone")

    def glob(path, pattern):
        assert path == Path("/proc") and pattern == "[0-9]*/io"
        return paths

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(runtime.subprocess, "run", Mock(return_value=SimpleNamespace(stdout="0")))
    assert runtime._utilization_snapshot() == (0, 42)


def test_output_fingerprint_ignores_logs_and_tolerates_checkpoint_rotation(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for name in ("PROGRESS.json", "ACTIVITY.json", "private-trainer.log"):
        (tmp_path / name).write_text("synthetic")
    live, gone = checkpoints / "live", checkpoints / "gone"
    live.write_bytes(b"1234")
    gone.write_bytes(b"rotating")
    stat = Path.stat

    def racing_stat(path, *args, **kwargs):
        if path == gone:
            raise FileNotFoundError
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", racing_stat)
    markers, files = runtime._output_progress(tmp_path)
    assert [row[0] for row in markers] == ["ACTIVITY.json", "PROGRESS.json"]
    assert files == (("checkpoints/live", 4, stat(live).st_mtime_ns),)
    assert all(row[2] > 0 for row in markers)
    (tmp_path / "private-trainer.log").write_text("different synthetic text")
    assert runtime._output_progress(tmp_path) == (markers, files)


@pytest.mark.parametrize("error", [None, RuntimeError("synthetic worker defect")])
def test_ready_result_does_not_poll_telemetry_or_cancel(tmp_path, monkeypatch, error):
    task = object()
    ray = SimpleNamespace(
        wait=Mock(return_value=([task], [])), get=Mock(return_value={"step": 1}), cancel=Mock()
    )
    if error:
        ray.get.side_effect = error
    probe = Mock(side_effect=AssertionError("no telemetry after terminal result"))
    monkeypatch.setattr(runtime, "_utilization_snapshot", probe)
    if error:
        with pytest.raises(RuntimeError, match="synthetic worker defect"):
            runtime._wait_for_training(ray, task, tmp_path)
    else:
        assert runtime._wait_for_training(ray, task, tmp_path) == {"step": 1}
    ray.wait.assert_called_once_with([task], timeout=runtime.WATCHDOG_POLL_SECONDS)
    ray.get.assert_called_once_with(task)
    ray.cancel.assert_not_called()
    assert not (tmp_path / "WATCHDOG.json").exists()


@pytest.mark.parametrize("unavailable", [None, "gpu", "io"])
def test_waiter_cancels_only_confirmed_idle_and_preserves_a_receipt(
    tmp_path, monkeypatch, unavailable
):
    task = object()
    # First sample establishes the marker; second begins the idle interval.
    start = runtime.WATCHDOG_STARTUP_SECONDS + 1
    moments = iter([0, start, start + 1, start + 1 + runtime.WATCHDOG_IDLE_SECONDS])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(runtime.time, "time", lambda: 1234.0)
    monkeypatch.setattr(
        runtime,
        "_utilization_snapshot",
        lambda: (
            None if unavailable == "gpu" else 0,
            None if unavailable == "io" else 0,
        ),
    )
    ray = SimpleNamespace(
        wait=Mock(side_effect=[([], [task])] * 3 + [([task], [])]),
        get=Mock(return_value={"step": 2}),
        cancel=Mock(),
    )
    if unavailable:
        assert runtime._wait_for_training(ray, task, tmp_path) == {"step": 2}
        ray.cancel.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="confirmed_no_progress_idle"):
            runtime._wait_for_training(ray, task, tmp_path)
        ray.cancel.assert_called_once_with(task, force=True, recursive=True)
        ray.get.assert_not_called()
    proof = json.loads((tmp_path / "WATCHDOG.json").read_text())
    assert proof.pop("receipt_sha256") == runtime._unsigned_digest(proof)
    assert proof["state"] == ("monitoring" if unavailable else "confirmed_no_progress_idle")
    assert proof["observed_at_unix"] == 1234.0
    assert proof["process_io_available"] == (unavailable != "io")
    assert "synthetic" not in json.dumps(proof)


def test_wandb_configuration_requires_injection_before_creating_output(tmp_path, monkeypatch):
    plan = {
        "output_root": str(tmp_path),
        "wandb": {
            "entity": "synthetic",
            "group": "group",
            "run_id": "run",
        },
    }
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="secret injection missing"):
        runtime._configure_wandb(plan)
    assert not (tmp_path / "wandb").exists()


@pytest.mark.parametrize("activity", ["checkpoint", "stale_checkpoint", "episode", "none"])
def test_rl_runtime_grants_only_checkpoint_writes_a_bounded_final_drain(
    tmp_path, monkeypatch, activity
):
    from training import rl_runtime

    hard, drain = runtime.WATCHDOG_HARD_SECONDS, runtime.WATCHDOG_DRAIN_SECONDS
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(rl_runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.setattr(runtime, "_utilization_snapshot", lambda: (100, 100))
    backend = SimpleNamespace(
        MODULE="training.synthetic_training",
        native_source=Mock(),
        native_result=Mock(side_effect=AssertionError("must not accept an unfinished run")),
    )
    samples = []
    child = SimpleNamespace(pid=123456, returncode=None, poll=lambda: None)
    terminate = Mock()
    monkeypatch.setattr(rl_runtime.os, "killpg", terminate)

    def wait(*, timeout):
        if timeout == 30:  # bounded cleanup, not a watchdog sample
            return -15
        assert timeout == runtime.WATCHDOG_POLL_SECONDS
        moments = [hard - 1, hard, hard + drain]
        clock.now = moments[len(samples)]
        samples.append(clock.now)
        if activity != "none" and (activity != "stale_checkpoint" or len(samples) == 1):
            folder = "checkpoints" if "checkpoint" in activity else "episodes"
            path = tmp_path / folder / "synthetic-state"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"x" * len(samples))
        raise subprocess.TimeoutExpired("synthetic", timeout)

    child.wait = wait
    monkeypatch.setattr(rl_runtime.subprocess, "Popen", Mock(return_value=child))
    with pytest.raises(RuntimeError, match="no automatic retry"):
        rl_runtime.run({"output_root": str(tmp_path)}, tmp_path / "plan.json", backend=backend)
    assert samples == (
        [hard - 1, hard, hard + drain] if activity == "checkpoint" else [hard - 1, hard]
    )
    proof = json.loads((tmp_path / "FAILED.json").read_text())
    assert proof["error_class"] == "TimeoutError"
    assert proof["watchdog_reason"] == "hard_runtime_bound"
    assert proof.pop("sha256") == rl_runtime.digest(proof)
    backend.native_result.assert_not_called()
    assert not (tmp_path / "NATIVE_TRAINING_COMPLETE.json").exists()
    assert terminate.call_args_list == [
        ((child.pid, rl_runtime.signal.SIGTERM),),
        ((child.pid, rl_runtime.signal.SIGKILL),),
    ]
