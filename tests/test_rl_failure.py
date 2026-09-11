"""Distributed failures retain code locations without copying private messages."""

import json

import pytest

from training.rl_episode import BUDGET_STOPS, EpisodeBudgetExceeded, InvalidEpisode, budget_stop
from training.rl_runtime import native_failure, native_rejection, sealed


@pytest.mark.parametrize("reason", sorted(BUDGET_STOPS))
@pytest.mark.parametrize("grouped", [False, True])
def test_explicit_budget_rejection_is_private_and_not_training(tmp_path, reason, grouped):
    error = EpisodeBudgetExceeded(reason)
    if grouped:
        error = ExceptionGroup("private transport context", [error, error])
    assert budget_stop(error) == reason
    assert native_rejection({"output_root": str(tmp_path)}, error)
    receipt = json.loads((tmp_path / "NATIVE_REJECTED.json").read_bytes())
    sealed(receipt, "cyber_rl_native_rejection_v1")
    assert receipt["status"] == "rejected" and receipt["reason"] == reason
    assert "private" not in json.dumps(receipt)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["NATIVE_REJECTED.json"]


@pytest.mark.parametrize("fault", ["generic", "unknown", "nonstring", "mixed", "cause"])
def test_real_errors_cannot_be_clean_rejections(tmp_path, fault):
    limit = EpisodeBudgetExceeded("generation_incomplete_length")
    error = {
        "generic": InvalidEpisode("generation_incomplete_length"),
        "unknown": EpisodeBudgetExceeded("private unknown stop"),
        "nonstring": EpisodeBudgetExceeded({"private": "data"}),
        "mixed": ExceptionGroup("private", [limit, OSError("private")]),
        "cause": RuntimeError("private"),
    }[fault]
    error.__cause__ = limit
    assert budget_stop(error) is None
    assert not native_rejection({"output_root": str(tmp_path)}, error)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("grouped", [False, True])
def test_actual_ray_serialization_preserves_only_typed_budget_stops(tmp_path, grouped):
    ray = pytest.importorskip("ray.exceptions")
    pickle = pytest.importorskip("ray.cloudpickle")
    cause = EpisodeBudgetExceeded("generation_incomplete_length")
    if grouped:
        cause = ExceptionGroup("private", [cause])
    wrapped = ray.RayTaskError("generate", "private remote text", cause).as_instanceof_cause()
    restored = pickle.loads(pickle.dumps(wrapped))
    assert budget_stop(restored) == "generation_incomplete_length"
    assert native_rejection({"output_root": str(tmp_path)}, restored)
    assert "private" not in (tmp_path / "NATIVE_REJECTED.json").read_text()
    # Actor death or another runtime failure must never unwrap a budget cause.
    died = ray.ActorDiedError(ray.RayTaskError("generate", "private", cause))
    assert budget_stop(died) is None


def test_remote_error_chain_is_sanitized_bounded_and_sealed(tmp_path):
    error = RuntimeError("private prompt and credential")
    error.actor_init_failed = True
    error.cause = AssertionError("another private message")
    error.cause.traceback_str = (
        'File "/root/miles/data.py", line 235, in __init__\n  private prompt and credential\n'
    )
    error.cause.__context__ = error
    receipt = native_failure({"output_root": str(tmp_path)}, error)
    assert receipt["causes"] == [
        {
            "error_class": "RuntimeError",
            "actor_init_failed": True,
            "remote_error_classes": [],
            "remote_frames": [],
            "local_frames": [],
        },
        {
            "error_class": "AssertionError",
            "actor_init_failed": False,
            "remote_error_classes": [],
            "remote_frames": [{"file": "data.py", "line": 235, "function": "__init__"}],
            "local_frames": [],
        },
    ]
    assert "private" not in json.dumps(receipt)
    assert json.loads((tmp_path / "NATIVE_FAILURE.json").read_text()) == receipt


def test_long_error_chain_has_a_fixed_bound(tmp_path):
    error = current = RuntimeError()
    for _ in range(12):
        current.__cause__ = RuntimeError()
        current = current.__cause__
    receipt = native_failure({"output_root": str(tmp_path)}, error)
    assert len(receipt["causes"]) == 8


def test_local_exception_retains_locations_without_payload(tmp_path):
    try:
        raise PermissionError("private path and credential")
    except PermissionError as error:
        receipt = native_failure({"output_root": str(tmp_path)}, error)
    frame = receipt["causes"][0]["local_frames"][0]
    assert frame["file"] == "test_rl_failure.py"
    assert frame["function"] == "test_local_exception_retains_locations_without_payload"
    assert frame["line"] > 0
    assert "private" not in json.dumps(receipt)
    assert "credential" not in json.dumps(receipt)


@pytest.mark.parametrize("wrapper", [None, "__cause__", "cause"])
def test_grouped_failures_retain_leaf_types_without_messages(tmp_path, wrapper):
    error = ExceptionGroup(
        "private group message",
        [TimeoutError("private tool"), ExceptionGroup("private inner", [ValueError("private")])],
    )
    if wrapper:
        outer = RuntimeError("private wrapper")
        setattr(outer, wrapper, error)
        error = outer
    receipt = native_failure({"output_root": str(tmp_path)}, error)
    classes = [cause["error_class"] for cause in receipt["causes"]]
    assert classes.count("ExceptionGroup") == 2
    assert "TimeoutError" in classes and "ValueError" in classes
    assert "private" not in json.dumps(receipt)


def test_grouped_failures_are_bounded_and_cycle_safe(tmp_path):
    leaf = TimeoutError("private")
    error = ExceptionGroup("private", [leaf] * 20 + [ValueError("private")])
    leaf.__context__ = error
    receipt = native_failure({"output_root": str(tmp_path)}, error)
    assert [cause["error_class"] for cause in receipt["causes"]] == [
        "ExceptionGroup",
        "TimeoutError",
        "ValueError",
    ]
    # The existing bound applies to groups as well as ordinary cause chains.
    other = tmp_path / "wide"
    other.mkdir()
    group = ExceptionGroup("private", [RuntimeError("private") for _ in range(20)])
    receipt = native_failure({"output_root": str(other)}, group)
    assert len(receipt["causes"]) == 8


def test_actual_ray_actor_error_flattens_cause_to_message(tmp_path):
    ray = pytest.importorskip("ray.exceptions")
    detail = 'File "/root/training/miles_text.py", line 27, in __init__\nValueError: private data'
    error = ray.ActorDiedError(ray.RayTaskError("Actor.__init__", detail, ValueError("private")))
    receipt = native_failure({"output_root": str(tmp_path)}, error)
    cause = receipt["causes"][0]
    assert cause["actor_init_failed"]
    assert cause["remote_error_classes"] == ["ValueError"]
    assert cause["remote_frames"] == [{"file": "miles_text.py", "line": 27, "function": "__init__"}]
    assert "private" not in json.dumps(receipt)
