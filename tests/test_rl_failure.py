"""Distributed failures retain code locations without copying private messages."""

import json

from training.rl_runtime import native_failure


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
        },
        {
            "error_class": "AssertionError",
            "actor_init_failed": False,
            "remote_error_classes": [],
            "remote_frames": [{"file": "data.py", "line": 235, "function": "__init__"}],
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


def test_actual_ray_actor_error_flattens_cause_to_message(tmp_path):
    import pytest

    ray = pytest.importorskip("ray.exceptions")
    detail = 'File "/root/training/miles_text.py", line 27, in __init__\nValueError: private data'
    error = ray.ActorDiedError(ray.RayTaskError("Actor.__init__", detail, ValueError("private")))
    receipt = native_failure({"output_root": str(tmp_path)}, error)
    cause = receipt["causes"][0]
    assert cause["actor_init_failed"]
    assert cause["remote_error_classes"] == ["ValueError"]
    assert cause["remote_frames"] == [{"file": "miles_text.py", "line": 27, "function": "__init__"}]
    assert "private" not in json.dumps(receipt)
