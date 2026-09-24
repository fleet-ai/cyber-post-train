from __future__ import annotations

import copy

import pytest

from training import miles_signal_wave as wave


def test_four_fresh_zero_update_signal_lanes_are_exactly_bound() -> None:
    plan = wave.load()
    assert [row["rank"] for row in plan["candidates"]] == [1, 2, 3, 4]
    assert plan["phase1"]["optimizer_steps"] == 0
    assert plan["phase1"]["samples_per_candidate"] == 8
    assert plan["cluster"]["parallel_candidates"] == 4
    assert sum(plan["cluster"]["gpus_per_node"] for _ in plan["candidates"]) == 32
    assert all(
        set(wave.task_binding(plan, row))
        == {
            "task_key",
            "task_version_id",
            "verifier_version_id",
            "task_set_sha256",
            "tool_catalog_sha256",
        }
        for row in plan["candidates"]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["phase1"].__setitem__("optimizer_steps", 1), "seal"),
        (lambda p: p["cluster"].__setitem__("failureAlerts", True), "seal"),
        (lambda p: p["candidates"][1].__setitem__("split", "dev"), "seal"),
        (
            lambda p: p["candidates"][0]["verifier"].__setitem__("version_id", "wrong"),
            "seal",
        ),
    ],
)
def test_wave_fails_closed_on_drift(mutate, message: str) -> None:
    plan = copy.deepcopy(wave.load())
    mutate(plan)
    with pytest.raises(ValueError, match=message):
        wave.validate(plan)
