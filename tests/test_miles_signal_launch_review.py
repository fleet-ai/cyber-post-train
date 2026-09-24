from __future__ import annotations

import copy
import json

import pytest

from training import miles96_mechanics_canary as mechanics
from training import miles_signal_launch_review as review


def _write(tmp_path, value: dict):
    body = {key: item for key, item in value.items() if key != "sha256"}
    value["sha256"] = "sha256:" + mechanics.digest(body)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(value))
    return path


def test_score_blind_review_rebuilds_all_four_exact_lanes() -> None:
    value = review.load()
    assert value["launchable"] is False
    assert [row["name"] for row in value["candidate_bindings"]] == [
        "chris-q38-m96-sig-2c50-a1",
        "chris-q38-m96-sig-7317-a1",
        "chris-q38-m96-sig-8095-a1",
        "chris-q38-m96-sig-f294-a1",
    ]
    assert value["execution_contract"]["sample_indexes"] == list(range(8))
    assert value["execution_contract"]["optimizer_steps"] == 0
    assert value["execution_contract"]["checkpoint"] is False


def test_review_contains_no_candidate_score_or_reward_prior() -> None:
    value = review.load()
    forbidden = {
        "selection_prior",
        "campaign_rank",
        "completed_accepted_episodes",
        "passes",
        "mixed_binary_outcomes",
        "verified_outcomes_sha256",
        "cell_set_sha256",
        "reward_values",
    }
    for row in value["candidate_bindings"]:
        assert not forbidden.intersection(row)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("launchable", True),
        lambda value: value["candidate_bindings"][0].__setitem__(
            "plan_sha256", "sha256:" + "0" * 64
        ),
        lambda value: value["candidate_bindings"][0].__setitem__("selection_prior", {"passes": 2}),
        lambda value: value["required_fresh_at_launch_evidence"][0].__setitem__(
            "maximum_age_seconds", 901
        ),
        lambda value: value["required_fresh_at_launch_evidence"][1].__setitem__(
            "requirement", "duplicates are absent"
        ),
        lambda value: value["runtime_fail_closed_gates"].__setitem__(
            "live_raw_tool_catalog_must_match_authority", False
        ),
        lambda value: value["candidate_bindings"].reverse(),
        lambda value: value["execution_contract"].__setitem__("optimizer_steps", 1),
    ],
)
def test_review_rejects_semantic_drift(tmp_path, mutate) -> None:
    value = copy.deepcopy(review.load())
    mutate(value)
    with pytest.raises(ValueError):
        review.load(_write(tmp_path, value))
