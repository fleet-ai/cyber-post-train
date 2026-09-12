import json
from pathlib import Path

import pytest

from training import exposure_control
from training.exposure_control import (
    MAX_REACHABILITY_BITS,
    SearchExhausted,
    _write_outputs,
    choose_exact,
)
from training.io import digest_json, file_sha256

ROOT = Path(__file__).parents[1]
LOCK = ROOT / "configs/studies/qwen-blackbox-teacher-exposure-matched-control-v1.json"


def _exposure(families, selected):
    rows = {
        episode_id: (family, segments, tokens)
        for family, episodes in families.items()
        for episode_id, segments, tokens in episodes
    }
    return (
        len(selected),
        sum(rows[episode_id][1] for episode_id in selected),
        sum(rows[episode_id][2] for episode_id in selected),
        {rows[episode_id][0] for episode_id in selected},
    )


def test_exact_control_is_input_order_invariant_and_keeps_every_family():
    families = {
        "family-b": [("b2", 4, 55), ("b1", 3, 30)],
        "family-a": [("a2", 2, 25), ("a1", 1, 10)],
    }
    expected = (2, 5, 55, {"family-a", "family-b"})
    selected, states = choose_exact(
        families,
        target_episodes=2,
        target_segments=5,
        target_supervised_tokens=55,
        seed="fixed-control",
    )
    reversed_input = {
        family: list(reversed(rows)) for family, rows in reversed(list(families.items()))
    }
    repeated, repeated_states = choose_exact(
        reversed_input,
        target_episodes=2,
        target_segments=5,
        target_supervised_tokens=55,
        seed="fixed-control",
    )
    assert _exposure(families, selected) == expected
    assert selected == repeated
    assert states == repeated_states


def test_exact_control_fails_closed_when_joint_target_is_impossible():
    families = {"a": [("a1", 1, 10)], "b": [("b1", 2, 20)]}
    with pytest.raises(ValueError, match="no exact whole-episode"):
        choose_exact(
            families,
            target_episodes=2,
            target_segments=3,
            target_supervised_tokens=29,
            seed="fixed-control",
        )


def test_exact_control_has_an_explicit_search_bound():
    families = {"a": [("a1", 1, 10)], "b": [("b1", 2, 20)]}
    with pytest.raises(SearchExhausted):
        choose_exact(
            families,
            target_episodes=2,
            target_segments=3,
            target_supervised_tokens=30,
            seed="fixed-control",
            max_search_states=1,
        )


def test_exact_control_falls_back_from_a_locally_preferred_dead_end():
    families = {
        "a": [("a1", 1, 1), ("a2", 2, 1)],
        "b": [("b1", 3, 1), ("b2", 4, 1)],
    }
    selected, _ = choose_exact(
        families,
        target_episodes=3,
        target_segments=6,
        target_supervised_tokens=3,
        seed="fallback",
    )
    assert selected == ["a1", "a2", "b1"]


def test_exact_control_chooses_the_first_feasible_local_share_preference():
    families = {
        "a": [("a1", 1, 1), ("a2", 1, 1), ("a3", 1, 1)],
        "b": [("b1", 1, 1), ("b2", 3, 3)],
    }
    selected, _ = choose_exact(
        families,
        target_episodes=4,
        target_segments=6,
        target_supervised_tokens=6,
        seed="local-rank",
    )
    # Selecting all three A episodes is closer to A's parent exposure share;
    # selecting two A episodes plus both B episodes is also exactly feasible.
    assert selected == ["a1", "a2", "a3", "b2"]


def test_exact_control_uses_the_seeded_digest_for_exact_ties():
    families = {"a": [("a1", 1, 1), ("a2", 1, 1)]}
    seed = "seeded-tie"
    expected = min(
        ("a1", "a2"),
        key=lambda episode_id: digest_json(
            {
                "seed": seed,
                "family_id": "a",
                "selected_episode_ids": [episode_id],
            }
        ),
    )
    selected, _ = choose_exact(
        families,
        target_episodes=1,
        target_segments=1,
        target_supervised_tokens=1,
        seed=seed,
    )
    assert selected == [expected]


@pytest.mark.parametrize(
    ("families", "target", "message"),
    [
        ({1: [("a1", 1, 10)]}, (1, 1, 10), "family identities"),
        ({"a": [(1, 1, 10)]}, (1, 1, 10), "episode exposures"),
        ({"a": [("a1", 1, 10)]}, (2, 1, 10), "candidate universe"),
        (
            {"a": [("a1", 1, MAX_REACHABILITY_BITS + 1)]},
            (1, 1, MAX_REACHABILITY_BITS + 1),
            "reachability-bitset bound",
        ),
    ],
)
def test_exact_control_rejects_malformed_or_unbounded_inputs(families, target, message):
    with pytest.raises(ValueError, match=message):
        choose_exact(
            families,
            target_episodes=target[0],
            target_segments=target[1],
            target_supervised_tokens=target[2],
            seed="fixed-control",
        )


def test_create_once_output_pair_rejects_aliases_and_rolls_back(monkeypatch, tmp_path):
    same = tmp_path / "same.json"
    with pytest.raises(ValueError, match="distinct"):
        _write_outputs(same, same, {"a": 1}, {"b": 2})
    assert not same.exists()

    calls = 0
    original = exposure_control._private_write_once

    def fail_second(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second-write failure")
        original(path, value)

    monkeypatch.setattr(exposure_control, "_private_write_once", fail_second)
    selection, receipt = tmp_path / "selection.json", tmp_path / "receipt.json"
    with pytest.raises(OSError, match="second-write"):
        _write_outputs(selection, receipt, {"a": 1}, {"b": 2})
    assert not selection.exists() and not receipt.exists()


def test_frozen_control_is_exact_dose_matched_and_nonlaunchable():
    value = json.loads(LOCK.read_text())
    assert value["sha256"] == digest_json(
        {key: row for key, row in value.items() if key != "sha256"}
    )
    assert value["execution"] == {
        "kind": "metadata_only_not_a_job_request",
        "new_jobs_authorized": False,
        "dev_cluster_qualification_required": True,
    }
    assert value["implementation"]["selection_code_sha256"] == file_sha256(
        ROOT / "training/exposure_control.py"
    )
    acceptance = value["materialization_acceptance"]
    assert acceptance["validator_code_sha256"] == file_sha256(
        ROOT / "training/corpus_acceptance.py"
    )
    assert acceptance["read_policy"] == "opaque_sha256_only_no_parquet_or_token_decoding"
    assert set(acceptance["variants"]) == {"a", "b"}
    for name, variant in value["variants"].items():
        assert (
            variant["outcome_protocol_sha256"]
            == value["evaluation"]["variants"][name]["protocol_sha256"]
        )
        assert (
            variant["dev_task_set_sha256"]
            == value["evaluation"]["variants"][name]["task_set_sha256"]
        )
        assert (
            variant["corpus_embedded_fleet_dev_protocol_sha256"]
            != variant["outcome_protocol_sha256"]
        )
        assert variant["target"] == {
            key: variant["matched_control"][key]
            for key in ("episodes", "segments", "supervised_tokens")
        }
        assert variant["target"]["episodes"] == variant["balanced"]["episodes"]
        assert variant["target"]["segments"] == variant["balanced"]["segments"]
        assert variant["target"]["supervised_tokens"] == variant["balanced"]["supervised_tokens"]
        assert variant["matched_control"]["families"] == variant["balanced"]["families"]
        for field in ("episodes", "segments", "supervised_tokens"):
            assert (
                variant["matched_control"]["tv_to_available"][field]
                < variant["balanced"]["tv_to_available"][field]
            )
