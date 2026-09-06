from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from evals.fleet import exact_pass4_ledger_snapshot_v49 as snapshot
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def _checked_in() -> dict:
    value = json.loads((ROOT / snapshot.OUTPUT_PATH).read_text())
    assert isinstance(value, dict)
    return value


def _cell(value: dict, model: str, rank: int, attempt: int) -> dict:
    return next(
        row
        for row in value["cells"]
        if (row["model"], row["selection_rank"], row["attempt"]) == (model, rank, attempt)
    )


def test_checked_in_snapshot_is_deterministic_and_self_digesting() -> None:
    expected = _checked_in()
    assert snapshot.build(ROOT) == expected
    assert expected["receipt_sha256"] == self_hosted.digest_without(expected, "receipt_sha256")


def test_all_800_exact_cells_are_enumerated_once() -> None:
    value = _checked_in()
    cells = value["cells"]
    assert len(cells) == 800
    assert len({row["cell_id"] for row in cells}) == 800
    assert len({(row["model"], row["selection_rank"], row["attempt"]) for row in cells}) == 800
    assert Counter(row["model"] for row in cells) == {
        "glm-5.3": 400,
        "qwen3.8-27b": 400,
    }
    assert {
        (model, rank, attempt)
        for model in ("glm-5.3", "qwen3.8-27b")
        for rank in range(1, 101)
        for attempt in range(1, 5)
    } == {(row["model"], row["selection_rank"], row["attempt"]) for row in cells}


def test_strict_tallies_are_derived_from_cell_states() -> None:
    value = _checked_in()
    for model, expected in snapshot.STRICT_V48_TALLY.items():
        counter = Counter(row["state"] for row in value["cells"] if row["model"] == model)
        assert {state: counter[state] for state in expected} == expected
    assert value["authoritative_tally"]["total"] == {
        "accepted": 66,
        "active": 1,
        "blocked_nonrepeatable": 12,
        "unstarted": 721,
    }


def test_post_v48_aggregate_claim_is_not_silently_projected() -> None:
    value = _checked_in()
    assert value["post_v48_qwen_tally_claim"] == {
        "claimed": snapshot.POST_V48_QWEN_TALLY_CLAIM,
        "adopted": False,
        "reason": (
            "the later receipts carry an aggregate constant but do not bind the nine "
            "additional accepted cell receipts, and the rank18 block does not name an attempt"
        ),
    }
    for rank in (15, 16):
        for attempt in range(1, 5):
            row = _cell(value, "qwen3.8-27b", rank, attempt)
            assert row["state"] == "unstarted"
            assert row["post_v48_observations"][0]["scientific_effect"] == (
                "UNRESOLVED_WITHOUT_CELL_ACCEPTED_RECEIPT"
            )
    rank99_a4 = _cell(value, "qwen3.8-27b", 99, 4)
    assert rank99_a4["state"] == "active"
    assert rank99_a4["post_v48_observations"][0]["scientific_effect"] == (
        "NOT_ADOPTED_WITHOUT_CELL_ACCEPTED_RECEIPT"
    )
    for attempt in range(1, 5):
        row = _cell(value, "qwen3.8-27b", 18, attempt)
        assert row["state"] == "unstarted"
        assert row["post_v48_observations"][0]["scientific_effect"] == ("NOT_PROJECTED_TO_ANY_CELL")


def test_every_non_unstarted_cell_has_v48_cell_evidence() -> None:
    for row in _checked_in()["cells"]:
        if row["state"] == "unstarted":
            assert row["v48_evidence"] == []
        else:
            assert row["v48_evidence"]
            assert all(item["receipt_sha256"].startswith("sha256:") for item in row["v48_evidence"])


def test_later_lineage_is_explicitly_aggregate_only() -> None:
    value = _checked_in()
    assert len(value["later_qwen_evidence"]) == 8
    assert all(row["cell_authority"] is False for row in value["later_qwen_evidence"])
    assert all(row["file_sha256"].startswith("sha256:") for row in value["later_qwen_evidence"])
    assert all(row["receipt_sha256"].startswith("sha256:") for row in value["later_qwen_evidence"])


def test_path_parser_uses_terminal_attempt_component() -> None:
    assert snapshot._accepted_coordinates(
        "/mnt/sfs/jobs/glm-example-r003-a2a4/accepted/glm-example-r003-a4-g1.json"
    ) == ("glm-5.3", 3, 4)


def test_snapshot_contains_no_prohibited_payload_keys() -> None:
    prohibited = {
        "answer",
        "answers",
        "flag",
        "flags",
        "prompt",
        "prompts",
        "reward",
        "rewards",
        "score",
        "scores",
        "solution",
        "solutions",
        "trace",
        "traces",
        "transcript",
        "transcripts",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert not (set(value) & prohibited)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(_checked_in())
