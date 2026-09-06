from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_rank18_g23_terminal_v1 as terminal
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]


def test_terminal_receipt_is_exact_digest_valid_and_zero_effect() -> None:
    failure = terminal.failure_receipt()
    terminal.validate_failure(failure)
    assert failure["receipt_sha256"] == self_hosted.digest_without(
        failure, "receipt_sha256"
    )
    value = terminal.load(ROOT)
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert value["status"] == "TERMINAL_BLOCKED_NONREPEATABLE"
    assert value["authority"]["retry_authorized"] is False
    assert value["authority"]["scored_successor_authorized"] is False
    assert value["authority"]["rank_walk_authorized"] is False
    assert set(value["side_effects"].values()) == {0}
    assert value["tally_after"] == {
        "accepted": 51,
        "active": 0,
        "blocked_nonrepeatable": 9,
        "unstarted": 340,
    }


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("authority", "retry_authorized", True),
        ("authority", "scored_successor_authorized", True),
        ("tally_after", "blocked_nonrepeatable", 8),
        ("observer", "failure_code", "clear"),
        ("side_effects", "model_calls", 1),
        ("privacy", "scores_included", True),
    ],
)
def test_terminal_receipt_rejects_rehashed_mutation(
    section: str, field: str, replacement: object
) -> None:
    value = copy.deepcopy(terminal.terminal_receipt())
    value[section][field] = replacement
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    with pytest.raises(ValueError, match="terminal authority drifted"):
        terminal.validate(value)
