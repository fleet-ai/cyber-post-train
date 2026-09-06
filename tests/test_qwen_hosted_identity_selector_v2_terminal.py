from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import qwen_hosted_identity_selector_v2_terminal as terminal
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as base

ROOT = Path(__file__).parents[1]


def test_terminal_selector_v2_evidence_is_exact_and_score_blind() -> None:
    observation, receipt = terminal.validate(ROOT)
    assert observation["receipt_sha256"] == base.digest(observation)
    assert receipt["receipt_sha256"] == base.digest(receipt)
    assert receipt["job"] == terminal.JOB
    assert receipt["pod"] == terminal.POD
    assert receipt["classification_counts"] == {
        "AMBIGUOUS_BLOCK": 100,
        "EXACT_MODEL_COLLISION": 0,
        "IDENTITY_CLEAR": 0,
    }
    assert receipt["earliest_clear_rank"] is None
    assert receipt["retry_v2_authorized"] is False
    assert receipt["scored_successor_authorized"] is False
    assert all(
        receipt[key] == 0
        for key in (
            "model_calls",
            "task_calls",
            "session_mutations",
            "verifier_calls",
            "scoring_calls",
            "api_mutations",
        )
    )
    assert receipt["protected_values_materialized"] is False
    assert receipt["scores_included"] is False
    assert receipt["prompts_traces_flags_included"] is False
    assert receipt["credentials_included"] is False


def test_terminal_builder_rejects_rehashed_outcome_or_identity_drift() -> None:
    observation = json.loads((ROOT / terminal.OBSERVATION_PATH).read_text())
    expected = terminal.expected_terminal(observation)
    for mutate in (
        lambda value: value["classification_counts"].__setitem__("IDENTITY_CLEAR", 1),
        lambda value: value["job"].__setitem__("uid", "11111111-1111-4111-8111-111111111111"),
        lambda value: value.__setitem__("scored_successor_authorized", True),
        lambda value: value.__setitem__("protected", "SECRET"),
    ):
        changed = copy.deepcopy(expected)
        mutate(changed)
        changed["receipt_sha256"] = base.digest(changed)
        assert changed != terminal.expected_terminal(observation)


def test_observation_file_digest_is_byte_exact(tmp_path: Path) -> None:
    observation = (ROOT / terminal.OBSERVATION_PATH).read_text()
    changed = tmp_path / terminal.OBSERVATION_PATH
    changed.parent.mkdir(parents=True)
    changed.write_text(observation + " ")
    persisted = json.loads((ROOT / terminal.TERMINAL_PATH).read_text())
    target = tmp_path / terminal.TERMINAL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(persisted))
    with pytest.raises(ValueError, match="observation file drifted"):
        terminal.validate(tmp_path)


@pytest.mark.parametrize(
    "injected",
    [
        b'"protected":"SECRET","protected":false,',
        b'"status":"FORGED","status":"NO_CLEAR_CANDIDATE_TERMINAL",',
        (
            b'"job":{"uid":"SECRET"},'
            b'"job":{"uid":"6cd4e7cd-6f58-442c-b57a-db1d46ea075a"},'
        ),
    ],
)
def test_terminal_duplicate_keys_fail_closed_before_interpretation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: bytes
) -> None:
    binding = terminal.package.build_binding(ROOT)
    monkeypatch.setattr(terminal.package, "build_binding", lambda _root: binding)
    observation_target = tmp_path / terminal.OBSERVATION_PATH
    observation_target.parent.mkdir(parents=True)
    observation_target.write_bytes((ROOT / terminal.OBSERVATION_PATH).read_bytes())
    raw = (ROOT / terminal.TERMINAL_PATH).read_bytes()
    malicious = b"{" + injected + raw[1:]
    terminal_target = tmp_path / terminal.TERMINAL_PATH
    terminal_target.parent.mkdir(parents=True, exist_ok=True)
    terminal_target.write_bytes(malicious)
    monkeypatch.setattr(terminal, "TERMINAL_FILE_SHA256", base.sha256(malicious))
    with pytest.raises(base.GateError, match="^duplicate_json_key$") as caught:
        terminal.validate(tmp_path)
    assert "SECRET" not in str(caught.value)
