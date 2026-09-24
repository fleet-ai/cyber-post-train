from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import pytest

from evals.fleet.outcome_validity import (
    OutcomeValidityError,
    require_complete_passk,
    require_normal_completion,
)


class _ScoreTrap(Mapping[str, Any]):
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def __getitem__(self, key: str) -> Any:
        if key == "score":
            raise AssertionError("score was opened before the validity gate")
        return self.value[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self.value else default


@pytest.mark.parametrize(
    "exit_code,termination",
    [(0, "output_limit"), (1, "process_error"), (0, "execution_timeout")],
)
def test_only_normal_completion_is_a_model_outcome(exit_code: int, termination: str) -> None:
    with pytest.raises(OutcomeValidityError, match="held or infrastructure-invalid"):
        require_normal_completion({"agent_exit_code": exit_code, "agent_termination": termination})


def test_complete_passk_gate_is_score_blind() -> None:
    rows = [
        _ScoreTrap(
            {
                "task_version_id": task,
                "arm": arm,
                "seed": seed,
                "agent_exit_code": 0,
                "agent_termination": "completed",
                "score": 1,
            }
        )
        for task in ("task-a", "task-b")
        for arm in ("base", "candidate")
        for seed in (48, 54, 61, 62)
    ]

    assert require_complete_passk(
        rows,
        task_versions=("task-a", "task-b"),
        arms=("base", "candidate"),
        seeds=(48, 54, 61, 62),
    ) == {"complete": True, "pass_k": 4, "valid_attempts": 16}


@pytest.mark.parametrize("termination,exit_code", [("output_limit", 0), ("process_error", 1)])
def test_passk_never_turns_a_held_attempt_into_zero(termination: str, exit_code: int) -> None:
    rows = [
        {
            "task_version_id": "task-a",
            "arm": arm,
            "seed": seed,
            "agent_exit_code": 0,
            "agent_termination": "completed",
            "score": 0,
        }
        for arm in ("base", "candidate")
        for seed in (48, 54, 61, 62)
    ]
    rows[-1].update(agent_exit_code=exit_code, agent_termination=termination, score=0)

    with pytest.raises(OutcomeValidityError, match=f"held={termination}=1"):
        require_complete_passk(
            rows,
            task_versions=("task-a",),
            arms=("base", "candidate"),
            seeds=(48, 54, 61, 62),
        )


def test_passk_rejects_missing_duplicate_or_unexpected_cells() -> None:
    normal = {
        "task_version_id": "task-a",
        "arm": "base",
        "seed": 48,
        "agent_exit_code": 0,
        "agent_termination": "completed",
    }
    with pytest.raises(OutcomeValidityError, match="pass@k is incomplete"):
        require_complete_passk(
            [normal, normal],
            task_versions=("task-a",),
            arms=("base",),
            seeds=(48, 54),
        )
