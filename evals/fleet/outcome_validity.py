"""Score-blind validity gates for fixed-seed Fleet evaluations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


class OutcomeValidityError(ValueError):
    """An attempt roster is incomplete or contains a held outcome."""


def require_normal_completion(row: Mapping[str, Any], *, label: str = "attempt") -> None:
    """Require the only OpenCode lifecycle that is a valid model outcome."""

    exit_code = row.get("agent_exit_code")
    termination = row.get("agent_termination")
    if type(exit_code) is not int or exit_code != 0 or termination != "completed":
        raise OutcomeValidityError(
            f"{label} is held or infrastructure-invalid, not a capability outcome"
        )


def require_complete_passk(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_versions: Sequence[str],
    arms: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, int | bool]:
    """Prove an exact fixed-seed pass@k roster before any score is read.

    The function deliberately never reads a score. An output-limit stop,
    process error, duplicate, missing cell, or unexpected cell keeps the whole
    study incomplete; callers may then apply the frozen replacement-seed policy.
    """

    tasks = tuple(task_versions)
    arm_names = tuple(arms)
    seed_values = tuple(seeds)
    if (
        not tasks
        or not arm_names
        or not seed_values
        or len(set(tasks)) != len(tasks)
        or len(set(arm_names)) != len(arm_names)
        or len(set(seed_values)) != len(seed_values)
        or any(not isinstance(value, str) or not value for value in (*tasks, *arm_names))
        or any(type(value) is not int for value in seed_values)
    ):
        raise OutcomeValidityError("expected pass@k roster is invalid")

    expected = {
        (task_version, arm, seed)
        for task_version in tasks
        for arm in arm_names
        for seed in seed_values
    }
    observed: Counter[tuple[object, object, object]] = Counter()
    held = Counter()
    for row in rows:
        identity = (row.get("task_version_id"), row.get("arm"), row.get("seed"))
        observed[identity] += 1
        try:
            require_normal_completion(row)
        except OutcomeValidityError:
            held[str(row.get("agent_termination") or "unknown")] += 1

    if set(observed) != expected or any(count != 1 for count in observed.values()) or held:
        reasons = ",".join(f"{key}={held[key]}" for key in sorted(held)) or "none"
        raise OutcomeValidityError(
            "pass@k is incomplete: "
            f"expected={len(expected)} observed={sum(observed.values())} held={reasons}"
        )
    return {"complete": True, "pass_k": len(seed_values), "valid_attempts": len(expected)}
