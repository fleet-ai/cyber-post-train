"""Deterministic reward composition for execution-verified cyber episodes."""

from __future__ import annotations

from typing import Any


def _bounded(value: Any, default: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return default
    return min(1.0, max(0.0, float(value)))


def compute_reward(
    *,
    verifier_score: Any,
    infra_valid: bool,
    behavior_score: Any = 1.0,
    reward_hack_detected: bool = False,
) -> dict[str, Any]:
    """Compose reward without allowing safety or infrastructure failures through.

    Exploit success remains the learning signal.  A session with invalid
    infrastructure, detected grader access, or other reward hacking receives
    zero irrespective of its apparent verifier score.  Behavior constraints
    are a bounded multiplier, which keeps them auditable as a separate metric.
    """
    exploit = _bounded(verifier_score)
    behavior = _bounded(behavior_score)
    gate = 1.0 if infra_valid and not reward_hack_detected else 0.0
    total = exploit * behavior * gate
    return {
        "total": total,
        "exploit": exploit,
        "behavior": behavior,
        "integrity_gate": gate,
        "reward_hack_detected": bool(reward_hack_detected),
    }
