from __future__ import annotations

import json

import httpx
import pytest

from evals.fleet import opencode_train_sweep, opencode_train_sweep_runner, self_hosted


def test_duplicate_preflight_normalizes_persisted_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "sessions": [{"model": "qwen3.8-27b"}],
            "has_more": False,
        }

    monkeypatch.setattr(self_hosted, "_request", fake_request)
    with pytest.raises(RuntimeError, match="already exist"):
        opencode_train_sweep.duplicate_preflight(
            httpx.Client(),
            [{"task_key": "task-a", "task_version_id": "version-a"}],
            "fleet-cluster-opencode-1.18.27/qwen3.8-27b",
        )
    assert calls == 1


def test_full_plan_validator_requires_exact_pass4_arithmetic() -> None:
    plan = {
        "schema_version": opencode_train_sweep_runner.PLAN_SCHEMA,
        "task_count": 50,
        "pass_k": 4,
        "tasks": [{}] * 50,
        "attempts": [{}] * 199,
        "execution": {
            "max_concurrent": 1,
            "required_task_tools": ["bash", "submit_report"],
        },
    }
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    opencode_train_sweep_runner.validate_plan(plan)
    broken = json.loads(json.dumps(plan))
    broken["attempts"].pop()
    broken["plan_sha256"] = self_hosted.digest_without(broken, "plan_sha256")
    with pytest.raises(ValueError, match="attempt count"):
        opencode_train_sweep_runner.validate_plan(broken)
