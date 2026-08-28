from __future__ import annotations

import pytest

from evals.fleet.models import (
    DEFAULT_RUNTIME_MODEL,
    DEFAULT_SALES_PRODUCT_ID,
    MAX_SESSIONS_PER_JOB,
    build_plan,
    validate_task_keys,
)


def _keys(count: int) -> list[str]:
    return [f"cysec1-2-demo-gen_blackbox-{index:024x}__blackbox_ctf_v1" for index in range(count)]


def test_full_plan_is_split_below_session_cap() -> None:
    plan = build_plan(_keys(160), pass_k=1)

    assert len(plan.batches) == 27
    assert plan.task_count == 160
    assert plan.planned_sessions == 160
    assert all(batch.planned_sessions <= MAX_SESSIONS_PER_JOB for batch in plan.batches)


def test_pass_k_changes_tasks_per_batch() -> None:
    plan = build_plan(_keys(5), pass_k=3)

    assert [len(batch.task_keys) for batch in plan.batches] == [2, 2, 1]
    assert all(batch.planned_sessions <= 6 for batch in plan.batches)


def test_job_payload_selects_agent_runtime_without_legacy_harness_or_cyber_tag() -> None:
    batch = build_plan(_keys(1)).batches[0]
    payload = batch.payload()

    assert payload["models"] == [DEFAULT_RUNTIME_MODEL]
    assert payload["agent_runtime"] is True
    assert "harness" not in payload
    assert payload["mode"] == "tool-use"
    assert payload["sales_product_id"] == DEFAULT_SALES_PRODUCT_ID
    assert "required_key_capabilities" not in payload


def test_idempotency_key_is_stable_and_request_sensitive() -> None:
    one = build_plan(_keys(1)).batches[0]
    again = build_plan(_keys(1)).batches[0]
    different = build_plan(_keys(1), max_steps=301).batches[0]

    assert one.idempotency_key == again.idempotency_key
    assert one.idempotency_key != different.idempotency_key


def test_rejects_non_cysec_task() -> None:
    with pytest.raises(ValueError, match="non-canonical"):
        build_plan(["demo-task"])


def test_rejects_pass_k_above_hard_cap() -> None:
    with pytest.raises(ValueError, match="six-session"):
        build_plan(_keys(1), pass_k=7)


def test_source_jobs_known_legacy_key_is_preserved() -> None:
    key = "fakelook_blackbox-e1fabce2cfffde31d1e5ecba__blackbox_ctf_v1"
    assert validate_task_keys([key]) == (key,)
