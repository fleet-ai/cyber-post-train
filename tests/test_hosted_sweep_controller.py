from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.fleet import hosted_sweep_controller as hosted
from evals.fleet import self_hosted

SOURCE = Path(
    "docs/evidence/qwen38-study/2026-09-04-hosted-opencode-successor-source-v1.json"
)


@pytest.mark.parametrize(
    ("model", "source_path", "tasks", "new_sessions", "credits", "excluded_rank"),
    [
        (
            "qwen38",
            Path("evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json"),
            49,
            196,
            0,
            1,
        ),
        (
            "glm53",
            Path("evals/fleet/configs/glm53-opencode-train100-pass4-v4.json"),
            99,
            394,
            2,
            2,
        ),
    ],
)
def test_successor_owns_only_complete_task_boundaries(
    model: str,
    source_path: Path,
    tasks: int,
    new_sessions: int,
    credits: int,
    excluded_rank: int,
) -> None:
    plan = hosted.build_plan(
        hosted.load_object(source_path), hosted.load_object(SOURCE), model
    )
    hosted.validate_plan(plan)
    assert plan["task_count"] == tasks
    assert plan["new_session_count"] == new_sessions
    assert len(plan["credited_sessions"]) == credits
    assert plan["excluded_tasks"][0]["source_rank"] == excluded_rank
    assert excluded_rank not in {row["source_rank"] for row in plan["tasks"]}
    cells = {
        *( (row["rank"], row["attempt"]) for row in plan["credited_sessions"] ),
        *( (row["rank"], row["attempt"]) for row in plan["attempts"] ),
    }
    assert cells == {
        (rank, attempt)
        for rank in range(1, tasks + 1)
        for attempt in range(1, 5)
    }


def test_source_fence_cannot_be_retried_or_credited() -> None:
    source = hosted.load_object(SOURCE)
    source["runs"]["qwen38"]["fenced_noncreditable"][0]["retry_allowed"] = True
    source["receipt_sha256"] = self_hosted.digest_without(source, "receipt_sha256")
    with pytest.raises(ValueError, match="policy drifted"):
        hosted.build_plan(
            hosted.load_object(
                Path("evals/fleet/configs/qwen38-opencode-train50-pass4-v3.json")
            ),
            source,
            "qwen38",
        )


def test_glm_clean_shard_defers_all_rank1_history_and_fenced_rank2() -> None:
    plan = hosted.build_plan(
        hosted.load_object(
            Path("evals/fleet/configs/glm53-opencode-train100-pass4-v4.json")
        ),
        hosted.load_object(SOURCE),
        "glm53_clean",
    )
    hosted.validate_plan(plan)
    assert plan["task_count"] == 98
    assert plan["new_session_count"] == 392
    assert plan["credited_sessions"] == []
    assert {row["source_rank"] for row in plan["tasks"]} == set(range(3, 101))
    assert {row["source_rank"] for row in plan["excluded_tasks"]} == {1, 2}
    assert plan["execution"]["inventory_policy"] == (
        "conservative_no_same_model_session_for_task_key_v1"
    )


def test_glm_hosted_odd_shard_is_disjoint_from_dedicated_even_shard() -> None:
    plan = hosted.build_plan(
        hosted.load_object(
            Path("evals/fleet/configs/glm53-opencode-train100-pass4-v4.json")
        ),
        hosted.load_object(SOURCE),
        "glm53_hosted_odd",
    )
    hosted.validate_plan(plan)
    assert plan["task_count"] == 49
    assert plan["new_session_count"] == 196
    assert plan["credited_sessions"] == []
    assert {row["source_rank"] for row in plan["tasks"]} == set(range(3, 100, 2))
    assert {row["source_rank"] for row in plan["excluded_tasks"]} == {1, 2}
    assert {row["source_rank"] for row in plan["reserved_tasks"]} == set(
        range(4, 101, 2)
    )


def test_plan_rejects_partial_or_duplicate_cells() -> None:
    plan = hosted.load_object(
        Path("evals/fleet/configs/qwen38-opencode-hosted-complete49-pass4-v5.json")
    )
    plan["attempts"][0]["attempt"] = plan["attempts"][1]["attempt"]
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    with pytest.raises(ValueError, match="Cartesian"):
        hosted.validate_plan(plan)


def test_score_blind_worker_cap_is_fixed_until_new_headroom_review() -> None:
    assert hosted._worker_cap(0) == 1
    assert hosted._worker_cap(4) == 1
    assert hosted._worker_cap(16) == 1
    assert hosted._worker_cap(10_000) == 1
    assert hosted.SCHEDULE[0]["headroom_gate"] == (
        "fixed_at_launch_no_automatic_widening"
    )
    assert all("score" not in key for stage in hosted.SCHEDULE for key in stage)


def test_exact_treatment_inventory_requires_harness_and_tool_digest(monkeypatch) -> None:
    plan = hosted.load_object(
        Path("evals/fleet/configs/glm53-opencode-hosted-complete99-pass4-v5.json")
    )
    task_key = plan["tasks"][0]["task"]["key"]
    exact_metadata = {
        "self_hosted_harness": "opencode-1.18.27",
        "tool_catalog_sha256": plan["execution"]["required_task_tool_catalog_sha256"],
    }
    rows = [
        {"session_id": "exact", "model": "glm-5.3", "metadata": exact_metadata},
        {"session_id": "other-harness", "model": "glm-5.3", "metadata": {}},
        {"session_id": "other-model", "model": "qwen3.8-27b", "metadata": exact_metadata},
    ]
    monkeypatch.setattr(
        self_hosted,
        "_task_sessions",
        lambda _client, key: rows if key == task_key else [],
    )
    exact = hosted._exact_treatment_sessions(object(), plan, task_key)
    assert [row["session_id"] for row in exact] == ["exact"]


def test_generated_plans_are_reproducible() -> None:
    source = hosted.load_object(SOURCE)
    for model, source_name, generated_name in (
        (
            "qwen38",
            "qwen38-opencode-train50-pass4-v3.json",
            "qwen38-opencode-hosted-complete49-pass4-v5.json",
        ),
        (
            "glm53",
            "glm53-opencode-train100-pass4-v4.json",
            "glm53-opencode-hosted-complete99-pass4-v5.json",
        ),
    ):
        expected = hosted.load_object(Path("evals/fleet/configs") / generated_name)
        actual = hosted.build_plan(
            hosted.load_object(Path("evals/fleet/configs") / source_name), source, model
        )
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
