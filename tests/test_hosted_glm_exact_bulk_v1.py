from pathlib import Path

from evals.fleet import hosted_glm_exact_bulk_v1 as bulk

ROOT = Path(__file__).resolve().parents[1]


def test_exact_394_cell_reviewed_partition_is_disjoint() -> None:
    plans = bulk.validate_all(ROOT)
    assert [plan["new_session_count"] for plan in plans.values()] == [94, 100, 100, 100]
    cells = {
        (row["selection_rank"], row["attempt"])
        for plan in plans.values()
        for row in plan["attempts"]
    }
    assert len(cells) == 394
    assert all((12, attempt) not in cells for attempt in range(1, 5))
    assert (1, 1) not in cells
    assert (13, 1) not in cells
    assert all((1, attempt) in cells for attempt in (2, 3, 4))


def test_treatment_and_bounded_execution_are_frozen() -> None:
    for plan in bulk.validate_all(ROOT).values():
        assert plan["model"]["served_id"] == "glm-5.3"
        assert plan["harness"]["version"] == "1.18.27"
        assert plan["harness"]["context_window_size"] == 262144
        assert plan["harness"]["max_output_tokens"] == 32768
        assert plan["harness"]["compaction_headroom_tokens"] == 20000
        assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 4
        assert plan["execution"]["priority_class"] == "fleet-serve-low"
        assert plan["execution"]["preemption_policy"] == "Never"
        assert plan["launch_authorized"] is False
