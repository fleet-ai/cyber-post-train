from pathlib import Path

from evals.fleet import glm53_dedicated_v22_concurrency_qualification_v1 as held

ROOT = Path(__file__).resolve().parents[1]


def test_concurrency_qualification_is_score_free_and_fail_closed() -> None:
    value = held.render(ROOT)
    assert value["launch_authorized"] is False
    assert [phase["concurrency"] for phase in value["phases"]] == [1, 2, 4]
    assert value["treatment"]["tools"] == ["bash", "submit_report"]
    assert value["treatment"]["context_window_size"] == 262144
    assert value["score_free_boundary"]["fleet_task_instance_calls"] == 0
    assert value["score_free_boundary"]["fleet_session_calls"] == 0
    assert value["score_free_boundary"]["verifier_calls"] == 0
    assert value["score_free_boundary"]["scoring_calls"] == 0
    assert value["fail_closed_ramp"]["never_overlap_scored_controller"] is True
    assert value["launch_prerequisites"]["rank51_attempt2_terminal_accepted"] is True
    assert value["scored_concurrency_change_authorized"] is False
