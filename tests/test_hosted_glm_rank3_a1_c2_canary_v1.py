from pathlib import Path

from evals.fleet import hosted_glm_rank3_a1_c2_canary_v1 as canary
from evals.fleet import self_hosted


ROOT = Path(__file__).resolve().parents[1]


def test_rank3_canary_is_held_one_cell_on_a_wholly_unstarted_task_boundary() -> None:
    plan = canary.validate_all(ROOT)[canary.CONTROLLER]
    assert plan["launch_authorized"] is False
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (3, 1)
    ]
    boundary = plan["partition"]["whole_task_cells"]
    assert [(row["attempt"], row["cell_id"]) for row in boundary] == [
        (1, "sha256:7568e59b6949088e67f3a98a566a22927640771abfd331ce5094364eb7cbac04"),
        (2, "sha256:57f0ec78f93760d980f31df4dac3fc6daf38f2849ee2949d943e3a41f4a61e2d"),
        (3, "sha256:85d8da4373e2dbd30100aca88b08b884f3ef65febf71f77060ae67246df37c8b"),
        (4, "sha256:f5b26b9037dd0f02ad2bb22401c9d0be8b9ab4d98641fdb091cc5f1bf6cddb44"),
    ]
    assert plan["partition"]["remaining_attempts_held"] == [2, 3, 4]
    assert plan["plan_sha256"] == self_hosted.digest_without(plan, "plan_sha256")


def test_rank3_canary_binds_exact_live_s2_runner_and_failure_evidence() -> None:
    plan = canary.validate_all(ROOT)[canary.CONTROLLER]
    runtime = plan["known_good_s2_runtime"]
    assert runtime["uid"] == "1b9b171e-5ea8-4f9d-b014-e4e0bd106789"
    assert runtime["immutable"] is True
    assert runtime["critical_data_sha256"]["engine.py"] == (
        "sha256:06f9857c99ee1773d505ea60c06c335b373d91b7dbf267d2074bbdc39c4aa20f"
    )
    assert plan["failure_authority"]["commit"] == "640f430"
    assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
    assert plan["execution"]["preemption_policy"] == "Never"
