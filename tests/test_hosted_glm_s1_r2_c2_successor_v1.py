from pathlib import Path

from evals.fleet import hosted_glm_s1_r2_c2_package_v1 as package
from evals.fleet import hosted_glm_s1_r2_c2_release_package_v1 as release_package
from evals.fleet import hosted_glm_s1_r2_c2_successor_v1 as successor

ROOT = Path(__file__).resolve().parents[1]


def test_successor_is_one_wholly_unstarted_complete_task():
    plan = successor.validate_all(ROOT)
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (2, 1), (2, 2), (2, 3), (2, 4)
    ]
    assert plan["new_session_count"] == 4
    assert plan["partition"]["whole_task_rank"] == 2
    assert plan["partition"]["rank1_attempt4_blocked_and_excluded"] is True
    assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert plan["harness"]["version"] == "1.18.27"


def test_held_controller_and_release_preserve_nonpreemption():
    held = package.render(ROOT)
    assert held["launch_authorized"] is False
    job = held["objects"]["items"][1]
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    released = release_package.render(ROOT)
    observer = released["objects"]["items"][1]
    assert observer["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert released["launch_authorized"] is True


def test_packages_install_exact_successor_entrypoints():
    controller_data = package.render(ROOT)["objects"]["items"][0]["data"]
    assert "successor.py:hosted_glm_s1_r2_c2_successor_v1.py" in controller_data["run.sh"]
    assert "successor_runtime.py:hosted_glm_s1_r2_c2_runtime_v1.py" in controller_data["run.sh"]
    release_data = release_package.render(ROOT)["objects"]["items"][0]["data"]
    assert "successor_release.py:hosted_glm_s1_r2_c2_release_v1.py" in release_data["run.sh"]
