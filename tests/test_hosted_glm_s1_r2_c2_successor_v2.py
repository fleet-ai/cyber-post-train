from pathlib import Path

from evals.fleet import hosted_glm_s1_r2_c2_package_v2 as package
from evals.fleet import hosted_glm_s1_r2_c2_release_package_v2 as release_package
from evals.fleet import hosted_glm_s1_r2_c2_release_package_v3 as release_package_v3
from evals.fleet import hosted_glm_s1_r2_c2_successor_v2 as successor

ROOT = Path(__file__).resolve().parents[1]


def test_v2_is_fresh_complete_rank2_and_retains_max_two_lease():
    plan = successor.validate_all(ROOT)
    assert plan["job_name"].endswith("successor-v2")
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (2, 1), (2, 2), (2, 3), (2, 4)
    ]
    assert plan["partition"]["v1_preclaim_failure_excluded"] is True
    assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2


def test_v2_packages_are_create_once_nonpreempting_and_held_until_release():
    held = package.render(ROOT)
    assert held["launch_authorized"] is False
    job = held["objects"]["items"][1]
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    observer = release_package.render(ROOT)["objects"]["items"][1]
    assert observer["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_v2_runtime_accepts_zero_or_one_preexisting_slot_but_never_two():
    source = (ROOT / "evals/fleet/hosted_glm_s1_r2_c2_release_v2.py").read_text()
    runtime = (ROOT / "evals/fleet/hosted_glm_s1_r2_c2_runtime_v2.py").read_text()
    assert "occupied not in (0, 1)" in source
    assert "_active_lease_slots() > 1" in runtime
    assert '"maximum_streams": 2' in (ROOT / "evals/fleet/hosted_glm_s1_r2_c2_successor_v1.py").read_text()


def test_fresh_release_installs_the_full_immutable_config_set():
    data = release_package_v3.render(ROOT)["objects"]["items"][0]["data"]
    script = data["run.sh"]
    assert "mkdir -p \"$ROOT/evals/fleet/configs\"" in script
    assert "campaign.json:q38-glm53-exact-easiest100-pass4-campaign-v1.json" in script
    assert "successor_release_v2.py:hosted_glm_s1_r2_c2_release_v2.py" in script
