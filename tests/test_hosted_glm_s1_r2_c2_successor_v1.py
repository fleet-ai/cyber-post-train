from pathlib import Path

from evals.fleet import hosted_glm_s1_r2_c2_package_v1 as package
from evals.fleet import hosted_glm_s1_r2_c2_release_package_v1 as release_package
from evals.fleet import hosted_glm_s1_r2_c2_runtime_v1 as runtime
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


def test_authorized_package_checks_static_release_but_runtime_rebinds_live_plan(tmp_path):
    plan = successor.validate_all(ROOT)
    receipt = {
        "schema_version": "fleet-hosted-glm-s1-r2-c2-successor-release-v1",
        "status": "CLEAR",
        "successor_job": successor.JOB_NAME,
        "successor_configmap": successor.CONFIGMAP_NAME,
        "plan_sha256": "runtime-only-plan-sha",
        "cell_ids": ["runtime-cell-1", "runtime-cell-2", "runtime-cell-3", "runtime-cell-4"],
        "execution_ids": ["runtime-exec-1", "runtime-exec-2", "runtime-exec-3", "runtime-exec-4"],
        "selection_rank": 2,
        "attempts": [1, 2, 3, 4],
        "whole_task_boundary": True,
        "s2_job_uid": "faf01255-0696-43e6-a47f-67802184986e",
        "s2_pod_uid": "72ce8163-fb2d-40ac-92da-da0dd578658d",
        "active_scored_lease_slots_before_create": 1,
        "maximum_scored_streams": 2,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "checked_immediately_before_create": True,
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    from evals.fleet import self_hosted
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    release_path = tmp_path / "RELEASE.json"
    release_path.write_bytes(self_hosted.canonical_json(receipt))
    assert package.render(ROOT, release_path=release_path)["launch_authorized"] is True
    runtime.validate_release_static(receipt)
    try:
        runtime.validate_release(plan, receipt)
    except RuntimeError as exc:
        assert "runtime plan drifted" in str(exc)
    else:
        raise AssertionError("live runtime plan binding was not enforced")
