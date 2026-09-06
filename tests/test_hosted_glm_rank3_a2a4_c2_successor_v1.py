from pathlib import Path

from evals.fleet import hosted_glm_rank3_a2a4_c2_package_v1 as package
from evals.fleet import hosted_glm_rank3_a2a4_c2_successor_v1 as successor
from evals.fleet import self_hosted


def test_successor_is_exact_rank3_tail() -> None:
    plan = successor.validate_all(Path.cwd())[successor.CONTROLLER]
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [
        (3, 2),
        (3, 3),
        (3, 4),
    ]
    assert plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert plan["execution"]["preemption_policy"] == "Never"
    assert plan["plan_sha256"] == self_hosted.digest_without(plan, "plan_sha256")


def test_release_package_is_create_once_and_nonpreempting() -> None:
    value = package.render_release(Path.cwd())
    configmap, job = value["objects"]["items"]
    assert configmap["immutable"] is True
    assert configmap["metadata"]["name"] == package.release.CONFIGMAP_NAME
    assert job["metadata"]["name"] == package.release.JOB_NAME
    assert job["spec"]["backoffLimit"] == 0
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    assert job["spec"]["template"]["spec"]["priorityClassName"] == "fleet-infra-quiet"
    assert "hosted_glm_rank3_a2a4_c2_release_v1" in configmap["data"]["run.sh"]
