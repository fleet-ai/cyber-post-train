from __future__ import annotations

from pathlib import Path

from evals.fleet import hosted_glm_s2_c2_canary_v1 as c2
from evals.fleet import hosted_glm_s2_c2_package_v1 as package
from evals.fleet import hosted_glm_s2_c2_release_package_v1 as release_package

ROOT = Path(__file__).resolve().parents[1]


def test_c2_canary_is_one_exact_held_cell_with_shared_lease() -> None:
    plan = c2.validate_all(ROOT)[c2.CONTROLLER]
    assert [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]] == [(26, 1)]
    assert plan["new_session_count"] == 1
    assert plan["launch_authorized"] is False
    assert plan["execution"]["endpoint_lease"] == {
        "lease_root": str(c2.LEASE_ROOT),
        "endpoint_key": c2.LEASE_ENDPOINT_KEY,
        "maximum_streams": 2,
    }
    assert plan["concurrency_treatment"]["pool_with_concurrency_one_without_review"] is False


def test_c2_scored_package_is_held_without_release() -> None:
    rendered = package.render(ROOT)
    configmap, job = rendered["objects"]["items"]
    assert rendered["launch_authorized"] is False
    assert configmap["immutable"] is True
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    required = {
        line.strip().rstrip("\\").strip().split(":", 1)[0]
        for line in configmap["data"]["run.sh"].splitlines()
        if ":" in line
        and line.strip().rstrip("\\").strip().split(":", 1)[0].endswith(
            (".py", ".json")
        )
    }
    assert required <= set(configmap["data"])


def test_c2_release_observer_is_also_held_until_s1_accepts() -> None:
    rendered = release_package.render(ROOT)
    configmap, job = rendered["objects"]["items"]
    assert rendered["launch_authorized"] is False
    assert configmap["immutable"] is True
    assert job["metadata"]["annotations"]["cyber-post-train.fleet.ai/launch-authorized"] == "false"
    assert job["spec"]["backoffLimit"] == 0
