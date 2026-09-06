from pathlib import Path

from evals.fleet import glm53_dedicated_v22_a2_package_v1 as package
from evals.fleet import glm53_dedicated_v22_a2_release_package_v1 as release
from evals.fleet import glm53_dedicated_v22_a2_v1 as controller

ROOT = Path(__file__).resolve().parents[1]


def test_v22_controller_exports_route_evidence_validator() -> None:
    assert callable(controller._validate_evidence)


def test_v22_controller_package_binds_only_v22_runtime_identity() -> None:
    value = package.render(ROOT)
    assert value["status"] == "READY_HELD"
    assert value["launch_authorized"] is False
    assert len(value["reserved_cell_ids"]) == 4
    assert len(value["reserved_execution_ids"]) == 4
    source, job = value["objects"]["items"]
    assert source["metadata"]["name"].startswith("chris-glm53-dedicated-v22-")
    assert job["metadata"]["name"].startswith("chris-glm53-dedicated-v22-")
    env = {
        row["name"]: row.get("value")
        for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["DEDICATED_SERVICE_ORIGIN"] == package.ORIGIN
    assert "v21-head-svc" not in env["DEDICATED_SERVICE_ORIGIN"]


def test_v22_release_package_binds_current_origin_and_controller() -> None:
    value = release.render(ROOT)
    configmap, job = value["objects"]["items"]
    assert value["launch_authorized"] is True
    assert configmap["metadata"]["name"].startswith("chris-glm53-dedicated-v22-")
    assert job["metadata"]["name"].startswith("chris-glm53-dedicated-v22-")
    env = {
        row["name"]: row.get("value")
        for row in job["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["DEDICATED_SERVICE_ORIGIN"] == package.ORIGIN
    assert "glm53_dedicated_v22_a2_release_v1" in configmap["data"]["run.sh"]
