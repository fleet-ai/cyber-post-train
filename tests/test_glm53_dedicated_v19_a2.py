from pathlib import Path

from evals.fleet import glm53_dedicated_v19_a2_bootstrap_v1 as bootstrap
from evals.fleet import glm53_dedicated_v19_a2_package_v1 as package

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path("/private/tmp/glm53-v19-a2-release.json")


def _configmap_references(pod: dict) -> set[str]:
    refs: set[str] = set()
    for volume in pod.get("volumes", []):
        if "configMap" in volume:
            refs.add(volume["configMap"]["name"])
    for container in [*pod.get("initContainers", []), *pod.get("containers", [])]:
        for source in container.get("envFrom", []):
            if "configMapRef" in source:
                refs.add(source["configMapRef"]["name"])
        for env in container.get("env", []):
            ref = env.get("valueFrom", {}).get("configMapKeyRef")
            if ref:
                refs.add(ref["name"])
    return refs


def test_bootstrap_v2_resolves_every_configmap_reference() -> None:
    built = bootstrap.render(ROOT, RELEASE)
    configmaps = {
        item["metadata"]["name"]
        for item in built["objects"]["items"]
        if item["kind"] == "ConfigMap"
    }
    job = next(item for item in built["objects"]["items"] if item["kind"] == "Job")
    refs = _configmap_references(job["spec"]["template"]["spec"])
    assert refs == configmaps
    assert bootstrap.JOB_NAME.endswith("bootstrap-v2")
    assert job["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"


def test_attempt2_package_is_single_cell_and_scoring_held() -> None:
    built = package.render(ROOT)
    assert built["launch_authorized"] is False
    assert built["reserved_cell_ids"][1].startswith("sha256:")
