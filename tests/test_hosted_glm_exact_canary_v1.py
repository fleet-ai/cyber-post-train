from pathlib import Path

import yaml

from evals.fleet import hosted_glm_exact_canary_package_v1 as package
from evals.fleet import hosted_glm_exact_canary_v1 as canary

ROOT = Path(__file__).resolve().parents[1]


def test_canary_is_exact_rank1_attempt1_and_disjoint_from_dedicated_rank12() -> None:
    canary._validate_parity(ROOT)
    authority = canary.bulk.validate_all(ROOT)[canary.CONTROLLER]
    item = canary._item(authority)
    assert (item["selection_rank"], item["attempt"]) == (1, 1)
    assert item["cell_id"].startswith("sha256:")
    assert item["execution_id"].startswith("sha256:")
    assert all(row["selection_rank"] != 12 for row in authority["attempts"])


def test_package_is_small_create_once_nonpreempting_cpu_job() -> None:
    rendered = package.render(ROOT)
    assert rendered["configmap_json_bytes"] < 900_000
    configmap, job = rendered["objects"]["items"]
    assert configmap["metadata"]["name"] == canary.CONFIGMAP_NAME
    assert job["metadata"]["name"] == canary.JOB_NAME
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    assert pod["preemptionPolicy"] == "Never"
    assert pod["priorityClassName"] == "fleet-train-high"
    assert pod["nodeSelector"]["workload"] == "fleetai-training-ng-cpu"
    assert not any("nvidia.com/gpu" in str(row) for row in pod.get("containers", []))
    manifest = ROOT / "evals/fleet/cluster/hosted-glm-exact-r001-a1-canary-v1.yaml"
    yaml.safe_load(manifest.read_text())
