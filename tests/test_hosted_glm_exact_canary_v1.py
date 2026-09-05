import subprocess
import sys
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
    assert configmap["immutable"] is True
    assert "inventory.py" in configmap["data"]
    assert "exact_pass4_task_inventory.py" in configmap["data"]["run.sh"]
    assert job["metadata"]["name"] == canary.JOB_NAME
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    assert pod["preemptionPolicy"] == "Never"
    # The admission controller derives PreemptLowerPriority for fleet-train-high
    # and rejects an explicit Never override before Pod creation.
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["nodeSelector"]["workload"] == "fleetai-training-ng-cpu"
    assert not any("nvidia.com/gpu" in str(row) for row in pod.get("containers", []))
    evaluator = next(row for row in pod["containers"] if row["name"] == "evaluator")
    startup = evaluator["args"][0]
    assert "apt-get update" in startup
    assert "docker.io=20.10.24+dfsg1-1+deb12u1+b6" in startup
    assert startup.index("docker.io=") < startup.index("exec /bin/bash /bootstrap/run.sh")
    manifest = ROOT / "evals/fleet/cluster/hosted-glm-exact-r001-a1-canary-v1.yaml"
    yaml.safe_load(manifest.read_text())


def test_exact_bundle_imports_in_isolated_tree(tmp_path: Path) -> None:
    configmap = package.render(ROOT)["objects"]["items"][0]
    data = configmap["data"]
    for name, relative in package.INSTALL_PATHS.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(data[name])
    (tmp_path / "evals/__init__.py").touch()
    (tmp_path / "evals/fleet/__init__.py").touch()
    code = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        "from evals.fleet import exact_pass4_task_inventory; "
        "from evals.fleet import hosted_glm_exact_canary_v1 as c; "
        f"root=Path({str(tmp_path)!r}); c._validate_parity(root); c.bulk.validate_all(root)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
