from pathlib import Path

from evals.fleet import glm53_dedicated_v20_a2_bootstrap_v1 as bootstrap
from evals.fleet import glm53_dedicated_v20_a2_package_v1 as controller_package
from evals.fleet import glm53_dedicated_v20_a2_release_package_v1 as release_package
from evals.fleet import configmap_python_closure_v1 as closure

ROOT = Path(__file__).resolve().parents[1]


def test_release_package_contains_complete_transitive_python_closure():
    data = release_package.render(ROOT)["objects"]["items"][0]["data"]
    for name in ("base_canary.py", "base_runtime.py", "prior_canary.py", "prior_runtime.py", "dedicated_canary.py", "dedicated_runtime.py", "old_release.py", "prior_release.py", "release.py"):
        assert data[name]


def test_bootstrap_stages_release_from_sfs_in_own_pod():
    built = bootstrap.render(ROOT)
    assert built["launch_authorized"] is False
    assert bootstrap.JOB_NAME.endswith("bootstrap-v3")
    job = built["objects"]["items"][2]
    pod = job["spec"]["template"]["spec"]
    stager = pod["initContainers"][0]
    assert stager["name"] == "stage-evidence"
    assert str(bootstrap.RELEASE_PATH) in stager["args"][0]
    assert str(bootstrap.PREDECESSOR_PATH) in stager["args"][0]
    evidence = next(v for v in pod["volumes"] if v["name"] == "evidence")
    assert evidence == {"name": "evidence", "emptyDir": {}}
    assert pod["preemptionPolicy"] == "Never"


def test_controller_package_carries_and_installs_v19_canary_dependency():
    data = controller_package.render(ROOT)["objects"]["items"][0]["data"]
    assert "from evals.fleet import glm53_dedicated_v19_a2_v1 as prior" in data["dedicated_canary.py"]
    assert data["prior_canary.py"]
    assert "prior_canary.py:glm53_dedicated_v19_a2_v1.py" in data["run.sh"]
    assert "dedicated_canary.py:glm53_dedicated_v20_a2_v1.py" in data["run.sh"]


def test_controller_package_transitive_eval_imports_are_installed():
    data = controller_package.render(ROOT)["objects"]["items"][0]["data"]
    graph = closure.validate(
        data, dynamic_import_allowlist=controller_package.DORMANT_BUILD_ONLY_IMPORTS
    )
    assert "glm53_dedicated_v20_a2_v1" in graph
    assert "glm53_dedicated_v19_a2_v1" in graph["glm53_dedicated_v20_a2_v1"]
