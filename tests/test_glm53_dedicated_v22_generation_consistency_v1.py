from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v22_generation_canary_job_v1 as job
from evals.fleet import glm53_dedicated_v22_generation_consistency_v1 as gate

ROOT = Path(__file__).resolve().parents[1]


def test_generation_fixture_binds_all_current_identities() -> None:
    value = gate.qualification_fixture()
    gate.validate(value)
    assert value["server"]["api_run_id"] == value["controller"]["api_run_id"]
    assert value["server"]["service_origin"] == value["release"]["service_origin"]
    assert value["release"]["controller_job_name"] == value["controller"]["job_name"]
    assert value["controller"]["server_run_dir"] == value["server"]["run_dir"]


@pytest.mark.parametrize(
    ("section", "field", "stale"),
    [
        ("server", "run_dir", "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v21"),
        ("release", "service_origin", "http://ft-run-v21-head-svc:8000"),
        ("controller", "lease_root", "/mnt/sfs/endpoint-leases/opencode11827-dedicated-v20-v1"),
        ("controller", "job_name", "chris-glm53-dedicated-v21-r051-a2-canary-v1"),
    ],
)
def test_generation_fixture_rejects_stale_runtime_token(
    section: str, field: str, stale: str
) -> None:
    value = copy.deepcopy(gate.qualification_fixture())
    value[section][field] = stale
    with pytest.raises(RuntimeError, match="generation binding drifted"):
        gate.validate(value)


def test_cpu_canary_is_projected_create_once_and_non_scoring() -> None:
    package = job.render(ROOT)
    configmap, rendered_job = package["objects"]["items"]
    assert package["gpu_requests"] == 0
    assert package["scoring_calls"] == 0
    assert configmap["immutable"] is True
    pod = rendered_job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-infra-quiet"
    assert pod["preemptionPolicy"] == "Never"
    assert "nvidia.com/gpu" not in json.dumps(pod)
    assert pod["volumes"][0]["configMap"]["name"] == job.NAME
    assert {row["name"] for row in pod["volumes"] if "emptyDir" in row} == {"work"}
    container = pod["containers"][0]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert {row["mountPath"] for row in container["volumeMounts"]} >= {"/work", "/mnt/sfs"}
    assert 'root="/work/root"' in container["args"][0]
    assert "mktemp" not in container["args"][0]
    data = package["objects"]["items"][0]["data"]
    assert "self_hosted.py" not in data
    assert "from evals.fleet" not in data["gate.py"]


def test_projected_gate_imports_and_runs_with_stdlib_only(tmp_path: Path) -> None:
    data = job.render(ROOT)["objects"]["items"][0]["data"]
    projected = tmp_path / "projected"
    projected.mkdir()
    (projected / "gate.py").write_text(data["gate.py"])
    (projected / "fixture.json").write_text(data["fixture.json"])
    output = tmp_path / "result" / "CANARY.json"
    script = (
        "import importlib.util;from pathlib import Path;"
        f"s=importlib.util.spec_from_file_location('gate',{str(projected / 'gate.py')!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"m.run_canary(Path({str(projected / 'fixture.json')!r}),Path({str(output)!r}))"
    )
    env = {
        "PATH": os.environ["PATH"],
        "JOB_UID": "33333333-3333-4333-8333-333333333331",
        "POD_UID": "33333333-3333-4333-8333-333333333332",
    }
    subprocess.run([sys.executable, "-I", "-c", script], check=True, env=env)
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "PASSED_FULL_GENERATION_BINDING_PATH"
