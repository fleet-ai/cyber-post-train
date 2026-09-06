from __future__ import annotations

import copy
import json
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
