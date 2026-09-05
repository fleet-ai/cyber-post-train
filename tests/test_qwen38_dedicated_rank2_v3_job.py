from pathlib import Path

from evals.fleet import qwen38_dedicated_rank2_v3_job as job

ROOT = Path(__file__).resolve().parents[1]


def test_rank2_v3_job_is_create_once_nonpreempting_and_exact() -> None:
    value = job.render(ROOT)
    assert value["metadata"]["name"] == job.NAME
    assert value["spec"]["backoffLimit"] == 0
    pod = value["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["volumes"][0]["configMap"]["name"] == job.NAME
    assert pod["containers"][0]["args"] == [job.BOOTSTRAP]
    assert "qwen38_dedicated_rank2_v3.py" in job.BOOTSTRAP
    assert "qwen38-dedicated-v3-actual-opencode-parity.json" in job.BOOTSTRAP
