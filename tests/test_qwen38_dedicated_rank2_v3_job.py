from pathlib import Path

from evals.fleet import qwen38_dedicated_rank2_v3_job as job

ROOT = Path(__file__).resolve().parents[1]


def test_rank2_v3_job_is_create_once_nonpreempting_and_exact() -> None:
    value = job.render(ROOT)
    assert value["metadata"]["name"] == ("chris-cyber-q38-opencode11827-ded-tp1-r002-a2-v3")
    assert value["spec"]["backoffLimit"] == 0
    pod = value["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["volumes"][0]["configMap"]["name"] == value["metadata"]["name"]
    assert pod["containers"][0]["args"] == [job.BOOTSTRAP]
    assert {row["name"]: row.get("value") for row in pod["containers"][0]["env"]}[
        "QWEN_DEDICATED_ATTEMPT"
    ] == "2"
    assert "qwen38_dedicated_rank2_v3.py" in job.BOOTSTRAP
    assert "qwen38-dedicated-v3-actual-opencode-parity.json" in job.BOOTSTRAP


def test_rank2_v3_job_renders_fresh_attempt_identity() -> None:
    value = job.render(ROOT, 3)
    assert value["metadata"]["name"] == ("chris-cyber-q38-opencode11827-ded-tp1-r002-a3-v3")
    pod = value["spec"]["template"]["spec"]
    assert pod["volumes"][0]["configMap"]["name"] == value["metadata"]["name"]
    assert {row["name"]: row.get("value") for row in pod["containers"][0]["env"]}[
        "QWEN_DEDICATED_ATTEMPT"
    ] == "3"
