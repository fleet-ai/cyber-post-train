from pathlib import Path

from evals.fleet import qwen38_dedicated_rank3_v3_job as job

ROOT = Path(__file__).resolve().parents[1]


def test_rank3_job_is_nonpreempting_and_release_gated() -> None:
    value = job.render(ROOT, 1)
    assert value["metadata"]["name"] == ("chris-cyber-q38-opencode11827-ded-tp1-r003-a1-g21-v2")
    assert value["spec"]["backoffLimit"] == 0
    pod = value["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    env = {row["name"]: row.get("value") for row in pod["containers"][0]["env"]}
    assert env["QWEN_DEDICATED_ATTEMPT"] == "1"
    assert env["QWEN_RANK3_RELEASE_PATH"] == "/bootstrap/rank3-release.json"
    assert "qwen38_dedicated_rank3_v3.py" in job.BOOTSTRAP
    assert "qwen38-dedicated-tp1-b-v2-actual-opencode-parity.json" in job.BOOTSTRAP
    assert "qwen38-dedicated-v3-actual-opencode-parity.json" not in job.BOOTSTRAP


def test_rank3_run_script_changes_only_exact_controller_module() -> None:
    source = (
        'case "$QWEN_DEDICATED_ATTEMPT" in 2|3|4) ;; *) exit 64 ;; esac\n'
        "exec python -m evals.fleet.qwen38_dedicated_rank2_v3\n"
    )
    assert job.rank3_run_script(source) == (
        'case "$QWEN_DEDICATED_ATTEMPT" in 1|2|3|4) ;; *) exit 64 ;; esac\n'
        "exec python -m evals.fleet.qwen38_dedicated_rank3_v3\n"
    )
