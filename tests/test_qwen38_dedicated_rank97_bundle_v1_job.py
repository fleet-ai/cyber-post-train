from pathlib import Path

from evals.fleet import qwen38_dedicated_rank97_bundle_v1_job as job

ROOT = Path(__file__).resolve().parents[1]


def test_rank97_bundle_job_is_one_nonpreempting_cpu_controller() -> None:
    value = job.render(ROOT)
    assert value["metadata"]["name"] == job.NAME
    assert value["spec"]["backoffLimit"] == 0
    pod = value["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "fleet-serve-low"
    assert pod["preemptionPolicy"] == "Never"
    assert pod["containers"][0]["resources"]["requests"]["cpu"] == "100m"
    env = {row["name"]: row.get("value") for row in pod["containers"][0]["env"]}
    assert env["QWEN_RANK97_RELEASE_PATH"] == "/bootstrap/release.json"
    assert "qwen38_dedicated_rank97_bundle_v1.py" in job.BOOTSTRAP
    assert "qwen-hosted-generation19-qwen-a-v4.json" in job.BOOTSTRAP


def test_rank97_run_script_has_no_single_attempt_gate() -> None:
    value = (ROOT / job.RUN_SCRIPT).read_text()
    assert "QWEN_DEDICATED_ATTEMPT" not in value
    assert "qwen38_dedicated_rank97_bundle_v1" in value
    assert "QWEN_RANK97_RELEASE_PATH" in value


def test_rank97_configmap_package_is_closed(tmp_path: Path) -> None:
    release = tmp_path / "release.json"
    release.write_text("{}")
    data = job.configmap_data(ROOT, release)
    assert set(data) == {
        "self_hosted.py",
        "exact_pass4_crypto.py",
        "exact_pass4_universe.py",
        "legacy.py",
        "lane.py",
        "parity.json",
        "campaign.json",
        "source.json",
        "Dockerfile.opencode",
        "fixed_proxy.py",
        "run.sh",
        "release.json",
    }
    assert "from evals.fleet import qwen38_dedicated_scored_canary_v1" in data["lane.py"]
    assert "fleet-qwen-generation19-hosted-bulk-plan-v4" in data["source.json"]
