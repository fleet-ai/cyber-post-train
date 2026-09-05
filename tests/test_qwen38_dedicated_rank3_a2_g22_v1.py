from pathlib import Path

from evals.fleet import qwen38_dedicated_rank3_a2_g22_v1 as lane
from evals.fleet import qwen38_dedicated_rank3_a2_g22_v1_job as job

ROOT = Path(__file__).resolve().parents[1]


def test_g22_a2_is_fresh_exact_and_held() -> None:
    plan = lane.build_plan(ROOT)
    assert plan["item"]["selection_rank"] == 3
    assert plan["item"]["attempt"] == 2
    assert plan["item"]["execution_generation"] == 22
    assert plan["item"]["cell_id"] == lane.CELL_ID
    assert plan["item"]["execution_id"] == lane.EXECUTION_ID
    assert plan["item"]["run_id"] == lane.RUN_ID
    assert plan["launch_authorized"] is False


def test_g22_package_uses_exact_new_runner(tmp_path: Path) -> None:
    release = tmp_path / "release.json"
    release.write_text("{}")
    data = job.configmap_data(ROOT, release)
    assert "qwen38_dedicated_rank3_a2_g22_v1" in data["run.sh"]
    assert "QWEN_DEDICATED_ATTEMPT" not in data["run.sh"]
    assert set(data) == {
        "self_hosted.py",
        "exact_pass4_crypto.py",
        "exact_pass4_universe.py",
        "legacy.py",
        "prior.py",
        "lane.py",
        "parity.json",
        "rank3-release.json",
        "campaign.json",
        "selection.json",
        "source.json",
        "Dockerfile.opencode",
        "fixed_proxy.py",
        "run.sh",
        "release.json",
    }
    rendered = job.render(ROOT)
    assert rendered["metadata"]["name"] == job.NAME
    assert rendered["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
