import json
from pathlib import Path

from evals.fleet import qwen38_dedicated_rank3_a2_g22_v1 as lane
from evals.fleet import qwen38_dedicated_rank3_a2_g22_v1_job as job
from evals.fleet import self_hosted

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


def test_g21_preplan_failure_is_retry_safe_and_sealed() -> None:
    path = ROOT / (
        "docs/evidence/qwen38-study/"
        "2026-09-05-qwen38-dedicated-rank3-a2-g21-package-tombstone-v1.json"
    )
    value = json.loads(path.read_text())
    assert value["status"] == "RETRY_SAFE_INFRA_FAILURE"
    assert value["pod_phase_when_released"] == "Pending"
    assert value["output_root_created"] is False
    assert value["claim_created"] is False
    assert value["model_call_started"] is False
    assert value["scored_session_created"] is False
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
