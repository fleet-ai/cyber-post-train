import json
from pathlib import Path

from evals.fleet import qwen38_dedicated_dp8_canary_v1 as lane
from evals.fleet import qwen38_dedicated_dp8_canary_v1_job as job
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_dp8_canary_is_exact_held_and_parity_bound() -> None:
    plan = lane.build_plan(ROOT)
    assert plan["item"] == {
        "cell_id": lane.CELL_ID,
        "execution_id": lane.EXECUTION_ID,
        "execution_generation": 22,
        "run_id": lane.RUN_ID,
        "selection_rank": 4,
        "attempt": 2,
        "task_version_id": lane.TASK_VERSION_ID,
    }
    assert plan["launch_authorized"] is False
    assert plan["config"]["serving"]["serving_block"] == lane.SERVING_BLOCK
    assert plan["plan_sha256"] == self_hosted.digest_without(plan, "plan_sha256")
    assert lane._validate_parity(ROOT)["receipt_sha256"] == lane.PARITY_SHA256


def test_dp8_release_is_exact(tmp_path: Path, monkeypatch) -> None:
    value = {
        "schema_version": lane.RELEASE_SCHEMA,
        "status": "RELEASED_ONE_CANARY",
        "cell_id": lane.CELL_ID,
        "execution_id": lane.EXECUTION_ID,
        "selection_rank": 4,
        "attempt": 2,
        "serving_block": lane.SERVING_BLOCK,
        "parity_receipt_sha256": lane.PARITY_SHA256,
        "rayjob_uid": lane.RAYJOB_UID,
        "head_pod_uid": lane.HEAD_POD_UID,
        "service_uid": lane.SERVICE_UID,
        "fresh_global_ledger_clear": True,
        "fresh_authoritative_sessions_clear": True,
        "claim_clear": True,
        "output_root_clear": True,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = tmp_path / "release.json"
    path.write_text(json.dumps(value))
    monkeypatch.setenv("QWEN_DP8_RELEASE_PATH", str(path))
    assert lane.released_plan(ROOT)["launch_authorized"] is True


def test_dp8_job_is_create_once_cpu_nonpreempting() -> None:
    value = job.render(ROOT)
    assert value["metadata"]["name"] == job.NAME
    assert value["spec"]["template"]["spec"]["priorityClassName"] == "fleet-serve-low"
    assert value["spec"]["template"]["spec"]["preemptionPolicy"] == "Never"
    container = value["spec"]["template"]["spec"]["containers"][0]
    release = [row for row in container["env"] if row["name"] == "QWEN_DP8_RELEASE_PATH"]
    assert release == [{"name": "QWEN_DP8_RELEASE_PATH", "value": "/bootstrap/dp8-release.json"}]
