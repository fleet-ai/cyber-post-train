import json
from pathlib import Path

import httpx
import pytest

from evals.fleet import glm53_dedicated_v14 as v14
from evals.fleet import glm53_dedicated_v15 as v15
from evals.fleet import glm53_dedicated_v15_live as live
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_v15_preserves_exact_v14_runtime_with_fresh_identity() -> None:
    old = v14.payload(v14.spec(ROOT), ROOT)
    new = v15.payload(v15.spec(ROOT), ROOT)
    assert new["title"] == v15.TITLE
    assert new["run_dir"] == v15.RUN_DIR
    assert new["env"] == {**old["env"], "GLM53_RUN_DIR": v15.RUN_DIR}
    for field in set(old) - {"title", "run_dir", "env"}:
        assert new[field] == old[field]
    assert new["workers"] == 1
    assert new["gpus_per_worker"] == 8
    assert new["priority_class"] == "fleet-infra-quiet"


def test_v15_rejects_pre_admission_drift() -> None:
    value = v15.spec(ROOT)
    value["pre_admission"]["selection_rank"] = 52
    with pytest.raises(ValueError, match="pre-admission"):
        v15.validate(value, ROOT)


def _prepared() -> dict:
    value = {
        "schema_version": "fleet-glm53-dedicated-v14-scored-canary-preflight-v1",
        "status": "CLEAR_HELD",
        "launch_authorized": False,
        "controller_package_sha256": v15.PRE_ADMISSION["controller_package_sha256"],
        "controller_objects_sha256": "sha256:" + "1" * 64,
        "held_plan_sha256": v15.PRE_ADMISSION["held_plan_sha256"],
        "selection_rank": 51,
        "reserved_cell_ids": [f"sha256:{number:064x}" for number in range(1, 5)],
        "reserved_execution_ids": [f"sha256:{number:064x}" for number in range(5, 9)],
        "all_four_rank_cells_unstarted": True,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "observer_job_uid": "4cf6c1ae-6533-4a1d-bcf9-a206297bd240",
        "observer_pod_uid": "dc79b655-90a2-4828-8d74-dd363704583b",
        "checked_at_utc": "2026-09-05T00:00:00Z",
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "fresh_server_binding_required_after_admission": True,
        "fresh_non_scored_parity_required_after_admission": True,
        "fresh_release_required_immediately_before_canary_create": True,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_prepared_validator_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    value = _prepared()
    raw = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    monkeypatch.setitem(v15.PRE_ADMISSION, "file_sha256", self_hosted.sha256(raw))
    monkeypatch.setitem(v15.PRE_ADMISSION, "receipt_sha256", value["receipt_sha256"])
    assert live._validate_prepared(raw)["status"] == "CLEAR_HELD"
    changed = {**value, "fleet_session_collisions": 1}
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    changed_raw = json.dumps(changed, indent=2, sort_keys=True).encode() + b"\n"
    monkeypatch.setitem(v15.PRE_ADMISSION, "file_sha256", self_hosted.sha256(changed_raw))
    monkeypatch.setitem(v15.PRE_ADMISSION, "receipt_sha256", changed["receipt_sha256"])
    with pytest.raises(RuntimeError, match="fields drifted"):
        live._validate_prepared(changed_raw)


def test_project_shape_allows_only_exact_active_qwen_peer() -> None:
    row = {
        "name": "ft-run-qwen",
        "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
    }
    shape = live._project_shape([row])
    assert shape["planned_nodes"] == 2
    assert shape["planned_gpus"] == 9
    with pytest.raises(RuntimeError, match="unknown active"):
        live._project_shape(
            [{"name": "ft-run-other", "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-other"}]
        )


def test_live_rows_reconciles_stale_list_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("stale"):
            return httpx.Response(404)
        return httpx.Response(200, json={"status": "RUNNING"})

    rows = [
        {"name": "ft-run-stale", "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-old"},
        {
            "name": "ft-run-qwen",
            "run_dir": "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-b-v2",
        },
    ]
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="https://test") as client:
        assert [row["name"] for row in live._live_rows(client, rows)] == ["ft-run-qwen"]
