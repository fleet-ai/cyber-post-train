import json
from pathlib import Path

import pytest

from evals.fleet import glm53_dedicated_v17 as v18
from evals.fleet import glm53_dedicated_v19 as v19
from evals.fleet import glm53_dedicated_v19_live as live
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_v19_preserves_runtime_except_qualified_lifecycle_and_identity() -> None:
    old = v18.payload(v18.spec(ROOT), ROOT)
    new = v19.payload(v19.spec(ROOT), ROOT)
    assert new["title"] == v19.TITLE
    assert new["run_dir"] == v19.RUN_DIR
    assert new["env"] == {**old["env"], "GLM53_RUN_DIR": v19.RUN_DIR}
    assert new["command"] != old["command"]
    assert "local_sglang_inference_metrics" in new["command"]
    assert "sglang:num_requests_total" in new["command"]
    for field in set(old) - {"title", "run_dir", "env", "command"}:
        assert new[field] == old[field]
    assert new["workers"] == 1 and new["gpus_per_worker"] == 8
    assert new["priority_class"] == "fleet-infra-quiet"


def test_v19_fails_closed_on_watchdog_drift() -> None:
    value = v19.spec(ROOT)
    value["watchdog_qualification"]["job_uid"] = "00000000-0000-4000-8000-000000000000"
    with pytest.raises(ValueError, match="watchdog qualification"):
        v19.validate(value, ROOT)


def test_v19_watchdog_receipt_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    value = {
        "schema_version": "fleet-glm53-metric-watchdog-canary-v1",
        "status": "PASSED_NON_SCORED",
        "lifecycle_file_sha256": v19.WATCHDOG_QUALIFICATION["lifecycle_file_sha256"],
        "production_idle_seconds": 600,
        "canary_idle_seconds": 5,
        "health_only_released": True,
        "health_probes_counted_as_traffic": False,
        "actual_model_requests": 8,
        "active_survived_seconds": 8.533,
        "actual_model_traffic_prevented_release": True,
        "server_local_uid_scope": True,
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_read": False,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    monkeypatch.setitem(v19.WATCHDOG_QUALIFICATION, "file_sha256", self_hosted.sha256(raw))
    monkeypatch.setitem(v19.WATCHDOG_QUALIFICATION, "receipt_sha256", value["receipt_sha256"])
    class Done:
        stdout = raw
    monkeypatch.setattr(live.subprocess, "run", lambda *a, **k: Done())
    assert live._watchdog()[1]["status"] == "PASSED_NON_SCORED"


def test_v19_live_allowlist_is_exact_current_q_peer() -> None:
    assert live.ALLOWED == {
        "/mnt/sfs/jobs/chris-cyber-evalserve-q38-tp1-i-v1": {"nodes": 1, "gpus": 1}
    }
