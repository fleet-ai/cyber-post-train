from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from evals.fleet import opencode_self_hosted, rollout_ledger, rollout_worker

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
SNAPSHOT = ROOT / "docs/evidence/qwen38-study/2026-09-06-exact-pass4-ledger-v50.json"


def _campaign() -> dict:
    return json.loads(CAMPAIGN.read_text())


def test_frozen_universe_matches_every_v50_scientific_cell() -> None:
    expanded = rollout_worker._universe_index(_campaign(), ROOT)
    snapshot = json.loads(SNAPSHOT.read_text())
    observed = {
        (row["model"], row["task_version_id"], row["attempt"]): row["cell_id"]
        for row in snapshot["cells"]
    }
    assert len(expanded) == 800
    assert {key: row["cell_id"] for key, row in expanded.items()} == observed


def test_config_keeps_exact_autocontinue_treatment_and_route(monkeypatch) -> None:
    campaign = _campaign()
    expanded = rollout_worker._universe_index(campaign, ROOT)
    scientific = next(
        row
        for row in expanded.values()
        if row["model"] == "qwen3.8-27b" and row["selection_rank"] == 2
    )
    bindings = (
        {
            "key": scientific["task_key"],
            "version_id": scientific["task_version_id"],
            "prompt_sha256": "sha256:" + "1" * 64,
            "env_variables_sha256": "sha256:" + "2" * 64,
            "output_json_schema_sha256": "sha256:" + "3" * 64,
            "cyber_contract": rollout_worker.AUTHORITY["required_cyber_contract"],
        },
        {
            "id": "env",
            "version": "v1",
            "version_id": "11111111-1111-4111-8111-111111111111",
            "data_id": "data",
            "data_version": "v1",
            "runtime_seed_content_sha256": "4" * 64,
            "ttl_seconds": 32400,
        },
        {
            "id": "22222222-2222-4222-8222-222222222222",
            "version_id": "33333333-3333-4333-8333-333333333333",
            "version": 1,
            "sha256": "5" * 64,
            "function_name": "verify",
        },
    )
    monkeypatch.setattr(rollout_worker, "_task_binding", lambda *_args: bindings)
    ledger_cell = {
        "model_id": "qwen3.8-27b",
        "model_revision": campaign["models"]["qwen3.8-27b"]["revision"],
        "endpoint_model_id": "chris-cyber-qwen38-27b-dedicated-v1",
    }
    config = rollout_worker.build_config(campaign, ledger_cell, scientific, {}, object())
    assert config["model"]["served_id"] == "chris-cyber-qwen38-27b-dedicated-v1"
    assert config["model"]["session_model"] == "qwen/qwen3.8-27b"
    assert config["harness"]["context_management"].endswith("autocontinue_v1")
    assert config["harness"]["context_window_size"] == 262144
    assert config["harness"]["compaction_headroom_tokens"] == 20000
    assert config["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert config["execution"]["cell_id"] == scientific["cell_id"]
    assert config["execution"]["execution_id"] == scientific["initial_execution"]["execution_id"]
    settings = opencode_self_hosted.opencode_settings(config)
    model_settings = settings["provider"]["fleet-cluster"]["models"][config["model"]["served_id"]]
    assert settings["compaction"] == {"auto": True, "reserved": 20000}
    assert model_settings["limit"] == {
        "context": 262144,
        "input": 229376,
        "output": 32768,
    }
    assert settings["permission"] == {"*": "deny", "fleet_*": "allow"}
    assert all(enabled is False for enabled in settings["tools"].values())


def test_claim_receipt_is_score_and_content_blind() -> None:
    campaign = _campaign()
    scientific = next(iter(rollout_worker._universe_index(campaign, ROOT).values()))
    config = {
        "campaign_id": campaign["campaign_id"],
        "run_id": "chris-cyber-cell-example-g1",
        "execution": scientific["initial_execution"],
    }
    cell = {
        "cell_id": "ledger-id",
        "serving_block": "glm-shared",
        "model_revision": campaign["models"]["glm-5.3"]["revision"],
        "task_version_id": scientific["task_version_id"],
        "attempt": scientific["attempt"],
        "harness_id": "opencode-1.18.27-autocontinue-v1",
    }
    receipt = rollout_worker._claim_receipt(config, cell)
    assert receipt["model_call_started_when_claim_written"] is False
    assert receipt["receipt_sha256"].startswith("sha256:")
    for forbidden in ("prompt", "trace", "flag", "reward", "score", "response"):
        assert forbidden not in receipt


def test_local_result_indexes_private_artifacts_before_catalog_acceptance(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "experiment_id,task_key,task_version_id,model_id,model_revision,"
        "serving_block,endpoint_model_id,harness_id,attempt,max_retries\n"
        "experiment,task,task-version,qwen3.8-27b,revision,qwen-shared,"
        "qwen3.8-27b,opencode-1.18.27,1,1\n",
        encoding="utf-8",
    )
    database = tmp_path / "ledger.sqlite3"
    rollout_ledger.initialize(database, plan)
    cell = rollout_ledger.claim(database, worker_id="worker", serving_block="qwen-shared")
    assert cell is not None

    output_root = tmp_path / "outputs"
    out_dir = output_root / "attempts" / ("1" * 64)
    (out_dir / "agent").mkdir(parents=True)
    trace = out_dir / "agent" / "opencode.json"
    trace.write_text('{"type":"message"}\n', encoding="utf-8")
    (out_dir / "trace-manifest.json").write_text(
        json.dumps(
            {
                "canonical_trace": "agent/opencode.json",
                "canonical_trace_sha256": opencode_self_hosted.sha256(trace.read_bytes()),
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "result.json").write_text("{}", encoding="utf-8")
    (out_dir / "reward-result.json").write_text("{}", encoding="utf-8")
    (out_dir / "session-ingest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (out_dir / "cleanup.json").write_text("{}", encoding="utf-8")
    config = {
        "run_id": "chris-cyber-cell-example-g1",
        "config_sha256": "sha256:" + "2" * 64,
        "execution": {
            "execution_id": "sha256:" + "1" * 64,
            "execution_generation": 1,
        },
    }
    result = {
        "session_id": "session-1",
        "verifier_execution_id": "verifier-1",
        "score": 0.25,
        "agent_exit_code": 0,
        "agent_termination": "completed",
        "elapsed_seconds": 10.0,
    }

    stored = rollout_worker._record_local_result(
        database=database,
        config=config,
        ledger_cell=cell,
        result=result,
        out_dir=out_dir,
        output_root=output_root,
    )

    assert stored["created"] is True
    assert stored["score"] == 0.25
    assert stored["trace_path"] == "agent/opencode.json"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM rollout_local_results").fetchone()[0] == 1
