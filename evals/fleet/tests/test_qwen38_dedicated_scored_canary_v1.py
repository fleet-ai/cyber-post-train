from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evals.fleet import self_hosted
from evals.fleet.qwen38_dedicated_scored_canary_v1 import (
    EXPECTED_CELL_ID,
    EXPECTED_EXECUTION_ID,
    EXPECTED_IDENTITIES,
    MODEL_REVISION,
    RUN_ID,
    SERVICE_ORIGIN,
    _classify,
    build_plan,
)


def test_exact_rank2_attempt1_dedicated_plan() -> None:
    root = Path(__file__).parents[3]
    plan = build_plan(root)
    config = plan["config"]
    assert plan["item"]["cell_id"] == EXPECTED_CELL_ID
    assert plan["item"]["execution_id"] == EXPECTED_EXECUTION_ID
    assert plan["item"]["selection_rank"] == 2
    assert plan["item"]["attempt"] == 1
    assert config["run_id"] == RUN_ID
    assert config["model"]["revision"] == MODEL_REVISION
    assert config["model"]["endpoint_origin"] == SERVICE_ORIGIN
    assert config["serving"]["serving_block"] == "dedicated-qwen-tp1-v1"
    assert config["harness"]["context_window_size"] == 262144
    assert config["harness"]["compaction_headroom_tokens"] == 20000
    assert config["execution"]["required_task_tools"] == ["bash", "submit_report"]
    settings = self_hosted.opencode_settings(config)
    limit = settings["provider"]["fleet-cluster"]["models"]["qwen3.8-27b"]["limit"]
    assert limit == {"context": 262144, "input": 229376, "output": 32768}


def test_rank2_attempts_have_exact_fresh_identities() -> None:
    root = Path(__file__).parents[3]
    for attempt in (2, 3, 4):
        plan = build_plan(root, attempt)
        expected_cell, expected_execution = EXPECTED_IDENTITIES[attempt]
        assert plan["item"]["cell_id"] == expected_cell
        assert plan["item"]["execution_id"] == expected_execution
        assert plan["item"]["attempt"] == attempt
        assert plan["item"]["selection_rank"] == 2
        assert plan["item"]["run_id"].endswith(f"-a{attempt}-v1")
        assert plan["config"]["execution"]["network"] == plan["item"]["run_id"]
        assert plan["output_root"].endswith(plan["item"]["run_id"])


def test_successor_runner_binds_attempt_argument() -> None:
    script = (
        Path(__file__).parents[3] / "evals/fleet/scripts/run_qwen38_dedicated_scored_canary_v1.sh"
    ).read_text()
    assert 'case "$QWEN_DEDICATED_ATTEMPT" in 1|2|3|4)' in script
    assert '--attempt "$QWEN_DEDICATED_ATTEMPT"' in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_legacy_session_projection_may_omit_but_not_mismatch_bindings(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).parents[3]
    plan = build_plan(root)
    item, config = plan["item"], plan["config"]
    session_id = "89a321f1-51dd-4ace-832b-6782385f6bf0"
    verifier_id = "9ed9b1d5-2f96-4f6c-aaa3-3f2e82c67adb"
    files = {
        "result.json": {
            "run_id": item["run_id"],
            "task_version_id": item["task_version_id"],
            "agent_termination": "completed",
            "agent_exit_code": 0,
            "session_ingest_status": "completed",
            "session_id": session_id,
            "verifier_execution_id": verifier_id,
        },
        "reward-result.json": {
            "task_version_id": item["task_version_id"],
            "verifier_execution_id": verifier_id,
            "reward": 0.0,
        },
        "session-ingest.json": {"status": "completed", "session_id": session_id},
        "cleanup.json": {
            "instance_created": True,
            "instance_closed": True,
            "containers_removed": True,
        },
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    row = {
        "session_id": session_id,
        "status": "completed",
        "task_key": config["task"]["key"],
        "model": None,
        "verifier_execution": {"id": verifier_id},
    }
    monkeypatch.setattr(self_hosted, "_task_sessions", lambda *_args: [row])

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    module = __import__("evals.fleet.qwen38_dedicated_scored_canary_v1", fromlist=["httpx"])
    monkeypatch.setattr(module.httpx, "Client", lambda **_kwargs: Client())
    accepted = _classify(
        tmp_path,
        plan,
        {"receipt_sha256": "sha256:claim"},
        "unused",
    )
    assert accepted["accepted"] is True
    assert accepted["authoritative_projection_omissions"] == [
        "metadata",
        "model",
        "task_version_id",
    ]
    row["model"] = "wrong-model"
    with pytest.raises(RuntimeError, match="binding drifted"):
        _classify(tmp_path, plan, {"receipt_sha256": "sha256:claim"}, "unused")
