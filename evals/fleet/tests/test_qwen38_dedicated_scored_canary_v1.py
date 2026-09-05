from __future__ import annotations

from pathlib import Path

from evals.fleet import self_hosted
from evals.fleet.qwen38_dedicated_scored_canary_v1 import (
    EXPECTED_CELL_ID,
    EXPECTED_EXECUTION_ID,
    MODEL_REVISION,
    RUN_ID,
    SERVICE_ORIGIN,
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
