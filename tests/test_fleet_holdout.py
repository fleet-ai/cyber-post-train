from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import holdout, self_hosted

PLAN_PATH = Path("evals/fleet/configs/qwen36-27b-qwen-code-test20-base-v1.json")
SPLIT_PATH = Path("configs/data/fleet-a62-task-split-v1.json")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_holdout_plan_pins_exact_untouched_test20() -> None:
    plan = _json(PLAN_PATH)
    split = _json(SPLIT_PATH)
    rows = holdout.validate_plan(plan, split)
    assert len(rows) == 20
    assert {row["split"] for row in rows} == {"test"}
    assert plan["model"]["revision"] == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert plan["harness"]["version"] == "0.22.3"
    assert plan["harness"]["source_commit"] == "09825973e7d3c3fd07e17909c396aa62f48ce51f"
    assert plan["harness"]["max_model_requests"] == 300
    assert plan["harness"]["timeout_seconds"] == 7200


def test_holdout_plan_rejects_split_digest_drift() -> None:
    plan = _json(PLAN_PATH)
    split = _json(SPLIT_PATH)
    split["tasks"][0]["task_version_id"] = "drifted"
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        holdout.validate_plan(plan, split)


def test_task_config_varies_only_task_runtime_and_run_identity() -> None:
    plan = _json(PLAN_PATH)
    row = {
        "index": 3,
        "task_key": "task-key",
        "task_version_id": "task-version",
        "prompt_sha256": "sha256:prompt",
        "env_variables_sha256": "sha256:env",
        "output_json_schema_sha256": "sha256:schema",
        "env_key": "env",
        "env_version": "v1",
        "data_key": "data",
        "data_version": "v2",
        "runtime_seed_content_sha256": None,
        "verifier": {
            "id": "verifier",
            "version_id": "verifier-version",
            "version": 1,
            "sha256": "verifier-sha",
            "function_name": "verify",
        },
    }
    config = holdout.task_config(plan, row)
    assert config["task"]["version_id"] == "task-version"
    assert config["environment"]["runtime_seed_content_sha256"] is None
    assert "code_sha256" not in config["verifier"]
    assert config["model"] == plan["model"]
    assert config["harness"] == plan["harness"]
    assert config["authority"] == plan["authority"]
    assert config["execution"]["pass_k"] == 1
    assert config["execution"]["max_concurrent"] == 1
    assert config["execution"]["training_data_eligible"] is False


def test_frozen_receipt_requires_exactly_20_unique_versions() -> None:
    plan = _json(PLAN_PATH)
    tasks = [
        {"index": index, "task_version_id": f"version-{index}"}
        for index in range(1, 21)
    ]
    receipt = {
        "schema_version": holdout.RECEIPT_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "task_count": 20,
        "planned_sessions": 20,
        "tasks": tasks,
    }
    receipt["receipt_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
    holdout.validate_frozen_receipt(receipt)
    drifted = copy.deepcopy(receipt)
    drifted["tasks"][1]["task_version_id"] = "version-1"
    drifted["receipt_sha256"] = holdout._digest_without(drifted, "receipt_sha256")
    with pytest.raises(ValueError, match="not unique"):
        holdout.validate_frozen_receipt(drifted)
