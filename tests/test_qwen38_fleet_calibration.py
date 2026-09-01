from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import qwen38_calibration, self_hosted

PLAN = Path(
    "evals/fleet/configs/qwen38-27b-qwen-code-reward-calibration-pass1-v2.json"
)
SPLIT = Path("configs/data/fleet-a62-task-split-v1.json")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_plan_selects_twenty_exact_non_test_tasks() -> None:
    rows = qwen38_calibration.validate_plan(_json(PLAN), _json(SPLIT))
    assert len(rows) == 20
    assert {row["split"] for row in rows} <= {"train", "dev"}
    assert all(row["split"] != "test" for row in rows)
    assert len({row["task_version_id"] for row in rows}) == 20
    assert {
        family: sum(row["family"] == family for row in rows)
        for family in qwen38_calibration.EXPECTED_FAMILY_COUNTS
    } == qwen38_calibration.EXPECTED_FAMILY_COUNTS


def test_plan_binds_qwen38_qwen_code_and_long_horizon() -> None:
    plan = _json(PLAN)
    assert plan["model"] == {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "served_id": "qwen3.8-27b",
        "endpoint_origin": "https://inference.flt.build",
    }
    assert plan["harness"]["version"] == "0.22.3"
    assert plan["harness"]["max_model_requests"] == 600
    assert plan["harness"]["context_window_size"] == 262144
    assert plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert (
        plan["execution"]["required_task_tool_catalog_sha256"]
        == qwen38_calibration.EXPECTED_TOOL_CATALOG_SHA256
    )
    assert plan["execution"]["pass_k"] == 1


def test_plan_rejects_a_sealed_test_version() -> None:
    plan, split = _json(PLAN), _json(SPLIT)
    test_row = next(row for row in split["tasks"] if row["split"] == "test")
    drifted = copy.deepcopy(plan)
    drifted["tasks"][0]["task_key"] = test_row["task_key"]
    drifted["tasks"][0]["task_version_id"] = test_row["task_version_id"]
    with pytest.raises(ValueError, match="sealed test"):
        qwen38_calibration.validate_plan(drifted, split)


def test_task_config_keeps_exact_tool_surface_and_unique_network() -> None:
    plan = _json(PLAN)
    row = {
        "index": 7,
        "task_key": "task",
        "task_version_id": "version",
        "prompt_sha256": "sha256:p",
        "env_variables_sha256": "sha256:e",
        "output_json_schema_sha256": "sha256:o",
        "env_key": "env",
        "env_version": "v1",
        "data_key": "data",
        "data_version": "v2",
        "runtime_seed_content_sha256": "sha256:r",
        "verifier": {"id": "v", "version_id": "vv", "version": 1, "sha256": "v"},
    }
    config = qwen38_calibration.task_config(plan, row)
    assert config["execution"]["required_task_tools"] == ["bash", "submit_report"]
    assert config["execution"]["network"].startswith("q38cal-t07-")
    assert config["harness"]["max_model_requests"] == 600


def test_runtime_tool_allowlist_is_fail_closed() -> None:
    config = {"execution": {"required_task_tools": ["bash", "submit_report"]}}
    self_hosted.assert_required_task_tools(config, ["bash", "submit_report"], "sha256:any")
    with pytest.raises(RuntimeError, match="exact required tool surface"):
        self_hosted.assert_required_task_tools(
            config, ["bash", "submit_report", "other"], "sha256:any"
        )
    with pytest.raises(RuntimeError, match="exact required tool surface"):
        self_hosted.assert_required_task_tools(config, ["submit_report", "bash"], "sha256:any")

    digest_config = {
        "execution": {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": "sha256:expected",
        }
    }
    with pytest.raises(RuntimeError, match="exact required catalog"):
        self_hosted.assert_required_task_tools(
            digest_config, ["bash", "submit_report"], "sha256:drifted"
        )


def test_non_root_docker_desktop_controller_uses_its_own_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(self_hosted.os, "geteuid", lambda: 501)
    monkeypatch.setattr(self_hosted.os, "getuid", lambda: 501)
    monkeypatch.setattr(self_hosted.os, "getgid", lambda: 20)
    assert self_hosted.agent_container_user_args() == ["--user", "501:20"]


def test_root_controller_keeps_image_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(self_hosted.os, "geteuid", lambda: 0)
    assert self_hosted.agent_container_user_args() == []


def test_canary_failure_prevents_remaining_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _json(PLAN)
    tasks = [
        {
            "index": index,
            "family": "current",
            "split": "train",
            "task_key": f"task-{index}",
            "task_version_id": f"version-{index}",
        }
        for index in range(1, 21)
    ]
    receipt = {
        "schema_version": qwen38_calibration.RECEIPT_SCHEMA,
        "task_count": 20,
        "planned_sessions": 20,
        "tasks": tasks,
    }
    receipt["receipt_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
    calls: list[int] = []

    def fail_first(plan, row, out_dir, proxy_script):
        calls.append(row["index"])
        return {
            "index": row["index"],
            "family": row["family"],
            "split": row["split"],
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "status": "infrastructure_error",
            "error_type": "GateFailure",
        }

    monkeypatch.setattr(qwen38_calibration, "_one_task", fail_first)
    summary = qwen38_calibration.run_campaign(
        plan, receipt, tmp_path / "run", Path("evals/fleet/fixed_proxy.py")
    )
    assert calls == [1]
    assert summary["canary_gate"]["passed"] is False
    assert summary["model_outcomes"] == 0


def test_valid_zero_canary_allows_remaining_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _json(PLAN)
    tasks = [
        {
            "index": index,
            "family": "current",
            "split": "train",
            "task_key": f"task-{index}",
            "task_version_id": f"version-{index}",
        }
        for index in range(1, 21)
    ]
    receipt = {
        "schema_version": qwen38_calibration.RECEIPT_SCHEMA,
        "task_count": 20,
        "planned_sessions": 20,
        "tasks": tasks,
    }
    receipt["receipt_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))

    def valid_zero(plan, row, out_dir, proxy_script):
        return {
            "index": row["index"],
            "family": row["family"],
            "split": row["split"],
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "status": "model_outcome",
            "score": 0.0,
        }

    monkeypatch.setattr(qwen38_calibration, "_one_task", valid_zero)
    summary = qwen38_calibration.run_campaign(
        plan, receipt, tmp_path / "run", Path("evals/fleet/fixed_proxy.py")
    )
    assert summary["canary_gate"]["passed"] is True
    assert summary["model_outcomes"] == 20
    assert summary["successes"] == 0
