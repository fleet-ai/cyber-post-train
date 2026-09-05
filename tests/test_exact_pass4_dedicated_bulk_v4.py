from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_dedicated_bulk_v4 as bulk

ROOT = Path.cwd()


def test_held_authority_exactly_partitions_798_cells() -> None:
    value = bulk.load(ROOT / bulk.SPEC_PATH)
    bulk.validate_spec(value, ROOT)
    plans = bulk.plans(value, ROOT)
    assert {name: plan["cell_count"] for name, plan in plans.items()} == {
        "qwen-hosted-1": 199,
        "qwen-hosted-2": 200,
        "glm-hosted-1": 99,
        "glm-hosted-2": 100,
        "glm-dedicated-a-1": 51,
        "glm-dedicated-a-2": 48,
        "glm-dedicated-b-1": 51,
        "glm-dedicated-b-2": 48,
    }
    canaries = bulk.scored_canary_plans(value, ROOT)
    cells = [cell for plan in [*plans.values(), *canaries.values()] for cell in plan["cells"]]
    assert len(cells) == len({cell["cell_id"] for cell in cells}) == 798


def test_glm_tasks_never_cross_serving_treatment() -> None:
    value = bulk.load(ROOT / bulk.SPEC_PATH)
    plans = bulk.plans(value, ROOT)
    canaries = bulk.scored_canary_plans(value, ROOT)
    owners: dict[int, set[str]] = {}
    for plan in plans.values():
        if plan["model"] != "glm-5.3":
            continue
        for cell in plan["cells"]:
            owners.setdefault(cell["selection_rank"], set()).add(plan["serving_block"])
    assert set(owners) == set(range(1, 101))
    assert all(len(blocks) == 1 for blocks in owners.values())
    assert [
        cell["attempt"] for cell in plans["glm-hosted-1"]["cells"] if cell["selection_rank"] == 13
    ] == [2, 3, 4]
    assert not any(
        cell["selection_rank"] == 51 and cell["attempt"] == 1
        for cell in plans["glm-dedicated-a-1"]["cells"]
    )
    assert not any(
        cell["selection_rank"] == 76 and cell["attempt"] == 1
        for cell in plans["glm-dedicated-b-1"]["cells"]
    )
    assert canaries["A"]["cells"][0]["selection_rank"] == 51
    assert canaries["B"]["cells"][0]["selection_rank"] == 76
    assert all(plan["runtime_gate_required_before_canary"] is False for plan in canaries.values())


def test_dedicated_controllers_bind_runtime_gate_and_heartbeat() -> None:
    value = bulk.load(ROOT / bulk.SPEC_PATH)
    plans = bulk.plans(value, ROOT)
    for name, plan in plans.items():
        assert plan["controller_resource_policy"] == {
            "cpu_only": True,
            "gpus": 0,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        }
        if "dedicated" in name:
            assert plan["dedicated_runtime_gate"].endswith("-runtime.json")
            assert plan["runtime_gate_required_before_any_scored_bulk_cell"] is True
            assert plan["heartbeat_script"].endswith("v7_controller_heartbeat.sh")
            assert plan["heartbeat_relative_path"].endswith(str(plan["stream"]))
            assert plan["qualified_controller_concurrency"] == 2
            assert plan["concurrency_qualification_receipt"].endswith("-parity.json")
            assert plan["dedicated_server"]["priority_class"] == "fleet-infra-quiet"
            assert plan["dedicated_server"]["preemption_policy"] == "Never"
            assert plan["dedicated_server"]["request_sha256"].startswith("sha256:")
        else:
            assert plan["dedicated_runtime_gate"] is None
            assert plan["heartbeat_script"] is None
            assert plan["heartbeat_relative_path"] is None
            assert plan["qualified_controller_concurrency"] is None
            assert plan["concurrency_qualification_receipt"] is None
            assert plan["dedicated_server"] is None


def test_prebulk_is_fresh_fail_closed_and_content_blind() -> None:
    value = bulk.load(ROOT / bulk.SPEC_PATH)
    gate = bulk.prebulk_requirements(value, ROOT)
    assert gate["root"].endswith("prebulk-reconcile-v4")
    assert gate["launch_authorized"] is False
    assert len(gate["controller_plan_sha256"]) == 8
    assert all("prompt" not in key for key in gate)


def test_mutation_of_partition_fails_closed() -> None:
    value = bulk.load(ROOT / bulk.SPEC_PATH)
    changed = copy.deepcopy(value)
    changed["controllers"][4]["ranks"] = [50, 63]
    with pytest.raises((KeyError, ValueError)):
        bulk.validate_spec(changed, ROOT)


def test_submitter_exposes_executable_prebulk_modes() -> None:
    text = (ROOT / bulk.SUBMIT_PATH).read_text()
    assert "submit-prebulk-source" in text
    assert "submit-prebulk-accept" in text
    result = subprocess.run(
        [str(ROOT / bulk.SUBMIT_PATH), "submit"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
