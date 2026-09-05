from __future__ import annotations

import json
from pathlib import Path

from evals.fleet import qwen_bulk_generation16 as bulk
from evals.fleet import qwen_bulk_generation16_package as package
from evals.fleet import qwen_bulk_generation16_runtime as runtime

ROOT = Path(__file__).parents[1]
INVENTORY = Path("/private/tmp/exact100-inventory.XXXXXX.json")


def test_exact_hosted_partition_excludes_accepted_and_dedicated_cells() -> None:
    plans = bulk.validate_all(ROOT)
    assert {name: plan["new_session_count"] for name, plan in plans.items()} == {
        "qwen-a": 199,
        "qwen-b": 196,
    }
    rows = [row for plan in plans.values() for row in plan["attempts"]]
    identities = {(row["selection_rank"], row["attempt"]) for row in rows}
    assert len(rows) == len(identities) == 395
    assert not any(rank == 2 for rank, _attempt in identities)
    assert (4, 1) not in identities
    assert all(row["execution_generation"] == 16 for row in rows)
    assert all(
        plan["execution"]["endpoint_lease"]["maximum_streams"] == 2
        for plan in plans.values()
    )


def test_attempts_are_sequential_within_complete_task_partitions() -> None:
    plans = bulk.validate_all(ROOT)
    rank_owner: dict[int, str] = {}
    for controller, plan in plans.items():
        assert plan["execution"]["attempts_per_task_sequential"] is True
        pairs = [(row["selection_rank"], row["attempt"]) for row in plan["attempts"]]
        assert pairs == sorted(pairs)
        for rank, _attempt in pairs:
            assert rank_owner.setdefault(rank, controller) == controller


def test_g15_nullable_model_gate_and_package_are_valid() -> None:
    gate = bulk.load(ROOT / bulk.G15_GATE_PATH)
    runtime.validate_g15_gate(gate)
    assert gate["api_session"]["model_projection"] == "omitted"
    built = package.build_package(ROOT)
    assert set(built["controller_manifests"]) == {"qwen-a", "qwen-b"}
    assert all(
        size < package.prior.PACKAGE_OBJECT_LIMIT
        for size in built["object_json_bytes"].values()
    )


def test_runtime_plans_preserve_exact_treatment() -> None:
    inventory = json.loads(INVENTORY.read_text())
    plans = {
        name: bulk.build_runtime_plan(name, inventory, ROOT) for name in bulk.CONTROLLERS
    }
    assert sum(len(plan["attempts"]) for plan in plans.values()) == 395
    assert all(plan["model"]["served_id"] == "qwen3.8-27b" for plan in plans.values())
    assert all(
        plan["execution"]["required_task_tools"] == ["bash", "submit_report"]
        for plan in plans.values()
    )
    assert all(plan["harness"]["context_window_size"] == 262144 for plan in plans.values())
