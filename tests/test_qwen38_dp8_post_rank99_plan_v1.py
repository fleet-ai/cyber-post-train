from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evals.fleet import qwen38_dp8_post_rank99_plan_v1 as plan
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]


def test_held_dp8_plan_is_valid_and_non_authorizing() -> None:
    value = plan.load_held(ROOT)
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["bulk_gate"]["launch_authorized"] is False
    assert value["server"]["data_parallel_size"] == 8
    assert value["non_scored_qualification"]["concurrency_ladder"] == [1, 2, 4, 8]


@pytest.mark.parametrize(
    "path,value",
    [
        (("launch_authorized",), True),
        (("scoring_authorized",), True),
        (("server", "model_revision"), "drifted"),
        (("server", "context_length"), 131072),
        (("server", "gpus_per_worker"), 16),
        (("node_budget_gate", "max_project_gpu_nodes"), 3),
        (("non_scored_qualification", "concurrency_ladder"), [8]),
        (("scored_canary_gate", "formal_acceptance_required_before_bulk"), False),
        (("bulk_gate", "whole_task_boundaries_only"), False),
        (("lifecycle", "post_ready_idle_seconds"), 3600),
    ],
)
def test_held_dp8_plan_fails_closed(path: tuple[str, ...], value: object) -> None:
    changed = copy.deepcopy(plan.load_held(ROOT))
    cursor = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    changed["receipt_sha256"] = self_hosted.digest_without(changed, "receipt_sha256")
    with pytest.raises(ValueError):
        plan.validate(changed)
