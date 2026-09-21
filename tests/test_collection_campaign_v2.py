"""The v2 campaign changes execution safety without changing the science."""

from __future__ import annotations

import copy

import pytest
from test_collection_campaign import _bindings, _inventory, _request, _split

from evals.fleet import visible_action_collection_v2 as runtime
from training import collection_campaign as v1
from training import collection_campaign_v2 as v2


def _render(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    inventory = _inventory()
    split, anchor = _split(inventory, monkeypatch)
    request = _request()
    return (
        v1.render(request, inventory, split, _bindings(inventory), role_anchor=anchor),
        v2.render(request, inventory, split, _bindings(inventory), role_anchor=anchor),
    )


def test_v2_preserves_science_and_binds_exactly_once_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old, new = _render(monkeypatch)

    assert new["task-selection.json"] == old["task-selection.json"]
    for field in (
        "name",
        "task_set",
        "models",
        "routes",
        "harness",
        "images",
        "pass_k",
        "concurrency",
        "sampling",
        "training_data_eligible",
    ):
        assert new["eval-config.json"][field] == old["eval-config.json"][field]

    authorization = new["operation-authorization.json"]
    packet = new["collection-packet.json"]
    assert packet["schema"] == v2.PACKET_SCHEMA
    assert packet["operation_authorization_sha256"] == authorization["sha256"]
    assert packet["execution_safety"] == {
        "execution_mode": runtime.EXECUTION_MODE,
        "planned_cells": authorization["planned_cells"],
        "maximum_planned_cells": new["eval-config.json"]["collection_runtime"][
            "maximum_planned_cells"
        ],
        "operation_authorization_required": True,
        "operation_authorization_sha256": authorization["sha256"],
        "canonical_private_operation_root_required": True,
        "operation_root_name": authorization["operation_root_name"],
        "exclusive_pre_mutation_intent_required": True,
        "dedicated_empty_ledger_required": True,
        "dedicated_ledger_id": authorization["dedicated_ledger_id"],
        "identity_map_sha256": authorization["identity_map_sha256"],
        "automatic_replay_of_ambiguous_cells": False,
        "same_path_retry_allowed": False,
        "alternate_path_retry_allowed": False,
        "external_submission": False,
        "cluster_wrapper_supported": True,
        "cluster_job_execution_requirements": v2.JOB_EXECUTION_REQUIREMENTS,
    }
    assert packet["admission_policy"]["adapter_must_bind"] == [
        "collection_packet_sha256",
        "eval_plan_sha256",
        "operation_authorization_sha256",
    ]


def test_v2_operation_authorization_cannot_be_relabelled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _old, new = _render(monkeypatch)
    (tmp_path / "task-selection.json").write_bytes(v1.raw(new["task-selection.json"]))
    plan = runtime.compile_eval(new["eval-config.json"], relative_to=tmp_path)
    changed = copy.deepcopy(new["operation-authorization.json"])
    changed["dedicated_ledger_id"] = "sha256:" + "0" * 64
    changed.pop("sha256")
    changed["sha256"] = v1.canonical_digest(changed)
    with pytest.raises(ValueError, match="differs from the exact campaign identity"):
        runtime.validate_operation_authorization(changed, plan)
