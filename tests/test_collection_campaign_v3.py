"""The v3 packet changes only completion-budget execution safety."""

from __future__ import annotations

import pytest
from test_collection_campaign import _bindings, _inventory, _request, _split

from evals.fleet import visible_action_collection_v3 as runtime
from training import collection_campaign as v1
from training import collection_campaign_v3 as v3


def test_v3_preserves_science_and_binds_completion_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = _inventory()
    split, anchor = _split(inventory, monkeypatch)
    request = _request()
    old = v1.render(request, inventory, split, _bindings(inventory), role_anchor=anchor)
    new = v3.render(request, inventory, split, _bindings(inventory), role_anchor=anchor)

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
    packet = new["collection-packet.json"]
    authorization = new["operation-authorization.json"]
    assert packet["schema"] == v3.PACKET_SCHEMA
    assert packet["operation_authorization_sha256"] == authorization["sha256"]
    assert packet["execution_safety"]["completion_budget"] == (runtime.COMPLETION_BUDGET_POLICY)
    assert (
        packet["completion_budget_runtime_sha256"]
        == packet["execution_safety"]["completion_budget_runtime_sha256"]
    )
    assert packet["admission_policy"]["adapter_must_bind"][-1] == (
        "completion_budget_runtime_sha256"
    )
    assert packet["corpus_scope"]["visible_reasoning_included"] is False
    assert packet["training_data_eligible"] is True
