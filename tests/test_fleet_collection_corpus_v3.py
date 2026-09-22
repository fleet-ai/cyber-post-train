"""Completion-budget version guards for visible-action corpus materialization."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import visible_action_collection_v3 as runtime
from training import fleet_collection_admission as admission
from training import fleet_collection_corpus as corpus
from training.task_family_split import TRUSTED_FLEET_COLLECTION_ROOT_ID

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v3-canary"
BASE = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v3"


def _read(name: str) -> dict:
    path = CAMPAIGN / name
    return json.loads((path if path.is_file() else BASE / name).read_text())


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _sealed(value: dict) -> dict:
    return {**value, "sha256": "sha256:" + digest(value)}


def _selection() -> tuple[dict, dict, dict, dict]:
    packet = _read("collection-packet.json")
    config = _read("eval-config.json")
    tasks = _read("task-selection.json")
    operation = _read("operation-authorization.json")
    split = _read("family-split.json")
    plan = runtime.compile_eval(config, relative_to=CAMPAIGN)
    identity = runtime.identity_map(plan)[0]
    role = next(
        row
        for row in split["tasks"]
        if row["task_key"] == identity["task_key"]
        and row["task_version_id"] == identity["task_version_id"]
    )
    harness = {
        "treatment_sha256": "sha256:" + digest(config["harness"]),
        "tool_catalog_sha256": config["harness"]["tool_catalog_sha256"],
    }
    row = {
        "session_id": "synthetic-v3-session",
        "campaign_plan_sha256": packet["eval_plan_sha256"],
        "cell_sha256": identity["scientific_cell_id"],
        "task_key": identity["task_key"],
        "task_version_id": identity["task_version_id"],
        "group_id": role["group_id"],
        "attempt": identity["attempt"],
        "source_kind": packet["source"]["kind"],
        "model": packet["source"]["model"],
        "harness": harness,
        "template_sha256": packet["source"]["template_sha256"],
        "normalized_record_sha256": _sha("1"),
        "normalized_trajectory_sha256": _sha("2"),
        "transcript_sha256": _sha("3"),
        "operation_authorization_sha256": operation["sha256"],
        "ledger_cell_id": identity["ledger_cell_id"],
        "scientific_cell_id": identity["scientific_cell_id"],
        "execution_id": identity["execution_id"],
        "completion_budget_runtime_sha256": packet["completion_budget_runtime_sha256"],
    }
    selection = _sealed(
        {
            "schema": admission.SELECTION_SCHEMA_V3,
            "artifact_kind": admission.HANDOFF_KIND,
            "trainable_corpus_created": False,
            "parquet_created": False,
            "source_text_read": False,
            "next_required_gate": (
                "bind_private_normalized_records_then_run_existing_dense_sft_builder"
            ),
            "campaign_plan_sha256": packet["eval_plan_sha256"],
            "catalog_inventory_sha256": _read("metadata-inventory.json")["sha256"],
            "family_split_sha256": split["sha256"],
            "root_role_anchor_id": TRUSTED_FLEET_COLLECTION_ROOT_ID,
            "family_role_anchor_sha256": _read("role-anchor.json")["sha256"],
            "protected_family_lock_sha256": _read("protected-family-lock.json")["sha256"],
            "source_kind": packet["source"]["kind"],
            "source_model_alias": packet["source"]["model_alias"],
            "source_model": packet["source"]["model"],
            "harness_treatment_sha256": harness["treatment_sha256"],
            "tool_catalog_sha256": harness["tool_catalog_sha256"],
            "template_sha256": packet["source"]["template_sha256"],
            "max_sessions_per_task_version": config["pass_k"],
            "collection_packet_sha256": packet["sha256"],
            "operation_authorization_sha256": operation["sha256"],
            "identity_map_sha256": operation["identity_map_sha256"],
            "completion_budget_runtime_sha256": packet["completion_budget_runtime_sha256"],
            "selected": [row],
        }
    )
    return selection, packet, tasks, operation


def test_v3_materializer_binds_completion_budget_runtime() -> None:
    selection, packet, tasks, operation = _selection()
    config = _read("eval-config.json")
    selected = corpus._selection(selection)  # noqa: SLF001
    assert (
        selected["synthetic-v3-session"]["completion_budget_runtime_sha256"]
        == packet["completion_budget_runtime_sha256"]
    )
    corpus._packet(packet, selection, tasks, config, operation)  # noqa: SLF001

    changed = copy.deepcopy(selection)
    changed["completion_budget_runtime_sha256"] = _sha("9")
    changed["selected"][0]["completion_budget_runtime_sha256"] = _sha("9")
    changed["sha256"] = "sha256:" + digest(
        {key: value for key, value in changed.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="completion-budget runtime binding differs"):
        corpus._packet(packet, changed, tasks, config, operation)  # noqa: SLF001


def test_v3_materializer_rejects_v2_selection_relabel() -> None:
    selection, packet, tasks, operation = _selection()
    selection["schema"] = admission.SELECTION_SCHEMA_V2
    selection.pop("completion_budget_runtime_sha256")
    selection["selected"][0].pop("completion_budget_runtime_sha256")
    selection["sha256"] = "sha256:" + digest(
        {key: value for key, value in selection.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="versions or bindings differ"):
        corpus._packet(  # noqa: SLF001
            packet,
            selection,
            tasks,
            _read("eval-config.json"),
            operation,
        )
