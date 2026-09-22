"""V3 admission binds exact completion-budget runtime evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import visible_action_collection_v3 as runtime
from training import fleet_collection_admission as admission
from training.io import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v3-canary"
PARENT = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v3"


def _seal(value: dict) -> dict:
    return {**value, "sha256": "sha256:" + digest(value)}


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _fixture(tmp_path: Path) -> tuple[dict, dict]:
    config = json.loads((CAMPAIGN / "eval-config.json").read_text())
    plan = runtime.compile_eval(config, relative_to=CAMPAIGN)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")
    operation = json.loads((CAMPAIGN / "operation-authorization.json").read_text())
    packet = json.loads((CAMPAIGN / "collection-packet.json").read_text())
    identity = runtime.identity_map(plan)[0]
    treatment = plan["treatment"]
    record = _seal(
        {
            "schema": admission.ATTEMPT_SCHEMA_V3,
            "session_id": "synthetic-v3-success",
            "campaign_plan_sha256": "sha256:" + plan["sha256"],
            "operation_authorization_sha256": operation["sha256"],
            "completion_budget_runtime_sha256": packet["completion_budget_runtime_sha256"],
            "cell_sha256": identity["scientific_cell_id"],
            "ledger_cell_id": identity["ledger_cell_id"],
            "scientific_cell_id": identity["scientific_cell_id"],
            "execution_id": identity["execution_id"],
            "task_key": identity["task_key"],
            "task_version_id": identity["task_version_id"],
            "model_alias": identity["model_id"],
            "attempt": identity["attempt"],
            "model": plan["models"][identity["model_id"]],
            "harness": {
                "treatment_sha256": "sha256:" + digest(treatment),
                "tool_catalog_sha256": treatment["tool_catalog_sha256"],
            },
            "template_sha256": config["collection_runtime"]["source_template_sha256"],
            "outcome": {
                "status": "completed",
                "verifier_process_success": True,
                "score_at_least_one": True,
            },
            "ingestion": {
                "status": "complete",
                "normalized_record_sha256": "sha256:" + "1" * 64,
                "normalized_trajectory_sha256": "sha256:" + "2" * 64,
                "transcript_sha256": "sha256:" + "3" * 64,
            },
            "content_policy": {
                "target_mode": "visible_actions_only",
                "reasoning_visibility": "absent",
                "compaction": "none",
            },
        }
    )
    attempts = tmp_path / "attempts.jsonl"
    attempts.write_text(json.dumps(record, sort_keys=True) + "\n")
    request = {
        "schema": admission.REQUEST_SCHEMA_V3,
        "campaign": _reference(plan_path),
        "collection_packet": _reference(CAMPAIGN / "collection-packet.json"),
        "operation_authorization": _reference(CAMPAIGN / "operation-authorization.json"),
        "inventory": _reference(PARENT / "metadata-inventory.json"),
        "family_split": _reference(PARENT / "family-split.json"),
        "role_anchor": _reference(PARENT / "role-anchor.json"),
        "protected_family_lock": _reference(PARENT / "protected-family-lock.json"),
        "attempts": _reference(attempts),
        "source": {
            "kind": "self",
            "model_alias": "source",
            "template_sha256": config["collection_runtime"]["source_template_sha256"],
        },
        "max_sessions_per_task_version": 4,
        "output": str(tmp_path / "admission"),
    }
    return request, identity


def test_v3_admission_binds_completion_budget_runtime(tmp_path: Path) -> None:
    request, identity = _fixture(tmp_path)
    result = admission.build(request, relative_to=ROOT)
    selection = json.loads((Path(request["output"]) / "selection.private.json").read_text())
    receipt = json.loads((Path(request["output"]) / "ADMISSION.json").read_text())
    packet = json.loads((CAMPAIGN / "collection-packet.json").read_text())

    assert result["admitted_sessions"] == 1
    assert selection["schema"] == admission.SELECTION_SCHEMA_V3
    assert receipt["schema"] == admission.RECEIPT_SCHEMA_V3
    assert (
        selection["completion_budget_runtime_sha256"] == packet["completion_budget_runtime_sha256"]
    )
    assert selection["selected"][0]["scientific_cell_id"] == identity["scientific_cell_id"]


def test_v3_admission_rejects_wrong_completion_budget_runtime(tmp_path: Path) -> None:
    request, _identity = _fixture(tmp_path)
    attempts = Path(request["attempts"]["path"])
    row = json.loads(attempts.read_text())
    row["completion_budget_runtime_sha256"] = "sha256:" + "9" * 64
    row.pop("sha256")
    attempts.write_text(json.dumps(_seal(row), sort_keys=True) + "\n")
    request["attempts"] = _reference(attempts)
    result = admission.build(request, relative_to=ROOT)
    assert result["admitted_sessions"] == 0
    assert result["rejections"]["wrong_campaign_binding"] == 1


def test_v3_admission_rejects_packet_runtime_cross_binding(tmp_path: Path) -> None:
    request, _identity = _fixture(tmp_path)
    packet = json.loads((CAMPAIGN / "collection-packet.json").read_text())
    packet.pop("sha256")
    packet["execution_safety"]["completion_budget_runtime_sha256"] = "sha256:" + "9" * 64
    packet_path = tmp_path / "changed-packet.json"
    packet_path.write_text(json.dumps(_seal(packet), sort_keys=True) + "\n")
    request["collection_packet"] = _reference(packet_path)
    with pytest.raises(ValueError, match="completion-budget binding differs"):
        admission.build(request, relative_to=ROOT)
