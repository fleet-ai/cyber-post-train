"""V2 success evidence binds packet, operation and all cell identities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import visible_action_collection_v2 as runtime
from training import fleet_collection_admission as admission
from training import fleet_collection_corpus as corpus
from training.io import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v2"


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
    identity = runtime.identity_map(plan)[0]
    treatment = plan["treatment"]
    record = _seal(
        {
            "schema": admission.ATTEMPT_SCHEMA_V2,
            "session_id": "synthetic-v2-success",
            "campaign_plan_sha256": "sha256:" + plan["sha256"],
            "operation_authorization_sha256": operation["sha256"],
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
        "schema": admission.REQUEST_SCHEMA_V2,
        "campaign": _reference(plan_path),
        "collection_packet": _reference(CAMPAIGN / "collection-packet.json"),
        "operation_authorization": _reference(CAMPAIGN / "operation-authorization.json"),
        "inventory": _reference(CAMPAIGN / "metadata-inventory.json"),
        "family_split": _reference(CAMPAIGN / "family-split.json"),
        "role_anchor": _reference(CAMPAIGN / "role-anchor.json"),
        "protected_family_lock": _reference(CAMPAIGN / "protected-family-lock.json"),
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


def test_v2_admission_binds_packet_operation_and_exact_identity(tmp_path: Path) -> None:
    request, identity = _fixture(tmp_path)
    result = admission.build(request, relative_to=ROOT)
    output = Path(request["output"])
    selection = json.loads((output / "selection.private.json").read_text())
    receipt = json.loads((output / "ADMISSION.json").read_text())
    operation = json.loads((CAMPAIGN / "operation-authorization.json").read_text())
    packet = json.loads((CAMPAIGN / "collection-packet.json").read_text())

    assert result["admitted_sessions"] == 1
    assert selection["schema"] == admission.SELECTION_SCHEMA_V2
    assert receipt["schema"] == admission.RECEIPT_SCHEMA_V2
    assert selection["collection_packet_sha256"] == packet["sha256"]
    assert selection["operation_authorization_sha256"] == operation["sha256"]
    row = selection["selected"][0]
    assert row["ledger_cell_id"] == identity["ledger_cell_id"]
    assert row["scientific_cell_id"] == identity["scientific_cell_id"]
    assert row["execution_id"] == identity["execution_id"]
    assert corpus._selection(selection)[row["session_id"]] == row  # noqa: SLF001
    corpus._admission_receipt(receipt, selection)  # noqa: SLF001

    task_selection = json.loads((CAMPAIGN / "task-selection.json").read_text())
    eval_config = json.loads((CAMPAIGN / "eval-config.json").read_text())
    corpus._packet(  # noqa: SLF001
        packet, selection, task_selection, eval_config, operation
    )
    with pytest.raises(ValueError, match="requires its exact operation authorization"):
        corpus._packet(packet, selection, task_selection, eval_config)  # noqa: SLF001


def test_v2_admission_rejects_cross_operation_success_evidence(tmp_path: Path) -> None:
    request, _identity = _fixture(tmp_path)
    attempts = Path(request["attempts"]["path"])
    row = json.loads(attempts.read_text())
    row["operation_authorization_sha256"] = "sha256:" + "9" * 64
    row.pop("sha256")
    row = _seal(row)
    attempts.write_text(json.dumps(row, sort_keys=True) + "\n")
    request["attempts"] = _reference(attempts)
    result = admission.build(request, relative_to=ROOT)
    assert result["admitted_sessions"] == 0
    assert result["rejections"]["wrong_campaign_binding"] == 1


def test_v1_request_rejects_v2_plan(tmp_path: Path) -> None:
    request, _identity = _fixture(tmp_path)
    request["schema"] = admission.REQUEST_SCHEMA
    request.pop("collection_packet")
    request.pop("operation_authorization")
    with pytest.raises(ValueError, match="v1 admission cannot consume"):
        admission.build(request, relative_to=ROOT)
