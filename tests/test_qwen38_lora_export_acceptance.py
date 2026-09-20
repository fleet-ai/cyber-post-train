from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training.qwen38_lora_export_acceptance import (
    LoraExportAcceptanceError,
    load_acceptance,
    validate_acceptance,
)

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "configs/qualification/qwen38-lora-prod-step1-export-acceptance-v1.json"


def _handoff() -> dict:
    return load_acceptance(HANDOFF, repository_root=ROOT)


def _resign(value: dict) -> dict:
    result = copy.deepcopy(value)
    result["handoff_sha256"] = digest(
        {key: item for key, item in result.items() if key != "handoff_sha256"}
    ).removeprefix("sha256:")
    return result


def test_checked_in_acceptance_opens_only_inert_followup_work() -> None:
    handoff = _handoff()

    assert handoff["export_receipt"]["optimizer_steps_executed"] == 0
    assert handoff["export_receipt"]["base_tensor_count"] == 1199
    assert handoff["export_receipt"]["merged_tensor_count"] == 1199
    assert handoff["export_receipt"]["finite_logits"] is True
    assert handoff["resource_release"]["allocated_gpus"] == 0
    assert handoff["broad_training_anchor"]["allows_broad_lora_preflight"] is True
    assert handoff["broad_training_anchor"]["allows_external_submit"] is False
    assert handoff["serving_and_evaluation"]["external_mutation_authorized"] is False
    assert handoff["export_receipt"]["remote_revalidation_required_before_stage"] is True


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("export_receipt", "optimizer_steps_executed"), 1, "export_receipt_reference_mismatch"),
        (("export_receipt", "file_sha256"), "0" * 64, "export_receipt_reference_mismatch"),
        (("resource_release", "allocated_gpus"), 8, "resource_release_mismatch"),
        (("broad_training_anchor", "allows_external_submit"), True, "broad_training_anchor"),
        (
            ("serving_and_evaluation", "external_mutation_authorized"),
            True,
            "serving_and_evaluation",
        ),
    ],
)
def test_acceptance_fails_closed_on_boundary_drift(
    path: tuple[str, str], value: object, reason: str
) -> None:
    handoff = _handoff()
    handoff[path[0]][path[1]] = value

    with pytest.raises(LoraExportAcceptanceError, match=reason):
        validate_acceptance(_resign(handoff), repository_root=ROOT)


def test_acceptance_rejects_source_plan_file_drift() -> None:
    handoff = _handoff()
    handoff["source_export_plan"]["file_sha256"] = "0" * 64

    with pytest.raises(LoraExportAcceptanceError, match="plan_file_digest_mismatch"):
        validate_acceptance(_resign(handoff), repository_root=ROOT)


def test_acceptance_is_json_serializable() -> None:
    json.dumps(_handoff(), sort_keys=True)
