"""Fail-closed release gate for the sealed Qwen3.8 Fleet holdout baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from evals.fleet import self_hosted

GATE_SCHEMA = "qwen38-fleet-holdout-release-gate-v1"
GATE_RECEIPT_SCHEMA = "qwen38-fleet-holdout-release-receipt-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _digest_without(value: dict[str, Any], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    return self_hosted.sha256(self_hosted.canonical_json(unsigned))


def _uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} is not a UUID")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a UUID") from exc
    if parsed.int == 0:
        raise ValueError(f"{label} is a zero UUID")
    return value


def _score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not a numeric model outcome")
    result = float(value)
    if not math.isfinite(result) or result not in {0.0, 1.0}:
        raise ValueError(f"{label} is not a binary model outcome")
    return result


def validate_launch_gate(plan: dict[str, Any]) -> dict[str, Any]:
    gate = plan.get("launch_gate")
    if not isinstance(gate, dict) or gate.get("schema_version") != GATE_SCHEMA:
        raise ValueError("Qwen3.8 holdout launch gate is missing or unsupported")
    expected_fields = {
        "schema_version",
        "campaign_id",
        "frozen_receipt_sha256",
        "task_index",
        "task_version_id",
        "required_outcome_status",
        "required_verifier_execution_id",
        "required_cleanup_verified",
    }
    if set(gate) != expected_fields:
        raise ValueError("Qwen3.8 holdout launch gate fields drifted")
    if gate["campaign_id"] != "chris-cyber-q38-qcode-positive-gate-p1-v3":
        raise ValueError("Qwen3.8 holdout does not bind the active V3 campaign")
    if not isinstance(gate["frozen_receipt_sha256"], str) or not gate[
        "frozen_receipt_sha256"
    ].startswith("sha256:"):
        raise ValueError("Qwen3.8 holdout gate receipt digest is invalid")
    if gate["task_index"] != 1 or gate["required_outcome_status"] != "model_outcome":
        raise ValueError("Qwen3.8 holdout gate must require the exact V3 canary outcome")
    if gate["required_verifier_execution_id"] != "nonzero_uuid":
        raise ValueError("Qwen3.8 holdout gate must require an authoritative verifier UUID")
    if gate["required_cleanup_verified"] is not True:
        raise ValueError("Qwen3.8 holdout gate must require verified cleanup")
    _uuid(gate["task_version_id"], "Qwen3.8 holdout gate task version")
    return gate


def build_gate_receipt(
    plan: dict[str, Any], calibration_receipt_path: Path, campaign_root: Path
) -> dict[str, Any]:
    gate = validate_launch_gate(plan)
    calibration_receipt = _load(calibration_receipt_path)
    if calibration_receipt.get("receipt_sha256") != _digest_without(
        calibration_receipt, "receipt_sha256"
    ):
        raise ValueError("V3 frozen calibration receipt digest mismatch")
    if calibration_receipt["receipt_sha256"] != gate["frozen_receipt_sha256"]:
        raise ValueError("V3 frozen calibration receipt does not match the holdout gate")
    if calibration_receipt.get("campaign_id") != gate["campaign_id"]:
        raise ValueError("V3 frozen calibration receipt campaign drifted")
    tasks = calibration_receipt.get("tasks") or []
    if len(tasks) != 20 or tasks[0].get("index") != gate["task_index"]:
        raise ValueError("V3 frozen calibration receipt lacks the exact canary slot")
    canary = tasks[0]
    if canary.get("task_version_id") != gate["task_version_id"]:
        raise ValueError("V3 frozen calibration canary task version drifted")

    campaign_state_path = campaign_root / "campaign-state.json"
    task_root = campaign_root / "task-01"
    paths = {
        "campaign_state": campaign_state_path,
        "binding": task_root / "binding.json",
        "runtime_binding": task_root / "runtime-binding.json",
        "reward_result": task_root / "reward-result.json",
        "result": task_root / "result.json",
        "cleanup": task_root / "cleanup.json",
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"V3 canary terminal evidence is incomplete: {', '.join(missing)}")
    values = {label: _load(path) for label, path in paths.items()}

    campaign_state = values["campaign_state"]
    if campaign_state.get("campaign_id") != gate["campaign_id"]:
        raise ValueError("V3 campaign-state identity drifted")
    outcomes = campaign_state.get("outcomes") or []
    matches = [row for row in outcomes if row.get("index") == gate["task_index"]]
    if len(matches) != 1:
        raise ValueError("V3 campaign-state does not contain one exact canary outcome")
    outcome = matches[0]
    if outcome.get("status") != gate["required_outcome_status"]:
        raise ValueError("V3 canary is not an authoritative model outcome")
    if outcome.get("cleanup_verified") is not True:
        raise ValueError("V3 canary campaign-state does not verify cleanup")
    if outcome.get("task_version_id") != gate["task_version_id"]:
        raise ValueError("V3 canary campaign-state task version drifted")

    binding = values["binding"]
    runtime = values["runtime_binding"]
    reward = values["reward_result"]
    result = values["result"]
    cleanup = values["cleanup"]
    expected_task = {
        "key": canary["task_key"],
        "version_id": canary["task_version_id"],
    }
    if {field: (binding.get("task") or {}).get(field) for field in expected_task} != expected_task:
        raise ValueError("V3 binding does not match the frozen canary task")
    for field in ("repository", "revision", "served_id", "endpoint_origin"):
        if (binding.get("model") or {}).get(field) != plan["model"].get(field):
            raise ValueError(f"V3 canary model {field} does not match the holdout")
    for field in ("name", "version", "source_commit"):
        if (binding.get("harness") or {}).get(field) != plan["harness"].get(field):
            raise ValueError(f"V3 canary harness {field} does not match the holdout")

    run_id = binding.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith(f"{gate['campaign_id']}-t01-"):
        raise ValueError("V3 canary run identity drifted")
    instance_id = _uuid(runtime.get("instance_id"), "V3 runtime instance")
    evidence_run_id = _uuid(runtime.get("evidence_run_id"), "V3 evidence run")
    if reward.get("task_key") != canary["task_key"]:
        raise ValueError("V3 reward task key drifted")
    if reward.get("task_version_id") != gate["task_version_id"]:
        raise ValueError("V3 reward task version drifted")
    if reward.get("instance_id") != instance_id:
        raise ValueError("V3 reward instance identity drifted")
    score = _score(reward.get("reward"), "V3 reward")
    verifier_execution_id = _uuid(
        reward.get("verifier_execution_id"), "V3 verifier execution"
    )
    if result.get("run_id") != run_id:
        raise ValueError("V3 result run identity drifted")
    if _score(result.get("score"), "V3 result score") != score:
        raise ValueError("V3 result score disagrees with authoritative reward")
    if result.get("verifier_execution_id") != verifier_execution_id:
        raise ValueError("V3 result verifier identity disagrees with authoritative reward")
    if _score(outcome.get("score"), "V3 campaign outcome score") != score:
        raise ValueError("V3 campaign outcome disagrees with authoritative reward")
    if outcome.get("verifier_execution_id") != verifier_execution_id:
        raise ValueError("V3 campaign outcome verifier identity drifted")
    if cleanup.get("containers_removed") is not True:
        raise ValueError("V3 canary containers were not removed")
    if cleanup.get("instance_created") is True and cleanup.get("instance_closed") is not True:
        raise ValueError("V3 canary instance was not closed")

    receipt: dict[str, Any] = {
        "schema_version": GATE_RECEIPT_SCHEMA,
        "launch_gate": gate,
        "canary": {
            "run_id": run_id,
            "task_version_id": gate["task_version_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "outcome_status": "model_outcome",
            "score": score,
            "verifier_execution_id": verifier_execution_id,
            "cleanup_verified": True,
        },
        "source_digests": {
            "calibration_receipt": self_hosted.sha256(calibration_receipt_path.read_bytes()),
            **{label: self_hosted.sha256(path.read_bytes()) for label, path in paths.items()},
        },
    }
    receipt["receipt_sha256"] = self_hosted.sha256(self_hosted.canonical_json(receipt))
    return receipt


def validate_gate_receipt(plan: dict[str, Any], receipt: dict[str, Any]) -> None:
    gate = validate_launch_gate(plan)
    if receipt.get("schema_version") != GATE_RECEIPT_SCHEMA:
        raise ValueError("unsupported Qwen3.8 holdout release receipt")
    if receipt.get("receipt_sha256") != _digest_without(receipt, "receipt_sha256"):
        raise ValueError("Qwen3.8 holdout release receipt digest mismatch")
    if receipt.get("launch_gate") != gate:
        raise ValueError("Qwen3.8 holdout release receipt gate drifted")
    canary = receipt.get("canary") or {}
    if canary.get("task_version_id") != gate["task_version_id"]:
        raise ValueError("Qwen3.8 holdout release receipt task drifted")
    if canary.get("outcome_status") != "model_outcome":
        raise ValueError("Qwen3.8 holdout release receipt lacks a model outcome")
    _score(canary.get("score"), "Qwen3.8 holdout release receipt score")
    _uuid(canary.get("verifier_execution_id"), "Qwen3.8 holdout release receipt verifier")
    _uuid(canary.get("instance_id"), "Qwen3.8 holdout release receipt instance")
    _uuid(canary.get("evidence_run_id"), "Qwen3.8 holdout release receipt evidence run")
    if canary.get("cleanup_verified") is not True:
        raise ValueError("Qwen3.8 holdout release receipt lacks verified cleanup")
    source_digests = receipt.get("source_digests") or {}
    expected_sources = {
        "calibration_receipt",
        "campaign_state",
        "binding",
        "runtime_binding",
        "reward_result",
        "result",
        "cleanup",
    }
    if set(source_digests) != expected_sources or any(
        not isinstance(value, str) or not value.startswith("sha256:")
        for value in source_digests.values()
    ):
        raise ValueError("Qwen3.8 holdout release receipt source digests are incomplete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--gate-receipt", type=Path)
    parser.add_argument("--receipt-out", type=Path)
    args = parser.parse_args()
    plan = _load(args.config)
    if args.command == "build":
        if not args.calibration_receipt or not args.campaign_root or not args.receipt_out:
            raise ValueError(
                "build requires --calibration-receipt, --campaign-root, and --receipt-out"
            )
        receipt = build_gate_receipt(plan, args.calibration_receipt, args.campaign_root)
        self_hosted.write_json_once(args.receipt_out, receipt)
    else:
        if not args.gate_receipt:
            raise ValueError("validate requires --gate-receipt")
        receipt = _load(args.gate_receipt)
        validate_gate_receipt(plan, receipt)
    print(
        json.dumps(
            {
                "ok": True,
                "campaign_id": plan["campaign_id"],
                "launch_gate_campaign_id": plan["launch_gate"]["campaign_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
