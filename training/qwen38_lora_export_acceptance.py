"""Validate the inert handoff from accepted LoRA export to later work.

The handoff records independently checked public receipt facts and exact
Kubernetes identities. It cannot stage or serve a model, launch an evaluation,
or submit broad training. Any later consumer must reopen the remote export
receipt, validate it with :mod:`training.qwen38_lora_artifacts`, and preserve
the same file and self digests before external mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest

SCHEMA = "cyber_qwen38_lora_export_acceptance_handoff_v1"


class LoraExportAcceptanceError(ValueError):
    """The accepted-export handoff is incomplete or inconsistent."""


def _need(condition: object, reason: str) -> None:
    if not condition:
        raise LoraExportAcceptanceError(reason)


def _exact(value: object, fields: set[str], reason: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping), reason)
    result = dict(value)
    _need(set(result) == fields, reason)
    return result


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_file(root: Path, value: object, expected_sha256: object) -> Path:
    _need(isinstance(value, str) and value and "\x00" not in value, "invalid_plan_path")
    relative = Path(value)
    _need(not relative.is_absolute() and ".." not in relative.parts, "unsafe_plan_path")
    path = (root / relative).resolve(strict=True)
    _need(path.is_file() and path.is_relative_to(root.resolve()), "missing_plan")
    _need(_file_sha256(path) == expected_sha256, "plan_file_digest_mismatch")
    return path


def validate_acceptance(value: object, *, repository_root: Path) -> dict[str, Any]:
    handoff = _exact(
        value,
        {
            "schema",
            "status",
            "source_export_plan",
            "execution",
            "export_receipt",
            "resource_release",
            "broad_training_anchor",
            "serving_and_evaluation",
            "remaining_gates",
            "handoff_sha256",
        },
        "invalid_handoff",
    )
    _need(handoff["schema"] == SCHEMA, "invalid_schema")
    _need(
        handoff["status"] == "export_accepted_waiting_for_immutable_stage",
        "invalid_status",
    )
    expected_handoff_sha256 = digest(
        {key: item for key, item in handoff.items() if key != "handoff_sha256"}
    ).removeprefix("sha256:")
    _need(handoff["handoff_sha256"] == expected_handoff_sha256, "handoff_digest_mismatch")

    plan_ref = _exact(
        handoff["source_export_plan"],
        {"path", "file_sha256", "plan_sha256", "request_sha256"},
        "invalid_source_export_plan",
    )
    plan_path = _repo_file(repository_root, plan_ref["path"], plan_ref["file_sha256"])
    plan = json.loads(plan_path.read_bytes())
    _need(
        digest(plan).removeprefix("sha256:") == plan_ref["plan_sha256"]
        and plan_ref["request_sha256"]
        == "9a9a9566c8554b2ba5a8e8c10b14bed24ad4dd158a442e5dff03028a01e76f11"
        and plan.get("run_name") == "chris-q38-lora-prod-exp-v1"
        and plan.get("optimizer_steps_executed") == 0
        and plan.get("priority_class") == "c1",
        "source_export_plan_mismatch",
    )

    execution = _exact(
        handoff["execution"],
        {
            "rayjob_name",
            "rayjob_uid",
            "workload_uid",
            "raycluster_name",
            "raycluster_uid",
            "pod_name",
            "pod_uid",
            "start_time",
            "end_time",
            "job_status",
            "restart_count",
        },
        "invalid_execution",
    )
    _need(
        execution
        == {
            "rayjob_name": "chris-q38-lora-prod-exp-v1-34e3dcf5",
            "rayjob_uid": "30279c0f-2ef5-4003-9045-eb318961c4aa",
            "workload_uid": "e5ad6724-379f-4bff-9992-7dca246b1a49",
            "raycluster_name": "chris-q38-lora-prod-exp-v1-34e3dcf5-fgm9s",
            "raycluster_uid": "f140b798-ca93-4151-b027-af49f8cf65c3",
            "pod_name": "chris-q38-lora-prod-exp-v1-34e3dcf5-fgm9s-head-zx65s",
            "pod_uid": "c621bd6c-cefd-4f4a-b206-df50c9515a84",
            "start_time": "2026-09-20T15:13:41Z",
            "end_time": "2026-09-20T15:40:48Z",
            "job_status": "SUCCEEDED",
            "restart_count": 0,
        },
        "execution_identity_mismatch",
    )

    export = _exact(
        handoff["export_receipt"],
        {
            "path",
            "file_sha256",
            "receipt_sha256",
            "schema",
            "output_root",
            "optimizer_step",
            "optimizer_steps_executed",
            "dtype",
            "merge_method",
            "base_tensor_count",
            "merged_tensor_count",
            "tensor_bytes",
            "source_base_unchanged",
            "source_checkpoint_unchanged",
            "adapter_checkpoint_reload_verified",
            "optimizer_resume_verified",
            "all_output_tensors_reopened_equal",
            "deterministic_merge",
            "merged_model_reload_verified",
            "finite_logits",
            "gpu_reload_verified",
            "remote_revalidation_required_before_stage",
        },
        "invalid_export_receipt_reference",
    )
    _need(
        export
        == {
            "path": "/mnt/sfs/jobs/chris-q38-lora-prod-exp-v1/QWEN38_LORA_MERGED_HF_EXPORT.json",
            "file_sha256": "bfeedcac2a82f25696e7d102de3b5f509ded9539a9a07d65c39a94a1c58bc7a2",
            "receipt_sha256": "237d90b34377599284c2358cd2611dfe2487a812a7179fc7460ed6ff3541ca76",
            "schema": "cyber_qwen38_megatron_lora_merged_hf_export_v1",
            "output_root": "/mnt/sfs/jobs/chris-q38-lora-prod-exp-v1/merged-hf",
            "optimizer_step": 1,
            "optimizer_steps_executed": 0,
            "dtype": "BF16",
            "merge_method": "megatron_bridge_lora_merge_v1",
            "base_tensor_count": 1199,
            "merged_tensor_count": 1199,
            "tensor_bytes": 55_562_855_904,
            "source_base_unchanged": True,
            "source_checkpoint_unchanged": True,
            "adapter_checkpoint_reload_verified": True,
            "optimizer_resume_verified": True,
            "all_output_tensors_reopened_equal": True,
            "deterministic_merge": True,
            "merged_model_reload_verified": True,
            "finite_logits": True,
            "gpu_reload_verified": True,
            "remote_revalidation_required_before_stage": True,
        },
        "export_receipt_reference_mismatch",
    )

    release = _exact(
        handoff["resource_release"],
        {"observed_at", "pod_count", "raycluster_count", "allocated_gpus"},
        "invalid_resource_release",
    )
    _need(
        release
        == {
            "observed_at": "2026-09-20T15:45:32Z",
            "pod_count": 0,
            "raycluster_count": 0,
            "allocated_gpus": 0,
        },
        "resource_release_mismatch",
    )

    _need(
        handoff["broad_training_anchor"]
        == {
            "operational_gate": "accepted_one_step_lora_and_reload",
            "allows_broad_lora_preflight": True,
            "allows_external_submit": False,
            "submit_blocker": "global_cluster_failure_budget_reconciliation",
        },
        "broad_training_anchor_mismatch",
    )
    _need(
        handoff["serving_and_evaluation"]
        == {
            "external_mutation_authorized": False,
            "serving_model_id": "chris-q38-lora-prod-s1-wbe-v1",
            "route_initial_state": "paused",
            "priority": "c1",
            "webexploitbench_canary_pair_id": "q38-lora-prod-s1-oc-wbe-c1",
            "fleet_development_campaign_id": "q38-lora-prod-s1-fleet-dev17-p1",
            "webexploitbench_collection": "score_free_immutable_rollout_bundles",
            "webexploitbench_scoring": "separate_resumable_stage",
            "fleet_final_test": "closed_until_development_selection",
        },
        "serving_and_evaluation_mismatch",
    )
    _need(
        handoff["remaining_gates"]
        == [
            "reopen_and_validate_remote_export_receipt",
            "immutable_stage_with_full_payload_manifest",
            "create_once_paused_c1_serving_route_with_new_uid",
            "live_base_candidate_serving_parity",
            "fresh_tensorlake_duplicate_census",
            "paired_score_free_webexploitbench_task0_canary",
            "frozen_fleet_development_evaluation",
            "separate_webexploitbench_scoring",
        ],
        "remaining_gates_mismatch",
    )
    return handoff


def load_acceptance(path: Path, *, repository_root: Path) -> dict[str, Any]:
    before = path.read_bytes()
    try:
        value = json.loads(before)
    except json.JSONDecodeError as exc:
        raise LoraExportAcceptanceError("invalid_json") from exc
    result = validate_acceptance(value, repository_root=repository_root)
    _need(path.read_bytes() == before, "handoff_changed_during_read")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    value = load_acceptance(args.handoff.resolve(), repository_root=args.repository_root.resolve())
    print(value["handoff_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
