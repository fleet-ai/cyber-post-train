"""Build and audit inert Fresh75 checkpoint export/reload/staging packets.

The packets contain no submission operation.  ``render`` and ``verify`` are
offline.  ``audit-sfs`` reads the exact saved checkpoint receipts and source
directories through an already-mounted SFS filesystem; it never creates or
changes cluster or SFS state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from training.checkpoints import checkpoint_files
from training.io import digest_json, file_sha256
from training.post_sft_artifacts import QWEN36_EXACT_MTP_OMISSION_KEYS
from training.sft_runtime import _unsigned_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = REPO_ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-queue-v1.json"
DEFAULT_PACKETS = (
    REPO_ROOT / "configs/qualification/qwen38-fresh75-v2-export-reload-stage-packets-v1.json"
)
SCHEMA = "qwen38_fresh75_v2_export_reload_stage_packets_v1"
PACKET_SCHEMA = "qwen38_fresh75_zero_step_export_reload_stage_packet_v1"
PRIORITY = (
    "b32-lr1e5-e2",
    "b16-lr1e5-e2",
    "b8-lr1e5-e2",
    "b8-lr3e6-e2",
    "b64-lr1e5-e2",
    "b8-lr1e6-e2",
    "b8-lr1e5-e1",
)
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
TRAIN_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@"
    "sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
STAGE_IMAGE = (
    "lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
)
MODEL = {
    "repository": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "weight_manifest_sha256": (
        "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
    ),
    "tokenizer_manifest_sha256": (
        "sha256:3938a9a8172f2738fed1be44efc11e2562059269d50d3721213f44802b53b4e1"
    ),
    "chat_template_sha256": (
        "sha256:c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    ),
}
CODE_FILES = (
    "training/fresh75_export_packets.py",
    "training/checkpoints.py",
    "training/export.py",
    "training/export_check.py",
    "training/model_stage.py",
    "training/sft_runtime.py",
    "training/post_sft_cast.py",
    "training/post_sft_artifacts.py",
    "training/post_sft_base_surface.py",
    "training/io.py",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a prefixed SHA-256")
    return value


def _reference(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "file_sha256": file_sha256(path),
    }


def _safe_slug(arm_id: str) -> str:
    value = arm_id.replace("-", "")
    if re.fullmatch(r"[a-z0-9]{1,20}", value) is None:
        raise ValueError("arm identifier cannot form a bounded resource slug")
    return value


def _build_arm(queue_arm: dict[str, Any], *, priority_rank: int, repo_root: Path) -> dict[str, Any]:
    arm_id = queue_arm["arm_id"]
    handoff_path = repo_root / queue_arm["handoff"]["path"]
    if file_sha256(handoff_path) != queue_arm["handoff"]["file_sha256"]:
        raise ValueError(f"{arm_id} handoff file changed")
    handoff = _read(handoff_path)
    source_ref = handoff["source_plan"]
    source_path = repo_root / source_ref["path"]
    if file_sha256(source_path) != source_ref["file_sha256"]:
        raise ValueError(f"{arm_id} source plan file changed")
    source = _read(source_path)
    source_digest = "sha256:" + _unsigned_digest(source)
    checkpoint = queue_arm["checkpoint"]
    if any(
        (
            handoff.get("arm_id") != arm_id,
            handoff.get("status") != "training_accepted_waiting_for_checkpoint_payload_seal",
            handoff.get("handoff_sha256") != queue_arm["handoff"]["handoff_sha256"],
            source_digest != source_ref["plan_sha256"],
            source.get("output_root") != str(Path(checkpoint["path"]).parents[1]),
            source.get("recipe", {}).get("max_steps") != checkpoint["optimizer_step"],
            source.get("model", {}).get("repo") != MODEL["repository"],
            source.get("model", {}).get("revision") != MODEL["revision"],
            source.get("model", {}).get("weight_manifest_sha256")
            != MODEL["weight_manifest_sha256"],
            source.get("execution", {}).get("image") != TRAIN_IMAGE,
        )
    ):
        raise ValueError(f"{arm_id} source plan differs from its accepted handoff")
    slug = _safe_slug(arm_id)
    step = checkpoint["optimizer_step"]
    run_root = source["output_root"]
    seal_root = f"{run_root}/checkpoint-seals-step{step}-fleet-v1"
    manifest = f"{seal_root}/step-{step}.json"
    export_root = f"{run_root}/hf-export-step{step}-fleet-v1"
    export_receipt = f"{export_root}/EXPORT.json"
    cpu_receipt = f"{run_root}/export-check-step{step}-cpu-fleet-v1.json"
    gpu_root = f"/mnt/sfs/jobs/chris-q38-f75v2-{slug}-gpu-reload-v1"
    gpu_receipt = f"{gpu_root}/GPU_CHECK.json"
    served_id = queue_arm["expected_served_id"]
    stage_root = f"/models/{served_id}"
    stage_plan = f"configs/qualification/generated/{served_id}-stage-v1.json"
    return {
        "schema": PACKET_SCHEMA,
        "arm_id": arm_id,
        "priority_rank": priority_rank,
        "status": "no_submit_blocked_waiting_sfs_payload_seal",
        "launchable": False,
        "submitted": False,
        "external_mutation_authorized": False,
        "accepted_handoff": {
            **_reference(handoff_path, repo_root),
            "handoff_sha256": handoff["handoff_sha256"],
            "training_observation_sha256": handoff["training_observation_sha256"],
        },
        "source_plan": {
            **_reference(source_path, repo_root),
            "plan_sha256": source_digest,
            "runtime_sha256": source_ref["runtime_sha256"],
            "reconstruction_status": source_ref["reconstruction_status"],
        },
        "source_checkpoint": {
            "path": checkpoint["path"],
            "optimizer_step": step,
            "world_size": source["recipe"]["nodes"] * source["recipe"]["gpus_per_node"],
            "saved_receipt_path": checkpoint["saved_receipt_path"],
            "saved_receipt_file_sha256": checkpoint["saved_receipt_file_sha256"],
            "saved_receipt_sha256": checkpoint["saved_receipt_sha256"],
            "payload_seal_status": checkpoint["payload_seal_status"],
            "payload_manifest_path": manifest,
            "payload_manifest_file_sha256": None,
            "payload_manifest_receipt_sha256": None,
            "read_only_preflight": {
                "require_saved_receipt_file_and_embedded_digests": True,
                "require_exact_plan_step_and_checkpoint_path": True,
                "reject_symlinks_and_unknown_checkpoint_files": True,
                "rehash_every_checkpoint_file": True,
                "prove_size_and_mtime_inventory_unchanged": True,
            },
        },
        "phase_state": {
            "seal": "blocked_until_read_only_saved_receipt_and_source_directory_preflight",
            "export": "blocked_until_payload_manifest_file_and_embedded_digests_are_frozen",
            "cpu_reload": "blocked_until_export_file_and_embedded_digests_are_frozen",
            "one_gpu_reload": "blocked_until_cpu_check_acceptance_and_new_failure_budget",
            "staging": "blocked_until_exact_export_and_gpu_reload_receipts_are_frozen",
            "serving": "blocked_until_staging_acceptance",
        },
        "destinations": {
            "checkpoint_manifest": manifest,
            "export_root": export_root,
            "export_receipt": export_receipt,
            "cpu_check_receipt": cpu_receipt,
            "gpu_reload_request_name": f"chris-q38-f75v2-{slug}-gpu-v1",
            "gpu_reload_root": gpu_root,
            "gpu_check_receipt": gpu_receipt,
            "stage_plan": stage_plan,
            "stage_source_export_receipt": export_receipt.removeprefix("/mnt/sfs"),
            "stage_source_gpu_check_receipt": gpu_receipt.removeprefix("/mnt/sfs"),
            "staged_model_root": stage_root,
            "staged_acceptance": f"{stage_root}/.fleet-acceptance.json",
            "served_id": served_id,
        },
    }


def build_packet_set(queue_path: Path = DEFAULT_QUEUE, *, repo_root: Path = REPO_ROOT) -> dict:
    queue = _read(queue_path)
    if (
        queue.get("schema") != "qwen38_fresh75_fleet_dev17_queue_v1"
        or queue.get("launch_authorized") is not False
        or queue.get("external_mutation_authorized") is not False
        or queue.get("failure_budget", {}).get("remaining") != 0
    ):
        raise ValueError("export packets require the frozen, mutation-free exhausted queue")
    arms = {arm["arm_id"]: arm for arm in queue.get("accepted_sft_arms", [])}
    if set(arms) != set(PRIORITY):
        raise ValueError("accepted Fresh75 arm set changed")
    code = {relative: file_sha256(repo_root / relative) for relative in CODE_FILES}
    unsigned = {
        "schema": SCHEMA,
        "status": "no_submit_blocked_waiting_sfs_payload_seals",
        "launchable": False,
        "submitted": False,
        "external_mutation_authorized": False,
        "failure_budget": {"limit": 10, "consumed": 10, "remaining": 0},
        "source_queue": _reference(queue_path, repo_root),
        "model": MODEL,
        "runtime": {
            "training_image": TRAIN_IMAGE,
            "staging_image": STAGE_IMAGE,
            "code_sha256": code,
        },
        "shared_contract": {
            "source_reads": "already-mounted SFS only; no reader workload creation",
            "checkpoint_seal": "full file-by-file rehash with source sizes/mtimes unchanged",
            "export": (
                "CPU-only BF16 reconstruction with zero optimizer steps and atomic "
                "no-replace publication"
            ),
            "reload": (
                "CPU meta-layout check followed by a separately authorized one-GPU "
                "finite synthetic check"
            ),
            "staging": "zero-GPU receipt-first stream copy with atomic no-replace destination",
            "registration": (
                "out of scope until staging acceptance; paused create-once route with a new UID"
            ),
            "external_benchmark_content_or_outcomes_used": False,
        },
        "phase_contracts": {
            "seal": {
                "operation": "training.checkpoints.seal",
                "gpu_count": 0,
                "create_once": True,
                "expected_schema": "cyber_skyrl_checkpoint_manifest_v1",
                "expected_world_size": 8,
                "expected_gpu_reload_verified": False,
            },
            "export": {
                "operation": "training.export.export",
                "gpu_count": 0,
                "create_once": True,
                "manifest_file_sha256_required": True,
                "expected": {
                    "schema": "cyber_native_checkpoint_hf_export_v1",
                    "model_repo": MODEL["repository"],
                    "model_revision": MODEL["revision"],
                    "optimizer_steps_executed": 0,
                    "gpu_reload_verified": False,
                    "trained_tensors": 1184,
                    "restored_base_tensors": sorted(QWEN36_EXACT_MTP_OMISSION_KEYS),
                    "tensor_count": 1199,
                    "tensor_bytes": 55562855904,
                    "dtype": "BF16",
                    "source_inventory_sizes_mtimes_unchanged": True,
                    "all_output_tensors_reopened_equal": True,
                },
            },
            "cpu_reload": {
                "operation": "training.export_check.check",
                "gpu_count": 0,
                "create_once": True,
                "export_file_sha256_required": True,
                "expected": {
                    "schema": "cyber_hf_export_check_v1",
                    "status": "passed",
                    "optimizer_steps_executed": 0,
                    "gpus": 0,
                    "gpu_reload_verified": False,
                    "serving_qualified": False,
                    "synthetic_only": True,
                    "source_unchanged": True,
                },
            },
            "one_gpu_reload": {
                "operation": "training.export_check.check",
                "workers": 1,
                "gpus_per_worker": 1,
                "priority_class": "c1",
                "expected_queue_priority_class": "q1",
                "requeue_if_preempted": False,
                "create_once": True,
                "export_file_sha256_required": True,
                "expected": {
                    "schema": "cyber_hf_export_check_v1",
                    "status": "passed",
                    "optimizer_steps_executed": 0,
                    "gpus": 1,
                    "gpu_reload_verified": True,
                    "serving_qualified": False,
                    "synthetic_only": True,
                    "source_unchanged": True,
                    "attention_implementation": "eager",
                    "finite_logits": True,
                    "generated_tokens": 2,
                },
                "terminal_gate": (
                    "UID-bound success plus exact Pod/RayCluster/workload and "
                    "GPU-allocation absence"
                ),
            },
            "staging": {
                "operation": "training.model_stage.run",
                "namespace": "inference",
                "priority_class": "c1",
                "gpu_count": 0,
                "restart_policy": "Never",
                "active_deadline_seconds": 5400,
                "image": STAGE_IMAGE,
                "runtime_source_sha256": (
                    "sha256:ab43d324d46e35712534f3ed8418807a924a7505feb30416538fc36b38fc01b6"
                ),
                "atomic_transaction": "directory_rename_noreplace_v1",
                "destination_must_be_absent": True,
                "payload": {
                    "mapping": "exact EXPORT.files object with bare per-file SHA-256 values",
                    "expected_file_count": 29,
                    "maximum_file_bytes": 4 * 1024**3,
                    "maximum_total_bytes": 64 * 1024**3,
                },
                "resource_shape": {
                    "cpu_request": "4",
                    "cpu_limit": "8",
                    "memory_request": "8Gi",
                    "memory_limit": "16Gi",
                    "ephemeral_storage_request": "1Gi",
                    "ephemeral_storage_limit": "2Gi",
                },
                "registration": None,
            },
            "serving": {
                "route_initial_state": "paused",
                "new_route_uid_required": True,
                "base_model_id": "qwen3.8-27b",
                "precision": "bf16",
                "quantization": "none",
                "max_context_size": 262144,
                "live_parity_required_before_evaluation": True,
            },
        },
        "ordered_gates": [
            "read_exact_saved_checkpoint_receipt_from_sfs_and_match_both_digests",
            "seal_and_full_rehash_exact_checkpoint_payload_create_once",
            "freeze_payload_manifest_file_and_embedded_receipt_digests",
            "cpu_zero_optimizer_step_bf16_export_create_once",
            "cpu_export_integrity_and_meta_reload_check",
            "new_failure_budget_and_explicit_one_gpu_reload_authorization",
            "one_gpu_zero_optimizer_step_finite_reload_and_complete_gpu_release",
            "freeze_exact_export_and_gpu_reload_receipts_into_stage_plan",
            "destination_absence_then_atomic_create_once_zero_gpu_staging",
            "paused_route_registration_with_new_uid_and_live_base_candidate_parity",
        ],
        "packets": [
            _build_arm(arms[arm_id], priority_rank=index, repo_root=repo_root)
            for index, arm_id in enumerate(PRIORITY, 1)
        ],
        "execution_record": {
            "kind": "offline_no_submit_packets",
            "external_api_calls_performed": 0,
            "cluster_calls_performed": 0,
            "jobs_submitted": 0,
            "resources_created": 0,
            "sfs_writes_performed": 0,
        },
    }
    return {**unsigned, "packet_set_sha256": digest_json(unsigned)}


def verify_packet_set(
    packet_path: Path = DEFAULT_PACKETS,
    *,
    queue_path: Path = DEFAULT_QUEUE,
    repo_root: Path = REPO_ROOT,
) -> dict:
    observed = _read(packet_path)
    expected = build_packet_set(queue_path, repo_root=repo_root)
    if observed != expected:
        raise ValueError("packet set differs from deterministic generator output")
    return {
        "status": "valid_no_submit_packet_set",
        "packet_set_sha256": observed["packet_set_sha256"],
        "file_sha256": file_sha256(packet_path),
        "packets": len(observed["packets"]),
        "launchable": False,
    }


def _map_sfs(logical: str, mount_root: Path) -> Path:
    path = PurePosixPath(logical)
    if not path.is_absolute() or path.parts[:3] != ("/", "mnt", "sfs") or ".." in path.parts:
        raise ValueError("packet source is not an absolute /mnt/sfs path")
    return mount_root.joinpath(*path.parts[3:])


def _read_regular_nofollow(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("SFS receipt must be a regular file")
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
            if len(payload) > 2 * 1024 * 1024:
                raise ValueError("SFS receipt exceeds its size bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if identity(before) != identity(after):
        raise ValueError("SFS receipt changed while read")
    return bytes(payload)


def audit_sfs_saved_receipts(
    packet_path: Path = DEFAULT_PACKETS, *, mount_root: Path = Path("/mnt/sfs")
) -> dict:
    """Read the seven exact saved receipts and source layouts without writing SFS."""
    packets = _read(packet_path)
    if packets.get("packet_set_sha256") != digest_json(
        {key: value for key, value in packets.items() if key != "packet_set_sha256"}
    ):
        raise ValueError("packet set digest changed")
    audited = []
    for packet in packets.get("packets", []):
        source = packet["source_checkpoint"]
        receipt_path = _map_sfs(source["saved_receipt_path"], mount_root)
        raw = _read_regular_nofollow(receipt_path)
        if "sha256:" + hashlib.sha256(raw).hexdigest() != source["saved_receipt_file_sha256"]:
            raise ValueError(f"{packet['arm_id']} saved receipt file digest changed")
        receipt = json.loads(raw)
        if (
            not isinstance(receipt, dict)
            or "sha256:" + str(receipt.get("receipt_sha256")) != source["saved_receipt_sha256"]
            or receipt.get("receipt_sha256")
            != _unsigned_digest(
                {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            )
            or receipt.get("optimizer_step") != source["optimizer_step"]
            or receipt.get("checkpoint_path") != source["path"]
            or "sha256:" + str(receipt.get("plan_sha256")) != packet["source_plan"]["plan_sha256"]
        ):
            raise ValueError(f"{packet['arm_id']} saved receipt binding changed")
        checkpoint = _map_sfs(source["path"], mount_root)
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError(f"{packet['arm_id']} checkpoint directory is absent or a symlink")
        inventory = checkpoint_files(checkpoint, source["world_size"])
        names = sorted(inventory)
        audited.append(
            {
                "arm_id": packet["arm_id"],
                "optimizer_step": source["optimizer_step"],
                "saved_receipt_file_sha256": source["saved_receipt_file_sha256"],
                "checkpoint_files_observed": len(names),
                "checkpoint_bytes_observed": sum(
                    path.stat().st_size for path in inventory.values()
                ),
                "checkpoint_names_sha256": digest_json(names),
                "source_bytes_modified": False,
            }
        )
    return {
        "status": "read_only_saved_receipt_and_layout_audit_passed",
        "external_api_calls_performed": 0,
        "sfs_writes_performed": 0,
        "audited": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="print deterministic packets to stdout")
    render.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    verify = subparsers.add_parser("verify", help="verify a committed packet set")
    verify.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    verify.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    audit = subparsers.add_parser("audit-sfs", help="read exact SFS receipts/layouts only")
    audit.add_argument("--packets", type=Path, default=DEFAULT_PACKETS)
    audit.add_argument("--mount-root", type=Path, default=Path("/mnt/sfs"))
    args = parser.parse_args()
    if args.command == "render":
        result = build_packet_set(args.queue)
    elif args.command == "verify":
        result = verify_packet_set(args.packets, queue_path=args.queue)
    else:
        result = audit_sfs_saved_receipts(args.packets, mount_root=args.mount_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
