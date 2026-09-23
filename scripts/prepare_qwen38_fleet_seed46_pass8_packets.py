#!/usr/bin/env python3
"""Seal matched pass@1 replicas for Qwen3.8 base versus Teacher3K step-1000.

This command is deliberately launch-inert.  It consumes one fresh, prompt-free
live-parity receipt, verifies that the existing immutable dev17 task roster was
excluded from the Teacher3K corpus, and writes eight matched pass@1 packet pairs
for fixed seeds 46 through 53.  This preserves exact pass@8 coverage without
placing 136 long sessions under one 48-hour Kubernetes Job deadline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import model_artifact, model_artifact_v3
from scripts import prepare_qwen38_fleet_seed44_two_arm_packets as shared

ROOT = Path(__file__).resolve().parents[1]
BASE_TEMPLATE = ROOT / "configs/evaluation/qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json"
TASK_SET = ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-task-set-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
CORPUS = ROOT / "configs/data/qwen38-teacher3k-32k-v1.manifest.json"
PROMOTION = ROOT / "docs/evidence/qwen38-teacher3k32-step1000-promotion-20260922.json"
STAGE_PLAN = ROOT / "configs/qualification/qwen38-teacher3k32-step1000-inference-stage-v1.json"
ARTIFACT_ACCEPTANCE = (
    ROOT / "docs/evidence/qwen38-teacher3k32-step1000-eval-artifact-accepted-20260923.json"
)
LEDGER = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v2.json"
BINDING_ROSTER = (
    ROOT / "configs/evaluation/qwen38-fleet-dev17-exact-binding-roster-20260922-v1.json"
)

PROTOCOL_STUDY_ID = "q38-dev17-seeds46to53-base-step1000-p8-v1"
SEEDS = tuple(range(46, 54))
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CANDIDATE_REVISION = "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
CANDIDATE_ID = "chris-q38-t3k32-s1000-v1"
ARTIFACT_ALIAS = "teacher3k32-step1000"
V3_SOURCE_FILES = {
    **shared.SOURCE_FILES,
    "model_artifact_v2.py": ROOT / "evals/fleet/model_artifact_v2.py",
    "model_artifact_v3.py": ROOT / "evals/fleet/model_artifact_v3.py",
    "run.sh": ROOT / "evals/fleet/scripts/run_qwen38_dev17_single_arm_v3.sh",
}


def _canonical(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _encoded(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_json(path: Path, label: str) -> dict[str, Any]:
    value = shared._read_json(path, label)  # noqa: SLF001
    expected = value.get("sha256")
    actual = _canonical({key: item for key, item in value.items() if key != "sha256"})
    if expected not in {actual, actual.removeprefix("sha256:")}:
        raise ValueError(f"{label} self digest differs")
    return value


def _inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    base = shared._read_json(BASE_TEMPLATE, "base template")  # noqa: SLF001
    task_set = shared._read_json(TASK_SET, "dev17 task set")  # noqa: SLF001
    split = _verified_json(SPLIT, "representative split")
    corpus = _verified_json(CORPUS, "Teacher3K 32K corpus")
    roster = _verified_json(BINDING_ROSTER, "dev17 exact binding roster")
    if (
        task_set.get("schema") != "cyber_eval_task_selection_v2"
        or task_set.get("selection_role") != "dev"
        or len(task_set.get("tasks", [])) != 17
        or task_set.get("selection_sha256")
        != "sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68"
        or corpus.get("split_sha256") != split["sha256"]
        or corpus.get("dev_windows") != 0
        or corpus.get("validation_mode") != "task_outcomes_only"
        or corpus.get("catalog_provenance", {}).get(
            "held_out_task_families_excluded_across_all_versions"
        )
        != 25
        or roster.get("binding_count") != 17
        or roster.get("bindings_sha256")
        != "sha256:32230df8759bb74eb306d16cd9e32686618b529aa7c0b0ac1fa1d751ecd38728"
    ):
        raise ValueError("held-out selection or corpus exclusion contract differs")
    corpus_keys = set(corpus.get("files", {}).get("train", {}).get("task_keys", []))
    dev_keys = {row["task_key"] for row in task_set["tasks"]}
    if dev_keys & corpus_keys:
        raise ValueError("Teacher3K corpus contains an exact dev17 task key")
    split_dev = {
        (row["task_key"], row["task_version_id"])
        for row in split.get("tasks", [])
        if row.get("split") == "dev"
    }
    selected = {(row["task_key"], row["task_version_id"]) for row in task_set["tasks"]}
    if split_dev != selected:
        raise ValueError("dev17 selection differs from the frozen split")
    roster_tasks = [row.get("task", {}).get("version_id") for row in roster["bindings"]]
    if roster_tasks != [row["task_version_id"] for row in task_set["tasks"]]:
        raise ValueError("exact binding roster differs from the frozen dev17 order")
    promotion = _verified_json(PROMOTION, "step-1000 promotion")
    if any(
        (
            promotion.get("status") != "accepted_through_live_serving_parity",
            promotion.get("checkpoint", {}).get("optimizer_step") != 1000,
            promotion.get("checkpoint", {}).get("receipt_sha256")
            != "sha256:bae4b7215a6b9ebd38461555a2de02dfe822013222280ee0ca6498ad27e892b7",
            promotion.get("export", {}).get("payload_manifest_sha256") != CANDIDATE_REVISION,
            promotion.get("serving_route", {}).get("model_id") != CANDIDATE_ID,
        )
    ):
        raise ValueError("step-1000 promotion identity differs")
    return base, task_set, split, corpus, roster


def _candidate_packet(campaign_name: str) -> dict[str, Any]:
    promotion = _verified_json(PROMOTION, "step-1000 promotion")
    stage_plan = shared._read_json(STAGE_PLAN, "step-1000 stage plan")  # noqa: SLF001
    if stage_plan.get("plan_sha256") != _canonical(
        {key: item for key, item in stage_plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("step-1000 stage plan self digest differs")
    acceptance = _verified_json(ARTIFACT_ACCEPTANCE, "step-1000 evaluation acceptance")
    checkpoint = promotion["checkpoint"]
    source = stage_plan["source"]
    export = source["export_receipt"]
    gpu = source["gpu_check_receipt"]
    payload = source["payload"]
    export_required = export["required_fields"]
    gpu_required = gpu["required_fields"]
    binding = {
        "schema": model_artifact_v3.BINDING_SCHEMA,
        "checkpoint_manifest": {
            "path": checkpoint["seal_path"],
            "file_sha256": checkpoint["seal_file_sha256"],
            "receipt_sha256": checkpoint["seal_sha256"],
            "required_fields": {
                "schema": "cyber_skyrl_checkpoint_manifest_v1",
                "checkpoint_path": checkpoint["path"],
                "optimizer_step": checkpoint["optimizer_step"],
                "world_size": 8,
                "total_bytes": 324627486795,
                "gpu_reload_verified": False,
            },
        },
        "export_receipt": {
            "path": "/mnt/sfs" + export["sfs_path"],
            "file_sha256": export["file_sha256"],
            "receipt_sha256": export["receipt_sha256"],
            "required_fields": export_required,
        },
        "gpu_reload_receipt": {
            "path": "/mnt/sfs" + gpu["sfs_path"],
            "file_sha256": gpu["file_sha256"],
            "receipt_sha256": gpu["receipt_sha256"],
            "required_fields": {
                key: gpu_required[key]
                for key in (
                    "schema",
                    "status",
                    "export_sha256",
                    "export_receipt_sha256",
                    "optimizer_steps_executed",
                    "gpus",
                    "gpu_reload_verified",
                    "serving_qualified",
                    "synthetic_only",
                    "source_unchanged",
                    "attention_implementation",
                    "finite_logits",
                    "generated_tokens",
                )
            },
        },
        "payload": {
            "served_revision": CANDIDATE_REVISION,
            "revision_basis": "export_files_canonical_sha256",
            "export_files_sha256": payload["manifest_sha256"],
            "stage_source_path": f"/models/{CANDIDATE_ID}",
            "file_count": payload["file_count"],
            "trained_tensors": 1184,
            "restored_mtp_tensors": sorted(model_artifact.EXACT_MTP_TENSORS),
            "tensor_count": 1199,
            "tensor_bytes": export_required["tensor_bytes"],
        },
        "qualification": {
            "acceptance_evidence_path": str(ARTIFACT_ACCEPTANCE.relative_to(ROOT)),
            "acceptance_evidence_file_sha256": _file_sha(ARTIFACT_ACCEPTANCE),
            "acceptance_evidence_sha256": acceptance["sha256"],
            "gpu_reload_workload_kind": "Pod",
            "gpu_reload_workload_uid": promotion["gpu_reload"]["dev_pod_uid"],
            "gpu_reload_pod_uid": promotion["gpu_reload"]["dev_pod_uid"],
            "gpu_allocation_released": promotion["gpu_reload"]["temporary_gpu_released"],
        },
    }
    value = {
        "schema": model_artifact_v3.PACKET_SCHEMA,
        "campaign_name": campaign_name,
        "models": {ARTIFACT_ALIAS: binding},
        "authorization": "provenance_only_not_launch_authorization",
    }
    value["sha256"] = hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def _candidate_binding(packet: dict[str, Any], packet_name: str) -> dict[str, Any]:
    acceptance = _verified_json(ARTIFACT_ACCEPTANCE, "step-1000 evaluation acceptance")
    return {
        "validator_file_sha256": _file_sha(Path(model_artifact_v3.__file__)),
        "packet_path": packet_name,
        "packet_file_sha256": "sha256:" + hashlib.sha256(_encoded(packet)).hexdigest(),
        "packet_sha256": "sha256:" + packet["sha256"],
        "acceptance_evidence_path": str(ARTIFACT_ACCEPTANCE.relative_to(ROOT)),
        "acceptance_evidence_file_sha256": _file_sha(ARTIFACT_ACCEPTANCE),
        "acceptance_evidence_sha256": acceptance["sha256"],
    }


def _configs(
    base_template: dict[str, Any], seed: int, *, candidate_generation: int = 1
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = copy.deepcopy(base_template)
    base["name"] = f"q38-dev17-s{seed}-base-p1-v1"
    base["pass_k"] = 1
    # Four concurrent sessions bound each 17-task replica to five waves.  The
    # 8-hour session ceiling therefore stays below the 48-hour Job deadline.
    base["concurrency"] = 4
    base["sampling"]["seed"] = seed
    base["max_reviewed_infrastructure_retries"] = 0

    candidate = copy.deepcopy(base)
    candidate["name"] = f"q38-dev17-s{seed}-t3k32s1000-p1-v{candidate_generation}"
    candidate["models"] = {
        "teacher3k32-step1000": {
            "repository": "/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/hf-export-step1000-v1",
            "revision": CANDIDATE_REVISION,
            "session_model": f"qwen/{CANDIDATE_ID}",
        }
    }
    packet_name = f"qwen38-step1000-seed{seed}-provenance-v{candidate_generation}.json"
    packet = _candidate_packet(candidate["name"])
    model_artifact_v3.validate_packet(candidate["models"], packet)
    candidate["model_artifact_binding"] = _candidate_binding(packet, packet_name)
    route = copy.deepcopy(next(iter(base["routes"].values())))
    route["model"] = "teacher3k32-step1000"
    route["served_id"] = CANDIDATE_ID
    route["model_info"]["model_path"] = f"/scratch/models/{CANDIDATE_ID}"
    route["server_info"]["model_path"] = f"/scratch/models/{CANDIDATE_ID}"
    candidate["routes"] = {"teacher3k32": route}
    return base, candidate


def _protocol(base: dict[str, Any], seed: int) -> dict[str, Any]:
    value = {
        "schema": "cyber_fleet_matched_evaluation_protocol_v1",
        "protocol_id": f"q38-dev17-s{seed}-base-t3k32s1000-p1-v1",
        "comparison_arms": ["base", "candidate"],
        "model_revisions": {"base": BASE_REVISION, "candidate": CANDIDATE_REVISION},
        "task_selection_sha256": _file_sha(TASK_SET),
        "split_manifest_file_sha256": _file_sha(SPLIT),
        "split_manifest_sha256": _verified_json(SPLIT, "representative split")["sha256"],
        "harness": base["harness"],
        "sampling": base["sampling"],
        "images": base["images"],
        "pass_k": 1,
        "retry_limit": 0,
    }
    value["sha256"] = _canonical(value)
    return value


def _base_provenance(
    config_path: Path, roster: dict[str, Any], campaign_name: str
) -> dict[str, Any]:
    lock = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
    weights = ROOT / "configs/models/qwen38-27b-1d4bf0f2.weights.json"
    value = {
        "schema": "cyber_fleet_eval_base_model_artifact_packet_v1",
        "campaign_name": campaign_name,
        "authorization": "provenance_only_not_launch_authorization",
        "model": {
            "alias": "qwen3.8-27b-base",
            "repository": "Qwen/Qwen3.8-27B",
            "revision": BASE_REVISION,
            "session_model": "qwen/qwen3.8-27b",
            "served_id": "qwen3.8-27b",
        },
        "evaluation_config": {"path": config_path.name, "file_sha256": _file_sha(config_path)},
        "model_lock": {"path": str(lock.relative_to(ROOT)), "file_sha256": _file_sha(lock)},
        "weight_manifest": {
            "path": str(weights.relative_to(ROOT)),
            "file_sha256": _file_sha(weights),
        },
        "scientific_boundary": {
            "fresh_live_serving_registration_required": True,
            "fresh_live_parity_required": True,
            "packet_does_not_authorize_serving_or_evaluation": True,
        },
        "exact_task_binding_roster": {
            "path": str(BINDING_ROSTER.relative_to(ROOT)),
            "file_sha256": _file_sha(BINDING_ROSTER),
            "sha256": roster["sha256"],
            "bindings_sha256": roster["bindings_sha256"],
            "bindings": roster["bindings"],
        },
    }
    value["sha256"] = _canonical(value)
    return value


def prepare(*, output: Path, live_parity: Path, now: datetime | None = None) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("packet output already exists")
    if not output.parent.is_dir():
        raise ValueError("packet output parent does not exist")
    base_template, task_set, split, corpus, roster = _inputs()
    parity_base, parity_candidate = _configs(base_template, SEEDS[0])
    parity_arms = {
        "base": {
            "model_revision": BASE_REVISION,
            "served_id": "qwen3.8-27b",
            "expected_source_path": f"/models/qwen3.8-27b/{BASE_REVISION}",
        },
        "candidate": {
            "model_revision": CANDIDATE_REVISION,
            "served_id": CANDIDATE_ID,
            "expected_source_path": f"/models/{CANDIDATE_ID}",
        },
    }
    parity = shared._live_parity(  # noqa: SLF001
        live_parity,
        readiness={"arms": {"base": parity_arms["base"], "fresh75": parity_arms["candidate"]}},
        configs={"base": parity_base, "fresh75": parity_candidate},
        now=now or datetime.now(UTC),
    )
    temporary = Path(tempfile.mkdtemp(prefix=".seed46-p8-", dir=output.parent))
    try:
        receipt_arms = []
        protocols = []
        for seed in SEEDS:
            seed_dir = temporary / f"seed{seed}"
            seed_dir.mkdir(mode=0o700)
            base, candidate = _configs(base_template, seed)
            protocol = _protocol(base, seed)
            base_path = seed_dir / (f"qwen38-base-fleet-dev17-opencode-seed{seed}-pass1-v1.json")
            candidate_path = seed_dir / (
                f"qwen38-teacher3k32-step1000-fleet-dev17-opencode-seed{seed}-pass1-v1.json"
            )
            protocol_path = seed_dir / (
                f"qwen38-fleet-dev17-seed{seed}-base-step1000-pass1-protocol-v1.json"
            )
            base_provenance = seed_dir / f"qwen38-base-seed{seed}-provenance-v1.json"
            candidate_provenance = seed_dir / (f"qwen38-step1000-seed{seed}-provenance-v1.json")
            _write_json(base_path, base)
            _write_json(candidate_path, candidate)
            _write_json(protocol_path, protocol)
            _write_json(base_provenance, _base_provenance(base_path, roster, base["name"]))
            candidate_evidence = _candidate_packet(candidate["name"])
            _write_json(candidate_provenance, candidate_evidence)
            arms = {
                "base": {
                    **parity_arms["base"],
                    "job_name": f"chris-q38-dev17-s{seed}-base-p1-v1",
                    "config_map_name": f"chris-q38-dev17-s{seed}-base-code-v1",
                    "output_root": (f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-base-p1-v1"),
                    "database": f"q38_dev17_s{seed}_base_p1_v1",
                },
                "candidate": {
                    **parity_arms["candidate"],
                    "job_name": f"chris-q38-dev17-s{seed}-t3k32s1000-p1-v1",
                    "config_map_name": f"chris-q38-dev17-s{seed}-t3k32s1000-code-v1",
                    "output_root": (
                        f"/mnt/sfs/jobs/chris-q38-fleet-dev17-s{seed}-t3k32s1000-p1-v1"
                    ),
                    "database": f"q38_dev17_s{seed}_t3k32s1000_p1_v1",
                },
            }
            for arm_id, config, config_path, checkpoint in (
                ("base", base, base_path, base_provenance),
                ("candidate", candidate, candidate_path, candidate_provenance),
            ):
                row = shared._prepare_arm(  # noqa: SLF001
                    seed_dir / arm_id,
                    arm_id=arm_id,
                    arm=arms[arm_id],
                    config_path=config_path,
                    config=config,
                    task_set_path=TASK_SET,
                    split_path=SPLIT,
                    protocol_path=protocol_path,
                    protocol=protocol,
                    checkpoint_path=checkpoint,
                    proof_path=live_parity,
                    ledger_path=LEDGER,
                    source_files=V3_SOURCE_FILES if arm_id == "candidate" else None,
                )
                row["seed"] = seed
                receipt_arms.append(row)
            protocols.append(
                {
                    "seed": seed,
                    "protocol_id": protocol["protocol_id"],
                    "sha256": protocol["sha256"],
                    "file_sha256": _file_sha(protocol_path),
                }
            )
        receipt = {
            "schema": "cyber_qwen38_fleet_seed46_pass8_packet_preparation_v1",
            "protocol_study_id": PROTOCOL_STUDY_ID,
            "seeds": list(SEEDS),
            "protocols": protocols,
            "live_parity_file_sha256": _file_sha(live_parity),
            "live_parity_receipt_sha256": parity["receipt_sha256"],
            "selection": {
                "task_count": len(task_set["tasks"]),
                "rollouts_per_seed_arm": len(task_set["tasks"]),
                "rollouts_per_aggregate_arm": len(task_set["tasks"]) * len(SEEDS),
                "total_rollouts": len(task_set["tasks"]) * len(SEEDS) * 2,
                "concurrency_per_replica_job": 4,
                "selection_sha256": task_set["selection_sha256"],
                "split_sha256": split["sha256"],
            },
            "leakage_gate": {
                "exact_dev_task_key_overlap": 0,
                "held_out_task_families_excluded_across_all_versions": corpus["catalog_provenance"][
                    "held_out_task_families_excluded_across_all_versions"
                ],
                "corpus_dev_windows": corpus["dev_windows"],
                "live_task_bindings_equal_frozen_roster_required_before_launch": True,
                "binding_roster_file_sha256": _file_sha(BINDING_ROSTER),
                "binding_roster_sha256": roster["sha256"],
                "bindings_sha256": roster["bindings_sha256"],
            },
            "arms": receipt_arms,
            "external_mutations": 0,
            "launch_performed": False,
            "credentials_read": False,
            "final8_opened": False,
        }
        receipt["sha256"] = _canonical(receipt)
        _write_json(temporary / "PREPARATION_RECEIPT.json", receipt)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(output=args.output, live_parity=args.live_parity), indent=2))


if __name__ == "__main__":
    main()
