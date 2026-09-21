#!/usr/bin/env python3
# ruff: noqa: E501
"""Render the source-only five-arm Qwen3.8 seed-45 Fleet configs.

This script has no cluster, Fleet, database, or serving client.  It derives the
new scientific configs from the already-reviewed seed-43/44 configs and seals
the exact local checkpoint/export/reload evidence for the three additional SFS
artifacts.  Live parity and create-once launch packets remain a later JIT gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import model_artifact, model_artifact_v2

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"
EVIDENCE = ROOT / "docs/evidence"
TASK_SET = EVAL / "qwen38-fresh75-fleet-dev17-task-set-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
PROTOCOL = EVAL / "qwen38-fleet-dev17-seed45-five-arm-matched-protocol-v1.json"
READINESS = EVAL / "qwen38-fleet-dev17-seed45-five-arm-readiness-v1.json"

ARMS = {
    "teacher": {
        "template": "qwen38-teacher-v5-fleet-dev17-opencode-seed43-pass1-v1.json",
        "config": "qwen38-teacher-v5-fleet-dev17-opencode-seed45-pass1-v1.json",
        "packet": "qwen38-teacher-v5-fleet-dev17-opencode-seed45-pass1-v1.artifact.json",
        "campaign": "q38-dev17-s45-teacher186-p1-v1",
        "alias": "teacher-v5-step186",
        "evidence": "qwen38-teacher186-eval-artifact-accepted-20260921.json",
        "checkpoint": {
            "path": "/mnt/sfs/jobs/chris-cyber-q38-teacher-dense-v5/checkpoint-seals-v1/checkpoint-step-186.json",
            "file_sha256": "sha256:1b671c6ec4b39d1141e0d28c530243624f6a97a7922f9f222ffcbe20b8434d3e",
            "receipt_sha256": "sha256:1d61918874d59421455577deca70c9784ce09f8f327a5031df9d39dfb5f3d7f2",
            "checkpoint_path": "/mnt/sfs/jobs/chris-cyber-q38-teacher-dense-v5/checkpoints/global_step_186",
            "optimizer_step": 186,
            "total_bytes": 324627486923,
        },
        "export": {
            "path": "/mnt/sfs/jobs/chris-cyber-q38-teacher-dense-v5/hf-export-v1/EXPORT.json",
            "file_sha256": "sha256:96b3181e05c501343c97cfe803aada6884e406c7a112e0dc6c77461f090b8bab",
            "receipt_sha256": "sha256:e6d59f5a58ce54b913d775cd71b81c64bd54637b53795df82baec0f71efefaf9",
            "output_root": "/mnt/sfs/jobs/chris-cyber-q38-teacher-dense-v5/hf-export-v1",
            "source_plan_sha256": "b4543b9173e799bbe155ab416923db57c9796b8734186b834e5109c5be6330fc",
            "source_checkpoint_receipt_sha256": "1d61918874d59421455577deca70c9784ce09f8f327a5031df9d39dfb5f3d7f2",
            "source_manifest_file_sha256": "1b671c6ec4b39d1141e0d28c530243624f6a97a7922f9f222ffcbe20b8434d3e",
            "optimizer_step": 186,
            "files_sha256": "sha256:91d98369bc415a4e471e1e7e46172f979341b44324de264a2cab16ee5172fa77",
        },
        "gpu": {
            "path": "/mnt/sfs/jobs/chris-q38-teacher-sft-hfcheck-v1/GPU_CHECK.json",
            "file_sha256": "sha256:7fbfcfe16875722012cc2654d979545e22aa1fdca093987495dfff65173dfd8e",
            "receipt_sha256": "sha256:9bd229e3c85a092ccc5fca8c3183f57554f1c8a72b2724221f5c7507718a3392",
            "rayjob_uid": "e5288ecc-ca64-41fd-a7a0-f569b4cc03ff",
            "pod_uid": "a367c717-23e2-4f6b-ac11-0ae6548d5bb2",
        },
        "payload": {
            "served_revision": "sha256:e6d59f5a58ce54b913d775cd71b81c64bd54637b53795df82baec0f71efefaf9",
            "revision_basis": "export_receipt_sha256",
            "stage_source_path": "/models/chris-q38-teacher-sft-v5-step186-v1",
        },
    },
    "self44": {
        "template": "qwen38-self-sft-step44-fleet-dev17-opencode-seed43-pass1-v1.json",
        "config": "qwen38-self-sft-step44-fleet-dev17-opencode-seed45-pass1-v1.json",
        "packet": "qwen38-self-sft-step44-fleet-dev17-opencode-seed45-pass1-v1.artifact.json",
        "campaign": "q38-dev17-s45-self44-p1-v1",
        "alias": "self-sft-step44",
        "evidence": "qwen38-self44-eval-artifact-accepted-20260921.json",
        "checkpoint": {
            "path": "/mnt/sfs/jobs/chris-cyber-q38-dev20-self-lr1-v2/checkpoint-seals-v1/checkpoint-step-44.json",
            "file_sha256": "sha256:0094cfeb62f120b55d9e2f7fdfd4217c4c0aab66f369e3c52c89fc5e7ccfff9b",
            "receipt_sha256": "sha256:bf8237a5837dd75bf1163b5a5e134e1349417599c1a28f022b91281c2c7b7885",
            "checkpoint_path": "/mnt/sfs/jobs/chris-cyber-q38-dev20-self-lr1-v2/checkpoints/global_step_44",
            "optimizer_step": 44,
            "total_bytes": 324627486795,
        },
        "export": {
            "path": "/mnt/sfs/jobs/chris-cyber-q38-dev20-self-lr1-v2/hf-export-v1/EXPORT.json",
            "file_sha256": "sha256:7d3d56da513488dab485114b49b4fe34c5efe4c2526577c0063a2cbc87413e50",
            "receipt_sha256": "sha256:12f06f12a35f9c0509610f8bed8cb685fc23864490ec5de7739ee1e6aa48f606",
            "output_root": "/mnt/sfs/jobs/chris-cyber-q38-dev20-self-lr1-v2/hf-export-v1",
            "source_plan_sha256": "4d97bb2e67f39aedd3bd49f85a49fa9f2c6c147c01628a612b24c3a5132678ed",
            "source_checkpoint_receipt_sha256": "bf8237a5837dd75bf1163b5a5e134e1349417599c1a28f022b91281c2c7b7885",
            "source_manifest_file_sha256": "0094cfeb62f120b55d9e2f7fdfd4217c4c0aab66f369e3c52c89fc5e7ccfff9b",
            "optimizer_step": 44,
            "files_sha256": "sha256:a0e55bebe9d78ac76c012446ae0147a421cd333e2a6bf2bb6f4fb3450fa776fe",
        },
        "gpu": {
            "path": "/mnt/sfs/jobs/chris-q38-self-sft-hfcheck-v1/GPU_CHECK.json",
            "file_sha256": "sha256:e866e6381a4a79bf1cb759016afde3aa5756bc707e047572e7edc59b18a4653c",
            "receipt_sha256": "sha256:9fed9b8368d58213af9999abe18d73dd5fb83cd273ee7923b57321f6c4d31e12",
            "rayjob_uid": "ca872fa6-6a2b-46aa-aa48-105e19115c19",
            "pod_uid": "b3d13079-86bb-4dfd-b51b-184cfcc4f0e3",
        },
        "payload": {
            "served_revision": "sha256:a0e55bebe9d78ac76c012446ae0147a421cd333e2a6bf2bb6f4fb3450fa776fe",
            "revision_basis": "export_files_canonical_sha256",
            "stage_source_path": "/models/chris-q38-self-sft-step44-v1",
        },
    },
    "lr30": {
        "template": "qwen38-lr30-step76-fleet-dev17-opencode-seed43-pass1-v1.json",
        "config": "qwen38-lr30-step76-fleet-dev17-opencode-seed45-pass1-v1.json",
        "packet": "qwen38-lr30-step76-fleet-dev17-opencode-seed45-pass1-v1.artifact.json",
        "campaign": "q38-dev17-s45-lr30s76-p1-v1",
        "alias": "lr30-step76",
        "evidence": "qwen38-lr30s76-eval-artifact-accepted-20260921.json",
        "checkpoint": {
            "path": "/mnt/sfs/jobs/chris-q38-lr30-resume76-prod-v1/checkpoint-seals-v1/step-76.json",
            "file_sha256": "sha256:cc4e2997e120a80bb1e209dc027f18e12ce1ebe12639594d28e23b08c459cf1a",
            "receipt_sha256": "sha256:1252a1a9d5556fa118f03d1b2fe5687e0e9505ccf622e2f86743a26b77dffa64",
            "checkpoint_path": "/mnt/sfs/jobs/chris-q38-lr30-resume76-prod-v1/checkpoints/global_step_76",
            "optimizer_step": 76,
            "total_bytes": 324627487179,
        },
        "export": {
            "path": "/mnt/sfs/jobs/chris-q38-lr30-resume76-prod-v1/hf-export-step76-v1/EXPORT.json",
            "file_sha256": "sha256:39cedfb644aa955b450e42eb98494276c1bdce85a9760b2b038b67823ff7b9e5",
            "receipt_sha256": "sha256:983bf6daa2f8b7680560c834cbfb76fb8fe7fba5a273f116b9b3e80ef9f97538",
            "output_root": "/mnt/sfs/jobs/chris-q38-lr30-resume76-prod-v1/hf-export-step76-v1",
            "source_plan_sha256": "7efcbbff380107220fb6a1d6680b0bbe8d0a303eff748e948a137dbd6d8524d3",
            "source_checkpoint_receipt_sha256": "1252a1a9d5556fa118f03d1b2fe5687e0e9505ccf622e2f86743a26b77dffa64",
            "source_manifest_file_sha256": "cc4e2997e120a80bb1e209dc027f18e12ce1ebe12639594d28e23b08c459cf1a",
            "optimizer_step": 76,
            "files_sha256": "sha256:3bef11697759b11db150b705e1882fb2d31a3efbdadbebd948aa917dcacc03a8",
        },
        "gpu": {
            "path": "/mnt/sfs/jobs/chris-q38-lr30-step76-hfcheck-v1/GPU_CHECK.json",
            "file_sha256": "sha256:8c6b4bc568fff6728dca2f7d4f7200d5c45b07a9b4b2c1ca9525c4cd16d97b2b",
            "receipt_sha256": "sha256:8cb7cbab5df338b2066825aad93ba52aed10a339464e2524fa1bb0ac2fec192a",
            "rayjob_uid": "e5da1477-9678-4f59-960c-97067ea7e9b9",
            "pod_uid": "6e3007c1-b0f0-4572-9cae-85290538bffa",
        },
        "payload": {
            "served_revision": "sha256:3bef11697759b11db150b705e1882fb2d31a3efbdadbebd948aa917dcacc03a8",
            "revision_basis": "export_files_canonical_sha256",
            "stage_source_path": "/models/chris-q38-available-a-lr30-step76-v1",
        },
    },
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def encoded(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def receipt_binding(spec: dict[str, Any]) -> dict[str, Any]:
    checkpoint, export, gpu = spec["checkpoint"], spec["export"], spec["gpu"]
    evidence_path = EVIDENCE / spec["evidence"]
    evidence = read(evidence_path)
    if evidence["artifact_alias"] != spec["alias"]:
        raise ValueError("acceptance evidence alias differs")
    if (
        evidence["sha256"]
        != "sha256:"
        + hashlib.sha256(
            canonical({key: item for key, item in evidence.items() if key != "sha256"})
        ).hexdigest()
    ):
        raise ValueError("acceptance evidence self digest differs")
    export_required = {
        "schema": "cyber_native_checkpoint_hf_export_v1",
        "source_checkpoint_receipt_sha256": export["source_checkpoint_receipt_sha256"],
        "source_manifest_file_sha256": export["source_manifest_file_sha256"],
        "source_plan_sha256": export["source_plan_sha256"],
        "model_repo": "Qwen/Qwen3.8-27B",
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "output_root": export["output_root"],
        "optimizer_step": export["optimizer_step"],
        "optimizer_steps_executed": 0,
        "gpu_reload_verified": False,
        "trained_tensors": 1184,
        "tensor_values": 27781427952,
        "tensor_bytes": 55562855904,
        "dtype": "BF16",
        "source_inventory_sizes_mtimes_unchanged": True,
        "all_output_tensors_reopened_equal": True,
    }
    gpu_required = {
        "schema": "cyber_hf_export_check_v1",
        "status": "passed",
        "export_sha256": export["file_sha256"].removeprefix("sha256:"),
        "export_receipt_sha256": export["receipt_sha256"].removeprefix("sha256:"),
        "optimizer_steps_executed": 0,
        "gpus": 1,
        "gpu_reload_verified": True,
        "serving_qualified": False,
        "synthetic_only": True,
        "source_unchanged": True,
        "attention_implementation": "eager",
        "finite_logits": True,
        "generated_tokens": 2,
    }
    payload = {
        **spec["payload"],
        "export_files_sha256": export["files_sha256"],
        "file_count": 29,
        "trained_tensors": 1184,
        "restored_mtp_tensors": sorted(model_artifact.EXACT_MTP_TENSORS),
        "tensor_count": 1199,
        "tensor_bytes": 55562855904,
    }
    return {
        "schema": model_artifact_v2.BINDING_SCHEMA,
        "checkpoint_manifest": {
            "path": checkpoint["path"],
            "file_sha256": checkpoint["file_sha256"],
            "receipt_sha256": checkpoint["receipt_sha256"],
            "required_fields": {
                "schema": "cyber_skyrl_checkpoint_manifest_v1",
                "checkpoint_path": checkpoint["checkpoint_path"],
                "optimizer_step": checkpoint["optimizer_step"],
                "world_size": 8,
                "total_bytes": checkpoint["total_bytes"],
                "gpu_reload_verified": False,
            },
        },
        "export_receipt": {
            "path": export["path"],
            "file_sha256": export["file_sha256"],
            "receipt_sha256": export["receipt_sha256"],
            "required_fields": export_required,
        },
        "gpu_reload_receipt": {
            "path": gpu["path"],
            "file_sha256": gpu["file_sha256"],
            "receipt_sha256": gpu["receipt_sha256"],
            "required_fields": gpu_required,
        },
        "payload": payload,
        "qualification": {
            "acceptance_evidence_path": str(evidence_path.relative_to(ROOT)),
            "acceptance_evidence_file_sha256": file_sha256(evidence_path),
            "acceptance_evidence_sha256": evidence["sha256"],
            "gpu_reload_rayjob_uid": gpu["rayjob_uid"],
            "gpu_reload_pod_uid": gpu["pod_uid"],
            "gpu_allocation_released": True,
        },
    }


def packet(spec: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema": model_artifact_v2.PACKET_SCHEMA,
        "campaign_name": spec["campaign"],
        "models": {spec["alias"]: receipt_binding(spec)},
        "authorization": "provenance_only_not_launch_authorization",
    }
    return {**value, "sha256": hashlib.sha256(canonical(value)).hexdigest()}


def render() -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {}
    base = read(EVAL / "qwen38-base-fleet-dev17-opencode-seed44-pass1-v1.json")
    base["name"] = "q38-dev17-s45-base-p1-v1"
    base["sampling"]["seed"] = 45
    base_config_path = EVAL / "qwen38-base-fleet-dev17-opencode-seed45-pass1-v1.json"
    base_config_bytes = encoded(base)
    outputs[base_config_path] = base_config_bytes
    base_artifact = read(EVAL / "qwen38-base-fleet-dev17-opencode-seed44-pass1-v1.artifact.json")
    base_artifact["campaign_name"] = base["name"]
    base_artifact["evaluation_config"] = {
        "path": str(base_config_path.relative_to(ROOT)),
        "file_sha256": "sha256:" + hashlib.sha256(base_config_bytes).hexdigest(),
    }
    base_artifact.pop("sha256")
    base_artifact["sha256"] = "sha256:" + hashlib.sha256(canonical(base_artifact)).hexdigest()
    base_artifact_path = EVAL / "qwen38-base-fleet-dev17-opencode-seed45-pass1-v1.artifact.json"
    outputs[base_artifact_path] = encoded(base_artifact)

    fresh_config = read(EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed44-pass1-v2.json")
    fresh_packet = read(EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed44-pass1-v2.artifact.json")
    fresh_config["name"] = fresh_packet["campaign_name"] = "q38-dev17-s45-fresh75-p1-v1"
    fresh_config["sampling"]["seed"] = 45
    fresh_packet.pop("sha256")
    fresh_packet["sha256"] = hashlib.sha256(canonical(fresh_packet)).hexdigest()
    fresh_packet_path = EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed45-pass1-v1.artifact.json"
    fresh_packet_bytes = encoded(fresh_packet)
    outputs[fresh_packet_path] = fresh_packet_bytes
    binding = fresh_config["model_artifact_binding"]
    binding["packet_path"] = str(fresh_packet_path.relative_to(ROOT))
    binding["packet_file_sha256"] = "sha256:" + hashlib.sha256(fresh_packet_bytes).hexdigest()
    binding["packet_sha256"] = "sha256:" + fresh_packet["sha256"]
    outputs[EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed45-pass1-v1.json"] = encoded(
        fresh_config
    )

    validator_sha256 = file_sha256(Path(model_artifact_v2.__file__))
    for spec in ARMS.values():
        artifact = packet(spec)
        packet_path = EVAL / spec["packet"]
        packet_bytes = encoded(artifact)
        outputs[packet_path] = packet_bytes
        config = copy.deepcopy(read(EVAL / spec["template"]))
        config["name"] = spec["campaign"]
        config["sampling"]["seed"] = 45
        evidence_path = EVIDENCE / spec["evidence"]
        evidence = read(evidence_path)
        config["model_artifact_binding"] = {
            "validator_file_sha256": validator_sha256,
            "packet_path": str(packet_path.relative_to(ROOT)),
            "packet_file_sha256": "sha256:" + hashlib.sha256(packet_bytes).hexdigest(),
            "packet_sha256": "sha256:" + artifact["sha256"],
            "acceptance_evidence_path": str(evidence_path.relative_to(ROOT)),
            "acceptance_evidence_file_sha256": file_sha256(evidence_path),
            "acceptance_evidence_sha256": evidence["sha256"],
        }
        outputs[EVAL / spec["config"]] = encoded(config)

    comparison_arms = ["base", "fresh75", "teacher186", "self44", "lr30s76"]
    protocol = {
        "schema": "cyber_fleet_matched_evaluation_protocol_v1",
        "protocol_id": "q38-dev17-seed45-five-arm-p1-v1",
        "comparison_arms": comparison_arms,
        "model_revisions": {
            "base": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "fresh75": "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029",
            "teacher186": "sha256:e6d59f5a58ce54b913d775cd71b81c64bd54637b53795df82baec0f71efefaf9",
            "self44": "sha256:a0e55bebe9d78ac76c012446ae0147a421cd333e2a6bf2bb6f4fb3450fa776fe",
            "lr30s76": "sha256:3bef11697759b11db150b705e1882fb2d31a3efbdadbebd948aa917dcacc03a8",
        },
        "task_selection_sha256": file_sha256(TASK_SET),
        "split_manifest_file_sha256": file_sha256(SPLIT),
        "split_manifest_sha256": "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c",
        "harness": base["harness"],
        "sampling": {"temperature": 0.6, "top_p": 0.95, "seed": 45},
        "images": base["images"],
        "pass_k": 1,
        "retry_limit": 1,
    }
    protocol["sha256"] = "sha256:" + hashlib.sha256(canonical(protocol)).hexdigest()
    protocol_bytes = encoded(protocol)
    outputs[PROTOCOL] = protocol_bytes

    sources = {
        "base": (base_config_path, base_artifact_path),
        "fresh75": (
            EVAL / "qwen38-fresh75-fleet-dev17-opencode-seed45-pass1-v1.json",
            fresh_packet_path,
        ),
        "teacher186": (EVAL / ARMS["teacher"]["config"], EVAL / ARMS["teacher"]["packet"]),
        "self44": (EVAL / ARMS["self44"]["config"], EVAL / ARMS["self44"]["packet"]),
        "lr30s76": (EVAL / ARMS["lr30"]["config"], EVAL / ARMS["lr30"]["packet"]),
    }
    runtime_identities = {
        "base": {
            "job_name": "chris-q38-dev17-s45-base-p1-v1",
            "config_map_name": "chris-q38-dev17-s45-base-code-v1",
            "output_root": "/mnt/sfs/jobs/chris-q38-fleet-dev17-s45-base-p1-v1",
            "database": "q38_dev17_s45_base_p1_v1",
            "served_id": "qwen3.8-27b",
            "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "expected_source_path": "/models/qwen3.8-27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
        "fresh75": {
            "job_name": "chris-q38-dev17-s45-fresh75-p1-v1",
            "config_map_name": "chris-q38-dev17-s45-fresh75-code-v1",
            "output_root": "/mnt/sfs/jobs/chris-q38-fleet-dev17-s45-fresh75-p1-v1",
            "database": "q38_dev17_s45_fresh75_p1_v1",
            "served_id": "chris-q38-fresh75-step230-web-v2",
            "model_revision": "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029",
            "expected_source_path": "/models/chris-q38-fresh75-step230-v1",
        },
        "teacher186": {
            "job_name": "chris-q38-dev17-s45-teacher186-p1-v1",
            "config_map_name": "chris-q38-dev17-s45-teacher186-code-v1",
            "output_root": "/mnt/sfs/jobs/chris-q38-fleet-dev17-s45-teacher186-p1-v1",
            "database": "q38_dev17_s45_teacher186_p1_v1",
            "served_id": "chris-q38-teacher-v5-step186-web-v1",
            "model_revision": "sha256:e6d59f5a58ce54b913d775cd71b81c64bd54637b53795df82baec0f71efefaf9",
            "expected_source_path": "/models/chris-q38-teacher-sft-v5-step186-v1",
        },
        "self44": {
            "job_name": "chris-q38-dev17-s45-self44-p1-v1",
            "config_map_name": "chris-q38-dev17-s45-self44-code-v1",
            "output_root": "/mnt/sfs/jobs/chris-q38-fleet-dev17-s45-self44-p1-v1",
            "database": "q38_dev17_s45_self44_p1_v1",
            "served_id": "chris-q38-self-sft-step44-v1",
            "model_revision": "sha256:a0e55bebe9d78ac76c012446ae0147a421cd333e2a6bf2bb6f4fb3450fa776fe",
            "expected_source_path": "/models/chris-q38-self-sft-step44-v1",
        },
        "lr30s76": {
            "job_name": "chris-q38-dev17-s45-lr30s76-p1-v1",
            "config_map_name": "chris-q38-dev17-s45-lr30s76-code-v1",
            "output_root": "/mnt/sfs/jobs/chris-q38-fleet-dev17-s45-lr30s76-p1-v1",
            "database": "q38_dev17_s45_lr30s76_p1_v1",
            "served_id": "chris-q38-lr30-step76-web-v1",
            "model_revision": "sha256:3bef11697759b11db150b705e1882fb2d31a3efbdadbebd948aa917dcacc03a8",
            "expected_source_path": "/models/chris-q38-available-a-lr30-step76-v1",
        },
    }
    readiness_arms = {}
    for arm in comparison_arms:
        config_path, artifact_path = sources[arm]
        config_bytes = outputs[config_path]
        artifact_bytes = outputs[artifact_path]
        readiness_arms[arm] = {
            "config_path": str(config_path.relative_to(ROOT)),
            "config_file_sha256": "sha256:" + hashlib.sha256(config_bytes).hexdigest(),
            "checkpoint_provenance_path": str(artifact_path.relative_to(ROOT)),
            "checkpoint_provenance_file_sha256": (
                "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
            ),
            **runtime_identities[arm],
        }
    readiness = {
        "schema": "cyber_qwen38_fleet_dev17_seed45_five_arm_readiness_v1",
        "status": "source_prepared_not_launchable",
        "launchable": False,
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "file_sha256": "sha256:" + hashlib.sha256(protocol_bytes).hexdigest(),
            "sha256": protocol["sha256"],
        },
        "selection": {
            "task_set_path": str(TASK_SET.relative_to(ROOT)),
            "task_set_file_sha256": file_sha256(TASK_SET),
            "split_path": str(SPLIT.relative_to(ROOT)),
            "split_file_sha256": file_sha256(SPLIT),
            "task_count": 17,
            "split_role": "dev_validation",
            "final8_sealed": True,
        },
        "arms": readiness_arms,
        "live_gate": {
            "required_candidate_parity_receipts": [
                "fresh75",
                "teacher186",
                "self44",
                "lr30s76",
            ],
            "all_candidate_routes_current_phase": "must_be_observed_jit",
            "maximum_receipt_age_seconds": 3600,
            "route_mutations_performed_by_this_source_preparation": 0,
        },
        "sequence_gate": {
            "seed44_base_and_fresh75_terminal_reconciliation_required": True,
            "seed44_accepted_cells_must_not_be_replayed": True,
            "candidate_routes_are_jit_only": True,
        },
        "create_once_gate": {
            "job_config_map_workload_pod_output_database_absence_required": True,
            "server_preview_required": True,
            "root_failure_alert_annotation": "off",
            "automatic_retry": False,
        },
        "live_evidence_binding": {
            "kubernetes_context": "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6",
            "namespace": "fleet-train-jobs",
            "assert_before_every_live_read": True,
            "object_absence_valid_only_after_exact_binding_assertion": True,
        },
        "scientific_boundary": {
            "descriptive_train50_is_a_separate_nonheldout_campaign": True,
            "final8_must_not_be_opened_by_this_protocol": True,
            "scores_or_task_content_required_for_preparation": False,
        },
        "capacity_gate": {
            "owned_node_cap": 8,
            "owned_gpu_cap": 64,
            "candidate_route_nodes": 4,
            "candidate_route_gpus": 32,
            "fresh_read_only_owned_capacity_census_required_before_resume": True,
        },
        "eligibility": {
            "seed44_terminal_reconciled": False,
            "fresh_live_parity_complete": False,
            "duplicate_absence_complete": False,
            "root_alert_off_server_preview_complete": False,
            "eligible_now": False,
        },
        "operation": {
            "jobs_created": 0,
            "config_maps_created": 0,
            "routes_resumed": 0,
            "databases_created": 0,
            "outputs_created": 0,
        },
    }
    readiness["sha256"] = "sha256:" + hashlib.sha256(canonical(readiness)).hexdigest()
    outputs[READINESS] = encoded(readiness)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = render()
    changed = [
        path
        for path, content in outputs.items()
        if not path.exists() or path.read_bytes() != content
    ]
    if args.check:
        if changed:
            raise SystemExit("seed45 generated files differ: " + ", ".join(map(str, changed)))
        print(json.dumps({"status": "passed", "files": len(outputs)}, sort_keys=True))
        return 0
    for path, content in outputs.items():
        path.write_bytes(content)
    print(json.dumps({"status": "written", "files": len(outputs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
