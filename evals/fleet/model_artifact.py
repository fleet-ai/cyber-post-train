"""Fail-closed local checkpoint provenance for Fleet evaluations.

Model routes can prove that a served revision is live, but a revision alone does
not prove which local export produced those bytes.  Before a local SFS-backed
model can be evaluated, reopen the three small producer receipts that bind the
native checkpoint, complete BF16 export, and finite zero-update GPU reload.
No weights are read here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "cyber_fleet_eval_model_artifact_v1"
PACKET_SCHEMA = "cyber_fleet_eval_model_artifact_packet_v1"
RECEIPT_LIMIT = 4 * 1024 * 1024
LOCAL_ROOT = "/mnt/sfs/jobs/"
RETIRED_CAMPAIGNS = {"q38-dev17-s43-fresh75-p1-v2"}
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
UID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")

EXACT_MTP_TENSORS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one prefixed SHA-256")
    return value


def _safe_path(value: Any, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value.startswith(LOCAL_ROOT)
        or ".." in PurePosixPath(value).parts
        or not value.endswith(".json")
    ):
        raise ValueError(f"{label} must be one safe absolute SFS receipt path")
    return Path(value)


def _safe_artifact_root(value: Any, label: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value.startswith(LOCAL_ROOT)
        or ".." in PurePosixPath(value).parts
    ):
        raise ValueError(f"{label} must be one safe absolute SFS artifact path")
    return PurePosixPath(value)


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _stable_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_stable_bytes(path: Path, label: str) -> tuple[bytes, str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        raw = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            raw.extend(chunk)
            if len(raw) > RECEIPT_LIMIT:
                raise ValueError(f"{label} exceeds its size bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(before.st_mode) or _stable_identity(before) != _stable_identity(after):
        raise ValueError(f"{label} changed while it was read")
    return bytes(raw), _digest(raw)


def _read_stable_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw, file_sha256 = _read_stable_bytes(path, label)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value, file_sha256


def _read_receipt(binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = _safe_path(binding.get("path"), f"{label} path")
    expected_file = _sha(binding.get("file_sha256"), f"{label} file digest")
    expected_receipt = _sha(binding.get("receipt_sha256"), f"{label} embedded digest")
    value, actual_file = _read_stable_json(path, label)
    if actual_file != expected_file:
        raise ValueError(f"{label} bytes differ from the accepted binding")
    actual = value.get("receipt_sha256")
    if (
        not isinstance(actual, str)
        or re.fullmatch(r"[0-9a-f]{64}", actual) is None
        or _digest(
            _canonical({key: item for key, item in value.items() if key != "receipt_sha256"})
        )
        != "sha256:" + actual
        or expected_receipt != "sha256:" + actual
    ):
        raise ValueError(f"{label} embedded digest differs from the accepted binding")
    required = _mapping(binding.get("required_fields"), f"{label} required fields")
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"{label} field {field} differs from the accepted binding")
    return value


def validate_binding(model: Mapping[str, Any], binding: Any) -> Mapping[str, Any]:
    """Validate one config binding without touching its SFS receipts."""

    repository = model.get("repository")
    artifact = _mapping(binding, "model artifact binding")
    if (
        set(artifact)
        != {
            "schema",
            "checkpoint_manifest",
            "export_receipt",
            "gpu_reload_receipt",
            "payload",
            "qualification",
        }
        or artifact.get("schema") != SCHEMA
    ):
        raise ValueError("model artifact binding has an unsupported shape or schema")
    checkpoint = _mapping(artifact["checkpoint_manifest"], "checkpoint manifest binding")
    export = _mapping(artifact["export_receipt"], "export receipt binding")
    gpu = _mapping(artifact["gpu_reload_receipt"], "GPU reload receipt binding")
    for label, receipt in (
        ("checkpoint manifest", checkpoint),
        ("export receipt", export),
        ("GPU reload receipt", gpu),
    ):
        if set(receipt) != {"path", "file_sha256", "receipt_sha256", "required_fields"}:
            raise ValueError(f"{label} binding has unknown or missing fields")
        _safe_path(receipt["path"], f"{label} path")
        _sha(receipt["file_sha256"], f"{label} file digest")
        _sha(receipt["receipt_sha256"], f"{label} embedded digest")
        _mapping(receipt["required_fields"], f"{label} required fields")
    checkpoint_required = checkpoint["required_fields"]
    export_required = export["required_fields"]
    gpu_required = gpu["required_fields"]
    if set(checkpoint_required) != {
        "schema",
        "checkpoint_path",
        "optimizer_step",
        "world_size",
        "total_bytes",
        "gpu_reload_verified",
    }:
        raise ValueError("checkpoint manifest binding is not an exact native checkpoint gate")
    checkpoint_root = _safe_artifact_root(
        checkpoint_required.get("checkpoint_path"), "checkpoint path"
    )
    if (
        checkpoint_required.get("schema") != "cyber_skyrl_checkpoint_manifest_v1"
        or checkpoint_required.get("gpu_reload_verified") is not False
    ):
        raise ValueError("checkpoint manifest binding is not a sealed native checkpoint")
    _positive_int(checkpoint_required.get("optimizer_step"), "checkpoint optimizer step")
    _positive_int(checkpoint_required.get("world_size"), "checkpoint world size")
    _positive_int(checkpoint_required.get("total_bytes"), "checkpoint total bytes")
    if set(export_required) != {
        "schema",
        "source_checkpoint_receipt_sha256",
        "source_manifest_file_sha256",
        "source_plan_sha256",
        "model_repo",
        "model_revision",
        "output_root",
        "optimizer_step",
        "optimizer_steps_executed",
        "gpu_reload_verified",
        "trained_tensors",
        "tensor_values",
        "tensor_bytes",
        "dtype",
        "source_inventory_sizes_mtimes_unchanged",
        "all_output_tensors_reopened_equal",
    }:
        raise ValueError("export receipt binding is not an exact complete BF16 export gate")
    export_root = _safe_artifact_root(export_required.get("output_root"), "export root")
    if (
        export_required.get("schema") != "cyber_native_checkpoint_hf_export_v1"
        or export_required.get("source_checkpoint_receipt_sha256")
        != checkpoint["receipt_sha256"].removeprefix("sha256:")
        or export_required.get("source_manifest_file_sha256")
        != checkpoint["file_sha256"].removeprefix("sha256:")
        or export_required.get("optimizer_step") != checkpoint_required["optimizer_step"]
        or export_required.get("optimizer_steps_executed") != 0
        or export_required.get("gpu_reload_verified") is not False
        or export_required.get("dtype") != "BF16"
        or export_required.get("source_inventory_sizes_mtimes_unchanged") is not True
        or export_required.get("all_output_tensors_reopened_equal") is not True
        or export_root == checkpoint_root
        or export_root.is_relative_to(checkpoint_root)
    ):
        raise ValueError("export receipt binding is not a zero-update checkpoint-bound export")
    _sha("sha256:" + str(export_required.get("source_plan_sha256")), "source plan")
    if (
        not isinstance(export_required.get("model_repo"), str)
        or not export_required["model_repo"]
        or not isinstance(export_required.get("model_revision"), str)
        or not export_required["model_revision"]
    ):
        raise ValueError("export base model binding is incomplete")
    _positive_int(export_required.get("trained_tensors"), "export trained tensors")
    _positive_int(export_required.get("tensor_values"), "export tensor values")
    _positive_int(export_required.get("tensor_bytes"), "export tensor bytes")
    if set(gpu_required) != {
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
    }:
        raise ValueError("GPU reload receipt binding is not an exact finite reload gate")
    if gpu_required != {
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
    }:
        raise ValueError("GPU reload receipt binding is not finite, zero-update, and one-GPU")
    if export["path"] != str(repository).rstrip("/") + "/EXPORT.json":
        raise ValueError("export receipt path does not belong to the configured repository")
    payload = _mapping(artifact["payload"], "export payload binding")
    if set(payload) != {
        "manifest_sha256",
        "file_count",
        "trained_tensors",
        "restored_mtp_tensors",
        "tensor_count",
        "tensor_bytes",
    }:
        raise ValueError("export payload binding has unknown or missing fields")
    if (
        _sha(payload.get("manifest_sha256"), "payload manifest") != model.get("revision")
        or type(payload.get("file_count")) is not int
        or payload["file_count"] <= 0
        or type(payload.get("trained_tensors")) is not int
        or payload["trained_tensors"] <= 0
        or payload.get("restored_mtp_tensors") != sorted(EXACT_MTP_TENSORS)
        or payload.get("tensor_count") != payload["trained_tensors"] + len(EXACT_MTP_TENSORS)
        or type(payload.get("tensor_bytes")) is not int
        or payload["tensor_bytes"] <= 0
    ):
        raise ValueError("export payload binding is incomplete or inconsistent")
    if (
        export_required["output_root"] != repository
        or export_required["trained_tensors"] != payload["trained_tensors"]
        or export_required["tensor_values"] * 2 != payload["tensor_bytes"]
        or export_required["tensor_bytes"] != payload["tensor_bytes"]
    ):
        raise ValueError("export receipt and payload binding disagree")
    qualification = _mapping(artifact["qualification"], "qualification binding")
    if set(qualification) != {
        "acceptance_evidence_path",
        "acceptance_evidence_file_sha256",
        "acceptance_evidence_sha256",
        "gpu_reload_rayjob_uid",
        "gpu_reload_pod_uid",
        "gpu_allocation_released",
    }:
        raise ValueError("qualification binding has unknown or missing fields")
    if (
        not isinstance(qualification["acceptance_evidence_path"], str)
        or not qualification["acceptance_evidence_path"].startswith("docs/evidence/")
        or qualification["acceptance_evidence_path"].startswith("/")
        or ".." in PurePosixPath(qualification["acceptance_evidence_path"]).parts
        or UID.fullmatch(str(qualification["gpu_reload_rayjob_uid"])) is None
        or UID.fullmatch(str(qualification["gpu_reload_pod_uid"])) is None
        or qualification["gpu_allocation_released"] is not True
    ):
        raise ValueError("qualification binding is incomplete")
    _sha(qualification["acceptance_evidence_file_sha256"], "acceptance evidence file digest")
    _sha(qualification["acceptance_evidence_sha256"], "acceptance evidence embedded digest")
    return artifact


def validate_packet(models: Mapping[str, Any], packet: Mapping[str, Any] | None) -> dict[str, Any]:
    """Bind one self-digested packet to the complete local-model alias set."""

    local = {
        str(alias): model
        for alias, model in models.items()
        if isinstance(model, Mapping)
        and isinstance(model.get("repository"), str)
        and model["repository"].startswith(LOCAL_ROOT)
    }
    if not local:
        if packet is not None:
            raise ValueError("model artifact packet is not allowed without a local SFS model")
        return {}
    if packet is None:
        raise ValueError("local SFS model requires an exact accepted artifact binding packet")
    value = _mapping(packet, "model artifact packet")
    if set(value) != {"schema", "campaign_name", "models", "authorization", "sha256"}:
        raise ValueError("model artifact packet has unknown or missing fields")
    if value.get("schema") != PACKET_SCHEMA:
        raise ValueError("unsupported model artifact packet schema")
    if value.get("campaign_name") in RETIRED_CAMPAIGNS:
        raise ValueError("model artifact packet belongs to a retired evaluation campaign")
    actual = value.get("sha256")
    if (
        not isinstance(actual, str)
        or re.fullmatch(r"[0-9a-f]{64}", actual) is None
        or hashlib.sha256(
            _canonical({key: item for key, item in value.items() if key != "sha256"})
        ).hexdigest()
        != actual
    ):
        raise ValueError("model artifact packet digest does not validate")
    if value.get("authorization") != "provenance_only_not_launch_authorization":
        raise ValueError("model artifact packet cannot authorize an evaluation")
    bindings = _mapping(value.get("models"), "model artifact packet models")
    if set(bindings) != set(local):
        raise ValueError("model artifact packet must cover every local model exactly once")
    for alias, model in local.items():
        validate_binding(model, bindings[alias])
    return dict(value)


def _read_runtime_documents(
    config_binding: Any,
    packet: Mapping[str, Any],
    packet_path: Path | None,
    acceptance_path: Path | None,
) -> dict[str, Any]:
    binding = _mapping(config_binding, "config model artifact binding")
    if set(binding) != {
        "validator_file_sha256",
        "packet_path",
        "packet_file_sha256",
        "packet_sha256",
        "acceptance_evidence_path",
        "acceptance_evidence_file_sha256",
        "acceptance_evidence_sha256",
    }:
        raise ValueError("config model artifact binding has unknown or missing fields")
    if packet_path is None or acceptance_path is None:
        raise ValueError("model artifact packet and acceptance evidence must both be staged")
    _, validator_file_sha256 = _read_stable_bytes(Path(__file__), "model artifact validator")
    if validator_file_sha256 != _sha(
        binding["validator_file_sha256"], "model artifact validator file digest"
    ):
        raise ValueError("model artifact validator differs from the config-bound code")
    staged_packet, packet_file_sha256 = _read_stable_json(packet_path, "model artifact packet")
    if staged_packet != packet:
        raise ValueError("parsed model artifact packet differs from its staged bytes")
    if packet_file_sha256 != _sha(
        binding["packet_file_sha256"], "packet file digest"
    ) or "sha256:" + str(packet.get("sha256")) != _sha(
        binding["packet_sha256"], "packet semantic digest"
    ):
        raise ValueError("model artifact packet differs from the config-bound packet")
    evidence, evidence_file_sha256 = _read_stable_json(
        acceptance_path, "model artifact acceptance evidence"
    )
    expected_file = _sha(
        binding["acceptance_evidence_file_sha256"], "acceptance evidence file digest"
    )
    expected_self = _sha(binding["acceptance_evidence_sha256"], "acceptance evidence digest")
    if evidence_file_sha256 != expected_file or evidence.get("sha256") != expected_self:
        raise ValueError("acceptance evidence differs from the config-bound evidence")
    if _digest(_canonical({key: item for key, item in evidence.items() if key != "sha256"})) != (
        expected_self
    ):
        raise ValueError("acceptance evidence self-digest does not validate")
    return evidence


def _validate_acceptance_evidence(
    evidence: Mapping[str, Any],
    binding: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    export: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> None:
    qualification = binding["qualification"]
    checkpoint_binding = binding["checkpoint_manifest"]
    export_binding = binding["export_receipt"]
    gpu_binding = binding["gpu_reload_receipt"]
    training = _mapping(evidence.get("training"), "acceptance training evidence")
    exported = _mapping(evidence.get("export"), "acceptance export evidence")
    cpu = _mapping(evidence.get("cpu_check"), "acceptance CPU evidence")
    gpu_check = _mapping(evidence.get("gpu_check"), "acceptance GPU evidence")
    release = _mapping(evidence.get("release"), "acceptance release evidence")
    if (
        evidence.get("schema") != "cyber_qwen38_fresh75_step230_reload_acceptance_v1"
        or training.get("status") != "SUCCEEDED"
        or training.get("checkpoint_path") != checkpoint.get("checkpoint_path")
        or training.get("optimizer_steps_executed") != checkpoint.get("optimizer_step")
        or training.get("checkpoint_manifest_file_sha256") != checkpoint_binding["file_sha256"]
        or training.get("checkpoint_receipt_sha256") != checkpoint_binding["receipt_sha256"]
    ):
        raise ValueError("acceptance evidence does not bind the native checkpoint")
    payload = binding["payload"]
    if (
        exported.get("receipt_path") != export_binding["path"]
        or exported.get("receipt_file_sha256") != export_binding["file_sha256"]
        or exported.get("receipt_sha256") != export_binding["receipt_sha256"]
        or exported.get("source_plan_sha256") != "sha256:" + str(export.get("source_plan_sha256"))
        or exported.get("payload_manifest_sha256") != payload["manifest_sha256"]
        or exported.get("dtype") != "BF16"
        or exported.get("tensor_count") != payload["tensor_count"]
        or exported.get("tensor_bytes") != payload["tensor_bytes"]
        or exported.get("payload_files") != payload["file_count"]
        or exported.get("source_unchanged") is not True
        or exported.get("all_output_tensors_reopened_equal") is not True
        or exported.get("optimizer_steps_executed") != 0
    ):
        raise ValueError("acceptance evidence does not bind the complete BF16 export")
    if (
        cpu.get("status") != "passed"
        or cpu.get("gpus") != 0
        or cpu.get("gpu_reload_verified") is not False
        or cpu.get("source_unchanged") is not True
        or cpu.get("optimizer_steps_executed") != 0
    ):
        raise ValueError("acceptance evidence does not bind the CPU integrity reload")
    if (
        gpu_check.get("jobs_api_status") != "SUCCEEDED"
        or gpu_check.get("jobs_api_output") != str(PurePosixPath(gpu_binding["path"]).parent)
        or gpu_check.get("rayjob_uid") != qualification["gpu_reload_rayjob_uid"]
        or gpu_check.get("pod_uid") != qualification["gpu_reload_pod_uid"]
        or gpu_check.get("gpu_check_file_sha256") != gpu_binding["file_sha256"]
        or gpu_check.get("gpu_check_receipt_sha256") != gpu_binding["receipt_sha256"]
        or gpu_check.get("status") != "passed"
        or gpu_check.get("gpus") != 1
        or gpu_check.get("gpu_reload_verified") is not True
        or gpu_check.get("finite_logits") is not True
        or gpu_check.get("generated_tokens") != 2
        or gpu_check.get("source_unchanged") is not True
        or gpu_check.get("serving_qualified") is not False
        or gpu_check.get("optimizer_steps_executed") != 0
    ):
        raise ValueError("acceptance evidence does not bind the finite zero-update GPU reload")
    if (
        release.get("raycluster_absent") is not True
        or release.get("pod_absent") is not True
        or release.get("workload_finished") is not True
        or release.get("gpu_allocation_released") is not True
        or qualification["gpu_allocation_released"] is not True
    ):
        raise ValueError("acceptance evidence does not prove UID-bound GPU release")


def validate_live_models(
    models: Mapping[str, Any],
    packet: Mapping[str, Any] | None,
    *,
    config_binding: Mapping[str, Any] | None = None,
    packet_path: Path | None = None,
    acceptance_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Reopen exact producer receipts for every local model before side effects."""

    value = validate_packet(models, packet)
    if not value:
        if config_binding is not None or packet_path is not None or acceptance_path is not None:
            raise ValueError("model artifact runtime inputs are not allowed without a local model")
        return {}
    bindings = value.get("models", {})
    evidence = _read_runtime_documents(config_binding, value, packet_path, acceptance_path)
    proof: dict[str, dict[str, Any]] = {}
    for alias, raw in models.items():
        model = _mapping(raw, f"model {alias}")
        repository = model.get("repository")
        if not isinstance(repository, str) or not repository.startswith(LOCAL_ROOT):
            continue
        binding = bindings[str(alias)]
        checkpoint = _read_receipt(binding["checkpoint_manifest"], "checkpoint manifest")
        export = _read_receipt(binding["export_receipt"], "export receipt")
        gpu = _read_receipt(binding["gpu_reload_receipt"], "GPU reload receipt")
        qualification = binding["qualification"]
        if (
            qualification["acceptance_evidence_path"] != config_binding["acceptance_evidence_path"]
            or qualification["acceptance_evidence_file_sha256"]
            != config_binding["acceptance_evidence_file_sha256"]
            or qualification["acceptance_evidence_sha256"]
            != config_binding["acceptance_evidence_sha256"]
        ):
            raise ValueError("packet qualification differs from the config-bound evidence")
        _validate_acceptance_evidence(evidence, binding, checkpoint, export, gpu)
        payload = binding["payload"]
        if (
            export.get("source_checkpoint_receipt_sha256")
            != binding["checkpoint_manifest"]["receipt_sha256"].removeprefix("sha256:")
            or export.get("source_manifest_file_sha256")
            != binding["checkpoint_manifest"]["file_sha256"].removeprefix("sha256:")
            or export.get("optimizer_step") != checkpoint.get("optimizer_step")
            or export.get("output_root") != repository
            or export.get("trained_tensors") != payload["trained_tensors"]
            or export.get("restored_base_tensors") != sorted(payload["restored_mtp_tensors"])
            or export.get("tensor_bytes") != payload["tensor_bytes"]
            or len(export.get("files", {})) != payload["file_count"]
            or _digest(_canonical(export.get("files"))) != payload["manifest_sha256"]
            or gpu.get("export_sha256")
            != binding["export_receipt"]["file_sha256"].removeprefix("sha256:")
            or gpu.get("export_receipt_sha256")
            != binding["export_receipt"]["receipt_sha256"].removeprefix("sha256:")
        ):
            raise ValueError("accepted checkpoint/export/reload chain is inconsistent")
        proof[str(alias)] = {
            "repository": repository,
            "revision": model["revision"],
            "checkpoint_path": checkpoint["checkpoint_path"],
            "checkpoint_manifest_file_sha256": binding["checkpoint_manifest"]["file_sha256"],
            "checkpoint_manifest_receipt_sha256": binding["checkpoint_manifest"]["receipt_sha256"],
            "export_receipt_file_sha256": binding["export_receipt"]["file_sha256"],
            "export_receipt_sha256": binding["export_receipt"]["receipt_sha256"],
            "payload_manifest_sha256": payload["manifest_sha256"],
            "gpu_reload_receipt_file_sha256": binding["gpu_reload_receipt"]["file_sha256"],
            "gpu_reload_receipt_sha256": binding["gpu_reload_receipt"]["receipt_sha256"],
            "acceptance_evidence_path": qualification["acceptance_evidence_path"],
            "acceptance_evidence_file_sha256": qualification["acceptance_evidence_file_sha256"],
            "acceptance_evidence_sha256": qualification["acceptance_evidence_sha256"],
            "gpu_reload_rayjob_uid": qualification["gpu_reload_rayjob_uid"],
            "gpu_reload_pod_uid": qualification["gpu_reload_pod_uid"],
            "gpu_allocation_released": True,
        }
    return proof
