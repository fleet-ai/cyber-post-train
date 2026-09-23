"""Checkpoint provenance for accepted reloads run as a Pod or RayJob.

The earlier v2 packet required every GPU reload to have a RayJob UID.  Some
reviewed qualification paths, including Teacher3K step 1000, use one bounded
development Pod instead.  This schema preserves the exact workload kind and
UID while reusing every v2 checkpoint, export, stage, and receipt check.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import model_artifact as v1
from evals.fleet import model_artifact_v2 as v2

BINDING_SCHEMA = "cyber_fleet_eval_model_artifact_v3"
PACKET_SCHEMA = "cyber_fleet_eval_model_artifact_packet_v3"
ACCEPTANCE_SCHEMA = "cyber_qwen38_checkpoint_eval_acceptance_v2"
WORKLOAD_KINDS = {"Pod", "RayJob"}


def _qualification(binding: Mapping[str, Any]) -> Mapping[str, Any]:
    value = v1._mapping(binding.get("qualification"), "qualification binding")  # noqa: SLF001
    if set(value) != {
        "acceptance_evidence_path",
        "acceptance_evidence_file_sha256",
        "acceptance_evidence_sha256",
        "gpu_reload_workload_kind",
        "gpu_reload_workload_uid",
        "gpu_reload_pod_uid",
        "gpu_allocation_released",
    }:
        raise ValueError("v3 qualification binding has unknown or missing fields")
    kind = value.get("gpu_reload_workload_kind")
    workload_uid = str(value.get("gpu_reload_workload_uid"))
    pod_uid = str(value.get("gpu_reload_pod_uid"))
    if (
        kind not in WORKLOAD_KINDS
        or v1.UID.fullmatch(workload_uid) is None
        or v1.UID.fullmatch(pod_uid) is None
        or (kind == "Pod" and workload_uid != pod_uid)
        or value.get("gpu_allocation_released") is not True
    ):
        raise ValueError("v3 GPU reload workload binding is invalid")
    return value


def _v2_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Project only common checks into v2; no projected identity is emitted."""

    value = copy.deepcopy(dict(binding))
    qualification = _qualification(value)
    value["schema"] = v2.BINDING_SCHEMA
    value["qualification"] = {
        "acceptance_evidence_path": qualification["acceptance_evidence_path"],
        "acceptance_evidence_file_sha256": qualification["acceptance_evidence_file_sha256"],
        "acceptance_evidence_sha256": qualification["acceptance_evidence_sha256"],
        # v2 uses this field only as a syntactic UID and later compares it with
        # an in-memory acceptance projection.  v3 reports the truthful kind.
        "gpu_reload_rayjob_uid": qualification["gpu_reload_workload_uid"],
        "gpu_reload_pod_uid": qualification["gpu_reload_pod_uid"],
        "gpu_allocation_released": True,
    }
    return value


def validate_binding(model: Mapping[str, Any], binding: Any) -> Mapping[str, Any]:
    artifact = v1._mapping(binding, "model artifact binding")  # noqa: SLF001
    if artifact.get("schema") != BINDING_SCHEMA:
        raise ValueError("model artifact binding has an unsupported schema")
    _qualification(artifact)
    v2.validate_binding(model, _v2_binding(artifact))
    return artifact


def validate_packet(models: Mapping[str, Any], packet: Mapping[str, Any] | None) -> dict[str, Any]:
    local = {
        str(alias): model
        for alias, model in models.items()
        if isinstance(model, Mapping)
        and isinstance(model.get("repository"), str)
        and model["repository"].startswith(v1.LOCAL_ROOT)
    }
    if not local:
        if packet is not None:
            raise ValueError("model artifact packet is not allowed without a local SFS model")
        return {}
    if len(local) != 1:
        raise ValueError("v3 model artifact packets cover exactly one local model")
    if packet is None:
        raise ValueError("local SFS model requires an exact accepted artifact binding packet")
    value = v1._mapping(packet, "model artifact packet")  # noqa: SLF001
    if set(value) != {"schema", "campaign_name", "models", "authorization", "sha256"}:
        raise ValueError("model artifact packet has unknown or missing fields")
    if value.get("schema") != PACKET_SCHEMA:
        raise ValueError("unsupported model artifact packet schema")
    actual = value.get("sha256")
    if (
        not isinstance(actual, str)
        or re.fullmatch(r"[0-9a-f]{64}", actual) is None
        or hashlib.sha256(
            v1._canonical({key: item for key, item in value.items() if key != "sha256"})  # noqa: SLF001
        ).hexdigest()
        != actual
    ):
        raise ValueError("model artifact packet digest does not validate")
    if value.get("authorization") != "provenance_only_not_launch_authorization":
        raise ValueError("model artifact packet cannot authorize an evaluation")
    bindings = v1._mapping(value.get("models"), "model artifact packet models")  # noqa: SLF001
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
    """Bind the staged packet/evidence to this exact v3 validator."""

    binding = v1._mapping(config_binding, "config model artifact binding")  # noqa: SLF001
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
    _, validator_file_sha256 = v1._read_stable_bytes(  # noqa: SLF001
        Path(__file__), "model artifact validator"
    )
    if validator_file_sha256 != v1._sha(  # noqa: SLF001
        binding["validator_file_sha256"], "model artifact validator file digest"
    ):
        raise ValueError("model artifact validator differs from the config-bound code")
    staged_packet, packet_file_sha256 = v1._read_stable_json(  # noqa: SLF001
        packet_path, "model artifact packet"
    )
    if staged_packet != packet:
        raise ValueError("parsed model artifact packet differs from its staged bytes")
    if packet_file_sha256 != v1._sha(  # noqa: SLF001
        binding["packet_file_sha256"], "packet file digest"
    ) or "sha256:" + str(packet.get("sha256")) != v1._sha(  # noqa: SLF001
        binding["packet_sha256"], "packet semantic digest"
    ):
        raise ValueError("model artifact packet differs from the config-bound packet")
    evidence, evidence_file_sha256 = v1._read_stable_json(  # noqa: SLF001
        acceptance_path, "model artifact acceptance evidence"
    )
    expected_file = v1._sha(  # noqa: SLF001
        binding["acceptance_evidence_file_sha256"], "acceptance evidence file digest"
    )
    expected_self = v1._sha(  # noqa: SLF001
        binding["acceptance_evidence_sha256"], "acceptance evidence digest"
    )
    if evidence_file_sha256 != expected_file or evidence.get("sha256") != expected_self:
        raise ValueError("acceptance evidence differs from the config-bound evidence")
    if (
        v1._digest(  # noqa: SLF001
            v1._canonical(  # noqa: SLF001
                {key: item for key, item in evidence.items() if key != "sha256"}
            )
        )
        != expected_self
    ):
        raise ValueError("acceptance evidence self-digest does not validate")
    return evidence


def _v2_acceptance(evidence: Mapping[str, Any], binding: Mapping[str, Any]) -> tuple[dict, dict]:
    projected_evidence = copy.deepcopy(dict(evidence))
    if projected_evidence.get("schema") != ACCEPTANCE_SCHEMA:
        raise ValueError("acceptance evidence has an unsupported schema")
    gpu = v1._mapping(projected_evidence.get("gpu_check"), "acceptance GPU evidence")  # noqa: SLF001
    if set(gpu) != {
        "workload_kind",
        "workload_uid",
        "pod_uid",
        "gpu_check_path",
        "gpu_check_file_sha256",
        "gpu_check_receipt_sha256",
        "status",
        "gpus",
        "gpu_reload_verified",
        "finite_logits",
        "generated_tokens",
        "source_unchanged",
        "serving_qualified",
        "optimizer_steps_executed",
    }:
        raise ValueError("acceptance GPU evidence has an unsupported shape")
    qualification = _qualification(binding)
    if (
        gpu.get("workload_kind") != qualification["gpu_reload_workload_kind"]
        or gpu.get("workload_uid") != qualification["gpu_reload_workload_uid"]
        or gpu.get("pod_uid") != qualification["gpu_reload_pod_uid"]
    ):
        raise ValueError("acceptance evidence differs from the GPU workload binding")
    projected_evidence["schema"] = v2.ACCEPTANCE_SCHEMA
    projected_evidence["gpu_check"] = {
        **{key: item for key, item in gpu.items() if key not in {"workload_kind", "workload_uid"}},
        "rayjob_uid": gpu["workload_uid"],
    }
    return projected_evidence, _v2_binding(binding)


def validate_live_models(
    models: Mapping[str, Any],
    packet: Mapping[str, Any] | None,
    *,
    config_binding: Mapping[str, Any] | None = None,
    packet_path: Path | None = None,
    acceptance_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Reopen the exact receipt chain before any evaluator side effect."""

    value = validate_packet(models, packet)
    if not value:
        if config_binding is not None or packet_path is not None or acceptance_path is not None:
            raise ValueError("model artifact runtime inputs are not allowed without a local model")
        return {}
    evidence = _read_runtime_documents(config_binding, value, packet_path, acceptance_path)
    proof: dict[str, dict[str, Any]] = {}
    for alias, raw in models.items():
        model = v1._mapping(raw, f"model {alias}")  # noqa: SLF001
        repository = model.get("repository")
        if not isinstance(repository, str) or not repository.startswith(v1.LOCAL_ROOT):
            continue
        binding = value["models"][str(alias)]
        qualification = _qualification(binding)
        if (
            qualification["acceptance_evidence_path"] != config_binding["acceptance_evidence_path"]
            or qualification["acceptance_evidence_file_sha256"]
            != config_binding["acceptance_evidence_file_sha256"]
            or qualification["acceptance_evidence_sha256"]
            != config_binding["acceptance_evidence_sha256"]
        ):
            raise ValueError("packet qualification differs from the config-bound evidence")
        checkpoint = v1._read_receipt(  # noqa: SLF001
            binding["checkpoint_manifest"], "checkpoint manifest"
        )
        export = v1._read_receipt(binding["export_receipt"], "export receipt")  # noqa: SLF001
        gpu = v1._read_receipt(binding["gpu_reload_receipt"], "GPU reload receipt")  # noqa: SLF001
        projected_evidence, projected_binding = _v2_acceptance(evidence, binding)
        v2._validate_acceptance_evidence(  # noqa: SLF001
            str(alias), projected_evidence, projected_binding, checkpoint, export, gpu
        )
        payload = binding["payload"]
        export_files_sha256 = v1._digest(v1._canonical(export.get("files")))  # noqa: SLF001
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
            or export_files_sha256 != payload["export_files_sha256"]
            or gpu.get("export_sha256")
            != binding["export_receipt"]["file_sha256"].removeprefix("sha256:")
            or gpu.get("export_receipt_sha256")
            != binding["export_receipt"]["receipt_sha256"].removeprefix("sha256:")
        ):
            raise ValueError("accepted checkpoint/export/reload chain is inconsistent")
        proof[str(alias)] = {
            "repository": repository,
            "revision": model["revision"],
            "revision_basis": payload["revision_basis"],
            "export_files_sha256": export_files_sha256,
            "stage_source_path": payload["stage_source_path"],
            "checkpoint_path": checkpoint["checkpoint_path"],
            "checkpoint_manifest_file_sha256": binding["checkpoint_manifest"]["file_sha256"],
            "checkpoint_manifest_receipt_sha256": binding["checkpoint_manifest"]["receipt_sha256"],
            "export_receipt_file_sha256": binding["export_receipt"]["file_sha256"],
            "export_receipt_sha256": binding["export_receipt"]["receipt_sha256"],
            "gpu_reload_receipt_file_sha256": binding["gpu_reload_receipt"]["file_sha256"],
            "gpu_reload_receipt_sha256": binding["gpu_reload_receipt"]["receipt_sha256"],
            "acceptance_evidence_path": qualification["acceptance_evidence_path"],
            "acceptance_evidence_file_sha256": qualification["acceptance_evidence_file_sha256"],
            "acceptance_evidence_sha256": qualification["acceptance_evidence_sha256"],
            "gpu_reload_workload_kind": qualification["gpu_reload_workload_kind"],
            "gpu_reload_workload_uid": qualification["gpu_reload_workload_uid"],
            "gpu_reload_pod_uid": qualification["gpu_reload_pod_uid"],
            "gpu_allocation_released": True,
        }
    return proof
