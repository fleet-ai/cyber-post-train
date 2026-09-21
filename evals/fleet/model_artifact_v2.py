"""Checkpoint provenance for Fleet evaluations beyond the Fresh75 artifact.

Version 1 intentionally remains byte-stable because existing evaluation
configs bind its file digest.  This successor reuses the same strict native
checkpoint, BF16 export, and finite one-GPU reload gates, while adding an
explicit binding between the canonical export inventory and the immutable
revision used by the serving registration.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from evals.fleet import model_artifact as v1

BINDING_SCHEMA = "cyber_fleet_eval_model_artifact_v2"
PACKET_SCHEMA = "cyber_fleet_eval_model_artifact_packet_v2"
ACCEPTANCE_SCHEMA = "cyber_qwen38_checkpoint_eval_acceptance_v1"
REVISION_BASES = {"export_files_canonical_sha256", "export_receipt_sha256"}


def _legacy_binding(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Project the common v2 fields through the already-reviewed v1 gates."""

    payload = v1._mapping(artifact.get("payload"), "export payload binding")  # noqa: SLF001
    return {
        **artifact,
        "schema": v1.SCHEMA,
        "payload": {
            "manifest_sha256": payload.get("served_revision"),
            "file_count": payload.get("file_count"),
            "trained_tensors": payload.get("trained_tensors"),
            "restored_mtp_tensors": payload.get("restored_mtp_tensors"),
            "tensor_count": payload.get("tensor_count"),
            "tensor_bytes": payload.get("tensor_bytes"),
        },
    }


def validate_binding(model: Mapping[str, Any], binding: Any) -> Mapping[str, Any]:
    """Validate one v2 artifact binding without opening its SFS receipts."""

    artifact = v1._mapping(binding, "model artifact binding")  # noqa: SLF001
    if artifact.get("schema") != BINDING_SCHEMA:
        raise ValueError("model artifact binding has an unsupported schema")
    payload = v1._mapping(artifact.get("payload"), "export payload binding")  # noqa: SLF001
    if set(payload) != {
        "served_revision",
        "revision_basis",
        "export_files_sha256",
        "stage_source_path",
        "file_count",
        "trained_tensors",
        "restored_mtp_tensors",
        "tensor_count",
        "tensor_bytes",
    }:
        raise ValueError("v2 export payload binding has unknown or missing fields")
    served_revision = v1._sha(payload.get("served_revision"), "served revision")  # noqa: SLF001
    export_files = v1._sha(  # noqa: SLF001
        payload.get("export_files_sha256"), "export files manifest"
    )
    basis = payload.get("revision_basis")
    source_path = payload.get("stage_source_path")
    if (
        served_revision != model.get("revision")
        or basis not in REVISION_BASES
        or not isinstance(source_path, str)
        or not source_path.startswith("/models/")
        or ".." in PurePosixPath(source_path).parts
    ):
        raise ValueError("v2 serving revision provenance is incomplete")
    export_binding = v1._mapping(artifact.get("export_receipt"), "export receipt binding")  # noqa: SLF001
    if basis == "export_files_canonical_sha256" and served_revision != export_files:
        raise ValueError("served revision does not equal the canonical export inventory")
    if basis == "export_receipt_sha256" and served_revision != export_binding.get("receipt_sha256"):
        raise ValueError("served revision does not equal the accepted export receipt")
    v1.validate_binding(model, _legacy_binding(artifact))
    return artifact


def validate_packet(models: Mapping[str, Any], packet: Mapping[str, Any] | None) -> dict[str, Any]:
    """Bind one self-digested v2 packet to exactly one local model alias."""

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
        raise ValueError("v2 model artifact packets cover exactly one local model")
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
            v1._canonical({key: item for key, item in evidence.items() if key != "sha256"})  # noqa: SLF001
        )
        != expected_self
    ):
        raise ValueError("acceptance evidence self-digest does not validate")
    return evidence


def _validate_acceptance_evidence(
    alias: str,
    evidence: Mapping[str, Any],
    binding: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    export: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> None:
    if set(evidence) != {
        "schema",
        "artifact_alias",
        "observed_at",
        "source_evidence",
        "training",
        "export",
        "gpu_check",
        "stage",
        "release",
        "privacy",
        "sha256",
    } or not isinstance(evidence.get("observed_at"), str):
        raise ValueError("acceptance evidence has an unsupported shape")
    qualification = binding["qualification"]
    checkpoint_binding = binding["checkpoint_manifest"]
    export_binding = binding["export_receipt"]
    gpu_binding = binding["gpu_reload_receipt"]
    payload = binding["payload"]
    training = v1._mapping(evidence.get("training"), "acceptance training evidence")  # noqa: SLF001
    exported = v1._mapping(evidence.get("export"), "acceptance export evidence")  # noqa: SLF001
    gpu_check = v1._mapping(evidence.get("gpu_check"), "acceptance GPU evidence")  # noqa: SLF001
    stage = v1._mapping(evidence.get("stage"), "acceptance stage evidence")  # noqa: SLF001
    release = v1._mapping(evidence.get("release"), "acceptance release evidence")  # noqa: SLF001
    privacy = v1._mapping(evidence.get("privacy"), "acceptance privacy evidence")  # noqa: SLF001
    if (
        set(training)
        != {
            "status",
            "checkpoint_path",
            "optimizer_steps_executed",
            "checkpoint_manifest_file_sha256",
            "checkpoint_receipt_sha256",
        }
        or set(exported)
        != {
            "receipt_path",
            "receipt_file_sha256",
            "receipt_sha256",
            "source_plan_sha256",
            "export_files_sha256",
            "dtype",
            "tensor_count",
            "tensor_bytes",
            "payload_files",
            "source_unchanged",
            "all_output_tensors_reopened_equal",
            "optimizer_steps_executed",
        }
        or set(gpu_check)
        != {
            "rayjob_uid",
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
        }
    ):
        raise ValueError("acceptance evidence receipt projection has an unsupported shape")
    if (
        evidence.get("schema") != ACCEPTANCE_SCHEMA
        or evidence.get("artifact_alias") != alias
        or training.get("status") != "SUCCEEDED"
        or training.get("checkpoint_path") != checkpoint.get("checkpoint_path")
        or training.get("optimizer_steps_executed") != checkpoint.get("optimizer_step")
        or training.get("checkpoint_manifest_file_sha256") != checkpoint_binding["file_sha256"]
        or training.get("checkpoint_receipt_sha256") != checkpoint_binding["receipt_sha256"]
    ):
        raise ValueError("acceptance evidence does not bind the native checkpoint")
    if (
        exported.get("receipt_path") != export_binding["path"]
        or exported.get("receipt_file_sha256") != export_binding["file_sha256"]
        or exported.get("receipt_sha256") != export_binding["receipt_sha256"]
        or exported.get("source_plan_sha256") != "sha256:" + str(export.get("source_plan_sha256"))
        or exported.get("export_files_sha256") != payload["export_files_sha256"]
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
        gpu_check.get("rayjob_uid") != qualification["gpu_reload_rayjob_uid"]
        or gpu_check.get("pod_uid") != qualification["gpu_reload_pod_uid"]
        or gpu_check.get("gpu_check_path") != gpu_binding["path"]
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
        stage
        != {
            "status": "accepted",
            "source_export_root": export.get("output_root"),
            "source_export_receipt_file_sha256": export_binding["file_sha256"],
            "source_export_receipt_sha256": export_binding["receipt_sha256"],
            "model_source_path": payload["stage_source_path"],
            "served_revision": payload["served_revision"],
            "revision_basis": payload["revision_basis"],
            "stage_receipt_sha256": stage.get("stage_receipt_sha256"),
            "resources_released": True,
        }
        or v1.SHA256.fullmatch(str(stage.get("stage_receipt_sha256"))) is None
    ):
        raise ValueError("acceptance evidence does not bind the staged serving payload")
    if (
        release != {"gpu_allocation_released": True}
        or qualification["gpu_allocation_released"] is not True
        or privacy
        != {
            "credentials_included": False,
            "prompts_traces_flags_answers_or_scores_included": False,
            "raw_logs_included": False,
            "tensor_values_read": False,
            "generated_token_text_read": False,
        }
    ):
        raise ValueError("acceptance evidence does not prove release and privacy boundaries")
    sources = evidence.get("source_evidence")
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(item, str) or not item for item in sources)
    ):
        raise ValueError("acceptance evidence has no immutable source citations")


def validate_live_models(
    models: Mapping[str, Any],
    packet: Mapping[str, Any] | None,
    *,
    config_binding: Mapping[str, Any] | None = None,
    packet_path: Path | None = None,
    acceptance_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Reopen the exact v2 receipt chain before evaluator side effects."""

    value = validate_packet(models, packet)
    if not value:
        if config_binding is not None or packet_path is not None or acceptance_path is not None:
            raise ValueError("model artifact runtime inputs are not allowed without a local model")
        return {}
    bindings = value["models"]
    evidence = _read_runtime_documents(config_binding, value, packet_path, acceptance_path)
    proof: dict[str, dict[str, Any]] = {}
    for alias, raw in models.items():
        model = v1._mapping(raw, f"model {alias}")  # noqa: SLF001
        repository = model.get("repository")
        if not isinstance(repository, str) or not repository.startswith(v1.LOCAL_ROOT):
            continue
        binding = bindings[str(alias)]
        checkpoint = v1._read_receipt(  # noqa: SLF001
            binding["checkpoint_manifest"], "checkpoint manifest"
        )
        export = v1._read_receipt(binding["export_receipt"], "export receipt")  # noqa: SLF001
        gpu = v1._read_receipt(binding["gpu_reload_receipt"], "GPU reload receipt")  # noqa: SLF001
        qualification = binding["qualification"]
        if (
            qualification["acceptance_evidence_path"] != config_binding["acceptance_evidence_path"]
            or qualification["acceptance_evidence_file_sha256"]
            != config_binding["acceptance_evidence_file_sha256"]
            or qualification["acceptance_evidence_sha256"]
            != config_binding["acceptance_evidence_sha256"]
        ):
            raise ValueError("packet qualification differs from the config-bound evidence")
        _validate_acceptance_evidence(str(alias), evidence, binding, checkpoint, export, gpu)
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
            "gpu_reload_rayjob_uid": qualification["gpu_reload_rayjob_uid"],
            "gpu_reload_pod_uid": qualification["gpu_reload_pod_uid"],
            "gpu_allocation_released": True,
        }
    return proof
