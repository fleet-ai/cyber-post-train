"""Fail-closed receipts for selecting and evaluating an SFT checkpoint.

This module deliberately does not call Fleet, Kubernetes, or an inference endpoint.  Operators
capture those systems' read-only responses and pass them here.  That keeps checkpoint selection
independent of benchmark results and makes every mutating step (export, registration, evaluation)
conditional on a reviewable immutable receipt.
"""

from __future__ import annotations

import copy
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .io import digest_json

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
IMAGE_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
MODEL_ID_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
CHECKPOINT_NAMESPACE = uuid.UUID("bc7f7c3f-38ba-5835-840d-ecb75c2d2f64")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Mapping[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return item.strip()


def _sha256(value: Mapping[str, Any], field: str) -> str:
    item = _text(value, field)
    if not SHA256_RE.fullmatch(item):
        raise ValueError(f"{field} must be sha256:<64 hex>")
    return item


def _digest_pinned_image(value: Mapping[str, Any], field: str) -> str:
    item = _text(value, field)
    if not IMAGE_RE.fullmatch(item):
        raise ValueError(f"{field} must be an image pinned by sha256 digest")
    return item


def _with_digest(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result[field] = digest_json(result)
    return result


def _validate_embedded_digest(value: Mapping[str, Any], field: str) -> str:
    expected = _sha256(value, field)
    unsigned = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    actual = digest_json(unsigned)
    if expected != actual:
        raise ValueError(f"{field} digest mismatch")
    return actual


def checkpoint_uuid(run_name: str, step: int) -> str:
    """Return Fleet's deterministic training-checkpoint UUID."""

    return str(uuid.uuid5(CHECKPOINT_NAMESPACE, f"{run_name}/{step}"))


def _validate_export_binding(
    binding: Mapping[str, Any], selection: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str]:
    """Bind the one predeclared zero-step export job and its collision-free destination."""

    if binding.get("strategy") != "resume_final_checkpoint_with_zero_optimizer_steps_v1":
        raise ValueError("unsupported export strategy")
    run = _mapping(selection.get("run"), "selection.run")
    checkpoint = _mapping(selection.get("checkpoint"), "selection.checkpoint")
    source_run = _text(run, "name")
    step = checkpoint.get("step")
    source_path = _text(checkpoint, "sfs_path")
    if _text(binding, "source_run_name") != source_run:
        raise ValueError("export binding names a different source run")
    if binding.get("source_global_step") != step:
        raise ValueError("export binding names a different source global step")
    if _text(binding, "source_checkpoint_path") != source_path:
        raise ValueError("export binding names a different source checkpoint path")
    output_root = _text(binding, "output_root")
    expected_output_path = f"{output_root}/global_step_{step}/policy"
    if _text(binding, "expected_output_path") != expected_output_path:
        raise ValueError("export binding output path is inconsistent with its output root")

    export_run = _mapping(binding.get("export_run"), "export binding export_run")
    _text(export_run, "name")
    _text(export_run, "run_id")
    _text(export_run, "rayjob_uid")
    _text(export_run, "trainer_version_id")
    _digest_pinned_image(export_run, "trainer_image")
    if _text(export_run, "resume_from") != source_path:
        raise ValueError("export run does not resume from the selected checkpoint")
    if export_run.get("num_steps") != step:
        raise ValueError("export run num_steps must equal the selected resume step")
    if export_run.get("hf_save_interval") != step + 1:
        raise ValueError("export run save interval must be one past the selected step")
    if export_run.get("optimizer_steps_expected") != 0:
        raise ValueError("export run must predeclare zero optimizer steps")

    preflight = _mapping(binding.get("destination_preflight"), "export destination preflight")
    _text(preflight, "observed_at")
    if preflight.get("state") != "absent":
        raise ValueError("export destination was not proven absent before submission")
    matches = preflight.get("matching_rayjobs")
    if not isinstance(matches, list) or len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise ValueError("export destination must be uniquely assigned to one RayJob")
    match = matches[0]
    if _text(match, "name") != _text(export_run, "name") or _text(match, "uid") != _text(
        export_run, "rayjob_uid"
    ):
        raise ValueError("export destination is assigned to a different RayJob")
    return export_run, expected_output_path


def validate_selection_receipt(selection: Mapping[str, Any]) -> str:
    """Validate the selected checkpoint's complete identity before any downstream render."""

    schema = selection.get("schema")
    if schema not in {"cyber_sft_checkpoint_selection_v1", "cyber_sft_checkpoint_selection_v2"}:
        raise ValueError("unsupported checkpoint selection receipt schema")
    if selection.get("selection_policy") != "final_promoted_checkpoint_after_successful_run_v1":
        raise ValueError("unsupported checkpoint selection policy")
    run = _mapping(selection.get("run"), "selection.run")
    checkpoint = _mapping(selection.get("checkpoint"), "selection.checkpoint")
    if _text(run, "terminal_status") != "succeeded":
        raise ValueError("selection receipt is not for a successful run")
    run_name = _text(run, "name")
    step = checkpoint.get("step")
    if not isinstance(step, int) or step < 1:
        raise ValueError("selected checkpoint step must be a positive integer")
    if _text(checkpoint, "uuid") != checkpoint_uuid(run_name, step):
        raise ValueError("selected checkpoint UUID does not match run and step")
    if _text(checkpoint, "fleet_model") != f"fleet/{run_name}-step-{step}":
        raise ValueError("selected checkpoint Fleet model does not match run and step")
    if checkpoint.get("complete") is not True or checkpoint.get("promoted") is not True:
        raise ValueError("selected checkpoint must be complete and promoted")
    if schema == "cyber_sft_checkpoint_selection_v1":
        _sha256(checkpoint, "archive_manifest_sha256")
    else:
        _sha256(checkpoint, "source_manifest_sha256")
        if checkpoint.get("source_manifest_kind") != "sfs_sha256_all_files_v1":
            raise ValueError("SFS selection must hash every checkpoint file")
        evidence = _mapping(checkpoint.get("sfs_evidence"), "selection.checkpoint.sfs_evidence")
        if evidence.get("api_checkpoint_rows") != 0:
            raise ValueError("SFS selection requires the observed empty checkpoint API index")
        pipeline = _mapping(evidence.get("checkpoint_pipeline"), "checkpoint pipeline evidence")
        _text(pipeline, "application_uid")
        if not re.fullmatch(r"[0-9a-f]{40}", _text(pipeline, "sync_revision")):
            raise ValueError("checkpoint pipeline revision must be an exact Git commit")
        _sha256(pipeline, "helm_values_sha256")
        _text(pipeline, "observed_at")
        if pipeline.get("apply") is not False or pipeline.get("archive_enabled") is not False:
            raise ValueError("empty API index is not explained by a disabled checkpoint pipeline")
        markers = _mapping(evidence.get("markers"), "SFS checkpoint markers")
        for field in ("promoted", "milestone"):
            if markers.get(field) is not True:
                raise ValueError(f"SFS checkpoint marker {field} is missing")
        expected_shards = markers.get("expected_shards")
        complete_shards = markers.get("complete_shards")
        if not isinstance(expected_shards, int) or expected_shards < 1:
            raise ValueError("SFS checkpoint expected_shards must be positive")
        if complete_shards != expected_shards:
            raise ValueError("SFS checkpoint shard completion markers are incomplete")
        if markers.get("latest_step") != step:
            raise ValueError("SFS checkpoint is not the run's latest step")
        before = _sha256(evidence, "structural_manifest_before_sha256")
        after = _sha256(evidence, "structural_manifest_after_sha256")
        if before != after:
            raise ValueError("SFS checkpoint structure changed during conversion")
        file_count = evidence.get("full_manifest_file_count")
        total_bytes = evidence.get("full_manifest_total_bytes")
        if not isinstance(file_count, int) or file_count < 1:
            raise ValueError("SFS full manifest file count must be positive")
        if not isinstance(total_bytes, int) or total_bytes < 1:
            raise ValueError("SFS full manifest byte count must be positive")
    sfs_path = _text(checkpoint, "sfs_path")
    if sfs_path != f"/mnt/sfs/checkpoints/{run_name}/global_step_{step}":
        raise ValueError("selected checkpoint SFS path does not match run and step")
    _sha256(run, "run_config_sha256")
    _sha256(run, "entrypoint_sha256")
    _digest_pinned_image(run, "trainer_image")
    return _validate_embedded_digest(selection, "selection_receipt_sha256")


def _selection_source_manifest(selection: Mapping[str, Any]) -> str:
    checkpoint = _mapping(selection.get("checkpoint"), "selection.checkpoint")
    if selection.get("schema") == "cyber_sft_checkpoint_selection_v2":
        return _sha256(checkpoint, "source_manifest_sha256")
    return _sha256(checkpoint, "archive_manifest_sha256")


def build_zero_step_hf_export_request(
    sft_config: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    expected_trainer_version_id: str,
    expected_trainer_image: str,
    expected_export_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Render a review-only SFT request that resumes at its terminal step and only exports.

    The pinned SkyRL loop begins at ``start_step + 1``. Setting ``num_steps`` to the selected
    checkpoint step therefore executes zero optimizer steps, while a save interval greater than
    the final step forces the built-in final FSDP-to-HF safetensors export. The Training API still
    creates a separate run and checkpoint directory; this request never points ``ckpt_path`` at the
    source run.
    """

    validate_selection_receipt(selection)
    run = _mapping(selection.get("run"), "selection.run")
    checkpoint = _mapping(selection.get("checkpoint"), "selection.checkpoint")
    if _text(run, "trainer_image") != expected_trainer_image:
        raise ValueError("selection trainer image differs from the planned exporter image")
    step = checkpoint["step"]
    source_run = _text(run, "name")
    checkpoint_id = _text(checkpoint, "uuid")
    export_run, expected_output_path = _validate_export_binding(expected_export_binding, selection)
    if _text(export_run, "trainer_version_id") != expected_trainer_version_id:
        raise ValueError("export binding trainer version differs from the planned exporter")
    if _text(export_run, "trainer_image") != expected_trainer_image:
        raise ValueError("export binding trainer image differs from the planned exporter")

    request = copy.deepcopy(dict(sft_config))
    if request.get("kind") != "sft":
        raise ValueError("export source config must be an SFT request")
    model = _mapping(request.get("model"), "sft model")
    if _text(model, "precision").lower() not in {"bf16", "bfloat16"}:
        raise ValueError("zero-step export source must use BF16")
    sft = _mapping(request.get("sft"), "sft settings")
    if _text(sft, "strategy") != "fsdp":
        raise ValueError("zero-step export currently supports only the proven FSDP path")
    trainer = _mapping(request.get("trainer"), "sft trainer")
    if _text(trainer, "trainer_version_id") != expected_trainer_version_id:
        raise ValueError("SFT config trainer version differs from the planned exporter")
    args = trainer.get("args")
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("SFT trainer args must be an array of strings")
    reserved = {"resume_from", "hf_save_interval", "export_path", "num_steps", "num_epochs"}
    collisions = sorted(item.split("=", 1)[0] for item in args if item.split("=", 1)[0] in reserved)
    if collisions:
        raise ValueError("SFT trainer args already set export-owned keys: " + ", ".join(collisions))

    source_path = _text(checkpoint, "sfs_path")
    if checkpoint.get("sfs_available") is not True:
        raise ValueError("selected checkpoint must be staged on SFS before rendering export")
    export_root = _text(expected_export_binding, "output_root")
    request.pop("name", None)
    request["title"] = f"Chris cyber zero-step HF export of {source_run} step {step}"
    request["sft"]["num_epochs"] = None
    request["sft"]["max_steps"] = step
    request["eval"] = {"task_keys": [], "interval": 1, "before_train": False}
    request["trainer"]["args"] = [
        *args,
        f"resume_from={source_path}",
        f"hf_save_interval={step + 1}",
        f"export_path={export_root}",
    ]

    receipt = {
        "schema": "cyber_sft_zero_step_export_request_v1",
        "request": request,
        "source_selection_sha256": _sha256(selection, "selection_receipt_sha256"),
        "source_checkpoint": {
            "path": source_path,
            "uuid": checkpoint_id,
            "source_manifest_sha256": _selection_source_manifest(selection),
        },
        "expected_output": {
            "path": expected_output_path,
            "format": "huggingface_safetensors",
            "dtype": "bf16",
        },
        "trainer": {
            "version_id": expected_trainer_version_id,
            "image": expected_trainer_image,
            "skyrl_source_commit": "f5bc3b78dfddfb352870d5d7430cd226e5785838",
        },
        "export_run": copy.deepcopy(dict(export_run)),
        "destination_preflight": copy.deepcopy(expected_export_binding["destination_preflight"]),
        "proof_obligations": {
            "source_checkpoint_present_before_submit": True,
            "rendered_entrypoint_resume_step_equals_num_steps": True,
            "observed_optimizer_steps_equal_zero": True,
            "source_checkpoint_directory_unchanged": True,
            "output_manifest_and_load_checks_required": True,
            "inference_filesystem_staging_is_separate": True,
        },
        "submit": False,
    }
    return _with_digest(receipt, "export_request_receipt_sha256")


def freeze_final_promoted_checkpoint(
    run: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    expected_run_name: str,
    expected_run_config_sha256: str,
    expected_rayjob_uid: str,
    expected_trainer_image: str,
    expected_entrypoint_sha256: str,
) -> dict[str, Any]:
    """Select the one final promoted checkpoint after a successful run.

    Intermediate loss values are intentionally absent from the inputs.  A checkpoint therefore
    cannot be selected by looking at WebExploitBench, the Fleet test split, or even by choosing the
    most flattering development loss after the fact.
    """

    if run.get("schema") != "fleet_training_run_observation_v1":
        raise ValueError("unsupported run observation schema")
    run_name = _text(run, "run_name")
    if run_name != expected_run_name:
        raise ValueError("run observation names a different training run")
    status = _text(run, "status").lower()
    if status != "succeeded":
        raise ValueError("training run must be succeeded before checkpoint selection")
    if _text(run, "rayjob_uid") != expected_rayjob_uid:
        raise ValueError("RayJob UID differs from the predeclared run")
    if _sha256(run, "run_config_sha256") != expected_run_config_sha256:
        raise ValueError("training run config digest differs from the predeclared run")
    if _digest_pinned_image(run, "trainer_image") != expected_trainer_image:
        raise ValueError("trainer image differs from the observed immutable image")
    entrypoint_sha256 = _sha256(run, "entrypoint_sha256")
    if entrypoint_sha256 != expected_entrypoint_sha256:
        raise ValueError("RayJob entrypoint digest differs from the predeclared run")

    matching = [row for row in checkpoints if row.get("run_name") == run_name]
    promoted = [row for row in matching if row.get("is_promoted") is True]
    if len(promoted) != 1:
        raise ValueError("exactly one checkpoint for this run must be promoted")
    selected = promoted[0]
    step = selected.get("step")
    if not isinstance(step, int) or step < 1:
        raise ValueError("promoted checkpoint step must be a positive integer")
    if selected.get("complete") is not True:
        raise ValueError("promoted checkpoint is not complete")
    if selected.get("s3_available") is not True:
        raise ValueError("promoted checkpoint has no verified durable archive")
    archive_uri = _text(selected, "s3_uri")
    if not archive_uri.startswith("s3://"):
        raise ValueError("promoted checkpoint archive must be an s3 URI")
    archive_manifest_sha256 = _sha256(selected, "archive_manifest_sha256")
    sfs_path = _text(selected, "sfs_path")
    expected_sfs_path = f"/mnt/sfs/checkpoints/{run_name}/global_step_{step}"
    if sfs_path != expected_sfs_path:
        raise ValueError("promoted checkpoint has an unexpected SFS path")

    latest_step = run.get("latest_checkpoint_step")
    if latest_step != step:
        raise ValueError("promoted checkpoint is not the run's final checkpoint")

    checkpoint_id = checkpoint_uuid(run_name, step)
    receipt = {
        "schema": "cyber_sft_checkpoint_selection_v1",
        "selection_policy": "final_promoted_checkpoint_after_successful_run_v1",
        "run": {
            "name": run_name,
            "rayjob_uid": expected_rayjob_uid,
            "run_config_sha256": expected_run_config_sha256,
            "entrypoint_sha256": entrypoint_sha256,
            "trainer_image": expected_trainer_image,
            "terminal_status": "succeeded",
        },
        "checkpoint": {
            "step": step,
            "uuid": checkpoint_id,
            "fleet_model": f"fleet/{run_name}-step-{step}",
            "archive_uri": archive_uri,
            "archive_manifest_sha256": archive_manifest_sha256,
            "sfs_path": sfs_path,
            "sfs_available": selected.get("sfs_available") is True,
            "complete": True,
            "promoted": True,
        },
        "selection_excludes": [
            "webexploitbench_results",
            "fleet_test_results",
            "intermediate_dev_loss_ranking",
        ],
    }
    return _with_digest(receipt, "selection_receipt_sha256")


def freeze_final_promoted_sfs_checkpoint(
    run: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    expected_run_name: str,
    expected_run_config_sha256: str,
    expected_rayjob_uid: str,
    expected_trainer_image: str,
    expected_entrypoint_sha256: str,
    expected_pipeline: Mapping[str, Any],
    expected_structural_manifest_before_sha256: str,
) -> dict[str, Any]:
    """Freeze a full-hash SFS checkpoint when the deployed index/archive pipeline was disabled."""

    if run.get("schema") != "fleet_training_run_observation_v1":
        raise ValueError("unsupported run observation schema")
    if _text(run, "run_name") != expected_run_name or _text(run, "status").lower() != "succeeded":
        raise ValueError("training run identity/status does not match the successful planned run")
    if _text(run, "rayjob_uid") != expected_rayjob_uid:
        raise ValueError("RayJob UID differs from the predeclared run")
    if _sha256(run, "run_config_sha256") != expected_run_config_sha256:
        raise ValueError("training run config digest differs from the predeclared run")
    if _digest_pinned_image(run, "trainer_image") != expected_trainer_image:
        raise ValueError("trainer image differs from the observed immutable image")
    if _sha256(run, "entrypoint_sha256") != expected_entrypoint_sha256:
        raise ValueError("RayJob entrypoint digest differs from the predeclared run")
    if observation.get("schema") != "fleet_sft_sfs_checkpoint_observation_v1":
        raise ValueError("unsupported SFS checkpoint observation schema")
    if _text(observation, "run_name") != expected_run_name:
        raise ValueError("SFS checkpoint observation names a different run")
    step = observation.get("step")
    if not isinstance(step, int) or step < 1 or run.get("latest_checkpoint_step") != step:
        raise ValueError("SFS checkpoint is not the run's final step")
    expected_path = f"/mnt/sfs/checkpoints/{expected_run_name}/global_step_{step}"
    if _text(observation, "sfs_path") != expected_path:
        raise ValueError("SFS checkpoint observation has an unexpected path")
    pipeline = _mapping(observation.get("checkpoint_pipeline"), "checkpoint pipeline observation")
    for field in (
        "argocd_application",
        "application_uid",
        "sync_revision",
        "helm_values_sha256",
        "apply",
        "archive_enabled",
        "observed_at",
    ):
        if pipeline.get(field) != expected_pipeline.get(field):
            raise ValueError(f"checkpoint pipeline {field} differs from the frozen plan")
    evidence = {
        "api_checkpoint_rows": observation.get("api_checkpoint_rows"),
        "checkpoint_pipeline": copy.deepcopy(dict(pipeline)),
        "markers": copy.deepcopy(observation.get("markers")),
        "structural_manifest_before_sha256": observation.get("structural_manifest_before_sha256"),
        "structural_manifest_after_sha256": observation.get("structural_manifest_after_sha256"),
        "full_manifest_file_count": observation.get("full_manifest_file_count"),
        "full_manifest_total_bytes": observation.get("full_manifest_total_bytes"),
    }
    if _sha256(observation, "structural_manifest_before_sha256") != (
        expected_structural_manifest_before_sha256
    ):
        raise ValueError("SFS pre-conversion structural manifest differs from the frozen plan")
    receipt = {
        "schema": "cyber_sft_checkpoint_selection_v2",
        "selection_policy": "final_promoted_checkpoint_after_successful_run_v1",
        "run": {
            "name": expected_run_name,
            "terminal_status": "succeeded",
            "rayjob_uid": expected_rayjob_uid,
            "run_config_sha256": expected_run_config_sha256,
            "trainer_image": expected_trainer_image,
            "entrypoint_sha256": expected_entrypoint_sha256,
        },
        "checkpoint": {
            "step": step,
            "uuid": checkpoint_uuid(expected_run_name, step),
            "fleet_model": f"fleet/{expected_run_name}-step-{step}",
            "source_manifest_sha256": _sha256(observation, "full_file_manifest_sha256"),
            "source_manifest_kind": "sfs_sha256_all_files_v1",
            "sfs_path": expected_path,
            "sfs_available": True,
            "complete": True,
            "promoted": True,
            "sfs_evidence": evidence,
        },
        "selection_excludes": [
            "webexploitbench_results",
            "fleet_test_results",
            "intermediate_dev_loss_ranking",
        ],
    }
    signed = _with_digest(receipt, "selection_receipt_sha256")
    validate_selection_receipt(signed)
    return signed


def validate_hf_export_receipt(
    export: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    expected_tokenizer_manifest_sha256: str,
    expected_chat_template_sha256: str,
    expected_config_sha256: str,
    expected_export_binding: Mapping[str, Any],
    expected_runtime_sidecar_sha256: Mapping[str, str] | None = None,
) -> str:
    """Validate a serving-format export without trusting a directory name as identity."""

    validate_selection_receipt(selection)
    if export.get("schema") != "cyber_sft_hf_export_v1":
        raise ValueError("unsupported HF export receipt schema")
    checkpoint = _mapping(selection.get("checkpoint"), "selection.checkpoint")
    source = _mapping(export.get("source_checkpoint"), "export.source_checkpoint")
    if _text(source, "uuid") != _text(checkpoint, "uuid"):
        raise ValueError("HF export came from a different checkpoint UUID")
    if _sha256(source, "source_manifest_sha256") != _selection_source_manifest(selection):
        raise ValueError("HF export source manifest differs from the selected checkpoint")

    output = _mapping(export.get("output"), "export.output")
    if _text(output, "format") != "safetensors":
        raise ValueError("HF export must use safetensors")
    if _text(output, "dtype").lower() not in {"bf16", "bfloat16"}:
        raise ValueError("HF export must preserve BF16 model weights")
    source_path = _text(output, "source_path")
    if source_path != expected_export_binding.get("inference_staging_destination"):
        raise ValueError("HF export source_path differs from the frozen inference destination")
    weights_manifest_sha256 = _sha256(output, "weights_manifest_sha256")
    _sha256(output, "files_manifest_sha256")
    if _sha256(output, "tokenizer_manifest_sha256") != expected_tokenizer_manifest_sha256:
        raise ValueError("HF export tokenizer differs from the base checkpoint")
    if _sha256(output, "chat_template_sha256") != expected_chat_template_sha256:
        raise ValueError("HF export chat template differs from the base checkpoint")
    if _sha256(output, "config_sha256") != expected_config_sha256:
        raise ValueError("HF export model configuration differs from the base checkpoint")
    if expected_runtime_sidecar_sha256 is not None:
        observed_sidecars = _mapping(output.get("sidecar_sha256"), "export.output.sidecar_sha256")
        if dict(observed_sidecars) != dict(expected_runtime_sidecar_sha256):
            raise ValueError("HF export runtime sidecars differ from the base checkpoint")

    conversion = _mapping(export.get("conversion"), "export.conversion")
    export_run, expected_conversion_output = _validate_export_binding(
        expected_export_binding, selection
    )
    conversion_image = _digest_pinned_image(conversion, "image")
    selected_run = _mapping(selection.get("run"), "selection.run")
    if conversion_image != _text(selected_run, "trainer_image"):
        raise ValueError("HF export must use the selected run's exact trainer image")
    if conversion.get("optimizer_steps") != 0:
        raise ValueError("HF export run must execute zero optimizer steps")
    observed_run = _mapping(conversion.get("run"), "export.conversion.run")
    for observed_field, expected_field in (
        ("name", "name"),
        ("run_id", "run_id"),
        ("rayjob_uid", "rayjob_uid"),
        ("trainer_version_id", "trainer_version_id"),
    ):
        if _text(observed_run, observed_field) != _text(export_run, expected_field):
            raise ValueError(f"HF export run {observed_field} differs from the predeclared job")
    if conversion_image != _text(export_run, "trainer_image"):
        raise ValueError("HF export image differs from the predeclared exporter image")
    if _text(conversion, "resume_from") != _text(export_run, "resume_from"):
        raise ValueError("HF export resume path differs from the predeclared source")
    if conversion.get("num_steps") != export_run.get("num_steps"):
        raise ValueError("HF export num_steps differs from the predeclared zero-step run")
    if conversion.get("hf_save_interval") != export_run.get("hf_save_interval"):
        raise ValueError("HF export save interval differs from the predeclared run")
    destination_preflight = _mapping(
        conversion.get("destination_preflight"), "export.conversion.destination_preflight"
    )
    if destination_preflight != expected_export_binding.get("destination_preflight"):
        raise ValueError("HF export destination preflight differs from the frozen collision check")
    _sha256(conversion, "export_request_receipt_sha256")
    _sha256(conversion, "command_sha256")
    if _text(conversion, "output_path") != expected_conversion_output:
        raise ValueError("HF conversion output path differs from the selected checkpoint")

    staging = _mapping(export.get("staging"), "export.staging")
    if _text(staging, "source_path") != expected_conversion_output:
        raise ValueError("inference staging source differs from the conversion output")
    if _text(staging, "destination_path") != source_path:
        raise ValueError("inference staging destination differs from the served source path")
    _digest_pinned_image(staging, "image")
    _sha256(staging, "command_sha256")
    if _sha256(staging, "source_manifest_sha256") != weights_manifest_sha256:
        raise ValueError("staging source manifest differs from the exported weights")
    if _sha256(staging, "destination_manifest_sha256") != weights_manifest_sha256:
        raise ValueError("staging destination manifest differs from the exported weights")
    if staging.get("byte_identical") is not True:
        raise ValueError("inference staging did not prove byte-identical transfer")
    _sha256(staging, "acceptance_manifest_sha256")
    verification = _mapping(export.get("verification"), "export.verification")
    for field in ("all_shards_present", "safetensors_load_passed", "parameter_count_matches"):
        if verification.get(field) is not True:
            raise ValueError(f"HF export verification {field} did not pass")

    return _validate_embedded_digest(export, "export_receipt_sha256")


def derive_post_sft_registration(
    base_registration: Mapping[str, Any],
    selection: Mapping[str, Any],
    export: Mapping[str, Any],
    *,
    expected_tokenizer_manifest_sha256: str,
    expected_chat_template_sha256: str,
    expected_config_sha256: str,
    expected_export_binding: Mapping[str, Any],
    expected_runtime_sidecar_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Clone the baseline serving contract, changing only checkpoint identity and paths."""

    validate_hf_export_receipt(
        export,
        selection,
        expected_tokenizer_manifest_sha256=expected_tokenizer_manifest_sha256,
        expected_chat_template_sha256=expected_chat_template_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_export_binding=expected_export_binding,
        expected_runtime_sidecar_sha256=expected_runtime_sidecar_sha256,
    )
    base = copy.deepcopy(dict(base_registration))
    spec = _mapping(base.get("spec"), "base registration spec")
    model = _mapping(spec.get("model"), "base registration spec.model")
    _mapping(spec.get("runtime"), "base registration spec.runtime")
    output = _mapping(export.get("output"), "export.output")
    checkpoint = _mapping(selection.get("checkpoint"), "selection.checkpoint")

    base_id = _text(base, "id")
    source_path = _text(model, "sourcePath")
    serving_path = _text(model, "path")
    new_source_path = _text(output, "source_path")
    new_serving_path = "/scratch/models/" + new_source_path.removeprefix("/models/")
    new_id = _text(checkpoint, "fleet_model").removeprefix("fleet/")
    if not MODEL_ID_RE.fullmatch(new_id):
        raise ValueError("checkpoint model name is not a valid inference model id")

    result = copy.deepcopy(base)
    result["id"] = new_id
    result_spec = result["spec"]
    result_spec["displayName"] = f"Qwen3.6 27B cyber SFT {new_id}"
    result_spec["model"]["sourcePath"] = new_source_path
    result_spec["model"]["path"] = new_serving_path
    result_spec["model"]["revision"] = _text(checkpoint, "uuid")

    replacements = {source_path: new_source_path, serving_path: new_serving_path, base_id: new_id}
    result_args: list[str] = []
    for arg in result_spec["runtime"]["args"]:
        replaced = str(arg)
        for old, new in replacements.items():
            replaced = replaced.replace(old, new)
        result_args.append(replaced)
    result_spec["runtime"]["args"] = result_args

    if result_spec["model"].get("precision") != "bf16":
        raise ValueError("post-SFT serving must preserve the baseline BF16 precision")
    if result_spec["runtime"].get("engine") != "sglang":
        raise ValueError("post-SFT serving must preserve the baseline SGLang engine")
    if result_spec["runtime"].get("image") != spec["runtime"].get("image"):
        raise ValueError("post-SFT serving image differs from the baseline")

    receipt = {
        "schema": "cyber_post_sft_serving_registration_v1",
        "registration": result,
        "base_registration_sha256": digest_json(base),
        "selection_receipt_sha256": _sha256(selection, "selection_receipt_sha256"),
        "export_receipt_sha256": _sha256(export, "export_receipt_sha256"),
        "allowed_differences": [
            "id",
            "display_name",
            "model_source_path",
            "model_serving_path",
            "model_revision",
            "served_model_name",
        ],
    }
    return _with_digest(receipt, "serving_receipt_sha256")


def derive_webexploit_config(
    base_config: Mapping[str, Any], *, served_model_id: str, run_id: str
) -> dict[str, Any]:
    """Create a paired post-SFT WebExploitBench config.

    The model id and run id are the only permitted changes.  In particular this preserves the
    exact Qwen Code 0.22.3 harness used by the canonical 10/110 baseline, plus its judge, prompt
    level, context windows, timeouts, pass@k, and concurrency.
    """

    if not MODEL_ID_RE.fullmatch(served_model_id):
        raise ValueError("served_model_id is not a valid inference model id")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", run_id):
        raise ValueError("run_id is not a valid CAGE run id")
    result = copy.deepcopy(dict(base_config))
    result["model"] = served_model_id
    result["run_id"] = run_id
    changed = {key for key in base_config if base_config.get(key) != result.get(key)}
    if changed != {"model", "run_id"}:
        raise ValueError("WebExploitBench paired config changed uncontrolled fields")
    if result.get("agent") != "qwen_code" or result.get("agent_version") != "0.22.3":
        raise ValueError("paired WebExploitBench evaluation must preserve Qwen Code 0.22.3")
    return result


def build_fleet_test_holdout_receipt(
    split_manifest: Mapping[str, Any], sft_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the exact 20 untouched Fleet task versions without copying prompt content."""

    if split_manifest.get("schema") != "fleet_rl_task_split_v1":
        raise ValueError("unsupported Fleet split manifest schema")
    tasks = split_manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Fleet split manifest tasks must be an array")
    test_rows = [row for row in tasks if isinstance(row, Mapping) and row.get("split") == "test"]
    expected_test = (
        _mapping(split_manifest.get("counts"), "split counts").get("splits", {}).get("test")
    )
    if len(test_rows) != expected_test or len(test_rows) != 20:
        raise ValueError("Fleet test split must contain exactly the declared 20 tasks")

    train = set(_mapping(sft_config.get("data"), "sft data").get("task_keys", []))
    dev = set(_mapping(sft_config.get("eval"), "sft eval").get("task_keys", []))
    test = {row.get("task_key") for row in test_rows}
    if None in test or len(test) != 20:
        raise ValueError("Fleet test split task keys must be unique and non-empty")
    if train & dev or train & test or dev & test:
        raise ValueError("Fleet train, development, and test task lineages overlap")

    bindings = []
    required = (
        "task_key",
        "task_version_id",
        "task_version",
        "environment_version_id",
        "env_key",
        "env_version",
        "data_key",
        "data_version",
    )
    for row in sorted(test_rows, key=lambda item: str(item["task_key"])):
        binding = {field: row.get(field) for field in required}
        if any(value is None or value == "" for value in binding.values()):
            raise ValueError(f"Fleet test binding for {row.get('task_key')} is incomplete")
        bindings.append(binding)
    if len({row["task_version_id"] for row in bindings}) != len(bindings):
        raise ValueError("Fleet test task-version IDs must be unique")

    receipt = {
        "schema": "cyber_fleet_test_holdout_v1",
        "source": copy.deepcopy(split_manifest.get("source")),
        "split_policy": copy.deepcopy(split_manifest.get("policy")),
        "task_count": len(bindings),
        "tasks": bindings,
        "training_overlap": 0,
        "development_overlap": 0,
        "evaluation_policy": {
            "task_group_members_must_pin_eval_task_version_id": True,
            "resolve_current_task_version": False,
            "pass_k": 1,
            "primary_harness": "claude_code",
        },
    }
    return _with_digest(receipt, "holdout_receipt_sha256")


def build_post_sft_comparison_receipt(
    *,
    selection: Mapping[str, Any],
    export: Mapping[str, Any],
    serving: Mapping[str, Any],
    base_webexploit_config_sha256: str,
    post_webexploit_config_sha256: str,
    webexploit_paired_identity: Mapping[str, Any],
    fleet_holdout: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the immutable handoff shared by both paired evaluation suites."""

    for field, value in {
        "base_webexploit_config_sha256": base_webexploit_config_sha256,
        "post_webexploit_config_sha256": post_webexploit_config_sha256,
    }.items():
        if not SHA256_RE.fullmatch(value):
            raise ValueError(f"{field} must be sha256:<64 hex>")
    if webexploit_paired_identity.get("schema") != (
        "webexploitbench_qwen_code_paired_identity_v1"
    ):
        raise ValueError("unsupported WebExploitBench paired identity schema")
    paired_identity_sha256 = _sha256(
        webexploit_paired_identity, "paired_identity_receipt_sha256"
    )
    paired_undigested = {
        key: value
        for key, value in webexploit_paired_identity.items()
        if key != "paired_identity_receipt_sha256"
    }
    if digest_json(paired_undigested) != paired_identity_sha256:
        raise ValueError("WebExploitBench paired identity receipt digest does not validate")
    receipt = {
        "schema": "cyber_post_sft_comparison_v1",
        "checkpoint_selection_sha256": _sha256(selection, "selection_receipt_sha256"),
        "hf_export_sha256": _sha256(export, "export_receipt_sha256"),
        "serving_registration_sha256": _sha256(serving, "serving_receipt_sha256"),
        "webexploitbench": {
            "base_config_sha256": base_webexploit_config_sha256,
            "post_config_sha256": post_webexploit_config_sha256,
            "allowed_config_differences": ["model", "run_id"],
            "paired_identity_receipt_sha256": paired_identity_sha256,
            "harness": "qwen_code",
            "harness_version": "0.22.3",
            "external_results_used_for_selection": False,
        },
        "fleet": {
            "holdout_receipt_sha256": _sha256(fleet_holdout, "holdout_receipt_sha256"),
            "exact_task_versions": fleet_holdout.get("task_count"),
            "training_overlap": fleet_holdout.get("training_overlap"),
            "development_overlap": fleet_holdout.get("development_overlap"),
        },
        "allowed_model_difference": "checkpoint_weights_only",
    }
    return _with_digest(receipt, "comparison_receipt_sha256")
