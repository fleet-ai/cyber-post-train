"""Fail-closed receipts for selecting and evaluating an SFT checkpoint.

This module deliberately does not call Fleet, Kubernetes, or an inference endpoint.  Operators
capture those systems' read-only responses and pass them here.  That keeps checkpoint selection
independent of benchmark results and makes every mutating step (export, registration, evaluation)
conditional on a reviewable immutable receipt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from .io import digest_json

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
IMAGE_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
MODEL_ID_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
CHECKPOINT_NAMESPACE = uuid.UUID("bc7f7c3f-38ba-5835-840d-ecb75c2d2f64")
STAGING_CODE_PATHS = {
    "training/__init__.py",
    "training/io.py",
    "training/post_sft_artifacts.py",
    "training/post_sft_staging.py",
}
STAGING_ACCEPTANCE_RECEIPT = ".fleet-acceptance.json"
STORED_CONFIG_NORMALIZATION_SCHEMA = "fleet_training_stored_config_normalization_v1"


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


def _canonical_pretty_json_sha256(value: Mapping[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _exact_sha256_map(value: Any, field: str, expected_paths: set[str]) -> dict[str, str]:
    mapping = _mapping(value, field)
    if set(mapping) != expected_paths:
        raise ValueError(f"{field} paths differ from the exact expected set")
    result: dict[str, str] = {}
    for path, digest in mapping.items():
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"{field} has an invalid SHA-256 for {path}")
        result[str(path)] = digest
    return result


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


def _validate_stored_config_normalization(
    export_run: Mapping[str, Any],
) -> Mapping[str, Any]:
    normalization = _mapping(
        export_run.get("server_normalization"), "export run server_normalization"
    )
    if normalization.get("schema") != STORED_CONFIG_NORMALIZATION_SCHEMA:
        raise ValueError("unsupported stored-config normalization schema")
    if _text(normalization, "node_pool") != "fleetai-training-ng-gpu":
        raise ValueError("stored-config node pool differs from the immutable export job")
    _text(normalization, "submitted_by_email")
    data_defaults = _mapping(
        normalization.get("data_defaults"), "stored-config data defaults"
    )
    expected_data_defaults = {
        "env_keys": None,
        "models": None,
        "session_ids": None,
        "since": None,
        "team_ids": ["a1025f0b-ad67-49fc-a023-51800ab43e84"],
        "until": None,
    }
    if dict(data_defaults) != expected_data_defaults:
        raise ValueError("stored-config data normalization differs from the reviewed Fleet job")
    trainer_defaults = _mapping(
        normalization.get("trainer_defaults"), "stored-config trainer defaults"
    )
    if dict(trainer_defaults) != {"command": None, "env": {}}:
        raise ValueError("stored-config trainer normalization differs from the reviewed Fleet job")
    return normalization


def normalize_zero_step_stored_config(
    request: Mapping[str, Any], export_run: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply only the exact server-owned normalization observed on the immutable export job."""

    normalization = _validate_stored_config_normalization(export_run)
    result = copy.deepcopy(dict(request))
    if result.get("title") != _text(export_run, "title"):
        raise ValueError("zero-step export title differs from the immutable submitted job")
    if any(field in result for field in ("name", "run_id", "submitted_by_email", "node_pool")):
        raise ValueError("client export request contains server-owned identity fields")
    result.update(
        {
            "name": _text(export_run, "name"),
            "run_id": _text(export_run, "run_id"),
            "submitted_by_email": _text(normalization, "submitted_by_email"),
            "node_pool": _text(normalization, "node_pool"),
        }
    )
    data = copy.deepcopy(dict(_mapping(result.get("data"), "zero-step request data")))
    for field, value in _mapping(
        normalization.get("data_defaults"), "stored-config data defaults"
    ).items():
        if field in data and data[field] != value:
            raise ValueError(
                f"client export data field {field} conflicts with server normalization"
            )
        data[field] = copy.deepcopy(value)
    result["data"] = data
    trainer = copy.deepcopy(dict(_mapping(result.get("trainer"), "zero-step request trainer")))
    for field, value in _mapping(
        normalization.get("trainer_defaults"), "stored-config trainer defaults"
    ).items():
        if field in trainer and trainer[field] != value:
            raise ValueError(
                f"client export trainer field {field} conflicts with server normalization"
            )
        trainer[field] = copy.deepcopy(value)
    result["trainer"] = trainer
    return result


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


def render_zero_step_sft_command(request: Mapping[str, Any], run_name: str) -> str:
    """Render the exact Fleet Train SFT entrypoint owned by the frozen request.

    This intentionally mirrors the deployed SFT command contract for the one pinned SkyRL
    exporter.  The resulting digest is frozen before evidence collection and compared with the
    RayJob's actual ``spec.entrypoint``; an arbitrary syntactically valid command is never enough.
    """

    if request.get("kind") != "sft":
        raise ValueError("zero-step export request kind must be sft")
    model = _mapping(request.get("model"), "zero-step request model")
    if _text(model, "precision").lower() != "bf16":
        raise ValueError("zero-step export request model precision must be exactly bf16")
    staged_model = _text(model, "staged_model")
    sft = _mapping(request.get("sft"), "zero-step request sft")
    if _text(sft, "strategy") != "fsdp":
        raise ValueError("zero-step export request strategy must be fsdp")
    objective = _mapping(request.get("objective"), "zero-step request objective")
    trainer = _mapping(request.get("trainer"), "zero-step request trainer")
    trainer_args = trainer.get("args")
    if not isinstance(trainer_args, list) or not all(
        isinstance(value, str) and value for value in trainer_args
    ):
        raise ValueError("zero-step export trainer args must be non-empty strings")
    wandb = _mapping(request.get("wandb"), "zero-step request wandb")
    evaluation = _mapping(request.get("eval"), "zero-step request eval")
    if evaluation.get("task_keys") != []:
        raise ValueError("zero-step export command cannot include evaluation tasks")

    overrides: dict[str, Any] = {
        "strategy": sft.get("strategy"),
        "model.path": f"/mnt/sfs/models/{staged_model}",
        "max_length": objective.get("max_length"),
        "train_on_what": objective.get("train_on_what"),
        "batch_size": sft.get("batch_size"),
        "micro_train_batch_size_per_gpu": sft.get("micro_train_batch_size_per_gpu"),
        "optimizer_config.lr": sft.get("learning_rate"),
        "ckpt_path": f"/mnt/sfs/checkpoints/{run_name}",
        "ckpt_interval": sft.get("checkpoint_interval"),
        "max_ckpts_to_keep": sft.get("max_checkpoints_to_keep"),
        "logger": "wandb",
        "project_name": wandb.get("project"),
        "run_name": run_name,
        "placement.num_nodes": request.get("num_workers"),
        "placement.num_gpus_per_node": request.get("gpus_per_worker"),
        "num_steps": sft.get("max_steps"),
        "eval_before_train": False,
        "eval_interval": 0,
    }
    if sft.get("sequence_parallel_size") is not None:
        overrides["sequence_parallel_size"] = sft.get("sequence_parallel_size")

    parts = ["python", "-m", "rl_rollout.sft_entrypoint"]
    for key, value in sorted(overrides.items()):
        if value is None:
            raise ValueError(f"zero-step export request is missing command field {key}")
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        if any(character in rendered for character in " \t'\"$`\\;|&<>()"):
            raise ValueError(f"zero-step export command field {key} contains shell metacharacters")
        parts.append(f"{key}={rendered}")
    parts.extend(trainer_args)
    return " ".join(parts)


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
    _text(export_run, "title")
    _validate_stored_config_normalization(export_run)
    request.pop("name", None)
    request["title"] = _text(export_run, "title")
    request["sft"]["num_epochs"] = None
    request["sft"]["max_steps"] = step
    request["eval"] = {"task_keys": [], "interval": 1, "before_train": False}
    request["trainer"]["args"] = [
        *args,
        f"resume_from={source_path}",
        f"hf_save_interval={step + 1}",
        f"export_path={export_root}",
    ]
    expected_command = render_zero_step_sft_command(request, _text(export_run, "name"))
    stored_config = normalize_zero_step_stored_config(request, export_run)
    expected_execution = {
        "request_config_sha256": digest_json(request),
        "stored_config_sha256": digest_json(stored_config),
        "kind": "sft",
        "model_precision": "bf16",
        "strategy": "fsdp",
        "trainer_version_id": expected_trainer_version_id,
        "command_sha256": digest_json({"entrypoint": expected_command}),
    }

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
        "expected_execution": expected_execution,
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

    correction = _mapping(export.get("precision_correction"), "export.precision_correction")
    expected_cast_output = _text(expected_export_binding, "bf16_cast_destination")
    if (
        correction.get("schema") != "cyber_sft_fp32_to_bf16_precision_correction_v1"
        or _text(correction, "source_path") != expected_conversion_output
        or _text(correction, "destination_path") != expected_cast_output
        or _text(correction, "source_dtype") != "F32"
        or _text(correction, "destination_dtype") != "BF16"
        or correction.get("policy") != "deterministic_sorted_tensor_fp32_to_bf16_v1"
        or correction.get("exact_cast_bits_verified") is not True
    ):
        raise ValueError("HF export precision correction is incomplete or inconsistent")
    if _sha256(correction, "destination_weights_manifest_sha256") != weights_manifest_sha256:
        raise ValueError("precision correction output differs from final exported weights")
    for field in (
        "source_weights_manifest_sha256",
        "cast_rows_sha256",
        "source_layout_sha256",
        "cast_receipt_sha256",
        "cast_full_manifest_sha256",
    ):
        _sha256(correction, field)

    staging = _mapping(export.get("staging"), "export.staging")
    if _text(staging, "source_path") != expected_cast_output:
        raise ValueError("inference staging source differs from the verified BF16 cast")
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


def assemble_hf_export_receipt(
    selection: Mapping[str, Any],
    export_request: Mapping[str, Any],
    export_run_observation: Mapping[str, Any],
    export_observation: Mapping[str, Any],
    cast_receipt: Mapping[str, Any],
    cast_full_manifest: Mapping[str, Any],
    stage_input: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    *,
    expected_tokenizer_manifest_sha256: str,
    expected_chat_template_sha256: str,
    expected_config_sha256: str,
    expected_export_binding: Mapping[str, Any],
    expected_runtime_sidecar_sha256: Mapping[str, str],
    expected_tokenizer_equivalence_evidence_sha256: str,
    expected_cast_execution: Mapping[str, Any],
    expected_staging_image: str,
    expected_staging_command_sha256: str,
) -> dict[str, Any]:
    """Assemble the final export receipt from independently captured evidence.

    This function performs no filesystem, Kubernetes, or Fleet API reads. Each input is an
    immutable, digest-bound observation produced by an earlier gate. The function deliberately
    reconstructs the public receipt instead of accepting an operator-authored aggregate.
    """

    validate_selection_receipt(selection)
    checkpoint = _mapping(selection.get("checkpoint"), "selection.checkpoint")
    selected_run = _mapping(selection.get("run"), "selection.run")
    export_run, expected_raw_path = _validate_export_binding(expected_export_binding, selection)
    expected_destination = _text(expected_export_binding, "inference_staging_destination")

    if export_request.get("schema") != "cyber_sft_zero_step_export_request_v1":
        raise ValueError("unsupported zero-step export request receipt schema")
    request_digest = _validate_embedded_digest(export_request, "export_request_receipt_sha256")
    if _sha256(export_request, "source_selection_sha256") != _sha256(
        selection, "selection_receipt_sha256"
    ):
        raise ValueError("export request names a different checkpoint selection")
    request_source = _mapping(
        export_request.get("source_checkpoint"), "export request source_checkpoint"
    )
    if _text(request_source, "uuid") != _text(checkpoint, "uuid"):
        raise ValueError("export request names a different checkpoint UUID")
    if _sha256(request_source, "source_manifest_sha256") != _selection_source_manifest(selection):
        raise ValueError("export request source manifest differs from the selection")
    request_output = _mapping(
        export_request.get("expected_output"), "export request expected_output"
    )
    if _text(request_output, "path") != expected_raw_path:
        raise ValueError("export request expected output path differs from the frozen plan")
    if _text(request_output, "format") != "huggingface_safetensors":
        raise ValueError("export request must expect Hugging Face safetensors")
    if _text(request_output, "dtype").lower() not in {"bf16", "bfloat16"}:
        raise ValueError("export request must preserve BF16 weights")
    if _mapping(export_request.get("export_run"), "export request export_run") != export_run:
        raise ValueError("export request run identity differs from the frozen plan")
    if export_request.get("destination_preflight") != expected_export_binding.get(
        "destination_preflight"
    ):
        raise ValueError("export request destination preflight differs from the frozen plan")
    if export_request.get("submit") is not False:
        raise ValueError("export request receipt must remain a non-submitting review artifact")
    rendered_request = _mapping(export_request.get("request"), "export request request")
    expected_execution = _mapping(
        export_request.get("expected_execution"), "export request expected_execution"
    )
    if expected_execution != {
        "request_config_sha256": digest_json(rendered_request),
        "stored_config_sha256": digest_json(
            normalize_zero_step_stored_config(rendered_request, export_run)
        ),
        "kind": "sft",
        "model_precision": "bf16",
        "strategy": "fsdp",
        "trainer_version_id": _text(export_run, "trainer_version_id"),
        "command_sha256": digest_json(
            {
                "entrypoint": render_zero_step_sft_command(
                    rendered_request, _text(export_run, "name")
                )
            }
        ),
    }:
        raise ValueError("export request expected execution identity is not exact")
    if rendered_request.get("kind") != expected_execution["kind"]:
        raise ValueError("export request kind differs from expected execution")
    rendered_model = _mapping(rendered_request.get("model"), "export request request.model")
    if _text(rendered_model, "precision").lower() != expected_execution["model_precision"]:
        raise ValueError("export request model precision differs from expected execution")
    rendered_sft = _mapping(rendered_request.get("sft"), "export request request.sft")
    if _text(rendered_sft, "strategy") != expected_execution["strategy"]:
        raise ValueError("export request strategy differs from expected execution")
    if (
        rendered_sft.get("max_steps") != checkpoint.get("step")
        or rendered_sft.get("num_epochs") is not None
    ):
        raise ValueError("export request does not encode the selected zero-step boundary")
    rendered_eval = _mapping(rendered_request.get("eval"), "export request request.eval")
    if rendered_eval.get("task_keys") != []:
        raise ValueError("zero-step export request must not run evaluations")
    rendered_trainer = _mapping(rendered_request.get("trainer"), "export request request.trainer")
    if _text(rendered_trainer, "trainer_version_id") != expected_execution[
        "trainer_version_id"
    ]:
        raise ValueError("export request trainer version differs from expected execution")
    args = rendered_trainer.get("args")
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("zero-step export trainer args must be an array of strings")
    owned_args = {
        "resume_from": _text(export_run, "resume_from"),
        "hf_save_interval": str(export_run.get("hf_save_interval")),
        "export_path": _text(expected_export_binding, "output_root"),
    }
    for key, expected in owned_args.items():
        matches = [item.split("=", 1)[1] for item in args if item.startswith(key + "=")]
        if matches != [expected]:
            raise ValueError(f"zero-step export request has an invalid {key} argument")

    if export_run_observation.get("schema") != "cyber_sft_zero_step_export_run_observation_v1":
        raise ValueError("unsupported zero-step export run observation schema")
    _validate_embedded_digest(export_run_observation, "observation_sha256")
    if _text(export_run_observation, "terminal_status") != "SUCCEEDED":
        raise ValueError("zero-step export run is not terminally SUCCEEDED")
    observed_run = _mapping(export_run_observation.get("run"), "export run observation run")
    for field in ("name", "run_id", "rayjob_uid", "trainer_version_id"):
        if _text(observed_run, field) != _text(export_run, field):
            raise ValueError(f"zero-step export run {field} differs from the frozen plan")
    if _digest_pinned_image(export_run_observation, "trainer_image") != _text(
        export_run, "trainer_image"
    ):
        raise ValueError("zero-step export trainer image differs from the frozen plan")
    for field in ("resume_from", "num_steps", "hf_save_interval"):
        if export_run_observation.get(field) != export_run.get(field):
            raise ValueError(f"zero-step export {field} differs from the frozen plan")
    if export_run_observation.get("optimizer_steps") != 0:
        raise ValueError("zero-step export run executed optimizer steps")
    if _sha256(export_run_observation, "export_request_receipt_sha256") != request_digest:
        raise ValueError("export run observation names a different request receipt")
    command_sha256 = _sha256(export_run_observation, "command_sha256")
    if command_sha256 != expected_execution["command_sha256"]:
        raise ValueError("zero-step export command differs from the exact expected command")
    request_identity = _mapping(
        export_run_observation.get("request_identity"), "export run request identity"
    )
    if dict(request_identity) != dict(expected_execution):
        raise ValueError("export run request identity differs from the exact request")
    rayjob_evidence = _mapping(
        export_run_observation.get("rayjob"), "export run RayJob evidence"
    )
    if (
        _text(rayjob_evidence, "uid") != _text(export_run, "rayjob_uid")
        or _text(rayjob_evidence, "entrypoint")
        != render_zero_step_sft_command(rendered_request, _text(export_run, "name"))
        or _sha256(rayjob_evidence, "entrypoint_sha256") != command_sha256
    ):
        raise ValueError("export run RayJob evidence differs from the exact execution")
    _sha256(rayjob_evidence, "spec_sha256")
    if _text(rayjob_evidence, "queue") != "training-lq":
        raise ValueError("export run RayJob used an unexpected queue")
    execution_projection = _mapping(
        rayjob_evidence.get("reviewed_execution_projection"),
        "export run reviewed RayJob execution projection",
    )
    if _sha256(rayjob_evidence, "reviewed_execution_projection_sha256") != digest_json(
        execution_projection
    ):
        raise ValueError("export run reviewed RayJob projection digest differs")
    projected_head = _mapping(execution_projection.get("head"), "projected Ray head")
    projected_worker = _mapping(execution_projection.get("worker"), "projected Ray worker")
    projected_submitter = _mapping(
        execution_projection.get("submitter"), "projected Ray submitter"
    )
    if (
        projected_head.get("service_account_name") != "default"
        or _text(projected_head, "image") != _text(export_run, "trainer_image")
        or projected_head.get("command") is not None
        or projected_head.get("args") is not None
        or projected_head.get("resources")
        != {
            "requests": {"cpu": "2", "memory": "8Gi"},
            "limits": {"cpu": "4", "memory": "16Gi"},
        }
    ):
        raise ValueError("export run Ray head projection differs from reviewed policy")
    if (
        projected_worker.get("group_name") != "gpu-worker"
        or projected_worker.get("replicas") != rendered_request.get("num_workers")
        or projected_worker.get("min_replicas") != rendered_request.get("num_workers")
        or projected_worker.get("max_replicas") != rendered_request.get("num_workers")
        or projected_worker.get("service_account_name") != "default"
        or _text(projected_worker, "image") != _text(export_run, "trainer_image")
        or projected_worker.get("command") is not None
        or projected_worker.get("args") is not None
        or projected_worker.get("resources")
        != {
            "requests": {"cpu": "184", "memory": "2560Gi", "nvidia.com/gpu": "8"},
            "limits": {"cpu": "184", "memory": "2560Gi", "nvidia.com/gpu": "8"},
        }
    ):
        raise ValueError("export run Ray worker projection differs from reviewed policy")
    if (
        projected_submitter.get("service_account_name") != "default"
        or _text(projected_submitter, "image") != "anyscale/ray:2.56.0-slim-py312"
        or projected_submitter.get("command") is not None
        or projected_submitter.get("args") is not None
        or projected_submitter.get("resources")
        != {
            "requests": {"cpu": "200m", "ephemeral-storage": "2Gi", "memory": "512Mi"},
            "limits": {"cpu": "1", "memory": "1Gi"},
        }
    ):
        raise ValueError("export run Ray submitter projection differs from reviewed policy")
    raycluster_evidence = _mapping(
        export_run_observation.get("raycluster"), "export run RayCluster evidence"
    )
    if _text(raycluster_evidence, "name") != _text(rayjob_evidence, "cluster_name"):
        raise ValueError("export run RayCluster identity differs from the RayJob")
    _text(raycluster_evidence, "uid")
    _sha256(raycluster_evidence, "spec_sha256")
    pod_evidence = _mapping(export_run_observation.get("pod"), "export run Pod evidence")
    _text(pod_evidence, "name")
    _text(pod_evidence, "uid")
    if _text(pod_evidence, "requested_image") != _text(export_run, "trainer_image"):
        raise ValueError("export run Pod resolved image differs from the trainer image")
    expected_image_digest = _text(export_run, "trainer_image").rsplit("@", 1)[-1]
    if _text(pod_evidence, "resolved_image_digest") != expected_image_digest:
        raise ValueError("export run Pod resolved image digest differs from the trainer image")
    _text(pod_evidence, "resolved_image_id")
    submitter_evidence = _mapping(
        export_run_observation.get("submitter"), "export run submitter evidence"
    )
    if (
        _text(submitter_evidence, "job_name") != _text(export_run, "name")
        or _text(submitter_evidence, "container_name") != "ray-job-submitter"
        or _text(submitter_evidence, "image") != "anyscale/ray:2.56.0-slim-py312"
    ):
        raise ValueError("export run submitter evidence differs from reviewed policy")
    for field in ("job_uid", "pod_name", "pod_uid"):
        _text(submitter_evidence, field)
    _sha256(submitter_evidence, "job_spec_sha256")
    _sha256(submitter_evidence, "command_sha256")
    zero_step_evidence = _mapping(
        export_run_observation.get("zero_step_evidence"), "zero-step execution evidence"
    )
    if (
        zero_step_evidence.get("resume_global_step") != checkpoint.get("step")
        or zero_step_evidence.get("configured_final_step") != checkpoint.get("step")
        or zero_step_evidence.get("optimizer_step_events") != 0
    ):
        raise ValueError("export run immutable evidence does not prove zero optimizer steps")
    logs = _mapping(zero_step_evidence.get("logs"), "zero-step log evidence")
    metrics = _mapping(zero_step_evidence.get("metrics"), "zero-step metric evidence")
    _sha256(logs, "sha256")
    _sha256(metrics, "api_observation_sha256")
    if (
        logs.get("contains_exact_entrypoint") is not True
        or logs.get("contains_terminal_success") is not True
        or not isinstance(logs.get("bytes"), int)
        or logs["bytes"] < 1
        or logs.get("optimizer_events") != 0
        or logs.get("steps_after_resume") != []
        or not isinstance(logs.get("observed_steps"), list)
        or any(
            not isinstance(step, int) or step > checkpoint.get("step")
            for step in logs.get("observed_steps", [])
        )
        or not isinstance(metrics.get("checked_step_fields"), Mapping)
        or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in metrics.get("checked_step_fields", {}).values()
        )
        or metrics.get("reported_optimizer_steps") != 0
    ):
        raise ValueError("export run log/metric evidence does not prove zero optimizer steps")
    if _text(export_run_observation, "output_path") != expected_raw_path:
        raise ValueError("zero-step export run observed an unexpected output path")

    if export_observation.get("schema") != "fleet_sft_sfs_checkpoint_observation_v1":
        raise ValueError("unsupported post-export filesystem observation schema")
    export_observation_sha256 = _validate_embedded_digest(export_observation, "observation_sha256")
    if _text(export_observation, "run_name") != _text(selected_run, "name"):
        raise ValueError("post-export observation names a different source run")
    if export_observation.get("step") != checkpoint.get("step"):
        raise ValueError("post-export observation names a different checkpoint step")
    if _text(export_observation, "sfs_path") != _text(checkpoint, "sfs_path"):
        raise ValueError("post-export observation names a different source checkpoint path")
    if _sha256(export_observation, "full_file_manifest_sha256") != (
        _selection_source_manifest(selection)
    ):
        raise ValueError("post-export source manifest differs from the selected checkpoint")
    raw_inspection = _mapping(
        export_observation.get("output_inspection"), "post-export output inspection"
    )
    if _text(raw_inspection, "root") != expected_raw_path:
        raise ValueError("post-export inspection names an unexpected raw export path")
    if _text(raw_inspection, "format") != "safetensors":
        raise ValueError("raw export inspection did not verify safetensors")
    if _text(raw_inspection, "dtype").lower() not in {"f32", "float32"}:
        raise ValueError("raw export inspection did not verify the observed FP32 weights")
    for field in ("all_shards_present", "safetensors_load_passed", "parameter_count_matches"):
        if raw_inspection.get(field) is not True:
            raise ValueError(f"raw export verification {field} did not pass")
    raw_weights_sha256 = _sha256(raw_inspection, "weights_manifest_sha256")
    raw_files_sha256 = _sha256(raw_inspection, "files_manifest_sha256")
    if _sha256(export_observation, "raw_export_full_manifest_sha256") != raw_files_sha256:
        raise ValueError("post-export observation full manifest differs from its inspection")
    raw_layout = _mapping(
        export_observation.get("weight_layout_equivalence"), "raw weight layout evidence"
    )
    if (
        raw_layout.get("schema") != "cyber_sft_safetensors_layout_equivalence_v2"
        or raw_layout.get("all_keys_and_shapes_match") is not True
        or raw_layout.get("dtype_match_required") is not False
        or raw_layout.get("shape_mismatch_count") != 0
        or raw_layout.get("missing_key_count") != 0
        or raw_layout.get("unexpected_key_count") != 0
        or raw_layout.get("dtype_mismatch_count") != raw_layout.get("tensor_count")
    ):
        raise ValueError("raw FP32 export layout does not match the frozen BF16 architecture")
    _sha256(raw_layout, "base_layout_sha256")
    raw_layout_sha256 = _sha256(raw_layout, "candidate_layout_sha256")
    architecture = _mapping(
        export_observation.get("model_config_architecture_equivalence"),
        "raw model architecture evidence",
    )
    if architecture.get("all_architecture_and_vocab_fields_identical") is not True:
        raise ValueError("raw export model architecture differs from the frozen base")
    _sha256(architecture, "normalized_architecture_sha256")

    if cast_receipt.get("schema") != "cyber_sft_fp32_to_bf16_cast_receipt_v1":
        raise ValueError("unsupported FP32-to-BF16 cast receipt schema")
    cast_receipt_sha256 = _validate_embedded_digest(cast_receipt, "cast_receipt_sha256")
    cast_source = _mapping(cast_receipt.get("source"), "cast receipt source")
    if (
        _text(cast_source, "path") != expected_raw_path
        or _sha256(cast_source, "observation_sha256") != export_observation_sha256
        or _sha256(cast_source, "checkpoint_full_manifest_sha256")
        != _selection_source_manifest(selection)
        or _sha256(cast_source, "raw_full_manifest_sha256") != raw_files_sha256
        or _sha256(cast_source, "raw_full_manifest_after_sha256") != raw_files_sha256
        or cast_source.get("raw_source_stable_during_cast") is not True
        or _sha256(cast_source, "raw_weights_manifest_sha256") != raw_weights_sha256
        or _text(cast_source, "dtype") != "F32"
    ):
        raise ValueError("FP32-to-BF16 cast source differs from immutable export evidence")
    cast_conversion = _mapping(cast_receipt.get("conversion"), "cast conversion proof")
    if (
        cast_conversion.get("schema") != "cyber_sft_fp32_to_bf16_cast_proof_v1"
        or cast_conversion.get("policy") != "deterministic_sorted_tensor_fp32_to_bf16_v1"
        or cast_conversion.get("source_dtype") != "F32"
        or cast_conversion.get("destination_dtype") != "BF16"
        or cast_conversion.get("all_source_values_finite") is not True
        or cast_conversion.get("all_destination_bits_equal_direct_bf16_cast") is not True
        or cast_conversion.get("parameter_count") != raw_inspection.get("parameter_count")
    ):
        raise ValueError("FP32-to-BF16 cast proof is incomplete or inconsistent")
    if _sha256(cast_conversion, "source_layout_sha256") != raw_layout_sha256:
        raise ValueError("cast source layout differs from the independently inspected raw export")
    cast_rows_sha256 = _sha256(cast_conversion, "cast_rows_sha256")
    cast_rows = cast_conversion.get("cast_rows")
    if not isinstance(cast_rows, list) or digest_json(cast_rows) != cast_rows_sha256:
        raise ValueError("FP32-to-BF16 per-tensor cast proof digest does not validate")
    if (
        len(cast_rows) != cast_conversion.get("tensor_count")
        or sum(row.get("elements", 0) for row in cast_rows if isinstance(row, Mapping))
        != cast_conversion.get("parameter_count")
        or any(
            not isinstance(row, Mapping)
            or row.get("source_dtype") != "F32"
            or row.get("destination_dtype") != "BF16"
            or row.get("exact_cast_bits_verified") is not True
            or not isinstance(row.get("key"), str)
            or not isinstance(row.get("shape"), list)
            or not row.get("shape")
            or any(not isinstance(size, int) or size < 1 for size in row.get("shape", []))
            or not isinstance(row.get("source_shard"), str)
            or not isinstance(row.get("destination_shard"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("source_tensor_sha256")))
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(row.get("destination_tensor_sha256"))
            )
            for row in cast_rows
        )
    ):
        raise ValueError("FP32-to-BF16 per-tensor cast proof is incomplete")
    if _mapping(cast_receipt.get("execution_plan"), "cast execution plan") != (
        expected_cast_execution
    ):
        raise ValueError("FP32-to-BF16 cast execution plan differs from the frozen plan")
    cast_runtime = _mapping(cast_receipt.get("execution"), "cast runtime execution")
    if cast_runtime.get("schema") != "cyber_sft_fp32_to_bf16_cast_execution_v1":
        raise ValueError("unsupported FP32-to-BF16 cast runtime schema")
    for runtime_field, plan_field in (
        ("image", "image"),
        ("resolved_image_digest", "image_digest"),
        ("command_sha256", "command_sha256"),
        ("service_account_name", "service_account_name"),
        ("container_name", "container_name"),
    ):
        if cast_runtime.get(runtime_field) != expected_cast_execution.get(plan_field):
            raise ValueError(f"FP32-to-BF16 cast runtime {runtime_field} differs from plan")
    if _sha256(cast_runtime, "cast_input_sha256") != _sha256(
        cast_receipt, "cast_input_sha256"
    ):
        raise ValueError("FP32-to-BF16 cast runtime names a different cast input")
    runtime_job = _mapping(cast_runtime.get("job"), "cast runtime Job")
    runtime_pod = _mapping(cast_runtime.get("pod"), "cast runtime Pod")
    runtime_config_map = _mapping(cast_runtime.get("config_map"), "cast runtime ConfigMap")
    if (
        _text(runtime_job, "namespace") != expected_cast_execution.get("namespace")
        or _text(runtime_job, "name") != expected_cast_execution.get("job_name")
        or _text(runtime_config_map, "name") != expected_cast_execution.get("config_map_name")
        or runtime_config_map.get("immutable") is not True
    ):
        raise ValueError("FP32-to-BF16 cast runtime resources differ from plan")
    for runtime_object in (runtime_job, runtime_pod, runtime_config_map):
        _text(runtime_object, "uid")
        _text(runtime_object, "resource_version")
    _sha256(runtime_job, "spec_sha256")
    _sha256(runtime_pod, "spec_sha256")
    mounted = _mapping(runtime_config_map.get("mounted_file_sha256"), "cast mounted files")
    for path, digest in _mapping(
        expected_cast_execution.get("config_map_code_sha256"), "cast reviewed code"
    ).items():
        if mounted.get(path) != digest:
            raise ValueError(f"FP32-to-BF16 cast mounted code differs for {path}")
    cast_destination = _mapping(cast_receipt.get("destination"), "cast destination")
    expected_cast_path = _text(expected_export_binding, "bf16_cast_destination")
    cast_inspection = _mapping(cast_destination.get("inspection"), "cast output inspection")
    if (
        _text(cast_destination, "path") != expected_cast_path
        or _text(cast_destination, "dtype") != "BF16"
        or _text(cast_inspection, "root") != expected_cast_path
        or _text(cast_inspection, "dtype").lower() not in {"bf16", "bfloat16"}
    ):
        raise ValueError("FP32-to-BF16 cast destination differs from the frozen plan")
    cast_weights_sha256 = _sha256(cast_inspection, "weights_manifest_sha256")
    cast_rows = cast_full_manifest.get("files")
    if (
        cast_full_manifest.get("schema") != "cyber_sft_full_file_manifest_v1"
        or _text(cast_full_manifest, "root") != expected_cast_path
        or not isinstance(cast_rows, list)
    ):
        raise ValueError("BF16 cast full manifest is malformed")
    cast_full_manifest_sha256 = _sha256(cast_full_manifest, "manifest_sha256")
    if digest_json(cast_rows) != cast_full_manifest_sha256:
        raise ValueError("BF16 cast full manifest digest does not validate")
    acceptance_rows = [
        row
        for row in cast_rows
        if isinstance(row, Mapping) and row.get("path") == ".fleet-bf16-cast-acceptance.json"
    ]
    if len(acceptance_rows) != 1:
        raise ValueError("BF16 cast full manifest does not contain one acceptance receipt")
    payload_rows = [row for row in cast_rows if row not in acceptance_rows]
    if _sha256(cast_destination, "payload_manifest_sha256") != digest_json(payload_rows):
        raise ValueError("BF16 cast payload manifest differs from its receipt")

    if stage_input.get("schema") != "cyber_sft_inference_stage_input_v1":
        raise ValueError("unsupported inference stage input schema")
    stage_input_sha256 = _validate_embedded_digest(stage_input, "stage_input_sha256")
    stage_source = _mapping(stage_input.get("source"), "stage input source")
    if _text(stage_source, "sfs_path") != expected_cast_path:
        raise ValueError("inference stage input names an unexpected BF16 cast path")
    if _sha256(stage_source, "observation_sha256") != export_observation_sha256:
        raise ValueError("inference stage input names a different export observation")
    if _sha256(stage_source, "cast_receipt_sha256") != cast_receipt_sha256:
        raise ValueError("inference stage input names a different BF16 cast receipt")
    if stage_source.get("bf16_inspection") != cast_inspection:
        raise ValueError("inference stage input BF16 inspection differs from cast evidence")
    stage_manifest = _mapping(
        stage_source.get("bf16_full_manifest"), "stage input BF16 full manifest"
    )
    if _sha256(stage_manifest, "manifest_sha256") != cast_full_manifest_sha256:
        raise ValueError("inference stage input BF16 manifest differs from cast evidence")
    stage_composition = _mapping(stage_input.get("composition"), "stage input composition")
    expected_composition = {
        "runtime_sidecar_sha256": dict(expected_runtime_sidecar_sha256),
        "expected_tokenizer_manifest_sha256": expected_tokenizer_manifest_sha256,
        "expected_chat_template_sha256": expected_chat_template_sha256,
        "expected_config_sha256": expected_config_sha256,
        "tokenizer_equivalence_evidence_sha256": (
            expected_tokenizer_equivalence_evidence_sha256
        ),
    }
    for field, expected in expected_composition.items():
        if stage_composition.get(field) != expected:
            raise ValueError(f"inference stage composition {field} differs from the frozen plan")
    stage_execution = _mapping(stage_input.get("execution"), "stage input execution")
    if stage_execution.get("schema") != "cyber_sft_inference_stage_execution_plan_v1":
        raise ValueError("unsupported inference stage execution plan schema")
    for field in (
        "namespace",
        "job_name",
        "config_map_name",
        "service_account_name",
        "container_name",
    ):
        _text(stage_execution, field)
    planned_staging_image = _digest_pinned_image(stage_execution, "image")
    if planned_staging_image != expected_staging_image:
        raise ValueError("inference stage planned image differs from the frozen plan")
    expected_image_digest = planned_staging_image.rsplit("@", 1)[1]
    if _sha256(stage_execution, "image_digest") != expected_image_digest:
        raise ValueError("inference stage planned image digest is inconsistent")
    if _sha256(stage_execution, "command_sha256") != expected_staging_command_sha256:
        raise ValueError("inference stage planned command differs from the frozen plan")
    reviewed_code_sha256 = _exact_sha256_map(
        stage_execution.get("config_map_code_sha256"),
        "stage input reviewed staging code",
        STAGING_CODE_PATHS,
    )
    stage_destination = _mapping(stage_input.get("destination"), "stage input destination")
    if (
        _text(stage_destination, "path") != expected_destination
        or stage_destination.get("must_be_absent") is not True
    ):
        raise ValueError("inference stage destination differs from the collision-free plan")

    if staging_receipt.get("schema") != "cyber_sft_inference_stage_receipt_v1":
        raise ValueError("unsupported inference stage receipt schema")
    staging_digest = _validate_embedded_digest(staging_receipt, "staging_receipt_sha256")
    if _sha256(staging_receipt, "stage_input_sha256") != stage_input_sha256:
        raise ValueError("inference staging executed a different stage input")
    if _sha256(staging_receipt, "source_observation_sha256") != export_observation_sha256:
        raise ValueError("inference staging names a different export observation")
    if _sha256(staging_receipt, "source_bf16_manifest_sha256") != cast_full_manifest_sha256:
        raise ValueError("inference staging names a different BF16 cast manifest")
    if _sha256(staging_receipt, "source_cast_receipt_sha256") != cast_receipt_sha256:
        raise ValueError("inference staging names a different BF16 cast receipt")
    execution = _mapping(staging_receipt.get("execution"), "staging receipt execution")
    if execution.get("schema") != "cyber_sft_inference_stage_execution_v1":
        raise ValueError("unsupported inference staging execution schema")
    staging_image = _digest_pinned_image(execution, "image")
    if staging_image != planned_staging_image:
        raise ValueError("inference staging image differs from the frozen plan")
    if _sha256(execution, "resolved_image_digest") != expected_image_digest:
        raise ValueError("inference staging resolved image digest differs from the frozen plan")
    image_id = _text(execution, "image_id")
    image_id_match = re.fullmatch(
        r"(?:[a-z][a-z0-9+.-]*://)?(?:[^@\s]+@)?(sha256:[0-9a-f]{64})", image_id
    )
    if image_id_match is None or image_id_match.group(1) != expected_image_digest:
        raise ValueError("inference staging imageID is not the exact frozen digest")
    staging_command_sha256 = _sha256(execution, "command_sha256")
    if staging_command_sha256 != _sha256(stage_execution, "command_sha256"):
        raise ValueError("inference staging command differs from the frozen plan")
    if (
        _text(execution, "service_account_name")
        != _text(stage_execution, "service_account_name")
        or _text(execution, "container_name") != _text(stage_execution, "container_name")
    ):
        raise ValueError("inference staging runtime identity differs from the stage input")
    if _sha256(execution, "stage_input_sha256") != stage_input_sha256:
        raise ValueError("inference staging provenance names a different stage input")
    execution_job = _mapping(execution.get("job"), "staging execution Job")
    execution_pod = _mapping(execution.get("pod"), "staging execution Pod")
    execution_config_map = _mapping(
        execution.get("config_map"), "staging execution ConfigMap"
    )
    if (
        _text(execution_job, "namespace") != _text(stage_execution, "namespace")
        or _text(execution_job, "name") != _text(stage_execution, "job_name")
        or _text(execution_pod, "namespace") != _text(stage_execution, "namespace")
        or _text(execution_config_map, "namespace") != _text(stage_execution, "namespace")
        or _text(execution_config_map, "name")
        != _text(stage_execution, "config_map_name")
    ):
        raise ValueError("inference staging runtime object names differ from the stage input")
    for runtime_object, label in (
        (execution_job, "Job"),
        (execution_pod, "Pod"),
        (execution_config_map, "ConfigMap"),
    ):
        _text(runtime_object, "uid")
        _text(runtime_object, "resource_version")
        if label != "ConfigMap":
            _sha256(runtime_object, "spec_sha256")
    _text(execution_pod, "name")
    if execution_config_map.get("immutable") is not True:
        raise ValueError("inference staging ConfigMap was not immutable")
    if _exact_sha256_map(
        execution_config_map.get("reviewed_code_sha256"),
        "staging execution reviewed code",
        STAGING_CODE_PATHS,
    ) != reviewed_code_sha256:
        raise ValueError("inference staging reviewed code differs from the stage input")
    expected_stage_input_file_sha256 = _canonical_pretty_json_sha256(stage_input)
    if _sha256(execution_config_map, "stage_input_file_sha256") != (
        expected_stage_input_file_sha256
    ):
        raise ValueError("mounted stage-input bytes differ from the frozen stage input")
    expected_mounted_sha256 = {
        **reviewed_code_sha256,
        "stage-input.json": expected_stage_input_file_sha256,
    }
    if _exact_sha256_map(
        execution_config_map.get("mounted_file_sha256"),
        "staging execution mounted files",
        set(expected_mounted_sha256),
    ) != expected_mounted_sha256:
        raise ValueError("inference staging mounted bytes differ from the stage input")
    staged_composition = _mapping(staging_receipt.get("composition"), "staging receipt composition")
    if staged_composition.get("policy") != (
        "verified_bf16_cast_weights_and_index_plus_exact_base_runtime_sidecars_v1"
    ):
        raise ValueError("unsupported inference bundle composition policy")
    if staged_composition.get("tokenizer_equivalence_evidence_sha256") != (
        stage_composition.get("tokenizer_equivalence_evidence_sha256")
    ):
        raise ValueError("staging used different tokenizer-equivalence evidence")
    composed = _mapping(staged_composition.get("inspection"), "composed output inspection")
    if _text(composed, "root") != expected_destination:
        raise ValueError("composed output inspection names an unexpected destination")
    if _sha256(composed, "weights_manifest_sha256") != cast_weights_sha256:
        raise ValueError("composed output weights differ from the verified BF16 cast")
    if _sha256(composed, "tokenizer_manifest_sha256") != expected_tokenizer_manifest_sha256:
        raise ValueError("composed tokenizer differs from the frozen base tokenizer")
    if _sha256(composed, "chat_template_sha256") != expected_chat_template_sha256:
        raise ValueError("composed chat template differs from the frozen base template")
    if _sha256(composed, "config_sha256") != expected_config_sha256:
        raise ValueError("composed model config differs from the frozen base config")
    if _mapping(composed.get("sidecar_sha256"), "composed sidecar hashes") != dict(
        expected_runtime_sidecar_sha256
    ):
        raise ValueError("composed runtime sidecars differ from the frozen base files")
    for field in ("all_shards_present", "safetensors_load_passed", "parameter_count_matches"):
        if composed.get(field) is not True:
            raise ValueError(f"composed output verification {field} did not pass")
    staged_destination = _mapping(staging_receipt.get("destination"), "staging destination")
    if (
        _text(staged_destination, "path") != expected_destination
        or _text(staged_destination, "acceptance_receipt_path")
        != f"{expected_destination}/{STAGING_ACCEPTANCE_RECEIPT}"
        or staged_destination.get("payload_manifest_excludes")
        != [STAGING_ACCEPTANCE_RECEIPT]
        or _sha256(staged_destination, "payload_manifest_sha256")
        != _sha256(composed, "files_manifest_sha256")
        or not isinstance(staged_destination.get("payload_file_count"), int)
        or staged_destination["payload_file_count"] < 1
        or not isinstance(staged_destination.get("payload_total_bytes"), int)
        or staged_destination["payload_total_bytes"] < 1
        or staged_destination.get("atomic_transaction") != "directory_rename_noreplace_v1"
        or staged_destination.get("atomic_promotion") is not True
    ):
        raise ValueError("staging receipt does not prove atomic promotion to the frozen path")

    receipt = {
        "schema": "cyber_sft_hf_export_v1",
        "source_checkpoint": {
            "uuid": _text(checkpoint, "uuid"),
            "source_manifest_sha256": _selection_source_manifest(selection),
        },
        "output": {
            "format": "safetensors",
            "dtype": "bf16",
            "source_path": expected_destination,
            "weights_manifest_sha256": cast_weights_sha256,
            "files_manifest_sha256": _sha256(composed, "files_manifest_sha256"),
            "tokenizer_manifest_sha256": _sha256(composed, "tokenizer_manifest_sha256"),
            "chat_template_sha256": _sha256(composed, "chat_template_sha256"),
            "config_sha256": _sha256(composed, "config_sha256"),
            "sidecar_sha256": copy.deepcopy(dict(composed["sidecar_sha256"])),
        },
        "conversion": {
            "image": _text(export_run_observation, "trainer_image"),
            "optimizer_steps": 0,
            "run": copy.deepcopy(dict(observed_run)),
            "resume_from": export_run_observation.get("resume_from"),
            "num_steps": export_run_observation.get("num_steps"),
            "hf_save_interval": export_run_observation.get("hf_save_interval"),
            "export_request_receipt_sha256": request_digest,
            "command_sha256": command_sha256,
            "output_path": expected_raw_path,
            "destination_preflight": copy.deepcopy(
                expected_export_binding["destination_preflight"]
            ),
            "observed_raw_dtype": "F32",
            "raw_weights_manifest_sha256": raw_weights_sha256,
            "raw_full_manifest_sha256": raw_files_sha256,
        },
        "precision_correction": {
            "schema": "cyber_sft_fp32_to_bf16_precision_correction_v1",
            "source_path": expected_raw_path,
            "destination_path": expected_cast_path,
            "source_dtype": "F32",
            "destination_dtype": "BF16",
            "policy": cast_conversion["policy"],
            "source_weights_manifest_sha256": raw_weights_sha256,
            "destination_weights_manifest_sha256": cast_weights_sha256,
            "cast_rows_sha256": _sha256(cast_conversion, "cast_rows_sha256"),
            "source_layout_sha256": _sha256(cast_conversion, "source_layout_sha256"),
            "cast_receipt_sha256": cast_receipt_sha256,
            "cast_full_manifest_sha256": cast_full_manifest_sha256,
            "exact_cast_bits_verified": True,
        },
        "staging": {
            "source_path": expected_cast_path,
            "destination_path": expected_destination,
            "image": staging_image,
            "command_sha256": staging_command_sha256,
            "source_manifest_sha256": cast_weights_sha256,
            "destination_manifest_sha256": _sha256(composed, "weights_manifest_sha256"),
            "byte_identical": True,
            "acceptance_manifest_sha256": staging_digest,
        },
        "verification": {
            "all_shards_present": True,
            "safetensors_load_passed": True,
            "parameter_count_matches": True,
        },
        "evidence": {
            "export_run_observation_sha256": _sha256(export_run_observation, "observation_sha256"),
            "export_filesystem_observation_sha256": export_observation_sha256,
            "bf16_cast_receipt_sha256": cast_receipt_sha256,
            "bf16_cast_full_manifest_sha256": cast_full_manifest_sha256,
            "stage_input_sha256": stage_input_sha256,
            "staging_receipt_sha256": staging_digest,
        },
    }
    signed = _with_digest(receipt, "export_receipt_sha256")
    validate_hf_export_receipt(
        signed,
        selection,
        expected_tokenizer_manifest_sha256=expected_tokenizer_manifest_sha256,
        expected_chat_template_sha256=expected_chat_template_sha256,
        expected_config_sha256=expected_config_sha256,
        expected_export_binding=expected_export_binding,
        expected_runtime_sidecar_sha256=expected_runtime_sidecar_sha256,
    )
    return signed


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
