"""Exact SkyRL checkpoint recovery; native workers still own loading and updates.

Validation restores every rank and the sampler, then evaluates without calling
the training loop. Continuation starts at the next optimizer step in a new owned
output directory. Neither mode writes or deletes the source checkpoint.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import re
from pathlib import Path

from . import checkpoints
from .sft_runtime import (
    _unsigned_digest,
    digest,
    selection_evidence,
    selection_policy,
    uses_reference_ce,
    write_receipt,
)

_RECOVERY_FIELDS = {"manifest", "sha256", "mode"}
_LEGACY_REQUEST_FIELDS = {"source_request", "source_request_sha256"}


def _image_identity(value: str) -> str:
    """Normalize an OCI tag away while retaining the exact repository+digest."""
    if not isinstance(value, str) or value.count("@sha256:") != 1:
        raise ValueError("producer request image must be pinned by immutable digest")
    name, digest_value = value.rsplit("@sha256:", 1)
    head, separator, leaf = name.rpartition("/")
    repository_leaf = leaf.split(":", 1)[0]
    if (
        not separator
        or not repository_leaf
        or not re.fullmatch(r"[a-f0-9]{64}", digest_value)
        or any(character.isspace() for character in name)
    ):
        raise ValueError("producer request image must be pinned by immutable digest")
    return f"{head}/{repository_leaf}@sha256:{digest_value}"


def _validate_request_value(manifest: dict, request: dict, image_identity: str) -> None:
    source = manifest["source_plan"]
    plan_file = manifest.get("source_plan_file")
    recipe = source["recipe"]
    runtime_sha256 = source.get("runtime_sha256")
    command = request.get("command")
    if (
        not isinstance(plan_file, dict)
        or not isinstance(runtime_sha256, str)
        or request.get("name") != source["run_name"]
        or request.get("run_dir") != source["output_root"]
        or request.get("workers") != recipe["nodes"]
        or request.get("gpus_per_worker") != recipe["gpus_per_node"]
        or request.get("requeueIfPreempted") is not False
        or not isinstance(command, str)
        or plan_file["sha256"] not in command
        or runtime_sha256 not in command
        or _image_identity(request.get("image")) != image_identity
    ):
        raise ValueError("producer request differs from the exact source plan/runtime")


def _bind_legacy_request(
    manifest: dict,
    path: Path,
    expected_sha256: str,
    *,
    manifest_file_sha256: str,
    target_image: str,
) -> dict:
    """Carry exact producer request bytes into a legacy recovery plan."""
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("source request needs an exact raw SHA-256")
    if path.is_symlink() or not path.is_file():
        raise ValueError("source request must be a real file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("source request file digest mismatch")
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("source request is not valid JSON") from error
    if not isinstance(request, dict):
        raise ValueError("source request must contain a JSON object")
    from .sft import IMAGE

    qualified_identity = _image_identity(IMAGE)
    if _image_identity(target_image) != qualified_identity:
        raise ValueError("legacy recovery requires the qualified immutable trainer image")
    _validate_request_value(manifest, request, qualified_identity)
    binding = {
        "schema": "cyber_sft_legacy_producer_request_binding_v1",
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": expected_sha256,
        "canonical_json_sha256": _unsigned_digest(request),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
        "image": request["image"],
        "image_identity": qualified_identity,
        "source_manifest_file_sha256": manifest_file_sha256,
        "source_manifest_receipt_sha256": manifest["receipt_sha256"],
        "source_plan_file_sha256": manifest["source_plan_file"]["sha256"],
    }
    binding["binding_sha256"] = _unsigned_digest(binding)
    return binding


def _validate_legacy_request(plan: dict, recovery: dict, manifest: dict) -> str:
    binding = recovery.get("source_request")
    keys = {
        "schema",
        "path",
        "bytes",
        "sha256",
        "canonical_json_sha256",
        "raw_base64",
        "image",
        "image_identity",
        "source_manifest_file_sha256",
        "source_manifest_receipt_sha256",
        "source_plan_file_sha256",
        "binding_sha256",
    }
    if (
        not isinstance(binding, dict)
        or set(binding) != keys
        or binding["schema"] != "cyber_sft_legacy_producer_request_binding_v1"
        or not isinstance(binding["path"], str)
        or not Path(binding["path"]).is_absolute()
        or type(binding["bytes"]) is not int
        or binding["bytes"] <= 0
        or any(
            not isinstance(binding[name], str) or not re.fullmatch(r"[a-f0-9]{64}", binding[name])
            for name in (
                "sha256",
                "canonical_json_sha256",
                "source_manifest_file_sha256",
                "source_manifest_receipt_sha256",
                "source_plan_file_sha256",
                "binding_sha256",
            )
        )
        or binding["binding_sha256"]
        != _unsigned_digest({k: v for k, v in binding.items() if k != "binding_sha256"})
        or binding["source_manifest_file_sha256"] != recovery["manifest_file_sha256"]
        or binding["source_manifest_receipt_sha256"] != manifest["receipt_sha256"]
        or binding["source_plan_file_sha256"] != manifest["source_plan_file"]["sha256"]
    ):
        raise ValueError("legacy producer request binding is invalid")
    try:
        raw = base64.b64decode(binding["raw_base64"], validate=True)
        request = json.loads(raw)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("legacy producer request binding is invalid") from error
    if (
        len(raw) != binding["bytes"]
        or hashlib.sha256(raw).hexdigest() != binding["sha256"]
        or not isinstance(request, dict)
        or _unsigned_digest(request) != binding["canonical_json_sha256"]
        or _image_identity(binding["image"]) != binding["image_identity"]
        or request.get("image") != binding["image"]
        or _image_identity(plan["execution"]["image"]) != binding["image_identity"]
    ):
        raise ValueError("legacy producer request binding is invalid")
    _validate_request_value(manifest, request, binding["image_identity"])
    return binding["image"]


def _legacy_model_equal(target: object, source: object) -> bool:
    """Legacy producers recorded the same immutable file set in a different order."""
    if not isinstance(target, dict) or not isinstance(source, dict):
        return False
    target_model, source_model = dict(target), dict(source)
    target_files = target_model.pop("files", None)
    source_files = source_model.pop("files", None)
    if (
        target_model != source_model
        or not isinstance(target_files, list)
        or not isinstance(source_files, list)
    ):
        return False

    def ordered(files: list) -> list | None:
        if any(
            not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in files
        ):
            return None
        paths = [item["path"] for item in files]
        return (
            None if len(set(paths)) != len(paths) else sorted(files, key=lambda item: item["path"])
        )

    return len(target_files) == len(source_files) and ordered(target_files) == ordered(source_files)


def bind(plan: dict, config: dict, *, relative_to: Path) -> None:
    fields = set(config)
    if fields != _RECOVERY_FIELDS and fields != _RECOVERY_FIELDS | _LEGACY_REQUEST_FIELDS:
        raise ValueError("recovery needs manifest, file SHA-256 and validate/resume mode")
    path = relative_to / config["manifest"]
    if digest(path) != config["sha256"]:
        raise ValueError("recovery manifest file digest mismatch")
    manifest = json.loads(path.read_text())
    checkpoints.verify(manifest, check_files=False)
    source_image = manifest["source_plan"].get("execution", {}).get("image")
    has_request = fields >= _LEGACY_REQUEST_FIELDS
    if source_image is None and not has_request:
        raise ValueError("legacy recovery requires an exact producer request")
    if source_image is not None and has_request:
        raise ValueError("producer request compatibility binding is legacy-only")
    value = {
        "mode": config["mode"],
        "checkpoint": manifest,
        "manifest_file_sha256": config["sha256"],
    }
    if has_request:
        value["source_request"] = _bind_legacy_request(
            manifest,
            relative_to / config["source_request"],
            config["source_request_sha256"],
            manifest_file_sha256=config["sha256"],
            target_image=plan["execution"]["image"],
        )
    plan["recovery"] = value
    plan["recovery_runtime_sha256"] = digest(Path(__file__))
    validate(plan, check_files=False)


def validate(plan: dict, *, check_files: bool) -> None:
    recovery = plan["recovery"]
    manifest = recovery["checkpoint"]
    source_candidate = manifest.get("source_plan") if isinstance(manifest, dict) else None
    legacy = (
        isinstance(source_candidate, dict)
        and source_candidate.get("execution", {}).get("image") is None
    )
    checkpoints.verify(
        manifest,
        check_files=check_files,
        check_source_plan_file=not legacy,
    )
    source = manifest["source_plan"]
    if recovery["mode"] not in {"validate", "resume"}:
        raise ValueError("recovery mode must be validate or resume")
    for key in (
        "schema",
        "model",
        "datasets",
        "lora",
        "split_manifest_sha256",
        "corpus_manifest_sha256",
    ):
        equal = (
            _legacy_model_equal(plan.get(key), source.get(key))
            if key == "model" and legacy and "source_request" in recovery
            else plan.get(key) == source.get(key)
        )
        if not equal:
            raise ValueError("recovery cannot change model, data, topology or scientific recipe")
    source_recipe, target_recipe = source["recipe"], plan["recipe"]
    if target_recipe != source_recipe:
        source_science = {
            key: value
            for key, value in source_recipe.items()
            if key not in {"nodes", "gpus_per_node"}
        }
        target_science = {
            key: value
            for key, value in target_recipe.items()
            if key not in {"nodes", "gpus_per_node"}
        }
        source_world_size = source_recipe["nodes"] * source_recipe["gpus_per_node"]
        target_world_size = target_recipe["nodes"] * target_recipe["gpus_per_node"]
        if (
            recovery["mode"] != "validate"
            or "lora" in plan
            or target_science != source_science
            or target_world_size != source_world_size
            or target_world_size != manifest["world_size"]
        ):
            raise ValueError("recovery cannot change model, data, topology or scientific recipe")
    if selection_policy(plan) != selection_policy(source):
        raise ValueError("recovery cannot change selection mode or the Fleet dev protocol")
    target_image = plan["execution"]["image"]
    source_image = source.get("execution", {}).get("image")
    if source_image is None:
        source_image = _validate_legacy_request(plan, recovery, manifest)
    elif "source_request" in recovery or target_image != source_image:
        raise ValueError("recovery requires the source trainer image")
    if any(plan[k] == source[k] for k in ("run_name", "output_root")) or (
        plan["wandb"]["run_id"] == source["wandb"]["run_id"]
    ):
        raise ValueError("recovery requires a new run, output directory and W&B identity")
    if Path(plan["output_root"]).is_relative_to(Path(source["output_root"])) or Path(
        source["output_root"]
    ).is_relative_to(Path(plan["output_root"])):
        raise ValueError("recovery output must be outside the source run")
    if recovery["mode"] == "resume":
        if manifest["optimizer_step"] >= plan["recipe"]["max_steps"]:
            raise ValueError("training already completed; only zero-step validation is allowed")
        progress = manifest.get("training_progress")
        if not isinstance(progress, dict) or type(progress.get("supervised_tokens")) is not int:
            raise ValueError("continuation needs a checkpoint with saved training progress")
    if check_files and digest(Path(__file__)) != plan["recovery_runtime_sha256"]:
        raise ValueError("recovery runtime differs from the frozen plan")


def equal_state(left, right) -> bool:
    """Exact trusted native state comparison, including tensor/RNG values."""
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            equal_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def worker_class(plan: dict):
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    parent = FSDPPolicyWorkerBase
    if "lora" in plan:
        from .glm_runtime import worker_class as glm_worker

        parent = glm_worker(plan)
    manifest = plan["recovery"]["checkpoint"]
    expected = manifest["optimizer_step"]

    class RecoveryWorker(parent):
        def load_checkpoint(
            self, ckpt_dir, load_optimizer_states=True, load_lr_scheduler_states=True
        ):
            import torch
            import torch.distributed as dist

            if str(Path(ckpt_dir)) != str(Path(manifest["checkpoint_path"]) / "policy") or (
                not load_optimizer_states or not load_lr_scheduler_states
            ):
                raise ValueError("recovery must restore exact model, optimizer and scheduler")
            super().load_checkpoint(ckpt_dir, True, True)
            steps = [float(s["step"].item()) for s in self.optimizer.state.values() if "step" in s]
            if not steps or any(s != expected for s in steps):
                raise ValueError("loaded optimizer counters differ from the checkpoint")
            if self.scheduler.state_dict().get("last_epoch") != expected:
                raise ValueError("loaded scheduler step differs from the checkpoint")
            if "lora" not in plan:
                rank, size = dist.get_rank(), dist.get_world_size()
                extra = torch.load(
                    Path(ckpt_dir) / f"extra_state_world_size_{size}_rank_{rank}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                if not equal_state(
                    self.scheduler.state_dict(), extra["lr_scheduler"]
                ) or not equal_state(self.strategy.get_rng_state(), extra["rng"]):
                    raise ValueError("loaded scheduler or RNG state differs from the checkpoint")
            return {
                "rank": dist.get_rank(),
                "optimizer_step": expected,
                "optimizer_states": len(steps),
                "scheduler_restored": True,
            }

        def optim_step(self, *args, **kwargs):
            if plan["recovery"]["mode"] == "validate":
                raise ValueError("zero-step validation must never call the optimizer")
            return super().optim_step(*args, **kwargs)

    return RecoveryWorker


@contextlib.contextmanager
def use_worker(plan: dict):
    import ray
    from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

    original = fsdp_worker.PolicyWorker
    fsdp_worker.PolicyWorker = ray.remote(num_gpus=1)(worker_class(plan))
    try:
        yield
    finally:
        fsdp_worker.PolicyWorker = original


def load(trainer) -> int:
    """Native all-rank reload plus strict sampler restoration, not warning fallback."""
    import ray
    import torch

    plan = trainer.plan
    manifest = plan["recovery"]["checkpoint"]
    root = Path(manifest["checkpoint_path"])
    trainer_state = torch.load(root / "trainer_state.pt", map_location="cpu", weights_only=False)
    sampler = torch.load(root / "data.pt", map_location="cpu", weights_only=False)
    if trainer_state["global_step"] != manifest["optimizer_step"] or (
        sampler.get("_num_yielded") != manifest["sampler_batches_in_epoch"]
    ):
        raise ValueError("checkpoint metadata changed after sealing")
    trainer.dispatch._ensure_on_gpu("policy", need_optimizer=True, need_model=True)
    actor = trainer.dispatch._actor_groups["policy"]
    replies = ray.get(
        actor.async_run_ray_method(
            "pass_through",
            "load_checkpoint",
            ckpt_dir=str(root / "policy"),
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
    )
    if sorted(r["rank"] for r in replies) != list(range(manifest["world_size"])) or any(
        r["optimizer_step"] != manifest["optimizer_step"] for r in replies
    ):
        raise ValueError("incomplete all-rank checkpoint reload")
    trainer.train_dataloader.load_state_dict(sampler)
    if not equal_state(trainer.train_dataloader.state_dict(), sampler):
        raise ValueError("sampler did not restore exactly")
    progress = manifest.get("training_progress", {})
    trainer.target_tokens_seen = progress.get("supervised_tokens", 0)
    trainer.best = progress.get("best")
    write_receipt(
        trainer.output / "RECOVERED.json",
        {
            "source_manifest_sha256": manifest["receipt_sha256"],
            "source_plan_sha256": manifest["source_plan_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "optimizer_step": manifest["optimizer_step"],
            "ranks": replies,
            "sampler_restored": True,
            **selection_evidence(plan),
        },
    )
    return manifest["optimizer_step"]


def validate_only(trainer) -> dict:
    """Restore state, with CE only for legacy plans; never train/save/optimize."""
    trainer.train_dataloader = trainer.build_train_dataloader(trainer.load_dataset())
    reference_ce = uses_reference_ce(trainer.plan)
    if reference_ce:
        trainer.eval_dataloader = trainer.build_eval_dataloader(trainer.load_eval_dataset())
    trainer.global_step = load(trainer)
    metrics = trainer.run_eval()[0] if reference_ce else {}
    result = {
        "status": "reload_validated",
        "optimizer_steps_executed": 0,
        "optimizer_step": trainer.global_step,
        **({"held_out": metrics} if reference_ce else {}),
        "validation_scope": (
            "checkpoint_sampler_reload_and_reference_ce"
            if reference_ce
            else "checkpoint_and_sampler_reload_only_no_ce"
        ),
        **selection_evidence(trainer.plan),
        "plan_sha256": trainer.plan["plan_sha256"],
        "source_manifest_sha256": trainer.plan["recovery"]["checkpoint"]["receipt_sha256"],
    }
    trainer.tracker.log(
        {"train/global_step": trainer.global_step, **{"eval/" + k: v for k, v in metrics.items()}},
        step=trainer.global_step,
    )
    trainer.shutdown()
    return result
