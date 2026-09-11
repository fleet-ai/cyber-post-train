"""Exact SkyRL checkpoint recovery; native workers still own loading and updates.

Validation restores every rank and the sampler, then evaluates without calling
the training loop. Continuation starts at the next optimizer step in a new owned
output directory. Neither mode writes or deletes the source checkpoint.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from . import checkpoints
from .sft_runtime import digest, write_receipt


def bind(plan: dict, config: dict, *, relative_to: Path) -> None:
    if set(config) != {"manifest", "sha256", "mode"}:
        raise ValueError("recovery needs manifest, file SHA-256 and validate/resume mode")
    path = relative_to / config["manifest"]
    if digest(path) != config["sha256"]:
        raise ValueError("recovery manifest file digest mismatch")
    plan["recovery"] = {
        "mode": config["mode"],
        "checkpoint": json.loads(path.read_text()),
        "manifest_file_sha256": config["sha256"],
    }
    plan["recovery_runtime_sha256"] = digest(Path(__file__))
    validate(plan, check_files=False)


def validate(plan: dict, *, check_files: bool) -> None:
    recovery = plan["recovery"]
    manifest = recovery["checkpoint"]
    checkpoints.verify(manifest, check_files=check_files)
    source = manifest["source_plan"]
    if recovery["mode"] not in {"validate", "resume"}:
        raise ValueError("recovery mode must be validate or resume")
    for key in (
        "schema",
        "model",
        "datasets",
        "recipe",
        "lora",
        "split_manifest_sha256",
        "corpus_manifest_sha256",
    ):
        if plan.get(key) != source.get(key):
            raise ValueError("recovery cannot change model, data, topology or scientific recipe")
    if plan["execution"]["image"] != source["execution"]["image"]:
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
        },
    )
    return manifest["optimizer_step"]


def validate_only(trainer) -> dict:
    """Load and evaluate; deliberately never call train(), save() or optimizer.step()."""
    trainer.train_dataloader = trainer.build_train_dataloader(trainer.load_dataset())
    trainer.eval_dataloader = trainer.build_eval_dataloader(trainer.load_eval_dataset())
    trainer.global_step = load(trainer)
    metrics, _ = trainer.run_eval()
    result = {
        "status": "reload_validated",
        "optimizer_steps_executed": 0,
        "optimizer_step": trainer.global_step,
        "held_out": metrics,
        "plan_sha256": trainer.plan["plan_sha256"],
        "source_manifest_sha256": trainer.plan["recovery"]["checkpoint"]["receipt_sha256"],
    }
    trainer.tracker.log(
        {"train/global_step": trainer.global_step, **{"eval/" + k: v for k, v in metrics.items()}},
        step=trainer.global_step,
    )
    trainer.shutdown()
    return result
