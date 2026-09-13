"""Seal and reload a trained Miles checkpoint without performing an update.

This is an operational recovery gate, not an RL result.  The CPU sealer binds a
terminal one-step Miles run to an immutable distributed-checkpoint inventory.
The GPU validator then asks Miles' own Megatron actor initializer to restore
that checkpoint on every rank.  It never creates rollout engines and never
calls a train, save, evaluation, or weight-transfer method.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import bundled_request, digest, quantity

from . import miles
from .miles_conversion import _hash, _write
from .rl_runtime import sealed

SOURCE_SCHEMA = "cyber_miles_training_v1"
CHECKPOINT_SCHEMA = "cyber_miles_training_checkpoint_v1"
CONFIG_SCHEMA = "cyber_miles_rl_reload_config_v1"
RELOAD_SCHEMA = "cyber_miles_rl_reload_v1"
PREFLIGHT_SCHEMA = "cyber_miles_rl_reload_cpu_preflight_v1"
RESULT_SCHEMA = "cyber_miles_rl_reload_validation_v1"
NATIVE_DRIVER_SHA256 = "85dbfd31d41a84f9c2e79a2918583851fb53925630afa229e9cd0a154b170f46"
DEADLINE_SECONDS = 1800
RUNTIME_FILES = (
    "training/miles_reload.py",
    "training/miles.py",
    "training/miles_conversion.py",
    "training/rl_runtime.py",
    "cyber_post_train/jobs.py",
)


def _runtime() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def _receipt(path: Path, *, schema: str | None = None) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("receipt must be an object")
    if schema is None:
        if value.get("sha256", "").removeprefix("sha256:") != digest(
            {k: v for k, v in value.items() if k != "sha256"}
        ):
            raise ValueError("receipt digest mismatch")
    else:
        sealed(value, schema)
    return value


def _checkpoint_inventory(root: Path, index: int, world_size: int) -> list[dict]:
    """Inventory one numeric Megatron DCP checkpoint without following links."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("trained checkpoint root is missing or indirect")
    tracker = root / "latest_checkpointed_iteration.txt"
    if tracker.is_symlink() or tracker.read_text().strip() != str(index):
        raise ValueError("trained checkpoint tracker/index mismatch")
    generation = root / f"iter_{index:07d}"
    if generation.is_symlink() or not generation.is_dir():
        raise ValueError("trained checkpoint generation is missing or indirect")
    if {path.name for path in root.iterdir()} != {tracker.name, generation.name}:
        raise ValueError("trained checkpoint has unexpected generations or root entries")
    files: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("trained checkpoint contains an indirect payload")
        if path.is_file():
            size = path.stat().st_size
            if size < 1:
                raise ValueError("trained checkpoint contains an empty payload")
            files.append({"path": str(path.relative_to(root)), "size": size})
    names = {item["path"] for item in files}
    prefix = generation.name + "/"
    shards = [name for name in names if name.startswith(prefix) and name.endswith(".distcp")]
    if prefix + ".metadata" not in names or len(shards) < world_size:
        raise ValueError("trained checkpoint lacks all-rank distributed state")
    return files


def _checkpoint_snapshot(root: Path, files: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Record identity/mtime only; CPU preflight separately re-hashes every file."""
    snapshot = []
    for item in files:
        path = root / str(item["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError("trained checkpoint inventory changed")
        stat = path.stat()
        snapshot.append(
            {
                "path": item["path"],
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        )
    return snapshot


def _validate_source_plan(plan: dict) -> tuple[int, int]:
    from .miles_training import job_request as source_request

    # Reconstructing the original bundled request catches a stale/edited plan
    # before the sealer accepts any checkpoint bytes.
    source_request(plan)
    if plan.get("schema") != SOURCE_SCHEMA or plan.get("execution", {}).get("image") != miles.IMAGE:
        raise ValueError("checkpoint sealing requires the exact Miles training plan")
    args = plan.get("arguments", {})
    nodes, gpus = args.get("nodes"), args.get("gpus_per_node")
    if (
        type(nodes) is not int
        or type(gpus) is not int
        or nodes * gpus != 8
        or args.get("steps") != 1
        or args.get("checkpoint_interval") != 1
        or plan.get("output_root") != args.get("output_root")
        or plan.get("native_driver_sha256") != NATIVE_DRIVER_SHA256
    ):
        raise ValueError("reload qualification requires one saved update across eight ranks")
    return nodes, gpus


def seal_training_checkpoint(plan: dict, output: Path) -> dict:
    """Hash a successful one-step Miles checkpoint on CPU; no GPU/reload claim."""
    import torch

    if torch.cuda.is_available():
        raise ValueError("trained Miles checkpoint sealing is CPU-only")
    nodes, gpus = _validate_source_plan(plan)
    source = Path(plan["output_root"])
    completion = _receipt(source / "NATIVE_TRAINING_COMPLETE.json")
    conflicts = (
        "FAILED.json",
        "REJECTED.json",
        "NATIVE_FAILURE.json",
        "NATIVE_REJECTED.json",
        "ACCEPTED.json",
    )
    if (
        any((source / name).exists() for name in conflicts)
        or completion.get("status") != "native_loop_returned"
        or completion.get("plan_sha256") != digest(plan)
        or completion.get("checkpoint_rollout_index") != 0
        or completion.get("optimizer_update_independently_verified") is not False
        or completion.get("checkpoint_reload_verified") is not False
    ):
        raise ValueError("Miles source run is conflicting or not terminally eligible")
    checkpoint = source / "checkpoints"
    if output in (source, checkpoint) or source in output.parents or checkpoint in output.parents:
        raise ValueError("checkpoint seal must be outside the immutable source run")
    inventory = _checkpoint_inventory(checkpoint, 0, nodes * gpus)
    before = _checkpoint_snapshot(checkpoint, inventory)
    hashed = [{**item, "sha256": _hash(checkpoint / item["path"])} for item in inventory]
    if (
        _checkpoint_inventory(checkpoint, 0, nodes * gpus) != inventory
        or _checkpoint_snapshot(checkpoint, inventory) != before
    ):
        raise ValueError("trained checkpoint changed while sealing")
    return _write(
        output,
        {
            "schema": CHECKPOINT_SCHEMA,
            "image": miles.IMAGE,
            "root": str(checkpoint),
            "rollout_index": 0,
            "next_rollout_id": 1,
            "world_size": nodes * gpus,
            "topology": {"nodes": nodes, "gpus_per_node": gpus},
            "model": plan["model"],
            "source": {
                "run_name": plan["run_name"],
                "output_root": plan["output_root"],
                "plan_sha256": digest(plan),
                "completion_sha256": completion["sha256"].removeprefix("sha256:"),
                "arguments": plan["arguments"],
                "execution": plan["execution"],
                "native_driver_sha256": plan["native_driver_sha256"],
            },
            "files": hashed,
            "source_optimizer_update_claimed": False,
            "gpu_reload_verified": False,
            "optimizer_update_during_reload": False,
        },
    )


def _verify_checkpoint(manifest: dict, *, hashes: bool) -> list[dict]:
    sealed(manifest, CHECKPOINT_SCHEMA)
    root = Path(manifest["root"])
    inventory = _checkpoint_inventory(root, manifest["rollout_index"], manifest["world_size"])
    expected = [{key: item[key] for key in ("path", "size")} for item in manifest["files"]]
    if inventory != expected:
        raise ValueError("sealed Miles checkpoint inventory changed")
    if hashes:
        for item in manifest["files"]:
            if _hash(root / item["path"]) != item["sha256"]:
                raise ValueError("sealed Miles checkpoint payload changed")
    return _checkpoint_snapshot(root, manifest["files"])


def compile_reload(config: dict, *, relative_to: Path) -> dict:
    from .sft import _known, _sfs_root, read_mapping

    _known(config, {"schema", "name", "output_root", "checkpoint", "cluster"}, "Miles reload")
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Miles reload configuration schema mismatch")
    checkpoint = config["checkpoint"]
    cluster = config["cluster"]
    _known(checkpoint, {"manifest", "sha256"}, "checkpoint")
    _known(cluster, {"target", "priority", "resources"}, "cluster")
    if cluster.get("target") != "dev" or cluster.get("priority") != "c1":
        raise ValueError("Miles reload qualification is dev-only at c1")
    manifest_path = relative_to / checkpoint["manifest"]
    expected_digest = checkpoint["sha256"].removeprefix("sha256:")
    if len(expected_digest) != 64 or _hash(manifest_path) != expected_digest:
        raise ValueError("Miles trained-checkpoint manifest digest mismatch")
    manifest = read_mapping(manifest_path)
    sealed(manifest, CHECKPOINT_SCHEMA)
    if (
        manifest.get("image") != miles.IMAGE
        or manifest.get("world_size") != 8
        or manifest.get("next_rollout_id") != manifest.get("rollout_index") + 1
        or manifest.get("source_optimizer_update_claimed") is not False
        or manifest.get("gpu_reload_verified") is not False
    ):
        raise ValueError("trained-checkpoint manifest is outside the qualified Miles contract")
    source_args = miles.MilesConfig(**manifest["source"]["arguments"])
    source_args.validate()
    if (
        source_args.name != manifest["source"]["run_name"]
        or source_args.output_root != manifest["source"]["output_root"]
        or Path(manifest["root"]) != Path(source_args.output_root) / "checkpoints"
        or source_args.model != manifest["model"]["repo"]
        or source_args.nodes != manifest["topology"]["nodes"]
        or source_args.gpus_per_node != manifest["topology"]["gpus_per_node"]
        or source_args.steps != 1
        or manifest["source"]["execution"].get("image") != miles.IMAGE
        or manifest["source"]["native_driver_sha256"] != NATIVE_DRIVER_SHA256
    ):
        raise ValueError("source topology/recipe differs from the sealed checkpoint")
    output = _sfs_root(config["output_root"], "output root")
    source_root, checkpoint_root = Path(manifest["source"]["output_root"]), Path(manifest["root"])
    output_path = Path(output)
    if any(
        output_path == path or output_path in path.parents or path in output_path.parents
        for path in (source_root, checkpoint_root)
    ):
        raise ValueError("reload output must be disjoint from its immutable source")
    source_resources = manifest["source"]["execution"]["resources"]
    resources = {**source_resources, **cluster.get("resources", {})}
    for key in ("cpu_request", "cpu_limit", "memory_request", "memory_limit"):
        if quantity(resources[key]) < quantity(source_resources[key]):
            raise ValueError("reload cannot underreserve the successful source topology")
    plan = {
        "schema": RELOAD_SCHEMA,
        "run_name": config["name"],
        "output_root": output,
        "source_manifest": manifest,
        "source_manifest_file_sha256": expected_digest,
        "runtime_sha256": digest(_runtime()),
        "native_driver_sha256": NATIVE_DRIVER_SHA256,
        "optimizer_updates": 0,
        "rollouts": 0,
        "deadline_seconds": DEADLINE_SECONDS,
        "execution": {
            "cluster_target": "dev",
            "image": miles.IMAGE,
            "priority": "c1",
            "resources": resources,
        },
    }
    job_request(plan)
    return plan


_REMOVE_ONE_VALUE = frozenset(
    {
        "--ref-load",
        "--save",
        "--save-interval",
        "--eval-interval",
        "--wandb-team",
        "--wandb-project",
        "--wandb-group",
        "--wandb-run-id",
        "--wandb-mode",
        "--wandb-dir",
    }
)
_REMOVE_FLAGS = frozenset(
    {
        "--colocate",
        "--offload",
        "--offload-train",
        "--offload-rollout",
        "--use-kl-loss",
        "--use-wandb",
        "--disable-wandb-random-suffix",
    }
)
_FORBIDDEN_FLAGS = frozenset(
    {
        "--debug-disable-optimizer",
        "--finetune",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--override-opt-param-scheduler",
    }
)


def reload_arguments(source: miles.MilesConfig, checkpoint_root: str) -> list[str]:
    """Transform the exact source recipe into a load-only actor configuration."""
    original = miles.arguments(source)
    result: list[str] = []
    index = 0
    while index < len(original):
        token = original[index]
        if token == "--eval-prompt-data":
            index += 3
        elif token in _REMOVE_ONE_VALUE:
            index += 2
        elif token in _REMOVE_FLAGS:
            index += 1
        elif token == "--load":
            result.extend((token, checkpoint_root))
            index += 2
        else:
            result.append(token)
            index += 1
    flags = {token for token in result if token.startswith("--")}
    if _FORBIDDEN_FLAGS & flags or "--load" not in flags:
        raise ValueError("source Miles arguments disable exact optimizer/RNG recovery")
    result.extend(("--debug-train-only", "--use-checkpoint-opt-param-scheduler"))
    flags = [token for token in result if token.startswith("--")]
    if len(flags) != len(set(flags)):
        raise ValueError("reload argument transformation introduced duplicate flags")
    return result


def native_source() -> Path:
    from miles.utils.external_utils.command_utils import repo_base_dir

    path = Path(repo_base_dir) / "train.py"
    if _hash(path) != NATIVE_DRIVER_SHA256:
        raise ValueError("native Miles training driver changed")
    return path


def native_args(plan: dict):
    os.environ["MILES_USE_LEGACY_ROLLOUT_V1"] = "0"
    os.environ["MILES_EXPERIMENTAL_FT_TRAINER"] = "0"
    from miles.utils.arguments import parse_args

    source = miles.MilesConfig(**plan["source_manifest"]["source"]["arguments"])
    argv = reload_arguments(source, plan["source_manifest"]["root"])
    previous = sys.argv
    try:
        sys.argv = [str(native_source()), *argv]
        args = parse_args()
    finally:
        sys.argv = previous
    expected = {
        "load": plan["source_manifest"]["root"],
        "save": None,
        "debug_train_only": True,
        "no_load_optim": False,
        "no_load_rng": False,
        "finetune": False,
        "use_checkpoint_opt_param_scheduler": True,
        "use_wandb": False,
        "use_kl_loss": False,
        "offload_train": False,
        "offload_rollout": False,
        "colocate": False,
        "actor_num_nodes": plan["source_manifest"]["topology"]["nodes"],
        "actor_num_gpus_per_node": plan["source_manifest"]["topology"]["gpus_per_node"],
    }
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError("native parser changed zero-update reload semantics")
    if args.num_rollout != 1 or args.eval_interval is not None or args.eval_prompt_data is not None:
        raise ValueError("reload must preserve scheduler shape without enabling rollout/evaluation")
    return args


def job_request(plan: dict) -> dict:
    manifest = plan["source_manifest"]
    if (
        plan.get("schema") != RELOAD_SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("native_driver_sha256") != NATIVE_DRIVER_SHA256
        or plan.get("optimizer_updates") != 0
        or plan.get("rollouts") != 0
        or plan.get("deadline_seconds") != DEADLINE_SECONDS
        or plan.get("execution", {}).get("cluster_target") != "dev"
        or plan["execution"].get("image") != miles.IMAGE
        or plan["execution"].get("priority") != "c1"
        or manifest.get("world_size") != 8
    ):
        raise ValueError("Miles reload runtime/plan drift")
    sealed(manifest, CHECKPOINT_SCHEMA)
    resources = plan["execution"]["resources"]
    source_resources = manifest["source"]["execution"]["resources"]
    if any(
        quantity(resources[key]) < quantity(source_resources[key])
        for key in ("cpu_request", "cpu_limit", "memory_request", "memory_limit")
    ):
        raise ValueError("Miles reload resource envelope drift")
    files = _runtime()
    files.update({"training/__init__.py": "", "cyber_post_train/__init__.py": ""})
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    topology = manifest["topology"]
    return bundled_request(
        {
            "name": plan["run_name"],
            "title": plan["run_name"] + " all-rank zero-update Miles reload",
            "run_dir": plan["output_root"],
            "image": miles.IMAGE,
            "workers": topology["nodes"],
            "gpus_per_worker": topology["gpus_per_node"],
            "resources": resources,
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "PYTHONPATH": "/root/Megatron-LM",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "MILES_USE_LEGACY_ROLLOUT_V1": "0",
                "MILES_EXPERIMENTAL_FT_TRAINER": "0",
                "WANDB_MODE": "disabled",
                "WANDB_DISABLED": "true",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        "training.miles_reload",
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def preflight(plan: dict) -> dict:
    import torch

    if torch.cuda.is_available():
        raise ValueError("Miles reload preflight must not allocate GPUs")
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("Miles reload output already exists")
    snapshot = _verify_checkpoint(plan["source_manifest"], hashes=True)
    source = miles.MilesConfig(**plan["source_manifest"]["source"]["arguments"])
    argv = reload_arguments(source, plan["source_manifest"]["root"])
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "checkpoint_files": len(snapshot),
        "checkpoint_sha256_verified": True,
        "native_arguments_sha256": digest(argv),
        "native_parser_checked": False,
        "optimizer_updates": 0,
        "rollouts": 0,
        "gpu_reload_verified": False,
    }


def _state_summary(value: Any, *, depth: int = 0) -> Any:
    """Return tensor metadata and scalar state only; never tensor/model values."""
    import torch

    if depth > 6:
        return {"type": type(value).__name__}
    if isinstance(value, torch.Tensor):
        return {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "requires_grad": bool(value.requires_grad),
            "version": int(getattr(value, "_version", 0)),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return [
            {
                "key": key if isinstance(key, (bool, int, float, str)) else type(key).__name__,
                "value": _state_summary(item, depth=depth + 1),
            }
            for key, item in value.items()
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_state_summary(item, depth=depth + 1) for item in value]
    return {"type": type(value).__name__}


def _model_probe(model: Sequence[Any]) -> dict:
    rows = []
    for chunk_index, chunk in enumerate(model):
        for name, parameter in chunk.named_parameters():
            rows.append(
                {
                    "chunk": chunk_index,
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "requires_grad": bool(parameter.requires_grad),
                    "version": int(getattr(parameter, "_version", 0)),
                }
            )
    return {
        "tensors": len(rows),
        "local_numel": sum(math.prod(item["shape"]) for item in rows),
        "structure_sha256": digest(rows),
    }


def _optimizer_probe(optimizer: Any) -> dict:
    pending, seen, state_rows, group_rows = [optimizer], set(), [], []
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        state = getattr(current, "state", None)
        if isinstance(state, Mapping):
            state_rows.append(_state_summary(state))
        groups = getattr(current, "param_groups", None)
        if isinstance(groups, Sequence):
            group_rows.append(_state_summary(groups))
        for attr in ("optimizer", "optimizers", "chained_optimizers"):
            child = getattr(current, attr, None)
            if isinstance(child, Sequence):
                pending.extend(child)
            elif child is not None:
                pending.append(child)
    state_entries = sum(len(row) for row in state_rows)
    return {
        "objects": len(seen),
        "state_entries": state_entries,
        "parameter_groups": sum(len(row) for row in group_rows),
        "structure_sha256": digest({"state": state_rows, "groups": group_rows}),
    }


def _positive_scheduler_counters(value: Any, key: str = "") -> int:
    if isinstance(value, Mapping):
        return sum(_positive_scheduler_counters(item, str(name)) for name, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_positive_scheduler_counters(item, key) for item in value)
    return int(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
        and any(token in key.lower() for token in ("step", "sample", "consum"))
    )


def _scheduler_probe(scheduler: Any) -> dict:
    state = scheduler.state_dict()
    return {
        "positive_progress_counters": _positive_scheduler_counters(state),
        "state_sha256": digest(_state_summary(state)),
    }


def _rng_probe() -> str:
    import torch

    payload = hashlib.sha256(repr(random.getstate()).encode())
    with suppress(ImportError):
        import numpy

        payload.update(repr(numpy.random.get_state()).encode())
    payload.update(bytes(torch.random.get_rng_state().cpu().tolist()))
    for state in torch.cuda.get_rng_state_all():
        payload.update(bytes(state.cpu().tolist()))
    return payload.hexdigest()


def _actor_probe(self) -> dict:
    """Runs inside one Miles actor; output contains no tensor values."""
    import torch.distributed as dist

    dist.barrier()
    result = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "load": self.args.load,
        "save_is_none": self.args.save is None,
        "debug_train_only": self.args.debug_train_only is True,
        "no_load_optim": self.args.no_load_optim is True,
        "no_load_rng": self.args.no_load_rng is True,
        "finetune": self.args.finetune is True,
        "use_checkpoint_opt_param_scheduler": self.args.use_checkpoint_opt_param_scheduler is True,
        "model": _model_probe(self.model),
        "optimizer": _optimizer_probe(self.optimizer),
        "scheduler": _scheduler_probe(self.opt_param_scheduler),
        "rng_sha256": _rng_probe(),
    }
    dist.barrier()
    return result


def validate_rank_probes(
    manifest: dict, start_ids: Sequence[int], before: Sequence[dict], after: Sequence[dict]
) -> dict:
    world = manifest["world_size"]
    if len(start_ids) != world or set(start_ids) != {manifest["next_rollout_id"]}:
        raise ValueError("all Miles ranks did not restore the exact rollout index")
    if len(before) != world or len(after) != world:
        raise ValueError("all-rank reload probes are incomplete")
    ordered_before = sorted(before, key=lambda row: row["rank"])
    ordered_after = sorted(after, key=lambda row: row["rank"])
    if [row["rank"] for row in ordered_before] != list(range(world)):
        raise ValueError("all-rank reload probe identity mismatch")
    if ordered_before != ordered_after:
        raise ValueError("model/optimizer/scheduler/RNG state changed during reload validation")
    for row in ordered_before:
        if (
            row["world_size"] != world
            or row["load"] != manifest["root"]
            or row["save_is_none"] is not True
            or row["debug_train_only"] is not True
            or row["no_load_optim"] is not False
            or row["no_load_rng"] is not False
            or row["finetune"] is not False
            or row["use_checkpoint_opt_param_scheduler"] is not True
            or row["model"]["tensors"] < 1
            or row["model"]["local_numel"] < 1
            or row["optimizer"]["state_entries"] < 1
            or row["optimizer"]["parameter_groups"] < 1
            or row["scheduler"]["positive_progress_counters"] < 1
        ):
            raise ValueError("one Miles rank lacks exact recoverable training state")
    return {
        "world_size": world,
        "ranks": list(range(world)),
        "restored_rollout_index": manifest["rollout_index"],
        "next_rollout_id": manifest["next_rollout_id"],
        "probe_set_sha256": digest(ordered_before),
        "all_rank_model_loaded": True,
        "all_rank_optimizer_loaded": True,
        "all_rank_scheduler_loaded": True,
        "all_rank_rng_loaded": True,
        "state_stable_across_zero_updates": True,
    }


async def _load_all_ranks(args, manifest: dict) -> dict:
    import ray
    from miles.backends.megatron_utils import actor as actor_module
    from miles.ray.placement_group import allocate_train_group, create_placement_groups

    original_actor = actor_module.MegatronTrainRayActor
    validator_actor = type(
        "CyberMilesReloadActor",
        (original_actor,),
        {"cyber_reload_probe": _actor_probe},
    )
    pgs = None
    group = None
    try:
        actor_module.MegatronTrainRayActor = validator_actor
        pgs = create_placement_groups(args)
        group = allocate_train_group(
            args=args,
            num_nodes=args.actor_num_nodes,
            num_gpus_per_node=args.actor_num_gpus_per_node,
            pg=pgs["actor"],
            role="actor",
            with_ref=False,
            rollout_manager=None,
        )
        actor_module.MegatronTrainRayActor = original_actor
        start_ids = await group.init()
        before = await group._broadcast("cyber_reload_probe")
        after = await group._broadcast("cyber_reload_probe")
        return validate_rank_probes(manifest, start_ids, before, after)
    finally:
        actor_module.MegatronTrainRayActor = original_actor
        if group is not None:
            for handle in group._actor_handles:
                with suppress(Exception):
                    ray.kill(handle, no_restart=True)
        if pgs is not None and pgs["actor"][0] is not None:
            with suppress(Exception):
                ray.util.remove_placement_group(pgs["actor"][0])


def _native(plan: dict) -> dict:
    import ray

    manifest = plan["source_manifest"]
    before = _verify_checkpoint(manifest, hashes=False)
    args = native_args(plan)
    ray.init(
        address="auto",
        log_to_driver=False,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(Path(__file__).resolve().parents[1])
                + ":"
                + os.environ.get("PYTHONPATH", ""),
                "MILES_USE_LEGACY_ROLLOUT_V1": "0",
                "MILES_EXPERIMENTAL_FT_TRAINER": "0",
                "WANDB_MODE": "disabled",
                "WANDB_DISABLED": "true",
            }
        },
    )
    try:
        result = asyncio.run(_load_all_ranks(args, manifest))
    finally:
        ray.shutdown()
    if _verify_checkpoint(manifest, hashes=False) != before:
        raise ValueError("source Miles checkpoint changed during reload")
    return result


def run(plan: dict) -> dict:
    if os.environ.get("RUN_DIR") != plan["output_root"]:
        raise ValueError("Jobs API output binding drift")
    job_request(plan)
    root = Path(plan["output_root"])
    if any((root / name).exists() for name in ("RELOAD_VALIDATED.json", "FAILED.json")):
        raise FileExistsError("Miles reload terminal evidence already exists")
    _write(
        root / "STARTED.json",
        {
            "schema": RELOAD_SCHEMA,
            "plan_sha256": digest(plan),
            "started_at": time.time(),
            "optimizer_updates": 0,
            "rollouts": 0,
        },
    )
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError("Miles reload exceeded its fixed deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(DEADLINE_SECONDS)
    try:
        proof = _native(plan)
        return _write(
            root / "RELOAD_VALIDATED.json",
            {
                "schema": RESULT_SCHEMA,
                "status": "reload_validated",
                "plan_sha256": digest(plan),
                "source_manifest_sha256": plan["source_manifest"]["sha256"].removeprefix("sha256:"),
                **proof,
                "optimizer_updates": 0,
                "rollouts": 0,
                "forwards": 0,
                "backwards": 0,
                "checkpoint_writes": 0,
                "source_checkpoint_unchanged": True,
                "external_gpu_release_verified": False,
                "scientific_rl_acceptance": False,
                "completed_at": time.time(),
            },
        )
    except BaseException as exc:
        _write(
            root / "FAILED.json",
            {
                "schema": RELOAD_SCHEMA,
                "status": "failed",
                "plan_sha256": digest(plan),
                "error_class": type(exc).__name__,
                "optimizer_updates": 0,
                "rollouts": 0,
            },
        )
        raise RuntimeError("Miles reload failed; preserve evidence and do not retry") from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        result = run(plan)
        print(json.dumps({key: result[key] for key in ("status", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
