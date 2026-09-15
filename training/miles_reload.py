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
LONG_SOURCE_SCHEMA = "cyber_miles_training_v2"
CHECKPOINT_SCHEMA = "cyber_miles_training_checkpoint_v1"
CONFIG_SCHEMA = "cyber_miles_rl_reload_config_v1"
RELOAD_SCHEMA = "cyber_miles_rl_reload_v1"
PREFLIGHT_SCHEMA = "cyber_miles_rl_reload_cpu_preflight_v1"
RESULT_SCHEMA = "cyber_miles_rl_reload_validation_v1"
RANK_STATE_COMMITMENT_METHOD = "miles_all_rank_model_optimizer_scheduler_rng_value_sha256_v1"
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
    value, _ = _json_snapshot(path)
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


def _json_snapshot(path: Path) -> tuple[dict, str]:
    """Read one regular JSON file once and reject replacement during the read."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("receipt is missing or indirect")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError("receipt changed while being read")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("receipt is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("receipt must be an object")
    return value, hashlib.sha256(payload).hexdigest()


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
    expected = {
        tracker.name,
        prefix + ".metadata",
        prefix + "common.pt",
        *(prefix + f"__{rank}_0.distcp" for rank in range(world_size)),
    }
    if names != expected:
        raise ValueError("trained checkpoint is not one complete exact-rank DCP generation")
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
    # This sealer may run after the repository evolves.  Historical request
    # identity is validated from the exact submitted source bytes by
    # miles_acceptance; never reconstruct a past POST with current code here.
    values = plan.get("arguments", {})
    long_horizon = plan.get("schema") == LONG_SOURCE_SCHEMA
    expected_layout = miles.LONG_CONTEXT_LAYOUT if long_horizon else None
    expected_driver = miles.LONG_NATIVE_DRIVER_SHA256 if long_horizon else NATIVE_DRIVER_SHA256
    runtime_image = values.get("runtime_image", miles.IMAGE)
    if plan.get("schema") not in {SOURCE_SCHEMA, LONG_SOURCE_SCHEMA}:
        raise ValueError("checkpoint sealing requires the exact Miles training plan")
    nodes, gpus = values.get("nodes"), values.get("gpus_per_node")
    if (
        type(nodes) is not int
        or type(gpus) is not int
        or (
            (nodes, gpus) != expected_layout
            if long_horizon
            else (nodes, gpus) not in miles.NATIVE_LAYOUTS
        )
        or (
            long_horizon
            and (
                values.get("harness") != "opencode"
                or values.get("native_profile") != miles.LONG_CONTEXT_PROFILE
                or values.get("context_tokens") != 262_144
                or values.get("response_tokens") != 245_760
                or values.get("tokens_per_turn") != 32_768
                or values.get("max_tokens_per_gpu") != 65_536
                or values.get("session_node_cap") != 4_096
                or runtime_image == miles.IMAGE
            )
        )
        or (not long_horizon and runtime_image != miles.IMAGE)
        or plan.get("execution", {}).get("image") != runtime_image
        or values.get("steps") != 1
        or values.get("checkpoint_interval") != 1
        or plan.get("output_root") != values.get("output_root")
        or plan.get("native_driver_sha256") != expected_driver
    ):
        raise ValueError("reload qualification requires one saved update on a native Miles layout")
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
            "image": plan["execution"]["image"],
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


def _sha256(value: object, label: str) -> str:
    result = str(value).removeprefix("sha256:")
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} is not an exact SHA-256")
    return result


def _terminal_binding(
    locator: Mapping[str, Any],
    *,
    relative_to: Path,
    manifest_path: Path,
    manifest_file_sha256: str,
    manifest: dict,
) -> dict:
    """Bind reload to a fully reopened terminal acceptance and saved rank state."""
    from . import miles_acceptance

    if not isinstance(locator, Mapping) or set(locator) != {
        "receipt",
        "file_sha256",
        "receipt_sha256",
    }:
        raise ValueError("terminal acceptance locator fields changed")
    path = relative_to / str(locator["receipt"])
    terminal, terminal_file_sha256 = _json_snapshot(path)
    if terminal_file_sha256 != _sha256(locator["file_sha256"], "terminal file digest"):
        raise ValueError("Miles terminal acceptance file digest mismatch")
    sealed(terminal, miles_acceptance.TERMINAL_SCHEMA)
    terminal_receipt_sha256 = _sha256(terminal["sha256"], "terminal receipt digest")
    if terminal_receipt_sha256 != _sha256(
        locator["receipt_sha256"], "configured terminal receipt digest"
    ):
        raise ValueError("Miles terminal acceptance receipt digest mismatch")
    miles_acceptance.validate_terminal(terminal, check_files=True)

    checkpoint_reference = terminal.get("checkpoint_manifest")
    policy_reference = terminal.get("policy_delta_observation")
    if (
        not isinstance(checkpoint_reference, dict)
        or set(checkpoint_reference) != miles_acceptance._REFERENCE_FIELDS
        or Path(checkpoint_reference.get("path", "")) != manifest_path
        or _sha256(checkpoint_reference.get("file_sha256"), "accepted checkpoint file digest")
        != manifest_file_sha256
        or _sha256(checkpoint_reference.get("receipt_sha256"), "accepted checkpoint digest")
        != _sha256(manifest.get("sha256"), "checkpoint manifest digest")
        or not isinstance(policy_reference, dict)
        or set(policy_reference) != miles_acceptance._REFERENCE_FIELDS
    ):
        raise ValueError("terminal acceptance does not bind the exact reload checkpoint")

    policy_path = Path(policy_reference["path"])
    policy, policy_file_sha256 = _json_snapshot(policy_path)
    if policy_file_sha256 != _sha256(
        policy_reference["file_sha256"], "policy observation file digest"
    ) or _sha256(policy.get("sha256"), "policy observation digest") != _sha256(
        policy_reference["receipt_sha256"], "accepted policy observation digest"
    ):
        raise ValueError("accepted policy-state observation digest changed")
    commitments = []
    for rank, row in enumerate(policy["ranks"]):
        restored = row["trained_reload"]["restored"]
        commitments.append(
            {
                "rank": rank,
                "model_tensor_count": row["policy_tensor_count"],
                "model_local_numel": row["local_policy_numel"],
                "model_structure_sha256": _sha256(
                    restored["model"]["structure_sha256"], "trained policy structure digest"
                ),
                "model_value_sha256": _sha256(
                    restored["model"]["value_sha256"], "trained policy value digest"
                ),
                "optimizer_value_sha256": _sha256(
                    restored["optimizer"]["value_sha256"], "trained optimizer digest"
                ),
                "scheduler_value_sha256": _sha256(
                    restored["scheduler"]["value_sha256"], "trained scheduler digest"
                ),
                "rng_value_sha256": _sha256(restored["rng_sha256"], "trained RNG digest"),
            }
        )
    return {
        "source_terminal_acceptance": {
            "path": str(path),
            "file_sha256": terminal_file_sha256,
            "receipt_sha256": terminal_receipt_sha256,
        },
        "source_policy_delta_observation": {
            "path": str(policy_path),
            "file_sha256": policy_file_sha256,
            "receipt_sha256": _sha256(policy["sha256"], "policy observation digest"),
        },
        "rank_state_commitment_method": RANK_STATE_COMMITMENT_METHOD,
        "expected_rank_state_commitments": commitments,
    }


def compile_reload(config: dict, *, relative_to: Path) -> dict:
    from .sft import _known, _sfs_root

    _known(
        config,
        {"schema", "name", "output_root", "checkpoint", "terminal_acceptance", "cluster"},
        "Miles reload",
    )
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Miles reload configuration schema mismatch")
    checkpoint = config["checkpoint"]
    cluster = config["cluster"]
    _known(checkpoint, {"manifest", "sha256"}, "checkpoint")
    _known(cluster, {"target", "priority", "resources"}, "cluster")
    target = cluster.get("target")
    if target not in {"dev", "prod"} or cluster.get("priority") != "c1":
        raise ValueError("Miles reload qualification requires its source cluster at c1")
    manifest_path = relative_to / checkpoint["manifest"]
    manifest, observed_manifest_file_sha256 = _json_snapshot(manifest_path)
    expected_digest = _sha256(checkpoint["sha256"], "trained-checkpoint manifest file digest")
    if observed_manifest_file_sha256 != expected_digest:
        raise ValueError("Miles trained-checkpoint manifest digest mismatch")
    sealed(manifest, CHECKPOINT_SCHEMA)
    if (
        manifest.get("world_size")
        != manifest["topology"]["nodes"] * manifest["topology"]["gpus_per_node"]
        or manifest.get("next_rollout_id") != manifest.get("rollout_index") + 1
        or manifest.get("source_optimizer_update_claimed") is not False
        or manifest.get("gpu_reload_verified") is not False
    ):
        raise ValueError("trained-checkpoint manifest is outside the qualified Miles contract")
    source_args = miles.MilesConfig(**manifest["source"]["arguments"])
    source_args.validate()
    source_target = manifest["source"]["execution"].get("cluster_target", "dev")
    source_driver = (
        miles.LONG_NATIVE_DRIVER_SHA256
        if source_args.harness == "opencode"
        else NATIVE_DRIVER_SHA256
    )
    if (
        source_args.name != manifest["source"]["run_name"]
        or source_args.output_root != manifest["source"]["output_root"]
        or Path(manifest["root"]) != Path(source_args.output_root) / "checkpoints"
        or source_args.model != manifest["model"]["repo"]
        or source_args.nodes != manifest["topology"]["nodes"]
        or source_args.gpus_per_node != manifest["topology"]["gpus_per_node"]
        or source_args.steps != 1
        or manifest.get("image") != source_args.runtime_image
        or manifest["source"]["execution"].get("image") != source_args.runtime_image
        or source_target != target
        or manifest["source"]["native_driver_sha256"] != source_driver
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
    terminal = _terminal_binding(
        config["terminal_acceptance"],
        relative_to=relative_to,
        manifest_path=manifest_path,
        manifest_file_sha256=expected_digest,
        manifest=manifest,
    )
    plan = {
        "schema": RELOAD_SCHEMA,
        "run_name": config["name"],
        "output_root": output,
        "source_manifest": manifest,
        "source_manifest_file_sha256": expected_digest,
        **terminal,
        "runtime_sha256": digest(_runtime()),
        "native_driver_sha256": source_driver,
        "optimizer_updates": 0,
        "rollouts": 0,
        "deadline_seconds": DEADLINE_SECONDS,
        "execution": {
            "cluster_target": target,
            "image": source_args.runtime_image,
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

_RANK_COMMITMENT_FIELDS = frozenset(
    {
        "rank",
        "model_tensor_count",
        "model_local_numel",
        "model_structure_sha256",
        "model_value_sha256",
        "optimizer_value_sha256",
        "scheduler_value_sha256",
        "rng_value_sha256",
    }
)


def _rank_commitments(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict]:
    rows = plan.get("expected_rank_state_commitments")
    world = manifest.get("world_size")
    if not isinstance(rows, list) or type(world) is not int or len(rows) != world:
        raise ValueError("Miles reload lacks all saved rank-state commitments")
    for rank, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != _RANK_COMMITMENT_FIELDS
            or row.get("rank") != rank
            or type(row.get("model_tensor_count")) is not int
            or row["model_tensor_count"] < 1
            or type(row.get("model_local_numel")) is not int
            or row["model_local_numel"] < 1
            or any(
                _sha256(row.get(key), f"rank {rank} {key}") != row.get(key)
                for key in _RANK_COMMITMENT_FIELDS
                if key.endswith("_sha256")
            )
        ):
            raise ValueError("Miles reload saved rank-state commitment is invalid")
    return rows


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
    source = miles.MilesConfig(**manifest.get("source", {}).get("arguments", {}))
    source.validate()
    terminal = plan.get("source_terminal_acceptance")
    policy = plan.get("source_policy_delta_observation")
    if (
        plan.get("schema") != RELOAD_SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("native_driver_sha256")
        != manifest.get("source", {}).get("native_driver_sha256")
        or plan.get("optimizer_updates") != 0
        or plan.get("rollouts") != 0
        or plan.get("deadline_seconds") != DEADLINE_SECONDS
        or plan.get("execution", {}).get("cluster_target")
        != manifest.get("source", {}).get("execution", {}).get("cluster_target", "dev")
        or plan["execution"].get("image") != manifest.get("image")
        or plan["execution"].get("priority") != "c1"
        or manifest.get("image") != source.runtime_image
        or manifest.get("topology")
        != {"nodes": source.nodes, "gpus_per_node": source.gpus_per_node}
        or manifest.get("world_size")
        != manifest["topology"]["nodes"] * manifest["topology"]["gpus_per_node"]
        or not isinstance(terminal, dict)
        or set(terminal) != {"path", "file_sha256", "receipt_sha256"}
        or not isinstance(policy, dict)
        or set(policy) != {"path", "file_sha256", "receipt_sha256"}
        or plan.get("rank_state_commitment_method") != RANK_STATE_COMMITMENT_METHOD
    ):
        raise ValueError("Miles reload runtime/plan drift")
    for label, reference in (("terminal", terminal), ("policy", policy)):
        if not isinstance(reference["path"], str) or not reference["path"].startswith("/"):
            raise ValueError(f"Miles reload {label} receipt path is not absolute")
        _sha256(reference["file_sha256"], f"{label} file digest")
        _sha256(reference["receipt_sha256"], f"{label} receipt digest")
    _rank_commitments(plan, manifest)
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
            "image": plan["execution"]["image"],
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


def _preflight_receipt(
    plan: dict, request: dict, *, checkpoint_files: int, native_arguments: Sequence[str]
) -> dict:
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "checkpoint_files": checkpoint_files,
        "checkpoint_sha256_verified": True,
        "native_arguments_sha256": digest(native_arguments),
        "native_parser_checked": False,
        "optimizer_updates": 0,
        "rollouts": 0,
        "gpu_reload_verified": False,
    }


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
    return _preflight_receipt(plan, request, checkpoint_files=len(snapshot), native_arguments=argv)


def validate_preflight_receipt(plan: dict, request: dict, proof: dict) -> None:
    """Require the complete zero-work checkpoint proof before the only POST."""
    if request != job_request(plan):
        raise ValueError("Miles reload request does not match its immutable plan")
    source = miles.MilesConfig(**plan["source_manifest"]["source"]["arguments"])
    argv = reload_arguments(source, plan["source_manifest"]["root"])
    expected = _preflight_receipt(
        plan,
        request,
        checkpoint_files=len(plan["source_manifest"]["files"]),
        native_arguments=argv,
    )
    body = {key: value for key, value in proof.items() if key != "sha256"}
    if body != expected or proof.get("sha256") != digest(body):
        raise ValueError("missing or mismatched Miles reload CPU preflight")


def _tensor_value_sha256(value: Any) -> str:
    """Hash raw tensor bytes in bounded chunks without returning any values."""
    import torch

    tensor = value.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    raw = tensor.reshape(-1).view(torch.uint8)
    result = hashlib.sha256()
    chunk_bytes = 16 * 1024 * 1024
    for start in range(0, raw.numel(), chunk_bytes):
        chunk = raw[start : start + chunk_bytes].cpu()
        result.update(chunk.numpy().tobytes())
    return result.hexdigest()


def _state_summary(value: Any, *, include_values: bool, depth: int = 0) -> Any:
    """Return a canonical state tree; tensor payloads appear only as SHA-256."""
    import torch

    if depth > 32:
        raise ValueError("Miles recoverable state is unexpectedly deeply nested")
    if isinstance(value, torch.Tensor):
        result = {
            "tensor": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "requires_grad": bool(value.requires_grad),
        }
        if include_values:
            result["value_sha256"] = _tensor_value_sha256(value)
        return result
    if value is None or isinstance(value, (bool, int, float, str)):
        return value if include_values else {"type": type(value).__name__}
    if isinstance(value, Mapping):
        return [
            {
                "key": key if isinstance(key, (bool, int, float, str)) else type(key).__name__,
                "value": _state_summary(item, include_values=include_values, depth=depth + 1),
            }
            for key, item in value.items()
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _state_summary(item, include_values=include_values, depth=depth + 1) for item in value
        ]
    return {"type": type(value).__name__}


def _model_probe(model: Sequence[Any]) -> dict:
    structure, values = [], []
    for chunk_index, chunk in enumerate(model):
        for name, parameter in chunk.named_parameters():
            row = {
                "chunk": chunk_index,
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            structure.append(row)
            values.append({**row, "value_sha256": _tensor_value_sha256(parameter)})
    return {
        "tensors": len(structure),
        "local_numel": sum(math.prod(item["shape"]) for item in structure),
        "structure_sha256": digest(structure),
        "value_sha256": digest(values),
    }


def _optimizer_probe(optimizer: Any) -> dict:
    pending, seen, structure_rows, value_rows = [optimizer], set(), [], []
    state_entries = parameter_groups = 0
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        state_dict = getattr(current, "state_dict", None)
        snapshot = state_dict() if callable(state_dict) else None
        if not isinstance(snapshot, Mapping):
            snapshot = {
                "state": getattr(current, "state", None),
                "param_groups": getattr(current, "param_groups", None),
            }
        state = snapshot.get("state")
        groups = snapshot.get("param_groups")
        state_entries += len(state) if isinstance(state, Mapping) else 0
        parameter_groups += (
            len(groups)
            if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes, bytearray))
            else 0
        )
        structure_rows.append(_state_summary(snapshot, include_values=False))
        value_rows.append(_state_summary(snapshot, include_values=True))
        for attr in ("optimizer", "optimizers", "chained_optimizers"):
            child = getattr(current, attr, None)
            if isinstance(child, Sequence):
                pending.extend(child)
            elif child is not None:
                pending.append(child)
    return {
        "objects": len(seen),
        "state_entries": state_entries,
        "parameter_groups": parameter_groups,
        "structure_sha256": digest(structure_rows),
        "value_sha256": digest(value_rows),
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
        "structure_sha256": digest(_state_summary(state, include_values=False)),
        "value_sha256": digest(_state_summary(state, include_values=True)),
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
    plan: dict, start_ids: Sequence[int], before: Sequence[dict], after: Sequence[dict]
) -> dict:
    manifest = plan["source_manifest"]
    world = manifest["world_size"]
    expected = _rank_commitments(plan, manifest)
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
    for rank, row in enumerate(ordered_before):
        wanted = expected[rank]
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
            or row["model"]["tensors"] != wanted["model_tensor_count"]
            or row["model"]["local_numel"] != wanted["model_local_numel"]
            or row["model"]["structure_sha256"] != wanted["model_structure_sha256"]
            or row["model"]["value_sha256"] != wanted["model_value_sha256"]
            or row["optimizer"]["state_entries"] < 1
            or row["optimizer"]["parameter_groups"] < 1
            or row["optimizer"]["value_sha256"] != wanted["optimizer_value_sha256"]
            or row["scheduler"]["positive_progress_counters"] < 1
            or row["scheduler"]["value_sha256"] != wanted["scheduler_value_sha256"]
            or row["rng_sha256"] != wanted["rng_value_sha256"]
        ):
            raise ValueError("one Miles rank differs from the exact saved training state")
    return {
        "world_size": world,
        "ranks": list(range(world)),
        "restored_rollout_index": manifest["rollout_index"],
        "next_rollout_id": manifest["next_rollout_id"],
        "probe_set_sha256": digest(ordered_before),
        "rank_state_commitments_sha256": digest(expected),
        "all_rank_model_loaded": True,
        "all_rank_optimizer_loaded": True,
        "all_rank_scheduler_loaded": True,
        "all_rank_rng_loaded": True,
        "all_rank_state_commitments_match": True,
        "state_stable_across_zero_updates": True,
    }


async def _load_all_ranks(args, plan: dict) -> dict:
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
        return validate_rank_probes(plan, start_ids, before, after)
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
        result = asyncio.run(_load_all_ranks(args, plan))
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
                "source_terminal_acceptance_sha256": plan["source_terminal_acceptance"][
                    "receipt_sha256"
                ],
                "source_policy_delta_observation_sha256": plan["source_policy_delta_observation"][
                    "receipt_sha256"
                ],
                "rank_state_commitment_method": RANK_STATE_COMMITMENT_METHOD,
                **proof,
                "optimizer_updates": 0,
                "rollouts": 0,
                "verifier_calls": 0,
                "forwards": 0,
                "backwards": 0,
                "checkpoint_writes": 0,
                "wandb_events": 0,
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
