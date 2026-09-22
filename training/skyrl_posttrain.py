"""Fail-closed evidence, checkpoint, and BF16 handoff for native SkyRL RL.

This module deliberately does not use the SFT checkpoint-plan validator.  A
native SkyRL run has different scientific evidence: authoritative Fleet reward
receipts, complete grouped rollouts, native optimizer telemetry, and a native
FSDP checkpoint.  Sealing is CPU-only, reads no trainer log, changes no source,
and publishes one create-once manifest only after every binding agrees.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as fleet

from . import skyrl
from .checkpoints import _validate_names, checkpoint_files, receipt
from .miles_conversion import _hash
from .rl_episode import _validate as validate_episode_config
from .rl_runtime import sealed
from .sft_runtime import write_receipt

MANIFEST_SCHEMA = "cyber_native_skyrl_rl_checkpoint_manifest_v1"
EXPORT_SCHEMA = "cyber_native_skyrl_rl_hf_export_v1"

FORBIDDEN_TERMINAL_MARKERS = (
    "FAILED.json",
    "REJECTED.json",
    "NATIVE_FAILURE.json",
    "NATIVE_REJECTED.json",
)


def _hex(value: str) -> str:
    value = value.removeprefix("sha256:") if isinstance(value, str) else ""
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid SHA-256")
    return value


def _json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or indirect evidence file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("evidence file must contain one object")
    return value


def _validate_plan(plan: dict) -> skyrl.SkyRLConfig:
    from .skyrl_training import SCHEMA as HISTORICAL_SCHEMA

    if not isinstance(plan, dict) or plan.get("schema") not in {
        HISTORICAL_SCHEMA,
        "cyber_skyrl_prod9_training_v1",
    }:
        raise ValueError("not a native SkyRL training plan")
    required = {
        "schema",
        "run_name",
        "output_root",
        "model",
        "data",
        "arguments",
        "native_overrides",
        "native_sources",
        "runtime_sha256",
        "execution",
    }
    optional = {"qualification"}
    if plan.get("schema") == "cyber_skyrl_prod9_training_v1":
        optional.add("prod9_runtime")
    if not required <= set(plan) or set(plan) - required - optional:
        raise ValueError("native SkyRL plan fields changed")
    args = skyrl.SkyRLConfig(**plan["arguments"])
    args.validate()
    sealed(plan["data"], "cyber_skyrl_data_v1")
    model = plan["model"]
    execution = plan["execution"]
    if plan.get("schema") == "cyber_skyrl_prod9_training_v1":
        from . import skyrl_prod9_training

        if plan.get("prod9_runtime") != skyrl_prod9_training._binding() or plan.get(
            "runtime_sha256"
        ) != digest(skyrl_prod9_training._runtime()):
            raise ValueError("prod9 SkyRL runtime binding disagrees")
    if (
        args.model != "Qwen/Qwen3.8-27B"
        or model.get("repo") != args.model
        or model.get("root") != args.model_root
        or plan["run_name"] != args.name
        or plan["output_root"] != args.output_root
        or plan["data"].get("name") != args.name
        or plan["data"].get("files", {}).get("train", {}).get("rows") != args.train_rows
        or plan["data"].get("files", {}).get("dev", {}).get("rows") != args.dev_rows
        or plan["native_overrides"] != skyrl.overrides(args)
        or execution.get("priority") != "c1"
        or type(args.nodes) is not int
        or args.nodes not in (1, 2)
        or _hex(plan.get("runtime_sha256", "")) != plan["runtime_sha256"]
    ):
        raise ValueError("native SkyRL plan bindings disagree")
    return args


def _source_files(plan: dict) -> dict[str, Path]:
    args, files = skyrl.SkyRLConfig(**plan["arguments"]), {}
    base = Path(plan["model"]["root"])
    for item in plan["model"].get("files", []):
        name = item.get("path")
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe model inventory path")
        path = base / name
        if _hash(path) != _hex(item.get("sha256", "")):
            raise ValueError("base model differs from its frozen inventory")
        files["model/" + name] = path
    if not files:
        raise ValueError("base model inventory is empty")
    data = Path(args.data_manifest).parent
    expected_data = {
        "manifest.json": plan["data"]["sha256"],
        "train.jsonl": plan["data"]["files"]["train"]["sha256"],
        "dev.jsonl": plan["data"]["files"]["dev"]["sha256"],
        "split.json": plan["data"]["split_sha256"],
        "task-set.json": plan["data"]["selection_sha256"],
    }
    for name, expected in expected_data.items():
        path = data / name
        value = _json(path) if name.endswith(".json") else None
        if value is not None and name != "manifest.json" and value.get("sha256") != expected:
            raise ValueError("private data receipt self-digest differs from the plan")
        if name == "manifest.json" and value != plan["data"]:
            raise ValueError("staged data manifest differs from the plan")
        # JSON receipts bind their canonical object, while their file bytes are
        # retained separately below. Train/dev are byte-bound directly.
        if name.endswith(".jsonl") and _hash(path) != _hex(expected):
            raise ValueError("private training data differs from the plan")
        files["data/" + name] = path
    return files


def _inventory(files: dict[str, Path]) -> tuple[dict, dict]:
    specs, stats = {}, {}
    for name, path in sorted(files.items()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("inventory contains a missing or indirect file")
        before = path.stat()
        sha = _hash(path)
        after = path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(before, field) != getattr(after, field) for field in fields)
            or after.st_size <= 0
        ):
            raise ValueError("inventory file changed or is empty")
        specs[name] = {"bytes": after.st_size, "sha256": sha}
        stats[name] = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    return specs, stats


def _rehash(files: dict[str, Path], specs: dict, stats: dict) -> None:
    if set(files) != set(specs) or set(files) != set(stats):
        raise ValueError("inventory membership changed")
    for name, path in files.items():
        stat = path.stat()
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if (
            current != stats[name]
            or stat.st_size != specs[name]["bytes"]
            or _hash(path) != specs[name]["sha256"]
        ):
            raise ValueError("source changed during stable full-file rehash")


def _reject_aliases(*groups: dict[str, Path]) -> None:
    seen = set()
    for files in groups:
        for path in files.values():
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                raise ValueError("checkpoint, source, and evidence inventories contain an alias")
            seen.add(identity)


def _expected_batches(args: skyrl.SkyRLConfig) -> set[tuple[str, int]]:
    return (
        {("train", step) for step in range(1, args.steps + 1)}
        | {("eval", 0), ("eval", args.steps)}
        | {("eval", step) for step in range(1, args.steps + 1) if step % args.eval_interval == 0}
    )


def _episode(
    plan: dict,
    batch_dir: Path,
    index: int,
    batch: dict,
    source_rows: dict[str, list[dict]],
) -> tuple[float, str, dict[str, Path]]:
    directory = batch_dir / f"episode-{index}"
    if not directory.is_dir() or (directory / "failure.json").exists():
        raise ValueError("counted rollout is missing or failed")
    accepted_path = directory / "ACCEPTED.json"
    accepted = _json(accepted_path)
    if accepted.get("sha256") != fleet.digest_without(accepted, "sha256"):
        raise ValueError("rollout acceptance digest mismatch")
    expected_files = {
        "binding.json",
        "instance.json",
        "conversation.json",
        "reward.json",
        "cleanup.json",
        "recording.json",
    }
    if set(accepted.get("files", {})) != expected_files:
        raise ValueError("rollout acceptance payload is incomplete")
    files = {"ACCEPTED.json": accepted_path}
    for name, expected in accepted["files"].items():
        path = directory / name
        if fleet.sha256(path.read_bytes()) != expected:
            raise ValueError("rollout payload digest mismatch")
        files[name] = path
    binding, instance_receipt, reward, cleanup = (
        _json(directory / name)
        for name in (
            "binding.json",
            "instance.json",
            "reward.json",
            "cleanup.json",
        )
    )
    validate_episode_config(binding)
    if binding.get("config_sha256") != fleet.digest_without(binding, "config_sha256"):
        raise ValueError("rollout binding digest mismatch")
    trajectory = batch["trajectory_ids"][index]
    split = "dev" if batch["phase"] == "eval" else "train"
    source = json.loads(source_rows[split][int(trajectory[0])]["cyber_config_json"])
    batch_binding = {
        "phase": batch["phase"],
        "global_step": batch["global_step"],
        "trajectory_ids": batch["trajectory_ids"],
    }
    exact_run_id = f"{plan['run_name']}-{batch_dir.name}-{index}"
    for key in ("task", "model", "environment", "verifier", "authority", "execution", "rl"):
        if binding.get(key) != source.get(key):
            raise ValueError("rollout task/runtime binding differs from private source data")
    if binding.get("run_id") != exact_run_id or binding.get("native_batch") != batch_binding:
        raise ValueError("rollout run/batch identity mismatch")
    instance = accepted.get("instance_id")
    execution = accepted.get("verifier_execution_id")
    evidence_run_id = instance_receipt.get("evidence_run_id")
    if (
        reward.get("task_key") != binding["task"]["key"]
        or reward.get("task_version_id") != binding["task"]["version_id"]
        or reward.get("instance_id") != instance
        or reward.get("verifier_execution_id") != execution
        or accepted.get("task_version_id") != binding["task"]["version_id"]
        or accepted.get("config_sha256") != binding["config_sha256"]
        or instance_receipt.get("instance_id") != instance
        or cleanup.get("instance_id") != instance
        or cleanup.get("instance_closed") is not True
        or cleanup.get("possible_instance_leak") is not False
    ):
        raise ValueError("reward, verifier, instance, cleanup, and rollout bindings disagree")
    value = reward.get("reward")
    if (
        isinstance(value, bool)
        or type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("authoritative reward is invalid")
    if not isinstance(execution, str) or not execution:
        raise ValueError("authoritative verifier execution ID is missing")
    attestation = reward.get("direct_authority_attestation")
    attestation = attestation if isinstance(attestation, dict) else {}
    context = attestation.get("context")
    activity = attestation.get("activity")
    shadow = attestation.get("shadow")
    direct = shadow.get("direct_verifier") if isinstance(shadow, dict) else None
    minimization = attestation.get("data_minimization")
    minimization_keys = {
        "components_included",
        "diagnostics_included",
        "evidence_payloads_included",
        "prompts_included",
        "traces_included",
        "flags_included",
    }
    if (
        attestation.get("schema_version") != fleet.DIRECT_AUTHORITY_ATTESTATION_SCHEMA
        or context
        != {
            "task_key": binding["task"]["key"],
            "task_version_id": binding["task"]["version_id"],
            "instance_id": instance,
            "evidence_run_id": evidence_run_id,
            "verifier_version_id": binding["verifier"]["version_id"],
            "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
        }
        or not isinstance(activity, dict)
        or activity.get("result_schema_version") != "cyber_verification_result_v3"
        or activity.get("reward") != float(value)
        or activity.get("task_version_id") != binding["task"]["version_id"]
        or activity.get("verifier_execution_id") != execution
        or not isinstance(shadow, dict)
        or shadow.get("mode") != "authoritative"
        or shadow.get("status") != "authoritative"
        or shadow.get("match") is not True
        or shadow.get("production_execution_id") != execution
        or not isinstance(direct, dict)
        or direct.get("status") != "authoritative"
        or direct.get("match") is not True
        or direct.get("execution_id") != execution
        or direct.get("verifier_contract_version")
        != binding["authority"]["required_cyber_contract"]["verifier_contract"]
        or direct.get("context_schema_version") != "cyber_verification_context_v1"
        or not isinstance(minimization, dict)
        or set(minimization) != minimization_keys
        or any(value is not False for value in minimization.values())
    ):
        raise ValueError("Fleet direct-authority reward attestation is incomplete or mismatched")
    return (
        float(value),
        execution,
        {str(path.relative_to(Path(plan["output_root"]))): path for path in files.values()},
    )


def _rollout_evidence(plan: dict, args: skyrl.SkyRLConfig) -> tuple[dict, dict[str, Path]]:
    # This rechecks exact staged rows and split/task receipts without returning
    # any prompt, flag, answer, or trajectory content in the public manifest.
    from .skyrl_training import check_artifacts

    rows = check_artifacts(plan)
    root = Path(plan["output_root"])
    expected, seen = _expected_batches(args), set()
    rewards, executions, evidence_files = [], set(), {}
    batch_root = root / "episodes/batches"
    for directory in batch_root.iterdir():
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("unexpected native batch entry")
        if any((directory / marker).exists() for marker in ("FAILED.json", "REJECTED.json")):
            raise ValueError("counted native batch has a failure/rejection marker")
        path = directory / "COLLECTED.json"
        batch = _json(path)
        sealed(batch, "cyber_skyrl_batch_v1")
        key = (batch.get("phase"), batch.get("global_step"))
        trajectories = batch.get("trajectory_ids")
        expected_count = (
            args.groups * args.samples_per_prompt if key[0] == "train" else args.dev_rows
        )
        if (
            key not in expected
            or key in seen
            or batch.get("data_sha256") != plan["data"]["sha256"]
            or batch.get("optimizer_step_verified") is not False
            or not isinstance(trajectories, list)
            or len(trajectories) != expected_count
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or not pair[0].isdecimal()
                or type(pair[1]) is not int
                for pair in trajectories
            )
        ):
            raise ValueError("native batch receipt differs from the plan")
        groups = {}
        for uid, repetition in trajectories:
            groups.setdefault(uid, []).append(repetition)
        repetitions = args.samples_per_prompt if key[0] == "train" else 1
        if (
            len(groups) != (args.groups if key[0] == "train" else args.dev_rows)
            or any(sorted(values) != list(range(repetitions)) for values in groups.values())
            or any(int(uid) >= len(rows["dev" if key[0] == "eval" else "train"]) for uid in groups)
        ):
            raise ValueError("native rollout group is incomplete or duplicated")
        batch_files = {str(path.relative_to(root)): path}
        for index in range(len(trajectories)):
            reward, execution, files = _episode(plan, directory, index, batch, rows)
            rewards.append((key[0], reward))
            if execution in executions:
                raise ValueError("verifier execution ID reused across rollouts")
            executions.add(execution)
            batch_files.update(files)
        evidence_files.update(batch_files)
        seen.add(key)
    if seen != expected:
        raise ValueError("planned native train/development batch is missing")
    train_rewards = [value for phase, value in rewards if phase == "train"]
    if len(train_rewards) != args.steps * args.groups * args.samples_per_prompt or min(
        train_rewards
    ) == max(train_rewards):
        raise ValueError("real train rewards are missing variation")
    summary = {
        "batches": len(seen),
        "train_batches": args.steps,
        "train_rollouts": len(train_rewards),
        "development_rollouts": len(rewards) - len(train_rewards),
        "reward_min": min(train_rewards),
        "reward_max": max(train_rewards),
        "unique_verifier_executions": len(executions),
        "all_instances_closed": True,
        "private_payloads_included": False,
    }
    return summary, evidence_files


def _update_evidence(root: Path, step: int) -> tuple[dict, dict[str, Path]]:
    path = root / "metrics.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("native scalar stream is missing")
    by_step = {number: [] for number in range(1, step + 1)}
    for raw in path.read_bytes().splitlines():
        value = json.loads(raw)
        current = value.pop("optimizer_step", None)
        timestamp = value.pop("time", None)
        if current in by_step and type(timestamp) in (int, float) and math.isfinite(timestamp):
            by_step[current].append(value)
        elif current not in (0,):
            raise ValueError("native scalar step is outside the training plan")
        if any(
            not isinstance(key, str)
            or isinstance(number, bool)
            or type(number) not in (int, float)
            or not math.isfinite(number)
            for key, number in value.items()
        ):
            raise ValueError("native scalar stream contains nonfinite telemetry")
    required = {
        "loss": lambda key: "loss" in key.lower(),
        "kl": lambda key: "kl" in key.lower(),
        "entropy": lambda key: "entropy" in key.lower(),
        "gradient": lambda key: "grad" in key.lower(),
    }
    for number, records in by_step.items():
        keys = {key for record in records for key in record}
        if not records or any(not any(test(key) for key in keys) for test in required.values()):
            raise ValueError(
                f"optimizer step {number} lacks finite loss/KL/entropy/gradient evidence"
            )
    return {
        "optimizer_steps": step,
        "all_steps_have_finite_loss_kl_entropy_gradient": True,
        "metrics_sha256": _hash(path),
    }, {"metrics.jsonl": path}


def _changed_parameters(plan: dict, root: Path, world_size: int) -> tuple[int, int]:
    import torch

    from .export import FROZEN_MTP_KEYS, _load_tensor, _safetensor_layout, reassemble

    base = Path(plan["model"]["root"])
    layout, _ = _safetensor_layout(base)
    try:
        states = [
            torch.load(
                root / "policy" / f"model_world_size_{world_size}_rank_{rank}.pt",
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            for rank in range(world_size)
        ]
    except Exception as exc:
        raise ValueError("native model checkpoint payload is malformed") from exc
    keys = set(states[0])
    if (
        any(set(state) != keys for state in states)
        or set(layout) - keys != set(FROZEN_MTP_KEYS)
        or keys - set(layout)
    ):
        raise ValueError("native checkpoint tensor layout differs from the exact base")
    changed = 0
    for key in sorted(keys):
        trained = reassemble([state[key] for state in states], layout[key]["shape"])
        base_value = _load_tensor(base, layout[key]["shard"], key)
        if not torch.equal(trained, base_value):
            changed += 1
    if changed == 0:
        raise ValueError("checkpoint contains no independently verified parameter update")
    return changed, len(keys)


def seal_checkpoint(plan: dict, output: Path, *, progress=None) -> dict:
    """Create one public RL checkpoint manifest from an exact completed run."""
    import torch

    if torch.cuda.is_available():
        raise ValueError("RL checkpoint sealing does not need a GPU")
    args = _validate_plan(plan)
    root = Path(plan["output_root"])
    checkpoint = root / "checkpoints" / f"global_step_{args.steps}"
    if output.resolve().is_relative_to(checkpoint.resolve()):
        raise ValueError("never write a manifest inside its source checkpoint")
    if output.exists() or output.is_symlink():
        raise FileExistsError("RL checkpoint manifest already exists")
    if any((root / marker).exists() for marker in FORBIDDEN_TERMINAL_MARKERS):
        raise ValueError("RL run has a conflicting failure/rejection marker")
    terminal_path = root / "NATIVE_TRAINING_COMPLETE.json"
    terminal = _json(terminal_path)
    if terminal.get("sha256", "").removeprefix("sha256:") != digest(
        {k: v for k, v in terminal.items() if k != "sha256"}
    ):
        raise ValueError("native terminal receipt digest mismatch")
    expected_batches = len(_expected_batches(args))
    if (
        terminal.get("status") != "native_loop_returned"
        or terminal.get("plan_sha256") != digest(plan)
        or terminal.get("checkpoint_global_step") != args.steps
        or terminal.get("completed_batches") != expected_batches
        or terminal.get("optimizer_update_independently_verified") is not False
        or terminal.get("checkpoint_reload_verified") is not False
    ):
        raise ValueError("native terminal receipt differs from the plan")
    pointer = root / "checkpoints/latest_ckpt_global_step.txt"
    saves = sorted(
        set(range(args.checkpoint_interval, args.steps + 1, args.checkpoint_interval))
        | {args.steps}
    )
    retained = saves[-args.keep_checkpoints :]
    present = sorted(
        int(path.name.removeprefix("global_step_"))
        for path in (root / "checkpoints").glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdecimal()
    )
    if pointer.read_text().strip() != str(args.steps) or present != retained:
        raise ValueError("native checkpoint cadence/retention/pointer differs from the plan")
    world_size = args.nodes * 8
    for retained_step in retained:
        retained_files = checkpoint_files(
            root / "checkpoints" / f"global_step_{retained_step}", world_size
        )
        try:
            retained_trainer = torch.load(
                retained_files["trainer_state.pt"], map_location="cpu", weights_only=False
            )
            retained_sampler = torch.load(
                retained_files["data.pt"], map_location="cpu", weights_only=False
            )
        except Exception as exc:
            raise ValueError("retained native checkpoint metadata is malformed") from exc
        retained_cursor = (retained_step - 1) % (args.train_rows // args.groups) + 1
        if (
            retained_trainer.get("global_step") != retained_step
            or retained_sampler.get("_num_yielded") != retained_cursor
        ):
            raise ValueError("retained checkpoint step/cursor differs from the plan")
    checkpoint_paths = checkpoint_files(checkpoint, world_size)
    source_paths = _source_files(plan)
    checkpoint_specs, checkpoint_stats = _inventory(checkpoint_paths)
    source_specs, source_stats = _inventory(source_paths)
    terminal_files = {"NATIVE_TRAINING_COMPLETE.json": terminal_path}
    rollouts, rollout_files = _rollout_evidence(plan, args)
    updates, metric_files = _update_evidence(root, args.steps)
    evidence_paths = {**terminal_files, **rollout_files, **metric_files}
    _reject_aliases(checkpoint_paths, source_paths, evidence_paths)
    evidence_specs, evidence_stats = _inventory(evidence_paths)
    try:
        trainer = torch.load(
            checkpoint_paths["trainer_state.pt"], map_location="cpu", weights_only=False
        )
        sampler = torch.load(checkpoint_paths["data.pt"], map_location="cpu", weights_only=False)
        for name, path in checkpoint_paths.items():
            if name.startswith(("policy/optim_", "policy/extra_state_")):
                value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                if not isinstance(value, dict) or not value:
                    raise ValueError("native optimizer/RNG checkpoint payload is malformed")
    except Exception as exc:
        raise ValueError("native checkpoint metadata/optimizer/RNG payload is malformed") from exc
    batches_per_epoch = args.train_rows // args.groups
    expected_cursor = (args.steps - 1) % batches_per_epoch + 1
    if trainer.get("global_step") != args.steps or sampler.get("_num_yielded") != expected_cursor:
        raise ValueError("native checkpoint step or sampler cursor differs from the plan")
    changed, trained = _changed_parameters(plan, checkpoint, world_size)
    updates.update(changed_parameter_tensors=changed, trained_parameter_tensors=trained)
    if progress:
        progress(
            "rehash", len(checkpoint_paths), sum(v["bytes"] for v in checkpoint_specs.values())
        )
    # A second complete hash pass catches corruption, replacement and source
    # drift before the create-once public handoff is written.
    _rehash(checkpoint_paths, checkpoint_specs, checkpoint_stats)
    _rehash(source_paths, source_specs, source_stats)
    _rehash(evidence_paths, evidence_specs, evidence_stats)
    result = {
        "schema": MANIFEST_SCHEMA,
        "source_plan_sha256": digest(plan),
        "source_plan": copy.deepcopy(plan),
        "terminal_receipt_sha256": terminal["sha256"],
        "checkpoint_path": str(checkpoint),
        "optimizer_step": args.steps,
        "world_size": world_size,
        "checkpoint_interval": args.checkpoint_interval,
        "expected_retained_steps": retained,
        "sampler_batches_in_epoch": expected_cursor,
        "rollout_evidence": rollouts,
        "update_evidence": updates,
        "files": checkpoint_specs,
        "source_files": source_specs,
        "evidence_files": evidence_specs,
        "total_bytes": sum(v["bytes"] for v in checkpoint_specs.values()),
        "source_inputs_unchanged": True,
        "optimizer_update_verified": True,
        "gpu_reload_verified": False,
    }
    write_receipt(output, result)
    return receipt(output)


def verify_manifest(manifest: dict, *, check_files: bool = True) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("not a native SkyRL RL checkpoint manifest")
    if manifest.get("receipt_sha256") != digest(
        {k: v for k, v in manifest.items() if k != "receipt_sha256"}
    ):
        raise ValueError("RL checkpoint manifest digest mismatch")
    plan = manifest.get("source_plan")
    args = _validate_plan(plan)
    step, world_size = args.steps, args.nodes * 8
    checkpoint = Path(plan["output_root"]) / "checkpoints" / f"global_step_{step}"
    retained = sorted(
        set(range(args.checkpoint_interval, step + 1, args.checkpoint_interval)) | {step}
    )[-args.keep_checkpoints :]
    cursor = (step - 1) % (args.train_rows // args.groups) + 1
    if (
        manifest.get("source_plan_sha256") != digest(plan)
        or manifest.get("checkpoint_path") != str(checkpoint)
        or manifest.get("optimizer_step") != step
        or manifest.get("world_size") != world_size
        or manifest.get("checkpoint_interval") != args.checkpoint_interval
        or manifest.get("expected_retained_steps") != retained
        or manifest.get("sampler_batches_in_epoch") != cursor
        or manifest.get("source_inputs_unchanged") is not True
        or manifest.get("optimizer_update_verified") is not True
        or manifest.get("gpu_reload_verified") is not False
        or manifest.get("total_bytes")
        != sum(v["bytes"] for v in manifest.get("files", {}).values())
        or manifest.get("rollout_evidence", {}).get("train_rollouts")
        != step * args.groups * args.samples_per_prompt
        or manifest.get("rollout_evidence", {}).get("reward_max", 0)
        <= manifest.get("rollout_evidence", {}).get("reward_min", 0)
        or manifest.get("update_evidence", {}).get("optimizer_steps") != step
        or manifest.get("update_evidence", {}).get("changed_parameter_tensors", 0) <= 0
    ):
        raise ValueError("RL checkpoint manifest bindings disagree")
    for inventory in (
        manifest.get("files"),
        manifest.get("source_files"),
        manifest.get("evidence_files"),
    ):
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError("RL checkpoint manifest inventory is incomplete")
        for name, spec in inventory.items():
            path = Path(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) != name
                or set(spec) != {"bytes", "sha256"}
            ):
                raise ValueError("unsafe RL checkpoint inventory entry")
            if (
                type(spec["bytes"]) is not int
                or spec["bytes"] <= 0
                or _hex(spec["sha256"]) != spec["sha256"]
            ):
                raise ValueError("invalid RL checkpoint inventory entry")
    _validate_names(set(manifest["files"]), world_size)
    if not check_files:
        return
    current = checkpoint_files(checkpoint, world_size)
    for name, path in current.items():
        spec = manifest["files"].get(name)
        if spec is None or path.stat().st_size != spec["bytes"] or _hash(path) != spec["sha256"]:
            raise ValueError("RL checkpoint payload changed after sealing")
    if set(current) != set(manifest["files"]):
        raise ValueError("RL checkpoint payload is partial or has extra files")
    for name, path in _source_files(plan).items():
        spec = manifest["source_files"].get(name)
        if spec is None or path.stat().st_size != spec["bytes"] or _hash(path) != spec["sha256"]:
            raise ValueError("RL model/data source changed after sealing")
    if set(_source_files(plan)) != set(manifest["source_files"]):
        raise ValueError("RL model/data source inventory is partial or has extra files")
    root = Path(plan["output_root"])
    evidence = {name: root / name for name in manifest["evidence_files"]}
    for name, path in evidence.items():
        spec = manifest["evidence_files"][name]
        if path.stat().st_size != spec["bytes"] or _hash(path) != spec["sha256"]:
            raise ValueError("RL terminal/rollout/update evidence changed after sealing")


def export_checkpoint(
    manifest_path: Path, expected_sha256: str, output: Path, *, progress=None
) -> dict:
    """Create a complete BF16 export after RL-native manifest verification."""
    from .export import _export_verified

    return _export_verified(
        manifest_path,
        expected_sha256,
        output,
        verify_manifest=verify_manifest,
        receipt_schema=EXPORT_SCHEMA,
        code_files=("skyrl_posttrain.py",),
        progress=progress,
    )
