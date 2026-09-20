"""Create-once Fleet V1 Miles reward-acquisition canary.

This is a narrow adapter around Fleet's maintained FTI/Miles trainer image.  It
stages exact, train-only task-version rows and preserves each verifier execution
ID in Miles sample metadata.  FTI still owns generation, grading and training.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import struct
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "cyber_fti_v1_rl_canary_v1"
PRODUCTION_SCHEMA = "cyber_fti_v1_rl_production_v1"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "0ca02d9bc920e104d67c249172eb0e8b0cc34fc409e738f76f2efee44610dc59"
)
FTI_VERSION = "0.9.2"
FTI_SOURCE_COMMIT = "0b1af5684310ee244bf7bdb0028e5ef78c08098b"
RUN_SH_SHA256 = "7fba883763eead8ab8ad7411c59e06f25dc19c81a970c6f225a6ad3cad65e0f8"
RUN_FLEET_SHA256 = "5efb5a99c9e07bf555703bd627b54651667f4d654b3010b4641059c4a5bab6c4"
EVIDENCE_SCHEMA = "cyber_fti_v1_reward_evidence_v1"
SITECUSTOMIZE = (
    "import os\n"
    "try:\n"
    " from training.fti_v1_training import install_safe_wandb\n"
    " install_safe_wandb()\n"
    "except BaseException:\n"
    " os.write(2,b'safe W&B environment hook failed\\n')\n"
    " os._exit(78)\n"
)
RESOURCES = {
    "cpu_request": "48",
    "cpu_limit": "64",
    "memory_request": "1500Gi",
    "memory_limit": "2650Gi",
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sfs_run_dir(value: str) -> None:
    path = PurePosixPath(value)
    if (
        path.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(path.parts) != 5
        or str(path) != value
        or ".." in path.parts
    ):
        raise ValueError("run_dir must be one create-once directory under /mnt/sfs/jobs")


def _read_bound_json(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    if root.resolve() not in path.parents or file_sha256(path) != binding["file_sha256"]:
        raise ValueError("bound data manifest changed")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("bound data manifest must be an object")
    return value


def validate_plan(plan: dict[str, Any], *, root: Path) -> dict[str, Any]:
    # Work on a JSON copy: production expands its digest-bound split selector
    # into exact task-version rows without mutating the reviewed source plan.
    plan = json.loads(json.dumps(plan))
    if set(plan) != {
        "schema",
        "name",
        "run_dir",
        "trainer",
        "data",
        "episode",
        "optimization",
        "wandb",
        "cluster",
        "acceptance",
    } or plan.get("schema") not in {SCHEMA, PRODUCTION_SCHEMA}:
        raise ValueError("unsupported FTI V1 plan")
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", plan["name"]):
        raise ValueError("invalid create-once run name")
    _require_sfs_run_dir(plan["run_dir"])
    if PurePosixPath(plan["run_dir"]).name != plan["name"]:
        raise ValueError("run name and output directory must agree")

    trainer = plan["trainer"]
    if trainer != {
        "model": "Qwen/Qwen3.8-27B",
        "recipe": "qwen3.8-27b-256k",
        "image": IMAGE,
        "fti_version": FTI_VERSION,
        "fti_source_commit": FTI_SOURCE_COMMIT,
        "run_sh_sha256": RUN_SH_SHA256,
        "run_fleet_sha256": RUN_FLEET_SHA256,
    }:
        raise ValueError("trainer identity drift")

    data = plan["data"]
    if set(data) != {"study_split", "eligible_inventory", "tasks"}:
        raise ValueError("unexpected data-plan fields")
    split = _read_bound_json(root, data["study_split"])
    eligible = _read_bound_json(root, data["eligible_inventory"])
    training_tasks = [
        {"task_key": row["task_key"], "task_version_id": row["task_version_id"]}
        for row in split["training_split"]["tasks"]
    ]
    split_rows = {
        (row["task_key"], row["task_version_id"]): row["split"]
        for row in split["training_split"]["tasks"]
    }
    eligible_rows = {(row["task_key"], row["task_version_id"]) for row in eligible["task_versions"]}
    tasks = data["tasks"]
    if plan["schema"] == PRODUCTION_SCHEMA:
        if tasks == "all_locked_train":
            tasks = training_tasks
            data["tasks"] = tasks
        elif tasks != training_tasks:
            raise ValueError("production RL must use every task in the locked training split")
        counts = {name: 0 for name in ("train", "dev", "final_test")}
        for row in split["tasks"]:
            if row.get("split") not in counts:
                raise ValueError("unexpected study-split assignment")
            counts[row["split"]] += 1
        if counts != {"train": 59, "dev": 20, "final_test": 10} or len(tasks) != 59:
            raise ValueError("locked 59/20/10 study split drift")
    elif not isinstance(tasks, list):
        raise ValueError("canary tasks must be exact task-version rows")
    identities = [(row["task_key"], row["task_version_id"]) for row in tasks]
    if not tasks or len(identities) != len(set(identities)):
        raise ValueError("RL plan needs unique exact task versions")
    # The inventory's split is provenance from the earlier rollout campaign.
    # Study Split A is the experiment assignment authority; eligibility only
    # answers whether the exact version is runnable and correctly graded.
    if any(split_rows.get(item) != "train" or item not in eligible_rows for item in identities):
        raise ValueError("RL tasks must be eligible members of the locked training split")

    episode = plan["episode"]
    expected_episode = {
        "max_turns": 160,
        "max_tokens_per_turn": 4096,
        "max_concurrent_envs": 8 if plan["schema"] == SCHEMA else 32,
        "request_timeout_s": 120,
        "ready_timeout_s": 600,
        "episode_timeout_s": 2400,
        "tool_timeout_s": 330,
        "grade_timeout_s": 900,
        "ttl_seconds": 7200,
        "tool_output_max_chars": 50000,
        "vision": False,
        "pass_conversation_to_verifier": False,
    }
    if episode != expected_episode:
        raise ValueError("episode contract drift")

    optim = plan["optimization"]
    expected_optimization = (
        {
            "optimizer_steps": 1,
            "prompt_groups": 1,
            "samples_per_prompt": 8,
            "learning_rate": 1e-6,
            "checkpoint_interval": 1,
            "seed": 20260913,
        }
        if plan["schema"] == SCHEMA
        else {
            "optimizer_steps": 8,
            "prompt_groups": 8,
            "samples_per_prompt": 8,
            "learning_rate": 1e-6,
            "checkpoint_interval": 1,
            "seed": 20260914,
        }
    )
    if optim != expected_optimization or (
        plan["schema"] == SCHEMA and len(tasks) != optim["prompt_groups"]
    ):
        raise ValueError("optimizer recipe drift")
    if plan["wandb"] != {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "run_id": plan["name"],
    }:
        raise ValueError("W&B identity drift")
    if plan["cluster"] != {
        "nodes": 4,
        "gpus_per_node": 8,
        "priority": "c1",
        "topology_mode": "preferred",
        "resources": RESOURCES,
    }:
        raise ValueError("cluster shape or priority drift")
    required = {
        "exact_task_versions",
        "finite_rewards",
        "verifier_execution_ids",
        "mixed_reward_group",
        "optimizer_step",
        "finite_nonzero_gradient",
        "checkpoint",
        "all_instances_released",
    }
    if set(plan["acceptance"]) != required or any(plan["acceptance"].values()):
        raise ValueError("acceptance gates must begin false")
    return plan


def task_rows(plan: dict[str, Any]) -> bytes:
    rows = []
    for task in plan["data"]["tasks"]:
        rows.append(
            {
                "messages": [{"role": "user", "content": "Run the versioned Fleet task."}],
                "label": task["task_key"],
                "metadata": {
                    "task_key": task["task_key"],
                    "task_version_id": task["task_version_id"],
                    "fleet": plan["episode"],
                },
            }
        )
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
    )


def native_arguments(plan: dict[str, Any]) -> list[str]:
    optim, wandb = plan["optimization"], plan["wandb"]
    extra = " ".join(
        (
            f"--num-rollout {optim['optimizer_steps']}",
            f"--save-interval {optim['checkpoint_interval']}",
            f"--lr {optim['learning_rate']}",
            f"--seed {optim['seed']}",
            f"--rollout-seed {optim['seed']}",
            "--custom-generate-function-path training.fti_v1_training.generate",
            f"--wandb-team {wandb['entity']}",
            f"--wandb-project {wandb['project']}",
            f"--wandb-group {plan['name']}",
            f"--wandb-run-id {wandb['run_id']}",
        )
    )
    return [
        "/opt/fleet/run.sh",
        "--model-name",
        "qwen3.8-27b-256k",
        "--platform",
        "v1",
        "--mode",
        "normal",
        "--num-nodes",
        "4",
        "--num-gpus-per-node",
        "8",
        "--rollout-batch-size",
        str(optim["prompt_groups"]),
        "--n-samples-per-prompt",
        str(optim["samples_per_prompt"]),
        "--output-dir",
        "/mnt/sfs/jobs",
        "--extra-args",
        extra,
    ]


def job_request(plan: dict[str, Any], *, root: Path) -> dict[str, Any]:
    from cyber_post_train.jobs import bundled_request

    plan = validate_plan(plan, root=root)
    files = {
        "sitecustomize.py": SITECUSTOMIZE,
        "training/__init__.py": "",
        "training/fti_v1_training.py": Path(__file__).read_text(),
        "plan.json": json.dumps(plan, sort_keys=True, separators=(",", ":")),
    }
    plan_sha256 = digest(plan)
    request = {
        "name": plan["name"],
        "title": (
            "Qwen3.8 Fleet V1 RL reward canary"
            if plan["schema"] == SCHEMA
            else "Qwen3.8 Fleet V1 RL production run"
        ),
        "run_dir": plan["run_dir"],
        "image": IMAGE,
        "command": "placeholder",
        "workers": 4,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "topology_mode": "preferred",
        "requeueIfPreempted": False,
        "failureAlerts": False,
        "resources": RESOURCES,
        "secrets": ["fleet-api", "wandb-api"],
        "image_pull_secrets": ["ecr-pull"],
        "env": {
            "V1_DATASET_DIR": plan["run_dir"] + "/data",
            "PYTHONPATH": plan["run_dir"] + "/.runtime:/root/Megatron-LM",
            "WANDB_ENTITY": "thefleet",
            "WANDB_MODE": "online",
            "WANDB_DISABLE_CODE": "true",
            "WANDB_CONSOLE": "off",
            "CYBER_PLAN_SHA256": plan_sha256,
            "CYBER_REWARD_EVIDENCE_DIR": plan["run_dir"] + "/evidence/verifier-executions",
            "CYBER_REWARD_HMAC_KEY": plan["run_dir"] + "/.private/reward-hmac.key",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        },
    }
    return bundled_request(
        request,
        files,
        "training.fti_v1_training",
        ["--plan", "plan.json", "--sha256", plan_sha256],
    )


_CLIENT_RECORDING = None


def _environment_only_wandb_args(*_args, **_kwargs) -> str:
    """Enable W&B without copying its environment credential into argv."""
    return "--use-wandb --disable-wandb-random-suffix" if os.environ.get("WANDB_API_KEY") else ""


def install_safe_wandb() -> None:
    """Patch the exact Miles argv builder before it can serialize secrets."""
    import miles.utils.external_utils.command_utils as command_utils

    command_utils.get_default_wandb_args = _environment_only_wandb_args


def _client_recording_with_execution_ids():
    """Keep FTI behavior, but retain the async verifier job UUID it drops."""
    global _CLIENT_RECORDING
    if _CLIENT_RECORDING is not None:
        return _CLIENT_RECORDING
    from fti.fleet import GradeResult
    from fti.miles.v1 import client_recording
    from fti.miles.v1.common import TaskSession, numeric_reward

    class EvidenceTaskSession(TaskSession):
        verifier_execution_id: str | None = None

        def grade(self, answer, reset_ack=None, close_final_step=False):
            if self.closed.is_set() or self.instance is None or self.verifier_version_id is None:
                raise RuntimeError("verifier called before a live bound task session")
            result = self.client.execute_verifier(
                self.verifier_version_id,
                self.instance.instance_id,
                final_answer=answer,
                conversation=self.conversation if self.cfg.pass_conversation_to_verifier else None,
                timeout_s=self.cfg.grade_timeout_s,
            )
            if result.get("success") is not True:
                raise RuntimeError("registered verifier execution failed")
            execution_id = (
                result.get("verifier_execution_id") or result.get("job_id") or result.get("id")
            )
            try:
                self.verifier_execution_id = str(uuid.UUID(str(execution_id)))
            except (ValueError, TypeError, AttributeError):
                raise RuntimeError("registered verifier omitted its execution UUID") from None
            verdict = result["result"]
            value = verdict["result"] if isinstance(verdict, dict) else verdict
            return GradeResult(reward=numeric_reward(value))

        def details(self):
            return {**super().details(), "verifier_execution_id": self.verifier_execution_id}

    client_recording.TaskSession = EvidenceTaskSession
    _CLIENT_RECORDING = client_recording
    return client_recording


def _uuid(value: Any, field: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"invalid {field}") from None


def _finite_reward(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("verifier reward is not numeric")
    reward = float(value)
    if not math.isfinite(reward):
        raise ValueError("verifier reward is not finite")
    return reward


def _evidence_payload(result: Any, expected: dict[str, Any], key: bytes) -> tuple[str, bytes]:
    samples = result.samples if isinstance(result.samples, list) else [result.samples]
    records = []
    for sample in samples:
        fleet = (sample.metadata or {}).get("fleet_v1")
        if not isinstance(fleet, dict) or fleet.get("graded") is not True:
            raise ValueError("completed generation omitted its grading metadata")
        reward = _finite_reward(fleet.get("reward"))
        if _finite_reward(sample.reward) != reward:
            raise ValueError("sample and verifier rewards disagree")
        if fleet.get("cleanup_error") is not None:
            raise ValueError("Fleet instance cleanup was not confirmed")
        task_key = fleet.get("task_key")
        if not isinstance(task_key, str) or not task_key:
            raise ValueError("invalid task key")
        record = {
            "task_key": task_key,
            "task_version_id": _uuid(fleet.get("task_version_id"), "task version UUID"),
            "verifier_version_id": _uuid(fleet.get("verifier_version_id"), "verifier version UUID"),
            "verifier_execution_id": _uuid(
                fleet.get("verifier_execution_id"), "verifier execution UUID"
            ),
            "reward": reward,
        }
        if record["task_key"] != expected.get("task_key") or record["task_version_id"] != _uuid(
            expected.get("task_version_id"), "expected task version UUID"
        ):
            raise ValueError("graded task identity differs from the requested exact version")
        records.append(record)
    if not records:
        raise ValueError("completed generation returned no samples")
    first = records[0]
    if any(record != first for record in records[1:]):
        raise ValueError("segments disagree on verifier evidence")
    plan_sha256 = os.environ["CYBER_PLAN_SHA256"]
    if not re.fullmatch(r"[a-f0-9]{64}", plan_sha256):
        raise ValueError("invalid runtime plan digest")
    reward_fingerprint = hmac.new(
        key, struct.pack("!d", first["reward"]), hashlib.sha256
    ).hexdigest()
    body = {
        "schema": EVIDENCE_SCHEMA,
        "plan_sha256": plan_sha256,
        "task_key": first["task_key"],
        "task_version_id": first["task_version_id"],
        "verifier_version_id": first["verifier_version_id"],
        "verifier_execution_id": first["verifier_execution_id"],
        "reward_class": "finite_numeric",
        "reward_fingerprint": reward_fingerprint,
        "instance_cleanup_confirmed": True,
    }
    body["receipt_sha256"] = digest(body)
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    return first["verifier_execution_id"], payload


def _spool_reward_evidence(result: Any, expected: dict[str, Any]) -> None:
    key_path = Path(os.environ["CYBER_REWARD_HMAC_KEY"])
    evidence_dir = Path(os.environ["CYBER_REWARD_EVIDENCE_DIR"])
    key = key_path.read_bytes()
    if len(key) != 32:
        raise ValueError("invalid private reward fingerprint key")
    execution_id, payload = _evidence_payload(result, expected, key)
    path = evidence_dir / f"{execution_id}.json"
    try:
        _write_once(path, payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("verifier evidence identity collision") from None


def _abort_unspooled(result: Any) -> Any:
    from miles.utils.types import Sample

    samples = result.samples if isinstance(result.samples, list) else [result.samples]
    for sample in samples:
        sample.reward = None
        sample.status = Sample.Status.ABORTED
        sample.metadata = {**(sample.metadata or {}), "evidence_error": "unavailable"}
    return result


async def generate(input):
    result = await _client_recording_with_execution_ids().generate(input)
    samples = result.samples if isinstance(result.samples, list) else [result.samples]
    # Upstream returns ABORTED samples for recoverable episode errors. They do
    # not carry rewards and must not create a verifier receipt.
    if not samples or not any(
        isinstance((sample.metadata or {}).get("fleet_v1"), dict)
        and sample.metadata["fleet_v1"].get("graded") is True
        for sample in samples
    ):
        return result
    try:
        _spool_reward_evidence(result, input.sample.metadata or {})
    except Exception:
        # Fail closed without crashing the Ray job: Miles discards ABORTED
        # samples, so no update can consume a reward that lacks durable proof.
        return _abort_unspooled(result)
    return result


def _add_generate_arguments(parser):
    return _client_recording_with_execution_ids().generate.add_arguments(parser)


generate.add_arguments = _add_generate_arguments


def _write_once(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def run(plan: dict[str, Any], expected_digest: str) -> None:
    if digest(plan) != expected_digest:
        raise ValueError("runtime plan digest mismatch")
    run_dir = Path(os.environ["RUN_DIR"])
    if str(run_dir) != plan["run_dir"]:
        raise ValueError("runtime output identity drift")
    if os.environ.get("CYBER_PLAN_SHA256") != expected_digest:
        raise ValueError("runtime evidence-plan identity drift")
    expected_evidence_dir = run_dir / "evidence" / "verifier-executions"
    expected_key_path = run_dir / ".private" / "reward-hmac.key"
    if (
        Path(os.environ.get("CYBER_REWARD_EVIDENCE_DIR", "")) != expected_evidence_dir
        or Path(os.environ.get("CYBER_REWARD_HMAC_KEY", "")) != expected_key_path
    ):
        raise ValueError("runtime evidence paths drift")

    import fti
    from fti.trainers.miles import run_fleet

    if fti.__version__ != FTI_VERSION:
        raise ValueError("FTI version drift")
    if file_sha256(Path("/opt/fleet/run.sh")) != RUN_SH_SHA256:
        raise ValueError("trainer entrypoint bytes drift")
    if file_sha256(Path(run_fleet.__file__)) != RUN_FLEET_SHA256:
        raise ValueError("native Miles recipe bytes drift")
    recipe = run_fleet._RECIPES.get("qwen3.8-27b-256k")
    if not recipe or (recipe.backend, recipe.max_context_len, recipe.max_response_len) != (
        "megatron",
        262144,
        245760,
    ):
        raise ValueError("Qwen native-context recipe drift")

    data_dir = run_dir / "data"
    data_dir.mkdir(mode=0o700)
    expected_evidence_dir.mkdir(parents=True, mode=0o700)
    expected_key_path.parent.mkdir(mode=0o700)
    _write_once(expected_key_path, os.urandom(32))
    rows = task_rows(plan)
    _write_once(data_dir / "train.jsonl", rows)
    launch = {
        "schema": "cyber_fti_v1_rl_launch_v1",
        "plan_sha256": expected_digest,
        "dataset_sha256": hashlib.sha256(rows).hexdigest(),
        "task_version_count": len(plan["data"]["tasks"]),
        "trainer_image": IMAGE,
        "fti_version": FTI_VERSION,
    }
    _write_once(
        run_dir / "LAUNCH.json",
        json.dumps(launch, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    os.execvpe("bash", ["bash", *native_arguments(plan)], os.environ)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    run(plan, args.sha256)


if __name__ == "__main__":
    main()
