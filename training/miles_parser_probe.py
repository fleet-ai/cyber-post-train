"""Dev-only CUDA parser probe for a prepared Miles RL plan.

The probe calls the same native argument parser as training, then exits before
Ray, rollout engines, task access, weight loading, reward, or optimization.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path, PurePosixPath

from cyber_post_train.jobs import bundled_request, digest

from . import miles
from .miles_conversion import _write
from .miles_training import (
    NATIVE_DRIVER_SHA256,
    native_args,
)
from .miles_training import (
    RUNTIME_FILES as TRAINING_RUNTIME_FILES,
)
from .miles_training import (
    SCHEMA as TRAINING_SCHEMA,
)
from .miles_training import (
    job_request as training_request,
)
from .rl_runtime import hard_deadline

SCHEMA = "cyber_miles_native_parser_probe_v1"
RECEIPT_SCHEMA = "cyber_miles_native_parser_probe_result_v1"
PREFLIGHT_SCHEMA = "cyber_miles_native_parser_probe_cpu_preflight_v1"
DEADLINE_SECONDS = 300
RESOURCES = {
    "cpu_request": "4",
    "cpu_limit": "8",
    "memory_request": "16Gi",
    "memory_limit": "32Gi",
}
RUNTIME_FILES = (*TRAINING_RUNTIME_FILES, "training/miles_parser_probe.py")


class ProbeRejected(RuntimeError):
    """A clean contract rejection that keeps the training gate closed."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _runtime() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = {name: (root / name).read_text() for name in RUNTIME_FILES}
    packages = (
        "training/__init__.py",
        "evals/__init__.py",
        "evals/fleet/__init__.py",
        "cyber_post_train/__init__.py",
    )
    files.update({name: "" for name in packages})
    return files


def _sfs_output(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("parser-probe output must be a string")
    path = PurePosixPath(value)
    if (
        path.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(path.parts) < 5
        or ".." in path.parts
        or str(path) != value
    ):
        raise ValueError("parser-probe output must be a canonical jobs path")
    return value


def _run_root(plan: dict) -> Path:
    """Bind receipts to the exact directory mounted by the Jobs API."""
    value = os.environ.get("RUN_DIR")
    if value != plan.get("output_root"):
        raise RuntimeError("RUN_DIR does not match parser-probe output")
    root = Path(value)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("parser-probe output is not a direct directory")
    return root


def _expected(source: dict) -> dict:
    args = source["arguments"]
    return {
        "source_run_name": source["run_name"],
        "source_output_root": source["output_root"],
        "source_checkpoint_root": source["checkpoint"]["root"],
        "tokens_per_turn": args["tokens_per_turn"],
        "context_tokens": args["context_tokens"],
        "response_tokens": args["response_tokens"],
        "nodes": args["nodes"],
        "gpus_per_node": args["gpus_per_node"],
        "steps": args["steps"],
        "global_batch_size": args["groups"] * args["samples_per_prompt"],
    }


def compile_probe(source: dict, *, source_file_sha256: str, name: str, output_root: str) -> dict:
    """Bind one already-prepared dev Miles plan to a one-GPU parser-only check."""
    request = training_request(source)
    args = miles.MilesConfig(**source["arguments"])
    if (
        source.get("schema") != TRAINING_SCHEMA
        or source.get("execution", {}).get("cluster_target") != "dev"
        or request["priority_class"] != "c1"
        or request["workers"] != 1
        or request["gpus_per_worker"] != 8
        or args.tokens_per_turn != 8192
        or args.nodes != 1
        or args.gpus_per_node != 8
        or args.steps != 1
    ):
        raise ValueError("source is not the bounded Dev8 8192-token Miles plan")
    output_root = _sfs_output(output_root)
    if output_root == source["output_root"]:
        raise ValueError("parser probe and training output must be distinct")
    plan = {
        "schema": SCHEMA,
        "run_name": name,
        "output_root": output_root,
        "source_plan": source,
        "source_plan_sha256": "sha256:" + digest(source),
        "source_plan_file_sha256": source_file_sha256,
        "expected": _expected(source),
        "native_driver_sha256": NATIVE_DRIVER_SHA256,
        "deadline_seconds": DEADLINE_SECONDS,
        "execution": {
            "cluster_target": "dev",
            "image": miles.IMAGE,
            "priority": "c1",
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": RESOURCES,
        },
        "operations": {
            "rollouts": 0,
            "task_requests": 0,
            "verifier_calls": 0,
            "reward_batches": 0,
            "optimizer_steps": 0,
            "checkpoints_written": 0,
            "weights_loaded": 0,
        },
    }
    plan["runtime_sha256"] = digest(_runtime())
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> dict:
    source = plan.get("source_plan")
    if not isinstance(source, dict):
        raise ValueError("parser-probe source plan is missing")
    source_request = training_request(source)
    expected = _expected(source)
    if (
        set(plan)
        != {
            "schema",
            "run_name",
            "output_root",
            "source_plan",
            "source_plan_sha256",
            "source_plan_file_sha256",
            "expected",
            "native_driver_sha256",
            "deadline_seconds",
            "execution",
            "operations",
            "runtime_sha256",
        }
        or plan["schema"] != SCHEMA
        or plan["source_plan_sha256"] != "sha256:" + digest(source)
        or not isinstance(plan["source_plan_file_sha256"], str)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", plan["source_plan_file_sha256"])
        or plan["expected"] != expected
        or expected["tokens_per_turn"] != 8192
        or plan["native_driver_sha256"] != NATIVE_DRIVER_SHA256
        or plan["deadline_seconds"] != DEADLINE_SECONDS
        or plan["runtime_sha256"] != digest(_runtime())
        or _sfs_output(plan["output_root"]) != plan["output_root"]
        or plan["output_root"] == source["output_root"]
        or plan["execution"]
        != {
            "cluster_target": "dev",
            "image": miles.IMAGE,
            "priority": "c1",
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": RESOURCES,
        }
        or plan["operations"]
        != {
            "rollouts": 0,
            "task_requests": 0,
            "verifier_calls": 0,
            "reward_batches": 0,
            "optimizer_steps": 0,
            "checkpoints_written": 0,
            "weights_loaded": 0,
        }
        or source_request["workers"] != 1
        or source_request["gpus_per_worker"] != 8
        or source_request["priority_class"] != "c1"
        or source_request["requeueIfPreempted"] is not False
    ):
        raise ValueError("Miles parser-probe plan drift")
    return source


def job_request(plan: dict) -> dict:
    validate_plan(plan)
    files = _runtime()
    files["probe-plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": plan["run_name"],
            "title": plan["run_name"] + " Miles parser-only dev check",
            "run_dir": plan["output_root"],
            "image": plan["execution"]["image"],
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": plan["execution"]["resources"],
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "PYTHONPATH": "/root/Megatron-LM",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "WANDB_MODE": "disabled",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        "training.miles_parser_probe",
        ["--plan", "probe-plan.json", "--sha256", digest(plan)],
    )


def preflight(plan: dict) -> dict:
    import torch

    if torch.cuda.is_available():
        raise ValueError("parser-probe preflight is CPU-only")
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("parser-probe output already exists")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "source_plan_sha256": plan["source_plan_sha256"],
        "native_argument_builder_tokens_per_turn": plan["expected"]["tokens_per_turn"],
        "native_parser_checked": False,
        "planned_gpu_probe_workers": 1,
        "planned_gpu_probe_gpus_per_worker": 1,
        "training_authorized": False,
    }


def run(plan: dict, *, run_root: Path | None = None) -> dict:
    import torch

    source = validate_plan(plan)
    run_root = _run_root(plan) if run_root is None else run_root
    source_output = Path(source["output_root"])
    if source_output.exists():
        raise ProbeRejected("source_training_output_exists")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ProbeRejected("exactly_one_cuda_device_not_visible")
    if torch.distributed.is_initialized():
        raise ProbeRejected("distributed_state_initialized_before_parser")
    with hard_deadline(plan["deadline_seconds"], "native_parser"):
        parsed = native_args(source)
    observed = {
        "fleet_max_tokens_per_turn": parsed.fleet_max_tokens_per_turn,
        "rollout_max_context_len": parsed.rollout_max_context_len,
        "rollout_max_response_len": parsed.rollout_max_response_len,
        "actor_num_nodes": parsed.actor_num_nodes,
        "actor_num_gpus_per_node": parsed.actor_num_gpus_per_node,
        "num_rollout": parsed.num_rollout,
        "global_batch_size": parsed.global_batch_size,
        "load_matches_source_checkpoint": parsed.load == source["checkpoint"]["root"],
        "ref_load_matches_source_checkpoint": parsed.ref_load == source["checkpoint"]["root"],
    }
    expected = plan["expected"]
    if observed != {
        "fleet_max_tokens_per_turn": expected["tokens_per_turn"],
        "rollout_max_context_len": expected["context_tokens"],
        "rollout_max_response_len": expected["response_tokens"],
        "actor_num_nodes": expected["nodes"],
        "actor_num_gpus_per_node": expected["gpus_per_node"],
        "num_rollout": expected["steps"],
        "global_batch_size": expected["global_batch_size"],
        "load_matches_source_checkpoint": True,
        "ref_load_matches_source_checkpoint": True,
    }:
        raise ProbeRejected("native_parser_contract_mismatch")
    if torch.distributed.is_initialized() or source_output.exists():
        raise ProbeRejected("parser_probe_crossed_execution_boundary")
    result = {
        "schema": RECEIPT_SCHEMA,
        "status": "validated",
        "plan_sha256": "sha256:" + digest(plan),
        "source_plan_sha256": plan["source_plan_sha256"],
        "source_plan_file_sha256": plan["source_plan_file_sha256"],
        "native_driver_sha256": plan["native_driver_sha256"],
        "runtime_sha256": plan["runtime_sha256"],
        "cuda_device_count": 1,
        "native_parser_observation": observed,
        "distributed_initialized_before_or_after": False,
        **plan["operations"],
    }
    return _write(run_root / "PARSER_VALIDATED.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if digest(plan) != args.sha256:
        raise SystemExit(2)
    run_root = _run_root(plan)
    try:
        result = run(plan, run_root=run_root)
    except ProbeRejected as exc:
        result = _write(
            run_root / "PARSER_REJECTED.json",
            {
                "schema": RECEIPT_SCHEMA,
                "status": "rejected",
                "reason": exc.reason,
                "plan_sha256": "sha256:" + digest(plan),
                **plan["operations"],
            },
        )
    except BaseException as exc:
        _write(
            run_root / "FAILED.json",
            {
                "schema": RECEIPT_SCHEMA,
                "status": "failed",
                "error_class": type(exc).__name__,
                "plan_sha256": "sha256:" + digest(plan),
                **plan["operations"],
            },
        )
        raise SystemExit(1) from None
    print(json.dumps({"status": result["status"], "sha256": result["sha256"]}))


if __name__ == "__main__":
    main()
