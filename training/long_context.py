"""Offline guard for an unqualified four-node Qwen3.8 capacity canary.

This validates a proposal and sanitized evidence, never submits or loads a model.
The exact historical hooks remain in Git; a CPU check cannot prove GPU fit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/runs/qwen38-262k-four-node-canary.json"
PARENT_PLAN_SHA256 = "3b9e81acb301327c40007a40ab13c3bf5c164152655fb49f96a8db52eb117690"
PARENT_RUNTIME_SHA256 = "b6b81c9876ddf8379a0837a0aaf5f0426acf8ccc4388d79bd93bad94c59e38d3"
HOOKS = {
    "chunked_sft_forward": "554cae31621e5f98eda3c0ad1fae3d64cac8af73f35086b494189784cb47792d",
    "checkpointed_qwen35_gdn_rule": "b27d5d2b2b76ff00b58dbe2442ed5d1af032e64b65ca7a47c42ee8301f72781a",
    "chunked_qwen35_mlp_forward": "2fcc9e9e4206ae8a141ba248b9af8f2ff39f92ddb7f2f561f877be7dc2b70f2b",
    "chunked_qwen35_rmsnorm_forward": "97d07b8ea509208727fb26c33840098bd9b1412efaadc87de3961a2be888e53d",
    "chunked_qwen35_rmsnorm_gated_forward": "af9f01ad825870a06486942a1a221ddfd2e484f921b0d752c4ea1ad1b27e3a57",
    "grouped_qwen35_text_forward": "e75bda3427314c0bd1dbe5d54fe438c470817212b39dc115d9bcf95a22acf9ec",
    "install_chunked_sft_worker": "f11637a5fbbf39bf689afd246ee2eb617b67675b2fd41af53dc20ba9d32a624a",
}
RECIPE = {
    "full_weight": True,
    "nodes": 4,
    "gpus_per_node": 8,
    "world_size": 32,
    "batch_size": 32,
    "microbatch_per_gpu": 1,
    "gradient_accumulation": 1,
    "max_length": 262144,
    "sequence_parallel_size": 1,
    "layer_checkpoint_group_size": 1,
    "outer_checkpoint_reentrant": True,
    "gdn_chunk_tokens": 512,
    "lm_head_chunk_tokens": 1024,
    "mlp_chunk_tokens": 1024,
    "rmsnorm_chunk_tokens": 1024,
    "cpu_parameter_offload": False,
    "cpu_optimizer_offload": False,
    "learning_rate": 3e-6,
    "epochs": 1,
    "max_steps": 4,
    "pause_after_optimizer_step": 1,
    "eval_interval": 0,
    "seed": 20260916,
    "checkpoint_interval": 1,
    "keep_checkpoints": 2,
    "reload_optimizer_steps": 0,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_spec(spec: dict) -> None:
    """Check immutable intent, not execution or capacity."""
    _require(set(spec) == {"schema", "status", "accepted", "submission_authorized", "name", "output_root", "parent", "model", "data", "length_audit", "recipe", "cluster", "example_gate"}, "unexpected canary spec fields")
    _require(spec["schema"] == "qwen38_262k_four_node_capacity_canary_v1", "wrong canary schema")
    _require(spec["status"] == "unqualified_hypothesis" and spec["accepted"] is False and spec["submission_authorized"] is False, "unproven acceptance or submission claim")
    _require(spec["name"] == "chris-q38-t3k262-4n-can-v1" and spec["output_root"] == "/mnt/sfs/jobs/chris-q38-t3k262-4n-can-v1", "wrong create-once identity")
    parent = spec["parent"]
    _require(set(parent) == {"run", "api_run_id", "rayjob_uid", "plan_commit", "plan_path", "plan_sha256", "runtime_commit", "runtime_path", "runtime_sha256", "hook_ast_sha256"}, "unexpected historical parent fields")
    _require(parent["run"] == "chris-q38-t3k262-can-v12" and parent["plan_sha256"] == PARENT_PLAN_SHA256, "wrong historical parent")
    _require(parent["plan_commit"] == "c908d3a828d070c6b27611fc388b1e7e3b4049dd" and parent["plan_path"] == "configs/qualification/qwen38-teacher3k-262k-v12-parent-plan.json", "wrong historical plan locator")
    _require(parent["api_run_id"] == "a421f0a3-5ed2-48cb-99ee-f830fa395cb7" and parent["rayjob_uid"] == "36f915ca-d882-46b3-9d8b-5e4e2bfddae7", "wrong historical run identity")
    _require(parent["runtime_commit"] == "45c04d709f855e20d931456233b85f427558525f" and parent["runtime_path"] == "training/sft_runtime.py" and parent["runtime_sha256"] == PARENT_RUNTIME_SHA256 and parent["hook_ast_sha256"] == HOOKS, "wrong historical runtime")
    _require(spec["model"] == {"repo": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "weight_manifest_sha256": "06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"}, "wrong model")
    _require(spec["data"] == {"path": "/mnt/sfs/jobs/chris-q38-study-corpora-v1/teacher3k-262k-v1/capacity-v5/train.parquet", "sha256": "2359c54e5c5cd756761a0f6e8c250ec8b32b87c8f84f0888252ddac932cfc5ec", "rows": 112, "purpose": "capacity_only_not_scientific_training"}, "wrong capacity corpus")
    _require(spec["length_audit"] == {"path": "docs/evidence/qwen38-262k-capacity-lengths-20260925.json", "file_sha256": "81ea8ddc237cf3ea66ae614e09bacde5f7aabe831ca76b5286a96711c1d2b42e"}, "wrong length audit")
    _require(spec["recipe"] == RECIPE and RECIPE["world_size"] == RECIPE["nodes"] * RECIPE["gpus_per_node"] and RECIPE["batch_size"] == RECIPE["world_size"] * RECIPE["microbatch_per_gpu"] * RECIPE["gradient_accumulation"], "wrong four-node recipe")
    cluster = spec["cluster"]
    _require(cluster == {"image": "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4", "priority_class": "c1", "queue_priority": "q1", "root_annotation": {"fleet.ai/failure-alerts": "off"}, "shutdown_after_finish": True, "ttl_seconds_after_finish": 0}, "wrong cluster binding")
    _require(spec["example_gate"] == {"sequence_tokens": 262144, "minimum_nonpadding_tokens": 250000, "source_kind": "real_training_row"}, "wrong real-example gate")


def validate_historical_hooks(spec: dict, *, cwd: Path = ROOT) -> None:
    """Confirm the parent plan and runtime match immutable historical Git blobs."""
    parent = spec["parent"]
    plan = subprocess.run(
        ["git", "show", f"{parent['plan_commit']}:{parent['plan_path']}"],
        cwd=cwd, check=True, capture_output=True,
    ).stdout
    canonical = json.dumps(json.loads(plan), sort_keys=True, separators=(",", ":")).encode()
    _require(_sha256(canonical) == parent["plan_sha256"], "historical parent plan changed")
    blob = subprocess.run(
        ["git", "show", f"{parent['runtime_commit']}:{parent['runtime_path']}"],
        cwd=cwd, check=True, capture_output=True,
    ).stdout
    _require(_sha256(blob) == parent["runtime_sha256"], "historical runtime bytes changed")
    # The whole-blob digest binds implementation bytes. Python's ast.dump format
    # changes across interpreters, so the older per-hook digests are provenance,
    # not portable evidence to recompute under a newer local Python.
    functions = {node.name for node in ast.parse(blob).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    _require(HOOKS.keys() <= functions, "historical long-context hooks missing")


def validate_example(spec: dict, example: dict) -> None:
    """Check a sanitized row summary; its producer must separately verify the row."""
    gate = spec["example_gate"]
    _require(example.get("schema") == "qwen38_262k_real_row_summary_v1", "wrong example summary")
    _require(example.get("source_kind") == gate["source_kind"] and example.get("dataset_sha256") == spec["data"]["sha256"], "example not bound to training corpus")
    _require(example.get("sequence_tokens") == gate["sequence_tokens"], "example does not exercise full sequence")
    nonpadding = example.get("nonpadding_tokens")
    supervised = example.get("supervised_tokens")
    _require(type(nonpadding) is int and gate["minimum_nonpadding_tokens"] <= nonpadding <= gate["sequence_tokens"], "no real near-max context evidence")
    _require(type(supervised) is int and 0 < supervised <= nonpadding, "no supervised target")
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", str(example.get("row_sha256", "")))), "missing source row digest")


def validate_length_audit(spec: dict) -> None:
    """Bind the aggregate read-only data observation; this is not row proof."""
    path = ROOT / spec["length_audit"]["path"]
    blob = path.read_bytes()
    _require(_sha256(blob) == spec["length_audit"]["file_sha256"], "length audit digest mismatch")
    audit = json.loads(blob)
    _require(
        audit.get("schema") == "qwen38_262k_capacity_length_observation_v1"
        and audit.get("train_parquet_sha256") == spec["data"]["sha256"]
        and audit.get("rows") == spec["data"]["rows"]
        and audit.get("max_input_tokens") == spec["recipe"]["max_length"]
        and audit.get("rows_with_at_least_250000_input_tokens") == 61
        and audit.get("input_length_equals_mask_length_equals_token_count") is True
        and audit.get("loss_mask_sum_equals_target_token_count") is True,
        "length audit does not bind the capacity corpus",
    )


def validate_rendered_job(spec: dict, job: dict) -> None:
    """Inspect a server-rendered root RayJob, not a request flag or Pod label."""
    _require(job.get("kind") == "RayJob", "preview is not a RayJob")
    _require(job.get("metadata", {}).get("annotations", {}).get("fleet.ai/failure-alerts") == "off", "root failed-job alert opt-out missing")
    _require(job.get("metadata", {}).get("labels", {}).get("kueue.x-k8s.io/priority-class") == "q1", "wrong queue priority")
    ray = job.get("spec", {})
    _require(ray.get("shutdownAfterJobFinishes") is True and ray.get("ttlSecondsAfterFinished") == 0, "job will not release promptly")
    groups = ray.get("rayClusterSpec", {})
    head, workers = groups.get("headGroupSpec", {}), groups.get("workerGroupSpecs", [])
    _require(isinstance(workers, list) and len(workers) == 1 and workers[0].get("replicas") == 3, "preview is not four nodes")
    for group in (head, workers[0]):
        pod = group.get("template", {}).get("spec", {})
        _require(pod.get("priorityClassName") == spec["cluster"]["priority_class"], "wrong effective pod priority")
        containers = pod.get("containers", [])
        _require(isinstance(containers, list) and len(containers) == 1, "unexpected container layout")
        container = containers[0]
        _require(container.get("image") == spec["cluster"]["image"], "wrong immutable image")
        resources = container.get("resources", {})
        _require(resources.get("requests", {}).get("nvidia.com/gpu") == 8 and resources.get("limits", {}).get("nvidia.com/gpu") == 8, "wrong GPU request")


def preflight(spec: dict, example: dict, rendered_job: dict) -> dict:
    validate_spec(spec)
    validate_historical_hooks(spec)
    validate_length_audit(spec)
    validate_example(spec, example)
    validate_rendered_job(spec, rendered_job)
    return {"metadata_checks": "passed", "aggregate_lengths_verified": True, "real_row_verified": False, "gpu_qualified": False, "checkpoint_reload_proven": False, "submission_authorized": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--example-summary", type=Path, required=True)
    parser.add_argument("--rendered-job", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(*(json.loads(path.read_text()) for path in (args.spec, args.example_summary, args.rendered_job)))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
