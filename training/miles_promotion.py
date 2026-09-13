"""Fail-closed dev-to-production gate for the exact Qwen3.8 Miles RL arm."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import JobsError, digest

SCHEMA = "cyber_qwen38_miles_production_promotion_v1"
PROD_NAME = "chris-q38-miles-rl-prod1"
PROD_OUTPUT = "/mnt/sfs/jobs/chris-q38-miles-rl-prod1"
PROD_DATA_ROOT = "/mnt/sfs/jobs/chris-q38-miles-rl-prod1-inputs/data"
PROD_DATA_MANIFEST = PROD_DATA_ROOT + "/manifest.json"
PROD_WANDB = {"entity": "thefleet", "project": "cyber-post-train", "run_id": PROD_NAME}
PROD_MODEL_SHA256 = "dcfdcd6ecb6661741cd3a4b24dc5af7259642c8a6824773e0de70d55d7501179"
BASE_CHECKPOINT = {
    "root": "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/torch-dist",
    "receipt_sha256": "19c8e93482530170e0f648815ab74233719e6f2b3bb7879a6564b42c3abec371",
}
PROD_KUBE_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
PROD_NAMESPACE_UID = "fd6d2fcd-687a-4257-9dba-a034bb381e6b"
EXPECTED_CANDIDATE_SHA256 = "007d60256013bc982c28f59a719b21bb3fc6afed5a123160a9c546959b2d2fce"
EXPECTED_RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "128",
    "memory_request": "1536Gi",
    "memory_limit": "2048Gi",
}
DEV3 = {
    "source_commit": "d69fd01e4b435adc5492c8aedba9cf7f3af5e40e",
    "source_plan_sha256": "245b404f9507ac2603c53969dd4506aee811e1c02ebf468543994cff76e3e95e",
    "source_request_sha256": "d2f2514a33bdf05610c5765c11efaecfcd6eea39e4297115cc39ce3c86d3f927",
    "runtime_bundle_sha256": "0e3ff1646344abb1b4be13ea064143014e94c13bc22c6c7e8c1a552f8bbe32a6",
    "api_base_url": "https://api.ft.dev.flt.build",
    "api_run_id": "6be68393-f032-40db-b640-d2d23473e85d",
    "api_run_name": "chris-q38-miles-rlreward-dev3-6be68393",
    "rayjob_uid": "803770c7-d830-4f47-8f1a-22d7e75362c3",
    "workload_uid": "a20b093e-93cd-4c26-b81c-33586d3bf6d3",
}
EXPECTED_DATA = {
    "name": PROD_NAME,
    "selection_sha256": "sha256:50d6052187ff2f5f085b9de402982d4421fba0c73bb1bd54fc4bc850516c5d68",
    "split_sha256": "sha256:c8c0083e08df55179a5acbd10602dbce5178257484b471c5ba7f4cbb04bdb35c",
    "tool_catalog_sha256": (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    ),
    "limits": {
        "context_tokens": 98304,
        "response_tokens": 81920,
        "max_tokens_per_turn": 4096,
        "max_turns": 600,
        "episode_seconds": 2400,
        "tool_seconds": 330,
        "tool_result_chars": 50000,
    },
    "rows": {"train": 59, "dev": 20},
}
BENCHMARK_ISOLATION = {
    "optimizer_split": "train",
    "fleet_dev_is_evaluation_only": True,
    "final_test_rows": 0,
    "external_benchmark_training_rows": 0,
    "external_benchmark_reward_inputs": 0,
    "external_benchmark_hpo_or_checkpoint_inputs": 0,
    "external_benchmark_retry_inputs": 0,
}
LIVE_REQUIREMENTS = [
    "prod_jobs_api_name_and_output_absent",
    "prod_output_and_submission_journal_absent",
    "prod_wandb_run_id_absent",
    "active_prod_experiment_nodes_plus_candidate_at_most_eight",
    "warning_free_one_by_eight_c1_no_requeue_preview",
]
_REF_FIELDS = {"path", "file_sha256", "receipt_sha256"}
_FIELDS = {
    "schema",
    "status",
    "candidate_run_sha256",
    "active_dev3",
    "reward_terminal",
    "native_reload",
    "production_data_manifest",
    "benchmark_isolation",
    "live_requirements",
    "sha256",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed(value: Any, schema: str | None = None) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or (schema is not None and value.get("schema") != schema)
        or value.get("sha256", "").removeprefix("sha256:")
        != digest({key: item for key, item in value.items() if key != "sha256"})
    ):
        raise ValueError("Miles production evidence is not digest-valid")
    return value


def _snapshot(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Miles production evidence path is not an exact regular file")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise ValueError("Miles production evidence changed while reading")
    if hashlib.sha256(payload).hexdigest() != expected_file_sha256.removeprefix("sha256:"):
        raise ValueError("Miles production evidence file digest changed")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Miles production evidence is not JSON") from error
    return _sealed(value)


def _reference(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    value = _snapshot(path, _sha256(path))
    return (
        {
            "path": str(path),
            "file_sha256": _sha256(path),
            "receipt_sha256": value["sha256"].removeprefix("sha256:"),
        },
        value,
    )


def _reopen(reference: Any, *, check_files: bool) -> dict[str, Any] | None:
    if (
        not isinstance(reference, dict)
        or set(reference) != _REF_FIELDS
        or not all(
            re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", str(reference.get(key, "")))
            for key in ("file_sha256", "receipt_sha256")
        )
    ):
        raise ValueError("Miles production evidence reference changed")
    path = reference.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("Miles production evidence path is not absolute")
    if not check_files:
        return None
    value = _snapshot(Path(path), reference["file_sha256"])
    if value["sha256"].removeprefix("sha256:") != reference["receipt_sha256"].removeprefix(
        "sha256:"
    ):
        raise ValueError("Miles production evidence self-digest changed")
    return value


def _exact_data(value: Any) -> None:
    _sealed(value, "cyber_miles_data_v1")
    files = value.get("files", {})
    if (
        value.get("name") != EXPECTED_DATA["name"]
        or value.get("selection_sha256") != EXPECTED_DATA["selection_sha256"]
        or value.get("split_sha256") != EXPECTED_DATA["split_sha256"]
        or value.get("tool_catalog_sha256") != EXPECTED_DATA["tool_catalog_sha256"]
        or value.get("tokenizer")
        != {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        }
        or value.get("template_sha256")
        != "sha256:38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe"
        or value.get("limits") != EXPECTED_DATA["limits"]
        or value.get("gpus") != 0
        or value.get("environment_creates") != 0
        or set(files) != {"train", "dev"}
        or any(
            not isinstance(files[split], dict)
            or files[split].get("path") != split + ".jsonl"
            or files[split].get("rows") != rows
            for split, rows in EXPECTED_DATA["rows"].items()
        )
    ):
        raise ValueError("Miles production data differs from the exact 59/20 split")


def _candidate_without_promotion(config: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in config.items() if key != "production_promotion"}


def requires_production_promotion(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    cluster = value.get("cluster", {})
    execution = value.get("execution", {})
    return (isinstance(cluster, dict) and cluster.get("target") == "prod") or (
        isinstance(execution, dict) and execution.get("cluster_target") == "prod"
    )


def _exact_candidate(config: dict[str, Any]) -> None:
    if digest(_candidate_without_promotion(config)) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("Miles production candidate differs from the exact reviewed run")


def _exact_dev3(terminal: dict[str, Any]) -> None:
    submission = _snapshot(
        Path(terminal["submission_binding"]["path"]),
        terminal["submission_binding"]["file_sha256"],
    )
    controller = _snapshot(
        Path(terminal["controller_observation"]["path"]),
        terminal["controller_observation"]["file_sha256"],
    )
    api = submission.get("api", {})
    kube = controller.get("kubernetes", {})
    if (
        terminal.get("source_run_name") != "chris-q38-miles-rlreward-dev3"
        or terminal.get("source_plan_sha256", "").removeprefix("sha256:")
        != DEV3["source_plan_sha256"]
        or terminal.get("source_request_sha256", "").removeprefix("sha256:")
        != DEV3["source_request_sha256"]
        or submission.get("source_commit") != DEV3["source_commit"]
        or submission.get("runtime_bundle_sha256", "").removeprefix("sha256:")
        != DEV3["runtime_bundle_sha256"]
        or api.get("base_url") != DEV3["api_base_url"]
        or api.get("run_id") != DEV3["api_run_id"]
        or api.get("run_name") != DEV3["api_run_name"]
        or kube.get("rayjob", {}).get("uid") != DEV3["rayjob_uid"]
        or kube.get("workload", {}).get("uid") != DEV3["workload_uid"]
    ):
        raise ValueError("Miles production proof is not the exact active dev3 run")


def validate_promotion(value: dict[str, Any], *, check_files: bool) -> dict[str, Any]:
    """Reopen the reward, native-reload and production-data chain."""
    _sealed(value, SCHEMA)
    if (
        set(value) != _FIELDS
        or value.get("status") != "qualified_for_fresh_live_checks"
        or value.get("candidate_run_sha256") != EXPECTED_CANDIDATE_SHA256
        or value.get("active_dev3") != DEV3
        or value.get("benchmark_isolation") != BENCHMARK_ISOLATION
        or value.get("live_requirements") != LIVE_REQUIREMENTS
    ):
        raise ValueError("Miles production promotion invariant changed")
    references = {
        name: _reopen(value.get(name), check_files=check_files)
        for name in ("reward_terminal", "native_reload", "production_data_manifest")
    }
    if not check_files:
        return {"candidate_run_sha256": EXPECTED_CANDIDATE_SHA256}

    from . import miles_acceptance, miles_reload_acceptance

    terminal = references["reward_terminal"]
    native_reload = references["native_reload"]
    data = references["production_data_manifest"]
    miles_acceptance.validate_terminal(terminal, check_files=True)
    miles_reload_acceptance.validate_accepted(native_reload, check_files=True)
    _exact_dev3(terminal)
    if (
        native_reload.get("source_terminal_acceptance_sha256")
        != terminal["sha256"].removeprefix("sha256:")
        or native_reload.get("source_manifest_sha256")
        != terminal["checkpoint_manifest"]["receipt_sha256"].removeprefix("sha256:")
    ):
        raise ValueError("Miles reward and native-reload chain is not cross-bound")
    _exact_data(data)
    if value["production_data_manifest"]["path"] != PROD_DATA_MANIFEST:
        raise ValueError("Miles production data path changed")
    return {
        "candidate_run_sha256": EXPECTED_CANDIDATE_SHA256,
        "dev_source_plan_sha256": DEV3["source_plan_sha256"],
        "dev_checkpoint": terminal["checkpoint_manifest"],
    }


def accept_promotion(
    *,
    reward_terminal: Path,
    native_reload: Path,
    production_data_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    """Create one promotion receipt after every immutable dev gate exists."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Miles production promotion destination already exists")
    refs = {}
    for name, path in {
        "reward_terminal": reward_terminal,
        "native_reload": native_reload,
        "production_data_manifest": production_data_manifest,
    }.items():
        refs[name], _ = _reference(path)
    value = {
        "schema": SCHEMA,
        "status": "qualified_for_fresh_live_checks",
        "candidate_run_sha256": EXPECTED_CANDIDATE_SHA256,
        "active_dev3": DEV3,
        **refs,
        "benchmark_isolation": BENCHMARK_ISOLATION,
        "live_requirements": LIVE_REQUIREMENTS,
    }
    value["sha256"] = digest(value)
    validate_promotion(value, check_files=True)
    from .miles_conversion import _write

    return _write(output, value)


def bind_production_promotion(
    config: dict[str, Any], relative_to: Path
) -> dict[str, Any] | None:
    if not requires_production_promotion(config):
        if config.get("production_promotion") is not None:
            raise ValueError("Miles production promotion cannot attach to a dev run")
        return None
    _exact_candidate(config)
    reference = config.get("production_promotion")
    if not isinstance(reference, dict) or set(reference) != _REF_FIELDS:
        raise ValueError("exact Miles production promotion receipt is required")
    path = Path(reference["path"])
    if not path.is_absolute():
        path = (relative_to / path).resolve()
    normalized = {**reference, "path": str(path)}
    receipt = _reopen(normalized, check_files=True)
    validate_promotion(receipt, check_files=True)
    return {**normalized, "receipt": receipt}


def _exact_plan(plan: dict[str, Any]) -> None:
    from . import miles

    args = plan.get("arguments", {})
    execution = plan.get("execution", {})
    checkpoint = plan.get("checkpoint", {})
    expected_args = {
        "name": PROD_NAME,
        "output_root": PROD_OUTPUT,
        "model_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "torch_dist_root": checkpoint.get("root"),
        "train_data": PROD_DATA_ROOT + "/train.jsonl",
        "dev_data": PROD_DATA_ROOT + "/dev.jsonl",
        "data_manifest": PROD_DATA_MANIFEST,
        "wandb_entity": PROD_WANDB["entity"],
        "wandb_project": PROD_WANDB["project"],
        "wandb_run_id": PROD_WANDB["run_id"],
        "model": "Qwen/Qwen3.8-27B",
        "nodes": 1,
        "gpus_per_node": 8,
        "steps": 59,
        "groups": 1,
        "samples_per_prompt": 8,
        "lr": 1e-6,
        "eval_interval": 59,
        "checkpoint_interval": 10,
        "seed": 42,
        "context_tokens": 98304,
        "response_tokens": 81920,
        "tokens_per_turn": 4096,
    }
    if (
        plan.get("run_name") != PROD_NAME
        or plan.get("output_root") != PROD_OUTPUT
        or args != expected_args
        or checkpoint.get("schema") != "cyber_miles_checkpoint_v1"
        or checkpoint.get("optimizer_steps") != 0
        or checkpoint.get("root") != BASE_CHECKPOINT["root"]
        or checkpoint.get("sha256", "").removeprefix("sha256:")
        != BASE_CHECKPOINT["receipt_sha256"]
        or checkpoint.get("image") != miles.IMAGE
        or checkpoint.get("model") != plan.get("model")
        or digest(plan.get("model")) != PROD_MODEL_SHA256
        or execution.get("image") != miles.IMAGE
        or execution.get("cluster_target") != "prod"
        or execution.get("priority") != "c1"
        or execution.get("resources") != EXPECTED_RESOURCES
        or set(execution)
        != {"image", "priority", "resources", "cluster_target", "production_promotion"}
    ):
        raise ValueError("compiled Miles production plan differs from the exact candidate")
    _exact_data(plan.get("data"))


def validate_embedded_promotion(
    plan: dict[str, Any], *, check_files: bool = True
) -> bool:
    required = requires_production_promotion(plan)
    proof = plan.get("execution", {}).get("production_promotion")
    if not required:
        if proof is not None:
            raise ValueError("Miles production promotion attached to a dev plan")
        return False
    if not isinstance(proof, dict) or set(proof) != {*_REF_FIELDS, "receipt"}:
        raise ValueError("embedded Miles production promotion is absent")
    if not isinstance(proof["receipt"], dict):
        raise ValueError("embedded Miles production promotion receipt is invalid")
    reference = {key: proof[key] for key in _REF_FIELDS}
    observed = _reopen(reference, check_files=check_files)
    if observed is not None and observed != proof["receipt"]:
        raise ValueError("embedded Miles production promotion receipt changed")
    if (
        proof["receipt"].get("sha256", "").removeprefix("sha256:")
        != proof["receipt_sha256"].removeprefix("sha256:")
    ):
        raise ValueError("embedded Miles production promotion digest changed")
    validate_promotion(proof["receipt"], check_files=check_files)
    _exact_plan(plan)
    return True


def validate_production_preview(
    plan: dict[str, Any], request: dict[str, Any], preview: dict[str, Any]
) -> dict[str, Any]:
    if not validate_embedded_promotion(plan, check_files=True):
        return {}
    from cyber_post_train.jobs import validate_preview

    rendered = validate_preview(request, preview)
    if (
        request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("priority_class") != "c1"
        or request.get("requeueIfPreempted") is not False
        or request.get("resources") != EXPECTED_RESOURCES
    ):
        raise JobsError("Miles production preview request is not exact 1x8 c1/no-requeue")
    return {
        "production_promotion": "validated",
        "rendered_nodes": rendered["nodes"],
        "effective_priority_expected": 10000,
    }


def require_live_files(plan: dict[str, Any], prepared_directory: Path) -> None:
    if not validate_embedded_promotion(plan, check_files=True):
        return
    if (prepared_directory / "SUBMISSION.jsonl").exists() or (
        prepared_directory / "SUBMISSION.jsonl"
    ).is_symlink():
        raise JobsError("Miles production submission journal already exists")
    if Path(PROD_OUTPUT).exists() or Path(PROD_OUTPUT).is_symlink():
        raise JobsError("Miles production output already exists")
    manifest = Path(PROD_DATA_MANIFEST)
    observed = _snapshot(
        manifest,
        plan["execution"]["production_promotion"]["receipt"]["production_data_manifest"][
            "file_sha256"
        ],
    )
    _exact_data(observed)


def _kubectl_json(*arguments: str) -> dict[str, Any]:
    result = subprocess.run(
        ["kubectl", "--context", PROD_KUBE_CONTEXT, *arguments, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise JobsError("production Kubernetes read-only gate failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise JobsError("production Kubernetes read-only result is invalid") from error
    if not isinstance(value, dict):
        raise JobsError("production Kubernetes read-only result is not an object")
    return value


def require_live_external(plan: dict[str, Any], client, *, wandb_api=None) -> dict[str, Any]:
    """Recheck Jobs, W&B and the eight-node ceiling immediately before POST."""
    if not validate_embedded_promotion(plan, check_files=True):
        return {}
    if any(
        not isinstance(row, dict)
        or row.get("name") == PROD_NAME
        or str(row.get("name", "")).startswith(PROD_NAME + "-")
        or row.get("run_dir") == PROD_OUTPUT
        for row in client.all_runs()
    ):
        raise JobsError("Miles production Jobs API identity/output exists or is malformed")
    if wandb_api is None:
        import wandb

        wandb_api = wandb.Api()
    try:
        observed = wandb_api.run(
            f"{PROD_WANDB['entity']}/{PROD_WANDB['project']}/{PROD_WANDB['run_id']}"
        )
    except Exception as error:  # W&B has no stable not-found exception across pins.
        message = str(error).lower()
        if "not found" not in message and "could not find run" not in message:
            raise JobsError("Miles production W&B duplicate check failed closed") from None
    else:
        if observed is not None:
            raise JobsError("Miles production W&B run ID already exists")
    namespace = _kubectl_json("get", "namespace", "fleet-train-jobs")
    if namespace.get("metadata", {}).get("uid") != PROD_NAMESPACE_UID:
        raise JobsError("production namespace identity changed")
    priority = _kubectl_json("get", "priorityclass", "c1")
    if (
        priority.get("metadata", {}).get("name") != "c1"
        or priority.get("value") != 10000
        or priority.get("preemptionPolicy") != "PreemptLowerPriority"
    ):
        raise JobsError("production c1 effective priority changed")
    pods = _kubectl_json("get", "pods", "-n", "fleet-train-jobs").get("items")
    if not isinstance(pods, list):
        raise JobsError("production Pod inventory is malformed")
    active_nodes, unscheduled = set(), 0
    for pod in pods:
        if not isinstance(pod, dict):
            raise JobsError("production Pod inventory is malformed")
        if pod.get("status", {}).get("phase") not in {"Pending", "Running"}:
            continue
        try:
            gpu = sum(
                int(container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
                for container in pod.get("spec", {}).get("containers", [])
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise JobsError("production Pod GPU inventory is malformed") from error
        if gpu:
            node = pod.get("spec", {}).get("nodeName")
            if node:
                active_nodes.add(node)
            else:
                unscheduled += 1
    active = len(active_nodes) + unscheduled
    if active + 1 > 8:
        raise JobsError("Miles production node budget would exceed eight")
    return {
        "jobs_api_absent": True,
        "wandb_absent": True,
        "active_experiment_nodes": active,
        "candidate_nodes": 1,
        "node_limit": 8,
        "effective_priority": priority["value"],
    }
