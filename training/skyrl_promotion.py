"""Fail-closed promotion contract for the exact Qwen3.8 SkyRL production arm.

Dev qualification is immutable; duplicate, output, W&B, preview, and node-budget
checks are intentionally live and must be repeated by the submitter immediately
before its one create-once POST.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from cyber_post_train.jobs import JobsError, digest

SCHEMA = "cyber_qwen38_skyrl_production_promotion_v1"
PROD_NAME = "chris-q38-rl-prod1"
PROD_OUTPUT = "/mnt/sfs/jobs/chris-q38-rl-prod1"
PROD_DATA_ROOT = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rl-prod1/data"
PROD_DATA_MANIFEST = PROD_DATA_ROOT + "/manifest.json"
PROD_WANDB = {"entity": "thefleet", "project": "cyber-post-train", "run_id": PROD_NAME}
PROD_KUBE_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
PROD_NAMESPACE_UID = "fd6d2fcd-687a-4257-9dba-a034bb381e6b"
EXCLUDED_INFERENCE_ENDPOINTS = [
    {
        "name": "inference-chris-cyber-glm53-dedicated-v1",
        "uid": "e8c2d55d-f4b6-45e9-9f0e-30516be8cd7a",
    },
    {
        "name": "inference-chris-cyber-qwen38-27b-dedicated-v1",
        "uid": "47db8bac-005b-43ce-992e-889a8273d50a",
    },
    {
        "name": "inference-glm-5-3-9691a990",
        "uid": "6699a52e-4d66-45a0-b441-11d4948d573a",
    },
    {
        "name": "inference-qwen3-8-27b-66fc7c05",
        "uid": "c4dff72c-3c99-4b31-adee-a13ba153676a",
    },
]
EXPECTED_CANDIDATE_SHA256 = "e0bd20a352c76a0c25a402cfac3a7098e719642f25018535a0f2ccb97c6188ac"
STALE_PRIVACY_INCOMPLETE_IMAGE_SHA256 = (
    "e48827529b1cf5fafa153b2aed1b774c2eec86905baf5ccb62b36300533e252b"
)
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
EXPECTED_ARGUMENTS = {
    "name": PROD_NAME,
    "output_root": PROD_OUTPUT,
    "model": "Qwen/Qwen3.8-27B",
    "model_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
    "train_data": PROD_DATA_ROOT + "/train.jsonl",
    "dev_data": PROD_DATA_ROOT + "/dev.jsonl",
    "data_manifest": PROD_DATA_MANIFEST,
    "train_rows": 59,
    "dev_rows": 20,
    "wandb_entity": "thefleet",
    "wandb_project": "cyber-post-train",
    "wandb_run_id": PROD_NAME,
    "nodes": 1,
    "steps": 59,
    "groups": 1,
    "samples_per_prompt": 8,
    "lr": 1e-6,
    "eval_interval": 59,
    "checkpoint_interval": 10,
    "keep_checkpoints": 3,
    "seed": 42,
    "context_tokens": 98304,
    "response_tokens": 81920,
    "tokens_per_turn": 4096,
    "max_turns": 600,
    "engine_start_timeout_seconds": 1800,
    "engine_cleanup_timeout_seconds": 300,
}
LIVE_REQUIREMENTS = [
    "prod_jobs_api_name_and_run_id_absent",
    "prod_output_root_absent",
    "prod_data_manifest_exact_and_unchanged",
    "wandb_run_id_absent",
    "submission_journal_absent",
    "active_experiment_nodes_plus_candidate_at_most_eight",
    "warning_free_exact_c1_no_requeue_nonroot_preview",
]
_REF_FIELDS = {"path", "file_sha256", "receipt_self_sha256"}
_FIELDS = {
    "schema",
    "status",
    "candidate_run_sha256",
    "qualified_image",
    "source_plan_sha256",
    "dev8_terminal",
    "reward_terminal",
    "checkpoint_manifest",
    "reload_accepted",
    "production_data_manifest",
    "runtime_user",
    "benchmark_isolation",
    "live_requirements",
    "excluded_inference_endpoints",
    "sha256",
}


def _sha256(path: Path) -> str:
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def _sealed(value: object, schema: str | None = None) -> dict:
    if (
        not isinstance(value, dict)
        or (schema is not None and value.get("schema") != schema)
        or value.get("sha256")
        != digest({key: item for key, item in value.items() if key != "sha256"})
    ):
        raise ValueError("production promotion evidence is not digest-valid")
    return value


def _ref(value: object, *, check_files: bool) -> dict | None:
    if not isinstance(value, dict) or set(value) != _REF_FIELDS:
        raise ValueError("production promotion evidence reference changed")
    if not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", str(value.get("file_sha256", ""))):
        raise ValueError("production promotion file digest is invalid")
    if not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", str(value.get("receipt_self_sha256", ""))):
        raise ValueError("production promotion receipt digest is invalid")
    if not check_files:
        return None
    path = Path(str(value["path"]))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("production promotion evidence path is not an exact file")
    receipt = json.loads(path.read_bytes())
    _sealed(receipt)
    if _sha256(path) != str(value["file_sha256"]).removeprefix("sha256:") or receipt[
        "sha256"
    ] != str(value["receipt_self_sha256"]).removeprefix("sha256:"):
        raise ValueError("production promotion evidence bytes changed")
    return receipt


def _candidate_without_promotion(config: dict) -> dict:
    return {key: item for key, item in config.items() if key != "production_promotion"}


def requires_production_promotion(value: object) -> bool:
    """Require promotion for every production SkyRL plan, without a rename bypass."""
    if not isinstance(value, dict):
        return False
    args = value.get("arguments", {})
    data = value.get("data", {})
    wandb = value.get("wandb", {})
    cluster = value.get("cluster", {})
    execution = value.get("execution", {})
    if (isinstance(cluster, dict) and cluster.get("target") == "prod") or (
        isinstance(execution, dict) and execution.get("cluster_target") == "prod"
    ):
        return True
    identities = (
        value.get("name"),
        value.get("run_name"),
        args.get("name") if isinstance(args, dict) else None,
        value.get("output_root"),
        args.get("output_root") if isinstance(args, dict) else None,
        data.get("root") if isinstance(data, dict) else None,
        data.get("manifest") if isinstance(data, dict) else None,
        args.get("data_manifest") if isinstance(args, dict) else None,
        wandb.get("run_id") if isinstance(wandb, dict) else None,
        args.get("wandb_run_id") if isinstance(args, dict) else None,
    )
    return any(
        item in {PROD_NAME, PROD_OUTPUT, PROD_DATA_ROOT, PROD_DATA_MANIFEST} for item in identities
    ) or (
        isinstance(data, dict) and data.get("selection_sha256") == EXPECTED_DATA["selection_sha256"]
    )


def _exact_candidate(config: dict) -> None:
    if digest(_candidate_without_promotion(config)) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("production candidate differs from the exact reviewed run")


def _exact_data(metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("production data manifest is absent")
    files = metadata.get("files", {})
    if (
        metadata.get("schema") != "cyber_skyrl_data_v1"
        or metadata.get("name") != EXPECTED_DATA["name"]
        or metadata.get("selection_sha256") != EXPECTED_DATA["selection_sha256"]
        or metadata.get("split_sha256") != EXPECTED_DATA["split_sha256"]
        or metadata.get("tool_catalog_sha256") != EXPECTED_DATA["tool_catalog_sha256"]
        or metadata.get("limits") != EXPECTED_DATA["limits"]
        or set(files) != {"train", "dev"}
        or any(
            not isinstance(files.get(split), dict)
            or files[split].get("path") != split + ".jsonl"
            or files[split].get("rows") != rows
            for split, rows in EXPECTED_DATA["rows"].items()
        )
    ):
        raise ValueError("production data differs from the exact benchmark-isolated split")
    _sealed(metadata, "cyber_skyrl_data_v1")


def validate_promotion(value: dict, *, check_files: bool) -> dict:
    """Reopen the qualification chain and recompute every immutable predicate."""
    from . import skyrl_rl_checkpoint as checkpoint

    _sealed(value, SCHEMA)
    if set(value) != _FIELDS:
        raise ValueError("production promotion fields changed")
    refs = {
        name: _ref(value.get(name), check_files=check_files)
        for name in (
            "dev8_terminal",
            "reward_terminal",
            "checkpoint_manifest",
            "reload_accepted",
            "production_data_manifest",
        )
    }
    image = value.get("qualified_image")
    endpoints = value.get("excluded_inference_endpoints")
    if (
        value.get("status") != "qualified_for_fresh_live_checks"
        or value.get("candidate_run_sha256") != EXPECTED_CANDIDATE_SHA256
        or not isinstance(image, str)
        or re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image) is None
        or image.endswith(STALE_PRIVACY_INCOMPLETE_IMAGE_SHA256)
        or value.get("runtime_user") != {"uid": 1000, "gid": 100, "run_as_non_root": True}
        or value.get("benchmark_isolation")
        != {
            "optimizer_split": "train",
            "dev_is_evaluation_only": True,
            "final_test_rows": 0,
            "webexploitbench_rows": 0,
        }
        or value.get("live_requirements") != LIVE_REQUIREMENTS
        or endpoints != EXCLUDED_INFERENCE_ENDPOINTS
    ):
        raise ValueError("production promotion invariant changed")
    if not check_files:
        return {"candidate_run_sha256": EXPECTED_CANDIDATE_SHA256, "qualified_image": image}
    accepted = refs["reload_accepted"]
    validated_reload = checkpoint.validate_reload_accepted(accepted, check_files=True)
    plan = accepted["plan"]
    source = accepted["source"]
    source_plan = plan["source_manifest"]["source_plan"]
    prerequisite = source_plan["execution"]["engine_diagnostic_prerequisite"]
    expected_refs = {
        "dev8_terminal": (
            prerequisite["terminal_receipt_path"],
            prerequisite["terminal_receipt_file_sha256"],
            prerequisite["terminal_receipt_self_sha256"],
        ),
        "reward_terminal": (
            source["reward_terminal_receipt_path"],
            source["reward_terminal_receipt_file_sha256"],
            source["reward_terminal_receipt_self_sha256"],
        ),
        "checkpoint_manifest": (
            source["checkpoint_manifest_path"],
            source["checkpoint_manifest_file_sha256"],
            source["checkpoint_manifest_self_sha256"],
        ),
    }
    if any(
        tuple(
            str(value[name][key]).removeprefix("sha256:")
            for key in ("path", "file_sha256", "receipt_self_sha256")
        )
        != tuple(str(item).removeprefix("sha256:") for item in expected)
        for name, expected in expected_refs.items()
    ):
        raise ValueError("production promotion qualification chain is cross-bound incorrectly")
    _exact_data(refs["production_data_manifest"])
    if (
        value["production_data_manifest"]["path"] != PROD_DATA_MANIFEST
        or value.get("source_plan_sha256") != digest(source_plan)
        or source_plan["execution"]["image"] != image
        or plan["execution"]["image"] != image
        or validated_reload["reward_terminal_receipt_self_sha256"]
        != str(value["reward_terminal"]["receipt_self_sha256"]).removeprefix("sha256:")
    ):
        raise ValueError("production promotion source/image/data linkage changed")
    return {
        "candidate_run_sha256": EXPECTED_CANDIDATE_SHA256,
        "qualified_image": image,
        "reload_accepted": accepted,
    }


def bind_production_promotion(config: dict, relative_to: Path) -> dict | None:
    """Load the exact immutable promotion receipt during compilation."""
    if not requires_production_promotion(config):
        if config.get("production_promotion") is not None:
            raise ValueError("production promotion cannot be attached to another RL run")
        return None
    _exact_candidate(config)
    ref = config.get("production_promotion")
    if not isinstance(ref, dict) or set(ref) != _REF_FIELDS:
        raise ValueError("exact production promotion receipt is required")
    path = Path(str(ref["path"]))
    if not path.is_absolute():
        path = (relative_to / path).resolve()
    normalized = {**ref, "path": str(path)}
    receipt = _ref(normalized, check_files=True)
    validate_promotion(receipt, check_files=True)
    return {**normalized, "receipt": receipt}


def _plan_candidate_projection(plan: dict) -> dict:
    args, execution = plan.get("arguments", {}), plan.get("execution", {})
    recipe_keys = (
        "nodes",
        "steps",
        "groups",
        "samples_per_prompt",
        "lr",
        "eval_interval",
        "checkpoint_interval",
        "keep_checkpoints",
        "seed",
        "engine_start_timeout_seconds",
        "engine_cleanup_timeout_seconds",
    )
    return {
        "backend": "skyrl",
        "name": plan.get("run_name"),
        "output_root": plan.get("output_root"),
        "model": {
            "lock": "../models/qwen38-27b-1d4bf0f2.lock.json",
            "weights": "../models/qwen38-27b-1d4bf0f2.weights.json",
            "root": args.get("model_root"),
        },
        "data": {"manifest": args.get("data_manifest"), "root": PROD_DATA_ROOT},
        "recipe": {key: args.get(key) for key in recipe_keys},
        "wandb": {
            "entity": args.get("wandb_entity"),
            "project": args.get("wandb_project"),
            "run_id": args.get("wandb_run_id"),
        },
        "cluster": {
            "target": execution.get("cluster_target"),
            "priority": execution.get("priority"),
            "resources": execution.get("resources"),
        },
    }


def _exact_compiled_plan(plan: dict, accepted_receipt: dict) -> None:
    from . import skyrl

    args, execution = plan.get("arguments", {}), plan.get("execution", {})
    if (
        digest(_plan_candidate_projection(plan)) != EXPECTED_CANDIDATE_SHA256
        or args != EXPECTED_ARGUMENTS
        or not isinstance(accepted_receipt, dict)
        or plan.get("model")
        != accepted_receipt.get("plan", {})
        .get("source_manifest", {})
        .get("source_plan", {})
        .get("model")
        or plan.get("native_sources")
        != accepted_receipt.get("plan", {})
        .get("source_manifest", {})
        .get("source_plan", {})
        .get("native_sources")
        or plan.get("native_overrides") != skyrl.overrides(skyrl.SkyRLConfig(**args))
        or set(execution)
        != {
            "image",
            "image_cpu_qualification",
            "runtime_user",
            "production_promotion",
            "cluster_target",
            "priority",
            "resources",
        }
        or execution.get("runtime_user") != {"uid": 1000, "gid": 100, "run_as_non_root": True}
    ):
        raise ValueError("compiled production plan differs from the exact promoted candidate")


def validate_embedded_promotion(plan: dict) -> bool:
    """Fail closed in request preparation and again inside the allocated Pod."""
    required = requires_production_promotion(plan)
    proof = plan.get("execution", {}).get("production_promotion")
    if not required:
        if proof is not None:
            raise ValueError("production promotion attached to a non-production plan")
        return False
    if not isinstance(proof, dict) or set(proof) != {*_REF_FIELDS, "receipt"}:
        raise ValueError("embedded production promotion is absent")
    validated = validate_promotion(proof["receipt"], check_files=True)
    _exact_data(plan.get("data"))
    args = plan.get("arguments", {})
    if (
        plan.get("run_name") != PROD_NAME
        or plan.get("output_root") != PROD_OUTPUT
        or args.get("data_manifest") != PROD_DATA_MANIFEST
        or args.get("wandb_entity") != PROD_WANDB["entity"]
        or args.get("wandb_project") != PROD_WANDB["project"]
        or args.get("wandb_run_id") != PROD_WANDB["run_id"]
        or plan.get("execution", {}).get("cluster_target") != "prod"
        or plan["execution"].get("priority") != "c1"
        or plan["execution"].get("image") != proof["receipt"]["qualified_image"]
    ):
        raise ValueError("compiled production plan differs from the promoted candidate")
    _exact_compiled_plan(plan, validated["reload_accepted"])
    return True


def validate_production_preview(plan: dict, request: dict, preview: dict) -> dict:
    """Require the effective non-root c1/no-requeue one-node production render."""
    if not validate_embedded_promotion(plan):
        return {}
    import yaml

    try:
        obj = yaml.safe_load(preview["manifest_yaml"])
        cluster = obj["spec"]["rayClusterSpec"]
        groups = [(1, cluster["headGroupSpec"]["template"])] + [
            (row["replicas"], row["template"]) for row in cluster.get("workerGroupSpecs", [])
        ]
        pods = 0
        for replicas, template in groups:
            if replicas == 0:
                continue
            pod = template["spec"]
            container = pod["containers"]
            if len(container) != 1:
                raise JobsError("production preview must have one container per Pod")
            pctx, cctx = pod.get("securityContext", {}), container[0].get("securityContext", {})
            effective = {
                key: cctx.get(key, pctx.get(key))
                for key in ("runAsUser", "runAsGroup", "runAsNonRoot")
            }
            if effective != {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}:
                raise JobsError("production preview runtime user differs from 1000:100")
            pods += replicas
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise JobsError("malformed production Jobs API preview") from error
    if (
        request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("priority_class") != "c1"
        or request.get("requeueIfPreempted") is not False
        or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_UID") != "1000"
        or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_GID") != "100"
        or pods != 1
    ):
        raise JobsError("production preview topology/priority/runtime binding changed")
    return {"production_promotion": "validated", "runtime_user": {"uid": 1000, "gid": 100}}


def require_live_files(plan: dict, prepared_directory: Path) -> None:
    """Repeat create-once local/SFS gates immediately before the POST."""
    if not validate_embedded_promotion(plan):
        return
    if (prepared_directory / "SUBMISSION.jsonl").exists() or (
        prepared_directory / "SUBMISSION.jsonl"
    ).is_symlink():
        raise JobsError("production submission journal already exists")
    output = Path(PROD_OUTPUT)
    if output.exists() or output.is_symlink():
        raise JobsError("production output root already exists")
    manifest = Path(PROD_DATA_MANIFEST)
    if manifest.is_symlink() or not manifest.is_file():
        raise JobsError("exact production data manifest is absent")
    observed = json.loads(manifest.read_bytes())
    _exact_data(observed)
    ref = plan["execution"]["production_promotion"]["receipt"]["production_data_manifest"]
    if _sha256(manifest) != str(ref["file_sha256"]).removeprefix("sha256:"):
        raise JobsError("production data manifest changed after promotion")


def _kubectl_json(*arguments: str) -> dict:
    result = subprocess.run(
        ["kubectl", "--context", PROD_KUBE_CONTEXT, *arguments, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise JobsError("production Kubernetes read-only admission check failed")
    try:
        value = json.loads(result.stdout)
    except ValueError as error:
        raise JobsError("production Kubernetes read-only result is invalid") from error
    if not isinstance(value, dict):
        raise JobsError("production Kubernetes read-only result is not an object")
    return value


def require_live_external(plan: dict, client, *, wandb_api=None) -> dict:
    """Recheck authoritative Jobs/W&B/Kubernetes state immediately before POST."""
    if not validate_embedded_promotion(plan):
        return {}
    runs = client.all_runs()
    if any(
        not isinstance(row, dict)
        or row.get("name") == PROD_NAME
        or str(row.get("name", "")).startswith(PROD_NAME + "-")
        or row.get("run_dir") == PROD_OUTPUT
        for row in runs
    ):
        raise JobsError("production Jobs API identity/output is present or malformed")
    if wandb_api is None:
        import wandb

        wandb_api = wandb.Api()
    try:
        observed_run = wandb_api.run(
            f"{PROD_WANDB['entity']}/{PROD_WANDB['project']}/{PROD_WANDB['run_id']}"
        )
    except Exception as error:  # W&B has no stable public not-found exception across pins.
        if "not found" not in str(error).lower() and "could not find run" not in str(error).lower():
            raise JobsError("W&B duplicate check failed closed") from None
    else:
        if observed_run is not None:
            raise JobsError("production W&B run ID already exists")
    namespace = _kubectl_json("get", "namespace", "fleet-train-jobs")
    if namespace.get("metadata", {}).get("uid") != PROD_NAMESPACE_UID:
        raise JobsError("production namespace identity changed")
    deployments = _kubectl_json("get", "deployments", "-n", "inference")
    observed_endpoints = sorted(
        (
            {
                "name": item.get("metadata", {}).get("name"),
                "uid": item.get("metadata", {}).get("uid"),
            }
            for item in deployments.get("items", [])
            if item.get("metadata", {}).get("name")
            in {endpoint["name"] for endpoint in EXCLUDED_INFERENCE_ENDPOINTS}
        ),
        key=lambda item: item["name"],
    )
    if observed_endpoints != sorted(EXCLUDED_INFERENCE_ENDPOINTS, key=lambda item: item["name"]):
        raise JobsError("reviewed inference endpoint identities changed")
    pods = _kubectl_json("get", "pods", "-n", "fleet-train-jobs")
    items = pods.get("items")
    if not isinstance(items, list):
        raise JobsError("production Pod inventory is malformed")
    active_nodes, unscheduled_gpu_pods = set(), 0
    for pod in items:
        if not isinstance(pod, dict):
            raise JobsError("production Pod inventory is malformed")
        spec, status = pod.get("spec", {}), pod.get("status", {})
        if status.get("phase") not in {"Pending", "Running"}:
            continue
        containers = spec.get("containers", [])
        if not isinstance(containers, list):
            raise JobsError("production Pod containers are malformed")
        gpu = 0
        for container in containers:
            try:
                resources = container.get("resources", {})
                gpu += int(resources.get("limits", {}).get("nvidia.com/gpu", 0))
            except (AttributeError, TypeError, ValueError) as error:
                raise JobsError("production Pod GPU allocation is malformed") from error
        node = spec.get("nodeName")
        if gpu:
            if node:
                active_nodes.add(node)
            else:
                unscheduled_gpu_pods += 1
    active = len(active_nodes) + unscheduled_gpu_pods
    if active + 1 > 8:
        raise JobsError("production experiment node budget would exceed eight")
    return {
        "jobs_api_absent": True,
        "wandb_absent": True,
        "active_experiment_nodes": active,
        "candidate_nodes": 1,
        "node_limit": 8,
        "kube_context": PROD_KUBE_CONTEXT,
        "namespace_uid": namespace["metadata"]["uid"],
    }


def require_production_runtime_identity(plan: dict) -> dict:
    if not validate_embedded_promotion(plan):
        return {}
    if (
        os.environ.get("CYBER_EXPECTED_RUNTIME_UID") != "1000"
        or os.environ.get("CYBER_EXPECTED_RUNTIME_GID") != "100"
        or (os.geteuid(), os.getegid()) != (1000, 100)
    ):
        raise ValueError("production SkyRL runtime must be the pinned non-root user 1000:100")
    return {"uid": os.geteuid(), "gid": os.getegid()}
