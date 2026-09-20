"""Closed production-queue bindings for the Qwen3.8 SkyRL full arms.

This module contains no launch command.  It validates the sanitized manifest,
the create-once SFS package, the scientific controls, and the two-phase release
observer contract used by :mod:`training.skyrl_production_training`.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath

from cyber_post_train.jobs import JobsError, digest

PROFILE = "qwen38_skyrl_production_queue_v1"
QUALIFICATION_SCHEMA = "cyber_qwen38_skyrl_production_queue_v1"
PLAN_BINDING_SCHEMA = "cyber_qwen38_skyrl_production_plan_binding_v1"
MANIFEST_SCHEMA = "cyber_skyrl_data_v1"
RELEASE_CONTRACT_SCHEMA = "cyber_skyrl_release_observer_contract_v1"
OFFLINE_PREVIEW_SCHEMA = "cyber_skyrl_offline_request_preview_v1"

IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
PROD_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"
API_URL = "https://api.ft.flt.build"
MODEL = {
    "repo": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
    "weight_manifest_sha256": (
        "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
    ),
}
MODEL_INVENTORY_SHA256 = "sha256:dcfdcd6ecb6661741cd3a4b24dc5af7259642c8a6824773e0de70d55d7501179"
SPLIT_SHA256 = "sha256:1f0da5054df25a7d67476f591d8c9c8d42d44de65c0b9d4b542ba9e33b132472"
TASK_SET_SHA256 = "sha256:5de88eada94119fd21ac715bb1956a745376d775c13ce7d5584936e70b3a003a"
TOOL_CATALOG_SHA256 = "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
TEMPLATE_SHA256 = "sha256:c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
LIMITS = {
    "context_tokens": 98304,
    "response_tokens": 81920,
    "max_tokens_per_turn": 4096,
    "max_turns": 600,
    "episode_seconds": 2400,
    "tool_seconds": 330,
    "tool_result_chars": 50000,
}
RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "64",
    "memory_request": "512Gi",
    "memory_limit": "768Gi",
}
COMMON_RECIPE = {
    "nodes": 1,
    "groups": 1,
    "samples_per_prompt": 8,
    "eval_interval": 10,
    "checkpoint_interval": 10,
    "keep_checkpoints": 2,
}
FIXED_CONTROLS = {
    "model": MODEL["repo"] + "@" + MODEL["revision"],
    "task_set_sha256": TASK_SET_SHA256,
    "split_sha256": SPLIT_SHA256,
    "counts": {"train": 59, "dev": 20, "test_untouched": 10},
    "harness": "Fleet native SkyRL exact-version task-tool runtime",
    "ordered_tools": ["bash", "submit_report"],
    "reward": "authoritative Fleet partial cyber score",
    "context_tokens": LIMITS["context_tokens"],
    "response_tokens": LIMITS["response_tokens"],
    "max_tokens_per_turn": LIMITS["max_tokens_per_turn"],
    "max_turns": LIMITS["max_turns"],
    "episode_seconds": LIMITS["episode_seconds"],
    "groups": COMMON_RECIPE["groups"],
    "samples_per_prompt": COMMON_RECIPE["samples_per_prompt"],
    "kl_coefficient": 0.001,
    "nodes": COMMON_RECIPE["nodes"],
    "gpus": 8,
    "priority": "c1",
}
ARMS = {
    "a1": {
        "name": "chris-q38-skyrl10-a1",
        "steps": 10,
        "lr": 1e-6,
        "seed": 42,
        "dispatch_order": 1,
    },
    "lr3e7": {
        "name": "chris-q38-skyrl10-lr3e7",
        "steps": 10,
        "lr": 3e-7,
        "seed": 42,
        "dispatch_order": 2,
    },
    "lr3e6": {
        "name": "chris-q38-skyrl10-lr3e6",
        "steps": 10,
        "lr": 3e-6,
        "seed": 42,
        "dispatch_order": 3,
    },
    "seed43": {
        "name": "chris-q38-skyrl10-seed43",
        "steps": 10,
        "lr": 1e-6,
        "seed": 43,
        "dispatch_order": 4,
    },
    "dose50": {
        "name": "chris-q38-skyrl50-a1",
        "steps": 50,
        "lr": 1e-6,
        "seed": 42,
        "dispatch_order": 5,
    },
}
DATA_BINDINGS = {
    "a1": {
        "manifest_sha256": (
            "sha256:09eae3253cf1224b329e83c4512da3d877ef1755d17efb7934713bb7e5d322d5"
        ),
        "manifest_file": (
            1814,
            "sha256:10703876ec2fa75c454bbc791150cf223697dbceee828f1ce4adc5af41b207a0",
        ),
        "train": (
            250141,
            "sha256:b4aac05771a7b8764d5960e0c99a96bd6b2369721390f8377c4d6dbde69a1280",
        ),
        "dev": (86777, "sha256:e02bf77148cfb7e90030aa2b63ccf044794caebbd82b3f616a628e5ee5586682"),
    },
    "lr3e7": {
        "manifest_sha256": (
            "sha256:4f3772182bbc08ab43024208311d957d66ef67be1f55106184111687bc06b6f8"
        ),
        "manifest_file": (
            1817,
            "sha256:31f63eae5cc1745787fa61f05f25eb0a2ae9e2585573783ffa199377c8e65391",
        ),
        "train": (
            250318,
            "sha256:633be6a16303287b936d71aa8271ce03c11d86ed44e51f9c550ed80c57d65472",
        ),
        "dev": (86837, "sha256:c793ef90bb69ec8217402d5337ae774465ae846de91f432408187bef1d41a65f"),
    },
    "lr3e6": {
        "manifest_sha256": (
            "sha256:7fbbd04ef14e88518e6046e9a35d4ab8ca5a5a44face27e77c6d1fa97ff55c75"
        ),
        "manifest_file": (
            1817,
            "sha256:23de3b5c2acd450071b51030c414328f853cfe27ec4a7a918687ec4cbe444e27",
        ),
        "train": (
            250318,
            "sha256:0ab9216284c6ed50efd26f0991f8298695ebd5404259aa33441cab5ccdb041e2",
        ),
        "dev": (86837, "sha256:79a6aad26f41a560ad2dc7126500926580565ab068243d2e2b0d03126e1fc0f9"),
    },
    "seed43": {
        "manifest_sha256": (
            "sha256:6db196e724999be81476ec15b5fe5ec46e87e05ae99b0d7ffbadaba3e5870e38"
        ),
        "manifest_file": (
            1818,
            "sha256:cbe5e8d6f5ec7a6ef6855348409da29b6d35e5eeff493df10fe60f1fc26b3e4c",
        ),
        "train": (
            250377,
            "sha256:e6e9c3c8cf00620b97d90c1cb0f14248a892b366ad17237093108557fc4792b9",
        ),
        "dev": (86857, "sha256:b187b3488bc105748a98f8fc25a96b45662606bdde5f26800180531786f1c9ea"),
    },
    "dose50": {
        "manifest_sha256": (
            "sha256:03ca037492b7fd758f748f53d2659dbc3cbcac96de43be39f74efa04e1e34368"
        ),
        "manifest_file": (
            1814,
            "sha256:472c3154671abc378185e2e2aa5f53178a6db8134f76534d1d2439c5c9894402",
        ),
        "train": (
            250141,
            "sha256:2ab774005a37c52299accf0bf0d8ff77c3e90fc8b852e8ebd4827d6656264381",
        ),
        "dev": (86777, "sha256:56b147a69340fc2f70ce0dd58d9627f3cfd391f52b888fa0d9c862181d00b675"),
    },
}
SHARED_STAGED_FILES = {
    "split.json": (
        16980,
        "sha256:e4cb912b1ed87122a57d7ed948765d892efa53c34f18ecd8916f4d59698f69ee",
    ),
    "task-set.json": (
        38991,
        "sha256:29a83428a81bdc15277094afb25e81fd28c02e8038ae0cc253543db5f438569e",
    ),
}
WATCHDOGS = {
    10: {
        "poll_seconds": 60,
        "startup_seconds": 1800,
        "idle_seconds": 1200,
        "hard_seconds": 43200,
        "drain_seconds": 300,
        "episode_ceiling_seconds": 38400,
    },
    50: {
        "poll_seconds": 60,
        "startup_seconds": 1800,
        "idle_seconds": 1200,
        "hard_seconds": 172800,
        "drain_seconds": 300,
        "episode_ceiling_seconds": 163200,
    },
}


def _seal(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = "sha256:" + digest(result)
    return result


def _validate_seal(value: object, schema: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError("SkyRL production closure must be an object")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(body):
        raise ValueError("SkyRL production closure digest mismatch")
    return value


def _recipe(arm: dict) -> dict:
    return {
        **COMMON_RECIPE,
        "steps": arm["steps"],
        "lr": arm["lr"],
        "seed": arm["seed"],
    }


def _expected_staged_files(arm_id: str) -> list[dict]:
    binding = DATA_BINDINGS[arm_id]
    values = {
        "dev.jsonl": (*binding["dev"], 20),
        "manifest.json": (*binding["manifest_file"], None),
        "split.json": (*SHARED_STAGED_FILES["split.json"], None),
        "task-set.json": (*SHARED_STAGED_FILES["task-set.json"], None),
        "train.jsonl": (*binding["train"], 59),
    }
    result = []
    for path, (size, sha256, rows) in values.items():
        item = {"path": path, "bytes": size, "sha256": sha256, "mode": "0600"}
        if rows is not None:
            item["rows"] = rows
        result.append(item)
    return result


def _expected_arguments(arm: dict, staged_root: str) -> dict:
    return {
        "name": arm["name"],
        "output_root": f"/mnt/sfs/jobs/{arm['name']}",
        "model": MODEL["repo"],
        "model_root": MODEL["root"],
        "train_data": staged_root + "/train.jsonl",
        "dev_data": staged_root + "/dev.jsonl",
        "data_manifest": staged_root + "/manifest.json",
        "train_rows": 59,
        "dev_rows": 20,
        "wandb_entity": "thefleet",
        "wandb_project": "cyber-post-train",
        "wandb_run_id": arm["name"],
        "context_tokens": LIMITS["context_tokens"],
        "response_tokens": LIMITS["response_tokens"],
        "tokens_per_turn": LIMITS["max_tokens_per_turn"],
        "max_turns": LIMITS["max_turns"],
        **_recipe(arm),
    }


def _expected_arm(arm_id: str) -> dict:
    try:
        return ARMS[arm_id]
    except KeyError as exc:
        raise ValueError("unknown SkyRL production arm") from exc


def validate_qualification(value: object) -> dict:
    """Validate the generated queue closure without trusting mutable defaults."""
    qualification = _validate_seal(value, QUALIFICATION_SCHEMA)
    expected_names = [arm["name"] for arm in ARMS.values()]
    arms = qualification.get("production_arms")
    if (
        qualification.get("profile") != PROFILE
        or qualification.get("execution")
        != {
            "cluster_target": "prod",
            "jobs_api_base_url": API_URL,
            "kubernetes_context": PROD_CONTEXT,
            "namespace": NAMESPACE,
            "image": IMAGE,
            "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        }
        or not isinstance(arms, list)
        or [item.get("name") for item in arms] != expected_names
        or qualification.get("fixed_controls") != FIXED_CONTROLS
        or qualification.get("release_observer")
        != {
            "contract_schema": RELEASE_CONTRACT_SCHEMA,
            "armed_receipt_schema": "cyber_skyrl_release_observer_armed_v1",
            "arm_before_jobs_post": True,
            "bind_server_run_and_kubernetes_uids_after_post": True,
            "release_requires_bound_descendants_absent_and_active_gpus_zero": True,
            "peer_workload_mutation_authorized": False,
        }
    ):
        raise ValueError("SkyRL production qualification controls changed")
    gate = qualification.get("submission_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("preview_authorized") is not False
        or gate.get("submission_authorized") is not False
        or not isinstance(gate.get("blockers"), list)
        or not gate["blockers"]
    ):
        raise ValueError("SkyRL production queue must remain externally blocked")
    for item in arms:
        arm = _expected_arm(item.get("id", ""))
        data_binding = DATA_BINDINGS[item["id"]]
        manifest = _validate_seal(item.get("manifest"), MANIFEST_SCHEMA)
        watchdog = item.get("watchdog")
        staged = item.get("staged_data")
        if (
            item.get("name") != arm["name"]
            or item.get("dispatch_order") != arm["dispatch_order"]
            or item.get("recipe") != _recipe(arm)
            or watchdog != WATCHDOGS[arm["steps"]]
            or watchdog["hard_seconds"]
            < watchdog["episode_ceiling_seconds"]
            + watchdog["startup_seconds"]
            + watchdog["drain_seconds"]
            or manifest.get("name") != arm["name"]
            or manifest.get("limits") != LIMITS
            or manifest.get("sha256") != data_binding["manifest_sha256"]
            or manifest.get("selection_sha256") != TASK_SET_SHA256
            or manifest.get("split_sha256") != SPLIT_SHA256
            or manifest.get("tool_catalog_sha256") != TOOL_CATALOG_SHA256
            or manifest.get("template_sha256") != TEMPLATE_SHA256
            or manifest.get("environment_creates") != 0
            or manifest.get("gpus") != 0
            or manifest.get("files")
            != {
                "train": {
                    "path": "train.jsonl",
                    "rows": 59,
                    "max_prompt_tokens": 1256,
                    "sha256": data_binding["train"][1],
                },
                "dev": {
                    "path": "dev.jsonl",
                    "rows": 20,
                    "max_prompt_tokens": 1686,
                    "sha256": data_binding["dev"][1],
                },
            }
            or manifest.get("tokenizer", {}).get("repo") != MODEL["repo"]
            or manifest.get("tokenizer", {}).get("revision") != MODEL["revision"]
            or manifest.get("tokenizer", {}).get("chat_template_sha256")
            != TEMPLATE_SHA256.removeprefix("sha256:")
            or not isinstance(staged, dict)
            or staged.get("root")
            != f"/mnt/sfs/jobs/chris-q38-study-corpora-v1/{arm['name']}-inputs-v1/data"
            or staged.get("manifest_self_sha256") != manifest["sha256"]
            or staged.get("manifest_file_sha256") != data_binding["manifest_file"][1]
            or staged.get("files") != _expected_staged_files(item["id"])
        ):
            raise ValueError("SkyRL production arm closure changed")
    return qualification


def validate_run_config(
    config: dict,
    metadata: dict,
    bound_model: dict,
    qualification: object,
) -> dict:
    """Bind one exact arm while keeping the external release gates closed."""
    closure = validate_qualification(qualification)
    matches = [item for item in closure["production_arms"] if item["name"] == config.get("name")]
    if len(matches) != 1:
        raise ValueError("SkyRL production run does not select exactly one arm")
    item = matches[0]
    arm = _expected_arm(item["id"])
    data = config.get("data", {})
    output = PurePosixPath(config.get("output_root", ""))
    data_root = PurePosixPath(data.get("root", ""))
    if (
        metadata != item["manifest"]
        or any(bound_model.get(key) != value for key, value in MODEL.items())
        or "sha256:" + digest(bound_model) != MODEL_INVENTORY_SHA256
        or config.get("recipe") != _recipe(arm)
        or config.get("cluster") != {"priority": "c1", "resources": RESOURCES, "target": "prod"}
        or config.get("wandb")
        != {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": arm["name"],
        }
        or output != PurePosixPath("/mnt/sfs/jobs") / arm["name"]
        or str(data_root) != item["staged_data"]["root"]
        or data.get("manifest") != str(data_root / "manifest.json")
        or output == data_root
        or output in data_root.parents
        or data_root in output.parents
    ):
        raise ValueError("SkyRL production model, data, recipe, resource, or identity drift")
    binding = {
        "schema": PLAN_BINDING_SCHEMA,
        "profile": PROFILE,
        "arm_id": item["id"],
        "dispatch_order": item["dispatch_order"],
        "image": closure["execution"]["image"],
        "environment": closure["execution"]["environment"],
        "cluster_target": closure["execution"]["cluster_target"],
        "jobs_api_base_url": closure["execution"]["jobs_api_base_url"],
        "kubernetes_context": closure["execution"]["kubernetes_context"],
        "namespace": closure["execution"]["namespace"],
        "watchdog": item["watchdog"],
        "staged_data": item["staged_data"],
        "model_sha256": MODEL_INVENTORY_SHA256,
        "manifest_sha256": metadata["sha256"],
        "submission_gate": closure["submission_gate"],
        "release_observer": closure["release_observer"],
    }
    return _seal(binding)


def validate_plan_binding(
    binding: object,
    metadata: dict,
    arguments: dict,
    model: dict,
) -> dict:
    """Revalidate the self-contained production binding inside the GPU bundle."""
    value = _validate_seal(binding, PLAN_BINDING_SCHEMA)
    arm = _expected_arm(value.get("arm_id", ""))
    watchdog = WATCHDOGS[arm["steps"]]
    if (
        value.get("profile") != PROFILE
        or value.get("dispatch_order") != arm["dispatch_order"]
        or value.get("image") != IMAGE
        or value.get("environment") != {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
        or value.get("cluster_target") != "prod"
        or value.get("jobs_api_base_url") != API_URL
        or value.get("kubernetes_context") != PROD_CONTEXT
        or value.get("namespace") != NAMESPACE
        or value.get("watchdog") != watchdog
        or value.get("model_sha256") != MODEL_INVENTORY_SHA256
        or "sha256:" + digest(model) != MODEL_INVENTORY_SHA256
        or value.get("manifest_sha256") != metadata.get("sha256")
        or value.get("staged_data", {}).get("manifest_self_sha256") != metadata.get("sha256")
        or metadata.get("sha256") != DATA_BINDINGS[value["arm_id"]]["manifest_sha256"]
        or arguments != _expected_arguments(arm, value["staged_data"]["root"])
    ):
        raise ValueError("SkyRL production plan binding changed")
    return value


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def validate_staged_data(plan: dict) -> dict:
    """Reopen one SFS package as runtime UID 1000:100 and hash every file."""
    binding = validate_plan_binding(
        plan.get("qualification"), plan["data"], plan["arguments"], plan["model"]
    )
    staged = binding["staged_data"]
    root = Path(staged["root"])
    root_stat = root.lstat()
    if (
        root.is_symlink()
        or not root.is_dir()
        or (root_stat.st_uid, root_stat.st_gid) != (1000, 100)
        or stat.S_IMODE(root_stat.st_mode) != 0o700
        or (os.geteuid(), os.getegid()) != (1000, 100)
    ):
        raise ValueError("SkyRL production data root must be private and owned by 1000:100")
    expected = {item["path"]: item for item in staged["files"]}
    found = {path.name: path for path in root.iterdir()}
    if set(found) != set(expected):
        raise ValueError("SkyRL production staged file set changed")
    observed = []
    for name in sorted(expected):
        path, item = found[name], expected[name]
        info = path.lstat()
        observed_sha256 = _file_sha256(path)
        after = path.lstat()
        if (
            path.is_symlink()
            or not path.is_file()
            or (info.st_uid, info.st_gid) != (1000, 100)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size != item["bytes"]
            or observed_sha256 != item["sha256"]
            or any(
                getattr(info, field) != getattr(after, field)
                for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            )
        ):
            raise ValueError("SkyRL production staged file identity changed")
        observed.append({"path": name, "bytes": info.st_size, "sha256": item["sha256"]})
    try:
        manifest = json.loads((root / "manifest.json").read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("SkyRL production staged manifest is unreadable") from exc
    if manifest != plan["data"]:
        raise ValueError("SkyRL production staged manifest differs from the plan")
    final_root_stat = root.lstat()
    if any(
        getattr(root_stat, field) != getattr(final_root_stat, field)
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    ):
        raise ValueError("SkyRL production data root changed during validation")
    output = Path(plan["output_root"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("SkyRL production output already exists")
    proof = {
        "schema": "cyber_skyrl_production_staged_data_validation_v1",
        "status": "passed",
        "plan_sha256": digest(plan),
        "manifest_sha256": manifest["sha256"],
        "root": str(root),
        "files": observed,
        "output_absent": True,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
    }
    return _seal(proof)


def release_observer_contract(plan: dict, request: dict) -> dict:
    """Return the immutable two-phase observer contract for one future POST."""
    binding = validate_plan_binding(
        plan.get("qualification"), plan["data"], plan["arguments"], plan["model"]
    )
    watchdog = binding["watchdog"]
    value = {
        "schema": RELEASE_CONTRACT_SCHEMA,
        "status": "contract_only_not_armed",
        "arm_id": binding["arm_id"],
        "run_name_prefix": request["name"],
        "server_run_name_pattern": "^" + request["name"] + "-[a-f0-9]{8}$",
        "run_dir": request["run_dir"],
        "plan_sha256": "sha256:" + digest(plan),
        "request_sha256": "sha256:" + digest(request),
        "kubernetes_context": binding["kubernetes_context"],
        "namespace": binding["namespace"],
        "expected_gpus": 8,
        "requested_image": request["image"],
        "arm_before_jobs_post": True,
        "bind_after_jobs_response": [
            "jobs_run_name",
            "jobs_run_id",
            "rayjob_uid",
            "raycluster_uid",
            "workload_uid",
            "pod_uids",
            "runtime_image_ids",
        ],
        "release_acceptance": {
            "controller_terminal_observed": True,
            "all_bound_pod_uids_absent": True,
            "bound_raycluster_uid_absent": True,
            "bound_workload_uid_absent": True,
            "active_gpus": 0,
            "shutdown_after_job_finishes_required_in_server_preview": True,
        },
        "maximum_observation_seconds": watchdog["hard_seconds"] + watchdog["drain_seconds"] + 1800,
        "peer_workload_mutation_authorized": False,
    }
    return _seal(value)


def offline_preview(plan: dict, request: dict) -> dict:
    """Summarize an exact request locally; this is not a server render."""
    binding = validate_plan_binding(
        plan.get("qualification"), plan["data"], plan["arguments"], plan["model"]
    )
    if (
        request.get("name") != plan["run_name"]
        or request.get("run_dir") != plan["output_root"]
        or request.get("image") != binding["image"]
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("priority_class") != "c1"
        or request.get("requeueIfPreempted") is not False
        or request.get("secrets") != ["fleet-api", "wandb-api"]
    ):
        raise JobsError("SkyRL production request differs from the offline plan")
    return _seal(
        {
            "schema": OFFLINE_PREVIEW_SCHEMA,
            "status": "locally_rendered_not_server_previewed",
            "submitted": False,
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "name": request["name"],
            "run_dir": request["run_dir"],
            "cluster_target": binding["cluster_target"],
            "image": request["image"],
            "resources": {
                "priority": request["priority_class"],
                "nodes": request["workers"],
                "gpus_per_node": request["gpus_per_worker"],
                "gpus": request["workers"] * request["gpus_per_worker"],
                **request["resources"],
            },
            "watchdog": binding["watchdog"],
            "wandb": {
                "entity": plan["arguments"]["wandb_entity"],
                "project": plan["arguments"]["wandb_project"],
                "run_id": plan["arguments"]["wandb_run_id"],
                "resume": "never",
            },
            "server_preview_requested": False,
            "external_reads": 0,
            "external_mutations": 0,
        }
    )
