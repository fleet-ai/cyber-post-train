"""Prepare configurable SFT plans using the qualified SkyRL runtime.

The editable configuration names model/data manifests and hyperparameters.
The resulting plan contains their resolved identities; no mutable model name,
data path alone, or recipe override is resolved after submission.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import re
import shlex
from pathlib import Path, PurePosixPath

import yaml

from cyber_post_train.jobs import digest, quantity, validate_request

from .sft_runtime import DENSE_FORMAT, DENSE_SCHEMA, validate_plan

IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
RECIPE = {
    "epochs": 1,
    "batch_size": 8,
    "microbatch_per_gpu": 1,
    "nodes": 1,
    "gpus_per_node": 8,
    "lr": 3e-6,
    "max_length": 16384,
    "eval_interval": 50,
    "checkpoint_interval": 50,
    "keep_checkpoints": 3,
    "seed": 42,
}
RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "64",
    "memory_request": "512Gi",
    "memory_limit": "768Gi",
}


def read_mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration/manifest must be a mapping")
    return value


def _known(value: dict, names: set[str], label: str) -> None:
    if not isinstance(value, dict) or value.keys() - names:
        raise ValueError(f"unknown fields in {label}; check the documented configuration")


def _sfs_root(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if (
        path.parts[:3] != ("/", "mnt", "sfs")
        or len(path.parts) < 5
        or ".." in path.parts
        or str(path) != value
    ):
        raise ValueError(f"{label} must be a canonical, specific directory under /mnt/sfs/")
    return str(path)


def compile_sft(config: dict, *, relative_to: Path) -> dict:
    _known(
        config,
        {
            "backend",
            "name",
            "output_root",
            "model",
            "data",
            "recipe",
            "wandb",
            "cluster",
            "lora",
            "recovery",
        },
        "SFT",
    )
    if config.get("backend", "skyrl") != "skyrl":
        raise ValueError("this compiler targets SkyRL; Miles requires its own backend recipe")
    model = config["model"]
    _known(model, {"lock", "weights", "root"}, "model")
    lock = read_mapping(relative_to / model["lock"])
    weights = read_mapping(relative_to / model["weights"])
    if (
        weights["revision"] != lock["revision"]
        or "sha256:" + digest(weights["files"]) != lock["weights"]["manifest_sha256"]
        or len(weights["files"]) != lock["weights"]["shards"]
    ):
        raise ValueError("weight inventory differs from the exact model lock")
    files = [
        *lock["tokenizer"]["files"],
        *weights["files"],
        {"path": "model.safetensors.index.json", "sha256": lock["weights"]["index_sha256"]},
    ]
    for key, sha in lock["configuration"].items():
        if not key.endswith("_sha256"):
            raise ValueError("configuration identity must contain exact file digests")
        files.append({"path": key.removesuffix("_sha256") + ".json", "sha256": sha})
    if len({f["path"] for f in files}) != len(files):
        raise ValueError("duplicate model inventory file")
    if any(not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", f["sha256"]) for f in files):
        raise ValueError("model files require SHA-256 identities")
    data = config["data"]
    _known(data, {"manifest", "root"}, "data")
    manifest = read_mapping(relative_to / data["manifest"])
    if manifest.get("sha256") != "sha256:" + digest(
        {k: v for k, v in manifest.items() if k != "sha256"}
    ):
        raise ValueError("corpus manifest digest mismatch")
    if (manifest["tokenizer"]["repo"], manifest["tokenizer"]["revision"]) != (
        lock["repo"],
        lock["revision"],
    ):
        raise ValueError("corpus tokenizer differs from model")
    datasets = {}
    for split in ("train", "dev"):
        entry = manifest["files"][split]
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("dataset file escapes corpus root")
        datasets[split] = {
            **entry,
            "path": str(PurePosixPath(_sfs_root(data["root"], "data root")) / path),
        }
    overrides = config.get("recipe", {})
    _known(overrides, set(RECIPE), "recipe")
    recipe = {**RECIPE, **overrides}
    if any(type(recipe[k]) is not int or recipe[k] <= 0 for k in RECIPE if k not in {"lr", "seed"}):
        raise ValueError("recipe counts must be positive integers")
    if type(datasets["train"]["rows"]) is not int or datasets["train"]["rows"] <= 0:
        raise ValueError("training corpus must have a positive integer row count")
    recipe["max_steps"] = (
        math.ceil(datasets["train"]["rows"] / recipe["batch_size"]) * recipe["epochs"]
    )
    cluster = config.get("cluster", {})
    _known(cluster, {"priority", "resources"}, "cluster")
    runtime = Path(__file__).with_name("sft_runtime.py").read_bytes()
    plan = {
        "schema": DENSE_SCHEMA
        if datasets["train"].get("format") == DENSE_FORMAT
        else "cyber_sft_runtime_v2",
        "run_name": config["name"],
        "output_root": config["output_root"],
        "model": {
            "repo": lock["repo"],
            "revision": lock["revision"],
            "root": _sfs_root(model["root"], "model root"),
            "files": files,
            "weight_manifest_sha256": lock["weights"]["manifest_sha256"],
        },
        "datasets": datasets,
        "recipe": recipe,
        "wandb": dict(config["wandb"]),
        "split_manifest_sha256": manifest["split_sha256"],
        "corpus_manifest_sha256": manifest["sha256"],
        "runtime_sha256": hashlib.sha256(runtime).hexdigest(),
        "execution": {
            "image": IMAGE,
            "priority": cluster.get("priority", "c1"),
            "resources": {**RESOURCES, **cluster.get("resources", {})},
        },
    }
    # The native GDN compatibility hook is currently qualified only for these
    # Qwen architectures. Do not silently substitute the GLM Flash model or
    # claim the full GLM FP8 checkpoint uses this ordinary full-weight loader.
    if lock["repo"] == "zai-org/GLM-5.3":
        # Rank zero materializes the full BF16 base before FSDP sharding. Never
        # inherit the much smaller Qwen CPU reservation for this ~1.5 TB load.
        if (
            recipe["nodes"] < 2
            or recipe["gpus_per_node"] != 8
            or quantity(plan["execution"]["resources"]["memory_request"]) < quantity("2048Gi")
        ):
            raise ValueError("full GLM LoRA requires at least two 8-GPU nodes and 2048Gi per node")
        plan["lora"] = config.get("lora")
        plan["model"]["tokenizer_manifest_sha256"] = lock["tokenizer"]["manifest_sha256"]
        plan["glm_runtime_sha256"] = hashlib.sha256(
            Path(__file__).with_name("glm_runtime.py").read_bytes()
        ).hexdigest()
    elif "lora" in config or lock["repo"] not in {"Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-27B"}:
        raise ValueError("model needs a qualified SkyRL loader profile before GPU submission")
    if "recovery" in config:
        from .recovery import bind

        bind(plan, config["recovery"], relative_to=relative_to)
    validate_plan(plan, check_files=False)
    validate_request(job_request(plan))
    return plan


def job_request(plan: dict) -> dict:
    """Embed the small immutable runtime/plan, not data, in the generic request.

    This avoids a separate GPU or cluster Job just to copy a Python script.
    The bundle is checked before unpacking into a create-once owned directory.
    """
    runtime = Path(__file__).with_name("sft_runtime.py").read_bytes()
    if hashlib.sha256(runtime).hexdigest() != plan["runtime_sha256"]:
        raise ValueError("local runtime changed since this plan was compiled")
    plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    contents = {"runtime": runtime.decode(), "plan": plan_bytes.decode()}
    if "lora" in plan:
        helper = Path(__file__).with_name("glm_runtime.py").read_bytes()
        if hashlib.sha256(helper).hexdigest() != plan["glm_runtime_sha256"]:
            raise ValueError("GLM runtime changed since this plan was compiled")
        contents["extra_files"] = {
            "training/__init__.py": "",
            "training/glm_runtime.py": helper.decode(),
        }
    if "recovery" in plan:
        extras = contents.setdefault("extra_files", {})
        for name in ("__init__.py", "recovery.py", "checkpoints.py", "sft_runtime.py"):
            extras["training/" + name] = (
                "" if name == "__init__.py" else Path(__file__).with_name(name).read_text()
            )
        if (
            hashlib.sha256(extras["training/recovery.py"].encode()).hexdigest()
            != plan["recovery_runtime_sha256"]
        ):
            raise ValueError("recovery runtime changed since this plan was compiled")
    bundle = json.dumps(
        contents,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    compressed = gzip.compress(bundle, mtime=0)
    bundle_sha = hashlib.sha256(compressed).hexdigest()
    bootstrap = (
        "import base64,gzip,hashlib,importlib,json,os,pathlib,runpy,sys;"
        "b=base64.b64decode(os.environ.pop('CYBER_SFT_BUNDLE'),validate=True);"
        f"assert hashlib.sha256(b).hexdigest()=={bundle_sha!r};"
        "v=json.loads(gzip.decompress(b));"
        "p=pathlib.Path(os.environ['RUN_DIR'])/'.runtime';p.mkdir(mode=0o700);"
        "(p/'sft_runtime.py').write_text(v['runtime']);(p/'plan.json').write_text(v['plan']);"
        "[((p/n).parent.mkdir(parents=True,exist_ok=True),(p/n).write_text(t)) "
        "for n,t in v.get('extra_files',{}).items()];"
        "sys.path.insert(0,str(p));importlib.invalidate_caches();"
        f"sys.argv=['sft_runtime','--plan',str(p/'plan.json'),'--plan-sha256',{hashlib.sha256(plan_bytes).hexdigest()!r}];"
        "runpy.run_path(str(p/'sft_runtime.py'),run_name='__main__')"
    )
    w, execution, r = plan["wandb"], plan["execution"], plan["recipe"]
    return {
        "name": plan["run_name"],
        "title": w["name"],
        "run_dir": plan["output_root"],
        "image": execution["image"],
        "command": "python -c " + shlex.quote(bootstrap),
        "workers": r["nodes"],
        "gpus_per_worker": r["gpus_per_node"],
        "resources": execution["resources"],
        "priority_class": execution["priority"],
        "requeueIfPreempted": False,
        "secrets": ["wandb-api"],
        "env": {
            "CYBER_SFT_BUNDLE": base64.b64encode(compressed).decode(),
            **(
                {"PYTHONPATH": str(Path(plan["output_root"]) / ".runtime")}
                if "extra_files" in contents
                else {}
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "online",
            "WANDB_ENTITY": w["entity"],
            "WANDB_PROJECT": w["project"],
            "WANDB_RUN_ID": w["run_id"],
            "WANDB_NAME": w["name"],
            "WANDB_RUN_GROUP": w["group"],
            "WANDB_TAGS": ",".join(w.get("tags", [])),
            "WANDB_DISABLE_CODE": "true",
            "WANDB_CONSOLE": "off",
        },
    }


def preflight(plan: dict) -> dict:
    """CPU-only checks in the pinned image, with staged inputs mounted.

    Check every bound file, native config and actual token/mask accounting before
    requesting a GPU. CUDA/distributed changes still require a real canary.
    """
    import pyarrow.parquet as pq
    import torch
    from skyrl.train.sft_trainer import tokenize_chat_example
    from transformers import AutoConfig, AutoTokenizer

    from .sft_runtime import build_runtime_configs, prepare_rows, validate_runtime_sources

    if torch.cuda.is_available():
        raise ValueError("run data/runtime preflight without GPU allocation")
    validate_plan(plan)
    validate_runtime_sources()
    build_runtime_configs(plan)
    AutoConfig.from_pretrained(
        plan["model"]["root"], local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["root"], local_files_only=True, trust_remote_code=False
    )
    counts = {}
    for split, spec in plan["datasets"].items():
        rows = prepare_rows(
            pq.read_table(spec["path"]).to_pylist(),
            spec,
            tokenizer,
            tokenize_chat_example,
            max_length=plan["recipe"]["max_length"],
        )
        counts[split] = {
            "rows": len(rows),
            "tasks": len(spec["task_keys"]),
            "supervised_tokens": sum(sum(row["loss_mask"]) for row in rows),
        }
    return {
        "schema": "cyber_sft_cpu_preflight_v1",
        "request_sha256": digest(job_request(plan)),
        "plan_sha256": digest(plan),
        "status": "passed",
        "gpus": 0,
        "checked": [
            "native_sources",
            "model_files",
            "dataset_files",
            "native_config",
            "tokenization",
            "target_accounting",
        ],
        "counts": counts,
    }
