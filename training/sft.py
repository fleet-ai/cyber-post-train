"""Prepare configurable SFT plans using the qualified SkyRL runtime.

The editable configuration names model/data manifests and hyperparameters.
The resulting plan contains their resolved identities; no mutable model name,
data path alone, or recipe override is resolved after submission.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import shlex
from pathlib import Path, PurePosixPath

import yaml

from cyber_post_train.jobs import canonical_gzip, digest, quantity, validate_request

from .models import bound_model
from .sft_runtime import (
    DENSE_FORMAT,
    DENSE_SCHEMA,
    QWEN38_LORA_BROAD_FULL_PLANS,
    QWEN38_LORA_PRODUCTION_QUALIFICATION,
    QWEN38_LORA_QUALIFICATION,
    qwen38_megatron_binding,
    validate_plan,
)

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
    text = path.read_text()
    # PyYAML's YAML 1.1 resolver reads JSON's 1e-06 as a string. Preserve
    # JSON scalar types before falling back to human-authored YAML.
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = yaml.safe_load(text)
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
            "runtime",
            "recovery",
            "pause_after_step",
        },
        "SFT",
    )
    if config.get("backend", "skyrl") != "skyrl":
        raise ValueError("this compiler targets SkyRL; Miles requires its own backend recipe")
    model = config["model"]
    _known(model, {"lock", "weights", "root"}, "model")
    lock = read_mapping(relative_to / model["lock"])
    weights = read_mapping(relative_to / model["weights"])
    bound = bound_model(lock, weights, _sfs_root(model["root"], "model root"))
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
    validation_mode = manifest.get("validation_mode", "teacher_cross_entropy")
    if validation_mode not in {"teacher_cross_entropy", "task_outcomes_only"}:
        raise ValueError("unsupported corpus validation mode")
    required_splits = ("train", "dev") if validation_mode == "teacher_cross_entropy" else ("train",)
    if set(manifest["files"]) != set(required_splits):
        raise ValueError("corpus files differ from its validation mode")
    datasets = {}
    for split in required_splits:
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
    if any(
        type(recipe[k]) is not int or recipe[k] < (0 if k == "eval_interval" else 1)
        for k in RECIPE
        if k not in {"lr", "seed"}
    ):
        raise ValueError("recipe counts must be positive integers; eval_interval may be zero")
    if (validation_mode == "task_outcomes_only") != (recipe["eval_interval"] == 0):
        raise ValueError("task-outcome evaluation requires eval_interval: 0 and no CE dev file")
    if type(datasets["train"]["rows"]) is not int or datasets["train"]["rows"] <= 0:
        raise ValueError("training corpus must have a positive integer row count")
    recipe["max_steps"] = (
        math.ceil(datasets["train"]["rows"] / recipe["batch_size"]) * recipe["epochs"]
    )
    cluster = config.get("cluster", {})
    _known(cluster, {"priority", "resources"}, "cluster")
    runtime = Path(__file__).with_name("sft_runtime.py").read_bytes()
    plan = {
        "schema": (
            DENSE_SCHEMA
            if datasets["train"].get("format") == DENSE_FORMAT
            else "cyber_sft_runtime_v2"
        ),
        "run_name": config["name"],
        "output_root": config["output_root"],
        "model": bound,
        "datasets": datasets,
        "recipe": recipe,
        "wandb": dict(config["wandb"]),
        "split_manifest_sha256": manifest["split_sha256"],
        "corpus_manifest_sha256": manifest["sha256"],
        "validation_mode": validation_mode,
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
        if "runtime" in config:
            raise ValueError("the explicit Megatron runtime binding is only valid for Qwen3.8 LoRA")
        # Rank zero materializes the full BF16 base before FSDP sharding. Never
        # inherit the much smaller Qwen CPU reservation for this ~1.5 TB load.
        if (
            recipe["nodes"] < 2
            or recipe["gpus_per_node"] != 8
            or quantity(plan["execution"]["resources"]["memory_request"]) < quantity("2048Gi")
            or quantity(plan["execution"]["resources"]["cpu_request"]) < 16
        ):
            raise ValueError(
                "full GLM LoRA requires at least two 8-GPU nodes, 2048Gi and 16 CPUs per node"
            )
        plan["lora"] = config.get("lora")
        plan["model"]["tokenizer_manifest_sha256"] = lock["tokenizer"]["manifest_sha256"]
        plan["glm_runtime_sha256"] = hashlib.sha256(
            Path(__file__).with_name("glm_runtime.py").read_bytes()
        ).hexdigest()
    elif lock["repo"] == "Qwen/Qwen3.8-27B" and "lora" in config:
        source_commit, image, source_files = qwen38_megatron_binding()
        runtime_config = config.get("runtime")
        _known(runtime_config, {"skyrl_source_commit", "image"}, "runtime")
        if set(runtime_config) != {"skyrl_source_commit", "image"}:
            raise ValueError("Qwen3.8 LoRA runtime needs exact source and image bindings")
        if runtime_config["skyrl_source_commit"] != source_commit:
            raise ValueError("Qwen3.8 LoRA SkyRL source commit differs from the reviewed revision")
        if runtime_config["image"] != image:
            raise ValueError("Qwen3.8 LoRA image differs from the reviewed immutable digest")
        _known(
            config["lora"],
            {"type", "target_modules", "rank", "alpha", "init_method", "dropout"},
            "Qwen3.8 LoRA",
        )
        if set(config["lora"]) != {
            "type",
            "target_modules",
            "rank",
            "alpha",
            "init_method",
            "dropout",
        }:
            raise ValueError("Qwen3.8 LoRA needs every exact adapter binding")
        plan["lora"] = dict(config["lora"])
        plan["execution"]["image"] = image
        plan["skyrl_runtime"] = {
            "source_commit": source_commit,
            "source_files_sha256": source_files,
        }
        # Production qualification admits only the exact create-once broad
        # identities reviewed in the runtime. Selecting one by its frozen run
        # identity does not open a parameter menu: ``validate_plan`` still
        # compares every scientific, data, resource, runtime and W&B field.
        qualification = (
            QWEN38_LORA_PRODUCTION_QUALIFICATION
            if config["name"] in QWEN38_LORA_BROAD_FULL_PLANS or "recovery" in config
            else QWEN38_LORA_QUALIFICATION
        )
        plan["qualification_gate"] = copy.deepcopy(qualification)
    elif "lora" in config or lock["repo"] not in {
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen3.6-27B",
    }:
        raise ValueError("model needs a qualified SkyRL loader profile before GPU submission")
    elif "runtime" in config:
        raise ValueError("the explicit Megatron runtime binding requires Qwen3.8 LoRA")
    if "recovery" in config:
        from .recovery import bind

        bind(plan, config["recovery"], relative_to=relative_to)
    if "pause_after_step" in config:
        plan["pause_after_step"] = config["pause_after_step"]
    validate_plan(plan, check_files=False)
    validate_request(job_request(plan))
    return plan


def job_request(plan: dict) -> dict:
    """Embed the small immutable runtime/plan, not data, in the generic request.

    This avoids a separate GPU or cluster Job just to copy a Python script.
    The bundle is checked before unpacking into a create-once owned directory.
    """
    # ``plan_sha256`` is attached only inside the runtime after the staged plan
    # file has been verified.  A prepared/submitted plan containing that field
    # would hash different bytes when the runtime attaches its real file digest.
    if "plan_sha256" in plan:
        raise ValueError("plan_sha256 is a runtime-only evidence field")
    # Submission re-renders a prepared request through this function. Recheck
    # the full plan so a hand-written or stale prepared directory cannot bypass
    # the Qwen LoRA source/image/c1/evidence gates.
    validate_plan(plan, check_files=False)
    qwen38_lora = plan.get("model", {}).get("repo") == "Qwen/Qwen3.8-27B" and "lora" in plan
    runtime = Path(__file__).with_name("sft_runtime.py").read_bytes()
    if hashlib.sha256(runtime).hexdigest() != plan["runtime_sha256"]:
        raise ValueError("local runtime changed since this plan was compiled")
    plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    contents = {"runtime": runtime.decode(), "plan": plan_bytes.decode()}
    if qwen38_lora:
        # The terminal one-step gate validates its signed checkpoint receipt
        # through this package after the GPU update and checkpoint flush.  The
        # trainer image intentionally contains SkyRL, not this repository, so
        # stage the complete import surface rather than discovering a missing
        # helper only after consuming a node.
        extras = contents.setdefault("extra_files", {})
        extras.update(
            {
                "training/__init__.py": Path(__file__).with_name("__init__.py").read_text(),
                "training/io.py": Path(__file__).with_name("io.py").read_text(),
                "training/qwen38_lora_artifacts.py": Path(__file__)
                .with_name("qwen38_lora_artifacts.py")
                .read_text(),
                "training/sft_runtime.py": runtime.decode(),
            }
        )
    if plan.get("model", {}).get("repo") == "zai-org/GLM-5.3" and "lora" in plan:
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
    compressed = canonical_gzip(bundle)
    bundle_sha = hashlib.sha256(compressed).hexdigest()
    encoded = base64.b64encode(compressed).decode()
    transport = {"CYBER_SFT_BUNDLE": encoded}
    bundle_expression = "os.environ.pop('CYBER_SFT_BUNDLE')"
    if len(encoded) > 120000:
        # Linux limits each argument/environment value independently. Recovery
        # plans embed their sealed checkpoint inventory, so split only the
        # transport representation and reassemble the exact digest-bound bytes
        # before decoding.
        parts = [encoded[index : index + 48000] for index in range(0, len(encoded), 48000)]
        transport = {f"CYBER_SFT_BUNDLE_{index}": value for index, value in enumerate(parts)}
        bundle_expression = (
            f"''.join(os.environ.pop('CYBER_SFT_BUNDLE_'+str(i)) for i in range({len(parts)}))"
        )
    bootstrap = (
        "import base64,gzip,hashlib,importlib,json,os,pathlib,runpy,sys;"
        f"b=base64.b64decode({bundle_expression},validate=True);"
        f"assert hashlib.sha256(b).hexdigest()=={bundle_sha!r};"
        "v=json.loads(gzip.decompress(b));"
        "p=pathlib.Path(os.environ['RUN_DIR'])/'.runtime';p.mkdir(parents=True,mode=0o700);"
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
        "failureAlerts": False,
        "secrets": ["wandb-api"],
        **(
            {"image_pull_secrets": ["ghcr-pull"]}
            if plan.get("model", {}).get("repo") == "Qwen/Qwen3.8-27B" and "lora" in plan
            else {}
        ),
        "env": {
            **transport,
            **(
                {
                    "PYTHONPATH": str(Path(plan["output_root"]) / ".runtime")
                    + (":/opt/skyrl" if qwen38_lora else "")
                }
                if "extra_files" in contents
                else {}
            ),
            **({"SKYRL_PYTHONPATH_EXPORT": "1"} if qwen38_lora else {}),
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
            **({"FLA_TILELANG": "0"} if qwen38_lora else {}),
        },
    }


def _check_native_dataset_loader(plan: dict, prepared_rows: dict[str, list[dict]]) -> None:
    """Exercise the exact staged loader against rows prepared by the CPU gate.

    The full-weight FSDP and Qwen3.8 Megatron-LoRA images intentionally carry
    different SkyRL dataset contracts.  Running the staged loader inside the
    pinned image catches an import/interface mismatch before a GPU is claimed.
    """
    from .sft_runtime import _make_trainer_class

    trainer_class = _make_trainer_class()
    trainer = trainer_class.__new__(trainer_class)
    trainer.plan = plan
    trainer._load_split = prepared_rows.__getitem__
    train_rows = prepared_rows["train"]
    dataset = trainer.load_dataset()
    if len(dataset) != len(train_rows):
        raise ValueError("native training loader changed the prepared row count")
    if "lora" in plan and plan.get("model", {}).get("repo") == "Qwen/Qwen3.8-27B":
        lengths = [int(value) for value in dataset.sequence_lengths]
        if len(lengths) != len(train_rows) or any(value <= 0 for value in lengths):
            raise ValueError("native Qwen3.8 LoRA dataset lengths are invalid")
    elif dataset is not train_rows:
        raise ValueError("native full-weight loader did not preserve the prepared rows")
    eval_dataset = trainer.load_eval_dataset()
    if "dev" in plan["datasets"]:
        if eval_dataset is not prepared_rows["dev"]:
            raise ValueError("native eval loader changed the prepared development rows")
    elif eval_dataset is not None:
        raise ValueError("task-outcome training unexpectedly produced an eval dataset")


def preflight(plan: dict) -> dict:
    """CPU-only checks in the pinned image, with staged inputs mounted.

    Check every bound file, native config and actual token/mask accounting before
    requesting a GPU. CUDA/distributed changes still require a real canary.
    """
    import pyarrow.parquet as pq
    import torch
    from skyrl.train.sft_trainer import tokenize_chat_example
    from transformers import AutoConfig, AutoTokenizer

    from .sft_runtime import (
        _validate_sft_forward_backward_adapter,
        build_runtime_configs,
        prepare_rows,
        validate_runtime_sources,
    )

    if torch.cuda.is_available():
        raise ValueError("run data/runtime preflight without GPU allocation")
    validate_plan(plan)
    validate_runtime_sources(plan)
    build_runtime_configs(plan)
    _validate_sft_forward_backward_adapter(plan)
    AutoConfig.from_pretrained(
        plan["model"]["root"], local_files_only=True, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["root"], local_files_only=True, trust_remote_code=False
    )
    counts = {}
    prepared_rows = {}
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
        prepared_rows[split] = rows
    if "train" not in prepared_rows:
        raise ValueError("CPU preflight did not materialize the training split")
    _check_native_dataset_loader(plan, prepared_rows)
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
            "native_forward_backward_signature",
            "native_train_only_loader",
            "tokenization",
            "target_accounting",
        ],
        "counts": counts,
    }
