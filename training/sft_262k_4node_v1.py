"""Exact four-node derivative of the recovered Qwen3.8 262K parent.

This compiler accepts no scientific menu.  It reopens the retained v12 plan,
verifies its canonical digest, changes only the reviewed four-node hypothesis,
and binds the current runtime bytes.  The resulting plan may be previewed but
remains submission-blocked until its zero-GPU preflight and root review exist.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import shlex
from pathlib import Path

from cyber_post_train.jobs import API_URLS, canonical_gzip, digest, validate_request

from . import sft
from .sft_262k_runtime import QUALIFICATION, SUBMISSION_GATE, validate_plan

VARIANT = "qwen38_sft_262k_4node_v1"
PARENT_PLAN_SHA256 = "3b9e81acb301327c40007a40ab13c3bf5c164152655fb49f96a8db52eb117690"
PARENT_RUNTIME_SHA256 = "b6b81c9876ddf8379a0837a0aaf5f0426acf8ccc4388d79bd93bad94c59e38d3"
IMAGE = sft.IMAGE


def _known(value: dict, names: set[str], label: str) -> None:
    if not isinstance(value, dict) or value.keys() - names:
        raise ValueError(f"unknown fields in {label}; this qualification is immutable")


def compile_sft(config: dict, *, relative_to: Path) -> dict:
    _known(
        config,
        {
            "runtime_variant",
            "parent_plan",
            "name",
            "output_root",
            "wandb",
            "cluster",
        },
        "four-node 262K SFT",
    )
    if config.get("runtime_variant") != VARIANT:
        raise ValueError("four-node 262K runtime variant changed")
    parent_path = relative_to / config["parent_plan"]
    parent = sft.read_mapping(parent_path)
    if digest(parent) != PARENT_PLAN_SHA256:
        raise ValueError("recovered v12 parent plan digest changed")
    if (
        parent.get("runtime_sha256") != PARENT_RUNTIME_SHA256
        or parent.get("execution", {}).get("image") != IMAGE
    ):
        raise ValueError("recovered v12 runtime or image changed")
    cluster = config.get("cluster")
    if cluster != {"priority": "c1", "target": "prod"}:
        raise ValueError("four-node 262K candidate requires exact production c1 binding")
    name = config.get("name")
    output_root = config.get("output_root")
    if (
        name != "chris-q38-t3k262-4n-can-v1"
        or output_root != "/mnt/sfs/jobs/chris-q38-t3k262-4n-can-v1"
    ):
        raise ValueError("four-node 262K create-once run identity changed")
    wandb = config.get("wandb")
    if not isinstance(wandb, dict) or wandb.get("run_id") != name or wandb.get("name") != name:
        raise ValueError("four-node 262K W&B identity changed")

    plan = copy.deepcopy(parent)
    plan.update(
        {
            "runtime_variant": VARIANT,
            "run_name": name,
            "output_root": output_root,
            "runtime_sha256": hashlib.sha256(
                Path(sft.__file__).with_name("sft_262k_runtime.py").read_bytes()
            ).hexdigest(),
            "wandb": copy.deepcopy(wandb),
            "long_context_qualification": copy.deepcopy(QUALIFICATION),
            "qualification": {
                "schema": "qwen38_262k_four_node_submission_gate_v1",
                "submission_gate": copy.deepcopy(SUBMISSION_GATE),
            },
        }
    )
    plan["recipe"].update({"nodes": 4, "batch_size": 32, "max_steps": 4})
    plan["execution"].update(
        {
            "priority": "c1",
            "cluster_target": "prod",
            "jobs_api_base_url": API_URLS["prod"],
        }
    )
    validate_plan(plan, check_files=False)
    request = job_request(plan)
    validate_request(request)
    return plan


def job_request(plan: dict) -> dict:
    if plan.get("runtime_variant") != VARIANT:
        raise ValueError("four-node 262K plan selected the wrong compiler")
    validate_plan(plan, check_files=False)
    if "plan_sha256" in plan:
        raise ValueError("plan_sha256 is a runtime-only evidence field")

    runtime_path = Path(__file__).with_name("sft_262k_runtime.py")
    base_runtime_path = Path(__file__).with_name("sft_runtime.py")
    runtime = runtime_path.read_bytes()
    base_runtime = base_runtime_path.read_bytes()
    if hashlib.sha256(runtime).hexdigest() != plan["runtime_sha256"]:
        raise ValueError("four-node 262K runtime changed since this plan was compiled")
    if (
        hashlib.sha256(base_runtime).hexdigest()
        != plan["long_context_qualification"]["base_runtime_sha256"]
    ):
        raise ValueError("four-node 262K base runtime changed since qualification")

    plan_bytes = json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    contents = {
        "runtime": runtime.decode(),
        "plan": plan_bytes.decode(),
        "extra_files": {
            "training/__init__.py": "",
            "training/sft_runtime.py": base_runtime.decode(),
            "training/sft_262k_runtime.py": runtime.decode(),
        },
    }
    compressed = canonical_gzip(
        json.dumps(contents, sort_keys=True, separators=(",", ":")).encode()
    )
    bundle_sha = hashlib.sha256(compressed).hexdigest()
    encoded = base64.b64encode(compressed).decode()
    transport = {"CYBER_SFT_BUNDLE": encoded}
    bundle_expression = "os.environ.pop('CYBER_SFT_BUNDLE')"
    if len(encoded) > 120000:
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
    wandb, execution, recipe = plan["wandb"], plan["execution"], plan["recipe"]
    return {
        "name": plan["run_name"],
        "title": wandb["name"],
        "run_dir": plan["output_root"],
        "image": execution["image"],
        "command": "python -c " + shlex.quote(bootstrap),
        "workers": recipe["nodes"],
        "gpus_per_worker": recipe["gpus_per_node"],
        "resources": execution["resources"],
        "priority_class": execution["priority"],
        "requeueIfPreempted": False,
        "failureAlerts": False,
        "secrets": ["wandb-api"],
        "env": {
            **transport,
            "PYTHONPATH": str(Path(plan["output_root"]) / ".runtime"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "online",
            "WANDB_ENTITY": wandb["entity"],
            "WANDB_PROJECT": wandb["project"],
            "WANDB_RUN_ID": wandb["run_id"],
            "WANDB_NAME": wandb["name"],
            "WANDB_RUN_GROUP": wandb["group"],
            "WANDB_TAGS": ",".join(wandb.get("tags", [])),
            "WANDB_DISABLE_CODE": "true",
            "WANDB_CONSOLE": "off",
        },
    }


def preflight(plan: dict) -> dict:
    from .sft_262k_runtime import install_runtime

    validate_plan(plan, check_files=True)
    install_runtime()
    return sft.preflight(plan)
