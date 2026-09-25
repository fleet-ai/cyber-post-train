"""Small bridge to the qualified Qwen3.8 96k SkyRL SFT runtime.

This checkout intentionally does not vendor the old trainer. Exact historical
Git objects are verified and used only in an isolated temporary directory.
Prepare and CPU preflight do not allocate GPUs; preview is read-only. There is
no POST/submit command: live capacity and create-once SFS checks are separate
operator gates, not something a local preview can prove.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

COMMIT = "c908d3a828d070c6b27611fc388b1e7e3b4049dd"
ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "training/__init__.py": "ecf358039bbb9b6cbab6546c9b1e61b9bc06c5b2d5b19907ff303a9277d77e27",
    "training/io.py": "7a0b734a4ab7fb8b702430094c58c72f19ac8fc7cc5e056eb8410267e6bfdfe3",
    "training/models.py": "712adce5f073de13d02168cd17c2f7c396ba094a1ebc9a34decafb8639b8375e",
    "training/sft.py": "447dcaac2b610c1b6c124a13e7d541edc4c3145d26d8ff31577d75b37d8dd67d",
    "training/sft_runtime.py": "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17",
    "cyber_post_train/__init__.py": "3e70d0bf91f68f0190f2eb6d08e4e83d1ba58c1cbff29eb78e600c24e89c96a6",
    "cyber_post_train/jobs.py": "bc0d08a7a27b48ff5a21a5d824356d6e92e4856b17cc7296a976aa18a2baa40d",
    "configs/models/qwen38-27b-1d4bf0f2.lock.json": "f3926fe675263b25dc79c2b3881a9c463d6b7931e9d61aeb777d15efc61e35ac",
    "configs/models/qwen38-27b-1d4bf0f2.weights.json": "80a5e9de066e068abb012a0e9bd144676813a31042bb578226472803f8e0e84f",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def historical_source(name: str) -> bytes:
    if name not in SOURCES:
        raise ValueError("unreviewed historical source")
    result = subprocess.run(
        ["git", "show", f"{COMMIT}:{name}"], cwd=ROOT, capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError("qualified historical source is unavailable")
    if SOURCES[name] and sha(result.stdout) != SOURCES[name]:
        raise ValueError("qualified historical source digest changed")
    return result.stdout


def _legacy(mode: str, value: dict, *, manifest: bytes | None = None) -> dict:
    """Run the historical compiler/CPU gate in its own import path."""
    script = """
import json,os,sys
from pathlib import Path
from training.sft import compile_sft,job_request,preflight
from cyber_post_train.jobs import Jobs,digest,validate_preview
v=json.load(sys.stdin)
try:
    if os.environ['SFT_BRIDGE_MODE']=='compile':
        p=compile_sft(v['config'],relative_to=Path('configs/runs'))
        result={'plan':p,'request':job_request(p)}
    elif os.environ['SFT_BRIDGE_MODE']=='request':
        result={'request':job_request(v['plan'])}
    elif os.environ['SFT_BRIDGE_MODE']=='preflight':
        result=preflight(v['plan'])
    elif os.environ['SFT_BRIDGE_MODE']=='validate_preview':
        result=validate_preview(v['request'],v['preview'])
    elif os.environ['SFT_BRIDGE_MODE']=='preview':
        token=os.environ.get('FLEET_API_KEY')
        if not token: raise ValueError('missing API credential')
        r=v['request']
        with Jobs(token) as jobs:
            for existing in jobs.all_runs():
                if (existing.get('run_dir')==r['run_dir'] or
                    existing.get('name')==r['name'] or
                    existing.get('name','').startswith(r['name']+'-') or
                    existing.get('title')==r['title']):
                    raise ValueError('duplicate API run identity')
            result=validate_preview(r,jobs.preview(r))
    else: raise ValueError('unsupported bridge stage')
    print(json.dumps(result,sort_keys=True,allow_nan=False))
except BaseException as exc:
    # Historical errors can contain private source paths or trainer details.
    print(type(exc).__name__,file=sys.stderr)
    sys.exit(2)
"""
    with tempfile.TemporaryDirectory(prefix="q38-sft-bridge-") as temporary:
        root = Path(temporary)
        (root / "configs/runs").mkdir(parents=True)
        for name in SOURCES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(historical_source(name))
        if manifest is not None:
            path = root / "configs/data/corpus.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(manifest)
        env = {**os.environ, "PYTHONPATH": str(root), "SFT_BRIDGE_MODE": mode}
        result = subprocess.run(
            [sys.executable, "-c", script], input=canonical(value), cwd=root,
            env=env, capture_output=True, check=False,
        )
    if result.returncode:
        # Captured stderr may include private trainer data; expose only a type.
        reason = result.stderr.decode(errors="replace").splitlines()[-1:] or [""]
        reason = reason[0].split(":", 1)[0].strip()
        if reason not in {"ValueError", "KeyError", "FileNotFoundError", "JobsError",
                          "ImportError", "ModuleNotFoundError", "RuntimeError", "TypeError"}:
            reason = "runtime_error"
        raise ValueError(f"historical {mode} gate rejected input ({reason})")
    lines = result.stdout.splitlines()
    if not lines:
        raise ValueError(f"historical {mode} gate returned no receipt")
    return json.loads(lines[-1])


def _require_debug_config(config: dict) -> None:
    recipe, cluster = config.get("recipe", {}), config.get("cluster", {})
    expected = {
        "nodes": 1, "gpus_per_node": 8, "max_length": 98304,
        "batch_size": 8, "microbatch_per_gpu": 1,
    }
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("debug recipe differs from qualified one-node 96k shape")
    if cluster.get("priority") != "c1" or config.get("backend") != "skyrl":
        raise ValueError("debug run requires SkyRL and c1")
    if recipe.get("eval_interval") != recipe.get("checkpoint_interval"):
        raise ValueError("save a checkpoint at every teacher-loss evaluation")
    if config.get("data", {}).get("manifest", "").endswith("96k-v1.manifest.json"):
        raise ValueError("the old clipped/leaky corpus is not a corrected debug corpus")
    if config.get("model", {}).get("root") != "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2":
        raise ValueError("model root differs from the qualified base")


def _require_converted_manifest(manifest: dict) -> None:
    if (manifest.get("schema") != "qwen38_tool_aware_parquet_v1"
        or manifest.get("trainer_ready") is not True
        or manifest.get("validation_mode") != "teacher_cross_entropy"
        or set(manifest.get("files", {})) != {"train", "dev"}
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", manifest.get("source_receipt_sha256", ""))
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", manifest.get("split_sha256", ""))
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", manifest.get("tokenizer", {}).get("sha256", ""))
        or any(manifest["files"][split].get("format") != "chat_messages_last_assistant_v2"
               for split in ("train", "dev"))):
        raise ValueError("tool-aware converter receipt/readback is missing")


def prepare(config_path: Path, destination: Path) -> dict:
    if destination.exists() or destination.is_symlink():
        raise ValueError("prepared destination already exists")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    _require_debug_config(config)
    source = Path(config["data"]["manifest"])
    manifest_path = source if source.is_absolute() else config_path.parent / source
    manifest_bytes = manifest_path.read_bytes()
    _require_converted_manifest(json.loads(manifest_bytes))
    staged = json.loads(json.dumps(config))
    staged["model"]["lock"] = "../models/qwen38-27b-1d4bf0f2.lock.json"
    staged["model"]["weights"] = "../models/qwen38-27b-1d4bf0f2.weights.json"
    staged["data"]["manifest"] = "../data/corpus.json"
    compiled = _legacy("compile", {"config": staged}, manifest=manifest_bytes)
    plan, request = compiled["plan"], compiled["request"]
    if (plan["model"]["revision"] != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
        or plan["validation_mode"] != "teacher_cross_entropy"
        or plan["datasets"]["train"].get("format") != "chat_messages_last_assistant_v2"
        or plan["datasets"]["dev"].get("format") != "chat_messages_last_assistant_v2"
        or plan["recipe"]["max_steps"] > 32
        or plan["recipe"]["keep_checkpoints"] <
            (plan["recipe"]["max_steps"] + plan["recipe"]["checkpoint_interval"] - 1)
            // plan["recipe"]["checkpoint_interval"]
        or request["failureAlerts"] is not False
        or request["priority_class"] != "c1"
        or request["workers"] != 1
        or request["gpus_per_worker"] != 8):
        raise ValueError("compiled debug request failed an immutable safety/science gate")
    receipt = {
        "schema": "qwen38_96k_debug_prepared_v1", "historical_commit": COMMIT,
        "config_sha256": sha(config_bytes), "manifest_file_sha256": sha(manifest_bytes),
        "plan_sha256": sha(canonical(plan)), "request_sha256": sha(canonical(request)),
        "status": "prepared_not_submitted",
    }
    destination.mkdir(parents=True)
    for name, value in (("plan.json", plan), ("request.json", request), ("PREPARED.json", receipt)):
        (destination / name).write_bytes(canonical(value) + b"\n")
    return receipt


def prepared(directory: Path) -> tuple[dict, dict, dict]:
    receipt = json.loads((directory / "PREPARED.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if (receipt.get("historical_commit") != COMMIT
        or receipt.get("plan_sha256") != sha(canonical(plan))
        or receipt.get("request_sha256") != sha(canonical(request))
        or _legacy("request", {"plan": plan})["request"] != request):
        raise ValueError("prepared binding changed")
    return plan, request, receipt


def _preflight_matches(result: dict, receipt: dict) -> bool:
    required = {
        "native_sources", "model_files", "dataset_files", "native_config",
        "native_forward_backward_signature", "native_train_only_loader",
        "tokenization", "target_accounting",
    }
    counts = result.get("counts", {})
    return (
        result.get("status") == "passed" and result.get("gpus") == 0
        and result.get("request_sha256") == receipt["request_sha256"]
        and result.get("plan_sha256") == receipt["plan_sha256"]
        and required <= set(result.get("checked", []))
        and set(counts) == {"train", "dev"}
        and all(counts[split].get("rows", 0) > 0 and
                counts[split].get("supervised_tokens", 0) > 0
                for split in ("train", "dev"))
    )


def preflight(directory: Path) -> dict:
    plan, request, receipt = prepared(directory)
    path = directory / "PREFLIGHT.json"
    if path.exists() or path.is_symlink():
        raise ValueError("CPU preflight receipt already exists")
    result = _legacy("preflight", {"plan": plan})
    if not _preflight_matches(result, receipt):
        raise ValueError("CPU preflight did not pass for this exact request")
    path.write_bytes(canonical(result) + b"\n")
    return {"status": "passed", "plan_sha256": receipt["plan_sha256"]}


def preview(directory: Path) -> dict:
    _plan, request, receipt = prepared(directory)
    gate = json.loads((directory / "PREFLIGHT.json").read_text())
    if not _preflight_matches(gate, receipt):
        raise ValueError("exact CPU preflight is absent")
    proof = _legacy("preview", {"request": request})
    return {**proof, "status": "previewed_not_submitted", "request_sha256": receipt["request_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare", help="compile a create-once exact request off-GPU")
    p.add_argument("config", type=Path)
    p.add_argument("destination", type=Path)
    for action in ("preflight", "preview"):
        p = commands.add_parser(action)
        p.add_argument("prepared_directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.config, args.destination)
        elif args.action == "preflight":
            result = preflight(args.prepared_directory)
        else:
            result = preview(args.prepared_directory)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"{args.action} rejected: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
