"""Exact HF → native Miles checkpoint preparation, never an optimizer step.

The pinned native converter owns model construction and distributed saving.
We own input identity, one bounded child process group, and create-once evidence.
No downloads, model edits, Ray lifecycle commands or process-name kills.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from cyber_post_train.jobs import bundled_request, digest, quantity

from . import miles

SCHEMA = "cyber_miles_conversion_v1"
CONVERTER_SHA256 = "0c2541d30073777a30344273a3773844a70ca1961287520c0496a1cec18d43f6"
DEADLINE_SECONDS = 1800


def _hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or indirect input file")
    before = path.stat()
    with path.open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if any(
        getattr(before, k) != getattr(after, k)
        for k in (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    ):
        raise ValueError("file changed during hashing")
    return result


def _write(path: Path, value: dict) -> dict:
    value = {**value, "sha256": digest(value)}
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return value


def compile_conversion(config: dict, *, relative_to: Path) -> dict:
    from .models import bound_model
    from .sft import RESOURCES, _known, _sfs_root, read_mapping

    _known(config, {"name", "output_root", "model", "cluster", "profile"}, "Miles conversion")
    prof = miles.profile_for(config.get("profile", "qwen3.8-27b"))
    model = config["model"]
    _known(model, {"lock", "weights", "root"}, "model")
    cluster = config.get("cluster", {})
    _known(cluster, {"priority", "resources"}, "cluster")
    bound = bound_model(
        read_mapping(relative_to / model["lock"]),
        read_mapping(relative_to / model["weights"]),
        _sfs_root(model["root"], "model root"),
    )
    output = _sfs_root(config["output_root"], "output root")
    if bound["repo"] != "Qwen/Qwen3.8-27B":
        raise ValueError("only the exact native Qwen3.8 text conversion is supported")
    if any(
        a == b or a in b.parents
        for a, b in (
            (Path(output), Path(bound["root"])),
            (Path(bound["root"]), Path(output)),
        )
    ):
        raise ValueError("model and output directories must not overlap")
    plan = {
        "schema": SCHEMA,
        "run_name": config["name"],
        "output_root": output,
        "model": bound,
        "profile": prof.recipe,
        "runtime_sha256": _hash(Path(__file__)),
        "native_converter_sha256": CONVERTER_SHA256,
        "optimizer_steps": 0,
        "deadline_seconds": DEADLINE_SECONDS,
        "execution": {
            "image": prof.image,
            "priority": cluster.get("priority", "c1"),
            "resources": {**RESOURCES, **cluster.get("resources", {})},
        },
    }
    job_request(plan)
    return plan


def job_request(plan: dict) -> dict:
    if (
        plan["schema"] != SCHEMA
        or plan["runtime_sha256"] != _hash(Path(__file__))
        or plan["native_converter_sha256"] != CONVERTER_SHA256
        or plan["execution"]["image"] != miles.image_for(plan.get("profile", "qwen3.8-27b"))
        or plan["optimizer_steps"] != 0
        or plan["deadline_seconds"] != DEADLINE_SECONDS
    ):
        raise ValueError("conversion runtime/plan drift")
    resources = plan["execution"]["resources"]
    if quantity(resources["cpu_request"]) < 64 or quantity(resources["memory_request"]) < quantity(
        "512Gi"
    ):
        raise ValueError("conversion requires its reviewed 64 CPU / 512Gi loading envelope")
    root = Path(__file__).resolve().parents[1]
    files = {
        name: (root / name).read_text()
        for name in (
            "training/miles_conversion.py",
            "training/miles.py",
            "cyber_post_train/jobs.py",
        )
    }
    files.update({"training/__init__.py": "", "cyber_post_train/__init__.py": ""})
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": plan["run_name"],
            "title": plan["run_name"] + " zero-step native conversion",
            "run_dir": plan["output_root"],
            "image": plan["execution"]["image"],
            "workers": 1,
            "gpus_per_worker": 8,
            "resources": plan["execution"]["resources"],
            "priority_class": plan["execution"]["priority"],
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "WANDB_MODE": "disabled",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        "training.miles_conversion",
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def check_inputs(plan: dict) -> None:
    root = Path(plan["model"]["root"])
    for item in plan["model"]["files"]:
        if _hash(root / item["path"]) != item["sha256"].removeprefix("sha256:"):
            raise ValueError("staged model differs from exact inventory")
    config = json.loads((root / "config.json").read_text())
    if config.get("auto_map") or config.get("text_config", {}).get("auto_map"):
        raise ValueError("native conversion must not resolve remote model code")


def native_arguments(plan: dict) -> list[str]:
    from fti.trainers.miles.run_fleet import _RECIPES
    from miles.utils.external_utils.command_utils import repo_base_dir
    from miles.utils.external_utils.model_args_utils import load_model_args

    profile = _RECIPES["qwen3.8-27b"]
    source = Path(repo_base_dir) / "tools/convert_hf_to_torch_dist.py"
    if (
        _hash(source) != CONVERTER_SHA256
        or profile.backend != "megatron"
        or profile.vision
        or profile.megatron_model_type != "qwen3.8-27B"
    ):
        raise ValueError("native conversion source/model profile changed")
    # Native auto-PP splits eight ranks rather than building eight whole models.
    # The source has no optimizer and calls save_checkpoint with optimizer=None.
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=8",
        str(source),
        *shlex.split(load_model_args(profile.megatron_model_type)),
        "--hf-checkpoint",
        plan["model"]["root"],
        "--save",
        plan["output_root"] + "/torch-dist",
    ]


def preflight(plan: dict) -> dict:
    import torch
    from transformers import AutoConfig

    if torch.cuda.is_available():
        raise ValueError("conversion preflight must not allocate GPUs")
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("conversion output already exists")
    check_inputs(plan)
    config = AutoConfig.from_pretrained(
        plan["model"]["root"],
        trust_remote_code=False,
        local_files_only=True,
    )
    text = config.get_text_config()
    if (text.num_hidden_layers, text.hidden_size, text.vocab_size) != (64, 5120, 248320):
        raise ValueError("staged architecture differs from the Qwen conversion profile")
    argv = native_arguments(plan)
    return {
        "schema": "cyber_miles_conversion_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "native_argv_sha256": digest(argv),
        "model_files": len(plan["model"]["files"]),
        "gpu_conversion_verified": False,
    }


def _child(argv: list[str], log: Path, timeout: float) -> None:
    if timeout <= 0:
        raise TimeoutError("conversion startup allowance exhausted")
    env = {**os.environ, "PYTHONPATH": str(Path(argv[6]).parent.parent) + ":/root/Megatron-LM"}
    env.pop("CONVERT_KEEP_PP1", None)
    with log.open("x") as stream:
        os.chmod(log, 0o600)
        process = subprocess.Popen(
            argv,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if process.wait(timeout=timeout) != 0:
                raise RuntimeError("native conversion failed; private output preserved")
        finally:
            # Reap only our own torchrun group, including on interruption. Never
            # address processes by name or touch the Jobs API's Ray processes.
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=30)
            # A dead torchrun parent is not proof every child exited. Reap the
            # remaining owned group even if its parent has already returned.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def inventory(root: Path) -> list[dict]:
    if root.is_symlink():
        raise ValueError("indirect checkpoint root")
    if (root / "latest_checkpointed_iteration.txt").read_text().strip() != "release":
        raise ValueError("native release checkpoint tracker absent")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("indirect checkpoint payload")
        if path.is_file():
            size = path.stat().st_size
            if size == 0:
                raise ValueError("empty checkpoint payload")
            files.append({"path": str(path.relative_to(root)), "size": size})
    names = {r["path"] for r in files}
    if "release/.metadata" not in names or not any(n.endswith(".distcp") for n in names):
        raise ValueError("incomplete native distributed checkpoint")
    return files


def run(plan: dict) -> dict:
    import torch

    if os.environ.get("RUN_DIR") != plan["output_root"]:
        raise ValueError("Jobs API output binding drift")
    if torch.cuda.device_count() != 8:
        raise ValueError("native conversion must own exactly eight visible GPUs")
    started = time.monotonic()
    root = Path(plan["output_root"])
    _write(root / "STARTED.json", {"plan_sha256": digest(plan), "started_at": time.time()})
    destination = root / "torch-dist"

    def interrupted(signum, frame):
        raise TimeoutError("conversion interrupted or exceeded its fixed deadline")

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(DEADLINE_SECONDS)
    try:
        destination.mkdir(mode=0o700, exist_ok=False)
        check_inputs(plan)
        argv = native_arguments(plan)
        _child(
            argv, root / "private-conversion.log", DEADLINE_SECONDS - (time.monotonic() - started)
        )
        files = inventory(destination)
        return _write(
            root / "CONVERSION_COMPLETE.json",
            {
                "schema": SCHEMA,
                "status": "native_conversion_complete",
                "plan_sha256": digest(plan),
                "native_argv_sha256": digest(argv),
                "optimizer_steps": 0,
                "files": files,
                "completed_at": time.time(),
                "checkpoint_sha256_verified": False,
                "gpu_reload_verified": False,
            },
        )
    except BaseException as exc:
        _write(
            root / "FAILED.json",
            {
                "status": "failed",
                "error_class": type(exc).__name__,
                "plan_sha256": digest(plan),
            },
        )
        raise RuntimeError("conversion failed; preserve output, no automatic retry") from None
    finally:
        signal.alarm(0)
        for number, handler in previous.items():
            signal.signal(number, handler)


def seal(plan: dict, output: Path) -> dict:
    """Hash a terminal owned conversion on CPU after GPU release; not reload proof."""
    import torch

    if torch.cuda.is_available():
        raise ValueError("checkpoint sealing must not allocate GPUs")
    root = Path(plan["output_root"])
    receipt = json.loads((root / "CONVERSION_COMPLETE.json").read_text())
    if (
        (root / "FAILED.json").exists()
        or receipt["sha256"] != digest({k: v for k, v in receipt.items() if k != "sha256"})
        or receipt["plan_sha256"] != digest(plan)
        or receipt["optimizer_steps"] != 0
        or receipt["status"] != "native_conversion_complete"
    ):
        raise ValueError("unaccepted conversion completion")
    check_inputs(plan)
    checkpoint = root / "torch-dist"
    if output == checkpoint or checkpoint in output.parents:
        raise ValueError("seal must not alter the native checkpoint directory")
    files = inventory(checkpoint)
    if files != receipt["files"]:
        raise ValueError("checkpoint inventory changed after completion")
    hashed = [{**f, "sha256": _hash(checkpoint / f["path"])} for f in files]
    if inventory(checkpoint) != files:
        raise ValueError("checkpoint changed while sealing")
    return _write(
        output,
        {
            "schema": "cyber_miles_checkpoint_v1",
            "model": plan["model"],
            "image": miles.image_for(plan.get("profile", "qwen3.8-27b")),
            "root": str(checkpoint),
            "plan_sha256": digest(plan),
            "conversion_receipt_sha256": receipt["sha256"],
            "files": hashed,
            "optimizer_steps": 0,
            "gpu_reload_verified": False,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    ray = None
    task = None
    try:
        plan = json.loads(args.plan.read_text())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        import ray

        # HTTPMode starts a zero-GPU driver. Claim all eight GPUs for this one
        # native torchrun worker inside the existing allocation; do not start
        # another Ray cluster or override CUDA visibility by hand.
        ray.init(
            address="auto",
            log_to_driver=False,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1])
                    + ":"
                    + os.environ.get("PYTHONPATH", ""),
                }
            },
        )
        task = ray.remote(num_cpus=1, num_gpus=8)(run).remote(plan)
        result = ray.get(task, timeout=DEADLINE_SECONDS + 120)
        print(json.dumps({k: result[k] for k in ("status", "optimizer_steps", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    finally:
        if ray is not None and ray.is_initialized():
            if task is not None:
                ray.cancel(task, force=True)  # no-op when done; only this task
            ray.shutdown()  # disconnect this driver, not a node-wide Ray stop


if __name__ == "__main__":
    main()
