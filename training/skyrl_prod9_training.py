"""Fresh prod9-only SkyRL compiler and runtime entrypoint.

This module binds the long-horizon canary to the token-safe prod9 recorder
without changing the sealed historical SkyRL compiler or rollout module.  It
is intentionally preparation-only: the historical direct-RayJob rail rejects
this schema, so a future launch needs a separately reviewed fresh direct rail.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

from cyber_post_train.jobs import JobsError, bundled_request, digest

from . import skyrl, skyrl_episode
from . import skyrl_prod9_hardening as hardening
from . import skyrl_training as historical
from .skyrl_prod9_rollout import (
    RECORDER_IMPLEMENTATION,
    Generator,
    offline_token_safe_tool_probe,
    runtime_binding,
)

SCHEMA = "cyber_skyrl_prod9_training_v1"
MODULE = "training.skyrl_prod9_training"
PREFLIGHT_RECEIPT = Path("/dev/termination-log")
RUNTIME_FILES = (
    *historical.RUNTIME_FILES,
    "training/skyrl_prod9_hardening.py",
    "training/skyrl_prod9_rollout.py",
    "training/skyrl_prod9_training.py",
)


def _runtime() -> dict[str, str]:
    """Read the exact fresh source closure embedded in a prod9 bundle."""
    if len(RUNTIME_FILES) != len(set(RUNTIME_FILES)):
        raise ValueError("prod9 runtime file list contains duplicates")
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def _binding() -> dict[str, str]:
    """Keep the plan's declared runtime equal to the executed source binding."""
    return {"module": MODULE, **runtime_binding()}


def _historical_plan(plan: dict) -> dict:
    """Adapt a validated prod9 plan only for pure historical helper checks."""
    result = copy.deepcopy(plan)
    result.pop("prod9_runtime", None)
    result["schema"] = historical.SCHEMA
    result["runtime_sha256"] = digest(historical._runtime())
    return result


def compile_rl(config: dict, *, relative_to: Path) -> dict:
    """Compile a fresh runtime after the unchanged science/configuration checks."""
    checked = historical.compile_rl(config, relative_to=relative_to)
    plan = {
        **checked,
        "schema": SCHEMA,
        "runtime_sha256": digest(_runtime()),
        "prod9_runtime": _binding(),
    }
    job_request(plan)
    return plan


def _base_request(plan: dict) -> dict:
    if (
        plan.get("schema") != SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("prod9_runtime") != _binding()
    ):
        raise ValueError("prod9 SkyRL plan/runtime binding changed")
    return historical.job_request(_historical_plan(plan))


def reject_historical_direct_rail(plan: dict) -> dict[str, str]:
    """Prove the historical direct rail cannot render a fresh prod9 plan.

    That rail starts by calling ``historical.job_request``.  Keep this check
    explicit in offline preparation so a future launcher cannot accidentally
    pass the fresh plan through a historical entrypoint.
    """
    if plan.get("schema") != SCHEMA:
        raise ValueError("not a fresh prod9 plan")
    try:
        historical.job_request(plan)
    except ValueError as exc:
        if str(exc) != "SkyRL plan/runtime drift":
            raise ValueError("historical direct rail rejected prod9 unexpectedly") from exc
        return {
            "status": "rejected",
            "reason": "historical_direct_rail_cannot_render_fresh_prod9_runtime",
        }
    raise ValueError("historical direct rail accepted a fresh prod9 plan")


def _bundled_request(plan: dict, extra_argv: list[str]) -> dict:
    """Build the one source closure used by both GPU and CPU prod9 gates."""
    base_request = _base_request(plan)
    files = _runtime()
    files.update(
        {
            path + "/__init__.py": ""
            for path in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    request = {
        key: base_request[key]
        for key in (
            "name",
            "title",
            "run_dir",
            "image",
            "workers",
            "gpus_per_worker",
            "resources",
            "priority_class",
            "requeueIfPreempted",
            "failureAlerts",
            "secrets",
        )
    }
    request["env"] = {
        key: value
        for key, value in base_request["env"].items()
        if not key.startswith("CYBER_RUNTIME_BUNDLE")
    }
    return bundled_request(
        request,
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan), *extra_argv],
    )


def job_request(plan: dict) -> dict:
    """Build one fresh GPU bundle whose entrypoint is this module, not historical."""
    return _bundled_request(plan, [])


def preflight_request(plan: dict, *, receipt: str = "/dev/termination-log") -> dict:
    """Build the same fresh closure for the CPU-only exact-image preflight."""
    if receipt != "/dev/termination-log":
        raise ValueError("prod9 CPU preflight receipt path changed")
    return _bundled_request(plan, ["--preflight", "--receipt", receipt])


def _write_preflight_receipt(path: Path, value: dict) -> None:
    """Replace Kubernetes' pre-created termination file with a sealed receipt.

    ``/dev/termination-log`` already exists in a Kubernetes container, so the
    create-once receipt writer used for durable SFS artifacts is deliberately
    inappropriate here.  This helper has one fixed destination and cannot
    create a receipt at an arbitrary path.
    """
    if path != PREFLIGHT_RECEIPT:
        raise ValueError("prod9 CPU preflight receipt path changed")
    payload = {**value, "receipt_sha256": digest(value)}
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def validate_preview(plan: dict, request: dict, preview: dict) -> dict:
    """Use the shared GPU identity checks only after fresh request equality."""
    if job_request(plan) != request:
        raise JobsError("prod9 preview request differs from its immutable plan")
    return historical.validate_gpu_runtime_preview(request, preview)


def preflight(plan: dict) -> dict:
    """CPU-only exact-image check for the fresh runtime closure.

    It deliberately does not construct a Kubernetes object.  A future direct
    rail must wrap this entrypoint in a fresh root-annotated zero-GPU Job,
    rather than reuse the historical rail that cannot prove this runtime.
    """
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("prod9 CPU preflight must use image user 1000:100")
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        raise ValueError("prod9 CPU preflight is CPU-only")
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("prod9 output already exists")
    rows = historical.check_artifacts(plan)
    modules = historical.native_source()
    args = skyrl.SkyRLConfig(**plan["arguments"])
    skyrl.native_config(args)
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["root"], trust_remote_code=False, local_files_only=True
    )
    if "sha256:" + historical.digest_template(tokenizer) != plan["data"]["template_sha256"]:
        raise ValueError("native template changed")
    for split in rows:
        historical.dataset(plan, tokenizer, split, rows[split])
    multi_tool_probe = skyrl_episode.parse(
        '<tool_call>{"name":"bash","arguments":{"script":"true"}}</tool_call>'
        '<tool_call>{"name":"submit_report","arguments":{"flags":[],"explanation":""}}</tool_call>'
    )
    if multi_tool_probe != [
        {"name": "bash", "arguments": {"script": "true"}},
        {"name": "submit_report", "arguments": {"flags": [], "explanation": ""}},
    ]:
        raise ValueError("native ordered multi-tool parser changed")
    hardening.validate_episode_limits(plan["data"]["limits"])
    horizon = asyncio.run(
        skyrl_episode.offline_long_horizon_probe(
            plan["model"],
            tokenizer,
            Path(modules["skyrl.train.generators.utils"].__file__),
        )
    )
    fresh = asyncio.run(
        offline_token_safe_tool_probe(
            plan["model"],
            tokenizer,
            Path(modules["skyrl.train.generators.utils"].__file__),
        )
    )
    return {
        "schema": "cyber_skyrl_prod9_training_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "prod9_runtime": _binding(),
        "native_parser_checked": True,
        "ordered_multi_tool_parser_checked": True,
        **horizon,
        **fresh,
        "counts": {key: len(value) for key, value in rows.items()},
        "planned_steps": plan["arguments"]["steps"],
        "rl_qualified": False,
    }


def native_result(plan: dict) -> dict:
    """Require every terminal batch receipt to name the fresh recorder."""
    root = Path(plan["output_root"])
    for directory in (root / "episodes/batches").iterdir():
        value = json.loads((directory / "COLLECTED.json").read_bytes())
        if value.get("recorder_implementation") != RECORDER_IMPLEMENTATION:
            raise ValueError("prod9 batch did not use the token-safe recorder")
    return historical.native_result(plan)


def _native(plan: dict) -> None:
    """Run native SkyRL with the fresh prod9 rollout generator."""
    import ray
    import wandb
    from skyrl.backends.skyrl_train.utils.ppo_utils import sync_registries

    rows, modules = historical.check_artifacts(plan), historical.native_source()
    args = skyrl.SkyRLConfig(**plan["arguments"])
    cfg = skyrl.native_config(args)
    base = modules["skyrl.train.entrypoints.main_base"].BasePPOExp

    class Experiment(base):
        def get_train_dataset(self):
            return historical.dataset(plan, self.tokenizer, "train", rows["train"])

        def get_eval_dataset(self):
            return historical.dataset(plan, self.tokenizer, "dev", rows["dev"])

        def get_generator(self, cfg, tokenizer, engine):
            return Generator(
                args.data_manifest,
                plan["data"]["sha256"],
                tokenizer,
                engine,
                Path(args.output_root) / "episodes",
                response_tokens=args.response_tokens,
                repetitions={"train": args.samples_per_prompt, "eval": 1},
                concurrency=args.groups * args.samples_per_prompt,
            )

        def get_tracker(self):
            return historical.ScalarTracking(plan)

        def get_trajectory_logger(self):
            return None

    env = modules["skyrl.train.utils.utils"].prepare_runtime_environment(cfg)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1]) + ":" + os.environ.get("PYTHONPATH", "")
    )
    ray.init(address="auto", log_to_driver=False, runtime_env={"env_vars": env})
    code = 1
    try:
        sync_registries()
        experiment = Experiment(cfg)
        experiment.run()
        if experiment.trainer.global_step != args.steps:
            raise ValueError("native optimizer step limit changed")
        native_result(plan)
        code = 0
    finally:
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=code)
        finally:
            ray.shutdown()


def run(plan: dict, plan_path: Path) -> dict:
    from .rl_runtime import run as supervised_run

    return supervised_run(plan, plan_path, backend=sys.modules[__name__])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.native and args.preflight:
            raise ValueError("prod9 native and CPU preflight modes are exclusive")
        if args.receipt is not None and not args.preflight:
            raise ValueError("prod9 receipt is only valid for CPU preflight")
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        job_request(plan)
        historical.validate_gpu_runtime_user()
        if args.preflight:
            result = preflight(plan)
            if args.receipt is not None:
                _write_preflight_receipt(args.receipt, result)
            print(json.dumps({"status": result["status"], "sha256": digest(result)}))
        elif args.native:
            try:
                _native(plan)
            except BaseException as exc:
                from .rl_runtime import native_failure, native_rejection

                if native_rejection(plan, exc):
                    return
                with suppress(Exception):
                    native_failure(plan, exc)
                raise
        else:
            result = run(plan, args.plan)
            print(json.dumps({key: result[key] for key in ("status", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
