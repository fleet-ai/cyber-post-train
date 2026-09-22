"""Current-runtime compiler and entrypoint for the independent lane2 canary."""

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
from . import skyrl_lane2_authority as authority
from . import skyrl_prod9_hardening as hardening
from . import skyrl_prod9_training as current
from . import skyrl_training as historical
from .skyrl_prod9_rollout import offline_token_safe_tool_probe, runtime_binding

SCHEMA = "cyber_skyrl_lane2_training_v1"
MODULE = "training.skyrl_lane2_training"
PREFLIGHT_RECEIPT = Path("/dev/termination-log")
RUNTIME_FILES = (
    *current.RUNTIME_FILES,
    "training/skyrl_lane2_authority.py",
    "training/skyrl_lane2_data.py",
    "training/skyrl_lane2_training.py",
)


def _runtime() -> dict[str, str]:
    if len(RUNTIME_FILES) != len(set(RUNTIME_FILES)):
        raise ValueError("lane2 runtime file list contains duplicates")
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def _binding() -> dict[str, str]:
    return {"module": MODULE, **runtime_binding()}


def _historical_plan(plan: dict) -> dict:
    """Adapt only for immutable helper validation and base request fields."""
    result = copy.deepcopy(plan)
    result.pop("lane2_runtime", None)
    result.pop("qualification", None)
    result["schema"] = historical.SCHEMA
    result["runtime_sha256"] = digest(historical._runtime())
    result["execution"]["image"] = historical.IMAGE
    return result


def compile_rl(config: dict, *, relative_to: Path) -> dict:
    """Compile after independent lane2 authority checks, without old qualifiers."""
    source = copy.deepcopy(config)
    qualification = source.pop("qualification")
    checked = historical.compile_rl(source, relative_to=relative_to)
    binding = authority.validate_run_config(
        config,
        checked["data"],
        checked["model"],
        relative_to=relative_to,
    )
    if qualification != config["qualification"]:
        raise ValueError("lane2 qualification path changed")
    plan = {
        **checked,
        "schema": SCHEMA,
        "runtime_sha256": digest(_runtime()),
        "lane2_runtime": _binding(),
        "qualification": binding,
        "execution": {
            **checked["execution"],
            "image": authority.IMAGE,
            "cluster_target": binding["cluster_target"],
            "jobs_api_base_url": binding["jobs_api_base_url"],
        },
    }
    job_request(plan)
    return plan


def _base_request(plan: dict) -> dict:
    if (
        plan.get("schema") != SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("lane2_runtime") != _binding()
        or plan.get("execution", {}).get("image") != authority.IMAGE
    ):
        raise ValueError("lane2 plan/runtime binding changed")
    authority.validate_plan_binding(plan.get("qualification"), plan["data"], plan["arguments"])
    return historical.job_request(_historical_plan(plan))


def _bundled_request(plan: dict, extra_argv: list[str]) -> dict:
    base = _base_request(plan)
    files = _runtime()
    files.update(
        {
            path + "/__init__.py": ""
            for path in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    request = {
        key: base[key]
        for key in (
            "name",
            "title",
            "run_dir",
            "workers",
            "gpus_per_worker",
            "resources",
            "priority_class",
            "requeueIfPreempted",
            "failureAlerts",
            "secrets",
        )
    }
    request["image"] = authority.IMAGE
    request["title"] = plan["run_name"] + " current-runtime SkyRL lane2"
    request["env"] = {
        key: value
        for key, value in base["env"].items()
        if not key.startswith("CYBER_RUNTIME_BUNDLE")
    }
    request["env"].update(plan["qualification"]["environment"])
    return bundled_request(
        request,
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan), *extra_argv],
    )


def job_request(plan: dict) -> dict:
    return _bundled_request(plan, [])


def preflight_request(plan: dict, *, receipt: str = "/dev/termination-log") -> dict:
    if receipt != str(PREFLIGHT_RECEIPT):
        raise ValueError("lane2 CPU preflight receipt path changed")
    return _bundled_request(plan, ["--preflight", "--receipt", receipt])


def _write_preflight_receipt(path: Path, value: dict) -> None:
    if path != PREFLIGHT_RECEIPT:
        raise ValueError("lane2 CPU preflight receipt path changed")
    payload = {**value, "receipt_sha256": digest(value)}
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def validate_preview(plan: dict, request: dict, preview: dict) -> dict:
    if job_request(plan) != request:
        raise JobsError("lane2 preview request differs from its immutable plan")
    return historical.validate_gpu_runtime_preview(request, preview)


def preflight(plan: dict) -> dict:
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("lane2 CPU preflight must use image user 1000:100")
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        raise ValueError("lane2 CPU preflight is CPU-only")
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("lane2 output already exists")
    rows = historical.check_artifacts(plan)
    modules = historical.native_source()
    args = skyrl.SkyRLConfig(**plan["arguments"])
    skyrl.native_config(args)
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["root"], trust_remote_code=False, local_files_only=True
    )
    if "sha256:" + historical.digest_template(tokenizer) != plan["data"]["template_sha256"]:
        raise ValueError("lane2 native template changed")
    for split in rows:
        historical.dataset(plan, tokenizer, split, rows[split])
    parsed = skyrl_episode.parse(
        '<tool_call>{"name":"bash","arguments":{"script":"true"}}</tool_call>'
        '<tool_call>{"name":"submit_report","arguments":{"flags":[],"explanation":""}}</tool_call>'
    )
    if parsed != [
        {"name": "bash", "arguments": {"script": "true"}},
        {"name": "submit_report", "arguments": {"flags": [], "explanation": ""}},
    ]:
        raise ValueError("lane2 ordered multi-tool parser changed")
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
        "schema": "cyber_skyrl_lane2_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "lane2_runtime": _binding(),
        **horizon,
        **fresh,
        "counts": {key: len(value) for key, value in rows.items()},
        "planned_steps": plan["arguments"]["steps"],
        "rl_qualified": False,
    }


def native_result(plan: dict) -> dict:
    return current.native_result(plan)


def native_source():
    """Expose the native loader required by the shared supervised lifecycle."""
    return historical.native_source()


def _native(plan: dict) -> None:
    current._native(plan)


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
            raise ValueError("lane2 native and CPU preflight modes are exclusive")
        if args.receipt is not None and not args.preflight:
            raise ValueError("lane2 receipt is only valid for CPU preflight")
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.sha256:
            raise ValueError("lane2 plan digest mismatch")
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
