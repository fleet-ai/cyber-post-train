"""Append-only Fast3 compiler and runtime over the frozen prod9 implementation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
from contextlib import suppress
from pathlib import Path

from cyber_post_train.jobs import JobsError, bundled_request, digest, validate_request

from . import rl_reward_fast3 as reward
from . import skyrl, skyrl_fast3_rollout, skyrl_posttrain
from . import skyrl_prod9_training as historical

SCHEMA = "cyber_skyrl_fast3_training_v1"
MODULE = "training.skyrl_fast3_training"
PREFLIGHT_RECEIPT = historical.PREFLIGHT_RECEIPT
RUNTIME_FILES = reward.FAST3_PORT_FILES
_HISTORICAL_POSTTRAIN_VALIDATE = skyrl_posttrain._validate_plan


def _runtime() -> dict[str, str]:
    if len(RUNTIME_FILES) != len(set(RUNTIME_FILES)):
        raise ValueError("Fast3 runtime file list contains duplicates")
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def _binding() -> dict:
    return {
        "module": MODULE,
        **skyrl_fast3_rollout.runtime_binding(),
    }


def _legacy_plan(plan: dict) -> dict:
    result = copy.deepcopy(plan)
    result.pop("fast3_runtime", None)
    result["schema"] = historical.SCHEMA
    result["runtime_sha256"] = digest(historical._runtime())
    result["qualification"] = plan["qualification"]["historical_binding"]
    return result


def _validated(plan: dict) -> skyrl.SkyRLConfig:
    if type(plan) is not dict:
        raise ValueError("Fast3 plan is not an object")
    binding = reward.validate_plan_binding(
        plan.get("qualification"), plan.get("data", {}), plan.get("arguments", {})
    )
    args = skyrl.SkyRLConfig(**plan["arguments"])
    if (
        plan.get("schema") != SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("fast3_runtime") != _binding()
        or plan.get("prod9_runtime") != historical._binding()
        or plan.get("native_overrides") != skyrl.overrides(args)
        or plan.get("execution", {}).get("image") != binding["image"]
        or plan.get("execution", {}).get("cluster_target") != binding["cluster_target"]
        or plan.get("execution", {}).get("jobs_api_base_url") != binding["jobs_api_base_url"]
        or plan.get("run_name") != args.name
        or plan.get("output_root") != args.output_root
    ):
        raise ValueError("Fast3 plan/runtime binding changed")
    legacy = _legacy_plan(plan)
    historical._base_request(legacy)
    _HISTORICAL_POSTTRAIN_VALIDATE(legacy)
    return args


def compile_rl(config: dict, *, relative_to: Path) -> dict:
    normalized = reward.normalized_historical_config(config)
    checked = historical.compile_rl(normalized, relative_to=relative_to)
    qualification = reward.validate_run_config(
        config,
        checked["data"],
        checked["model"],
        relative_to=relative_to,
    )
    plan = {
        **checked,
        "schema": SCHEMA,
        "runtime_sha256": digest(_runtime()),
        "qualification": qualification,
        "fast3_runtime": _binding(),
    }
    job_request(plan)
    return plan


def _bundled_request(plan: dict, extra_argv: list[str]) -> dict:
    _validated(plan)
    base = historical.job_request(_legacy_plan(plan))
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
        for key, value in base["env"].items()
        if not key.startswith("CYBER_RUNTIME_BUNDLE")
    }
    return bundled_request(
        request,
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan), *extra_argv],
    )


def job_request(plan: dict) -> dict:
    request = _bundled_request(plan, [])
    claim = plan["output_root"] + "/.fast3-training-create-claim-v1"
    request["command"] = "mkdir " + shlex.quote(claim) + " && exec " + request["command"]
    validate_request(request)
    return request


def preflight_request(plan: dict, *, receipt: str = "/dev/termination-log") -> dict:
    if receipt != str(PREFLIGHT_RECEIPT):
        raise ValueError("Fast3 CPU preflight receipt path changed")
    return _bundled_request(plan, ["--preflight", "--receipt", receipt])


def validate_preview(plan: dict, request: dict, preview: dict) -> dict:
    if job_request(plan) != request:
        raise JobsError("Fast3 preview request differs from its immutable plan")
    return historical.historical.validate_gpu_runtime_preview(request, preview)


def check_artifacts(plan: dict):
    _validated(plan)
    legacy = historical._historical_plan(_legacy_plan(plan))
    return historical.historical.check_artifacts(legacy)


def preflight(plan: dict) -> dict:
    _validated(plan)
    result = historical.preflight(_legacy_plan(plan))
    result.update(
        {
            "schema": "cyber_skyrl_fast3_training_cpu_preflight_v1",
            "plan_sha256": digest(plan),
            "request_sha256": digest(job_request(plan)),
            "fast3_runtime": _binding(),
            "generation_retry_policy_sha256": plan["qualification"]["fast3"][
                "generation_retry_policy"
            ]["sha256"],
        }
    )
    return result


def native_result(plan: dict) -> dict:
    _validated(plan)
    return historical.native_result(plan)


def native_source():
    return historical.native_source()


def _generator(plan: dict, args: skyrl.SkyRLConfig, tokenizer, engine):
    """Construct the one plan-bound Fast3 rollout generator."""
    policy = plan["qualification"]["fast3"]["generation_retry_policy"]
    return skyrl_fast3_rollout.Generator(
        args.data_manifest,
        plan["data"]["sha256"],
        tokenizer,
        engine,
        Path(args.output_root) / "episodes",
        response_tokens=args.response_tokens,
        repetitions={"train": args.samples_per_prompt, "eval": 1},
        concurrency=args.groups * args.samples_per_prompt,
        generation_retry_policy=policy,
    )


def _native(plan: dict) -> None:
    import ray
    import wandb
    from skyrl.backends.skyrl_train.utils.ppo_utils import sync_registries

    rows, modules = check_artifacts(plan), native_source()
    args = skyrl.SkyRLConfig(**plan["arguments"])
    cfg = skyrl.native_config(args)
    base = modules["skyrl.train.entrypoints.main_base"].BasePPOExp

    class Experiment(base):
        def get_train_dataset(self):
            return historical.historical.dataset(plan, self.tokenizer, "train", rows["train"])

        def get_eval_dataset(self):
            return historical.historical.dataset(plan, self.tokenizer, "dev", rows["dev"])

        def get_generator(self, cfg, tokenizer, engine):
            return _generator(plan, args, tokenizer, engine)

        def get_tracker(self):
            return historical.historical.ScalarTracking(plan)

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
    plan: object = None
    try:
        if args.native and args.preflight:
            raise ValueError("Fast3 native and CPU preflight modes are exclusive")
        if args.receipt is not None and not args.preflight:
            raise ValueError("Fast3 receipt is only valid for CPU preflight")
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        job_request(plan)
        historical.historical.validate_gpu_runtime_user()
        if args.preflight:
            result = preflight(plan)
            if args.receipt is not None:
                historical._write_preflight_receipt(args.receipt, result)
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
            historical._write_gpu_termination_receipt(result)
            print(json.dumps({key: result[key] for key in ("status", "sha256")}))
    except BaseException as exc:
        if args.preflight and args.receipt == PREFLIGHT_RECEIPT:
            with suppress(Exception):
                historical._write_preflight_failure_receipt(args.receipt, plan, exc)
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
