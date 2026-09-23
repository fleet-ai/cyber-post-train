"""Fresh prod9-only SkyRL compiler and runtime entrypoint.

This module binds the long-horizon canary to the token-safe prod9 recorder
without changing the sealed historical SkyRL compiler or rollout module.  It
is accepted only through the separately reviewed generic Jobs API rail; the
historical direct-RayJob rail continues to reject this schema.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import shlex
import sys
from contextlib import suppress
from pathlib import Path

from cyber_post_train.jobs import JobsError, bundled_request, digest, validate_request

from . import skyrl, skyrl_episode
from . import skyrl_prod9_hardening as hardening
from . import skyrl_reward_rayjob as legacy_direct
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
STAGE_SCHEMA = "cyber_skyrl_prod9_rebind_stage_v1"
STAGE_RECEIPT_SCHEMA = "cyber_skyrl_prod9_rebind_stage_receipt_v1"
TERMINATION_MESSAGE_MAX_BYTES = 16384
PREFLIGHT_FAILURE_SCHEMA = "cyber_skyrl_prod9_training_cpu_preflight_failure_v1"
WATCHDOG_HARD_SECONDS = legacy_direct.MAXIMUM_SECONDS
_PREFLIGHT_STAGE = "not_started"
RUNTIME_FILES = (
    *historical.RUNTIME_FILES,
    "training/skyrl_reward_rayjob.py",
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


def _seal(value: dict) -> dict:
    """Return a canonical, self-digesting public receipt or stage specification."""
    body = {key: item for key, item in value.items() if key not in {"sha256", "receipt_sha256"}}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: object, schema: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError("prod9 rebind evidence is not an object")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(body):
        raise ValueError("prod9 rebind evidence digest/schema changed")
    return value


def stage_spec(identity: legacy_direct.RailIdentity, predecessor_manifest: dict) -> dict:
    """Seal the zero-GPU, SFS-local transformation needed before compilation.

    The predecessor package is the only permitted source.  This specification
    deliberately contains only public manifest metadata, never its task rows.
    The resulting successor manifest becomes the sole data input to the later
    training-plan compilation.
    """
    if not isinstance(identity, legacy_direct.RailIdentity):
        raise ValueError("prod9 rebind identity is invalid")
    body = {key: item for key, item in predecessor_manifest.items() if key != "sha256"}
    files = predecessor_manifest.get("files") if isinstance(predecessor_manifest, dict) else None
    if (
        predecessor_manifest.get("schema") != "cyber_skyrl_data_v1"
        or predecessor_manifest.get("name") != identity.predecessor_run_name
        or predecessor_manifest.get("sha256") != "sha256:" + digest(body)
        or not isinstance(files, dict)
        or set(files) != {"train", "dev"}
        or any(
            not isinstance(files[split], dict)
            or files[split].get("path") != split + ".jsonl"
            or type(files[split].get("rows")) is not int
            or files[split]["rows"] < 1
            or not isinstance(files[split].get("sha256"), str)
            or not files[split]["sha256"].startswith("sha256:")
            for split in ("train", "dev")
        )
    ):
        raise ValueError("prod9 predecessor manifest is not an exact private-data source")
    return _seal(
        {
            "schema": STAGE_SCHEMA,
            "name": identity.stage_name,
            "identity": identity.sealed_mapping(),
            "source": identity.predecessor_data_root,
            "destination": identity.data_root,
            "predecessor_manifest": predecessor_manifest,
            "predecessor_manifest_sha256": predecessor_manifest["sha256"],
            "image": historical.IMAGE,
            "gpus": 0,
            "scientific_work": {
                "task_rows_read": 0,
                "rollout_episodes": 0,
                "optimizer_steps": 0,
                "checkpoints": 0,
            },
        }
    )


def _stage_identity(value: object) -> tuple[dict, legacy_direct.RailIdentity]:
    stage = _validate_seal(value, STAGE_SCHEMA)
    identity = legacy_direct.identity_from_mapping(stage.get("identity"))
    predecessor = stage.get("predecessor_manifest")
    expected = stage_spec(identity, predecessor)
    if stage != expected:
        raise ValueError("prod9 rebind stage specification changed")
    return stage, identity


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
    request = _bundled_request(plan, [])
    claim = plan["output_root"] + "/.prod9-training-create-claim-v1"
    request["command"] = "mkdir " + shlex.quote(claim) + " && exec " + request["command"]
    validate_request(request)
    return request


def preflight_request(plan: dict, *, receipt: str = "/dev/termination-log") -> dict:
    """Build the same fresh closure for the CPU-only exact-image preflight."""
    if receipt != "/dev/termination-log":
        raise ValueError("prod9 CPU preflight receipt path changed")
    return _bundled_request(plan, ["--preflight", "--receipt", receipt])


def stage_request(stage: dict) -> dict:
    """Bundle the exact-image, zero-GPU SFS-local rebind entrypoint.

    This bundle intentionally includes the historical pure rebind helper but
    never calls its historical renderer, authorizer, or create rail.  It is
    the narrow bridge that lets an SFS-mounted Pod turn an old private input
    package into a fresh run-ID-bound package without exposing its rows to the
    operator machine.
    """
    checked, identity = _stage_identity(stage)
    files = _runtime()
    files.update(
        {
            path + "/__init__.py": ""
            for path in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    files["stage.json"] = json.dumps(checked, sort_keys=True, separators=(",", ":"))
    files["identity.json"] = json.dumps(
        identity.sealed_mapping(), sort_keys=True, separators=(",", ":")
    )
    return bundled_request(
        {
            "name": identity.stage_name,
            "title": identity.stage_name + " zero-GPU SFS rebind",
            "run_dir": "/mnt/sfs/jobs/" + identity.stage_name,
            "image": checked["image"],
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": {
                "cpu_request": "4",
                "cpu_limit": "8",
                "memory_request": "32Gi",
                "memory_limit": "48Gi",
            },
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "failureAlerts": False,
            "secrets": [],
            "env": {"PYTHONUNBUFFERED": "1"},
        },
        files,
        MODULE,
        [
            "--stage",
            "--stage-spec",
            "stage.json",
            "--identity",
            "identity.json",
            "--receipt",
            str(PREFLIGHT_RECEIPT),
        ],
    )


def stage_rebind(stage: dict, *, identity: legacy_direct.RailIdentity) -> dict:
    """Rebind only private row run IDs inside the exact SFS source package."""
    checked, bound = _stage_identity(stage)
    if identity != bound:
        raise ValueError("prod9 rebind identity differs from the sealed stage specification")
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("prod9 rebind must use image user 1000:100")
    source, destination = Path(bound.predecessor_data_root), Path(bound.data_root)
    if (
        source.is_symlink()
        or not source.is_dir()
        or destination.exists()
        or destination.is_symlink()
        or destination.parent.is_symlink()
    ):
        raise ValueError("prod9 rebind SFS source or destination changed")
    try:
        observed_predecessor = json.loads((source / "manifest.json").read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("prod9 rebind predecessor manifest is unreadable") from exc
    if observed_predecessor != checked["predecessor_manifest"]:
        raise ValueError("prod9 rebind predecessor manifest differs from the sealed source")
    successor = legacy_direct.rebind_private_source_for_identity(source, destination, bound)
    successor_body = {key: item for key, item in successor.items() if key != "sha256"}
    expected_names = {"manifest.json", "split.json", "task-set.json", "train.jsonl", "dev.jsonl"}
    found = {path.name: path for path in destination.iterdir()}
    if (
        successor.get("schema") != "cyber_skyrl_data_v1"
        or successor.get("name") != bound.run_name
        or successor.get("sha256") != "sha256:" + digest(successor_body)
        or set(found) != expected_names
        or any(path.is_symlink() or not path.is_file() for path in found.values())
    ):
        raise ValueError("prod9 rebind successor package is malformed")
    try:
        persisted = json.loads((destination / "manifest.json").read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("prod9 rebind successor manifest is unreadable") from exc
    if persisted != successor:
        raise ValueError("prod9 rebind successor manifest changed")
    files = []
    for name in sorted(found):
        path = found[name]
        payload = path.read_bytes()
        files.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )
    by_name = {item["path"]: item for item in files}
    if any(
        by_name[successor["files"][split]["path"]]["sha256"] != successor["files"][split]["sha256"]
        for split in ("train", "dev")
    ):
        raise ValueError("prod9 rebind successor payload digest changed")
    return {
        "schema": STAGE_RECEIPT_SCHEMA,
        "status": "published",
        "stage_spec_sha256": checked["sha256"],
        "identity_sha256": bound.sealed_mapping()["sha256"],
        "source": str(source),
        "destination": str(destination),
        "predecessor_manifest_sha256": checked["predecessor_manifest_sha256"],
        "successor_manifest": successor,
        "successor_manifest_sha256": successor["sha256"],
        "files": files,
        "gpus": 0,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
        **checked["scientific_work"],
    }


def _write_termination_receipt(path: Path, value: dict) -> None:
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
    if len(encoded) > TERMINATION_MESSAGE_MAX_BYTES:
        raise ValueError("prod9 CPU termination receipt is too large")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _write_preflight_receipt(path: Path, value: dict) -> None:
    """Keep the fixed-path preflight writer as a named public test seam."""
    _write_termination_receipt(path, value)


def _write_preflight_failure_receipt(path: Path, plan: object, exc: BaseException) -> None:
    """Persist only a bounded phase and exception class for failed CPU probes."""
    value = {
        "schema": PREFLIGHT_FAILURE_SCHEMA,
        "status": "failed",
        "stage": _PREFLIGHT_STAGE,
        "error_class": type(exc).__name__,
        "plan_sha256": digest(plan) if isinstance(plan, dict) else "",
        "gpus": 0,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
    }
    _write_termination_receipt(path, value)


def _write_stage_receipt(path: Path, value: dict) -> None:
    """Write only the zero-GPU stage receipt to Kubernetes' fixed file."""
    _write_termination_receipt(path, value)


def _write_gpu_termination_receipt(value: dict) -> None:
    """Expose the sealed native completion to the exact-UID observer."""
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("status") != "native_loop_returned" or value.get("sha256") != digest(body):
        raise ValueError("prod9 GPU termination receipt is not a sealed native completion")
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > TERMINATION_MESSAGE_MAX_BYTES:
        raise ValueError("prod9 GPU termination receipt is too large")
    fd = os.open(PREFLIGHT_RECEIPT, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def validate_preview(plan: dict, request: dict, preview: dict) -> dict:
    """Use the shared GPU identity checks only after fresh request equality."""
    if job_request(plan) != request:
        raise JobsError("prod9 preview request differs from its immutable plan")
    return historical.validate_gpu_runtime_preview(request, preview)


def _preflight_native_config(args: skyrl.SkyRLConfig):
    """Run the native parse/validation path with bounded diagnostic phases."""
    global _PREFLIGHT_STAGE
    _PREFLIGHT_STAGE = "native_config_overrides"
    values = skyrl.overrides(args)
    _PREFLIGHT_STAGE = "native_config_modules"
    modules = {name: skyrl_episode._module(name, sha) for name, sha in skyrl.NATIVE_SOURCES.items()}
    _PREFLIGHT_STAGE = "native_config_parse"
    cfg = modules["skyrl.train.config.config"].SkyRLTrainConfig.from_cli_overrides(values)
    _PREFLIGHT_STAGE = "native_config_validate"
    modules["skyrl.train.utils.utils"].validate_cfg(cfg)
    return cfg


def preflight(plan: dict) -> dict:
    """CPU-only exact-image check for the fresh runtime closure.

    It deliberately does not construct a Kubernetes object.  A future direct
    rail must wrap this entrypoint in a fresh root-annotated zero-GPU Job,
    rather than reuse the historical rail that cannot prove this runtime.
    """
    global _PREFLIGHT_STAGE
    _PREFLIGHT_STAGE = "runtime_identity"
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("prod9 CPU preflight must use image user 1000:100")
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        raise ValueError("prod9 CPU preflight is CPU-only")
    _PREFLIGHT_STAGE = "request_and_artifacts"
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("prod9 output already exists")
    rows = historical.check_artifacts(plan)
    modules = historical.native_source()
    _PREFLIGHT_STAGE = "native_config_arguments"
    args = skyrl.SkyRLConfig(**plan["arguments"])
    _preflight_native_config(args)
    _PREFLIGHT_STAGE = "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["root"], trust_remote_code=False, local_files_only=True
    )
    if "sha256:" + historical.digest_template(tokenizer) != plan["data"]["template_sha256"]:
        raise ValueError("native template changed")
    _PREFLIGHT_STAGE = "dataset"
    for split in rows:
        historical.dataset(plan, tokenizer, split, rows[split])
    _PREFLIGHT_STAGE = "parser_and_limits"
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
    _PREFLIGHT_STAGE = "long_horizon_probe"
    horizon = asyncio.run(
        skyrl_episode.offline_long_horizon_probe(
            plan["model"],
            tokenizer,
            Path(modules["skyrl.train.generators.utils"].__file__),
        )
    )
    _PREFLIGHT_STAGE = "token_safe_tool_probe"
    fresh = asyncio.run(
        offline_token_safe_tool_probe(
            plan["model"],
            tokenizer,
            Path(modules["skyrl.train.generators.utils"].__file__),
        )
    )
    _PREFLIGHT_STAGE = "passed"
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
        "output_absent": True,
        "wandb_create_once": {
            "entity": plan["arguments"]["wandb_entity"],
            "project": plan["arguments"]["wandb_project"],
            "run_id": plan["arguments"]["wandb_run_id"],
            "resume": "never",
        },
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


def native_source():
    """Expose the native loader required by the shared supervised lifecycle."""
    return historical.native_source()


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
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--stage-spec", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    plan: object = None
    try:
        modes = sum((args.native, args.preflight, args.stage))
        if modes > 1:
            raise ValueError("prod9 native, CPU preflight, and stage modes are exclusive")
        if args.stage:
            if (
                args.plan is not None
                or args.sha256 is not None
                or args.stage_spec is None
                or args.identity is None
            ):
                raise ValueError(
                    "prod9 stage accepts only its sealed stage specification and identity"
                )
            stage = json.loads(args.stage_spec.read_bytes())
            identity = legacy_direct.load_identity(args.identity)
            result = stage_rebind(stage, identity=identity)
            if args.receipt is not None:
                _write_stage_receipt(args.receipt, result)
            print(json.dumps({"status": result["status"], "sha256": digest(result)}))
            return
        if args.plan is None or args.sha256 is None:
            raise ValueError("prod9 training and CPU preflight require an immutable plan digest")
        if args.stage_spec is not None or args.identity is not None:
            raise ValueError("prod9 stage inputs are not valid for training or CPU preflight")
        if args.receipt is not None and not args.preflight:
            raise ValueError("prod9 receipt is only valid for CPU preflight or stage")
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
            _write_gpu_termination_receipt(result)
            print(json.dumps({key: result[key] for key in ("status", "sha256")}))
    except BaseException as exc:
        if args.preflight and args.receipt == PREFLIGHT_RECEIPT:
            with suppress(Exception):
                _write_preflight_failure_receipt(args.receipt, plan, exc)
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
