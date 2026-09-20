"""One-node SkyRL setup probe with no task, rollout, optimizer or checkpoint path.

The probe is deliberately a different schema, name, output directory and
entrypoint from scientific training.  It may only target the development Jobs
API.  Its 20 minute setup budget plus five minute cleanup budget is a hard
25-minute process limit; Kubernetes release is accepted only by a separate
post-terminal receipt bound to the exact API and Kubernetes identities.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from contextlib import suppress
from pathlib import Path

from cyber_post_train.jobs import API_URLS, JobsError, bundled_request, digest, quantity

from . import skyrl

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cyber_skyrl_topology_probe_v1"
CONFIG_SCHEMA = "cyber_skyrl_topology_probe_config_v1"
RECEIPT_SCHEMA = "cyber_skyrl_topology_probe_receipt_v1"
RELEASE_SCHEMA = "cyber_skyrl_topology_probe_release_v1"
MODULE = "training.skyrl_topology_probe"
CONFIG_PATH = ROOT / "configs/qualification/qwen38-skyrl-topology-probe-dev-v1.json"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
RUNTIME_FILES = (
    "training/skyrl_topology_probe.py",
    "training/skyrl.py",
    "training/skyrl_episode.py",
    "training/dense.py",
    "training/io.py",
    "training/qwen_tools.py",
    "training/rl_episode.py",
    "evals/fleet/opencode_self_hosted.py",
    "cyber_post_train/jobs.py",
)


def _seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: dict, schema: str) -> None:
    if value.get("schema") != schema or value != _seal(value):
        raise ValueError("SkyRL topology probe digest changed")


def _runtime() -> dict[str, str]:
    return {name: (ROOT / name).read_text() for name in RUNTIME_FILES}


def _config(path: Path) -> dict:
    from .sft import read_mapping

    if path.resolve() != CONFIG_PATH.resolve() or path.is_symlink():
        raise ValueError("unknown SkyRL topology probe configuration")
    value = read_mapping(path)
    _validate_seal(value, CONFIG_SCHEMA)
    expected = {
        "cluster_target": "dev",
        "jobs_api_base_url": API_URLS["dev"],
        "priority": "c1",
        "workers": 1,
        "gpus_per_worker": 8,
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    }
    if (
        value["execution"] != expected
        or value["engine"]
        != {"num_engines": 2, "tensor_parallel_size": 4, "context_tokens": 98304}
        or value["deadlines"]
        != {"setup_seconds": 1200, "cleanup_seconds": 300, "total_seconds": 1500}
        or value["deadlines"]["setup_seconds"] + value["deadlines"]["cleanup_seconds"]
        != value["deadlines"]["total_seconds"]
        or value["submission_gate"].get("preview_authorized") is not True
        or value["submission_gate"].get("submission_authorized") is not False
    ):
        raise ValueError("SkyRL topology probe contract changed")
    return value


def compile_probe(path: Path) -> dict:
    from .models import bound_model
    from .sft import read_mapping

    value = _config(path)
    model = value["model"]
    bound = bound_model(
        read_mapping(path.parent / model["lock"]),
        read_mapping(path.parent / model["weights"]),
        model["root"],
    )
    arguments = skyrl.SkyRLConfig(
        name=value["name"],
        output_root=value["output_root"],
        model_root=bound["root"],
        train_data="/mnt/sfs/datasets/q38-probe/unused-train.jsonl",
        dev_data="/mnt/sfs/datasets/q38-probe/unused-dev.jsonl",
        data_manifest="/mnt/sfs/datasets/q38-probe/unused-manifest.json",
        train_rows=2,
        dev_rows=1,
        wandb_entity="disabled",
        wandb_project="disabled",
        wandb_run_id=value["name"],
        groups=1,
        samples_per_prompt=8,
        context_tokens=value["engine"]["context_tokens"],
        response_tokens=81920,
        tokens_per_turn=4096,
        max_turns=2,
    )
    plan = {
        "schema": SCHEMA,
        "run_name": value["name"],
        "output_root": value["output_root"],
        "model": bound,
        "arguments": arguments.__dict__,
        "native_overrides": skyrl.overrides(arguments),
        "engine": value["engine"],
        "deadlines": value["deadlines"],
        "execution": {"image": IMAGE, **value["execution"]},
        "qualification": {
            "profile": "qwen38_skyrl_topology_probe_dev_v1",
            "submission_gate": value["submission_gate"],
        },
        "runtime_sha256": digest(_runtime()),
        "scientific_work": {
            "task_rows": 0,
            "rollout_episodes": 0,
            "verifier_calls": 0,
            "optimizer_steps": 0,
            "checkpoints": 0,
        },
    }
    request(plan)
    return plan


def _validate(plan: dict) -> skyrl.SkyRLConfig:
    arguments = skyrl.SkyRLConfig(**plan["arguments"])
    overrides = skyrl.overrides(arguments)
    execution = plan["execution"]
    if (
        plan.get("schema") != SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("run_name") != arguments.name
        or plan.get("output_root") != arguments.output_root
        or plan.get("native_overrides") != overrides
        or plan.get("engine")
        != {"num_engines": 2, "tensor_parallel_size": 4, "context_tokens": 98304}
        or plan.get("deadlines")
        != {"setup_seconds": 1200, "cleanup_seconds": 300, "total_seconds": 1500}
        or execution.get("image") != IMAGE
        or execution.get("cluster_target") != "dev"
        or execution.get("jobs_api_base_url") != API_URLS["dev"]
        or (execution.get("workers"), execution.get("gpus_per_worker")) != (1, 8)
        or plan.get("qualification")
        != {
            "profile": "qwen38_skyrl_topology_probe_dev_v1",
            "submission_gate": {
                "preview_authorized": True,
                "submission_authorized": False,
                "blockers": [
                    "cpu_preflight_not_yet_recorded",
                    "runtime_user_preview_not_yet_validated",
                    "external_release_observer_not_yet_bound",
                ],
            },
        }
        or plan.get("scientific_work")
        != {
            "task_rows": 0,
            "rollout_episodes": 0,
            "verifier_calls": 0,
            "optimizer_steps": 0,
            "checkpoints": 0,
        }
    ):
        raise ValueError("SkyRL topology probe plan changed")
    return arguments


def request(plan: dict) -> dict:
    arguments = _validate(plan)
    execution = plan["execution"]
    files = _runtime()
    files.update(
        {
            "training/__init__.py": "",
            "evals/__init__.py": "",
            "evals/fleet/__init__.py": "",
            "cyber_post_train/__init__.py": "",
        }
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": arguments.name,
            "title": arguments.name + " zero-update topology probe",
            "run_dir": arguments.output_root,
            "image": IMAGE,
            "workers": 1,
            "gpus_per_worker": 8,
            "resources": execution["resources"],
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "CYBER_EXPECTED_RUNTIME_UID": "1000",
                "CYBER_EXPECTED_RUNTIME_GID": "100",
            },
        },
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def preflight(plan: dict) -> dict:
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("probe preflight requires image user 1000:100")
    import torch
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args

    if torch.cuda.is_available():
        raise ValueError("probe preflight is CPU-only")
    arguments = _validate(plan)
    cfg = skyrl.diagnostic_native_config(arguments)
    build_vllm_cli_args(cfg)
    req = request(plan)
    return {
        "schema": "cyber_skyrl_topology_probe_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": 1000, "gid": 100},
        "plan_sha256": digest(plan),
        "request_sha256": digest(req),
        "task_rows_read": 0,
        "rollout_episodes": 0,
        "optimizer_steps": 0,
    }


def validate_preview(plan: dict, req: dict, preview: dict) -> dict:
    from .skyrl_training import validate_gpu_runtime_preview

    if request(plan) != req:
        raise JobsError("topology probe request differs from its immutable plan")
    return validate_gpu_runtime_preview(req, preview)


class _Deadline:
    def __init__(self, seconds: int, phase: str):
        self.seconds, self.phase = seconds, phase

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("topology-probe deadline requires the main thread")
        if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
            raise RuntimeError("topology-probe deadline cannot replace an existing timer")
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def _expired(self, *_):
        raise TimeoutError(f"{self.phase} exceeded its plan-bound deadline")

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)


def _stop_setup(setup, ray) -> None:
    if setup is None:
        return
    errors = []
    try:
        setup.router.shutdown()
    except Exception as exc:
        errors.append(exc)
    actors, placement_groups = [], []
    for group in reversed(tuple(setup.server_groups)):
        try:
            actors.extend(tuple(group.get_actors()))
        except Exception as exc:
            errors.append(exc)
        for name in ("_internal_pg", "_external_pg"):
            value = getattr(group, name, None)
            if value is not None:
                placement_groups.append(getattr(value, "pg", value))
    refs = []
    for actor in actors:
        with suppress(Exception):
            refs.append(actor.shutdown.remote())
    if refs:
        with suppress(Exception):
            ray.get(refs, timeout=30)
    for actor in actors:
        with suppress(Exception):
            ray.kill(actor, no_restart=True)
    if placement_groups:
        from ray.util.placement_group import remove_placement_group

        for group in placement_groups:
            with suppress(Exception):
                remove_placement_group(group)
    if errors:
        raise ExceptionGroup("one or more topology-probe resources failed to stop", errors)


def run(plan: dict) -> dict:
    """Start both TP4 engines and release them; no scientific path is imported."""
    arguments = _validate(plan)
    if os.environ.get("RUN_DIR") != plan["output_root"]:
        raise ValueError("Jobs API output binding mismatch")
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("probe GPU runtime requires image user 1000:100")
    root = Path(plan["output_root"])
    forbidden = ("episodes", "checkpoints", "exports")
    if any((root / name).exists() for name in forbidden):
        raise FileExistsError("probe output contains a scientific artifact")

    import ray
    from skyrl.backends.skyrl_train.inference_servers.setup import create_inference_servers
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args

    cfg = skyrl.diagnostic_native_config(arguments)
    setup = None
    started = time.time()
    setup_error = None
    try:
        with _Deadline(plan["deadlines"]["setup_seconds"], "engine setup"):
            ray.init(address="auto", log_to_driver=False)
            nodes = [
                row
                for row in ray.nodes()
                if row.get("Alive") and row.get("Resources", {}).get("GPU")
            ]
            if len(nodes) != 1 or quantity(nodes[0]["Resources"]["GPU"]) != 8:
                raise ValueError("probe requires exactly one live eight-GPU Ray node")
            setup = create_inference_servers(
                cfg.generator.inference_engine,
                build_vllm_cli_args(cfg),
                log_path=cfg.trainer.log_path,
            )
            if len(tuple(setup.server_groups)) != 2 or len(tuple(setup.server_urls)) != 2:
                raise ValueError("probe did not start both TP4 engine groups")
    except BaseException as exc:
        setup_error = exc
    try:
        with _Deadline(plan["deadlines"]["cleanup_seconds"], "engine cleanup"):
            _stop_setup(setup, ray)
            ray.shutdown()
    finally:
        if ray.is_initialized():
            ray.shutdown()
    if setup_error is not None:
        raise setup_error
    if any((root / name).exists() for name in forbidden):
        raise ValueError("probe created a scientific artifact")
    return _seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "setup_and_internal_cleanup_passed",
            "plan_sha256": digest(plan),
            "elapsed_seconds": time.time() - started,
            **plan["scientific_work"],
            "engines_started": 2,
            "tensor_parallel_size": 4,
            "ray_shutdown_called": True,
            "external_release_required": True,
        }
    )


def validate_release(plan: dict, receipt: dict, observation: dict) -> dict:
    """Seal a later read-only observation that the exact allocation disappeared."""
    _validate_seal(receipt, RECEIPT_SCHEMA)
    required = {
        "api_run_id",
        "rayjob_uid",
        "workload_uid",
        "pod_uid",
        "terminal_status",
        "rayjob_present",
        "workload_present",
        "pod_present",
        "active_gpus",
    }
    if (
        set(observation) != required
        or any(
            not isinstance(observation[key], str) or not observation[key]
            for key in required
            - {"rayjob_present", "workload_present", "pod_present", "active_gpus"}
        )
        or observation["terminal_status"] not in {"SUCCEEDED", "FAILED", "CANCELLED"}
        or any(
            observation[key] is not False
            for key in ("rayjob_present", "workload_present", "pod_present")
        )
        or observation["active_gpus"] != 0
    ):
        raise ValueError("exact topology-probe release was not proven")
    return _seal(
        {
            "schema": RELEASE_SCHEMA,
            "status": "released",
            "plan_sha256": digest(plan),
            "probe_receipt_sha256": receipt["sha256"],
            "observation": observation,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        request(plan)
        receipt = run(plan)
        path = Path(plan["output_root"]) / "TOPOLOGY_PROBE.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
        print(json.dumps({"status": receipt["status"], "sha256": receipt["sha256"]}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
