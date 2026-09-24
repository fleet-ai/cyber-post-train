"""Minimal zero-update Miles96 qualifier for the exact task7317 binding."""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import hashlib
import inspect
import json
import math
import os
import shlex
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from training import miles96_mechanics_canary as mechanics

SCHEMA = "cyber_qwen38_miles96_phase1a_plan_v1"
PRIVATE_SCHEMA = "cyber_qwen38_miles96_phase1a_private_sample_v1"
PUBLIC_SCHEMA = "cyber_qwen38_miles96_phase1a_terminal_v1"
PREFLIGHT_SCHEMA = "cyber_qwen38_miles96_phase1a_runtime_preflight_v1"
MANIFEST_PATH = "configs/qualification/qwen38-miles96-phase1a-7317-v1.json"
MANIFEST_SHA256 = "sha256:0800ba0cb20acf9c1cac85fcae4d4cb7311fd4c23aac84e3e9df7995862faab2"
PRIVATE_DIR = ".private-phase1a"
PUBLIC_FILE = "PHASE1A_TERMINAL.json"
PREFLIGHT_FILE = "PHASE1A_RUNTIME_PREFLIGHT.json"
SAMPLES = 8
NORMAL_DONE_REASONS = {"answered", "submitted", "boundary_stop", "max_turns"}
FTI_V1_SHA256 = "0524f19dcc886b20d17b39c21bd6417f537423eec6ef2359487fc3911dad441c"
MILES_EVAL_SHA256 = "7c13e0e1ab49cc1cb7224c3f9f91245d4c0bd46c0b7c1785cba8b633ca587a26"
EXPECTED_TASK_KEY = "cysec1-2-fakelook-gen_blackbox-7317189e8fefb9033c5be097__blackbox_ctf_v1"
EXPECTED_TASK_VERSION = "2fdc9511-f5fe-4386-a4a5-a5a16a30f359"
EXPECTED_TASK_ID = "fdd4e7b3-6671-4187-ae04-f26c38272c65"
EXPECTED_VERIFIER_VERSION = "928e5c13-b15a-412d-a037-fbed7389572a"
EXPECTED_COMPONENT = "sha256:b4ff90e50a914ae470b531340bd6f8ad939a60eb5f5d37405a2f68afeda2570e"
EXPECTED_GROUP = "sha256:3a65efd8bbf117ce66d40256df760eb89535832242757ae4e20857a24b480819"
HF_MODEL_ROOT = "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
HF_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HF_MODEL_BINDING = "sha256:dcfdcd6ecb6661741cd3a4b24dc5af7259642c8a6824773e0de70d55d7501179"
CYBER_CONTRACT = {
    "evidence_schema": "1.0.0",
    "submission_protocol": "2.0.0",
    "verifier_contract": "3.0.0",
}

_SLOT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "phase1a_slot", default=None
)
_CLAIMED: set[int] = set()
_CLAIM_LOCK = threading.Lock()
_SESSION_CLASS: type | None = None


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sealed(path: Path, expected: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"authority source is absent or unsafe: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("sha256") != expected:
        raise ValueError(f"authority source seal changed: {path.name}")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value["sha256"] != "sha256:" + digest(body):
        raise ValueError(f"authority source body changed: {path.name}")
    return value


def _manifest_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _task_authority_receipt(manifest: dict[str, Any]) -> str:
    value = {
        "task_key": manifest["task"]["key"],
        "task_version_id": manifest["task"]["version_id"],
        "verifier_version_id": manifest["verifier"]["version_id"],
        "task_set_sha256": manifest["train_authority"]["task_set_sha256"],
        "tool_catalog_sha256": manifest["tools"]["raw_catalog_sha256"],
    }
    return "sha256:" + digest(value)


def _live_binding(manifest: dict[str, Any]) -> dict[str, Any]:
    task = {
        key: manifest["task"][key]
        for key in (
            "key",
            "version_id",
            "prompt_sha256",
            "env_variables_sha256",
            "output_json_schema_sha256",
        )
    }
    task["cyber_contract"] = CYBER_CONTRACT
    return {
        "task": task,
        "environment": manifest["environment"],
        "verifier": manifest["verifier"],
    }


def load_manifest(root: Path | None = None) -> dict[str, Any]:
    """Load the fixed task and prove its current train-only lineage from source bodies."""
    root = _manifest_root() if root is None else root
    manifest = _sealed(root / MANIFEST_PATH, MANIFEST_SHA256)
    if manifest.get("schema") != "cyber_qwen38_miles96_phase1a_manifest_v1":
        raise ValueError("unsupported phase1-A manifest")
    task, verifier, tools, authority, contract = (
        manifest["task"],
        manifest["verifier"],
        manifest["tools"],
        manifest["train_authority"],
        manifest["contract"],
    )
    if (
        task.get("key") != EXPECTED_TASK_KEY
        or task.get("version_id") != EXPECTED_TASK_VERSION
        or task.get("id") != EXPECTED_TASK_ID
        or verifier.get("version_id") != EXPECTED_VERIFIER_VERSION
        or authority.get("role") != "train"
        or authority.get("group_id") != EXPECTED_GROUP
        or authority.get("component_id") != EXPECTED_COMPONENT
        or tools.get("transform_source_sha256") != "sha256:" + FTI_V1_SHA256
        or tools.get("ordered_names") != ["bash", "submit_report"]
    ):
        raise ValueError("task7317 immutable binding changed")
    expected_contract = {
        "mode": "eval",
        "nodes": 1,
        "gpus_per_node": 8,
        "tensor_parallel": 4,
        "context_parallel": 2,
        "context_tokens": 98304,
        "samples": 8,
        "max_concurrent_envs": 2,
        "terminal_slots_required": 8,
        "outer_episode_replacements": 0,
        "unique_verifier_execution_ids": True,
        "gradeable_completion": "normal_only",
        "all_instances_released": True,
        "minimum_completed_gradeable_episodes": 2,
        "minimum_distinct_finite_rewards": 2,
        "optimizer_steps": 0,
        "checkpoint_files": 0,
        "sampling": {"temperature": 1, "top_p": 1, "top_k": -1},
        "priority_class": "c1",
        "queue_priority_class": "q1",
    }
    if contract != expected_contract:
        raise ValueError("phase1-A zero-update contract changed")
    catalog_path = root / tools["path"]
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    if (
        catalog_path.is_symlink()
        or "sha256:" + hashlib.sha256(catalog_bytes).hexdigest() != tools["file_sha256"]
        or "sha256:" + digest(catalog) != tools["raw_catalog_sha256"]
        or [item.get("name") for item in catalog] != tools["ordered_names"]
    ):
        raise ValueError("reviewed task tool catalog changed")
    if (
        _task_authority_receipt(manifest) != authority["task_authority_receipt_sha256"]
        or "sha256:" + digest(_live_binding(manifest)) != authority["live_binding_sha256"]
        or task["cyber_contract_sha256"] != "sha256:" + digest(CYBER_CONTRACT)
    ):
        raise ValueError("task7317 task/live authority digest changed")
    _validate_train_sources(root, manifest)
    return manifest


def validate_live_task_response(response: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the complete frozen binding from one immediate Fleet task GET."""
    from evals.fleet import opencode_self_hosted as fleet

    manifest = load_manifest()
    task, env = manifest["task"], manifest["environment"]
    if (
        response.get("id") != task["id"]
        or response.get("environment_version_id") != env["version_id"]
        or response.get("task_lifecycle_status") != "production"
    ):
        raise ValueError("live task identity/lifecycle differs from task7317")
    selected = {
        "task_key": task["key"],
        "task_version_id": task["version_id"],
        "env_key": env["id"],
        "env_version": env["version"],
        "environment_version_id": env["version_id"],
        "data_key": env["data_id"],
        "data_version": env["data_version"],
    }
    live_task, live_env, live_verifier = fleet.bind_task(response, selected)
    actual = {"task": live_task, "environment": live_env, "verifier": live_verifier}
    if actual != _live_binding(manifest):
        raise ValueError("live task/env/verifier binding differs from task7317")
    return {
        "live_binding_sha256": "sha256:" + digest(actual),
        "tool_catalog_sha256": manifest["tools"]["raw_catalog_sha256"],
    }


def _validate_train_sources(root: Path, manifest: dict[str, Any]) -> None:
    task = manifest["task"]
    env = manifest["environment"]
    authority = manifest["train_authority"]
    split = _sealed(root / authority["split_path"], authority["split_sha256"])
    rows = [
        row
        for row in split.get("tasks", [])
        if row.get("task_key") == task["key"] or row.get("task_version_id") == task["version_id"]
    ]
    expected_split = {
        "group_id": authority["group_id"],
        "shared_atom_component_id": authority["component_id"],
        "split": "train",
        "task_key": task["key"],
        "task_version_id": task["version_id"],
    }
    component_rows = [
        row
        for row in split.get("tasks", [])
        if row.get("shared_atom_component_id") == authority["component_id"]
    ]
    if (
        rows != [expected_split]
        or not component_rows
        or any(row.get("split") != "train" for row in component_rows)
    ):
        raise ValueError("task7317 is not uniquely current-train in the split authority")
    census = _sealed(root / authority["census_path"], authority["census_sha256"])
    components = [
        value
        for value in census.get("components", [])
        if value.get("component_id") == authority["component_id"]
    ]
    expected_version = {
        "source": "receipt_proven",
        "task_key": task["key"],
        "task_version_id": task["version_id"],
    }
    if (
        len(components) != 1
        or components[0].get("inherited_role") != "train"
        or components[0].get("task_versions") != [expected_version]
    ):
        raise ValueError("task7317 transitive train component changed")
    task_set = _sealed(root / authority["task_set_path"], authority["task_set_sha256"])
    selected = [row for row in task_set.get("tasks", []) if row.get("task_key") == task["key"]]
    if len(selected) != 1 or selected[0] != {
        "data_key": env["data_id"],
        "data_version": env["data_version"],
        "env_key": env["id"],
        "env_version": env["version"],
        "environment_version_id": env["version_id"],
        "lineage": {"application": "fakelook", "task_family": authority["family"]},
        "task_key": task["key"],
        "task_version_id": task["version_id"],
    }:
        raise ValueError("task7317 production task-set binding changed")
    inventory = _sealed(root / authority["inventory_path"], authority["inventory_sha256"])
    entries = [
        row
        for row in inventory.get("tasks", [])
        if row.get("task_key") == task["key"] or row.get("task_version_id") == task["version_id"]
    ]
    if len(entries) != 1 or entries[0].get("task_id") != task["id"]:
        raise ValueError("task7317 current inventory identity changed")


def runtime_source_files() -> dict[str, str]:
    root = _manifest_root()
    manifest = load_manifest(root)
    paths = {
        "cyber_post_train/__init__.py",
        "cyber_post_train/jobs.py",
        "training/__init__.py",
        "training/miles96_mechanics_canary.py",
        "training/miles96_phase1a_qualifier.py",
        "training/models.py",
        MANIFEST_PATH,
        manifest["tools"]["path"],
        manifest["train_authority"]["split_path"],
        manifest["train_authority"]["census_path"],
        manifest["train_authority"]["task_set_path"],
        manifest["train_authority"]["inventory_path"],
        "configs/models/qwen38-27b-1d4bf0f2.lock.json",
        "configs/models/qwen38-27b-1d4bf0f2.weights.json",
    }
    return {path: (root / path).read_text() for path in sorted(paths)}


def _source_manifest() -> dict[str, str]:
    return {
        path: "sha256:" + hashlib.sha256(content.encode()).hexdigest()
        for path, content in runtime_source_files().items()
    }


def _model_manifest() -> dict[str, Any]:
    from training.models import bound_model

    root = _manifest_root()
    lock = json.loads((root / "configs/models/qwen38-27b-1d4bf0f2.lock.json").read_text())
    weights = json.loads((root / "configs/models/qwen38-27b-1d4bf0f2.weights.json").read_text())
    value = bound_model(lock, weights, HF_MODEL_ROOT)
    if "sha256:" + digest(value) != HF_MODEL_BINDING:
        raise ValueError("reviewed model inventory binding changed")
    return value


def _sample_slots(name: str) -> list[str]:
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f"miles96-phase1a:{name}:{i}")) for i in range(8)]


def _plan(manifest: dict[str, Any]) -> dict[str, Any]:
    task, env, verifier, tools, authority = (
        manifest["task"],
        manifest["environment"],
        manifest["verifier"],
        manifest["tools"],
        manifest["train_authority"],
    )
    name = manifest["identity"]["name"]
    return {
        "schema": SCHEMA,
        "manifest_sha256": manifest["sha256"],
        "identity": manifest["identity"],
        "execution": {
            "cluster_target": "prod",
            "jobs_api_base_url": mechanics.PROD_JOBS_API,
            "kubernetes_context": mechanics.PROD_CONTEXT,
            "namespace": mechanics.NAMESPACE,
        },
        "trainer": {
            "image": mechanics.IMAGE,
            "recipe": mechanics.RECIPE,
            "fti_version": mechanics.FTI_VERSION,
            "theseus_commit": mechanics.THESEUS_COMMIT,
            "miles_commit": mechanics.MILES_COMMIT,
            "tensor_parallel": 4,
            "context_parallel": 2,
            "context_tokens": mechanics.CONTEXT_TOKENS,
        },
        "model": {
            "root": HF_MODEL_ROOT,
            "revision": HF_MODEL_REVISION,
            "binding_sha256": HF_MODEL_BINDING,
        },
        "task": {
            "id": task["id"],
            "key": task["key"],
            "version_id": task["version_id"],
            "environment": env,
            "verifier": verifier,
            "tool_contract": tools,
        },
        "train_authority": {
            key: authority[key]
            for key in (
                "role",
                "family",
                "group_id",
                "component_id",
                "split_sha256",
                "census_sha256",
                "task_set_sha256",
                "inventory_sha256",
                "task_authority_receipt_sha256",
                "live_binding_sha256",
            )
        },
        "episode": {
            "max_turns": 32,
            "max_tokens_per_turn": 8192,
            "max_concurrent_envs": 2,
            "request_timeout_s": 120,
            "ready_timeout_s": 600,
            "episode_timeout_s": 2400,
            "tool_timeout_s": 330,
            "grade_timeout_s": 900,
            "ttl_seconds": 7200,
            "tool_output_max_chars": 50000,
            "vision": False,
            "pass_conversation_to_verifier": False,
        },
        "qualification": {
            **manifest["contract"],
            "seed": 20260924,
            "sample_slots": _sample_slots(name),
        },
        "runtime_sources": _source_manifest(),
    }


def build_plan() -> dict[str, Any]:
    return _plan(load_manifest())


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(plan))
    if value != build_plan():
        raise ValueError("phase1-A plan differs from the sole reviewed task7317 plan")
    return value


def task_rows(plan: dict[str, Any]) -> bytes:
    plan = validate_plan(plan)
    row = {
        "messages": [{"role": "user", "content": "Run the versioned Fleet task."}],
        "label": plan["task"]["key"],
        "metadata": {
            "task_key": plan["task"]["key"],
            "task_version_id": plan["task"]["version_id"],
            "fleet": plan["episode"],
        },
    }
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _native_contract(plan: dict[str, Any]) -> list[str]:
    run_dir = plan["identity"]["run_dir"]
    q = plan["qualification"]
    runtime_env = json.dumps(
        {
            "PYTHONPATH": run_dir + "/.runtime",
            "CYBER_RUNTIME_DIR": run_dir + "/.runtime",
            "CYBER_PLAN_SHA256": digest(plan),
        },
        separators=(",", ":"),
    )
    extra = " ".join(
        (
            "--custom-generate-function-path training.miles96_phase1a_qualifier.generate",
            f"--seed {q['seed']}",
            f"--rollout-seed {q['seed']}",
            f"--hf-checkpoint {plan['model']['root']}",
            f"--temperature {q['sampling']['temperature']}",
            f"--top-p {q['sampling']['top_p']}",
            f"--top-k {q['sampling']['top_k']}",
        )
    )
    return [
        "-m",
        "fti.trainers.miles.run_fleet",
        "--model-name",
        mechanics.RECIPE,
        "--platform",
        "v1",
        "--mode",
        "eval",
        "--num-nodes",
        "1",
        "--num-gpus-per-node",
        "8",
        "--rollout-batch-size",
        "1",
        "--n-samples-per-prompt",
        "8",
        "--eval-dataset-name",
        "phase1a-7317",
        "--run-id",
        plan["identity"]["name"],
        "--dataset-dir",
        run_dir + "/data",
        "--model-dir",
        plan["model"]["root"],
        "--data-dir",
        run_dir,
        "--output-dir",
        run_dir,
        "--checkpoint-dir",
        run_dir + "/model-output/checkpoints",
        "--extra-env-vars",
        runtime_env,
        "--extra-args",
        extra,
    ]


def native_arguments(plan: dict[str, Any]) -> list[str]:
    plan = validate_plan(plan)
    arguments = _native_contract(plan)
    if arguments != _native_contract(plan):
        raise ValueError("native phase1-A argv is not closed-world")
    extra = shlex.split(arguments[arguments.index("--extra-args") + 1])
    forbidden = {"--save", "--save-interval", "--optimizer-steps", "--post-save-hook"}
    if any(flag in forbidden for flag in arguments + extra):
        raise ValueError("zero-update phase1-A argv contains a training/save flag")
    return [sys.executable, *arguments]


def _runtime_files(plan: dict[str, Any]) -> dict[str, str]:
    return {
        **runtime_source_files(),
        "plan.json": json.dumps(plan, sort_keys=True, separators=(",", ":")),
        "model.json": json.dumps(_model_manifest(), sort_keys=True, separators=(",", ":")),
    }


def job_request(plan: dict[str, Any]) -> dict[str, Any]:
    from cyber_post_train.jobs import bundled_request

    plan = validate_plan(plan)
    name, run_dir = plan["identity"]["name"], plan["identity"]["run_dir"]
    request = {
        "name": name,
        "title": f"Qwen3.8 Miles96 phase1-A task7317: {name}",
        "run_dir": run_dir,
        "image": mechanics.IMAGE,
        "command": "placeholder",
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "topology_mode": "preferred",
        "requeueIfPreempted": False,
        "failureAlerts": False,
        "privileged": True,
        "resources": mechanics.TRAIN_RESOURCES,
        "secrets": ["fleet-api"],
        "image_pull_secrets": ["ecr-pull"],
        "env": {
            "MODEL_DIR": plan["model"]["root"],
            "FLEET_MODEL_DIR": run_dir + "/model-output",
            "PYTHONPATH": run_dir + "/.runtime",
            "MILES_SCRIPT_EXTERNAL_RAY": "1",
            "CYBER_PLAN_SHA256": digest(plan),
            "CYBER_TASK_BINDING_SHA256": "sha256:" + digest(plan["task"]),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        },
    }
    return bundled_request(
        request,
        _runtime_files(plan),
        "training.miles96_phase1a_qualifier",
        ["--run", "--plan", "plan.json", "--sha256", digest(plan)],
    )


def _runtime_plan() -> dict[str, Any]:
    root = Path(os.environ.get("CYBER_RUNTIME_DIR", ""))
    if not root.is_dir():
        raise ValueError("phase1-A runtime bundle is absent")
    return validate_plan(json.loads((root / "plan.json").read_text()))


def _runtime_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    mechanics._runtime_recipe_binding()
    from fti.fleet import v1 as fleet_v1
    from fti.miles.v1 import client_recording, common
    from miles.rollout.inference_rollout import inference_rollout_eval

    if mechanics.file_sha256(Path(fleet_v1.__file__)) != FTI_V1_SHA256:
        raise ValueError("FTI tool transformation source changed")
    if mechanics.file_sha256(Path(inference_rollout_eval.__file__)) != MILES_EVAL_SHA256:
        raise ValueError("Miles eval sampling source changed")
    source = inspect.getsource(inference_rollout_eval.eval_rollout_single_dataset)
    for fragment in (
        "sample_index = 0",
        "sample.index = sample_index",
        "sample_index += 1",
        "data.sort(key=lambda sample: sample.index)",
    ):
        if fragment not in source:
            raise ValueError("Miles sample-slot control flow changed")
    if "async with env_gate(cfg.max_concurrent_envs):" not in inspect.getsource(
        common.run_episode
    ) or "await asyncio.shield(asyncio.to_thread(session.close))" not in inspect.getsource(
        common.run_episode
    ):
        raise ValueError("FTI concurrency/release control flow changed")
    cfg = common.Config(**plan["episode"])
    generate_parameters = tuple(inspect.signature(client_recording.generate).parameters)
    if cfg.max_concurrent_envs != 2 or generate_parameters != ("input",):
        raise ValueError("phase1-A runtime interface changed")
    return _preflight_receipt(plan)


def _preflight_receipt(plan: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "image": mechanics.IMAGE,
        "fti_tool_transform_sha256": "sha256:" + FTI_V1_SHA256,
        "miles_eval_sha256": "sha256:" + MILES_EVAL_SHA256,
        "sampling": plan["qualification"]["sampling"],
        "sample_count": 8,
        "max_concurrent_envs": 2,
        "outer_episode_replacements": 0,
        "shielded_close": True,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def _session_class() -> type:
    global _SESSION_CLASS
    if _SESSION_CLASS is not None:
        return _SESSION_CLASS
    from fti.fleet import GradeResult
    from fti.fleet.v1 import PlatformError, openai_tools
    from fti.miles.v1.common import TaskSession, numeric_reward

    class Phase1ASession(TaskSession):
        verifier_execution_id: str | None = None
        release_confirmed = False

        def open(self) -> None:
            captured = None
            restored: list[tuple[Any, object]] = []
            missing = object()
            original_create = self.client.create_reward_instance
            client_override = self.client.__dict__.get("create_reward_instance", missing)

            def create(*args: Any, **kwargs: Any):
                instance = original_create(*args, **kwargs)
                state = _SLOT.get()
                if state is not None:
                    state["instance_id"] = instance.instance_id
                    state["cleanup_confirmed"] = False
                original_list = instance.list_tools
                instance_override = instance.__dict__.get("list_tools", missing)

                def list_tools():
                    nonlocal captured
                    if captured is not None:
                        raise ValueError("live task tools were read more than once")
                    raw = original_list()
                    captured = json.loads(json.dumps(raw, sort_keys=True, allow_nan=False))
                    return raw

                instance.list_tools = list_tools
                restored.append((instance, instance_override))
                return instance

            self.client.create_reward_instance = create
            try:
                super().open()
            finally:
                if client_override is missing:
                    self.client.__dict__.pop("create_reward_instance", None)
                else:
                    self.client.__dict__["create_reward_instance"] = client_override
                for instance, override in restored:
                    if override is missing:
                        instance.__dict__.pop("list_tools", None)
                    else:
                        instance.__dict__["list_tools"] = override
            try:
                plan = _runtime_plan()
                task, tools = plan["task"], plan["task"]["tool_contract"]
                visible = openai_tools(captured) if isinstance(captured, list) else None
                if (
                    self.task_key != task["key"]
                    or self.task_version_id != task["version_id"]
                    or self.verifier_version_id != task["verifier"]["version_id"]
                    or "sha256:" + digest(captured) != tools["raw_catalog_sha256"]
                    or visible != self.tools
                    or "sha256:" + digest(self.tools) != tools["openai_catalog_sha256"]
                ):
                    raise ValueError("live task/version/verifier/tools differ from task7317")
            except BaseException:
                self.close()
                raise

        def grade(self, answer, reset_ack=None, close_final_step=False):
            del reset_ack, close_final_step
            if self.closed.is_set() or self.instance is None or self.verifier_version_id is None:
                raise PlatformError("episode authority is incomplete")
            result = self.client.execute_verifier(
                self.verifier_version_id,
                self.instance.instance_id,
                final_answer=answer,
                conversation=self.conversation if self.cfg.pass_conversation_to_verifier else None,
                timeout_s=self.cfg.grade_timeout_s,
            )
            if result.get("success") is not True:
                raise PlatformError("registered verifier execution failed")
            execution = (
                result.get("verifier_execution_id") or result.get("job_id") or result.get("id")
            )
            self.verifier_execution_id = str(uuid.UUID(str(execution)))
            verdict = result.get("result")
            value = verdict.get("result") if isinstance(verdict, dict) else verdict
            return GradeResult(reward=numeric_reward(value))

        def close(self) -> None:
            instance = self.instance
            state = _SLOT.get()
            if state is not None:
                state["instance_id"] = None if instance is None else instance.instance_id
                state["verifier_execution_id"] = self.verifier_execution_id
            try:
                super().close()
            finally:
                if state is not None:
                    state["instance_id"] = None if instance is None else instance.instance_id
                    state["verifier_execution_id"] = self.verifier_execution_id
            if instance is None or not self.deleted or self.cleanup_error is not None:
                return
            deadline = time.monotonic() + self.cfg.request_timeout_s
            while time.monotonic() < deadline:
                try:
                    self.client.get_instance(
                        instance.instance_id,
                        timeout_s=min(10, max(1, deadline - time.monotonic())),
                    )
                except PlatformError as error:
                    if str(error) == "GET request returned HTTP 404":
                        self.release_confirmed = True
                        if state is not None:
                            state["cleanup_confirmed"] = True
                        return
                    self.cleanup_error = "instance absence check failed"
                    return
                time.sleep(1)
            self.cleanup_error = "instance absence check timed out"

    _SESSION_CLASS = Phase1ASession
    return Phase1ASession


def _episode_metadata(session: Any, result: Any, stats: Any) -> dict[str, Any]:
    state = _SLOT.get()
    if state is None:
        raise ValueError("episode has no fixed sample-slot binding")
    reward = None if result.grade is None else float(result.grade.reward)
    return {
        "slot_id": state["slot_id"],
        "sample_index": state["sample_index"],
        "instance_id": None if session.instance is None else session.instance.instance_id,
        "verifier_execution_id": session.verifier_execution_id,
        "reward": reward,
        "graded": result.grade is not None and reward is not None and math.isfinite(reward),
        "authority_verified": True,
        "cleanup_confirmed": session.release_confirmed,
        "done_reason": result.done_reason,
        "tool_calls": stats.tool_calls,
    }


def _private_root(plan: dict[str, Any]) -> Path:
    path = Path(plan["identity"]["run_dir"]) / PRIVATE_DIR
    if not path.exists():
        path.mkdir(mode=0o700)
    if path.is_symlink() or not path.is_dir() or stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise ValueError("private phase1-A evidence directory is unsafe")
    return path


def _write_once(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    temporary = f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd
        )
        os.fchmod(descriptor, 0o600)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("evidence write made no progress")
            written += count
        os.fsync(descriptor)
        os.link(
            temporary,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(OSError):
            os.unlink(temporary, dir_fd=directory_fd)
        os.close(directory_fd)


def _slot_receipt(plan: dict[str, Any], index: int, output: Any) -> dict[str, Any]:
    state = _SLOT.get()
    if (
        state is None
        or state.get("sample_index") != index
        or state.get("slot_id") != plan["qualification"]["sample_slots"][index]
    ):
        raise ValueError("returned episode lacks its fixed sample-slot state")
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    if len(samples) != 1:
        raise ValueError("one fixed sample slot returned other than one episode")
    sample = samples[0]
    metadata = getattr(sample, "metadata", None) or {}
    fleet = metadata.get("fleet_v1") if isinstance(metadata, dict) else None
    fleet = fleet if isinstance(fleet, dict) else {}
    status = getattr(getattr(sample, "status", None), "value", None)
    reward = getattr(sample, "reward", None)
    finite = (
        not isinstance(reward, bool)
        and isinstance(reward, (int, float))
        and math.isfinite(float(reward))
    )
    normal = (
        status == "completed"
        and finite
        and fleet.get("graded") is True
        and fleet.get("authority_verified") is True
        and fleet.get("cleanup_confirmed") is True
        and fleet.get("done_reason") in NORMAL_DONE_REASONS
        and type(fleet.get("tool_calls")) is int
        and fleet["tool_calls"] >= 1
    )
    metadata_instance = fleet.get("instance_id")
    state_instance = state.get("instance_id")
    if metadata_instance is not None and state_instance is not None:
        if str(metadata_instance) != str(state_instance):
            raise ValueError("returned episode instance disagrees with runtime state")
    elif metadata_instance is not None:
        raise ValueError("returned episode claims an instance absent from runtime state")
    instance_id = state_instance
    verifier_id = fleet.get("verifier_execution_id")
    if verifier_id is not None and state.get("verifier_execution_id") not in {None, verifier_id}:
        raise ValueError("returned verifier execution disagrees with runtime state")
    instance_created = instance_id is not None
    if instance_created and state.get("cleanup_confirmed") is not True:
        raise ValueError("sample instance was not proven released")
    instance_id = str(uuid.UUID(str(instance_id))) if instance_created else None
    verifier_id = str(uuid.UUID(str(verifier_id))) if normal else None
    body = {
        "schema": PRIVATE_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "sample_index": index,
        "slot_id": plan["qualification"]["sample_slots"][index],
        "terminally_accounted": True,
        "terminal_reason": "normally_completed" if normal else "returned_ungradeable",
        "normally_completed_gradeable": normal,
        "instance_created": instance_created,
        "instance_id": instance_id,
        "verifier_execution_id": verifier_id,
        "reward": float(reward) if normal else None,
        "cleanup_confirmed": True if instance_created else None,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def _exception_receipt(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    instance = state.get("instance_id")
    if instance is not None:
        instance = str(uuid.UUID(str(instance)))
    body = {
        "schema": PRIVATE_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "sample_index": state["sample_index"],
        "slot_id": state["slot_id"],
        "terminally_accounted": True,
        "terminal_reason": "runtime_exception",
        "normally_completed_gradeable": False,
        "instance_created": instance is not None,
        "instance_id": instance,
        "verifier_execution_id": None,
        "reward": None,
        "cleanup_confirmed": state.get("cleanup_confirmed") if instance is not None else None,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


async def generate(input: Any) -> Any:
    """Run one Miles-provided index exactly once and seal its private terminal receipt."""
    from fti.miles.v1 import client_recording

    plan = _runtime_plan()
    sampling = getattr(input, "sampling_params", None)
    if not isinstance(sampling, dict) or any(
        sampling.get(key) != value for key, value in plan["qualification"]["sampling"].items()
    ):
        raise ValueError("effective sampling differs from the reviewed phase1-A plan")
    index = getattr(getattr(input, "sample", None), "index", None)
    if type(index) is not int or not 0 <= index < SAMPLES:
        raise ValueError("Miles sample index is outside the fixed eight slots")
    with _CLAIM_LOCK:
        if index in _CLAIMED:
            raise ValueError("a phase1-A sample slot was replayed")
        _CLAIMED.add(index)
    state = {"sample_index": index, "slot_id": plan["qualification"]["sample_slots"][index]}
    token = _SLOT.set(state)
    client_recording.TaskSession = _session_class()
    client_recording.episode_metadata = _episode_metadata
    try:
        try:
            output = await client_recording.generate(input)
        except BaseException:
            _write_once(
                _private_root(plan) / f"sample-{index}.json",
                _exception_receipt(plan, state),
            )
            raise
        receipt = _slot_receipt(plan, index, output)
        _write_once(_private_root(plan) / f"sample-{index}.json", receipt)
        return output
    finally:
        _SLOT.reset(token)


def _generate_add_arguments(parser: Any) -> None:
    from fti.miles.v1.client_recording import add_arguments

    add_arguments(parser)


generate.add_arguments = _generate_add_arguments


def _checkpoint_absent(output: Path) -> bool:
    checkpoints = output / "checkpoints"
    if output.is_symlink() or not output.is_dir() or checkpoints.is_symlink():
        return False
    if not checkpoints.exists():
        return True
    return checkpoints.is_dir() and all(
        path.is_dir() and not path.is_symlink() for path in checkpoints.rglob("*")
    )


def _read_private(plan: dict[str, Any]) -> list[dict[str, Any]]:
    root = _private_root(plan)
    values = []
    for index in range(SAMPLES):
        path = root / f"sample-{index}.json"
        if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError("one private sample receipt is absent or unsafe")
        value = json.loads(path.read_text())
        body = {key: item for key, item in value.items() if key != "sha256"}
        if (
            set(value)
            != {
                "schema",
                "plan_sha256",
                "sample_index",
                "slot_id",
                "terminally_accounted",
                "terminal_reason",
                "normally_completed_gradeable",
                "instance_created",
                "instance_id",
                "verifier_execution_id",
                "reward",
                "cleanup_confirmed",
                "sha256",
            }
            or value.get("sha256") != "sha256:" + digest(body)
            or value.get("schema") != PRIVATE_SCHEMA
            or value.get("plan_sha256") != "sha256:" + digest(plan)
            or value.get("sample_index") != index
            or value.get("slot_id") != plan["qualification"]["sample_slots"][index]
            or value.get("terminally_accounted") is not True
            or value.get("terminal_reason")
            not in {"normally_completed", "returned_ungradeable", "runtime_exception"}
        ):
            raise ValueError("one private sample receipt is invalid")
        if value["instance_created"]:
            uuid.UUID(value["instance_id"])
            if value["cleanup_confirmed"] is not True:
                raise ValueError("one created sample instance was not released")
        elif value["instance_id"] is not None or value["cleanup_confirmed"] is not None:
            raise ValueError("no-instance terminal receipt claims cleanup evidence")
        if value["normally_completed_gradeable"]:
            if value["terminal_reason"] != "normally_completed" or not value["instance_created"]:
                raise ValueError("gradeable sample lacks a normal terminal instance")
            uuid.UUID(value["verifier_execution_id"])
            if isinstance(value["reward"], bool) or not math.isfinite(float(value["reward"])):
                raise ValueError("gradeable reward is not finite")
        elif value["verifier_execution_id"] is not None or value["reward"] is not None:
            raise ValueError("ungradeable sample carries grade evidence")
        values.append(value)
    if len(list(root.glob("sample-*.json"))) != SAMPLES:
        raise ValueError("private sample receipt set contains an extra slot")
    return values


def aggregate(plan: dict[str, Any], preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = validate_plan(plan)
    run_dir = Path(plan["identity"]["run_dir"])
    if preflight is None:
        preflight_path = run_dir / PREFLIGHT_FILE
        if (
            preflight_path.is_symlink()
            or not preflight_path.is_file()
            or stat.S_IMODE(preflight_path.stat().st_mode) != 0o600
        ):
            raise ValueError("runtime preflight receipt is absent or unsafe")
        preflight = json.loads(preflight_path.read_text())
    if preflight != _preflight_receipt(plan):
        raise ValueError("runtime preflight receipt differs from the exact reviewed runtime")
    if not _checkpoint_absent(run_dir / "model-output"):
        raise ValueError("checkpoint absence is not proven")
    receipts = _read_private(plan)
    gradeable = [value for value in receipts if value["normally_completed_gradeable"]]
    verifier_ids = [value["verifier_execution_id"] for value in gradeable]
    instances = [value["instance_id"] for value in receipts if value["instance_created"]]
    rewards = [value["reward"] for value in gradeable]
    if (
        len(receipts) != 8
        or len(instances) != len(set(instances))
        or len(verifier_ids) != len(set(verifier_ids))
        or len(gradeable) < 2
        or len(set(rewards)) < 2
    ):
        raise ValueError("phase1-A reward/release qualification did not pass")
    private_refs = [value["sha256"] for value in receipts]
    body = {
        "schema": PUBLIC_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "manifest_sha256": plan["manifest_sha256"],
        "run_name": plan["identity"]["name"],
        "task_key": plan["task"]["key"],
        "task_version_id": plan["task"]["version_id"],
        "verifier_version_id": plan["task"]["verifier"]["version_id"],
        "terminal_slot_count": 8,
        "gradeable_episode_count": len(gradeable),
        "minimum_gradeable_episode_count": 2,
        "distinct_finite_reward_count": len(set(rewards)),
        "all_instances_released": True,
        "outer_episode_replacements": 0,
        "unique_verifier_execution_ids": True,
        "optimizer_steps": 0,
        "checkpoint_artifacts_absent": True,
        "private_receipts_sha256": "sha256:" + digest(sorted(private_refs)),
        "runtime_preflight_sha256": preflight["sha256"],
    }
    value = {**body, "sha256": "sha256:" + digest(body)}
    _write_once(Path(plan["identity"]["run_dir"]) / PUBLIC_FILE, value)
    return value


def _prepared_model(plan: dict[str, Any]) -> None:
    runtime = Path(os.environ["CYBER_RUNTIME_DIR"])
    model = json.loads((runtime / "model.json").read_text())
    if "sha256:" + digest(model) != plan["model"]["binding_sha256"]:
        raise ValueError("runtime model inventory differs from the reviewed binding")
    root = Path(model["root"])
    if root.is_symlink() or not root.is_dir():
        raise ValueError("reviewed model root is absent or unsafe")
    for item in model["files"]:
        path = root / item["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("one reviewed model file is absent or unsafe")
        if mechanics.file_sha256(path) != str(item["sha256"]).removeprefix("sha256:"):
            raise ValueError("one reviewed model file changed")


def run(plan: dict[str, Any], expected: str) -> dict[str, Any]:
    plan = validate_plan(plan)
    if digest(plan) != expected:
        raise ValueError("phase1-A runtime plan digest changed")
    run_dir = Path(plan["identity"]["run_dir"])
    data, output = run_dir / "data", run_dir / "model-output"
    if (
        os.environ.get("RUN_DIR") != str(run_dir)
        or os.environ.get("MODEL_DIR") != plan["model"]["root"]
        or os.environ.get("FLEET_MODEL_DIR") != str(output)
        or os.environ.get("CYBER_PLAN_SHA256") != expected
        or os.environ.get("CYBER_TASK_BINDING_SHA256") != "sha256:" + digest(plan["task"])
    ):
        raise ValueError("phase1-A runtime environment binding changed")
    if not run_dir.is_dir() or any(
        path.exists() for path in (data, output, run_dir / PREFLIGHT_FILE, run_dir / PUBLIC_FILE)
    ):
        raise FileExistsError("phase1-A output is not create-once")
    run_dir.chmod(0o700)
    _prepared_model(plan)
    preflight = _runtime_preflight(plan)
    _write_once(run_dir / PREFLIGHT_FILE, preflight)
    data.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    _private_root(plan)
    eval_path = data / "eval.jsonl"
    eval_path.write_bytes(task_rows(plan))
    try:
        completed = subprocess.run(native_arguments(plan), check=False, env=os.environ)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, completed.args)
        if not _checkpoint_absent(output):
            raise ValueError("zero-update phase1-A run wrote checkpoint artifacts")
        return aggregate(plan, preflight)
    finally:
        eval_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    plan = validate_plan(json.loads(Path(args.plan).read_text()))
    if digest(plan) != args.sha256:
        raise ValueError("phase1-A plan digest changed")
    value = run(plan, args.sha256) if args.run else aggregate(plan)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
