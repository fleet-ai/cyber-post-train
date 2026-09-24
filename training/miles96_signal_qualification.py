"""Zero-update reward-signal qualification for the maintained Miles96 recipe."""

from __future__ import annotations

import argparse
import base64
import contextvars
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from training import miles96_mechanics_canary as mechanics

SCHEMA = "cyber_qwen38_miles96_signal_qualification_v1"
PRIVATE_GROUP_SCHEMA = "cyber_qwen38_miles96_signal_private_group_v1"
PRIVATE_ATTEMPT_SCHEMA = "cyber_qwen38_miles96_signal_attempt_v1"
PRIVATE_SLOT_SCHEMA = "cyber_qwen38_miles96_signal_slot_v1"
PRIVATE_CLAIM_SCHEMA = "cyber_qwen38_miles96_signal_slot_claim_v1"
REJECTION_SCHEMA = "cyber_qwen38_miles96_signal_rejection_v1"
LAUNCH_FILE = "SIGNAL_LAUNCH.json"
EVIDENCE_FILE = "TASK_SIGNAL_EVIDENCE.json"
REJECTION_FILE = "TASK_SIGNAL_REJECTED.json"
PRIVATE_GROUP_FILE = "SIGNAL_GROUP.json"
NATIVE_TERMINAL_FILE = "SIGNAL_NATIVE_TERMINAL.json"
NATIVE_TERMINAL_SCHEMA = "cyber_qwen38_miles96_signal_native_terminal_v1"
RUNTIME_PREFLIGHT_FILE = "SIGNAL_RUNTIME_PREFLIGHT.json"
RUNTIME_PREFLIGHT_SCHEMA = "cyber_qwen38_miles96_signal_runtime_preflight_v1"
SAMPLES = 8
FTI_V1_SHA256 = "0524f19dcc886b20d17b39c21bd6417f537423eec6ef2359487fc3911dad441c"
MILES_INFERENCE_EVAL_SHA256 = "7c13e0e1ab49cc1cb7224c3f9f91245d4c0bd46c0b7c1785cba8b633ca587a26"
HF_MODEL_ROOT = "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
HF_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HF_MODEL_BINDING_SHA256 = "sha256:dcfdcd6ecb6661741cd3a4b24dc5af7259642c8a6824773e0de70d55d7501179"
_CURRENT_ATTEMPT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "miles96_signal_attempt", default=None
)


def _sample_slots(name: str) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": index,
            "slot_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"fleet-miles96-signal:{name}:{index}")),
        }
        for index in range(SAMPLES)
    ]


def _trainer_binding() -> dict[str, str]:
    return {
        "model": mechanics.MODEL,
        "recipe": mechanics.RECIPE,
        "image": mechanics.IMAGE,
        "theseus_commit": mechanics.THESEUS_COMMIT,
        "image_source_commit": mechanics.IMAGE_SOURCE_COMMIT,
        "miles_commit": mechanics.MILES_COMMIT,
        "fti_version": mechanics.FTI_VERSION,
        "run_fleet_sha256": mechanics.RUN_FLEET_SHA256,
        "fti_common_sha256": mechanics.FTI_COMMON_SHA256,
        "fti_client_recording_sha256": mechanics.FTI_CLIENT_RECORDING_SHA256,
        "fti_v1_sha256": FTI_V1_SHA256,
        "miles_inference_rollout_sha256": mechanics.MILES_INFERENCE_ROLLOUT_SHA256,
        "miles_inference_eval_sha256": MILES_INFERENCE_EVAL_SHA256,
        "miles_http_utils_sha256": mechanics.MILES_HTTP_UTILS_SHA256,
        "miles_megatron_actor_sha256": mechanics.MILES_MEGATRON_ACTOR_SHA256,
        "miles_hf_export_sha256": mechanics.MILES_HF_EXPORT_SHA256,
        "miles_wandb_utils_sha256": mechanics.MILES_WANDB_UTILS_SHA256,
    }


def runtime_source_files() -> dict[str, str]:
    """Return the complete custom runtime and immutable authority snapshot."""
    root = Path(__file__).resolve().parents[1]
    wave_path = "configs/qualification/qwen38-miles-signal-wave-v1.json"
    wave = json.loads((root / wave_path).read_text())
    paths = {
        "training/miles96_mechanics_canary.py",
        "training/miles96_signal_qualification.py",
        "training/miles_signal_wave.py",
        wave_path,
        "configs/models/qwen38-27b-1d4bf0f2.lock.json",
        "configs/models/qwen38-27b-1d4bf0f2.weights.json",
    }
    paths.update(
        authority["path"] for authority in wave["authorities"].values() if "path" in authority
    )
    files = {path: (root / path).read_text() for path in sorted(paths)}
    files["training/__init__.py"] = ""
    return files


def runtime_source_manifest() -> dict[str, str]:
    return {
        path: "sha256:" + hashlib.sha256(content.encode()).hexdigest()
        for path, content in runtime_source_files().items()
    }


def _runtime_source_manifest_sha256(plan: dict[str, Any]) -> str:
    return "sha256:" + mechanics.digest(plan["runtime_sources"])


def build_plan(
    *,
    name: str,
    model_root: str,
    model_binding_sha256: str,
    task_binding: dict[str, str],
    authority_config_sha256: str,
    current_binding_sha256: str,
    production_split_sha256: str,
) -> dict[str, Any]:
    """Bind one exact task to eight fresh episodes and no optimizer work."""
    return validate_plan(
        {
            "schema": SCHEMA,
            "identity": {"name": name, "run_dir": f"/mnt/sfs/jobs/{name}"},
            "trainer": _trainer_binding(),
            "runtime_sources": runtime_source_manifest(),
            "execution": {
                "cluster_target": "prod",
                "jobs_api_base_url": mechanics.PROD_JOBS_API,
                "kubernetes_context": mechanics.PROD_CONTEXT,
                "namespace": mechanics.NAMESPACE,
            },
            "prepared_model": {
                "root": model_root,
                "revision": HF_MODEL_REVISION,
                "binding_sha256": model_binding_sha256,
            },
            "task_binding": task_binding,
            "selection_authority": {
                "authority_config_sha256": authority_config_sha256,
                "current_binding_sha256": current_binding_sha256,
                "production_split_sha256": production_split_sha256,
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
                "samples": SAMPLES,
                "optimizer_steps": 0,
                "minimum_completed_gradeable_episodes": 2,
                "minimum_distinct_finite_rewards": 2,
                "outer_episode_replacements": 0,
                "seed": 20260924,
                "sample_slots": _sample_slots(name),
            },
            "cluster": {
                "nodes": 1,
                "gpus_per_node": 8,
                "priority": "c1",
                "topology_mode": "preferred",
                "resources": mechanics.TRAIN_RESOURCES,
            },
            "acceptance": {
                "eight_fresh_samples": False,
                "finite_rewards": False,
                "reward_variation": False,
                "all_instances_released": False,
                "optimizer_steps": 0,
            },
        }
    )


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(plan))
    if (
        set(value)
        != {
            "schema",
            "identity",
            "trainer",
            "runtime_sources",
            "execution",
            "prepared_model",
            "task_binding",
            "selection_authority",
            "episode",
            "qualification",
            "cluster",
            "acceptance",
        }
        or value.get("schema") != SCHEMA
    ):
        raise ValueError("unsupported Miles96 signal plan")

    identity = value["identity"]
    if set(identity) != {"name", "run_dir"}:
        raise ValueError("signal identity fields changed")
    name = mechanics._name(identity["name"], "signal name")
    mechanics._sfs_path(identity["run_dir"], "signal run_dir", exact_leaf=name)
    if value["trainer"] != _trainer_binding():
        raise ValueError("maintained Miles96 runtime binding drift")
    if value["runtime_sources"] != runtime_source_manifest():
        raise ValueError("custom runtime source manifest drift")
    if value["execution"] != {
        "cluster_target": "prod",
        "jobs_api_base_url": mechanics.PROD_JOBS_API,
        "kubernetes_context": mechanics.PROD_CONTEXT,
        "namespace": mechanics.NAMESPACE,
    }:
        raise ValueError("production execution binding drift")

    prepared = value["prepared_model"]
    if set(prepared) != {"root", "revision", "binding_sha256"}:
        raise ValueError("prepared-model binding fields changed")
    if prepared["root"] != HF_MODEL_ROOT or prepared["revision"] != HF_MODEL_REVISION:
        raise ValueError("signal base-model binding drift")
    if prepared["binding_sha256"] != HF_MODEL_BINDING_SHA256:
        raise ValueError("signal model inventory binding drift")
    if prepared["root"].startswith(identity["run_dir"] + "/"):
        raise ValueError("prepared model must not be a signal output")

    task = value["task_binding"]
    task_fields = {
        "task_key",
        "task_version_id",
        "verifier_version_id",
        "task_set_sha256",
        "tool_catalog_sha256",
        "authority_receipt_sha256",
    }
    if not isinstance(task, dict) or set(task) != task_fields:
        raise ValueError("task binding fields changed")
    if not isinstance(task["task_key"], str) or not task["task_key"].strip():
        raise ValueError("task binding needs an exact task key")
    mechanics._uuid(task["task_version_id"], "task version")
    mechanics._uuid(task["verifier_version_id"], "verifier version")
    for field in task_fields - {"task_key", "task_version_id", "verifier_version_id"}:
        mechanics._sha256(task[field], field)
    authority = {key: task[key] for key in task_fields if key != "authority_receipt_sha256"}
    if task["authority_receipt_sha256"] != "sha256:" + mechanics.digest(authority):
        raise ValueError("task authority receipt digest is not self-consistent")

    selection = value["selection_authority"]
    if set(selection) != {
        "authority_config_sha256",
        "current_binding_sha256",
        "production_split_sha256",
    }:
        raise ValueError("selection authority fields changed")
    for field, item in selection.items():
        mechanics._sha256(item, field)
    from training import miles_signal_wave

    wave = miles_signal_wave.load()
    candidate = next(
        (row for row in wave["candidates"] if row["identity"] == identity),
        None,
    )
    expected_task = None if candidate is None else miles_signal_wave.task_binding(wave, candidate)
    if (
        selection["authority_config_sha256"] != wave["sha256"]
        or selection["production_split_sha256"]
        != wave["authorities"]["production_split"]["self_sha256"]
        or candidate is None
        or selection["current_binding_sha256"] != candidate["live_binding_receipt_sha256"]
        or task
        != {
            **expected_task,
            "authority_receipt_sha256": candidate["authority_receipt_sha256"],
        }
    ):
        raise ValueError("signal candidate differs from the frozen wave authority")

    expected_episode = {
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
    }
    if value["episode"] != expected_episode:
        raise ValueError("bounded V1 episode contract drift")
    if value["qualification"] != {
        "samples": SAMPLES,
        "optimizer_steps": 0,
        "minimum_completed_gradeable_episodes": 2,
        "minimum_distinct_finite_rewards": 2,
        "outer_episode_replacements": 0,
        "seed": 20260924,
        "sample_slots": _sample_slots(name),
    }:
        raise ValueError("zero-update signal contract drift")
    if value["cluster"] != {
        "nodes": 1,
        "gpus_per_node": 8,
        "priority": "c1",
        "topology_mode": "preferred",
        "resources": mechanics.TRAIN_RESOURCES,
    }:
        raise ValueError("one-node c1 cluster contract drift")
    if value["acceptance"] != {
        "eight_fresh_samples": False,
        "finite_rewards": False,
        "reward_variation": False,
        "all_instances_released": False,
        "optimizer_steps": 0,
    }:
        raise ValueError("signal acceptance must start closed")
    return value


def task_rows(plan: dict[str, Any]) -> bytes:
    plan = validate_plan(plan)
    row = {
        "messages": [{"role": "user", "content": "Run the versioned Fleet task."}],
        "label": plan["task_binding"]["task_key"],
        "metadata": {
            "task_key": plan["task_binding"]["task_key"],
            "task_version_id": plan["task_binding"]["task_version_id"],
            "fleet": plan["episode"],
        },
    }
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def native_arguments(plan: dict[str, Any]) -> list[str]:
    plan = validate_plan(plan)
    run_dir = plan["identity"]["run_dir"]
    runtime_env = json.dumps(
        {
            "PYTHONPATH": run_dir + "/.runtime",
            "CYBER_RUNTIME_DIR": run_dir + "/.runtime",
            "CYBER_EVIDENCE_RUN_DIR": run_dir,
            "CYBER_PLAN_SHA256": mechanics.digest(plan),
        },
        separators=(",", ":"),
    )
    extra = " ".join(
        (
            "--custom-generate-function-path "
            "training.miles96_signal_qualification.generate_with_evidence",
            f"--seed {plan['qualification']['seed']}",
            f"--rollout-seed {plan['qualification']['seed']}",
            f"--hf-checkpoint {plan['prepared_model']['root']}",
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
        str(SAMPLES),
        "--eval-dataset-name",
        "fresh-signal",
        "--run-id",
        plan["identity"]["name"],
        "--dataset-dir",
        run_dir + "/data",
        "--model-dir",
        plan["prepared_model"]["root"],
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


def _source_order(source: str, *needles: str) -> None:
    positions = [source.index(needle) for needle in needles]
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise ValueError("maintained signal runtime control flow drift")


def _runtime_signal_binding(plan: dict[str, Any]) -> dict[str, Any]:
    """Prove the exact-image sample, concurrency, and release contracts."""
    mechanics._runtime_recipe_binding()
    from fti.fleet import v1 as fleet_v1
    from fti.miles.v1 import client_recording, common
    from miles.rollout.inference_rollout import inference_rollout_eval

    if mechanics.file_sha256(Path(fleet_v1.__file__)) != FTI_V1_SHA256:
        raise ValueError("maintained Fleet V1 client source drift")
    if mechanics.file_sha256(Path(inference_rollout_eval.__file__)) != MILES_INFERENCE_EVAL_SHA256:
        raise ValueError("maintained Miles eval sample-index source drift")
    if tuple(inspect.signature(client_recording.generate).parameters) != ("input",):
        raise ValueError("maintained V1 generate API drift")

    _source_order(
        inspect.getsource(inference_rollout_eval.eval_rollout_single_dataset),
        "sample_index = 0",
        "for j in range(dataset_cfg.n_samples_per_eval_prompt):",
        "sample.index = sample_index",
        "sample_index += 1",
        "asyncio.create_task(",
        "data.sort(key=lambda sample: sample.index)",
    )
    _source_order(
        inspect.getsource(client_recording.generate),
        "base_sample = deepcopy(input.sample)",
        'cfg = Config(**metadata.get("fleet", {}))',
        "session = TaskSession(metadata, cfg, stats.trajectory)",
        "result = await run_episode(session, recorder, cfg, stats)",
        "episode_metadata(session, result, stats)",
        "recorder.finalize",
    )
    episode_source = inspect.getsource(common.run_episode)
    if (
        "async with env_gate(cfg.max_concurrent_envs):" not in episode_source
        or "await asyncio.shield(asyncio.to_thread(session.close))" not in episode_source
    ):
        raise ValueError("maintained V1 concurrency or cleanup contract drift")
    request_source = inspect.getsource(fleet_v1.Client.request)
    instance_source = inspect.getsource(fleet_v1.Client.get_instance)
    if (
        'raise PlatformError(f"{method} request returned HTTP {error.code}")' not in request_source
        or "self.base_url + f\"/v1/env/instances/{quote(instance_id, safe='')}\""
        not in instance_source
    ):
        raise ValueError("maintained Fleet V1 release-observation contract drift")
    cfg = common.Config(**plan["episode"])
    if cfg.max_concurrent_envs != 2:
        raise ValueError("signal environment concurrency is not exactly two")
    session_source = inspect.getsource(mechanics._evidence_session_class)
    if (
        '"sha256:" + digest(self.tools) != binding["tool_catalog_sha256"]' not in session_source
        or "live V1 task authority differs from the immutable plan" not in session_source
    ):
        raise ValueError("live session-open tool-schema gate drift")
    body = {
        "schema": RUNTIME_PREFLIGHT_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "image": mechanics.IMAGE,
        "fti_v1_sha256": "sha256:" + FTI_V1_SHA256,
        "miles_eval_sha256": "sha256:" + MILES_INFERENCE_EVAL_SHA256,
        "sample_index_start": 0,
        "sample_index_end": SAMPLES - 1,
        "sample_count": SAMPLES,
        "max_concurrent_envs": cfg.max_concurrent_envs,
        "shielded_close": True,
        "release_absence_http_status": 404,
        "tool_catalog_sha256": plan["task_binding"]["tool_catalog_sha256"],
        "live_tool_schema_gate_at_session_open": True,
        "outer_episode_replacements": 0,
    }
    return {**body, "sha256": "sha256:" + mechanics.digest(body)}


_SIGNAL_SESSION_CLASS: type | None = None


def _signal_session_class() -> type:
    global _SIGNAL_SESSION_CLASS
    if _SIGNAL_SESSION_CLASS is not None:
        return _SIGNAL_SESSION_CLASS
    base = mechanics._evidence_session_class()

    class SignalSession(base):
        def __init__(self, *args: Any, **kwargs: Any):
            state = _CURRENT_ATTEMPT.get()
            if state is None:
                raise ValueError("signal session lacks a sample-slot binding")
            self._signal_attempt_id = state["attempt_id"]
            self._signal_slot_id = state["slot_id"]
            self._signal_sample_index = state["sample_index"]
            self._signal_closed_written = False
            self._signal_release_confirmed = False
            self._signal_event_lock = threading.Lock()
            super().__init__(*args, **kwargs)
            self._write_signal_event("initialized")

        def _write_signal_event(self, stage: str) -> None:
            plan = mechanics._load_runtime_plan()
            state = _CURRENT_ATTEMPT.get()
            if state is None:
                raise ValueError("signal event lacks a sample-slot binding")
            instance_id = self.instance.instance_id if self.instance is not None else None
            body = {
                "schema": PRIVATE_ATTEMPT_SCHEMA,
                "plan_sha256": "sha256:" + mechanics.digest(plan),
                "slot_id": state["slot_id"],
                "attempt_number": state["attempt_number"],
                "attempt_id": self._signal_attempt_id,
                "stage": stage,
                "instance_id": instance_id,
                "cleanup_confirmed": bool(
                    stage == "closed"
                    and self.instance is not None
                    and self.deleted
                    and self.cleanup_error is None
                    and self._signal_release_confirmed
                ),
            }
            payload = {**body, "sha256": "sha256:" + mechanics.digest(body)}
            root = mechanics._private_directory(Path(plan["identity"]["run_dir"]))
            mechanics._write_once(
                root / f"signal-{self._signal_attempt_id}-{stage}.json",
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
            )

        def open(self) -> None:
            super().open()
            self._write_signal_event("opened")

        def close(self) -> None:
            super().close()
            with self._signal_event_lock:
                if not self._signal_closed_written and self.instance is not None:
                    if self.deleted and self.cleanup_error is None:
                        from fti.fleet.v1 import PlatformError

                        deadline = time.monotonic() + self.cfg.request_timeout_s
                        while time.monotonic() < deadline:
                            try:
                                self.client.get_instance(
                                    self.instance.instance_id,
                                    timeout_s=min(10, max(1, deadline - time.monotonic())),
                                )
                            except PlatformError as error:
                                if str(error) == "GET request returned HTTP 404":
                                    self._signal_release_confirmed = True
                                    break
                                self.cleanup_error = "instance absence check failed"
                                break
                            time.sleep(1)
                        if not self._signal_release_confirmed and self.cleanup_error is None:
                            self.cleanup_error = "instance absence check timed out"
                    self._write_signal_event("closed")
                    state = _CURRENT_ATTEMPT.get()
                    if state is not None:
                        state.update(
                            {
                                "instance_id": (
                                    self.instance.instance_id if self.instance is not None else None
                                ),
                                "verifier_execution_id": self.verifier_execution_id,
                                "cleanup_confirmed": bool(
                                    self.instance is not None
                                    and self.deleted
                                    and self.cleanup_error is None
                                    and self._signal_release_confirmed
                                ),
                            }
                        )
                    self._signal_closed_written = True

    _SIGNAL_SESSION_CLASS = SignalSession
    return SignalSession


def _accepted_episode_receipt(output: Any) -> str | None:
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    references = set()
    for sample in samples:
        metadata = getattr(sample, "metadata", None) or {}
        fleet = metadata.get("fleet_v1") if isinstance(metadata, dict) else None
        status = getattr(getattr(sample, "status", None), "value", None)
        reward = getattr(sample, "reward", None)
        if (
            status == "completed"
            and not isinstance(reward, bool)
            and isinstance(reward, (int, float))
            and math.isfinite(float(reward))
            and isinstance(fleet, dict)
            and fleet.get("graded") is True
            and fleet.get("authority_verified") is True
            and fleet.get("cleanup_confirmed") is True
            and fleet.get("done_reason") in mechanics.NORMAL_DONE_REASONS
            and type(fleet.get("tool_calls")) is int
            and fleet["tool_calls"] >= 1
        ):
            reference = fleet.get("authority_evidence_sha256")
            mechanics._sha256(reference, "accepted episode receipt")
            references.add(reference)
    if len(references) > 1:
        raise ValueError("one sample slot returned multiple accepted episodes")
    return next(iter(references), None)


def _write_slot_terminal(plan: dict[str, Any], state: dict[str, Any], status: str) -> None:
    body = {
        "schema": PRIVATE_SLOT_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "slot_id": state["slot_id"],
        "attempt_number": state["attempt_number"],
        "attempt_id": state.get("attempt_id"),
        "instance_id": state.get("instance_id"),
        "verifier_execution_id": state.get("verifier_execution_id"),
        "accepted_episode_receipt_sha256": state.get("accepted_episode_receipt_sha256"),
        "cleanup_confirmed": state.get("cleanup_confirmed") is True,
        "status": status,
    }
    payload = {**body, "sha256": "sha256:" + mechanics.digest(body)}
    root = mechanics._private_directory(Path(plan["identity"]["run_dir"]))
    mechanics._write_once(
        root / f"slot-{state['slot_id']}-{state['attempt_number']}.json",
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )


def _claim_slot(plan: dict[str, Any], sample_index: int) -> dict[str, Any]:
    slot = plan["qualification"]["sample_slots"][sample_index]
    body = {
        "schema": PRIVATE_CLAIM_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "sample_index": sample_index,
        "slot_id": slot["slot_id"],
        "attempt_number": 1,
        "attempt_id": str(uuid.uuid4()),
    }
    payload = {**body, "sha256": "sha256:" + mechanics.digest(body)}
    root = mechanics._private_directory(Path(plan["identity"]["run_dir"]))
    mechanics._write_once(
        root / f"claim-{slot['slot_id']}.json",
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    return body


async def generate_with_evidence(input: Any) -> Any:
    """Run one predeclared slot once through the maintained FTI path."""
    from fti.miles.v1 import client_recording

    client_recording.TaskSession = _signal_session_class()
    client_recording.episode_metadata = mechanics._evidence_episode_metadata
    plan = mechanics._load_runtime_plan()
    sample_index = getattr(getattr(input, "sample", None), "index", None)
    if type(sample_index) is not int or not 0 <= sample_index < SAMPLES:
        raise ValueError("Miles sample index is outside the predeclared signal slots")
    state = _claim_slot(plan, sample_index)
    token = _CURRENT_ATTEMPT.set(state)
    try:
        try:
            output = await client_recording.generate(input)
            accepted = _accepted_episode_receipt(output)
        except BaseException:
            _write_slot_terminal(plan, state, "unaccepted_stop")
            raise
        state["accepted_episode_receipt_sha256"] = accepted
        # The maintained callback sanitizes ordinary exceptions into an aborted
        # sample. Unknown or ungraded cells stop the lane and never replay.
        status = "accepted" if accepted is not None else "unaccepted_stop"
        _write_slot_terminal(plan, state, status)
        return output
    finally:
        _CURRENT_ATTEMPT.reset(token)


def _generate_add_arguments(parser: Any) -> None:
    from fti.miles.v1.client_recording import add_arguments

    add_arguments(parser)


generate_with_evidence.add_arguments = _generate_add_arguments


def _runtime_files(plan: dict[str, Any]) -> dict[str, str]:
    files = runtime_source_files()
    if runtime_source_manifest() != plan["runtime_sources"]:
        raise ValueError("custom runtime source manifest drift")
    return {
        **files,
        "plan.json": json.dumps(plan, sort_keys=True, separators=(",", ":")),
        "model.json": json.dumps(_model_manifest(), sort_keys=True, separators=(",", ":")),
    }


def _model_manifest() -> dict[str, Any]:
    from training.models import bound_model

    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "configs/models/qwen38-27b-1d4bf0f2.lock.json").read_text())
    weights = json.loads((root / "configs/models/qwen38-27b-1d4bf0f2.weights.json").read_text())
    value = bound_model(lock, weights, HF_MODEL_ROOT)
    if "sha256:" + mechanics.digest(value) != HF_MODEL_BINDING_SHA256:
        raise ValueError("repository model inventory binding drift")
    return value


def _prepared_model_exists(plan: dict[str, Any]) -> None:
    runtime = Path(os.environ["CYBER_RUNTIME_DIR"])
    manifest = json.loads((runtime / "model.json").read_text())
    if (
        "sha256:" + mechanics.digest(manifest) != plan["prepared_model"]["binding_sha256"]
        or manifest.get("root") != plan["prepared_model"]["root"]
        or manifest.get("revision") != plan["prepared_model"]["revision"]
    ):
        raise ValueError("signal HF model manifest differs from the plan")
    root = Path(manifest["root"])
    if root.is_symlink() or not root.is_dir():
        raise ValueError("signal HF model root is absent or unsafe")
    for item in manifest["files"]:
        path = root / item["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("signal HF model file is absent or unsafe")
        if type(item.get("size")) is int and path.stat().st_size != item["size"]:
            raise ValueError("signal HF model file size changed")
        if mechanics.file_sha256(path) != str(item["sha256"]).removeprefix("sha256:"):
            raise ValueError("signal HF model file digest changed")


def job_request(plan: dict[str, Any]) -> dict[str, Any]:
    from cyber_post_train.jobs import bundled_request

    plan = validate_plan(plan)
    identity = plan["identity"]
    request = {
        "name": identity["name"],
        "title": f"Qwen3.8 Miles96 fresh signal: {identity['name']}",
        "run_dir": identity["run_dir"],
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
            "MODEL_DIR": plan["prepared_model"]["root"],
            "FLEET_MODEL_DIR": identity["run_dir"] + "/model-output",
            "PYTHONPATH": identity["run_dir"] + "/.runtime",
            "MILES_SCRIPT_EXTERNAL_RAY": "1",
            "CYBER_PLAN_SHA256": mechanics.digest(plan),
            "CYBER_MODEL_BINDING_SHA256": plan["prepared_model"]["binding_sha256"],
            "CYBER_TASK_BINDING_SHA256": "sha256:" + mechanics.digest(plan["task_binding"]),
            "CYBER_RUNTIME_SOURCE_MANIFEST_SHA256": _runtime_source_manifest_sha256(plan),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        },
    }
    return _seal_runtime_request(
        bundled_request(
            request,
            _runtime_files(plan),
            "training.miles96_signal_qualification",
            ["--run", "--plan", "plan.json", "--sha256", mechanics.digest(plan)],
        )
    )


def _runtime_bundle_sha256(request: dict[str, Any]) -> str:
    env = request["env"]
    if "CYBER_RUNTIME_BUNDLE" in env:
        encoded = env["CYBER_RUNTIME_BUNDLE"]
    else:
        chunks = sorted(
            (
                int(key.rsplit("_", 1)[1]),
                value,
            )
            for key, value in env.items()
            if key.startswith("CYBER_RUNTIME_BUNDLE_")
            and key.removeprefix("CYBER_RUNTIME_BUNDLE_").isdigit()
        )
        encoded = "".join(value for _, value in chunks)
    return "sha256:" + hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest()


def _request_binding_sha256(request: dict[str, Any]) -> str:
    value = json.loads(json.dumps(request))
    value["env"].pop("CYBER_REQUEST_BINDING_SHA256", None)
    return "sha256:" + mechanics.digest(value)


def _seal_runtime_request(request: dict[str, Any]) -> dict[str, Any]:
    request["env"]["CYBER_RUNTIME_BUNDLE_SHA256"] = _runtime_bundle_sha256(request)
    request["env"]["CYBER_REQUEST_BINDING_SHA256"] = _request_binding_sha256(request)
    return request


def _private_episode_receipts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(plan["identity"]["run_dir"]) / mechanics.PRIVATE_EVIDENCE_DIR
    if not root.is_dir() or root.is_symlink():
        return []
    result = []
    for path in sorted(root.glob("*.json")):
        if not re.fullmatch(r"[a-f0-9]{64}\.json", path.name) or path.is_symlink():
            continue
        value = json.loads(path.read_text())
        body = {key: item for key, item in value.items() if key != "sha256"}
        sample_index = value.get("sample_index")
        expected_slot = (
            plan["qualification"]["sample_slots"][sample_index]["slot_id"]
            if type(sample_index) is int and 0 <= sample_index < SAMPLES
            else None
        )
        if (
            value.get("schema") != mechanics.PRIVATE_EPISODE_SCHEMA
            or value.get("sha256") != "sha256:" + mechanics.digest(body)
            or value.get("plan_sha256") != "sha256:" + mechanics.digest(plan)
            or value.get("task_key") != plan["task_binding"]["task_key"]
            or value.get("task_version_id") != plan["task_binding"]["task_version_id"]
            or value.get("verifier_version_id") != plan["task_binding"]["verifier_version_id"]
            or expected_slot is None
            or value.get("slot_id") != expected_slot
            or value.get("done_reason") not in mechanics.NORMAL_DONE_REASONS
            or type(value.get("tool_calls")) is not int
            or value["tool_calls"] < 1
            or value.get("cleanup_confirmed") is not True
        ):
            raise ValueError("private signal episode evidence is invalid")
        try:
            uuid.UUID(str(value.get("attempt_id")))
        except (TypeError, ValueError, AttributeError):
            raise ValueError("private signal episode evidence is invalid") from None
        reward = value.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError("private signal reward is not numeric")
        if not math.isfinite(float(reward)):
            raise ValueError("private signal reward is not finite")
        result.append(value)
    return result


def _private_attempt_receipts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(plan["identity"]["run_dir"]) / mechanics.PRIVATE_EVIDENCE_DIR
    if not root.is_dir() or root.is_symlink():
        return []
    result = []
    fields = {
        "schema",
        "plan_sha256",
        "slot_id",
        "attempt_number",
        "attempt_id",
        "stage",
        "instance_id",
        "cleanup_confirmed",
        "sha256",
    }
    for path in sorted(root.glob("signal-*.json")):
        if path.is_symlink():
            raise ValueError("private signal attempt evidence is unsafe")
        value = json.loads(path.read_text())
        body = {key: item for key, item in value.items() if key != "sha256"}
        try:
            attempt_id = str(uuid.UUID(str(value.get("attempt_id"))))
            slot_id = str(uuid.UUID(str(value.get("slot_id"))))
        except (TypeError, ValueError, AttributeError):
            raise ValueError("private signal attempt identity is invalid") from None
        if (
            set(value) != fields
            or value.get("schema") != PRIVATE_ATTEMPT_SCHEMA
            or value.get("sha256") != "sha256:" + mechanics.digest(body)
            or value.get("plan_sha256") != "sha256:" + mechanics.digest(plan)
            or value.get("slot_id") != slot_id
            or value.get("attempt_number") != 1
            or value.get("attempt_id") != attempt_id
            or value.get("stage") not in {"initialized", "opened", "closed"}
            or not (
                value.get("instance_id") is None
                or mechanics._uuid(value["instance_id"], "instance identity")
            )
            or type(value.get("cleanup_confirmed")) is not bool
        ):
            raise ValueError("private signal attempt evidence is invalid")
        result.append(value)
    return result


def _private_claim_receipts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(plan["identity"]["run_dir"]) / mechanics.PRIVATE_EVIDENCE_DIR
    if not root.is_dir() or root.is_symlink():
        return []
    result = []
    fields = {
        "schema",
        "plan_sha256",
        "sample_index",
        "slot_id",
        "attempt_number",
        "attempt_id",
        "sha256",
    }
    expected = {
        item["sample_index"]: item["slot_id"] for item in plan["qualification"]["sample_slots"]
    }
    for path in sorted(root.glob("claim-*.json")):
        if path.is_symlink():
            raise ValueError("private signal claim evidence is unsafe")
        value = json.loads(path.read_text())
        body = {key: item for key, item in value.items() if key != "sha256"}
        sample_index = value.get("sample_index")
        try:
            slot_id = str(uuid.UUID(str(value.get("slot_id"))))
            attempt_id = str(uuid.UUID(str(value.get("attempt_id"))))
        except (TypeError, ValueError, AttributeError):
            raise ValueError("private signal claim identity is invalid") from None
        if (
            set(value) != fields
            or value.get("schema") != PRIVATE_CLAIM_SCHEMA
            or value.get("sha256") != "sha256:" + mechanics.digest(body)
            or value.get("plan_sha256") != "sha256:" + mechanics.digest(plan)
            or type(sample_index) is not int
            or expected.get(sample_index) != slot_id
            or value.get("attempt_number") != 1
            or value.get("attempt_id") != attempt_id
            or path.name != f"claim-{slot_id}.json"
        ):
            raise ValueError("private signal claim evidence is invalid")
        result.append(value)
    return result


def _private_slot_receipts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(plan["identity"]["run_dir"]) / mechanics.PRIVATE_EVIDENCE_DIR
    if not root.is_dir() or root.is_symlink():
        return []
    result = []
    fields = {
        "schema",
        "plan_sha256",
        "slot_id",
        "attempt_number",
        "attempt_id",
        "instance_id",
        "verifier_execution_id",
        "accepted_episode_receipt_sha256",
        "cleanup_confirmed",
        "status",
        "sha256",
    }
    statuses = {"accepted", "infra_invalid_excluded", "unaccepted_stop"}
    for path in sorted(root.glob("slot-*.json")):
        if path.is_symlink():
            raise ValueError("private signal slot evidence is unsafe")
        value = json.loads(path.read_text())
        body = {key: item for key, item in value.items() if key != "sha256"}
        try:
            slot_id = str(uuid.UUID(str(value.get("slot_id"))))
            attempt_id = str(uuid.UUID(str(value.get("attempt_id"))))
        except (TypeError, ValueError, AttributeError):
            raise ValueError("private signal slot identity is invalid") from None
        for field in ("instance_id", "verifier_execution_id"):
            if value.get(field) is not None:
                mechanics._uuid(value[field], field)
        reference = value.get("accepted_episode_receipt_sha256")
        if reference is not None:
            mechanics._sha256(reference, "accepted episode receipt")
        if (
            set(value) != fields
            or value.get("schema") != PRIVATE_SLOT_SCHEMA
            or value.get("sha256") != "sha256:" + mechanics.digest(body)
            or value.get("plan_sha256") != "sha256:" + mechanics.digest(plan)
            or value.get("slot_id") != slot_id
            or value.get("attempt_id") != attempt_id
            or value.get("attempt_number") != 1
            or value.get("status") not in statuses
            or type(value.get("cleanup_confirmed")) is not bool
        ):
            raise ValueError("private signal slot evidence is invalid")
        status = value["status"]
        if status == "accepted" and (
            value["cleanup_confirmed"] is not True
            or value["instance_id"] is None
            or value["verifier_execution_id"] is None
            or reference is None
        ):
            raise ValueError("private signal slot terminal category is inconsistent")
        if status == "infra_invalid_excluded" and (
            value["cleanup_confirmed"] is not True
            or value["instance_id"] is None
            or value["verifier_execution_id"] is not None
            or reference is not None
        ):
            raise ValueError("private signal slot terminal category is inconsistent")
        if status == "unaccepted_stop" and reference is not None:
            raise ValueError("private signal slot terminal category is inconsistent")
        result.append(value)
    return result


def _lifecycle_valid(
    attempts: list[dict[str, Any]], slots: list[dict[str, Any]]
) -> tuple[bool, bool]:
    events: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for item in attempts:
        key = (item["slot_id"], item["attempt_number"])
        if item["stage"] in events.setdefault(key, {}):
            return False, False
        events[key][item["stage"]] = item
    terminals = {(item["slot_id"], item["attempt_number"]): item for item in slots}
    if len(terminals) != len(slots) or set(events) != set(terminals):
        return False, False
    attempt_ids = [item["attempt_id"] for item in slots]
    instance_ids = [item["instance_id"] for item in slots if item["instance_id"] is not None]
    verifier_ids = [
        item["verifier_execution_id"] for item in slots if item["verifier_execution_id"] is not None
    ]
    if (
        len(attempt_ids) != len(set(attempt_ids))
        or len(instance_ids) != len(set(instance_ids))
        or len(verifier_ids) != len(set(verifier_ids))
    ):
        return False, False
    all_released = True
    for key, terminal in terminals.items():
        stages = events[key]
        if set(stages) != {"initialized", "opened", "closed"}:
            return False, False
        initialized, opened, closed = (stages[name] for name in ("initialized", "opened", "closed"))
        if not (
            initialized["attempt_id"]
            == opened["attempt_id"]
            == closed["attempt_id"]
            == terminal["attempt_id"]
            and initialized["instance_id"] is None
            and opened["instance_id"] == closed["instance_id"] == terminal["instance_id"]
            and opened["instance_id"] is not None
            and initialized["cleanup_confirmed"] is False
            and opened["cleanup_confirmed"] is False
            and closed["cleanup_confirmed"] is terminal["cleanup_confirmed"]
        ):
            return False, False
        all_released = all_released and terminal["cleanup_confirmed"] is True
    return True, all_released


def _write_or_verify(path: Path, payload: bytes) -> None:
    try:
        mechanics._write_once(path, payload)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f"existing terminal evidence differs: {path.name}") from None


def _validate_runtime_preflight(plan: dict[str, Any], value: dict[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "sha256"}
    if (
        set(value)
        != {
            "schema",
            "plan_sha256",
            "image",
            "fti_v1_sha256",
            "miles_eval_sha256",
            "sample_index_start",
            "sample_index_end",
            "sample_count",
            "max_concurrent_envs",
            "shielded_close",
            "release_absence_http_status",
            "tool_catalog_sha256",
            "live_tool_schema_gate_at_session_open",
            "outer_episode_replacements",
            "sha256",
        }
        or value.get("sha256") != "sha256:" + mechanics.digest(body)
        or value.get("schema") != RUNTIME_PREFLIGHT_SCHEMA
        or value.get("plan_sha256") != "sha256:" + mechanics.digest(plan)
        or value.get("image") != mechanics.IMAGE
        or value.get("fti_v1_sha256") != "sha256:" + FTI_V1_SHA256
        or value.get("miles_eval_sha256") != "sha256:" + MILES_INFERENCE_EVAL_SHA256
        or value.get("sample_index_start") != 0
        or value.get("sample_index_end") != SAMPLES - 1
        or value.get("sample_count") != SAMPLES
        or value.get("max_concurrent_envs") != 2
        or value.get("shielded_close") is not True
        or value.get("release_absence_http_status") != 404
        or value.get("tool_catalog_sha256") != plan["task_binding"]["tool_catalog_sha256"]
        or value.get("live_tool_schema_gate_at_session_open") is not True
        or value.get("outer_episode_replacements") != 0
    ):
        raise ValueError("exact-image signal runtime preflight receipt is invalid")


def _runtime_preflight_receipt(plan: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads((Path(plan["identity"]["run_dir"]) / RUNTIME_PREFLIGHT_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("exact-image signal runtime preflight is absent") from exc
    _validate_runtime_preflight(plan, value)
    return value


def aggregate(plan: dict[str, Any]) -> dict[str, Any]:
    plan = validate_plan(plan)
    run_dir = Path(plan["identity"]["run_dir"])
    preflight = _runtime_preflight_receipt(plan)
    try:
        native = json.loads((run_dir / NATIVE_TERMINAL_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("native FTI signal run has not terminated") from exc
    native_body = {key: item for key, item in native.items() if key != "sha256"}
    native_fields = {
        "schema",
        "plan_sha256",
        "status",
        "returncode",
        "optimizer_steps",
        "checkpoint_artifacts_absent",
        "runtime_source_manifest_sha256",
        "runtime_bundle_sha256",
        "request_binding_sha256",
        "runtime_preflight_sha256",
        "sha256",
    }
    if (
        set(native) != native_fields
        or native.get("schema") != NATIVE_TERMINAL_SCHEMA
        or native.get("sha256") != "sha256:" + mechanics.digest(native_body)
        or native.get("plan_sha256") != "sha256:" + mechanics.digest(plan)
        or native.get("status") != "succeeded"
        or native.get("optimizer_steps") != 0
        or native.get("checkpoint_artifacts_absent") is not True
        or native.get("runtime_source_manifest_sha256") != _runtime_source_manifest_sha256(plan)
        or native.get("runtime_preflight_sha256") != preflight["sha256"]
        or type(native.get("returncode")) is not int
    ):
        raise ValueError("native FTI signal terminal receipt is invalid")
    mechanics._sha256(native["runtime_bundle_sha256"], "runtime bundle")
    mechanics._sha256(native["request_binding_sha256"], "request binding")
    episodes = _private_episode_receipts(plan)
    claims = _private_claim_receipts(plan)
    attempts = _private_attempt_receipts(plan)
    slots = _private_slot_receipts(plan)
    refs = [item["sha256"] for item in episodes]
    execution_ids = [item["verifier_execution_id"] for item in episodes]
    instance_ids = [item["instance_id"] for item in episodes]
    rewards = [float(item["reward"]) for item in episodes]
    unique = (
        len(refs) == len(set(refs))
        and len(execution_ids) == len(set(execution_ids))
        and len(instance_ids) == len(set(instance_ids))
    )
    lifecycle_valid, all_released = _lifecycle_valid(attempts, slots)
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for item in slots:
        by_slot.setdefault(item["slot_id"], []).append(item)
    planned_slots = {item["slot_id"] for item in plan["qualification"]["sample_slots"]}
    claim_slots = {item["slot_id"] for item in claims}
    claim_attempts = {item["slot_id"]: item["attempt_id"] for item in claims}
    terminal_attempts = {item["slot_id"]: item["attempt_id"] for item in slots}
    slot_accounting = (
        len(claims) == SAMPLES
        and claim_slots == planned_slots
        and set(by_slot) == planned_slots
        and all(len(records) == 1 for records in by_slot.values())
        and len(set(claim_attempts.values())) == SAMPLES
        and terminal_attempts == claim_attempts
    )
    accepted = []
    stopped = False
    for records in by_slot.values():
        records.sort(key=lambda item: item["attempt_number"])
        statuses = [item["status"] for item in records]
        numbers = [item["attempt_number"] for item in records]
        if statuses == ["accepted"] and numbers == [1]:
            accepted.append(records[0])
        elif statuses == ["infra_invalid_excluded"] and numbers == [1]:
            pass
        else:
            stopped = True
    claims_by_slot = {item["slot_id"]: item for item in claims}
    episodes_by_ref = {item["sha256"]: item for item in episodes}
    admitted = []
    evidence_linked = len(episodes_by_ref) == len(episodes)
    for terminal in accepted:
        episode = episodes_by_ref.get(terminal["accepted_episode_receipt_sha256"])
        claim = claims_by_slot.get(terminal["slot_id"])
        if (
            episode is None
            or claim is None
            or episode["slot_id"] != terminal["slot_id"]
            or episode["sample_index"] != claim["sample_index"]
            or not (episode["attempt_id"] == terminal["attempt_id"] == claim["attempt_id"])
            or episode["verifier_execution_id"] != terminal["verifier_execution_id"]
            or episode["instance_id"] != terminal["instance_id"]
        ):
            evidence_linked = False
            continue
        admitted.append(
            {
                "sample_index": claim["sample_index"],
                "slot_id": claim["slot_id"],
                "attempt_id": claim["attempt_id"],
                "instance_id": episode["instance_id"],
                "verifier_version_id": episode["verifier_version_id"],
                "verifier_execution_id": episode["verifier_execution_id"],
                "episode_receipt_sha256": episode["sha256"],
            }
        )
    evidence_linked = evidence_linked and set(episodes_by_ref) == {
        item["accepted_episode_receipt_sha256"] for item in accepted
    }
    if not slot_accounting:
        raise ValueError("all eight predeclared signal slots are not terminally accounted")
    all_released = all_released and lifecycle_valid and slot_accounting
    variation = len(set(rewards)) >= 2
    qualified = (
        native["returncode"] == 0
        and all_released
        and unique
        and evidence_linked
        and not stopped
        and 2 <= len(episodes) <= SAMPLES
        and variation
    )

    private_body = {
        "schema": PRIVATE_GROUP_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "runtime_source_manifest_sha256": native["runtime_source_manifest_sha256"],
        "runtime_bundle_sha256": native["runtime_bundle_sha256"],
        "request_binding_sha256": native["request_binding_sha256"],
        "runtime_preflight_sha256": native["runtime_preflight_sha256"],
        "episode_receipts": sorted(refs),
        "claim_receipts": sorted(item["sha256"] for item in claims),
        "attempt_receipts": sorted(item["sha256"] for item in attempts),
        "slot_receipts": sorted(item["sha256"] for item in slots),
        "verifier_execution_ids": sorted(execution_ids),
        "instance_ids": sorted(instance_ids),
        "rewards": rewards,
        "admitted_episode_bindings": sorted(admitted, key=lambda item: item["sample_index"]),
    }
    private = {**private_body, "sha256": "sha256:" + mechanics.digest(private_body)}
    private_dir = mechanics._private_directory(run_dir)
    _write_or_verify(
        private_dir / PRIVATE_GROUP_FILE,
        json.dumps(private, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    common = {
        "phase1_plan_sha256": "sha256:" + mechanics.digest(plan),
        "phase1_run_name": plan["identity"]["name"],
        "model_revision": plan["prepared_model"]["revision"],
        "model_binding_sha256": plan["prepared_model"]["binding_sha256"],
        **plan["selection_authority"],
        "planned_slot_count": SAMPLES,
        "terminal_slot_count": len(slots),
        "excluded_slot_count": sum(item["status"] == "infra_invalid_excluded" for item in slots),
        "outer_replacement_count": 0,
        "task_key": plan["task_binding"]["task_key"],
        "task_version_id": plan["task_binding"]["task_version_id"],
        "verifier_version_id": plan["task_binding"]["verifier_version_id"],
        "task_set_sha256": plan["task_binding"]["task_set_sha256"],
        "tool_catalog_sha256": plan["task_binding"]["tool_catalog_sha256"],
        "max_turns": plan["episode"]["max_turns"],
        "max_tokens_per_turn": plan["episode"]["max_tokens_per_turn"],
        "episode_timeout_s": plan["episode"]["episode_timeout_s"],
        "completed_episode_count": len(episodes),
        "finite_rewards": bool(rewards),
        "reward_variation": variation,
        "all_instances_released": all_released,
        "source_receipt_sha256": private["sha256"],
        "native_terminal_sha256": native["sha256"],
        "runtime_source_manifest_sha256": native["runtime_source_manifest_sha256"],
        "runtime_bundle_sha256": native["runtime_bundle_sha256"],
        "request_binding_sha256": native["request_binding_sha256"],
        "runtime_preflight_sha256": native["runtime_preflight_sha256"],
        "optimizer_steps": 0,
        "checkpoint_artifacts_absent": True,
    }
    if qualified:
        body = {"schema": mechanics.TASK_SIGNAL_EVIDENCE_SCHEMA, **common}
        payload = {**body, "sha256": "sha256:" + mechanics.digest(body)}
        path = run_dir / EVIDENCE_FILE
    else:
        body = {
            "schema": REJECTION_SCHEMA,
            "plan_sha256": "sha256:" + mechanics.digest(plan),
            "expected_episode_count": SAMPLES,
            **common,
        }
        payload = {**body, "sha256": "sha256:" + mechanics.digest(body)}
        path = run_dir / REJECTION_FILE
    _write_or_verify(
        path,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    return payload


def _checkpoint_artifacts_absent(output_dir: Path) -> bool:
    checkpoint_dir = output_dir / "checkpoints"
    return not (
        checkpoint_dir.exists() and any(path.is_file() for path in checkpoint_dir.rglob("*"))
    )


def _runtime_receipt_binding(plan: dict[str, Any]) -> dict[str, str]:
    source = _runtime_source_manifest_sha256(plan)
    if os.environ.get("CYBER_RUNTIME_SOURCE_MANIFEST_SHA256") != source:
        raise ValueError("runtime source manifest binding drift")
    bundle = mechanics._sha256(os.environ.get("CYBER_RUNTIME_BUNDLE_SHA256"), "runtime bundle")
    request = mechanics._sha256(os.environ.get("CYBER_REQUEST_BINDING_SHA256"), "request binding")
    return {
        "runtime_source_manifest_sha256": source,
        "runtime_bundle_sha256": bundle,
        "request_binding_sha256": request,
    }


def _write_native_terminal(
    plan: dict[str, Any],
    *,
    status: str,
    returncode: int | None,
    checkpoint_artifacts_absent: bool,
    binding: dict[str, str],
    runtime_preflight_sha256: str,
) -> dict[str, Any]:
    if status not in {
        "succeeded",
        "native_nonzero",
        "native_execution_error",
        "zero_update_assertion_failed",
    }:
        raise ValueError("unsupported native signal terminal status")
    body = {
        "schema": NATIVE_TERMINAL_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "status": status,
        "returncode": returncode,
        "optimizer_steps": 0 if status == "succeeded" else None,
        "checkpoint_artifacts_absent": checkpoint_artifacts_absent,
        "runtime_preflight_sha256": runtime_preflight_sha256,
        **binding,
    }
    payload = {**body, "sha256": "sha256:" + mechanics.digest(body)}
    _write_or_verify(
        Path(plan["identity"]["run_dir"]) / NATIVE_TERMINAL_FILE,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    return payload


def _write_native_rejection(plan: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema": REJECTION_SCHEMA,
        "plan_sha256": "sha256:" + mechanics.digest(plan),
        "status": native["status"],
        "returncode": native["returncode"],
        "optimizer_steps": native["optimizer_steps"],
        "checkpoint_artifacts_absent": native["checkpoint_artifacts_absent"],
        "native_terminal_sha256": native["sha256"],
        "runtime_source_manifest_sha256": native["runtime_source_manifest_sha256"],
        "runtime_bundle_sha256": native["runtime_bundle_sha256"],
        "request_binding_sha256": native["request_binding_sha256"],
        "runtime_preflight_sha256": native["runtime_preflight_sha256"],
    }
    payload = {**body, "sha256": "sha256:" + mechanics.digest(body)}
    _write_or_verify(
        Path(plan["identity"]["run_dir"]) / REJECTION_FILE,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    return payload


def run(plan: dict[str, Any], expected_digest: str) -> dict[str, Any]:
    plan = validate_plan(plan)
    if mechanics.digest(plan) != expected_digest:
        raise ValueError("runtime signal plan digest mismatch")
    run_dir = Path(plan["identity"]["run_dir"])
    data_dir = run_dir / "data"
    output_dir = run_dir / "model-output"
    if (
        os.environ.get("RUN_DIR") != str(run_dir)
        or os.environ.get("MODEL_DIR") != plan["prepared_model"]["root"]
        or os.environ.get("FLEET_MODEL_DIR") != str(output_dir)
        or os.environ.get("CYBER_PLAN_SHA256") != expected_digest
        or os.environ.get("CYBER_MODEL_BINDING_SHA256") != plan["prepared_model"]["binding_sha256"]
        or os.environ.get("CYBER_TASK_BINDING_SHA256")
        != "sha256:" + mechanics.digest(plan["task_binding"])
        or os.environ.get("CYBER_RUNTIME_SOURCE_MANIFEST_SHA256")
        != _runtime_source_manifest_sha256(plan)
    ):
        raise ValueError("runtime signal identity drift")
    if not run_dir.is_dir() or any(
        path.exists() for path in (data_dir, output_dir, run_dir / LAUNCH_FILE)
    ):
        raise FileExistsError("signal output is not create-once")
    run_dir.chmod(0o700)
    binding = _runtime_receipt_binding(plan)
    preflight = _runtime_signal_binding(plan)
    mechanics._write_once(
        run_dir / RUNTIME_PREFLIGHT_FILE,
        json.dumps(preflight, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    _prepared_model_exists(plan)
    data_dir.mkdir(mode=0o700)
    output_dir.mkdir(mode=0o700)
    rows = task_rows(plan)
    mechanics._write_once(data_dir / "eval.jsonl", rows)
    launch = {
        "schema": "cyber_qwen38_miles96_signal_launch_v1",
        "plan_sha256": expected_digest,
        "dataset_sha256": "sha256:" + hashlib.sha256(rows).hexdigest(),
        "row_count": 1,
        "sample_count": SAMPLES,
        "optimizer_steps": 0,
        "trainer_image": mechanics.IMAGE,
        "fti_version": mechanics.FTI_VERSION,
    }
    mechanics._write_once(
        run_dir / LAUNCH_FILE,
        json.dumps(launch, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    try:
        try:
            completed = subprocess.run(
                [sys.executable, *native_arguments(plan)], check=False, env=os.environ
            )
        except BaseException:
            native = _write_native_terminal(
                plan,
                status="native_execution_error",
                returncode=None,
                checkpoint_artifacts_absent=_checkpoint_artifacts_absent(output_dir),
                binding=binding,
                runtime_preflight_sha256=preflight["sha256"],
            )
            _write_native_rejection(plan, native)
            raise
        checkpoint_absent = _checkpoint_artifacts_absent(output_dir)
        status = (
            "native_nonzero"
            if completed.returncode
            else "succeeded"
            if checkpoint_absent
            else "zero_update_assertion_failed"
        )
        native = _write_native_terminal(
            plan,
            status=status,
            returncode=completed.returncode,
            checkpoint_artifacts_absent=checkpoint_absent,
            binding=binding,
            runtime_preflight_sha256=preflight["sha256"],
        )
        if completed.returncode:
            _write_native_rejection(plan, native)
            raise subprocess.CalledProcessError(completed.returncode, completed.args)
        if not checkpoint_absent:
            _write_native_rejection(plan, native)
            raise ValueError("zero-update signal run unexpectedly wrote a checkpoint")
        return aggregate(plan)
    finally:
        (data_dir / "eval.jsonl").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true")
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if mechanics.digest(validate_plan(plan)) != args.sha256:
        raise ValueError("signal plan digest mismatch")
    value = run(plan, args.sha256) if args.run else aggregate(plan)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
