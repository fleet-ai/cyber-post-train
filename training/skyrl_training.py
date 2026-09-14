"""Exact Fleet GRPO through native SkyRL, using the shared Jobs API lifecycle.

Preparation/CPU checks are not real reward, optimizer or recovery qualification.
The initial profile starts from the pinned base; it never auto-resumes a run.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib
import json
import math
import numbers
import os
import re
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from pathlib import Path

from cyber_post_train.jobs import (
    API_URLS,
    JobsError,
    bundled_request,
    digest,
    quantity,
    validate_request,
)

from . import skyrl
from .miles_conversion import _hash, _write, check_inputs
from .rl_runtime import HardDeadlineExceeded, hard_deadline, sealed
from .skyrl_episode import _module

SCHEMA = "cyber_skyrl_training_v1"
ENGINE_DIAGNOSTIC_SCHEMA = "cyber_skyrl_engine_start_diagnostic_v1"
ENGINE_DIAGNOSTIC_ACCEPTANCE_SCHEMA = "cyber_skyrl_engine_diagnostic_terminal_v1"
ENGINE_DIAGNOSTIC_WORKERS = 2
ENGINE_DIAGNOSTIC_GPUS_PER_WORKER = 4
# The unqualified names below bind the completed dev8 acceptance closure. They
# are historical evidence inputs, not a launch selector; never retarget them to
# a successor whose API/Kubernetes identities do not exist yet.
ENGINE_DIAGNOSTIC_CONFIG_PATH = "qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v8.json"
ENGINE_DIAGNOSTIC_CONFIG_NAME = "chris-q38-rldiag-dev8"
ENGINE_DIAGNOSTIC_OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-rldiag-dev8"
ENGINE_DIAGNOSTIC_CONFIG_FILE_SHA256 = (
    "sha256:d5d1f3d32abbb674fef4beea85d77b1c48fb99a53ab8672fcb55a469180a7164"
)
DEV8_ENGINE_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "e48827529b1cf5fafa153b2aed1b774c2eec86905baf5ccb62b36300533e252b"
)
DEV8_ENGINE_IMAGE_CPU_QUALIFICATION = {
    "schema": "cyber_q38_rl_image_cleanpull_cpu_qualification_v1",
    "status": "qualified",
    "classification": "operational_gate",
    "source_commit": "34de8d5753b8dfe44460ff9656db4ddc9a85a62c",
    "receipt_sha256": "4326ec6a28f1f2deee6d6ebaf9c04f80ec8ba5ac8856ae636e91aca5e4e53837",
    "evidence_path": (
        "docs/evidence/qwen38-study/2026-09-12-skyrl-startup-relay-image-cpu-qualification-v1.json"
    ),
}
DEV_KUBERNETES_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
DEV_KUBERNETES_NAMESPACE = "fleet-train-jobs"
DEV_KUBERNETES_NAMESPACE_UID = "10394b76-e1d4-40b1-a8e2-7575e95df216"
PROD_KUBERNETES_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
PROD_KUBERNETES_NAMESPACE = "fleet-train-jobs"
PROD_KUBERNETES_NAMESPACE_UID = "fd6d2fcd-687a-4257-9dba-a034bb381e6b"
ENGINE_DIAGNOSTIC_SOURCE_COMMIT = "6724567675252c2010ca6830a27d5c33ce562afc"
ENGINE_DIAGNOSTIC_PLAN_FILE_SHA256 = (
    "sha256:ae3648c649e768a7c41d2d067e7b39b3ab5bd85af8095e4976950a0c9849fb3b"
)
ENGINE_DIAGNOSTIC_REQUEST_FILE_SHA256 = (
    "sha256:7261e3f6473b5b8e4fae3435c0fe028dfc8e8d9455682aa9d18fa799cfd540cd"
)
ENGINE_DIAGNOSTIC_PREPARED_FILE_SHA256 = (
    "sha256:bf8730940c80f539269388588da2b8f89bc546469ef17f07e57cd8e16f2dcb43"
)
ENGINE_DIAGNOSTIC_PREFLIGHT_FILE_SHA256 = (
    "sha256:6585db1585cd255e29b93fa2d567363d2f468af87fda90cdaae8818a1ff7f448"
)
ENGINE_DIAGNOSTIC_SUBMISSION_JOURNAL_FILE_SHA256 = (
    "sha256:211cc69381d809a72a726e271d20ef55df92bb4cb151303ac6f754238ec39a0c"
)
ENGINE_DIAGNOSTIC_PLAN_SHA256 = "e99f022c6ea2c23b2effbd0669a8de1b21dc9e72703aa6a6f7279f20674f3535"
ENGINE_DIAGNOSTIC_REQUEST_SHA256 = (
    "6a26d6011d668a1deb8c6c9cd6a707fcf01e94d3876a6d6e52d0dba4807477bd"
)
ENGINE_DIAGNOSTIC_MANIFEST_SHA256 = (
    "80bd359ac7ba9d55076b93987c3935dcf8b16fb4cbc220e67eec16bc8ade0d81"
)
ENGINE_DIAGNOSTIC_API_RUN_NAME = "chris-q38-rldiag-dev8-0d7bed25"
ENGINE_DIAGNOSTIC_API_JOB_ID = "chris-q38-rldiag-dev8-0d7bed25-bbsbq"
ENGINE_DIAGNOSTIC_RAYJOB_UID = "6279d187-0f36-4c1a-be94-8a64e29559da"
ENGINE_DIAGNOSTIC_WORKLOAD_NAME = "rayjob-chris-q38-rldiag-dev8-0d7bed25-37b74"
ENGINE_DIAGNOSTIC_WORKLOAD_UID = "b31ce6e5-de9d-4292-bae9-bfab6f7dd561"
ENGINE_DIAGNOSTIC_RAYCLUSTER_NAME = "chris-q38-rldiag-dev8-0d7bed25-j8v7b"
ENGINE_DIAGNOSTIC_RAYCLUSTER_UID = "c4be4319-3333-43a6-b8e6-ef49ecfb6fab"
ENGINE_DIAGNOSTIC_PODS = (
    (
        "chris-q38-rldiag-dev8-0d7bed25-j8v7b-gpu-worker-jh8k8",
        "ec007189-b2a2-4902-b87f-d68b3c9f0eac",
    ),
    (
        "chris-q38-rldiag-dev8-0d7bed25-j8v7b-head-48cvs",
        "b98ce606-2a74-4b1a-8d7f-cbf3c3402f06",
    ),
)
ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_PATH = "qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v9.json"
ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_NAME = "chris-q38-rldiag-dev9"
ENGINE_DIAGNOSTIC_SUCCESSOR_WANDB_RUN_ID = "chris-q38-rldiag-dev9"
ENGINE_DIAGNOSTIC_SUCCESSOR_OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-rldiag-dev9"
ENGINE_DIAGNOSTIC_SUCCESSOR_DATA_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev9/data"
)
ENGINE_DIAGNOSTIC_SUCCESSOR_PREPARED_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev9/prepared-v1"
)
ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_FILE_SHA256 = (
    "sha256:6b17e668cde7eaded5bb5df5a010568702f8260d2c4f71e61cb9396383515538"
)
ENGINE_DIAGNOSTIC_SUCCESSOR_DATA_CONFIG_FILE_SHA256 = (
    "sha256:084bb9a65b77482e46f0218ff990a920abed2712de1e8cea4f8053dca594846e"
)
ENGINE_DIAGNOSTIC_REJECTED_CONFIG_PATH = "qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v10.json"
ENGINE_DIAGNOSTIC_REJECTED_CONFIG_NAME = "chris-q38-rldiag-dev10"
ENGINE_DIAGNOSTIC_REJECTED_WANDB_RUN_ID = "chris-q38-rldiag-dev10"
ENGINE_DIAGNOSTIC_REJECTED_OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-rldiag-dev10"
ENGINE_DIAGNOSTIC_REJECTED_DATA_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev10/data"
)
ENGINE_DIAGNOSTIC_REJECTED_PREPARED_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev10/prepared-v1"
)
ENGINE_DIAGNOSTIC_REJECTED_CONFIG_FILE_SHA256 = (
    "sha256:9cfbb2b4bbbf97e84c1dc32f5dc4ebad5e6ddf670467b28e2019d4262f85ba9f"
)
ENGINE_DIAGNOSTIC_REJECTED_DATA_CONFIG_FILE_SHA256 = (
    "sha256:b349aa746c333f98bdebcc0d4147b0c85080008c0d0b81bea6050bfcf847083a"
)
ENGINE_DIAGNOSTIC_CURRENT_CONFIG_PATH = "qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v11.json"
ENGINE_DIAGNOSTIC_CURRENT_CONFIG_NAME = "chris-q38-rldiag-dev11"
ENGINE_DIAGNOSTIC_CURRENT_WANDB_RUN_ID = "chris-q38-rldiag-dev11"
ENGINE_DIAGNOSTIC_CURRENT_OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-rldiag-dev11"
ENGINE_DIAGNOSTIC_CURRENT_DATA_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev11/data"
)
ENGINE_DIAGNOSTIC_CURRENT_PREPARED_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev11/prepared-v1"
)
ENGINE_DIAGNOSTIC_CURRENT_CONFIG_FILE_SHA256 = (
    "sha256:079160bac1cfd9a900a5d1d3e27236d58e9448c4fa59197c99ad3787ff317478"
)
ENGINE_DIAGNOSTIC_CURRENT_DATA_CONFIG_FILE_SHA256 = (
    "sha256:b95b896a0cee0813751d63d939d73580598d51dc25e06142bfb6cfbae9244dbf"
)
ENGINE_DIAGNOSTIC_PROD_CONFIG_NAME = "chris-q38-rldiag-prod1"
ENGINE_DIAGNOSTIC_PROD_WANDB_RUN_ID = ENGINE_DIAGNOSTIC_PROD_CONFIG_NAME
ENGINE_DIAGNOSTIC_PROD_OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-rldiag-prod1"
ENGINE_DIAGNOSTIC_PROD_DATA_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-prod1/data"
)
ENGINE_DIAGNOSTIC_PROD_PREPARED_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-prod1/prepared-v1"
)
ENGINE_DIAGNOSTIC_PROD_CONFIG_SHA256 = (
    "480dae102f5f047cd68cd7e07acede5afebcbf1b22e39c60f94742ce70e4c989"
)
ENGINE_DIAGNOSTIC_PROD2_CONFIG_NAME = "chris-q38-rldiag-prod2"
ENGINE_DIAGNOSTIC_PROD2_WANDB_RUN_ID = ENGINE_DIAGNOSTIC_PROD2_CONFIG_NAME
ENGINE_DIAGNOSTIC_PROD2_OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-rldiag-prod2"
ENGINE_DIAGNOSTIC_PROD2_DATA_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-prod2/data"
)
ENGINE_DIAGNOSTIC_PROD2_PREPARED_ROOT = (
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-prod2/prepared-v1"
)
ENGINE_DIAGNOSTIC_PROD2_CONFIG_SHA256 = (
    "f5602c336f98dd772e0e26009fa8c8902bc2e4a67ba17e211ff9e88b5649109b"
)
VLLM_SAMPLER_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
ENGINE_DIAGNOSTIC_MODEL_SHA256 = "dcfdcd6ecb6661741cd3a4b24dc5af7259642c8a6824773e0de70d55d7501179"
ENGINE_DIAGNOSTIC_DATA_IDENTITY = {
    "selection_sha256": "sha256:8672a1bcb7073ee93d30c6cbc5b6a140d21571c8b58fc6100d7007a6f2e56a9e",
    "split_sha256": "sha256:22b2a68908e46ad6060779126186c466cee8c56fc3df3ea763cc77985bc5ae86",
    "tool_catalog_sha256": (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    ),
    "template_sha256": ("sha256:c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "tokenizer_sha256": "61463c912eb610ee47b0ade3837572e5c048310b5be04bdf8164b9c15aa149e5",
    "limits": {
        "context_tokens": 98304,
        "response_tokens": 81920,
        "max_tokens_per_turn": 4096,
        "max_turns": 80,
        "episode_seconds": 2400,
        "tool_seconds": 330,
        "tool_result_chars": 50000,
    },
}
REWARD_CANARY_DATA_CONTRACT = {
    "selection_sha256": "sha256:15400d5e58c53c8c2258c826c39aa5545b14d1705bece3d927d09d38de76b157",
    "split_sha256": "sha256:8279ea19808ad1accb00d3f3145c3ec087030677786e0188251cfc197f306adf",
    "tool_catalog_sha256": (
        "sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"
    ),
    "limits": {
        "context_tokens": 98304,
        "response_tokens": 81920,
        "max_tokens_per_turn": 4096,
        "max_turns": 600,
        "episode_seconds": 2400,
        "tool_seconds": 330,
        "tool_result_chars": 50000,
    },
    "rows": {"train": 1, "dev": 1},
}
REWARD_CANARY_RUNTIME_USER = {"uid": 1000, "gid": 100, "run_as_non_root": True}
REWARD_CANARY_ARGUMENTS = {
    "name": "chris-q38-rlreward-dev1",
    "output_root": "/mnt/sfs/jobs/chris-q38-rlreward-dev1",
    "model": "Qwen/Qwen3.8-27B",
    "model_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
    "train_data": (
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-dev1/data/train.jsonl"
    ),
    "dev_data": "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-dev1/data/dev.jsonl",
    "data_manifest": (
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-dev1/data/manifest.json"
    ),
    "train_rows": 1,
    "dev_rows": 1,
    "wandb_entity": "thefleet",
    "wandb_project": "cyber-post-train",
    "wandb_run_id": "chris-q38-rlreward-dev1",
    "context_tokens": 98304,
    "response_tokens": 81920,
    "tokens_per_turn": 4096,
    "max_turns": 600,
    "nodes": 1,
    "steps": 1,
    "groups": 1,
    "samples_per_prompt": 8,
    "lr": 1e-6,
    "eval_interval": 1,
    "checkpoint_interval": 1,
    "keep_checkpoints": 2,
    "seed": 42,
    "engine_start_timeout_seconds": 1800,
    "engine_cleanup_timeout_seconds": 300,
}
REWARD_CANARY_RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "64",
    "memory_request": "512Gi",
    "memory_limit": "768Gi",
}
REWARD_CANARY_MODEL_IDENTITY = {
    "repo": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
    "weight_manifest_sha256": (
        "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
    ),
}
REWARD_CANARY_EPISODE_AUDIT_SCHEMA = "cyber_skyrl_reward_canary_episode_audit_v1"
REWARD_CANARY_BASE_POLICY_RANK_SCHEMA = "cyber_skyrl_reward_canary_base_policy_rank_v1"
REWARD_CANARY_UPDATE_PROOF_SCHEMA = "cyber_skyrl_reward_canary_optimizer_update_v1"
REWARD_CANARY_WANDB_SCHEMA = "cyber_skyrl_reward_canary_wandb_history_v1"
REWARD_CANARY_TERMINAL_SCHEMA = "cyber_skyrl_rl_reward_canary_terminal_v1"
_CREDENTIAL_ENV_NAME = re.compile(
    r"(?:^|_)(?:TOKENS?|PASSWORDS?|PASSWD|CREDENTIALS?|SECRETS?|API_KEYS?|"
    r"ACCESS_KEYS?|PRIVATE_KEYS?|DATABASE_URL|AUTH(?:ORIZATION)?)(?:_|$)",
    re.IGNORECASE,
)
_ALWAYS_SCRUB_WORKER_ENV = frozenset({"FLEET_API_KEY", "WANDB_API_KEY"})
MODULE = "training.skyrl_training"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f"
)
ENGINE_IMAGE_CPU_QUALIFICATION = {
    "schema": "cyber_q38_rl_image_cleanpull_cpu_qualification_v1",
    "status": "qualified",
    "classification": "operational_gate",
    "source_commit": "8d62868d6dc00eee793d83efe5738dc21e42758d",
    "receipt_sha256": "28244d695896b8b9766df66caecd117a33fd5d9c2c5df35faba12bcc785d09f1",
    "evidence_path": (
        "docs/evidence/qwen38-study/"
        "2026-09-12-skyrl-worker-rpc-relay-image-cpu-qualification-v1.json"
    ),
}
_ENGINE_START_DISQUALIFIED = frozenset(
    {
        (
            "Qwen/Qwen3.8-27B",
            "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
            "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4",
        ),
        (
            "Qwen/Qwen3.8-27B",
            "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
            "9b6f43938f9b28aaff7ba91edd59be3d18b01f9a22078ba5475cdac5e6bfcca6",
        ),
    }
)
_ENGINE_PRIVACY_DISQUALIFIED = frozenset({("Qwen/Qwen3.8-27B", DEV8_ENGINE_IMAGE)})
NATIVE = {
    **skyrl.NATIVE_SOURCES,
    "skyrl.train.entrypoints.main_base": (
        "aee8976aa5d18a0c19be93e0b99fc1d1868688af0fe26b08b8a863c0aed027d7"
    ),
    # The immutable image adds phase-utilization and log-probability telemetry
    # to upstream f5bc3b78; optimizer and checkpoint semantics are unchanged.
    "skyrl.train.trainer": "0e3e2e4d1bbe28f8920ddd3c73999fae27308f949b60c8ea8062d41527dc3b21",
    "skyrl.train.dataset.dataset": (
        "ff041e24a24e7d99c9c20052ae643b137acb0f1777016ddba79260a42226466c"
    ),
}
RUNTIME_FILES = (
    "training/skyrl_training.py",
    "training/skyrl_promotion.py",
    "training/skyrl.py",
    "training/skyrl_rollout.py",
    "training/skyrl_episode.py",
    "training/rl_episode.py",
    "training/rl_runtime.py",
    "training/rl_data.py",
    "training/rl_reward_canary.py",
    "training/sft_runtime.py",
    "training/dense.py",
    "training/io.py",
    "training/sft.py",
    "training/models.py",
    "training/corpus.py",
    "training/source_coverage.py",
    "training/study_data.py",
    "training/splits.py",
    "training/qwen_tools.py",
    "training/miles_conversion.py",
    "training/miles.py",
    "evals/fleet/opencode_self_hosted.py",
    "cyber_post_train/jobs.py",
)


def _runtime():
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def _canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _semantic_state_digest(value: object) -> str:
    """Hash tensor state by value, dtype, shape and canonical container topology."""
    import numpy as np
    import torch

    hasher = hashlib.sha256()

    def update(item: object) -> None:
        if hasattr(item, "to_local"):
            item = item.to_local()
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous().cpu()
            metadata = _canonical_json({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
            hasher.update(b"tensor")
            hasher.update(len(metadata).to_bytes(8, "big"))
            hasher.update(metadata)
            raw = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
            hasher.update(len(raw).to_bytes(8, "big"))
            hasher.update(raw)
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            metadata = _canonical_json({"dtype": str(array.dtype), "shape": list(array.shape)})
            hasher.update(b"numpy")
            hasher.update(len(metadata).to_bytes(8, "big"))
            hasher.update(metadata)
            raw = memoryview(array).cast("B")
            hasher.update(len(raw).to_bytes(8, "big"))
            hasher.update(raw)
        elif isinstance(item, dict):
            if any(type(key) not in {str, int} for key in item):
                raise ValueError("state dictionary key type changed")
            hasher.update(b"dict")
            ordered = sorted(item, key=lambda key: (type(key).__name__, str(key)))
            hasher.update(len(ordered).to_bytes(8, "big"))
            for key in ordered:
                update(key)
                update(item[key])
        elif isinstance(item, tuple):
            hasher.update(b"tuple")
            hasher.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
        elif isinstance(item, list):
            hasher.update(b"list")
            hasher.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
        elif item is None or type(item) in {bool, int, float, str}:
            payload = json.dumps(
                {"type": type(item).__name__, "value": item},
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            hasher.update(b"scalar")
            hasher.update(len(payload).to_bytes(8, "big"))
            hasher.update(payload)
        else:
            raise ValueError("unsupported policy state value in reward-canary proof")

    update(value)
    return hasher.hexdigest()


def _policy_state_digest(state: object) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("policy state must be a nonempty mapping")
    return _semantic_state_digest(dict(state))


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def is_prod_engine_diagnostic_config(config: object) -> bool:
    """Identify the exact source config allowed to skip training-promotion gates."""
    return isinstance(config, dict) and digest(config) in {
        ENGINE_DIAGNOSTIC_PROD_CONFIG_SHA256,
        ENGINE_DIAGNOSTIC_PROD2_CONFIG_SHA256,
    }


def is_prod_engine_diagnostic(plan: object) -> bool:
    """Recognize only the two create-once c1 production engine gates."""
    if not isinstance(plan, Mapping):
        return False
    arguments = plan.get("arguments")
    data = plan.get("data")
    execution = plan.get("execution")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    data = data if isinstance(data, Mapping) else {}
    execution = execution if isinstance(execution, Mapping) else {}
    profiles = (
        (
            ENGINE_DIAGNOSTIC_PROD_CONFIG_NAME,
            ENGINE_DIAGNOSTIC_PROD_WANDB_RUN_ID,
            ENGINE_DIAGNOSTIC_PROD_OUTPUT_ROOT,
            ENGINE_DIAGNOSTIC_PROD_DATA_ROOT,
        ),
        (
            ENGINE_DIAGNOSTIC_PROD2_CONFIG_NAME,
            ENGINE_DIAGNOSTIC_PROD2_WANDB_RUN_ID,
            ENGINE_DIAGNOSTIC_PROD2_OUTPUT_ROOT,
            ENGINE_DIAGNOSTIC_PROD2_DATA_ROOT,
        ),
    )
    markers = (
        plan.get("run_name"),
        arguments.get("name"),
        data.get("name"),
        arguments.get("wandb_run_id"),
        plan.get("output_root"),
        arguments.get("output_root"),
        arguments.get("train_data"),
        arguments.get("dev_data"),
        arguments.get("data_manifest"),
    )
    profile = next(
        (
            profile
            for profile in profiles
            if any(
                value == expected
                for value, expected in zip(
                    markers,
                    (
                        profile[0],
                        profile[0],
                        profile[0],
                        profile[1],
                        profile[2],
                        profile[2],
                        profile[3] + "/train.jsonl",
                        profile[3] + "/dev.jsonl",
                        profile[3] + "/manifest.json",
                    ),
                )
            )
        ),
        None,
    )
    if profile is None:
        return False
    name, wandb_run_id, output_root, data_root = profile
    expected_args = {
        "name": name,
        "output_root": output_root,
        "model": "Qwen/Qwen3.8-27B",
        "model_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "train_data": data_root + "/train.jsonl",
        "dev_data": data_root + "/dev.jsonl",
        "data_manifest": data_root + "/manifest.json",
        "train_rows": 2,
        "dev_rows": 1,
        "wandb_entity": "thefleet",
        "wandb_project": "cyber-post-train",
        "wandb_run_id": wandb_run_id,
        "context_tokens": 98304,
        "response_tokens": 81920,
        "tokens_per_turn": 4096,
        "max_turns": 80,
        "nodes": 1,
        "steps": 1,
        "groups": 2,
        "samples_per_prompt": 4,
        "lr": 1e-6,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "keep_checkpoints": 2,
        "seed": 42,
        "engine_start_timeout_seconds": 1800,
        "engine_cleanup_timeout_seconds": 300,
    }
    expected_execution = {
        "image": IMAGE,
        "image_cpu_qualification": ENGINE_IMAGE_CPU_QUALIFICATION,
        "cluster_target": "prod",
        "priority": "c1",
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    }
    expected_markers = (
        name,
        name,
        name,
        wandb_run_id,
        output_root,
        output_root,
        data_root + "/train.jsonl",
        data_root + "/dev.jsonl",
        data_root + "/manifest.json",
    )
    files = data.get("files") if isinstance(data.get("files"), Mapping) else {}
    if (
        tuple(markers) != expected_markers
        or arguments != expected_args
        or execution != expected_execution
        or digest(plan.get("model")) != ENGINE_DIAGNOSTIC_MODEL_SHA256
        or any(
            data.get(key) != value
            for key, value in ENGINE_DIAGNOSTIC_DATA_IDENTITY.items()
            if key != "tokenizer_sha256"
        )
        or digest(data.get("tokenizer")) != ENGINE_DIAGNOSTIC_DATA_IDENTITY["tokenizer_sha256"]
        or not _exact_integer(data.get("gpus"), 0)
        or not _exact_integer(data.get("environment_creates"), 0)
        or set(files) != {"train", "dev"}
        or any(
            not isinstance(files.get(split), Mapping)
            or files[split].get("path") != split + ".jsonl"
            or not _exact_integer(files[split].get("rows"), rows)
            for split, rows in (("train", 2), ("dev", 1))
        )
    ):
        raise ValueError(f"{name} engine diagnostic identity is incomplete or mixed")
    return True


def _require_fresh_engine_diagnostic_identity(plan: object) -> None:
    """Retire dev8-dev10 and reject partial reuse of the fresh dev11 identity."""
    if not isinstance(plan, Mapping):
        return
    arguments = plan.get("arguments")
    data = plan.get("data")
    execution = plan.get("execution")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    data = data if isinstance(data, Mapping) else {}
    execution = execution if isinstance(execution, Mapping) else {}
    observed = {
        "run_name": plan.get("run_name"),
        "argument_name": arguments.get("name"),
        "data_name": data.get("name"),
        "wandb_run_id": arguments.get("wandb_run_id"),
        "output_root": plan.get("output_root"),
        "argument_output_root": arguments.get("output_root"),
        "train_data": arguments.get("train_data"),
        "dev_data": arguments.get("dev_data"),
        "data_manifest": arguments.get("data_manifest"),
        "cluster_target": execution.get("cluster_target"),
    }
    retired_root = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev8/data"
    retired = {
        "run_name": ENGINE_DIAGNOSTIC_CONFIG_NAME,
        "argument_name": ENGINE_DIAGNOSTIC_CONFIG_NAME,
        "data_name": ENGINE_DIAGNOSTIC_CONFIG_NAME,
        "wandb_run_id": ENGINE_DIAGNOSTIC_CONFIG_NAME,
        "output_root": ENGINE_DIAGNOSTIC_OUTPUT_ROOT,
        "argument_output_root": ENGINE_DIAGNOSTIC_OUTPUT_ROOT,
        "train_data": retired_root + "/train.jsonl",
        "dev_data": retired_root + "/dev.jsonl",
        "data_manifest": retired_root + "/manifest.json",
    }
    if any(observed.get(key) == value for key, value in retired.items()):
        raise ValueError("terminal dev8 identity cannot be replayed")

    terminal_successor = {
        "run_name": ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_NAME,
        "argument_name": ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_NAME,
        "data_name": ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_NAME,
        "wandb_run_id": ENGINE_DIAGNOSTIC_SUCCESSOR_WANDB_RUN_ID,
        "output_root": ENGINE_DIAGNOSTIC_SUCCESSOR_OUTPUT_ROOT,
        "argument_output_root": ENGINE_DIAGNOSTIC_SUCCESSOR_OUTPUT_ROOT,
        "train_data": ENGINE_DIAGNOSTIC_SUCCESSOR_DATA_ROOT + "/train.jsonl",
        "dev_data": ENGINE_DIAGNOSTIC_SUCCESSOR_DATA_ROOT + "/dev.jsonl",
        "data_manifest": ENGINE_DIAGNOSTIC_SUCCESSOR_DATA_ROOT + "/manifest.json",
        "cluster_target": "dev",
    }
    terminal_markers = {
        key: value for key, value in terminal_successor.items() if key != "cluster_target"
    }
    if any(observed.get(key) == value for key, value in terminal_markers.items()):
        raise ValueError("terminal dev9 identity cannot be replayed")

    terminal_rejected = {
        "run_name": ENGINE_DIAGNOSTIC_REJECTED_CONFIG_NAME,
        "argument_name": ENGINE_DIAGNOSTIC_REJECTED_CONFIG_NAME,
        "data_name": ENGINE_DIAGNOSTIC_REJECTED_CONFIG_NAME,
        "wandb_run_id": ENGINE_DIAGNOSTIC_REJECTED_WANDB_RUN_ID,
        "output_root": ENGINE_DIAGNOSTIC_REJECTED_OUTPUT_ROOT,
        "argument_output_root": ENGINE_DIAGNOSTIC_REJECTED_OUTPUT_ROOT,
        "train_data": ENGINE_DIAGNOSTIC_REJECTED_DATA_ROOT + "/train.jsonl",
        "dev_data": ENGINE_DIAGNOSTIC_REJECTED_DATA_ROOT + "/dev.jsonl",
        "data_manifest": ENGINE_DIAGNOSTIC_REJECTED_DATA_ROOT + "/manifest.json",
    }
    if any(observed.get(key) == value for key, value in terminal_rejected.items()):
        raise ValueError("terminal dev10 identity cannot be replayed")

    current = {
        "run_name": ENGINE_DIAGNOSTIC_CURRENT_CONFIG_NAME,
        "argument_name": ENGINE_DIAGNOSTIC_CURRENT_CONFIG_NAME,
        "data_name": ENGINE_DIAGNOSTIC_CURRENT_CONFIG_NAME,
        "wandb_run_id": ENGINE_DIAGNOSTIC_CURRENT_WANDB_RUN_ID,
        "output_root": ENGINE_DIAGNOSTIC_CURRENT_OUTPUT_ROOT,
        "argument_output_root": ENGINE_DIAGNOSTIC_CURRENT_OUTPUT_ROOT,
        "train_data": ENGINE_DIAGNOSTIC_CURRENT_DATA_ROOT + "/train.jsonl",
        "dev_data": ENGINE_DIAGNOSTIC_CURRENT_DATA_ROOT + "/dev.jsonl",
        "data_manifest": ENGINE_DIAGNOSTIC_CURRENT_DATA_ROOT + "/manifest.json",
        "cluster_target": "dev",
    }
    current_markers = {key: value for key, value in current.items() if key != "cluster_target"}
    if any(observed.get(key) == value for key, value in current_markers.items()):
        resources = execution.get("resources")
        critical_args = {
            "nodes": 1,
            "steps": 1,
            "groups": 2,
            "samples_per_prompt": 4,
            "engine_start_timeout_seconds": 1800,
            "engine_cleanup_timeout_seconds": 300,
        }
        exact_resources = {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        }
        if (
            any(observed.get(key) != value for key, value in current.items())
            or any(arguments.get(key) != value for key, value in critical_args.items())
            or arguments.get("wandb_entity") != "thefleet"
            or arguments.get("wandb_project") != "cyber-post-train"
            or digest(plan.get("model")) != ENGINE_DIAGNOSTIC_MODEL_SHA256
            or any(
                data.get(key) != value
                for key, value in ENGINE_DIAGNOSTIC_DATA_IDENTITY.items()
                if key != "tokenizer_sha256"
            )
            or digest(data.get("tokenizer")) != ENGINE_DIAGNOSTIC_DATA_IDENTITY["tokenizer_sha256"]
            or execution.get("image") != IMAGE
            or execution.get("image_cpu_qualification") != ENGINE_IMAGE_CPU_QUALIFICATION
            or execution.get("priority") != "c1"
            or resources != exact_resources
        ):
            raise ValueError("dev11 identity is incomplete or mixed")
    is_prod_engine_diagnostic(plan)


def _engine_evidence_root() -> Path:
    return Path(ENGINE_DIAGNOSTIC_OUTPUT_ROOT)


def _engine_durable_root() -> Path:
    return Path("/mnt/sfs/jobs")


def _exact_integer(value, expected: int) -> bool:
    return type(value) is int and value == expected


def _observed_epoch(value: object) -> float:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("dev8 evidence needs an exact UTC observation time")
    try:
        return dt.datetime.fromisoformat(value[:-1] + "+00:00").timestamp()
    except ValueError as error:
        raise ValueError("dev8 evidence observation time is invalid") from error


def _snapshot(path: Path) -> tuple[bytes, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("dev8 evidence file is missing or indirect")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if any(
        getattr(before, key) != getattr(after, key)
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    ):
        raise ValueError("dev8 evidence file changed while it was read")
    return payload, "sha256:" + hashlib.sha256(payload).hexdigest()


def _json_snapshot(path: Path) -> tuple[dict, str]:
    payload, file_sha256 = _snapshot(path)
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("dev8 evidence file is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("dev8 evidence file must be a JSON object")
    return value, file_sha256


def _bound_evidence(binding: object, expected_path: Path, schema: str) -> tuple[dict, str]:
    if not isinstance(binding, dict) or set(binding) != {"path", "file_sha256", "value"}:
        raise ValueError("dev8 evidence binding fields changed")
    if binding.get("path") != str(expected_path):
        raise ValueError("dev8 evidence path changed")
    value, file_sha256 = _json_snapshot(expected_path)
    if binding.get("file_sha256") != file_sha256 or binding.get("value") != value:
        raise ValueError("dev8 evidence file digest or value changed")
    try:
        sealed(value, schema)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("dev8 evidence receipt is not digest-valid") from error
    return value, file_sha256


def _image_digest(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("dev8 runtime imageID is absent")
    match = re.fullmatch(
        r"(?:(?:containerd|docker-pullable)://(?:[^@\s]+@)?|[^@\s]+@)"
        r"sha256:([a-f0-9]{64})",
        value,
    )
    if match is None:
        raise ValueError("dev8 runtime imageID is not an immutable observed digest")
    return match.group(1)


def _validate_exact_engine_diagnostic_plan(plan: object) -> None:
    """Recognize only the exact dev8 config closure before trusting its self-digest."""
    if not isinstance(plan, dict):
        raise ValueError("dev8 prepared plan is not an object")
    data, args, execution = (
        plan.get("data"),
        plan.get("arguments"),
        plan.get("execution"),
    )
    data_root = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev8/data"
    expected_args = {
        "name": ENGINE_DIAGNOSTIC_CONFIG_NAME,
        "output_root": ENGINE_DIAGNOSTIC_OUTPUT_ROOT,
        "model": "Qwen/Qwen3.8-27B",
        "model_root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "train_data": data_root + "/train.jsonl",
        "dev_data": data_root + "/dev.jsonl",
        "data_manifest": data_root + "/manifest.json",
        "train_rows": 2,
        "dev_rows": 1,
        "wandb_entity": "thefleet",
        "wandb_project": "cyber-post-train",
        "wandb_run_id": ENGINE_DIAGNOSTIC_CONFIG_NAME,
        "context_tokens": 98304,
        "response_tokens": 81920,
        "tokens_per_turn": 4096,
        "max_turns": 80,
        "nodes": 1,
        "steps": 1,
        "groups": 2,
        "samples_per_prompt": 4,
        "lr": 1e-6,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "keep_checkpoints": 2,
        "seed": 42,
        "engine_start_timeout_seconds": 1800,
        "engine_cleanup_timeout_seconds": 300,
    }
    expected_execution = {
        "image": DEV8_ENGINE_IMAGE,
        "image_cpu_qualification": DEV8_ENGINE_IMAGE_CPU_QUALIFICATION,
        "priority": "c1",
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    }
    try:
        sealed(data, "cyber_skyrl_data_v1")
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("dev8 prepared data manifest is not digest-valid") from error
    files = data.get("files", {}) if isinstance(data, dict) else {}
    identity = ENGINE_DIAGNOSTIC_DATA_IDENTITY
    if (
        set(plan)
        != {
            "schema",
            "run_name",
            "output_root",
            "model",
            "data",
            "arguments",
            "native_overrides",
            "native_sources",
            "runtime_sha256",
            "execution",
        }
        or plan.get("schema") != SCHEMA
        or plan.get("run_name") != ENGINE_DIAGNOSTIC_CONFIG_NAME
        or plan.get("output_root") != ENGINE_DIAGNOSTIC_OUTPUT_ROOT
        or digest(plan.get("model")) != ENGINE_DIAGNOSTIC_MODEL_SHA256
        or args != expected_args
        or execution != expected_execution
        or not isinstance(data, dict)
        or set(data)
        != {
            "schema",
            "name",
            "gpus",
            "environment_creates",
            "selection_sha256",
            "split_sha256",
            "tool_catalog_sha256",
            "tokenizer",
            "template_sha256",
            "limits",
            "files",
            "sha256",
        }
        or data.get("name") != ENGINE_DIAGNOSTIC_CONFIG_NAME
        or not _exact_integer(data.get("gpus"), 0)
        or not _exact_integer(data.get("environment_creates"), 0)
        or any(data.get(key) != identity[key] for key in identity if key != "tokenizer_sha256")
        or digest(data.get("tokenizer")) != identity["tokenizer_sha256"]
        or set(files) != {"train", "dev"}
        or any(
            not isinstance(files.get(split), dict)
            or set(files[split]) != {"path", "rows", "max_prompt_tokens", "sha256"}
            or files[split].get("path") != split + ".jsonl"
            or not _exact_integer(files[split].get("rows"), rows)
            or not _exact_integer(files[split].get("max_prompt_tokens"), prompt_tokens)
            or _SHA256.fullmatch(str(files[split].get("sha256", ""))) is None
            for split, rows, prompt_tokens in (("train", 2, 1241), ("dev", 1, 1256))
        )
    ):
        raise ValueError("dev8 prepared plan differs from the exact source config closure")


def _validate_prepared_inputs(
    prepared: object,
    *,
    expected_plan_sha256: str,
    expected_request_sha256: str,
) -> dict:
    fields = {
        "directory",
        "source_commit",
        "config_file_sha256",
        "plan_file_sha256",
        "request_file_sha256",
        "prepared_receipt_file_sha256",
        "preflight_file_sha256",
        "submission_journal_file_sha256",
    }
    if not isinstance(prepared, dict) or set(prepared) != fields:
        raise ValueError("dev8 prepared-input binding fields changed")
    directory = Path(str(prepared.get("directory", "")))
    durable_root = _engine_durable_root().resolve()
    resolved = directory.resolve()
    output_root = _engine_evidence_root().resolve()
    if (
        not directory.is_absolute()
        or directory.is_symlink()
        or directory != resolved
        or not resolved.is_relative_to(durable_root)
        or not resolved.is_dir()
        or resolved == output_root
        or resolved.is_relative_to(output_root)
        or output_root.is_relative_to(resolved)
    ):
        raise ValueError("dev8 prepared inputs are not on exact durable storage")
    if prepared.get("config_file_sha256") != ENGINE_DIAGNOSTIC_CONFIG_FILE_SHA256:
        raise ValueError("dev8 prepared inputs name a different source config")
    if prepared.get("source_commit") != ENGINE_DIAGNOSTIC_SOURCE_COMMIT:
        raise ValueError("dev8 prepared inputs name a different source commit")
    exact_file_digests = {
        "plan_file_sha256": ENGINE_DIAGNOSTIC_PLAN_FILE_SHA256,
        "request_file_sha256": ENGINE_DIAGNOSTIC_REQUEST_FILE_SHA256,
        "prepared_receipt_file_sha256": ENGINE_DIAGNOSTIC_PREPARED_FILE_SHA256,
        "preflight_file_sha256": ENGINE_DIAGNOSTIC_PREFLIGHT_FILE_SHA256,
        "submission_journal_file_sha256": ENGINE_DIAGNOSTIC_SUBMISSION_JOURNAL_FILE_SHA256,
    }
    if any(prepared.get(key) != expected for key, expected in exact_file_digests.items()):
        raise ValueError("dev8 prepared artifact differs from the immutable active closure")
    values: dict[str, dict] = {}
    for name, field in (
        ("plan.json", "plan_file_sha256"),
        ("request.json", "request_file_sha256"),
        ("PREPARED.json", "prepared_receipt_file_sha256"),
        ("PREFLIGHT.json", "preflight_file_sha256"),
    ):
        values[name], file_sha256 = _json_snapshot(resolved / name)
        if prepared.get(field) != file_sha256:
            raise ValueError(f"dev8 {name} digest changed")
    plan, request = values["plan.json"], values["request.json"]
    _validate_exact_engine_diagnostic_plan(plan)
    validate_request(request)
    plan_sha256, request_sha256 = digest(plan), digest(request)
    if (
        plan_sha256 != ENGINE_DIAGNOSTIC_PLAN_SHA256
        or request_sha256 != ENGINE_DIAGNOSTIC_REQUEST_SHA256
        or expected_plan_sha256 != ENGINE_DIAGNOSTIC_PLAN_SHA256
        or expected_request_sha256 != ENGINE_DIAGNOSTIC_REQUEST_SHA256
    ):
        raise ValueError("dev8 prepared plan/request differ from the compiled exact identities")
    if values["PREPARED.json"] != {
        "plan_sha256": plan_sha256,
        "request_sha256": request_sha256,
    }:
        raise ValueError("dev8 PREPARED receipt differs from plan/request bytes")
    preflight = values["PREFLIGHT.json"]
    preflight_body = {key: value for key, value in preflight.items() if key != "sha256"}
    required_false = (
        "rollouts",
        "verifier_calls",
        "optimizer_updates",
        "checkpoints",
        "wandb",
        "engine_start_qualified",
    )
    if (
        set(preflight)
        != {
            "schema",
            "status",
            "gpus",
            "runtime_user",
            "plan_sha256",
            "request_sha256",
            "native_parser_checked",
            "engine_cli_args_checked",
            "startup_error_transport_checked",
            "model_files",
            "task_rows_read",
            *required_false,
            "diagnostic_workers",
            "diagnostic_gpus_per_worker",
            "diagnostic_total_gpus",
            "num_engines",
            "tensor_parallel_size",
            "sha256",
        }
        or preflight.get("sha256") != digest(preflight_body)
        or preflight.get("schema") != "cyber_skyrl_engine_diagnostic_cpu_preflight_v1"
        or preflight.get("status") != "passed"
        or not _exact_integer(preflight.get("gpus"), 0)
        or preflight.get("runtime_user") != {"uid": 1000, "gid": 100}
        or preflight.get("plan_sha256") != plan_sha256
        or preflight.get("request_sha256") != request_sha256
        or preflight.get("native_parser_checked") is not True
        or preflight.get("engine_cli_args_checked") is not True
        or preflight.get("startup_error_transport_checked") is not True
        or not _exact_integer(preflight.get("model_files"), len(plan["model"]["files"]))
        or not _exact_integer(preflight.get("task_rows_read"), 0)
        or any(preflight.get(key) is not False for key in required_false)
        or not all(
            _exact_integer(preflight.get(key), expected)
            for key, expected in (
                ("diagnostic_workers", ENGINE_DIAGNOSTIC_WORKERS),
                ("diagnostic_gpus_per_worker", ENGINE_DIAGNOSTIC_GPUS_PER_WORKER),
                ("diagnostic_total_gpus", 8),
                ("num_engines", ENGINE_DIAGNOSTIC_WORKERS),
                ("tensor_parallel_size", ENGINE_DIAGNOSTIC_GPUS_PER_WORKER),
            )
        )
    ):
        raise ValueError("dev8 CPU preflight is absent, changed, or not zero-work")

    journal_path = resolved / "SUBMISSION.jsonl"
    payload, journal_sha256 = _snapshot(journal_path)
    if prepared.get("submission_journal_file_sha256") != journal_sha256:
        raise ValueError("dev8 submission journal digest changed")
    try:
        rows = [json.loads(line) for line in payload.splitlines()]
    except (TypeError, ValueError) as error:
        raise ValueError("dev8 submission journal is not valid JSONL") from error
    if len(rows) != 2 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("dev8 submission journal needs exactly one intent and response")
    intent, response = rows
    if (
        set(intent)
        != {
            "state",
            "api_base_url",
            "request_sha256",
            "manifest_sha256",
            "nodes",
            "gpus",
            "image",
        }
        or intent.get("state") != "POST_INTENT_DO_NOT_RETRY"
        or intent.get("api_base_url") != API_URLS["dev"]
        or intent.get("request_sha256") != request_sha256
        or intent.get("manifest_sha256") != ENGINE_DIAGNOSTIC_MANIFEST_SHA256
        or not _exact_integer(intent.get("nodes"), ENGINE_DIAGNOSTIC_WORKERS)
        or not _exact_integer(intent.get("gpus"), 8)
        or intent.get("image") != DEV8_ENGINE_IMAGE
        or set(response)
        != {
            "state",
            "name",
            "job_id",
            "run_dir",
            "status",
            "created_at",
            "finished_at",
        }
        or response.get("state") != "POST_RESPONSE"
        or response
        != {
            "state": "POST_RESPONSE",
            "name": ENGINE_DIAGNOSTIC_API_RUN_NAME,
            "job_id": None,
            "run_dir": ENGINE_DIAGNOSTIC_OUTPUT_ROOT,
            "status": None,
            "created_at": None,
            "finished_at": None,
        }
    ):
        raise ValueError("dev8 submission journal differs from the exact dev request")
    return {
        "directory": str(directory),
        "plan": plan,
        "request": request,
        "plan_sha256": plan_sha256,
        "request_sha256": request_sha256,
        "preflight": preflight,
        "response": response,
        "preflight_file_sha256": prepared["preflight_file_sha256"],
        "submission_journal_file_sha256": journal_sha256,
    }


def _validate_engine_diagnostic_receipt(
    receipt,
    *,
    config_file_sha256,
    expected_plan_sha256,
    expected_request_sha256,
):
    """Validate exact dev8 preparation, execution ownership, and GPU release."""
    if (
        not isinstance(receipt, dict)
        or set(receipt)
        != {
            "schema",
            "classification",
            "prepared_inputs",
            "diagnostic_receipt",
            "controller_audit",
            "release_evidence",
            "sha256",
        }
        or receipt.get("schema") != ENGINE_DIAGNOSTIC_ACCEPTANCE_SCHEMA
        or receipt.get("classification") != "accepted_engine_start_zero_work"
        or receipt.get("sha256")
        != digest({key: value for key, value in receipt.items() if key != "sha256"})
        or config_file_sha256 != ENGINE_DIAGNOSTIC_CONFIG_FILE_SHA256
    ):
        raise ValueError("dev8 terminal receipt is not digest-valid")
    prepared = _validate_prepared_inputs(
        receipt["prepared_inputs"],
        expected_plan_sha256=expected_plan_sha256,
        expected_request_sha256=expected_request_sha256,
    )
    output_root = _engine_evidence_root()
    if (
        not output_root.is_absolute()
        or output_root.is_symlink()
        or not output_root.is_dir()
        or output_root != output_root.resolve()
    ):
        raise ValueError("dev8 evidence output is not an exact durable directory")
    native, native_file_sha256 = _bound_evidence(
        receipt["diagnostic_receipt"],
        output_root / "ENGINE_DIAGNOSTIC.json",
        ENGINE_DIAGNOSTIC_SCHEMA,
    )
    cleanup = native.get("cleanup", {})
    output = native.get("output_postconditions", {})
    required_native_zero = (
        "task_rows_read",
        "rollouts",
        "verifier_calls",
        "optimizer_steps",
        "checkpoints_created",
        "registry_actors_created",
    )
    required_output_zero = (
        "checkpoint_artifacts",
        "episode_artifacts",
        "task_artifacts",
        "unexpected_output_artifacts",
    )
    if (
        native.get("plan_sha256") != prepared["plan_sha256"]
        or native.get("runtime_user") != {"uid": 1000, "gid": 100}
        or native.get("status") != "passed"
        or native.get("engine_start_state") != "all"
        or native.get("engine_started") is not True
        or native.get("diagnostic_completed") is not True
        or native.get("engine_start_qualified") is not True
        or native.get("training_qualified") is not False
        or native.get("production_training_shape_qualified") is not False
        or native.get("checkpoint_created") is not False
        or native.get("wandb_initialized") is not False
        or any(not _exact_integer(native.get(key), 0) for key in required_native_zero)
        or not all(
            _exact_integer(native.get(key), expected)
            for key, expected in (
                ("diagnostic_workers", ENGINE_DIAGNOSTIC_WORKERS),
                ("diagnostic_gpus_per_worker", ENGINE_DIAGNOSTIC_GPUS_PER_WORKER),
                ("diagnostic_total_gpus", 8),
                ("num_engines", ENGINE_DIAGNOSTIC_WORKERS),
                ("tensor_parallel_size", ENGINE_DIAGNOSTIC_GPUS_PER_WORKER),
                ("ray_gpu_nodes_expected", ENGINE_DIAGNOSTIC_WORKERS),
                ("ray_gpu_nodes_discovered", ENGINE_DIAGNOSTIC_WORKERS),
                ("ray_gpu_nodes_probed", ENGINE_DIAGNOSTIC_WORKERS),
                ("ray_actor_environment_probes_passed", ENGINE_DIAGNOSTIC_WORKERS),
                ("ray_actor_environment_probe_failures", 0),
                ("ray_actor_nonempty_scrubbed_credentials", 0),
                ("router_start_attempts", 1),
                ("router_environment_probes_passed", 1),
                ("router_environment_probe_failures", 0),
            )
        )
        or native.get("credential_environment_isolation_proven") is not True
        or native.get("ray_actor_environment_isolation_proven") is not True
        or native.get("ray_initialization_attempted") is not True
        or native.get("router_child_credential_environment_isolation") != "proven"
        or not isinstance(output, dict)
        or any(not _exact_integer(output.get(key), 0) for key in required_output_zero)
        or output.get("runtime_files_unchanged") is not True
        or not isinstance(cleanup, dict)
        or not _exact_integer(cleanup.get("tracked_engine_actors"), ENGINE_DIAGNOSTIC_WORKERS)
        or not _exact_integer(cleanup.get("active_owned_actors"), 0)
        or not _exact_integer(cleanup.get("active_owned_placement_groups"), 0)
        or cleanup.get("cleanup_proven") is not True
        or not isinstance(native.get("completed_at"), numbers.Real)
        or isinstance(native.get("completed_at"), bool)
    ):
        raise ValueError("dev8 native receipt does not prove accepted zero-work engine start")

    audit, audit_file_sha256 = _bound_evidence(
        receipt["controller_audit"],
        output_root / "CONTROLLER_AUDIT.json",
        "cyber_skyrl_engine_diagnostic_controller_audit_v1",
    )
    audit_fields = {
        "schema",
        "cluster",
        "api_base_url",
        "kubernetes_context",
        "namespace",
        "namespace_uid",
        "config_name",
        "plan_sha256",
        "request_sha256",
        "api_run_name",
        "api_job_id",
        "rayjob_name",
        "rayjob_uid",
        "workload_name",
        "workload_uid",
        "workload_owner_rayjob_uid",
        "raycluster_name",
        "raycluster_uid",
        "raycluster_owner_rayjob_uid",
        "pods",
        "effective_priority",
        "automatic_requeue",
        "workers",
        "gpus_per_worker",
        "total_gpus",
        "observed_at",
        "sha256",
    }
    pods = audit.get("pods")
    pod_fields = {
        "name",
        "uid",
        "owner_raycluster_uid",
        "runtime_image_id",
        "runtime_uid",
        "runtime_gid",
        "effective_security_context",
        "phase",
        "container_exit_code",
        "termination_reason",
        "terminated_at",
        "container_restarts",
        "gpus",
    }
    image_sha256 = DEV8_ENGINE_IMAGE.rsplit("@sha256:", 1)[1]
    if (
        set(audit) != audit_fields
        or not isinstance(pods, list)
        or len(pods) != ENGINE_DIAGNOSTIC_WORKERS
        or any(not isinstance(pod, dict) or set(pod) != pod_fields for pod in pods)
        or [(pod["name"], pod["uid"]) for pod in pods] != list(ENGINE_DIAGNOSTIC_PODS)
        or any(
            not isinstance(pod.get("name"), str)
            or not pod["name"]
            or _UUID.fullmatch(str(pod.get("uid", ""))) is None
            or pod.get("owner_raycluster_uid") != audit.get("raycluster_uid")
            or _image_digest(pod.get("runtime_image_id")) != image_sha256
            or not _exact_integer(pod.get("runtime_uid"), 1000)
            or not _exact_integer(pod.get("runtime_gid"), 100)
            or pod.get("effective_security_context")
            != {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}
            or pod.get("phase") != "Succeeded"
            or not _exact_integer(pod.get("container_exit_code"), 0)
            or pod.get("termination_reason") != "Completed"
            or _observed_epoch(pod.get("terminated_at")) > _observed_epoch(audit.get("observed_at"))
            or not _exact_integer(pod.get("container_restarts"), 0)
            or not _exact_integer(pod.get("gpus"), ENGINE_DIAGNOSTIC_GPUS_PER_WORKER)
            for pod in pods
        )
        or audit.get("cluster") != "dev"
        or audit.get("api_base_url") != API_URLS["dev"]
        or audit.get("kubernetes_context") != DEV_KUBERNETES_CONTEXT
        or audit.get("namespace") != DEV_KUBERNETES_NAMESPACE
        or audit.get("namespace_uid") != DEV_KUBERNETES_NAMESPACE_UID
        or audit.get("config_name") != ENGINE_DIAGNOSTIC_CONFIG_NAME
        or audit.get("plan_sha256") != prepared["plan_sha256"]
        or audit.get("request_sha256") != prepared["request_sha256"]
        or audit.get("api_run_name") != ENGINE_DIAGNOSTIC_API_RUN_NAME
        or audit.get("api_run_name") != prepared["response"].get("name")
        or audit.get("api_job_id") != ENGINE_DIAGNOSTIC_API_JOB_ID
        or audit.get("rayjob_name") != ENGINE_DIAGNOSTIC_API_RUN_NAME
        or audit.get("rayjob_uid") != ENGINE_DIAGNOSTIC_RAYJOB_UID
        or audit.get("workload_name") != ENGINE_DIAGNOSTIC_WORKLOAD_NAME
        or audit.get("workload_uid") != ENGINE_DIAGNOSTIC_WORKLOAD_UID
        or audit.get("raycluster_name") != ENGINE_DIAGNOSTIC_RAYCLUSTER_NAME
        or audit.get("raycluster_uid") != ENGINE_DIAGNOSTIC_RAYCLUSTER_UID
        or audit.get("workload_owner_rayjob_uid") != audit.get("rayjob_uid")
        or audit.get("raycluster_owner_rayjob_uid") != audit.get("rayjob_uid")
        or not _exact_integer(audit.get("effective_priority"), 10000)
        or audit.get("automatic_requeue") is not False
        or not all(
            _exact_integer(audit.get(key), expected)
            for key, expected in (
                ("workers", ENGINE_DIAGNOSTIC_WORKERS),
                ("gpus_per_worker", ENGINE_DIAGNOSTIC_GPUS_PER_WORKER),
                ("total_gpus", 8),
            )
        )
    ):
        raise ValueError("dev8 controller audit does not prove exact dev ownership")
    identities = [
        audit.get("namespace_uid"),
        audit.get("rayjob_uid"),
        audit.get("workload_uid"),
        audit.get("raycluster_uid"),
        *(pod["uid"] for pod in pods),
    ]
    if any(_UUID.fullmatch(str(item)) is None for item in identities) or len(
        set(identities)
    ) != len(identities):
        raise ValueError("dev8 controller ownership UIDs are absent, reused, or malformed")

    release, release_file_sha256 = _bound_evidence(
        receipt["release_evidence"],
        output_root / "RELEASE.json",
        "cyber_skyrl_engine_diagnostic_release_v1",
    )
    release_fields = {
        "schema",
        "status",
        "cluster",
        "api_base_url",
        "kubernetes_context",
        "namespace",
        "namespace_uid",
        "config_name",
        "plan_sha256",
        "request_sha256",
        "prepared_directory",
        "preflight_file_sha256",
        "submission_journal_file_sha256",
        "diagnostic_receipt_file_sha256",
        "controller_audit_file_sha256",
        "controller_audit_self_sha256",
        "api_run_name",
        "api_job_id",
        "rayjob_name",
        "rayjob_uid",
        "workload_name",
        "workload_uid",
        "raycluster_name",
        "raycluster_uid",
        "pod_uids",
        "runtime_image_ids",
        "runtime_uids",
        "runtime_gids",
        "effective_security_contexts",
        "pod_terminal_observations",
        "api_status",
        "controller_status",
        "effective_priority",
        "automatic_requeue",
        "workers",
        "gpus_per_worker",
        "total_gpus",
        "container_restarts",
        "raycluster_present",
        "gpu_pods_present",
        "active_gpus",
        "gpu_release_proven",
        "observed_at",
        "sha256",
    }
    if (
        set(release) != release_fields
        or release.get("status") != "released"
        or release.get("cluster") != "dev"
        or release.get("api_base_url") != API_URLS["dev"]
        or release.get("kubernetes_context") != DEV_KUBERNETES_CONTEXT
        or release.get("namespace") != DEV_KUBERNETES_NAMESPACE
        or release.get("namespace_uid") != DEV_KUBERNETES_NAMESPACE_UID
        or release.get("config_name") != ENGINE_DIAGNOSTIC_CONFIG_NAME
        or release.get("plan_sha256") != prepared["plan_sha256"]
        or release.get("request_sha256") != prepared["request_sha256"]
        or release.get("prepared_directory") != prepared["directory"]
        or release.get("preflight_file_sha256") != prepared["preflight_file_sha256"]
        or release.get("submission_journal_file_sha256")
        != prepared["submission_journal_file_sha256"]
        or release.get("diagnostic_receipt_file_sha256") != native_file_sha256
        or release.get("controller_audit_file_sha256") != audit_file_sha256
        or release.get("controller_audit_self_sha256") != audit.get("sha256")
        or any(
            release.get(key) != audit.get(key)
            for key in (
                "api_run_name",
                "api_job_id",
                "rayjob_name",
                "rayjob_uid",
                "workload_name",
                "workload_uid",
                "raycluster_name",
                "raycluster_uid",
                "effective_priority",
                "automatic_requeue",
                "workers",
                "gpus_per_worker",
                "total_gpus",
            )
        )
        or release.get("pod_uids") != [pod["uid"] for pod in pods]
        or release.get("runtime_image_ids") != [pod["runtime_image_id"] for pod in pods]
        or release.get("runtime_uids") != [pod["runtime_uid"] for pod in pods]
        or release.get("runtime_gids") != [pod["runtime_gid"] for pod in pods]
        or release.get("effective_security_contexts")
        != [pod["effective_security_context"] for pod in pods]
        or release.get("pod_terminal_observations")
        != [
            {
                key: pod[key]
                for key in (
                    "uid",
                    "phase",
                    "container_exit_code",
                    "termination_reason",
                    "terminated_at",
                )
            }
            for pod in pods
        ]
        or release.get("api_status") != "SUCCEEDED"
        or release.get("controller_status") != "SUCCEEDED"
        or not _exact_integer(release.get("effective_priority"), 10000)
        or release.get("automatic_requeue") is not False
        or not all(
            _exact_integer(release.get(key), expected)
            for key, expected in (
                ("workers", ENGINE_DIAGNOSTIC_WORKERS),
                ("gpus_per_worker", ENGINE_DIAGNOSTIC_GPUS_PER_WORKER),
                ("total_gpus", 8),
            )
        )
        or not _exact_integer(release.get("container_restarts"), 0)
        or release.get("raycluster_present") is not False
        or release.get("gpu_pods_present") is not False
        or not _exact_integer(release.get("active_gpus"), 0)
        or release.get("gpu_release_proven") is not True
    ):
        raise ValueError("dev8 release evidence is not exact, terminal, and UID-bound")
    if _observed_epoch(audit.get("observed_at")) > _observed_epoch(
        release.get("observed_at")
    ) or float(native["completed_at"]) > _observed_epoch(release.get("observed_at")):
        raise ValueError("dev8 GPU release observation predates terminal evidence")
    return {
        "config_file_sha256": config_file_sha256,
        "plan_sha256": prepared["plan_sha256"],
        "request_sha256": prepared["request_sha256"],
        "image_sha256": image_sha256,
        "namespace_uid": DEV_KUBERNETES_NAMESPACE_UID,
        "api_run_name": audit["api_run_name"],
        "api_job_id": audit["api_job_id"],
        "rayjob_uid": audit["rayjob_uid"],
        "workload_uid": audit["workload_uid"],
        "raycluster_uid": audit["raycluster_uid"],
        "pod_uids": [pod["uid"] for pod in pods],
        "runtime_user": {"uid": 1000, "gid": 100},
        "release_file_sha256": release_file_sha256,
    }


def _compile_exact_engine_diagnostic(config, *, relative_to):
    plan = compile_rl(config, relative_to=relative_to)
    return plan, engine_diagnostic_request(plan)


def _accepted_engine_diagnostic(prerequisites, config, *, relative_to):
    """Bind a successful zero-work dev diagnostic before a real reward run."""
    if not isinstance(prerequisites, dict) or set(prerequisites) != {"engine_diagnostic"}:
        raise ValueError("exact accepted engine diagnostic prerequisite required")
    gate = prerequisites["engine_diagnostic"]
    fields = {
        "config_path",
        "config_file_sha256",
        "terminal_receipt_path",
        "terminal_receipt_file_sha256",
    }
    if not isinstance(gate, dict) or set(gate) != fields:
        raise ValueError("engine diagnostic prerequisite schema changed")
    if (
        gate.get("config_path") == ENGINE_DIAGNOSTIC_CONFIG_PATH
        or gate.get("config_file_sha256") == ENGINE_DIAGNOSTIC_CONFIG_FILE_SHA256
    ):
        raise ValueError(
            "terminal dev8 is historical and privacy-disqualified; an exact accepted "
            "dev11 terminal receipt is required"
        )
    if (
        gate.get("config_path") == ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_PATH
        or gate.get("config_file_sha256") == ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_FILE_SHA256
    ):
        raise ValueError("terminal dev9 engine diagnostic cannot be promoted")
    if (
        gate.get("config_path") == ENGINE_DIAGNOSTIC_REJECTED_CONFIG_PATH
        or gate.get("config_file_sha256") == ENGINE_DIAGNOSTIC_REJECTED_CONFIG_FILE_SHA256
    ):
        raise ValueError("terminal rejected dev10 engine diagnostic cannot be promoted")
    if (
        gate.get("config_path") != ENGINE_DIAGNOSTIC_CURRENT_CONFIG_PATH
        or gate.get("config_file_sha256") != ENGINE_DIAGNOSTIC_CURRENT_CONFIG_FILE_SHA256
    ):
        raise ValueError("exact accepted dev11 engine diagnostic prerequisite is not bound")
    diagnostic_path = relative_to / gate["config_path"]
    if not diagnostic_path.is_file() or diagnostic_path.is_symlink():
        raise ValueError("exact engine diagnostic config is unavailable")
    diagnostic_payload, diagnostic_file_sha256 = _snapshot(diagnostic_path)
    if diagnostic_file_sha256 != gate.get("config_file_sha256"):
        raise ValueError("engine diagnostic config digest mismatch")
    diagnostic = json.loads(diagnostic_payload)
    recipe = diagnostic.get("recipe", {})
    if (
        diagnostic.get("backend") != "skyrl"
        or diagnostic.get("name") != ENGINE_DIAGNOSTIC_CURRENT_CONFIG_NAME
        or diagnostic.get("output_root") != ENGINE_DIAGNOSTIC_CURRENT_OUTPUT_ROOT
        or diagnostic.get("model") != config.get("model")
        or diagnostic.get("cluster", {}).get("target") != "dev"
        or diagnostic.get("cluster", {}).get("priority") != "c1"
        or {key: recipe.get(key) for key in ("nodes", "steps", "groups", "samples_per_prompt")}
        != {"nodes": 1, "steps": 1, "groups": 2, "samples_per_prompt": 4}
        or recipe.get("engine_start_timeout_seconds", 1800) != 1800
        or recipe.get("engine_cleanup_timeout_seconds", 300) != 300
    ):
        raise ValueError("engine diagnostic is not the exact dev11 zero-work shape")

    receipt_name, receipt_sha = (
        gate.get("terminal_receipt_path"),
        gate.get("terminal_receipt_file_sha256"),
    )
    if not isinstance(receipt_name, str) or not isinstance(receipt_sha, str):
        raise ValueError("exact accepted dev11 terminal receipt is not bound")
    raise ValueError("exact dev11 terminal acceptance bindings are not frozen yet")


def _validate_embedded_engine_prerequisite(proof, *, required=False):
    if proof is None:
        if required:
            raise ValueError("exact accepted engine diagnostic prerequisite required")
        return
    if isinstance(proof, dict) and (
        proof.get("config_path") == ENGINE_DIAGNOSTIC_CONFIG_PATH
        or proof.get("config_file_sha256") == ENGINE_DIAGNOSTIC_CONFIG_FILE_SHA256
    ):
        raise ValueError(
            "terminal dev8 prerequisite is historical and privacy-disqualified; "
            "an exact accepted dev11 terminal receipt is required"
        )
    if isinstance(proof, dict) and (
        proof.get("config_path") == ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_PATH
        or proof.get("config_file_sha256") == ENGINE_DIAGNOSTIC_SUCCESSOR_CONFIG_FILE_SHA256
    ):
        raise ValueError("terminal dev9 engine prerequisite cannot be replayed")
    if isinstance(proof, dict) and (
        proof.get("config_path") == ENGINE_DIAGNOSTIC_REJECTED_CONFIG_PATH
        or proof.get("config_file_sha256") == ENGINE_DIAGNOSTIC_REJECTED_CONFIG_FILE_SHA256
    ):
        raise ValueError("terminal rejected dev10 engine prerequisite cannot be replayed")
    fields = {
        "schema",
        "status",
        "config_path",
        "config_file_sha256",
        "terminal_receipt_path",
        "terminal_receipt_file_sha256",
        "terminal_receipt_self_sha256",
        "diagnostic_plan_sha256",
        "diagnostic_request_sha256",
        "image_sha256",
        "workers",
        "gpus_per_worker",
        "kubernetes_context",
        "namespace",
        "namespace_uid",
        "api_run_name",
        "api_job_id",
        "rayjob_uid",
        "workload_uid",
        "raycluster_uid",
        "pod_uids",
        "runtime_user",
        "release_file_sha256",
        "gpu_release_proven",
        "terminal_receipt",
        "sha256",
    }
    if (
        not isinstance(proof, dict)
        or set(proof) != fields
        or proof.get("schema") != "cyber_skyrl_engine_prerequisite_v1"
        or proof.get("status") != "accepted"
        or proof.get("config_path") != ENGINE_DIAGNOSTIC_CONFIG_PATH
        or proof.get("config_file_sha256") != ENGINE_DIAGNOSTIC_CONFIG_FILE_SHA256
        or proof.get("image_sha256") != IMAGE.rsplit("@sha256:", 1)[1]
        or proof.get("workers") != ENGINE_DIAGNOSTIC_WORKERS
        or proof.get("gpus_per_worker") != ENGINE_DIAGNOSTIC_GPUS_PER_WORKER
        or proof.get("kubernetes_context") != DEV_KUBERNETES_CONTEXT
        or proof.get("namespace") != DEV_KUBERNETES_NAMESPACE
        or proof.get("namespace_uid") != DEV_KUBERNETES_NAMESPACE_UID
        or proof.get("api_run_name") != ENGINE_DIAGNOSTIC_API_RUN_NAME
        or proof.get("api_job_id") != ENGINE_DIAGNOSTIC_API_JOB_ID
        or proof.get("rayjob_uid") != ENGINE_DIAGNOSTIC_RAYJOB_UID
        or proof.get("workload_uid") != ENGINE_DIAGNOSTIC_WORKLOAD_UID
        or proof.get("raycluster_uid") != ENGINE_DIAGNOSTIC_RAYCLUSTER_UID
        or proof.get("runtime_user") != {"uid": 1000, "gid": 100}
        or proof.get("gpu_release_proven") is not True
        or proof.get("sha256")
        != "sha256:" + digest({key: value for key, value in proof.items() if key != "sha256"})
        or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", str(proof.get(key, "")))
            for key in ("terminal_receipt_file_sha256", "release_file_sha256", "sha256")
        )
        or any(
            not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", str(proof.get(key, "")))
            for key in (
                "terminal_receipt_self_sha256",
                "diagnostic_plan_sha256",
                "diagnostic_request_sha256",
            )
        )
        or any(
            not isinstance(proof.get(key), str) or not proof[key]
            for key in (
                "config_path",
                "config_file_sha256",
                "terminal_receipt_path",
                "terminal_receipt_file_sha256",
                "terminal_receipt_self_sha256",
                "diagnostic_plan_sha256",
                "diagnostic_request_sha256",
            )
        )
        or any(
            _UUID.fullmatch(str(proof.get(key, ""))) is None
            for key in (
                "namespace_uid",
                "rayjob_uid",
                "workload_uid",
                "raycluster_uid",
            )
        )
        or not isinstance(proof.get("pod_uids"), list)
        or proof["pod_uids"] != [uid for _, uid in ENGINE_DIAGNOSTIC_PODS]
        or any(_UUID.fullmatch(str(item)) is None for item in proof["pod_uids"])
    ):
        raise ValueError("embedded engine diagnostic prerequisite changed")
    immutable = _validate_engine_diagnostic_receipt(
        proof["terminal_receipt"],
        config_file_sha256=proof["config_file_sha256"],
        expected_plan_sha256=proof["diagnostic_plan_sha256"],
        expected_request_sha256=proof["diagnostic_request_sha256"],
    )
    if (
        proof["terminal_receipt_self_sha256"] != proof["terminal_receipt"]["sha256"]
        or proof["image_sha256"] != immutable["image_sha256"]
        or proof["diagnostic_plan_sha256"] != immutable["plan_sha256"]
        or proof["diagnostic_request_sha256"] != immutable["request_sha256"]
        or proof["namespace_uid"] != immutable["namespace_uid"]
        or proof["api_run_name"] != immutable["api_run_name"]
        or proof["api_job_id"] != immutable["api_job_id"]
        or proof["rayjob_uid"] != immutable["rayjob_uid"]
        or proof["workload_uid"] != immutable["workload_uid"]
        or proof["raycluster_uid"] != immutable["raycluster_uid"]
        or proof["pod_uids"] != immutable["pod_uids"]
        or proof["runtime_user"] != immutable["runtime_user"]
        or proof["release_file_sha256"] != immutable["release_file_sha256"]
        or proof["terminal_receipt_file_sha256"]
        != "sha256:" + hashlib.sha256(_canonical_json(proof["terminal_receipt"])).hexdigest()
    ):
        raise ValueError("embedded engine diagnostic prerequisite changed")


def _validate_reward_canary_data(metadata) -> bool:
    """Recognize the canary by immutable selection identity, then require all bindings."""
    expected = REWARD_CANARY_DATA_CONTRACT
    if not isinstance(metadata, dict):
        return False
    if metadata.get("selection_sha256") != expected["selection_sha256"]:
        if metadata.get("name") == REWARD_CANARY_ARGUMENTS["name"]:
            raise ValueError("reward canary name is bound to a different immutable selection")
        return False
    files = metadata.get("files", {})
    if (
        metadata.get("split_sha256") != expected["split_sha256"]
        or metadata.get("tool_catalog_sha256") != expected["tool_catalog_sha256"]
        or metadata.get("limits") != expected["limits"]
        or set(files) != set(expected["rows"])
        or any(
            not isinstance(files.get(split), dict)
            or files[split].get("path") != f"{split}.jsonl"
            or files[split].get("rows") != rows
            for split, rows in expected["rows"].items()
        )
    ):
        raise ValueError("reward canary data differs from its exact reviewed contract")
    return True


def is_reward_canary(plan: object) -> bool:
    """Recognize and validate the exact one-update dev reward-canary plan."""
    if not isinstance(plan, dict) or not _validate_reward_canary_data(plan.get("data")):
        return False
    args = plan.get("arguments")
    execution = plan.get("execution", {})
    model = plan.get("model", {})
    fixed_execution = {
        "image": IMAGE,
        "image_cpu_qualification": ENGINE_IMAGE_CPU_QUALIFICATION,
        "runtime_user": REWARD_CANARY_RUNTIME_USER,
        "cluster_target": "dev",
        "priority": "c1",
        "resources": REWARD_CANARY_RESOURCES,
    }
    if (
        args != REWARD_CANARY_ARGUMENTS
        or plan.get("run_name") != REWARD_CANARY_ARGUMENTS["name"]
        or plan.get("output_root") != REWARD_CANARY_ARGUMENTS["output_root"]
        or digest(model) != ENGINE_DIAGNOSTIC_MODEL_SHA256
        or any(model.get(key) != value for key, value in REWARD_CANARY_MODEL_IDENTITY.items())
        or not isinstance(model.get("files"), list)
        or len(model["files"]) != 28
        or plan.get("native_overrides")
        != skyrl.overrides(skyrl.SkyRLConfig(**REWARD_CANARY_ARGUMENTS))
        or set(execution)
        != {
            *fixed_execution,
            "reward_canary_source",
            "engine_diagnostic_prerequisite",
        }
        or any(execution.get(key) != value for key, value in fixed_execution.items())
        or not isinstance(execution.get("reward_canary_source"), dict)
        or not isinstance(execution.get("engine_diagnostic_prerequisite"), dict)
    ):
        raise ValueError("reward canary differs from the complete exact reviewed recipe")
    return True


def _require_reward_canary_runtime_identity(plan: dict) -> dict:
    """Fail inside the allocated GPU Pod unless its real process is 1000:100."""
    if not is_reward_canary(plan):
        return {}
    expected = REWARD_CANARY_RUNTIME_USER
    if (
        os.environ.get("CYBER_EXPECTED_RUNTIME_UID") != str(expected["uid"])
        or os.environ.get("CYBER_EXPECTED_RUNTIME_GID") != str(expected["gid"])
        or (os.geteuid(), os.getegid()) != (expected["uid"], expected["gid"])
    ):
        raise ValueError("reward canary GPU runtime must be the pinned non-root user 1000:100")
    return {"uid": os.geteuid(), "gid": os.getegid()}


def compile_rl(config, *, relative_to):
    from .models import bound_model
    from .sft import RESOURCES, _known, _sfs_root, read_mapping

    _known(
        config,
        {
            "backend",
            "name",
            "output_root",
            "model",
            "data",
            "prerequisites",
            "recipe",
            "wandb",
            "cluster",
            "production_promotion",
        },
        "RL",
    )
    if config["backend"] != "skyrl":
        raise ValueError("no silent RL backend substitution")
    model, data, w = (config[k] for k in ("model", "data", "wandb"))
    cluster, recipe = config.get("cluster", {}), config.get("recipe", {})
    _known(model, {"lock", "weights", "root"}, "model")
    _known(data, {"manifest", "root"}, "data")
    _known(w, {"entity", "project", "run_id"}, "W&B")
    _known(cluster, {"target", "priority", "resources"}, "cluster")
    _known(
        recipe,
        {
            "nodes",
            "steps",
            "groups",
            "samples_per_prompt",
            "lr",
            "eval_interval",
            "checkpoint_interval",
            "keep_checkpoints",
            "seed",
            "engine_start_timeout_seconds",
            "engine_cleanup_timeout_seconds",
        },
        "SkyRL recipe",
    )
    target = cluster.get("target")
    if target not in {None, "dev", "prod"}:
        raise ValueError("cluster target must be dev or prod")
    from .skyrl_promotion import bind_production_promotion

    production_promotion = (
        None
        if is_prod_engine_diagnostic_config(config)
        else bind_production_promotion(config, relative_to)
    )
    prerequisite = None
    if "prerequisites" in config:
        if target != "dev":
            raise ValueError("engine-qualified reward canary is dev-cluster-only")
        prerequisite = _accepted_engine_diagnostic(
            config["prerequisites"], config, relative_to=relative_to
        )
    bound = bound_model(
        read_mapping(relative_to / model["lock"]),
        read_mapping(relative_to / model["weights"]),
        _sfs_root(model["root"], "model root"),
    )
    metadata = read_mapping(relative_to / data["manifest"])
    sealed(metadata, "cyber_skyrl_data_v1")
    reward_canary = _validate_reward_canary_data(metadata)
    reward_source_proof = None
    if reward_canary:
        if prerequisite is None:
            raise ValueError("exact accepted engine diagnostic prerequisite required")
        from .rl_reward_canary import validate_repository_source

        reward_source_proof = validate_repository_source(metadata)
    elif prerequisite is not None:
        raise ValueError("engine diagnostic prerequisite is only valid for exact canary data")
    if set(metadata["files"]) != {"train", "dev"} or any(
        item["path"] != split + ".jsonl" for split, item in metadata["files"].items()
    ):
        raise ValueError("exact native train/dev files required")
    root, limits = Path(_sfs_root(data["root"], "data root")), metadata["limits"]
    args = skyrl.SkyRLConfig(
        name=config["name"],
        output_root=_sfs_root(config["output_root"], "output root"),
        model=bound["repo"],
        model_root=bound["root"],
        train_data=str(root / "train.jsonl"),
        dev_data=str(root / "dev.jsonl"),
        data_manifest=str(root / "manifest.json"),
        train_rows=metadata["files"]["train"]["rows"],
        dev_rows=metadata["files"]["dev"]["rows"],
        wandb_entity=w["entity"],
        wandb_project=w["project"],
        wandb_run_id=w["run_id"],
        context_tokens=limits["context_tokens"],
        response_tokens=limits["response_tokens"],
        tokens_per_turn=limits["max_tokens_per_turn"],
        max_turns=limits["max_turns"],
        **recipe,
    )
    if metadata["name"] != args.name or any(
        metadata["tokenizer"][k] != bound[k] for k in ("repo", "revision")
    ):
        raise ValueError("run/model/data identity mismatch")
    plan = {
        "schema": SCHEMA,
        "run_name": args.name,
        "output_root": args.output_root,
        "model": bound,
        "data": metadata,
        "arguments": dataclasses.asdict(args),
        "native_overrides": skyrl.overrides(args),
        "native_sources": NATIVE,
        "runtime_sha256": digest(_runtime()),
        "execution": {
            "image": IMAGE,
            "image_cpu_qualification": dict(ENGINE_IMAGE_CPU_QUALIFICATION),
            **(
                {"reward_canary_source": reward_source_proof}
                if reward_source_proof is not None
                else {}
            ),
            **(
                {"runtime_user": dict(REWARD_CANARY_RUNTIME_USER)}
                if reward_source_proof is not None or production_promotion is not None
                else {}
            ),
            **(
                {"production_promotion": production_promotion}
                if production_promotion is not None
                else {}
            ),
            **(
                {"engine_diagnostic_prerequisite": prerequisite} if prerequisite is not None else {}
            ),
            **({"cluster_target": target} if target is not None else {}),
            "priority": cluster.get("priority", "c1"),
            "resources": {**RESOURCES, **cluster.get("resources", {})},
        },
    }
    job_request(plan)
    return plan


def job_request(plan):
    from .skyrl_promotion import validate_embedded_promotion

    _require_fresh_engine_diagnostic_identity(plan)
    args = skyrl.SkyRLConfig(**plan["arguments"])
    sealed(plan.get("data", {}), "cyber_skyrl_data_v1")
    reward_canary = _validate_reward_canary_data(plan.get("data"))
    _validate_embedded_engine_prerequisite(
        plan.get("execution", {}).get("engine_diagnostic_prerequisite"),
        required=reward_canary,
    )
    source_proof = plan.get("execution", {}).get("reward_canary_source")
    if reward_canary:
        from .rl_reward_canary import validate_source_proof

        validate_source_proof(source_proof, plan["data"])
    elif source_proof is not None:
        raise ValueError("reward source closure is only valid for exact canary data")
    if reward_canary:
        is_reward_canary(plan)
    production = None if is_prod_engine_diagnostic(plan) else validate_embedded_promotion(plan)
    if (
        plan["schema"] != SCHEMA
        or plan["runtime_sha256"] != digest(_runtime())
        or plan["native_sources"] != NATIVE
        or plan["native_overrides"] != skyrl.overrides(args)
        or plan["execution"]["image"] != IMAGE
        or plan["execution"].get("image_cpu_qualification") != ENGINE_IMAGE_CPU_QUALIFICATION
        or plan["run_name"] != args.name
        or plan["output_root"] != args.output_root
        or plan["execution"].get("cluster_target") not in {None, "dev", "prod"}
    ):
        raise ValueError("SkyRL plan/runtime drift")
    resources = plan["execution"]["resources"]
    if quantity(resources["cpu_request"]) < 64 or quantity(resources["memory_request"]) < quantity(
        "512Gi"
    ):
        raise ValueError("native Qwen RL requires its reviewed loading reservation")
    files = _runtime()
    files.update(
        {p + "/__init__.py": "" for p in ("training", "evals", "evals/fleet", "cyber_post_train")}
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        **VLLM_SAMPLER_ENV,
        "WANDB_MODE": "online",
        "WANDB_RUN_ID": args.wandb_run_id,
        "WANDB_DISABLE_CODE": "true",
        "WANDB_CONSOLE": "off",
        "PYTHONUNBUFFERED": "1",
    }
    if reward_canary or production:
        env.update(
            CYBER_EXPECTED_RUNTIME_UID=str(REWARD_CANARY_RUNTIME_USER["uid"]),
            CYBER_EXPECTED_RUNTIME_GID=str(REWARD_CANARY_RUNTIME_USER["gid"]),
        )
    return bundled_request(
        {
            "name": args.name,
            "title": args.name + " native SkyRL RL",
            "run_dir": args.output_root,
            "image": IMAGE,
            "workers": args.nodes,
            "gpus_per_worker": 8,
            "resources": resources,
            "priority_class": plan["execution"]["priority"],
            "requeueIfPreempted": False,
            "secrets": ["fleet-api", "wandb-api"],
            "env": env,
        },
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def validate_reward_canary_preview(plan: dict, request: dict, preview: dict) -> dict:
    """Require every rendered canary Pod to declare the effective 1000:100 user."""
    if not is_reward_canary(plan):
        return {}
    import yaml

    if request != job_request(plan):
        raise JobsError("reward canary preview request differs from its exact plan")
    try:
        obj = yaml.safe_load(preview["manifest_yaml"])
        cluster = obj["spec"]["rayClusterSpec"]
        groups = [(1, cluster["headGroupSpec"]["template"])] + [
            (group["replicas"], group["template"]) for group in cluster.get("workerGroupSpecs", [])
        ]
        pods = 0
        for replicas, template in groups:
            if type(replicas) is not int or replicas < 0:
                raise JobsError("invalid reward canary preview replica count")
            if replicas == 0:
                continue
            pod = template["spec"]
            containers = pod["containers"]
            if not isinstance(containers, list) or len(containers) != 1:
                raise JobsError("reward canary preview must have one container per Pod")
            pod_context = pod.get("securityContext", {})
            container_context = containers[0].get("securityContext", {})
            effective = {
                key: container_context.get(key, pod_context.get(key))
                for key in ("runAsUser", "runAsGroup", "runAsNonRoot")
            }
            if effective != {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}:
                raise JobsError("reward canary preview runtime user differs from 1000:100")
            pods += replicas
        if (
            obj["metadata"]["namespace"] != DEV_KUBERNETES_NAMESPACE
            or pods != 1
            or request.get("workers") != 1
            or request.get("gpus_per_worker") != 8
            or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_UID") != "1000"
            or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_GID") != "100"
        ):
            raise JobsError("reward canary preview topology or runtime binding changed")
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise JobsError("malformed reward canary Jobs API preview") from error
    return {"runtime_user": {"uid": 1000, "gid": 100}, "pods": pods}


def _diagnostic_shape(plan, cfg=None):
    """Bind the fragmented-dev diagnostic shape, never the training shape."""
    overrides = plan["native_overrides"]
    if (
        plan["arguments"]["nodes"] != 1
        or overrides.get("generator.inference_engine.num_engines") != 2
        or overrides.get("generator.inference_engine.tensor_parallel_size") != 4
    ):
        raise ValueError("engine diagnostic requires the exact one-node 2xTP4 engine plan")
    if cfg is not None:
        engine = cfg.generator.inference_engine
        if (
            engine.num_engines != 2
            or engine.tensor_parallel_size != 4
            or engine.pipeline_parallel_size != 1
            or engine.data_parallel_size != 1
        ):
            raise ValueError("native engine shape differs from diagnostic cluster binding")
    return {
        "diagnostic_workers": ENGINE_DIAGNOSTIC_WORKERS,
        "diagnostic_gpus_per_worker": ENGINE_DIAGNOSTIC_GPUS_PER_WORKER,
        "diagnostic_total_gpus": (ENGINE_DIAGNOSTIC_WORKERS * ENGINE_DIAGNOSTIC_GPUS_PER_WORKER),
        "num_engines": 2,
        "tensor_parallel_size": 4,
    }


def engine_diagnostic_request(plan):
    """Build a no-rollout, no-secret engine-start diagnostic for a fresh plan.

    The caller must compile the source configuration under a new diagnostic
    name/output root.  Reusing a failed run directory is intentionally not
    supported.
    """
    job_request(plan)  # exact plan/runtime/image/resource validation
    shape = _diagnostic_shape(plan)
    args = skyrl.SkyRLConfig(**plan["arguments"])
    resources = plan["execution"]["resources"]
    files = _runtime()
    files.update(
        {p + "/__init__.py": "" for p in ("training", "evals", "evals/fleet", "cyber_post_train")}
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": args.name,
            "title": args.name + " SkyRL engine-start diagnostic",
            "run_dir": args.output_root,
            "image": IMAGE,
            "workers": shape["diagnostic_workers"],
            "gpus_per_worker": shape["diagnostic_gpus_per_worker"],
            "resources": resources,
            "priority_class": plan["execution"]["priority"],
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                **VLLM_SAMPLER_ENV,
                "CYBER_EXPECTED_RUNTIME_UID": str(REWARD_CANARY_RUNTIME_USER["uid"]),
                "CYBER_EXPECTED_RUNTIME_GID": str(REWARD_CANARY_RUNTIME_USER["gid"]),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan), "--engine-diagnostic"],
    )


def validate_engine_diagnostic_preview(plan: dict, request: dict, preview: dict) -> dict:
    """Require every rendered diagnostic Pod to be the exact non-root GPU identity."""
    import yaml

    if request != engine_diagnostic_request(plan):
        raise JobsError("engine diagnostic preview request differs from its exact plan")
    target = plan.get("execution", {}).get("cluster_target")
    if target == "prod" and not is_prod_engine_diagnostic(plan):
        raise JobsError("production engine diagnostic is not the exact authorized fallback")
    expected_namespace = (
        PROD_KUBERNETES_NAMESPACE if target == "prod" else DEV_KUBERNETES_NAMESPACE
    )
    try:
        obj = yaml.safe_load(preview["manifest_yaml"])
        cluster = obj["spec"]["rayClusterSpec"]
        groups = [(1, cluster["headGroupSpec"]["template"])] + [
            (group["replicas"], group["template"]) for group in cluster.get("workerGroupSpecs", [])
        ]
        pods = 0
        for replicas, template in groups:
            if type(replicas) is not int or replicas < 0:
                raise JobsError("invalid engine diagnostic preview replica count")
            if replicas == 0:
                continue
            pod = template["spec"]
            containers = pod["containers"]
            if not isinstance(containers, list) or len(containers) != 1:
                raise JobsError("engine diagnostic preview must have one container per Pod")
            pod_context = pod.get("securityContext", {})
            container_context = containers[0].get("securityContext", {})
            effective = {
                key: container_context.get(key, pod_context.get(key))
                for key in ("runAsUser", "runAsGroup", "runAsNonRoot")
            }
            if effective != {"runAsUser": 1000, "runAsGroup": 100, "runAsNonRoot": True}:
                raise JobsError("engine diagnostic preview runtime user differs from 1000:100")
            pods += replicas
        if (
            obj["metadata"]["namespace"] != expected_namespace
            or pods != ENGINE_DIAGNOSTIC_WORKERS
            or request.get("workers") != ENGINE_DIAGNOSTIC_WORKERS
            or request.get("gpus_per_worker") != ENGINE_DIAGNOSTIC_GPUS_PER_WORKER
            or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_UID") != "1000"
            or request.get("env", {}).get("CYBER_EXPECTED_RUNTIME_GID") != "100"
        ):
            raise JobsError("engine diagnostic preview topology or runtime binding changed")
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise JobsError("malformed engine diagnostic Jobs API preview") from error
    return {"runtime_user": {"uid": 1000, "gid": 100}, "pods": pods}


def check_artifacts(plan):
    """Hash payloads and reconcile exact task families without exposing task text."""
    from .rl_data import selection
    from .rl_episode import _validate

    check_inputs(plan)
    data, args = plan["data"], plan["arguments"]
    root = Path(args["data_manifest"]).parent
    if json.loads((root / "manifest.json").read_bytes()) != data:
        raise ValueError("staged data manifest changed")
    split, task_set = (json.loads((root / p).read_bytes()) for p in ("split.json", "task-set.json"))
    if split["sha256"] != data["split_sha256"] or task_set["sha256"] != data["selection_sha256"]:
        raise ValueError("reviewed task-set/split changed")
    expected = {
        (r["split"], r["task_key"], r["task_version_id"]) for r in selection(task_set, split)
    }
    rows, selected = {}, set()
    for group, item in data["files"].items():
        path = root / item["path"]
        if _hash(path) != item["sha256"].removeprefix("sha256:"):
            raise ValueError("RL data payload changed")
        rows[group] = [json.loads(line) for line in path.read_bytes().splitlines()]
        if len(rows[group]) != item["rows"] or not rows[group]:
            raise ValueError("RL data count changed")
        for row in rows[group]:
            cfg = json.loads(row["cyber_config_json"])
            _validate(cfg)
            key = (group, cfg["task"]["key"], cfg["task"]["version_id"])
            if (
                key in selected
                or row["split"] != group
                or row["env_class"] != cfg["environment"]["id"]
                or cfg["run_id"] != args["name"]
                or len(row["prompt"]) != 1
                or row["prompt"][0]["role"] != "user"
                or "sha256:" + hashlib.sha256(row["prompt"][0]["content"].encode()).hexdigest()
                != cfg["task"]["prompt_sha256"]
                or any(cfg["model"][k] != plan["model"][k] for k in ("repo", "revision", "root"))
                or cfg["model"]["runtime_chat_template_sha256"] != data["template_sha256"]
                or cfg["rl"] != {k: v for k, v in data["limits"].items() if k != "response_tokens"}
                or cfg["execution"]["required_task_tool_catalog_sha256"]
                != data["tool_catalog_sha256"]
            ):
                raise ValueError("RL episode identity changed")
            selected.add(key)
    if selected != expected:
        raise ValueError("selected RL task families differ from frozen split")
    return rows


def native_source():
    from .sft_runtime import validate_runtime_sources

    validate_runtime_sources()
    return {name: _module(name, sha) for name, sha in NATIVE.items()}


def dataset(plan, tokenizer, split, rows):
    native = _module("skyrl.train.dataset.dataset", NATIVE["skyrl.train.dataset.dataset"])
    value = native.PromptDataset(
        plan["arguments"][split + "_data"],
        tokenizer,
        plan["native_overrides"]["trainer.max_prompt_length"],
        num_workers=1,
    )
    if len(value) != len(rows) or any(
        value[i]
        != (
            row["prompt"],
            row["env_class"],
            {k: v for k, v in row.items() if k not in {"prompt", "env_class"}},
            str(i),
        )
        for i, row in enumerate(rows)
    ):
        raise ValueError("native SkyRL silently filtered or changed a task")
    return value


def _require_engine_start_qualified_image(plan):
    """Reject exact model/image pairs disproven by terminal or privacy evidence."""
    identity = (plan["model"]["repo"], plan["execution"]["image"])
    if identity in _ENGINE_START_DISQUALIFIED:
        raise ValueError(
            "Qwen3.8 SkyRL model/image pair is engine-start disqualified by sealed "
            "terminal dev evidence; qualify and pin a replacement image before GPU submission"
        )
    if identity in _ENGINE_PRIVACY_DISQUALIFIED:
        raise ValueError(
            "Qwen3.8 SkyRL model/image pair is privacy-disqualified by sealed worker-RPC "
            "evidence; qualify and pin a sanitized replacement image before GPU submission"
        )


def _validate_vllm_startup_error_transport():
    """Prove the installed safe startup wrapper crosses Ray's pickle boundary.

    vLLM raises this exception inside a Ray actor.  Ray 2.56 checks only that a
    cause can be serialized before placing it in ``RayTaskError``; the driver
    can still lose the cause later if reconstructing the exception fails.  A
    GPU engine smoke cannot be the first place that constructor mismatch is
    discovered.
    """
    import pickle

    ray_errors = importlib.import_module("ray.exceptions")
    ray_pickle = importlib.import_module("ray.cloudpickle")
    startup = importlib.import_module("vllm.v1.engine.fleet_startup_error")
    error_type = getattr(startup, "FleetVllmStartupError", None)
    schema = getattr(startup, "SCHEMA", None)
    if not isinstance(error_type, type) or schema != "fleet_vllm_startup_error_v1":
        raise ValueError("installed vLLM startup exception contract changed")

    expected = {
        "schema": schema,
        "exception_class": "TypeError",
        "frame": {"source_id": "model", "line": 1},
    }
    secret = "SYNTHETIC_PRIVATE_STARTUP_TRANSPORT_PROBE"
    poisoned = {
        "schema": schema,
        "exception_class": secret,
        "frame": {"source_id": secret, "line": secret},
        "message": secret,
    }
    probes = (
        ("engine_core", expected, expected),
        (
            secret,
            poisoned,
            {"schema": schema, "exception_class": "Other", "frame": None},
        ),
    )
    for codec_name, codec in (("pickle", pickle), ("ray.cloudpickle", ray_pickle)):
        for stage, cause, normalized in probes:
            error = error_type(stage, cause)
            try:
                payload = codec.dumps(error)
                restored = codec.loads(payload)
            except Exception as exc:
                raise ValueError(
                    f"installed vLLM startup exception failed {codec_name} round-trip"
                ) from exc
            if (
                type(restored) is not error_type
                or getattr(restored, "sanitized_cause", None) != normalized
                or secret.encode() in payload
                or secret in str(restored)
                or secret in repr(getattr(restored, "sanitized_cause", None))
            ):
                raise ValueError(
                    f"installed vLLM startup exception failed {codec_name} privacy contract"
                )
    for stage, cause, normalized in probes:
        error = error_type(stage, cause)
        envelope = ray_errors.RayTaskError(
            "engine_start",
            "synthetic safe traceback",
            error,
            proctitle="engine-start-preflight",
            pid=1,
            ip="127.0.0.1",
        )
        try:
            payload = envelope.to_bytes()
            restored = ray_errors.RayError.from_bytes(payload)
        except Exception as exc:
            raise ValueError(
                "installed vLLM startup exception failed RayTaskError round-trip"
            ) from exc
        restored_cause = getattr(restored, "cause", None)
        if (
            isinstance(restored, ray_errors.UnserializableException)
            or type(restored_cause) is not error_type
            or getattr(restored_cause, "sanitized_cause", None) != normalized
            or secret.encode() in payload
            or secret in str(restored_cause)
            or secret in repr(getattr(restored_cause, "sanitized_cause", None))
        ):
            raise ValueError(
                "installed vLLM startup exception failed RayTaskError privacy contract"
            )


def _validate_vllm_sampler_fallback_contract() -> None:
    """Prove the exact image selects vLLM's PyTorch sampler on Blackwell."""
    if any(os.environ.get(key) != value for key, value in VLLM_SAMPLER_ENV.items()):
        raise ValueError("vLLM sampler fallback environment is not exact")
    vllm_envs = importlib.import_module("vllm.envs")
    sampler = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
    if vllm_envs.VLLM_USE_FLASHINFER_SAMPLER is not False:
        raise ValueError("vLLM did not parse the FlashInfer sampler fallback")

    class CudaPlatform:
        @staticmethod
        def is_cuda():
            return True

        @staticmethod
        def is_cpu():
            return False

        @staticmethod
        def is_xpu():
            return False

    original = sampler.current_platform
    try:
        sampler.current_platform = CudaPlatform()
        instance = sampler.TopKTopPSampler()
    finally:
        sampler.current_platform = original
    if instance.forward.__func__ is not instance.forward_native.__func__:
        raise ValueError("vLLM did not select its PyTorch-native sampler fallback")


def preflight(plan):
    # The pinned GPU image is UID 1000/GID 100. Root can read private staging
    # files that its trainer cannot; such a preflight is not representative.
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("SkyRL CPU preflight must use the pinned image user 1000:100, not root")
    _require_engine_start_qualified_image(plan)
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        raise ValueError("SkyRL preflight is CPU-only")
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("RL output already exists")
    rows = check_artifacts(plan)
    native_source()
    _validate_vllm_startup_error_transport()
    _validate_vllm_sampler_fallback_contract()
    skyrl.native_config(skyrl.SkyRLConfig(**plan["arguments"]))
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["root"], trust_remote_code=False, local_files_only=True
    )
    if "sha256:" + digest_template(tokenizer) != plan["data"]["template_sha256"]:
        raise ValueError("native template changed")
    for split in rows:
        dataset(plan, tokenizer, split, rows[split])
    return {
        "schema": "cyber_skyrl_training_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "native_parser_checked": True,
        "startup_error_transport_checked": True,
        "vllm_sampler_fallback_checked": True,
        "counts": {k: len(v) for k, v in rows.items()},
        "planned_steps": plan["arguments"]["steps"],
        "rl_qualified": False,
    }


def engine_diagnostic_preflight(plan):
    """Validate the engine-only request without reading task rows or using GPUs."""
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("SkyRL CPU preflight must use the pinned image user 1000:100, not root")
    _require_engine_start_qualified_image(plan)
    import torch

    if torch.cuda.is_available():
        raise ValueError("SkyRL engine diagnostic preflight is CPU-only")
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args

    request = engine_diagnostic_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("RL diagnostic output already exists")
    check_inputs(plan)
    native_source()
    _validate_vllm_startup_error_transport()
    _validate_vllm_sampler_fallback_contract()
    cfg = skyrl.diagnostic_native_config(skyrl.SkyRLConfig(**plan["arguments"]))
    shape = _diagnostic_shape(plan, cfg)
    build_vllm_cli_args(cfg)
    return {
        "schema": "cyber_skyrl_engine_diagnostic_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "native_parser_checked": True,
        "engine_cli_args_checked": True,
        "startup_error_transport_checked": True,
        "vllm_sampler_fallback_checked": True,
        "model_files": len(plan["model"]["files"]),
        "task_rows_read": 0,
        "rollouts": False,
        "verifier_calls": False,
        "optimizer_updates": False,
        "checkpoints": False,
        "wandb": False,
        "engine_start_qualified": False,
        **shape,
    }


def digest_template(tokenizer):
    return hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()


class ScalarTracking:
    """Native tracker interface with no text/tables and explicit outer finalization."""

    def __init__(self, plan):
        import wandb

        from .sft_runtime import _configure_wandb

        args = plan["arguments"]
        _configure_wandb(
            {
                "output_root": plan["output_root"],
                "wandb": {
                    "entity": args["wandb_entity"],
                    "group": args["name"],
                    "run_id": args["wandb_run_id"],
                },
            }
        )
        self.root, self.logger = Path(plan["output_root"]), wandb
        run = wandb.init(
            entity=args["wandb_entity"],
            project=args["wandb_project"],
            id=args["wandb_run_id"],
            name=args["name"],
            resume="never",
            config={"plan_sha256": digest(plan), "arguments": args},
        )
        if (run.id, run.entity, run.project) != (
            args["wandb_run_id"],
            args["wandb_entity"],
            args["wandb_project"],
        ):
            raise ValueError("W&B run identity mismatch")

    def log(self, data, step, commit=False):
        if (
            type(step) is not int
            or step < 0
            or any(
                not isinstance(k, str)
                or not re.fullmatch(r"[A-Za-z0-9_./@() -]+", k)
                or isinstance(v, bool)
                or not isinstance(v, numbers.Real)
                or not math.isfinite(v)
                for k, v in data.items()
            )
            or "cyber/optimizer_step" in data
        ):
            raise ValueError("native telemetry must contain finite scalars only")
        data = {
            **{k: float(v) for k, v in data.items()},
            "cyber/optimizer_step": float(step),
        }
        with (self.root / "metrics.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"optimizer_step": step, "time": time.time(), **data}, allow_nan=False)
                + "\n"
            )
            stream.flush()
        self.logger.log(data=data, step=step, commit=commit)

    def finish(self):
        pass  # only the outer run knows whether checkpoint/evidence checks passed

    def log_exception(self, error, step=0):
        pass  # native otherwise uploads complete private tracebacks to W&B

    def log_samples_to_table(self, *args, **kwargs):
        raise ValueError("trajectory uploads are disabled")


def _prepare_infra_log(plan):
    """Create the one private shared file used by SkyRL infrastructure actors."""
    directory = Path(plan["output_root"]) / "private-native-logs"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / "infra.log"
    with path.open("x"):
        pass
    os.chmod(path, 0o600)
    return path


def _infra_log_evidence(path):
    """Return metadata only; the infrastructure log always remains private."""
    signature = b"Engine core initialization failed. See root cause above."
    hasher = hashlib.sha256()
    size = 0
    overlap = b""
    signature_present = False
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            hasher.update(chunk)
            window = overlap + chunk
            signature_present = signature_present or signature in window
            overlap = window[-(len(signature) - 1) :]
    return {
        "path": "private-native-logs/infra.log",
        "bytes": size,
        "sha256": hasher.hexdigest(),
        "engine_failure_signature_present": signature_present,
    }


def _string_environment(env: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(env, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
    ):
        raise ValueError("SkyRL Ray environment must contain only string variables")
    return dict(env)


def _credential_names(*environments: Mapping[str, str]) -> tuple[str, ...]:
    names = set(_ALWAYS_SCRUB_WORKER_ENV)
    for environment in environments:
        names.update(
            key for key in _string_environment(environment) if _CREDENTIAL_ENV_NAME.search(key)
        )
    return tuple(sorted(names))


def _scrubbed_actor_environment(environment, scrubbed, overlay=None):
    """Merge an explicit actor environment and blank credentials last."""
    value = _string_environment(environment)
    if overlay is not None:
        value.update(_string_environment(overlay))
    names = set(scrubbed)
    names.update(_credential_names(value))
    value.update(dict.fromkeys(sorted(names), ""))
    return value


@contextmanager
def _scrubbed_process_environment(names):
    """Temporarily blank credentials while a router child is launched."""
    missing = object()
    previous = {name: os.environ.get(name, missing) for name in names}
    try:
        for name in names:
            os.environ[name] = ""
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _DiagnosticContractRejected(ValueError):
    """An explicitly recognized diagnostic incompatibility, not a runtime defect."""


def _router_child_entry(target, names, probe_state, start_method, args, kwargs):
    """Picklable entrypoint: prove isolation before calling the unchanged target.

    The original native target must remain its module's global function so that
    Python's spawn pickler can resolve it.  No credentials or environment values
    are copied into the child arguments or the shared, integer-only probe.
    """
    import multiprocessing

    if multiprocessing.get_start_method() != start_method:
        probe_state.value = -2
        raise RuntimeError("router child process context changed")
    if any(os.environ.get(name) for name in set(names) | set(_credential_names(os.environ))):
        probe_state.value = -1
        raise RuntimeError("router child credential isolation failed")
    probe_state.value = 1
    return target(*args, **kwargs)


class _RouterMultiprocessing:
    """Intercept only this native router module's Process constructor."""

    def __init__(self, native, target, names, probe_state, start_method):
        self._native = native
        self._target = target
        self._names = tuple(names)
        self._probe_state = probe_state
        self._start_method = start_method

    def __getattr__(self, name):
        return getattr(self._native, name)

    def Process(self, *, target, args=(), kwargs=None, **options):  # noqa: N802
        if target is not self._target:
            raise _DiagnosticContractRejected("native router process target changed")
        if self._native.get_start_method() != self._start_method:
            raise _DiagnosticContractRejected("native router process context changed")
        return self._native.Process(
            target=_router_child_entry,
            args=(
                target,
                self._names,
                self._probe_state,
                self._start_method,
                args,
                {} if kwargs is None else kwargs,
            ),
            **options,
        )


def _ray_environment(plan, cfg, native, *, diagnostic=False):
    env = _string_environment(native["skyrl.train.utils.utils"].prepare_runtime_environment(cfg))
    env.update(VLLM_SAMPLER_ENV)
    scrubbed = _credential_names(os.environ, env) if diagnostic else ()
    for name in scrubbed:
        env[name] = ""
    log = _prepare_infra_log(plan)
    env["SKYRL_LOG_FILE"] = str(log)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1]) + ":" + os.environ.get("PYTHONPATH", "")
    )
    return env, log, scrubbed


def _diagnostic_runtime_files(plan):
    files = _runtime()
    files.update(
        {p + "/__init__.py": "" for p in ("training", "evals", "evals/fleet", "cyber_post_train")}
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return files


def _diagnostic_output_evidence(plan, *, started):
    """Prove the immutable bundle and absence of task/training artifacts."""
    root = Path(plan["output_root"])
    if not root.is_dir() or root.is_symlink():
        raise ValueError("engine diagnostic output root is not a regular directory")
    allowed = {".runtime"}
    if started:
        allowed.update({"ENGINE_DIAGNOSTIC_STARTED.json", "private-native-logs"})
    if {path.name for path in root.iterdir()} != allowed:
        raise ValueError("engine diagnostic output contains an unexpected artifact")

    runtime = root / ".runtime"
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError("engine diagnostic runtime bundle is not a regular directory")
    expected = _diagnostic_runtime_files(plan)
    actual = {}
    actual_directories = set()
    for path in runtime.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("engine diagnostic runtime bundle contains an unsafe entry")
        if path.is_file():
            actual[str(path.relative_to(runtime))] = path
        else:
            actual_directories.add(str(path.relative_to(runtime)))
    expected_directories = {
        str(parent) for name in expected for parent in Path(name).parents if str(parent) != "."
    }
    if (
        set(actual) != set(expected)
        or actual_directories != expected_directories
        or any(actual[name].read_text() != value for name, value in expected.items())
    ):
        raise ValueError("engine diagnostic runtime bundle changed")
    if digest({name: actual[name].read_text() for name in RUNTIME_FILES}) != plan.get(
        "runtime_sha256"
    ):
        raise ValueError("engine diagnostic runtime sources changed")

    if started:
        value = json.loads((root / "ENGINE_DIAGNOSTIC_STARTED.json").read_bytes())
        sealed(value, ENGINE_DIAGNOSTIC_SCHEMA)
        if (
            value.get("status") != "started"
            or value.get("plan_sha256") != digest(plan)
            or value.get("runtime_user") != {"uid": 1000, "gid": 100}
        ):
            raise ValueError("engine diagnostic start receipt changed")
        logs = root / "private-native-logs"
        if logs.is_symlink() or not logs.is_dir() or (logs.stat().st_mode & 0o777) != 0o700:
            raise ValueError("engine diagnostic private log directory changed")
        for path in logs.rglob("*"):
            if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                raise ValueError("engine diagnostic private logs contain an unsafe entry")
            if path.is_dir() or (
                path.name != "infra.log"
                and re.fullmatch(r"router-\d{6}_\d{6}\.log", path.name) is None
            ):
                raise ValueError("engine diagnostic private logs contain an unexpected entry")
        infra = logs / "infra.log"
        if not infra.is_file() or (infra.stat().st_mode & 0o777) != 0o600:
            raise ValueError("engine diagnostic infrastructure log changed")
    return {
        "runtime_files_unchanged": True,
        "unexpected_output_artifacts": 0,
        "checkpoint_artifacts": 0,
        "episode_artifacts": 0,
        "task_artifacts": 0,
    }


def _add_identity(values, value):
    if value is not None and all(existing is not value for existing in values):
        values.append(value)


@dataclasses.dataclass
class _DiagnosticOwnership:
    groups: list = dataclasses.field(default_factory=list)
    routers: list = dataclasses.field(default_factory=list)
    actors: list = dataclasses.field(default_factory=list)
    engine_actors: list = dataclasses.field(default_factory=list)
    placement_groups: list = dataclasses.field(default_factory=list)
    job_id: str | None = None
    ray_gpu_nodes_discovered: int | None = None
    ray_gpu_nodes_probed: int = 0
    ray_actor_environment_probes_passed: int | None = None
    ray_actor_environment_probe_failures: int | None = None
    ray_actor_nonempty_scrubbed_credentials: int | None = None
    router_start_attempts: int = 0
    router_credential_probes_passed: int = 0
    router_credential_probe_failures: int = 0
    router_multiprocessing_start_method: str | None = None

    def absorb_setup(self, setup):
        if setup is None:
            return
        try:
            _add_identity(self.routers, setup.router)
        except HardDeadlineExceeded:
            raise
        except Exception:
            pass
        try:
            for group in setup.server_groups:
                _add_identity(self.groups, group)
        except HardDeadlineExceeded:
            raise
        except Exception:
            pass

    def absorb_group_resources(self):
        for group in tuple(self.groups):
            try:
                for actor in group.get_actors():
                    _add_identity(self.actors, actor)
                    _add_identity(self.engine_actors, actor)
            except HardDeadlineExceeded:
                raise
            except Exception:
                pass
            for name in ("_internal_pg", "_external_pg"):
                try:
                    value = getattr(group, name)
                    _add_identity(self.placement_groups, getattr(value, "pg", value))
                except HardDeadlineExceeded:
                    raise
                except Exception:
                    pass


class _TrackedActorClass:
    def __init__(self, actor_class, ownership):
        self._actor_class = actor_class
        self._ownership = ownership

    def remote(self, *args, **kwargs):
        actor = self._actor_class.remote(*args, **kwargs)
        _add_identity(self._ownership.actors, actor)
        _add_identity(self._ownership.engine_actors, actor)
        return actor

    def options(self, *args, **kwargs):
        return type(self)(self._actor_class.options(*args, **kwargs), self._ownership)

    def __getattr__(self, name):
        return getattr(self._actor_class, name)


@contextmanager
def _instrument_engine_ownership(ownership, environment, scrubbed):
    """Track exact native objects even when setup loses its partial locals."""
    setup_module = importlib.import_module("skyrl.backends.skyrl_train.inference_servers.setup")
    group_module = importlib.import_module(
        "skyrl.backends.skyrl_train.inference_servers.server_group"
    )
    router_module = importlib.import_module(
        "skyrl.backends.skyrl_train.inference_servers.vllm_router"
    )
    patches = []

    def patch(module, name, value):
        patches.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    original_group = group_module.ServerGroup
    original_router = router_module.VLLMRouter
    original_engine_env = group_module.build_engine_runtime_env
    original_router_target = router_module._run_router_with_logging
    original_multiprocessing = router_module.multiprocessing
    start_method = original_multiprocessing.get_start_method()
    ownership.router_multiprocessing_start_method = start_method
    # A pre-existing forkserver can inherit an older, unsanitized environment.
    # Neither changing the framework's context nor trusting that server is safe.
    if start_method not in {"fork", "spawn"}:
        raise _DiagnosticContractRejected("unsupported native router process context")
    router_probe_state = original_multiprocessing.RawValue("b", 0)

    class TrackedServerGroup(original_group):
        def __init__(self, *args, **kwargs):
            _add_identity(ownership.groups, self)
            super().__init__(*args, **kwargs)

        def _create_actor_class(self, *args, **kwargs):
            actor_class = super()._create_actor_class(*args, **kwargs)
            return _TrackedActorClass(actor_class, ownership)

    class TrackedRouter(original_router):
        def __init__(self, *args, **kwargs):
            _add_identity(ownership.routers, self)
            super().__init__(*args, **kwargs)

        def start(self, *args, **kwargs):
            ownership.router_start_attempts += 1
            router_probe_state.value = 0
            try:
                with _scrubbed_process_environment(scrubbed):
                    return super().start(*args, **kwargs)
            finally:
                if router_probe_state.value == 1:
                    ownership.router_credential_probes_passed += 1
                elif router_probe_state.value in {-1, -2}:
                    ownership.router_credential_probe_failures += 1

    def tracked_placement_group(original):
        def create(*args, **kwargs):
            value = original(*args, **kwargs)
            _add_identity(ownership.placement_groups, value)
            return value

        return create

    def scrubbed_engine_environment(*args, **kwargs):
        value = original_engine_env(*args, **kwargs) or {}
        if not isinstance(value, Mapping):
            raise ValueError("native engine runtime environment changed")
        value = dict(value)
        value["env_vars"] = _scrubbed_actor_environment(
            environment,
            scrubbed,
            value.get("env_vars", {}),
        )
        return value

    try:
        patch(group_module, "ServerGroup", TrackedServerGroup)
        patch(router_module, "VLLMRouter", TrackedRouter)
        patch(
            router_module,
            "multiprocessing",
            _RouterMultiprocessing(
                original_multiprocessing,
                original_router_target,
                scrubbed,
                router_probe_state,
                start_method,
            ),
        )
        # Patch any import-time aliases too.  The pinned implementation reads
        # these module globals while ``create_inference_servers`` is running.
        if getattr(setup_module, "ServerGroup", None) is original_group:
            patch(setup_module, "ServerGroup", TrackedServerGroup)
        if getattr(setup_module, "VLLMRouter", None) is original_router:
            patch(setup_module, "VLLMRouter", TrackedRouter)
        if getattr(group_module, "VLLMRouter", None) is original_router:
            patch(group_module, "VLLMRouter", TrackedRouter)
        patch(
            setup_module,
            "ray_placement_group",
            tracked_placement_group(setup_module.ray_placement_group),
        )
        patch(
            group_module,
            "placement_group",
            tracked_placement_group(group_module.placement_group),
        )
        patch(group_module, "build_engine_runtime_env", scrubbed_engine_environment)
        yield setup_module
    finally:
        for module, name, value in reversed(patches):
            setattr(module, name, value)


@contextmanager
def _bounded_ray_get(ray, deadline):
    """Give every native bare ``ray.get`` the current hard-deadline remainder."""
    original = ray.get

    def bounded(refs, *, timeout=None, **kwargs):
        remaining = deadline.remaining()
        plan_limited = timeout is None or remaining <= float(timeout)
        timeout = remaining if timeout is None else min(remaining, float(timeout))
        try:
            return original(refs, timeout=timeout, **kwargs)
        except BaseException as exc:
            if plan_limited and (
                isinstance(exc, TimeoutError)
                or type(exc).__name__ in {"GetTimeoutError", "RayTimeoutError"}
            ):
                deadline.raise_expired()
            raise

    ray.get = bounded
    try:
        yield
    finally:
        ray.get = original


class _CredentialProbe:
    def inspect(self, names):
        import ray

        return {
            "node_id": str(ray.get_runtime_context().get_node_id()),
            "nonempty": sum(bool(os.environ.get(name)) for name in names),
        }

    def shutdown(self):
        return None


def _live_gpu_node_ids(ray, expected_nodes, expected_gpus_per_node):
    """Return the exact sorted live GPU node IDs or fail on topology uncertainty."""
    rows = ray.nodes()
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Ray node table changed")
    node_ids = []
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("Alive")) is not bool:
            raise ValueError("Ray node record changed")
        if not row["Alive"]:
            continue
        resources = row.get("Resources")
        if not isinstance(resources, Mapping):
            raise ValueError("Ray live-node resources changed")
        gpus = resources.get("GPU", 0)
        if (
            isinstance(gpus, bool)
            or not isinstance(gpus, numbers.Real)
            or not math.isfinite(gpus)
            or gpus < 0
        ):
            raise ValueError("Ray live-node GPU resources changed")
        if gpus == 0:
            continue
        node_id = row.get("NodeID")
        if (
            not isinstance(node_id, str)
            or re.fullmatch(r"[0-9a-fA-F]+", node_id) is None
            or len(node_id) % 2
            or gpus != expected_gpus_per_node
        ):
            raise ValueError("Ray GPU node topology differs from diagnostic request")
        node_ids.append(node_id.lower())
    if len(node_ids) != expected_nodes or len(set(node_ids)) != len(node_ids):
        raise ValueError("Ray GPU node topology differs from diagnostic request")
    return tuple(sorted(node_ids))


def _node_affinity_strategy(ray, node_id):
    strategies = getattr(getattr(ray, "util", None), "scheduling_strategies", None)
    value = getattr(strategies, "NodeAffinitySchedulingStrategy", None)
    if value is None:
        value = importlib.import_module(
            "ray.util.scheduling_strategies"
        ).NodeAffinitySchedulingStrategy
    return value(node_id=node_id, soft=False)


def _probe_worker_credentials(
    ray,
    ownership,
    environment,
    scrubbed,
    *,
    expected_nodes,
    expected_gpus_per_node,
):
    node_ids = _live_gpu_node_ids(ray, expected_nodes, expected_gpus_per_node)
    ownership.ray_gpu_nodes_discovered = len(node_ids)
    actor_environment = _scrubbed_actor_environment(environment, scrubbed)
    actors = []
    for node_id in node_ids:
        actor = (
            ray.remote(_CredentialProbe)
            .options(
                num_cpus=0,
                num_gpus=0,
                max_restarts=0,
                runtime_env={"env_vars": dict(actor_environment)},
                scheduling_strategy=_node_affinity_strategy(ray, node_id),
            )
            .remote()
        )
        _add_identity(ownership.actors, actor)
        actors.append(actor)
    results = ray.get([actor.inspect.remote(scrubbed) for actor in actors])
    if not isinstance(results, (list, tuple)) or len(results) != len(node_ids):
        raise ValueError("Ray credential probe result changed")
    normalized = []
    for result in results:
        if (
            not isinstance(result, Mapping)
            or set(result) != {"node_id", "nonempty"}
            or not isinstance(result["node_id"], str)
            or re.fullmatch(r"[0-9a-fA-F]+", result["node_id"]) is None
            or len(result["node_id"]) % 2
            or type(result["nonempty"]) is not int
            or not 0 <= result["nonempty"] <= len(scrubbed)
        ):
            raise ValueError("Ray credential probe result changed")
        normalized.append((result["node_id"].lower(), result["nonempty"]))
    ownership.ray_gpu_nodes_probed = len(normalized)
    ownership.ray_actor_nonempty_scrubbed_credentials = sum(count for _, count in normalized)
    ownership.ray_actor_environment_probes_passed = sum(count == 0 for _, count in normalized)
    ownership.ray_actor_environment_probe_failures = sum(count != 0 for _, count in normalized)
    if tuple(sorted(node_id for node_id, _ in normalized)) != node_ids:
        raise ValueError("Ray credential probes did not cover the exact GPU nodes")
    if _live_gpu_node_ids(ray, expected_nodes, expected_gpus_per_node) != node_ids:
        raise ValueError("Ray GPU node topology changed during credential probes")
    if ownership.ray_actor_nonempty_scrubbed_credentials != 0:
        raise ValueError("Ray actor retained a scrubbed credential value")
    return 0


def _validate_complete_engine_setup(setup, engine_config):
    expected_groups = engine_config.num_engines
    expected_servers = expected_groups * engine_config.data_parallel_size
    groups = tuple(setup.server_groups)
    urls = tuple(setup.server_urls)
    actor_counts = tuple(len(tuple(group.get_actors())) for group in groups)
    if (
        type(expected_groups) is not int
        or type(expected_servers) is not int
        or expected_groups < 1
        or expected_servers < 1
        or len(groups) != expected_groups
        or len(urls) != expected_servers
        or actor_counts != (engine_config.data_parallel_size,) * expected_groups
        or any(not isinstance(url, str) or not url for url in urls)
        or setup.router is None
        or not isinstance(setup.proxy_url, str)
        or not setup.proxy_url
    ):
        raise ValueError("native engine setup returned a partial topology")


def _active_owned_resources(job_id, timeout):
    """Read exact current-job ownership without Ray's dashboard server.

    Ray 2.56's public state list APIs route through the optional dashboard at
    port 8265, which the pinned training cluster does not run.  Its internal
    actor table and raw GCS placement-group table are dashboard-independent.
    The caller's process alarm bounds both raw calls because they expose no
    per-request timeout; an error or deadline remains fatal uncertainty.
    Treat every state except the one terminal state as active so a new or
    unfamiliar state can never be mistaken for successful cleanup.
    """
    job_id = _ray_job_id_hex(job_id)
    if timeout <= 0:
        raise ValueError("invalid Ray cleanup ownership query")
    ray_state = importlib.import_module("ray._private.state")
    actors = ray_state.actors()
    if not isinstance(actors, Mapping):
        raise ValueError("Ray actor ownership table changed")
    active_actors = 0
    for value in actors.values():
        if not isinstance(value, Mapping) or "JobID" not in value or "State" not in value:
            raise ValueError("Ray actor ownership record changed")
        if _ray_job_id_hex(value["JobID"]) == job_id and value["State"] != "DEAD":
            active_actors += 1

    gcs_pb2 = importlib.import_module("ray.core.generated.gcs_pb2")
    binary_to_hex = importlib.import_module("ray._common.utils").binary_to_hex
    accessor = ray_state.state._connect_and_get_accessor()
    payloads = accessor.get_placement_group_table()
    if not isinstance(payloads, (list, tuple)):
        raise ValueError("Ray placement-group ownership table changed")
    active_groups = 0
    for payload in payloads:
        row = gcs_pb2.PlacementGroupTableData.FromString(payload)
        creator_job_id = _ray_job_id_hex(binary_to_hex(row.creator_job_id))
        if creator_job_id != job_id:
            continue
        try:
            state_name = gcs_pb2.PlacementGroupTableData.PlacementGroupState.Name(row.state)
        except (TypeError, ValueError) as exc:
            raise ValueError("Ray placement-group state changed") from exc
        if state_name != "REMOVED":
            active_groups += 1
    return active_actors, active_groups


def _ray_job_id_hex(value):
    """Normalize Ray's JobID object without trusting its display string.

    Ray 2.56 renders ``str(JobID)`` as ``JobID(<hex>)``.  The raw ``hex()``
    method is the stable identity used by the actor and placement-group tables.
    """
    if isinstance(value, str):
        result = value
    else:
        method = getattr(value, "hex", None)
        if not callable(method):
            raise ValueError("Ray job identity changed")
        result = method()
    if (
        not isinstance(result, str)
        or re.fullmatch(r"[0-9a-fA-F]+", result) is None
        or len(result) % 2
    ):
        raise ValueError("Ray job identity changed")
    return result.lower()


def _prove_owned_resources_released(ownership, deadline):
    if not ownership.job_id:
        raise ValueError("Ray job identity unavailable for cleanup proof")
    while True:
        actors, groups = _active_owned_resources(ownership.job_id, deadline.remaining())
        if actors == groups == 0:
            return {"active_owned_actors": 0, "active_owned_placement_groups": 0}
        time.sleep(min(0.1, deadline.remaining() / 2))


def _router_released(router, shutdown_returned):
    marker = object()
    process = getattr(router, "_process", marker)
    if process is marker:
        return shutdown_returned
    if process is not None and process.is_alive():
        return False
    return all(
        getattr(router, name, None) is None
        for name in ("_port_reservation", "_prometheus_port_reservation")
    )


def _cleanup_engine_diagnostic(setup, ray, ownership, timeout_seconds):
    """Bound, force, and independently prove all diagnostic-owned cleanup."""
    with (
        hard_deadline(timeout_seconds, "SkyRL engine cleanup") as deadline,
        _bounded_ray_get(ray, deadline),
    ):
        ownership.absorb_setup(setup)
        ownership.absorb_group_resources()
        router_results = {}
        for router in ownership.routers:
            try:
                router.shutdown()
            except HardDeadlineExceeded:
                raise
            except Exception:
                router_results[id(router)] = False
            else:
                router_results[id(router)] = True

        shutdown_refs = []
        graceful = True
        for actor in ownership.actors:
            try:
                shutdown_refs.append(actor.shutdown.remote())
            except HardDeadlineExceeded:
                raise
            except Exception:
                graceful = False
        if shutdown_refs:
            try:
                ray.get(
                    shutdown_refs,
                    timeout=min(30.0, max(0.1, deadline.remaining() / 3)),
                )
            except HardDeadlineExceeded:
                raise
            except Exception:
                graceful = False
        for actor in ownership.actors:
            deadline.remaining()
            try:
                ray.kill(actor, no_restart=True)
            except HardDeadlineExceeded:
                raise
            except Exception:
                pass

        remove_placement_group = importlib.import_module(
            "ray.util.placement_group"
        ).remove_placement_group
        for group in ownership.placement_groups:
            deadline.remaining()
            try:
                remove_placement_group(group)
            except HardDeadlineExceeded:
                raise
            except Exception:
                pass

        if ownership.job_id:
            proof = _prove_owned_resources_released(ownership, deadline)
        elif any(
            (ownership.actors, ownership.groups, ownership.placement_groups, ownership.routers)
        ):
            raise ValueError("Ray job identity unavailable for cleanup proof")
        else:
            proof = {"active_owned_actors": 0, "active_owned_placement_groups": 0}
        if any(
            not _router_released(router, router_results.get(id(router), False))
            for router in ownership.routers
        ):
            raise ValueError("engine diagnostic router cleanup could not be proven")
        ray.shutdown()
        deadline.remaining()
    return {
        **proof,
        "tracked_ray_actors": len(ownership.actors),
        "tracked_engine_actors": len(ownership.engine_actors),
        "tracked_placement_groups": len(ownership.placement_groups),
        "tracked_routers": len(ownership.routers),
        "graceful_actor_shutdown": graceful,
        "cleanup_proven": True,
    }


def _reward_canary_policy_worker(plan: dict):
    """Instrument the native worker only to seal its pre-update policy state."""
    import torch.distributed as dist
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    class RewardCanaryPolicyWorker(FSDPPolicyWorkerBase):
        def init_model(self, model_path, num_training_steps=None):
            result = super().init_model(model_path, num_training_steps=num_training_steps)
            if (
                model_path != plan["model"]["root"]
                or num_training_steps != 1
                or self.optimizer is None
                or not isinstance(self.optimizer.state, dict)
                or self.optimizer.state
            ):
                raise ValueError("reward canary lacks an exact fresh policy initialization")
            rank, world_size = dist.get_rank(), dist.get_world_size()
            if world_size != 8 or rank not in range(world_size):
                raise ValueError("reward canary policy world topology changed")
            root = Path(plan["output_root"]) / "reward_canary_base_policy"
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if root.is_symlink() or not root.is_dir():
                raise ValueError("reward canary base-policy evidence root is indirect")
            _write(
                root / f"rank-{rank}.json",
                {
                    "schema": REWARD_CANARY_BASE_POLICY_RANK_SCHEMA,
                    "source_plan_sha256": digest(plan),
                    "model_root": plan["model"]["root"],
                    "model_revision": plan["model"]["revision"],
                    "weight_manifest_sha256": plan["model"]["weight_manifest_sha256"],
                    "rank": rank,
                    "world_size": world_size,
                    "initial_optimizer_states": len(self.optimizer.state),
                    "policy_state_sha256": "sha256:"
                    + _policy_state_digest(self.model.model.state_dict()),
                },
            )
            self._reward_canary_base_policy_captured = True
            self._reward_canary_optimizer_steps = 0
            return result

        def optim_step(self, *args, **kwargs):
            if not getattr(self, "_reward_canary_base_policy_captured", False):
                raise ValueError("optimizer cannot run before base policy is sealed")
            if self._reward_canary_optimizer_steps != 0:
                raise ValueError("reward canary performed more than one optimizer update")
            result = super().optim_step(*args, **kwargs)
            self._reward_canary_optimizer_steps += 1
            return result

    return RewardCanaryPolicyWorker


@contextmanager
def _reward_canary_policy_probe(plan: dict, ray):
    """Scope the audited worker replacement to the exact canary native loop."""
    if not is_reward_canary(plan):
        yield
        return
    from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker

    original = fsdp_worker.PolicyWorker
    fsdp_worker.PolicyWorker = ray.remote(num_gpus=1)(_reward_canary_policy_worker(plan))
    try:
        yield
    finally:
        fsdp_worker.PolicyWorker = original


def _native(plan):
    import ray
    import wandb
    from skyrl.backends.skyrl_train.utils.ppo_utils import sync_registries

    from .skyrl_promotion import require_production_runtime_identity
    from .skyrl_rollout import Generator

    _require_engine_start_qualified_image(plan)
    require_production_runtime_identity(plan)
    rows, modules = check_artifacts(plan), native_source()
    _validate_vllm_sampler_fallback_contract()
    args = skyrl.SkyRLConfig(**plan["arguments"])
    cfg = skyrl.native_config(args)
    base = modules["skyrl.train.entrypoints.main_base"].BasePPOExp

    class Experiment(base):
        def get_train_dataset(self):
            return dataset(plan, self.tokenizer, "train", rows["train"])

        def get_eval_dataset(self):
            return dataset(plan, self.tokenizer, "dev", rows["dev"])

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
            return ScalarTracking(plan)

        def get_trajectory_logger(self):
            return None

    env, _, _ = _ray_environment(plan, cfg, modules)
    ray.init(address="auto", log_to_driver=False, runtime_env={"env_vars": env})
    code = 1
    try:
        sync_registries()
        experiment = Experiment(cfg)
        with _reward_canary_policy_probe(plan, ray):
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


def _require_engine_diagnostic_runtime_identity(plan: dict) -> dict:
    """Fail in the allocated diagnostic GPU Pod unless its process is 1000:100."""
    _diagnostic_shape(plan)
    if (
        os.environ.get("CYBER_EXPECTED_RUNTIME_UID") != "1000"
        or os.environ.get("CYBER_EXPECTED_RUNTIME_GID") != "100"
        or (os.geteuid(), os.getegid()) != (1000, 100)
    ):
        raise ValueError("engine diagnostic GPU runtime must be the pinned non-root user 1000:100")
    return {"uid": os.geteuid(), "gid": os.getegid()}


def engine_diagnostic(plan):
    """Start and stop only the exact vLLM engines; never load task rows or train."""
    root = Path(plan["output_root"])
    if os.environ.get("RUN_DIR") != str(root):
        raise ValueError("Jobs API output binding mismatch")
    _require_engine_start_qualified_image(plan)
    runtime_user = _require_engine_diagnostic_runtime_identity(plan)
    args = skyrl.SkyRLConfig(**plan["arguments"])
    ownership = _DiagnosticOwnership()
    setup = None
    ray = None
    log = None
    scrubbed = ()
    shape = None
    create_called = False
    ray_initialization_attempted = False
    startup_error = None
    startup_deadline = None
    startup_phase = "plan_validation"
    ray_actor_environment_isolation_proven = False
    try:
        # One wall-clock budget covers all work done after GPU allocation,
        # including the exact 27B model re-hash.  Cleanup owns a separate alarm
        # only after this context has restored the process signal state.
        with hard_deadline(
            args.engine_start_timeout_seconds, "SkyRL engine startup"
        ) as startup_deadline:
            job_request(plan)
            startup_phase = "runtime_validation"
            _diagnostic_output_evidence(plan, started=False)
            startup_phase = "input_validation"
            check_inputs(plan)
            startup_phase = "native_validation"
            modules = native_source()
            _validate_vllm_sampler_fallback_contract()
            cfg = skyrl.diagnostic_native_config(args)
            shape = _diagnostic_shape(plan, cfg)
            _write(
                root / "ENGINE_DIAGNOSTIC_STARTED.json",
                {
                    "schema": ENGINE_DIAGNOSTIC_SCHEMA,
                    "status": "started",
                    "plan_sha256": digest(plan),
                    "runtime_user": runtime_user,
                    "optimizer_steps": 0,
                    "rollouts": 0,
                    "verifier_calls": 0,
                    "checkpoints_created": 0,
                    "engine_start_timeout_seconds": args.engine_start_timeout_seconds,
                    "engine_cleanup_timeout_seconds": args.engine_cleanup_timeout_seconds,
                    **shape,
                },
            )
            startup_phase = "ray_environment"
            import ray as ray_module
            from skyrl.backends.skyrl_train.inference_servers.utils import (
                build_vllm_cli_args,
            )

            ray = ray_module
            env, log, scrubbed = _ray_environment(plan, cfg, modules, diagnostic=True)
            startup_phase = "ownership_instrumentation"
            with (
                _instrument_engine_ownership(ownership, env, scrubbed) as setup_module,
                _bounded_ray_get(ray, startup_deadline),
            ):
                startup_phase = "ray_initialization"
                ray_initialization_attempted = True
                ray.init(address="auto", log_to_driver=False, runtime_env={"env_vars": env})
                ownership.job_id = _ray_job_id_hex(ray.get_runtime_context().get_job_id())
                startup_phase = "ray_actor_environment_probe"
                _probe_worker_credentials(
                    ray,
                    ownership,
                    env,
                    scrubbed,
                    expected_nodes=shape["diagnostic_workers"],
                    expected_gpus_per_node=shape["diagnostic_gpus_per_worker"],
                )
                ray_actor_environment_isolation_proven = True
                startup_phase = "engine_argument_build"
                vllm_args = build_vllm_cli_args(cfg)
                engine_config = cfg.generator.inference_engine
                engine_log_path = cfg.trainer.log_path
                create_called = True
                startup_phase = "engine_creation"
                setup = setup_module.create_inference_servers(
                    engine_config,
                    vllm_args,
                    log_path=engine_log_path,
                )
                startup_phase = "engine_validation"
                _validate_complete_engine_setup(setup, engine_config)
                if (
                    ownership.router_start_attempts != 1
                    or ownership.router_credential_probes_passed != 1
                    or ownership.router_credential_probe_failures != 0
                ):
                    raise ValueError("router child credential isolation was not proven")
                startup_deadline.remaining()
                startup_phase = "complete"
    except BaseException as exc:
        startup_error = exc

    ownership.absorb_setup(setup)
    if startup_error is None:
        engine_start_state, engine_started = "all", True
        status = "passed"
    elif setup is not None or ownership.groups or ownership.engine_actors or ownership.routers:
        engine_start_state, engine_started = "partial_or_unknown", None
        status = None
    else:
        engine_start_state, engine_started = "none", False
        status = None

    timed_out = bool(startup_deadline is not None and startup_deadline.expired)
    router_isolation_proven = ownership.router_start_attempts == 0 or (
        ownership.router_start_attempts == ownership.router_credential_probes_passed
        and ownership.router_credential_probe_failures == 0
    )
    # Only a typed, recognized contract rejection before any Ray initialization
    # is a clean pre-Ray outcome.  Import errors and arbitrary ValueErrors still
    # fail, and any ambiguous allocation continues to require exact cleanup.
    pre_ray_contract_rejection = (
        isinstance(startup_error, _DiagnosticContractRejected)
        and startup_phase == "ownership_instrumentation"
        and not ray_initialization_attempted
        and not any(
            (
                ownership.job_id,
                ownership.groups,
                ownership.routers,
                ownership.actors,
                ownership.engine_actors,
                ownership.placement_groups,
            )
        )
    )
    terminal_evidence_candidate = pre_ray_contract_rejection or (
        isinstance(startup_error, Exception) and bool(ownership.job_id)
    )
    clean_candidate = pre_ray_contract_rejection or (
        terminal_evidence_candidate
        and ray_actor_environment_isolation_proven
        and router_isolation_proven
    )
    if terminal_evidence_candidate:
        if pre_ray_contract_rejection:
            status = "diagnostic_contract_rejected"
        elif timed_out:
            status = "engine_start_timeout"
        elif not ray_actor_environment_isolation_proven or not router_isolation_proven:
            status = "environment_isolation_rejected"
        elif create_called:
            status = "engine_start_rejected"
        else:
            status = "pre_engine_rejected"

    if ray is not None:
        try:
            cleanup = _cleanup_engine_diagnostic(
                setup,
                ray,
                ownership,
                args.engine_cleanup_timeout_seconds,
            )
        except BaseException as cleanup_error:
            if startup_error is not None:
                raise BaseExceptionGroup(
                    "engine diagnostic startup and cleanup failed",
                    [startup_error, cleanup_error],
                ) from None
            raise
    elif startup_error is None:
        raise RuntimeError("engine diagnostic did not initialize Ray")

    if startup_error is not None and not terminal_evidence_candidate:
        raise startup_error

    from .rl_runtime import sanitized_causes

    output = _diagnostic_output_evidence(plan, started=True)
    if log is None:
        raise RuntimeError("engine diagnostic infrastructure log unavailable")
    if shape is None:
        raise RuntimeError("engine diagnostic cluster shape unavailable")
    router_isolation = (
        "proven"
        if ownership.router_start_attempts == ownership.router_credential_probes_passed == 1
        and ownership.router_credential_probe_failures == 0
        else "not_started"
        if ownership.router_start_attempts == 0
        else "failed_or_unknown"
    )
    result = {
        "schema": ENGINE_DIAGNOSTIC_SCHEMA,
        "status": status,
        "engine_start_state": engine_start_state,
        "engine_started": engine_started,
        "startup_phase": startup_phase,
        "causes": [] if startup_error is None else sanitized_causes(startup_error),
    }
    result.update(
        {
            "plan_sha256": digest(plan),
            "runtime_user": runtime_user,
            "private_log": _infra_log_evidence(log),
            "task_rows_read": 0,
            "verifier_calls": 0,
            "optimizer_steps": 0,
            "rollouts": 0,
            "checkpoints_created": 0,
            "checkpoint_created": False,
            "wandb_initialized": False,
            "diagnostic_completed": True,
            "engine_start_qualified": result["status"] == "passed",
            "training_qualified": False,
            "production_training_shape_qualified": False,
            "engine_start_timeout_seconds": args.engine_start_timeout_seconds,
            "engine_cleanup_timeout_seconds": args.engine_cleanup_timeout_seconds,
            "credential_variables_scrubbed": len(scrubbed),
            "credential_environment_isolation_proven": (
                ray_actor_environment_isolation_proven and router_isolation_proven
            ),
            "ray_actor_environment_isolation_proven": (ray_actor_environment_isolation_proven),
            "ray_initialization_attempted": ray_initialization_attempted,
            "ray_gpu_nodes_expected": shape["diagnostic_workers"],
            "ray_gpu_nodes_discovered": ownership.ray_gpu_nodes_discovered,
            "ray_gpu_nodes_probed": ownership.ray_gpu_nodes_probed,
            "ray_actor_environment_probes_passed": (ownership.ray_actor_environment_probes_passed),
            "ray_actor_environment_probe_failures": (
                ownership.ray_actor_environment_probe_failures
            ),
            "ray_actor_nonempty_scrubbed_credentials": (
                ownership.ray_actor_nonempty_scrubbed_credentials
            ),
            "router_child_credential_environment_isolation": router_isolation,
            "router_start_attempts": ownership.router_start_attempts,
            "router_multiprocessing_start_method": ownership.router_multiprocessing_start_method,
            "router_environment_probes_passed": ownership.router_credential_probes_passed,
            "router_environment_probe_failures": ownership.router_credential_probe_failures,
            "registry_actors_created": 0,
            "service_account_token_isolation_proven": False,
            "service_account_rbac_write_access_tested": False,
            **shape,
            "cleanup": cleanup,
            "output_postconditions": output,
            "completed_at": time.time(),
        }
    )
    result = _write(root / "ENGINE_DIAGNOSTIC.json", result)
    if startup_error is not None and not clean_candidate:
        raise startup_error
    return result


def native_result(plan):
    root, args = Path(plan["output_root"]), plan["arguments"]
    expected = (
        {("train", i) for i in range(1, args["steps"] + 1)}
        | {("eval", 0), ("eval", args["steps"])}
        | {("eval", i) for i in range(1, args["steps"] + 1) if i % args["eval_interval"] == 0}
    )
    seen = set()
    for directory in (root / "episodes/batches").iterdir():
        value = json.loads((directory / "COLLECTED.json").read_bytes())
        sealed(value, "cyber_skyrl_batch_v1")
        key = (value["phase"], value["global_step"])
        if (
            key not in expected
            or key in seen
            or (directory / "FAILED.json").exists()
            or value["data_sha256"] != plan["data"]["sha256"]
        ):
            raise ValueError("native batch completion evidence mismatch")
        seen.add(key)
    if seen != expected:
        raise ValueError("native training/dev batch missing")
    checkpoint = root / f"checkpoints/global_step_{args['steps']}"
    if (
        (root / "checkpoints/latest_ckpt_global_step.txt").read_text().strip() != str(args["steps"])
        or any(
            not (checkpoint / p).is_file() or (checkpoint / p).stat().st_size == 0
            for p in ("data.pt", "trainer_state.pt")
        )
        or not any(p.is_file() and p.stat().st_size for p in (checkpoint / "policy").rglob("*"))
    ):
        raise ValueError("native final checkpoint/sampler missing")
    return {
        "status": "native_loop_returned",
        "plan_sha256": digest(plan),
        "checkpoint_global_step": args["steps"],
        "completed_batches": len(seen),
        "completed_at": time.time(),
        "optimizer_update_independently_verified": False,
        "checkpoint_reload_verified": False,
    }


def _evidence_reference(path: Path, value: dict, file_sha256: str | None = None) -> dict:
    if file_sha256 is None:
        _, file_sha256 = _json_snapshot(path)
    self_sha256 = value.get("sha256")
    if not isinstance(self_sha256, str):
        raise ValueError("evidence receipt lacks a self digest")
    return {
        "path": str(path),
        "file_sha256": file_sha256,
        "receipt_self_sha256": "sha256:" + self_sha256.removeprefix("sha256:"),
    }


def _reward_canary_source_rows(plan: dict) -> dict[str, dict]:
    rows = {}
    for phase, split in (("train", "train"), ("eval", "dev")):
        item = plan["data"]["files"][split]
        path = Path(plan["arguments"][f"{split}_data"])
        payload, file_sha256 = _snapshot(path)
        if file_sha256 != item["sha256"] or item["rows"] != 1:
            raise ValueError("reward canary staged row identity changed")
        values = [json.loads(line) for line in payload.splitlines()]
        if len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError("reward canary requires one exact row per split")
        config = json.loads(values[0]["cyber_config_json"])
        from .rl_episode import _validate

        _validate(config)
        if values[0].get("split") != split or config.get("run_id") != plan["run_name"]:
            raise ValueError("reward canary staged row binding changed")
        rows[phase] = config
    return rows


def _validate_authoritative_reward(config: dict, reward: object) -> tuple[float, str, str]:
    from evals.fleet import opencode_self_hosted as fleet

    if not isinstance(reward, dict) or set(reward) != {
        "task_key",
        "task_version_id",
        "instance_id",
        "reward",
        "verifier_execution_id",
        "direct_authority_attestation",
    }:
        raise ValueError("episode reward is not the sanitized direct-authority receipt")
    value, execution_id = reward.get("reward"), reward.get("verifier_execution_id")
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or not 0 <= value <= 1
        or _UUID.fullmatch(str(execution_id)) is None
        or execution_id == "00000000-0000-0000-0000-000000000000"
        or reward.get("task_key") != config["task"]["key"]
        or reward.get("task_version_id") != config["task"]["version_id"]
    ):
        raise ValueError("authoritative reward identity or value changed")
    attestation = reward["direct_authority_attestation"]
    expected_context = {
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": reward["instance_id"],
        "evidence_run_id": None,
        "verifier_version_id": config["verifier"]["version_id"],
        "scoring_payload_mode": fleet.RUNTIME_EVIDENCE_ONLY_V3,
    }
    context = attestation.get("context") if isinstance(attestation, dict) else None
    expected_context["evidence_run_id"] = context.get("evidence_run_id") if context else None
    direct = (
        attestation.get("shadow", {}).get("direct_verifier")
        if isinstance(attestation, dict)
        else None
    )
    if (
        not isinstance(context, dict)
        or _UUID.fullmatch(str(context.get("evidence_run_id"))) is None
        or context != expected_context
        or attestation.get("schema_version") != fleet.DIRECT_AUTHORITY_ATTESTATION_SCHEMA
        or attestation.get("activity")
        != {
            "result_schema_version": "cyber_verification_result_v3",
            "reward": float(value),
            "task_version_id": config["task"]["version_id"],
            "verifier_execution_id": execution_id,
        }
        or attestation.get("shadow")
        != {
            "mode": "authoritative",
            "status": "authoritative",
            "match": True,
            "production_execution_id": execution_id,
            "direct_verifier": direct,
        }
        or direct
        != {
            "status": "authoritative",
            "match": True,
            "execution_id": execution_id,
            "verifier_contract_version": config["authority"]["required_cyber_contract"][
                "verifier_contract"
            ],
            "context_schema_version": "cyber_verification_context_v1",
        }
        or attestation.get("data_minimization")
        != {
            "components_included": False,
            "diagnostics_included": False,
            "evidence_payloads_included": False,
            "prompts_included": False,
            "traces_included": False,
            "flags_included": False,
        }
    ):
        raise ValueError("authoritative verifier execution attestation changed")
    return float(value), execution_id, context["evidence_run_id"]


def reward_canary_episode_audit(plan: dict) -> dict:
    """Reopen all ten private episodes and derive reward/termination facts."""
    from evals.fleet import opencode_self_hosted as fleet

    if not is_reward_canary(plan):
        raise ValueError("episode audit requires the exact reward canary")
    source_rows = _reward_canary_source_rows(plan)
    root = Path(plan["output_root"]) / "episodes/batches"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("reward canary batch directory is missing or indirect")
    directories = [path for path in root.iterdir() if path.is_dir() and not path.is_symlink()]
    if len(directories) != 3 or any(not path.is_dir() for path in root.iterdir()):
        raise ValueError("reward canary needs exactly three direct batch directories")
    expected = {
        ("eval", 0): [["0", 0]],
        ("train", 1): [["0", index] for index in range(8)],
        ("eval", 1): [["0", 0]],
    }
    batches = {}
    verifier_ids = set()
    for directory in directories:
        collected_path = directory / "COLLECTED.json"
        collected, collected_file_sha256 = _json_snapshot(collected_path)
        sealed(collected, "cyber_skyrl_batch_v1")
        key = (collected.get("phase"), collected.get("global_step"))
        trajectories = collected.get("trajectory_ids")
        binding = {
            "phase": key[0],
            "global_step": key[1],
            "trajectory_ids": trajectories,
        }
        identifier = fleet.sha256(fleet.canonical_json(binding)).removeprefix("sha256:")[:24]
        if (
            key not in expected
            or key in batches
            or trajectories != expected[key]
            or directory.name != identifier
            or collected.get("data_sha256") != plan["data"]["sha256"]
            or collected.get("optimizer_step_verified") is not False
            or (directory / "FAILED.json").exists()
            or (directory / "REJECTED.json").exists()
        ):
            raise ValueError("reward canary batch identity or completion changed")
        episodes = []
        for index, trajectory in enumerate(trajectories):
            episode_root = directory / f"episode-{index}"
            names = {
                "binding.json",
                "create-intent.json",
                "instance.json",
                "conversation.json",
                "score-intent.json",
                "reward.json",
                "cleanup.json",
                "recording.json",
                "ACCEPTED.json",
            }
            if (
                episode_root.is_symlink()
                or not episode_root.is_dir()
                or {path.name for path in episode_root.iterdir()} != names
                or any(path.is_symlink() or not path.is_file() for path in episode_root.iterdir())
            ):
                raise ValueError("reward canary episode status is incomplete or ambiguous")
            values = {}
            file_digests = {}
            for name in (
                "binding.json",
                "create-intent.json",
                "instance.json",
                "score-intent.json",
                "reward.json",
                "cleanup.json",
                "ACCEPTED.json",
            ):
                values[name], file_digests[name] = _json_snapshot(episode_root / name)
            config, create_intent, instance, score_intent, reward, cleanup, accepted = (
                values[name]
                for name in (
                    "binding.json",
                    "create-intent.json",
                    "instance.json",
                    "score-intent.json",
                    "reward.json",
                    "cleanup.json",
                    "ACCEPTED.json",
                )
            )
            base = source_rows[key[0]]
            dynamic = {"run_id", "native_batch", "sampling", "config_sha256"}
            if (
                config.get("config_sha256") != fleet.digest_without(config, "config_sha256")
                or {name: value for name, value in config.items() if name not in dynamic}
                != {name: value for name, value in base.items() if name not in dynamic}
                or config.get("run_id") != f"{base['run_id']}-{identifier}-{index}"
                or config.get("native_batch") != binding
                or not isinstance(config.get("sampling"), dict)
            ):
                raise ValueError("reward canary episode configuration identity changed")
            reward_value, execution_id, evidence_run_id = _validate_authoritative_reward(
                config, reward
            )
            if execution_id in verifier_ids:
                raise ValueError("reward canary verifier execution ID was reused")
            verifier_ids.add(execution_id)
            try:
                fleet.validate_scoring_payload(config, score_intent)
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                raise ValueError("reward canary score intent changed") from error
            if (
                create_intent != {"run_id": config["run_id"]}
                or instance
                != {
                    "instance_id": reward["instance_id"],
                    "evidence_run_id": evidence_run_id,
                }
                or score_intent
                != fleet.build_scoring_payload(
                    config,
                    instance_id=reward["instance_id"],
                    final_answer="",
                    messages=[],
                )
                or score_intent.get("instance_id") != reward["instance_id"]
                or cleanup
                != {
                    "create_attempted": True,
                    "instance_created": True,
                    "instance_closed": True,
                    "instance_id": reward["instance_id"],
                    "possible_instance_leak": False,
                }
                or accepted.get("sha256")
                != fleet.sha256(
                    fleet.canonical_json(
                        {name: value for name, value in accepted.items() if name != "sha256"}
                    )
                )
                or accepted.get("config_sha256") != config["config_sha256"]
                or accepted.get("task_version_id") != config["task"]["version_id"]
                or accepted.get("instance_id") != reward["instance_id"]
                or accepted.get("verifier_execution_id") != execution_id
                or accepted.get("sample_count") != 1
                or accepted.get("done_reason") not in {"model_stop", "report_submitted"}
                or set(accepted.get("files", {}))
                != {
                    "binding.json",
                    "create-intent.json",
                    "instance.json",
                    "conversation.json",
                    "score-intent.json",
                    "reward.json",
                    "cleanup.json",
                    "recording.json",
                }
                or any(
                    accepted["files"][name] != "sha256:" + _hash(episode_root / name)
                    for name in accepted["files"]
                )
            ):
                raise ValueError("reward canary ACCEPTED receipt is not evidence-derived")
            episodes.append(
                {
                    "index": index,
                    "trajectory_id": trajectory,
                    "status": "accepted",
                    "truncation_reason": None,
                    "task_key": config["task"]["key"],
                    "task_version_id": config["task"]["version_id"],
                    "environment_version_id": config["environment"]["version_id"],
                    "config_sha256": config["config_sha256"],
                    "run_id": config["run_id"],
                    "instance_id": reward["instance_id"],
                    "evidence_run_id": evidence_run_id,
                    "verifier_execution_id": execution_id,
                    "done_reason": accepted["done_reason"],
                    "reward": reward_value,
                    "accepted": _evidence_reference(
                        episode_root / "ACCEPTED.json",
                        accepted,
                        file_digests["ACCEPTED.json"],
                    ),
                    "binding_file_sha256": file_digests["binding.json"],
                    "create_intent_file_sha256": file_digests["create-intent.json"],
                    "instance_file_sha256": file_digests["instance.json"],
                    "score_intent_file_sha256": file_digests["score-intent.json"],
                    "reward_file_sha256": file_digests["reward.json"],
                    "cleanup_file_sha256": file_digests["cleanup.json"],
                }
            )
        batches[key] = {
            "phase": key[0],
            "global_step": key[1],
            "trajectory_ids": trajectories,
            "collected": _evidence_reference(
                collected_path,
                collected,
                collected_file_sha256,
            ),
            "episodes": episodes,
        }
    if set(batches) != set(expected):
        raise ValueError("reward canary train/dev batch coverage changed")
    ordered = [batches[key] for key in (("eval", 0), ("train", 1), ("eval", 1))]
    rewards = [episode["reward"] for batch in ordered for episode in batch["episodes"]]
    train_rewards = [episode["reward"] for episode in batches[("train", 1)]["episodes"]]
    mean = math.fsum(train_rewards) / len(train_rewards)
    variance = math.fsum((value - mean) ** 2 for value in train_rewards) / len(train_rewards)
    nonzero = sum(value != 0 for value in train_rewards)
    if (
        len(rewards) != 10
        or len(train_rewards) != 8
        or any(not math.isfinite(value) for value in rewards)
        or nonzero < 1
        or not math.isfinite(variance)
        or variance <= 0
    ):
        raise ValueError("reward canary lacks finite nonzero within-group reward variance")
    return {
        "schema": REWARD_CANARY_EPISODE_AUDIT_SCHEMA,
        "source_plan_sha256": digest(plan),
        "data_sha256": plan["data"]["sha256"],
        "batches": ordered,
        "reward_vector": rewards,
        "train_reward_vector": train_rewards,
        "train_nonzero_count": nonzero,
        "train_reward_mean": mean,
        "train_reward_population_variance": variance,
        "episode_status_counts": {"accepted": 10, "truncated": 0},
        "source_use_counts": {
            "reward_canary_train": 8,
            "reward_canary_dev": 2,
            "webexploitbench": 0,
            "final_test": 0,
        },
    }


def reward_canary_update_proof(plan: dict, manifest_path: Path, manifest: dict) -> dict:
    """Derive one update from independently decoded optimizer/checkpoint state."""
    from . import skyrl_rl_checkpoint as checkpoint

    if not is_reward_canary(plan) or manifest.get("source_plan") != plan:
        raise ValueError("optimizer proof source plan changed")
    checkpoint.verify_manifest(manifest, check_files=True)
    state = manifest["checkpoint"]
    ranks = [
        {
            key: rank[key]
            for key in (
                "rank",
                "policy_state_sha256",
                "optimizer_state_sha256",
                "optimizer_states",
                "optimizer_step_states",
                "optimizer_step",
                "scheduler_last_epoch",
            )
        }
        for rank in state["ranks"]
    ]
    if (
        state.get("step") != 1
        or plan["native_overrides"].get("trainer.resume_mode") != "none"
        or plan["native_overrides"].get("trainer.max_training_steps") != 1
        or [rank["rank"] for rank in ranks] != list(range(8))
        or any(
            rank["optimizer_states"] < 1
            or rank["optimizer_step_states"] < 1
            or rank["optimizer_step"] != 1
            or rank["scheduler_last_epoch"] != 1
            for rank in ranks
        )
    ):
        raise ValueError("checkpoint state does not prove exactly one optimizer update")
    base_policy_ranks = []
    compared_ranks = []
    expected_base_fields = {
        "schema",
        "source_plan_sha256",
        "model_root",
        "model_revision",
        "weight_manifest_sha256",
        "rank",
        "world_size",
        "initial_optimizer_states",
        "policy_state_sha256",
        "sha256",
    }
    for rank, checkpoint_rank in enumerate(ranks):
        path = Path(plan["output_root"]) / "reward_canary_base_policy" / f"rank-{rank}.json"
        base, file_sha256 = _json_snapshot(path)
        sealed(base, REWARD_CANARY_BASE_POLICY_RANK_SCHEMA)
        base_digest = str(base.get("policy_state_sha256", ""))
        checkpoint_digest = str(checkpoint_rank["policy_state_sha256"])
        if (
            set(base) != expected_base_fields
            or base.get("source_plan_sha256") != digest(plan)
            or base.get("model_root") != plan["model"]["root"]
            or base.get("model_revision") != plan["model"]["revision"]
            or base.get("weight_manifest_sha256") != plan["model"]["weight_manifest_sha256"]
            or base.get("rank") != rank
            or base.get("world_size") != 8
            or base.get("initial_optimizer_states") != 0
            or re.fullmatch(r"sha256:[0-9a-f]{64}", base_digest) is None
            or re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", checkpoint_digest) is None
        ):
            raise ValueError("pre-update policy state receipt is incomplete or mismatched")
        changed = base_digest.removeprefix("sha256:") != checkpoint_digest.removeprefix("sha256:")
        base_policy_ranks.append(
            {
                "rank": rank,
                **_evidence_reference(path, base, file_sha256),
                "policy_state_sha256": base_digest,
            }
        )
        compared_ranks.append(
            {
                **checkpoint_rank,
                "base_policy_state_sha256": base_digest,
                "checkpoint_policy_state_sha256": checkpoint_digest,
                "policy_changed": changed,
            }
        )
    changed_policy_ranks = sum(rank["policy_changed"] for rank in compared_ranks)
    if changed_policy_ranks < 1:
        raise ValueError("step-1 checkpoint has no semantic policy delta from the frozen base")
    return {
        "schema": REWARD_CANARY_UPDATE_PROOF_SCHEMA,
        "source_plan_sha256": digest(plan),
        "checkpoint_manifest": _evidence_reference(manifest_path, manifest),
        "base_policy_ranks": base_policy_ranks,
        "initial_optimizer_step": 0,
        "final_optimizer_step": 1,
        "optimizer_updates": 1,
        "changed_optimizer_ranks": len(ranks),
        "delta_basis": "decoded_step1_optimizer_state_after_exact_no_resume_initialization",
        "changed_policy_ranks": changed_policy_ranks,
        "policy_delta_basis": (
            "native_rank_state_digest_before_first_optim_step_vs_sealed_step1_checkpoint"
        ),
        "ranks": compared_ranks,
    }


def _finite_scalar_rows(rows: object, *, label: str) -> list[dict]:
    if not hasattr(rows, "__iter__") or isinstance(rows, (str, bytes, Mapping)):
        raise ValueError(f"{label} scalar history is not an iterable of rows")
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} history row is not an object")
        clean = {}
        for key, value in row.items():
            if value is None:
                continue
            if (
                not isinstance(key, str)
                or re.fullmatch(r"[A-Za-z0-9_./@() -]+", key) is None
                or isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{label} history contains non-scalar or non-finite data")
            clean[key] = float(value)
        if clean:
            result.append(clean)
    if not result:
        raise ValueError(f"{label} scalar history is empty")
    return result


def _scalar_step_view(rows: list[dict], *, remote: bool) -> dict[int, dict[str, float]]:
    """Normalize W&B and local JSONL rows into exact application scalars by step."""
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        raw_step = row.get("_step") if remote else row.get("optimizer_step")
        if (
            isinstance(raw_step, bool)
            or not isinstance(raw_step, numbers.Real)
            or not math.isfinite(raw_step)
            or not float(raw_step).is_integer()
            or raw_step < 0
        ):
            raise ValueError("scalar history has an invalid optimizer step")
        step = int(raw_step)
        values = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
            and key not in ({"optimizer_step", "time"} if not remote else set())
        }
        target = result.setdefault(step, {})
        for key, value in values.items():
            if key in target and target[key] != value:
                raise ValueError("scalar history rewrites a metric within one optimizer step")
            target[key] = value
    return result


def _wandb_has_rich_payloads(run) -> bool:
    """Inspect every user-visible W&B payload surface, not a claimed boolean."""
    try:
        artifacts = list(run.logged_artifacts())
        files = [str(item.name) for item in run.files()]
        summary = dict(run.summary)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("W&B run cannot prove its scalar-only payload surface") from error
    allowed_system_files = {
        "config.yaml",
        "requirements.txt",
        "wandb-history.jsonl",
        "wandb-metadata.json",
        "wandb-summary.json",
    }
    if artifacts or any(name not in allowed_system_files for name in files):
        return True
    return any(
        not key.startswith("_")
        and value is not None
        and (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        )
        for key, value in summary.items()
    )


def reward_canary_wandb_history(
    plan: dict,
    episode_audit: dict,
    update_proof: dict,
    api=None,
) -> dict:
    """Reopen exact W&B scalars and cross-bind reward and optimizer evidence."""
    if not is_reward_canary(plan):
        raise ValueError("W&B evidence requires the exact reward canary")
    args = plan["arguments"]
    if api is None:
        import wandb

        api = wandb.Api()
    run = api.run(f"{args['wandb_entity']}/{args['wandb_project']}/{args['wandb_run_id']}")
    config = dict(run.config)
    if (
        (run.entity, run.project, run.id, run.name)
        != (
            args["wandb_entity"],
            args["wandb_project"],
            args["wandb_run_id"],
            args["name"],
        )
        or run.state != "finished"
        or config.get("plan_sha256") != digest(plan)
        or config.get("arguments") != args
    ):
        raise ValueError("W&B run identity/configuration is absent or changed")
    if _wandb_has_rich_payloads(run):
        raise ValueError("W&B run contains tables, artifacts, media, text, or rich payloads")
    remote = _finite_scalar_rows(run.scan_history(), label="W&B")
    if len(remote) > 10_000:
        raise ValueError("W&B scalar history exceeds its reviewed bound")
    local_path = Path(plan["output_root"]) / "metrics.jsonl"
    payload, local_file_sha256 = _snapshot(local_path)
    try:
        local_raw = [json.loads(line) for line in payload.splitlines()]
    except (TypeError, ValueError) as error:
        raise ValueError("local scalar history is invalid JSONL") from error
    local = _finite_scalar_rows(local_raw, label="local")
    remote_by_step = _scalar_step_view(remote, remote=True)
    local_by_step = _scalar_step_view(local, remote=False)
    if remote_by_step != local_by_step:
        raise ValueError("remote W&B and local scalar histories are not exactly cross-bound")
    step = remote_by_step.get(1, {})
    expected = {
        "trainer/global_step": 1.0,
        "reward/avg_raw_reward": float(episode_audit.get("train_reward_mean", math.nan)),
        "cyber/train/episodes": 8.0,
        "cyber/train/reward_mean": float(episode_audit.get("train_reward_mean", math.nan)),
        "cyber/train/reward_nonzero_count": float(
            episode_audit.get("train_nonzero_count", math.nan)
        ),
        "cyber/train/reward_population_variance": float(
            episode_audit.get("train_reward_population_variance", math.nan)
        ),
        "cyber/optimizer_step": float(update_proof.get("final_optimizer_step", math.nan)),
    }
    required_policy = {
        "policy/policy_loss",
        "policy/policy_lr",
        "policy/grad_norm",
    }
    update_ranks = update_proof.get("ranks")
    base_ranks = update_proof.get("base_policy_ranks")
    changed_policy_ranks = update_proof.get("changed_policy_ranks")
    if (
        episode_audit.get("schema") != REWARD_CANARY_EPISODE_AUDIT_SCHEMA
        or episode_audit.get("sha256")
        != digest({key: value for key, value in episode_audit.items() if key != "sha256"})
        or episode_audit.get("source_plan_sha256") != digest(plan)
        or update_proof.get("schema") != REWARD_CANARY_UPDATE_PROOF_SCHEMA
        or update_proof.get("sha256")
        != digest({key: value for key, value in update_proof.items() if key != "sha256"})
        or update_proof.get("source_plan_sha256") != digest(plan)
        or update_proof.get("initial_optimizer_step") != 0
        or update_proof.get("final_optimizer_step") != 1
        or update_proof.get("optimizer_updates") != 1
        or type(changed_policy_ranks) is not int
        or not 1 <= changed_policy_ranks <= 8
        or update_proof.get("policy_delta_basis")
        != "native_rank_state_digest_before_first_optim_step_vs_sealed_step1_checkpoint"
        or not isinstance(update_ranks, list)
        or not isinstance(base_ranks, list)
        or [rank.get("rank") for rank in update_ranks] != list(range(8))
        or [rank.get("rank") for rank in base_ranks] != list(range(8))
        or sum(rank.get("policy_changed") is True for rank in update_ranks) != changed_policy_ranks
        or any(
            rank.get("base_policy_state_sha256") != base_ranks[index].get("policy_state_sha256")
            or rank.get("checkpoint_policy_state_sha256") != rank.get("policy_state_sha256")
            or rank.get("policy_changed")
            is not (
                str(rank.get("base_policy_state_sha256", "")).removeprefix("sha256:")
                != str(rank.get("checkpoint_policy_state_sha256", "")).removeprefix("sha256:")
            )
            for index, rank in enumerate(update_ranks)
        )
        or set(remote_by_step) != {0, 1}
        or any(not math.isfinite(value) for value in expected.values())
        or any(step.get(key) != value for key, value in expected.items())
        or not required_policy <= set(step)
    ):
        raise ValueError("W&B history lacks exact reward, rollout, policy, or optimizer telemetry")
    return {
        "schema": REWARD_CANARY_WANDB_SCHEMA,
        "source_plan_sha256": digest(plan),
        "identity": {
            "entity": run.entity,
            "project": run.project,
            "run_id": run.id,
            "name": run.name,
        },
        "state": run.state,
        "remote_scalar_history": remote,
        "remote_scalar_history_sha256": "sha256:" + digest(remote),
        "remote_history_rows": len(remote),
        "local_metrics_path": str(local_path),
        "local_metrics_file_sha256": local_file_sha256,
        "local_scalar_history_sha256": "sha256:" + digest(local),
        "local_history_rows": len(local),
        "local_remote_step_history_sha256": "sha256:" + digest(local_by_step),
        "required_step_1_scalars": {**expected, **{key: step[key] for key in required_policy}},
        "episode_audit_sha256": episode_audit.get("sha256"),
        "optimizer_update_proof_sha256": update_proof.get("sha256"),
        "logged_artifact_count": 0,
        "rich_payload_count": 0,
    }


def seal_reward_canary_terminal(
    plan: dict,
    checkpoint_manifest_path: Path,
    release_evidence: Path,
    output: Path,
    *,
    wandb_api=None,
) -> dict:
    """Create the thin terminal receipt from independently reopenable evidence."""
    import torch

    from . import skyrl_rl_checkpoint as checkpoint

    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("reward canary terminal sealing must use pinned image user 1000:100")
    if torch.cuda.is_available():
        raise ValueError("reward canary terminal sealing is CPU-only after GPU release")
    if not is_reward_canary(plan):
        raise ValueError("terminal sealing requires the exact reward canary")
    root = Path(plan["output_root"])
    expected_paths = {
        "episode": root / "REWARD_CANARY_EPISODE_AUDIT.json",
        "update": root / "REWARD_CANARY_UPDATE_PROOF.json",
        "wandb": root / "WANDB_SCALAR_HISTORY.json",
        "terminal": root / "REWARD_CANARY_TERMINAL.json",
    }
    if output != expected_paths["terminal"] or any(
        path.exists() or path.is_symlink() for path in expected_paths.values()
    ):
        raise FileExistsError("reward canary terminal package path exists or is not exact")
    manifest, manifest_file_sha256 = checkpoint._json_snapshot(checkpoint_manifest_path)
    checkpoint.verify_manifest(manifest, check_files=True)
    if (
        manifest.get("source_plan") != plan
        or checkpoint_manifest_path != root / "RL_CHECKPOINT_STEP_1_MANIFEST.json"
        or release_evidence != root / "SOURCE_RELEASE.json"
        or manifest.get("release", {}).get("path") != str(release_evidence)
    ):
        raise ValueError("reward canary checkpoint/release source binding changed")
    release, release_file_sha256 = checkpoint._json_snapshot(release_evidence)
    completion_path = root / "NATIVE_TRAINING_COMPLETE.json"
    completion, completion_file_sha256 = checkpoint._json_snapshot(completion_path)
    checkpoint._validate_completion(completion, plan, 1)
    audit, audit_file_sha256 = checkpoint._controller_audit(plan, release)
    audit_path = Path(release["controller_audit_path"])
    pods = audit.get("pods")
    if (
        not isinstance(pods, list)
        or len(pods) != 1
        or pods[0].get("runtime_uid") != 1000
        or pods[0].get("runtime_gid") != 100
        or release.get("source_runtime_uid") != 1000
        or release.get("source_runtime_gid") != 100
    ):
        raise ValueError("source controller/release lacks observed Pod runtime user 1000:100")

    episode = reward_canary_episode_audit(plan)
    update = reward_canary_update_proof(plan, checkpoint_manifest_path, manifest)
    episode = _write(expected_paths["episode"], episode)
    update = _write(expected_paths["update"], update)
    wandb = reward_canary_wandb_history(plan, episode, update, api=wandb_api)
    wandb = _write(expected_paths["wandb"], wandb)
    terminal = {
        "schema": REWARD_CANARY_TERMINAL_SCHEMA,
        "status": "accepted",
        "source_run_name": plan["run_name"],
        "source_plan_sha256": digest(plan),
        "source_request_sha256": digest(job_request(plan)),
        "runtime_user": {"uid": pods[0]["runtime_uid"], "gid": pods[0]["runtime_gid"]},
        "native_completion": _evidence_reference(
            completion_path,
            completion,
            "sha256:" + completion_file_sha256.removeprefix("sha256:"),
        ),
        "episode_audit": _evidence_reference(expected_paths["episode"], episode),
        "optimizer_update_proof": _evidence_reference(expected_paths["update"], update),
        "wandb_scalar_history": _evidence_reference(expected_paths["wandb"], wandb),
        "checkpoint_manifest": _evidence_reference(
            checkpoint_manifest_path,
            manifest,
            "sha256:" + manifest_file_sha256.removeprefix("sha256:"),
        ),
        "source_controller_audit": _evidence_reference(
            audit_path,
            audit,
            "sha256:" + audit_file_sha256.removeprefix("sha256:"),
        ),
        "source_external_release": _evidence_reference(
            release_evidence,
            release,
            "sha256:" + release_file_sha256.removeprefix("sha256:"),
        ),
    }
    return _write(output, terminal)


def run(plan, plan_path):
    from .rl_runtime import run as supervised_run

    return supervised_run(plan, plan_path, backend=sys.modules[__name__])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--native", action="store_true")
    mode.add_argument("--engine-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        if args.engine_diagnostic:
            result = engine_diagnostic(plan)
            print(json.dumps({k: result[k] for k in ("status", "sha256")}))
        else:
            job_request(plan)
            _require_reward_canary_runtime_identity(plan)
            from .skyrl_promotion import require_production_runtime_identity

            require_production_runtime_identity(plan)
            if args.native:
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
                print(json.dumps({k: result[k] for k in ("status", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
