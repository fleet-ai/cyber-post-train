"""Pinned SkyRL SFT with scalar W&B telemetry and recoverable saves.

The FSDP path was inspected at SkyRL f5bc3b78. The Qwen3.8 Megatron-LoRA
qualification producer is committed at exact SkyRL revision
7e9356c8e02e7382e84b8484638baccdd1bbf680 and its immutable trainer image
passed the bounded CPU image qualification recorded in
configs/qualification/qwen38-lora-megatron-trainer-image-2026-09-20-v1.json.
The accepted production one-step checkpoint and zero-update BF16 merge/export
now admit three exact broad-data plans: the anchor and two learning-rate
controls. Every other broader treatment remains fail-closed. This module
deliberately keeps the native training loop and
optimizer. It adds
metadata-preserving tokenization, one-pass task-macro validation, scalar
telemetry, and checkpoint retention. HF export is a separate zero-step job:
the old step-318 inline export could time out before the final checkpoint.

No task text, token array, private traceback, or sample table is sent to W&B.
Run as a staged module with --plan and its externally bound --plan-sha256.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import numbers
import os
import re
import shutil
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path

SKYRL_REVISION = "f5bc3b78dfddfb352870d5d7430cd226e5785838"
# The evidence API changes the load-bearing worker bytes. Bind only the exact
# independently qualified source/image pair; the retired prior pair must not
# launch a producer whose evidence API it does not contain.
QWEN38_MEGATRON_SKYRL_REVISION: str | None = "7e9356c8e02e7382e84b8484638baccdd1bbf680"
QWEN38_MEGATRON_IMAGE: str | None = (
    "ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:"
    "7da4adba80d032509dba69fb4dd23bedca17fde3e2b88643815f80d6ee6c5317"
)
QWEN38_MEGATRON_IMAGE_RE = re.compile(
    r"ghcr\.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:[a-f0-9]{64}"
)
DENSE_SCHEMA = "cyber_sft_runtime_dense_v1"
DENSE_FORMAT = "pretokenized_assistant_segments_v1"
DENSE_EXCLUSION_REASONS = {
    "overlength_assistant_target",
    "overlength_required_previous_round",
}
PUBLIC_RUNTIME_STAGES = {
    "trainer_constructed",
    "tracker_initializing",
    "tracker_ready",
    "tracker_bypassed_for_setup_probe",
    "native_worker_initializing",
    "native_worker_ready",
    "device_backload_started",
    "device_ready",
}
QWEN38_QUALIFICATION_STAGES = {
    "forward_backward_started",
    "forward_backward_complete",
    "rank_lr_query_started",
    "rank_lr_query_complete",
    "worker_lr_validation_started",
    "worker_lr_validated",
    "lr_consensus_validated",
    "optimizer_update_started",
    "optimizer_update_returned",
    "gradient_validated",
    "step_evidence_ready",
    "checkpoint_finalization_started",
    "checkpoint_finalization_complete",
    "post_update_snapshot_started",
    "post_update_snapshot_complete",
    "qualification_receipt_preparation_started",
    "checkpoint_inventory_validated",
    "trainer_step_evidence_validated",
    "adapter_reconciliation_started",
    "adapter_reconciliation_complete",
    "source_revalidation_started",
    "source_revalidation_complete",
    "runtime_revalidation_complete",
    "source_immutability_validated",
    "source_plan_validated",
    "qualification_receipt_prepared",
    "terminal_result_validated",
    "wandb_finish_started",
    "wandb_finished",
    "trainer_shutdown_complete",
    "qualification_receipt_validated",
}
WATCHDOG_POLL_SECONDS = 60
WATCHDOG_STARTUP_SECONDS = 30 * 60
WATCHDOG_IDLE_SECONDS = 20 * 60
WATCHDOG_HARD_SECONDS = 8 * 60 * 60
WATCHDOG_DRAIN_SECONDS = 5 * 60
SOURCE_SHA256 = {
    "skyrl/train/sft_trainer.py": (
        "a5ef8a2e22de785b6760abffdd9353f1246a5898983b4d27b4aadd8089a3579a"
    ),
    "skyrl/train/config/sft_config.py": (
        "930812a153f8575b989d1492679f476d7a6495832bc08e623372d76efc407cd0"
    ),
    "skyrl/train/config/config.py": (
        "a3b36099c5308fc9bc658394f10cfff0da3f0cadcc067aa2b44e95870a609c90"
    ),
    "skyrl/train/dataset/collators.py": (
        "891fe6fdbeef5ede64514a265e8b573f9ca803115f64d705302ed235f4359e32"
    ),
    "skyrl/train/utils/callbacks.py": (
        "0b86c51d71db4a21f3fbd58b81e28a42ed00e1fe0aec5890a9f50a034c1be903"
    ),
    "skyrl/train/utils/tracking.py": (
        "51f6515201f654fd1c6a182bd62fa9840ee8c5f537ff3ae74e485b852faf7524"
    ),
    "skyrl/backends/skyrl_train/workers/worker.py": (
        "a3db942bcfb8f4a2f9cb6eae5717a2adde6699592ed469e164d60173e2bb2eab"
    ),
    "skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py": (
        "20710cfa5ac38bdc2333c05cc8d141ef03c9544ddfc73ffa58d719c01c355760"
    ),
    # The ba288751 image adds the already-approved opt-in Torch GDN constructor
    # patch to f5bc3b78. Its forward/loss implementation remains unchanged.
    "skyrl/backends/skyrl_train/workers/model_wrapper.py": (
        "015b532717df5a3492ebf42fa03982bdeaa820c658c9207689bc64f26c07f8fa"
    ),
    "skyrl/backends/skyrl_train/patches/fleet_qwen_torch_gdn.py": (
        "36414572e7dd70f2cc098f062e4f5c0544f72b58c3c3c71927ea67edb9d5f07e"
    ),
    "skyrl/backends/skyrl_train/utils/torch_utils.py": (
        "c145636be365b9513197cd0947f24ef24f0c007919fc04a512e14f6b35d2763e"
    ),
    "skyrl/backends/skyrl_train/workers/worker_dispatch.py": (
        "e2da7e5d351699b2e38f2d326b905af2b7bedf0c04e6799c3eed260db6ca45aa"
    ),
    "skyrl/backends/skyrl_train/distributed/dispatch.py": (
        "165db3891b96b3fb92a98727f9290989f9f67b0a0dc8398a622b249762c9fe2e"
    ),
    "skyrl/backends/skyrl_train/distributed/fsdp_strategy.py": (
        "a987a4bd509a43af1d27a92b1a933455f1f812e0da8a5ad733a68bcf2d67cd63"
    ),
    "skyrl/backends/skyrl_train/distributed/fsdp_utils.py": (
        "68fa165b08a6341430468ffa7961bab76f2c87cbb250df800a68a5e1505cb726"
    ),
    "skyrl/backends/skyrl_train/training_batch.py": (
        "7668f7007dc983549a661a166c0948d12c8db74fbd7f8f4d37889f8cbf6fa00f"
    ),
    "skyrl/backends/skyrl_train/workers/worker_utils.py": (
        "effc71cb05327d74bf9163ca3de14370c0bf8fa51302bfd1ff4331a8bdd94fd0"
    ),
    "skyrl/backends/skyrl_train/utils/ppo_utils.py": (
        "0d7ae59af59b16cc3d0b6593251bd3ab5dbfd421d9e3ae1bfde00783ba6323d7"
    ),
}

# Load-bearing bytes from exact committed revision
# 7e9356c8e02e7382e84b8484638baccdd1bbf680. This census is complete enough to
# build and verify the successor image. Dependency manifests and Dockerfile
# remain load-bearing because the earlier image failed when its build omitted
# the Megatron extra. The qualified pair above is accepted only for the exact
# one-step identities and production-qualified broad identity enforced by
# ``validate_plan``.
QWEN38_MEGATRON_SOURCE_SHA256: dict[str, str] = {
    "pyproject.toml": "ec1f6edaf83c3b5299d455c858409cdace4a7b3cc2948d208264ae0937077182",
    "uv.lock": "7843814ce42bdc1d34173138f00037108ea4181909e7fac5af14578c3c4551a6",
    "integrations/fleet_v2/rl1/Dockerfile": (
        "69c490cde076f531b2d5cd566aa6e9ab58f65b74b8c1fb741af3290c2bf9881c"
    ),
    "skyrl/train/sft_trainer.py": (
        "fbb12f4f0bd118848dc3c0eec43c37414f9b182204d578b6efe8d5d06e91eb26"
    ),
    "skyrl/train/config/sft_config.py": (
        "d5972bc61b3ec90e342a2395137942a22f5ef23294f662def51f95bacd65dd97"
    ),
    "skyrl/train/config/config.py": (
        "01ccf7c73f51dd168f06b94480b2c6938d8a5458e79ab4d044d6eecc3007e663"
    ),
    "skyrl/backends/skyrl_train/workers/worker_dispatch.py": (
        "5461b909155f87b0564efb30cafb56b971fa0e812f73a50307789ae6537cfb33"
    ),
    "skyrl/backends/skyrl_train/distributed/megatron/megatron_strategy.py": (
        "936580a50050f4a2dc82516e6d187909445cec79652a06ea8b10366280a69142"
    ),
    "skyrl/backends/skyrl_train/workers/megatron/adapter_store.py": (
        "198367c682105a930e800ab7cd6c4157654428dbf649161b9f8c45e4cb5134de"
    ),
    "skyrl/backends/skyrl_train/workers/megatron/lora_targets.py": (
        "fe3101241b16b37eb897675620b91e59f4c2b80b66e25d96040cd73b2565cdd7"
    ),
    "skyrl/backends/skyrl_train/workers/megatron/lora_evidence.py": (
        "f3452dc57d13b5c062ba05f43f40a74788057700c6b53a8849ec0337d21e649e"
    ),
    "skyrl/backends/skyrl_train/workers/megatron/megatron_model_wrapper.py": (
        "9cafe09ab2aa5755b7b707973e53df47b7bd51c335708468e701905f406e658a"
    ),
    "skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py": (
        "7e4a98279ba1b46db5ea3043f0b66486366656cbc61df7a3febbda5e2d83b174"
    ),
    "skyrl/backends/skyrl_train/workers/megatron/model_bridges.py": (
        "fa96f2d5563094d719975d0ee68eea1df20da34ac7d479e80bffd632fe3e00b8"
    ),
}
QWEN38_MEGATRON_SOURCE_CENSUS_SHA256 = (
    "a88c512b3c330b4cf21ef16318b98bcdb90a2d850936d1858d7b79343df94989"
)

QWEN38_LORA = {
    "type": "lora",
    "target_modules": "all-linear",
    "rank": 64,
    "alpha": 32,
    "init_method": "kaiming",
    "dropout": 0.0,
}
QWEN38_LORA_QUALIFICATION = {
    "schema": "qwen38_megatron_lora_one_step_gate_v1",
    "training_job_evidence": [
        "complete_target_census",
        "finite_forward_loss",
        "finite_nonzero_lora_gradients",
        "one_optimizer_update",
        "changed_adapter_tensors",
        "unchanged_frozen_base_tensors",
        "adapter_checkpoint",
        "wandb_scalar_run",
    ],
    "later_zero_step_evidence": [
        "adapter_checkpoint_reload",
        "deterministic_merge_and_export",
        "merged_model_and_tokenizer_reload",
    ],
    "training_success_is_acceptance": False,
    "accepted_for_production": False,
}

# This is the complete public handoff from the accepted production one-step
# checkpoint to its zero-update BF16 merge/export.  The file digest binds all
# receipt fields; the selected fields below keep the admission decision legible
# and make the source checkpoint and source plan links independently explicit.
# Broad training must reopen these exact bytes at runtime before model setup.
QWEN38_LORA_PRODUCTION_QUALIFICATION = {
    "schema": "qwen38_megatron_lora_production_gate_v1",
    "acceptance_handoff_sha256": (
        "8e1370ee28f8e366a3306446ab5a328b513e1d76fa0a317c9d70ca315fbbc1f2"
    ),
    "source_plan_sha256": ("33fc02619b07e9dd2bf21fc886ed4e41069151e32d70824e0a572223c86ce677"),
    "source_checkpoint_receipt_sha256": (
        "ab8d0e48566a4d4ddd4017fa14db8733122e5d8447cbcee535976c110c4354c2"
    ),
    "export_receipt": {
        "path": ("/mnt/sfs/jobs/chris-q38-lora-prod-exp-v1/QWEN38_LORA_MERGED_HF_EXPORT.json"),
        "file_sha256": ("bfeedcac2a82f25696e7d102de3b5f509ded9539a9a07d65c39a94a1c58bc7a2"),
        "receipt_sha256": ("237d90b34377599284c2358cd2611dfe2487a812a7179fc7460ed6ff3541ca76"),
        "schema": "cyber_qwen38_megatron_lora_merged_hf_export_v1",
        "output_root": "/mnt/sfs/jobs/chris-q38-lora-prod-exp-v1/merged-hf",
        "optimizer_step": 1,
        "optimizer_steps_executed": 0,
        "dtype": "BF16",
        "merge_method": "megatron_bridge_lora_merge_v1",
        "base_tensor_count": 1199,
        "merged_tensor_count": 1199,
        "tensor_bytes": 55_562_855_904,
        "source_base_unchanged": True,
        "source_checkpoint_unchanged": True,
        "adapter_checkpoint_reload_verified": True,
        "optimizer_resume_verified": True,
        "all_output_tensors_reopened_equal": True,
        "adapter_payloads_absent": True,
        "deterministic_merge": True,
        "merged_model_reload_verified": True,
        "finite_logits": True,
        "gpu_reload_verified": True,
    },
    "training_success_is_acceptance": False,
    "accepted_for_production": True,
}

# This is deliberately the complete identity of the one reviewed GPU canary,
# not a menu of values which happens to include it.  The qualified SkyRL image
# has not yet produced a checkpoint, so accepting a different dataset, model,
# recipe, resource envelope, or telemetry destination would turn the first paid
# run into an unreviewed experiment.  Compact canonical digests bind the large
# model inventory and dataset/task inventory without copying those lists into
# executable code; their human-readable roots and manifest identities remain
# explicit here.
QWEN38_LORA_ONE_STEP_PLAN = {
    "plan_keys": [
        "corpus_manifest_sha256",
        "datasets",
        "execution",
        "lora",
        "model",
        "output_root",
        "pause_after_step",
        "qualification_gate",
        "recipe",
        "run_name",
        "runtime_sha256",
        "schema",
        "skyrl_runtime",
        "split_manifest_sha256",
        "validation_mode",
        "wandb",
    ],
    "schema": DENSE_SCHEMA,
    "run_name": "chris-q38-lora-sft-c1-v10",
    "output_root": "/mnt/sfs/jobs/chris-q38-lora-sft-c1-v10",
    "model": {
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "weight_manifest_sha256": (
            "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
        ),
        "files_sha256": ("c4dab1bcd885ef426cc310853ef8f6fed451158cf83de17e8c625d703202940b"),
    },
    # The predecessor corpus overlaps two final-test task families.  This binds
    # the independently verified create-once V2 successor and its exact
    # payload-derived dataset/corpus identities and step count.
    "datasets_sha256": "219efad0257ff29b3057c91de88144542f8617a124079c5a20c79cf80b071f72",
    "split_manifest_sha256": (
        "sha256:b7b536940995a0d4b8674a4bdadd6240ef88dc1c07200a93f27b592ac8c046ea"
    ),
    "corpus_manifest_sha256": (
        "sha256:5db5600ac9f403fd147b2ecbf6068b514d075d07ac683d97bc83319c6e89d149"
    ),
    "recipe": {
        "epochs": 1,
        "batch_size": 1,
        "microbatch_per_gpu": 1,
        "nodes": 1,
        "gpus_per_node": 8,
        "lr": 3e-5,
        "max_length": 16384,
        "eval_interval": 0,
        "checkpoint_interval": 1,
        "keep_checkpoints": 3,
        "seed": 20260919,
        "max_steps": 866,
    },
    "pause_after_step": 1,
    "validation_mode": "task_outcomes_only",
    "execution": {
        "priority": "c1",
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    },
    "wandb": {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "group": "qwen38-lora-sft-goal-v1",
        "run_id": "chris-q38-lora-sft-c1-v10",
        "name": "chris-q38-lora-sft-c1-v10",
        "tags": [
            "qwen38",
            "lora",
            "teacher-sft",
            "rank64",
            "alpha32",
            "exact-model-gate",
            "planned-pause-step1",
            "task-outcomes-only",
            "runtime-path-repair",
            "native-dataset-contract-repair",
            "all-rank-lr-evidence-repair",
            "dev-init-contract-repair",
            "step-boundary-evidence-repair",
            "float32-lr-consensus-repair",
            "post-checkpoint-boundary-evidence-repair",
            "receipt-invariant-boundary-evidence-repair",
            "native-adapter-rank-coordinate-repair",
        ],
    },
}

# The exact development gate above completed one finite optimizer update and
# its separate zero-update merge/export/model-reload gate was independently
# accepted.  Production admission still starts with a create-once one-step
# canary.  Only launch identity and W&B identity change; every scientific,
# data, model, runtime and resource field remains byte-for-byte identical.
QWEN38_LORA_PRODUCTION_CANARY = {
    "run_name": "chris-q38-lora-prod-can-v1",
    "output_root": "/mnt/sfs/jobs/chris-q38-lora-prod-can-v1",
    "wandb": {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "group": "qwen38-lora-sft-goal-v1",
        "run_id": "chris-q38-lora-prod-can-v1",
        "name": "chris-q38-lora-prod-can-v1",
        "tags": [
            "qwen38",
            "lora",
            "teacher-sft",
            "rank64",
            "alpha32",
            "production-canary",
            "planned-pause-step1",
            "task-outcomes-only",
            "dev-training-gate-accepted",
            "dev-merge-reload-gate-accepted",
        ],
    },
}

# The first broad-data BF16 LoRA run is a single immutable treatment, not a
# newly opened parameter menu.  Its model and runtime match the accepted
# production canary, while its already-reviewed anchor changes only the corpus,
# context, batch, stopping rule, checkpoint cadence, and create-once identities
# recorded here.  Compact digests bind the full model and 496-task data
# inventories without copying either inventory into executable code.
QWEN38_LORA_BROAD_FULL_PLAN = {
    "plan_keys": [
        "corpus_manifest_sha256",
        "datasets",
        "execution",
        "lora",
        "model",
        "output_root",
        "qualification_gate",
        "recipe",
        "run_name",
        "runtime_sha256",
        "schema",
        "skyrl_runtime",
        "split_manifest_sha256",
        "validation_mode",
        "wandb",
    ],
    "schema": DENSE_SCHEMA,
    "run_name": "chris-q38-lora-sft-a1-v1",
    "output_root": "/mnt/sfs/jobs/chris-q38-lora-sft-a1-v1",
    "model": {
        "repo": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "root": "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2",
        "weight_manifest_sha256": (
            "sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352"
        ),
        "files_sha256": "c4dab1bcd885ef426cc310853ef8f6fed451158cf83de17e8c625d703202940b",
    },
    "datasets_sha256": "c8fa0d8e21500269afce7437241b4a9fd349d4714cbedeee6d39452dc0dd359e",
    "split_manifest_sha256": (
        "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
    ),
    "corpus_manifest_sha256": (
        "sha256:a8d08609991d7960cdcdab826bd07c3216bfcd3aa8f6fb33ab7a5fd29648d6f5"
    ),
    "recipe": {
        "epochs": 1,
        "batch_size": 8,
        "microbatch_per_gpu": 1,
        "nodes": 1,
        "gpus_per_node": 8,
        "lr": 3e-5,
        "max_length": 32768,
        "eval_interval": 0,
        "checkpoint_interval": 20,
        "keep_checkpoints": 3,
        "seed": 20260919,
        "max_steps": 1837,
    },
    "pause_after_step": None,
    "validation_mode": "task_outcomes_only",
    "execution": {
        "priority": "c1",
        "resources": {
            "cpu_request": "64",
            "cpu_limit": "64",
            "memory_request": "512Gi",
            "memory_limit": "768Gi",
        },
    },
    "wandb": {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "group": "qwen38-lora-sft-goal-v1",
        "run_id": "chris-q38-lora-sft-a1-v1",
        "name": "chris-q38-lora-sft-a1-v1",
        "tags": [
            "qwen38",
            "lora",
            "teacher-sft",
            "rank64",
            "alpha32",
            "57m-unique-supervised-tokens",
            "32k",
            "anchor",
            "task-outcomes-only",
        ],
    },
}


def _reviewed_broad_lora_lr_plan(
    *, run_name: str, learning_rate: float, rate_tag: str, purpose_tag: str
) -> dict:
    """Construct one reviewed LR-only broad treatment from the exact anchor."""
    return {
        **QWEN38_LORA_BROAD_FULL_PLAN,
        "run_name": run_name,
        "output_root": f"/mnt/sfs/jobs/{run_name}",
        "recipe": {
            **QWEN38_LORA_BROAD_FULL_PLAN["recipe"],
            "lr": learning_rate,
        },
        "wandb": {
            **QWEN38_LORA_BROAD_FULL_PLAN["wandb"],
            "run_id": run_name,
            "name": run_name,
            "tags": [
                "qwen38",
                "lora",
                "teacher-sft",
                "rank64",
                "alpha32",
                "57m-unique-supervised-tokens",
                "32k",
                rate_tag,
                purpose_tag,
                "task-outcomes-only",
            ],
        },
    }


QWEN38_LORA_BROAD_LOWER_LR_PLAN = _reviewed_broad_lora_lr_plan(
    run_name="chris-q38-lora-lr1-v1",
    learning_rate=1e-5,
    rate_tag="lr1e-5",
    purpose_tag="lower-rate-control",
)
QWEN38_LORA_BROAD_UPPER_LR_PLAN = _reviewed_broad_lora_lr_plan(
    run_name="chris-q38-lora-lr100-v1",
    learning_rate=1e-4,
    rate_tag="lr1e-4",
    purpose_tag="upper-rate-exploration",
)
QWEN38_LORA_BROAD_FULL_PLANS = {
    plan["run_name"]: plan
    for plan in (
        QWEN38_LORA_BROAD_FULL_PLAN,
        QWEN38_LORA_BROAD_LOWER_LR_PLAN,
        QWEN38_LORA_BROAD_UPPER_LR_PLAN,
    )
}


def qwen38_megatron_binding() -> tuple[str, str, dict[str, str]]:
    """Return the one reviewed source/image pair, or fail closed.

    A repository-qualified image name is not enough: submission must bind the
    exact digest built from the same reviewed source commit and must retain a
    non-empty census of load-bearing source bytes for in-image preflight.
    """
    if (
        not isinstance(QWEN38_MEGATRON_SKYRL_REVISION, str)
        or re.fullmatch(r"[a-f0-9]{40}", QWEN38_MEGATRON_SKYRL_REVISION) is None
        or not isinstance(QWEN38_MEGATRON_IMAGE, str)
        or QWEN38_MEGATRON_IMAGE_RE.fullmatch(QWEN38_MEGATRON_IMAGE) is None
        or not QWEN38_MEGATRON_SOURCE_SHA256
        or hashlib.sha256(
            json.dumps(
                QWEN38_MEGATRON_SOURCE_SHA256,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        != QWEN38_MEGATRON_SOURCE_CENSUS_SHA256
        or any(
            not isinstance(path, str)
            or not path
            or re.fullmatch(r"[a-f0-9]{64}", source_digest) is None
            for path, source_digest in QWEN38_MEGATRON_SOURCE_SHA256.items()
        )
    ):
        raise ValueError("Qwen3.8 Megatron source/image qualification binding is unresolved")
    return (
        QWEN38_MEGATRON_SKYRL_REVISION,
        QWEN38_MEGATRON_IMAGE,
        dict(QWEN38_MEGATRON_SOURCE_SHA256),
    )


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unsigned_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _qwen38_lora_one_step_identity(plan: dict) -> dict:
    """Return every reviewed scientific/resource field of the paid canary."""
    # The immutable plan file deliberately does not contain its own digest.
    # The runtime verifies that file against ``--plan-sha256`` and then adds
    # ``plan_sha256`` for receipts.  Keep that transport/evidence field out of
    # the reviewed scientific identity so the same exact plan is accepted both
    # immediately before and immediately after runtime enrichment.
    source_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    model = source_plan.get("model", {})
    execution = source_plan.get("execution", {})
    return {
        "plan_keys": sorted(source_plan),
        "schema": source_plan.get("schema"),
        "run_name": source_plan.get("run_name"),
        "output_root": source_plan.get("output_root"),
        "model": {
            "repo": model.get("repo"),
            "revision": model.get("revision"),
            "root": model.get("root"),
            "weight_manifest_sha256": model.get("weight_manifest_sha256"),
            "files_sha256": _unsigned_digest(model.get("files", [])),
        },
        "datasets_sha256": _unsigned_digest(source_plan.get("datasets", {})),
        "split_manifest_sha256": source_plan.get("split_manifest_sha256"),
        "corpus_manifest_sha256": source_plan.get("corpus_manifest_sha256"),
        "recipe": source_plan.get("recipe"),
        "pause_after_step": source_plan.get("pause_after_step"),
        "validation_mode": source_plan.get("validation_mode"),
        "execution": {
            "priority": execution.get("priority"),
            "resources": execution.get("resources"),
        },
        "wandb": source_plan.get("wandb"),
    }


def qwen38_lora_one_step_plan_binding() -> dict:
    """Return the exact canary identity only after its corpus is published."""
    if (
        QWEN38_LORA_ONE_STEP_PLAN["datasets_sha256"] is None
        or QWEN38_LORA_ONE_STEP_PLAN["corpus_manifest_sha256"] is None
        or QWEN38_LORA_ONE_STEP_PLAN["recipe"]["max_steps"] is None
    ):
        raise ValueError("Qwen3.8 one-step leak-free corpus binding is unresolved")
    return QWEN38_LORA_ONE_STEP_PLAN


def qwen38_lora_production_canary_plan_binding() -> dict:
    """Return the one production canary admitted by the accepted dev gates."""
    source = qwen38_lora_one_step_plan_binding()
    return {
        **source,
        "run_name": QWEN38_LORA_PRODUCTION_CANARY["run_name"],
        "output_root": QWEN38_LORA_PRODUCTION_CANARY["output_root"],
        "wandb": QWEN38_LORA_PRODUCTION_CANARY["wandb"],
    }


def qwen38_lora_broad_full_plan_binding(
    run_name: str = QWEN38_LORA_BROAD_FULL_PLAN["run_name"],
) -> dict:
    """Return one exact reviewed broad plan after the production receipt chain."""
    qualification = QWEN38_LORA_PRODUCTION_QUALIFICATION
    export = qualification["export_receipt"]
    if (
        qualification.get("accepted_for_production") is not True
        or not re.fullmatch(r"[a-f0-9]{64}", qualification.get("source_plan_sha256", ""))
        or not re.fullmatch(
            r"[a-f0-9]{64}", qualification.get("source_checkpoint_receipt_sha256", "")
        )
        or not isinstance(export, dict)
        or not re.fullmatch(r"[a-f0-9]{64}", export.get("file_sha256", ""))
        or not re.fullmatch(r"[a-f0-9]{64}", export.get("receipt_sha256", ""))
    ):
        raise ValueError("Qwen3.8 broad LoRA production qualification is unresolved")
    try:
        return QWEN38_LORA_BROAD_FULL_PLANS[run_name]
    except KeyError as exc:
        raise ValueError("Qwen3.8 broad LoRA plan is not reviewed") from exc


def write_receipt(path: Path, value: dict, *, replace: bool = False) -> None:
    payload = {**value, "receipt_sha256": _unsigned_digest(value)}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp") if replace else path
    with temporary.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        temporary.replace(path)


def _verified_file_observation(path: Path, expected: str) -> dict:
    """Read one immutable file once and return its verified public metadata."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("immutable file missing or digest mismatch")
    before = path.stat()
    observed = digest(path)
    after = path.stat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if observed != expected.removeprefix("sha256:") or any(
        getattr(before, field) != getattr(after, field) for field in stable_fields
    ):
        raise ValueError("immutable file missing or digest mismatch")
    return {"bytes": after.st_size, "sha256": observed}


def _verified_json_file(path: Path, expected: str) -> dict:
    """Parse the same stable bytes whose exact file digest was verified."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("immutable JSON file missing or digest mismatch")
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if hashlib.sha256(payload).hexdigest() != expected.removeprefix("sha256:") or any(
        getattr(before, field) != getattr(after, field) for field in stable_fields
    ):
        raise ValueError("immutable JSON file missing or digest mismatch")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("immutable JSON file is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("immutable JSON file is invalid")
    return value


def _validate_qwen38_production_export_receipt(
    receipt: dict,
    qualification: dict = QWEN38_LORA_PRODUCTION_QUALIFICATION,
) -> None:
    """Validate the public receipt fields that authorize the exact broad plan."""
    if not isinstance(receipt, dict) or not isinstance(qualification, dict):
        raise ValueError("Qwen3.8 production export qualification is invalid")
    export = qualification.get("export_receipt")
    if not isinstance(export, dict):
        raise ValueError("Qwen3.8 production export qualification is invalid")
    expected = {key: value for key, value in export.items() if key not in {"path", "file_sha256"}}
    expected.update(
        {
            "source_checkpoint_receipt_sha256": qualification.get(
                "source_checkpoint_receipt_sha256"
            ),
            "source_plan_sha256": qualification.get("source_plan_sha256"),
        }
    )
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if any(receipt.get(key) != value for key, value in expected.items()) or receipt.get(
        "receipt_sha256"
    ) != _unsigned_digest(unsigned):
        raise ValueError("Qwen3.8 production export receipt differs from its accepted binding")


def _verify_qwen38_production_qualification(plan: dict) -> None:
    """Reopen the exact accepted public receipt before broad model setup."""
    # A recovery has a deliberately fresh run/output/W&B identity.  Its sealed
    # checkpoint carries the immutable production-qualified source plan, and
    # ``validate_plan`` has already proved that the successor preserves every
    # scientific, data, model and topology field from that source.  Reopen the
    # reviewed source identity here instead of trying to find the fresh
    # successor name in the create-once broad-plan table.
    source_plan = plan["recovery"]["checkpoint"]["source_plan"] if "recovery" in plan else plan
    if (
        _qwen38_lora_one_step_identity(source_plan)
        != qwen38_lora_broad_full_plan_binding(source_plan.get("run_name", ""))
        or plan.get("qualification_gate") != QWEN38_LORA_PRODUCTION_QUALIFICATION
    ):
        raise ValueError("Qwen3.8 broad LoRA plan lost its production qualification binding")
    reference = plan["qualification_gate"]["export_receipt"]
    receipt = _verified_json_file(Path(reference["path"]), reference["file_sha256"])
    _validate_qwen38_production_export_receipt(receipt)


def _checked_file(path: Path, expected: str) -> None:
    _verified_file_observation(path, expected)


def _is_qwen38_lora(plan: dict) -> bool:
    return "lora" in plan and plan.get("model", {}).get("repo") == "Qwen/Qwen3.8-27B"


def _is_qwen38_lora_one_step_gate(plan: dict) -> bool:
    """Distinguish the historical one-step evidence producer from full SFT."""
    return _is_qwen38_lora(plan) and plan.get("qualification_gate") == QWEN38_LORA_QUALIFICATION


def _reconcile_worker_learning_rates(values: list, *, expected_ranks: int) -> float:
    """Require one finite, positive, identical optimizer LR from every rank."""
    if type(expected_ranks) is not int or expected_ranks <= 0:
        raise ValueError("learning-rate rank count is invalid")
    if not isinstance(values, list) or len(values) != expected_ranks:
        raise ValueError("learning-rate evidence does not cover every rank")
    rates = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError("learning-rate evidence is not numeric")
        rate = float(value)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("learning-rate evidence is not finite and positive")
        rates.append(rate)
    if any(rate != rates[0] for rate in rates[1:]):
        raise ValueError("learning-rate evidence differs across ranks")
    return rates[0]


def _qwen38_policy_learning_rate(dispatch, plan: dict) -> float:
    """Read the impending optimizer LR from every exact Qwen3.8 policy rank."""
    if not _is_qwen38_lora(plan):
        raise ValueError("all-rank learning-rate evidence is Qwen3.8-LoRA-only")
    expected = plan["recipe"]["nodes"] * plan["recipe"]["gpus_per_node"]
    groups = getattr(dispatch, "_actor_groups", None)
    group = groups.get("policy") if isinstance(groups, dict) else None
    actor_infos = getattr(group, "actor_infos", None)
    if not isinstance(actor_infos, list) or len(actor_infos) != expected:
        raise ValueError("policy actor group does not contain every planned rank")
    refs = group.async_run_ray_method("pass_through", "get_lr")
    if not isinstance(refs, list) or len(refs) != expected:
        raise ValueError("learning-rate query did not dispatch to every rank")
    import ray

    return _reconcile_worker_learning_rates(ray.get(refs), expected_ranks=expected)


def _scalar_sft_forward_backward(dispatch, batch):
    """Run SFT without constructing unused per-token worker outputs."""
    return dispatch.forward_backward(
        "policy",
        batch,
        loss_fn="cross_entropy",
        return_per_token_outputs=False,
    )


def validate_plan(plan: dict, *, check_files: bool = True) -> None:
    if plan.get("schema") not in ("cyber_sft_runtime_v2", DENSE_SCHEMA):
        raise ValueError("unsupported SFT runtime plan")
    recipe = plan["recipe"]
    for key in (
        "epochs",
        "batch_size",
        "microbatch_per_gpu",
        "nodes",
        "gpus_per_node",
        "max_length",
        "eval_interval",
        "checkpoint_interval",
        "keep_checkpoints",
        "max_steps",
    ):
        minimum = 0 if key == "eval_interval" else 1
        if type(recipe[key]) is not int or recipe[key] < minimum:
            raise ValueError("recipe counts must be positive integers; eval_interval may be zero")
    if type(recipe["seed"]) is not int or not 0 <= recipe["seed"] < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if (
        type(recipe["lr"]) not in {int, float}
        or not math.isfinite(recipe["lr"])
        or not 0 < recipe["lr"] <= 1e-4
    ):
        raise ValueError("learning rate outside reviewed SFT range")
    data_parallel_size = recipe["nodes"] * recipe["gpus_per_node"]
    if _is_qwen38_lora(plan):
        # The qualification topology is TP=8, so eight model-parallel ranks
        # collectively consume one sample. They are not eight data replicas.
        if data_parallel_size < 8 or data_parallel_size % 8:
            raise ValueError("Qwen3.8 Megatron world size must be divisible by TP=8")
        data_parallel_size //= 8
    if recipe["batch_size"] % (data_parallel_size * recipe["microbatch_per_gpu"]):
        raise ValueError("global batch must divide evenly across GPU microbatches")
    if recipe["eval_interval"] and recipe["checkpoint_interval"] != recipe["eval_interval"]:
        raise ValueError("every periodic validation needs a corresponding saved checkpoint")
    if plan.get("resume_from"):
        raise ValueError(
            "v2 arms start from their exact base; resume needs a separately bound plan"
        )
    if "recovery" in plan:
        from training.recovery import validate

        validate(plan, check_files=check_files)
    if "pause_after_step" in plan:
        pause = plan["pause_after_step"]
        recovery = plan.get("recovery", {})
        start = recovery.get("checkpoint", {}).get("optimizer_step", 0)
        if (
            type(pause) is not int
            or not start < pause < recipe["max_steps"]
            or recovery.get("mode") == "validate"
        ):
            raise ValueError("planned pause must follow new work and precede full completion")
    training_only = plan.get("validation_mode") == "task_outcomes_only"
    if training_only != (recipe["eval_interval"] == 0):
        raise ValueError("task-outcome evaluation and zero CE interval must be selected together")
    if training_only and set(plan["datasets"]) != {"train"}:
        raise ValueError("task-outcome training must not load a teacher-reference dev artifact")
    if not training_only and set(plan["datasets"]) != {"train", "dev"}:
        raise ValueError("teacher CE validation requires train and dev artifacts")
    if training_only and plan.get("recovery", {}).get("mode") == "validate":
        raise ValueError("zero-step CE validation is unavailable without a dev artifact")
    train = plan["datasets"]["train"]
    dev = plan["datasets"].get("dev")
    for dataset in plan["datasets"].values():
        if not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", dataset["sha256"]):
            raise ValueError("dataset digest missing or invalid")
        if any(not isinstance(k, str) or not k.strip() for k in dataset["task_keys"]):
            raise ValueError("task identities must be nonempty strings")
    if plan["schema"] == DENSE_SCHEMA:
        if train.get("format") != DENSE_FORMAT:
            raise ValueError("dense training requires explicitly masked segments")
        for key in (
            "supervised_tokens",
            "assistant_responses",
            "source_sessions",
            "source_total_assistant_responses",
        ):
            if type(train.get(key)) is not int or train[key] <= 0:
                raise ValueError("dense teacher source and target totals must be bound")
        excluded = train.get("excluded_assistant_responses")
        if (
            type(excluded) is not int
            or excluded < 0
            or train["assistant_responses"] + excluded != train["source_total_assistant_responses"]
        ):
            raise ValueError("dense eligible and excluded response totals must exhaust the source")
    elif train.get("format") == DENSE_FORMAT:
        raise ValueError("dense teacher data requires its separately versioned plan")
    # Pinned worker per-row diagnostics are suffix-only. Keep the bound dev set
    # on the qualified contiguous last-assistant path, never sparse-mask eval.
    if dev and dev.get("format") == DENSE_FORMAT:
        raise ValueError("held-out evaluation must remain contiguous last-assistant windows")
    train_keys = set(train["task_keys"])
    dev_keys = set(dev["task_keys"]) if dev else set()
    if dev and (not dev_keys or len(dev_keys) != len(dev["task_keys"])):
        raise ValueError("validation must bind distinct nonempty held-out tasks")
    if not train_keys or train_keys & dev_keys:
        raise ValueError("train/dev task overlap or empty training task set")
    if (dev and train["path"] == dev["path"]) or any(
        type(x["rows"]) is not int or x["rows"] <= 0 for x in plan["datasets"].values()
    ):
        raise ValueError("every selected dataset must be nonempty and paths must be distinct")
    if recipe["max_steps"] != math.ceil(train["rows"] / recipe["batch_size"]) * recipe["epochs"]:
        raise ValueError("max_steps must equal complete epochs with the native kept tail batch")
    model = plan["model"]
    if _is_qwen38_lora(plan):
        source_commit, image, source_files = qwen38_megatron_binding()
        one_step_plans = (
            qwen38_lora_one_step_plan_binding(),
            qwen38_lora_production_canary_plan_binding(),
        )
        identity = _qwen38_lora_one_step_identity(plan)
        broad_identity = (
            qwen38_lora_broad_full_plan_binding(plan["run_name"])
            if plan.get("run_name") in QWEN38_LORA_BROAD_FULL_PLANS
            else None
        )
        if identity in one_step_plans:
            expected_qualification = QWEN38_LORA_QUALIFICATION
        elif identity == broad_identity:
            expected_qualification = QWEN38_LORA_PRODUCTION_QUALIFICATION
        elif "recovery" in plan:
            # A continuation needs a fresh run/output/W&B identity, so it
            # cannot equal one of the frozen broad-plan identities.  Admit it
            # only when its sealed source independently reopens as one of
            # those exact production-qualified plans; recovery.validate above
            # already requires every scientific/data/topology field to remain
            # identical to that source.
            source_plan = plan["recovery"]["checkpoint"]["source_plan"]
            source_identity = _qwen38_lora_one_step_identity(source_plan)
            source_broad_identity = (
                qwen38_lora_broad_full_plan_binding(source_plan.get("run_name", ""))
                if source_plan.get("run_name") in QWEN38_LORA_BROAD_FULL_PLANS
                else None
            )
            expected_qualification = (
                QWEN38_LORA_PRODUCTION_QUALIFICATION
                if source_identity == source_broad_identity
                else None
            )
        else:
            expected_qualification = None
        runtime = plan.get("skyrl_runtime")
        expected_runtime = {
            "source_commit": source_commit,
            "source_files_sha256": source_files,
        }
        if (
            plan["lora"] != QWEN38_LORA
            or runtime != expected_runtime
            or plan.get("execution", {}).get("image") != image
            or expected_qualification is None
            or plan.get("qualification_gate") != expected_qualification
        ):
            raise ValueError(
                "Qwen3.8 LoRA requires the exact digest-bound one-step or "
                "production-qualified broad Megatron plan"
            )
    elif "lora" in plan:
        lora = plan["lora"]
        if (
            model["repo"] != "zai-org/GLM-5.3"
            or not isinstance(lora, dict)
            or set(lora) != {"rank", "alpha"}
            or type(lora["rank"]) is not int
            or not 1 <= lora["rank"] <= 64
            or type(lora["alpha"]) is not int
            or lora["alpha"] <= 0
            or not re.fullmatch(r"[a-f0-9]{64}", plan.get("glm_runtime_sha256", ""))
            or not re.fullmatch(r"sha256:[a-f0-9]{64}", model.get("tokenizer_manifest_sha256", ""))
        ):
            raise ValueError("GLM LoRA needs an exact bounded adapter/runtime identity")
        if check_files:
            from training import glm_runtime

            _checked_file(Path(glm_runtime.__file__), plan["glm_runtime_sha256"])
    elif model["repo"] == "zai-org/GLM-5.3":
        raise ValueError("full GLM requires the separately qualified LoRA loader")
    if not re.fullmatch(r"[a-f0-9]{40}", model["revision"]) or not model["files"]:
        raise ValueError("exact model revision and file bindings required")
    if not {"config.json", "tokenizer_config.json"}.issubset({x["path"] for x in model["files"]}):
        raise ValueError("model config/tokenizer config bindings required")
    for item in model["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("model inventory path escapes model root")
    for key in ("project", "entity", "group", "run_id", "name"):
        if not isinstance(plan["wandb"][key], str) or not plan["wandb"][key].strip():
            raise ValueError("complete W&B run identity required")
    if check_files:
        for spec in plan["datasets"].values():
            _checked_file(Path(spec["path"]), spec["sha256"])
        for item in model["files"]:
            _checked_file(Path(model["root"]) / item["path"], item["sha256"])
        # Optional full model staging proof, beyond runtime sidecar bindings.
        if plan.get("staged_receipt"):
            _checked_file(Path(plan["staged_receipt"]["path"]), plan["staged_receipt"]["sha256"])


def validate_runtime_sources(plan: dict | None = None, root: Path | None = None) -> None:
    if root is None:
        spec = importlib.util.find_spec("skyrl")
        if spec is None or not spec.submodule_search_locations:
            raise ValueError("pinned SkyRL package is unavailable")
        root = Path(next(iter(spec.submodule_search_locations))).parent
    sources = qwen38_megatron_binding()[2] if plan and _is_qwen38_lora(plan) else SOURCE_SHA256
    for path, expected in sources.items():
        _checked_file(root / path, expected)


def _validate_entrypoint_sources(plan: dict, *, verify_qwen_files: bool) -> None:
    """Validate a staged plan without duplicating Qwen model hash passes."""
    qwen38_lora = _is_qwen38_lora(plan)
    validate_plan(plan, check_files=not qwen38_lora)
    if qwen38_lora:
        if plan.get("qualification_gate") == QWEN38_LORA_PRODUCTION_QUALIFICATION:
            _verify_qwen38_production_qualification(plan)
        if verify_qwen_files:
            _qwen38_source_inventory(plan)
            _qwen38_runtime_source_inventory(plan)
    else:
        validate_runtime_sources(plan)


def sft_overrides(plan: dict) -> dict:
    r, w = plan["recipe"], plan["wandb"]
    output = Path(plan["output_root"])
    training_only = plan.get("validation_mode") == "task_outcomes_only"
    options = {
        "strategy": "fsdp",
        "model.path": plan["model"]["root"],
        "max_length": r["max_length"],
        "train_on_what": (
            "all_assistant_messages" if plan["schema"] == DENSE_SCHEMA else "last_assistant_message"
        ),
        "batch_size": r["batch_size"],
        "micro_train_batch_size_per_gpu": r["microbatch_per_gpu"],
        "optimizer_config.lr": r["lr"],
        "optimizer_config.scheduler": "constant_with_warmup",
        "optimizer_config.num_warmup_steps": 0,
        "placement.num_nodes": r["nodes"],
        "placement.num_gpus_per_node": r["gpus_per_node"],
        "sequence_parallel_size": 1,
        "model_config_kwargs.fleet_force_qwen35_torch_gdn": True,
        "logger": "wandb",
        "project_name": w["project"],
        "run_name": w["name"],
        "tags": w.get("tags", []),
        "seed": r["seed"],
        "dataset_name": plan["datasets"]["train"]["path"],
        "dataset_split": "train",
        "eval_before_train": not training_only,
        "eval_interval": r["eval_interval"],
        "ckpt_path": str(output / "checkpoints"),
        "ckpt_interval": r["checkpoint_interval"],
        # This module retains latest N plus best; native deletion must be disabled.
        "max_ckpts_to_keep": -1,
        "hf_save_interval": 0,
        "cache_dir": str(output / "tokenized_cache"),
        "disable_cache": True,
        "num_epochs": r["epochs"],
        "max_training_steps": r["max_steps"],
        "num_workers": 0,
        "dataloader_num_workers": 0,
    }
    if not training_only:
        options.update(
            {
                "eval_dataset_name": plan["datasets"]["dev"]["path"],
                "eval_dataset_split": "validation",
            }
        )
    if plan["schema"] == DENSE_SCHEMA:
        options.update(
            {
                "fsdp_config.cpu_offload": False,
                "optimizer_config.offload_after_step": False,
            }
        )
    if _is_qwen38_lora(plan):
        options.pop("model_config_kwargs.fleet_force_qwen35_torch_gdn")
        lora = plan["lora"]
        options.update(
            {
                "strategy": "megatron",
                "language_model_only": True,
                "megatron_config.tensor_model_parallel_size": 8,
                "megatron_config.pipeline_model_parallel_size": 1,
                "megatron_config.context_parallel_size": 1,
                "megatron_config.lora_config.lora_type": lora["type"],
                "megatron_config.lora_config.merge_lora": True,
                "model.lora.rank": lora["rank"],
                "model.lora.alpha": lora["alpha"],
                "model.lora.dropout": lora["dropout"],
                "model.lora.init_method": lora["init_method"],
                "model.lora.target_modules": lora["target_modules"],
                "optimizer_config.weight_decay": 0.01,
                "optimizer_config.max_grad_norm": 1.0,
                "remove_microbatch_padding": True,
                "use_sequence_packing": False,
            }
        )
    elif "lora" in plan:
        from training.glm_runtime import TARGETS

        options.pop("model_config_kwargs.fleet_force_qwen35_torch_gdn")
        options.update(
            {
                "model.lora.rank": plan["lora"]["rank"],
                "model.lora.alpha": plan["lora"]["alpha"],
                "model.lora.dropout": 0.0,
                "model.lora.target_modules": list(TARGETS),
                "remove_microbatch_padding": False,
                "use_sequence_packing": False,
                "fsdp_config.cpu_offload": False,
                "optimizer_config.offload_after_step": False,
            }
        )
    return options


def build_runtime_configs(plan: dict):
    """Use native SFT configuration, with explicit single-policy dense residency."""
    from skyrl.train.config.sft_config import SFTConfig, build_skyrl_config_for_sft

    cfg = SFTConfig.from_cli_overrides(sft_overrides(plan))
    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    if "lora" in plan and not _is_qwen38_lora(plan):
        # SFTConfig has no flash_attn field; this is an explicit native bridge
        # setting, not an ignored CLI override. The GLM loader uses HF SDPA.
        skyrl_cfg.trainer.flash_attn = False
    if plan["schema"] == DENSE_SCHEMA or ("lora" in plan and not _is_qwen38_lora(plan)):
        # The native bridge disables colocate_all but inherits the RL default
        # colocate_policy_ref=True. This arm has no reference/inference actor.
        skyrl_cfg.trainer.placement.colocate_policy_ref = False
        assert not skyrl_cfg.trainer.placement.colocate_all
        assert not skyrl_cfg.trainer.policy.fsdp_config.cpu_offload
        assert not skyrl_cfg.trainer.policy.optimizer_config.offload_after_step
    return cfg, skyrl_cfg


def tokenize_rows(
    rows: list[dict], spec: dict, tokenizer, tokenize, *, max_length: int
) -> list[dict]:
    """Apply the pinned native target mask and reject any lost/truncated row."""
    if len(rows) != spec["rows"] or {r["task_key"] for r in rows} != set(spec["task_keys"]):
        raise ValueError("dataset row count or task identity mismatch")
    if len({r["window_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate training window")
    result = []
    for row in rows:
        if (
            not isinstance(row["messages"], list)
            or not row["messages"]
            or row["messages"][-1]["role"] != "assistant"
        ):
            raise ValueError("window must contain native messages and end at assistant target")
        if any(
            message.get(key)
            for message in row["messages"]
            for key in ("reasoning_content", "reasoning", "thinking")
        ):
            raise ValueError("visible-action objective must not contain private reasoning fields")
        item = tokenize(row, tokenizer, max_length=None)
        if item is None or not 0 < item["num_actions"] < len(item["input_ids"]) <= max_length:
            raise ValueError("empty target or overlength window; truncation is prohibited")
        if item["loss_mask"] != [1] * item["num_actions"]:
            raise ValueError("expected contiguous last-assistant target mask")
        if "token_count" in row and row["token_count"] != len(item["input_ids"]):
            raise ValueError("materialized and executed tokenization disagree")
        result.append({**item, "task_key": row["task_key"], "window_id": row["window_id"]})
    return result


def dense_rows(rows: list[dict], spec: dict, *, max_length: int, vocab_size: int) -> list[dict]:
    """Validate once-only eligible targets and adapt full masks to native suffix masks.

    ``num_actions`` is the WIDTH of the suffix, including masked interior tool
    tokens. It is not the number of supervised tokens. The native collator
    supplies token-normalization from the positive mask count.
    """
    if spec.get("format") != DENSE_FORMAT:
        raise ValueError("dense row format not explicitly selected")
    if len(rows) != spec["rows"] or {r["task_key"] for r in rows} != set(spec["task_keys"]):
        raise ValueError("dense dataset row count or task identity mismatch")
    if len({r["window_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate dense segment")
    sessions = {}
    supervised = responses = 0
    result = []
    for row in rows:
        ids, mask = row["input_ids"], row["loss_mask"]
        if (
            not isinstance(ids, list)
            or not 1 < len(ids) <= max_length
            or any(type(x) is not int or not 0 <= x < vocab_size for x in ids)
        ):
            raise ValueError("invalid or overlength pretokenized segment")
        if (
            not isinstance(mask, list)
            or len(mask) != len(ids)
            or mask[0] != 0
            or any(type(x) is not int or x not in (0, 1) for x in mask)
            or not sum(mask)
        ):
            raise ValueError("invalid full-token assistant mask or missing causal context")
        if row.get("token_count") != len(ids) or row.get("target_token_count") != sum(mask):
            raise ValueError("dense token inventory disagrees with payload")
        sid, count = row["source_session_id"], row["source_assistant_count"]
        if not isinstance(sid, str) or not sid or type(count) is not int or count <= 0:
            raise ValueError("dense source identity or response count missing")
        eligible, exclusions = (
            row["eligible_assistant_indices"],
            row["excluded_assistant_targets"],
        )
        if (
            not isinstance(eligible, list)
            or not eligible
            or any(type(x) is not int or not 0 <= x < count for x in eligible)
            or eligible != sorted(set(eligible))
            or not isinstance(exclusions, list)
        ):
            raise ValueError("invalid eligible-response inventory")
        excluded = {}
        for entry in exclusions:
            index, message, reason = (
                entry["assistant_index"],
                entry["source_message_index"],
                entry["reason"],
            )
            if (
                type(index) is not int
                or not 0 <= index < count
                or index in excluded
                or type(message) is not int
                or message < 0
                or reason not in DENSE_EXCLUSION_REASONS
            ):
                raise ValueError("invalid excluded-response provenance")
            excluded[index] = (message, reason)
        if (
            list(excluded) != sorted(excluded)
            or set(eligible) & set(excluded)
            or set(eligible) | set(excluded) != set(range(count))
        ):
            raise ValueError("eligible and excluded inventories are not disjoint and exhaustive")
        source = sessions.setdefault(
            sid,
            {
                "count": count,
                "task": row["task_key"],
                "targets": {},
                "eligible": tuple(eligible),
                "excluded": excluded,
            },
        )
        if (
            source["count"] != count
            or source["task"] != row["task_key"]
            or source["eligible"] != tuple(eligible)
            or source["excluded"] != excluded
        ):
            raise ValueError("dense source metadata changes across segments")
        copied = row["copied_context_assistant_indices"]
        if (
            not isinstance(copied, list)
            or len(set(copied)) != len(copied)
            or any(type(x) is not int or not 0 <= x < count for x in copied)
        ):
            raise ValueError("invalid copied-context assistant inventory")
        spans = row["target_spans"]
        if not isinstance(spans, list) or not spans:
            raise ValueError("dense segment lacks assistant target spans")
        expected_mask = [0] * len(ids)
        previous_end = 0
        previous_assistant = -1
        previous_message = -1
        for span in spans:
            index, message = span["assistant_index"], span["source_message_index"]
            start, end = span["token_start"], span["token_end"]
            if (
                any(type(x) is not int for x in (index, message, start, end))
                or not 0 <= index < count
                or message < 0
                or not 1 <= start < end <= len(ids)
                or start < previous_end
                or index <= previous_assistant
                or message <= previous_message
            ):
                raise ValueError("invalid, unordered or overlapping assistant target spans")
            if index not in source["eligible"] or index in source["targets"] or index in copied:
                raise ValueError("excluded, repeated, or copied-context assistant targeted")
            if span["source_target_sha256"].removeprefix("sha256:") != _unsigned_digest(
                ids[start:end]
            ):
                raise ValueError("assistant target tokens differ from source span digest")
            source["targets"][index] = message
            expected_mask[start:end] = [1] * (end - start)
            previous_end, previous_assistant, previous_message = end, index, message
        if mask != expected_mask:
            raise ValueError("loss mask trains outside declared assistant targets")
        first = mask.index(1)
        result.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "num_actions": len(ids) - first,
                "loss_mask": mask[first:],
                "task_key": row["task_key"],
                "window_id": row["window_id"],
            }
        )
        supervised += sum(mask)
        responses += len(spans)
    for source in sessions.values():
        if set(source["targets"]) != set(source["eligible"]):
            raise ValueError("eligible source assistant response coverage is incomplete")
        provenance = {
            **source["targets"],
            **{i: v[0] for i, v in source["excluded"].items()},
        }
        messages = [provenance[i] for i in range(source["count"])]
        if messages != sorted(set(messages)):
            raise ValueError("source assistant order or uniqueness differs across segments")
    if (supervised, responses, len(sessions)) != (
        spec["supervised_tokens"],
        spec["assistant_responses"],
        spec["source_sessions"],
    ):
        raise ValueError("dense supervised-token, response or source total mismatch")
    if (
        sum(s["count"] for s in sessions.values()),
        sum(len(s["excluded"]) for s in sessions.values()),
    ) != (
        spec["source_total_assistant_responses"],
        spec["excluded_assistant_responses"],
    ):
        raise ValueError("dense original-source or excluded-response total mismatch")
    return result


def prepare_rows(rows, spec, tokenizer, tokenize, *, max_length):
    if spec.get("format") == DENSE_FORMAT:
        return dense_rows(rows, spec, max_length=max_length, vocab_size=len(tokenizer))
    return tokenize_rows(rows, spec, tokenizer, tokenize, max_length=max_length)


class EvalAccumulator:
    """Recover unnormalized NLL from native per-row, globally scaled losses."""

    def __init__(self):
        self.tasks = defaultdict(lambda: [0.0, 0, 0])

    def add_batch(
        self,
        metadata: list[dict],
        counts: list[int],
        outputs: list[dict],
        batch_loss: float,
    ) -> None:
        if (
            len(metadata) != len(counts)
            or len(outputs) < len(metadata)
            or any(n <= 0 for n in counts)
        ):
            raise ValueError("eval row/target accounting mismatch")
        count = sum(counts)
        summed_scaled = 0.0
        for row, n, output in zip(metadata, counts, outputs, strict=False):
            values = output["elementwise_loss"]
            if len(values) != n or not all(
                math.isfinite(float(x)) and float(x) >= 0 for x in values
            ):
                raise ValueError("invalid native per-target loss output")
            scaled = math.fsum(float(x) for x in values)
            summed_scaled += scaled
            totals = self.tasks[row["task_key"]]
            totals[0] += scaled * count  # native mask scale is 1 / real batch targets
            totals[1] += n
            totals[2] += 1
        if any(item["elementwise_loss"] for item in outputs[len(metadata) :]):
            raise ValueError("padding contributed to validation loss")
        if not math.isfinite(batch_loss) or not math.isclose(
            summed_scaled, batch_loss, rel_tol=2e-4, abs_tol=2e-5
        ):
            raise ValueError("per-task loss does not reconcile with native global loss")

    def metrics(self, expected_tasks: set[str]) -> dict:
        if not expected_tasks or set(self.tasks) != expected_tasks:
            raise ValueError("validation task coverage differs from the frozen task set")
        nll = math.fsum(value[0] for value in self.tasks.values())
        tokens = sum(value[1] for value in self.tasks.values())
        macro = math.fsum(value[0] / value[1] for value in self.tasks.values()) / len(self.tasks)
        return {
            "eval_loss": nll / tokens,
            "task_macro_loss": macro,
            "supervised_tokens": tokens,
            "windows": sum(v[2] for v in self.tasks.values()),
            "tasks": len(self.tasks),
        }


def retention_steps(saved: set[int], best_step: int, keep_latest: int) -> set[int]:
    return set(sorted(saved)[-keep_latest:]) | ({best_step} if best_step in saved else set())


class ProgressWatchdog:
    """Bounded fail-closed resource watchdog; unavailable telemetry is not idle."""

    def __init__(self, started_at: float):
        self.started_at = started_at
        self.last_progress = started_at
        self.previous_marker = None
        self.previous_io = None
        self.idle_since = None

    def observe(
        self,
        now: float,
        marker,
        gpu_mean: float | None,
        process_io: int | None,
        *,
        checkpoint_advancing: bool = False,
    ) -> str | None:
        changed = marker != self.previous_marker
        if changed:
            self.last_progress = now
        self.previous_marker = marker
        io_advanced = (
            process_io is not None
            and self.previous_io is not None
            and process_io - self.previous_io >= 1024 * 1024
        )
        self.previous_io = process_io
        elapsed = now - self.started_at
        if elapsed >= WATCHDOG_HARD_SECONDS and (
            not checkpoint_advancing or elapsed >= WATCHDOG_HARD_SECONDS + WATCHDOG_DRAIN_SECONDS
        ):
            return "hard_runtime_bound"
        if changed or io_advanced or gpu_mean is None or process_io is None or gpu_mean >= 1:
            self.idle_since = None
        elif self.idle_since is None:
            self.idle_since = now
        if (
            elapsed >= WATCHDOG_STARTUP_SECONDS
            and self.idle_since is not None
            and now - self.idle_since >= WATCHDOG_IDLE_SECONDS
            and now - self.last_progress >= WATCHDOG_IDLE_SECONDS
        ):
            return "confirmed_no_progress_idle"
        return None


def _output_progress(output: Path) -> tuple[tuple, tuple]:
    def fingerprint(paths):
        found = []
        for path in paths:
            try:
                stat = path.stat()
                if path.is_file():
                    found.append((str(path.relative_to(output)), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:  # an atomic write/prune can race the observation
                continue
        return tuple(sorted(found))

    receipts = fingerprint(output / name for name in ("PROGRESS.json", "ACTIVITY.json"))
    checkpoints = fingerprint((output / "checkpoints").rglob("*"))
    return receipts, checkpoints


def _utilization_snapshot() -> tuple[float | None, int | None]:
    # No command lines, environment, logs or process contents are read or printed.
    gpu_mean = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        if values and all(math.isfinite(value) and 0 <= value <= 100 for value in values):
            gpu_mean = sum(values) / len(values)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    io_bytes = 0
    observed = False
    # This container belongs entirely to this job; /proc PIDs are namespace local.
    # Physical I/O ignores normal tiny Ray heartbeat/log writes (<1MiB/sample).
    for path in Path("/proc").glob("[0-9]*/io"):
        try:
            counters = dict(line.split(":", 1) for line in path.read_text().splitlines())
            io_bytes += int(counters["read_bytes"]) + int(counters["write_bytes"])
            observed = True
        except (OSError, ValueError, KeyError):
            continue
    return gpu_mean, io_bytes if observed else None


def _wait_for_training(ray, task, output: Path) -> dict:
    watchdog = ProgressWatchdog(time.monotonic())
    previous_checkpoint = None
    while True:
        ready, _ = ray.wait([task], timeout=WATCHDOG_POLL_SECONDS)
        if ready:
            return ray.get(task)
        marker, checkpoint = _output_progress(output)
        gpu_mean, io_bytes = _utilization_snapshot()
        reason = watchdog.observe(
            time.monotonic(),
            (marker, checkpoint),
            gpu_mean,
            io_bytes,
            checkpoint_advancing=checkpoint != previous_checkpoint,
        )
        previous_checkpoint = checkpoint
        write_receipt(
            output / "WATCHDOG.json",
            {
                "observed_at_unix": time.time(),
                "gpu_mean_utilization_pct": gpu_mean,
                "process_io_available": io_bytes is not None,
                "state": reason or "monitoring",
            },
            replace=True,
        )
        if reason:
            ray.cancel(task, force=True, recursive=True)
            raise RuntimeError(reason)


def _configure_wandb(plan: dict) -> None:
    for key, value in {
        "WANDB_ENTITY": plan["wandb"]["entity"],
        "WANDB_RUN_GROUP": plan["wandb"]["group"],
        "WANDB_RUN_ID": plan["wandb"]["run_id"],
        "WANDB_RESUME": "never",
        "WANDB_MODE": "online",
        "WANDB_CONSOLE": "off",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_SAVE_CODE": "false",
        "WANDB_DISABLE_GIT": "true",
        "WANDB_DIR": str(Path(plan["output_root"]) / "wandb"),
    }.items():
        os.environ[key] = value
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("W&B secret injection missing")
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True, mode=0o700)


def explicit_tracking_class(base):
    """Native Tracking.__del__ assumes success; only our outer outcome may finish."""

    class ExplicitTracking(base):
        def __del__(self):
            pass

    return ExplicitTracking


def public_failure_details(error: BaseException) -> dict:
    """Return a narrow, non-secret fingerprint for setup contract failures."""
    chain = []
    current = error
    seen = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = getattr(current, "cause", None) or current.__cause__ or current.__context__

    missing_key = None
    for item in chain:
        if isinstance(item, KeyError) and len(item.args) == 1 and isinstance(item.args[0], str):
            candidate = item.args[0]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", candidate):
                missing_key = candidate
                break
    if missing_key is None:
        for item in chain:
            match = re.search(
                r"(?:^|\n)KeyError: ['\"]([A-Za-z_][A-Za-z0-9_.-]{0,127})['\"](?:\n|$)",
                str(item),
            )
            if match:
                missing_key = match.group(1)
                break

    missing_attribute = None
    attribute_error = re.compile(
        r"(?:'[^'\r\n]{1,128}' object|type object '[^'\r\n]{1,128}'|"
        r"module '[A-Za-z_][A-Za-z0-9_.]{0,255}') has no attribute "
        r"'([A-Za-z_][A-Za-z0-9_]{0,127})'"
    )
    for item in chain:
        for line in str(item).splitlines():
            if line.startswith("AttributeError: "):
                line = line.removeprefix("AttributeError: ")
            elif not isinstance(item, AttributeError):
                continue
            match = attribute_error.fullmatch(line)
            if match:
                missing_attribute = match.group(1)
                break
        if missing_attribute:
            break

    error_class = (
        "KeyError"
        if missing_key
        else "AttributeError"
        if missing_attribute
        else type(chain[-1]).__name__
    )
    details = {"error_class": error_class}
    if missing_key:
        details["missing_key"] = missing_key
    elif missing_attribute:
        details["missing_attribute"] = missing_attribute
    return details


def finalize_failed_run(trainer, output: Path, error: BaseException) -> list[str]:
    """Best-effort independent cleanup: storage/SDK defects cannot skip exit1."""
    failures = []
    try:
        stage = getattr(trainer, "public_runtime_stage", None)
        if stage not in PUBLIC_RUNTIME_STAGES:
            stage = "trainer_constructed"
        qualification_stage = getattr(trainer, "public_qualification_stage", None)
        qualification = (
            {"qualification_stage": qualification_stage}
            if qualification_stage in QWEN38_QUALIFICATION_STAGES
            else {}
        )
        write_receipt(
            output / "FAILURE_STAGE.json",
            {
                **public_failure_details(error),
                **qualification,
                "optimizer_step": getattr(trainer, "global_step", 0),
                "plan_sha256": trainer.plan["plan_sha256"],
                "stage": stage,
            },
        )
    except BaseException:
        failures.append("failure_stage")
    try:
        with (output / "private-runtime-failure.log").open("a") as stream:
            traceback.print_exc(file=stream)
    except BaseException:
        failures.append("private_log")
    try:
        if trainer._ray_gpu_monitor is not None:
            trainer._ray_gpu_monitor.stop()
    except BaseException:
        failures.append("gpu_monitor")
    try:
        import wandb
    except ImportError:
        return failures + ["tracking_import"]
    try:
        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    "status": "failed",
                    "error_class": type(error).__name__,
                    "optimizer_step": trainer.global_step,
                }
            )
    except BaseException:
        failures.append("tracking_summary")
    try:
        # Also handles a run created before native tracker construction failed.
        if wandb.run is not None:
            wandb.finish(exit_code=1)
    except BaseException:
        failures.append("tracking_finish")
    return failures


class PlannedPause(Exception):
    """Exit the native loop only after the requested save, validation and log."""


def _qwen38_rank_snapshots(snapshots: list[dict], *, stage: str) -> dict[int, dict]:
    """Validate and index one complete TP8 snapshot without tensor values."""
    if not isinstance(snapshots, list) or len(snapshots) != 8:
        raise ValueError(f"{stage} LoRA evidence must contain exactly eight ranks")
    indexed = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "rank",
            "target_census",
            "adapters",
            "trainable",
            "frozen_base",
            "successful_optimizer_updates",
            "last_gradient_norm",
        }:
            raise ValueError(f"{stage} LoRA rank evidence has unknown or missing fields")
        rank = snapshot["rank"]
        expected_rank_fields = {
            "world_rank",
            "tp_rank",
            "tp_size",
            "pp_rank",
            "pp_size",
            "cp_rank",
            "cp_size",
            "dp_rank",
            "dp_size",
        }
        if not isinstance(rank, dict) or set(rank) != expected_rank_fields:
            raise ValueError(f"{stage} LoRA rank topology is incomplete")
        tp_rank = rank["tp_rank"]
        if (
            type(tp_rank) is not int
            or any(type(value) is not int for value in rank.values())
            or not 0 <= tp_rank < 8
            or tp_rank in indexed
            or rank
            != {
                "world_rank": tp_rank,
                "tp_rank": tp_rank,
                "tp_size": 8,
                "pp_rank": 0,
                "pp_size": 1,
                "cp_rank": 0,
                "cp_size": 1,
                "dp_rank": 0,
                "dp_size": 1,
            }
        ):
            raise ValueError(f"{stage} LoRA evidence has duplicate or invalid TP ranks")
        adapters = snapshot["adapters"]
        trainable = snapshot["trainable"]
        frozen = snapshot["frozen_base"]
        if not isinstance(adapters, dict) or not adapters:
            raise ValueError(f"{stage} LoRA rank has no adapter tensors")
        for name, row in adapters.items():
            if (
                not isinstance(name, str)
                or ".adapter" not in name.lower()
                or not isinstance(row, dict)
                or set(row)
                != {
                    "logical_target",
                    "dtype",
                    "global_shape",
                    "local_shape",
                    "sharding",
                    "sha256",
                }
                or not isinstance(row["logical_target"], str)
                or row["dtype"] != "BF16"
                or any(
                    not isinstance(shape, list)
                    or not shape
                    or any(type(size) is not int or size <= 0 for size in shape)
                    for shape in (row["global_shape"], row["local_shape"])
                )
                or not isinstance(row["sharding"], dict)
                or not isinstance(row["sha256"], str)
                or re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is None
            ):
                raise ValueError(f"{stage} LoRA adapter tensor evidence is invalid")
        if not isinstance(trainable, dict) or set(trainable) != {
            "parameter_count",
            "elements",
            "manifest_sha256",
            "unexpected_parameters",
            "nontrainable_adapter_parameters",
        }:
            raise ValueError(f"{stage} trainable census is incomplete")
        if not isinstance(frozen, dict) or set(frozen) != {
            "parameter_count",
            "elements",
            "bytes",
            "manifest_sha256",
        }:
            raise ValueError(f"{stage} frozen-base census is incomplete")
        expected_trainable_rows = [
            {
                "name": name,
                "dtype": row["dtype"],
                "shape": row["local_shape"],
                "numel": math.prod(row["local_shape"]),
            }
            for name, row in sorted(adapters.items())
        ]
        expected_trainable_elements = sum(row["numel"] for row in expected_trainable_rows)
        if (
            trainable["unexpected_parameters"] != []
            or trainable["nontrainable_adapter_parameters"] != []
            or trainable["parameter_count"] != len(adapters)
            or trainable["elements"] != expected_trainable_elements
            or trainable["manifest_sha256"] != _unsigned_digest(expected_trainable_rows)
            or any(
                type(trainable[field]) is not int or trainable[field] <= 0
                for field in ("parameter_count", "elements")
            )
            or any(
                type(frozen[field]) is not int or frozen[field] <= 0
                for field in ("parameter_count", "elements", "bytes")
            )
            or any(
                not isinstance(value["manifest_sha256"], str)
                or re.fullmatch(r"[a-f0-9]{64}", value["manifest_sha256"]) is None
                for value in (trainable, frozen)
            )
        ):
            raise ValueError(f"{stage} LoRA rank violates adapter-only trainability")
        indexed[tp_rank] = snapshot
    if set(indexed) != set(range(8)):
        raise ValueError(f"{stage} LoRA evidence is missing one or more TP ranks")
    return indexed


def _qwen38_checkpoint_inventory(checkpoint: Path) -> tuple[dict, dict]:
    """Reopen one finalized checkpoint and return exact file and role inventories."""
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("qualified checkpoint root is missing or a symlink")
    files = {}
    roles = {"adapter": [], "optimizer_and_rng": [], "metadata": []}
    adapter_pattern = re.compile(r"policy/adapter_tp([0-7])_pp0_cp0_dp0_ep0_etp([0-7])\.pt")
    optimizer_pattern = re.compile(r"policy/__\d+_\d+\.distcp")
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint inventory contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("checkpoint inventory contains a non-file entry")
        relative = path.relative_to(checkpoint).as_posix()
        size = path.stat().st_size
        if size <= 0:
            raise ValueError("checkpoint inventory contains an empty file")
        files[relative] = {"bytes": size, "sha256": digest(path)}
        if adapter_pattern.fullmatch(relative):
            roles["adapter"].append(relative)
        elif relative == "policy/.metadata" or optimizer_pattern.fullmatch(relative):
            roles["optimizer_and_rng"].append(relative)
        else:
            roles["metadata"].append(relative)
    roles["adapter"].sort(key=lambda name: int(adapter_pattern.fullmatch(name).group(1)))
    for name in ("optimizer_and_rng", "metadata"):
        roles[name].sort()
    # Megatron Bridge writes the no-expert-parallel TP8 adapter shards with
    # the expert-tensor rank label equal to the tensor-parallel rank. This is
    # a filename coordinate emitted by the native writer, not an assertion
    # that expert tensor parallelism has size eight.
    expected_adapters = [
        f"policy/adapter_tp{rank}_pp0_cp0_dp0_ep0_etp{rank}.pt" for rank in range(8)
    ]
    if (
        roles["adapter"] != expected_adapters
        or not roles["optimizer_and_rng"]
        or not roles["metadata"]
        or set(files) != {item for values in roles.values() for item in values}
    ):
        raise ValueError("finalized checkpoint file roles are incomplete")
    return files, roles


def _qwen38_checkpoint_finalization(acknowledgements: list[dict]) -> dict:
    """Require an explicit successful async-writer drain from every TP rank."""
    if not isinstance(acknowledgements, list) or len(acknowledgements) != 8:
        raise ValueError("checkpoint finalization must acknowledge every TP8 rank")
    ranks = set()
    rows = []
    for acknowledgement in acknowledgements:
        if not isinstance(acknowledgement, dict) or set(acknowledgement) != {
            "world_rank",
            "tp_rank",
            "finalized",
        }:
            raise ValueError("checkpoint finalization acknowledgement is malformed")
        rank = acknowledgement["tp_rank"]
        if (
            type(rank) is not int
            or acknowledgement["world_rank"] != rank
            or rank in ranks
            or not 0 <= rank < 8
            or acknowledgement["finalized"] is not True
        ):
            raise ValueError("checkpoint finalization has duplicate or unsuccessful ranks")
        ranks.add(rank)
        rows.append(dict(acknowledgement))
    if ranks != set(range(8)):
        raise ValueError("checkpoint finalization is missing one or more TP ranks")
    return {
        "async_writes_finalized": True,
        "rank_acknowledgements": sorted(rows, key=lambda row: row["tp_rank"]),
    }


def _tensor_sha256(tensor, *, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    """Independently hash exact dense tensor bytes with bounded host memory."""
    import torch

    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise TypeError("checkpoint evidence supports dense strided tensors only")
    if tensor.numel() <= 0 or chunk_bytes <= 0:
        raise ValueError("checkpoint evidence rejects empty tensors and invalid chunks")
    value = tensor.detach().contiguous().reshape(-1)
    elements = max(1, chunk_bytes // max(1, value.element_size()))
    result = hashlib.sha256()
    for start in range(0, value.numel(), elements):
        host = value[start : start + elements].to(device="cpu", non_blocking=False)
        result.update(host.view(torch.uint8).numpy().tobytes(order="C"))
    return result.hexdigest()


def _qwen38_reconcile_evidence(
    before_snapshots: list[dict],
    after_snapshots: list[dict],
    checkpoint: Path,
    *,
    forward_loss: float,
    trainer_gradient_norm: float,
) -> dict:
    """Reconcile live TP8 bytes with independently reopened adapter shards."""
    import torch

    before = _qwen38_rank_snapshots(before_snapshots, stage="before")
    after = _qwen38_rank_snapshots(after_snapshots, stage="after")
    census = before[0]["target_census"]
    names = set(before[0]["adapters"])
    if not names:
        raise ValueError("LoRA adapter census is empty")
    gradients = []
    checkpoint_hashes = {}
    for rank in range(8):
        first, last = before[rank], after[rank]
        if first["target_census"] != census or last["target_census"] != census:
            raise ValueError("all-linear target census differs across rank or time")
        if set(first["adapters"]) != names or set(last["adapters"]) != names:
            raise ValueError("adapter keys differ across rank or time")
        if (
            type(first["successful_optimizer_updates"]) is not int
            or type(last["successful_optimizer_updates"]) is not int
            or first["successful_optimizer_updates"] != 0
            or first["last_gradient_norm"] is not None
            or last["successful_optimizer_updates"] != 1
        ):
            raise ValueError("every TP rank must report exactly one successful optimizer update")
        gradient = last["last_gradient_norm"]
        if type(gradient) not in {int, float} or not math.isfinite(gradient) or gradient <= 0:
            raise ValueError("every TP rank must report one finite positive LoRA gradient norm")
        gradients.append(float(gradient))
        if (
            first["trainable"]["parameter_count"] != last["trainable"]["parameter_count"]
            or first["trainable"]["elements"] != last["trainable"]["elements"]
            or first["trainable"]["manifest_sha256"] != last["trainable"]["manifest_sha256"]
            or first["frozen_base"] != last["frozen_base"]
        ):
            raise ValueError("trainable metadata or frozen base bytes changed")
        for name in names:
            first_row, last_row = first["adapters"][name], last["adapters"][name]
            metadata = {
                "logical_target",
                "dtype",
                "global_shape",
                "local_shape",
                "sharding",
            }
            if (
                set(first_row) != metadata | {"sha256"}
                or set(last_row) != metadata | {"sha256"}
                or any(first_row[key] != last_row[key] for key in metadata)
                or first_row["dtype"] != "BF16"
                or not first_row["local_shape"]
                or any(type(size) is not int or size <= 0 for size in first_row["local_shape"])
            ):
                raise ValueError("adapter tensor metadata changed or is invalid")

        adapter_path = checkpoint / f"policy/adapter_tp{rank}_pp0_cp0_dp0_ep0_etp{rank}.pt"
        payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"model_state_dict"}:
            raise ValueError("adapter checkpoint payload has unknown or missing fields")
        state = payload["model_state_dict"]
        if not isinstance(state, dict) or set(state) != names:
            raise ValueError("adapter checkpoint keys differ from the live model")
        checkpoint_hashes[rank] = {}
        for name, tensor in state.items():
            live = last["adapters"][name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided
                or tensor.numel() <= 0
                or list(tensor.shape) != live["local_shape"]
                or str(tensor.dtype).removeprefix("torch.").upper().replace("BFLOAT16", "BF16")
                != live["dtype"]
            ):
                raise ValueError("adapter checkpoint tensor metadata differs from the live model")
            saved_hash = _tensor_sha256(tensor)
            if saved_hash != live["sha256"]:
                raise ValueError(
                    "adapter checkpoint tensor differs from the live post-update tensor"
                )
            checkpoint_hashes[rank][name] = saved_hash

    if any(
        not math.isclose(value, gradients[0], rel_tol=1e-6, abs_tol=1e-12) for value in gradients
    ):
        raise ValueError("gradient norm differs across TP ranks")
    if (
        type(trainer_gradient_norm) not in {int, float}
        or not math.isfinite(trainer_gradient_norm)
        or trainer_gradient_norm <= 0
        or not math.isclose(trainer_gradient_norm, gradients[0], rel_tol=1e-6, abs_tol=1e-12)
    ):
        raise ValueError("driver and rank-local gradient evidence differ")
    if (
        type(forward_loss) not in {int, float}
        or not math.isfinite(forward_loss)
        or forward_loss < 0
    ):
        raise ValueError("forward loss evidence is invalid")

    parameters = {}
    changed = 0
    for name in sorted(names):
        reference = before[0]["adapters"][name]
        shards = []
        for rank in range(8):
            first = before[rank]["adapters"][name]
            last = after[rank]["adapters"][name]
            if any(
                first[key] != reference[key]
                for key in ("logical_target", "dtype", "global_shape", "sharding")
            ):
                raise ValueError("adapter global metadata differs across TP ranks")
            changed += first["sha256"] != last["sha256"]
            shards.append(
                {
                    "tp_rank": rank,
                    "shape": first["local_shape"],
                    "before_sha256": first["sha256"],
                    "after_sha256": last["sha256"],
                    "checkpoint_sha256": checkpoint_hashes[rank][name],
                }
            )
        parameters[name] = {
            "logical_target": reference["logical_target"],
            "dtype": reference["dtype"],
            "global_shape": reference["global_shape"],
            "sharding": reference["sharding"],
            "rank_shards": shards,
        }
    if changed <= 0:
        raise ValueError("one successful optimizer update changed no adapter tensor")
    trainable_manifests = [
        {
            "tp_rank": rank,
            "before_sha256": before[rank]["trainable"]["manifest_sha256"],
            "after_sha256": after[rank]["trainable"]["manifest_sha256"],
        }
        for rank in range(8)
    ]
    frozen_manifests = [
        {
            "tp_rank": rank,
            "before_sha256": before[rank]["frozen_base"]["manifest_sha256"],
            "after_sha256": after[rank]["frozen_base"]["manifest_sha256"],
        }
        for rank in range(8)
    ]
    return {
        "target_census": census,
        "adapter_parameters": parameters,
        "trainable_parameter_census": {
            "parameter_count": len(parameters),
            "elements": sum(math.prod(row["global_shape"]) for row in parameters.values()),
            "rank_manifests": trainable_manifests,
        },
        "frozen_base": {
            "parameter_count": sum(
                before[rank]["frozen_base"]["parameter_count"] for rank in range(8)
            ),
            "elements": sum(before[rank]["frozen_base"]["elements"] for rank in range(8)),
            "bytes": sum(before[rank]["frozen_base"]["bytes"] for rank in range(8)),
            "rank_manifests": frozen_manifests,
        },
        "forward_loss": float(forward_loss),
        "lora_gradient_norm": gradients[0],
        "adapter_updated_tensor_count": changed,
    }


def _qwen38_wandb_evidence(trainer) -> dict:
    metrics_path = trainer.output / "metrics.jsonl"
    if metrics_path.is_symlink() or not metrics_path.is_file() or metrics_path.stat().st_size <= 0:
        raise ValueError("local W&B scalar stream is missing")
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    step = [record for record in records if record.get("optimizer_step") == 1]
    if len(step) != 1 or any(record.get("optimizer_step") != 1 for record in records):
        raise ValueError("local scalar stream does not contain exactly one optimizer step")
    record = step[0]
    scalar_keys = sorted(key for key in record if key not in {"optimizer_step", "time"})
    if not {"train/loss", "train/grad_norm", "train/lr"}.issubset(scalar_keys) or any(
        type(record[key]) not in {int, float} or not math.isfinite(record[key])
        for key in scalar_keys
    ):
        raise ValueError("local W&B scalar stream is incomplete or nonfinite")
    return {
        **trainer.wandb_binding,
        "optimizer_step": 1,
        "local_metrics_sha256": digest(metrics_path),
        "scalar_keys": scalar_keys,
        "finish_succeeded": True,
    }


def _qwen38_source_inventory(plan: dict) -> dict:
    """Hash the exact staged model/data bytes without publishing their contents."""
    rows = []
    for item in plan["model"]["files"]:
        path = Path(plan["model"]["root"]) / item["path"]
        observation = _verified_file_observation(path, item["sha256"])
        rows.append(
            {
                "kind": "model",
                "name": item["path"],
                **observation,
            }
        )
    for split, spec in sorted(plan["datasets"].items()):
        path = Path(spec["path"])
        observation = _verified_file_observation(path, spec["sha256"])
        rows.append(
            {
                "kind": "dataset",
                "name": split,
                **observation,
            }
        )
    if not rows or any(row["bytes"] <= 0 for row in rows):
        raise ValueError("staged model/data source inventory is empty")
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "manifest_sha256": _unsigned_digest(rows),
    }


def _qwen38_runtime_source_inventory(plan: dict) -> dict:
    """Hash every load-bearing installed SkyRL source named by the binding."""
    spec = importlib.util.find_spec("skyrl")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("pinned SkyRL package is unavailable")
    root = Path(next(iter(spec.submodule_search_locations))).parent
    rows = []
    for name, expected in sorted(qwen38_megatron_binding()[2].items()):
        path = root / name
        rows.append({"name": name, **_verified_file_observation(path, expected)})
    if not rows or any(row["bytes"] <= 0 for row in rows):
        raise ValueError("installed SkyRL source inventory is empty")
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "manifest_sha256": _unsigned_digest(rows),
    }


def _prepare_qwen38_checkpoint_receipt(
    trainer,
    before_snapshots,
    after_snapshots,
    checkpoint_finalization,
    source_inventory_before,
    runtime_inventory_before,
) -> dict:
    """Prepare (but do not publish) the strict post-finalization receipt."""
    plan = trainer.plan
    checkpoint = Path(plan_checkpoint(plan, 1))
    if checkpoint != trainer.output / "checkpoints" / "global_step_1":
        raise ValueError("qualified checkpoint path differs from the one-step plan")
    files, roles = _qwen38_checkpoint_inventory(checkpoint)
    trainer._record_qualification_stage("checkpoint_inventory_validated")
    step = getattr(trainer, "last_step_evidence", None)
    if not isinstance(step, dict) or set(step) != {
        "forward_loss",
        "lora_gradient_norm",
    }:
        raise ValueError("one-step trainer evidence is incomplete")
    trainer._record_qualification_stage("trainer_step_evidence_validated")
    trainer._record_qualification_stage("adapter_reconciliation_started")
    reconciled = _qwen38_reconcile_evidence(
        before_snapshots,
        after_snapshots,
        checkpoint,
        forward_loss=step["forward_loss"],
        trainer_gradient_norm=step["lora_gradient_norm"],
    )
    trainer._record_qualification_stage("adapter_reconciliation_complete")
    # Re-hash every bound model/data source exactly once after the optimizer
    # step.  Structural validation is independent of file reads, and the
    # complete pre-step inventories were already collected immediately before
    # trainer setup.  This preserves before/after immutability evidence without
    # multiplying a 55.6 GB model read.
    trainer._record_qualification_stage("source_revalidation_started")
    validate_plan(plan, check_files=False)
    source_inventory_after = _qwen38_source_inventory(plan)
    trainer._record_qualification_stage("source_revalidation_complete")
    runtime_inventory_after = _qwen38_runtime_source_inventory(plan)
    trainer._record_qualification_stage("runtime_revalidation_complete")
    if source_inventory_before != source_inventory_after:
        raise ValueError("staged model/data source inventory changed during training")
    if runtime_inventory_before != runtime_inventory_after:
        raise ValueError("installed SkyRL source inventory changed during training")
    trainer._record_qualification_stage("source_immutability_validated")
    source_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if _unsigned_digest(source_plan) != plan["plan_sha256"]:
        raise ValueError("runtime source plan differs from the staged plan bytes")
    trainer._record_qualification_stage("source_plan_validated")
    return {
        "schema": "cyber_qwen38_megatron_lora_checkpoint_manifest_v1",
        "source_plan_sha256": plan["plan_sha256"],
        "source_plan": source_plan,
        "checkpoint_path": str(checkpoint),
        "optimizer_step": 1,
        "topology": {
            "world_size": 8,
            "tensor_parallel": 8,
            "pipeline_parallel": 1,
            "context_parallel": 1,
            "data_parallel": 1,
            "expert_parallel": 1,
            "expert_tensor_parallel": 1,
        },
        "target_census": reconciled["target_census"],
        "adapter_parameters": reconciled["adapter_parameters"],
        "trainable_parameter_census": reconciled["trainable_parameter_census"],
        "frozen_base": reconciled["frozen_base"],
        "checkpoint_finalization": _qwen38_checkpoint_finalization(checkpoint_finalization),
        "source_inventory": {
            "model_and_data": {
                "file_count": source_inventory_before["file_count"],
                "total_bytes": source_inventory_before["total_bytes"],
                "before_sha256": source_inventory_before["manifest_sha256"],
                "after_sha256": source_inventory_after["manifest_sha256"],
            },
            "runtime": {
                "file_count": runtime_inventory_before["file_count"],
                "total_bytes": runtime_inventory_before["total_bytes"],
                "before_sha256": runtime_inventory_before["manifest_sha256"],
                "after_sha256": runtime_inventory_after["manifest_sha256"],
            },
        },
        "files": files,
        "file_roles": roles,
        "total_bytes": sum(row["bytes"] for row in files.values()),
        "evidence": {
            "forward_loss": reconciled["forward_loss"],
            "lora_gradient_norm": reconciled["lora_gradient_norm"],
            "optimizer_updates": 1,
            "adapter_tensors_changed": True,
            "adapter_updated_tensor_count": reconciled["adapter_updated_tensor_count"],
            "frozen_base_tensors_unchanged": True,
            "unexpected_trainable_parameters": [],
            "source_inventory_unchanged": True,
            "wandb_run_id": plan["wandb"]["run_id"],
        },
    }


def _make_trainer_class():
    from skyrl.backends.skyrl_train.training_batch import pad_training_input_batch
    from skyrl.train.sft_trainer import SFTTrainer, tokenize_chat_example
    from skyrl.train.utils.callbacks import TrainingCallback
    from skyrl.train.utils.tracking import Tracking
    from skyrl.train.utils.utils import Timer

    class EvidenceCallback(TrainingCallback):
        def on_log(self, trainer, event, control):
            event.logs["train/global_step"] = event.global_step
            event.logs["train/epoch_fraction"] = event.global_step / max(event.steps_per_epoch, 1)
            event.logs.update(trainer.extra_train_metrics if "train/loss" in event.logs else {})
            # Durable local scalar stream also covers W&B network interruptions.
            with (trainer.output / "metrics.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "optimizer_step": event.global_step,
                            "time": time.time(),
                            **event.logs,
                        }
                    )
                    + "\n"
                )
            write_receipt(
                trainer.output / "PROGRESS.json",
                {
                    "optimizer_step": event.global_step,
                    "phase": (
                        "validation"
                        if any(k.startswith("eval/") for k in event.logs)
                        else "training"
                    ),
                    "observed_at_unix": time.time(),
                },
                replace=True,
            )
            if "train/loss" in event.logs and event.global_step == trainer.plan.get(
                "pause_after_step"
            ):
                # Native on_log runs after checkpoint + evaluation, but before
                # tracker.log. Commit this scalar event exactly once, then leave
                # the native loop without changing its scheduler/epoch horizon.
                trainer.tracker.log(event.logs, step=event.global_step, commit=True)
                raise PlannedPause

        def on_step_end(self, trainer, event, control):
            # Final checkpoint exists before final validation, including a tail batch.
            if event.global_step == event.total_steps:
                control.should_save = True
            if event.global_step == trainer.plan.get("pause_after_step"):
                control.should_save = True
                control.should_evaluate = "dev" in trainer.plan["datasets"]

        def on_eval_end(self, trainer, event, control):
            metrics = event.metrics
            selected = (
                trainer.best is None or metrics["task_macro_loss"] < trainer.best["task_macro_loss"]
            )
            if selected:
                trainer.best = {
                    "optimizer_step": event.global_step,
                    "task_macro_loss": metrics["task_macro_loss"],
                    "token_weighted_loss": metrics["eval_loss"],
                    "checkpoint_path": plan_checkpoint(trainer.plan, event.global_step),
                }
                write_receipt(trainer.output / "BEST_CHECKPOINT.json", trainer.best, replace=True)
            write_receipt(
                trainer.output / "validation" / f"step-{event.global_step:06d}.json",
                {
                    "optimizer_step": event.global_step,
                    **metrics,
                    "plan_sha256": trainer.plan["plan_sha256"],
                },
            )
            trainer.prune_checkpoints()

    class AuditedSFTTrainer(SFTTrainer):
        def __init__(self, cfg, skyrl_cfg, plan):
            super().__init__(cfg, skyrl_cfg=skyrl_cfg, callbacks=[EvidenceCallback()])
            self.plan = plan
            self.output = Path(plan["output_root"])
            self.saved_steps = set()
            self.best = None
            self.extra_train_metrics = {}
            self.target_tokens_seen = 0
            self.last_step_evidence = None
            self.wandb_binding = None
            self.public_runtime_stage = "trainer_constructed"
            self.public_qualification_stage = None

        def _record_runtime_stage(self, stage):
            if stage not in PUBLIC_RUNTIME_STAGES:
                raise ValueError("unknown public runtime stage")
            self.public_runtime_stage = stage
            write_receipt(
                self.output / "RUNTIME_STAGE.json",
                {
                    "optimizer_step": self.global_step,
                    "plan_sha256": self.plan["plan_sha256"],
                    "stage": stage,
                    "observed_at_unix": time.time(),
                },
                replace=True,
            )

        def _record_qualification_stage(self, stage):
            """Expose only the last completed one-step boundary, never model data."""
            if not (_is_qwen38_lora(self.plan) and self.plan.get("pause_after_step") == 1):
                return
            if stage not in QWEN38_QUALIFICATION_STAGES:
                raise ValueError("unknown public qualification stage")
            self.public_qualification_stage = stage
            write_receipt(
                self.output / "QUALIFICATION_STAGE.json",
                {
                    "optimizer_step": self.global_step,
                    "plan_sha256": self.plan["plan_sha256"],
                    "stage": stage,
                    "observed_at_unix": time.time(),
                },
                replace=True,
            )

        def _init_workers(self):
            self._record_runtime_stage("native_worker_initializing")
            selection = contextlib.nullcontext()
            if "recovery" in self.plan:
                from training.recovery import use_worker

                selection = use_worker(self.plan)
            elif "lora" in self.plan and not _is_qwen38_lora(self.plan):
                from training.glm_runtime import use_worker

                selection = use_worker(self.plan)
            with selection:
                super()._init_workers()
            self._record_runtime_stage("native_worker_ready")
            if self.plan["schema"] == DENSE_SCHEMA or (
                "lora" in self.plan and not _is_qwen38_lora(self.plan)
            ):
                # Pinned FSDP2 initialization broadcasts non-persistent buffers
                # (including RoPE inv_freq) back to CPU. Turning off colocation
                # skips the dispatcher's usual initial backload as well as its
                # repeated offloads. Explicitly finish initialization ONCE via
                # the native all-rank API; never recreate buffers or cast values.
                self._record_runtime_stage("device_backload_started")
                actor = self.dispatch._actor_groups["policy"]
                replies = actor.backload_to_gpu(backload_optimizer=False, backload_model=True)
                expected = self.plan["recipe"]["nodes"] * self.plan["recipe"]["gpus_per_node"]
                if (
                    not isinstance(replies, list)
                    or len(replies) != expected
                    or any(x is not None for x in replies)
                ):
                    raise ValueError("initial all-rank model backload was not acknowledged")
                write_receipt(
                    self.output / "DEVICE_INITIALIZED.json",
                    {
                        "plan_sha256": self.plan["plan_sha256"],
                        "ranks_acknowledged": expected,
                        "method": "native_backload_to_gpu_once_after_init",
                        "optimizer_steps": 0,
                        "observed_at_unix": time.time(),
                    },
                )
            self._record_runtime_stage("device_ready")

        def _init_tracker(self):
            self._record_runtime_stage("tracker_initializing")
            self.tracker = explicit_tracking_class(Tracking)(
                project_name=self.cfg.trainer.project_name,
                experiment_name=self.cfg.trainer.run_name,
                backend=self.cfg.trainer.logger,
                config=self.sft_cfg,
                tags=self.cfg.trainer.tags,
            )
            import wandb

            run = wandb.run
            expected = self.plan["wandb"]
            if (
                run is None
                or run.id != expected["run_id"]
                or run.entity != expected["entity"]
                or run.project != expected["project"]
                or run.group != expected["group"]
                or run.name != expected["name"]
                or not isinstance(run.url, str)
                or not run.url.startswith("https://")
            ):
                raise ValueError("W&B run identity mismatch")
            self.wandb_binding = {
                "entity": run.entity,
                "project": run.project,
                "group": run.group,
                "run_id": run.id,
                "name": run.name,
                "url": run.url,
            }
            run.define_metric("train/global_step")
            run.define_metric("train/*", step_metric="train/global_step")
            run.define_metric("eval/*", step_metric="train/global_step")
            run.define_metric("eval/task_macro_loss", summary="min")
            run.config.update(
                {
                    "experiment_plan_sha256": self.plan["plan_sha256"],
                    "model_repo": self.plan["model"]["repo"],
                    "model_revision": self.plan["model"]["revision"],
                    "split_manifest_sha256": self.plan.get("split_manifest_sha256"),
                    "train_rows": self.plan["datasets"]["train"]["rows"],
                    "dev_rows": self.plan["datasets"].get("dev", {}).get("rows", 0),
                    "dev_tasks": len(self.plan["datasets"].get("dev", {}).get("task_keys", [])),
                    "target_policy": (
                        "visible_all_assistant_once"
                        if self.plan["schema"] == DENSE_SCHEMA
                        else "visible_last_assistant_message"
                    ),
                    "dev_target_policy": (
                        "visible_last_assistant_message"
                        if "dev" in self.plan["datasets"]
                        else "none_fresh_task_outcomes_after_training"
                    ),
                    "train_expected_supervised_tokens": self.plan["datasets"]["train"].get(
                        "supervised_tokens"
                    ),
                    "train_source_sessions": self.plan["datasets"]["train"].get("source_sessions"),
                    "train_assistant_responses": self.plan["datasets"]["train"].get(
                        "assistant_responses"
                    ),
                    "train_original_assistant_responses": self.plan["datasets"]["train"].get(
                        "source_total_assistant_responses"
                    ),
                    "train_excluded_assistant_responses": self.plan["datasets"]["train"].get(
                        "excluded_assistant_responses"
                    ),
                    "corpus_manifest_sha256": self.plan.get("corpus_manifest_sha256"),
                    "execution_resources": self.plan.get("execution", {}).get("resources"),
                    "selection_metric": (
                        "eval/task_macro_loss"
                        if "dev" in self.plan["datasets"]
                        else "fresh_fleet_dev_task_success_rate"
                    ),
                    "inline_hf_export": False,
                },
                allow_val_change=False,
            )
            write_receipt(
                self.output / "WANDB.json",
                {
                    "url": run.url,
                    "run_id": run.id,
                    "project": run.project,
                    "entity": run.entity,
                },
            )
            self._record_runtime_stage("tracker_ready")

        def _load_split(self, split):
            import pyarrow.parquet as pq

            spec = self.plan["datasets"][split]
            _checked_file(Path(spec["path"]), spec["sha256"])
            rows = pq.read_table(spec["path"]).to_pylist()
            return prepare_rows(
                rows,
                spec,
                self.tokenizer,
                tokenize_chat_example,
                max_length=self.sft_cfg.max_length,
            )

        def load_dataset(self):
            rows = self._load_split("train")
            if not _is_qwen38_lora(self.plan):
                # The exact FSDP image is SkyRL f5bc3b78.  Its native
                # SFTTrainer consumes ``list[dict]`` directly and does not ship
                # ``skyrl.train.dataset.sft_dataset``.  Importing the newer
                # wrapper on this path fails before the first optimizer step.
                return rows

            # The separately pinned Qwen3.8 Megatron-LoRA image requires every
            # training source to expose ``sequence_lengths`` for its
            # pre-dataloader statistics pass.  Use that image's native wrapper
            # only for the exact plan family whose source census proves it.
            from skyrl.train.dataset.sft_dataset import TextDataset

            return TextDataset(rows)

        def load_eval_dataset(self):
            if "dev" not in self.plan["datasets"]:
                return None
            self.dev_rows = self._load_split("dev")
            return self.dev_rows

        def load_checkpoint(self):
            if "recovery" in self.plan:
                from training.recovery import load

                return load(self)
            return super().load_checkpoint()

        def _policy_learning_rate(self):
            return _qwen38_policy_learning_rate(self.dispatch, self.plan)

        def run_eval(self):
            accumulator = EvalAccumulator()
            cursor = batches = 0
            for batch in self.eval_dataloader:
                n = batch["sequences"].shape[0]
                counts = (batch["loss_mask"] > 0).sum(dim=1).tolist()
                metadata = self.dev_rows[cursor : cursor + n]
                if n < self.eval_dataloader.batch_size:
                    batch = pad_training_input_batch(batch, self.eval_dataloader.batch_size - n)
                output = self.dispatch.forward(
                    "policy", batch, loss_fn="cross_entropy", loss_fn_config=None
                )
                accumulator.add_batch(
                    metadata,
                    counts,
                    output.loss_fn_outputs,
                    float(output.metrics["loss"]),
                )
                cursor += n
                batches += 1
                write_receipt(
                    self.output / "ACTIVITY.json",
                    {
                        "phase": "validation",
                        "optimizer_step": self.global_step,
                        "completed_batches": batches,
                        "observed_at_unix": time.time(),
                    },
                    replace=True,
                )
            if cursor != len(self.dev_rows):
                raise ValueError("validation did not consume the complete held-out dataset")
            return (
                accumulator.metrics(set(self.plan["datasets"]["dev"]["task_keys"])),
                batches,
            )

        def train_step(self, batch, step):
            # Exact upstream calls.  Megatron emits ``policy_lr`` before the
            # update, while the generic dispatcher drops its all-rank shape.
            # Query every native optimizer rank before mutation so missing,
            # divergent or invalid evidence cannot produce an unaudited step.
            timings = {}
            if _is_qwen38_lora(self.plan):
                self._record_qualification_stage("forward_backward_started")
            with Timer("forward_backward", timings):
                # SFT consumes scalar metrics only.  The pinned SkyRL trainer
                # explicitly disables its per-token outputs here; keep the
                # audited override identical so 32K batches do not construct
                # and transport unused logprob/loss arrays every step.
                output = _scalar_sft_forward_backward(self.dispatch, batch)
            if _is_qwen38_lora(self.plan):
                self._record_qualification_stage("forward_backward_complete")
            loss = float(output.metrics.get("final_loss", output.metrics.get("loss", float("nan"))))
            if _is_qwen38_lora(self.plan):
                self._record_qualification_stage("rank_lr_query_started")
                lr = self._policy_learning_rate()
                self._record_qualification_stage("rank_lr_query_complete")
                self._record_qualification_stage("worker_lr_validation_started")
                metric_lr = _reconcile_worker_learning_rates(
                    [output.metrics.get("policy_lr")], expected_ranks=1
                )
                self._record_qualification_stage("worker_lr_validated")
                # ``policy_lr`` passes through SkyRL's scalar all-reduce, which
                # materializes Python values as float32 before returning them.
                # The direct optimizer query remains a Python float.  Compare
                # within a bound far tighter than any meaningful schedule
                # change while accepting that one expected float32 round trip.
                if not math.isclose(metric_lr, lr, rel_tol=1e-6, abs_tol=0.0):
                    raise ValueError("worker metric learning rate differs from optimizer ranks")
                self._record_qualification_stage("lr_consensus_validated")
            else:
                lr = float(output.metrics["lr"])
            if not math.isfinite(loss) or not math.isfinite(lr) or lr <= 0:
                raise ValueError("nonfinite training metric")
            if _is_qwen38_lora(self.plan):
                self._record_qualification_stage("optimizer_update_started")
            with Timer("optim_step", timings):
                grad_norm = self.dispatch.optim_step("policy")
            if _is_qwen38_lora(self.plan):
                self._record_qualification_stage("optimizer_update_returned")
            if self._torch_profiler_enabled:
                self.dispatch.profile_step("policy")
            gradient = float(grad_norm)
            if not math.isfinite(gradient) or (_is_qwen38_lora(self.plan) and gradient <= 0):
                raise ValueError("nonfinite training metric")
            if _is_qwen38_lora(self.plan):
                self._record_qualification_stage("gradient_validated")
            self.last_step_evidence = {
                "forward_loss": loss,
                "lora_gradient_norm": gradient,
            }
            if _is_qwen38_lora(self.plan):
                self._record_qualification_stage("step_evidence_ready")
            targets = int((batch["loss_mask"] > 0).sum().item())
            self.target_tokens_seen += targets
            self.extra_train_metrics = {
                "train/lr": lr,
                "train/supervised_tokens": targets,
                "train/total_supervised_tokens": self.target_tokens_seen,
            }
            return {"loss": loss, "grad_norm": gradient, "timings": timings}

        def save_checkpoint(self):
            import torch

            path = Path(super().save_checkpoint())
            # Native save catches sampler-write errors, including a partially
            # written file. Reopen only the two small, trusted metadata files;
            # hashing the full model remains a separate CPU sealing operation.
            metadata = []
            for name in ("data.pt", "trainer_state.pt"):
                source = path / name
                if source.is_symlink() or not source.is_file():
                    raise ValueError("checkpoint lacks sampler/trainer resume state")
                metadata.append(torch.load(source, map_location="cpu", weights_only=False))
            batches = math.ceil(
                self.plan["datasets"]["train"]["rows"] / self.plan["recipe"]["batch_size"]
            )
            if metadata[0].get("_num_yielded") != (self.global_step - 1) % batches + 1 or (
                metadata[1].get("global_step") != self.global_step
            ):
                raise ValueError("checkpoint sampler/trainer counters differ from completed step")
            self.saved_steps.add(self.global_step)
            write_receipt(
                self.output / "checkpoint_receipts" / f"step-{self.global_step:06d}.json",
                {
                    "optimizer_step": self.global_step,
                    "checkpoint_path": str(path),
                    "plan_sha256": self.plan["plan_sha256"],
                    "saved_at_unix": time.time(),
                    "training_progress": {
                        "supervised_tokens": self.target_tokens_seen,
                        "best": self.best,
                    },
                },
            )
            self.prune_checkpoints()
            return str(path)

        def prune_checkpoints(self):
            keep = retention_steps(
                self.saved_steps,
                self.best["optimizer_step"] if self.best else 0,
                self.plan["recipe"]["keep_checkpoints"],
            )
            for step in sorted(self.saved_steps - keep):
                path = self.output / "checkpoints" / f"global_step_{step}"
                if (
                    path.is_symlink()
                    or path.resolve().parent != (self.output / "checkpoints").resolve()
                ):
                    raise ValueError("checkpoint retention target escaped owned root")
                shutil.rmtree(path)
                self.saved_steps.remove(step)

        def save_hf_model(self):
            raise RuntimeError(
                "inline HF export is prohibited; "
                "use the completed checkpoint in a separate export job"
            )

    return AuditedSFTTrainer


def plan_checkpoint(plan: dict, step: int) -> str:
    if "recovery" in plan and step == plan["recovery"]["checkpoint"]["optimizer_step"]:
        return plan["recovery"]["checkpoint"]["checkpoint_path"]
    return (
        plan["model"]["root"]
        if step == 0
        else str(Path(plan["output_root"]) / "checkpoints" / f"global_step_{step}")
    )


def training_result(trainer, *, paused: bool) -> dict:
    """A planned pause is a recoverable partial run, never full completion."""
    plan = trainer.plan
    expected = plan.get("pause_after_step") if paused else plan["recipe"]["max_steps"]
    if expected is None or ("pause_after_step" in plan and not paused):
        raise ValueError("native loop did not honor the planned lifecycle")
    pointer = trainer.output / "checkpoints/latest_ckpt_global_step.txt"
    if int(pointer.read_text()) != expected or trainer.global_step != expected:
        raise ValueError("final checkpoint optimizer step mismatch")
    # A native step counter/pointer alone does not prove a saved, evaluated
    # checkpoint. Both complete and deliberately paused runs need the receipts.
    terminal_receipts = ["checkpoint_receipts"]
    if "dev" in plan["datasets"]:
        terminal_receipts.append("validation")
    for kind in terminal_receipts:
        path = trainer.output / kind / f"step-{expected:06d}.json"
        if path.is_symlink():
            raise ValueError("terminal receipt must not be a symlink")
        proof = json.loads(path.read_text())
        if (
            proof.get("receipt_sha256")
            != _unsigned_digest({k: v for k, v in proof.items() if k != "receipt_sha256"})
            or proof.get("optimizer_step") != expected
            or proof.get("plan_sha256") != plan["plan_sha256"]
            or (
                kind == "checkpoint_receipts"
                and proof.get("checkpoint_path") != plan_checkpoint(plan, expected)
            )
        ):
            raise ValueError("terminal result lacks a matching save or validation receipt")
    if (
        not paused
        and plan["schema"] == DENSE_SCHEMA
        and trainer.target_tokens_seen
        != plan["datasets"]["train"]["supervised_tokens"] * plan["recipe"]["epochs"]
    ):
        raise ValueError("completed epochs did not train every assistant target once per epoch")
    return {
        "optimizer_step": expected,
        "planned_optimizer_steps": plan["recipe"]["max_steps"],
        "optimizer_steps_executed": expected
        - plan.get("recovery", {}).get("checkpoint", {}).get("optimizer_step", 0),
        "checkpoint_path": plan_checkpoint(plan, expected),
        "best": trainer.best,
        "export_status": "pending_separate_zero_step_export",
        "supervised_tokens": trainer.target_tokens_seen,
        "plan_sha256": plan["plan_sha256"],
        "status": "training_paused" if paused else "training_complete",
    }


def _run_training(plan: dict) -> dict:
    _configure_wandb(plan)
    cfg, skyrl_cfg = build_runtime_configs(plan)
    skyrl_cfg.trainer.log_path = str(Path(plan["output_root"]) / "private_logs")
    trainer = _make_trainer_class()(cfg, skyrl_cfg, plan)
    try:
        one_step_gate = _is_qwen38_lora_one_step_gate(plan)
        qwen38_source_before = _qwen38_source_inventory(plan) if one_step_gate else None
        qwen38_runtime_before = _qwen38_runtime_source_inventory(plan) if one_step_gate else None
        trainer.setup()
        qwen38_lora_before = (
            trainer.dispatch.collect_lora_qualification_snapshots("policy")
            if one_step_gate
            else None
        )
        if plan.get("recovery", {}).get("mode") == "validate":
            from training.recovery import validate_only

            return validate_only(trainer)
        paused = False
        qwen38_checkpoint_finalization = None
        try:
            trainer.train()
        except PlannedPause:
            paused = True
            # PlannedPause is raised from on_log before the native loop's
            # postamble.  Explicitly drain the async dist-checkpoint writer
            # before reading any file, collecting post-update hashes, or
            # preparing a qualification receipt.
            if one_step_gate:
                trainer._record_qualification_stage("checkpoint_finalization_started")
                qwen38_checkpoint_finalization = (
                    trainer.dispatch.finalize_lora_qualification_checkpoint("policy")
                )
                trainer._record_qualification_stage("checkpoint_finalization_complete")
            else:
                trainer.dispatch.finalize_pending_saves("policy")
        qwen38_receipt = None
        if one_step_gate:
            if not paused or qwen38_lora_before is None:
                raise ValueError("Qwen3.8 qualification did not stop at its one-step gate")
            if qwen38_source_before is None or qwen38_runtime_before is None:
                raise ValueError("Qwen3.8 pre-step source inventory is missing")
            trainer._record_qualification_stage("post_update_snapshot_started")
            qwen38_lora_after = trainer.dispatch.collect_lora_qualification_snapshots("policy")
            trainer._record_qualification_stage("post_update_snapshot_complete")
            trainer._record_qualification_stage("qualification_receipt_preparation_started")
            qwen38_receipt = _prepare_qwen38_checkpoint_receipt(
                trainer,
                qwen38_lora_before,
                qwen38_lora_after,
                qwen38_checkpoint_finalization,
                qwen38_source_before,
                qwen38_runtime_before,
            )
            trainer._record_qualification_stage("qualification_receipt_prepared")
        result = training_result(trainer, paused=paused)
        if _is_qwen38_lora(plan):
            trainer._record_qualification_stage("terminal_result_validated")
        import wandb

        wandb.run.summary.update(
            {
                "completed_optimizer_steps": result["optimizer_step"],
                "status": result["status"],
                "planned_optimizer_steps": result["planned_optimizer_steps"],
                "best_checkpoint": trainer.best,
                "export_status": result["export_status"],
            }
        )
        if qwen38_receipt is not None:
            # Qualification requires an observed successful flush, not the
            # upstream best-effort Tracking.finish() wrapper.
            trainer._record_qualification_stage("wandb_finish_started")
            wandb.finish(exit_code=0)
            if wandb.run is not None:
                raise ValueError("W&B run remained active after finish")
            trainer.tracker = None
            trainer._record_qualification_stage("wandb_finished")
        trainer.shutdown()
        if qwen38_receipt is not None:
            from training.qwen38_lora_artifacts import validate_checkpoint_receipt

            trainer._record_qualification_stage("trainer_shutdown_complete")
            if wandb.run is not None:
                raise ValueError("W&B run reappeared or remained active after trainer shutdown")
            qwen38_receipt["wandb"] = _qwen38_wandb_evidence(trainer)
            signed = {
                **qwen38_receipt,
                "receipt_sha256": _unsigned_digest(qwen38_receipt),
            }
            validate_checkpoint_receipt(signed)
            trainer._record_qualification_stage("qualification_receipt_validated")
            receipt_path = trainer.output / "QWEN38_LORA_CHECKPOINT.json"
            write_receipt(receipt_path, qwen38_receipt)
            written = json.loads(receipt_path.read_text())
            identity = validate_checkpoint_receipt(written)
            result["qwen38_lora_checkpoint_receipt"] = {
                "path": str(receipt_path),
                "file_sha256": digest(receipt_path),
                "receipt_sha256": identity["receipt_sha256"],
            }
        return result
    except BaseException as exc:
        # Do not use native log_exception: it uploads raw traceback and finishes exit0.
        incomplete = finalize_failed_run(trainer, Path(plan["output_root"]), exc)
        if incomplete:
            print(
                json.dumps({"status": "failure_cleanup_incomplete", "components": incomplete}),
                flush=True,
            )
        raise RuntimeError(f"SFT runtime failed: {type(exc).__name__}") from None


def _run_setup_probe(plan: dict, *, with_tracker: bool = False) -> dict:
    """Exercise exact model/FSDP setup and the no-dev loader without optimizer steps."""
    cfg, skyrl_cfg = build_runtime_configs(plan)
    skyrl_cfg.trainer.log_path = str(Path(plan["output_root"]) / "private_logs")
    trainer_class = _make_trainer_class()

    if not with_tracker:

        def bypass_tracker(trainer):
            trainer.tracker = None
            trainer._record_runtime_stage("tracker_bypassed_for_setup_probe")

        trainer_class._init_tracker = bypass_tracker
    trainer = trainer_class(cfg, skyrl_cfg, plan)
    try:
        trainer.setup()
        snapshot_rank_count = None
        qwen38_dataset_rows = None
        qwen38_initial_lr = None
        if _is_qwen38_lora(plan):
            snapshots = trainer.dispatch.collect_lora_qualification_snapshots("policy")
            snapshot_rank_count = len(
                _qwen38_rank_snapshots(snapshots, stage="setup probe pre-step")
            )
            qwen38_initial_lr = trainer._policy_learning_rate()
            if not math.isclose(
                qwen38_initial_lr,
                float(plan["recipe"]["lr"]),
                rel_tol=1e-12,
                abs_tol=0.0,
            ):
                raise ValueError("Qwen3.8 native optimizer LR does not match the plan")
            # Exercise the exact pinned image's post-setup dataset interface.
            # SkyRL reads ``sequence_lengths`` before it constructs the first
            # dataloader; an earlier probe stopped just short of that boundary
            # and therefore missed a list-vs-SFTDataset contract drift.
            training_dataset = trainer.load_dataset()
            trainer._log_dataset_stats(training_dataset)
            lengths = [int(v) for v in training_dataset.sequence_lengths]
            qwen38_dataset_rows = len(training_dataset)
            if (
                qwen38_dataset_rows != plan["datasets"]["train"]["rows"]
                or len(lengths) != qwen38_dataset_rows
                or not lengths
                or min(lengths) <= 0
                or max(lengths) > plan["recipe"]["max_length"]
            ):
                raise ValueError("Qwen3.8 native dataset contract does not match the plan")
        if "dev" not in plan["datasets"] and trainer.load_eval_dataset() is not None:
            raise ValueError("task-outcome training unexpectedly produced an eval dataset")
        result = {
            "optimizer_steps": 0,
            "plan_sha256": plan["plan_sha256"],
            "stage": trainer.public_runtime_stage,
            "status": "setup_validated",
        }
        if snapshot_rank_count is not None:
            result["snapshot_rank_count"] = snapshot_rank_count
            result["learning_rate_rank_count"] = (
                plan["recipe"]["nodes"] * plan["recipe"]["gpus_per_node"]
            )
            result["initial_learning_rate"] = qwen38_initial_lr
        if qwen38_dataset_rows is not None:
            result["dataset_contract_rows"] = qwen38_dataset_rows
        return result
    except BaseException as exc:
        return {
            **public_failure_details(exc),
            "optimizer_steps": 0,
            "plan_sha256": plan["plan_sha256"],
            "stage": trainer.public_runtime_stage,
            "status": "setup_rejected",
        }
    finally:
        with contextlib.suppress(BaseException):
            trainer.shutdown()


def _setup_probe_plan(
    plan: dict,
    output_root: Path | None,
    *,
    jobs_root: Path = Path("/mnt/sfs/jobs"),
) -> dict:
    """Bind setup-only evidence to a new narrow output without changing the plan.

    The immutable training plan is validated before this function is called. A
    setup probe must not write into that plan's eventual training destination,
    nor into a previous failed run. Keep the original plan digest in the copied
    runtime value while redirecting setup-only caches and receipts to one
    create-once sibling directory.
    """
    if output_root is None:
        raise ValueError("setup probe requires an explicit output root")
    root = Path(output_root)
    if (
        root.parent != jobs_root
        or not re.fullmatch(r"chris-q38-lora-setup-probe-[a-z0-9-]{1,80}", root.name)
        or str(root) == plan["output_root"]
    ):
        raise ValueError("setup probe output must be a new specific owned SFS directory")
    if root.exists() or root.is_symlink():
        raise ValueError("setup probe output already exists")
    return {**plan, "output_root": str(root)}


def _create_runtime_output(output: Path, *, setup_probe: bool) -> None:
    """Create the runtime output, atomically refusing probe-path reuse."""
    if setup_probe:
        output.mkdir(mode=0o700)
        return
    output.mkdir(parents=True, exist_ok=True, mode=0o700)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preflight-tokenize", action="store_true")
    parser.add_argument("--setup-probe", action="store_true")
    parser.add_argument("--setup-probe-with-tracker", action="store_true")
    parser.add_argument("--probe-output-root", type=Path)
    args = parser.parse_args()
    setup_probe = args.setup_probe or args.setup_probe_with_tracker
    if setup_probe != (args.probe_output_root is not None):
        raise ValueError("probe output root and setup-probe mode must be selected together")
    _checked_file(args.plan, args.plan_sha256)
    plan = json.loads(args.plan.read_text())
    plan["plan_sha256"] = args.plan_sha256.removeprefix("sha256:")
    # Qwen3.8 training records one complete source read immediately before
    # trainer setup and one after the optimizer update.  Keep this entrypoint
    # validation structural so it does not add another 55.6 GB hash pass.
    _validate_entrypoint_sources(
        plan,
        verify_qwen_files=(
            args.preflight_tokenize
            or args.validate_only
            or args.setup_probe
            or args.setup_probe_with_tracker
        ),
    )
    if args.preflight_tokenize:
        import pyarrow.parquet as pq
        from skyrl.train.sft_trainer import tokenize_chat_example
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            plan["model"]["root"], local_files_only=True, trust_remote_code=True
        )
        results = {}
        for split, spec in plan["datasets"].items():
            rows = prepare_rows(
                pq.read_table(spec["path"]).to_pylist(),
                spec,
                tokenizer,
                tokenize_chat_example,
                max_length=plan["recipe"]["max_length"],
            )
            results[split] = {
                "rows": len(rows),
                "tasks": len(spec["task_keys"]),
                "supervised_tokens": sum(sum(x["loss_mask"]) for x in rows),
            }
        print(json.dumps({"status": "tokenization_preflight_passed", **results}))
        return
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "dev_tasks": len(plan["datasets"].get("dev", {}).get("task_keys", [])),
                    "max_steps": plan["recipe"]["max_steps"],
                }
            )
        )
        return
    if setup_probe:
        plan = _setup_probe_plan(plan, args.probe_output_root)
    output = Path(plan["output_root"])
    _create_runtime_output(output, setup_probe=setup_probe)
    if setup_probe:
        write_receipt(
            output / "SETUP_PROBE_STARTED.json",
            {"plan_sha256": plan["plan_sha256"], "started_at_unix": time.time()},
        )
        ray = None
        try:
            if args.setup_probe_with_tracker:
                _configure_wandb(plan)
            import ray
            from skyrl.train.utils.utils import initialize_ray

            _, cfg = build_runtime_configs(plan)
            cfg.trainer.log_path = str(output / "private_logs")
            initialize_ray(cfg)
            result = _run_setup_probe(plan, with_tracker=args.setup_probe_with_tracker)
        except BaseException as exc:
            result = {
                **public_failure_details(exc),
                "optimizer_steps": 0,
                "plan_sha256": plan["plan_sha256"],
                "stage": "probe_bootstrap",
                "status": "setup_rejected",
            }
        finally:
            if ray is not None and ray.is_initialized():
                ray.shutdown()
        terminal = (
            "SETUP_VALIDATED.json"
            if result["status"] == "setup_validated"
            else "SETUP_REJECTED.json"
        )
        write_receipt(output / terminal, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    write_receipt(
        output / "STARTED.json",
        {"plan_sha256": plan["plan_sha256"], "started_at_unix": time.time()},
    )
    ray = None
    try:
        import ray
        from skyrl.train.utils.utils import initialize_ray

        _, cfg = build_runtime_configs(plan)
        cfg.trainer.log_path = str(output / "private_logs")
        initialize_ray(cfg)
        task = ray.remote(num_cpus=1)(_run_training).remote(plan)
        result = _wait_for_training(ray, task, output)
        terminal = (
            "RELOAD_VALIDATED.json"
            if result["status"] == "reload_validated"
            else (
                "TRAINING_PAUSED.json"
                if result["status"] == "training_paused"
                else "TRAINING_COMPLETE.json"
            )
        )
        write_receipt(output / terminal, result)
        print(json.dumps({"status": result["status"], "optimizer_step": result["optimizer_step"]}))
    except BaseException as exc:
        write_receipt(
            output / "FAILED.json",
            {
                "error_class": type(exc).__name__,
                "plan_sha256": plan["plan_sha256"],
                "failed_at_unix": time.time(),
                "status": "failed",
            },
        )
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    finally:
        if ray is not None and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
