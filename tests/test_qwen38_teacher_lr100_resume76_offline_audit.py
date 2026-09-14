"""Offline launch-audit checks for the blocked LR1e-4 step-21 continuation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from cyber_post_train.jobs import validate_request
from training import checkpoints, recovery
from training.io import digest_json, file_sha256
from training.sft import IMAGE, job_request
from training.sft_runtime import (
    DENSE_FORMAT,
    DENSE_SCHEMA,
    OUTCOME_WANDB_HISTORY_KEYS,
    WATCHDOG_DRAIN_SECONDS,
    WATCHDOG_HARD_SECONDS,
    WATCHDOG_IDLE_SECONDS,
    WATCHDOG_POLL_SECONDS,
    WATCHDOG_STARTUP_SECONDS,
    optimizer_schedule,
    retention_steps,
    selection_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT / "configs/qualification/qwen38-teacher-lr100-step21-resume-to76-dev-v1.template.json"
)
AUDIT = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-14-lr100-step21-resume76-offline-launch-audit-v1.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sealed(path: Path) -> dict:
    value = read(path)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def set_dotted(value: dict, dotted: str, item: object) -> None:
    target = value
    pieces = dotted.split(".")
    for piece in pieces[:-1]:
        target = target[piece]
    target[pieces[-1]] = copy.deepcopy(item)


def blocked_config_projection(template: dict) -> dict:
    source = read(ROOT / template["source"]["config"]["path"])
    result = copy.deepcopy(source)
    transform = template["materialization"]["source_config_transform"]
    for key in transform["remove"]:
        del result[key]
    for dotted, item in transform["replace"].items():
        set_dotted(result, dotted, item)
    result.update(copy.deepcopy(transform["add"]))
    return result


def synthetic_plan(config: dict) -> dict:
    """Small private-data-free plan used only to exercise the request boundary."""
    return {
        "runtime_sha256": file_sha256(ROOT / "training/sft_runtime.py").removeprefix("sha256:"),
        "recovery_runtime_sha256": file_sha256(ROOT / "training/recovery.py").removeprefix(
            "sha256:"
        ),
        "run_name": config["name"],
        "output_root": config["output_root"],
        "recipe": {
            "nodes": config["recipe"]["nodes"],
            "gpus_per_node": config["recipe"]["gpus_per_node"],
        },
        "wandb": config["wandb"],
        "execution": {
            "image": IMAGE,
            "priority": config["cluster"]["priority"],
            "resources": config["cluster"]["resources"],
        },
        "recovery": {
            "mode": "resume",
            "manifest_file_sha256": "blocked-by-live-pair-proof",
            "checkpoint": {"optimizer_step": 21},
        },
    }


def fixed_request_projection(request: dict) -> dict:
    result = copy.deepcopy(request)
    result["command"] = None
    result["env"]["CYBER_SFT_BUNDLE"] = None
    return result


def synthetic_recovery_plan() -> dict:
    """A schema-valid exact-topology fixture; no source checkpoint bytes are implied."""
    recipe = {
        "epochs": 1,
        "batch_size": 8,
        "microbatch_per_gpu": 1,
        "nodes": 1,
        "gpus_per_node": 8,
        "lr": 1e-4,
        "max_length": 16384,
        "eval_interval": 0,
        "checkpoint_interval": 20,
        "keep_checkpoints": 3,
        "seed": 42,
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "max_steps": 76,
    }
    source = {
        "schema": DENSE_SCHEMA,
        "run_name": "source-run",
        "output_root": "/mnt/sfs/jobs/source-run",
        "model": {
            "repo": "Qwen/Qwen3.5-27B",
            "revision": "a" * 40,
            "root": "/mnt/sfs/models/exact-model",
            "files": [
                {"path": "config.json", "sha256": "7" * 64},
                {"path": "tokenizer_config.json", "sha256": "8" * 64},
            ],
        },
        "datasets": {
            "train": {
                "path": "/mnt/sfs/jobs/corpus/train.parquet",
                "sha256": "1" * 64,
                "task_keys": ["task-family"],
                "rows": 602,
                "format": DENSE_FORMAT,
                "supervised_tokens": 700359,
                "assistant_responses": 3242,
                "source_sessions": 71,
                "source_total_assistant_responses": 3457,
                "excluded_assistant_responses": 215,
            }
        },
        "recipe": recipe,
        "wandb": {
            "project": "cyber-post-train",
            "entity": "thefleet",
            "group": "offline-test",
            "run_id": "source-run",
            "name": "source-run",
        },
        "split_manifest_sha256": "sha256:" + "2" * 64,
        "corpus_manifest_sha256": "sha256:" + "3" * 64,
        "validation_mode": "task_outcomes_only",
        "fleet_dev_protocol_sha256": "sha256:" + "4" * 64,
        "runtime_sha256": file_sha256(ROOT / "training/sft_runtime.py").removeprefix("sha256:"),
        "execution": {"image": IMAGE},
    }
    names = {"data.pt", "trainer_state.pt", "policy/fsdp_config.json"}
    names |= {
        f"policy/{kind}_world_size_8_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state")
        for rank in range(8)
    }
    names.add("policy/huggingface/config.json")
    inventory = {name: {"bytes": 1, "sha256": "5" * 64} for name in sorted(names)}
    manifest = {
        "schema": "cyber_skyrl_checkpoint_manifest_v1",
        "source_plan_sha256": digest_json(source).removeprefix("sha256:"),
        "source_plan": source,
        "checkpoint_path": "/mnt/sfs/jobs/source-run/checkpoints/global_step_21",
        "optimizer_step": 21,
        "world_size": 8,
        "sampler_batches_in_epoch": 21,
        "files": inventory,
        "total_bytes": len(inventory),
        "gpu_reload_verified": False,
        "training_progress": {"supervised_tokens": 200000, "best": None},
        **selection_evidence(source),
    }
    manifest["receipt_sha256"] = digest_json(manifest).removeprefix("sha256:")
    target = copy.deepcopy(source)
    target.update(
        run_name="continuation-run",
        output_root="/mnt/sfs/jobs/continuation-run",
        wandb={
            "project": "cyber-post-train",
            "entity": "thefleet",
            "group": "offline-test",
            "run_id": "continuation-run",
            "name": "continuation-run",
        },
        recovery={
            "mode": "resume",
            "checkpoint": manifest,
            "manifest_file_sha256": "6" * 64,
        },
        recovery_runtime_sha256=file_sha256(ROOT / "training/recovery.py").removeprefix("sha256:"),
    )
    return target


def test_audit_is_self_digesting_and_exactly_source_bound() -> None:
    audit = sealed(AUDIT)
    template = sealed(TEMPLATE)
    for key in (
        "template",
        "producer_config",
        "producer_terminal_evidence",
        "checkpoint_pair_verifier",
        "step21_reload_template",
        "model_lock",
        "model_weights",
    ):
        binding = audit["source"][key]
        assert binding["file_sha256"] == file_sha256(ROOT / binding["path"])
    assert audit["source"]["template"]["embedded_sha256"] == template["sha256"]
    handoff = sealed(ROOT / audit["source"]["producer_terminal_evidence"]["path"])
    assert audit["source"]["producer_terminal_evidence"]["embedded_sha256"] == handoff["sha256"]
    assert handoff["acceptance"]["independent_checkpoint_pair_verification_accepted"] is False
    assert handoff["acceptance"]["zero_optimizer_reload_accepted"] is False


def test_blocked_materialization_cannot_fabricate_an_exact_request() -> None:
    audit = sealed(AUDIT)
    template = sealed(TEMPLATE)
    config = blocked_config_projection(template)
    assert (
        digest_json(config)
        == audit["conditional_materialization"]["blocked_config_projection_canonical_sha256"]
    )
    assert config["recovery"] == {
        "manifest": template["source"]["checkpoint_manifest"],
        "sha256": None,
        "mode": "resume",
    }
    blockers = audit["hard_blockers_preserved"]
    assert blockers["checkpoint_pair_terminal_proof_accepted"] is False
    assert blockers["step21_zero_update_all_rank_reload_accepted"] is False
    assert blockers["materialized_continuation_config_created"] is False
    assert blockers["continuation_plan_created"] is False
    assert blockers["continuation_request_created"] is False
    assert blockers["continuation_runtime_bundle_created"] is False
    assert all(
        audit["conditional_materialization"][name] is None
        for name in (
            "exact_materialized_config_file_sha256",
            "exact_plan_sha256",
            "exact_request_sha256",
            "exact_runtime_bundle_sha256",
            "exact_command_sha256",
        )
    )


def test_fixed_request_projection_comes_from_the_current_generator() -> None:
    audit = sealed(AUDIT)
    config = blocked_config_projection(sealed(TEMPLATE))
    request = job_request(synthetic_plan(config))
    validate_request(request)
    projection = fixed_request_projection(request)
    expected = audit["conditional_request_projection"]
    assert projection == expected["request"]
    assert digest_json(projection) == expected["fixed_field_projection_canonical_sha256"]
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == ["wandb-api"]
    assert not any(name.endswith(("TOKEN", "PASSWORD", "API_KEY")) for name in request["env"])
    assert expected["exact_request_bytes_available_before_live_blockers_pass"] is False


def test_runtime_bundle_file_bindings_are_current_and_non_secret() -> None:
    audit = sealed(AUDIT)
    runtime = audit["runtime_binding"]
    assert runtime["compiler"]["file_sha256"] == file_sha256(ROOT / runtime["compiler"]["path"])
    for relative, expected in runtime["runtime_files"].items():
        observed = (
            "sha256:" + hashlib.sha256(b"").hexdigest()
            if relative == "training/__init__.py"
            else file_sha256(ROOT / relative)
        )
        assert expected == observed
    assert runtime["package_initializer_is_intentionally_empty_in_the_bundle"] is True
    assert runtime["exact_bundle_and_bootstrap_hashes_depend_on_the_accepted_manifest"] is True


def test_resume_requires_the_exact_one_by_eight_source_topology() -> None:
    target = synthetic_recovery_plan()
    checkpoints.verify(target["recovery"]["checkpoint"], check_files=False)
    recovery.validate(target, check_files=False)
    target["recipe"] = {**target["recipe"], "nodes": 2, "gpus_per_node": 4}
    with pytest.raises(ValueError, match="topology"):
        recovery.validate(target, check_files=False)


def test_resume_math_checkpoint_cadence_and_wandb_are_exact() -> None:
    audit = sealed(AUDIT)
    science = audit["scientific_contract"]
    cadence = audit["resume_math_and_cadence"]
    assert cadence["terminal_optimizer_step"] == math.ceil(602 / 8) == 76
    assert cadence["continuation_optimizer_steps"] == 76 - 21 == 55
    assert cadence["continuation_global_step_range"] == [22, 76]
    assert cadence["new_checkpoint_steps"] == [40, 60, 76]
    assert retention_steps({40, 60, 76}, 0, 3) == {40, 60, 76}
    assert cadence["expected_retained_new_checkpoint_steps"] == [40, 60, 76]
    recipe = {
        "max_steps": 76,
        "scheduler": science["scheduler"],
        "warmup_ratio": science["warmup_ratio"],
    }
    assert optimizer_schedule(recipe)["num_warmup_steps"] == science["warmup_steps"] == 4
    wandb = audit["wandb_contract"]
    assert wandb["resume_policy"] == "never"
    assert wandb["expected_scalar_events"] == 55
    assert wandb["expected_global_step_range"] == [22, 76]
    assert tuple(wandb["required_scalar_fields"]) == OUTCOME_WANDB_HISTORY_KEYS
    assert wandb["tracking_incomplete_is_training_acceptance"] is False
    assert science["training_loss_selection_eligible"] is False
    assert science["webexploitbench_selection_eligible"] is False


def test_watchdog_failure_release_and_readiness_remain_fail_closed() -> None:
    audit = sealed(AUDIT)
    cadence = audit["resume_math_and_cadence"]
    assert (
        cadence["watchdog_poll_seconds"],
        cadence["watchdog_startup_seconds"],
        cadence["watchdog_idle_seconds"],
        cadence["watchdog_hard_seconds"],
        cadence["watchdog_drain_seconds"],
    ) == (
        WATCHDOG_POLL_SECONDS,
        WATCHDOG_STARTUP_SECONDS,
        WATCHDOG_IDLE_SECONDS,
        WATCHDOG_HARD_SECONDS,
        WATCHDOG_DRAIN_SECONDS,
    )
    failure = audit["terminal_failure_and_release_contract"]
    assert failure["unexpected_failure_exit_code"] == "nonzero"
    assert failure["unexpected_failure_may_be_automatically_retried"] is False
    assert failure["runtime_writes_a_gpu_release_receipt"] is False
    assert failure["preview_must_prove_shutdown_after_job_finishes"] is True
    assert audit["execution_contract"]["automatic_requeue"] is False
    assert audit["readiness"] == {
        "scientific_contract_qualified": True,
        "fixed_request_envelope_qualified": True,
        "exact_plan_request_command_and_bundle_sealed": False,
        "direct_submission_ready": False,
        "checkpoint_pair_and_reload_blockers_preserved": True,
        "production_submission_authorized": False,
    }
    assert len(audit["ordered_live_gates"]) == 16
    assert audit["static_validation"]["defects_found_in_rank_two_template"] == 0
    assert audit["static_validation"]["live_or_network_calls_performed"] == 0
    assert audit["static_validation"]["jobs_api_posts"] == 0
    assert not any(audit["privacy"].values())
