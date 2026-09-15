"""Offline launch-audit checks for the LR3e-5 step-6 continuation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import validate_request
from training import recovery
from training.io import digest_json, file_sha256
from training.sft import IMAGE, job_request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qualification/qwen38-teacher-lr30-step6-resume-to76-dev-v1.json"
TEMPLATE = CONFIG.with_name(CONFIG.stem + ".template.json")
AUDIT = (
    ROOT / "docs/evidence/qwen38-study/2026-09-14-lr30-step6-resume76-offline-launch-audit-v1.json"
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


def test_materialized_config_is_the_exact_rank_one_transform() -> None:
    template = sealed(TEMPLATE)
    source_binding = template["source"]["config"]
    source_path = ROOT / source_binding["path"]
    source = read(source_path)
    expected = copy.deepcopy(source)
    transform = template["materialization"]["source_config_transform"]

    assert source_binding["file_sha256"] == file_sha256(source_path)
    for key in transform["remove"]:
        del expected[key]
    for dotted, item in transform["replace"].items():
        set_dotted(expected, dotted, item)
    expected.update(copy.deepcopy(transform["add"]))

    config = read(CONFIG)
    assert config == expected
    assert "pause_after_step" not in config
    assert config["recovery"] == {
        "manifest": template["source"]["checkpoint_manifest"],
        "sha256": template["source"]["checkpoint_manifest_file_sha256"].removeprefix("sha256:"),
        "mode": "resume",
    }
    assert config["recipe"] == source["recipe"]
    assert config["model"] == source["model"]
    assert config["data"] == source["data"]


def test_audit_is_self_digesting_and_source_bound() -> None:
    audit = sealed(AUDIT)
    for key in (
        "materialized_config",
        "template",
        "producer_config",
        "producer_terminal_evidence",
        "model_lock",
        "model_weights",
    ):
        binding = audit["source"][key]
        assert binding["file_sha256"] == file_sha256(ROOT / binding["path"])

    config = read(CONFIG)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    assert audit["source"]["materialized_config"]["canonical_sha256"] == (
        "sha256:" + hashlib.sha256(canonical).hexdigest()
    )
    terminal = sealed(ROOT / audit["source"]["producer_terminal_evidence"]["path"])
    assert terminal["acceptance"]["zero_update_reload_accepted"] is True
    assert (
        terminal["independent_post_reload_verification"]["source_checkpoint_unchanged_after_reload"]
        is True
    )
    assert (
        terminal["checkpoint_seal"]["manifest_file_sha256"]
        == audit["source"]["checkpoint_manifest"]["file_sha256"]
    )
    assert (
        terminal["checkpoint_seal"]["manifest_receipt_sha256"]
        == audit["source"]["checkpoint_manifest"]["receipt_sha256"]
    )


def test_resume_shape_and_step_math_are_exact() -> None:
    audit = sealed(AUDIT)
    execution = audit["execution_contract"]
    cadence = audit["resume_math_and_cadence"]

    assert execution["priority_request"] == "c1"
    assert execution["expected_rendered_queue"] == "q1"
    assert execution["expected_effective_priority"] == 10000
    assert (execution["workers"], execution["gpus_per_worker"]) == (2, 4)
    assert execution["world_size"] == execution["total_gpus"] == 8
    assert execution["automatic_requeue"] is False
    assert execution["one_worker_by_eight_gpu_request_qualified"] is False

    assert cadence["terminal_optimizer_step"] == (602 + 8 - 1) // 8 == 76
    assert cadence["continuation_optimizer_steps"] == 76 - 6 == 70
    assert cadence["continuation_global_step_range"] == [7, 76]
    expected = [step for step in range(12, 77, 6)] + [76]
    assert cadence["new_checkpoint_steps"] == expected
    assert cadence["expected_retained_new_checkpoint_steps"] == expected[-3:]
    assert cadence["expected_new_wandb_scalar_events"] == 70
    assert audit["scientific_contract"]["warmup_steps"] == 4
    assert audit["scientific_contract"]["scheduler_state_is_restored_not_restarted"] is True


def test_resume_rejects_a_one_by_eight_topology_change(monkeypatch) -> None:
    config = read(CONFIG)
    recipe = {**config["recipe"], "max_steps": 76, "eval_interval": 0}
    source = {
        "schema": "cyber_sft_runtime_dense_v1",
        "run_name": "source-run",
        "output_root": "/mnt/sfs/jobs/source-run",
        "model": {"identity": "exact-model"},
        "datasets": {"train": {"identity": "exact-data"}},
        "recipe": recipe,
        "wandb": {"run_id": "source-run"},
        "split_manifest_sha256": "sha256:" + "1" * 64,
        "corpus_manifest_sha256": "sha256:" + "2" * 64,
        "validation_mode": "task_outcomes_only",
        "fleet_dev_protocol_sha256": "sha256:" + "3" * 64,
        "execution": {"image": IMAGE},
    }
    manifest = {
        "source_plan": source,
        "world_size": 8,
        "optimizer_step": 6,
        "training_progress": {"supervised_tokens": 70862},
    }
    target = copy.deepcopy(source)
    target.update(
        run_name=config["name"],
        output_root=config["output_root"],
        wandb={"run_id": config["wandb"]["run_id"]},
        recovery={"mode": "resume", "checkpoint": manifest},
    )
    monkeypatch.setattr(recovery.checkpoints, "verify", lambda *_args, **_kwargs: None)

    recovery.validate(target, check_files=False)
    target["recipe"] = {**target["recipe"], "nodes": 1, "gpus_per_node": 8}
    with pytest.raises(ValueError, match="topology"):
        recovery.validate(target, check_files=False)

    target["recovery"]["mode"] = "validate"
    recovery.validate(target, check_files=False)


def test_request_boundary_and_runtime_bindings_remain_fail_closed() -> None:
    audit = sealed(AUDIT)
    config = read(CONFIG)
    runtime_files = audit["offline_prepared_artifact"]["runtime_files"]
    for relative, expected in runtime_files.items():
        observed = (
            "sha256:" + hashlib.sha256(b"").hexdigest()
            if relative == "training/__init__.py"
            else file_sha256(ROOT / relative)
        )
        assert expected == observed

    plan = {
        "run_name": config["name"],
        "output_root": config["output_root"],
        "runtime_sha256": file_sha256(ROOT / "training/sft_runtime.py").removeprefix("sha256:"),
        "recovery_runtime_sha256": file_sha256(ROOT / "training/recovery.py").removeprefix(
            "sha256:"
        ),
        "recovery": {"mode": "resume"},
        "recipe": {
            "nodes": config["recipe"]["nodes"],
            "gpus_per_node": config["recipe"]["gpus_per_node"],
        },
        "wandb": config["wandb"],
        "execution": {
            "image": audit["execution_contract"]["immutable_image"],
            "priority": config["cluster"]["priority"],
            "resources": config["cluster"]["resources"],
        },
    }
    request = job_request(plan)
    validate_request(request)
    assert request["workers"] == 2
    assert request["gpus_per_worker"] == 4
    assert request["requeueIfPreempted"] is False
    assert request["priority_class"] == "c1"
    assert request["secrets"] == ["wandb-api"]
    assert '"WANDB_RESUME": "never"' in (ROOT / "training/sft_runtime.py").read_text()
    assert not any(name.endswith(("TOKEN", "PASSWORD", "API_KEY")) for name in request["env"])


def test_acceptance_stays_closed_until_live_gates_and_release() -> None:
    audit = sealed(AUDIT)
    assert audit["classification"] == "offline_contract_qualified_live_submission_gates_required"
    assert audit["offline_prepared_artifact"]["live_or_network_calls_performed"] == 0
    assert audit["offline_prepared_artifact"]["jobs_api_posts"] == 0
    assert audit["create_once_contract"]["maximum_jobs_api_posts"] == 1
    assert audit["readiness"] == {
        "scientific_and_static_runtime_contract_ready": True,
        "direct_submission_without_live_rechecks_ready": False,
        "ready_for_one_post_login_dev_submission_after_all_live_gates": True,
        "production_submission_authorized": False,
        "one_by_eight_request_authorized": False,
    }
    assert len(audit["remaining_live_only_gates"]) == 10
    assert (
        audit["terminal_failure_and_release_contract"][
            "unexpected_failure_may_be_automatically_retried"
        ]
        is False
    )
    assert (
        audit["terminal_failure_and_release_contract"]["runtime_writes_a_gpu_release_receipt"]
        is False
    )
    assert not any(audit["privacy"].values())
