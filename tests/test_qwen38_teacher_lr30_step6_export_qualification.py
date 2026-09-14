from __future__ import annotations

import copy
import hashlib
import json
import shlex
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest, validate_request
from training import miles_serving_dev, serving_registration
from training.io import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/qualification/qwen38-teacher-lr30-step6-hf-export-dev-v1.template.json"
DEV_PATH = ROOT / "configs/qualification/qwen38-teacher-lr30-step6-serving-dev-v1.template.json"
PROD_PATH = ROOT / "configs/qualification/qwen38-teacher-lr30-step6-serving-prod-v1.template.json"
SOURCE_EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-14-lr30-2x4-reload-terminal-v1.json"
PREPARATION_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-study/2026-09-14-lr30-step6-export-preparation-v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _null_paths(prefix: str, value: object) -> set[str]:
    if value is None:
        return {prefix}
    if not isinstance(value, dict):
        return set()
    result: set[str] = set()
    for key, item in value.items():
        if key != "unresolved_bindings":
            result |= _null_paths(f"{prefix}.{key}", item)
    return result


def test_plan_binds_exact_accepted_lr30_step6_source_and_current_export_code() -> None:
    plan = _load(PLAN_PATH)
    source = plan["source"]
    evidence = _load(SOURCE_EVIDENCE)
    unsigned = {key: value for key, value in plan.items() if key != "sha256"}

    assert plan["schema"] == "cyber_qwen38_sft_export_qualification_plan_v1"
    assert plan["sha256"] == "sha256:" + digest(unsigned)
    assert source["accepted_reload_evidence"]["file_sha256"] == file_sha256(SOURCE_EVIDENCE)
    assert source["accepted_reload_evidence"]["receipt_sha256"] == evidence["sha256"]
    assert source["accepted_reload_evidence"]["classification"] == evidence["classification"]
    assert source["checkpoint"]["path"] == evidence["source"]["checkpoint_path"]
    assert source["checkpoint"]["optimizer_step"] == evidence["source"]["optimizer_step"] == 6
    assert source["checkpoint"]["world_size"] == evidence["source"]["world_size"] == 8
    assert source["checkpoint"]["files"] == evidence["source"]["files"] == 33
    assert source["checkpoint"]["bytes"] == evidence["source"]["bytes"] == 324627486731
    assert source["checkpoint"]["manifest_path"] == evidence["checkpoint_seal"]["manifest_path"]
    assert (
        source["checkpoint"]["manifest_file_sha256"]
        == evidence["checkpoint_seal"]["manifest_file_sha256"]
    )
    assert (
        source["checkpoint"]["manifest_receipt_sha256"]
        == evidence["checkpoint_seal"]["manifest_receipt_sha256"]
    )
    assert source["checkpoint"]["native_all_rank_reload_accepted"] is True
    assert evidence["acceptance"]["zero_update_reload_accepted"] is True
    assert evidence["acceptance"]["hf_export_accepted"] is False

    for name, expected in plan["cpu_export"]["code_sha256"].items():
        assert file_sha256(ROOT / name) == expected
    assert (
        file_sha256(ROOT / plan["cpu_check"]["checker_path"])
        == plan["cpu_check"]["checker_file_sha256"]
    )
    assert file_sha256(ROOT / source["model"]["lock_path"]) == source["model"]["lock_file_sha256"]
    assert (
        file_sha256(ROOT / source["model"]["weights_path"])
        == source["model"]["weights_file_sha256"]
    )
    assert (
        file_sha256(ROOT / source["training_config"]["path"])
        == source["training_config"]["file_sha256"]
    )


def test_export_and_checks_are_create_once_zero_update_and_dev_only() -> None:
    plan = _load(PLAN_PATH)
    export = plan["cpu_export"]
    cpu = plan["cpu_check"]
    gpu = plan["gpu_check"]
    source = Path(plan["source"]["checkpoint"]["path"])
    base = Path(plan["source"]["model"]["root"])
    destination = Path(export["destination"])

    assert plan["status"] == "prepared_not_executed" and plan["launchable"] is False
    assert plan["cluster_or_api_mutations_performed"] == 0
    assert export["resources"]["gpus"] == cpu["resources"]["gpus"] == 0
    assert export["atomic_no_replace"] and export["destination_and_partial_must_be_absent"]
    assert Path(export["partial_destination"]) == destination.with_name(
        destination.name + ".partial"
    )
    assert not destination.is_relative_to(source) and not destination.is_relative_to(base)
    assert export["argv"] == [
        "cyber-post-train",
        "checkpoint-export",
        plan["source"]["checkpoint"]["manifest_path"],
        "--sha256",
        plan["source"]["checkpoint"]["manifest_file_sha256"].removeprefix("sha256:"),
        "--output",
        export["destination"],
    ]
    assert export["expected_optimizer_step"] == 6
    assert export["expected_optimizer_steps_executed"] == 0
    assert export["expected_dtype"] == "BF16"
    assert export["expected_tensor_count"] == 1199
    assert export["expected_tensor_bytes"] == 55562855904

    for stage in (cpu, gpu):
        argv = (
            stage["argv_template"]
            if "argv_template" in stage
            else stage["request"]["argv_template"]
        )
        assert argv.count("{export_receipt_file_sha256}") == 1
        assert "--gpu" in argv if stage is gpu else "--gpu" not in argv
        assert stage["must_report_optimizer_steps_executed"] == 0
        assert stage["must_report_source_unchanged"] is True
    assert gpu["cluster"] == "dev"
    assert gpu["jobs_api_base"] == "https://api.ft.dev.flt.build"
    assert gpu["request"]["priority_class"] == "c1"
    assert gpu["request"]["expected_derived_queue_priority_class"] == "q1"
    assert gpu["request"]["expected_effective_priority"] == 10000
    assert gpu["request"]["workers"] == gpu["request"]["gpus_per_worker"] == 1
    assert gpu["request"]["requeueIfPreempted"] is False
    assert gpu["terminal_and_release_evidence_required"] is True

    request = copy.deepcopy(gpu["request"])
    argv = [
        "a" * 64 if value == "{export_receipt_file_sha256}" else value
        for value in request.pop("argv_template")
    ]
    request["command"] = shlex.join(argv)
    request.pop("expected_derived_queue_priority_class")
    request.pop("expected_effective_priority")
    validate_request(request)


def test_every_live_binding_is_explicitly_unresolved_and_templates_fail_closed(
    tmp_path: Path,
) -> None:
    plan, dev, prod = _load(PLAN_PATH), _load(DEV_PATH), _load(PROD_PATH)
    actual = (
        _null_paths("plan", plan)
        | _null_paths("dev_template", dev)
        | _null_paths("prod_template", prod)
    )
    assert actual == set(plan["unresolved_bindings"])
    assert file_sha256(DEV_PATH) == plan["dev_serving"]["template_file_sha256"]
    assert file_sha256(PROD_PATH) == plan["persistent_serving"]["template_file_sha256"]
    assert dev["schema"] == miles_serving_dev.SFT_CONFIG_SCHEMA
    assert dev["cluster"] == {"target": "dev", "priority": "c1"}
    assert prod["schema"] == serving_registration.SFT_SCHEMA
    assert (
        dev["export"]["path"]
        == prod["export"]["path"]
        == plan["cpu_export"]["expected_receipt_path"]
    )
    assert (
        dev["gpu_check"]["path"]
        == prod["gpu_check"]["path"]
        == plan["gpu_check"]["expected_receipt_path"]
    )
    assert prod["model_id"] == plan["persistent_serving"]["model_id"]

    with pytest.raises((TypeError, ValueError)):
        miles_serving_dev.compile_plan(dev)
    output = tmp_path / "must-not-exist"
    with pytest.raises((TypeError, ValueError)):
        serving_registration.prepare(prod, output)
    assert not output.exists()


def test_wbe_and_capability_gates_stay_closed_until_live_serving_parity() -> None:
    plan = _load(PLAN_PATH)
    accepted = plan["acceptance"]
    assert accepted == {
        "native_checkpoint_reload_accepted": True,
        "bf16_export_accepted": False,
        "cpu_export_check_accepted": False,
        "one_gpu_export_reload_accepted": False,
        "dev_sglang_serving_accepted": False,
        "inference_staging_accepted": False,
        "persistent_registration_accepted": False,
        "matched_live_parity_accepted": False,
        "webexploitbench_eligible": False,
        "model_quality_claimed": False,
    }
    assert plan["dev_serving"]["non_scored"] is True
    assert plan["dev_serving"]["benchmark_attempts"] == 0
    assert plan["execution"] == {
        "prepared_at": "2026-09-14T10:50:00Z",
        "prepared_offline": True,
        "kubernetes_calls": 0,
        "jobs_api_calls": 0,
        "inference_api_calls": 0,
        "wandb_calls": 0,
        "tensorlake_calls": 0,
        "gpu_allocations": 0,
    }


def test_template_hashes_are_plain_file_sha256_not_self_digests() -> None:
    plan = _load(PLAN_PATH)
    for path, recorded in (
        (DEV_PATH, plan["dev_serving"]["template_file_sha256"]),
        (PROD_PATH, plan["persistent_serving"]["template_file_sha256"]),
    ):
        assert recorded == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_preparation_evidence_is_self_digesting_and_claims_no_live_execution() -> None:
    evidence = _load(PREPARATION_EVIDENCE)
    unsigned = {key: value for key, value in evidence.items() if key != "sha256"}
    assert evidence["schema"] == "cyber_qwen38_lr30_step6_export_preparation_v1"
    assert evidence["status"] == "prepared_offline_not_launched"
    assert evidence["sha256"] == "sha256:" + digest(unsigned)
    artifacts = evidence["prepared_artifacts"]
    assert artifacts["export_qualification"]["file_sha256"] == file_sha256(PLAN_PATH)
    assert artifacts["dev_serving"]["file_sha256"] == file_sha256(DEV_PATH)
    assert artifacts["persistent_serving"]["file_sha256"] == file_sha256(PROD_PATH)
    assert all(value == 0 for value in evidence["execution"].values())
    assert evidence["acceptance"]["webexploitbench_eligible"] is False
    assert evidence["acceptance"]["capability_claimed"] is False
