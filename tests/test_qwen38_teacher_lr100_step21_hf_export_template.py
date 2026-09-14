"""The LR100 step-21 HF export remains inert behind checkpoint reload gates."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/qualification/qwen38-teacher-lr100-step21-hf-export-dev-v1.template.json"
SOURCE = ROOT / "configs/qualification/qwen38-teacher-lr100-cosine-layout-dev-v2.json"
HANDOFF = ROOT / "docs/evidence/qwen38-study/2026-09-14-lr100-training-seal-handoff-v1.json"
PAIR = ROOT / "configs/qualification/qwen38-teacher-lr100-step20-step21-seal-verify-dev-v2.pod.yaml"
RELOAD = (
    ROOT
    / "configs/qualification/qwen38-teacher-lr100-cosine-1x8-step21-reload-dev-v1.template.json"
)
PROVEN = ROOT / "docs/evidence/qwen38-study/2026-09-13-sft-export-gpu-acceptance-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-14-lr100-step21-export-preparation-v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sealed(path: Path) -> dict:
    value = read(path)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def nested(value: dict, path: str):
    current = value
    for part in path.split("."):
        current = current[part]
    return current


def null_paths(value, prefix: str = "") -> set[str]:
    paths = set()
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            paths.update(null_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.update(null_paths(item, f"{prefix}[{index}]"))
    elif value is None:
        paths.add(prefix)
    return paths


def test_template_is_self_digesting_inert_and_enumerates_every_null() -> None:
    value = sealed(TEMPLATE)
    assert value["status"] == "blocked_waiting_checkpoint_pair_and_native_reload_acceptance"
    assert value["launchable"] is False
    assert value["cluster_or_api_mutations_performed"] is False
    assert null_paths(value) == set(value["unresolved_bindings"])
    assert len(value["unresolved_bindings"]) == len(set(value["unresolved_bindings"]))
    assert all(nested(value, path) is None for path in value["unresolved_bindings"])
    assert value["execution"] == {
        "kind": "offline_preparation_only_not_a_job_request",
        "kubernetes_calls_performed": 0,
        "jobs_api_preview_calls_performed": 0,
        "jobs_api_submit_calls_performed": 0,
        "wandb_calls_performed": 0,
        "tensorlake_calls_performed": 0,
    }


def test_source_is_exact_step21_but_not_yet_accepted_for_export() -> None:
    value, source, handoff = sealed(TEMPLATE), read(SOURCE), sealed(HANDOFF)
    bound = value["source"]
    assert bound["training"]["config_file_sha256"] == file_sha256(SOURCE)
    assert bound["training"]["output_root"] == source["output_root"]
    assert bound["training"]["handoff_file_sha256"] == file_sha256(HANDOFF)
    assert bound["training"]["handoff_receipt_sha256"] == handoff["sha256"]
    assert bound["training"]["optimizer_step"] == 21
    assert bound["training"]["resource_release_verified"] is True
    checkpoint = bound["checkpoint"]
    assert checkpoint["path"] == f"{source['output_root']}/checkpoints/global_step_21"
    assert checkpoint["files"] == 33
    assert checkpoint["bytes"] == 324_627_486_731
    assert checkpoint["seal_file_sha256"] is None
    assert checkpoint["pair_verification_accepted"] is False
    assert checkpoint["native_zero_update_reload_accepted"] is False
    assert checkpoint["accepted_for_export"] is False


def test_prerequisites_and_proven_qwen38_mechanism_are_digest_bound() -> None:
    value = sealed(TEMPLATE)
    source = value["source"]
    assert source["pair_verification"]["artifact_file_sha256"] == file_sha256(PAIR)
    assert source["native_reload"]["template_file_sha256"] == file_sha256(RELOAD)
    assert source["native_reload"]["template_receipt_sha256"] == sealed(RELOAD)["sha256"]
    proven = value["proven_mechanism"]
    assert proven["evidence_file_sha256"] == file_sha256(PROVEN)
    assert proven["evidence_receipt_sha256"] == sealed(PROVEN)["sha256"]
    assert proven["same_exact_model_revision"] is True
    assert proven["previous_qwen38_checkpoints_accepted"] == [44, 186]
    assert proven["scope"] == "mechanism_reuse_only_not_lr100_acceptance"
    assert value["gate_order"][:2] == [
        "accept_one_terminal_read_only_step20_step21_pair_verification_proof",
        "accept_exact_step21_one_by_eight_native_reload_with_zero_optimizer_updates_and_full_release",
    ]


def test_cpu_export_and_check_are_zero_gpu_create_once_dev_stages() -> None:
    value = sealed(TEMPLATE)
    export, check = value["cpu_export"], value["cpu_check"]
    source_checkpoint = Path(value["source"]["checkpoint"]["path"])
    model_root = Path(value["source"]["model"]["root"])
    output = Path(export["output_root"])
    assert export["cluster"] == check["cluster"] == "dev"
    assert export["priority_request"] == check["priority_request"] == "c1"
    assert export["resources"]["gpus"] == check["resources"]["gpus"] == 0
    assert (export["resources"]["memory_request"], export["resources"]["memory_limit"]) == (
        "32Gi",
        "48Gi",
    )
    assert not output.is_relative_to(source_checkpoint)
    assert not output.is_relative_to(model_root)
    assert Path(export["partial_root"]) == output.with_name(output.name + ".partial")
    assert not Path(check["output_path"]).is_relative_to(output)
    assert export["command"]["verb"] == "checkpoint-export"
    assert export["source_manifest_file_sha256"] is None
    assert export["output_absent_verified"] is None
    assert check["export_receipt_file_sha256"] is None
    assert export["accepted"] is check["accepted"] is False
    for path, expected in export["code_sha256"].items():
        assert expected == file_sha256(ROOT / path)
    assert check["checker_file_sha256"] == file_sha256(ROOT / "training/export_check.py")


def test_one_gpu_check_is_c1_q1_dev_only_and_all_later_gates_stay_closed() -> None:
    value = sealed(TEMPLATE)
    check = value["gpu_check"]
    assert check["cluster"] == "dev"
    assert check["execution_rail"] == "fleet_jobs_api"
    assert (check["priority_class"], check["expected_queue_priority"]) == ("c1", "q1")
    assert check["expected_priority_value"] == 10_000
    assert (check["workers"], check["gpus_per_worker"], check["total_gpus"]) == (1, 1, 1)
    assert check["requeue_if_preempted"] is False
    assert check["optimizer_steps_authorized"] == 0
    assert check["synthetic_only"] is True
    assert check["generated_token_text_recorded"] is False
    assert check["api_post_calls_performed"] == 0
    assert check["accepted"] is False
    assert set(value["acceptance"].values()) == {False}
    assert "webexploitbench" not in TEMPLATE.read_text(encoding="utf-8").lower()


def test_evidence_is_self_digesting_and_does_not_upgrade_checkpoint_status() -> None:
    value, evidence = sealed(TEMPLATE), sealed(EVIDENCE)
    assert evidence["prepared_plan"] == {
        "path": str(TEMPLATE.relative_to(ROOT)),
        "file_sha256": file_sha256(TEMPLATE),
        "receipt_sha256": value["sha256"],
        "launchable": False,
        "cluster": "dev",
        "priority_ceiling": "c1/q1",
        "cpu_export_gpus": 0,
        "cpu_check_gpus": 0,
        "gpu_check_gpus": 1,
        "production_authorized": False,
    }
    assert evidence["checkpoint_audit"]["accepted_for_export"] is False
    assert set(evidence["closed_gates"].values()) == {False}
    assert evidence["execution"]["gpu_nodes_allocated"] == 0
