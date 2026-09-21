import hashlib
import json
from pathlib import Path

import pytest

from training import sft

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "configs" / "runs"
DATA = ROOT / "configs" / "data"
EVIDENCE = ROOT / "docs" / "evidence"

MANIFEST_SHA256 = "sha256:a8d08609991d7960cdcdab826bd07c3216bfcd3aa8f6fb33ab7a5fd29648d6f5"
TRAIN_SHA256 = "sha256:bf245db073fcd5a39ff6f90f6fc322710a3694408845bc27bfb6d9eaa6c41e0c"
SOURCE_MANIFEST_SHA256 = "sha256:d0641cc7baf082c0b1d9bd494112ad8562b9c332cb5be5a7696249d9101a96a8"


def test_teacher3k_32k_manifest_preserves_unique_targets_and_exclusions():
    manifest = json.loads((DATA / "qwen38-teacher3k-32k-v1.manifest.json").read_text())
    receipt = json.loads(
        (EVIDENCE / "qwen38-teacher3k-32k-materialization-receipt-20260920.json").read_text()
    )
    verification = json.loads(
        (EVIDENCE / "qwen38-teacher3k-32k-independent-verification-20260920.json").read_text()
    )

    assert manifest["sha256"] == MANIFEST_SHA256
    assert manifest["sha256"] == "sha256:" + sft.digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    assert manifest["validation_mode"] == "task_outcomes_only"
    assert manifest["max_length"] == 32_768
    assert manifest["context_tokens"] == 8_192
    assert manifest["files"]["train"] == {
        **manifest["files"]["train"],
        "sha256": TRAIN_SHA256,
        "rows": 14_693,
        "source_sessions": 2_886,
        "assistant_responses": 176_654,
        "supervised_tokens": 57_384_881,
    }
    assert len(manifest["files"]["train"]["task_keys"]) == 496
    assert manifest["rechunk_provenance"]["task_versions_preserved"] == 1_176
    assert (
        manifest["rechunk_provenance"]["held_out_task_families_excluded_across_all_versions"] == 25
    )
    assert not any("webexploit" in key.lower() for key in manifest["files"]["train"]["task_keys"])
    assert receipt["manifest_sha256"] == verification["manifest_sha256"] == MANIFEST_SHA256
    assert receipt["train_parquet_sha256"] == verification["train_sha256"] == TRAIN_SHA256
    assert receipt["supervised_tokens"] == verification["supervised_tokens"] == 57_384_881
    assert verification["target_identity_set_sha256"] == (
        "sha256:d510d970a57b6b62e6f05dd21203892ef3c0bd68e7c29f1a0cab3f1765371b85"
    )


def test_teacher3k_96k_manifest_preserves_the_exact_32k_target_set():
    manifest = json.loads((DATA / "qwen38-teacher3k-96k-v1.manifest.json").read_text())
    receipt = json.loads(
        (EVIDENCE / "qwen38-teacher3k-96k-materialization-receipt-20260920.json").read_text()
    )
    verification = json.loads(
        (EVIDENCE / "qwen38-teacher3k-96k-independent-verification-20260920.json").read_text()
    )

    assert manifest["sha256"] == (
        "sha256:8297f035f4c5b0446578cfa46897733273286f76e55d486c9598452e5585a479"
    )
    assert manifest["sha256"] == "sha256:" + sft.digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    assert manifest["max_length"] == 98_304
    assert manifest["context_tokens"] == 24_576
    assert manifest["validation_mode"] == "task_outcomes_only"
    assert manifest["files"]["train"] == {
        **manifest["files"]["train"],
        "sha256": "sha256:d292e4cd9c7ec0e45a63274fa7b7bdb1be18509b1a512b09eb9180e050357824",
        "rows": 6_847,
        "source_sessions": 2_886,
        "assistant_responses": 176_654,
        "supervised_tokens": 57_384_881,
    }
    assert receipt["manifest_sha256"] == verification["manifest_sha256"] == manifest["sha256"]
    assert (
        receipt["train_sha256"]
        == verification["train_sha256"]
        == (manifest["files"]["train"]["sha256"])
    )
    assert receipt["supervised_tokens"] == verification["supervised_tokens"] == 57_384_881
    assert verification["target_identity_set_sha256"] == (
        "sha256:d510d970a57b6b62e6f05dd21203892ef3c0bd68e7c29f1a0cab3f1765371b85"
    )


def test_teacher3k_64k_manifest_preserves_the_exact_32k_target_set():
    manifest = json.loads((DATA / "qwen38-teacher3k-64k-v1.manifest.json").read_text())
    receipt = json.loads(
        (EVIDENCE / "qwen38-teacher3k-64k-materialization-receipt-20260920.json").read_text()
    )
    verification = json.loads(
        (EVIDENCE / "qwen38-teacher3k-64k-independent-verification-20260920.json").read_text()
    )

    assert manifest["sha256"] == (
        "sha256:377720bd39a7dcb2f5c0dfd152fc9f955d4726f3f13c358dd57cac18bc84a2ae"
    )
    assert manifest["sha256"] == "sha256:" + sft.digest(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    assert manifest["max_length"] == 65_536
    assert manifest["context_tokens"] == 16_384
    assert manifest["validation_mode"] == "task_outcomes_only"
    assert manifest["files"]["train"] == {
        **manifest["files"]["train"],
        "sha256": "sha256:2ff865659cc0ed6a21efff3c6cf75999470a2d9bbbc86940425f45f728f73168",
        "rows": 8_953,
        "source_sessions": 2_886,
        "assistant_responses": 176_654,
        "supervised_tokens": 57_384_881,
    }
    assert receipt["manifest_sha256"] == verification["manifest_sha256"] == manifest["sha256"]
    assert (
        receipt["train_sha256"]
        == verification["train_sha256"]
        == (manifest["files"]["train"]["sha256"])
    )
    assert receipt["supervised_tokens"] == verification["supervised_tokens"] == 57_384_881
    assert verification["target_identity_set_sha256"] == (
        "sha256:d510d970a57b6b62e6f05dd21203892ef3c0bd68e7c29f1a0cab3f1765371b85"
    )


@pytest.mark.parametrize(
    ("filename", "name", "steps", "lr", "batch", "interval", "pause"),
    [
        (
            "qwen38-teacher3k-32k-canary-b8-lr3e6-v1.json",
            "chris-q38-t3k32-can-v1",
            1_837,
            3e-6,
            8,
            450,
            1,
        ),
        (
            "qwen38-teacher3k-32k-full-b8-lr3e6-v1.json",
            "chris-q38-t3k32-b8-v1",
            1_837,
            3e-6,
            8,
            450,
            None,
        ),
        (
            "qwen38-teacher3k-32k-full-b8-lr1e6-v1.json",
            "chris-q38-t3k32-lr1-v1",
            1_837,
            1e-6,
            8,
            450,
            None,
        ),
        (
            "qwen38-teacher3k-32k-full-b16-lr3e6-v1.json",
            "chris-q38-t3k32-b16-v1",
            919,
            3e-6,
            16,
            225,
            None,
        ),
    ],
)
def test_teacher3k_32k_runs_compile_to_one_node_c1_jobs(
    filename, name, steps, lr, batch, interval, pause
):
    config = json.loads((RUNS / filename).read_text())
    plan = sft.compile_sft(config, relative_to=RUNS)
    request = sft.job_request(plan)

    assert plan["run_name"] == name
    assert plan["output_root"] == f"/mnt/sfs/jobs/{name}"
    assert plan["corpus_manifest_sha256"] == MANIFEST_SHA256
    assert plan["validation_mode"] == "task_outcomes_only"
    assert plan["recipe"]["max_steps"] == steps
    assert plan["recipe"]["max_length"] == 32_768
    assert plan["recipe"]["lr"] == lr
    assert plan["recipe"]["batch_size"] == batch
    assert plan["recipe"]["epochs"] == 1
    assert plan["recipe"]["checkpoint_interval"] == interval
    assert plan["recipe"]["keep_checkpoints"] == 5
    assert plan.get("pause_after_step") == pause
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False


def test_teacher3k_32k_runs_have_unique_external_identities():
    paths = sorted(RUNS.glob("qwen38-teacher3k-32k-*-v1.json"))
    configs = [json.loads(path.read_text()) for path in paths]
    assert len(configs) == 5
    for field in ("name", "output_root"):
        values = [config[field] for config in configs]
        assert len(values) == len(set(values))
    run_ids = [config["wandb"]["run_id"] for config in configs]
    assert len(run_ids) == len(set(run_ids))


def test_teacher3k_native_resume_gate_changes_only_identity_and_lifecycle():
    source = json.loads((RUNS / "qwen38-teacher3k-32k-canary-b8-lr3e6-v1.json").read_text())
    resume = json.loads((RUNS / "qwen38-teacher3k-32k-resume-canary-b8-lr3e6-v1.json").read_text())

    for key in ("model", "data", "recipe", "cluster"):
        assert resume[key] == source[key]
    assert resume["name"] != source["name"]
    assert resume["output_root"] != source["output_root"]
    assert resume["wandb"]["run_id"] != source["wandb"]["run_id"]
    assert resume["recovery"] == {
        "manifest": "/mnt/sfs/jobs/chris-q38-t3k32-can-v1/checkpoint-step-1-seal.json",
        "sha256": "fce35242686027f1e4db7065fd1ec47142cb669930025a0fc8b287d7acc11d83",
        "mode": "resume",
    }
    assert source["pause_after_step"] == 1
    assert resume["pause_after_step"] == 2


@pytest.mark.parametrize(
    ("filename", "max_length", "context_tokens"),
    [
        ("qwen38-teacher3k-64k-rechunk-v1.request.json", 65_536, 16_384),
        ("qwen38-teacher3k-96k-rechunk-v1.request.json", 98_304, 24_576),
    ],
)
def test_teacher3k_later_context_treatments_are_sealed_and_distinct(
    filename, max_length, context_tokens
):
    request = json.loads((DATA / filename).read_text())

    assert request["sha256"] == "sha256:" + sft.digest(
        {key: value for key, value in request.items() if key != "sha256"}
    )
    assert request["source"]["manifest_sha256"] == SOURCE_MANIFEST_SHA256
    assert request["source"]["train_parquet_sha256"] == (
        "sha256:86452a28af5c77b48c1e53e13a6d8fd9d483eec350d22214cd7717127502a7b1"
    )
    assert request["minimum_supervised_tokens"] == 20_000_000
    assert request["max_length"] == max_length
    assert request["context_tokens"] == context_tokens
    assert request["destination"]["create_once"] is True
    assert request["destination"]["output_root"].endswith(
        f"/teacher3k-{max_length // 1024}k-v1/data-v1"
    )


@pytest.mark.parametrize(
    (
        "filename",
        "name",
        "manifest_sha256",
        "max_length",
        "steps",
        "checkpoint_interval",
        "keep_checkpoints",
        "pause",
    ),
    [
        (
            "qwen38-teacher3k-64k-canary-b8-lr3e6-v1.json",
            "chris-q38-t3k64-can-v1",
            "sha256:377720bd39a7dcb2f5c0dfd152fc9f955d4726f3f13c358dd57cac18bc84a2ae",
            65_536,
            1_120,
            275,
            5,
            1,
        ),
        (
            "qwen38-teacher3k-64k-full-b8-lr3e6-v1.json",
            "chris-q38-t3k64-b8-v1",
            "sha256:377720bd39a7dcb2f5c0dfd152fc9f955d4726f3f13c358dd57cac18bc84a2ae",
            65_536,
            1_120,
            275,
            5,
            None,
        ),
        (
            "qwen38-teacher3k-96k-canary-b8-lr3e6-v1.json",
            "chris-q38-t3k96-can-v1",
            "sha256:8297f035f4c5b0446578cfa46897733273286f76e55d486c9598452e5585a479",
            98_304,
            856,
            210,
            5,
            1,
        ),
        (
            "qwen38-teacher3k-96k-full-b8-lr3e6-v1.json",
            "chris-q38-t3k96-b8-v1",
            "sha256:8297f035f4c5b0446578cfa46897733273286f76e55d486c9598452e5585a479",
            98_304,
            856,
            210,
            5,
            None,
        ),
        (
            "qwen38-teacher3k-96k-full-b8-lr3e6-v3.json",
            "chris-q38-t3k96-b8-v3",
            "sha256:8297f035f4c5b0446578cfa46897733273286f76e55d486c9598452e5585a479",
            98_304,
            856,
            50,
            2,
            None,
        ),
    ],
)
def test_teacher3k_later_context_runs_compile_as_one_node_canary_first_arms(
    filename,
    name,
    manifest_sha256,
    max_length,
    steps,
    checkpoint_interval,
    keep_checkpoints,
    pause,
):
    config = json.loads((RUNS / filename).read_text())
    plan = sft.compile_sft(config, relative_to=RUNS)
    request = sft.job_request(plan)

    assert plan["run_name"] == name
    assert plan["corpus_manifest_sha256"] == manifest_sha256
    assert plan["validation_mode"] == "task_outcomes_only"
    assert plan["recipe"] == {
        **plan["recipe"],
        "max_length": max_length,
        "max_steps": steps,
        "checkpoint_interval": checkpoint_interval,
        "keep_checkpoints": keep_checkpoints,
        "batch_size": 8,
        "lr": 3e-6,
    }
    assert plan.get("pause_after_step") == pause
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False


@pytest.mark.parametrize("context", ["64", "96"])
def test_teacher3k_later_context_full_arms_change_only_identity_and_lifecycle(context):
    canary = json.loads((RUNS / f"qwen38-teacher3k-{context}k-canary-b8-lr3e6-v1.json").read_text())
    full = json.loads((RUNS / f"qwen38-teacher3k-{context}k-full-b8-lr3e6-v1.json").read_text())

    expected = json.loads(json.dumps(canary))
    expected["name"] = full["name"]
    expected["output_root"] = full["output_root"]
    expected.pop("pause_after_step")
    expected["wandb"]["run_id"] = full["wandb"]["run_id"]
    expected["wandb"]["name"] = full["wandb"]["name"]
    expected["wandb"]["tags"] = full["wandb"]["tags"]
    assert full == expected
    assert "one-step-canary" in canary["wandb"]["tags"]
    assert "one-epoch" in full["wandb"]["tags"]

    canary_plan = sft.compile_sft(canary, relative_to=RUNS)
    full_plan = sft.compile_sft(full, relative_to=RUNS)
    expected_plan = json.loads(json.dumps(canary_plan))
    expected_plan["run_name"] = full_plan["run_name"]
    expected_plan["output_root"] = full_plan["output_root"]
    expected_plan.pop("pause_after_step")
    expected_plan["wandb"]["run_id"] = full_plan["wandb"]["run_id"]
    expected_plan["wandb"]["name"] = full_plan["wandb"]["name"]
    expected_plan["wandb"]["tags"] = full_plan["wandb"]["tags"]
    assert full_plan == expected_plan


@pytest.mark.parametrize("context", ["64", "96"])
def test_teacher3k_context_runtime_repair_changes_only_external_identity(context):
    retired = json.loads((RUNS / f"qwen38-teacher3k-{context}k-full-b8-lr3e6-v1.json").read_text())
    successor = json.loads(
        (RUNS / f"qwen38-teacher3k-{context}k-full-b8-lr3e6-v2.json").read_text()
    )

    expected = json.loads(json.dumps(retired))
    expected["name"] = successor["name"]
    expected["output_root"] = successor["output_root"]
    expected["wandb"]["run_id"] = successor["wandb"]["run_id"]
    expected["wandb"]["name"] = successor["wandb"]["name"]
    expected["wandb"]["tags"].append("runtime-dataset-contract-repair")
    assert successor == expected

    retired_plan = sft.compile_sft(retired, relative_to=RUNS)
    successor_plan = sft.compile_sft(successor, relative_to=RUNS)
    expected_plan = json.loads(json.dumps(retired_plan))
    expected_plan["run_name"] = successor_plan["run_name"]
    expected_plan["output_root"] = successor_plan["output_root"]
    expected_plan["wandb"]["run_id"] = successor_plan["wandb"]["run_id"]
    expected_plan["wandb"]["name"] = successor_plan["wandb"]["name"]
    expected_plan["wandb"]["tags"].append("runtime-dataset-contract-repair")
    assert successor_plan == expected_plan


def test_teacher3k_64k_forward_adapter_repair_changes_only_external_identity():
    retired = json.loads((RUNS / "qwen38-teacher3k-64k-full-b8-lr3e6-v2.json").read_text())
    successor = json.loads((RUNS / "qwen38-teacher3k-64k-full-b8-lr3e6-v3.json").read_text())

    expected = json.loads(json.dumps(retired))
    expected["name"] = successor["name"]
    expected["output_root"] = successor["output_root"]
    expected["wandb"]["run_id"] = successor["wandb"]["run_id"]
    expected["wandb"]["name"] = successor["wandb"]["name"]
    expected["wandb"]["tags"][-1] = "runtime-forward-adapter-repair"
    assert successor == expected


def test_teacher3k_64k_v3_qualification_is_historical_after_runtime_repair():
    evidence = json.loads(
        (EVIDENCE / "qwen38-teacher3k-64k-v3-qualified-ready-20260921.json").read_text()
    )
    incident = json.loads(
        (EVIDENCE / "qwen38-broad-sft-eight-hour-runtime-bound-20260921.json").read_text()
    )
    config = json.loads((ROOT / evidence["source"]["config"]).read_text())
    plan = sft.compile_sft(config, relative_to=RUNS)
    request = sft.job_request(plan)

    assert evidence["status"] == "qualified_not_submitted"
    assert evidence["source"]["head"] == "aecb8d9deab66014eccaeeda0a86cbc9c1a98fac"
    bindings = evidence["immutable_bindings"]
    invalidated = incident["prepared_packets_invalidated_by_runtime_change"][0]
    assert invalidated == {
        "config": evidence["source"]["config"],
        "runtime_sha256": bindings["runtime_sha256"],
        "plan_sha256": bindings["plan_sha256"],
        "request_sha256": bindings["request_sha256"],
        "status": "historical_packet_was_submitted_once_and_is_not_reusable",
    }
    assert sft.digest(plan) != bindings["plan_sha256"]
    assert sft.digest(request) != bindings["request_sha256"]
    assert plan["runtime_sha256"] != bindings["runtime_sha256"]
    assert plan["corpus_manifest_sha256"] == bindings["corpus_manifest_sha256"]
    assert plan["split_manifest_sha256"] == bindings["split_manifest_sha256"]
    assert plan["datasets"]["train"]["sha256"] == bindings["train_parquet_sha256"]

    reuse = evidence["qualification_reuse_proof"]
    assert reuse["previous_source_head"] == "40077bb2902268519e2e07ac086737130b6df077"
    assert reuse["current_source_head"] == evidence["source"]["head"]
    assert reuse["changed_paths_affecting_sft_runtime_or_config"] == []
    assert reuse["regenerated_packet_matches_qualified_packet"] == {
        **evidence["prepared_packet"],
        "runtime_sha256": bindings["runtime_sha256"],
        "plan_sha256": bindings["plan_sha256"],
        "request_sha256": bindings["request_sha256"],
    }
    assert reuse["live_preview_reuse_proof"] == {
        "preview_source_head": "ddba5eacc6a460c7c0b98f085e5a50dfcee2bea3",
        "current_source_head": evidence["source"]["head"],
        "cyber_post_train/direct_submit.py": "f1247dfbab718a9819d21c6880c42928627d2e72",
    }

    treatment = evidence["treatment"]
    assert plan["recipe"] == {
        "epochs": treatment["epochs"],
        "batch_size": treatment["global_batch"],
        "microbatch_per_gpu": treatment["microbatch_per_gpu"],
        "lr": treatment["learning_rate"],
        "max_length": treatment["max_length"],
        "max_steps": treatment["planned_optimizer_steps"],
        "checkpoint_interval": treatment["checkpoint_interval"],
        "keep_checkpoints": treatment["keep_checkpoints"],
        "eval_interval": 0,
        "seed": treatment["seed"],
        "nodes": treatment["nodes"],
        "gpus_per_node": treatment["gpus_per_node"],
    }
    assert plan["datasets"]["train"]["rows"] == treatment["train_rows"]
    assert plan["datasets"]["train"]["supervised_tokens"] == treatment["unique_supervised_tokens"]
    assert plan["validation_mode"] == "task_outcomes_only"
    assert plan["wandb"]["entity"] == evidence["wandb"]["entity"]
    assert plan["wandb"]["project"] == evidence["wandb"]["project"]
    assert plan["wandb"]["group"] == evidence["wandb"]["group"]
    assert plan["wandb"]["run_id"] == evidence["wandb"]["run_id"]
    assert plan["wandb"]["name"] == evidence["wandb"]["name"]
    assert plan["wandb"]["tags"] == evidence["wandb"]["tags"]

    cpu = evidence["exact_image_cpu_qualification"]
    assert cpu["status"] == "passed_and_released"
    assert cpu["gpus"] == 0
    assert cpu["priority"] == "c1"
    assert cpu["root_failure_alert_annotation"] == "off"
    assert cpu["restarts"] == 0
    assert cpu["pod_absent_after_release"] is True
    assert "native_forward_backward_signature" in cpu["checks"]
    assert "native_train_only_loader" in cpu["checks"]

    preview = evidence["live_preview"]
    assert preview["status"] == "accepted_without_create"
    assert preview["root_failure_alert_annotation"] == "off"
    assert preview["pod_template_priority"] == "c1"
    assert preview["secret_names"] == ["wandb-api"]
    assert preview["api_duplicate_absent"] is True
    assert preview["kubernetes_duplicate_absent"] is True
    assert preview["output_absent"] is True
    assert preview["server_dry_run_passed"] is True
    assert preview["created"] is False
    assert request["failureAlerts"] is False
    assert request["priority_class"] == "c1"
    assert request["secrets"] == ["wandb-api"]
    assert evidence["submission"] == {
        "gpu_post_performed": False,
        "job_or_rayjob_created": False,
        "training_status": "not_submitted",
        "96k_arm_submitted": False,
    }


@pytest.mark.parametrize(("context", "position"), [("64", 0), ("96", 1)])
def test_teacher3k_later_context_launch_binds_current_inputs(context, position):
    evidence = json.loads(
        (EVIDENCE / "qwen38-next-sft-intentional-launch-20260920.json").read_text()
    )
    incident = json.loads(
        (EVIDENCE / "qwen38-broad-sft-eight-hour-runtime-bound-20260921.json").read_text()
    )
    row = evidence["arms"][position]
    config_path = ROOT / row["config"]
    config = json.loads(config_path.read_text())
    plan = sft.compile_sft(config, relative_to=RUNS)
    request = sft.job_request(plan)

    assert evidence["status"] == "v2_64k_failed_v3_64k_qualified_not_submitted_96k_not_submitted"
    invalidated = incident["prepared_packets_invalidated_by_runtime_change"][position]
    assert invalidated["config"] == row["config"]
    assert invalidated["runtime_sha256"] == row["runtime_sha256"]
    assert invalidated["plan_sha256"] == row["plan_sha256"]
    assert invalidated["request_sha256"] == row["request_sha256"]
    assert sft.digest(plan) != row["plan_sha256"]
    assert sft.digest(request) != row["request_sha256"]
    assert plan["runtime_sha256"] != row["runtime_sha256"]
    assert plan["recipe"]["max_length"] == int(context) * 1024
    assert plan["recipe"]["max_steps"] == row["planned_optimizer_steps"]
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["failureAlerts"] is False
    if context == "64":
        assert invalidated["status"] == "historical_packet_was_submitted_once_and_is_not_reusable"
        assert row["successor_status"].endswith("not_submitted")
        qualification = evidence["64k_successor_qualification"]
        assert qualification["source_head"] == "aecb8d9deab66014eccaeeda0a86cbc9c1a98fac"
        assert qualification["status"].endswith("not_submitted")
        assert qualification["gpu_post_performed"] is False
        assert qualification["root_failure_alert_annotation"] == "off"
        assert qualification["priority"] == "c1"
        retired = row["retired_runtime_contract_attempt"]
        assert retired["run_id"] == "2f111132-a6d5-407c-aa3f-6ceed1033b8e"
        assert retired["root_failure_alert_annotation"] == "off"
        assert retired["priority"] == "c1"
        assert retired["restarts"] == 0
        terminal = retired["terminal"]
        assert terminal["reported_loop_step"] == 1
        assert terminal["proven_optimizer_updates"] == 0
        assert terminal["metrics_receipt_present"] is False
        assert terminal["checkpoint_receipt_present"] is False
        assert terminal["raycluster_absent"] is True
        assert terminal["pod_absent"] is True
        assert terminal["active_gpus"] == 0
    else:
        assert (
            invalidated["status"]
            == "historical_packet_not_submitted_and_requires_fresh_qualification"
        )
        assert row["successor_status"].endswith("not_submitted")
        assert "retired_runtime_contract_attempt" not in row
    assert row["retired_attempt"]["optimizer_step"] == 0
    assert row["retired_attempt"]["root_failure_alert_annotation"] == "off"
    assert row["retired_attempt"]["resource_release"].endswith("zero_gpu_held")
    assert row["motivation"]

    gate_path = EVIDENCE / f"qwen38-teacher3k-{context}k-canary-accepted-20260920.json"
    gate_evidence = json.loads(gate_path.read_text())
    assert gate_evidence["status"] == "accepted_one_step_and_released"
    assert gate_evidence["treatment"]["context_length"] == int(context) * 1024
    assert gate_evidence["scientific_result"]["optimizer_step"] == 1
    assert gate_evidence["scientific_result"]["finite_metrics"] is True
    assert gate_evidence["resource_release"]["active_gpus"] == 0


def test_teacher3k_fullweight_launch_receipt_is_bound_and_nonterminal():
    receipt = json.loads(
        (EVIDENCE / "qwen38-teacher3k-fullweight-sweep-launch-20260920.json").read_text()
    )
    claimed = receipt.pop("receipt_sha256")
    observed = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()

    assert claimed == observed
    assert receipt["scientific_disposition"] == "nonterminal_training_evidence_no_capability_claim"
    assert receipt["interpretation"] == {
        **receipt["interpretation"],
        "finite_update_gate_passed": True,
        "checkpoint_gate_passed": False,
        "capability_claim": False,
    }
    assert [run["arm"] for run in receipt["runs"]] == [
        "batch8_lr3e-6_reference",
        "batch16_lr3e-6_batch_control",
        "batch8_lr1e-6_learning_rate_control",
    ]
    assert all(run["submission"]["submitted_once"] for run in receipt["runs"])
    assert all(
        run["live_evidence"]["optimizer_step_at_observation"] >= 1 for run in receipt["runs"]
    )
    assert all(run["live_evidence"]["restarts"] == 0 for run in receipt["runs"])
