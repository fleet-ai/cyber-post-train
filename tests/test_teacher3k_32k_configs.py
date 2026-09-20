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
        (
            EVIDENCE
            / "qwen38-teacher3k-96k-independent-verification-20260920.json"
        ).read_text()
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
    assert receipt["train_sha256"] == verification["train_sha256"] == (
        manifest["files"]["train"]["sha256"]
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
        (
            EVIDENCE
            / "qwen38-teacher3k-64k-independent-verification-20260920.json"
        ).read_text()
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
    assert receipt["train_sha256"] == verification["train_sha256"] == (
        manifest["files"]["train"]["sha256"]
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


def test_teacher3k_native_resume_gate_changes_only_identity_and_lifecycle():
    source = json.loads(
        (RUNS / "qwen38-teacher3k-32k-canary-b8-lr3e6-v1.json").read_text()
    )
    resume = json.loads(
        (
            RUNS / "qwen38-teacher3k-32k-resume-canary-b8-lr3e6-v1.json"
        ).read_text()
    )

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
            1,
        ),
        (
            "qwen38-teacher3k-64k-full-b8-lr3e6-v1.json",
            "chris-q38-t3k64-b8-v1",
            "sha256:377720bd39a7dcb2f5c0dfd152fc9f955d4726f3f13c358dd57cac18bc84a2ae",
            65_536,
            1_120,
            275,
            None,
        ),
        (
            "qwen38-teacher3k-96k-canary-b8-lr3e6-v1.json",
            "chris-q38-t3k96-can-v1",
            "sha256:8297f035f4c5b0446578cfa46897733273286f76e55d486c9598452e5585a479",
            98_304,
            856,
            210,
            1,
        ),
        (
            "qwen38-teacher3k-96k-full-b8-lr3e6-v1.json",
            "chris-q38-t3k96-b8-v1",
            "sha256:8297f035f4c5b0446578cfa46897733273286f76e55d486c9598452e5585a479",
            98_304,
            856,
            210,
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
        "keep_checkpoints": 5,
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
    canary = json.loads(
        (RUNS / f"qwen38-teacher3k-{context}k-canary-b8-lr3e6-v1.json").read_text()
    )
    full = json.loads(
        (RUNS / f"qwen38-teacher3k-{context}k-full-b8-lr3e6-v1.json").read_text()
    )

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


@pytest.mark.parametrize(("context", "position"), [("64", 0), ("96", 1)])
def test_teacher3k_later_context_ready_queue_binds_current_inputs(context, position):
    evidence = json.loads(
        (
            EVIDENCE
            / "qwen38-teacher3k-long-context-full-ready-queue-20260920.json"
        ).read_text()
    )
    row = evidence["full_run_queue"][position]
    config_path = ROOT / row["config"]
    config = json.loads(config_path.read_text())
    plan = sft.compile_sft(config, relative_to=RUNS)
    request = sft.job_request(plan)

    assert evidence["status"] == "scientifically_qualified_queue_held_no_submission"
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == row["config_file_sha256"]
    assert sft.digest(plan) == row["plan_sha256"]
    assert sft.digest(request) == row["request_sha256"]
    assert plan["recipe"]["max_length"] == int(context) * 1024
    assert plan["recipe"]["max_steps"] == row["planned_optimizer_steps"]
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False

    gate = evidence["accepted_canary_gates"][position]
    gate_path = ROOT / gate["evidence"]
    gate_evidence = json.loads(gate_path.read_text())
    assert hashlib.sha256(gate_path.read_bytes()).hexdigest() == gate["evidence_file_sha256"]
    assert gate_evidence["status"] == "accepted_one_step_and_released"
    assert gate_evidence["treatment"]["context_length"] == int(context) * 1024
    assert gate_evidence["scientific_result"]["optimizer_step"] == 1
    assert gate_evidence["scientific_result"]["finite_metrics"] is True
    assert gate_evidence["resource_release"]["active_gpus"] == 0
