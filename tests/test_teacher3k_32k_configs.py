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
    assert len(configs) == 4
    for field in ("name", "output_root"):
        values = [config[field] for config in configs]
        assert len(values) == len(set(values))
    run_ids = [config["wandb"]["run_id"] for config in configs]
    assert len(run_ids) == len(set(run_ids))
