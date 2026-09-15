import json
import math
from pathlib import Path

from training.io import digest_json, file_sha256
from training.sft_runtime import OUTCOME_WANDB_HISTORY_KEYS

ROOT = Path(__file__).parents[1]
TEMPLATE = (
    ROOT
    / "configs/qualification/qwen38-teacher-exposure-matched-sampling-pair-dev-v1.template.json"
)
EVIDENCE = (
    ROOT / "docs/evidence/qwen38-study/"
    "2026-09-14-teacher-exposure-matched-sampling-pair-preparation-v1.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_pair_is_self_digested_inert_and_blocked_on_lr_selection():
    value = load(TEMPLATE)
    assert value["sha256"] == digest_json(
        {key: row for key, row in value.items() if key != "sha256"}
    )
    assert value["launchable"] is False
    assert value["execution"] == {
        "kind": "offline_template_only_not_a_job_request",
        "kubernetes_calls_performed": 0,
        "jobs_api_preview_calls_performed": 0,
        "jobs_api_submit_calls_performed": 0,
        "wandb_runs_created": 0,
    }
    selection = value["prerequisites"]["learning_rate_selection"]
    assert selection["state"] == "blocking"
    assert selection["selection_decision_sha256"] is None
    assert selection["selected_learning_rate"] is None
    assert value["frozen_common_contract"]["recipe"]["learning_rate"] is None


def test_pair_binds_current_files_and_the_frozen_exposure_lock():
    value = load(TEMPLATE)
    common = value["frozen_common_contract"]
    for binding in (
        value["prerequisites"]["learning_rate_selection"],
        value["prerequisites"]["exposure_control"],
        common["model"],
        common["split"],
        common["runtime"],
        common["fleet_development_evaluation"],
    ):
        for key, expected in binding.items():
            if key.endswith("_path") and key.replace("_path", "_file_sha256") in binding:
                assert file_sha256(ROOT / expected) == binding[key.replace("_path", "_file_sha256")]

    exposure = load(ROOT / value["prerequisites"]["exposure_control"]["path"])
    assert (
        file_sha256(ROOT / value["prerequisites"]["exposure_control"]["path"])
        == (value["prerequisites"]["exposure_control"]["file_sha256"])
    )
    assert exposure["sha256"] == value["prerequisites"]["exposure_control"]["embedded_sha256"]
    split = load(ROOT / common["split"]["path"])
    assert file_sha256(ROOT / common["split"]["path"]) == common["split"]["file_sha256"]
    assert split["sha256"] == common["split"]["embedded_sha256"]
    protocol = load(ROOT / common["fleet_development_evaluation"]["protocol_path"])
    assert protocol["sha256"] == common["fleet_development_evaluation"]["protocol_embedded_sha256"]


def test_sampling_is_the_only_intended_scientific_difference_and_dose_is_exact():
    value = load(TEMPLATE)
    common = value["frozen_common_contract"]
    arms = value["arms"]
    exposure = load(ROOT / value["prerequisites"]["exposure_control"]["path"])["variants"]["a"]
    assert len(arms) == 2
    assert {arm["role"] for arm in arms} == {"control", "treatment"}
    assert all(set(arm) == {"id", "role", "data", "operational_identity"} for arm in arms)
    assert len({arm["data"]["sampling"] for arm in arms}) == 2
    assert len({arm["data"]["corpus_manifest_sha256"] for arm in arms}) == 2
    assert len({arm["operational_identity"]["run_name"] for arm in arms}) == 2
    assert len({arm["operational_identity"]["output_root"] for arm in arms}) == 2
    assert len({arm["operational_identity"]["wandb_run_id"] for arm in arms}) == 2

    control, treatment = arms
    assert control["data"]["corpus_manifest_sha256"] == exposure["matched_corpus_sha256"]
    assert control["data"]["train_parquet_sha256"] == exposure["matched_train_parquet_sha256"]
    assert control["data"]["source_selection_sha256"] == exposure["matched_source_selection_sha256"]
    assert control["data"]["selected_episode_ids_sha256"] == exposure["selected_episode_ids_sha256"]
    assert treatment["data"]["corpus_manifest_sha256"] == exposure["balanced_corpus_sha256"]
    assert treatment["data"]["train_parquet_sha256"] == exposure["balanced_train_parquet_sha256"]
    assert (
        treatment["data"]["source_selection_sha256"] == exposure["balanced_source_selection_sha256"]
    )
    assert (
        value["prerequisites"]["exposure_control"][
            "matched_control_materialization_acceptance_sha256"
        ]
        == load(ROOT / value["prerequisites"]["exposure_control"]["path"])[
            "materialization_acceptance"
        ]["variants"]["a"]["receipt_sha256"]
    )

    matched = ("covered_task_families", "episodes", "dense_windows", "supervised_tokens")
    assert {tuple(arm["data"][field] for field in matched) for arm in arms} == {
        (29, 50, 401, 419165)
    }
    recipe = common["recipe"]
    assert recipe["dense_windows"] == 401
    assert recipe["supervised_tokens"] == 419165
    assert recipe["optimizer_steps"] == math.ceil(401 / recipe["global_batch_size"])
    assert recipe["warmup_steps"] == math.ceil(recipe["optimizer_steps"] * recipe["warmup_ratio"])
    assert common["runtime"]["only_intended_scientific_difference"] == ("arm data sampling binding")
    assert value["pair_contract"]["allowed_differences"] == [
        "whole-episode data sampling binding",
        "create-once run, output and W&B identities",
    ]


def test_wandb_and_fleet_dev_protocol_are_exactly_shared_and_wbe_cannot_select():
    value = load(TEMPLATE)
    common = value["frozen_common_contract"]
    wandb = common["wandb"]
    assert wandb["history_keys"] == list(OUTCOME_WANDB_HISTORY_KEYS)
    assert wandb["series_step_axis"] == "train/total_supervised_tokens"
    assert wandb["expected_scalar_events_per_arm"] == 51
    assert wandb["expected_terminal_total_supervised_tokens_per_arm"] == 419165
    assert wandb["automatic_system_telemetry"] is False
    assert "external_benchmark_results" in wandb["forbidden_payloads"]
    assert common["fleet_development_evaluation"] == {
        "protocol_path": "configs/evaluation/qwen38-blackbox-fleet-dev-a-outcome-protocol-v2.json",
        "protocol_file_sha256": (
            "sha256:008d412bcdee122cd2b4f072e6f242496b3e8de1e8b07ad60e598de59202560e"
        ),
        "protocol_embedded_sha256": (
            "sha256:3c2c65eed748b16b50ef992327324b1fcaad84b190456c4aefe69c6efe748f75"
        ),
        "harness": "opencode",
        "harness_version": "1.18.27",
        "attempt_seeds": [42, 43, 44, 45],
        "task_count": 20,
        "pass_k": 4,
        "primary_metric": "fleet_dev_paired_mean_success_delta_4fixed",
        "selection_eligible": True,
    }
    assert "WebExploitBench input or selection" in value["pair_contract"]["prohibited"]


def test_preparation_evidence_is_self_digested_and_binds_template():
    value = load(EVIDENCE)
    assert value["sha256"] == digest_json(
        {key: row for key, row in value.items() if key != "sha256"}
    )
    assert value["template"] == {
        "path": str(TEMPLATE.relative_to(ROOT)),
        "file_sha256": file_sha256(TEMPLATE),
        "embedded_sha256": load(TEMPLATE)["sha256"],
    }
    assert value["status"] == "offline_prepared_not_submitted"
    assert value["live_actions"] == {
        "kubernetes_calls": 0,
        "jobs_api_preview_calls": 0,
        "jobs_api_submit_calls": 0,
        "wandb_runs_created": 0,
    }
    assert value["blockers"]
