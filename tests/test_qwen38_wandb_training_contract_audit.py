"""Offline contract for the current Qwen3.8 W&B training audit."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256
from training.sft_runtime import OUTCOME_WANDB_HISTORY_KEYS

ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-14-wandb-training-contract-audit-v1.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_audit_is_self_digesting_offline_and_binds_exact_inputs() -> None:
    audit = load(AUDIT)
    assert audit["sha256"] == digest_json(
        {key: value for key, value in audit.items() if key != "sha256"}
    )
    assert audit["status"] == "offline_audit_complete"
    assert audit["method"] == {
        "live_wandb_calls": 0,
        "private_data_reads": 0,
        "cluster_calls": 0,
        "jobs_api_calls": 0,
        "secrets_read": 0,
        "executable_or_prepared_config_bytes_changed": False,
        "reason_no_runtime_edit": (
            "Changing a compiler, runtime, or prepared config would invalidate the "
            "existing plan, request, runtime-bundle, and offline qualification digests. "
            "Proven comparison gaps are recorded as promotion blockers instead."
        ),
    }
    for binding in audit["bound_sources"].values():
        path = ROOT / binding["path"]
        assert binding["file_sha256"] == file_sha256(path)
        if "embedded_sha256" in binding:
            assert load(path)["sha256"] == binding["embedded_sha256"]


def test_sft_contract_has_one_scalar_schema_and_two_explicit_step_axes() -> None:
    audit = load(AUDIT)
    runtime = audit["runtime_contracts"]["sft"]
    assert tuple(runtime["history_keys"]) == OUTCOME_WANDB_HISTORY_KEYS
    assert runtime["scalar_only"] is True
    assert runtime["automatic_system_telemetry"] is False
    assert runtime["step_axes"] == {
        "optimizer_progress": "train/global_step, one-indexed",
        "comparison_curve": "train/total_supervised_tokens, cumulative",
        "reason": (
            "Dense windows contain different numbers of supervised tokens, so token "
            "exposure is the fair cross-arm x-axis."
        ),
    }
    source = (ROOT / "training/sft_runtime.py").read_text(encoding="utf-8")
    assert '"WANDB_RESUME": "never"' in source
    assert 'run.define_metric("train/total_supervised_tokens")' in source
    assert 'step_metric="train/total_supervised_tokens"' in source
    assert '"experiment_plan_sha256": self.plan["plan_sha256"]' in source
    request_source = (ROOT / "training/sft.py").read_text(encoding="utf-8")
    assert '"secrets": ["wandb-api"]' in request_source
    assert "WANDB_API_KEY" not in request_source


def test_sft_arm_identities_are_unique_and_comparison_groups_are_explicit() -> None:
    lr30 = load(
        ROOT / "configs/qualification/qwen38-teacher-lr30-step6-resume-to76-dev-v1.json"
    )
    lr100 = load(
        ROOT
        / "configs/qualification/qwen38-teacher-lr100-step21-resume-to76-dev-v1.template.json"
    )
    lr100_source = load(ROOT / lr100["source"]["config"]["path"])
    replacements = lr100["materialization"]["source_config_transform"]["replace"]
    lr100_wandb = {
        **lr100_source["wandb"],
        "group": replacements["wandb.group"],
        "run_id": replacements["wandb.run_id"],
        "name": replacements["wandb.name"],
        "tags": replacements["wandb.tags"],
    }
    assert lr30["wandb"]["entity"] == lr100_wandb["entity"] == "thefleet"
    assert lr30["wandb"]["project"] == lr100_wandb["project"] == "cyber-post-train"
    assert lr30["wandb"]["group"] == lr100_wandb["group"]
    assert lr30["wandb"]["run_id"] == lr30["wandb"]["name"] == lr30["name"]
    assert lr100_wandb["run_id"] == lr100_wandb["name"]
    assert lr30["wandb"]["run_id"] != lr100_wandb["run_id"]

    pair = load(
        ROOT
        / "configs/qualification/qwen38-teacher-exposure-matched-sampling-pair-dev-v1.template.json"
    )
    wandb = pair["frozen_common_contract"]["wandb"]
    identities = [arm["operational_identity"] for arm in pair["arms"]]
    assert wandb["group"] == "q38-teacher-exposure-matched-sampling-a-v1"
    assert wandb["history_keys"] == list(OUTCOME_WANDB_HISTORY_KEYS)
    assert wandb["series_step_axis"] == "train/total_supervised_tokens"
    assert wandb["expected_scalar_events_per_arm"] == 51
    assert wandb["expected_terminal_total_supervised_tokens_per_arm"] == 419165
    assert len({identity["wandb_run_id"] for identity in identities}) == 2
    for identity in identities:
        assert identity["run_name"] == identity["wandb_run_id"] == identity["wandb_name"]


def test_miles_contract_is_unique_scalar_only_but_not_yet_grouped_for_a_series() -> None:
    config = load(
        ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-dev-v8.json"
    )
    launch = load(
        ROOT / "configs/qualification/qwen38-miles-rl-reward-canary-launch-dev-v8.json"
    )
    wandb = config["wandb"]
    assert wandb == {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "run_id": config["name"],
    }
    assert launch["identities"]["wandb_run_id"] == config["name"]
    source = (ROOT / "training/miles.py").read_text(encoding="utf-8")
    assert '"wandb-group": config.name' in source
    assert '"wandb-run-id": config.wandb_run_id' in source
    assert '"disable-wandb-random-suffix"' in source
    assert '"--wandb-key"' in source
    compiler = (ROOT / "training/miles_training.py").read_text(encoding="utf-8")
    assert '"secrets": ["fleet-api", "wandb-api"]' in compiler
    assert "WANDB_API_KEY" not in compiler
    acceptance = (ROOT / "training/miles_acceptance.py").read_text(encoding="utf-8")
    assert 'value.get("observed_train_steps") != [0]' in acceptance
    assert 'value.get("logged_artifact_count") != 0' in acceptance
    assert 'value.get("rich_payload_count") != 0' in acceptance
    assert 'value.get("reward_values_included") is not False' in acceptance
    audited = load(AUDIT)["runtime_contracts"]["miles"]
    assert audited["comparison_limit"].startswith(
        "A one-run group is adequate for the Dev8 operational canary"
    )


def test_no_prepared_arm_is_safe_after_auth_alone_and_wbe_cannot_select_training() -> None:
    audit = load(AUDIT)
    arms = {item["arm"]: item for item in audit["arms"]}
    assert all(item["unsafe_after_auth_alone"] is True for item in arms.values())
    assert arms["chris-q38-miles-rlreward-dev8"]["status"].startswith(
        "conditionally_safe_as_one_dev_operational_canary"
    )
    for name in (
        "teacher.available-a.lr-1e-4.batch-8.epoch-1.seed-42.resume-step21-to76",
        "self.direct.split-a.lr-1e-5.batch-8.epoch-1.seed-42",
        "qwen38-teacher-exposure-matched-sampling-a-v1",
        "chris-q38-miles-rl-prod2",
        "chris-q38-rl-prod1",
    ):
        assert arms[name]["status"] == "unsafe_not_launchable"
    assert audit["conclusion"]["external_benchmark_selection_excluded"] is True
    lr30 = load(
        ROOT
        / "configs/qualification/qwen38-teacher-lr30-step6-resume-to76-dev-v1.template.json"
    )
    lr100 = load(
        ROOT
        / "configs/qualification/qwen38-teacher-lr100-step21-resume-to76-dev-v1.template.json"
    )
    for arm in (lr30, lr100):
        assert arm["selection_evaluation"]["training_loss_selection_eligible"] is False
        assert arm["selection_evaluation"]["external_benchmark_selection_eligible"] is False
    prod = load(ROOT / "configs/runs/qwen38-miles-rl-filtered-study-a-prod-v2.template.json")
    boundaries = prod["scientific_boundaries"]
    assert boundaries["external_benchmark_training_rows"] == 0
    assert boundaries["external_benchmark_reward_inputs"] == 0
    assert boundaries["external_benchmark_hpo_or_checkpoint_inputs"] == 0
    assert boundaries["external_benchmark_retry_inputs"] == 0
