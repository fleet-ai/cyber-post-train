"""Offline checks for the fail-closed next Qwen direct self-SFT dev arm."""

import json
from pathlib import Path

from cyber_post_train.jobs import digest
from training.io import digest_json, file_sha256
from training.sft import compile_sft, job_request
from training.sft_runtime import DENSE_FORMAT, DENSE_SCHEMA, optimizer_schedule

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "configs/qualification"
STUDY = ROOT / "configs/studies/qwen-blackbox-self-sft-next-dev-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-14-self-sft-next-dev-preparation-v1.json"
INDEX = QUALIFICATION / "qwen38-self-trace-index-a-v2.template.json"
CORPUS = QUALIFICATION / "qwen38-self-trace-corpus-a-v2.template.json"
SFT = QUALIFICATION / "qwen38-self-sft-direct-a-lr10-dev-v2.template.json"


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def assert_sealed(value: dict) -> None:
    assert value["sha256"] == digest_json({k: v for k, v in value.items() if k != "sha256"})


def test_audit_and_prepared_file_bindings_are_exact():
    study, evidence = load(STUDY), load(EVIDENCE)
    assert_sealed(study)
    assert_sealed(evidence)
    assert file_sha256(STUDY) == evidence["prepared_files"]["study"]["file_sha256"]
    assert study["sha256"] == evidence["prepared_files"]["study"]["embedded_sha256"]

    prepared = {
        "reviewed_index": INDEX,
        "corpus": CORPUS,
        "first_dev_arm": SFT,
    }
    evidence_names = {
        "reviewed_index": "index_template",
        "corpus": "corpus_template",
        "first_dev_arm": "sft_template",
    }
    for name, path in prepared.items():
        expected = study["prepared_templates"][name]["file_sha256"]
        assert file_sha256(path) == expected
        assert evidence["prepared_files"][evidence_names[name]]["file_sha256"] == expected

    request = ROOT / "configs/qualification/qwen38-self-trace-collection-request-v3.json"
    audit = study["source_audit"]["current_direct_collection_request"]
    assert file_sha256(request) == audit["file_sha256"]
    assert load(request)["sha256"] == audit["embedded_sha256"]
    assert audit["status"] == "blocked_external_bindings"

    historical = ROOT / "docs/evidence/qwen38-study/2026-09-11-self-sft-interface-audit-v2.json"
    historical_audit = study["source_audit"]["historical_interface_audit"]
    assert file_sha256(historical) == historical_audit["file_sha256"]
    assert load(historical)["sha256"] == historical_audit["embedded_sha256"]
    assert historical_audit["direct_interface_compatible_sessions"] == 0
    assert historical_audit["admission"] == "forbidden_for_this_direct_interface_arm"

    existing = study["source_audit"]["existing_filtered_templates"]
    for name in ("index", "corpus", "sft"):
        path = ROOT / existing[name]["path"]
        assert file_sha256(path) == existing[name]["file_sha256"]
    assert load(ROOT / existing["index"]["path"])["episodes"] == []
    assert existing["corpus"]["source_collection_bound"] is False
    assert existing["sft"]["data_manifest_bound"] is False

    teacher = study["teacher_comparison"]["teacher_reference"]
    teacher_path = ROOT / teacher["path"]
    assert file_sha256(teacher_path) == teacher["file_sha256"]
    assert load(teacher_path)["sha256"] == teacher["embedded_sha256"]


def test_templates_preserve_split_interface_and_fail_closed_bindings():
    study, index, corpus, sft = load(STUDY), load(INDEX), load(CORPUS), load(SFT)
    primary = study["representative_split"]["primary"]
    train = study["representative_split"]["train_projection"]

    assert primary["train_task_families"] == 59
    assert primary["dev_task_families"] == 20
    assert primary["final_task_families"] == 10
    assert file_sha256(ROOT / primary["path"]) == primary["file_sha256"]
    assert load(ROOT / primary["path"])["sha256"] == primary["embedded_sha256"]
    train_path = ROOT / train["path"]
    assert file_sha256(train_path) == train["file_sha256"] == corpus["split"]["sha256"]
    assert load(train_path)["sha256"] == train["embedded_sha256"]
    secondary = study["representative_split"]["secondary_confirmation_only"]
    final = study["representative_split"]["final_test_lock"]
    for binding in (secondary, final):
        path = ROOT / binding["path"]
        assert file_sha256(path) == binding["file_sha256"]
        assert load(path)["sha256"] == binding["embedded_sha256"]

    request = load(ROOT / study["source_audit"]["current_direct_collection_request"]["path"])
    assert index["producer_plan_sha256"] == request["sha256"]
    assert index["episodes"] == []
    assert index["interface"]["backend"] == "skyrl_direct"
    assert index["interface"]["required_task_tools"] == ["bash", "submit_report"]
    assert index["interface"]["compaction"] == "disabled"
    assert index["interface"]["recorder_source_sha256"] == file_sha256(
        ROOT / "training/skyrl_episode.py"
    )
    assert index["sha256"] == "PENDING_CANONICAL_SELF_DIGEST"
    assert "PENDING" in index["review_receipt_sha256"]
    assert "PENDING" in index["route_certificate_sha256"]

    assert corpus["max_episodes_per_task_version"] == 2
    assert corpus["max_supervised_tokens_per_task_version"] == 49152
    assert corpus["max_submit_token_share"] <= 0.5
    assert corpus["max_submit_response_share"] <= 0.5
    assert corpus["fleet_dev_protocol_sha256"].startswith("PENDING_MATCHED_SKYRL_DIRECT")
    assert sft["data"]["manifest_sha256"].startswith("PENDING_")
    assert sft["fleet_dev_protocol_sha256"] == corpus["fleet_dev_protocol_sha256"]
    assert sft["validation_mode"] == "task_outcomes_only"
    assert sft["recipe"]["eval_interval"] == 0
    assert sft["name"] == sft["wandb"]["run_id"] == sft["wandb"]["name"]
    assert sft["output_root"] == study["prepared_templates"]["first_dev_arm"]["create_once_output"]
    operational_templates = json.dumps({"index": index, "corpus": corpus, "sft": sft})
    assert "webexploitbench" not in operational_templates.lower()


def test_materialized_template_compiles_to_one_bounded_c1_dev_request(tmp_path):
    config = load(SFT)
    protocol = "sha256:" + "d" * 64
    manifest = {
        "schema": "cyber_dense_sft_corpus_v1",
        "tokenizer": {
            "repo": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        },
        "split_sha256": "sha256:" + "a" * 64,
        "validation_mode": "task_outcomes_only",
        "fleet_dev_protocol_sha256": protocol,
        "files": {
            "train": {
                "path": "train.parquet",
                "rows": 64,
                "sha256": "b" * 64,
                "task_keys": ["synthetic-offline-train-task"],
                "format": DENSE_FORMAT,
                "supervised_tokens": 6400,
                "assistant_responses": 128,
                "excluded_assistant_responses": 2,
                "source_sessions": 16,
                "source_total_assistant_responses": 130,
            }
        },
    }
    manifest["sha256"] = "sha256:" + digest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config["data"]["manifest"] = str(manifest_path)
    config["data"]["manifest_sha256"] = manifest["sha256"]
    config["fleet_dev_protocol_sha256"] = protocol

    plan = compile_sft(config, relative_to=SFT.parent)
    request = job_request(plan)
    assert plan["schema"] == DENSE_SCHEMA
    assert plan["validation_mode"] == "task_outcomes_only"
    assert set(plan["datasets"]) == {"train"}
    assert plan["fleet_dev_protocol_sha256"] == protocol
    assert plan["recipe"]["max_steps"] == 8
    assert plan["pause_after_step"] == 6
    assert optimizer_schedule(plan["recipe"])["num_warmup_steps"] == 1
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
    assert request["secrets"] == ["wandb-api"]
    assert request["env"]["WANDB_MODE"] == "online"
    assert request["env"]["WANDB_DISABLE_CODE"] == "true"


def test_selection_and_acceptance_boundaries_stay_closed():
    study, evidence = load(STUDY), load(EVIDENCE)
    assert study["teacher_comparison"]["ready"] is False
    assert study["teacher_comparison"]["source_comparison_claim_allowed"] is False
    assert "skyrl_direct" in " ".join(study["teacher_comparison"]["current_blockers"])
    assert "OpenCode 1.18.27" in " ".join(study["teacher_comparison"]["current_blockers"])
    assert study["selection_policy"]["training_loss"] == "fitting diagnostic only"
    external = study["selection_policy"]["webexploitbench"]
    assert external["source_selection_eligible"] is False
    assert external["hyperparameter_selection_eligible"] is False
    assert external["checkpoint_selection_eligible"] is False
    assert all(
        value is False for stage in study["acceptance_gates"].values() for value in stage.values()
    )
    assert all(value is False for value in study["claim_boundaries"].values())
    assert all(value is False for value in evidence["claim_boundaries"].values())
    assert evidence["execution"] == {
        "live_calls_performed": False,
        "cluster_api_registry_wandb_or_model_calls_performed": False,
        "private_prompts_traces_answers_flags_or_scores_read": False,
        "credentials_used": False,
    }
