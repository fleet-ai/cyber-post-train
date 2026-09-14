import json
import re
from pathlib import Path

from training.io import digest_json
from training.study_plan import compile_study

STUDIES = Path(__file__).parents[1] / "configs" / "studies"
CONFIG = STUDIES / "qwen-blackbox-teacher-balanced-exposure-v1.json"
PLAN_INPUT = STUDIES / "qwen-blackbox-teacher-balanced-v1.json"
PLAN = STUDIES / "qwen-blackbox-teacher-balanced-v1.plan.json"


def test_teacher_balance_lock_is_self_digested_and_nonlaunchable():
    value = json.loads(CONFIG.read_text())
    assert value["sha256"] == digest_json({k: v for k, v in value.items() if k != "sha256"})
    assert value["treatment"]["role"] == "teacher_availability_ablation"
    assert value["treatment"]["unit"] == "whole_certified_native_episode"
    assert value["launch_status"] == "blocked_until_dev_cluster_qualification"
    assert "partial episode truncation" in value["treatment"]["prohibited"]
    assert "duplicated token targets" in value["treatment"]["prohibited"]


def test_balanced_corpora_are_train_only_private_and_still_dev_gated():
    value = json.loads(CONFIG.read_text())
    assert value["corpus_contract"] == {
        "visibility": "private",
        "file_mode": "0600",
        "files": ["train.parquet"],
        "validation_mode": "task_outcomes_only",
        "teacher_cross_entropy": "not_computed",
    }
    assert value["launch_status"] == "blocked_until_dev_cluster_qualification"
    for variant in value["variants"].values():
        assert variant["balanced_corpus_status"] == "materialized_private_train_only"
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", variant["balanced_corpus_sha256"])
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", variant["balanced_train_parquet_sha256"])


def test_teacher_balance_preserves_coverage_while_reducing_family_inequality():
    value = json.loads(CONFIG.read_text())
    for variant in value["variants"].values():
        available, balanced = variant["available"], variant["balanced"]
        assert balanced["families"] == available["families"]
        assert balanced["episodes"] < available["episodes"]
        assert balanced["supervised_tokens"] < available["supervised_tokens"]
        assert balanced["family_token_gini"] < available["family_token_gini"]
        assert (
            balanced["family_token_coefficient_of_variation"]
            < available["family_token_coefficient_of_variation"]
        )
        for key, digest in variant.items():
            if key.endswith("sha256"):
                assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)


def test_balanced_plan_is_reproducible_nonlaunchable_and_binds_the_treatment():
    source = json.loads(PLAN_INPUT.read_text())
    frozen = json.loads(PLAN.read_text())
    exposure = json.loads(CONFIG.read_text())
    assert frozen == compile_study(source)
    assert frozen["sha256"] == digest_json(
        {key: value for key, value in frozen.items() if key != "sha256"}
    )
    assert frozen["execution"] == {
        "kind": "metadata_only_not_a_job_request",
        "dev_cluster_qualification_required": True,
        "new_jobs_authorized": False,
    }
    assert frozen["selection"]["teacher_reference_ce"] == ("not_computed_or_used_for_selection")
    arms = frozen["stages"][0]["arms"]
    assert {(arm["split"], arm["source"]) for arm in arms} == {
        ("a", "teacher"),
        ("b", "teacher"),
    }
    for variant in ("a", "b"):
        bound = frozen["sources"][variant]["teacher"]
        treatment = exposure["variants"][variant]
        assert bound["corpus_sha256"] == treatment["balanced_corpus_sha256"]
        assert bound["source_selection_sha256"] == treatment["balanced_source_selection_sha256"]
        assert bound["source_sessions"] == treatment["balanced"]["episodes"]
