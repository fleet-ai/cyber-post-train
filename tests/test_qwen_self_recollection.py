"""Static invariants for the blocked Qwen direct-interface recollection gate."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs/studies/qwen-blackbox-self-recollection-v1.json"
AUDIT = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-11-self-sft-interface-audit-v2.json"
)
PARITY = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-12-self-trace-recorder-dense-parity-v1.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_self_source_audit_is_sealed_metadata_only_and_still_blocked():
    audit = read(AUDIT)
    assert audit["sha256"] == digest_json(
        {key: value for key, value in audit.items() if key != "sha256"}
    )
    assert audit["status"] == "blocked_no_native_compatible_source"
    assert audit["interface_gate"]["admitted_native_compatible_sessions"] == 0
    assert audit["exact_ledger_audit"]["observed"]["eligible_89"] == {
        "sessions": 87,
        "task_versions": 43,
    }
    assert audit["catalog_corroboration"]["exact_ledger_successes_found"] == 87
    assert audit["catalog_corroboration"]["exact_ledger_successes_missing"] == 0
    assert audit["catalog_corroboration"]["positive_qwen38_candidates_after_prior_audit"] == 0
    assert audit["execution"] == {
        "cluster_or_api_mutations_performed": False,
        "model_requests_performed": 0,
        "prompts_or_transcripts_read": 0,
        "scores_persisted_or_reported": 0,
        "training_or_scored_evaluation_submitted": False,
    }


def test_recollection_plan_is_sealed_nonlaunchable_and_binds_offline_evidence():
    plan, audit, parity = read(PLAN), read(AUDIT), read(PARITY)
    assert plan["sha256"] == digest_json(
        {key: value for key, value in plan.items() if key != "sha256"}
    )
    assert plan["historical_source_gate"]["audit_sha256"] == audit["sha256"]
    assert plan["execution"]["launchable"] is False
    assert plan["execution"]["new_jobs_authorized"] is False
    assert plan["historical_source_gate"]["native_compatible_sessions"] == 0
    assert plan["route_treatments"]["selected"] is None
    assert plan["status"] == (
        "blocked_missing_exact_direct_collector_image_route_and_system_prompt"
    )
    assert plan["unresolved"] == {
        "direct_base_route_certificate_sha256": None,
        "direct_collector_image_digest": None,
        "direct_system_prompt_sha256": None,
    }
    offline = plan["offline_qualification"]
    assert offline["cluster_or_api_mutations_performed"] is False
    assert offline["target_model_or_runtime_used"] is False
    binding = offline["synthetic_recorder_dense_parity"]
    assert binding == {
        "path": PARITY.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(PARITY),
        "document_sha256": parity["sha256"],
    }
    assert parity["sha256"] == digest_json(
        {key: value for key, value in parity.items() if key != "sha256"}
    )
    assert plan["collection"]["tool_order"] == ["bash", "submit_report"]
    assert plan["collection"]["transcript_policy"]["tool_rewriting"] == "prohibited"


def test_recollection_uses_only_representative_train_tasks_and_no_holdout():
    plan = read(PLAN)
    eligible = read(ROOT / plan["task_universe"]["path"])
    assert file_sha256(ROOT / plan["model"]["lock_path"]) == plan["model"][
        "lock_file_sha256"
    ]
    assert eligible["sha256"] == plan["task_universe"]["sha256"]
    assert len(eligible["task_versions"]) == plan["task_universe"]["task_versions"] == 89

    train_sets = {}
    holdout_sets = {}
    for variant, binding in plan["splits"].items():
        if variant == "overlap":
            continue
        split = read(ROOT / binding["path"])
        assert split["sha256"] == binding["sha256"]
        assert split["training_split"]["sha256"] == binding["training_split_sha256"]
        train_sets[variant] = {
            (row["task_key"], row["task_version_id"])
            for row in split["training_split"]["tasks"]
        }
        holdout_sets[variant] = {
            (row["task_key"], row["task_version_id"])
            for row in split["tasks"]
            if row["split"] != "train"
        }
        assert len(train_sets[variant]) == binding["train_task_versions"] == 59
        assert train_sets[variant].isdisjoint(holdout_sets[variant])

    union = train_sets["a"] | train_sets["b"]
    intersection = train_sets["a"] & train_sets["b"]
    assert len(union) == plan["splits"]["overlap"]["train_union_task_versions"] == 74
    assert len(intersection) == plan["splits"]["overlap"][
        "train_intersection_task_versions"
    ] == 44
    assert plan["collection"]["maximum_sessions"] == (
        len(union) * plan["collection"]["attempts_per_task_version"]
    )
