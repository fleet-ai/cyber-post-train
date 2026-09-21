from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from evals.webexploitbench.cage import cage_patch_set_sha256
from evals.webexploitbench.tensorlake.benchmark_snapshot import (
    REVIEWED_TASK_ORDER_SHA256,
)

ROOT = Path(__file__).parents[1]
PLAN_RELATIVE = Path(
    "configs/evaluation/qwen38-important-checkpoints-opencode-wbe-campaign-v1.json"
)
PLAN_PATH = ROOT / PLAN_RELATIVE
EVIDENCE_PATH = (
    ROOT / "docs/evidence/qwen38-web-important-checkpoint-campaign-prepared-20260921.json"
)


def _load() -> dict:
    return json.loads(PLAN_PATH.read_bytes())


def _digest(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _campaigns(plan: dict) -> dict[str, int]:
    campaigns = {plan["shared_base"]["campaigns"]["canary"]: 1}
    campaigns.update(
        {campaign: 15 for campaign in plan["shared_base"]["campaigns"]["full_replicas"]}
    )
    for candidate in plan["candidates"]:
        campaigns[candidate["campaigns"]["canary"]] = 1
        campaigns.update({campaign: 15 for campaign in candidate["campaigns"]["full_replicas"]})
    return campaigns


def test_plan_is_self_digesting_inert_and_bound_to_reviewed_source() -> None:
    plan = _load()

    assert plan["schema"] == "cyber_qwen38_web_checkpoint_campaign_preparation_v1"
    assert plan["sha256"] == _digest(plan)
    assert plan["source"]["prepared_against_main_commit"] == (
        "276231a5e202cab0b2be2e06d5f5c0b99cb6e509"
    )
    for binding in plan["source"].values():
        if not isinstance(binding, dict) or "path" not in binding:
            continue
        assert _sha256(ROOT / binding["path"]) == binding["file_sha256"]

    status = plan["status"]
    assert status["state"] == "prepared_not_launchable"
    assert status["external_mutations_by_this_preparation"] == 0
    assert status["provider_api_calls_by_this_preparation"] == 0
    assert status["evaluation_attempts_by_this_preparation"] == 0
    assert status["capability_claimed"] is False
    assert plan["execution"]["launchable_now"] is False


def test_preparation_receipt_is_self_digesting_and_binds_the_exact_plan() -> None:
    receipt = json.loads(EVIDENCE_PATH.read_bytes())
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    assert receipt["receipt_sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest()
    assert receipt["plan"]["file_sha256"] == _sha256(PLAN_PATH)
    assert receipt["plan"]["plan_sha256"] == _load()["sha256"]
    assert receipt["readiness"]["state"] == "prepared_not_launchable"
    assert receipt["operation"] == {
        "provider_api_calls": 0,
        "sandboxes_created_resumed_snapshotted_or_released": 0,
        "inference_routes_created_resumed_modified_or_paused": 0,
        "benchmark_attempts": 0,
        "scoring_attempts": 0,
        "external_mutations": 0,
    }


def test_every_evidence_binding_exists_and_matches() -> None:
    for candidate in _load()["candidates"]:
        for evidence in candidate["evidence"]:
            assert _sha256(ROOT / evidence["path"]) == evidence["file_sha256"]


def test_exact_candidate_roster_and_checkpoint_bindings() -> None:
    candidates = _load()["candidates"]
    assert [row["artifact_id"] for row in candidates] == [
        "q38-fresh75-step230",
        "q38-teacher-dense-v5-step186",
        "q38-self-sft-step44",
        "q38-available-a-lr30-step76",
        "q38-lora-step60",
    ]
    assert [row["priority"] for row in candidates] == [1, 2, 3, 4, 5]

    expected = {
        "q38-fresh75-step230": (
            230,
            "sha256:36eec01f1dd3f0d47f6b099e79970d9533cf59c37d8e15478907413dd162e029",
            "chris-q38-fresh75-step230-web-v2",
        ),
        "q38-teacher-dense-v5-step186": (
            186,
            "sha256:e6d59f5a58ce54b913d775cd71b81c64bd54637b53795df82baec0f71efefaf9",
            "chris-q38-teacher-v5-step186-web-v1",
        ),
        "q38-self-sft-step44": (
            44,
            "sha256:a0e55bebe9d78ac76c012446ae0147a421cd333e2a6bf2bb6f4fb3450fa776fe",
            "chris-q38-self-sft-step44-v1",
        ),
        "q38-available-a-lr30-step76": (
            76,
            "sha256:3bef11697759b11db150b705e1882fb2d31a3efbdadbebd948aa917dcacc03a8",
            "chris-q38-lr30-step76-web-v1",
        ),
    }
    for candidate in candidates[:4]:
        step, revision, served_model = expected[candidate["artifact_id"]]
        assert candidate["checkpoint"]["optimizer_step"] == step
        assert candidate["export"]["payload_manifest_sha256"] == revision
        assert candidate["model_revision"] == revision
        assert candidate["served_model"] == served_model
        assert candidate["eligible_after_global_repair_gates"] is True

    lora = candidates[-1]
    assert lora["checkpoint"]["optimizer_step"] == 60
    assert lora["export"] is None
    assert lora["served_model"] is None
    assert lora["eligible_after_global_repair_gates"] is False
    assert len(lora["additional_gates"]) == 6


def test_protocol_is_current_opencode_only_and_score_free_during_collection() -> None:
    protocol = _load()["protocol"]

    assert protocol["harness"] == {
        "name": "opencode",
        "version": "1.18.27",
        "qwen_code_allowed": False,
        "tool_surface": ["web_bash", "web_submit_findings"],
    }
    assert protocol["benchmark"]["task_order_sha256"] == REVIEWED_TASK_ORDER_SHA256
    assert protocol["cage"]["patch_set_sha256"] == cage_patch_set_sha256()
    assert protocol["sampling"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_output_tokens": 32768,
        "scientific_attempt_retries": 0,
    }
    assert protocol["context"]["max_context_tokens"] == 262144
    assert protocol["context"]["max_input_tokens"] == 229376
    assert protocol["context"]["automatic_compaction"] is True
    assert protocol["context"]["automatic_continuation"] is True
    assert protocol["context"]["compaction_reserved_tokens"] == 20000
    assert protocol["budget"] == {
        "max_agent_steps": 600,
        "task_max_duration_minutes": 480,
        "sandbox_timeout_seconds": 28800,
    }

    split = protocol["collection_and_scoring"]
    assert split["collection_output"] == "immutable_score_free_rollout_bundle"
    assert split["judge_calls_during_collection"] == 0
    assert split["agent_calls_during_scoring"] == 0
    assert split["judge_identity_in_collection_plan"] is False
    assert split["recollection_required_after_scoring_failure"] is False
    forbidden_collection_fields = {
        "judge_model",
        "judge_provider",
        "judge_api_key",
        "judge_sampling",
    }
    assert forbidden_collection_fields.isdisjoint(split)


def test_campaign_shape_and_all_operational_identities_are_unique() -> None:
    plan = _load()
    shape = plan["comparison"]["aggregate_shape"]
    assert shape["canary"] == {
        "independent_replicas": 1,
        "tasks_per_replica": 1,
        "pass_k_per_replica": 1,
        "collections_per_arm": 1,
    }
    assert shape["full"]["independent_replicas"] == 4
    assert shape["full"]["tasks_per_replica"] == 15
    assert shape["full"]["pass_k_per_replica"] == 1
    assert shape["full"]["collections_per_arm"] == 60

    campaigns = _campaigns(plan)
    assert len(campaigns) == 30
    identities: set[str] = set()
    templates = plan["identity_templates"]
    for campaign, task_count in campaigns.items():
        for task_index in range(task_count):
            values = {
                templates[key].format(campaign_id=campaign, task_index=task_index)
                for key in (
                    "sandbox_name",
                    "run_id",
                    "lifecycle_path",
                    "result_root",
                    "collection_path",
                    "collection_acceptance_path",
                )
            }
            assert len(values) == 6
            assert identities.isdisjoint(values)
            identities.update(values)


def test_every_pair_reuses_the_same_base_and_changes_only_weights() -> None:
    plan = _load()
    candidates = {row["artifact_id"]: row for row in plan["candidates"]}
    base_campaigns = plan["shared_base"]["campaigns"]
    intervention = set(plan["comparison"]["weight_intervention_fields"])
    held_constant = set(plan["comparison"]["held_constant_student_fields"])
    assert intervention == {
        "served_model",
        "model_revision",
        "model_artifact_sha256",
        "models_file_sha256",
    }
    assert held_constant == {
        "endpoint_origin_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "max_context_size",
        "inference_precision",
        "quantization",
        "serving_runtime_sha256",
    }

    pair_ids: set[str] = set()
    for pairing in plan["pairings"]:
        candidate = candidates[pairing["artifact_id"]]
        assert pairing["canary"]["base_campaign_id"] == base_campaigns["canary"]
        assert pairing["canary"]["candidate_campaign_id"] == candidate["campaigns"]["canary"]
        for index, row in enumerate(pairing["full_replicas"]):
            assert row["base_campaign_id"] == base_campaigns["full_replicas"][index]
            assert row["candidate_campaign_id"] == candidate["campaigns"]["full_replicas"][index]
        new_pair_ids = {pairing["canary"]["pair_id"]} | {
            row["pair_id"] for row in pairing["full_replicas"]
        }
        assert pair_ids.isdisjoint(new_pair_ids)
        pair_ids.update(new_pair_ids)
    assert len(pair_ids) == 25


def test_failed_v23_is_permanently_excluded_and_relaunch_fails_closed() -> None:
    plan = _load()
    recovery = plan["recovery"]
    old = recovery["old_v23_disposition"]
    assert old["accepted_collections"] == 0
    assert old["failure_preserved_and_released"] == 60
    assert old["failure_classes"] == {"collection_export": 45, "collection_run": 15}
    assert recovery["exact16_review"]["accepted_collections"] == 0
    assert recovery["exact16_review"]["failure_preserved_and_released"] == 16
    assert recovery["exact16_review"]["provider_release_replay_allowed"] is False
    assert recovery["attach_client_repair"]["sufficient_for_relaunch"] is False
    assert len(recovery["fresh_launch_gates"]) == 6

    proposed = set(_campaigns(plan))
    forbidden = set(old["permanently_excluded_campaign_ids"])
    assert proposed.isdisjoint(forbidden)
    assert old["permanently_excluded_launch_receipt_sha256"] == (
        "sha256:1889b6ff0e80e7709841d23447f44fb9a87b587e4b6b7cd3f5e36b619d44db09"
    )
    assert plan["execution"]["live_provider_inventory_required"] is True
    assert plan["execution"]["fresh_duplicate_census_max_age_seconds"] == 600


def test_proposed_campaign_ids_were_absent_from_the_tracked_base() -> None:
    plan = _load()
    proposed = set(_campaigns(plan))
    excluded = {str(PLAN_RELATIVE), str(Path(__file__).relative_to(ROOT))}
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    hits: dict[str, list[str]] = {}
    for relative in tracked:
        if relative in excluded:
            continue
        path = ROOT / relative
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for campaign in proposed:
            if campaign in text:
                hits.setdefault(campaign, []).append(relative)
    assert hits == {}


def test_plan_contains_no_secret_or_private_result_payload() -> None:
    text = PLAN_PATH.read_text().lower()
    assert "tensorlake-api-key" not in text
    assert "fleet_api_key" not in text
    assert "aws_secret_access_key" not in text
    assert "private-reload.log" not in text
    privacy = _load()["privacy"]
    assert privacy == {
        "credentials_included": False,
        "prompts_traces_flags_answers_or_scores_included": False,
        "raw_logs_included": False,
        "tensor_values_read": False,
    }
