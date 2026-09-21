from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKLOG_RELATIVE = Path("configs/evaluation/qwen38-opencode-web-servable-backlog-v2.json")
BACKLOG = ROOT / BACKLOG_RELATIVE
LEDGER = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v1.json"
READINESS = ROOT / "configs/evaluation/qwen38-checkpoint-serving-readiness-v1.json"
HISTORICAL = ROOT / "configs/evaluation/qwen38-important-checkpoints-opencode-wbe-campaign-v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _self_digest(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("sha256")
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _campaign_ids(backlog: dict) -> set[str]:
    identities = set()
    for candidate in backlog["selected_candidates"]:
        campaigns = candidate["candidate_campaigns"]
        identities.add(campaigns["canary"])
        identities.update(campaigns["full_replicas"])
    return identities


def _historical_campaign_ids(plan: dict) -> set[str]:
    shared = plan["shared_base"]["campaigns"]
    identities = {shared["canary"], *shared["full_replicas"]}
    for candidate in plan["candidates"]:
        campaigns = candidate["campaigns"]
        identities.add(campaigns["canary"])
        identities.update(campaigns["full_replicas"])
    return identities


def test_backlog_is_self_digesting_score_free_and_not_a_launch_packet() -> None:
    backlog = _load(BACKLOG)

    assert backlog["schema"] == "cyber_qwen38_opencode_web_servable_backlog_v2"
    assert backlog["sha256"] == _self_digest(backlog)
    assert backlog["execution"] == {
        "state": "prepared_not_launchable",
        "launch_authority": False,
        "provider_api_calls_by_this_preparation": 0,
        "model_requests_by_this_preparation": 0,
        "judge_requests_by_this_preparation": 0,
        "benchmark_attempts_by_this_preparation": 0,
        "scoring_attempts_by_this_preparation": 0,
        "recommended_order": [
            "q38-fresh75-step230",
            "q38-teacher-dense-v5-step186",
            "q38-self-sft-step44",
        ],
        "route_lifecycle_rule": (
            "Keep paused routes paused until their matched evaluator is ready. Resume exactly "
            "one candidate route only for its accepted collection, then pause it after the "
            "terminal collection receipt is preserved."
        ),
    }
    assert backlog["scientific_boundary"]["evaluation_only"] is True
    assert backlog["scientific_boundary"]["qwen_code_allowed"] is False
    assert backlog["scientific_boundary"]["collection_and_scoring_separate"] is True
    assert backlog["scientific_boundary"]["judge_calls_during_collection"] == 0
    assert backlog["scientific_boundary"]["agent_calls_during_scoring"] == 0
    assert backlog["scientific_boundary"]["scoring_failure_does_not_require_recollection"] is True
    assert backlog["deferred_scoring"] == {
        "state": "not_sealed_no_accepted_collection_bundle",
        "model_agnostic": True,
        "agent_calls": 0,
        "recollection_on_scoring_failure": False,
        "requires": [
            "an_accepted_immutable_score_free_rollout_bundle",
            "a_fresh_scoring_only_packet_that_binds_the_collection_bundle_digest_and_exact_scoring_policy",
            "a_scoring_receipt_that_never_changes_or_reexecutes_the_collection",
        ],
    }
    assert backlog["privacy"] == {
        "credentials_included": False,
        "prompts_traces_flags_answers_or_scores_included": False,
        "raw_logs_included": False,
        "tensor_values_read": False,
    }


def test_backlog_binds_the_exact_ledger_and_readiness_inputs() -> None:
    backlog = _load(BACKLOG)
    sources = backlog["source"]

    assert sources["checkpoint_evaluation_ledger"] == {
        "path": str(LEDGER.relative_to(ROOT)),
        "file_sha256": _sha256(LEDGER),
    }
    readiness = _load(READINESS)
    assert sources["serving_readiness_matrix"] == {
        "path": str(READINESS.relative_to(ROOT)),
        "file_sha256": _sha256(READINESS),
        "matrix_sha256": readiness["sha256"],
    }


def test_only_accepted_servable_artifacts_are_selected() -> None:
    backlog = _load(BACKLOG)
    ledger = _load(LEDGER)
    readiness = {row["artifact_id"]: row for row in _load(READINESS)["artifacts"]}
    selected = backlog["selected_candidates"]

    assert [row["artifact_id"] for row in selected] == [
        "q38-fresh75-step230",
        "q38-teacher-dense-v5-step186",
        "q38-self-sft-step44",
    ]
    assert [row["priority"] for row in selected] == [1, 2, 3]
    assert {row["artifact_id"] for row in selected}.issubset(
        {row["artifact_id"] for row in ledger["accepted_checkpoints"]}
    )

    for candidate in selected:
        artifact = readiness[candidate["artifact_id"]]
        assert artifact["checkpoint"]["status"] == "accepted"
        assert artifact["export"]["status"] == "accepted"
        assert artifact["reload"]["status"] == "accepted_one_gpu_forward"
        assert artifact["stage"]["status"] == "accepted"
        assert candidate["checkpoint_step"] == artifact["checkpoint"]["optimizer_step"]
        assert (
            candidate["checkpoint_manifest_sha256"]
            == artifact["checkpoint"]["manifest_file_sha256"]
        )
        identity = candidate["export_identity"]
        if identity["kind"] == "payload_manifest_sha256":
            assert identity["value"] == artifact["export"]["payload_manifest_sha256"]
        elif identity["kind"] == "export_receipt_sha256":
            assert identity["value"] == artifact["export"]["receipt_sha256"]
        else:
            assert identity == {
                "kind": "stage_payload_manifest_sha256",
                "value": artifact["stage"]["payload_manifest_sha256"],
            }

    assert readiness["q38-available-a-lr30-step76"]["export_reload"]["status"] == (
        "missing_one_gpu_inference_forward_receipt"
    )
    assert "q38-available-a-lr30-step76" not in {row["artifact_id"] for row in selected}
    assert "q38-lora-recovery-step40-to-step42" in {
        row["artifact_id"] for row in backlog["deferred_artifacts"]
    }


def test_protocol_is_opencode_only_and_keeps_all_non_weight_inputs_fixed() -> None:
    backlog = _load(BACKLOG)
    protocol = backlog["frozen_protocol_template"]
    comparison = backlog["matched_comparison"]

    assert protocol["benchmark"]["name"] == "WebExploitBench"
    assert protocol["harness"] == {
        "name": "opencode",
        "version": "1.18.27",
        "tool_surface": ["web_bash", "web_submit_findings"],
    }
    assert protocol["context"]["max_context_tokens"] == 262144
    assert protocol["context"]["automatic_compaction"] is True
    assert protocol["context"]["automatic_continuation"] is True
    assert protocol["sampling"]["scientific_attempt_retries"] == 0
    assert comparison["shared_base_required"] is True
    assert comparison["allowed_difference"] == (
        "accepted_model_weight_manifest_and_its_freshly_proven_route_binding"
    )
    assert "scoring_policy" in comparison["held_constant_fields"]
    assert "runner" in protocol["execution_seal_requirement"]


def test_campaign_identities_are_new_unique_and_live_census_remains_mandatory() -> None:
    backlog = _load(BACKLOG)
    campaigns = _campaign_ids(backlog)

    assert len(campaigns) == 15
    assert all(identity.endswith(("-c1", "-r0", "-r1", "-r2", "-r3")) for identity in campaigns)
    assert campaigns.isdisjoint(backlog["deduplication"]["historical_campaigns_never_reusable"])
    historical_source = backlog["deduplication"]["historical_preparation_identity_source"]
    assert historical_source["path"] == str(HISTORICAL.relative_to(ROOT))
    historical = _load(HISTORICAL)
    assert historical_source["plan_sha256"] == historical["sha256"]
    assert campaigns.isdisjoint(_historical_campaign_ids(historical))
    assert (
        backlog["deduplication"]["live_provider_census_required_immediately_before_create"] is True
    )
    assert backlog["deduplication"]["live_provider_census_max_age_seconds"] == 600

    excluded = {str(BACKLOG_RELATIVE), str(Path(__file__).relative_to(ROOT))}
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    hits: dict[str, list[str]] = {}
    for relative in tracked:
        if relative in excluded:
            continue
        path = ROOT / relative
        if not path.is_file() or path.suffix not in {".json", ".md", ".py", ".yaml", ".yml"}:
            continue
        content = path.read_text(errors="replace")
        for campaign in campaigns:
            if campaign in content:
                hits.setdefault(campaign, []).append(relative)
    assert hits == {}


def test_public_backlog_does_not_contain_secret_or_private_trace_material() -> None:
    serialized = BACKLOG.read_text().lower()
    for forbidden in ("api_key", "bearer ", "authorization:"):
        assert forbidden not in serialized
